"""Final Week-5 pipeline (frozen configuration).

End-to-end: Tier-1 (finalized MLP artifacts) -> Tier-2 (frozen gating:
HTTP HC 0.60, corr 0.30, weights 50/20/15/15) -> Tier-3 (Qwen2.5-1.5B-Instruct
with evidence-grounded triage + grounding validation).

NEW implementation — does not modify Weeks 1-4 source files. The existing
week5/ modules from the earlier attempt are historical/reference code; the
FINAL implementation lives in this package.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Paths (final pipeline owns its own artifacts; Tier-1 source is tier1/)
# ---------------------------------------------------------------------------
_curr = Path(__file__).resolve().parent
while not (_curr / "pyproject.toml").exists() and _curr != _curr.parent:
    _curr = _curr.parent
PROJECT_ROOT = _curr
TIER1_DIR = PROJECT_ROOT / "tier1"
DATA_DIR = PROJECT_ROOT / "data"
WEEK4_FINAL = PROJECT_ROOT / "week4" / "final_llm_reasoning"
W4_FROZEN_INPUT = PROJECT_ROOT / "week4" / "frozen_inputs" / "improved_tier2_test_predictions.csv"
W4_TRIAGE_OUTPUTS = WEEK4_FINAL / "outputs" / "llm_triage_outputs_final.csv"
W4_GROUNDING_RESULTS = WEEK4_FINAL / "outputs" / "grounding_results_final.csv"
W5_FINAL = PROJECT_ROOT / "week5" / "final_pipeline"
OUT = W5_FINAL / "outputs"

# ---------------------------------------------------------------------------
# FROZEN Tier-2 configuration (do not change)
# ---------------------------------------------------------------------------
HC_THRESHOLD = 0.60
CORR_THRESHOLD = 0.30
WEIGHTS = {"http": 0.50, "file_device": 0.20, "logon": 0.15, "email": 0.15}
assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9
LLM_MODEL = "qwen2.5:1.5b-instruct"
CHANNELS = ["http", "file_device", "logon", "email"]

# scenario mapping (final Week-4 mapping — do not use older inverted docs)
SCENARIO_MAPPING = {
    1: "WikiLeaks uploader",
    2: "Job-hunter exfiltrator",
    3: "Keylogger sysadmin",
}


def _cuda_usable() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        x = torch.randn(4, 4, device="cuda")
        _ = x @ x
        torch.cuda.synchronize()
        return True
    except Exception:
        return False


DEVICE = torch.device("cuda" if _cuda_usable() else "cpu")


# ---------------------------------------------------------------------------
# Tier-1 (finalized MLPs)
# ---------------------------------------------------------------------------
class ModalityFFNN(nn.Module):
    """Architecture of the frozen Tier-1 modality models (unchanged)."""

    def __init__(self, input_dim, hidden_dim=64, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


def load_tier1(artifact_dir: Path = TIER1_DIR) -> dict:
    """Load the four frozen Tier-1 modality MLPs + scalers + metadata.

    Returns a bundle dict:
      models: {modality: ModalityFFNN (eval)}
      scalers: {modality: fitted StandardScaler}
      feature_cols: {modality: [feature names in model order]}
      input_dims: {modality: int}
      meta: {modality: full meta.json}
    """
    bundle = {"models": {}, "scalers": {}, "feature_cols": {}, "input_dims": {}, "meta": {}}
    for mod in CHANNELS:
        meta = json.loads((artifact_dir / mod / "meta.json").read_text())
        bundle["meta"][mod] = meta
        bundle["feature_cols"][mod] = list(meta["feature_cols"])
        bundle["input_dims"][mod] = meta["input_dim"]
        bundle["scalers"][mod] = joblib.load(artifact_dir / mod / "scaler.joblib")
        model = ModalityFFNN(input_dim=meta["input_dim"], hidden_dim=meta["hidden_dim"])
        model.load_state_dict(torch.load(artifact_dir / mod / "model.pt",
                                         map_location="cpu", weights_only=True))
        model.to(DEVICE).eval()
        bundle["models"][mod] = model
    return bundle


def score_mlp(bundle, modality: str, features: np.ndarray) -> np.ndarray:
    """Tier-1 sigmoid anomaly score for raw features (model feature order)."""
    sc = bundle["scalers"][modality]
    model = bundle["models"][modality]
    X = sc.transform(np.asarray(features, dtype=np.float64)).astype(np.float32)
    model.eval()
    with torch.no_grad():
        x = torch.tensor(X, dtype=torch.float32).to(DEVICE)
        return torch.sigmoid(model(x)).cpu().numpy()


# ---------------------------------------------------------------------------
# Tier-2 (frozen gating) — vectorized reference implementation
# ---------------------------------------------------------------------------
def risk_score(df: pd.DataFrame) -> pd.Series:
    """Frozen weighted fusion (50/20/15/15)."""
    return (df["http_score"] * WEIGHTS["http"] +
            df["file_device_score"] * WEIGHTS["file_device"] +
            df["logon_score"] * WEIGHTS["logon"] +
            df["email_score"] * WEIGHTS["email"])


def tier2_decision(df: pd.DataFrame, degraded_channels: list | None = None) -> pd.DataFrame:
    """Frozen Tier-2 sequential gating (vectorized reference implementation).

    Gate 1 (high_confidence): http_score > HC_THRESHOLD
    Gate 2 (correlation):     risk_score > CORR_THRESHOLD
    Escalation = gate1 | gate2. Identical to the LangGraph orchestrator's
    sequential decision (see tier2_langgraph.py).

    degraded_channels: optional list of unavailable channels. When non-empty,
    the weights are renormalized over the remaining channels and the HTTP
    high-confidence gate is disabled if HTTP itself is missing (the missing
    score is NEVER fabricated).
    """
    out = df.copy()
    missing = set(degraded_channels or [])
    remaining = [c for c in CHANNELS if c not in missing]
    w = {c: WEIGHTS[c] for c in remaining}
    s = sum(w.values())
    w = {c: v / s for c, v in w.items()}
    risk = sum(out[f"{c}_score"] * w[c] for c in remaining)
    out["risk_score"] = risk
    hc = out["http_score"] > HC_THRESHOLD if "http" in remaining else np.zeros(len(out), dtype=bool)
    corr = risk > CORR_THRESHOLD
    out["escalate"] = (hc | corr).astype(int)
    out["gate_triggered"] = np.where(hc, "high_confidence",
                                     np.where(corr, "correlation", "normal"))
    return out


def degraded_fusion(df: pd.DataFrame, missing: str) -> pd.DataFrame:
    """Single-channel degradation (weight renormalization, no fabrication)."""
    return tier2_decision(df, degraded_channels=[missing])
