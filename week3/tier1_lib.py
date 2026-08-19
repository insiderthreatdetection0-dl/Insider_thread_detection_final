"""Tier-1 shared library — SINGLE SOURCE OF TRUTH for the finalized Tier-1 MLP pipeline.

Finalized experimental design (DO NOT CHANGE):
    HTTP        -> ModalityFFNN on percentile-30 features, undersample 6:1
    File/Device -> ModalityFFNN on percentile-30 features, undersample 4:1
    Email       -> ModalityFFNN on percentile-30 features, undersample 4:1
    Logon       -> ModalityFFNN on percentile-30 features, undersample 4:1
    split       -> official 60/20/20 user-level split (data/user_splits/*.npy,
                   mirrored by data/user_split_formlp/*.pkl)
    scaler      -> StandardScaler fit on TRAIN users only (60%)
    loss/opt    -> BCEWithLogitsLoss, Adam(lr=0.001), batch 128, 20 epochs, SEED=42

Reproducibility guarantees enforced here:
  * feature columns are built deterministically (sorted) — never via set()
  * the exact ordered feature list is persisted in each modality meta.json
  * the fitted scaler is persisted (joblib) next to the model
  * scoring always loads the persisted scaler + meta (never refits)
  * every consumer asserts input_dim == len(feature_cols) == len(scaler.mean_)
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_curr = Path(__file__).resolve().parent
while not (_curr / "pyproject.toml").exists() and _curr != _curr.parent:
    _curr = _curr.parent
PROJECT_ROOT = _curr
DATA_DIR = PROJECT_ROOT / "data"
SPLIT_DIR = DATA_DIR / "user_split_formlp"
USER_SPLITS_DIR = DATA_DIR / "user_splits"
TIER1_DIR = PROJECT_ROOT / "tier1"
SCORES_DIR = TIER1_DIR / "scores"
MODALITIES = ["http", "file_device", "email", "logon"]

# ---------------------------------------------------------------------------
# Finalized config (single source of truth)
# ---------------------------------------------------------------------------
SEED = 42

# sampling_strategy passed to RandomUnderSampler (majority:minority ratio)
SAMPLING_RATIO = {
    "http": 1 / 6,          # 6:1
    "file_device": 0.25,    # 4:1
    "email": 0.25,          # 4:1
    "logon": 0.25,          # 4:1
}
RATIO_LABEL = {
    "http": "6:1",
    "file_device": "4:1",
    "email": "4:1",
    "logon": "4:1",
}

HIDDEN_DIM = 64
DROPOUT = 0.2
EPOCHS = 20
LR = 0.001
BATCH_SIZE = 128

PERSONALITY_COLS = ["O", "C", "E", "A", "N"]
META_COLS = ["user", "week", "day", "insider"]


def _cuda_usable() -> bool:
    """Probe CUDA with a real op; fall back to CPU (RTX 5090 / sm_120 issue)."""
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


def seed_all(seed: int = SEED) -> torch.Generator:
    """Seed every RNG + build the shared DataLoader generator."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    g = torch.Generator()
    g.manual_seed(seed)
    return g


# ---------------------------------------------------------------------------
# Deterministic feature selection (NO set() ordering)
# ---------------------------------------------------------------------------
_MODALITY_KEYWORDS = {
    "logon": ("logon", "logout"),
    "email": ("email", "mail"),
    "http": ("http", "url"),
    "file_device": ("file", "device", "comp", "disk", "phof", "docf", "txtf", "exef"),
}


def get_modality_columns(all_cols: list[str], modality: str) -> list[str]:
    """Deterministic, ordered feature columns for a modality.

    Modality feature columns are matched by keyword and SORTED; the five
    personality columns (O,C,E,A,N) are appended in their canonical order.
    This is the ONLY feature-ordering function used by the finalized pipeline.
    """
    if modality not in _MODALITY_KEYWORDS:
        raise ValueError(f"Unknown modality: {modality}")
    keywords = _MODALITY_KEYWORDS[modality]
    cols = sorted(c for c in all_cols if any(k in c for k in keywords))
    return cols + list(PERSONALITY_COLS)


def assert_feature_consistency(model, scaler, feature_cols, modality: str) -> None:
    """Fail loudly if model/scaler/feature list disagree (Step 7 of the task)."""
    input_dim = model.net[0].in_features if hasattr(model, "net") else None
    assert input_dim == len(feature_cols), (
        f"{modality}: model input dim {input_dim} != len(feature_cols) {len(feature_cols)}"
    )
    assert len(scaler.mean_) == len(feature_cols), (
        f"{modality}: scaler.mean_ len {len(scaler.mean_)} != {len(feature_cols)}"
    )
    assert len(scaler.scale_) == len(feature_cols), (
        f"{modality}: scaler.scale_ len {len(scaler.scale_)} != {len(feature_cols)}"
    )


# ---------------------------------------------------------------------------
# Model (verbatim Week-2 ModalityFFNN)
# ---------------------------------------------------------------------------
class ModalityFFNN(nn.Module):
    """3-layer MLP: input -> 64 -> 32 -> 1 (ReLU, dropout 0.2)."""

    def __init__(self, input_dim: int, hidden_dim: int = HIDDEN_DIM, dropout: float = DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


class FlatDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# ---------------------------------------------------------------------------
# Split loading / verification
# ---------------------------------------------------------------------------
def load_official_split():
    """Load the official 60/20/20 user-level split from data/user_splits/*.npy."""
    train = np.load(USER_SPLITS_DIR / "train_users.npy")
    val = np.load(USER_SPLITS_DIR / "val_users.npy")
    test = np.load(USER_SPLITS_DIR / "test_users.npy")
    return train, val, test


def load_pre_split_data():
    """Load train/val/test percentile DataFrames from data/user_split_formlp/."""
    files = {
        "train": SPLIT_DIR / "dayr4.2-percentile30-train.pkl",
        "val": SPLIT_DIR / "dayr4.2-percentile30-validation.pkl",
        "test": SPLIT_DIR / "dayr4.2-percentile30-test.pkl",
    }
    out = {}
    for key, p in files.items():
        if not p.exists():
            raise FileNotFoundError(f"Split file not found: {p}")
        df = pd.read_pickle(str(p))
        for col in df.select_dtypes(include=["float64"]).columns:
            df[col] = df[col].astype("float32")
        out[key] = df
    return out["train"], out["val"], out["test"]


def verify_split(train_df, val_df, test_df) -> dict:
    """Verify 600/200/200 users, zero overlap, malicious counts (Step 8)."""
    tr, va, te = set(train_df["user"].unique()), set(val_df["user"].unique()), set(test_df["user"].unique())
    assert len(tr) == 600 and len(va) == 200 and len(te) == 200, (
        f"split sizes {len(tr)}/{len(va)}/{len(te)} != 600/200/200"
    )
    assert not (tr & va) and not (tr & te) and not (va & te), "user overlap in split!"
    assert len(tr | va | te) == 1000
    malicious = {
        "train": int((train_df.groupby("user")["insider"].max() > 0).sum()),
        "val": int((val_df.groupby("user")["insider"].max() > 0).sum()),
        "test": int((test_df.groupby("user")["insider"].max() > 0).sum()),
    }
    return {"train_users": len(tr), "val_users": len(va), "test_users": len(te),
            "malicious": malicious}


# ---------------------------------------------------------------------------
# Training (Step 4: scaler fit on train only, then persisted)
# ---------------------------------------------------------------------------
def train_modality(modality: str, train_df: pd.DataFrame, seed: int = SEED,
                   epochs: int = EPOCHS, lr: float = LR, batch_size: int = BATCH_SIZE,
                   hidden_dim: int = HIDDEN_DIM):
    """Train ONE finalized MLP. Returns (model, scaler, feature_cols, metrics).

    Scaler is fitted on the 60% train users only; train is then undersampled.
    Validation/test data must NOT be passed here.
    """
    g = seed_all(seed)
    feature_cols = get_modality_columns(train_df.columns.tolist(), modality)
    ratio = SAMPLING_RATIO[modality]

    y_train = (train_df["insider"].values > 0).astype(int)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[feature_cols].fillna(0)).astype(np.float32)

    rus = RandomUnderSampler(sampling_strategy=ratio, random_state=seed)
    X_res, y_res = rus.fit_resample(X_train, y_train)
    print(f"  {modality:12s} features={len(feature_cols):3d} ratio={RATIO_LABEL[modality]:>3s} "
          f"train={len(X_train):6d} (pos={y_train.sum()}) -> resampled={len(X_res):6d} (pos={y_res.sum()})")

    train_loader = DataLoader(FlatDataset(X_res, y_res), batch_size=batch_size,
                              shuffle=True, generator=g)

    model = ModalityFFNN(input_dim=len(feature_cols), hidden_dim=hidden_dim).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total = 0.0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            optimizer.step()
            total += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"      epoch {epoch+1}/{epochs} | loss={total/len(train_loader):.4f}")

    metrics = {"epochs": epochs, "lr": lr, "batch_size": batch_size,
               "hidden_dim": hidden_dim, "seed": seed,
               "sampling_strategy": ratio, "ratio": RATIO_LABEL[modality]}
    return model, scaler, feature_cols, metrics


# ---------------------------------------------------------------------------
# Artifact IO (Step 3/4/15: tier1/<modality>/{model.pt,scaler.joblib,meta.json})
# ---------------------------------------------------------------------------
def modality_dir(modality: str) -> Path:
    return TIER1_DIR / modality


def build_meta(modality: str, feature_cols: list[str], scaler, model, metrics: dict,
               split_info: dict) -> dict:
    meta = {
        "modality": modality,
        "model_type": "MLP",
        "arch": "ModalityFFNN",
        "input_dim": len(feature_cols),
        "hidden_dim": metrics["hidden_dim"],
        "sampling_ratio": RATIO_LABEL[modality],
        "sampling_strategy": metrics["sampling_strategy"],
        "feature_cols": feature_cols,
        "scaler_type": "StandardScaler",
        "train_users": [int(u) for u in split_info["train_users"]],
        "validation_users": [int(u) for u in split_info["val_users"]],
        "test_users": [int(u) for u in split_info["test_users"]],
        "malicious_users": split_info["malicious"],
        "seed": metrics["seed"],
        "epochs": metrics["epochs"],
        "lr": metrics["lr"],
        "batch_size": metrics["batch_size"],
    }
    # compatibility checks recorded
    meta["checks"] = {
        "model_input_dim": len(feature_cols),
        "scaler_mean_len": int(len(scaler.mean_)),
        "scaler_scale_len": int(len(scaler.scale_)),
    }
    return meta


def save_artifacts(modality: str, model, scaler, feature_cols: list[str],
                   metrics: dict, split_info: dict) -> Path:
    """Persist model.pt + scaler.joblib + meta.json under tier1/<modality>/."""
    d = modality_dir(modality)
    d.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), d / "model.pt")
    joblib.dump(scaler, d / "scaler.joblib")
    meta = build_meta(modality, feature_cols, scaler, model, metrics, split_info)
    with open(d / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    return d


def load_artifacts(modality: str):
    """Load (model, scaler, feature_cols, meta) from tier1/<modality>/, verifying consistency."""
    d = modality_dir(modality)
    with open(d / "meta.json") as f:
        meta = json.load(f)
    feature_cols = list(meta["feature_cols"])
    scaler = joblib.load(d / "scaler.joblib")

    sd = torch.load(d / "model.pt", map_location="cpu", weights_only=True)
    model = ModalityFFNN(input_dim=meta["input_dim"], hidden_dim=meta["hidden_dim"])
    model.load_state_dict(sd)
    model.to(DEVICE).eval()

    # Step 6/7: fail loudly on any inconsistency
    assert meta["modality"] == modality
    assert meta["model_type"] == "MLP"
    assert len(feature_cols) == meta["input_dim"], f"{modality}: meta input_dim mismatch"
    assert len(scaler.mean_) == len(feature_cols), f"{modality}: scaler/feature len mismatch"
    assert len(scaler.scale_) == len(feature_cols), f"{modality}: scaler/feature len mismatch"
    assert_feature_consistency(model, scaler, feature_cols, modality)
    return model, scaler, feature_cols, meta


def load_all_artifacts():
    """Load all four finalized artifacts; returns dict modality -> (model, scaler, feature_cols, meta)."""
    return {m: load_artifacts(m) for m in MODALITIES}


# ---------------------------------------------------------------------------
# Inference (Step 5: use persisted scaler + feature order, never refit)
# ---------------------------------------------------------------------------
@torch.no_grad()
def score_dataframe(df: pd.DataFrame, model, scaler, feature_cols: list[str],
                    batch_size: int = 4096) -> np.ndarray:
    """Return sigmoid probabilities for df rows, features in EXACT saved order."""
    assert_feature_consistency(model, scaler, feature_cols, "?")
    X = scaler.transform(df[feature_cols].fillna(0)).astype(np.float32)
    out = []
    model.eval()
    Xt = torch.tensor(X, dtype=torch.float32).to(DEVICE)
    for i in range(0, len(Xt), batch_size):
        out.append(torch.sigmoid(model(Xt[i:i + batch_size])).cpu().numpy())
    return np.concatenate(out)


def score_mlp_row(model, scaler, feature_cols: list[str], row: np.ndarray) -> float:
    """Score one raw feature row (already in feature_cols order)."""
    X = scaler.transform(np.asarray(row, dtype=np.float64).reshape(1, -1)).astype(np.float32)
    model.eval()
    with torch.no_grad():
        x = torch.tensor(X, dtype=torch.float32).to(DEVICE)
        return float(torch.sigmoid(model(x)).cpu().numpy()[0])
