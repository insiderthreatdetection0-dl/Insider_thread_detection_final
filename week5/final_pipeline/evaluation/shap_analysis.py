"""Actual SHAP explanations for the Tier-1 models (final Week-5 implementation).

Uses the REAL `shap` package — `shap.DeepExplainer` (preference order #1 for
the PyTorch ModalityFFNN models), explaining the pre-sigmoid logit of each
frozen Tier-1 model. No gradient-only attribution is passed off as SHAP.

BACKGROUND DATA: training portion only (meta.train_users from the train
split), deterministic 100-sample sample (seed 42) per modality. Held-out test
samples are NEVER used as background.

SHAP OUTPUT: week5/final_pipeline/outputs/shap_attributions.csv with
case_id/user_id/date/channel/feature/feature_value/shap_value/abs_shap_value/
direction/rank/model_output/expected_value/shap_method, using the model's
canonical feature order.

SANITY: additive check logit == expected_value + sum(SHAP) per explained
(case, modality), reported in outputs/shap_sanity.csv.

Run with the SHAP-enabled interpreter, e.g.:
    .venv-shap/Scripts/python.exe -m week5.final_pipeline.evaluation.shap_analysis
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch

from ..pipeline import CHANNELS, DATA_DIR, OUT, W4_FROZEN_INPUT, load_tier1
from ..tier1 import assemble_feature_rows

SHAP_METHOD = "shap.DeepExplainer"
SHAP_VERSION = None
try:
    import shap
    SHAP_VERSION = shap.__version__
except Exception as e:  # pragma: no cover - import error path
    shap = None
    SHAP_IMPORT_ERROR = str(e)


class _LogitModel(torch.nn.Module):
    """2D-output wrapper so shap.DeepExplainer works on the frozen MLPs.

    The frozen models squeeze the logit to shape (batch,); DeepExplainer
    requires (batch, n_outputs). SHAP values are therefore defined w.r.t. the
    pre-sigmoid logit, which is what the additivity check validates.
    """

    def __init__(self, model):
        super().__init__()
        self.net = model.net

    def forward(self, x):
        return self.net(x)


def _background_sample(train_df: pd.DataFrame, meta: dict, seed: int = 42, n: int = 100) -> np.ndarray:
    """Deterministic training-only background (scaled) for one modality."""
    users = sorted(set(meta["train_users"]))
    bg = train_df[train_df["user"].isin(users)]
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(bg), n, replace=False)
    cols = meta["feature_cols"]
    X = bg.iloc[idx][cols].fillna(0).values.astype(np.float64)
    return X


def build_explainers(bundle: dict, train_df: pd.DataFrame,
                     background_n: int = 100, seed: int = 42) -> dict:
    """Instantiate shap.DeepExplainer per modality with training-only background."""
    if shap is None:
        raise RuntimeError(f"shap unavailable: {SHAP_IMPORT_ERROR}")
    explainers = {}
    for mod in CHANNELS:
        meta = bundle["meta"][mod]
        model = bundle["models"][mod]
        scaler = bundle["scalers"][mod]
        X_bg_raw = _background_sample(train_df, meta, seed=seed, n=background_n)
        X_bg = scaler.transform(X_bg_raw).astype(np.float32)
        torch.manual_seed(seed)
        explainers[mod] = shap.DeepExplainer(_LogitModel(model), torch.tensor(X_bg))
    return explainers


def explain_cases(bundle: dict, explainers: dict, cases: pd.DataFrame,
                  percentile_df: pd.DataFrame, top_k: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute SHAP values for every escalated case.

    Returns (attributions long DataFrame, sanity-check DataFrame).
    top_k: if set, keep only the |top_k| features per (case, channel) ranked by
           |shap_value|; otherwise keep all features.
    """
    rows = []
    sanity = []
    feats_df = assemble_feature_rows(percentile_df, cases, bundle)
    for _, r in feats_df.iterrows():
        cid = f"{int(r['user'])}_{int(r['day'])}"
        for mod in CHANNELS:
            meta = bundle["meta"][mod]
            scaler = bundle["scalers"][mod]
            model = bundle["models"][mod]
            expl = explainers[mod]
            raw = r[f"{mod}_features"]
            X = scaler.transform(raw.reshape(1, -1)).astype(np.float32)
            with torch.no_grad():
                logit = float(model(torch.tensor(X)).detach().numpy()[0])
            model_output = float(1.0 / (1.0 + np.exp(-logit)))
            sv = np.array(expl.shap_values(torch.tensor(X)))[0, :, 0]  # (n_features,)
            ev = float(np.array(expl.expected_value).reshape(-1)[0])
            residual = abs(logit - (ev + sv.sum()))
            sanity.append({"case_id": cid, "user_id": int(r["user"]), "date": int(r["day"]),
                           "channel": mod, "model_output": model_output, "logit": logit,
                           "expected_value": ev, "sum_shap": float(sv.sum()),
                           "residual": float(residual), "n_features": int(len(sv)),
                           "pass": bool(residual <= 1e-3)})
            order = np.argsort(-np.abs(sv))
            keep = order if top_k is None else order[:top_k]
            for rank, idx in enumerate(keep, start=1):
                feat = meta["feature_cols"][idx]
                rows.append({
                    "case_id": cid, "user_id": int(r["user"]), "date": int(r["day"]),
                    "channel": mod, "feature": feat, "feature_value": float(raw[idx]),
                    "shap_value": float(sv[idx]), "abs_shap_value": float(abs(sv[idx])),
                    "direction": "positive" if sv[idx] >= 0 else "negative",
                    "rank": int(rank), "model_output": model_output,
                    "expected_value": ev, "shap_method": SHAP_METHOD,
                    "shap_version": SHAP_VERSION,
                })
    return pd.DataFrame(rows), pd.DataFrame(sanity)


def run(background_n: int = 100, seed: int = 42, top_k: int | None = None) -> dict:
    """Full SHAP experiment over the 712 escalated cases."""
    assert shap is not None, f"shap import failed: {SHAP_IMPORT_ERROR}"
    print(f"SHAP version: {shap.__version__} | method: {SHAP_METHOD}")
    train_df = pd.read_pickle(str(DATA_DIR / "dayr4.2-percentile30-train.pkl"))
    test_df = pd.read_pickle(str(DATA_DIR / "dayr4.2-percentile30-test.pkl"))
    cases = pd.read_csv(W4_FROZEN_INPUT)
    cases = cases[cases["escalate"] == 1].copy()
    print(f"Escalated cases: {len(cases)} | background: train-only n={background_n} seed={seed}")

    bundle = load_tier1()
    explainers = build_explainers(bundle, train_df, background_n=background_n, seed=seed)
    attr, sanity = explain_cases(bundle, explainers, cases, test_df, top_k=top_k)

    OUT.mkdir(parents=True, exist_ok=True)
    attr.to_csv(OUT / "shap_attributions.csv", index=False)
    sanity.to_csv(OUT / "shap_sanity.csv", index=False)
    print(f"Saved shap_attributions.csv ({len(attr):,} rows) + shap_sanity.csv ({len(sanity)} rows)")

    # sanity summary
    n_pass = int(sanity["pass"].sum())
    n_fail = int((~sanity["pass"]).sum())
    summary = {
        "shap_method": SHAP_METHOD, "shap_version": SHAP_VERSION,
        "background_source": "training split only (meta.train_users)",
        "background_n": background_n, "background_seed": seed,
        "n_explained": len(sanity), "n_pass": n_pass, "n_fail": n_fail,
        "max_residual": float(sanity["residual"].max()), "mean_residual": float(sanity["residual"].mean()),
        "n_attribution_rows": len(attr),
    }
    print(f"\nSHAP sanity: {n_pass}/{len(sanity)} passed additivity (residual<=1e-3) | "
          f"max residual {summary['max_residual']:.3e} | mean {summary['mean_residual']:.3e}")
    with open(OUT / "shap_meta.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


if __name__ == "__main__":
    run(background_n=100, seed=42)
