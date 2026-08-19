"""Tier-1 modality agents (final Week-5 implementation).

Wraps the frozen Week-1-4 Tier-1 artifacts (tier1/<mod>/{meta.json, model.pt,
scaler.joblib}). Provides:
  - score_features: raw percentile feature row -> per-modality sigmoid scores
  - assemble_feature_rows: percentile DataFrame -> per-modality raw feature rows
    for a set of (user, day) cases, using the model's canonical feature order
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .pipeline import CHANNELS, load_tier1, score_mlp


def assemble_feature_rows(percentile_df: pd.DataFrame, cases: pd.DataFrame,
                          bundle: dict) -> pd.DataFrame:
    """Attach the raw Tier-1 feature values to escalated case rows.

    percentile_df: the full dayr4.2-percentile30 DataFrame (centered
                   percentile-minus-50 representation).
    cases: DataFrame with user/day (+ any extra columns).
    bundle: Tier-1 bundle from pipeline.load_tier1().

    Returns cases with one added column per modality: f"<mod>_features" holding
    the np.ndarray of raw feature values in the model's canonical order
    (missing percentile values filled with 0, as done by the frozen pipeline).
    """
    out = cases.copy()
    pct_idx = percentile_df.set_index(["user", "day"], drop=False)
    cols_by_mod = bundle["feature_cols"]
    for mod in CHANNELS:
        vals = []
        cols = cols_by_mod[mod]
        for _, r in out.iterrows():
            key = (int(r["user"]), int(r["day"]))
            try:
                row = pct_idx.loc[key]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
                vals.append(row[cols].fillna(0).values.astype(np.float64))
            except KeyError:
                vals.append(np.zeros(len(cols), dtype=np.float64))
        out[f"{mod}_features"] = vals
    return out


def score_cases(bundle: dict, feature_rows: pd.DataFrame) -> pd.DataFrame:
    """Tier-1 scores for every case row (uses the attached feature arrays)."""
    out = feature_rows.copy()
    for mod in CHANNELS:
        out[f"{mod}_score"] = [
            float(score_mlp(bundle, mod, np.asarray(v).reshape(1, -1))[0])
            for v in out[f"{mod}_features"]
        ]
    return out
