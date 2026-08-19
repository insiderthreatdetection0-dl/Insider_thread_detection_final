"""Aggregate grounding evaluation for the final Tier-3 module (Week 4 final)."""
from __future__ import annotations

import json

import pandas as pd


def summarize_grounding(results_df: pd.DataFrame) -> dict:
    """Aggregate per-case grounding metrics across the triage run.

    reports both macro (mean-of-per-case-F1) and aggregate-PR
    (harmonic F1 of the aggregate precision/recall) for channel and feature.
    """
    n = len(results_df)
    if n == 0:
        return {}

    def _count_unsup(x):
        if isinstance(x, list):
            return len(x)
        if isinstance(x, str):
            try:
                return len(json.loads(x)) if x.strip() else 0
            except Exception:
                return len(x.split(";")) if x.strip() else 0
        return 0

    def _f1_from_pr(p, r):
        return (2 * p * r / (p + r)) if (p + r) else 0.0

    unsup_count = int(results_df["unsupported_claims"].apply(_count_unsup).sum())

    ch_p = float(results_df["channel_precision"].mean())
    ch_r = float(results_df["channel_recall"].mean())
    f_p = float(results_df["feature_precision"].mean())
    f_r = float(results_df["feature_recall"].mean())

    return {
        "n_cases": n,
        # channel
        "channel_grounding_precision": ch_p,
        "channel_grounding_recall": ch_r,
        "channel_f1_macro_case": float(results_df["channel_f1"].mean()),
        "channel_f1_from_aggregate_pr": _f1_from_pr(ch_p, ch_r),
        # feature
        "feature_grounding_precision": f_p,
        "feature_grounding_recall": f_r,
        "feature_f1_macro_case": float(results_df["feature_f1"].mean()),
        "feature_f1_from_aggregate_pr": _f1_from_pr(f_p, f_r),
        # unsupported
        "unsupported_claim_count": unsup_count,
        "unsupported_claim_rate": unsup_count / n if n else 0.0,
        # severity/action
        "severity_consistency": float(results_df["severity_valid"].mean()),
        "action_consistency": float(results_df["action_valid"].mean()),
        # evidence IDs (P0-1 separated metrics)
        "grounding_valid_rate": float(results_df["grounding_valid"].mean()),
        "evidence_id_contract_compliance_rate": float(results_df["evidence_id_contract_compliant"].mean()),
        "evidence_id_extraction_fallback_rate": float(results_df["evidence_id_extraction_fallback_used"].mean()),
        "evidence_id_grounding_valid_rate": float(results_df["evidence_id_valid"].mean()),
    }
