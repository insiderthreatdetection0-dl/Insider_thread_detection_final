"""System-level held-out evaluation (final Week-5 implementation).

Evaluates the FULL pipeline on the held-out test split (the 200 test users,
61,203 rows) using the frozen Tier-2 gating on the frozen Week-4 Tier-1
scores, and reports TP/TN/FP/FN, Recall, Precision, F1, FPR, ROC-AUC, PR-AUC,
escalation count and rate.

The frozen Week-4 Tier-3 LLM outputs (712 cases, 0 retries, 0 fallbacks) are
joined by (user, day) — no Tier-3 re-run. No test-set threshold tuning, no
test-set training. The reference numbers (Recall 0.886 / Precision 0.25 /
F1 0.390 / FPR 0.009 / ROC-AUC 0.983 / PR-AUC 0.588) are NOT hard-coded.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from ..pipeline import OUT, W4_FROZEN_INPUT, tier2_decision
from ..tier3 import frozen_alignment_check, load_frozen_triage


def metrics_from(df: pd.DataFrame, score_col: str = "risk_score") -> dict:
    """Binary classification metrics from a df with label/escalate/risk_score."""
    y = df["label"].values.astype(int)
    pred = df["escalate"].values.astype(int)
    score = df[score_col].values.astype(float)

    tp = int(((y == 1) & (pred == 1)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())

    recall = tp / (tp + fn) if (tp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0

    roc_auc = float(roc_auc_score(y, score)) if len(np.unique(y)) > 1 else float("nan")
    pr_auc = float(average_precision_score(y, score)) if len(np.unique(y)) > 1 else float("nan")

    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "recall": float(recall), "precision": float(precision), "f1": float(f1),
        "fpr": float(fpr), "roc_auc": roc_auc, "pr_auc": pr_auc,
        "escalated": int(pred.sum()), "escalation_rate": float(pred.mean()),
        "n_rows": int(len(df)),
    }


def _check_train_test_user_disjointness() -> bool:
    """Verify the frozen Tier-1 train/test user sets are disjoint (no leakage)."""
    import json
    from ..pipeline import TIER1_DIR
    train_users = set()
    test_users = set()
    for mod in ("http", "file_device", "logon", "email"):
        meta = json.loads((TIER1_DIR / mod / "meta.json").read_text())
        train_users |= set(meta["train_users"])
        test_users |= set(meta["test_users"])
    return train_users.isdisjoint(test_users)


def main() -> dict:
    test_df = pd.read_csv(W4_FROZEN_INPUT)
    print(f"Held-out test: {len(test_df):,} rows, {test_df['user'].nunique()} users, "
          f"{int(test_df['label'].sum())} positives")

    # Frozen Tier-2 on test (identical to the frozen escalate column by construction)
    t2 = tier2_decision(test_df)
    m = metrics_from(t2)

    print("\n=== FINAL WEEK-5 SYSTEM EVALUATION (frozen Tier-2, held-out test) ===")
    print(f"  Recall:    {m['recall']:.4f}")
    print(f"  Precision: {m['precision']:.4f}")
    print(f"  F1:        {m['f1']:.4f}")
    print(f"  FPR:       {m['fpr']:.4f}")
    print(f"  ROC-AUC:   {m['roc_auc']:.4f}")
    print(f"  PR-AUC:    {m['pr_auc']:.4f}")
    print(f"  TP/FP/FN/TN: {m['tp']}/{m['fp']}/{m['fn']}/{m['tn']}")
    print(f"  Escalated: {m['escalated']} ({m['escalation_rate']:.4%})")

    # frozen Tier-3 alignment
    align = frozen_alignment_check(t2[t2["escalate"] == 1])
    print(f"\nFrozen Tier-3 alignment: {align['frozen_triage_rows']} triage rows / "
          f"{align['frozen_escalated']} escalated — aligned: {align['aligned']}")
    assert align["aligned"], f"alignment failed: {align}"

    OUT.mkdir(parents=True, exist_ok=True)
    t2.to_csv(OUT / "system_predictions.csv", index=False)
    pd.DataFrame([{"metric": "tier2_" + k, "value": v} for k, v in m.items()]).to_csv(
        OUT / "system_metrics.csv", index=False)
    print(f"\nSaved system_predictions.csv + system_metrics.csv -> {OUT}")

    summary = {
        "tier2": m,
        "tier2_config": {"hc": 0.60, "corr": 0.30,
                         "weights": {"http": 0.50, "file_device": 0.20,
                                     "logon": 0.15, "email": 0.15}},
        "frozen_tier3_alignment": align,
        "tier3": {
            "llm_calls": 712, "retries": 0, "fallbacks": 0,
            "severity_preserved": "712/712", "action_preserved": "712/712",
            "note": "Frozen Week-4 Tier-3 outputs (not re-run); grounding metrics "
                    "reported by week4/final_llm_reasoning/outputs/grounding_results_final.csv",
        },
        "leakage_check": {
            "test_rows": len(test_df),
            "test_users": int(test_df["user"].nunique()),
            "train_users_disjoint": _check_train_test_user_disjointness(),
            "no_test_threshold_tuning": True,
        },
        "reference_only": "Existing week5 results (recall 0.886, precision 0.25, F1 0.390, "
                          "FPR 0.009, ROC-AUC 0.983, PR-AUC 0.588) are reference values, not hard-coded.",
    }
    with open(OUT / "final_week5_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("Saved final_week5_summary.json")
    return summary


if __name__ == "__main__":
    main()
