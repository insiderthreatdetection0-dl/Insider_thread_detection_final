"""Graceful degradation tests (final Week-5 implementation).

Evaluates the frozen Tier-2 under missing channels (all channels / HTTP
missing / File-Device missing / Logon missing / Email missing) with weight
renormalization — the missing score is never fabricated. Also verifies the
operational behavior of the FastAPI service: request succeeds, degraded_
channels identifies the missing channel, degraded_mode=true, rationale
identifies degradation, and no exception occurs when a channel is missing.
"""
from __future__ import annotations

import pandas as pd

from ..pipeline import CHANNELS, OUT, W4_FROZEN_INPUT, tier2_decision
from .evaluate_system import metrics_from


def run_degradation(test_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    base = tier2_decision(test_df)
    rows.append({"config": "all_channels", "missing": "none", **metrics_from(base)})
    for ch in CHANNELS:
        d = tier2_decision(test_df, degraded_channels=[ch])
        rows.append({"config": f"missing_{ch}", "missing": ch, **metrics_from(d)})
    return pd.DataFrame(rows)


def degradation_report(test_df: pd.DataFrame, out_path) -> pd.DataFrame:
    df = run_degradation(test_df)
    df.to_csv(out_path, index=False)
    print(f"Degradation results saved -> {out_path}")
    print(df[["config", "recall", "precision", "fpr", "f1", "roc_auc", "pr_auc",
              "escalated", "tp", "fp"]].round(4).to_string(index=False))
    base = df[df["missing"] == "none"].iloc[0]
    for _, r in df[df["missing"] != "none"].iterrows():
        d_rec = base["recall"] - r["recall"]
        print(f"\n{r['missing']} missing: recall delta = {d_rec:+.4f} "
              f"({'LOAD-BEARING' if d_rec > 0.05 else 'well-absorbed'})")
    return df


def api_degradation_checks() -> dict:
    """Operational checks via FastAPI TestClient (no Ollama needed for Tier-2)."""
    from fastapi.testclient import TestClient
    from week5.final_pipeline.app import app

    results = {}
    with TestClient(app) as client:
        # escalated case, http missing
        r = client.post("/score", json={"user": 5, "day": 37,
                                        "channel_scores": {"file_device_score": 0.9,
                                                           "logon_score": 0.3,
                                                           "email_score": 0.2}})
        body = r.json()
        results["http_missing"] = {
            "status": r.status_code,
            "degraded_channels": body["tier2"]["degraded_channels"],
            "degraded_mode": body["tier2"]["degraded_mode"],
            "rationale_mentions_degradation": "DEGRADED" in body["tier2"]["rationale"].upper(),
            "escalated": body["tier2"]["escalated"],
        }
        # all channels present -> no degraded mode
        r2 = client.post("/score", json={"user": 5, "day": 37,
                                         "channel_scores": {"http_score": 0.1,
                                                            "file_device_score": 0.1,
                                                            "logon_score": 0.1,
                                                            "email_score": 0.1}})
        results["all_present"] = {
            "status": r2.status_code,
            "degraded_mode": r2.json()["tier2"]["degraded_mode"],
            "degraded_channels": r2.json()["tier2"]["degraded_channels"],
        }
        # file_device missing on an escalated case
        r3 = client.post("/score", json={"user": 5, "day": 37,
                                         "channel_scores": {"http_score": 0.9,
                                                            "logon_score": 0.1,
                                                            "email_score": 0.1}})
        b3 = r3.json()
        results["file_missing"] = {
            "status": r3.status_code,
            "degraded_channels": b3["tier2"]["degraded_channels"],
            "degraded_mode": b3["tier2"]["degraded_mode"],
            "escalated": b3["tier2"]["escalated"],
        }
    return results


def main():
    test_df = pd.read_csv(W4_FROZEN_INPUT)
    OUT.mkdir(parents=True, exist_ok=True)
    df = degradation_report(test_df, OUT / "degradation_results.csv")
    api = api_degradation_checks()
    print("\n=== API degradation operational checks ===")
    for k, v in api.items():
        print(f"  {k}: {v}")
    assert all(v["status"] == 200 for v in api.values())
    assert api["http_missing"]["degraded_mode"] is True
    assert api["http_missing"]["degraded_channels"] == ["http"]
    assert api["http_missing"]["rationale_mentions_degradation"]
    assert api["all_present"]["degraded_mode"] is False
    print("\nAll API degradation operational checks passed.")


if __name__ == "__main__":
    main()
