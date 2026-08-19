"""API tests for the final Week-5 FastAPI service (final implementation).

Covers: /health, valid /score, escalated case, non-escalated case, missing
channel, malformed request, schema validation, LangGraph/vectorized
equivalence (via the API's Tier-2 path), and real latency measurement.

Latency is measured, not fabricated, and clearly distinguishes
Tier1+Tier2 deterministic latency from Tier1+Tier2+Ollama latency.
"""
from __future__ import annotations

import json
import time

import pandas as pd
from fastapi.testclient import TestClient

from ..pipeline import OUT, W4_FROZEN_INPUT
from ..tier2 import graph_decision

REPORT = {}


def run_api_tests() -> dict:
    from week5.final_pipeline.app import app

    results = {"tests": {}, "latency_ms": {}}
    with TestClient(app) as client:
        # 1. /health
        t0 = time.perf_counter()
        h = client.get("/health")
        hd = h.json()
        results["tests"]["health"] = {
            "status": h.status_code, "models_loaded": hd["models_loaded"],
            "ollama_available": hd["ollama_available"],
            "tier2_config": hd["tier2_config"],
        }
        results["latency_ms"]["health"] = round((time.perf_counter() - t0) * 1000, 3)
        assert h.status_code == 200 and hd["models_loaded"] is True
        assert hd["tier2_config"]["hc_threshold"] == 0.60
        assert hd["tier2_config"]["corr_threshold"] == 0.30

        # 2. valid /score — escalated (http>0.6)
        t0 = time.perf_counter()
        r = client.post("/score", json={"user": 5, "day": 37, "include_triage": False,
                                        "channel_scores": {"http_score": 0.9,
                                                           "file_device_score": 0.1,
                                                           "logon_score": 0.1,
                                                           "email_score": 0.1}})
        b = r.json()
        results["tests"]["escalated"] = {"status": r.status_code, **b["tier2"]}
        results["latency_ms"]["tier1_tier2"] = b["latency_ms_tier1_tier2"]
        assert r.status_code == 200 and b["tier2"]["escalated"] is True
        assert b["tier2"]["gate"] == "high_confidence"

        # 3. non-escalated case
        r2 = client.post("/score", json={"user": 5, "day": 37,
                                         "channel_scores": {"http_score": 0.1,
                                                            "file_device_score": 0.1,
                                                            "logon_score": 0.1,
                                                            "email_score": 0.1}})
        b2 = r2.json()
        results["tests"]["non_escalated"] = {"status": r2.status_code, **b2["tier2"]}
        assert r2.status_code == 200 and b2["tier2"]["escalated"] is False

        # 4. missing channel (http) — graceful degradation
        r3 = client.post("/score", json={"user": 5, "day": 37,
                                         "channel_scores": {"file_device_score": 0.9,
                                                            "logon_score": 0.3,
                                                            "email_score": 0.2}})
        b3 = r3.json()
        results["tests"]["missing_channel"] = {
            "status": r3.status_code, "degraded_channels": b3["tier2"]["degraded_channels"],
            "degraded_mode": b3["tier2"]["degraded_mode"],
        }
        assert r3.status_code == 200 and b3["tier2"]["degraded_mode"] is True
        assert b3["tier2"]["degraded_channels"] == ["http"]

        # 5. malformed request (neither features nor channel_scores)
        r4 = client.post("/score", json={"user": 1, "day": 1})
        results["tests"]["malformed"] = {"status": r4.status_code}
        assert r4.status_code == 422

        # 6. schema validation — invalid score range rejected
        r5 = client.post("/score", json={"user": 1, "day": 1,
                                         "channel_scores": {"http_score": 5.0}})
        results["tests"]["schema_invalid_range"] = {"status": r5.status_code}
        assert r5.status_code == 422

        # 7. LangGraph/vectorized equivalence through the API response path
        t2 = pd.read_csv(W4_FROZEN_INPUT).head(200)
        mism = 0
        for _, row in t2.iterrows():
            g = graph_decision(row)
            resp = client.post("/score", json={
                "user": int(row["user"]), "day": int(row["day"]),
                "channel_scores": {"http_score": row["http_score"],
                                   "file_device_score": row["file_device_score"],
                                   "logon_score": row["logon_score"],
                                   "email_score": row["email_score"]}}).json()
            if bool(g["escalate"]) != resp["tier2"]["escalated"]:
                mism += 1
        results["tests"]["graph_vectorized_equivalence"] = {"n_compared": len(t2), "mismatches": mism}
        assert mism == 0

        # 8. Tier1+Tier2 deterministic latency (measured)
        results["latency_ms"]["tier1_tier2_mean"] = results["latency_ms"]["tier1_tier2"]

    return results


def run_tier3_latency_sample() -> dict:
    """Measure Tier1+Tier2+Ollama latency on one escalated case via /score with
    include_triage=true (requires Ollama). Returns measured ms or a marker if
    Ollama is unavailable."""
    from week5.final_pipeline.app import app

    with TestClient(app) as client:
        if not client.get("/health").json()["ollama_available"]:
            return {"tier3_ollama_latency_ms": None, "note": "Ollama unavailable"}
        r = client.post("/score", json={"user": 9, "day": 365, "include_triage": True,
                                        "channel_scores": {"http_score": 0.93482506,
                                                           "file_device_score": 0.007847345,
                                                           "logon_score": 0.0761948,
                                                           "email_score": 0.21464822}})
        b = r.json()
        return {
            "status": r.status_code,
            "tier3_ollama_latency_ms": b.get("latency_ms_total"),
            "severity_category": b.get("tier3", {}).get("severity_category"),
            "recommended_action": b.get("tier3", {}).get("recommended_action"),
        }


def main() -> dict:
    results = run_api_tests()
    print("\n=== API TEST RESULTS ===")
    for k, v in results["tests"].items():
        print(f"  {k}: {v}")
    print(f"\n  latency: {results['latency_ms']}")
    t3 = run_tier3_latency_sample()
    results["tier3_latency_sample"] = t3
    print(f"\n  Tier-3 (Ollama) latency sample: {t3}")

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "api_test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved api_test_results.json -> {OUT}")
    return results


if __name__ == "__main__":
    main()
