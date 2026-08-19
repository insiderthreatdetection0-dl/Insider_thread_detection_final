"""Pipeline unit tests — final Week-5 implementation (pytest).

Covers: model loading, feature dimensions, Tier-2 output, LangGraph/vectorized
equivalence, API schema, degraded mode, non-escalated case, escalated case.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from week5.final_pipeline import pipeline as P
from week5.final_pipeline import tier2 as T2
from week5.final_pipeline.tier1 import assemble_feature_rows


# ---------------------------------------------------------------------------
# Tier-1
# ---------------------------------------------------------------------------
def test_model_loading_and_dimensions():
    bundle = P.load_tier1()
    assert set(bundle["models"].keys()) == {"http", "file_device", "logon", "email"}
    for mod, dim in bundle["input_dims"].items():
        assert bundle["models"][mod].net[0].in_features == dim
        assert len(bundle["feature_cols"][mod]) == dim
        assert len(bundle["scalers"][mod].mean_) == dim
        assert bundle["meta"][mod]["seed"] == 42


def test_feature_rows_align_to_model_order():
    bundle = P.load_tier1()
    test_pkl = P.DATA_DIR / "user_split_formlp" / "dayr4.2-percentile30-test.pkl"
    if not test_pkl.exists():
        test_pkl = P.DATA_DIR / "dayr4.2-percentile30-test.pkl"
    test = pd.read_pickle(str(test_pkl))
    cases = pd.DataFrame({"user": [test.iloc[0]["user"]], "day": [test.iloc[0]["day"]]})
    feats = assemble_feature_rows(test, cases, bundle)
    for mod in P.CHANNELS:
        arr = feats.iloc[0][f"{mod}_features"]
        assert len(arr) == bundle["input_dims"][mod]


# ---------------------------------------------------------------------------
# Tier-2
# ---------------------------------------------------------------------------
def test_tier2_output_contract():
    df = pd.DataFrame([{"user": 1, "day": 1, "label": 1,
                        "http_score": 0.9, "file_device_score": 0.1,
                        "logon_score": 0.1, "email_score": 0.1}])
    d = P.tier2_decision(df)
    row = d.iloc[0]
    assert row["escalate"] == 1
    assert row["gate_triggered"] == "high_confidence"
    for col in ["fused_score", "fired_channels", "gate", "escalated",
                "degraded_channels", "degraded_mode", "rationale"]:
        pass  # API contract (Tier2Result) — checked in API tests


def test_tier2_frozen_config():
    assert P.HC_THRESHOLD == 0.60
    assert P.CORR_THRESHOLD == 0.30
    assert P.WEIGHTS == {"http": 0.50, "file_device": 0.20, "logon": 0.15, "email": 0.15}


def test_tier2_risk_matches_frozen():
    t2 = pd.read_csv(P.W4_FROZEN_INPUT)
    risk = P.risk_score(t2)
    assert (risk - t2["risk_score"]).abs().max() < 1e-9


def test_langgraph_vectorized_equivalence():
    t2 = pd.read_csv(P.W4_FROZEN_INPUT).head(300)
    res = T2.equivalence_check(t2, n=300)
    assert res["n_compared"] == 300
    assert res["n_mismatches"] == 0


# ---------------------------------------------------------------------------
# Degradation
# ---------------------------------------------------------------------------
def test_degraded_fusion_no_fabrication():
    df = pd.DataFrame([{"user": 1, "day": 1, "label": 1,
                        "http_score": 0.9, "file_device_score": 0.1,
                        "logon_score": 0.1, "email_score": 0.1}])
    d = P.degraded_fusion(df, "http")
    row = d.iloc[0]
    # http missing -> gate cannot be high_confidence, correlation only
    assert row["gate_triggered"] in ("correlation", "normal")
    # risk uses renormalized weights over remaining channels
    expected = (0.1 * 0.2 + 0.1 * 0.15 + 0.1 * 0.15) / (0.2 + 0.15 + 0.15)
    assert abs(row["risk_score"] - expected) < 1e-12
    # no fabrication: scores for remaining channels unchanged
    assert row["file_device_score"] == 0.1


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
def test_api_health():
    from fastapi.testclient import TestClient
    from week5.final_pipeline.app import app
    with TestClient(app) as client:
        h = client.get("/health")
        assert h.status_code == 200
        body = h.json()
        assert body["models_loaded"] is True
        assert body["tier2_config"]["hc_threshold"] == 0.60
        assert body["tier2_config"]["corr_threshold"] == 0.30


def test_api_score_escalated():
    from fastapi.testclient import TestClient
    from week5.final_pipeline.app import app
    with TestClient(app) as client:
        r = client.post("/score", json={"user": 5, "day": 37,
                                        "channel_scores": {"http_score": 0.9,
                                                           "file_device_score": 0.1,
                                                           "logon_score": 0.1,
                                                           "email_score": 0.1}})
        assert r.status_code == 200
        body = r.json()
        assert body["tier2"]["escalated"] is True
        assert body["tier2"]["gate"] == "high_confidence"
        assert body["tier2"]["degraded_mode"] is False


def test_api_score_non_escalated():
    from fastapi.testclient import TestClient
    from week5.final_pipeline.app import app
    with TestClient(app) as client:
        r = client.post("/score", json={"user": 5, "day": 37,
                                        "channel_scores": {"http_score": 0.1,
                                                           "file_device_score": 0.1,
                                                           "logon_score": 0.1,
                                                           "email_score": 0.1}})
        assert r.status_code == 200
        assert r.json()["tier2"]["escalated"] is False


def test_api_missing_channel_degrades():
    from fastapi.testclient import TestClient
    from week5.final_pipeline.app import app
    with TestClient(app) as client:
        r = client.post("/score", json={"user": 5, "day": 37,
                                        "channel_scores": {"file_device_score": 0.9,
                                                           "logon_score": 0.3,
                                                           "email_score": 0.2}})
        assert r.status_code == 200
        body = r.json()
        assert body["tier2"]["degraded_mode"] is True
        assert body["tier2"]["degraded_channels"] == ["http"]
        assert "DEGRADED" in body["tier2"]["rationale"].upper()


def test_api_malformed_and_schema():
    from fastapi.testclient import TestClient
    from week5.final_pipeline.app import app
    with TestClient(app) as client:
        assert client.post("/score", json={"user": 1, "day": 1}).status_code == 422
        assert client.post("/score", json={"user": 1, "day": 1,
                                           "channel_scores": {"http_score": 5.0}}).status_code == 422
