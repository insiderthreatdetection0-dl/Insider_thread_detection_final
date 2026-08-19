"""Week-6 deployment tests — additive /case endpoint + frozen-artifact layer.

Covers (WEEK6_IMPLEMENTATION_PLAN.md §F + approval validation list):
  - /health ok + frozen-artifact integrity flag
  - /score behavior preserved (unchanged)
  - /case escalated (demo user 247) / non-escalated / unknown 404 / malformed 422
  - frozen artifact validation (pass + fail-fast path)
  - frozen Tier-2 consistency on >=10 sampled escalated cases
  - 712-case escalation set reproduced
  - frozen SHAP + frozen LLM lookups byte-consistent with the frozen CSVs

All tests are read-only over frozen artifacts.
"""
from __future__ import annotations

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from week5.final_pipeline import pipeline as P
from week5.final_pipeline.app import app
from week6.artifact_integrity import EXPECTED_HASHES, verify_artifacts
from week6.case_service import get_case_service

DEMO_USER, DEMO_DAY = 247, 259        # plan §F end-to-end demo case (scenario 2)
NON_ESC_USER, NON_ESC_DAY = 5, 37     # non-escalated test-split case


# ---------------------------------------------------------------------------
# Frozen artifact validation
# ---------------------------------------------------------------------------
def test_artifact_integrity_passes():
    res = verify_artifacts()
    bad = [r for r in res["results"] if r["status"] not in ("ok", "present")]
    assert res["ok"], f"artifact integrity failed: {bad}"
    assert res["checked"] >= 22  # 21 hashed + presence checks


def test_artifact_integrity_failfast_on_missing(tmp_path):
    res = verify_artifacts(root=tmp_path)
    assert res["ok"] is False
    assert all(r["status"] == "missing" for r in res["results"])


def test_health_reports_verified_artifacts():
    with TestClient(app) as client:
        body = client.get("/health").json()
        assert body["status"] == "ok"
        assert body["artifacts_verified"] is True
        # pre-Week-6 fields preserved
        assert body["models_loaded"] is True
        assert body["tier2_config"]["hc_threshold"] == 0.60


# ---------------------------------------------------------------------------
# /score preserved (behavior must be unchanged by Week 6)
# ---------------------------------------------------------------------------
def test_score_endpoint_unchanged():
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
        assert abs(body["tier2"]["fused_score"]
                   - (0.5 * 0.9 + 0.2 * 0.1 + 0.15 * 0.1 + 0.15 * 0.1)) < 1e-12


# ---------------------------------------------------------------------------
# /case — escalated demo case (plan §F.5 / end-to-end chain)
# ---------------------------------------------------------------------------
def test_case_escalated_demo_user():
    with TestClient(app) as client:
        r = client.post("/case", json={"user": DEMO_USER, "day": DEMO_DAY})
        assert r.status_code == 200
        c = r.json()
        assert c["in_frozen_test_split"] is True
        assert set(c["tier1_scores"]) == {f"{m}_score" for m in P.CHANNELS}
        assert c["tier2"]["escalate"] is True
        assert c["tier2"]["weights"] == {"http": 0.50, "file_device": 0.20,
                                         "logon": 0.15, "email": 0.15}
        # frozen Tier-2 record consistency
        assert c["frozen_consistency"]["in_frozen_tier2"] is True
        assert c["frozen_consistency"]["consistent"] is True
        # explanations come from FROZEN outputs only
        assert c["evidence"] is not None and len(c["evidence"]["items"]) > 0
        assert all(it["id"] for it in c["evidence"]["items"])
        assert c["shap_top_features"], "frozen SHAP lookup returned nothing"
        assert c["llm_explanation"], "frozen LLM lookup returned nothing"
        assert c["llm_explanation"]["severity_category"]
        assert c["llm_explanation"]["recommended_action"]


def test_case_non_escalated():
    with TestClient(app) as client:
        r = client.post("/case", json={"user": NON_ESC_USER, "day": NON_ESC_DAY})
        assert r.status_code == 200
        c = r.json()
        assert c["tier2"]["escalate"] is False
        assert c["evidence"] is None
        assert c["shap_top_features"] is None
        assert c["llm_explanation"] is None


def test_case_unknown_returns_404():
    with TestClient(app) as client:
        r = client.post("/case", json={"user": 999999, "day": 1})
        assert r.status_code == 404


def test_case_malformed_returns_422():
    with TestClient(app) as client:
        assert client.post("/case", json={"user": -5, "day": 3}).status_code == 422
        assert client.post("/case", json={"user": 5}).status_code == 422


# ---------------------------------------------------------------------------
# Frozen Tier-2 consistency on >=10 sampled escalated cases (plan §F.6)
# ---------------------------------------------------------------------------
def _tier1_tier2_live(svc, df, user, day) -> tuple[float, bool]:
    """Live Tier-1 + Tier-2 exactly as /case does, minus the evidence step."""
    from week5.final_pipeline.tier1 import assemble_feature_rows, score_cases
    row = df[(df["user"].astype(int) == user) & (df["day"].astype(int) == day)].iloc[0]
    feats = assemble_feature_rows(df, pd.DataFrame([{"user": user, "day": day}]),
                                  svc.bundle())
    scored = score_cases(svc.bundle(), feats).iloc[0]
    tier1 = {f"{c}_score": float(scored[f"{c}_score"]) for c in P.CHANNELS}
    ddf = P.tier2_decision(pd.DataFrame([{"user": float(user), "day": float(day),
                                          "week": float(row.get("week", 0)), **tier1}]))
    return float(ddf.iloc[0]["risk_score"]), bool(ddf.iloc[0]["escalate"])


def test_tier2_matches_frozen_on_sampled_cases():
    svc = get_case_service()
    df = svc.test_df()
    frozen = svc.frozen_t2_index()
    keys = sorted(frozen)[:300:29]  # deterministic spread, 11 escalated cases
    assert len(keys) >= 10
    for user, day in keys:
        risk, escalate = _tier1_tier2_live(svc, df, user, day)
        rec = frozen[(user, day)]
        # frozen CSV stores ~7 significant digits -> 1e-6 tolerance
        assert abs(risk - rec["risk_score"]) < 1e-6, (user, day)
        assert escalate == bool(rec["escalate"]), (user, day)


# ---------------------------------------------------------------------------
# 712-case escalation set reproduced (plan §F.7)
# ---------------------------------------------------------------------------
def test_escalated_case_set_is_the_frozen_712():
    with TestClient(app) as client:
        lst = client.get("/cases").json()
    assert lst["n_cases"] == 712
    api_keys = {(c["user"], c["day"]) for c in lst["cases"]}
    frozen = pd.read_csv(P.W4_FROZEN_INPUT)
    esc = frozen[frozen["escalate"].astype(int) == 1]
    assert len(esc) == 712
    assert api_keys == set(zip(esc["user"].astype(int), esc["day"].astype(int)))


# ---------------------------------------------------------------------------
# Frozen SHAP + LLM lookups byte-consistent with the frozen CSVs (plan §F.8)
# ---------------------------------------------------------------------------
def test_shap_lookup_matches_frozen_csv():
    shap = pd.read_csv(P.OUT / "shap_attributions.csv")
    sub = shap[(shap["user_id"] == DEMO_USER) & (shap["date"] == DEMO_DAY)
               & (shap["rank"] <= 5)]
    from week6.case_service import META_COLS
    sub = sub[~sub["feature"].isin(META_COLS)]
    svc = get_case_service()
    got = svc.shap_index()[(DEMO_USER, DEMO_DAY)]
    assert len(got) == len(sub)
    expected = {(r.channel, r.feature, round(float(r.shap_value), 10))
                for r in sub.itertuples()}
    actual = {(g["channel"], g["feature"], round(g["shap_value"], 10)) for g in got}
    assert actual == expected


def test_llm_lookup_matches_frozen_csv():
    llm = pd.read_csv(P.W4_TRIAGE_OUTPUTS)
    row = llm[(llm["user"] == DEMO_USER) & (llm["day"] == DEMO_DAY)]
    assert len(row) == 1
    svc = get_case_service()
    got = svc.llm_index()[(DEMO_USER, DEMO_DAY)]
    assert got["explanation"] == row.iloc[0]["explanation"]
    assert got["severity_category"] == row.iloc[0]["severity_category"]
    assert got["recommended_action"] == row.iloc[0]["recommended_action"]


# ---------------------------------------------------------------------------
# No live LLM generation anywhere in the case path
# ---------------------------------------------------------------------------
def test_case_path_performs_no_llm_calls():
    """The case service only LOOKS UP frozen outputs — it must never touch the
    live Tier-3 LLM pipeline."""
    import week6.case_service as cs
    src = open(cs.__file__, encoding="utf-8").read()
    assert "Tier3Pipeline" not in src
    assert "triage(" not in src
    assert "ollama" not in src.lower()
    # hash registry is immutable evidence of frozen serving
    # 5 Week-4 outputs + 1 SHAP CSV + 12 Tier-1 files + 1 test-split pkl
    assert len(EXPECTED_HASHES) == 19
