"""Week-6 microservice tests — Tier-1 agents, orchestrator, equivalence.

MANDATORY output-equivalence coverage (task §8/§9/§10):
  - per-modality numerical equivalence: existing in-process score_mlp vs the
    agent service POST /score, on a fixed deterministic sample of frozen test
    rows (tolerance 1e-9 — same frozen weights/scaler/architecture, so the
    expectation is bit-identity; tolerance only guards JSON float round-trip);
  - Tier-2 decision equivalence: fired channels / gate / fused risk /
    escalation identical for the local vs microservice Tier-1 score sets;
  - frozen-record consistency of the orchestrator /case path.

Plus service behavior: feature-order contract, health endpoints, failure
handling (agent down -> 503, never a fabricated score), and the
plain-language explanation grounding rules.
"""
from __future__ import annotations

import socket
import threading
import time

import httpx
import numpy as np
import pandas as pd
import pytest
import uvicorn
from fastapi.testclient import TestClient

from week5.final_pipeline import pipeline as P
from week5.final_pipeline.tier1 import assemble_feature_rows, score_cases

# Documented equivalence tolerance: identical computation path, JSON round-trip.
TIER1_TOL = 1e-9
TIER2_RISK_TOL = 1e-12


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _agent_app(modality: str):
    """An INDEPENDENT Tier-1 agent app for one modality.

    Uses the tier1_agent.create_app factory (closure-captured state) so the
    four agents never share module-level state and can coexist in one process.
    """
    from week6.tier1_agent import create_app
    return create_app(modality)


@pytest.fixture(scope="module")
def local_bundle():
    return P.load_tier1()


@pytest.fixture(scope="module")
def sample_rows():
    """Fixed deterministic sample of frozen test rows (escalated + not)."""
    svc_keys = sorted(pd.read_csv(P.W4_FROZEN_INPUT)
                      .query("escalate == 1")[["user", "day"]].itertuples(index=False),
                      key=lambda r: (r.user, r.day))
    esc = [(int(u), int(d)) for u, d in svc_keys][::59][:10]
    esc.append((247, 259))  # demo case
    df = pd.read_pickle(str(P.DATA_DIR / "user_split_formlp"
                            / "dayr4.2-percentile30-test.pkl"))
    keys = set(zip(df["user"].astype(int), df["day"].astype(int)))
    non_esc = [(5, 37)] if (5, 37) in keys else []
    return df, esc + non_esc


@pytest.fixture(scope="module")
def agent_clients():
    clients = {}
    for mod in P.CHANNELS:
        c = TestClient(_agent_app(mod))
        c.__enter__()  # run lifespan: verify artifacts + load weights
        clients[mod] = c
    yield clients
    for c in clients.values():
        c.__exit__(None, None, None)


def _features_dict(df, bundle, user, day, mod) -> dict:
    feats = assemble_feature_rows(df, pd.DataFrame([{"user": user, "day": day}]),
                                  bundle)
    cols = bundle["feature_cols"][mod]
    arr = feats.iloc[0][f"{mod}_features"]
    return {name: float(v) for name, v in zip(cols, arr)}


# ---------------------------------------------------------------------------
# §8 — Tier-1 numerical equivalence (in-process vs agent service)
# ---------------------------------------------------------------------------
def test_tier1_score_equivalence(local_bundle, sample_rows, agent_clients):
    df, cases = sample_rows
    feats = assemble_feature_rows(df, pd.DataFrame(cases, columns=["user", "day"]),
                                  local_bundle)
    local_scores = score_cases(local_bundle, feats)
    max_diff = {mod: 0.0 for mod in P.CHANNELS}
    for i, (user, day) in enumerate(cases):
        for mod in P.CHANNELS:
            ref = float(local_scores.iloc[i][f"{mod}_score"])
            fd = {name: float(v) for name, v in
                  zip(local_bundle["feature_cols"][mod],
                      feats.iloc[i][f"{mod}_features"])}
            r = agent_clients[mod].post("/score",
                                        json={"user": user, "day": day, "features": fd})
            assert r.status_code == 200, r.text
            got = r.json()["score"]
            max_diff[mod] = max(max_diff[mod], abs(ref - got))
    for mod in P.CHANNELS:
        assert max_diff[mod] <= TIER1_TOL, f"{mod}: max abs diff {max_diff[mod]}"


# ---------------------------------------------------------------------------
# §9 — Tier-2 decision equivalence (local scores vs microservice scores)
# ---------------------------------------------------------------------------
def test_tier2_equivalence(local_bundle, sample_rows, agent_clients):
    df, cases = sample_rows
    feats = assemble_feature_rows(df, pd.DataFrame(cases, columns=["user", "day"]),
                                  local_bundle)
    local_scores = score_cases(local_bundle, feats)
    for i, (user, day) in enumerate(cases):
        row = df[(df["user"].astype(int) == user) & (df["day"].astype(int) == day)].iloc[0]
        ms_scores = {}
        for mod in P.CHANNELS:
            fd = {name: float(v) for name, v in
                  zip(local_bundle["feature_cols"][mod],
                      feats.iloc[i][f"{mod}_features"])}
            r = agent_clients[mod].post("/score",
                                        json={"user": user, "day": day, "features": fd})
            ms_scores[f"{mod}_score"] = r.json()["score"]
        base = {"user": float(user), "day": float(day),
                "week": float(row.get("week", 0))}
        d_local = P.tier2_decision(pd.DataFrame([{**base,
                                                  **{f"{m}_score": float(local_scores.iloc[i][f"{m}_score"])
                                                     for m in P.CHANNELS}}])).iloc[0]
        d_ms = P.tier2_decision(pd.DataFrame([{**base, **ms_scores}])).iloc[0]
        assert int(d_local["escalate"]) == int(d_ms["escalate"]), (user, day)
        assert str(d_local["gate_triggered"]) == str(d_ms["gate_triggered"]), (user, day)
        assert abs(float(d_local["risk_score"]) - float(d_ms["risk_score"])) <= TIER2_RISK_TOL


# ---------------------------------------------------------------------------
# Agent contract
# ---------------------------------------------------------------------------
def test_agent_health_and_feature_order_contract(agent_clients, local_bundle):
    for mod in P.CHANNELS:
        h = agent_clients[mod].get("/health").json()
        assert h["status"] == "ok" and h["model_loaded"] is True
        assert h["artifacts_verified"] is True
        assert h["input_dim"] == local_bundle["input_dims"][mod]
    # wrong feature set -> 422 (never silently reordered)
    r = agent_clients["logon"].post("/score",
                                    json={"user": 1, "day": 1,
                                          "features": {"bogus": 1.0}})
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# Orchestrator over REAL agent HTTP servers (routing + failure handling)
# ---------------------------------------------------------------------------
def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture(scope="module")
def agent_servers():
    servers, urls = [], {}
    for mod in P.CHANNELS:
        port = _free_port()
        cfg = uvicorn.Config(_agent_app(mod), host="127.0.0.1", port=port,
                             log_level="error")
        srv = uvicorn.Server(cfg)
        threading.Thread(target=srv.run, daemon=True).start()
        for _ in range(100):
            try:
                if httpx.get(f"http://127.0.0.1:{port}/health", timeout=1).status_code == 200:
                    break
            except httpx.HTTPError:
                time.sleep(0.2)
        servers.append(srv)
        urls[mod] = f"http://127.0.0.1:{port}"
    yield urls
    for srv in servers:
        srv.should_exit = True


@pytest.fixture(scope="module")
def orch_client(agent_servers, monkeypatch_module):
    import week6.orchestrator as orch
    # Explicit microservice mode: the orchestrator routes Tier-1 scoring to
    # the agent servers below (single mode scores in-process by default).
    monkeypatch_module.setenv("DEPLOYMENT_MODE", "microservice")
    monkeypatch_module.setattr(orch, "agent_url",
                               lambda mod: agent_servers[mod])
    with TestClient(orch.app) as client:
        yield client


@pytest.fixture(scope="module")
def monkeypatch_module():
    from _pytest.monkeypatch import MonkeyPatch
    mp = MonkeyPatch()
    yield mp
    mp.undo()


def test_orchestrator_health_and_agents(orch_client):
    h = orch_client.get("/health").json()
    assert h["status"] == "ok" and h["artifacts_verified"] is True
    assert h["tier2_config"]["weights"] == {"http": 0.50, "file_device": 0.20,
                                            "logon": 0.15, "email": 0.15}
    a = orch_client.get("/health/agents").json()
    assert a["all_agents_reachable"] is True


def test_orchestrator_case_escalated_equivalent_and_plain(orch_client):
    r = orch_client.post("/case", json={"user": 247, "day": 259})
    assert r.status_code == 200, r.text
    c = r.json()
    assert c["tier2"]["escalate"] is True
    assert c["frozen_consistency"]["consistent"] is True
    # microservice Tier-1 scores equal the frozen-record-driven local scores
    assert abs(c["tier2"]["risk_score"] - 0.8109975874889642) < 1e-6
    # plain-language explanation present, grounded, no raw feature names
    p = c["plain_explanation"]
    assert p["why_flagged"] and p["conclusion"]
    text = " ".join([p["why_flagged"], p["what_happened"], p["why_unusual"],
                     p["conclusion"]]).lower()
    for raw in ["email_mean_email_text_slen", "file_docf", "http_c_len", "afterhour"]:
        assert raw not in text
    # unknown -> 404 ; malformed case_id -> 422
    assert orch_client.post("/case", json={"user": 999999, "day": 1}).status_code == 404
    assert orch_client.get("/case/not-a-case").status_code == 422
    assert orch_client.get("/case/247-259").status_code == 200


def test_orchestrator_score_channel_path_unchanged(orch_client):
    r = orch_client.post("/score", json={"user": 5, "day": 37,
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
    # live LLM triage is not permitted in the deployment
    r2 = orch_client.post("/score", json={"user": 5, "day": 37,
                                          "channel_scores": {"http_score": 0.9},
                                          "include_triage": True})
    assert r2.status_code == 400


def test_orchestrator_agent_failure_is_explicit(orch_client, agent_servers):
    import week6.orchestrator as orch
    original = orch.agent_url

    def broken(mod):
        return "http://127.0.0.1:1" if mod == "email" else agent_servers[mod]

    orch.agent_url = broken
    try:
        r = orch_client.post("/case", json={"user": 247, "day": 259})
        assert r.status_code == 503
        assert "email" in r.json()["detail"].lower()
        assert "fabricat" in r.json()["detail"].lower()
    finally:
        orch.agent_url = original
