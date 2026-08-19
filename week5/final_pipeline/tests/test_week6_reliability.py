"""Week-6 deployment RELIABILITY tests (DEPLOYMENT_MODE + lifecycle fixes).

Regression coverage for the recurring dashboard "API error 500" incident:

  1.  /health in single-container mode (full component checks)
  2.  /case for a known frozen escalated case (single mode)
  3.  /case for a normal case (single mode)
  4.  all four Tier-1 in-process scorers
  5.  artifact integrity verification (pass)
  6.  missing/corrupt artifact handling (fail-fast detection)
  7.  unavailable HTTP agent in microservice mode (structured 503)
  8.  timeout handling (bounded, structured, no hang)
  9.  repeated /case calls (100+ cases, zero unexpected 500s)
  10. repeated calls after a delay (the "works, then fails later" repro)
  11. HTTP-agent vs in-process score/case equivalence
  12. structured errors — never a bare "Internal Server Error"
  13. path resolution independent of the current working directory

All tests are read-only over the frozen artifacts.
"""
from __future__ import annotations

import socket
import threading
import time

import httpx
import pandas as pd
import pytest
import uvicorn
from _pytest.monkeypatch import MonkeyPatch
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.responses import PlainTextResponse

from week5.final_pipeline import pipeline as P
from week5.final_pipeline.tier1 import assemble_feature_rows, score_cases
from week6.artifact_integrity import verify_artifacts, verify_modality
from week6.case_service import get_case_service

import week6.orchestrator as orch

DEMO_USER, DEMO_DAY = 247, 259      # frozen escalated demo case (scenario 2)
NORMAL_USER, NORMAL_DAY = 5, 37     # non-escalated test-split case
FROZEN_RISK_247_259 = 0.8109975874889642


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture(scope="module")
def mp_module():
    mp = MonkeyPatch()
    yield mp
    mp.undo()


@pytest.fixture(scope="module")
def single_client(mp_module):
    """Orchestrator in DEPLOYMENT_MODE=single (in-process Tier-1)."""
    mp_module.setenv("DEPLOYMENT_MODE", "single")
    with TestClient(orch.app) as client:
        yield client


# ---------------------------------------------------------------------------
# 1. /health — single-container mode (deployment-mode aware)
# ---------------------------------------------------------------------------
def test_health_single_mode_full_component_checks(single_client):
    r = single_client.get("/health")
    assert r.status_code == 200, r.text
    h = r.json()
    assert h["status"] == "ok"
    assert h["deployment_mode"] == "single"
    assert h["models_loaded"] is True
    assert h["artifacts_verified"] is True
    checks = h["checks"]
    assert checks["tier1_models"]["loaded"] is True
    assert sorted(checks["tier1_models"]["modalities"]) == sorted(P.CHANNELS)
    assert checks["tier1_scalers"]["loaded"] is True
    assert checks["artifact_sha256"]["verified"] is True
    assert checks["tier2_config"]["loaded"] is True
    assert checks["tier2_config"]["hc_threshold"] == 0.60
    assert checks["tier2_config"]["corr_threshold"] == 0.30
    assert checks["tier2_config"]["weights"] == P.WEIGHTS
    assert checks["tier3_frozen_lookup"]["loaded"] is True
    assert checks["tier3_frozen_lookup"]["llm_cases"] == 712
    assert checks["tier3_frozen_lookup"]["frozen_tier2_cases"] >= 712
    assert checks["case_data"]["loaded"] is True
    assert checks["case_data"]["test_rows"] > 0
    # single mode never depends on agent ports
    assert single_client.get("/health/agents").json()["status"] == "not_applicable"


# ---------------------------------------------------------------------------
# 2+3. /case — frozen escalated + normal case (single mode)
# ---------------------------------------------------------------------------
def test_case_escalated_frozen_single_mode(single_client):
    r = single_client.post("/case", json={"user": DEMO_USER, "day": DEMO_DAY})
    assert r.status_code == 200, r.text
    c = r.json()
    assert c["tier2"]["escalate"] is True
    assert abs(c["tier2"]["risk_score"] - FROZEN_RISK_247_259) < 1e-6
    assert c["frozen_consistency"]["in_frozen_tier2"] is True
    assert c["frozen_consistency"]["consistent"] is True
    assert c["evidence"] and len(c["evidence"]["items"]) > 0
    assert c["shap_top_features"], "frozen SHAP lookup returned nothing"
    assert c["llm_explanation"], "frozen LLM lookup returned nothing"
    assert c["plain_explanation"]["conclusion"]
    # unknown case -> structured 404 (never a 500)
    assert single_client.post("/case",
                              json={"user": 999999, "day": 1}).status_code == 404


def test_case_normal_single_mode(single_client):
    r = single_client.post("/case", json={"user": NORMAL_USER, "day": NORMAL_DAY})
    assert r.status_code == 200, r.text
    c = r.json()
    assert c["tier2"]["escalate"] is False
    assert c["evidence"] is None
    assert c["shap_top_features"] is None
    assert c["llm_explanation"] is None


# ---------------------------------------------------------------------------
# 4. all four Tier-1 in-process scorers
# ---------------------------------------------------------------------------
def test_inprocess_tier1_scorers_all_modalities(single_client):
    svc = get_case_service()
    df = svc.test_df()
    bundle = svc.bundle()
    feats = assemble_feature_rows(
        df, pd.DataFrame([{"user": DEMO_USER, "day": DEMO_DAY}]), bundle)
    ref = score_cases(bundle, feats).iloc[0]
    scorer = orch._local_scorer(DEMO_USER, DEMO_DAY)
    for mod in P.CHANNELS:
        cols = bundle["feature_cols"][mod]
        arr = feats.iloc[0][f"{mod}_features"]
        got = scorer(mod, {name: float(v) for name, v in zip(cols, arr)})
        expected = float(ref[f"{mod}_score"])
        assert 0.0 <= got <= 1.0, mod
        assert abs(got - expected) <= 1e-12, mod  # byte-identical frozen path
    # feature-contract violations are structured 4xx, never 500
    with pytest.raises(orch.CaseScoringFailure) as ei:
        scorer("logon", {"bogus": 1.0})
    assert ei.value.error_type == "feature_contract"
    assert ei.value.status_code == 422


# ---------------------------------------------------------------------------
# 5+6. artifact integrity: pass / missing / corrupt
# ---------------------------------------------------------------------------
def test_artifact_integrity_passes_full():
    res = verify_artifacts()
    assert res["ok"] is True
    assert res["checked"] >= 22


def test_artifact_integrity_missing_detected(tmp_path):
    res = verify_artifacts(root=tmp_path)
    assert res["ok"] is False
    assert all(r["status"] == "missing"
               for r in res["results"] if r["expected"] is not None)


def test_artifact_integrity_corrupt_detected(tmp_path):
    import shutil
    # stage a modality directory with a CORRUPTED meta.json
    for name in ("meta.json", "model.pt", "scaler.joblib"):
        src = P.TIER1_DIR / "http" / name
        dst = tmp_path / "tier1" / "http" / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dst)
    meta = tmp_path / "tier1" / "http" / "meta.json"
    meta.write_bytes(meta.read_bytes() + b" ")  # 1-byte corruption
    res = verify_modality("http", root=tmp_path)
    assert res["ok"] is False
    statuses = {r["path"]: r["status"] for r in res["results"]}
    assert statuses["tier1/http/meta.json"] == "hash_mismatch"
    assert statuses["tier1/http/model.pt"] == "ok"


# ---------------------------------------------------------------------------
# 7+8+12. microservice failure modes — structured, bounded, never bare 500
# ---------------------------------------------------------------------------
@pytest.fixture()
def micro_env():
    """Fresh per-test env + attribute patching for microservice mode."""
    mp = MonkeyPatch()
    mp.setenv("DEPLOYMENT_MODE", "microservice")
    mp.setenv("W6_AGENT_MAX_RETRIES", "0")
    yield mp
    mp.undo()


def test_microservice_dead_agent_structured_503_and_degraded_health(micro_env):
    micro_env.setattr(orch, "agent_url", lambda mod: "http://127.0.0.1:1")
    with TestClient(orch.app, raise_server_exceptions=False) as c:
        h = c.get("/health")
        assert h.status_code == 503
        assert h.json()["status"] == "degraded"
        assert h.json()["models_loaded"] is False
        agents = h.json()["agents"]
        assert all(a["reachable"] is False for a in agents.values())
        r = c.post("/case", json={"user": DEMO_USER, "day": DEMO_DAY})
        assert r.status_code == 503
        body = r.json()
        assert body["error_type"] == "tier1_agent_unavailable"
        assert body["modality"] == "http"
        assert "fabricat" in body["detail"]
        assert r.text != "Internal Server Error"


def test_microservice_read_timeout_is_bounded(micro_env):
    """A hanging agent must produce a structured dependency error within the
    bounded retry budget — the orchestrator never hangs indefinitely."""
    micro_env.setenv("W6_AGENT_READ_TIMEOUT", "0.3")
    micro_env.setenv("W6_AGENT_MAX_RETRIES", "1")  # 2 attempts total

    slow = FastAPI()

    @slow.post("/score")
    def slow_score():
        time.sleep(2.0)
        return {"score": 0.5}

    port = _free_port()
    srv = uvicorn.Server(uvicorn.Config(slow, host="127.0.0.1", port=port,
                                        log_level="error"))
    threading.Thread(target=srv.run, daemon=True).start()
    time.sleep(0.8)
    micro_env.setattr(orch, "agent_url", lambda mod: f"http://127.0.0.1:{port}")
    try:
        with TestClient(orch.app, raise_server_exceptions=False) as c:
            t0 = time.perf_counter()
            r = c.post("/case", json={"user": DEMO_USER, "day": DEMO_DAY})
            elapsed = time.perf_counter() - t0
        assert r.status_code == 503
        assert r.json()["error_type"] == "tier1_agent_unavailable"
        assert "ReadTimeout" in r.json()["detail"]
        # bounded: 2 attempts x 0.3s read timeout + small backoff, never 2s+
        assert elapsed < 4.0, f"orchestrator hung for {elapsed:.1f}s"
    finally:
        srv.should_exit = True


def test_microservice_malformed_agent_response_never_bare_500(micro_env):
    """ROOT-CAUSE REGRESSION: an agent answering 200 with a body that lacks
    the score (or is not JSON at all) used to leak an unhandled KeyError /
    JSONDecodeError as a bare HTTP 500 'Internal Server Error'. It must be a
    structured 503 dependency failure instead."""
    no_score = FastAPI()

    @no_score.post("/score")
    def _no_score():
        return {"status": "ok"}

    @no_score.get("/health")
    def _h():
        return {"status": "ok", "model_loaded": True, "artifacts_verified": True}

    html_body = FastAPI()

    @html_body.post("/score")
    def _html():
        return PlainTextResponse("<html>503 Service Unavailable</html>")

    @html_body.get("/health")
    def _h2():
        return {"status": "ok", "model_loaded": True, "artifacts_verified": True}

    p1, p2 = _free_port(), _free_port()
    servers = []
    for app_, port in ((no_score, p1), (html_body, p2)):
        srv = uvicorn.Server(uvicorn.Config(app_, host="127.0.0.1", port=port,
                                            log_level="error"))
        threading.Thread(target=srv.run, daemon=True).start()
        servers.append(srv)
    time.sleep(0.8)
    try:
        for port in (p1, p2):
            micro_env.setattr(orch, "agent_url",
                              lambda mod, p=port: f"http://127.0.0.1:{p}")
            with TestClient(orch.app, raise_server_exceptions=False) as c:
                r = c.post("/case", json={"user": DEMO_USER, "day": DEMO_DAY})
            assert r.status_code == 503, r.text
            assert r.json()["error_type"] == "tier1_agent_unavailable"
            assert r.text != "Internal Server Error"
    finally:
        for srv in servers:
            srv.should_exit = True


def test_unexpected_error_is_structured_not_bare_500(monkeypatch):
    """Any genuinely unexpected error is logged WITH its traceback server-side
    and returned as structured JSON — never a bare 'Internal Server Error'.

    (starlette's ServerErrorMiddleware always re-raises after invoking the
    installed handler, so the client here must opt out of re-raising.)"""
    svc = get_case_service()

    def boom(*a, **k):
        raise RuntimeError("simulated component failure")

    monkeypatch.setattr(svc, "build_case", boom)
    monkeypatch.setenv("DEPLOYMENT_MODE", "single")
    with TestClient(orch.app, raise_server_exceptions=False) as c:
        r = c.post("/case", json={"user": DEMO_USER, "day": DEMO_DAY})
    assert r.status_code == 500
    body = r.json()
    assert body["detail"] == "Case scoring failed"
    assert body["error_type"] == "RuntimeError"
    assert "simulated component failure" in body["message"]
    assert r.text != "Internal Server Error"


# ---------------------------------------------------------------------------
# 9+10. stress: repeated /case — zero unexpected 500s, deterministic
# ---------------------------------------------------------------------------
def _case_cycle() -> list[tuple[int, int]]:
    """10 deterministic cases: the demo pair + 8 frozen escalated cases."""
    svc = get_case_service()
    esc = sorted(k for k in svc.frozen_t2_index()
                 if k in svc._test_keys and k != (DEMO_USER, DEMO_DAY))
    return [(DEMO_USER, DEMO_DAY), (NORMAL_USER, NORMAL_DAY)] + esc[::len(esc) // 8][:8]


def test_stress_100_requests_zero_500_deterministic(single_client):
    svc = get_case_service()
    bundle_before = svc.bundle()
    cases = _case_cycle()
    assert len(cases) == 10
    reference, n_ok, n_500 = {}, 0, 0
    # 10 rounds x 10 cases = 100 requests
    for rnd in range(10):
        for user, day in cases:
            r = single_client.post("/case", json={"user": user, "day": day})
            if r.status_code == 500:
                n_500 += 1
            assert r.status_code == 200, (user, day, r.status_code, r.text[:200])
            n_ok += 1
            body = r.json()
            key = (user, day)
            if key not in reference:
                reference[key] = body
            else:
                assert body == reference[key], f"non-deterministic: {key}"
    assert n_ok == 100 and n_500 == 0
    # zero model reloads: the SAME bundle object served every request
    assert svc.bundle() is bundle_before
    assert single_client.get("/health").json()["status"] == "ok"


def test_stress_100_distinct_cases(single_client):
    """100 DISTINCT non-escalated test-split cases — zero unexpected errors."""
    svc = get_case_service()
    llm_keys = set(svc.llm_index())
    distinct = sorted(k for k in svc._test_keys if k not in llm_keys)[:100]
    assert len(distinct) == 100
    for user, day in distinct:
        r = single_client.post("/case", json={"user": user, "day": day})
        assert r.status_code == 200, (user, day, r.text[:200])
        assert r.json()["tier2"]["escalate"] in (True, False)


def test_repeated_calls_after_delay(single_client):
    """The 'works initially, then fails later' repro: identical requests
    separated by an idle delay must keep returning HTTP 200."""
    payload = {"user": DEMO_USER, "day": DEMO_DAY}
    first = single_client.post("/case", json=payload)
    assert first.status_code == 200
    for _ in range(3):
        time.sleep(2.0)  # idle gap between bursts
        r = single_client.post("/case", json=payload)
        assert r.status_code == 200, r.text[:200]
        assert r.json() == first.json()


# ---------------------------------------------------------------------------
# 11. HTTP-agent vs in-process equivalence (scores + full case payload)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def agent_servers():
    """Real socket servers for all four Tier-1 agents."""
    from week6.tier1_agent import create_app
    servers, urls = [], {}
    for mod in P.CHANNELS:
        port = _free_port()
        srv = uvicorn.Server(uvicorn.Config(create_app(mod), host="127.0.0.1",
                                            port=port, log_level="error"))
        threading.Thread(target=srv.run, daemon=True).start()
        for _ in range(100):
            try:
                if httpx.get(f"http://127.0.0.1:{port}/health",
                             timeout=1).status_code == 200:
                    break
            except httpx.HTTPError:
                time.sleep(0.2)
        servers.append(srv)
        urls[mod] = f"http://127.0.0.1:{port}"
    yield urls
    for srv in servers:
        srv.should_exit = True


def _case_payloads_by_mode(mp, agent_urls, cases):
    """Fetch identical /case payloads once per deployment mode."""
    out = {}
    # microservice: real HTTP Tier-1 agents
    mp.setenv("DEPLOYMENT_MODE", "microservice")
    mp.setattr(orch, "agent_url", lambda mod: agent_urls[mod])
    with TestClient(orch.app) as c:
        assert c.get("/health").json()["status"] == "ok"
        client_before = orch.STATE["http_client"]
        for user, day in cases:
            r = c.post("/case", json={"user": user, "day": day})
            assert r.status_code == 200, r.text
            out[("micro", user, day)] = r.json()
        # zero connection leaks: ONE shared pooled client for all requests
        assert orch.STATE["http_client"] is client_before
    # single: in-process Tier-1
    mp.setenv("DEPLOYMENT_MODE", "single")
    with TestClient(orch.app) as c:
        for user, day in cases:
            r = c.post("/case", json={"user": user, "day": day})
            assert r.status_code == 200, r.text
            out[("single", user, day)] = r.json()
    return out


def test_single_vs_microservice_case_equivalence(mp_module, agent_servers):
    cases = [(DEMO_USER, DEMO_DAY), (NORMAL_USER, NORMAL_DAY)] + _case_cycle()[2:6]
    payloads = _case_payloads_by_mode(mp_module, agent_servers, cases)
    for user, day in cases:
        a, b = payloads[("micro", user, day)], payloads[("single", user, day)]
        # Tier-1 scores: agent HTTP vs in-process (JSON round-trip tolerance)
        for mod in P.CHANNELS:
            assert abs(a["tier1_scores"][f"{mod}_score"]
                       - b["tier1_scores"][f"{mod}_score"]) <= 1e-9
        # fired channels / Tier-2 fused risk / escalation decision
        assert a["fired_channels"] == b["fired_channels"]
        assert abs(a["tier2"]["risk_score"] - b["tier2"]["risk_score"]) <= 1e-12
        assert a["tier2"]["escalate"] == b["tier2"]["escalate"]
        assert a["tier2"]["gate"] == b["tier2"]["gate"]
        # frozen case lookup + explanations
        assert a["frozen_consistency"] == b["frozen_consistency"]
        assert a["evidence"] == b["evidence"]
        assert a["shap_top_features"] == b["shap_top_features"]
        assert a["llm_explanation"] == b["llm_explanation"]
        assert a["plain_explanation"] == b["plain_explanation"]
    # demo case keeps the frozen scientific record in BOTH modes
    for mode in ("micro", "single"):
        c = payloads[(mode, DEMO_USER, DEMO_DAY)]
        assert abs(c["tier2"]["risk_score"] - FROZEN_RISK_247_259) < 1e-6
        assert c["frozen_consistency"]["consistent"] is True


def test_microservice_repeated_case_calls(mp_module, agent_servers):
    """Repeated /case over the real agent HTTP path stays healthy."""
    mp_module.setenv("DEPLOYMENT_MODE", "microservice")
    mp_module.setattr(orch, "agent_url", lambda mod: agent_servers[mod])
    with TestClient(orch.app) as c:
        first = None
        for i in range(15):
            r = c.post("/case", json={"user": DEMO_USER, "day": DEMO_DAY})
            assert r.status_code == 200, r.text
            if first is None:
                first = r.json()
            else:
                assert r.json() == first


# ---------------------------------------------------------------------------
# 13. paths resolve from the package location, not the shell CWD
# ---------------------------------------------------------------------------
def test_paths_and_integrity_independent_of_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # run from a completely different directory
    assert P.PROJECT_ROOT.is_absolute()
    assert (P.PROJECT_ROOT / "pyproject.toml").exists()
    assert (P.TIER1_DIR / "http" / "model.pt").exists()
    assert verify_artifacts()["ok"] is True
    svc = get_case_service()
    assert svc.has_case(DEMO_USER, DEMO_DAY)
