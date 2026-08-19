"""Week-6 STREAMLIT DASHBOARD request-path tests.

Regression coverage for the dashboard half of the recurring "API error 500"
incident:

  - the EXACT request path the dashboard uses (httpx calls to /health,
    /cases and /case — the same calls made by week6/dashboard/app.py) works
    against a REAL uvicorn FastAPI server in DEPLOYMENT_MODE=single;
  - repeated + idle-delayed "Load case" requests stay HTTP 200;
  - the real dashboard script (week6/dashboard/app.py) renders end-to-end
    via streamlit.testing.v1.AppTest with the API running, and shows NO
    "API error ..." message;
  - with the API down, the dashboard degrades to a clear "unreachable"
    message instead of crashing.

All tests are read-only over the frozen artifacts.
"""
from __future__ import annotations

import socket
import threading
import time

import httpx
import pytest
import uvicorn
from _pytest.monkeypatch import MonkeyPatch

import week6.orchestrator as orch
from week5.final_pipeline import pipeline as P

DEMO_USER, DEMO_DAY = 247, 259      # frozen escalated demo case
NORMAL_USER, NORMAL_DAY = 5, 37     # non-escalated test-split case
FROZEN_RISK_247_259 = 0.8109975874889642

DASHBOARD_PATH = P.PROJECT_ROOT / "week6" / "dashboard" / "app.py"
# The dashboard HTTP_KW: explicit timeout + no ambient proxy interception.
DASH_KW = {"timeout": httpx.Timeout(120.0), "trust_env": False}


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _wait_healthy(base: str, deadline: float = 90.0) -> None:
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < deadline:
        try:
            if httpx.get(f"{base}/health", timeout=2,
                         trust_env=False).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.3)
    raise AssertionError(f"orchestrator never became healthy at {base}")


@pytest.fixture(scope="module")
def live_single_server():
    """Real uvicorn server running week6.orchestrator:app in single mode."""
    mp = MonkeyPatch()
    mp.setenv("DEPLOYMENT_MODE", "single")
    port = _free_port()
    srv = uvicorn.Server(uvicorn.Config(orch.app, host="127.0.0.1", port=port,
                                        log_level="error"))
    threading.Thread(target=srv.run, daemon=True).start()
    base = f"http://127.0.0.1:{port}"
    try:
        _wait_healthy(base)
        yield base
    finally:
        srv.should_exit = True
        mp.undo()


# ---------------------------------------------------------------------------
# The exact dashboard request path against a real server (single mode)
# ---------------------------------------------------------------------------
def test_dashboard_api_path_health_cases_case(live_single_server):
    """api_health() / api_cases() / api_case() — exactly as the dashboard
    implements them — must succeed against the live single-mode server."""
    base = live_single_server
    # GET /health (sidebar status)
    r = httpx.get(f"{base}/health", **DASH_KW)
    assert r.status_code == 200
    h = r.json()
    assert h["status"] == "ok" and h["deployment_mode"] == "single"
    assert h["artifacts_verified"] is True
    # GET /cases (escalated case selector)
    r = httpx.get(f"{base}/cases", **DASH_KW)
    assert r.status_code == 200
    cases = r.json()["cases"]
    assert len(cases) == 712
    assert (DEMO_USER, DEMO_DAY) in {(c["user"], c["day"]) for c in cases}
    # POST /case ("Load case")
    r = httpx.post(f"{base}/case", json={"user": DEMO_USER, "day": DEMO_DAY},
                   **DASH_KW)
    assert r.status_code == 200, r.text[:300]
    c = r.json()
    assert c["tier2"]["escalate"] is True
    assert abs(c["tier2"]["risk_score"] - FROZEN_RISK_247_259) < 1e-6
    assert c["plain_explanation"]["conclusion"]
    # unknown case -> structured 404 with detail, never a 500
    r = httpx.post(f"{base}/case", json={"user": 999999, "day": 1}, **DASH_KW)
    assert r.status_code == 404 and "detail" in r.json()


def test_dashboard_load_case_repeated_and_after_delay(live_single_server):
    """"Load case" 20 times + after idle delays — all HTTP 200, identical
    payload every time (the recurring-500 'works then fails' condition)."""
    base = live_single_server
    payload = {"user": DEMO_USER, "day": DEMO_DAY}
    first = None
    for _ in range(20):
        r = httpx.post(f"{base}/case", json=payload, **DASH_KW)
        assert r.status_code == 200, r.text[:300]
        first = first or r.json()
        assert r.json() == first
    for _ in range(3):
        time.sleep(2.0)
        r = httpx.post(f"{base}/case", json=payload, **DASH_KW)
        assert r.status_code == 200, r.text[:300]
        assert r.json() == first
    assert httpx.get(f"{base}/health", **DASH_KW).json()["status"] == "ok"


def test_dashboard_load_10_distinct_cases(live_single_server):
    """The 'load at least 10 different cases' validation, through the same
    request path the dashboard uses."""
    base = live_single_server
    r = httpx.get(f"{base}/cases", **DASH_KW)
    cases = r.json()["cases"]
    picked = [(c["user"], c["day"]) for c in cases[::70]][:10]
    assert len(picked) == 10
    for user, day in picked:
        r = httpx.post(f"{base}/case", json={"user": user, "day": day},
                       **DASH_KW)
        assert r.status_code == 200, (user, day, r.text[:200])
        assert r.json()["tier2"]["escalate"] is True


# ---------------------------------------------------------------------------
# Real dashboard script via streamlit.testing.v1.AppTest
# ---------------------------------------------------------------------------
def _apptest(monkeypatch, api_url: str):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("W6_API_URL", api_url)
    at = AppTest.from_file(str(DASHBOARD_PATH), default_timeout=180)
    at.run()
    return at


def test_streamlit_dashboard_renders_with_api(live_single_server, monkeypatch):
    """The real week6/dashboard/app.py renders end-to-end with the API up:
    no exceptions, no 'API error ...', the demo case is shown with its
    frozen risk."""
    at = _apptest(monkeypatch, live_single_server)
    assert not at.exception, at.exception
    errors = [e.value for e in at.error]
    assert not any("API error" in str(v) for v in errors), errors
    assert not any("unreachable" in str(v) for v in errors), errors
    # the demo case is preselected (247/259) and fully rendered
    assert any("user 247" in str(s.value) and "day 259" in str(s.value)
               for s in at.subheader)
    assert any("0.8110" in str(m.value) for m in at.metric)
    # clicking "Load case" again (Streamlit rerun) still succeeds
    at.button[0].click().run()
    assert not at.exception, at.exception
    assert not any("API error" in str(e.value) for e in at.error)


def test_streamlit_dashboard_degrades_when_api_down(monkeypatch):
    """API down -> clear 'unreachable' message, NOT a crash and NOT the old
    'API error 500: Internal Server Error'."""
    port = _free_port()  # nothing listens here
    at = _apptest(monkeypatch, f"http://127.0.0.1:{port}")
    assert not at.exception, at.exception
    errors = [str(e.value) for e in at.error]
    assert any("unreachable" in v for v in errors), errors
    assert not any("API error" in v for v in errors)
