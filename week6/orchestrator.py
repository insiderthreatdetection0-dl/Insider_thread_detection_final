"""Week-6 orchestrator API (deployment layer).

Serves the public API (GET /health, GET /health/agents, POST /score,
POST /case, GET /case/{case_id}, GET /cases) over the FROZEN research
pipeline. Two explicit deployment modes (env var DEPLOYMENT_MODE):

  single (DEFAULT — reliable local/single-container demo)
      Dashboard -> FastAPI -> IN-PROCESS Tier-1 scoring -> frozen Tier-2
      gating -> frozen Tier-3 lookups -> response. The frozen Tier-1
      model/scaler artifacts are loaded directly in THIS process (SHA-256
      verified at startup); ports 8001-8004 are never used, so a dead
      auxiliary process can never break the dashboard.

  microservice (Docker Compose six-service architecture)
      Dashboard -> orchestrator-api -> {http, file-device, logon, email}
      -agent over the internal Docker network, with explicit connect/read
      timeouts and a small BOUNDED retry policy. This architecture is
      unchanged and stays supported.

Everything else is shared and frozen:
  - Tier-2 = pipeline.tier2_decision (HC 0.60 / corr 0.30 / weights
    50-20-15-15) — unchanged;
  - evidence = canonical Week-4 EvidenceBuilder; SHAP + LLM explanations are
    frozen lookups (no live SHAP, no live LLM generation);
  - plain-language summary = deterministic presentation transform of the same
    frozen evidence (week6.explain). No new facts, no LLM.

Health is deployment-mode aware: it reports degraded/unhealthy (HTTP 503)
whenever /case could not actually perform scoring — in single mode that means
models/scalers/artifacts/Tier-2 config/frozen lookups/case data; in
microservice mode additionally the reachability of all four Tier-1 agents.

Failure handling: an unreachable or misbehaving Tier-1 agent yields an
explicit structured 503 "dependency failure" — a score is NEVER fabricated
(and never substituted with 0.0). No endpoint ever leaks a bare
"Internal Server Error": unexpected exceptions are logged WITH their full
traceback and returned as structured JSON
({"detail": "Case scoring failed", "error_type": ...}).
"""
from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager

import httpx
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from week5.final_pipeline import pipeline as P
from week5.final_pipeline import tier2 as T2
from week5.final_pipeline.app import _build_rationale
from week5.final_pipeline.schemas import (CaseListResponse, CaseRequest,
                                          CaseResponse, HealthResponse,
                                          ScoreRequest, ScoreResponse,
                                          Tier2Result)
from week6.artifact_integrity import verify_artifacts
from week6.case_service import CaseNotFound, get_case_service
from week6.explain import build_plain_explanation

logger = logging.getLogger("week6.orchestrator")

# Agent endpoint registry. Docker Compose always supplies *_AGENT_URL
# explicitly (compose service names, internal ports); the fallback below is
# for local/manual MICROSERVICE execution where every agent listens on the
# same host. DEPLOYMENT_MODE=single never touches these ports.
_AGENT_SERVICE = {"http": "http-agent", "file_device": "file-device-agent",
                  "logon": "logon-agent", "email": "email-agent"}
_AGENT_PORT = {"http": 8001, "file_device": 8002, "logon": 8003, "email": 8004}

DEPLOYMENT_MODES = ("single", "microservice")


def resolve_deployment_mode() -> str:
    """DEPLOYMENT_MODE=single|microservice.

    Default is the reliable single-container mode (in-process Tier-1). The
    microservice mode is selected explicitly (Docker Compose sets it) or
    auto-detected when *_AGENT_URL endpoints are injected.
    """
    raw = os.environ.get("DEPLOYMENT_MODE", "").strip().lower()
    if raw in DEPLOYMENT_MODES:
        return raw
    if raw:
        logger.warning("[orchestrator] unknown DEPLOYMENT_MODE=%r — "
                       "falling back to 'single'", raw)
        return "single"
    if any(os.environ.get(f"{m.upper()}_AGENT_URL") for m in _AGENT_SERVICE):
        return "microservice"
    return "single"


def agent_url(modality: str) -> str:
    env = os.environ.get(f"{modality.upper()}_AGENT_URL")
    if env:
        return env.rstrip("/")
    # Local microservice default: the documented localhost agent ports
    # (http 8001 / file_device 8002 / logon 8003 / email 8004).
    return f"http://127.0.0.1:{_AGENT_PORT[modality]}"


class AgentUnavailable(RuntimeError):
    def __init__(self, modality: str, reason: str):
        super().__init__(f"{_AGENT_SERVICE[modality]} unavailable: {reason}")
        self.modality = modality


class CaseScoringFailure(RuntimeError):
    """Expected model/data/service failure on the /case path (structured)."""

    def __init__(self, error_type: str, message: str, status_code: int = 500):
        super().__init__(message)
        self.error_type = error_type
        self.status_code = status_code


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except ValueError:
        logger.warning("[orchestrator] invalid %s=%r — using %s",
                       name, os.environ.get(name), default)
        return default


STATE: dict = {}


def _warm_or_die(steps: dict) -> dict:
    """Run startup loaders one by one; on failure log the FULL traceback,
    name the failing component, and refuse to serve (fail-fast contract)."""
    out = {}
    for name, fn in steps.items():
        try:
            out[name] = fn()
        except Exception:
            logger.exception("[orchestrator] startup dependency FAILED: %s", name)
            raise RuntimeError(
                f"Required dependency '{name}' failed to load at startup — "
                "refusing to serve. See the server log for the traceback.")
    return out


@asynccontextmanager
async def lifespan(app: FastAPI):
    mode = resolve_deployment_mode()
    STATE["mode"] = mode
    svc = get_case_service()
    STATE["case_service"] = svc
    own_client = None  # HTTP client created by THIS lifespan (closed below)

    if mode == "single":
        # In-process Tier-1 OWNS the model/scaler artifacts here — verify
        # ALL frozen artifacts (no exclusions) before serving anything.
        integrity = verify_artifacts()
        STATE["artifacts_verified"] = integrity["ok"]
        if not integrity["ok"]:
            bad = [r for r in integrity["results"] if r["status"] not in ("ok", "present")]
            for r in bad:
                logger.error("[orchestrator] ARTIFACT %s: %s",
                             r["status"].upper(), r["path"])
            raise RuntimeError(
                f"Frozen-artifact integrity FAILED ({len(bad)} problem(s)). "
                "STOP: do not regenerate artifacts — report the mismatch.")
        # Eagerly load EVERYTHING the /case path needs so that no failure is
        # deferred to the first dashboard click (fail-fast at startup).
        STATE["warm"] = _warm_or_die({"case_artifacts": svc.warm})
        print(f"[orchestrator] DEPLOYMENT_MODE=single | frozen-artifact "
              f"integrity PASS ({integrity['checked']} artifacts) | "
              f"Tier-1 models loaded in-process: "
              f"{sorted(STATE['warm']['case_artifacts']['tier1_models'])}")
    else:
        # Microservice orchestrator: model weights/scalers live in the agent
        # containers and are SHA-256-verified THERE; here we verify the rest
        # (data + frozen outputs + meta.json feature-ordering contracts).
        integrity = verify_artifacts(exclude={"model.pt", "scaler.joblib"})
        STATE["artifacts_verified"] = integrity["ok"]
        if not integrity["ok"]:
            bad = [r for r in integrity["results"] if r["status"] not in ("ok", "present")]
            for r in bad:
                logger.error("[orchestrator] ARTIFACT %s: %s",
                             r["status"].upper(), r["path"])
            raise RuntimeError(
                f"Frozen-artifact integrity FAILED ({len(bad)} problem(s)). "
                "STOP: do not regenerate artifacts — report the mismatch.")
        STATE["warm"] = _warm_or_die({
            "test_split": lambda: int(len(svc.test_df())),
            "meta_contracts": lambda: sorted(svc.meta_bundle()["feature_cols"]),
            "shap_lookup": lambda: len(svc.shap_index()),
            "llm_lookup": lambda: len(svc.llm_index()),
            "frozen_tier2": lambda: len(svc.frozen_t2_index()),
            "evidence_builder": lambda: svc.evidence_builder().__class__.__name__,
        })
        # Explicit connect/read timeouts + bounded retries (env-overridable).
        STATE["timeout"] = httpx.Timeout(
            connect=_float_env("W6_AGENT_CONNECT_TIMEOUT", 5.0),
            read=_float_env("W6_AGENT_READ_TIMEOUT", 30.0),
            write=10.0, pool=5.0)
        STATE["max_attempts"] = max(
            1, int(_float_env("W6_AGENT_MAX_RETRIES", 2)) + 1)
        STATE["health_timeout"] = _float_env("W6_AGENT_HEALTH_TIMEOUT", 3.0)
        # ONE shared client: pooled keep-alive connections to the agents,
        # created/closed with the app lifecycle (no per-request churn/leaks).
        own_client = httpx.Client(
            timeout=STATE["timeout"],
            limits=httpx.Limits(max_connections=16, max_keepalive_connections=8),
            # agent calls are ALWAYS intra-stack — ambient HTTP(S)_PROXY env
            # vars (corporate proxies) must never intercept them.
            trust_env=False)
        STATE["http_client"] = own_client
        print(f"[orchestrator] DEPLOYMENT_MODE=microservice | frozen-artifact "
              f"integrity PASS ({integrity['checked']} artifacts) | agent "
              f"timeout connect={STATE['timeout'].connect}s "
              f"read={STATE['timeout'].read}s attempts={STATE['max_attempts']}")
    yield
    # Shutdown: close ONLY the HTTP client created by THIS lifespan. STATE
    # itself is never cleared (a later shutdown of an overlapping session
    # must not wipe state another live session still serves from).
    if own_client is not None:
        own_client.close()


app = FastAPI(title="Insider Threat Detection — Week-6 Orchestrator",
              description="Deployment layer over the frozen research pipeline. "
                          "DEPLOYMENT_MODE=single: in-process Tier-1 scoring "
                          "(reliable local/single-container demo). "
                          "DEPLOYMENT_MODE=microservice: Docker Compose "
                          "orchestrator-api + 4 Tier-1 agents + dashboard.",
              version="6.2.0", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Structured error reporting — never a bare "Internal Server Error"
# ---------------------------------------------------------------------------
@app.exception_handler(AgentUnavailable)
async def _agent_unavailable_handler(_, exc: AgentUnavailable):
    logger.error("[orchestrator] Tier-1 dependency failure: %s", exc)
    return JSONResponse(
        status_code=503,
        content={"detail": f"{exc} — dependency failure; no score fabricated.",
                 "error_type": "tier1_agent_unavailable",
                 "modality": exc.modality})


@app.exception_handler(CaseScoringFailure)
async def _case_scoring_failure_handler(_, exc: CaseScoringFailure):
    logger.error("[orchestrator] case scoring failed (%s): %s",
                 exc.error_type, exc)
    return JSONResponse(status_code=exc.status_code,
                        content={"detail": str(exc),
                                 "error_type": exc.error_type})


@app.exception_handler(Exception)
async def _unexpected_error_handler(_, exc: Exception):
    # Full traceback stays in the SERVER LOG; the client gets a structured,
    # actionable JSON body instead of an opaque "Internal Server Error".
    logger.exception("[orchestrator] unhandled error on request path: %s", exc)
    return JSONResponse(status_code=500,
                        content={"detail": "Case scoring failed",
                                 "error_type": type(exc).__name__,
                                 "message": str(exc)[:300]})


# ---------------------------------------------------------------------------
# Tier-1 scoring: in-process (single) or delegated over HTTP (microservice)
# ---------------------------------------------------------------------------
# Transient transport problems that are SAFE to retry (scoring is a pure,
# idempotent function of the frozen artifacts). Read errors / HTTP statuses
# are never retried indefinitely: the attempt count is bounded.
_RETRYABLE = (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout,
              httpx.PoolTimeout, httpx.RemoteProtocolError)


def _http_scorer(user: int, day: int):
    """Per-request Tier-1 scorer routing inference to the agent containers
    (microservice mode only). Bounded retries + explicit timeouts."""
    def scorer(modality: str, features: dict[str, float]) -> float:
        url = agent_url(modality)
        client = STATE.get("http_client")
        payload = {"user": user, "day": day, "features": features}
        attempts = STATE.get("max_attempts", 3)
        for attempt in range(1, attempts + 1):
            try:
                if client is not None:
                    r = client.post(f"{url}/score", json=payload)
                else:  # defensive fallback: identical explicit timeout
                    r = httpx.post(f"{url}/score", json=payload,
                                   timeout=httpx.Timeout(connect=5.0, read=30.0,
                                                         write=10.0, pool=5.0),
                                   trust_env=False)
            except _RETRYABLE as e:
                if attempt < attempts:
                    logger.warning(
                        "[orchestrator] %s-agent attempt %d/%d failed "
                        "(%s: %s) — retrying", _AGENT_SERVICE[modality],
                        attempt, attempts, type(e).__name__, e)
                    time.sleep(min(0.25 * attempt, 1.0))
                    continue
                raise AgentUnavailable(modality,
                                       f"{type(e).__name__}: {e}") from e
            except httpx.HTTPError as e:
                raise AgentUnavailable(modality,
                                       f"{type(e).__name__}: {e}") from e
            if r.status_code != 200:
                raise AgentUnavailable(modality,
                                       f"HTTP {r.status_code}: {r.text[:120]}")
            try:
                score = float(r.json()["score"])
            except (ValueError, KeyError, TypeError) as e:
                raise AgentUnavailable(
                    modality, f"invalid /score response: {r.text[:120]!r}") from e
            if not np.isfinite(score) or not (0.0 <= score <= 1.0):
                raise AgentUnavailable(modality,
                                       f"score out of [0,1] range: {score!r}")
            return score
        raise AgentUnavailable(modality, "retry budget exhausted")  # unreachable
    return scorer


def _local_scorer(user: int, day: int):
    """In-process Tier-1 scorer (single mode). Byte-identical to the agent
    /score path: same frozen weights/scaler, same canonical meta.json feature
    order, same pipeline.score_mlp computation."""
    def scorer(modality: str, features: dict[str, float]) -> float:
        bundle = STATE["case_service"].bundle()
        cols = bundle["feature_cols"][modality]
        got, want = set(features), set(cols)
        if got != want:
            raise CaseScoringFailure(
                "feature_contract",
                f"feature set mismatch for {modality}: "
                f"missing={sorted(want - got)[:8]} extra={sorted(got - want)[:8]}",
                status_code=422)
        x = np.asarray([features[c] for c in cols],
                       dtype=np.float64).reshape(1, -1)
        return float(P.score_mlp(bundle, modality, x)[0])
    return scorer


def _agent_health() -> dict:
    out = {}
    for mod in P.CHANNELS:
        url = agent_url(mod)
        try:
            r = httpx.get(f"{url}/health",
                          timeout=STATE.get("health_timeout", 3.0),
                          trust_env=False)
            body = r.json() if r.status_code == 200 else {}
            out[_AGENT_SERVICE[mod]] = {
                "reachable": r.status_code == 200,
                "model_loaded": body.get("model_loaded"),
                "artifacts_verified": body.get("artifacts_verified"),
            }
        except httpx.HTTPError:
            out[_AGENT_SERVICE[mod]] = {"reachable": False,
                                        "model_loaded": None,
                                        "artifacts_verified": None}
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    """Deployment-mode aware health. Reports degraded (HTTP 503) whenever
    /case could not actually perform scoring — never a false 'ok'."""
    mode = STATE.get("mode") or resolve_deployment_mode()
    svc = STATE.get("case_service")
    warm = STATE.get("warm") or {}
    tier2_config = {"hc_threshold": P.HC_THRESHOLD,
                    "corr_threshold": P.CORR_THRESHOLD,
                    "weights": P.WEIGHTS}
    tier2_cfg_ok = abs(sum(P.WEIGHTS.values()) - 1.0) < 1e-9

    if mode == "single":
        bundle = svc._bundle if svc is not None else None
        models_ok = (bundle is not None
                     and set(bundle["models"]) == set(P.CHANNELS)
                     and all(m is not None for m in bundle["models"].values()))
        scalers_ok = (bundle is not None
                      and set(bundle["scalers"]) == set(P.CHANNELS))
        ca = warm.get("case_artifacts") or {}
        data_ok = int(ca.get("test_rows", 0)) > 0
        lookups_ok = (int(ca.get("llm_cases", 0)) >= 712
                      and int(ca.get("shap_cases", 0)) > 0
                      and int(ca.get("frozen_tier2_cases", 0)) >= 712)
        checks = {
            "tier1_models": {"loaded": models_ok, "modalities": P.CHANNELS},
            "tier1_scalers": {"loaded": scalers_ok},
            "artifact_sha256": {"verified": bool(STATE.get("artifacts_verified"))},
            "tier2_config": {"loaded": tier2_cfg_ok, **tier2_config},
            "tier3_frozen_lookup": {
                "loaded": lookups_ok,
                "llm_cases": int(ca.get("llm_cases", 0)),
                "shap_cases": int(ca.get("shap_cases", 0)),
                "frozen_tier2_cases": int(ca.get("frozen_tier2_cases", 0)),
            },
            "case_data": {"loaded": data_ok,
                          "test_rows": int(ca.get("test_rows", 0))},
        }
        ok = all([models_ok, scalers_ok, data_ok, lookups_ok, tier2_cfg_ok,
                  bool(STATE.get("artifacts_verified"))])
        agents = None
        models_loaded = models_ok and scalers_ok
    else:
        agents = _agent_health()
        agents_ok = bool(agents) and all(a["reachable"] for a in agents.values())
        models_loaded = (agents_ok
                         and all(a.get("model_loaded") for a in agents.values()))
        lookups_ok = (int(warm.get("llm_lookup", 0)) >= 712
                      and int(warm.get("shap_lookup", 0)) > 0
                      and int(warm.get("frozen_tier2", 0)) >= 712)
        data_ok = int(warm.get("test_split", 0)) > 0
        checks = {
            "tier1_agents": {"reachable": agents_ok},
            "artifact_sha256": {"verified": bool(STATE.get("artifacts_verified"))},
            "tier2_config": {"loaded": tier2_cfg_ok, **tier2_config},
            "tier3_frozen_lookup": {
                "loaded": lookups_ok,
                "llm_cases": int(warm.get("llm_lookup", 0)),
                "shap_cases": int(warm.get("shap_lookup", 0)),
                "frozen_tier2_cases": int(warm.get("frozen_tier2", 0)),
            },
            "case_data": {"loaded": data_ok,
                          "test_rows": int(warm.get("test_split", 0))},
        }
        ok = all([agents_ok, lookups_ok, data_ok, tier2_cfg_ok,
                  bool(STATE.get("artifacts_verified"))])

    body = HealthResponse(
        status="ok" if ok else "degraded",
        models_loaded=bool(models_loaded),
        tier1_modalities=P.CHANNELS,
        ollama_available=False,   # deployment constraint: no live LLM generation
        llm_model=P.LLM_MODEL,
        tier2_config=tier2_config,
        artifacts_verified=bool(STATE.get("artifacts_verified")),
        deployment_mode=mode,
        checks=checks,
        agents=agents,
    ).model_dump()
    return JSONResponse(status_code=200 if ok else 503, content=body)


@app.get("/health/agents")
def health_agents():
    """Tier-1 dependency reachability (diagnostics)."""
    if STATE.get("mode") == "single":
        return {"status": "not_applicable",
                "deployment_mode": "single",
                "note": ("Tier-1 scoring runs in-process; ports 8001-8004 "
                         "are not used in DEPLOYMENT_MODE=single."),
                "all_agents_reachable": True,
                "agents": {}}
    agents = _agent_health()
    return {"status": "ok",
            "all_agents_reachable": all(a["reachable"] for a in agents.values()),
            "agents": agents}


def _scores_from_features_local(req: ScoreRequest) -> dict:
    """Features path of POST /score in single mode (in-process inference)."""
    bundle = STATE["case_service"].bundle()
    feats = {"http": req.features.http, "file_device": req.features.file_device,
             "logon": req.features.logon, "email": req.features.email}
    scores = {}
    for mod, ch in feats.items():
        if ch is None:
            continue
        cols = bundle["feature_cols"][mod]
        if len(ch.values) != len(cols):
            raise HTTPException(
                status_code=422,
                detail=(f"{mod} features length {len(ch.values)} != frozen "
                        f"feature_cols length {len(cols)}"))
        scores[f"{mod}_score"] = float(P.score_mlp(
            bundle, mod, np.asarray(ch.values, dtype=np.float64).reshape(1, -1))[0])
    return scores


def _scores_from_features_via_agents(req: ScoreRequest) -> dict:
    """Features path of POST /score, delegated to the Tier-1 agents."""
    svc = STATE["case_service"]
    cols_by_mod = svc.meta_bundle()["feature_cols"]
    scorer = _http_scorer(req.user, req.day)
    feats = {"http": req.features.http, "file_device": req.features.file_device,
             "logon": req.features.logon, "email": req.features.email}
    scores = {}
    for mod, ch in feats.items():
        if ch is None:
            continue
        cols = cols_by_mod[mod]
        if len(ch.values) != len(cols):
            raise HTTPException(
                status_code=422,
                detail=(f"{mod} features length {len(ch.values)} != frozen "
                        f"feature_cols length {len(cols)}"))
        scores[f"{mod}_score"] = scorer(
            mod, {name: float(v) for name, v in zip(cols, ch.values)})
    return scores


@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest):
    """Existing public /score contract. Features are scored in-process
    (single) or by the Tier-1 agents (microservice); precomputed
    channel_scores go straight to the frozen Tier-2."""
    if req.include_triage:
        raise HTTPException(
            status_code=400,
            detail="Live LLM triage is not available in the Week-6 deployment "
                   "(frozen outputs only). Use /case for frozen explanations.")
    t0 = time.perf_counter()
    if req.features is not None and any(v is not None for v in
                                        [req.features.http, req.features.file_device,
                                         req.features.logon, req.features.email]):
        if STATE.get("mode") == "single":
            scores = _scores_from_features_local(req)
        else:
            scores = _scores_from_features_via_agents(req)
    elif req.channel_scores is not None:
        s = req.channel_scores
        scores = {}
        if s.http_score is not None: scores["http_score"] = s.http_score
        if s.file_device_score is not None: scores["file_device_score"] = s.file_device_score
        if s.logon_score is not None: scores["logon_score"] = s.logon_score
        if s.email_score is not None: scores["email_score"] = s.email_score
    else:
        raise HTTPException(status_code=422, detail="Provide features or channel_scores")

    if not scores:
        raise HTTPException(status_code=422, detail="No channel scores available")

    # EXISTING frozen Tier-2 logic (unchanged): sequential gating + fusion.
    degraded = [c for c in P.CHANNELS if f"{c}_score" not in scores]
    row = {"user": float(req.user), "day": float(req.day),
           "week": float(req.week if req.week is not None else 0.0)}
    for c in P.CHANNELS:
        row[f"{c}_score"] = scores.get(f"{c}_score", 0.0)
    ddf = P.tier2_decision(pd.DataFrame([row]),
                           degraded_channels=degraded if degraded else None)
    r = ddf.iloc[0]
    latency = (time.perf_counter() - t0) * 1000.0

    gate = str(r["gate_triggered"])
    fired = T2._fired_channels({f"{c}_score": scores.get(f"{c}_score", 0.0)
                                for c in P.CHANNELS})
    tier2 = Tier2Result(
        fused_score=float(r["risk_score"]),
        escalated=bool(r["escalate"]),
        fired_channels=fired,
        gate=gate,
        degraded_channels=degraded,
        degraded_mode=bool(degraded),
        rationale=_build_rationale(gate, degraded, float(r["risk_score"])),
    )
    return ScoreResponse(user=req.user, day=req.day, week=req.week, tier2=tier2,
                         latency_ms_tier1_tier2=round(latency, 3))


def _case_payload(user: int, day: int) -> CaseResponse:
    svc = STATE.get("case_service")
    if svc is None:
        raise CaseScoringFailure(
            "service_not_initialized",
            "Case service not initialized (server starting or shutting down).",
            status_code=503)
    try:
        if STATE.get("mode") == "single":
            # In-process Tier-1 over the frozen bundle — no ports 8001-8004.
            data = svc.build_case(user, day)
        else:
            data = svc.build_case(user, day, tier1_scorer=_http_scorer(user, day))
    except CaseNotFound:
        raise HTTPException(
            status_code=404,
            detail=(f"Case (user={user}, day={day}) is not part of the frozen "
                    "held-out test split served by this deployment."))
    # AgentUnavailable / CaseScoringFailure propagate to their structured
    # handlers; anything unexpected is caught by the global handler (which
    # logs the full traceback) — never a bare "Internal Server Error".
    # Presentation-only: plain-language summary from the SAME frozen evidence.
    data["plain_explanation"] = build_plain_explanation(data)
    try:
        return CaseResponse(**data)
    except ValidationError as e:
        raise CaseScoringFailure(
            "response_validation",
            f"Case payload failed schema validation: {e.errors()[:3]}",
            status_code=500)


@app.post("/case", response_model=CaseResponse)
def case(req: CaseRequest):
    """Full case view: Tier-1 (in-process or agents, per DEPLOYMENT_MODE) ->
    frozen Tier-2 -> frozen evidence, SHAP and LLM lookups + plain-language
    summary."""
    return _case_payload(req.user, req.day)


@app.get("/case/{case_id}", response_model=CaseResponse)
def case_by_id(case_id: str):
    parts = case_id.split("-")
    if len(parts) != 2 or not all(p.isdigit() for p in parts):
        raise HTTPException(status_code=422,
                            detail="case_id must be '<user>-<day>', e.g. '247-259'")
    return _case_payload(int(parts[0]), int(parts[1]))


@app.get("/cases", response_model=CaseListResponse)
def cases():
    """The 712 frozen escalated cases (dashboard selector)."""
    svc = STATE.get("case_service")
    if svc is None:
        raise CaseScoringFailure(
            "service_not_initialized",
            "Case service not initialized (server starting or shutting down).",
            status_code=503)
    items = svc.escalated_cases()
    return CaseListResponse(
        n_cases=len(items),
        source="week4/final_llm_reasoning/outputs/llm_triage_outputs_final.csv (re-frozen W4-final)",
        cases=items,
    )
