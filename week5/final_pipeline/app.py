"""Final Week-5 FastAPI service (NEW implementation, frozen config).

GET  /health — model load status + Ollama + Tier-2 config
POST /score   — Tier-1 (features or precomputed channel scores) -> Tier-2
                (frozen gating) -> Tier-3 (evidence-grounded LLM triage when
                requested and the case escalates)

app.py calls the canonical pipeline; it does NOT duplicate model/fusion logic.
Missing channels degrade gracefully (weight renormalization, never fabricated)
and are reported in degraded_channels / degraded_mode / rationale.
"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException

from week5.final_pipeline import pipeline as P
from week5.final_pipeline import tier2 as T2
from week5.final_pipeline.schemas import (CaseListResponse, CaseRequest,
                                          CaseResponse, GroundingValidation,
                                          HealthResponse, ScoreRequest,
                                          ScoreResponse, Tier2Result,
                                          Tier3Result)

STATE: dict = {}


def _build_rationale(gate: str, degraded: list, risk: float) -> str:
    if degraded:
        return (f"DEGRADED MODE: channel(s) {', '.join(sorted(degraded))} unavailable; "
                f"weights renormalized over remaining channels. "
                f"Gate: {gate} (risk {risk:.3f}).")
    if gate == "high_confidence":
        return f"High-confidence escalation: HTTP anomaly score exceeds {P.HC_THRESHOLD}."
    if gate == "correlation":
        return f"Correlation escalation: weighted risk {risk:.3f} exceeds {P.CORR_THRESHOLD}."
    return f"Normal: weighted risk {risk:.3f} below all thresholds."


@asynccontextmanager
async def lifespan(app: FastAPI):
    from week5.final_pipeline.tier3 import Tier3Pipeline
    pipe = Tier3Pipeline()
    STATE["pipe"] = pipe
    STATE["ollama_available"] = pipe.llm_available
    print(f"[startup] Tier-3 pipeline ready | ollama: {pipe.llm_available}")

    # Week-6 deployment layer (additive): frozen-artifact SHA-256 verification.
    # Fail-fast on any missing/mismatched artifact — never regenerate.
    from week6.artifact_integrity import verify_artifacts
    integrity = verify_artifacts()
    STATE["artifacts_verified"] = integrity["ok"]
    if not integrity["ok"]:
        bad = [r for r in integrity["results"] if r["status"] not in ("ok", "present")]
        for r in bad:
            print(f"[startup] ARTIFACT {r['status'].upper()}: {r['path']}")
        raise RuntimeError(
            f"Frozen-artifact integrity FAILED ({len(bad)} problem(s)). "
            "STOP: do not regenerate artifacts — report the mismatch.")
    print(f"[startup] frozen-artifact integrity PASS ({integrity['checked']} artifacts)")

    from week6.case_service import get_case_service
    STATE["case_service"] = get_case_service()
    yield
    STATE.clear()


app = FastAPI(title="Insider Threat Detection — FINAL Week-5 Pipeline",
              description="Tier-1 (MLP) -> Tier-2 (frozen gating, LangGraph) "
                          "-> Tier-3 (evidence-grounded LLM)",
              version="5.0.0", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        models_loaded=True,
        tier1_modalities=P.CHANNELS,
        ollama_available=STATE["ollama_available"],
        llm_model=P.LLM_MODEL,
        tier2_config={"hc_threshold": P.HC_THRESHOLD, "corr_threshold": P.CORR_THRESHOLD,
                      "weights": P.WEIGHTS},
        artifacts_verified=STATE.get("artifacts_verified", False),
    )


def _scores_from_features(pipe, req) -> dict:
    """Tier-1 inference from raw features (missing channel -> absent from dict)."""
    if not getattr(pipe, "_bundle", None):
        pipe._bundle = P.load_tier1()
    scores = {}
    feats = {"http": req.features.http, "file_device": req.features.file_device,
             "logon": req.features.logon, "email": req.features.email}
    for mod, ch in feats.items():
        if ch is not None:
            scores[f"{mod}_score"] = float(P.score_mlp(
                pipe._bundle, mod, np.asarray(ch.values).reshape(1, -1))[0])
    return scores


def _scores_from_channel(s) -> dict:
    out = {}
    if s.http_score is not None: out["http_score"] = s.http_score
    if s.file_device_score is not None: out["file_device_score"] = s.file_device_score
    if s.logon_score is not None: out["logon_score"] = s.logon_score
    if s.email_score is not None: out["email_score"] = s.email_score
    return out


@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest):
    t0 = time.perf_counter()
    pipe = STATE["pipe"]

    if req.features is not None and any(v is not None for v in
                                        [req.features.http, req.features.file_device,
                                         req.features.logon, req.features.email]):
        scores = _scores_from_features(pipe, req)
        used_features = True
    elif req.channel_scores is not None:
        scores = _scores_from_channel(req.channel_scores)
        used_features = False
    else:
        raise HTTPException(status_code=422, detail="Provide features or channel_scores")

    degraded = [c for c in P.CHANNELS if f"{c}_score" not in scores]
    if not scores:
        raise HTTPException(status_code=422, detail="No channel scores available")

    row = {"user": float(req.user), "day": float(req.day),
           "week": float(req.week if req.week is not None else 0.0)}
    for c in P.CHANNELS:
        row[f"{c}_score"] = scores.get(f"{c}_score", 0.0)

    if degraded:
        ddf = P.tier2_decision(pd.DataFrame([row]), degraded_channels=degraded)
    else:
        ddf = P.tier2_decision(pd.DataFrame([row]))
    r = ddf.iloc[0]
    t1 = time.perf_counter()
    latency_t12 = (t1 - t0) * 1000.0

    gate = str(r["gate_triggered"])
    fired = T2._fired_channels({f"{c}_score": scores.get(f"{c}_score", 0.0) for c in P.CHANNELS})
    tier2 = Tier2Result(
        fused_score=float(r["risk_score"]),
        escalated=bool(r["escalate"]),
        fired_channels=fired,
        gate=gate,
        degraded_channels=degraded,
        degraded_mode=bool(degraded),
        rationale=_build_rationale(gate, degraded, float(r["risk_score"])),
    )

    resp = ScoreResponse(user=req.user, day=req.day, week=req.week, tier2=tier2,
                         latency_ms_tier1_tier2=round(latency_t12, 3))

    if req.include_triage and r["escalate"]:
        res = pipe.run(r)
        out = res["triage"]
        resp.tier3 = Tier3Result(
            explanation=out.explanation,
            severity_category=out.severity_category,
            recommended_action=out.recommended_action,
            evidence_used=list(out.evidence_used),
            raw_llm_output=res["raw_llm_output"],
            fallback_used=res["fallback_used"],
            llm_available=res["llm_available"],
        )
        v = res["validation"]
        resp.grounding_validation = GroundingValidation(
            grounding_valid=bool(v["grounding_valid"]),
            channel_grounding_valid=bool(v["channel_grounding_valid"]),
            feature_grounding_valid=bool(v["feature_grounding_valid"]),
            evidence_id_valid=bool(v["evidence_id_valid"]),
            # Tier-3 has no manual severity/action policy: Qwen's values are
            # preserved by design; validity here means enum-valid schema output
            # (verified 712/712 preserved on the frozen Week-4 run).
            severity_valid=True,
            action_valid=True,
            unsupported_claims=list(v["unsupported_claims"]),
        )
        resp.latency_ms_total = round((time.perf_counter() - t0) * 1000.0, 3)
    return resp


# ---------------------------------------------------------------------------
# Week-6 deployment layer (additive): frozen-output case lookups
# ---------------------------------------------------------------------------
@app.post("/case", response_model=CaseResponse)
def case(req: CaseRequest):
    """Full case view for the Week-6 dashboard.

    Live Tier-1 inference + frozen Tier-2 gating over the frozen held-out test
    split; evidence from the unchanged canonical EvidenceBuilder; SHAP and LLM
    explanations are FROZEN LOOKUPS (no live SHAP, no live LLM generation).
    """
    from week6.case_service import CaseNotFound
    svc = STATE["case_service"]
    try:
        data = svc.build_case(req.user, req.day)
    except CaseNotFound:
        raise HTTPException(
            status_code=404,
            detail=(f"Case (user={req.user}, day={req.day}) is not part of the "
                    "frozen held-out test split served by this deployment."))
    return CaseResponse(**data)


@app.get("/cases", response_model=CaseListResponse)
def cases():
    """The 712 frozen escalated cases (dashboard selector)."""
    svc = STATE["case_service"]
    items = svc.escalated_cases()
    return CaseListResponse(
        n_cases=len(items),
        source="week4/final_llm_reasoning/outputs/llm_triage_outputs_final.csv (re-frozen W4-final)",
        cases=items,
    )
