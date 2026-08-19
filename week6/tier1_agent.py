"""Week-6 Tier-1 microservice agent (deployment layer only).

Thin HTTP wrapper around the EXISTING frozen Tier-1 inference implementation
(`week5.final_pipeline.pipeline.ModalityFFNN` + `score_mlp`). One container per
modality; modality selected via the MODALITY env var:

    MODALITY=http|file_device|logon|email

Guarantees:
  - loads ONLY its own frozen artifacts (tier1/<mod>/{model.pt, scaler.joblib,
    meta.json}), SHA-256-verified at startup (refuses to serve on mismatch);
  - feature ordering comes from the frozen meta.json feature_cols — the request
    must supply exactly those feature names; the agent orders them canonically;
  - scoring reuses `pipeline.score_mlp` byte-for-byte (same scaler, same
    weights, same sigmoid output activation); no re-interpretation of features.

Endpoints: GET /health, POST /score.

The app is built by `create_app(modality)` with ALL state held in a closure so
that several independent agents can coexist in one process (equivalence tests)
while the container deployment simply runs one agent per MODALITY env var.
"""
from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager

import joblib
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from week5.final_pipeline import pipeline as P
from week6.artifact_integrity import verify_modality


def _load_single_modality(mod: str) -> dict:
    """Load ONE modality's frozen artifacts into a score_mlp-compatible bundle.

    Uses the exact same loading semantics as pipeline.load_tier1 (same class,
    same torch.load flags, same eval mode) without loading the other agents.
    """
    meta = json.loads((P.TIER1_DIR / mod / "meta.json").read_text())
    scaler = joblib.load(P.TIER1_DIR / mod / "scaler.joblib")
    model = P.ModalityFFNN(input_dim=meta["input_dim"], hidden_dim=meta["hidden_dim"])
    model.load_state_dict(torch.load(P.TIER1_DIR / mod / "model.pt",
                                     map_location="cpu", weights_only=True))
    model.to(P.DEVICE).eval()
    return {"models": {mod: model}, "scalers": {mod: scaler},
            "feature_cols": {mod: list(meta["feature_cols"])},
            "input_dims": {mod: meta["input_dim"]}, "meta": {mod: meta}}


class AgentScoreRequest(BaseModel):
    user: int = Field(ge=0)
    day: int = Field(ge=0)
    # {feature_name: value} — names must EXACTLY match the frozen meta.json
    # feature_cols; the agent applies the canonical model feature order itself.
    features: dict[str, float]


class AgentScoreResponse(BaseModel):
    user: int
    day: int
    modality: str
    input_dim: int
    score: float


def create_app(modality: str) -> FastAPI:
    """Build an independent Tier-1 agent app for ONE modality.

    Every piece of runtime state (`state`, the loaded bundle, the modality
    string) is captured in this closure — NOT held in module globals — so
    multiple agents built from this factory never interfere with each other.
    """
    if modality not in P.CHANNELS:
        raise RuntimeError(
            f"MODALITY must be one of {P.CHANNELS}, got {modality!r}")

    state: dict = {}

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        integrity = verify_modality(modality)
        if not integrity["ok"]:
            for r in integrity["results"]:
                if r["status"] != "ok":
                    print(f"[{modality}-agent] ARTIFACT "
                          f"{r['status'].upper()}: {r['path']}")
            raise RuntimeError(
                f"Frozen-artifact integrity FAILED for {modality}. "
                "STOP: do not regenerate artifacts — report the mismatch.")
        state["bundle"] = _load_single_modality(modality)
        state["artifacts_verified"] = True
        print(f"[{modality}-agent] ready | artifacts verified: {integrity['checked']}")
        yield
        state.clear()

    app = FastAPI(title=f"Insider Threat Tier-1 Agent — {modality}",
                  version="6.1.0", lifespan=lifespan)

    @app.get("/health")
    def health():
        bundle = state.get("bundle")
        return {
            "status": "ok",
            "service": f"{modality}-agent",
            "modality": modality,
            "model_loaded": bundle is not None,
            "input_dim": bundle["input_dims"][modality] if bundle else None,
            "artifacts_verified": bool(state.get("artifacts_verified")),
        }

    @app.post("/score", response_model=AgentScoreResponse)
    def score(req: AgentScoreRequest):
        bundle = state["bundle"]
        cols = bundle["feature_cols"][modality]
        got = set(req.features)
        want = set(cols)
        if got != want:
            missing, extra = sorted(want - got), sorted(got - want)
            raise HTTPException(
                status_code=422,
                detail=(f"feature set mismatch for {modality}: "
                        f"missing={missing[:8]}{'…' if len(missing) > 8 else ''} "
                        f"extra={extra[:8]}{'…' if len(extra) > 8 else ''}"))
        # Canonical frozen feature order — never the caller's order.
        x = np.asarray([req.features[c] for c in cols], dtype=np.float64).reshape(1, -1)
        s = float(P.score_mlp(bundle, modality, x)[0])
        return AgentScoreResponse(user=req.user, day=req.day, modality=modality,
                                  input_dim=len(cols), score=s)

    return app


# Container entrypoint target (`uvicorn week6.tier1_agent:app`): one modality
# per process, selected via MODALITY. Guarded so importing the module without
# a MODALITY (e.g. in tests that use create_app directly) never fails.
_ENV_MODALITY = os.environ.get("MODALITY", "").strip()
app = create_app(_ENV_MODALITY) if _ENV_MODALITY in P.CHANNELS else None
