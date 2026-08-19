"""Final Week-5 FastAPI schemas (NEW implementation).

Response model mirrors the pipeline outputs: tier2.fused_score / escalated /
fired_channels, tier3.explanation / severity_category / recommended_action,
degraded_channels / degraded_mode, and grounding/validation status.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class ChannelScores(BaseModel):
    http_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    file_device_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    logon_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    email_score: Optional[float] = Field(None, ge=0.0, le=1.0)


class ChannelFeatures(BaseModel):
    """Raw features for one modality, aligned with tier1/<mod>/meta.json feature_cols order."""
    values: list[float]


class FeaturePayload(BaseModel):
    http: Optional[ChannelFeatures] = None
    file_device: Optional[ChannelFeatures] = None
    logon: Optional[ChannelFeatures] = None
    email: Optional[ChannelFeatures] = None


class ScoreRequest(BaseModel):
    user: int = 0
    day: int = 0
    week: Optional[float] = None
    channel_scores: Optional[ChannelScores] = None
    features: Optional[FeaturePayload] = None
    include_triage: bool = False


class Tier2Result(BaseModel):
    fused_score: float
    escalated: bool
    fired_channels: list[str]
    gate: str
    degraded_channels: list[str]
    degraded_mode: bool
    rationale: str


class Tier3Result(BaseModel):
    explanation: Optional[str] = None
    severity_category: Optional[str] = None
    recommended_action: Optional[str] = None
    evidence_used: list[str] = Field(default_factory=list)
    raw_llm_output: Optional[str] = None
    fallback_used: bool = False
    llm_available: bool = False


class GroundingValidation(BaseModel):
    grounding_valid: bool
    channel_grounding_valid: bool
    feature_grounding_valid: bool
    evidence_id_valid: bool
    severity_valid: bool
    action_valid: bool
    unsupported_claims: list[str] = Field(default_factory=list)


class ScoreResponse(BaseModel):
    user: int
    day: int
    week: Optional[float] = None
    tier2: Tier2Result
    tier3: Optional[Tier3Result] = None
    grounding_validation: Optional[GroundingValidation] = None
    latency_ms_tier1_tier2: Optional[float] = None
    latency_ms_total: Optional[float] = None


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    tier1_modalities: list[str]
    ollama_available: bool
    llm_model: str
    tier2_config: dict
    artifacts_verified: bool = False  # Week-6 additive (frozen-artifact SHA-256 check)
    # Week-6 reliability upgrade (additive): deployment-mode aware health.
    # "single" = in-process Tier-1 scoring; "microservice" = HTTP Tier-1 agents.
    deployment_mode: Optional[str] = None
    # Per-component readiness detail (models/scalers/artifacts/tier2/tier3/data).
    checks: Optional[dict] = None
    # Microservice mode only: reachability of each Tier-1 agent service.
    agents: Optional[dict] = None


# ---------------------------------------------------------------------------
# Week-6 deployment layer (additive): /case + /cases
# Frozen-output lookups + live Tier-1/Tier-2 inference (frozen config).
# No live LLM generation; SHAP/LLM served from frozen artifacts.
# ---------------------------------------------------------------------------
class CaseRequest(BaseModel):
    user: int = Field(..., ge=0)
    day: int = Field(..., ge=0)


class CaseTier2(BaseModel):
    risk_score: float
    escalate: bool
    gate: str
    weights: dict
    hc_threshold: float
    corr_threshold: float
    rationale: str


class FrozenConsistency(BaseModel):
    in_frozen_tier2: bool
    consistent: bool
    frozen_risk_score: Optional[float] = None
    risk_score_match: Optional[bool] = None
    escalation_match: Optional[bool] = None
    note: Optional[str] = None


class EvidenceItem(BaseModel):
    id: str
    channel: str
    feature: str
    human_label: str
    baseline_percentile: int
    current_percentile: int
    delta_percentile: int
    direction: str
    strength: str


class CaseEvidence(BaseModel):
    fired_channels: list[str]
    items: list[EvidenceItem]


class ShapFeature(BaseModel):
    channel: str
    feature: str
    feature_value: float
    shap_value: float
    direction: str
    rank: int


class CaseLLMExplanation(BaseModel):
    explanation: str
    severity_category: str
    recommended_action: str
    channels_mentioned: list[str] = Field(default_factory=list)
    evidence_used: list[str] = Field(default_factory=list)


class CaseResponse(BaseModel):
    user: int
    day: int
    week: Optional[int] = None
    in_frozen_test_split: bool
    # Held-out test label; display-only, hidden by default in the dashboard.
    insider_ground_truth: Optional[int] = None
    tier1_scores: dict
    fired_channels: list[str]
    tier2: CaseTier2
    frozen_consistency: FrozenConsistency
    evidence: Optional[CaseEvidence] = None
    shap_top_features: Optional[list[ShapFeature]] = None
    llm_explanation: Optional[CaseLLMExplanation] = None
    # Week-6 microservice deployment (additive, presentation-only): plain-language
    # summary deterministically composed from the same frozen evidence. Always
    # null for the single-container deployment.
    plain_explanation: Optional[dict] = None


class EscalatedCaseSummary(BaseModel):
    user: int
    day: int
    severity_category: Optional[str] = None
    recommended_action: Optional[str] = None
    risk_score: Optional[float] = None
    gate: Optional[str] = None


class CaseListResponse(BaseModel):
    n_cases: int
    source: str
    cases: list[EscalatedCaseSummary]
