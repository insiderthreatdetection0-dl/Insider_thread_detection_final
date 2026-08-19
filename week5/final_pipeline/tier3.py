"""Tier-3 — canonical Week-4 LLM reasoning (final Week-5 implementation).

Re-uses the FROZEN Week-4 Tier-3 modules (evidence_builder.py,
grounding_validator.py, llm_triage_final.py) unchanged via re-exports in
week5/final_pipeline/evidence_builder.py and grounding_validator.py.

Tier 3 reasons ONLY about already-escalated cases (the Tier-2 decision is the
escalation authority). It MUST NOT introduce a new risk classifier, a manual
risk->severity/action classifier, or post-LLM severity/action overrides —
Qwen's severity and recommended action are preserved subject only to the
Week-4 schema/grounding contract.

The frozen Week-4 Tier-3 outputs (712 calls, 0 retries, 0 fallbacks) are the
scientific evaluation population and are NOT re-run for evaluation; the live
pipeline demonstrates Tier-3 execution on a sample request.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from .evidence_builder import EvidenceBuilder, feature_cols_by_modality
from .grounding_validator import GroundingValidator
from .pipeline import (DATA_DIR, W4_GROUNDING_RESULTS, W4_TRIAGE_OUTPUTS,
                       WEEK4_FINAL, LLM_MODEL)
from .llm_triage import LLMTriageFinal

PERCENTILE_PATH = DATA_DIR / "dayr4.2-percentile30.pkl"


def load_frozen_triage() -> pd.DataFrame:
    """Load and validate the frozen Week-4 Tier-3 outputs (712 cases)."""
    df = pd.read_csv(W4_TRIAGE_OUTPUTS)
    keys = df.groupby(["user", "day"]).size().reset_index(name="n")
    assert len(df) == 712, f"expected 712 frozen triage rows, got {len(df)}"
    assert (keys["n"] == 1).all(), "duplicate (user, day) in frozen triage outputs"
    assert df["user"].notna().all() and df["day"].notna().all(), "missing case ids"
    return df


def load_frozen_grounding() -> pd.DataFrame:
    """Load the frozen Week-4 grounding results (summary metrics)."""
    return pd.read_csv(W4_GROUNDING_RESULTS)


def make_evidence_builder(top_k: int = 3) -> EvidenceBuilder:
    """Canonical Week-4 EvidenceBuilder over the frozen percentile data."""
    cols = pd.read_pickle(str(PERCENTILE_PATH)).columns.tolist()
    fcm = feature_cols_by_modality(cols)
    return EvidenceBuilder(PERCENTILE_PATH, fcm, top_k=top_k)


def frozen_alignment_check(tier2_escalated: pd.DataFrame) -> dict:
    """Verify the 712 frozen Tier-3 cases align with the frozen Tier-2 escalations.

    Returns a dict with counts and any missing/extra keys.
    """
    triage = load_frozen_triage()
    esc_keys = set(zip(tier2_escalated["user"].astype(int),
                       tier2_escalated["day"].astype(int)))
    tr_keys = set(zip(triage["user"].astype(int), triage["day"].astype(int)))
    return {
        "frozen_triage_rows": len(triage),
        "frozen_escalated": len(esc_keys),
        "missing_in_triage": sorted(esc_keys - tr_keys),
        "extra_in_triage": sorted(tr_keys - esc_keys),
        "aligned": esc_keys == tr_keys,
    }


class Tier3Pipeline:
    """Live Tier-3 wrapper: evidence building -> LLM triage -> grounding validation.

    Only runs on escalated cases. Uses the canonical Week-4 LLMTriageFinal
    (Ollama Qwen2.5-1.5B-Instruct, temperature 0, structured JSON) with the
    deterministic evidence-grounded fallback when Ollama is unavailable.
    """

    def __init__(self, model: str = LLM_MODEL):
        self.evidence_builder = None  # lazy (needs percentile data)
        self.validator = GroundingValidator()
        self.triage = LLMTriageFinal(model=model)
        self.llm_available = self.triage.available

    def _ensure_builder(self):
        if self.evidence_builder is None:
            self.evidence_builder = make_evidence_builder(top_k=3)

    def run(self, row: pd.Series) -> dict:
        """Full Tier-3 for one escalated Tier-2 row.

        Returns the rich result dict (evidence, triage, validation, raw output,
        severity/action preserved from Qwen, fallback flags).
        """
        self._ensure_builder()
        ev = self.evidence_builder.build_evidence(row)
        res = self.triage.triage(ev)
        out = res["triage"]
        val = self.validator.validate(ev, out.explanation, out.severity_category,
                                      out.recommended_action, out.evidence_used)
        return {
            "evidence": ev,
            "triage": out,
            "validation": val,
            "raw_llm_output": res["raw_llm_output"],
            "parsed_successfully": res["parsed_successfully"],
            "fallback_used": res["fallback_used"],
            "severity_category": out.severity_category,
            "recommended_action": out.recommended_action,
            "llm_available": self.llm_available,
        }
