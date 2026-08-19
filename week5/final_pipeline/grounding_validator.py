"""Final Week-5 — re-export of the Week-4 final grounding validator (read-only).

NOTE: the FINAL Week-4 grounding_validator.py does NOT define
`deterministic_severity_action` (no manual severity/action policy exists in
Tier 3). Earlier week5 code importing it was broken reference code.
"""
from __future__ import annotations

import sys
from pathlib import Path

_W4 = Path(__file__).resolve().parent.parent.parent / "week4" / "final_llm_reasoning"
if str(_W4) not in sys.path:
    sys.path.insert(0, str(_W4))

from grounding_validator import (  # noqa: E402,F401
    GroundingValidator,
    extract_evidence_ids_from_explanation,
    canonical_key_for_feature,
    alias_tokens_for_feature,
    _mentioned_channels,
    _mentioned_features,
    _unsupported_feature_mentions,
)
