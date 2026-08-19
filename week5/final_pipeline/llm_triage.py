"""Final Week-5 — re-export of the Week-4 final LLM triage module (read-only)."""
from __future__ import annotations

import sys
from pathlib import Path

_W4 = Path(__file__).resolve().parent.parent.parent / "week4" / "final_llm_reasoning"
if str(_W4) not in sys.path:
    sys.path.insert(0, str(_W4))

from llm_triage_final import (  # noqa: E402,F401
    LLMTriageFinal,
    TriageOutput,
    Observation,
    build_prompt,
    build_canonical_inventory,
    inventory_text,
    generate_grounded_fallback,
    validate_output,
    SYSTEM_PROMPT,
)
