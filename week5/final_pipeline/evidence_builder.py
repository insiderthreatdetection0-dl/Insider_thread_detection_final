"""Final Week-5 — re-export of the Week-4 final evidence builder (read-only)."""
from __future__ import annotations

import sys
from pathlib import Path

_W4 = Path(__file__).resolve().parent.parent.parent / "week4" / "final_llm_reasoning"
if str(_W4) not in sys.path:
    sys.path.insert(0, str(_W4))

from evidence_builder import EvidenceBuilder, feature_cols_by_modality, evidence_id  # noqa: E402,F401
