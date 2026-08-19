"""Export/copy the frozen Tier-1 artifacts into week5 (reproducibility aid).

The FINAL Week-5 pipeline reads the Tier-1 models directly from tier1/ (the
single source of truth) — nothing is retrained. This module optionally copies
the four model+scaler+meta bundles into week5/final_pipeline/saved_models/ so
the Week-5 outputs are self-contained and the exact artifacts used are
recorded. The copy is byte-identical; it is NOT a retrain.
"""
from __future__ import annotations

import shutil
from pathlib import Path

from .pipeline import CHANNELS, TIER1_DIR, W5_FINAL

DEST = W5_FINAL / "saved_models"


def export_models(overwrite: bool = False) -> list[Path]:
    """Copy tier1/<mod>/{meta.json,model.pt,scaler.joblib} -> saved_models/.

    Returns the list of copied files. Idempotent unless overwrite=True.
    """
    DEST.mkdir(parents=True, exist_ok=True)
    copied = []
    for mod in CHANNELS:
        src = TIER1_DIR / mod
        for fname in ("meta.json", "model.pt", "scaler.joblib"):
            s = src / fname
            d = DEST / f"{mod}_{fname}"
            if d.exists() and not overwrite:
                continue
            shutil.copy2(s, d)
            copied.append(d)
    return copied


if __name__ == "__main__":
    files = export_models(overwrite=False)
    print(f"Exported {len(files)} Tier-1 artifact files -> {DEST}")
    for f in files:
        print("  ", f.name, f.stat().st_size)
