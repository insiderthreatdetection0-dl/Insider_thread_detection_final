"""Week-6 frozen-artifact integrity verification (deployment layer).

Verifies the SHA-256 of every frozen research artifact the Week-6 deployment
serves. Read-only: NEVER regenerates or modifies an artifact. On mismatch or
missing file the deployment must STOP (fail-fast at API startup) and report —
per the Week-6 approval: "Do not regenerate the artifact to make the check
pass."

Expected hashes come from:
  - week4/final_llm_reasoning/outputs/week4_freeze_manifest.json (D1-A re-freeze)
  - Phase-4 closure verification (SHAP CSV == MLflow artifact copy; tier1/ exports)

Override the artifact root via env var W6_ARTIFACT_ROOT (Docker mounts).
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

from week5.final_pipeline import pipeline as P

ARTIFACT_ROOT = Path(os.environ.get("W6_ARTIFACT_ROOT", str(P.PROJECT_ROOT)))

# Re-frozen Week-4 outputs + frozen Tier-2 input (week4_freeze_manifest.json).
WEEK4_OUTPUTS = {
    "week4/final_llm_reasoning/outputs/llm_triage_outputs_final.csv":
        "26f5c4c6d6616d04197d21072cd7ad179031a966df32ffc79d550844490a113d",
    "week4/final_llm_reasoning/outputs/grounding_results_final.csv":
        "6d3e93b06361be68f20975edd7d711297df613527b8611d5595b942e06ccbe14",
    "week4/final_llm_reasoning/outputs/scenario_results.csv":
        "bb1362088b95a73d92647dc075a193c11fe02b8fd7d0a5eac3d006312ed42912",
    "week4/final_llm_reasoning/outputs/final_week4_summary.json":
        "4fbe156641468be65116d2029ded3f977f4bb79281366074d9179eb55a3237d2",
    "week4/frozen_inputs/improved_tier2_test_predictions.csv":
        "9511923b41847e29ed85e4064827ec58253b9e526cc4e756a88d9dedbce55c26",
}

# Frozen Week-5 SHAP output (byte-identical to its MLflow artifact copy).
WEEK5_OUTPUTS = {
    "week5/final_pipeline/outputs/shap_attributions.csv":
        "a9f47db0f56afec0fcd95b53fd9ec7cc7912a9d9a8ed443d497e542fa984571b",
}

# Frozen Tier-1 exports (model.pt / scaler.joblib / meta.json per modality).
TIER1_ARTIFACTS = {
    "tier1/http/meta.json": "ad8bacdc110b680f514417d419a6a97990cf5f7b3143205af8b3b9363b3315dd",
    "tier1/http/model.pt": "1a4ea5b3b1d01754065569e0929699d04ff34e1e84ef6a6d199106b3a8202490",
    "tier1/http/scaler.joblib": "6c251a05590c26ef638391a482c895de2f4647f01f655206fbc19624f6a8b20f",
    "tier1/file_device/meta.json": "918d8bf8cd99d352599334cb3e275df21670f9b97c6813549507d4c722b27528",
    "tier1/file_device/model.pt": "ddfaee8a38e99aa50a48e3af512b169b8fc5f6917073d4e245dede8f9489eaa6",
    "tier1/file_device/scaler.joblib": "44d28112f96157ed5535f1b8b523dc5b88600f13a05616230c32441d04174e55",
    "tier1/logon/meta.json": "505868b2d123b4dcdb90e6af3a95bc30ca05ffe2d9c4a94bad9e6a74c3e4d1c1",
    "tier1/logon/model.pt": "39837d41463cbd370a623dbbe540468665307369d076e10e8034830402ac8ce1",
    "tier1/logon/scaler.joblib": "bf4087de11947860082c7c1dfa38107e15fb71579127b6ba94e9c94ef3d41568",
    "tier1/email/meta.json": "75452f6701c554e26b25faad818b8816092e31f56cc71fec274aede9aaa605f4",
    "tier1/email/model.pt": "17c98bd7641baf7916d024b05ce90a740b027f0e8d4222b1baeb68f832b5236b",
    "tier1/email/scaler.joblib": "f1bcc97f3e74d3d7608a2823e1625685141ffb935cbd89a36c01c97238f0d36f",
}

# Percentile-30 test split (frozen; provenance per docs/DATA_PROVENANCE.md —
# percentile conversion of the ENTIRE dataset before the 60/20/20 user split).
DATA_ARTIFACTS = {
    "data/user_split_formlp/dayr4.2-percentile30-test.pkl":
        "cb8484647f75d181200c0e7a3738c41fff36cd896f88cbb636eda02ca2b009e6",
}

# Presence-only checks (large files; hash verification would be too costly at
# startup — their consumers re-derive results that are hash-checked above).
PRESENCE_ONLY = [
    "data/dayr4.2-percentile30.pkl",
    "week4/final_llm_reasoning/evidence_builder.py",
    "week4/final_llm_reasoning/grounding_validator.py",
]

EXPECTED_HASHES: dict[str, str] = {}
EXPECTED_HASHES.update(WEEK4_OUTPUTS)
EXPECTED_HASHES.update(WEEK5_OUTPUTS)
EXPECTED_HASHES.update(TIER1_ARTIFACTS)
EXPECTED_HASHES.update(DATA_ARTIFACTS)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_modality(modality: str, root: Path | None = None) -> dict:
    """Verify only ONE Tier-1 modality's frozen artifacts (microservice agents).

    Each agent container verifies exactly its own {model.pt, scaler.joblib,
    meta.json}. Same fail-fast contract as verify_artifacts().
    """
    root = Path(root) if root else ARTIFACT_ROOT
    subset = {rel: exp for rel, exp in TIER1_ARTIFACTS.items()
              if rel.startswith(f"tier1/{modality}/")}
    results = []
    for rel, expected in subset.items():
        p = root / rel
        if not p.exists():
            results.append({"path": rel, "status": "missing",
                            "expected": expected, "actual": None})
            continue
        actual = sha256_file(p)
        results.append({"path": rel,
                        "status": "ok" if actual == expected else "hash_mismatch",
                        "expected": expected, "actual": actual})
    ok = bool(results) and all(r["status"] == "ok" for r in results)
    return {"ok": ok, "checked": len(results), "results": results}


def verify_artifacts(root: Path | None = None, exclude: set[str] = frozenset()) -> dict:
    """Verify every expected artifact. Returns:

    {"ok": bool, "checked": int, "results": [
        {"path", "status": "ok"|"missing"|"hash_mismatch"|"present",
         "expected", "actual"}]}

    exclude: optional set of path suffixes to skip (microservice orchestrator
    skips 'model.pt'/'scaler.joblib' — those are verified by the Tier-1 agent
    containers that own them; the orchestrator still verifies the meta.json
    feature-ordering contracts).

    NEVER modifies anything. On any failure the caller must STOP and report.
    """
    root = Path(root) if root else ARTIFACT_ROOT
    results = []
    for rel, expected in EXPECTED_HASHES.items():
        if any(rel.endswith(sfx) for sfx in exclude):
            continue
        p = root / rel
        if not p.exists():
            results.append({"path": rel, "status": "missing",
                            "expected": expected, "actual": None})
            continue
        actual = sha256_file(p)
        results.append({"path": rel,
                        "status": "ok" if actual == expected else "hash_mismatch",
                        "expected": expected, "actual": actual})
    for rel in PRESENCE_ONLY:
        p = root / rel
        results.append({"path": rel,
                        "status": "present" if p.exists() else "missing",
                        "expected": None, "actual": None})
    ok = all(r["status"] in ("ok", "present") for r in results)
    return {"ok": ok, "checked": len(results), "results": results}


def main() -> int:
    """CLI entrypoint (used by the Docker entrypoint + local checks)."""
    res = verify_artifacts()
    bad = [r for r in res["results"] if r["status"] not in ("ok", "present")]
    print(f"[artifact-integrity] root={ARTIFACT_ROOT}")
    print(f"[artifact-integrity] {res['checked'] - len(bad)}/{res['checked']} artifacts verified")
    for r in bad:
        print(f"[artifact-integrity] {r['status'].upper()}: {r['path']}"
              + (f" expected={r['expected']} actual={r['actual']}" if r["expected"] else ""))
    if not res["ok"]:
        print("[artifact-integrity] FAILED — STOP. Do NOT regenerate artifacts; "
              "report the mismatch.")
        return 1
    print("[artifact-integrity] PASS — all frozen artifacts verified read-only")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
