# Week 4 — Freeze Report

**Date:** 2026-08-11
**Status:** FROZEN
**Task:** Final freeze & documentation of the Week 4 Tier-3 LLM reasoning implementation. No implementation changes were made by this freeze task; only inspection, verification, and two documentation artifacts were created.

---

## 1. Week 4 status

**FROZEN.** The current Week 4 implementation is the final baseline and is to be treated as the reference for Week 5.

## 2. Exact final LLM / model

- **Qwen2.5-1.5B-Instruct** (`qwen2.5:1.5b-instruct`) served locally via **Ollama**.
- Temperature 0, JSON-schema-forced structured output (`format=TriageOutput.model_json_schema()`).
- Model unchanged; no model switch performed.

## 3. Exact frozen Tier-2 input

- `week4/frozen_inputs/improved_tier2_test_predictions.csv`
- 61,203 rows total; **712 escalated cases** (`escalate == 1`).
- SHA-256: `9511923b41847e29ed85e4064827ec58253b9e526cc4e756a88d9dedbce55c26`
- **Untouched** by this freeze task.

## 4. Number of escalated cases

**712** (correct; not questioned or regenerated).

## 5. Number of LLM calls

**712** (one per escalated case). **No LLM inference was re-run** by this freeze task.

## 6. Retries / fallbacks

**0 retries, 0 fallbacks.** All 712 calls succeeded on the initial attempt (`initial_call_success = 712/712`).

## 7. Structured output validation result

- **712/712** `raw_llm_output` entries are valid JSON containing `explanation`, `severity_category`, `recommended_action`.
- Enum values valid (`severity_category ∈ {critical,high,medium,low}`; `recommended_action ∈ {immediate_isolation,soc_escalation,monitor_enhanced,no_action}`).
- 0 parse failures; 0 invalid evidence IDs; 0 invalid channels; 0 invalid observations; 0 unsupported intent terms.

## 8. Ground-truth leakage result

**PASS — no leakage.** The LLM receives only behavioural/evidence information (fused score, channel scores, fired channels, gate, feature deltas, percentiles, direction, strength). The Tier-2 `label`/`insider` ground-truth is dropped by `EvidenceBuilder` and never reaches the prompt. Verified programmatically over all 712 evidence objects and sampled prompts.

## 9. Manual severity/action policy result

**None.** No risk→severity thresholds, no fired-channel→action rules, no `deterministic_severity`/`calibrate` functions exist in the active modules. Verified: 712/712 raw-vs-final severity and action match — **Qwen's severity/action are preserved as-is**.

## 10. Corrected scenario mapping status

**VERIFIED and FROZEN** (corrected in the previous audit pass, not changed again here). Official CERT r4.2 answer key mapping:
- Scenario 1 = WikiLeaks uploader (users 555/564/848 in the escalated set)
- Scenario 2 = job-hunter / job websites + thumb-drive theft (users 118/247/577/652/767/900)
- Scenario 3 = keylogger (users 11/861)

`week4/final_llm_reasoning/outputs/scenario_results.csv` (11 user rows, regenerated previously from existing outputs) is the final artifact — no LLM re-run. Job-hunter user 247: 40 escalated / 26 malicious, preserved.

## 11. Grounding-metric limitation

The 100% validated-observation / validated-evidence-ID metrics are **structural validation metrics of the deterministic evidence layer**. They must **not** be described as proof that 100% of Qwen's natural-language explanations are semantically grounded. Qwen2.5-1.5B returns only `explanation` + `severity_category` + `recommended_action` and does not populate the structured `observations`/`evidence_used` fields; the deterministic layer constructs the grounded observations from the supplied evidence inventory. This limitation is documented in `week4/final_llm_reasoning/outputs/week4_final_audit.md` and `week4_final_llm_reasoning/outputs/week4_freeze_manifest.json`.

## 12. Existing scenario-evaluation limitation

- Scenario-1 held-out test users **49, 620, 723** had malicious days that Tier-2 did **not** escalate; they are documented (not fabricated) as Tier-2 escalation misses and are not part of the 712-case input.
- The three scenario evaluations are all **held-out quantitative** (users in `test_users.npy`); no qualitative/non-held-out substitutes were needed for the Week 4 scenario requirement.
- The lightweight explanation audit (existing outputs only) is not a statistically rigorous hallucination benchmark — that is Week 5 scope.

## 13. Files created/modified by THIS freeze task

**Created:**
- `week4/final_llm_reasoning/outputs/week4_freeze_manifest.json`
- `week4/final_llm_reasoning/outputs/week4_freeze_report.md` (this report)

**Modified:** none. No implementation file, output file, scenario file, or frozen input was changed by this freeze task.

**Checked (unchanged, verified present & consistent):**
- `week4/final_llm_reasoning/outputs/llm_triage_outputs_final.csv` (712 rows; SHA-256 `26f5c4c6d6616d04197d21072cd7ad179031a966df32ffc79d550844490a113d` — current re-frozen artifact; original freeze hash `22f96a37...` superseded, see §17)
- `week4/final_llm_reasoning/outputs/final_week4_summary.json` (SHA-256 `4fbe156641468be65116d2029ded3f977f4bb79281366074d9179eb55a3237d2`)
- `week4/final_llm_reasoning/outputs/grounding_results_final.csv` (SHA-256 `6d3e93b06361be68f20975edd7d711297df613527b8611d5595b942e06ccbe14`)
- `week4/final_llm_reasoning/outputs/scenario_results.csv` (11 rows; SHA-256 `bb1362088b95a73d92647dc075a193c11fe02b8fd7d0a5eac3d006312ed42912`)
- `week4/final_llm_reasoning/outputs/week4_final_audit.md`
- `week4/final_llm_reasoning/final_llm_reasoning.ipynb` (canonical Week 4 notebook)
- Active modules `evidence_builder.py`, `grounding_validator.py`, `llm_triage_final.py`, `grounding_evaluation.py`, `scenario_evaluation.py` — all import cleanly; no deterministic severity/action policy symbols; no stray temp/broken artifacts.

## 14. No Tier-1/Tier-2 changes

**Explicitly:** no Tier-1 models, Tier-1 scores, Tier-2 models, Tier-2 fusion/escalation logic, or the frozen Tier-2 input were changed by this freeze task or the preceding audit. The 712-case input and the final LLM outputs are byte-identical to the audited baseline.

## 15. No LLM inference re-run

**Explicitly:** no LLM calls were re-run for this freeze task. All metrics above are from the existing final execution outputs.

## 16. Week 4 is now frozen

Week 4 is **FROZEN** and is the baseline for Week 5 (end-to-end quantitative evaluation, SHAP-vs-LLM comparison, FastAPI integration, graceful-degradation testing).

---

## 17. Re-freeze (D1-A) — 2026-08-11, Phase 4 audit closure

**Authorized by user decision D1-A.** The Phase 4 audit (see `PROJECT_AUDIT_REPORT.md`, finding AUD-001) established that `llm_triage_outputs_final.csv` was modified on disk after the original freeze (original SHA-256 `22f96a37...` no longer reproducible; the original bytes are not recoverable). `grounding_results_final.csv` was also modified post-freeze (it was not hash-tracked in the original manifest). `scenario_results.csv` and `final_week4_summary.json` still match the original freeze hashes exactly.

Per the approved decision: **the CURRENT on-disk Week-4 artifact is the artifact to preserve/re-freeze.**

- **No LLM re-run.** No new LLM responses were generated. No prompt or methodology change. Only the freeze manifest and this report were updated.
- **Re-frozen hashes (verified 2026-08-11):**
  - `llm_triage_outputs_final.csv` — `26f5c4c6d6616d04197d21072cd7ad179031a966df32ffc79d550844490a113d` (712 rows)
  - `grounding_results_final.csv` — `6d3e93b06361be68f20975edd7d711297df613527b8611d5595b942e06ccbe14`
  - `scenario_results.csv` — `bb1362088b95a73d92647dc075a193c11fe02b8fd7d0a5eac3d006312ed42912` (unchanged vs original freeze)
  - `final_week4_summary.json` — `4fbe156641468be65116d2029ded3f977f4bb79281366074d9179eb55a3237d2` (unchanged vs original freeze)
  - Frozen Tier-2 input — `9511923b41847e29ed85e4064827ec58253b9e526cc4e756a88d9dedbce55c26` (unchanged vs original freeze)
- **Current on-disk distributions** (recomputed from the re-frozen artifact): severity {high 666, medium 37, low 9}; actions {immediate_isolation 340, soc_escalation 326, monitor_enhanced 37, no_action 9}. The original freeze record was 341/325/37/9; the one-case shift between immediate_isolation and soc_escalation is part of the post-freeze modification documented above and is preserved (not corrected) per D1-A.
- **Two-generation clarification:** this re-frozen artifact (`llm_triage_outputs_final.csv`) is generation **W4-final**. The separate "controlled update" outputs in `week4/updated_outputs/updated_llm_outputs.csv` (different schema, includes `chk_*` citation columns) are generation **W4-updated** and are consumed by the Week-5 updated evaluation chain (`week5/final_pipeline/updated_outputs/`). The two generations are different LLM generations and must not be conflated; see `week5/OBSOLETE_OUTPUTS.md` / canonical-classification docs.
