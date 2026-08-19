# Week 5 — Final Implementation Report

**Date:** 2026-08-11
**Status:** FINAL
**Implementation:** `week5/final_pipeline/` (new, clean, leakage-free)
**Week-1–4 source files: UNTOUCHED** (verified — no modifications)

---

## 1. Objective

Demonstrate the complete insider-threat detection pipeline on the CERT r4.2
held-out test set:

```
CERT r4.2 held-out test data
        ↓
Tier 1 — four modality agents (HTTP, File/Device, Logon, Email)
        ↓
Tier 2 — LangGraph validated fusion / escalation
        ↓
Escalated cases (712)
        ↓
Week-4 canonical EvidenceBuilder
        ↓
Tier 3 — Ollama / Qwen2.5-1.5B-Instruct
        ↓
Grounding / schema validation
        ↓
Final triage output
```

exposed through `POST /score` and `GET /health`, with:

1. end-to-end integration
2. leakage-free held-out test evaluation
3. system precision/recall/F1/AUC/FPR/PR-AUC
4. MLflow tracking
5. **ACTUAL SHAP explanations** for Tier-1 models
6. SHAP-vs-LLM explanation comparison
7. graceful degradation when a channel is unavailable
8. API/schema/smoke tests
9. deterministic/vectorized-vs-LangGraph equivalence
10. reproducible final artifacts and report

## 2. Final architecture

The Week-4 architecture is **frozen** and consumed unchanged:

- **Tier 1**: the finalized per-modality MLPs in `tier1/{http,file_device,logon,email}/`
  (`meta.json` + `model.pt` + `scaler.joblib`). Feature counts 204 / 215 / 20 / 47.
  No retraining — the pipeline loads the frozen artifacts (byte-copied to
  `saved_models/` for reproducibility).
- **Tier 2**: frozen sequential gating. **Tier 2 makes the security escalation
  decision.** Real LangGraph orchestrator (`tier2.py`) + vectorized reference
  (`pipeline.tier2_decision`) — verified identical.
- **Tier 3**: the canonical Week-4 modules (`evidence_builder.py`,
  `grounding_validator.py`, `llm_triage_final.py`) re-exported, reasoning only
  about escalated cases. **No manual Tier-3 severity/action classifier exists**
  and there are no hand-written severity/action thresholds; Qwen's
  `severity_category` / `recommended_action` are preserved as-is.

## 3. Week-4 frozen dependencies

| Artifact | Path | Verified |
|---|---|---|
| Tier-2 input (frozen) | `week4/frozen_inputs/improved_tier2_test_predictions.csv` | 61,203 rows; **712 escalated** |
| Tier-3 LLM outputs | `week4/final_llm_reasoning/outputs/llm_triage_outputs_final.csv` | **712 cases**, 0 retries, 0 fallbacks |
| Grounding results | `week4/final_llm_reasoning/outputs/grounding_results_final.csv` | 712 calls, 712/712 validated |
| Evidence builder | `week4/final_llm_reasoning/evidence_builder.py` | canonical (re-exported) |
| Grounding validator | `week4/final_llm_reasoning/grounding_validator.py` | canonical (re-exported) |
| LLM triage | `week4/final_llm_reasoning/llm_triage_final.py` | canonical (re-exported) |

**Frozen config:** HC = 0.60, correlation = 0.30, weights HTTP 0.50 /
File-Device 0.20 / Logon 0.15 / Email 0.15.

**Scenario mapping (final):** Scenario 1 = WikiLeaks uploader; Scenario 2 =
Job-hunter exfiltrator; Scenario 3 = Keylogger sysadmin.

## 4. Leakage-free evaluation protocol

- Held-out test = the 200 frozen test users (`tier1/*/meta.json test_users`),
  61,203 rows, 201 positive user-days.
- The frozen Tier-2 input and frozen Tier-3 outputs were produced from this
  test split only; Tier-1 models never trained on test users (train/val/test
  user sets are disjoint; verified `train pkl users == meta.train_users`).
- **No test-set threshold tuning, no test-set training.**
- Tier-3 receives only behavioural/evidence information (fused score, channel
  scores, fired channels, gate, feature deltas, percentiles, direction,
  strength, deterministic evidence IDs) — never raw logs, never ground-truth
  labels.
- The frozen 712-case Tier-3 population was NOT re-run; Week-5 joins it by
  `(user, day)` (not row number) — **712/712 aligned, no duplicates, no
  missing IDs** (`frozen_tier3_alignment.aligned = true`).

## 5. Tier-1 implementation

`tier1.py` loads the frozen models/scalers/feature order and scores raw
percentile feature rows. `assemble_feature_rows` attaches per-modality raw
feature vectors in the exact model feature order from the percentile
DataFrame (missing values filled with 0 as in the frozen pipeline). Verified:
feature order == `meta.feature_cols`, scaler length == input_dim, model input
dimension == input_dim for all four modalities. No retraining performed.

## 6. Tier-2 implementation

`tier2.py` builds the **real LangGraph** orchestrator (StateGraph) with the
frozen sequential gating — mirroring the Week-3 finalized notebook:

- `compute_risk_score` → weighted fusion (50/20/15/15)
- `check_high_confidence` → escalate if `http_score > 0.60`
- `check_correlation` → escalate if `risk_score > 0.30`

Each case exposes `fused_score`, `fired_channels`, `gate`, `escalated`,
`degraded_channels`, `degraded_mode`, `rationale`.

**LangGraph vs vectorized equivalence: 0 mismatches over all 61,203 rows**
(`equivalence_check`, no arbitrary rounding — risk agreement to 1e-16, gate
and escalate agreement exact).

## 7. Tier-3 integration

`tier3.py` re-uses the canonical Week-4 pipeline:

1. `EvidenceBuilder` builds the deterministic evidence inventory (top-3
   feature deltas, IDs, percentiles, direction, strength) from escalated
   cases only.
2. `LLMTriageFinal` calls Ollama `qwen2.5:1.5b-instruct` (temperature 0,
   JSON-schema-forced) — the frozen 712 outputs are reused for evaluation;
   the live API demonstrates Tier-3 on a sample escalated request.
3. `GroundingValidator.validate` performs schema/evidence/grounding
   validation. Qwen's `explanation`, `severity_category`, `recommended_action`,
   `evidence_used` are preserved; no deterministic severity/action override.

Frozen Tier-3 re-validation (Week-5, canonical validator): 576/712 strict
`grounding_valid`; the 136 failures are all the same known Week-4 structural
artifact — `evidence_used` lists only the top-1 evidence while the explanation
cites a different evidence feature; no invalid evidence IDs/channels and no
unsupported intent terms anywhere (see §12 and §16).

## 8. FastAPI service

`app.py` → `GET /health`, `POST /score`:

- `/health`: status, models_loaded, tier1_modalities, ollama_available,
  llm_model, tier2_config.
- `/score`: accepts `channel_scores` (0–1 per modality) or raw `features`
  (model feature order); runs Tier-1 (if features) → Tier-2 → Tier-3 (if
  `include_triage` and the case escalates). Response exposes `tier2.fused_
  score`, `tier2.escalated`, `tier2.fired_channels`, `tier2.gate`,
  `tier3.explanation`, `tier3.severity_category`, `tier3.recommended_action`,
  `tier3.evidence_used`, `degraded_channels`, `degraded_mode`, and
  grounding/validation status.
- Missing channels never crash the request: weights are renormalized over the
  remaining channels and the missing score is **never fabricated**; the
  response reports `degraded_channels`, `degraded_mode=true`, and a rationale
  that identifies the degradation.
- `app.py` calls the canonical pipeline; it duplicates no model/fusion logic.

## 9. System-level results

Held-out test (61,203 rows, 200 users, 201 positives), frozen Tier-2:

| Metric | Value |
|---|---|
| TP / FP / FN / TN | 178 / 534 / 23 / 60,468 |
| Recall | 0.8856 |
| Precision | 0.2500 |
| F1 | 0.3899 |
| FPR | 0.0088 |
| ROC-AUC | 0.9829 |
| PR-AUC | 0.5878 |
| Escalations | 712 (1.16 %) |

These match the reference values exactly because both derive from the frozen
Tier-2 inputs; nothing is hard-coded (the evaluator recomputes everything from
the frozen CSV). Frozen Tier-3 alignment: 712/712.

### 9.1 Comparison vs. Week-1 flat baseline (added by audit 2026-08-11)

Canonical Week-1 flat baseline (`week1/week1_random_forest_binary.ipynb`:
RandomForest n=200, class_weight=balanced, seed 42, official 60/20/20 user
split, threshold 0.05 selected on validation via Youden's J, evaluated once on
the same held-out test split):

| Metric | Week-1 flat RF | Final hierarchical system | Δ |
|---|---|---|---|
| Recall | 0.6418 | **0.8856** | +0.2438 |
| Precision | 0.1812 | **0.2500** | +0.0688 |
| F1 | 0.2826 | **0.3899** | +0.1073 |
| ROC-AUC | 0.9410 | **0.9829** | +0.0419 |
| Accuracy | 0.9900 | 0.9909 | +0.0009 |

Note (audit): older Week-3 reports quote different RF numbers (e.g. Recall
0.2153 / AUC 0.5000, or 0.1875 / 0.9639) — those come from the *legacy* RF
variants (`week1_random_forest_fixed` / `_model_only`) and are obsolete; the
table above uses the canonical binary RF notebook results.

## 10. Actual SHAP implementation

- **Package:** `shap` **0.52.0** (real SHAP — not gradient attribution).
- **Explainer:** `shap.DeepExplainer` (preference order #1 for the PyTorch
  ModalityFFNN models), explaining the **pre-sigmoid logit** via a 2D-output
  wrapper (the frozen models squeeze to `(batch,)`, which DeepExplainer
  cannot consume). The additive check is performed on the logit.
- **Background source:** training split only — `meta.train_users` from
  `dayr4.2-percentile30-train.pkl` (600 train users; never test/validation
  rows).
- **Background size:** 100 samples per modality, deterministic **seed 42**.
- **Coverage:** all 712 escalated cases × 4 modalities = 2,848 explained
  (case, modality) pairs; 346,032 attribution rows in
  `outputs/shap_attributions.csv` (case_id, user_id, date, channel, feature,
  feature_value, shap_value, abs_shap_value, direction, rank, model_output,
  expected_value, shap_method, shap_version).
- **Feature alignment:** SHAP columns are the model's canonical
  `meta.feature_cols` order — no reordering, no invented names.

### SHAP environment note (reproducibility)

The project `.venv` runs numpy 2.5.1; `shap` 0.52.0 requires `numba`, which
requires numpy ≤ 2.4 — a genuine incompatibility. Rather than downgrade the
frozen project environment (whose numpy binaries are also loaded by running
Jupyter kernels), real SHAP runs in an isolated sandbox venv `.venv-shap`
(numpy 2.4.6, numba 0.66.0, torch CPU 2.13.0). The frozen project environment
was left fully intact. Exact resolution steps are in the README.

## 11. SHAP sanity / additivity results

`outputs/shap_sanity.csv` — per (case, modality):

```
model_output ≈ expected_value + sum(SHAP)   (on the logit; tolerance 1e-3)
```

| Metric | Value |
|---|---|
| Number explained | 2,848 |
| Number passing | **2,848 (100 %)** |
| Number failing | 0 |
| Maximum residual | 1.88e-5 |
| Mean residual | 5.40e-7 |

No NaN/Inf anywhere; SHAP values are reproducible within tolerance
(deterministic background + seeds).

## 12. SHAP-vs-LLM comparison

`outputs/explanation_comparison.csv` — per case, the LLM explanation's
mentioned channels/features vs the actual SHAP top attribution, with a
**three-way claim classification** so a valid Week-4 evidence-grounded claim is
never automatically "unsupported" just because it is not in the SHAP top-k:

1. supported by SHAP/top attribution,
2. supported by canonical Week-4 evidence but not top-k SHAP,
3. unsupported claim.

Semantics: SHAP `rank` is per (case, channel), so the top-k window is per
(case, channel); the models' non-behavioral organizational metadata columns
(CERT r4.2 org codes O/C/E/A/N) are excluded from the SHAP comparison set
(they are never shown to the LLM and are not part of the evidence vocabulary;
they remain in `shap_attributions.csv`). The SHAP/evidence feature vocabulary
is aligned through the canonical `feature_cols_by_modality` (the evidence
features are the same percentile columns the models consume — verified
199/199 for http, 15/15 logon, 42/42 email).

| Metric | Value |
|---|---|
| Channel precision / recall / F1 | 1.0000 / 0.2803 / 0.4332 |
| Feature precision / recall / F1 | 0.1014 / 0.0092 / 0.0166 |
| Channel Jaccard (mean / median) | 0.2803 / 0.2500 |
| Feature Jaccard (mean / median) | 0.0090 / 0.0000 |
| Total LLM feature claims | 734 |
| Supported by SHAP | 83 |
| Evidence-grounded only | 651 |
| Unsupported claims | **0** |
| Unsupported claim rate | 0.0000 |
| Supported claim rate | 1.0000 |

Interpretation: the LLM's claims are channel-consistent with SHAP (perfect
channel precision) and 100 % evidence-grounded (0 unsupported claims); the low
feature-level overlap reflects that the frozen Tier-3 explanation cites the
canonical evidence inventory (top-3 evidence), while SHAP per-channel top-5
attribution is a different ranking signal — the comparison intentionally does
not punish evidence-grounded-but-not-SHAP-top claims.

## 13. Graceful degradation

`outputs/degradation_results.csv` (weight renormalization, missing score never
fabricated):

| Config | Recall | Precision | FPR | F1 | ROC-AUC | PR-AUC | Escalated |
|---|---|---|---|---|---|---|---|
| all channels | 0.8856 | 0.2500 | 0.0088 | 0.3899 | 0.9829 | 0.5878 | 712 |
| HTTP missing | 0.6667 | 0.0245 | 0.0876 | 0.0472 | 0.8814 | 0.0874 | 5,480 |
| File/Device missing | 0.8657 | 0.3000 | 0.0067 | 0.4456 | 0.9668 | 0.5065 | 580 |
| Logon missing | 0.8905 | 0.2828 | 0.0074 | 0.4293 | 0.9850 | 0.6030 | 633 |
| Email missing | 0.8706 | 0.2210 | 0.0101 | 0.3525 | 0.9783 | 0.5491 | 792 |

HTTP is load-bearing (recall −0.22 when missing); the others are
well-absorbed. Operational checks (FastAPI TestClient): request succeeds,
`degraded_channels` identifies the missing channel, `degraded_mode=true`,
rationale identifies degradation, and **no exception occurs** when a channel
is missing.

## 14. API testing

`outputs/api_test_results.json` — all passed:

| Test | Result |
|---|---|
| `GET /health` | 200, models_loaded, frozen config |
| `/score` escalated | 200, gate=high_confidence, degraded_mode=false |
| `/score` non-escalated | 200, escalated=false |
| `/score` missing channel | 200, degraded_channels=[http], degraded_mode=true |
| malformed request | 422 |
| schema validation (score>1) | 422 |
| LangGraph/vectorized equivalence | 200 cases, **0 mismatches** |

**Measured latency** (real, not fabricated):
- Tier1 + Tier2 deterministic: **≈ 1.85 ms** per request.
- Tier1 + Tier2 + Ollama (escalated case, include_triage): **≈ 1,082 ms**
  (Ollama Qwen2.5-1.5B, temperature 0) — sample returned severity `high`,
  action `soc_escalation`.

## 15. MLflow tracking

Experiment `week5_final_pipeline`, run `final_week5` (run_id
`97bbb7574a454c21b2f62a1d2b9ebe77`):

- **Parameters (13):** seed=42, split=60/20/20, HC=0.60,
  correlation=0.30, fusion weights 50/20/15/15, llm_model=qwen2.5:1.5b-instruct,
  shap_method=shap.DeepExplainer, shap_version=0.52.0,
  shap_background_n=100, shap_background_source=train-only, python_version,
  tier2_input / tier3_outputs paths.
- **Metrics (23):** recall, precision, f1, fpr, roc_auc, pr_auc,
  escalation_rate, channel_jaccard_mean, feature_jaccard_mean,
  unsupported_claim_rate, supported_claim_rate, channel_f1, feature_f1,
  degraded_{http,file_device,logon,email}_recall, TP/FP/FN/TN.
- **Artifacts (8):** system_metrics.csv, system_predictions.csv,
  shap_attributions.csv, explanation_comparison.csv, degradation_results.csv,
  final_week5_summary.json, shap_meta.json, shap_sanity.csv.

## 16. Limitations

- **SHAP environment split:** shap cannot coexist with the frozen numpy 2.5.1
  env (numba constraint); real SHAP runs in `.venv-shap` (numpy 2.4.6). The
  main venv is intact. Full details and the exact SHAP versions are recorded.
- **Tier-3 grounding metric:** 576/712 strict `grounding_valid` on the frozen
  outputs — the 136 failures are the documented Week-4 structural artifact
  (evidence_used top-1 vs explanation top-3); no invalid evidence IDs /
  channels / unsupported intent terms anywhere. Severity/action preservation
  **verified programmatically**: 712/712 raw-vs-final match, all enums valid.
- **Feature-level SHAP-vs-LLM overlap is low** (feature Jaccard mean 0.009)
  because the frozen Tier-3 explanation cites the canonical evidence
  inventory, which is not the same ranking as SHAP per-channel top-5; the
  comparison separates "evidence-grounded-only" from "unsupported" to stay
  fair. SHAP `rank` is per (case, channel); the models' org metadata columns
  (O/C/E/A/N) are excluded from the comparison set (documented).
- **DeepExplainer** explains the pre-sigmoid logit (2D-output wrapper); the
  additivity check is on the logit.
- Qwen2.5-1.5B severity is coarse (666/712 escalated = high) — a documented
  Week-4 limitation, not altered here.

## 17. Final conclusion

The Week-5 final implementation is complete and verified:

- Pipeline runs end-to-end; **Week-1–4 sources untouched**.
- Frozen Tier-2 config intact (0.60 / 0.30 / 50-20-15-15); escalation decision
  lives in Tier-2; Tier-3 reasons only about escalated cases; **no manual
  Tier-3 severity/action classifier**.
- **LangGraph/vectorized equivalence: 0 mismatches** (61,203 rows).
- `GET /health` and `POST /score` work; missing channels degrade gracefully.
- Full system metrics generated (Recall 0.886, Precision 0.25, F1 0.390,
  FPR 0.009, ROC-AUC 0.983, PR-AUC 0.588 — recomputed, not hard-coded).
- **Actual SHAP** (shap 0.52.0, DeepExplainer) on all four Tier-1 models:
  2,848/2,848 additivity checks pass; training-only background; correct
  feature alignment; no NaN/Inf.
- SHAP-vs-LLM comparison with fair 3-way claim classification: **0
  unsupported claims**, 100 % supported rate.
- Degradation matrix + API operational checks pass.
- MLflow run logged with params/metrics/artifacts.
- 712 frozen Week-4 Tier-3 cases align 1:1 (no re-run).
- `requirements_freeze.txt`, `final_week5_summary.json`,
  `final_week5_report.md` generated.
- **Tests:** 12/12 pipeline tests + 7/7 SHAP tests pass.
- **No fallback attribution was used** — all attribution in `shap_attributions.csv`
  is genuine SHAP (method + version recorded per row).
