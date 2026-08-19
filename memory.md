# Project Memory — Insider Threat Detection (CERT r4.2)

> **Purpose:** Consolidated, agent-readable memory of this project — architecture, methodology, environment, conventions, results, and known pitfalls. Compiled from the project's persistent memory store. Read this (plus `.agents/AGENTS.md`) before modifying anything.
>
> **Sync contract:** this file is itself a registered project memory; when significant facts change (methodology, results after the re-run, thresholds in `pipeline.py`), update it to stay in sync. `.agents/AGENTS.md` is the canonical documentation structure file (project tree, tooling standards, notebook status) and cross-references this file.
>
> **Companion docs:** `.agents/AGENTS.md` (detailed project structure & weekly history), `README.md`, weekly reports in `docs/`, `ARTIFACT_CLASSIFICATION.md` (canonical FINAL/FROZEN/LEGACY/OBSOLETE classification of all implementations & outputs, per decision D2), `docs/DATA_PROVENANCE.md` (canonical data provenance, per decision D3).

---

## 1. What This Project Is

A university research project building a **Hierarchical Multi-Agent Insider Threat Detection System** on the **CERT r4.2 dataset** (Carnegie Mellon SEI). Spans five weekly milestones (week1–week5), culminates in a FastAPI service exposing the full pipeline, and is documented via weekly reports presented to a professor.

- **Dataset:** `data/dayr4.2.csv` (~552 MB; one row = one user/day) + `data/dayr4.2-percentile30.pkl` (percentile-30 features, ~1.26 GB) for the MLP agents.
- **Class distribution (highly imbalanced):** Normal 329,486 user-days · Insider-1 85 · Insider-2 861 · Insider-3 20 (966 malicious user-days total). **70 malicious users** in `dayr4.2.csv` (Insider-1: 30, Insider-2: 30, Insider-3: 10) — hence all splits are stratified at the *user* level.
- **Ground-truth attack patterns** (CERT r4.2 red-team scenarios, verified against the official answer key `answers/scenarios.txt`): **insider code 1 = WikiLeaks uploader** (uploads data to wikileaks.org; HTTP dominant), **insider code 2 = job-hunter exfiltrator** (job-site surfing + thumb-drive data theft; email + file), **insider code 3 = keylogger sysadmin** (downloads keylogger, transfers via thumb drive; file/device + HTTP). ⚠️ Earlier project docs (e.g. `docs/week4_llm_integration_report*.md`, `week4_final_report.md` pre-audit) used an **inverted** label mapping (1=keylogger, 2=WikiLeaks, 3=job-hunter); the Week-4 final audit (2026-08-11) corrected this from the official answer key + behavioral profiles. All three scenarios are represented in the held-out test split (see §6/§9).
- Each user also has static OCEAN psychometric features.

---

## 2. Three-Tier Architecture

- **Tier 1 — per-modality anomaly agents.** **All four final agents are MLPs** (`ModalityFFNN` on percentile-30 flat features): HTTP 6:1, File/Device 4:1, Email 4:1, **Logon 4:1** — the old **Logon UniLSTM is retired** from the production Tier-1 path (kept in `week2/unilstm/` as historical comparison). Week-2 lineage: BiLSTM was the primary temporal approach, UniLSTM the baseline; MLflow experiment tracking and model persistence in `saved_models/` + `mlruns/` artifacts throughout, Jupyter-driven training workflow. Scoring is now **inference-only** in `week3/tier1_score_generation.ipynb` (loads the 4 pretrained `.pt`, no training).
- **Tier 2 — fusion layer (two parallel systems):**
  - **Weighted-fusion baseline:** ~~domain-fixed weights HTTP 0.40 / File 0.40 / Logon 0.10 / Email 0.10 — never tuned on evaluation data~~ **[AUDIT CORRECTION 2026-08-11]** the 40/40/10/10 weights were the *early* baseline only. The final frozen weights **HTTP 0.50 / File 0.20 / Logon 0.15 / Email 0.15 (HC 0.60, corr 0.30) were selected on validation data** in `week3/improved_tier2_optimization.ipynb` (config C, val F1 0.4666; fine-search plateau confirmed in `week4/historical/.../tier2_fine_search_sensitivity.ipynb` (folder renamed from `week4/_archive/` on 2026-08-14)) and evaluated once on test. See `docs/tier2_fusion_weight_selection_report.md` for the earlier 5-strategy selection (which had picked 60/20/10/10 @ 0.26 before the improved optimization superseded it).
  - **LangGraph orchestrator:** `StateGraph(FusionState)`: `compute_risk_score → check_high_confidence → [escalated? generate_llm_rationale : check_correlation] → END`. The week3 notebooks originally used a uniform `HIGH_CONFIDENCE_THRESHOLD` (any channel > HC) and a **simple sum** Gate 2; thresholds were tuned by Youden's-J grid search (HC ∈ [0.50, 0.95] step 0.05; CORR ∈ [0.80, 2.00] step 0.10) on Validation. **`week5/pipeline.py` has since evolved** (see Known Pitfalls §8.4): Gate 2 now uses the **weighted sum**, and the HC gate uses **per-channel thresholds** — only HTTP is enabled (`PER_CHANNEL_HC_THRESHOLDS['http']=0.34`); logon/file_device/email are disabled (threshold 1.0) because their validation FPR exceeds ~2%.
- **Tier 3 — LLM triage.** Local Ollama LLM produces structured, analyst-ready explanations for escalated cases.

### Tier-1 model implementations (custom PyTorch classes)

- `ModalityFFNN` (**all four final agents** — HTTP 6:1, File/Device 4:1, Email 4:1, Logon 4:1): 3-layer FFNN, hidden 64 → 32 → 1, ReLU, dropout 0.2; `BCEWithLogitsLoss`, Adam(lr=0.001), batch 128, 20 epochs; percentile-30 flat features; undersampling 4:1–6:1. Pretrained state dicts in `week2/mlp/saved_models/` + `week2/mlp/mlruns/8/<run>/artifacts/*.pt`.
- `ModalityLSTM` (historical Logon UniLSTM): 7-day sliding window, 1:1 undersampling, raw daily features — **retired from the production Tier-1 path** (still in `week2/unilstm/` for comparison).

### Tier-1 per-channel performance (legacy 80/20 numbers — historical, pre-final-config)

| Channel | Model | Recall | FPR |
|---|---|---|---|
| HTTP | MLP (4:1) | 91.8% | 0.5% |
| File/Device | MLP (4:1) | 62.1% | 8.9% |
| Email | MLP (6:1) | 26.3% | 0.6% |
| Logon | UniLSTM (1:1, window=7) | 27.0% | 18.9% |

> These reflect the **old** Tier-1 config (HTTP/File 4:1, Email 6:1, Logon UniLSTM). The **final** config is HTTP 6:1 / File 4:1 / Email 4:1 / Logon 4:1 (all MLPs); scores are generated by the inference-only `tier1_score_generation.ipynb` (see §6).

### Tier-3 LLM contract (Week 4 final — `week4/final_llm_reasoning/`)

- **Inputs:** (1) fused risk score, (2) list of channels that exceeded threshold, (3) top feature deltas (percentile deviations from baseline), (4) canonical evidence inventory with deterministic evidence IDs. Raw log entries are explicitly excluded.
- **Output:** structured JSON — `explanation`, `observations[] {evidence_id, statement}`, `channels_mentioned`, `evidence_used`, `severity_category` ∈ {critical, high, medium, low}, `recommended_action` ∈ {immediate_isolation, soc_escalation, monitor_enhanced, no_action}.
- **Enforcement:** token-level, via Ollama's `format` parameter with the Pydantic schema (`format=TriageOutput.model_json_schema()`) — no post-hoc parsing fixes.
- **Known limitation (documented, frozen):** Qwen2.5-1.5B does **not** populate the structured `observations`/`evidence_used` fields (0/712); the final grounded observations are constructed deterministically from the supplied evidence inventory (`llm_triage_final._finalize` → `generate_grounded_fallback`). The 100% validated-observation/evidence-ID metrics are **structural** metrics of that deterministic layer, not proof that 100% of the LLM's prose is semantically grounded.
- **Freeze artifacts:** `week4/final_llm_reasoning/outputs/week4_final_audit.md`, `week4_freeze_manifest.json`, `week4_freeze_report.md` (2026-08-11) — Week 4 is **FROZEN** as the Week 5 baseline.

---

## 3. Experimental Methodology (CRITICAL)

**Data provenance (canonical — see `docs/DATA_PROVENANCE.md`):** Raw CERT r4.2 logs → MIT feature-extraction script → preprocessed `dayr4.2.csv` → MIT temporal-representation script → percentile-30 representation → 60/20/20 user split → downstream experiments. The percentile-30 files were generated by our team using the MIT temporal-representation script on the preprocessed `dayr4.2.csv`, and the **entire preprocessed dataset was converted to percentile-30 BEFORE the 60/20/20 user split** — do NOT claim the percentile transformation was train-only, and do NOT regenerate/modify the percentile files or retrain models (frozen per decision D3, 2026-08-11).

**60/20/20 stratified user-level split** (Train/Validation/Test), migrated from a leaky 80/20 to eliminate test-set leakage. Two representations of the same split exist:

1. **`data/user_splits/`** (user-ID `.npy` arrays, int) — generated by `week1/week1_official_split.ipynb` from `data/dayr4.2.csv`; canonical for Week-1 RF, Week-2 experiment/bilstm/unilstm/mlp notebooks, and Week-5. `train_users.npy` 600 / `val_users.npy` 200 / `test_users.npy` 200; 42/14/14 malicious users; zero overlap.
2. **`data/user_split_formlp/`** (pre-split **percentile DataFrames**, float user IDs) — generated by `week3/percentile_mlp_user_split.ipynb` (and the script `data/create_percentile_split.py`) from `data/dayr4.2-percentile30.pkl`; rows: train 185,744 / val 61,076 / test 61,203 = 308,023; **same users** as (1) (float(int) match). Used by the Week-2 MLP agents and the inference-only `tier1_score_generation.ipynb`.

The split logic is identical to `week3/tier1_score_generation.ipynb` (same seed 42, stratified on per-user max insider label). The Tier-2 implementations and this methodology were guided by the project proposal document in the `docs/` folder:

```python
user_labels = df_percentile.groupby('user')['insider'].max().astype(int)
train_users, temp_users = train_test_split(unique_users, test_size=0.4,
                                           random_state=42, stratify=user_labels.values)
val_users, test_users = train_test_split(temp_users, test_size=0.5,
                                         random_state=42, stratify=user_labels.loc[temp_users].values)
```

Hard rules:

1. Tier-1 trains **strictly on the 60% Train set** (scalers fit on train only).
2. Scores exported for both held-out splits: `week3/tier1_validation_scores.csv` + `week3/tier1_test_scores.csv`.
3. **All tuning happens on Validation only** (fusion threshold; LangGraph HC/CORR; Week-1 RF decision threshold). Tuned parameters are **frozen** before the Test set is ever touched.
4. Final metrics computed **once** on the untouched Test set, labeled "Baseline Held-Out Test Metrics" / "LangGraph Held-Out Test Metrics".
5. Fusion weights: **[AUDIT CORRECTION 2026-08-11]** originally domain-fixed; the FINAL frozen weights (50/20/15/15) were validation-selected — see the Tier-2 correction above. Test set was still touched only once, after freezing.
6. **No-Logic-Change constraint:** for methodology updates, all underlying ML logic, agent architectures, weighting logic, and LangGraph routing must remain byte-identical — only data splitting, tuning scope, and reporting locations may change.
7. **Split sources:** `week1/week1_official_split.ipynb` → `data/user_splits/*.npy` (user IDs); `week3/percentile_mlp_user_split.ipynb` / `data/create_percentile_split.py` → `data/user_split_formlp/*.pkl` (pre-split percentile DataFrames). `tier1_score_generation.ipynb` no longer splits or trains — it loads the pre-split data and the pretrained MLPs, and only scores.
8. **Legacy-notebook migration convention:** when aligning legacy variants (e.g., `tier2_fusion_agent.ipynb`) with canonical notebooks, apply the full Validation/Test treatment (load both CSVs; tune on Validation; freeze; evaluate on Test) while preserving original ML logic.
9. **Week-1 uses the same official split:** `week1_random_forest_binary.ipynb` loads `data/user_splits/` directly — so Week 1, Tier-1, Tier-2, and Tier-3 all evaluate on the exact same users.
10. **Week-2 experiment notebooks use the official split too:** `week2/experiments/*.ipynb` load `data/user_splits/` via `experiment_utils.load_official_user_split()`, validate on the validation split, and evaluate once on test.

---

## 4. Environment & Tooling

- **Package manager:** `uv` (declarative `pyproject.toml` + `uv.lock`). `uv sync` installs everything (auto-provisions Python 3.12 if missing).
- **`uv run` is the standard execution command** for all project-facing operations — it verifies the lockfile and syncs `.venv` first:
  ```powershell
  uv run python -m week5.export_models
  uv run uvicorn week5.app:app --port 8000
  uv run jupyter lab week3/tier1_score_generation.ipynb
  uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
  ```
- **Python:** 3.12 strictly pinned (`.python-version`); PyTorch CUDA 12.1 wheels via custom index, CPU fallback available; pandas capped `<3`; protobuf pinned `==3.20.3`. **GPU note:** the dev machine's RTX 5090 (sm_120) is **not** supported by the installed cu121 build — training/inference runs on CPU via the `_cuda_usable()` fallback (see Known Pitfalls §8.9).
- **Dependencies pre-declared:** langgraph, torch, imbalanced-learn, scikit-learn, pandas, numpy, matplotlib, seaborn, fastapi, uvicorn, httpx. `ollama` (>=0.4.0, installed 0.6.2) added via `uv add ollama`; its transitive deps (httpx, pydantic) were already satisfied.
- **Ollama (Tier 3):** must be installed and running locally at `http://localhost:11434`; default model **`qwen2.5:1.5b-instruct`** (`ollama pull qwen2.5:1.5b-instruct`), alternative `gemma3:4b`. The pipeline auto-detects availability and falls back to deterministic template triage.
- **Experiment tracking:** MLflow, SQLite backend `sqlite:///mlflow.db` (the default `./mlruns` FileStore is deprecated). Models persisted under `saved_models/` dirs. **Launch the UI with:** `uv run mlflow ui --backend-store-uri sqlite:///mlflow.db` — without the flag the UI reads the empty `./mlruns/` store and appears blank. Never launch via `uvicorn --workers 4`. Artifact store anchored to project-root `mlruns/` by `week2/experiments/experiment_utils.py`.
- **Week-2 MLflow logging:** every experiment notebook (`week2/experiments/*.ipynb`) logs via `experiment_utils.start_overwritable_run(EXPERIMENT_NAME, RUN_NAME)` — each notebook has a unique `RUN_NAME` (`logon_experiment_1..3`, `email_experiment_1..2`), and re-running a notebook **overwrites only its own** previous run (permanent purge by `exp_run_name` tag), never a sibling's. Experiments: `Week2_Logon_Experiments`, `Week2_Email_Experiments`.
- **Portability:** every notebook/script walks up the directory tree to find `pyproject.toml` and resolves all paths dynamically — the parent folder can be renamed freely. Workspace root contains a `.qoder` config; **planning documents live in `.qoder/plans/`**.
- **PowerShell:** user shell is PowerShell — `&&` is unsupported, use `;`. In Markdown docs, tag Windows-style CLI command blocks with the `powershell` language tag.
- **uv warning (harmless):** if the uv cache and project sit on different filesystems, uv falls back from hardlinking to copying — performance warning only, not an error.
- **Memory-constrained data loading:** `dayr4.2-percentile30.pkl` is ~1.26 GB. Standard protocol: load full pickle → extract only needed columns (~100–150) → `del df_percentile_full` → downcast to float32 → then load other datasets. Fallback: one-time Parquet conversion for column-wise lazy loading.

---

## 5. Reproducibility Practices

- **Seeding:** `random.seed`, `np.random.seed`, `torch.manual_seed`, `torch.cuda.manual_seed_all`, `cudnn.deterministic=True`, `cudnn.benchmark=False`; all `DataLoader`s use an explicit `generator=g`. Seed 42 throughout.
- **Deterministic feature ordering:** use `sorted(cols)`, never `list(set(cols))`. (Exception: the Week-2 MLP agents use `list(set(cols)) + personality` — the **same ordering must be reused at inference time** for the pretrained models.)
- **Tier-1 scores are now generated by INFERENCE ONLY** (no retraining): `week3/tier1_score_generation.ipynb` loads the 4 pretrained MLP `.pt` state dicts and scores val/test. The scaler (not saved by training) is refit on **train users only** — deterministic, no test leakage. Same checkpoints + same preprocessing + same data → same scores.
- **CPU nondeterminism caveat:** PyTorch CPU training loops are **not bit-reproducible across runs** (multi-threaded op reductions), even with the seed block — observed (e.g., 2-run test recall differed slightly). Split + undersampling + model init + eval forward ARE deterministic; only the training forward/backward reductions vary. This matches `tier1_score_generation`'s config level; document rather than silently fix.
- **Week 1 baseline config** (used in every comparison): binary Random Forest — `n_estimators=200`, `class_weight='balanced'`, `random_state=42`, **user-level split loaded from the official `data/user_splits/`**; target binarized as `insider > 0`. Canonical baseline notebook: `week1_random_forest_binary.ipynb` (loads processed features from `data/processed/features_clean.parquet` + official split; calibrates threshold on validation only; evaluates once on test). `week1_random_forest_model_only.ipynb` / `week1_random_forest_fixed.ipynb` are legacy binary variants.
- **Official split generation:** `week1/week1_official_split.ipynb` regenerates `data/user_splits/*.npy` deterministically (same `random_state=42` logic as week3) — rerunning reproduces the identical split. The percentile pre-split (`data/user_split_formlp/`) is regenerated by `week3/percentile_mlp_user_split.ipynb` / `data/create_percentile_split.py`.

---

## 6. Canonical Notebooks & Artifacts

- **Week 3 (Tier-2, canonical):** `tier1_score_generation.ipynb` is now **inference-only** — loads the 4 pretrained Week-2 MLPs (HTTP 6:1, File 4:1, Email 4:1, Logon 4:1 from `week2/mlp/mlruns/8/`), scores `data/user_split_formlp/` val/test, and writes `tier1_validation_scores.csv` + `tier1_test_scores.csv` (schema `user, week, day, label, logon_score, email_score, file_device_score, http_score`). `tier2_fusion_agent1.ipynb` is the **canonical weighted-fusion baseline** (per project owner). `tier2_langgraph_orchestrator.ipynb` is the canonical LangGraph implementation. Both consume the CSVs unchanged.
- **`week3/percentile_mlp_user_split.ipynb`** — produces `data/user_split_formlp/*.pkl` (stratified user-level 60/20/20 of the percentile pickle, SEED=42, 600/200/200 users, 42/14/14 malicious, zero overlap, deterministic + verified). Also exists as `data/create_percentile_split.py`.
- **`week3/tier2_fusion_agent.ipynb` status:** legacy copy, now **fully migrated** to the 60/20/20 protocol (loads both score CSVs; sweeps threshold on Validation only; freezes `FROZEN_THRESHOLD`; evaluates on Test). Results remain **non-canonical** — use `tier2_fusion_agent1.ipynb` for authoritative numbers — but no test-set leakage remains.
- **Dual-version report convention:** original reports are preserved unchanged as historical documentation; updated reports get an `_updated` suffix reflecting the 60/20/20 methodology and the `tier1_scores.csv` → `tier1_test_scores.csv` migration.
- **Legacy artifact:** `week3/tier1_scores.csv` (old 80/20) is kept only for historical `docs/` reports and `week3/historical_baseline_comparison/` notebooks (folder renamed from `week3/uneeded/` on 2026-08-14). All active consumers read `tier1_test_scores.csv`.

### Headline results (legacy 80/20 — to be replaced after the leakage-free re-run)

| Metric | Week-1 binary RF | Tier-2 weighted fusion | Tier-2 LangGraph |
|---|---|---|---|
| Recall | 0.1875 | 0.9424 | 0.9465 |
| FPR | 0.0001 | 0.0804 | 0.0538 |
| AUC | 0.9639 | 0.9934 | 0.9926 |

LangGraph operating point: HC = 0.95, CORR = 1.10 (legacy confusion: TN=54,517, FP=6,339, FN=15, TP=228; escalations 6,368 HC + 199 CORR). Week-5 full system: 5,120 escalations (3,387 HC + 1,733 CORR), single-request LangGraph latency ~1.13 ms/case, vectorized fusion ~155k rows/s.

### Week 4 final (FROZEN) — `week4/final_llm_reasoning/`

- **712 escalated cases** (frozen Tier-2 input `week4/frozen_inputs/improved_tier2_test_predictions.csv`), **712 LLM calls**, 0 retries, 0 fallbacks, 0 invalid evidence IDs/channels/observations, 0 unsupported intent terms.
- Severity {high 666, medium 37, low 9}; actions {immediate_isolation 341, soc_escalation 325, monitor_enhanced 37, no_action 9}. Qwen severity/action preserved (0 raw-vs-final mismatches) — no manual policy.
- Ground-truth leakage: **none** (label/insider dropped by EvidenceBuilder; verified across all 712 evidence objects and sampled prompts).
- **Scenario evaluation (corrected mapping, held-out test):** Scenario 1 WikiLeaks (users 555/564/848 → 12 escalated malicious), Scenario 2 job-hunter (118/247/577/652/767/900 → 162 escalated malicious; user 247 = 40 cases/26 malicious), Scenario 3 keylogger (11/861 → 4 escalated malicious). Scenario-1 test users 49/620/723 were Tier-2 escalation misses (documented, not fabricated).
- **Grounding limitation:** the 100% validated-observation/evidence-ID metrics are structural (deterministic evidence layer); Qwen 1.5B does not fill `observations`/`evidence_used`; the lightweight explanation audit (all 712 deliverable explanations: 0 invalid IDs, 0 channel/fired mismatches, 0 direction contradictions, 0 unsupported claims) is not a rigorous hallucination benchmark.

### Week 5 final — `week5/final_pipeline/` (frozen Tier-2 config, held-out test, evaluated once)

| Metric | Value |
|---|---|
| Recall | 0.8856 |
| Precision | 0.2500 |
| F1 | 0.3899 |
| ROC-AUC | 0.9829 |
| PR-AUC | 0.5878 |
| FPR | 0.0088 |
| TP/FP/FN/TN | 178 / 534 / 23 / 60,468 |
| Escalations | 712 (1.16%) |

- **Attribution vs LLM (712 cases):** channel precision 1.00 / recall 0.35 / F1 0.41; feature precision 1.00 / recall 0.00; Jaccard 0.35; **0 LLM claims unsupported by attribution** (conservative anti-hallucination trade-off). SHAP not installed → gradient-based attribution (documented).
- **Degradation (leave-one-channel-out):** HTTP is load-bearing (recall 0.886 → 0.667 when missing, escalations 5,480); file_device/logon/email loss well-absorbed (recall 0.87–0.89). API never fails — `degraded_channels` + `DEGRADED MODE` rationale.
- **API/tests:** `POST /score` + `GET /health`; 4/4 smoke test groups pass; LangGraph single-request latency **1.13 ms/case**; vectorized fusion ~155k rows/s; MLflow run `Week5_Final_EndToEnd_Evaluation`.

---

## 7. Production-Readiness Notes (for limitations sections)

Static global thresholds tuned on test data (e.g., 0.95 HC / 1.10 CORR) are not production-ready. Real deployments need: (1) tuning on validation data only, (2) per-user adaptive baselines (user-specific percentiles), (3) cost-sensitive optimization favoring recall over precision, (4) analyst feedback loops to adjust thresholds dynamically, (5) periodic recalibration for concept drift, (6) multi-tier confidence scoring instead of binary escalation.

---

## 8. Known Pitfalls

1. **Notebook JSON corruption:** `.ipynb` files are strict JSON. Any edit must be JSON-aware (`json.load`/`json.dump`; `source` = list of lines; nbformat 4.5 cells need an `id` field). NEVER raw-text-replace inside notebooks.
2. **[AUDIT CORRECTION 2026-08-11 — SUPERSEDED]** numba/shap incompatibility was resolved with an isolated interpreter: SHAP runs in `.venv-shap` (its own pinned numpy), not the main env. The final Week-5 pipeline uses real `shap.DeepExplainer` (`week5/final_pipeline/evaluation/shap_analysis.py`); run those tests with `.venv-shap/Scripts/python.exe -m pytest week5/final_pipeline/tests/test_shap.py -v`. The earlier gradient-based attribution was a Week-5 historical interim step.
3. **`tier1_test_scores.csv` / `tier1_validation_scores.csv` now exist** under the 60/20/20 split (61,203 test rows / 61,076 val rows, written by `week3/tier1_score_generation.ipynb`). They are generated artifacts, not tracked — if regenerated, downstream week4/week5 consumers depend on them.
4. **Week 5 thresholds have diverged from week3.** `week5/pipeline.py` module defaults are `HIGH_CONFIDENCE_THRESHOLD=0.95`, `CORRELATION_THRESHOLD=0.37` (weighted-sum), plus `PER_CHANNEL_HC_THRESHOLDS` (HTTP 0.34 enabled; others 1.0 disabled). `calibrate_thresholds()` / `calibrate_per_channel_thresholds()` re-tune these on the validation set and write back to the module globals, so `evaluate.py`/`app.py` pick up the calibrated values automatically. The **final frozen config** used by the Week-4 freeze and `week5/final_pipeline/` is HC=0.60 / corr=0.30 / weights 50/20/15/15 (see `week5/final_pipeline/pipeline.py`).
4b. **Scenario mapping correction (2026-08-11).** The official CERT r4.2 answer key (`answers/scenarios.txt` + `insiders.csv`) maps insider 1 = WikiLeaks uploader, 2 = job-hunter, 3 = keylogger. Earlier docs inverted this. The corrected `SCENARIO_MAP` in `week4/final_llm_reasoning/scenario_evaluation.py` and the regenerated `scenario_results.csv` are frozen; do not change them again.
5. **Cross-filesystem uv cache** → hardlink fallback warning; harmless.
6. **MLflow UI looks empty if launched wrong:** `mlflow ui` without `--backend-store-uri sqlite:///mlflow.db` reads the empty `./mlruns/` FileStore. Launch via `uv run mlflow ui --backend-store-uri sqlite:///mlflow.db`. Do NOT launch via `uvicorn --workers 4` (broken server that serves the shell but can fail to populate runs).
7. **MLflow artifact root:** experiments created from a notebook CWD (e.g. `week2/experiments/`) got `artifact_location` pointing at a missing dir, which made the UI runs list appear empty. `experiment_utils.start_overwritable_run()` now anchors experiments/runs to the project-root `mlruns/` and repairs stale locations on re-run.
8. **`torch.load` FutureWarning:** always load checkpoints with `weights_only=True` (all project checkpoints are plain state dicts). All week2 experiment notebooks already do this.
9. **RTX 5090 (sm_120) unsupported by the installed torch CUDA build** — `CUDA error: no kernel image is available`. `torch.cuda.is_available()` lies (True) but real ops crash. Every PyTorch notebook must use the `_cuda_usable()` device-probe fallback (tiny op probe → CPU fallback). Applied to the week-2 bilstm/unilstm/mlp notebooks + the week-3 inference notebook; keep it whenever touching PyTorch code.
10. **Week-2 notebooks may revert to their original state** (observed repeatedly: `bilstm_agents_imbalanced_4_to_1`, `mlp_agents_percentile_4to1/6to1`, `logon_unilstm_agent`, and the tier-3 notebook file `tier1_mlp_60_20_20` was deleted per user request). If a converted notebook regresses to `train_test_split` + no `_cuda_usable`, re-apply the conversion; check `.ipynb_checkpoints` too. Do not assume a previously-fixed file is still fixed — re-verify before relying on it.
11. **[AUDIT CORRECTION 2026-08-11 — RESOLVED]** MLP scalers ARE now persisted: `tier1/<mod>/scaler.joblib` (fit on train users only) ships with each agent alongside `model.pt` + `meta.json`, and `week5/final_pipeline` loads them at inference. The historical in-function refit in `tier1_score_generation.ipynb` predates this.
12. **`mlp_and_logon_results.txt`** — a generated text dump (project root) of the executed outputs of `week2/mlp/*` + `week2/unilstm/logon_unilstm_agent.ipynb`, created for review; it is a scratch artifact, not part of the pipeline.
