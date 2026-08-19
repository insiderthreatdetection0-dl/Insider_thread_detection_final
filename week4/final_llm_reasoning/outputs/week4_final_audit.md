# Week 4 Final Validation Audit

**Date:** 2026-08-11
**Scope:** Final audit of the existing Week 4 Tier-3 LLM reasoning implementation against the project proposal. Read-only except for one verified correction to the scenario mapping (see §8). No Tier-1/Tier-2 changes, no retraining, no LLM re-runs, no architecture changes.
**Audit inputs:** `week4/final_llm_reasoning/*.py`, `final_llm_reasoning.ipynb`, `outputs/*`, frozen Tier-2 input `week4/frozen_inputs/improved_tier2_test_predictions.csv`, `data/dayr4.2.csv`, `data/user_splits/*.npy`, official CERT r4.2 answer key (`answers.tar.bz2` → `scenarios.txt` + `insiders.csv`, figshare DOI 10.1184/R1/12841247.v1).

---

## 1. Requirement compliance table

| Requirement | Status | Evidence |
|---|---|---|
| Ollama/Qwen | PASS | Ollama server live, model `qwen2.5:1.5b-instruct`; 712/712 `initial_call_success`, 0 retries, 0 fallbacks. |
| fused score in prompt | PASS | `build_prompt` → `inventory_text` includes `FUSED RISK SCORE: <x.xxx>` for every case (verified on cases 0/5/400 and sampled). |
| fired channels in prompt | PASS | `FIRED CHANNELS: <list>` present; per-case. |
| feature deltas in prompt | PASS | `EVIDENCE INVENTORY` lists each evidence item with ID, channel, feature, percentile, delta, direction, strength. |
| raw logs excluded | PASS | Prompt contains only percentile-based evidence summaries; no raw log tokens (`starttime`/`endtime`/`http://`/activity rows) in any sampled prompt. |
| structured JSON | PASS | All 712 `raw_llm_output` are valid JSON with `explanation`, `severity_category`, `recommended_action` (and the schema fields); 712/712 parsed successfully; enums valid. |
| grounding validation | PARTIAL | Validator checks are structural/evidence-ID (IDs exist, channel fired, no unsupported-intent terms, numeric/direction agreement). It does **not** prove semantic correctness of the LLM's prose — and the 100% figures are computed on deterministically constructed observations, not the LLM's own explanation (see §3). |
| hallucination/unsupported-claim audit | PASS | 0 unsupported claims, 0 invalid evidence IDs, 0 contradictions in the deliverable explanations (all 712); raw LLM explanations are shallow but contain no invented specifics. |
| keylogger scenario | PASS (held-out) | Scenario 3 (keylogger sysadmin) **is** represented in the held-out test escalated set: users 11 & 861, 4 escalated malicious cases, all validated. Previously mis-documented as "not in test". |
| WikiLeaks scenario | PASS (held-out) | Scenario 1 (WikiLeaks uploader) held-out users 555/564/848, 12 escalated malicious cases, all validated. Previously mis-documented as "not in test". |
| job-hunter scenario | PASS | Scenario 2 (job-hunter) held-out users 118/247/577/652/767/900, 162 escalated malicious cases; user 247 = 40 escalated / 26 malicious (unchanged). |
| no manual severity policy | PASS | No risk/fired-channel→severity/action thresholds anywhere in the active modules; 0/712 raw-vs-final severity/action mismatches (Qwen preserved). |

---

## 2. Existing final execution metrics (reported, not re-run)

| Metric | Value |
|---|---|
| Escalated cases (frozen Tier-2 input) | 712 |
| LLM calls | 712 |
| Retries | 0 |
| Fallbacks | 0 |
| Cases with validated observations | 712 / 712 (100%) |
| Validated observation rate | 1.0 |
| Cases with validated evidence IDs | 712 / 712 (100%) |
| Validated evidence rate | 1.0 |
| Invalid evidence IDs | 0 |
| Invalid channels | 0 |
| Invalid observations | 0 |
| Unsupported intent terms | 0 |
| Severity distribution | high 666 · medium 37 · low 9 (critical 0) |
| Action distribution | immediate_isolation 341 · soc_escalation 325 · monitor_enhanced 37 · no_action 9 |

**Important interpretation (see §3):** the 100% figures measure the deterministic evidence-grounded layer, not the LLM's prose. They are correct as reported but must not be read as "the LLM's explanations are semantically 100% grounded".

---

## 3. Grounding-validation semantics (CH3)

What the reported metrics actually measure:

- `validated_observation_rate` / `validated_evidence_rate` / `invalid_evidence_ids` / `unsupported_intent_terms` are computed on the **final deliverable** `explanation` + `observations`, which are **constructed deterministically** by `llm_triage_final._finalize()` → `generate_grounded_fallback()` from the supplied evidence inventory. Every observation is bound to an evidence ID that exists in the case inventory; the explanation is a template (`<evidence statement>. Risk score <x> triggered the <gate> gate.`). Hence 100% by construction — this is structural validation of the deterministic layer, not semantic validation of LLM prose.
- Qwen2.5-1.5B returns only `{explanation, severity_category, recommended_action}`; it does **not** populate `observations[]`/`evidence_used`/`channels_mentioned` (0/712 in the structured fields). This is documented in the code as a KNOWN LIMITATION.
- Applying `GroundingValidator` to the **raw LLM explanations**: `grounding_valid` = 0.0, `evidence_id_valid` = 0.0, `feature_grounding_valid` = 0.017, `channel_grounding_valid` = 0.125, `numeric_grounding_valid` = 1.0, `direction_grounding_valid` = 1.0, unsupported claims = 0. The 0.0 values come from the LLM never emitting the structured evidence fields (schema/contract mismatch), and the low feature/channel values from the validator's exact-label matching vs the LLM's natural-language paraphrases — the same root causes identified in the earlier diagnostic audit (`week4/diagnostics/week4_grounding_audit.md`).

**Conclusion:** the validator checks evidence-ID existence, fired-channel membership, direction/numeric consistency, and unsupported-intent terms. It does **not** prove that the LLM's prose explanation is semantically grounded. The deliverable explanations are grounded because they are evidence-derived by construction; the LLM's own prose is shallow (mostly risk-score restatement) but contains no fabricated specifics.

---

## 4. Explanation-grounding audit (CH4)

Method: reused existing saved outputs only (no LLM re-runs). Full 712-case deterministic check of the deliverable explanations plus a 13-case stratified human-style sample (high/medium/low, 1 vs multi fired channels, http/file/email/logon combos).

**All 712 deliverable observations:**
- evidence IDs valid: 712/712 (0 invalid)
- evidence channel is fired: 712/712 (0 not-fired)
- statement direction matches evidence direction: 712/712 (0 mismatches)
- unsupported-intent terms: 0
- elevated-vs-reduced contradictions: 0

**Raw LLM explanations (712):** 327 risk-only, 55 risk+gate, 330 mention evidence-ish terms; 0/712 cite evidence IDs inline; 2/712 cite percentiles; 0 unsupported claims. They are shallow (mostly "FUSED risk score is high at X" / "indicates a moderate level of risk") but do not hallucinate activities, motives, events, or data-theft claims.

**Sample audit table (representative):**

| case_id | sev/action | fired | evidence supplied | deliverable explanation | raw LLM explanation | verdict |
|---|---|---|---|---|---|---|
| 9/365 | high/soc_escalation | http | HTTP_E1..E3 (content length/word count/url depth reduced) | "HTTP content length is reduced (0th percentile). Risk score 0.513 triggered the high_confidence gate." | "FUSED risk score is high and the GATE trigger indicates a strong likelihood of malicious activity." | PASS (supported; shallow but not contradictory) |
| 11/486 | high/immediate_isolation | http,file_device,logon,email | HTTP_E1 (afterhour http c len elevated) | "afterhourhttp mean http c len is elevated (100th percentile)…" | "FUSED risk score is high at 0.937, indicating a significant security threat." | PASS |
| 49/421 | medium/monitor_enhanced | http | HTTP_E1 (afterhour http c len elevated) | "afterhourhttp mean http c len is elevated (100th percentile)…" | "FUSED RISK SCORE: 0.500 indicates a moderate risk level." | PASS |
| 51/110 | low/no_action | http | HTTP_E1 (content length reduced) | "HTTP content length is reduced (0th percentile)…" | "FUSED risk score is low at 0.322, indicating a low likelihood of malicious activity." | PASS |
| 113/37 | medium/monitor_enhanced | file_device,logon,email | EMAIL_E1 (email n-Xemail1 reduced) | "email n-Xemail1 is reduced (6th percentile)…" | "FUSED risk score is elevated at 0.316, indicating a moderate level of risk." | PASS |

No FAIL cases. All sampled explanations are supported by the supplied evidence; the deliverable explanation is always tied to a real evidence ID; no unsupported malicious-intent/data-theft/motive claims, no events absent from evidence, no contradictions. Caveat: the raw LLM prose is largely generic risk restatement, so "grounded" here primarily reflects the deterministic evidence layer plus the absence of hallucination in the LLM's short prose. This is a lightweight audit, not a statistically rigorous hallucination benchmark.

---

## 5. Severity/action diagnostic (CH5)

- **Fused risk score:** min 0.300, mean 0.524, median 0.536, max 0.940 (all escalated cases ≥ 0.30).
- **Fired channels:** 1 channel 152, 2 310, 3 216, 4 34 cases.
- **Severity vs risk:** high mean 0.530 (range 0.301–0.940); medium 0.464 (0.305–0.682); low 0.314 (0.300–0.323). corr(severity_ordinal, risk) = 0.170.
- **Severity vs fired-channel count:** high mean 2.21, medium 1.78, low 2.33; corr = 0.067.
- **Severity×action crosstab:** perfectly diagonal (high→immediate_isolation 341 / soc_escalation 325; medium→monitor_enhanced 37; low→no_action 9). **This comes from the raw LLM output** (raw==final, 0 mismatches): Qwen emits `high` for most escalated cases with a fixed action pairing.

**Interpretation:** the High-heavy distribution (666/712) is **consistent with the supplied evidence**. Every escalated case has a fused risk ≥ 0.30 and ≥ 1 fired channel, and Qwen2.5-1.5B maps "risk score is high" → `high` (a coarse but defensible behaviour at the escalation-triage stage: by the time Tier 2 escalates, the module treats it as high-priority). The LLM is **not** blindly outputting High: the 46 non-high cases cluster at lower risk (≤ 0.682) while high spans the full range, and medium/low map to the risk-score wording ("moderate"/"low") in the raw explanations. No manual thresholds were introduced or needed; the architecture remains LLM-generated severity/action.

---

## 6. Scenario evaluation (CH6)

### Corrected scenario mapping (verified from the official answer key)

The official CERT r4.2 answer key (`answers.tar.bz2` → `scenarios.txt`) defines:
1. **WikiLeaks uploader** — "uploading data to wikileaks.org"
2. **Job-hunter exfiltrator** — "surfing job websites … use a thumb drive … to steal data"
3. **Keylogger sysadmin** — "Downloads a keylogger and uses a thumb drive to transfer it to his supervisor's machine"

This matches the `insider` codes in `data/dayr4.2.csv` and the user-level split (`data/user_splits/*.npy`), and the behavioural profiles (insider=1 users dominate leak-site visits; insider=2 job-site visits; insider=3 hack-site/executable/USB).

**The previous `SCENARIO_MAP` was factually wrong:** it labelled only user 247 as "job-hunter" and asserted keylogger/WikiLeaks users are "not in the held-out test". In fact **all three scenarios have held-out TEST users with escalated malicious days in the frozen 712-case set.** This was a documentation/evaluation bug, now fixed (see §8). No users were moved between splits, nothing was retrained, and the frozen 712-case input is unchanged.

### Held-out quantitative evaluation (existing saved outputs, no LLM re-runs)

| Scenario | Official code | Held-out test users in 712 set | Escalated cases | Escalated malicious (label=1) | Validated obs / IDs | Unsupported terms |
|---|---|---|---|---|---|---|
| 1 · WikiLeaks uploader | 1 | 555, 564, 848 | 65 | 12 | 65 / 65 | 0 |
| 2 · Job-hunter exfiltrator | 2 | 118, 247, 577, 652, 767, 900 | 245 | 162 | 245 / 245 | 0 |
| 3 · Keylogger sysadmin | 3 | 11, 861 | 8 | 4 | 8 / 8 | 0 |
| **Total** | | 11 users | 318 | **178** | 318 / 318 | 0 |

Job-hunter detail (unchanged from the previous report): **user 247 = 40 escalated cases, 26 malicious**, all validated; severity high 37 / medium 3; actions immediate_isolation 27 / soc_escalation 10 / monitor_enhanced 3. The full scenario rows (per user, with gates/fired channels/severity/actions) are in `outputs/scenario_results.csv` (regenerated from existing outputs).

### Qualitative / documented limitations

- Scenario-1 held-out test users **49, 620, 723** have malicious days that Tier-2 did **not** escalate, so they are not in the 712-case input and are documented (not fabricated) as Tier-2 escalation misses. User 49 also has 103 escalated *normal* days in the set (escalation-stage false positives).
- The three scenario evaluations above are **all genuine held-out test evaluations** (the users are in `test_users.npy`), so no qualitative/non-held-out substitute was needed for the Week 4 scenario requirement.
- The earlier project reports (`docs/week4_llm_integration_report*.md`) used an inverted scenario naming (calling users 30/70/136/199 "WikiLeaks/keylogger" and user 247 "job-hunter"); those older reports predate this audit. The frozen code + this audit use the official mapping. This naming discrepancy should be noted in the internship report.

---

## 7. Leakage audit (CH2)

**No ground-truth leakage.** Verified programmatically:
- The evidence object built per case contains only `channel_scores, day, feature_deltas, fired_channels, gate_triggered, risk_score, user, week` (all 712 cases). The Tier-2 `label` column exists in the CSV but is **dropped** by `EvidenceBuilder`; no `insider`/`scenario`/`malicious`/`label` field reaches the evidence or the prompt.
- No leak tokens (`insider`, `scenario`, `label`, `ground truth`, `benign`, `malicious`, `attack`, `threat`) appear in any sampled user prompt.
- The system prompt contains the word "malicious" only inside the anti-hallucination guardrail ("do NOT write … 'is malicious' unless the supplied evidence explicitly supports that specific claim"), not as a label.

The LLM receives only legitimate behavioural evidence: fused score, channel scores, fired channels, feature deltas, percentiles, gate trigger.

---

## 8. Files changed / created

| File | Change |
|---|---|
| `week4/final_llm_reasoning/scenario_evaluation.py` | **Fixed** (only code change). Corrected `SCENARIO_MAP` and `EXPECTED_CHANNELS` to the official CERT r4.2 mapping (scenario 1 = WikiLeaks users 555/564/848; scenario 2 = job-hunter users 118/247/577/652/767/900; scenario 3 = keylogger users 11/861), updated module docstring and the `scenario_report` printout to document scenario-1 test users 49/620/723 as Tier-2 escalation misses. No severity/action policy, no split changes, no retraining. |
| `week4/final_llm_reasoning/outputs/scenario_results.csv` | **Regenerated** from the existing saved outputs (no LLM calls) with the corrected mapping; previous user-247 row (40/26) preserved. |
| `week4/final_llm_reasoning/outputs/week4_final_audit.md` | **Created** — this audit report. |

No other files modified. Historical/archive files under `week4/_archive/` were not touched.

---

## 9. Final verdict

**READY WITH DOCUMENTED LIMITATIONS**

The existing Week 4 implementation satisfies the proposal: local Qwen2.5-1.5B via Ollama; structured prompt with fused score + fired channels + feature deltas (no raw logs); structured JSON triage output; deterministic grounding validation; no ground-truth leakage; no manual severity/action policy (Qwen's severity/action preserved); all three known malicious scenarios now have genuine held-out test evaluation. The only code change was correcting the scenario mapping, which was objectively wrong and is now verified against the official answer key.

**Documented limitations to carry into the final report/presentation:**
1. **The 100% grounding metrics are structural, not semantic.** They validate deterministically constructed evidence-derived observations, not the LLM's prose. Qwen2.5-1.5B does not populate the structured `observations`/`evidence_used` fields (contract compliance 0%); applied to raw LLM prose, the validator yields grounding_valid ≈ 0 mainly due to that schema mismatch. The LLM's own explanations are shallow (mostly risk-score restatement) though hallucination-free.
2. **Severity/action are coarse:** 93.5% of escalated cases are labelled `high`, with a near-perfect severity→action pairing from the model. This is defensible for an escalation-stage triage but not a nuanced risk ranking; severity correlates only weakly with risk (0.17) and fired-channel count (0.07).
3. **Scenario naming discrepancy with earlier reports:** older project docs (`docs/week4_llm_integration_report*.md`) used an inverted scenario label mapping; this audit and the frozen code use the official CERT r4.2 mapping.
4. **Tier-2 escalation misses:** scenario-1 test users 49/620/723 and scenario-3 user 11 (some days) were not fully escalated; the scenario evaluation covers the escalated malicious days present in the frozen 712-case set.
5. This is a lightweight explanation audit on existing outputs, not a statistically rigorous hallucination benchmark (Week 5 will do the broader end-to-end quantitative evaluation).
