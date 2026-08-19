"""Week-6 case service (deployment layer over the canonical frozen pipeline).

Serves ONE (user, day) case for the demo deployment. Everything is assembled
from the EXISTING frozen research implementation and artifacts:

  - Tier-1 scores: live inference with the frozen tier1/ MLP artifacts via
    week5.final_pipeline.pipeline.load_tier1/score_mlp + tier1.assemble_feature_rows
  - Tier-2: frozen gating via week5.final_pipeline.pipeline.tier2_decision
    (HC 0.60 / corr 0.30 / weights 50-20-15-15) and the LangGraph rationale
  - Evidence: canonical Week-4 EvidenceBuilder (unchanged, re-exported by
    week5.final_pipeline.evidence_builder)
  - SHAP: LOOKUP in the frozen week5/final_pipeline/outputs/shap_attributions.csv
  - LLM explanation: LOOKUP in the re-frozen Week-4 outputs
    (week4/final_llm_reasoning/outputs/llm_triage_outputs_final.csv)

NO live LLM generation, NO SHAP computation, NO retraining — read-only over
frozen artifacts. Cases outside the frozen held-out test split are not served
(404), because Tier-1 features for the demo come from the frozen test split.
"""
from __future__ import annotations

import json

import pandas as pd

from week5.final_pipeline import pipeline as P
from week5.final_pipeline import tier2 as T2
from week5.final_pipeline.tier1 import assemble_feature_rows, score_cases
from week5.final_pipeline.tier3 import make_evidence_builder

SHAP_TOP_K = 5  # display window; same per-(case, channel) convention as the
# frozen Week-5 explanation_comparison.py
META_COLS = {"role", "ITAdmin", "b_unit", "f_unit", "dept", "team",
             "O", "C", "E", "A", "N", "org_isweekday", "org_isweekend",
             "starttime", "endtime", "user", "day", "week", "insider"}

TEST_PKL = P.DATA_DIR / "user_split_formlp" / "dayr4.2-percentile30-test.pkl"
SHAP_CSV = P.OUT / "shap_attributions.csv"


class CaseNotFound(KeyError):
    """(user, day) is not part of the frozen held-out test split."""


class CaseService:
    """Read-only case lookups + live Tier-1/Tier-2 inference (frozen config)."""

    def __init__(self):
        self._bundle = None
        self._meta_bundle = None
        self._test_df = None
        self._test_keys: set[tuple[int, int]] = set()
        self._shap: dict[tuple[int, int], list[dict]] = {}
        self._llm: dict[tuple[int, int], dict] = {}
        self._frozen_t2: dict[tuple[int, int], dict] = {}
        self._builder = None

    # -- lazy loaders (each artifact loaded at most once) --------------------
    def bundle(self) -> dict:
        if self._bundle is None:
            self._bundle = P.load_tier1()
        return self._bundle

    def meta_bundle(self) -> dict:
        """Feature-column ordering ONLY (no model weights) from frozen meta.json.

        Used by the microservice orchestrator to assemble feature rows in the
        canonical order before delegating inference to the Tier-1 agents. The
        frozen feature_cols ordering is the contract; no inference happens here.
        """
        if self._meta_bundle is None:
            import json
            cols = {}
            for mod in P.CHANNELS:
                meta = json.loads((P.TIER1_DIR / mod / "meta.json").read_text())
                cols[mod] = list(meta["feature_cols"])
            self._meta_bundle = {"feature_cols": cols}
        return self._meta_bundle

    def test_df(self) -> pd.DataFrame:
        if self._test_df is None:
            df = pd.read_pickle(str(TEST_PKL))
            self._test_df = df
            self._test_keys = set(zip(df["user"].astype(int), df["day"].astype(int)))
        return self._test_df

    def has_case(self, user: int, day: int) -> bool:
        self.test_df()
        return (int(user), int(day)) in self._test_keys

    def shap_index(self) -> dict:
        if not self._shap:
            df = pd.read_csv(SHAP_CSV)
            df = df[~df["feature"].isin(META_COLS)]
            df = df[df["rank"] <= SHAP_TOP_K]
            for key, sub in df.groupby(["user_id", "date"]):
                self._shap[(int(key[0]), int(key[1]))] = [
                    {"channel": r.channel, "feature": r.feature,
                     "feature_value": float(r.feature_value),
                     "shap_value": float(r.shap_value),
                     "direction": r.direction, "rank": int(r.rank)}
                    for r in sub.sort_values(["channel", "rank"]).itertuples()
                ]
        return self._shap

    def llm_index(self) -> dict:
        if not self._llm:
            df = pd.read_csv(P.W4_TRIAGE_OUTPUTS)
            for r in df.itertuples():
                self._llm[(int(r.user), int(r.day))] = {
                    "explanation": r.explanation,
                    "severity_category": r.severity_category,
                    "recommended_action": r.recommended_action,
                    "channels_mentioned": _safe_list(r.channels_mentioned),
                    "evidence_used": _safe_list(r.evidence_used),
                }
        return self._llm

    def frozen_t2_index(self) -> dict:
        if not self._frozen_t2:
            df = pd.read_csv(P.W4_FROZEN_INPUT)
            for r in df.itertuples():
                self._frozen_t2[(int(r.user), int(r.day))] = {
                    "risk_score": float(r.risk_score), "escalate": int(r.escalate),
                    "gate_triggered": str(r.gate_triggered),
                }
        return self._frozen_t2

    def escalated_cases(self) -> list[dict]:
        """The 712 frozen escalated cases (dashboard selector)."""
        llm = self.llm_index()
        t2 = self.frozen_t2_index()
        out = []
        for (u, d), rec in sorted(llm.items()):
            frozen = t2.get((u, d), {})
            out.append({"user": u, "day": d,
                        "severity_category": rec["severity_category"],
                        "recommended_action": rec["recommended_action"],
                        "risk_score": frozen.get("risk_score"),
                        "gate": frozen.get("gate_triggered")})
        return out

    # -- startup warm-up (single-container deployment) ----------------------
    def warm(self) -> dict:
        """Eagerly load EVERY artifact the /case path needs (fail-fast).

        Used by the orchestrator lifespan in DEPLOYMENT_MODE=single so that
        no model/scaler/data loading is deferred to the first dashboard
        click: any failure surfaces at startup (refuse to serve) instead of
        as a runtime 500, and every later /case is pure in-memory work.

        Also pre-warms the canonical Week-4 EvidenceBuilder once with all
        held-out test users (single read of the frozen percentile data).
        Read-only — never modifies an artifact; outputs are unchanged.
        """
        bundle = self.bundle()                     # 4 frozen models + scalers
        df = self.test_df()                        # frozen held-out test split
        shap = self.shap_index()                   # frozen Week-5 SHAP lookup
        llm = self.llm_index()                     # frozen Week-4 LLM lookup
        t2 = self.frozen_t2_index()                # frozen Week-4 Tier-2 record
        builder = self.evidence_builder()          # canonical EvidenceBuilder
        return {
            "tier1_models": sorted(bundle["models"]),
            "tier1_scalers": sorted(bundle["scalers"]),
            "test_rows": int(len(df)),
            "shap_cases": len(shap),
            "llm_cases": len(llm),
            "frozen_tier2_cases": len(t2),
            "evidence_builder": builder.__class__.__name__,
        }

    def evidence_builder(self):
        """The canonical EvidenceBuilder, warmed ONCE for all test users.

        The frozen builder lazily re-reads the full percentile dataset for
        every user it has not seen; warming it with the whole held-out test
        split up front makes every /case a bounded in-memory lookup and
        removes repeated large-file handle churn from the request path.
        """
        if self._builder is None:
            self._builder = make_evidence_builder(top_k=3)
            self.test_df()  # ensure the held-out key set is populated
            users = sorted({u for u, _ in self._test_keys})
            if users:
                self._builder._ensure_loaded(users)  # read-only cache fill
        return self._builder

    # -- the case itself ------------------------------------------------------
    def build_case(self, user: int, day: int, tier1_scorer=None) -> dict:
        """Assemble the full case payload.

        tier1_scorer: optional callable(modality, {feature_name: value}) -> float.
        None (default) = in-process inference with the frozen local bundle
        (unchanged behavior). A scorer routes inference to the Tier-1
        microservice agents (Week-6 multi-container deployment); feature
        assembly still uses the frozen canonical column ordering.
        """
        user, day = int(user), int(day)
        if not self.has_case(user, day):
            raise CaseNotFound((user, day))

        df = self.test_df()
        row = df[(df["user"].astype(int) == user) & (df["day"].astype(int) == day)].iloc[0]

        # Tier-1 (live inference on the frozen percentile row, frozen models)
        cases_df = pd.DataFrame([{"user": user, "day": day}])
        if tier1_scorer is None:
            feats = assemble_feature_rows(df, cases_df, self.bundle())
            scored = score_cases(self.bundle(), feats).iloc[0]
            tier1 = {f"{c}_score": float(scored[f"{c}_score"]) for c in P.CHANNELS}
        else:
            feats = assemble_feature_rows(df, cases_df, self.meta_bundle())
            tier1 = {}
            for c in P.CHANNELS:
                cols = self.meta_bundle()["feature_cols"][c]
                arr = feats.iloc[0][f"{c}_features"]
                tier1[f"{c}_score"] = float(tier1_scorer(
                    c, {name: float(v) for name, v in zip(cols, arr)}))

        # Tier-2 (frozen gating, vectorized reference == LangGraph decision)
        ddf = P.tier2_decision(pd.DataFrame([{"user": float(user), "day": float(day),
                                              "week": float(row.get("week", 0)),
                                              **tier1}]))
        r = ddf.iloc[0]
        risk, escalate, gate = float(r["risk_score"]), bool(r["escalate"]), str(r["gate_triggered"])
        fired = T2._fired_channels({f"{c}_score": tier1[f"{c}_score"] for c in P.CHANNELS})

        # Consistency vs the frozen Tier-2 artifact (the scientific record).
        # The frozen CSV stores floats rounded to ~7 significant digits, so
        # match tolerance is 1e-6 (deployment-layer display check only).
        frozen = self.frozen_t2_index().get((user, day))
        consistency = None
        if frozen is not None:
            consistency = {
                "in_frozen_tier2": True,
                "frozen_risk_score": frozen["risk_score"],
                "risk_score_match": abs(risk - frozen["risk_score"]) < 1e-6,
                "escalation_match": escalate == bool(frozen["escalate"]),
            }
            consistency["consistent"] = (consistency["risk_score_match"]
                                         and consistency["escalation_match"])
        else:
            consistency = {"in_frozen_tier2": False, "consistent": True,
                           "note": "non-escalated test case; not in the frozen 712"}

        out = {
            "user": user, "day": day,
            "week": int(row["week"]) if "week" in row.index else None,
            "in_frozen_test_split": True,
            # Held-out test label — display-only, never used by any model.
            "insider_ground_truth": int(row["insider"]) if "insider" in row.index else None,
            "tier1_scores": tier1,
            "fired_channels": fired,
            "tier2": {"risk_score": risk, "escalate": escalate, "gate": gate,
                      "weights": P.WEIGHTS, "hc_threshold": P.HC_THRESHOLD,
                      "corr_threshold": P.CORR_THRESHOLD,
                      "rationale": _rationale(gate, risk)},
            "frozen_consistency": consistency,
            "evidence": None, "shap_top_features": None, "llm_explanation": None,
        }

        if escalate:
            # Evidence: canonical Week-4 EvidenceBuilder, unchanged.
            ev = self.evidence_builder().build_evidence(pd.Series({
                "user": user, "day": day, "week": row.get("week", 0),
                "risk_score": risk, "gate_triggered": gate, **tier1}))
            out["evidence"] = {
                "fired_channels": ev["fired_channels"],
                "items": [{"id": it["id"], "channel": it["channel"],
                           "feature": it["feature"], "human_label": it["human_label"],
                           "baseline_percentile": it["baseline_percentile"],
                           "current_percentile": it["current_percentile"],
                           "delta_percentile": it["delta_percentile"],
                           "direction": it["direction"], "strength": it["strength"]}
                          for it in ev["feature_deltas"]],
            }
            # SHAP: frozen lookup only.
            out["shap_top_features"] = self.shap_index().get((user, day))
            # LLM explanation: frozen Week-4 lookup only (no live generation).
            out["llm_explanation"] = self.llm_index().get((user, day))
        return out


def _rationale(gate: str, risk: float) -> str:
    if gate == "high_confidence":
        return f"High-confidence escalation: HTTP anomaly score exceeds {P.HC_THRESHOLD}."
    if gate == "correlation":
        return f"Correlation escalation: weighted risk {risk:.3f} exceeds {P.CORR_THRESHOLD}."
    return f"Normal: weighted risk {risk:.3f} below all thresholds."


def _safe_list(v) -> list:
    """Frozen CSV list columns are stored as JSON/Python-literal strings."""
    if v is None or (isinstance(v, float) and v != v):
        return []
    if isinstance(v, list):
        return v
    try:
        out = json.loads(str(v))
        return out if isinstance(out, list) else []
    except (json.JSONDecodeError, ValueError):
        return []


# --- module-level singleton: data caches survive repeated app lifecycles -----
_SERVICE: CaseService | None = None


def get_case_service() -> CaseService:
    global _SERVICE
    if _SERVICE is None:
        _SERVICE = CaseService()
    return _SERVICE
