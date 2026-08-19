"""SHAP-vs-LLM explanation comparison (final Week-5 implementation).

For each of the 712 frozen escalated cases, compares the channels/features the
Week-4 LLM explanation cites (via the canonical grounding validator's mention
extraction) against the ACTUAL SHAP top attribution from the Tier-1 models.

Three-way claim classification (scientifically fair — a valid Week-4
evidence-grounded claim is NOT automatically "unsupported" merely because that
feature is not among the top-k SHAP features):

  1. supported_by_shap      — mentioned AND in the case's top-k SHAP attribution
  2. evidence_grounded_only — mentioned AND in the canonical Week-4 evidence
                              inventory (feature_deltas) but NOT in top-k SHAP
  3. unsupported            — mentioned but supported by neither SHAP nor the
                              canonical evidence inventory

Semantics (important):
- SHAP `rank` in shap_attributions.csv is per (case, channel); the top-k window
  is therefore per (case, channel), and the channel top-k is the set of
  channels that have any feature in their own top-k.
- The models include non-behavioral organizational metadata columns
  (O/C/E/A/N — CERT r4.2 org codes). These are never shown to the LLM and are
  not part of the canonical evidence vocabulary, so they are EXCLUDED from the
  SHAP top-k comparison set (documented; they remain in shap_attributions.csv).
- Feature vocabulary is aligned through the canonical evidence-builder
  `feature_cols_by_modality`, which selects the same percentile columns the
  model consumes (verified 199/199 for http etc.).
"""
from __future__ import annotations

import json

import pandas as pd

from ..evidence_builder import feature_cols_by_modality
from ..grounding_validator import _mentioned_channels, _mentioned_features
from ..pipeline import DATA_DIR, OUT, W4_FROZEN_INPUT, W4_TRIAGE_OUTPUTS
from ..tier3 import make_evidence_builder

TOP_K = 5  # SHAP top-k window (per channel) for "supported by SHAP" claims

# Non-behavioral metadata columns present in the Tier-1 model inputs but never
# shown to the LLM / never part of the canonical evidence inventory.
META_COLS = {"role", "ITAdmin", "b_unit", "f_unit", "dept", "team",
             "O", "C", "E", "A", "N", "org_isweekday", "org_isweekend",
             "starttime", "endtime", "user", "day", "week", "insider"}


def _load_evidence_items() -> dict:
    """Rebuild the canonical Week-4 evidence inventory for every escalated case.

    Returns {(user, day): {"fired_channels": [...], "features": {feature: id}}}
    using the frozen EvidenceBuilder (deterministic, evidence-grounded).
    """
    t2 = pd.read_csv(W4_FROZEN_INPUT)
    esc = t2[t2["escalate"] == 1]
    builder = make_evidence_builder(top_k=3)
    out = {}
    for _, r in esc.iterrows():
        ev = builder.build_evidence(r)
        out[(int(r["user"]), int(r["day"]))] = {
            "fired_channels": ev["fired_channels"],
            "features": {it["feature"]: it["id"] for it in ev["feature_deltas"]},
        }
    return out


def _topk_channels(attr_df: pd.DataFrame, case_key) -> set:
    """Channels that have at least one behavioral feature in their per-channel
    top-TOP_K SHAP window."""
    sub = attr_df[(attr_df["user_id"] == case_key[0]) & (attr_df["date"] == case_key[1])]
    return set(sub[(sub["rank"] <= TOP_K) & (~sub["feature"].isin(META_COLS))]["channel"].unique())


def _topk_features(attr_df: pd.DataFrame, case_key) -> set:
    """Behavioral features in the per-(case,channel) top-TOP_K SHAP window."""
    sub = attr_df[(attr_df["user_id"] == case_key[0]) & (attr_df["date"] == case_key[1])]
    return set(sub[(sub["rank"] <= TOP_K) & (~sub["feature"].isin(META_COLS))]["feature"].unique())


def _prf(inter, mentioned, relevant):
    p = len(inter) / len(mentioned) if mentioned else 0.0
    r = len(inter) / len(relevant) if relevant else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def _jaccard(a: set, b: set) -> float:
    u = a | b
    return len(a & b) / len(u) if u else 0.0


def compare_all(attr_df: pd.DataFrame) -> pd.DataFrame:
    triage = pd.read_csv(W4_TRIAGE_OUTPUTS)
    evidence = _load_evidence_items()

    rows = []
    for _, c in triage.iterrows():
        key = (int(c["user"]), int(c["day"]))
        expl = str(c["explanation"])
        ev = evidence.get(key)
        if ev is None:
            continue

        llm_ch = _mentioned_channels(expl)
        # features the LLM cited: match via canonical evidence vocabulary
        # (technical names + human labels), exactly as the grounding validator does
        builder_items = [{"feature": f, "human_label": f.replace("_", " ")}
                         for f in ev["features"]]
        llm_feat = _mentioned_features(expl, builder_items)
        # also match against the raw feature names of the evidence inventory
        llm_feat |= _mentioned_features(expl, [{"feature": f, "human_label": f}
                                               for f in ev["features"]])

        shap_ch = _topk_channels(attr_df, key)
        shap_feat = _topk_features(attr_df, key)
        ev_feat = set(ev["features"].keys())

        # three-way classification at the feature level
        supported_by_shap = llm_feat & shap_feat
        evidence_grounded_only = (llm_feat & ev_feat) - shap_feat
        unsupported = llm_feat - shap_feat - ev_feat

        rows.append({
            "case_id": f"{key[0]}_{key[1]}", "user_id": key[0], "date": key[1],
            "llm_channels": sorted(llm_ch), "shap_channels": sorted(shap_ch),
            "llm_features": sorted(llm_feat), "shap_features": sorted(shap_feat),
            "evidence_features": sorted(ev_feat),
            "supported_by_shap_features": sorted(supported_by_shap),
            "evidence_grounded_only_features": sorted(evidence_grounded_only),
            "unsupported_features": sorted(unsupported),
            "channel_precision": _prf(llm_ch & shap_ch, llm_ch, shap_ch)[0],
            "channel_recall": _prf(llm_ch & shap_ch, llm_ch, shap_ch)[1],
            "channel_f1": _prf(llm_ch & shap_ch, llm_ch, shap_ch)[2],
            "feature_precision": _prf(supported_by_shap, llm_feat, shap_feat)[0],
            "feature_recall": _prf(supported_by_shap, llm_feat, shap_feat)[1],
            "feature_f1": _prf(supported_by_shap, llm_feat, shap_feat)[2],
            "channel_jaccard": _jaccard(llm_ch, shap_ch),
            "feature_jaccard": _jaccard(llm_feat, shap_feat),
            "unsupported_feature_claims": len(unsupported),
            "n_llm_features": len(llm_feat),
        })
    return pd.DataFrame(rows)


def summarize(cmp: pd.DataFrame) -> dict:
    n = len(cmp)
    total_claims = int(cmp["n_llm_features"].sum())
    unsup = int(cmp["unsupported_feature_claims"].sum())
    sup_shap = int(cmp["supported_by_shap_features"].apply(len).sum())
    ev_only = int(cmp["evidence_grounded_only_features"].apply(len).sum())
    return {
        "n_cases": n,
        "channel_precision": float(cmp["channel_precision"].mean()),
        "channel_recall": float(cmp["channel_recall"].mean()),
        "channel_f1": float(cmp["channel_f1"].mean()),
        "feature_precision": float(cmp["feature_precision"].mean()),
        "feature_recall": float(cmp["feature_recall"].mean()),
        "feature_f1": float(cmp["feature_f1"].mean()),
        "channel_jaccard_mean": float(cmp["channel_jaccard"].mean()),
        "channel_jaccard_median": float(cmp["channel_jaccard"].median()),
        "feature_jaccard_mean": float(cmp["feature_jaccard"].mean()),
        "feature_jaccard_median": float(cmp["feature_jaccard"].median()),
        "total_llm_feature_claims": total_claims,
        "supported_by_shap_claims": sup_shap,
        "evidence_grounded_only_claims": ev_only,
        "unsupported_claims": unsup,
        "unsupported_claim_rate": unsup / total_claims if total_claims else 0.0,
        "supported_claim_rate": (sup_shap + ev_only) / total_claims if total_claims else 0.0,
    }


def main() -> dict:
    attr = pd.read_csv(OUT / "shap_attributions.csv")
    print(f"SHAP attributions: {len(attr):,} rows | TOP_K={TOP_K} (per channel) | "
          f"metadata cols excluded: {sorted(META_COLS)}")
    cmp = compare_all(attr)
    cmp.to_csv(OUT / "explanation_comparison.csv", index=False)
    s = summarize(cmp)
    print("\n=== SHAP vs LLM EXPLANATION COMPARISON ===")
    for k, v in s.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    with open(OUT / "explanation_comparison_summary.json", "w") as f:
        json.dump(s, f, indent=2, default=str)
    return s


if __name__ == "__main__":
    main()
