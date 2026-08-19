"""Week-6 microservice OUTPUT-EQUIVALENCE report generator (task §8/§9/§10).

Deterministically proves that the multi-container deployment reproduces the
frozen research system EXACTLY. It is a deployment-layer validation only — it
never retrains, never modifies artifacts, and never regenerates SHAP/LLM.

Three equivalence dimensions, all over the frozen held-out test split and the
frozen 712-case escalated Tier-2 record:

  §8  Tier-1 numerical equivalence: in-process `pipeline.score_mlp` vs each
      Tier-1 agent's HTTP `POST /score`, per modality. Expectation is
      bit-identity (same frozen weights/scaler/architecture/code path); the
      documented tolerance 1e-9 only guards JSON float round-trip.
  §9  Tier-2 decision equivalence: fired channels / gate / fused risk /
      escalation identical between local-score-driven and agent-score-driven
      `pipeline.tier2_decision`.
  §10 Frozen-record consistency: agent-path Tier-2 risk/escalate/gate match the
      frozen Week-4 scientific record (1e-6 — CSV stores ~7 significant digits).

Outputs (week6/reports/):
  tier1_equivalence.csv   per-modality samples / max / mean abs diff / PASS-FAIL
  tier2_equivalence.csv   per-case local vs agent escalate/gate/risk agreement
  equivalence_report.md   human-readable PASS/FAIL summary

Run:  $env:PYTHONPATH=<repo>; .\\.venv\\Scripts\\python.exe -m week6.equivalence_report
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from week5.final_pipeline import pipeline as P
from week5.final_pipeline import tier2 as T2
from week5.final_pipeline.tier1 import assemble_feature_rows, score_cases
from week6.tier1_agent import create_app

# Documented tolerances (see module docstring).
TIER1_TOL = 1e-9       # identical computation; JSON round-trip guard only
TIER2_RISK_TOL = 1e-12 # identical fused-risk arithmetic
FROZEN_TOL = 1e-6      # frozen CSV rounds floats to ~7 significant digits

REPORT_DIR = Path(__file__).resolve().parent / "reports"


def _agents() -> dict:
    """Independent TestClients for all four Tier-1 agents (lifespan started)."""
    clients = {}
    for mod in P.CHANNELS:
        c = TestClient(create_app(mod))
        c.__enter__()
        clients[mod] = c
    return clients


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    bundle = P.load_tier1()
    df = pd.read_pickle(str(P.DATA_DIR / "user_split_formlp"
                            / "dayr4.2-percentile30-test.pkl"))
    test_keys = set(zip(df["user"].astype(int), df["day"].astype(int)))

    # Frozen 712-case escalated Tier-2 scientific record.
    frozen = pd.read_csv(P.W4_FROZEN_INPUT)
    esc_records = frozen.query("escalate == 1")
    esc_cases = [(int(r.user), int(r.day)) for r in esc_records.itertuples()]
    esc_cases = sorted(set(esc_cases), key=lambda t: (t[0], t[1]))
    in_split = [c for c in esc_cases if c in test_keys]
    print(f"[eq] frozen escalated cases: {len(esc_cases)} | "
          f"present in test split: {len(in_split)}")

    # --- local (in-process) reference: one vectorized batch ---------------
    cases_df = pd.DataFrame(in_split, columns=["user", "day"])
    feats = assemble_feature_rows(df, cases_df, bundle)
    local_scores = score_cases(bundle, feats)

    # --- agent (microservice) scores over HTTP ----------------------------
    agents = _agents()
    n = len(in_split)
    agent_scores = {f"{m}_score": [0.0] * n for m in P.CHANNELS}
    for i, (user, day) in enumerate(in_split):
        for mod in P.CHANNELS:
            cols = bundle["feature_cols"][mod]
            arr = feats.iloc[i][f"{mod}_features"]
            fd = {name: float(v) for name, v in zip(cols, arr)}
            r = agents[mod].post("/score",
                                 json={"user": user, "day": day, "features": fd})
            if r.status_code != 200:
                raise RuntimeError(f"agent {mod} failed for ({user},{day}): {r.text}")
            agent_scores[f"{mod}_score"][i] = r.json()["score"]
        if (i + 1) % 100 == 0:
            print(f"[eq]   scored {i + 1}/{n}")

    # --- §8 Tier-1 numerical equivalence ----------------------------------
    t1_rows = []
    for mod in P.CHANNELS:
        col = f"{mod}_score"
        ref = local_scores[col].to_numpy(dtype=float)
        got = pd.Series(agent_scores[col], dtype=float).to_numpy()
        diff = abs(ref - got)
        ok = bool(diff.max() <= TIER1_TOL)
        t1_rows.append({"modality": mod, "samples": int(n),
                        "max_abs_diff": float(diff.max()),
                        "mean_abs_diff": float(diff.mean()),
                        "tolerance": TIER1_TOL,
                        "status": "PASS" if ok else "FAIL"})

    # --- §9 Tier-2 decision equivalence + §10 frozen-record consistency ----
    week_by_key = {(int(r.user), int(r.day)): getattr(r, "week", 0)
                   for r in df.itertuples()}
    frozen_by_key = {(int(r.user), int(r.day)): r for r in esc_records.itertuples()}

    base_local = pd.DataFrame({
        "user": [float(u) for u, _ in in_split],
        "day": [float(d) for _, d in in_split],
        "week": [float(week_by_key.get(k, 0)) for k in in_split],
        **{f"{m}_score": local_scores[f"{m}_score"].to_numpy(dtype=float)
           for m in P.CHANNELS}})
    base_agent = base_local.copy()
    for m in P.CHANNELS:
        base_agent[f"{m}_score"] = agent_scores[f"{m}_score"]

    t2_local = P.tier2_decision(base_local)
    t2_agent = P.tier2_decision(base_agent)

    t2_rows = []
    esc_match = gate_match = risk_match = frozen_match = fired_match = 0
    for i, key in enumerate(in_split):
        rl, ra = t2_local.iloc[i], t2_agent.iloc[i]
        fr = frozen_by_key.get(key)
        e_ok = int(rl["escalate"]) == int(ra["escalate"])
        g_ok = str(rl["gate_triggered"]) == str(ra["gate_triggered"])
        r_ok = abs(float(rl["risk_score"]) - float(ra["risk_score"])) <= TIER2_RISK_TOL
        fired_l = T2._fired_channels({f"{m}_score": float(local_scores.iloc[i][f"{m}_score"])
                                      for m in P.CHANNELS})
        fired_a = T2._fired_channels({f"{m}_score": float(agent_scores[f"{m}_score"][i])
                                      for m in P.CHANNELS})
        fc_ok = fired_l == fired_a
        f_ok = (fr is not None
                and abs(float(ra["risk_score"]) - float(fr.risk_score)) <= FROZEN_TOL
                and int(ra["escalate"]) == int(fr.escalate)
                and str(ra["gate_triggered"]) == str(fr.gate_triggered))
        esc_match += e_ok; gate_match += g_ok; risk_match += r_ok
        fired_match += fc_ok; frozen_match += f_ok
        if not (e_ok and g_ok and r_ok and fc_ok and f_ok):
            t2_rows.append({
                "user": key[0], "day": key[1],
                "local_escalate": int(rl["escalate"]), "agent_escalate": int(ra["escalate"]),
                "local_gate": str(rl["gate_triggered"]), "agent_gate": str(ra["gate_triggered"]),
                "local_fired": ";".join(fired_l), "agent_fired": ";".join(fired_a),
                "local_risk": float(rl["risk_score"]), "agent_risk": float(ra["risk_score"]),
                "frozen_risk": float(fr.risk_score) if fr is not None else None,
            })

    # --- write CSV artifacts ----------------------------------------------
    pd.DataFrame(t1_rows).to_csv(REPORT_DIR / "tier1_equivalence.csv", index=False)
    pd.DataFrame(t2_rows).to_csv(REPORT_DIR / "tier2_equivalence.csv", index=False)

    all_t1_pass = all(r["status"] == "PASS" for r in t1_rows)
    all_t2_pass = (esc_match == gate_match == risk_match == fired_match
                   == frozen_match == n)

    # --- write markdown summary -------------------------------------------
    lines = ["# Week-6 Microservice Output-Equivalence Report", "",
             "Deployment-layer validation that the multi-container microservice "
             "system reproduces the frozen research pipeline EXACTLY. No "
             "retraining / artifact modification / SHAP / LLM regeneration.", "",
             f"- Escalated frozen cases: **{len(esc_cases)}** | in held-out test "
             f"split: **{len(in_split)}**",
             f"- Tier-1 tolerance: `{TIER1_TOL}` (identical computation; JSON "
             f"round-trip guard)",
             f"- Tier-2 fused-risk tolerance: `{TIER2_RISK_TOL}`",
             f"- Frozen-record tolerance: `{FROZEN_TOL}` (CSV ~7 sig. digits)", "",
             "## §8 Tier-1 numerical equivalence (in-process vs agent HTTP)", "",
             "| Modality | Samples | Max abs diff | Mean abs diff | Tolerance | Status |",
             "|---|---|---|---|---|---|"]
    for r in t1_rows:
        lines.append(f"| {r['modality']} | {r['samples']} | "
                     f"{r['max_abs_diff']:.3e} | {r['mean_abs_diff']:.3e} | "
                     f"{r['tolerance']:.0e} | {r['status']} |")
    lines += ["", f"**Tier-1 overall: {'PASS' if all_t1_pass else 'FAIL'}**", "",
              "## §9/§10 Tier-2 decision + frozen-record equivalence", "",
              f"| Check | Matches | Cases | Status |",
              f"|---|---|---|---|",
              f"| Fired channels identical (local vs agent) | {fired_match} | {n} | "
              f"{'PASS' if fired_match == n else 'FAIL'} |",
              f"| Escalation identical (local vs agent) | {esc_match} | {n} | "
              f"{'PASS' if esc_match == n else 'FAIL'} |",
              f"| Gate identical (local vs agent) | {gate_match} | {n} | "
              f"{'PASS' if gate_match == n else 'FAIL'} |",
              f"| Fused risk identical (≤{TIER2_RISK_TOL:.0e}) | {risk_match} | {n} | "
              f"{'PASS' if risk_match == n else 'FAIL'} |",
              f"| Matches frozen Week-4 record (≤{FROZEN_TOL:.0e}) | {frozen_match} | {n} | "
              f"{'PASS' if frozen_match == n else 'FAIL'} |", "",
              f"**Tier-2 overall: {'PASS' if all_t2_pass else 'FAIL'}**", ""]
    if t2_rows:
        lines += [f"### Mismatches ({len(t2_rows)})", "",
                  "user,day,local_escalate,agent_escalate,local_gate,agent_gate,"
                  "local_fired,agent_fired,local_risk,agent_risk,frozen_risk"]
        for r in t2_rows[:50]:
            lines.append(f"{r['user']},{r['day']},{r['local_escalate']},"
                         f"{r['agent_escalate']},{r['local_gate']},{r['agent_gate']},"
                         f"{r['local_fired']},{r['agent_fired']},"
                         f"{r['local_risk']:.7f},{r['agent_risk']:.7f},"
                         f"{'' if r['frozen_risk'] is None else format(r['frozen_risk'], '.7f')}")
    else:
        lines.append("_No Tier-2 mismatches._")

    (REPORT_DIR / "equivalence_report.md").write_text("\n".join(lines) + "\n",
                                                        encoding="utf-8")

    print(f"[eq] Tier-1: {'PASS' if all_t1_pass else 'FAIL'} | "
          f"Tier-2: {'PASS' if all_t2_pass else 'FAIL'}")
    for r in t1_rows:
        print(f"[eq]   {r['modality']:<12} max={r['max_abs_diff']:.3e} "
              f"mean={r['mean_abs_diff']:.3e} {r['status']}")
    print(f"[eq] report -> {REPORT_DIR}")
    if not (all_t1_pass and all_t2_pass):
        raise SystemExit("EQUIVALENCE FAILED — do NOT deploy.")


if __name__ == "__main__":
    main()
