"""Scenario evaluation for the final Tier-3 module (Week 4 final).

CERT r4.2 held-out test evaluation. The official answer key (answers.tar.bz2,
scenarios.txt + insiders.csv) maps the insider codes as:
  1 = WikiLeaks uploader (uploads data to wikileaks.org)
  2 = job-hunter exfiltrator (job-site surfing + thumb-drive data theft)
  3 = keylogger sysadmin (downloads keylogger, transfers it to supervisor's PC)
The dayr4.2.csv `insider` column uses these codes, and the user-level split
(data/user_splits/*.npy) keeps ALL THREE scenarios represented in the held-out
test split (scenario 1: 49/555/564/620/723/848, scenario 2: 118/247/577/652/
767/900, scenario 3: 11/861). SCENARIO_MAP below therefore lists the held-out
test users of each scenario that have escalated malicious days in the frozen
712-case set; users 49/620/723 (scenario 1) are documented separately as
Tier-2 misses (their malicious days were not escalated).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

SCENARIO_MAP = {
    # Scenario 1 — WikiLeaks uploader (HTTP; leak-site uploads)
    555: "Scenario 1: WikiLeaks uploader (HTTP; leak-site uploads)",
    564: "Scenario 1: WikiLeaks uploader (HTTP; leak-site uploads)",
    848: "Scenario 1: WikiLeaks uploader (HTTP; leak-site uploads)",
    # Scenario 2 — job-hunter exfiltrator (email + file; job-site surfing)
    118: "Scenario 2: Job-hunter exfiltrator (email + file)",
    247: "Scenario 2: Job-hunter exfiltrator (email + file)",
    577: "Scenario 2: Job-hunter exfiltrator (email + file)",
    652: "Scenario 2: Job-hunter exfiltrator (email + file)",
    767: "Scenario 2: Job-hunter exfiltrator (email + file)",
    900: "Scenario 2: Job-hunter exfiltrator (email + file)",
    # Scenario 3 — keylogger sysadmin (file/device + HTTP)
    11: "Scenario 3: Keylogger sysadmin (file/device + HTTP)",
    861: "Scenario 3: Keylogger sysadmin (file/device + HTTP)",
}

EXPECTED_CHANNELS = {
    555: ["http"], 564: ["http"], 848: ["http"],
    118: ["email", "file_device"], 247: ["email", "file_device"],
    577: ["email", "file_device"], 652: ["email", "file_device"],
    767: ["email", "file_device"], 900: ["email", "file_device"],
    11: ["file_device", "http"], 861: ["file_device", "http"],
}


def evaluate_scenarios(results_df: pd.DataFrame) -> pd.DataFrame:
    """Per-scenario report from the triage results (user-level)."""
    rows = []
    for user, scen in SCENARIO_MAP.items():
        sub = results_df[results_df["user"] == user]
        rows.append({
            "scenario": scen,
            "user": user,
            "cases": int(len(sub)),
            "malicious": int((sub["label"] == 1).sum()),
            "gates": sub["gate_triggered"].value_counts().to_dict(),
            "fired_channels": sub["fired_channels"].value_counts().head(3).to_dict(),
            "mean_risk": float(sub["risk_score"].mean()),
            "cases_with_validated_obs": int(sub["observations"].apply(
                lambda x: len(json.loads(x)) if isinstance(x, str) and x.strip() else 0).sum()),
            "validated_evidence_ids": int(sub["validated_evidence_ids"].apply(
                lambda x: len(json.loads(x)) if isinstance(x, str) and x.strip() else 0).sum()),
            "unsupported_intent_terms": int(sub["explanation"].apply(
                lambda x: sum(1 for t in ["stole","theft","deliberately","intentionally","exfiltrated data","intended to leak","is malicious"] if t in (x or "").lower())).sum()),
            "severity": sub["severity_category"].value_counts().to_dict(),
            "actions": sub["recommended_action"].value_counts().to_dict(),
        })
    return pd.DataFrame(rows)


def scenario_report(results_df: pd.DataFrame, out_path: Path) -> None:
    """Human-readable + CSV scenario report."""
    df = evaluate_scenarios(results_df)
    df.to_csv(out_path, index=False)
    print(f"Scenario report saved -> {out_path}")
    for _, r in df.iterrows():
        print(f"  {r['scenario']} (user {r['user']}): {r['cases']} cases, "
              f"{r['malicious']} malicious, validated_obs={r['cases_with_validated_obs']}, "
              f"validated_ids={r['validated_evidence_ids']}, unsupported_terms={r['unsupported_intent_terms']}")
    # Document the Tier-2 escalation misses (scenario users whose malicious days
    # did NOT escalate, hence cannot be evaluated from the frozen 712-case set).
    print("Scenario-1 test users 49/620/723 are held-out TEST users whose malicious")
    print("days were NOT escalated by Tier-2 (Tier-2 misses); they are not in the")
    print("712-case input and are documented, not fabricated.")
