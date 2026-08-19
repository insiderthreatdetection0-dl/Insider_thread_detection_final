"""Tier-2 LangGraph fusion/escalation (final Week-5 implementation).

The SECURITY ESCALATION DECISION lives in Tier 2. This module implements the
real LangGraph orchestrator (sequential gating, mirroring the frozen Week-3
tier2_langgraph_orchestrator notebook) and the frozen configuration:

    HC threshold  = 0.60
    Corr threshold = 0.30
    HTTP 0.50 / File-Device 0.20 / Logon 0.15 / Email 0.15

`graph_decision` runs one case through the compiled LangGraph graph;
`tier2_decision` is the vectorized reference implementation in pipeline.py.
The two must agree exactly (0 mismatches) — verified by `equivalence_check`.
"""
from __future__ import annotations

from typing import Literal, TypedDict

import numpy as np
import pandas as pd
from langgraph.graph import END, START, StateGraph

from .pipeline import CHANNELS, CORR_THRESHOLD, HC_THRESHOLD, WEIGHTS, risk_score


class FusionState(TypedDict, total=False):
    """State passed through the LangGraph orchestrator (frozen schema)."""
    user: float
    day: float
    http_score: float
    file_device_score: float
    logon_score: float
    email_score: float
    risk_score: float
    escalate: bool
    gate_triggered: str
    rationale: str
    fired_channels: list
    degraded_channels: list
    degraded_mode: bool


def _fired_channels(state: FusionState) -> list:
    """Channels with score > CHANNEL_FIRE_THRESHOLD (0.3, explanation convention)."""
    return [c for c in CHANNELS if float(state[f"{c}_score"]) > 0.3]


def compute_risk_score(state: FusionState) -> dict:
    """Frozen weighted fusion (50/20/15/15)."""
    risk = (float(state["http_score"]) * WEIGHTS["http"] +
            float(state["file_device_score"]) * WEIGHTS["file_device"] +
            float(state["logon_score"]) * WEIGHTS["logon"] +
            float(state["email_score"]) * WEIGHTS["email"])
    return {"risk_score": risk}


def check_high_confidence(state: FusionState) -> dict:
    """Gate 1 (HTTP-only): escalate if http_score > HC_THRESHOLD."""
    if float(state["http_score"]) > HC_THRESHOLD:
        rationale = (
            f"HIGH-CONFIDENCE ESCALATION: HTTP anomaly score "
            f"({state['http_score']:.3f}) exceeds threshold ({HC_THRESHOLD}). "
            f"Strong single-channel behavioral deviation consistent with "
            f"insider-threat activity (e.g. bulk data exfiltration via http)."
        )
        return {"escalate": True, "rationale": rationale,
                "gate_triggered": "high_confidence",
                "fired_channels": _fired_channels(state)}
    return {"escalate": False, "rationale": "", "gate_triggered": "",
            "fired_channels": _fired_channels(state)}


def check_correlation(state: FusionState) -> dict:
    """Gate 2: escalate if frozen weighted risk > CORRELATION_THRESHOLD."""
    total = risk_score(pd.DataFrame([{f"{c}_score": state[f"{c}_score"] for c in CHANNELS}]))[0]
    if total > CORR_THRESHOLD:
        contributions = [f"{name}={float(state[f'{c}_score']):.3f}"
                         for c, name in [("http", "HTTP"), ("file_device", "File"),
                                         ("logon", "Logon"), ("email", "Email")]
                         if float(state[f"{c}_score"]) > 0.1]
        rationale = (
            f"CORRELATION ESCALATION: Weighted anomaly signal (weighted_sum={total:.3f}) "
            f"exceeds threshold ({CORR_THRESHOLD:.2f}). Multiple channels show "
            f"coordinated deviation: [{', '.join(contributions)}]. This pattern "
            f"suggests a multi-stage insider operation."
        )
        return {"escalate": True, "rationale": rationale,
                "gate_triggered": "correlation",
                "fired_channels": _fired_channels(state)}
    return {"escalate": False,
            "rationale": f"NORMAL: Weighted combined signal ({total:.3f}) below threshold.",
            "gate_triggered": "normal",
            "fired_channels": _fired_channels(state)}


def route_after_confidence(state: FusionState) -> Literal["escalated", "check_correlation"]:
    return "escalated" if state["escalate"] else "check_correlation"


def route_after_correlation(state: FusionState) -> Literal["escalated", "normal"]:
    return "escalated" if state["escalate"] else "normal"


def build_graph():
    """Compile the frozen LangGraph orchestrator (sequential gating preserved)."""
    builder = StateGraph(FusionState)
    builder.add_node("compute_risk", compute_risk_score)
    builder.add_node("check_high_confidence", check_high_confidence)
    builder.add_node("check_correlation", check_correlation)
    builder.add_edge(START, "compute_risk")
    builder.add_edge("compute_risk", "check_high_confidence")
    builder.add_conditional_edges(
        "check_high_confidence", route_after_confidence,
        {"escalated": END, "check_correlation": "check_correlation"})
    builder.add_conditional_edges(
        "check_correlation", route_after_correlation,
        {"escalated": END, "normal": END})
    return builder.compile()


_GRAPH = build_graph()


def graph_decision(row: pd.Series | dict) -> dict:
    """Run ONE case through the real LangGraph orchestrator.

    Returns the final FusionState dict (includes escalate/gate_triggered/
    risk_score/rationale/fired_channels).
    """
    init = {f"{c}_score": float(row[f"{c}_score"]) for c in CHANNELS}
    init["user"] = float(row.get("user", 0))
    init["day"] = float(row.get("day", 0))
    init.update({"escalate": False, "gate_triggered": "", "rationale": "",
                 "fired_channels": [], "degraded_mode": False,
                 "degraded_channels": []})
    return _GRAPH.invoke(init)


def equivalence_check(test_df: pd.DataFrame, n: int | None = None) -> dict:
    """Compare the real LangGraph run vs the vectorized reference.

    n: number of rows to compare (default: all). Returns a dict with
    mismatches and the number compared. Zero mismatches required.
    """
    sub = test_df if n is None else test_df.head(n)
    vec = risk_score(sub)
    hc = sub["http_score"] > HC_THRESHOLD
    corr = vec > CORR_THRESHOLD
    vec_esc = (hc | corr).astype(int)
    vec_gate = np.where(hc, "high_confidence", np.where(corr, "correlation", "normal"))

    mism = []
    for i, (_, row) in enumerate(sub.iterrows()):
        g = graph_decision(row)
        if int(g["escalate"]) != int(vec_esc.iloc[i]) or g["gate_triggered"] != vec_gate[i]:
            mism.append({"idx": i, "user": row["user"], "day": row["day"],
                         "graph_esc": int(g["escalate"]), "vec_esc": int(vec_esc.iloc[i]),
                         "graph_gate": g["gate_triggered"], "vec_gate": vec_gate[i]})
        if abs(float(g["risk_score"]) - float(vec.iloc[i])) > 1e-9:
            mism.append({"idx": i, "user": row["user"], "day": row["day"],
                         "graph_risk": float(g["risk_score"]), "vec_risk": float(vec.iloc[i])})
    return {"n_compared": len(sub), "n_mismatches": len(mism), "mismatches": mism[:20]}
