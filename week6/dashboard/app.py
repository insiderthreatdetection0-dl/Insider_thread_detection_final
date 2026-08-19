"""Week-6 Streamlit dashboard (deployment layer only).

Thin read-only viewer over the FastAPI service. ALL data comes from the
FastAPI endpoints (``/health``, ``/case``, ``/cases``) — NO pipeline logic,
NO artifact loading inside Streamlit. See WEEK6_IMPLEMENTATION_PLAN.md §D.

Run locally (API already up on :8000):

    streamlit run week6/dashboard/app.py
"""
from __future__ import annotations

import os

import httpx
import streamlit as st

API_URL = os.environ.get(
    "W6_API_URL", f"http://localhost:{os.environ.get('API_PORT', '8000')}"
).rstrip("/")
SHOW_GROUND_TRUTH_DEFAULT = os.environ.get("SHOW_GROUND_TRUTH", "false").lower() in ("1", "true", "yes")

CHANNELS = ["http", "file_device", "logon", "email"]
TIMEOUT = httpx.Timeout(120.0)  # first escalated case lazy-loads the full pkl
# The dashboard talks ONLY to the orchestrator — ambient HTTP(S)_PROXY env
# vars (corporate proxies) must never intercept this intra-stack traffic.
HTTP_KW = {"timeout": TIMEOUT, "trust_env": False}


# ------------------------------------------------------------------ API calls
@st.cache_data(ttl=60, show_spinner=False)
def api_health() -> dict | None:
    try:
        r = httpx.get(f"{API_URL}/health", **HTTP_KW)
        return r.json() if r.status_code == 200 else None
    except httpx.HTTPError:
        return None


@st.cache_data(ttl=300, show_spinner=False)
def api_cases() -> list[dict]:
    try:
        r = httpx.get(f"{API_URL}/cases", **HTTP_KW)
        return r.json().get("cases", []) if r.status_code == 200 else []
    except httpx.HTTPError:
        return []


def api_case(user: int, day: int) -> tuple[dict | None, str | None]:
    try:
        r = httpx.post(f"{API_URL}/case", json={"user": user, "day": day}, **HTTP_KW)
    except httpx.HTTPError as e:
        return None, f"API unreachable at {API_URL}: {e}"
    if r.status_code == 404:
        return None, r.json().get("detail", "case not found")
    if r.status_code != 200:
        return None, f"API error {r.status_code}: {r.text[:200]}"
    return r.json(), None


# ------------------------------------------------------------------- helpers
def _bar(label: str, value: float, threshold: float = 0.3) -> None:
    st.progress(min(value, 1.0), text=f"{label}: {value:.4f}")
    if value >= threshold:
        st.caption(f":red[fired ≥ {threshold}]")


def _evidence_table(evidence: dict) -> None:
    rows = [
        {
            "ID": it["id"],
            "Channel": it["channel"],
            "Feature": it["feature"],
            "Label": it["human_label"],
            "Baseline pct": it["baseline_percentile"],
            "Current pct": it["current_percentile"],
            "Δ percentile": it["delta_percentile"],
            "Direction": it["direction"],
            "Strength": it["strength"],
        }
        for it in evidence["items"]
    ]
    st.dataframe(rows, use_container_width=True, hide_index=True)


def _highlight_ids(text: str, ids: list[str]) -> str:
    out = text
    for ev_id in sorted(ids, key=len, reverse=True):
        out = out.replace(ev_id, f"**`{ev_id}`**")
    return out


# ---------------------------------------------------------------------- page
st.set_page_config(page_title="Insider Threat — Case Viewer (Week 6)", layout="wide")
st.title("Insider Threat Detection — Case Viewer")
st.caption(
    "Week-6 demo deployment: **single container** (FastAPI + Streamlit) over the "
    "frozen research pipeline. Frozen artifacts only — no live LLM generation."
)

health = api_health()
with st.sidebar:
    st.subheader("API status")
    if health is None:
        st.error(f"FastAPI unreachable at `{API_URL}`")
        st.stop()
    st.success(f"status: {health.get('status')}")
    if health.get("artifacts_verified"):
        st.success("frozen-artifact SHA-256 integrity: verified")
    else:
        st.error("frozen-artifact integrity NOT verified — do not trust output")

    st.subheader("Case selection")
    cases = api_cases()
    mode = st.radio("Select case via", ["Escalated case list", "Manual user/day"])

case, err = None, None
with st.sidebar:
    if mode == "Escalated case list" and cases:
        labels = {
            f"user {c['user']} / day {c['day']} — {c.get('severity_category') or '?'}": (
                c["user"], c["day"])
            for c in cases
        }
        choice = st.selectbox(
            f"Frozen escalated cases ({len(cases)})", list(labels),
            index=next((i for i, c in enumerate(cases)
                        if c["user"] == 247 and c["day"] == 259),
                       next((i for i, c in enumerate(cases)
                             if c["user"] == 247), 0)))
        sel_user, sel_day = labels[choice]
    else:
        sel_user = st.number_input("user", min_value=0, value=247)
        sel_day = st.number_input("day", min_value=0, value=259)
    if st.button("Load case", type="primary") or "loaded" not in st.session_state:
        st.session_state.loaded = (int(sel_user), int(sel_day))
    user, day = st.session_state.loaded
    with st.spinner(f"Building case user={user} day={day} …"):
        case, err = api_case(user, day)

if err:
    st.error(err)
    st.stop()

t2 = case["tier2"]

# ------------------------------------------------- panel 1: header + truth
st.subheader(f"1 · Case: user {case['user']} — day {case['day']} (week {case['week']})")
if case.get("insider_ground_truth") is not None:
    show_gt = st.toggle("Show insider ground truth", value=SHOW_GROUND_TRUTH_DEFAULT)
    if show_gt:
        st.warning("Ground truth — NOT available at inference time (held-out test label).")
        st.write("Insider label:", ":red[insider]" if case["insider_ground_truth"] else "normal")

# ----------------------------------------------- panel 2: tier-1 scores
st.subheader("2 · Tier-1 channel anomaly scores")
cols = st.columns(4)
for col, ch in zip(cols, CHANNELS):
    with col:
        _bar(ch, float(case["tier1_scores"][f"{ch}_score"]))

# ----------------------------------------------- panel 3: gating
st.subheader("3 · Channel gating")
fired = case["fired_channels"]
st.write("Fired channels (score ≥ 0.3):",
          ", ".join(f":red[{c}]" for c in fired) if fired else "none")
gate = t2["gate"]
if gate == "high_confidence":
    st.info(f"Gate path: **high-confidence** (HTTP score > {t2['hc_threshold']})")
elif gate == "correlation":
    st.info(f"Gate path: **correlation** (weighted risk > {t2['corr_threshold']})")
else:
    st.info("Gate path: none (below all thresholds)")

# ----------------------------------------------- panel 4: tier-2 fused risk
st.subheader("4 · Tier-2 fused risk")
w = t2["weights"]
st.write(
    f"Weighted risk = Σ wᵢ·scoreᵢ with frozen weights "
    f"http {w['http']:.2f} / file_device {w['file_device']:.2f} / "
    f"logon {w['logon']:.2f} / email {w['email']:.2f}"
)
st.metric("Tier-2 risk score", f"{t2['risk_score']:.4f}")
fc = case["frozen_consistency"]
if fc.get("in_frozen_tier2"):
    if fc["consistent"]:
        st.success(
            f"Matches frozen Tier-2 record (frozen risk {fc['frozen_risk_score']:.7f})")
    else:
        st.error("MISMATCH vs frozen Tier-2 record — investigate before demo")
else:
    st.caption(fc.get("note", "Case not part of the frozen escalated 712."))

# ----------------------------------------------- panel 5: escalation decision
st.subheader("5 · Escalation decision")
if t2["escalate"]:
    st.error(f"**ESCALATE** — {t2['rationale']}")
else:
    st.success(f"NO escalation — {t2['rationale']}")

if not t2["escalate"]:
    st.info("Non-escalated case: evidence/SHAP/LLM panels are not produced "
            "(frozen explanations exist only for the 712 escalated test cases).")
    plain = case.get("plain_explanation")
    if plain:
        st.subheader("Why was this case NOT flagged?")
        st.write(plain["why_flagged"])
        st.write(plain["conclusion"])
    st.stop()

# ------------------------------- plain-language summary (microservice deploy)
plain = case.get("plain_explanation")
if plain:
    st.subheader("Why was this case flagged? — plain-language summary")
    st.write(plain["why_flagged"])
    st.markdown(f"**What happened?** {plain['what_happened']}")
    st.markdown(f"**Why is it unusual?** {plain['why_unusual']}")
    if plain.get("why_it_matters"):
        st.markdown(f"**Why does it matter?** {plain['why_it_matters']}")
    st.markdown(f"**What did the system conclude?** {plain['conclusion']}")
    cards = plain.get("evidence_cards") or []
    if cards:
        cols = st.columns(len(cards))
        for col, card in zip(cols, cards):
            with col:
                st.info(f"**{card['label']}**  \n{card['detail']}  \n`{card['id']}`")
    st.caption("Raw feature names are intentionally kept out of this summary; "
               "they remain available in the technical evidence sections below.")


# ----------------------------------------------- panel 6: evidence inventory
st.subheader("6 · Evidence inventory (EvidenceBuilder, percentile deviations)")
if case.get("evidence"):
    _evidence_table(case["evidence"])
    evidence_ids = [it["id"] for it in case["evidence"]["items"]]
else:
    st.write("No evidence items.")
    evidence_ids = []

# ----------------------------------------------- panel 7: SHAP explanation
st.subheader("7 · SHAP explanation (frozen Week-5 attributions, lookup only)")
shap = case.get("shap_top_features") or []
if shap:
    for ch in CHANNELS:
        rows = [s for s in shap if s["channel"] == ch]
        if not rows:
            continue
        st.markdown(f"**{ch}**")
        st.dataframe(
            [{"rank": s["rank"], "feature": s["feature"],
              "value": round(s["feature_value"], 4),
              "SHAP": round(s["shap_value"], 5),
              "direction": s["direction"]} for s in rows],
            use_container_width=True, hide_index=True)
    st.caption("Model-attribution view (SHAP). Intentionally different from the "
               "EvidenceBuilder behavioral-deviation view above — not aligned.")
else:
    st.write("No frozen SHAP rows for this case.")

# ----------------------------------------------- panel 8: LLM explanation
st.subheader("8 · LLM explanation (frozen Week-4 output, lookup only)")
llm = case.get("llm_explanation")
if llm:
    cited = llm.get("evidence_used") or []
    st.write("Evidence cited by the LLM:", ", ".join(f"`{i}`" for i in cited) or "—")
    st.markdown(_highlight_ids(llm["explanation"], set(cited) | set(evidence_ids)))
else:
    st.write("No frozen LLM explanation for this case.")

# ----------------------------------------------- panel 9: recommended action
st.subheader("9 · Recommended action")
if llm:
    sev = llm.get("severity_category") or "?"
    act = llm.get("recommended_action") or "?"
    color = {"high": "red", "medium": "orange", "low": "green"}.get(sev, "blue")
    st.markdown(f":{color}[**severity: {sev}**] &nbsp;·&nbsp; **action: {act}**")
    st.caption("Channels mentioned: " + (", ".join(llm.get("channels_mentioned") or []) or "—"))
else:
    st.write("—")
