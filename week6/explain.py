"""Plain-language explanation builder (Week-6 deployment layer, presentation only).

Converts the FROZEN, already-grounded case data (fired channels, EvidenceBuilder
inventory, Tier-2 risk/gate, frozen LLM severity/action) into a short
human-readable summary. This is a PRESENTATION transform only:

    same evidence · same risk · same decision · better communication

Rules enforced here:
  - deterministic template composition — NO LLM, NO new facts;
  - every sentence is grounded in fields of the case payload (fired channels,
    evidence items, percentiles, risk score, gate, decision);
  - raw internal feature names never appear in the main summary: the frozen
    Week-4 label fallback (underscore-separated technical names) is translated
    here into verified human vocabulary, or replaced by a safe channel-level
    description when a feature's meaning cannot be verified;
  - the underlying evidence is untouched and stays visible separately (the
    dashboard's technical evidence section keeps the raw names for audit);
  - percentile wording reflects the actual methodology: the frozen centered
    percentile-minus-50 representation uses a FIXED 50th-percentile reference
    over the population reference distribution — NOT a per-user baseline.

Human vocabulary provenance (all tokens verified from the existing project):
  - curated labels: week4/final_llm_reasoning/evidence_builder.py FEATURE_LABELS
    and week5/final_pipeline/verification_*/canonical_mapping_manifest.json;
  - file-type categories (docf/compf/exef/txtf/phof/otherf) and HTTP site
    categories (socnetf/cloudf/jobf/leakf/hackf/otherf):
    week2/feature_extraction_mit.py extension/domain classification;
  - workstation codes n-pc0..3 ("own pc, sharedpc, other's pc, supervisor's
    pc"): week2/feature_extraction_mit.py from_pc();
  - disk identifiers n-disk*: week2/feature_extraction_mit.py file_process()
    disk code derived from the file path;
  - email vocabulary (n_des/n_atts/n_exdes/n_bccdes/Xemail/exbccmail/
    email_text_slen): week2/feature_extraction_mit.py email_feature_names.
Features whose tokens cannot be fully verified fall back to a channel-level
description instead of guessing a meaning.
"""
from __future__ import annotations

CHANNEL_NOUNS = {
    "http": "web (HTTP) activity",
    "file_device": "file and device activity",
    "logon": "log-on activity",
    "email": "email activity",
}

# Verified workstation codes — week2/feature_extraction_mit.py from_pc():
# "code: 0,1,2,3:  own pc, sharedpc, other's pc, supervisor's pc".
_WORKSTATION = {
    "0": "on the user's own workstation",
    "1": "on a shared workstation",
    "2": "on another person's workstation",
    "3": "on a supervisor's workstation",
}

# Verified HTTP site categories (feature_extraction_mit.py domain classes +
# frozen Week-4 FEATURE_LABELS).
_SITE_CAT = {
    "otherf": "other websites",
    "socnetf": "social-networking sites",
    "cloudf": "cloud-storage sites",
    "jobf": "job-search sites",
    "leakf": "leak sites",
    "hackf": "hacking sites",
    "wikif": "wiki pages",
}

# Verified file-type categories (feature_extraction_mit.py extension classes;
# docf/exef/txtf also carry frozen curated labels).
_FILE_CAT = {"docf": "document", "compf": "compressed", "exef": "executable",
             "txtf": "text", "phof": "photo", "otherf": "other"}

_WEB_MEASURE = {"http_c_len": "web-page content length",
                "http_c_nwords": "web-page content word count",
                "url_len": "URL length", "url_depth": "URL depth"}

_FILE_MEASURE = {"file_len": "file size", "file_nwords": "file word count",
                 "file_depth": "file folder depth"}

_EMAIL_MEASURE = {"email_size": "email size",
                  "email_text_slen": "email text length",
                  "email_text_nwords": "email text word count",
                  "n_des": "number of email recipients",
                  "n_exdes": "number of external email recipients",
                  "n_bccdes": "number of BCC'd email recipients",
                  "n_atts": "number of email attachments"}


def _direction_word(direction) -> str:
    d = str(direction or "").lower()
    if "elev" in d or "increase" in d or "high" in d or d == "up":
        return "higher than the reference level"
    if "reduc" in d or "decrease" in d or "low" in d or d == "down":
        return "lower than the reference level"
    return "different from the reference level"


def _strength_word(delta) -> str:
    """Deviation wording keyed to the frozen EvidenceBuilder strength bands
    (|centered percentile| >= 50 strong, >= 30 moderate, else weak)."""
    try:
        d = abs(float(delta))
    except (TypeError, ValueError):
        return "a notable"
    if d >= 50:
        return "an extreme"
    if d >= 30:
        return "a very large"
    if d >= 20:
        return "a noticeable"
    return "a modest"


def _join_nouns(nouns: list[str]) -> str:
    if len(nouns) <= 1:
        return " ".join(nouns)
    return ", ".join(nouns[:-1]) + f" and {nouns[-1]}"


def _cap(txt: str) -> str:
    return txt[:1].upper() + txt[1:] if txt else txt


def _ordinal(v) -> str:
    try:
        n = int(round(float(v)))
    except (TypeError, ValueError):
        return str(v)
    if 10 <= n % 100 <= 20:
        suf = "th"
    else:
        suf = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suf}"


# ---------------------------------------------------------------------------
# Deterministic raw-feature -> plain-language translation (presentation only)
# ---------------------------------------------------------------------------
def _translate_http(name: str) -> str | None:
    if name.startswith("http_"):
        rest = name[5:]
    elif name in ("n_http",):
        return "web-browsing volume"
    else:
        return None
    if rest.startswith("n_") and rest[2:] in _SITE_CAT:
        return f"visits to {_SITE_CAT[rest[2:]]}"
    if rest.startswith("n-pc") and rest[4:] in _WORKSTATION:
        return f"web-browsing activity {_WORKSTATION[rest[4:]]}"
    if rest.startswith("mean_") and rest[5:] in _WEB_MEASURE:
        return _WEB_MEASURE[rest[5:]]
    for cat, cat_txt in _SITE_CAT.items():
        prefix = f"{cat}_"
        if rest.startswith(prefix):
            inner = rest[len(prefix):]
            if inner.startswith("mean_") and inner[5:] in _WEB_MEASURE:
                return f"{_WEB_MEASURE[inner[5:]]} on {cat_txt}"
            if inner.startswith("n-pc") and inner[4:] in _WORKSTATION:
                return f"visits to {cat_txt} {_WORKSTATION[inner[4:]]}"
    return None


def _translate_file(name: str) -> str | None:
    if name == "n_file":
        return "overall file activity volume"
    if not name.startswith("file_"):
        return None
    rest = name[5:]
    if rest.startswith("n_") and rest[2:] in _FILE_CAT:
        return f"{_FILE_CAT[rest[2:]]} file operations"
    if rest.startswith("n-disk") and rest[6:].isdigit():
        return f"file operations involving disk {rest[6:]}"
    if rest.startswith("n-pc") and rest[4:] in _WORKSTATION:
        return f"file operations {_WORKSTATION[rest[4:]]}"
    if rest.startswith("mean_") and rest[5:] in _FILE_MEASURE:
        return _FILE_MEASURE[rest[5:]]
    for cat, cat_txt in _FILE_CAT.items():
        prefix = f"{cat}_"
        if rest.startswith(prefix):
            inner = rest[len(prefix):]
            if inner.startswith("mean_") and inner[5:] in _FILE_MEASURE:
                return f"{_FILE_MEASURE[inner[5:]]} of {cat_txt} files"
            if inner.startswith("n-disk") and inner[6:].isdigit():
                return f"{cat_txt} file operations involving disk {inner[6:]}"
            if inner.startswith("n-pc") and inner[4:] in _WORKSTATION:
                return f"{cat_txt} file operations {_WORKSTATION[inner[4:]]}"
    return None


def _translate_email(name: str) -> str | None:
    if name == "n_email":
        return "email volume"
    if not name.startswith("email_"):
        return None
    rest = name[6:]
    if rest.startswith("mean_") and rest[5:] in _EMAIL_MEASURE:
        return _EMAIL_MEASURE[rest[5:]]
    if rest == "n-Xemail1":
        return "emails sent to external addresses"
    if rest == "n-exbccmail1":
        return "emails BCC'd to external addresses"
    if rest.startswith("n-pc") and rest[4:] in _WORKSTATION:
        return f"email activity {_WORKSTATION[rest[4:]]}"
    return None


def _translate_logon(name: str) -> str | None:
    if name == "n_logon":
        return "log-on volume"
    if not name.startswith("logon_"):
        return None
    rest = name[6:]
    if rest.startswith("n-pc") and rest[4:] in _WORKSTATION:
        return f"log-ons {_WORKSTATION[rest[4:]]}"
    return None


def _translate_usb(name: str) -> str | None:
    if name == "n_usb":
        return "USB device connections"
    if not name.startswith("usb_"):
        return None
    rest = name[4:]
    if rest == "mean_usb_dur":
        return "USB connection duration"
    if rest.startswith("n-pc") and rest[4:] in _WORKSTATION:
        return f"USB connections {_WORKSTATION[rest[4:]]}"
    return None


_TRANSLATORS = (_translate_http, _translate_file, _translate_email,
                _translate_logon, _translate_usb)
_TIME_PREFIX = (("workhour", "during work hours"), ("afterhour", "after hours"))


def plain_label(feature: str, channel: str, provided_label: str | None) -> str:
    """Human-readable description of one evidence feature.

    - curated frozen labels (FEATURE_LABELS) pass through unchanged;
    - the frozen technical fallback (underscore-separated raw name) is
      translated with verified vocabulary only;
    - anything not fully verifiable degrades to the safe channel noun.
    Never returns a raw/internal feature name.
    """
    raw_fallback = str(feature).replace("_", " ")
    if provided_label and str(provided_label) != raw_fallback:
        return str(provided_label)  # curated frozen label — already human

    name, time_txt = str(feature), ""
    for prefix, txt in _TIME_PREFIX:
        if name.startswith(prefix):
            name, time_txt = name[len(prefix):], txt
            break
        if name.startswith(f"n_{prefix}"):  # count features: n_workhourlogon, ...
            name, time_txt = f"n_{name[len(prefix) + 2:]}", txt
            break
    for tr in _TRANSLATORS:
        core = tr(name)
        if core is not None:
            return f"{core} {time_txt}".strip() if time_txt else core
    return CHANNEL_NOUNS.get(channel, "user activity")  # safe, no guessing


def build_plain_explanation(case: dict) -> dict:
    """Compose the user-facing summary from an existing /case payload.

    Pure function of the payload: it never mutates the case, never changes
    the risk score / decision / frozen LLM output, and invents no facts.

    Returns {"why_flagged": str, "what_happened": str, "why_unusual": str,
             "why_it_matters": str | None, "conclusion": str,
             "evidence_cards": [{label, detail, id}]}.
    """
    tier2 = case.get("tier2") or {}
    risk = tier2.get("risk_score")
    escalate = bool(tier2.get("escalate"))
    gate = tier2.get("gate") or "normal"
    fired = case.get("fired_channels") or []
    evidence = (case.get("evidence") or {}).get("items") or []
    llm = case.get("llm_explanation") or {}

    channel_nouns = [CHANNEL_NOUNS.get(c, c) for c in fired]

    # --- non-escalated: short, factual, nothing invented --------------------
    if not escalate:
        why = ("This case was reviewed by the detection system and did not "
               "show a strong enough combination of unusual signals to be "
               "escalated.")
        return {
            "why_flagged": why,
            "what_happened": "The user's activity stayed within the expected "
                             "range of the reference behavior.",
            "why_unusual": "No extreme deviation was detected across the "
                           "monitored channels.",
            "why_it_matters": None,
            "conclusion": (f"The combined risk score was {risk:.3f}, below the "
                           "escalation thresholds, so no alert was raised."),
            "evidence_cards": [],
        }

    # --- 1. why flagged: short channel-level summary (no evidence detail) ----
    if len(channel_nouns) > 1:
        why_flagged = ("This case was flagged because the system detected "
                       "unusually strong signals across multiple types of user "
                       f"activity, particularly {_join_nouns(channel_nouns)}.")
    elif channel_nouns:
        why_flagged = ("This case was flagged because the system detected an "
                       f"unusually strong signal in the user's {channel_nouns[0]}.")
    else:
        why_flagged = ("This case was flagged because the combined activity "
                       "signals exceeded the escalation threshold.")

    # --- 2. what happened: affected activity areas (fired channels only) -----
    per_channel = {c: sum(1 for it in evidence if it.get("channel") == c)
                   for c in fired}
    bits = []
    for c, noun in zip(fired, channel_nouns):
        n = per_channel.get(c, 0)
        if n > 1:
            bits.append(f"{_cap(noun)} contained {n} extreme deviations.")
        elif n == 1:
            bits.append(f"{_cap(noun)} contained an extreme deviation.")
        else:
            bits.append(f"{_cap(noun)} showed an elevated anomaly score.")
    what = " ".join(bits) if bits else \
        "The user's activity showed unusually strong signals."

    # --- 3. why unusual: grounded ONLY in the supplied evidence inventory ----
    cards = []
    unusual_bits = []
    for it in evidence[:3]:
        label = plain_label(it.get("feature", ""), it.get("channel", ""),
                            it.get("human_label"))
        cur, base = it.get("current_percentile"), it.get("baseline_percentile")
        delta = it.get("delta_percentile",
                       (float(cur) - float(base)) if cur is not None
                       and base is not None else None)
        unusual_bits.append(
            f"{_strength_word(delta)} deviation in {label} — at the "
            f"{_ordinal(cur)} percentile of the reference behavior, "
            f"against an expected {_ordinal(base)}")
        cards.append({"id": it.get("id"), "label": label,
                      "detail": (f"{_direction_word(it.get('direction'))} — at the "
                                 f"{_ordinal(cur)} percentile of the reference "
                                 f"behavior (expected {_ordinal(base)})")})
    if unusual_bits:
        why_unusual = ("Compared with the reference behavior: "
                       + "; ".join(unusual_bits) + ".")
    elif channel_nouns:
        why_unusual = (f"The {_join_nouns(channel_nouns)} deviated clearly "
                       "from the reference behavior.")
    else:
        why_unusual = "The activity deviated clearly from the reference behavior."

    # --- 4. why it matters (only when supported by the fused-risk record) ----
    matters = (f"These unusual activity patterns contributed to an elevated "
               f"overall risk score of {risk:.3f}.") if risk is not None else None

    # --- 5. conclusion: existing gate/escalation + frozen LLM assessment -----
    if gate == "high_confidence":
        gate_txt = "high-confidence alert rule"
    elif gate == "correlation":
        gate_txt = "combined-risk alert rule"
    else:
        gate_txt = "alert rule"
    conclusion = (f"The combined activity signals produced a risk score of "
                  f"{risk:.3f} and triggered the {gate_txt}, so the case was "
                  "escalated for human review.")
    sev, act = llm.get("severity_category"), llm.get("recommended_action")
    if sev:
        conclusion += (f" The frozen analyst assessment classified the case as "
                       f"{sev} severity"
                       + (f" and recommended '{act.replace('_', ' ')}'" if act else "")
                       + ".")

    return {
        "why_flagged": why_flagged,
        "what_happened": what,
        "why_unusual": why_unusual,
        "why_it_matters": matters,
        "conclusion": conclusion,
        "evidence_cards": cards,
    }
