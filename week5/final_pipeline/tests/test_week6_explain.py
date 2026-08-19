"""Week-6 plain-language explanation tests (presentation layer only).

Proves that week6.explain.build_plain_explanation is a deterministic,
grounded PRESENTATION transform of the frozen evidence:

  - the main summary contains NO raw/internal feature names (real frozen
    case 247/259 carries exactly the leaky features named in the task:
    email_mean_email_text_slen, file_docf_n-disk0, file_docf_n-pc0);
  - it uses ONLY the evidence supplied to it (no invented channels/facts);
  - it never changes the risk score, the escalation decision, or the frozen
    LLM output (the input case is not mutated at all);
  - it is deterministic;
  - unknown features degrade to a safe channel-level description instead of
    leaking their raw names.

Raw feature names remain allowed in the dashboard's technical evidence
section (case["evidence"]) — that audit view is NOT part of the plain
summary and is intentionally untouched here.
"""
from __future__ import annotations

import copy

import pytest

from week6.case_service import get_case_service
from week6.explain import build_plain_explanation, plain_label

# The real frozen demo case — its evidence items carry exactly the raw
# features whose plain-language translation this task requires.
DEMO_USER, DEMO_DAY = 247, 259
LEAKY_FEATURES = ["email_mean_email_text_slen", "file_docf_n-disk0",
                   "file_docf_n-pc0"]

# Fragments that must never appear in the user-facing summary (raw names,
# underscore fallbacks, and internal token fragments).
FORBIDDEN = LEAKY_FEATURES + [f.replace("_", " ") for f in LEAKY_FEATURES] + [
    "slen", "n-disk", "n-pc", "docf", "afterhour", "workhour",
    "mean email", "this user's normal behavior",
]

SECTIONS = ["why_flagged", "what_happened", "why_unusual",
            "why_it_matters", "conclusion"]


def _plain_text(p: dict) -> str:
    parts = [p[s] for s in SECTIONS if p.get(s)]
    parts += [c["label"] for c in p.get("evidence_cards", [])]
    parts += [c["detail"] for c in p.get("evidence_cards", [])]
    return " ".join(parts).lower()


@pytest.fixture(scope="module")
def demo_case():
    return get_case_service().build_case(DEMO_USER, DEMO_DAY)


# ---------------------------------------------------------------------------
# Real frozen case: no raw feature names in the main summary
# ---------------------------------------------------------------------------
def test_real_case_evidence_really_contains_the_leaky_features(demo_case):
    """Precondition: the frozen evidence of the demo case carries exactly the
    raw features named in the task (so the no-leak test below is meaningful)."""
    feats = {it["feature"] for it in demo_case["evidence"]["items"]}
    assert set(LEAKY_FEATURES) == feats
    # ... and the frozen builder's own fallback labels are the raw names
    assert any(it["human_label"] in ("email mean email text slen",
                                     "file docf n-disk0", "file docf n-pc0")
               for it in demo_case["evidence"]["items"])


def test_real_case_plain_summary_contains_no_raw_feature_names(demo_case):
    p = build_plain_explanation(demo_case)
    text = _plain_text(p)
    for raw in FORBIDDEN:
        assert raw.lower() not in text, f"leaked into plain summary: {raw}"


def test_real_case_percentile_wording_matches_methodology(demo_case):
    """Baseline is the fixed 50th-percentile population reference (centered
    percentile-minus-50), NOT a per-user baseline — the wording must say so."""
    p = build_plain_explanation(demo_case)
    assert "reference behavior" in p["why_unusual"].lower()
    assert "this user's normal" not in _plain_text(p)


def test_real_case_every_section_adds_distinct_content(demo_case):
    p = build_plain_explanation(demo_case)
    # no section repeats another verbatim
    vals = [p[s] for s in SECTIONS if p.get(s)]
    assert len(set(vals)) == len(vals)
    # why_flagged is a short channel-level summary, not a copy of why_unusual
    assert p["why_unusual"] not in p["why_flagged"]
    assert "0.811" in p["why_it_matters"] and "0.811" in p["conclusion"]


def test_real_case_conclusion_uses_frozen_decision_and_llm_verbatim(demo_case):
    p = build_plain_explanation(demo_case)
    assert "high-confidence alert rule" in p["conclusion"]
    assert "escalated for human review" in p["conclusion"]
    # frozen LLM severity/action stated as the existing assessment, unchanged
    llm = demo_case["llm_explanation"]
    assert llm["severity_category"] in p["conclusion"]
    assert llm["recommended_action"].replace("_", " ") in p["conclusion"]


# ---------------------------------------------------------------------------
# Grounding + purity: only supplied evidence, no mutation, deterministic
# ---------------------------------------------------------------------------
def test_explanation_is_pure_and_deterministic(demo_case):
    snapshot = copy.deepcopy(demo_case)
    first = build_plain_explanation(demo_case)
    second = build_plain_explanation(demo_case)
    assert first == second                      # deterministic
    assert demo_case == snapshot                # case payload not mutated
    # risk score / decision / frozen LLM output untouched
    assert demo_case["tier2"]["risk_score"] == snapshot["tier2"]["risk_score"]
    assert demo_case["tier2"]["escalate"] == snapshot["tier2"]["escalate"]
    assert demo_case["llm_explanation"] == snapshot["llm_explanation"]
    assert demo_case["evidence"] == snapshot["evidence"]  # raw names kept for audit


def test_explanation_uses_only_supplied_evidence():
    """A synthetic case with ONE http evidence item must not mention any
    other channel's deviations."""
    case = {
        "tier2": {"risk_score": 0.42, "escalate": True, "gate": "correlation"},
        "fired_channels": ["http"],
        "evidence": {"fired_channels": ["http"], "items": [{
            "id": "HTTP_E1", "channel": "http", "feature": "http_mean_url_depth",
            "human_label": "URL depth", "baseline_percentile": 50,
            "current_percentile": 96, "delta_percentile": 46,
            "direction": "elevated", "strength": "moderate"}]},
        "llm_explanation": {},
    }
    p = build_plain_explanation(case)
    text = _plain_text(p)
    assert "url depth" in text
    assert "email" not in text and "file and device" not in text
    assert "combined-risk alert rule" in p["conclusion"]


def test_unknown_feature_degrades_to_safe_channel_description():
    """A feature whose meaning cannot be verified must NOT leak its raw name;
    the builder falls back to the plain channel noun instead."""
    label = plain_label("email_totally_unknown_x1", "email",
                        "email totally unknown x1")
    assert label == "email activity"
    assert "unknown" not in label and "x1" not in label


def test_named_leaky_features_translate_to_verified_human_wording():
    assert plain_label("email_mean_email_text_slen", "email",
                       "email mean email text slen") == "email text length"
    assert plain_label("file_docf_n-disk0", "file_device",
                       "file docf n-disk0") == \
        "document file operations involving disk 0"
    assert plain_label("file_docf_n-pc0", "file_device",
                       "file docf n-pc0") == \
        "document file operations on the user's own workstation"
    # curated frozen labels pass through unchanged
    assert plain_label("http_mean_http_c_len", "http",
                       "HTTP content length") == "HTTP content length"


def test_non_escalated_case_is_short_and_clean():
    case = {"tier2": {"risk_score": 0.05, "escalate": False, "gate": "normal"},
            "fired_channels": [], "evidence": None, "llm_explanation": None}
    p = build_plain_explanation(case)
    assert p["evidence_cards"] == [] and p["why_it_matters"] is None
    assert "did not" in p["why_flagged"] and "0.050" in p["conclusion"]
    assert "this user's normal" not in _plain_text(p)
