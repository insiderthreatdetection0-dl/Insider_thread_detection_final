"""Grounding Validator for the final Tier-3 LLM reasoning module.

Automatically validates every LLM response against the supplied evidence:

  - feature grounding: every mentioned feature exists in the evidence
  - channel grounding: every mentioned channel fired AND has supporting evidence
  - numeric grounding: numerical claims match supplied evidence within tolerance
  - direction grounding: direction (elevated/reduced) matches the evidence
  - evidence-ID validity: every evidence_used ID exists
  - unsupported claims: phrases that cannot be mapped to supplied evidence

Also computes attribution-style precision/recall/F1 for channels and features:
  precision = correctly-grounded mentioned / all mentioned
  recall    = correctly-grounded mentioned / relevant supplied evidence
(High precision > high recall: it is better to mention few, accurate items than
 many ungrounded ones — preserves the 41 -> 2 unsupported-claim improvement.)
"""
from __future__ import annotations

import re

# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Deterministic matching helpers
# ---------------------------------------------------------------------------
CHANNEL_TERMS = {
    "http": ["http", "web", "website", "browser", "leak site", "upload", "url", "cloud"],
    "file_device": ["file", "usb", "device", "removable", "executable", "document", "drive"],
    "logon": ["logon", "login", "authentication", "after-hours", "access"],
    "email": ["email", "mail", "external email", "attachment", "recipient"],
}

UNSUPPORTED_TERMS = [
    "uploaded confidential files", "stole company data", "intended to leak",
    "transferred source code", "personal cloud account", "exfiltrated data",
    "stole", "theft", "deliberately", "intentionally",
]


def _mentioned_channels(explanation: str) -> set[str]:
    expl = explanation.lower()
    out = set()
    for ch, terms in CHANNEL_TERMS.items():
        if any(t in expl for t in terms):
            out.add(ch)
    return out


def _validate_evidence_ids(evidence_used, valid_ids) -> bool:
    # The strict contract requires citing the evidence IDs actually used.
    # Empty evidence_used = contract violation (NOT treated as valid).
    if not evidence_used:
        return False
    return all(eid in valid_ids for eid in evidence_used)


# ---------------------------------------------------------------------------
# Evidence-ID extraction from explanation (robust fallback, P0-1)
# ---------------------------------------------------------------------------
EVIDENCE_ID_PATTERN = re.compile(r"\b([A-Z]+_E\d+)\b")


def extract_evidence_ids_from_explanation(explanation: str) -> list[str]:
    """Extract evidence-ID tokens (e.g. HTTP_E1, FILE_E3) from the explanation."""
    return EVIDENCE_ID_PATTERN.findall(explanation)


# ---------------------------------------------------------------------------
# Canonical feature representation (P0-2)
# ---------------------------------------------------------------------------
# Canonical mapping: technical feature name -> canonical key + alias tokens the
# LLM commonly uses. Built from the actual feature definitions (improved_config
# FEATURE_LABELS + evidence-builder keywords), NOT invented.
CANONICAL_FEATURES = {
    "http_mean_http_c_len": {"canonical": "http_content_length",
                             "aliases": ["content length", "http content length", "payload content length", "content-length"]},
    "http_mean_http_c_nwords": {"canonical": "http_content_word_count",
                                "aliases": ["word count", "content word count", "number of words", "words"]},
    "http_mean_url_len": {"canonical": "url_length", "aliases": ["url length"]},
    "http_mean_url_depth": {"canonical": "url_depth", "aliases": ["url depth"]},
    "http_n_leakf": {"canonical": "leak_site_visits", "aliases": ["leak site", "leak-site", "leaked"]},
    "http_n_jobf": {"canonical": "job_search_visits", "aliases": ["job search", "job site", "job function"]},
    "http_n_cloudf": {"canonical": "cloud_uploads", "aliases": ["cloud storage", "cloud upload"]},
    "file_n_exef": {"canonical": "executable_operations", "aliases": ["executable", "exe"]},
    "file_n_docf": {"canonical": "document_operations", "aliases": ["document"]},
    "file_n_txtf": {"canonical": "text_file_operations", "aliases": ["text file"]},
    "usb_n_usb": {"canonical": "usb_connections", "aliases": ["usb"]},
    "usb_mean_usb_dur": {"canonical": "usb_duration", "aliases": ["usb duration", "usb device"]},
    "email_n_xemail": {"canonical": "external_email_sends", "aliases": ["external email", "external mail", "xemail"]},
    "email_mean_email_size": {"canonical": "email_size", "aliases": ["email size", "mail size"]},
    "email_mean_n_exdes": {"canonical": "external_recipients", "aliases": ["external recipient", "recipient"]},
    "email_mean_email_text_nwords": {"canonical": "email_text_word_count", "aliases": ["text word count", "email text"]},
    "logon_n_afterhourlogon": {"canonical": "after_hours_logons", "aliases": ["after-hours", "after hours", "afterhour"]},
    "logon_n_workhourlogon": {"canonical": "work_hours_logons", "aliases": ["workhour", "work hours"]},
}

# Channel-level suffix aliases: any *_len / *_nwords / *_depth feature
# is described by the LLM with the corresponding generic phrase.
CANONICAL_SUFFIX_ALIASES = {
    "_len": ["length", "content length"],
    "_nwords": ["word count", "words"],
    "_depth": ["depth"],
    "_n_pc0": ["pc0"], "_n_pc1": ["pc1"], "_n_pc2": ["pc2"], "_n_pc3": ["pc3"],
}


def canonical_key_for_feature(feature: str) -> str:
    """Deterministic canonical key for a technical feature name."""
    f = feature.lower()
    if f in CANONICAL_FEATURES:
        return CANONICAL_FEATURES[f]["canonical"]
    # suffix-based fallback: http_mean_http_c_len -> http_content_length via _len
    for suffix, canon in CANONICAL_SUFFIX_ALIASES.items():
        if f.endswith(suffix):
            base = f[: -len(suffix)].replace("_", " ")
            return f"{base} {canon[0]}".replace(" ", "_")
    return f.replace("_", " ")


def alias_tokens_for_feature(feature: str) -> list[str]:
    """All known alias tokens (lowercased) for a technical feature name."""
    f = feature.lower()
    if f in CANONICAL_FEATURES:
        return CANONICAL_FEATURES[f]["aliases"]
    # derive generic aliases from the suffix (e.g. _len -> "length")
    tokens = []
    for suffix, canon in CANONICAL_SUFFIX_ALIASES.items():
        if f.endswith(suffix):
            tokens.extend(canon)
    return tokens


def normalize_explanation(explanation: str) -> str:
    """Lowercase + collapse punctuation for alias matching."""
    return re.sub(r"[-_]", " ", explanation.lower())


def _mentioned_features(explanation: str, evidence_items: list) -> set[str]:
    """Features mentioned in the explanation — canonical-aware (P0-2).

    A feature is 'mentioned' if the explanation contains:
      - its human label (technical, as before), OR
      - its raw name, OR
      - any of its known alias tokens / canonical key.
    This lets natural-language paraphrases match without an uncontrolled matcher.
    """
    expl_norm = normalize_explanation(explanation)
    out = set()
    for it in evidence_items:
        feat = it["feature"]
        label = it["human_label"].lower()
        raw = feat.lower()
        # direct matches (as before)
        if label in expl_norm or raw in expl_norm or raw.replace("_", " ") in expl_norm:
            out.add(feat)
            continue
        # canonical/alias match
        canon = canonical_key_for_feature(feat)
        if canon.replace("_", " ") in expl_norm:
            out.add(feat)
            continue
        for alias in alias_tokens_for_feature(feat):
            if alias in expl_norm:
                out.add(feat)
                break
    return out


# Feature-domain keywords used to detect feature mentions NOT supported by evidence.
_FEATURE_PATTERNS = [
    ("leak site", "leak"), ("upload",), ("usb",), ("removable",), ("executable",),
    ("document",), ("attachment",), ("external email",), ("recipient",),
    ("after-hours",), ("login",), ("job search",), ("cloud",), ("url depth",),
    ("content length",), ("word count",), ("email size",),
]


def _unsupported_feature_mentions(explanation: str, evidence_items: list) -> list[str]:
    """Feature-like phrases in the explanation that NO evidence item supports."""
    expl = explanation.lower()
    # what the evidence actually supports (labels + raw names)
    supported = set()
    for it in evidence_items:
        supported.add(it["human_label"].lower())
        supported.add(it["feature"].lower())
    hits = []
    for pat in _FEATURE_PATTERNS:
        key = pat[0]
        if key in expl:
            # check if any supported label/raw contains this key or its alias
            alias = pat[1] if len(pat) > 1 else key
            if not any(alias in s or key in s for s in supported):
                hits.append(key)
    return hits


def _validate_evidence_ids(evidence_used, valid_ids) -> bool:
    # The strict contract requires citing the evidence IDs actually used.
    # Empty evidence_used = contract violation (NOT treated as valid).
    if not evidence_used:
        return False
    return all(eid in valid_ids for eid in evidence_used)


class GroundingValidator:
    """Validates one LLM triage response against one evidence object."""

    def __init__(self):
        self.channel_terms = CHANNEL_TERMS
        self.unsupported_terms = UNSUPPORTED_TERMS

    def validate(self, ev: dict, explanation: str, severity: str, action: str,
                 evidence_used: list | None = None) -> dict:
        evidence_items = ev["feature_deltas"]
        valid_ids = {it["id"] for it in evidence_items}
        fired = set(ev["fired_channels"])

        expl_l = explanation.lower()

        # --- unsupported claims (narrative + invented feature mentions) ---
        unsupported = [t for t in self.unsupported_terms if t in expl_l]
        invented_features = _unsupported_feature_mentions(explanation, evidence_items)
        unsupported = list(dict.fromkeys(unsupported + invented_features))

        # --- channel grounding (precision/recall) ---
        mentioned_ch = _mentioned_channels(explanation)
        supported_ch = {ch for ch in mentioned_ch if ch in fired}
        # Empty fired set or empty mention must not be rewarded with 1.0.
        if not fired:
            ch_prec = 0.0
            ch_rec = 0.0
        elif not mentioned_ch:
            ch_prec = 0.0
            ch_rec = 0.0
        else:
            ch_prec = len(supported_ch) / len(mentioned_ch)
            ch_rec = len(supported_ch) / len(fired)
        ch_f1 = (2 * ch_prec * ch_rec / (ch_prec + ch_rec)) if (ch_prec + ch_rec) else 0.0

        # --- feature grounding (canonical-aware, precision/recall) ---
        mentioned_feat = _mentioned_features(explanation, evidence_items)
        supported_feat = {f for f in mentioned_feat if any(it["feature"] == f for it in evidence_items)}
        if not evidence_items:
            f_prec = 0.0
            f_rec = 0.0
        elif not mentioned_feat:
            f_prec = 0.0
            f_rec = 0.0
        else:
            f_prec = len(supported_feat) / len(mentioned_feat)
            f_rec = len(supported_feat) / len(evidence_items)
        f_f1 = (2 * f_prec * f_rec / (f_prec + f_rec)) if (f_prec + f_rec) else 0.0

        # --- numeric grounding: percentile claims must match supplied evidence ---
        numeric_valid = True
        ev_numbers = set()
        for it in evidence_items:
            ev_numbers.add(it["current_percentile"])
            ev_numbers.add(it["delta_percentile"])
        percentile_tokens = re.findall(r"(\d{1,3})\s*(?:th percentile|%|percentile)", expl_l)
        for tok in percentile_tokens:
            n = int(tok)
            if not any(abs(n - e) <= 5 for e in ev_numbers):
                numeric_valid = False
                break

        # --- direction grounding ---
        direction_valid = True
        for it in evidence_items:
            if it["direction"] == "elevated" and ("reduced" in expl_l and it["human_label"].lower()[:20] in expl_l):
                direction_valid = False
                break

        # --- evidence IDs (P0-1): separate contract compliance from grounding ---
        structured_used = list(evidence_used or [])
        contract_compliant = bool(structured_used) and all(e in valid_ids for e in structured_used)

        # fallback: extract IDs from the explanation
        explanation_ids = extract_evidence_ids_from_explanation(explanation)
        # grounding-valid ID set = structured IDs (if valid) OR explanation-extracted valid IDs
        if contract_compliant:
            grounded_ids = [e for e in structured_used if e in valid_ids]
            extraction_fallback = False
        else:
            grounded_ids = [e for e in explanation_ids if e in valid_ids]
            extraction_fallback = bool(grounded_ids)
        id_grounding_valid = bool(grounded_ids)
        # contract compliance (strict) — separate metric
        id_contract_valid = contract_compliant

        grounding_valid = (not unsupported) and (ch_prec >= 0.5) and (f_prec >= 0.5) and numeric_valid and id_grounding_valid

        expected_features = [it["feature"] for it in evidence_items]
        detected_features = sorted(mentioned_feat)
        matched_features = sorted(supported_feat)
        unmatched_expected = sorted(set(expected_features) - set(supported_feat))
        unmatched_detected = sorted(set(mentioned_feat) - set(supported_feat))

        return {
            "grounding_valid": bool(grounding_valid),
            "feature_grounding_valid": bool(not unsupported and f_prec >= 0.5),
            "channel_grounding_valid": bool(ch_prec >= 0.5),
            "numeric_grounding_valid": bool(numeric_valid),
            "direction_grounding_valid": bool(direction_valid),
            # P0-1: separated evidence-ID metrics
            "evidence_id_valid": bool(id_grounding_valid),          # grounding (fallback allowed)
            "evidence_id_contract_compliant": bool(id_contract_valid),  # strict structured field
            "evidence_id_extraction_fallback_used": bool(extraction_fallback),
            "structured_evidence_used": structured_used,
            "explanation_evidence_ids": explanation_ids,
            "grounded_evidence_ids": grounded_ids,
            "unsupported_claims": unsupported,
            "channel_precision": ch_prec,
            "channel_recall": ch_rec,
            "channel_f1": ch_f1,
            "feature_precision": f_prec,
            "feature_recall": f_rec,
            "feature_f1": f_f1,
            "mentioned_channels": sorted(mentioned_ch),
            "supported_channels": sorted(supported_ch),
            "mentioned_features": mentioned_feat,
            "supported_features": supported_feat,
            "expected_features": expected_features,
            "detected_features": detected_features,
            "matched_features": matched_features,
            "unmatched_expected_features": unmatched_expected,
            "unmatched_detected_features": unmatched_detected,
        }
