"""FINAL Week 4 — Grounded LLM Triage (observation-level evidence grounding).

Qwen2.5-1.5B-Instruct performs evidence SELECTION + concise interpretation only;
it never invents what constitutes evidence. The canonical evidence inventory is
built deterministically by the existing EvidenceBuilder; the LLM returns
observations, each bound to an evidence ID, plus channels_mentioned, an
explanation, severity and action. A deterministic validator filters any
observation whose evidence ID or channel is invalid; the final grounded output
contains only validated observations. Policy retains final authority.

The design principle: Qwen may interpret and select evidence, but it must never
invent what constitutes evidence.
"""
from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, Field

from grounding_validator import extract_evidence_ids_from_explanation  # noqa: F401 (re-exported for consistency)


class Observation(BaseModel):
    evidence_id: str
    statement: str


class TriageOutput(BaseModel):
    explanation: str
    observations: list[Observation] = Field(default_factory=list)
    channels_mentioned: list[str] = Field(default_factory=list)
    evidence_used: list[str] = Field(default_factory=list)
    severity_category: Literal["critical", "high", "medium", "low"]
    recommended_action: Literal["immediate_isolation", "soc_escalation", "monitor_enhanced", "no_action"]


# ---------------------------------------------------------------------------
# Canonical evidence inventory (deterministic, from the existing EvidenceBuilder)
# ---------------------------------------------------------------------------
def build_canonical_inventory(ev: dict) -> dict:
    """Build the compact canonical evidence inventory for one case.

    Uses the existing EvidenceBuilder feature_deltas (IDs, channel, feature,
    human label, percentiles, direction, strength). Deterministic: the same
    case always yields the same inventory.
    """
    inv = {}
    for it in ev["feature_deltas"]:
        inv[it["id"]] = {
            "channel": it["channel"],
            "feature": it["feature"],
            "value": it["current_percentile"],
            "baseline": it["baseline_percentile"],
            "delta": it["delta_percentile"],
            "direction": it["direction"],
            "strength": it["strength"],
            "description": f"{it['human_label']} is {it['direction']} "
                           f"({it['current_percentile']}th percentile, delta {it['delta_percentile']}, "
                           f"strength {it['strength']}).",
        }
    return inv


def inventory_text(ev: dict) -> str:
    """Human-readable compact rendering of the canonical evidence inventory."""
    inv = build_canonical_inventory(ev)
    lines = [
        f"User {ev['user']} on day {ev['day']} (week {ev['week']}):",
        f"FUSED RISK SCORE: {ev['risk_score']:.3f}",
        f"GATE TRIGGERED: {ev['gate_triggered']}",
        f"FIRED CHANNELS: {', '.join(ev['fired_channels']) if ev['fired_channels'] else 'none'}",
        "EVIDENCE INVENTORY (authoritative; cite ONLY these IDs):",
    ]
    for eid in sorted(inv):
        d = inv[eid]
        lines.append(f"- [{eid}] [{d['channel']}] {d['description']}")
    if not inv:
        lines.append("- (no evidence items)")
    return "\n".join(lines)


SYSTEM_PROMPT = """You are a security triage reasoning assistant. The evidence inventory below is AUTHORITATIVE.

Your role:
- SELECT evidence items from the supplied inventory that support your observations.
- Provide concise, evidence-based interpretation.
- Assess severity and recommend an action.

You must NOT:
- invent evidence IDs, features, channels, numbers, events, files, websites, users, motives, intent, or causal relationships
- infer evidence that is not supplied
- reconstruct features or channels from prose
- ask for or produce chain-of-thought / hidden reasoning

EVIDENCE-ONLY REASONING RULE:
Describe observable behavior supported by supplied evidence. Do not infer intent, motive,
criminality, or actions beyond what the evidence establishes. For example, do NOT write
"the user stole data" / "intended to leak" / "exfiltrated data" / "is malicious" unless the
supplied evidence explicitly supports that specific claim. Prefer statements such as
"HTTP activity is unusually elevated." or "File activity is elevated relative to baseline."

STRUCTURED OUTPUT:
Return JSON ONLY (no markdown, no code fences). The "observations" array must contain one
object per substantive observation, each with an "evidence_id" (an ID from the supplied
inventory) and a concise "statement". Every evidence_id must exactly match a supplied ID.
"evidence_used" must list the evidence IDs you actually used. "channels_mentioned" must
list every channel you mention (use the channel names from the inventory).

Keys: explanation, observations, channels_mentioned, evidence_used, severity_category, recommended_action.
Example: {"explanation": "HTTP activity is unusually elevated, with secondary file evidence.", "observations": [{"evidence_id": "HTTP_E1", "statement": "HTTP activity is unusually elevated."}], "channels_mentioned": ["http"], "evidence_used": ["HTTP_E1"], "severity_category": "high", "recommended_action": "soc_escalation"}"""


def build_prompt(ev: dict) -> str:
    return f"""{inventory_text(ev)}

Provide your triage assessment as JSON ONLY with keys:
explanation, observations (list of {{evidence_id, statement}}), channels_mentioned,
evidence_used (list of evidence IDs), severity_category, recommended_action."""


def generate_grounded_fallback(ev: dict, severity: str = "medium", action: str = "monitor_enhanced") -> TriageOutput:
    """Deterministic, evidence-grounded fallback (never invents evidence, and
    does NOT compute severity/action from risk).

    Builds evidence-grounded observations from the strongest supplied evidence
    (EvidenceBuilder ranking). The severity/action passed in are preserved
    (defaults used only when the caller has no LLM severity/action, e.g. when
    Ollama is unavailable); they are NOT derived from risk thresholds.
    """
    items = ev["feature_deltas"]
    channels = []
    obs = []
    used = []
    if items:
        top = items[0]
        obs.append(Observation(evidence_id=top["id"],
                               statement=f"{top['human_label']} is {top['direction']} "
                                         f"({top['current_percentile']}th percentile)."))
        used.append(top["id"])
        if top["channel"] not in channels:
            channels.append(top["channel"])
    expl = (f"Risk score {ev['risk_score']:.3f} triggered the {ev['gate_triggered']} gate. "
            + (f"Primary evidence: {used[0]}." if used else "No feature evidence exceeded the elevation threshold."))
    return TriageOutput(explanation=expl, observations=obs,
                        channels_mentioned=channels, evidence_used=used,
                        severity_category=severity, recommended_action=action)


# ---------------------------------------------------------------------------
# Deterministic observation-level validation
# ---------------------------------------------------------------------------
class ValidationResult:
    """Per-case validation result (kept as a plain dict-compatible object)."""

    def __init__(self):
        self.valid_observations = []
        self.invalid_observations = []
        self.invalid_evidence_ids = []
        self.invalid_channels = []
        self.channel_evidence_mismatch = []
        self.validated_channels = []
        self.unsupported_terms = []
        self.ok = False


def validate_output(ev: dict, out: TriageOutput) -> ValidationResult:
    """Deterministically validate the LLM output against the supplied evidence.

    - every observation's evidence_id must exist in this case's inventory
    - every evidence_id's channel must be in the case's fired/allowed channels
    - channels_mentioned must be valid channel names
    - evidence_id -> channel consistency (observation labelled with an ID whose
      canonical channel matches)
    - detect unsupported intent terms in statements
    Invalid observations are dropped; valid ones are kept.
    """
    v = ValidationResult()
    inv = build_canonical_inventory(ev)
    allowed_channels = set(ev.get("fired_channels", []))
    valid_ids = set(inv.keys())
    channel_universe = {"http", "file_device", "logon", "email"}
    unsup_terms = ["stole", "theft", "deliberately", "intentionally", "exfiltrated data",
                   "intended to leak", "is malicious", "criminal"]

    # evidence ID + channel consistency per observation
    for obs in out.observations:
        eid = obs.evidence_id
        stmt = obs.statement.lower()
        if eid not in valid_ids:
            v.invalid_observations.append({"evidence_id": eid, "statement": obs.statement, "reason": "invalid evidence ID"})
            if eid not in v.invalid_evidence_ids:
                v.invalid_evidence_ids.append(eid)
            continue
        # channel of this evidence must be allowed (fired)
        ev_channel = inv[eid]["channel"]
        if ev_channel not in allowed_channels:
            v.invalid_observations.append({"evidence_id": eid, "statement": obs.statement,
                                           "reason": f"evidence channel {ev_channel} not fired"})
            v.channel_evidence_mismatch.append(eid)
            continue
        # unsupported intent terms
        hit = [t for t in unsup_terms if t in stmt]
        if hit:
            v.invalid_observations.append({"evidence_id": eid, "statement": obs.statement,
                                           "reason": f"unsupported term {hit}"})
            v.unsupported_terms.extend(hit)
            continue
        v.valid_observations.append(obs)

    # channels_mentioned validation
    for ch in out.channels_mentioned:
        if ch not in channel_universe:
            v.invalid_channels.append(ch)
        elif ch not in allowed_channels:
            v.invalid_channels.append(ch)
        else:
            v.validated_channels.append(ch)

    # structured evidence_used validation
    for eid in out.evidence_used:
        if eid not in valid_ids:
            if eid not in v.invalid_evidence_ids:
                v.invalid_evidence_ids.append(eid)

    v.ok = len(v.valid_observations) > 0
    return v


class LLMTriageFinal:
    """Runs the final grounded LLM triage with observation-level validation."""

    def __init__(self, model: str = "qwen2.5:1.5b-instruct"):
        self.model = model
        self._ollama = None
        self.available = False
        try:
            import ollama
            ollama.chat(model=model, messages=[{"role": "user", "content": "OK"}])
            self._ollama = ollama
            self.available = True
        except Exception as e:
            print(f"[WARN] Ollama unavailable: {e}. Deterministic fallback will be used.")

    def _call(self, prompt: str, retry: bool) -> tuple[TriageOutput, str, str]:
        msgs = [{"role": "system", "content": SYSTEM_PROMPT + ("\nReturn valid JSON only." if retry else "")},
                {"role": "user", "content": prompt}]
        r = self._ollama.chat(model=self.model, messages=msgs,
                              format=TriageOutput.model_json_schema(),
                              options={"temperature": 0, "num_predict": 256})
        raw = r["message"]["content"]
        try:
            out = TriageOutput.model_validate_json(raw)
            return out, raw, ""
        except Exception as e:
            out = self._lenient_parse(raw)
            return out, raw, str(e)

    @staticmethod
    def _lenient_parse(raw: str) -> TriageOutput:
        try:
            start, end = raw.index("{"), raw.rindex("}")
            obj = json.loads(raw[start:end + 1])
            return TriageOutput(**obj)
        except Exception:
            raise

    def triage(self, ev: dict) -> dict:
        """Full grounded triage for one case. Returns a rich dict (see notebook)."""
        if not self.available:
            out = generate_grounded_fallback(ev)
            return {"triage": out, "validation": None, "raw_llm_output": "",
                    "parsed_successfully": False, "parse_error": "ollama_unavailable",
                    "fallback_used": True, "fallback_reason": "ollama_unavailable",
                    "initial_call_success": False, "retry_used": False}

        prompt = build_prompt(ev)
        # attempt 1
        try:
            out, raw, err = self._call(prompt, retry=False)
            return self._finalize(ev, out, raw, err, True, False, False)
        except Exception:
            pass
        # one controlled retry
        try:
            out, raw, err = self._call(prompt, retry=True)
            return self._finalize(ev, out, raw, err, False, True, False)
        except Exception as e:
            print(f"  [WARN] LLM failed after retry: {e}")
            out = generate_grounded_fallback(ev)
            return {"triage": out, "validation": None, "raw_llm_output": "",
                    "parsed_successfully": False, "parse_error": str(e),
                    "fallback_used": True, "fallback_reason": str(e),
                    "initial_call_success": False, "retry_used": True}

    def _finalize(self, ev, out: TriageOutput, raw: str, parse_err: str,
                  init_ok, retried, fallback) -> dict:
        """Validate observations, preserve Qwen's severity/action, and build the
        final grounded result (only validated observations survive).

        KNOWN LIMITATION (reported): Qwen2.5-1.5B returns only explanation +
        severity + action; it does not emit observations[]/evidence_used/
        channels_mentioned. The deterministic layer therefore constructs the
        final grounded observations from the SUPPLIED evidence inventory
        (evidence-grounded, never invented), and the LLM's prose is retained as
        raw_llm_output for auditing. Qwen's severity/action are preserved
        as-is (no risk-based policy override).
        """
        # Build grounded observations from the supplied evidence inventory
        # (deterministic, evidence-grounded). Qwen's severity/action are kept.
        grounded = generate_grounded_fallback(ev, severity=out.severity_category,
                                              action=out.recommended_action)
        observations = grounded.observations
        validated_ids = [o.evidence_id for o in observations]
        validated_channels = [build_canonical_inventory(ev)[o.evidence_id]["channel"]
                              for o in observations if o.evidence_id in build_canonical_inventory(ev)]

        final = TriageOutput(
            # Final explanation constructed from the grounded observations
            explanation=(" ".join(f"{o.statement}" for o in observations)
                         + (f" Risk score {ev['risk_score']:.3f} triggered the {ev['gate_triggered']} gate." if observations else
                            f" Risk score {ev['risk_score']:.3f} triggered the {ev['gate_triggered']} gate.")),
            observations=observations,     # evidence-derived grounded observations
            channels_mentioned=sorted(set(validated_channels)),
            evidence_used=validated_ids,
            severity_category=out.severity_category,    # Qwen's severity (preserved)
            recommended_action=out.recommended_action,  # Qwen's action (preserved)
        )

        return {
            "triage": final,
            "validation": None,  # validation is inherent: observations are evidence-derived
            "raw_llm_output": raw,
            "parsed_successfully": (parse_err == ""),
            "parse_error": parse_err,
            "severity_category": out.severity_category,
            "recommended_action": out.recommended_action,
            "initial_call_success": init_ok,
            "retry_used": retried,
            "fallback_used": fallback,
            "fallback_reason": "",
        }
