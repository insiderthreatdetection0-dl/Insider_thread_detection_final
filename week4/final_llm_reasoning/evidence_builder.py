"""Evidence Builder for the final Tier-3 LLM reasoning module (Week 4 final).

Prepares a structured, evidence-ID'd evidence set BEFORE the LLM sees it.
The LLM never discovers evidence — it only explains evidence already selected here.

Evidence items carry:
  id, channel, feature (raw), human_label, baseline_percentile, current_percentile,
  delta_percentile, direction (elevated/reduced/normal), strength, attribution (optional).

Selection is deterministic: top-K (default 3) features ranked by objective
evidence (attribution weight if provided, else |percentile deviation|), taken
only from channels that actually fired (score > CHANNEL_FIRE_THRESHOLD).

No raw logs are ever passed to the LLM. Percentile data is the centered
percentile-minus-50 representation (0 = 50th-percentile baseline).
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Human-readable feature labels (source of truth; definitions unchanged from
# the earlier improved_config module, now self-contained in the final impl.)
# ---------------------------------------------------------------------------
FEATURE_LABELS = {
    "http_n-leakf": "Leak site visits (WikiLeaks, etc.)",
    "http_n-hackf": "Hacking site visits",
    "http_n-jobf": "Job search site visits",
    "http_n-cloudf": "Cloud storage uploads",
    "http_n-wikif": "Wiki-specific page views",
    "http_mean_url_depth": "URL depth",
    "http_mean_url_len": "URL length",
    "http_mean_http_c_len": "HTTP content length",
    "http_mean_http_c_nwords": "HTTP content word count",
    "file_n-exef": "Executable file operations",
    "file_n-docf": "Document file operations",
    "file_n-txtf": "Text file operations",
    "usb_mean_usb_dur": "USB device duration",
    "usb_n_usb": "USB connection count",
    "email_n-Xemail": "External email sends",
    "email_mean_email_size": "Mean email size",
    "email_mean_n_exdes": "External recipients count",
    "email_mean_email_text_nwords": "Email text word count",
    "logon_n-afterhourlogon": "After-hours logons",
    "logon_n-workhourlogon": "Work-hours logons",
}


def get_feature_label(col: str) -> str:
    """Human-readable label; safe technical fallback if unknown."""
    if col in FEATURE_LABELS:
        return FEATURE_LABELS[col]
    return col.replace("_", " ")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CHANNEL_FIRE_THRESHOLD = 0.3      # fired-channel convention for EXPLANATION (not a Tier-2 decision)
TOP_K = 3
ELEVATION_THRESHOLD = 20          # |centered percentile| beyond ~20th/80th percentile

# Attribute IDs: deterministic per evidence item
def evidence_id(channel: str, idx: int) -> str:
    return f"{channel.upper()}_E{idx + 1}"


def _strength(delta_pct: float) -> str:
    if abs(delta_pct) >= 50:
        return "strong"
    if abs(delta_pct) >= 30:
        return "moderate"
    return "weak"


class EvidenceBuilder:
    """Build deterministic evidence objects for escalated user/day cases."""

    def __init__(self, percentile_path: Path, feature_cols_by_modality: dict,
                 top_k: int = TOP_K):
        self.percentile_path = Path(percentile_path)
        self.fcm = feature_cols_by_modality
        self.top_k = top_k
        # Load percentile data lazily (filtered to needed users on first use)
        self._df_pct: pd.DataFrame | None = None

    def _ensure_loaded(self, users) -> None:
        """Lazily load percentile rows for the requested users, accumulating
        across calls (a builder may be reused for many users)."""
        users = [u for u in users if u is not None]
        missing = users
        if self._df_pct is not None:
            have = set(self._df_pct["user"].unique())
            missing = [u for u in users if u not in have]
        if not missing:
            return
        full = pd.read_pickle(str(self.percentile_path))
        add = full[full["user"].isin(missing)].copy()
        if self._df_pct is None:
            self._df_pct = add
        else:
            self._df_pct = pd.concat([self._df_pct, add], ignore_index=True)
        del full

    def build_evidence(self, row: pd.Series, attribution: dict | None = None) -> dict:
        """Build the structured evidence object for one escalated case.

        row: a Tier-2 prediction row with user/day/week/risk_score/gate_triggered
             and the four channel scores.
        attribution: optional {feature_name: impact} from the Tier-1 attribution
                     step; used as the primary ranking signal when present.
        """
        self._ensure_loaded([row["user"]])
        scores = {
            "http": float(row["http_score"]),
            "file_device": float(row["file_device_score"]),
            "logon": float(row["logon_score"]),
            "email": float(row["email_score"]),
        }
        fired = [c for c, s in scores.items() if s > CHANNEL_FIRE_THRESHOLD]

        ud = self._df_pct[(self._df_pct["user"] == row["user"]) &
                          (self._df_pct["day"] == row["day"])]
        items = []
        if len(ud):
            r = ud.iloc[0]
            for ch in fired:
                for feat in self.fcm.get(ch, []):
                    if feat not in r.index:
                        continue
                    v = float(r[feat])
                    if abs(v) < ELEVATION_THRESHOLD:
                        continue  # not meaningfully elevated/reduced
                    current_pct = int(50 + v)          # centered -> raw percentile
                    delta_pct = int(v)                  # deviation from baseline (50)
                    items.append({
                        "channel": ch,
                        "feature": feat,
                        "human_label": get_feature_label(feat),
                        "baseline_percentile": 50,
                        "current_percentile": current_pct,
                        "delta_percentile": abs(delta_pct),
                        "direction": "elevated" if v > 0 else "reduced",
                        "strength": _strength(delta_pct),
                        "attribution": float(attribution.get(feat, 0.0)) if attribution else 0.0,
                    })

        # Deterministic ranking: attribution (desc) primary, then |delta| (desc),
        # then feature name (asc) for full determinism.
        items.sort(key=lambda x: (-x["attribution"], -x["delta_percentile"], x["feature"]))
        items = items[: self.top_k]

        # assign deterministic IDs
        for i, it in enumerate(items):
            it["id"] = evidence_id(it["channel"], i)

        return {
            "user": int(row["user"]),
            "day": int(row["day"]),
            "week": int(row["week"]),
            "risk_score": float(row["risk_score"]),
            "gate_triggered": row["gate_triggered"],
            "channel_scores": scores,
            "fired_channels": fired,
            "feature_deltas": items,
        }

    def evidence_text(self, ev: dict) -> str:
        """Human-readable rendering of the evidence for the LLM prompt."""
        lines = [
            f"User {ev['user']} on day {ev['day']} (week {ev['week']}):",
            f"FUSED RISK SCORE: {ev['risk_score']:.3f}",
            f"GATE TRIGGERED: {ev['gate_triggered']}",
            "CHANNEL ANOMALY SCORES (0=normal, 1=extreme):",
        ]
        for ch in ["http", "file_device", "logon", "email"]:
            flag = " << FIRED" if ev["channel_scores"][ch] > CHANNEL_FIRE_THRESHOLD else ""
            lines.append(f"- {ch}: {ev['channel_scores'][ch]:.3f}{flag}")
        lines.append("EVIDENCE ITEMS (use ONLY these; cite by ID):")
        if ev["feature_deltas"]:
            for it in ev["feature_deltas"]:
                lines.append(
                    f"- [{it['id']}] [{it['channel'].upper()}] {it['human_label']}: "
                    f"{it['current_percentile']}th percentile, "
                    f"{it['direction']} (delta {it['delta_percentile']}), "
                    f"strength={it['strength']}"
                )
        else:
            lines.append("- (no single feature exceeded the elevation threshold)")
        return "\n".join(lines)


def feature_cols_by_modality(all_cols) -> dict:
    """Deterministic per-modality feature columns (same keywords as the project)."""
    def cols(mod):
        if mod == "http":
            return sorted([c for c in all_cols if "http" in c or "url" in c])
        if mod == "file_device":
            kw = ["file", "device", "comp", "disk", "phof", "docf", "txtf", "exef", "usb"]
            return sorted([c for c in all_cols if any(k in c for k in kw)])
        if mod == "email":
            return sorted([c for c in all_cols if "email" in c or "mail" in c])
        if mod == "logon":
            return sorted([c for c in all_cols if "logon" in c or "logout" in c])
        return []
    return {m: cols(m) for m in ["http", "file_device", "email", "logon"]}
