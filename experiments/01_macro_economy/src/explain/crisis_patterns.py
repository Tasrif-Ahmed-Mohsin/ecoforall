"""Historical Crisis Pattern Matcher — ground crisis probabilities in specific precedents.

Takes the existing crisis classifier probability and enriches it with
specific historical country-years that had similar macro profiles and
DID (or did NOT) experience crises. The LLM explains causal patterns.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.explain.deepseek_handler import (
    _call_deepseek_api,
    get_deepseek_api_key,
    logret_to_human,
)
from src.explain.llm_narrative import _humanize_indicator
from src.harmonize.common import FEATURES_DIR

# Crisis indicator columns from the GMD panel (sovereign debt, currency, banking)
CRISIS_COLS = [
    "sov_debt_crisis", "currency_crisis", "banking_crisis",
    "crisis_any",  # union of the three
]

# Key vulnerability indicators and their danger thresholds
VULNERABILITY_INDICATORS = {
    "gov_debt_gdp":           {"name": "Government Debt / GDP",           "threshold": 80.0,  "direction": "above", "unit": "%"},
    "current_account_gdp":    {"name": "Current Account / GDP",           "threshold": -5.0,  "direction": "below", "unit": "%"},
    "inflation_rate":         {"name": "Inflation Rate",                  "threshold": 10.0,  "direction": "above", "unit": "%"},
    "short_rate":             {"name": "Short-Term Interest Rate",        "threshold": 15.0,  "direction": "above", "unit": "%"},
    "unemployment_rate":      {"name": "Unemployment Rate",               "threshold": 15.0,  "direction": "above", "unit": "%"},
}


def _check_vulnerability(row: pd.Series) -> list[dict[str, Any]]:
    """Check a country-year row against known vulnerability thresholds.

    Returns a list of indicator assessments: {name, value, threshold, status}.
    """
    assessments = []
    for col, info in VULNERABILITY_INDICATORS.items():
        val = row.get(col)
        if val is None or (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
            assessments.append({
                "name": info["name"],
                "value": None,
                "threshold": info["threshold"],
                "unit": info["unit"],
                "status": "unknown",
                "icon": "❓",
            })
            continue

        val = float(val)
        if info["direction"] == "above":
            danger = val > info["threshold"]
            warning = val > info["threshold"] * 0.7
        else:
            danger = val < info["threshold"]
            warning = val < info["threshold"] * 0.7

        if danger:
            status, icon = "danger", "🔴"
        elif warning:
            status, icon = "warning", "⚠️"
        else:
            status, icon = "safe", "✅"

        assessments.append({
            "name": info["name"],
            "value": val,
            "threshold": info["threshold"],
            "unit": info["unit"],
            "status": status,
            "icon": icon,
        })

    return assessments


def find_crisis_precedents(
    query_iso3: str,
    query_year: int,
    analogs_df: pd.DataFrame,
    panel: pd.DataFrame,
    horizon: int = 5,
    search_k: int = 20,
) -> dict[str, Any]:
    """Find historical countries with similar macro profiles that experienced crises.

    Uses the FAISS analogs (already similar in feature space) and checks
    which ones had a crisis event within the next `horizon` years.

    Returns:
      - crisis_precedents: list of {iso3, year, crisis_type, realized_growth}
      - non_crisis_precedents: list of {iso3, year, realized_growth}
      - vulnerability_check: list of indicator assessments for the query
      - crisis_rate: fraction of analogs that experienced a crisis
    """
    target_col = f"gdp_pc_growth_{horizon}y_fwd"
    if target_col not in analogs_df.columns and "gdp_pc_growth_5y_fwd" in analogs_df.columns:
        target_col = "gdp_pc_growth_5y_fwd"

    crisis_precedents = []
    non_crisis_precedents = []

    for _, analog in analogs_df.iterrows():
        a_iso3 = str(analog.get("iso3", ""))
        a_year = int(analog.get("year", 0))
        realized = analog.get(target_col)
        realized_human = logret_to_human(realized, horizon) if realized is not None else "N/A"

        # Look for crisis events in the panel for this analog in the forward window
        crisis_found = False
        crisis_types = []
        for future_year in range(a_year + 1, a_year + horizon + 1):
            future_row = panel[(panel.iso3 == a_iso3) & (panel.year == future_year)]
            if future_row.empty:
                continue
            for ccol in CRISIS_COLS:
                if ccol in future_row.columns:
                    cval = future_row.iloc[0].get(ccol)
                    if cval is not None and float(cval) == 1.0:
                        crisis_found = True
                        crisis_types.append(ccol.replace("_crisis", "").replace("_", " ").title())

        if crisis_found:
            crisis_precedents.append({
                "iso3": a_iso3,
                "year": a_year,
                "crisis_types": list(set(crisis_types)),
                "realized": float(realized) if realized is not None else None,
                "realized_human": realized_human,
            })
        else:
            non_crisis_precedents.append({
                "iso3": a_iso3,
                "year": a_year,
                "realized": float(realized) if realized is not None else None,
                "realized_human": realized_human,
            })

    # Vulnerability check on the query row
    query_row = panel[(panel.iso3 == query_iso3) & (panel.year == query_year)]
    vulnerability = _check_vulnerability(query_row.iloc[0]) if not query_row.empty else []

    n_total = len(crisis_precedents) + len(non_crisis_precedents)
    crisis_rate = len(crisis_precedents) / max(n_total, 1)

    return {
        "crisis_precedents": crisis_precedents,
        "non_crisis_precedents": non_crisis_precedents,
        "vulnerability_check": vulnerability,
        "crisis_rate": crisis_rate,
        "n_analogs_checked": n_total,
    }


def build_crisis_narrative(
    crisis_data: dict[str, Any],
    query_iso3: str,
    query_year: int,
    horizon: int = 5,
    crisis_prob: float | None = None,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Generate LLM narrative explaining crisis risk based on historical precedents.

    Returns:
      - narrative: LLM-generated or fallback crisis risk narrative
      - success: bool
    """
    # Build fallback narrative (always available)
    fallback = _build_crisis_fallback(crisis_data, query_iso3, query_year, horizon, crisis_prob)

    api_key = api_key or get_deepseek_api_key()
    if not api_key:
        return {"narrative": fallback, "success": False, "reason": "no_api_key"}

    # Build context for LLM
    crisis_lines = []
    for p in crisis_data.get("crisis_precedents", []):
        types = ", ".join(p.get("crisis_types", ["unknown"]))
        crisis_lines.append(
            f"  • {p['iso3']} ({p['year']}): experienced {types} crisis. "
            f"Realized {horizon}y growth: {p['realized_human']}"
        )
    crisis_block = "\n".join(crisis_lines) if crisis_lines else "  No crisis precedents found among analogs."

    safe_lines = []
    for p in crisis_data.get("non_crisis_precedents", []):
        safe_lines.append(
            f"  • {p['iso3']} ({p['year']}): no crisis. Realized {horizon}y growth: {p['realized_human']}"
        )
    safe_block = "\n".join(safe_lines) if safe_lines else "  No non-crisis precedents found."

    vuln_lines = []
    for v in crisis_data.get("vulnerability_check", []):
        val_str = f"{v['value']:.1f}{v['unit']}" if v["value"] is not None else "N/A"
        vuln_lines.append(
            f"  {v['icon']} {v['name']}: {val_str} (threshold: {v['threshold']}{v['unit']})"
        )
    vuln_block = "\n".join(vuln_lines) if vuln_lines else "  Vulnerability data unavailable."

    system_prompt = (
        "You are an expert in financial crises and sovereign risk analysis. "
        "You analyze historical crisis patterns to assess current vulnerability. "
        "Every number you cite must come from the data context. Do NOT invent examples. "
        "Be honest about uncertainty — if the evidence is mixed, say so."
    )

    prob_str = f"{crisis_prob:.1%}" if crisis_prob is not None else "N/A"
    user_prompt = f"""Assess crisis risk for {query_iso3} in {query_year} over the next {horizon} years.

MODEL CRISIS PROBABILITY: {prob_str}
HISTORICAL CRISIS RATE AMONG ANALOGS: {crisis_data.get('crisis_rate', 0):.0%} ({len(crisis_data.get('crisis_precedents', []))} of {crisis_data.get('n_analogs_checked', 0)} similar economies)

HISTORICAL CRISIS PRECEDENTS (similar economies that DID have crises):
{crisis_block}

HISTORICAL SURVIVORS (similar economies that did NOT have crises):
{safe_block}

CURRENT VULNERABILITY INDICATORS for {query_iso3} ({query_year}):
{vuln_block}

Provide a structured crisis risk assessment in 2 sections:
1. **CRISIS PATTERN ANALYSIS**: What do the historical precedents tell us? Which crisis types are most relevant? What protected the survivors?
2. **VULNERABILITY ASSESSMENT** for {query_iso3}: Based on the current indicators and historical patterns, what is the overall risk level? What are the key watch indicators?
"""

    try:
        response = _call_deepseek_api(system_prompt, user_prompt, api_key, temperature=0.0)
        return {"narrative": response, "success": True}
    except Exception as e:
        return {"narrative": fallback, "success": False, "reason": str(e)}


def _build_crisis_fallback(
    crisis_data: dict[str, Any],
    query_iso3: str,
    query_year: int,
    horizon: int,
    crisis_prob: float | None,
) -> str:
    """Deterministic crisis narrative when DeepSeek is unavailable."""
    lines = [f"## Crisis Risk Assessment: {query_iso3} ({query_year}), {horizon}-year horizon\n"]

    if crisis_prob is not None:
        lines.append(f"**Model crisis probability:** {crisis_prob:.1%}\n")

    rate = crisis_data.get("crisis_rate", 0)
    n = crisis_data.get("n_analogs_checked", 0)
    lines.append(f"**Historical crisis rate among analogs:** {rate:.0%} ({int(rate * n)} of {n})\n")

    if crisis_data.get("crisis_precedents"):
        lines.append("**Crisis precedents:**")
        for p in crisis_data["crisis_precedents"]:
            types = ", ".join(p.get("crisis_types", ["unknown"]))
            lines.append(f"- {p['iso3']} ({p['year']}): {types} — {p['realized_human']}")

    if crisis_data.get("vulnerability_check"):
        lines.append("\n**Vulnerability indicators:**")
        for v in crisis_data["vulnerability_check"]:
            val_str = f"{v['value']:.1f}{v['unit']}" if v["value"] is not None else "N/A"
            lines.append(f"- {v['icon']} {v['name']}: {val_str} (threshold: {v['threshold']}{v['unit']})")

    return "\n".join(lines)
