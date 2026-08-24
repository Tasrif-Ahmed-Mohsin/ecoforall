"""Historical Pattern Analysis Engine — Scenario Trees from FAISS Analogs.

This module transforms raw FAISS retrieval output into structured scenario
analysis grounded in the "history repeats" thesis. Instead of showing one
point forecast, it clusters historical analogs by realized outcome and
presents probability-weighted scenario trees.

The LLM adds causal reasoning ("why does this cluster represent a distinct
pathway?") — it does NOT predict. The predictions come from data.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from src.explain.deepseek_handler import (
    _call_deepseek_api,
    get_deepseek_api_key,
    logret_to_human,
)
from src.explain.llm_narrative import HUMAN_NAMES, _humanize_indicator


# ── Scenario labels by percentile bucket ──────────────────────────────
SCENARIO_LABELS = {
    "high":   "Growth Acceleration",
    "mid":    "Steady-State Growth",
    "low":    "Stagnation / Decline",
    "crisis": "Crisis / Deep Contraction",
}


def _get_target_col(horizon: int) -> str:
    return f"gdp_pc_growth_{horizon}y_fwd"


# ─────────────────────────────────────────────────────────────────────
# 1. Cluster analog outcomes
# ─────────────────────────────────────────────────────────────────────
def cluster_analog_outcomes(
    analogs_df: pd.DataFrame,
    horizon: int = 5,
    n_clusters: int = 3,
) -> list[dict[str, Any]]:
    """Cluster FAISS analogs by realized growth into distinct scenario groups.

    Uses simple percentile splits (not K-means) for stability on small K.
    Each cluster gets a label, probability weight, and member list.

    Returns a list of cluster dicts, each with:
      - label: human-readable scenario name
      - members: list of (iso3, year, realized_growth) tuples
      - mean_growth: average realized log-return in this cluster
      - mean_growth_human: human-readable growth string
      - probability: n_members / n_total
      - growth_range: (min, max) realized growth in cluster
    """
    target_col = _get_target_col(horizon)
    # Fall back to 5y target if the horizon-specific one isn't present
    if target_col not in analogs_df.columns and "gdp_pc_growth_5y_fwd" in analogs_df.columns:
        target_col = "gdp_pc_growth_5y_fwd"

    valid = analogs_df.dropna(subset=[target_col]).copy()
    if valid.empty:
        return []

    realized = valid[target_col].astype(float).values
    n = len(realized)

    # For very small analog sets, reduce cluster count
    effective_clusters = min(n_clusters, n)
    if effective_clusters < 2:
        # Single cluster — everything is one scenario
        mean_g = float(np.mean(realized))
        return [{
            "label": "Historical Average",
            "members": [
                {"iso3": r.iso3, "year": int(r.year), "realized": float(r[target_col])}
                for _, r in valid.iterrows()
            ],
            "mean_growth": mean_g,
            "mean_growth_human": logret_to_human(mean_g, horizon),
            "probability": 1.0,
            "growth_range": (float(np.min(realized)), float(np.max(realized))),
        }]

    # Percentile-based splits
    if effective_clusters == 2:
        boundaries = [np.median(realized)]
    else:  # 3 clusters: p33, p66
        boundaries = [np.percentile(realized, 33), np.percentile(realized, 66)]

    # Assign each analog to a bucket
    bucket_labels = ["low", "mid", "high"] if effective_clusters == 3 else ["low", "high"]
    # For 3 clusters: < p33 = low, p33-p66 = mid, > p66 = high
    bucket_indices = np.digitize(realized, boundaries)

    clusters = []
    for i in range(effective_clusters):
        mask = bucket_indices == i
        if not mask.any():
            continue
        members_df = valid.iloc[mask]
        member_realized = realized[mask]
        mean_g = float(np.mean(member_realized))
        label_key = bucket_labels[i] if i < len(bucket_labels) else "mid"

        # Override label if the cluster has deeply negative growth
        if mean_g < -0.10:  # > 10% contraction
            label_key = "crisis"

        clusters.append({
            "label": SCENARIO_LABELS.get(label_key, "Mixed"),
            "members": [
                {"iso3": r.iso3, "year": int(r.year), "realized": float(r[target_col])}
                for _, r in members_df.iterrows()
            ],
            "mean_growth": mean_g,
            "mean_growth_human": logret_to_human(mean_g, horizon),
            "probability": float(mask.sum()) / n,
            "growth_range": (float(np.min(member_realized)), float(np.max(member_realized))),
        })

    # Sort by mean_growth descending (best scenario first)
    clusters.sort(key=lambda c: -c["mean_growth"])
    return clusters


# ─────────────────────────────────────────────────────────────────────
# 2. Build full scenario tree
# ─────────────────────────────────────────────────────────────────────
def build_scenario_tree(
    iso3: str,
    year: int,
    horizon: int,
    analogs_df: pd.DataFrame,
    ml_ensemble: float | None = None,
    n_clusters: int = 3,
) -> dict[str, Any]:
    """Build a complete scenario tree from FAISS analogs.

    Returns:
      - scenarios: list of cluster dicts (from cluster_analog_outcomes)
      - weighted_forecast: probability-weighted average of scenario outcomes
      - weighted_forecast_human: human-readable
      - ml_ensemble: ML ensemble for comparison
      - ml_ensemble_human: human-readable
      - n_analogs: total number of analogs used
    """
    clusters = cluster_analog_outcomes(analogs_df, horizon=horizon, n_clusters=n_clusters)
    if not clusters:
        return {
            "scenarios": [],
            "weighted_forecast": ml_ensemble,
            "weighted_forecast_human": logret_to_human(ml_ensemble, horizon) if ml_ensemble else "N/A",
            "ml_ensemble": ml_ensemble,
            "ml_ensemble_human": logret_to_human(ml_ensemble, horizon) if ml_ensemble else "N/A",
            "n_analogs": 0,
        }

    # Probability-weighted scenario forecast
    weighted = sum(c["mean_growth"] * c["probability"] for c in clusters)

    return {
        "scenarios": clusters,
        "weighted_forecast": weighted,
        "weighted_forecast_human": logret_to_human(weighted, horizon),
        "ml_ensemble": ml_ensemble,
        "ml_ensemble_human": logret_to_human(ml_ensemble, horizon) if ml_ensemble else "N/A",
        "n_analogs": sum(len(c["members"]) for c in clusters),
        "query": {"iso3": iso3, "year": year, "horizon": horizon},
    }


# ─────────────────────────────────────────────────────────────────────
# 3. Pattern divergence detector
# ─────────────────────────────────────────────────────────────────────
def detect_pattern_divergence(
    query_row: pd.Series,
    analog_row: pd.Series,
    feature_cols: list[str],
    top_k: int = 5,
) -> dict[str, Any]:
    """Identify WHERE the query country matches and diverges from its closest analog.

    For each feature, computes the absolute and relative difference. Returns
    the top-K matching dimensions (smallest difference) and top-K diverging
    dimensions (largest difference).

    Returns:
      - analog: {iso3, year}
      - matches: list of {feature, human_name, query_val, analog_val, diff}
      - divergences: list of {feature, human_name, query_val, analog_val, diff}
    """
    diffs = []
    for col in feature_cols:
        q_val = query_row.get(col)
        a_val = analog_row.get(col)
        if q_val is None or a_val is None:
            continue
        try:
            q_f = float(q_val)
            a_f = float(a_val)
        except (ValueError, TypeError):
            continue
        if np.isnan(q_f) or np.isnan(a_f):
            continue

        abs_diff = abs(q_f - a_f)
        # Normalize by the analog's value magnitude to get relative difference
        scale = max(abs(a_f), abs(q_f), 1e-6)
        rel_diff = abs_diff / scale

        diffs.append({
            "feature": col,
            "human_name": _humanize_indicator(col),
            "query_val": q_f,
            "analog_val": a_f,
            "abs_diff": abs_diff,
            "rel_diff": rel_diff,
        })

    if not diffs:
        return {
            "analog": {"iso3": str(analog_row.get("iso3", "?")), "year": int(analog_row.get("year", 0))},
            "matches": [],
            "divergences": [],
        }

    # Sort by relative difference
    diffs.sort(key=lambda d: d["rel_diff"])

    matches = diffs[:top_k]
    divergences = list(reversed(diffs[-top_k:]))

    return {
        "analog": {
            "iso3": str(analog_row.get("iso3", "?")),
            "year": int(analog_row.get("year", 0)),
        },
        "matches": [
            {
                "feature": d["feature"],
                "human_name": d["human_name"],
                "query_val": d["query_val"],
                "analog_val": d["analog_val"],
                "diff_pct": d["rel_diff"] * 100,
            }
            for d in matches
        ],
        "divergences": [
            {
                "feature": d["feature"],
                "human_name": d["human_name"],
                "query_val": d["query_val"],
                "analog_val": d["analog_val"],
                "diff_pct": d["rel_diff"] * 100,
            }
            for d in divergences
        ],
    }


# ─────────────────────────────────────────────────────────────────────
# 4. Divergent twins finder
# ─────────────────────────────────────────────────────────────────────
def find_divergent_twins(
    analogs_df: pd.DataFrame,
    horizon: int = 5,
    min_outcome_gap: float = 0.15,
) -> list[dict[str, Any]]:
    """Among the FAISS analogs, find pairs that started similarly but had very different outcomes.

    Returns a list of divergent pairs, each with:
      - twin_a: {iso3, year, realized, realized_human}
      - twin_b: {iso3, year, realized, realized_human}
      - outcome_gap: absolute difference in realized growth
      - outcome_gap_human: human-readable
    """
    target_col = _get_target_col(horizon)
    if target_col not in analogs_df.columns and "gdp_pc_growth_5y_fwd" in analogs_df.columns:
        target_col = "gdp_pc_growth_5y_fwd"

    valid = analogs_df.dropna(subset=[target_col])
    if len(valid) < 2:
        return []

    pairs = []
    rows = valid.to_dict("records")
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            r_a = float(rows[i][target_col])
            r_b = float(rows[j][target_col])
            gap = abs(r_a - r_b)
            if gap >= min_outcome_gap:
                # Put the better-performing twin first
                if r_a >= r_b:
                    a, b = rows[i], rows[j]
                    a_r, b_r = r_a, r_b
                else:
                    a, b = rows[j], rows[i]
                    a_r, b_r = r_b, r_a
                pairs.append({
                    "twin_a": {
                        "iso3": str(a.get("iso3", "?")),
                        "year": int(a.get("year", 0)),
                        "realized": a_r,
                        "realized_human": logret_to_human(a_r, horizon),
                    },
                    "twin_b": {
                        "iso3": str(b.get("iso3", "?")),
                        "year": int(b.get("year", 0)),
                        "realized": b_r,
                        "realized_human": logret_to_human(b_r, horizon),
                    },
                    "outcome_gap": gap,
                    "outcome_gap_human": logret_to_human(gap, horizon),
                })

    # Sort by largest gap first
    pairs.sort(key=lambda p: -p["outcome_gap"])
    return pairs[:3]  # Top 3 most divergent pairs


# ─────────────────────────────────────────────────────────────────────
# 5. LLM-powered scenario narrative generation
# ─────────────────────────────────────────────────────────────────────
def generate_scenario_narrative(
    scenario_tree: dict[str, Any],
    divergence: dict[str, Any] | None = None,
    divergent_twins: list[dict[str, Any]] | None = None,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Generate LLM narrative explaining the scenario tree and pattern analysis.

    The LLM is given the structured scenario data and asked to:
    1. Explain WHY each scenario cluster represents a distinct economic pathway
    2. Assess whether the structural divergences affect the forecast
    3. Draw lessons from divergent twins

    Returns:
      - narrative: full LLM response
      - success: bool
    """
    api_key = api_key or get_deepseek_api_key()
    if not api_key:
        return {
            "narrative": _build_fallback_narrative(scenario_tree, divergence, divergent_twins),
            "success": False,
            "reason": "no_api_key",
        }

    query = scenario_tree.get("query", {})
    iso3 = query.get("iso3", "?")
    year = query.get("year", "?")
    horizon = query.get("horizon", 5)

    system_prompt = (
        "You are an expert economic historian and macro-forecasting analyst. "
        "You analyze historical economic patterns to understand how countries "
        "develop over time. Your role is to explain WHY historical patterns "
        "matter for predicting the future — not to invent predictions. "
        "Every number you cite must come from the data context provided. "
        "Write in clear, compelling English accessible to policy-makers."
    )

    # Build the scenario context
    scenario_lines = []
    for i, s in enumerate(scenario_tree.get("scenarios", []), 1):
        members = ", ".join(
            f"{m['iso3']} ({m['year']}): {logret_to_human(m['realized'], horizon)}"
            for m in s["members"]
        )
        scenario_lines.append(
            f"SCENARIO {i}: \"{s['label']}\" — Probability {s['probability']:.0%}\n"
            f"  Average growth: {s['mean_growth_human']}\n"
            f"  Members: {members}\n"
        )
    scenario_block = "\n".join(scenario_lines)

    # Build divergence context
    divergence_block = ""
    if divergence and (divergence.get("matches") or divergence.get("divergences")):
        analog = divergence.get("analog", {})
        match_lines = "\n".join(
            f"  ✅ {m['human_name']}: query={m['query_val']:.2f} vs analog={m['analog_val']:.2f} (diff: {m['diff_pct']:.1f}%)"
            for m in divergence.get("matches", [])
        )
        div_lines = "\n".join(
            f"  ⚠️ {d['human_name']}: query={d['query_val']:.2f} vs analog={d['analog_val']:.2f} (diff: {d['diff_pct']:.1f}%)"
            for d in divergence.get("divergences", [])
        )
        divergence_block = (
            f"\nCLOSEST ANALOG: {analog.get('iso3', '?')} ({analog.get('year', '?')})\n"
            f"STRUCTURAL MATCHES (why they're similar):\n{match_lines}\n"
            f"STRUCTURAL DIVERGENCES (where they differ):\n{div_lines}\n"
        )

    # Build twins context
    twins_block = ""
    if divergent_twins:
        twin_lines = []
        for t in divergent_twins:
            a = t["twin_a"]
            b = t["twin_b"]
            twin_lines.append(
                f"  • {a['iso3']} ({a['year']}): {a['realized_human']}  vs  "
                f"{b['iso3']} ({b['year']}): {b['realized_human']}  "
                f"(gap: {t['outcome_gap_human']})"
            )
        twins_block = (
            "\nDIVERGENT TWINS (similar starting points, different outcomes):\n"
            + "\n".join(twin_lines) + "\n"
        )

    user_prompt = f"""Analyze the historical pattern for {iso3} in {year} with a {horizon}-year forecast horizon.

SCENARIO TREE (from historical analog clustering):
{scenario_block}

Weighted scenario forecast: {scenario_tree.get('weighted_forecast_human', 'N/A')}
ML ensemble forecast (for comparison): {scenario_tree.get('ml_ensemble_human', 'N/A')}
{divergence_block}{twins_block}

Please provide a structured analysis in 3 sections:

1. **SCENARIO ANALYSIS** (2-3 sentences per scenario): For each scenario, explain what economic pathway it represents and WHY the member countries followed that path. What were the structural conditions that led to their outcomes?

2. **PATTERN ASSESSMENT** for {iso3}: Based on the structural matches and divergences with the closest analog, which scenario is {iso3} most likely to follow? What are the key factors that will determine which path it takes?

3. **LESSONS FROM HISTORY**: What do the divergent twins teach us? Countries that started from similar positions but ended up with very different outcomes — what drove the divergence, and what does that imply for {iso3}?
"""

    try:
        response = _call_deepseek_api(system_prompt, user_prompt, api_key, temperature=0.0)
        return {"narrative": response, "success": True}
    except Exception as e:
        return {
            "narrative": _build_fallback_narrative(scenario_tree, divergence, divergent_twins),
            "success": False,
            "reason": str(e),
        }


def _build_fallback_narrative(
    scenario_tree: dict[str, Any],
    divergence: dict[str, Any] | None = None,
    divergent_twins: list[dict[str, Any]] | None = None,
) -> str:
    """Deterministic fallback when DeepSeek API is unavailable."""
    query = scenario_tree.get("query", {})
    lines = [
        f"## Historical Pattern Analysis for {query.get('iso3', '?')} ({query.get('year', '?')}), "
        f"{query.get('horizon', 5)}-year horizon\n",
    ]

    for i, s in enumerate(scenario_tree.get("scenarios", []), 1):
        members = ", ".join(f"{m['iso3']} ({m['year']})" for m in s["members"])
        lines.append(
            f"**Scenario {i}: {s['label']}** — Probability: {s['probability']:.0%}\n"
            f"Average growth: {s['mean_growth_human']}. "
            f"Based on: {members}.\n"
        )

    wf = scenario_tree.get("weighted_forecast_human", "N/A")
    ml = scenario_tree.get("ml_ensemble_human", "N/A")
    lines.append(f"\n**Weighted scenario forecast:** {wf}")
    lines.append(f"**ML ensemble (comparison):** {ml}")

    if divergence and divergence.get("divergences"):
        analog = divergence.get("analog", {})
        lines.append(
            f"\n**Top divergence vs closest analog ({analog.get('iso3', '?')} {analog.get('year', '?')}):**"
        )
        for d in divergence["divergences"][:3]:
            lines.append(f"- {d['human_name']}: query {d['query_val']:.2f} vs analog {d['analog_val']:.2f}")

    if divergent_twins:
        lines.append("\n**Most divergent historical twins:**")
        for t in divergent_twins[:2]:
            a, b = t["twin_a"], t["twin_b"]
            lines.append(
                f"- {a['iso3']} ({a['year']}): {a['realized_human']} vs "
                f"{b['iso3']} ({b['year']}): {b['realized_human']}"
            )

    return "\n".join(lines)
