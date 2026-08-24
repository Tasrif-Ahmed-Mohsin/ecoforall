"""
LLM-Gated Cross-Domain Feature Selection Engine (LGCF)
=======================================================
Core contribution: An LLM decides PER COUNTRY-YEAR whether to incorporate
political, environmental, or societal signals into economic forecasting.

The LLM doesn't forecast — it acts as a REGIME DETECTOR:
  "Given Brazil in 2014 with commodity bust + rising protests + drought,
   should the forecaster weight political and environmental signals?"

Architecture:
  1. Context Builder: Assembles macro context from the panel for each (country, year)
  2. Gate Prompt: Asks the LLM to rate domain relevance [0-1]
  3. Gate Parser: Extracts structured weights from LLM response
  4. Caching: All LLM responses cached for reproducibility and cost control
"""

from __future__ import annotations
import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────── Paths ───────────────────────────
ROOT = Path(r"e:\politics and economy")
CACHE_DIR = ROOT / "data" / "llm_gate_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────── Gate Output Schema ───────────────────────────

@dataclass
class GateWeights:
    """Output of the LLM gate function — domain relevance weights."""
    iso3: str
    year: int
    horizon: int
    economy: float      # weight for economic features
    politics: float     # weight for political features
    environment: float  # weight for environmental features
    human_society: float  # weight for human/society features
    confidence: float   # LLM's self-reported confidence (0-1)
    reasoning: str      # LLM's reasoning (for interpretability)
    raw_response: str   # full LLM response (for debugging)

    def as_array(self) -> np.ndarray:
        """Return domain weights as a normalized array [eco, pol, env, hum]."""
        w = np.array([self.economy, self.politics, self.environment, self.human_society])
        total = w.sum()
        if total > 0:
            w = w / total
        else:
            w = np.array([1.0, 0.0, 0.0, 0.0])  # fallback to eco-only
        return w.astype(np.float32)

    def dominant_domain(self) -> str:
        """Return the non-economic domain with highest weight, or 'economy' if eco dominates."""
        domains = {
            "economy": self.economy,
            "politics": self.politics,
            "environment": self.environment,
            "human_society": self.human_society,
        }
        return max(domains, key=domains.get)

    def is_cross_domain_relevant(self, threshold: float = 0.15) -> bool:
        """Does the LLM think any non-economic domain exceeds the threshold?"""
        return max(self.politics, self.environment, self.human_society) >= threshold


# ─────────────────────────── Context Builder ───────────────────────────

def build_context(df: pd.DataFrame, iso3: str, year: int) -> dict:
    """Build a structured context dict for the LLM from the panel data.
    
    Only uses data available AT OR BEFORE the given year (no future leakage).
    """
    country_data = df[(df["iso3"] == iso3) & (df["year"] <= year)].sort_values("year")
    
    if country_data.empty:
        return {"error": f"No data for {iso3} before {year}"}
    
    latest = country_data.iloc[-1]
    
    # ── Economic context ──
    eco_context = {}
    for col, label in [
        ("gdp_pc", "gdp_per_capita"),
        ("gdp_pc_real", "gdp_per_capita"),
        ("gdp_pc_real_usd", "gdp_per_capita"),
        ("inflation_rate", "inflation_rate"),
        ("gov_debt_gdp", "gov_debt_pct_gdp"),
        ("unemployment_rate", "unemployment_rate"),
        ("current_account_gdp", "current_account_pct_gdp"),
        ("investment_gdp", "investment_pct_gdp"),
        ("banking_crisis", "banking_crisis"),
        ("currency_crisis", "currency_crisis"),
        ("sov_debt_crisis", "sovereign_debt_crisis"),
    ]:
        if col in latest.index and pd.notna(latest[col]):
            eco_context[label] = float(latest[col])

    # GDP growth trajectory (last 3 years)
    if len(country_data) >= 2:
        gdp_col = None
        for candidate in ["gdp_pc", "gdp_pc_real", "gdp_pc_real_usd"]:
            if candidate in country_data.columns:
                gdp_col = candidate
                break
        if gdp_col:
            recent = country_data.tail(min(4, len(country_data)))[gdp_col].dropna()
            if len(recent) >= 2:
                growth_rates = recent.pct_change().dropna()
                eco_context["recent_growth_trajectory"] = [round(float(g), 4) for g in growth_rates]

    # ── Political context ──
    pol_context = {}
    for col, label in [
        ("stability_momentum_annual_mean", "political_stability"),
        ("conflict_intensity_annual_mean", "conflict_intensity"),
        ("protest_pressure_annual_mean", "protest_pressure"),
        ("protest_unrest_annual_sum", "protest_count"),
        ("material_conflict_annual_sum", "material_conflicts"),
        ("sanctions_coercion_annual_sum", "sanctions"),
        ("goldstein_annual_mean", "goldstein_index"),
        ("news_tone_annual_mean", "media_sentiment"),
    ]:
        if col in latest.index and pd.notna(latest[col]):
            pol_context[label] = round(float(latest[col]), 4)

    # ── Environmental context ──
    env_context = {}
    for col, label in [
        ("co2_emissions_per_capita", "co2_per_capita"),
        ("temp_anomaly_celsius", "temperature_anomaly_celsius"),
        ("disaster_economic_damage_usd", "disaster_damage_usd"),
        ("renewable_energy_pct_share", "renewable_energy_pct"),
        ("floods_count", "floods"),
        ("droughts_count", "droughts"),
        ("storms_count", "storms"),
        ("wildfires_count", "wildfires"),
        ("forest_cover_pct", "forest_cover_pct"),
        ("forest_area_pct_land", "forest_cover_pct"),
    ]:
        if col in latest.index and pd.notna(latest[col]):
            env_context[label] = round(float(latest[col]), 4)

    # ── Human/Society context ──
    hum_context = {}
    for col, label in [
        ("psychology_trust", "institutional_trust"),
        ("psychology_fear", "societal_fear"),
        ("psychology_optimism", "public_optimism"),
        ("psychology_confidence", "consumer_confidence"),
        ("psychology_social_cohesion", "social_cohesion"),
        ("psychology_nationalism", "nationalism_index"),
        ("society_education", "education_index"),
        ("society_urbanization", "urbanization_rate"),
        ("society_age", "median_age"),
        ("society_healthcare", "healthcare_index"),
        ("society_migration", "net_migration"),
    ]:
        if col in latest.index and pd.notna(latest[col]):
            hum_context[label] = round(float(latest[col]), 4)

    return {
        "iso3": iso3,
        "year": year,
        "economic": eco_context,
        "political": pol_context,
        "environmental": env_context,
        "human_society": hum_context,
    }


# ─────────────────────────── Prompt Construction ───────────────────────────

def build_gate_prompt(context: dict, horizon: int) -> str:
    """Construct the LLM prompt for domain gating."""
    
    iso3 = context["iso3"]
    year = context["year"]
    eco = context.get("economic", {})
    pol = context.get("political", {})
    env = context.get("environmental", {})
    hum = context.get("human_society", {})

    # Format each section
    eco_lines = "\n".join(f"  - {k}: {v}" for k, v in eco.items()) if eco else "  (limited data)"
    pol_lines = "\n".join(f"  - {k}: {v}" for k, v in pol.items()) if pol else "  (limited data)"
    env_lines = "\n".join(f"  - {k}: {v}" for k, v in env.items()) if env else "  (limited data)"
    hum_lines = "\n".join(f"  - {k}: {v}" for k, v in hum.items()) if hum else "  (limited data)"

    prompt = f"""You are a senior macroeconomist analyzing {iso3} in {year} to predict GDP per capita growth over the next {horizon} year(s).

CURRENT STATE OF {iso3} ({year}):

Economic indicators:
{eco_lines}

Political indicators:
{pol_lines}

Environmental indicators:
{env_lines}

Society & psychology indicators:
{hum_lines}

TASK: Rate how important each domain is for predicting GDP growth over the next {horizon} year(s).

REASONING GUIDELINES:
- In stable, high-income economies with no shocks: economy ≈ 0.75-0.90, others low
- During political crises, coups, sanctions, or wars: politics should be 0.20-0.50
- During major climate disasters, droughts, or energy transitions: environment should be 0.15-0.40
- During social upheaval, collapsing trust, or demographic shifts: human_society should be 0.15-0.40
- Multiple crises can co-occur (e.g., political + environmental stress)
- For longer horizons (5+ years), structural factors (institutions, demographics, climate) matter MORE

OUTPUT EXACTLY this JSON format (no other text):
{{
  "economy": <float 0.0-1.0>,
  "politics": <float 0.0-1.0>,
  "environment": <float 0.0-1.0>,
  "human_society": <float 0.0-1.0>,
  "confidence": <float 0.0-1.0>,
  "reasoning": "<one sentence explaining your weighting>"
}}

The four domain weights should sum to approximately 1.0."""

    return prompt


# ─────────────────────────── Response Parsing ───────────────────────────

def parse_gate_response(response_text: str) -> dict:
    """Parse the LLM response into structured gate weights."""
    
    # Try to extract JSON from the response
    # Handle cases where LLM wraps JSON in markdown code blocks
    text = response_text.strip()
    
    # Remove markdown code block markers if present
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*", "", text)
    
    # Try to find JSON object
    json_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group())
            
            # Validate and extract fields
            result = {
                "economy": float(parsed.get("economy", 0.7)),
                "politics": float(parsed.get("politics", 0.1)),
                "environment": float(parsed.get("environment", 0.1)),
                "human_society": float(parsed.get("human_society", 0.1)),
                "confidence": float(parsed.get("confidence", 0.5)),
                "reasoning": str(parsed.get("reasoning", "")),
            }
            
            # Clamp values to [0, 1]
            for key in ["economy", "politics", "environment", "human_society", "confidence"]:
                result[key] = max(0.0, min(1.0, result[key]))
            
            # Normalize domain weights to sum to 1
            total = result["economy"] + result["politics"] + result["environment"] + result["human_society"]
            if total > 0:
                for key in ["economy", "politics", "environment", "human_society"]:
                    result[key] = result[key] / total
            
            return result
            
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            log.warning(f"Failed to parse LLM response JSON: {e}")
    
    # Fallback: return eco-dominant weights
    log.warning(f"Using fallback gate weights for unparseable response")
    return {
        "economy": 0.85,
        "politics": 0.05,
        "environment": 0.05,
        "human_society": 0.05,
        "confidence": 0.0,
        "reasoning": "PARSE_FAILURE: defaulting to economy-dominant",
    }


# ─────────────────────────── Caching Layer ───────────────────────────

def _cache_key(iso3: str, year: int, horizon: int, model: str) -> str:
    """Generate a deterministic cache key."""
    raw = f"{iso3}_{year}_{horizon}_{model}"
    return hashlib.md5(raw.encode()).hexdigest()


def load_cached_gate(iso3: str, year: int, horizon: int, model: str = "deepseek") -> dict | None:
    """Load a cached gate response if it exists."""
    key = _cache_key(iso3, year, horizon, model)
    cache_file = CACHE_DIR / f"{key}.json"
    if cache_file.exists():
        try:
            with open(cache_file) as f:
                return json.load(f)
        except Exception:
            return None
    return None


def save_gate_cache(iso3: str, year: int, horizon: int, model: str,
                     gate_data: dict):
    """Save gate response to cache."""
    key = _cache_key(iso3, year, horizon, model)
    cache_file = CACHE_DIR / f"{key}.json"
    gate_data["_cache_key"] = key
    gate_data["_iso3"] = iso3
    gate_data["_year"] = year
    gate_data["_horizon"] = horizon
    gate_data["_model"] = model
    with open(cache_file, "w") as f:
        json.dump(gate_data, f, indent=2)


# ─────────────────────────── LLM API Interface ───────────────────────────

def call_deepseek_gate(prompt: str, api_key: str | None = None,
                        model: str = "deepseek-chat",
                        temperature: float = 0.0,
                        max_tokens: int = 300) -> str:
    """Call DeepSeek API for gate weights. Returns raw response text."""
    import httpx
    
    if api_key is None:
        # Try to load from standard locations
        key_paths = [
            ROOT / "projectresearch" / "deepseek.txt",
            Path(r"E:\projectresearch\deepseek.txt"),
            Path.home() / "deepseek.txt",
        ]
        for kp in key_paths:
            if kp.exists():
                text = kp.read_text().strip()
                # Handle api_key=VALUE format
                if "api_key=" in text:
                    m = re.search(r"api_key=([^\s]+)", text)
                    api_key = m.group(1).strip() if m else text
                else:
                    api_key = text
                break
        
        if api_key is None:
            api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        
        if not api_key:
            raise ValueError("No DeepSeek API key found. Set DEEPSEEK_API_KEY or place key in deepseek.txt")
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    
    resp = httpx.post(
        "https://api.deepseek.com/chat/completions",
        json=payload,
        headers=headers,
        timeout=30.0,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


# ─────────────────────────── Main Gate Function ───────────────────────────

def compute_gate_weights(df: pd.DataFrame, iso3: str, year: int, horizon: int,
                          api_key: str | None = None,
                          llm_model: str = "deepseek-chat",
                          use_cache: bool = True) -> GateWeights:
    """Compute LLM gate weights for a single (country, year, horizon).
    
    This is the core function of the LGCF framework.
    """
    
    # Check cache first
    if use_cache:
        cached = load_cached_gate(iso3, year, horizon, llm_model)
        if cached:
            return GateWeights(
                iso3=iso3, year=year, horizon=horizon,
                economy=cached["economy"], politics=cached["politics"],
                environment=cached["environment"], human_society=cached["human_society"],
                confidence=cached.get("confidence", 0.5),
                reasoning=cached.get("reasoning", ""),
                raw_response=cached.get("_raw_response", ""),
            )
    
    # Build context and prompt
    context = build_context(df, iso3, year)
    prompt = build_gate_prompt(context, horizon)
    
    # Call LLM
    try:
        raw_response = call_deepseek_gate(prompt, api_key=api_key, model=llm_model)
    except Exception as e:
        log.warning(f"LLM call failed for {iso3}/{year}/h={horizon}: {e}. Using fallback.")
        raw_response = ""
    
    # Parse response
    if raw_response:
        parsed = parse_gate_response(raw_response)
    else:
        parsed = parse_gate_response("")  # will return fallback
    
    # Cache the result
    cache_data = {**parsed, "_raw_response": raw_response}
    save_gate_cache(iso3, year, horizon, llm_model, cache_data)
    
    return GateWeights(
        iso3=iso3, year=year, horizon=horizon,
        economy=parsed["economy"], politics=parsed["politics"],
        environment=parsed["environment"], human_society=parsed["human_society"],
        confidence=parsed.get("confidence", 0.5),
        reasoning=parsed.get("reasoning", ""),
        raw_response=raw_response,
    )


def compute_gate_weights_batch(df: pd.DataFrame, 
                                 queries: list[tuple[str, int, int]],
                                 api_key: str | None = None,
                                 llm_model: str = "deepseek-chat",
                                 use_cache: bool = True,
                                 rate_limit_delay: float = 0.1) -> list[GateWeights]:
    """Compute gate weights for a batch of (iso3, year, horizon) queries.
    
    Handles rate limiting and progress logging.
    """
    results = []
    n_total = len(queries)
    n_cached = 0
    n_api_calls = 0
    
    for i, (iso3, year, horizon) in enumerate(queries):
        if use_cache and load_cached_gate(iso3, year, horizon, llm_model) is not None:
            n_cached += 1
        
        gate = compute_gate_weights(
            df, iso3, year, horizon,
            api_key=api_key, llm_model=llm_model, use_cache=use_cache,
        )
        results.append(gate)
        
        if gate.raw_response and not (use_cache and load_cached_gate(iso3, year, horizon, llm_model)):
            n_api_calls += 1
            if rate_limit_delay > 0:
                time.sleep(rate_limit_delay)
        
        if (i + 1) % 50 == 0 or (i + 1) == n_total:
            log.info(f"  Gate progress: {i+1}/{n_total} "
                     f"({n_cached} cached, {n_api_calls} API calls)")
    
    log.info(f"  Gate batch complete: {n_total} queries, "
             f"{n_cached} from cache, {n_api_calls} fresh API calls")
    
    return results


# ─────────────────────────── Heuristic Gate (Baseline) ───────────────────────────

def heuristic_gate(context: dict, horizon: int) -> dict:
    """A simple rule-based gate for comparison against the LLM gate.
    
    This serves as an ablation baseline — if the LLM can't beat hand-crafted rules,
    it's not contributing novel regime detection.
    """
    eco = context.get("economic", {})
    pol = context.get("political", {})
    env = context.get("environmental", {})
    hum = context.get("human_society", {})
    
    w_eco, w_pol, w_env, w_hum = 0.70, 0.10, 0.10, 0.10
    
    # Rule 1: If there's an active crisis, upweight politics
    if eco.get("banking_crisis", 0) > 0 or eco.get("currency_crisis", 0) > 0:
        w_pol += 0.15
        w_eco -= 0.10
        w_hum += 0.05
    
    # Rule 2: If conflict intensity is high, upweight politics
    if pol.get("conflict_intensity", 0) > 0.5:
        w_pol += 0.20
        w_eco -= 0.15
        w_hum += 0.05
    
    # Rule 3: If there's significant disaster damage, upweight environment
    if env.get("disaster_damage_usd", 0) > 1e8:
        w_env += 0.15
        w_eco -= 0.10
    
    # Rule 4: If temperature anomaly is extreme, upweight environment for long horizons
    if abs(env.get("temperature_anomaly_celsius", 0)) > 1.5 and horizon >= 5:
        w_env += 0.10
        w_eco -= 0.05
    
    # Rule 5: If trust is very low, upweight human/society
    if hum.get("institutional_trust", 100) < 30:
        w_hum += 0.15
        w_eco -= 0.10
    
    # Rule 6: For longer horizons, structural factors matter more
    if horizon >= 5:
        structural_boost = 0.05 * (horizon / 5)
        w_hum += structural_boost
        w_env += structural_boost / 2
        w_eco -= structural_boost * 1.5
    
    # Normalize
    total = w_eco + w_pol + w_env + w_hum
    return {
        "economy": max(0, w_eco / total),
        "politics": max(0, w_pol / total),
        "environment": max(0, w_env / total),
        "human_society": max(0, w_hum / total),
        "confidence": 0.5,
        "reasoning": "heuristic_rule_based_gate",
    }


# ─────────────────────────── Random Gate (Ablation) ───────────────────────────

def random_gate(seed: int = 42) -> dict:
    """Random gate weights for ablation — proves the LLM adds signal, not just noise."""
    rng = np.random.RandomState(seed)
    w = rng.dirichlet([1, 1, 1, 1])
    return {
        "economy": float(w[0]),
        "politics": float(w[1]),
        "environment": float(w[2]),
        "human_society": float(w[3]),
        "confidence": 0.0,
        "reasoning": "random_ablation_baseline",
    }


# ─────────────────────────── Pilot Test ───────────────────────────

PILOT_CASES = [
    # (iso3, year, horizon, expected_dominant_non_eco, description)
    ("BRA", 2014, 1, "politics", "Brazil commodity bust + protests + impeachment path"),
    ("BRA", 2014, 5, "environment", "Brazil long-term: Amazon deforestation + drought"),
    ("JPN", 2011, 1, "environment", "Japan Fukushima tsunami + nuclear disaster"),
    ("EGY", 2011, 1, "politics", "Egypt: Arab Spring revolution"),
    ("SYR", 2012, 1, "politics", "Syria: civil war escalation"),
    ("GRC", 2012, 1, "economy", "Greece: sovereign debt crisis (economic crisis)"),
    ("USA", 2019, 1, "economy", "USA 2019: stable economy pre-COVID"),
    ("NOR", 2018, 5, "environment", "Norway: energy transition leader, long horizon"),
    ("VEN", 2017, 1, "politics", "Venezuela: political/economic collapse"),
    ("IND", 2016, 5, "human_society", "India: demographic dividend, urbanization surge"),
]


def run_pilot_test(df: pd.DataFrame, api_key: str | None = None,
                    llm_model: str = "deepseek-chat") -> pd.DataFrame:
    """Run the pilot test on known historical events.
    
    Checks if the LLM correctly identifies the dominant non-economic domain.
    """
    log.info("=" * 70)
    log.info("  LLM GATE PILOT TEST — Known Historical Events")
    log.info("=" * 70)
    
    results = []
    for iso3, year, horizon, expected_domain, description in PILOT_CASES:
        log.info(f"\n  Testing: {iso3} {year} (h={horizon}) — {description}")
        
        gate = compute_gate_weights(
            df, iso3, year, horizon,
            api_key=api_key, llm_model=llm_model, use_cache=True,
        )
        
        dominant = gate.dominant_domain()
        correct = (dominant == expected_domain) or (
            expected_domain != "economy" and gate.is_cross_domain_relevant()
        )
        
        results.append({
            "iso3": iso3,
            "year": year,
            "horizon": horizon,
            "description": description,
            "expected_dominant": expected_domain,
            "actual_dominant": dominant,
            "correct": correct,
            "w_eco": round(gate.economy, 3),
            "w_pol": round(gate.politics, 3),
            "w_env": round(gate.environment, 3),
            "w_hum": round(gate.human_society, 3),
            "confidence": round(gate.confidence, 3),
            "reasoning": gate.reasoning[:100],
        })
        
        status = "✓" if correct else "✗"
        log.info(f"  {status} Expected: {expected_domain}, Got: {dominant}")
        log.info(f"    Weights: eco={gate.economy:.2f} pol={gate.politics:.2f} "
                 f"env={gate.environment:.2f} hum={gate.human_society:.2f}")
        log.info(f"    Reasoning: {gate.reasoning[:80]}")
    
    df_results = pd.DataFrame(results)
    accuracy = df_results["correct"].mean() * 100
    
    log.info(f"\n  PILOT ACCURACY: {accuracy:.0f}% ({df_results['correct'].sum()}/{len(df_results)})")
    
    if accuracy >= 70:
        log.info("  VERDICT: PROCEED with full-scale gate computation")
    elif accuracy >= 50:
        log.info("  VERDICT: MARGINAL — consider refining the prompt")
    else:
        log.info("  VERDICT: POOR — LLM cannot reliably detect regimes. Rethink approach.")
    
    return df_results


if __name__ == "__main__":
    # If run directly, execute pilot test
    log.info("Loading quad-domain panel for pilot test...")
    quad_path = ROOT / "data" / "quad_domain_annual_panel.parquet"
    if quad_path.exists():
        df = pd.read_parquet(quad_path)
        log.info(f"Loaded: {df.shape}")
        
        pilot_results = run_pilot_test(df)
        pilot_results.to_csv(ROOT / "data" / "llm_gate_pilot_results.csv", index=False)
        print(pilot_results[["iso3", "year", "horizon", "expected_dominant", 
                             "actual_dominant", "correct", "w_eco", "w_pol", 
                             "w_env", "w_hum"]].to_string(index=False))
    else:
        log.error(f"Missing {quad_path}")
