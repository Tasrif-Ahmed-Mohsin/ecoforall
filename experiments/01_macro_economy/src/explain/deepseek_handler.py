"""DeepSeek LLM Narrative Handler — Dual Mode (ML Grounded RAG + Raw World Knowledge)."""
from __future__ import annotations

import json
import math
import os
import re
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEEPSEEK_TXT = ROOT / "deepseek.txt"


def get_deepseek_api_key() -> str | None:
    """Read API key from environment variable DEEPSEEK_API_KEY or deepseek.txt."""
    if os.environ.get("DEEPSEEK_API_KEY"):
        return os.environ["DEEPSEEK_API_KEY"].strip()
    if DEEPSEEK_TXT.exists():
        text = DEEPSEEK_TXT.read_text().strip()
        if "api_key=" in text:
            m = re.search(r"api_key=([^\s]+)", text)
            if m:
                return m.group(1).strip()
        return text
    return None


def _fmt_val(val: Any, spec: str = "+.4f", default: str = "N/A") -> str:
    """Safely format any value into floating point string or string fallback."""
    if val is None or (isinstance(val, float) and val != val):
        return default
    try:
        f_val = float(val)
        return format(f_val, spec)
    except (ValueError, TypeError):
        return str(val)


def logret_to_human(y: Any, horizon: int = 5) -> str:
    """Convert log-return y to intuitive human-readable total % growth and annual % growth.
    
    Example: 0.2290 -> "+25.7% total growth (+4.7%/yr)"
    """
    if y is None:
        return "N/A"
    try:
        y_val = float(y)
        if math.isnan(y_val):
            return "N/A"
        total_pct = (math.exp(y_val) - 1.0) * 100.0
        annual_pct = (math.exp(y_val / max(horizon, 1)) - 1.0) * 100.0
        return f"{total_pct:+.1f}% total ({annual_pct:+.1f}%/yr)"
    except Exception:
        return str(y)


def _call_deepseek_api(system_prompt: str, user_prompt: str, api_key: str, temperature: float = 0.0) -> str:
    """Execute raw HTTPS request to DeepSeek API endpoint."""
    url = "https://api.deepseek.com/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": 1000,
    }
    
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = resp.read().decode("utf-8")
            parsed = json.loads(body)
            return parsed["choices"][0]["message"]["content"]
    except Exception as e:
        raise RuntimeError(f"DeepSeek API call failed: {e}")


def generate_grounded_explanation(
    iso3: str,
    year: int,
    horizon: int,
    ml_forecast: dict[str, Any],
    analogs: list[dict[str, Any]],
    macro_snapshot: dict[str, float],
    conformal_info: dict[str, Any],
    user_prompt: str,
    pos_drivers: list[dict[str, Any]] | None = None,
    neg_drags: list[dict[str, Any]] | None = None,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Generate DeepSeek narrative grounded STRICTLY in ML system numbers and FAISS analogs."""
    api_key = api_key or get_deepseek_api_key()
    if not api_key:
        return {
            "error": "No DeepSeek API key found in deepseek.txt or environment.",
            "grounded_narrative": "API key missing. Unable to generate narrative.",
        }

    sys_prompt = (
        "You are an expert quantitative macroeconomic analyst explaining outputs from a global machine learning model. "
        "CRITICAL RULE: DO NOT INVENT NUMBERS OR SELF-MADE ASSUMPTIONS. Every figure, percentage, growth rate, and "
        "country twin outcome mentioned MUST come strictly from the ML system context provided below. "
        "IMPORTANT: Present all growth forecasts in clear percentage terms (e.g. '+25.7% total growth over 5 years') so "
        "it is easy to understand for non-experts. "
        "Explain the country's economic context, key macro drivers, why the historical analog countries are relevant, "
        "and what the model forecasts in plain, compelling English."
    )

    ensemble = ml_forecast.get("ensemble", ml_forecast.get("lgbm", 0.0))
    ridge = ml_forecast.get("ridge", 0.0)
    lgbm = ml_forecast.get("lgbm", 0.0)
    q05 = ml_forecast.get("q05")
    q95 = ml_forecast.get("q95")
    
    analog_str_list = []
    for i, a in enumerate(analogs[:5], 1):
        country_code = str(a.get("iso3", "UNK"))
        try:
            a_year = int(float(a.get("year", 0)))
        except (ValueError, TypeError):
            a_year = 0
        realized_human = logret_to_human(a.get("gdp_pc_growth_5y_fwd"), horizon)
        match_score_str = _fmt_val(a.get("match_score", a.get("similarity")), ".3f")
        analog_str_list.append(
            f"  {i}. {country_code} ({a_year}): {horizon}y realized growth={realized_human}, match_score={match_score_str}"
        )
    analog_block = "\n".join(analog_str_list) if analog_str_list else "  No analogs found."

    macro_str_list = []
    for k, v in list(macro_snapshot.items())[:30]:
        if v is not None and not (isinstance(v, float) and (v != v)):  # check not NaN
            macro_str_list.append(f"  - {k}: {_fmt_val(v, ',.3f')}")
    macro_block = "\n".join(macro_str_list) if macro_str_list else "  Standard macro snapshot available."

    ens_human = logret_to_human(ensemble, horizon)
    lgbm_human = logret_to_human(lgbm, horizon)
    ridge_human = logret_to_human(ridge, horizon)
    q05_human = logret_to_human(q05, horizon)
    q95_human = logret_to_human(q95, horizon)
    cov_str = _fmt_val(conformal_info.get("calibrated_coverage_pct", 90.0), ".1f")

    pos_str = "\n".join([f"  + {d['human_name']}: value={d['val']:,.2f} (SHAP push={d['contribution']:+.4f})" for d in (pos_drivers or [])]) or "  (none)"
    neg_str = "\n".join([f"  - {d['human_name']}: value={d['val']:,.2f} (SHAP drag={d['contribution']:+.4f})" for d in (neg_drags or [])]) or "  (none)"
    driver_block = f"MATHEMATICAL SHAP FEATURE DRIVERS:\nTop Positive Push Factors:\n{pos_str}\nTop Negative Drag Factors:\n{neg_str}"

    year_desc = f"{year} (projecting forward from latest available 2024 macro baseline)" if year > 2024 else f"{year}"

    user_context = f"""User Question: "{user_prompt}"

ML SYSTEM GROUND TRUTH DATA:
  - Country: {iso3} (Query Year Target: {year_desc})
  - Horizon: {horizon}-year ahead forecast
  - ML Ensemble Forecast: {ens_human} [log return: {_fmt_val(ensemble, '+.4f')}]
  - LightGBM Model Prediction: {lgbm_human}
  - Ridge Model Prediction: {ridge_human}
  - Conformal 90% Uncertainty Interval: [{q05_human} to {q95_human}]
  - Conformal Coverage: {cov_str}%

{driver_block}

HISTORICAL SIMILAR TWINS (FAISS Rank-Euclidean Retrieval):
{analog_block}

KEY OBSERVED MACRO INDICATORS ({year}):
{macro_block}

Please provide a structured 3-paragraph economic summary using clear percentage terms:
1. Economic Situation & Key Drivers in {year} (explicitly cite the positive push factors and negative drags above)
2. ML Forecast Trend & Uncertainty Range over the next {horizon} years (explain in % growth)
3. Lessons from Historical Twin Economies (FAISS analogs)
"""

    try:
        response_text = _call_deepseek_api(sys_prompt, user_context, api_key, temperature=0.0)
        return {
            "grounded_narrative": response_text,
            "raw_context": user_context,
            "ensemble_pred": ensemble,
            "ensemble_human": ens_human,
            "lgbm_pred": lgbm,
            "lgbm_human": lgbm_human,
            "ridge_pred": ridge,
            "ridge_human": ridge_human,
            "q05": q05,
            "q05_human": q05_human,
            "q95": q95,
            "q95_human": q95_human,
            "success": True,
        }
    except Exception as e:
        return {
            "error": str(e),
            "grounded_narrative": f"Error querying DeepSeek API: {e}",
            "success": False,
        }


def generate_standalone_llm_answer(
    user_prompt: str,
    iso3: str | None = None,
    year: int | None = None,
    horizon: int = 5,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Generate DeepSeek zero-shot response purely based on its internal world knowledge, plus an estimated numerical forecast."""
    api_key = api_key or get_deepseek_api_key()
    if not api_key:
        return {
            "standalone_narrative": "DeepSeek API key missing.",
            "estimated_log_return": None,
            "estimated_human": "N/A",
            "success": False,
        }

    if iso3 and year:
        sys_prompt = (
            "You are Aurelius AI, an expert economic history and macro forecasting AI assistant. "
            "Provide a direct, thorough answer to the user's question based strictly on your general parametric world knowledge. "
            "Do not reference any external ML models.\n\n"
            f"CRITICAL REQUIREMENT: At the very end of your response, estimate the expected cumulative {horizon}-year real GDP PER CAPITA growth percentage for {iso3} from end-{year} to end-{year + horizon} (i.e. per-capita growth between year {year} and year {year + horizon}).\n"
            "Output your final percentage estimate on a dedicated final line in EXACTLY this format:\n"
            "ESTIMATED_GROWTH_PCT: <number>"
        )
    else:
        sys_prompt = (
            "You are Aurelius AI, an intelligent, classy, and highly capable Macroeconomic Intelligence & Causal Forecasting Assistant. "
            "Provide a helpful, polite, and direct answer to general user questions (e.g. 'who are you', greetings, general economic concepts) "
            "in plain, well-structured text."
        )

    try:
        response_text = _call_deepseek_api(sys_prompt, user_prompt, api_key, temperature=0.0)
        est_pct = None
        
        # 1. Search STRICTLY for the ESTIMATED_GROWTH_PCT: tag
        m_pct = re.search(r"ESTIMATED_GROWTH_PCT:\s*([+-]?\d+\.?\d*)", response_text, re.IGNORECASE)
        if m_pct:
            try:
                val = float(m_pct.group(1))
                # Validate reasonable annual growth bound (|annual_pct| <= 35%)
                annual_pct_est = val / max(horizon, 1) if abs(val) > 1.0 else (val * 100.0) / max(horizon, 1)
                if abs(annual_pct_est) <= 35.0:
                    est_pct = val
            except ValueError:
                pass

        # Clean prompt output text by removing trailing estimation tag
        clean_text = re.sub(r"ESTIMATED_GROWTH_PCT:.*", "", response_text, flags=re.DOTALL).strip()
        clean_text = re.sub(r"ESTIMATED_5Y_GROWTH_PCT:.*", "", clean_text, flags=re.DOTALL).strip()
        clean_text = clean_text.split("NUMERIC_ESTIMATE:")[0].strip()

        log_ret = None
        if est_pct is not None:
            pct_val = est_pct if abs(est_pct) > 1.0 else est_pct * 100.0
            log_ret = math.log(max(1.0 + pct_val / 100.0, 1e-4))
            est_human = logret_to_human(log_ret, horizon)
        else:
            est_human = "N/A"

        return {
            "standalone_narrative": clean_text,
            "estimated_log_return": log_ret,
            "estimated_human": est_human,
            "success": True,
        }
    except Exception as e:
        return {
            "error": str(e),
            "standalone_narrative": f"Error querying DeepSeek API: {e}",
            "estimated_log_return": None,
            "estimated_human": "N/A",
            "success": False,
        }
