"""LLM-driven narrative explainer for the per-country forecaster.

The LLM is the *explainer only* — never the predictor. It receives:

1. The deterministic model output (ensemble forecast + calibrated 80% band).
2. The current macro snapshot for the queried country-year (only the columns the
   model was trained on, no future leakage).
3. The FAISS-retrieved historical analogs, with their realized 5y outcomes.
4. The top contributing features (LGBM gain).

And it is asked to produce:

  {
    "summary":     "... 2-3 sentences in plain English ...",
    "key_drivers": ["...", "...", "..."],     # grounded in the indicators below
    "analog_notes": "... how the historical analogs relate to the query ...",
    "uncertainty_notes": "... what could push this prediction outside the band ..."
  }

The LLM is instructed NEVER to invent numbers; every claim in the summary must
be backed by data we provided in the prompt. If you want to swap providers,
override `call_llm()`.

When no API key is configured the module returns a deterministic fallback dict
that simply reformats the structured input into a stub narrative. This keeps
the pipeline testable end-to-end on a fresh machine.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

# --- Provider selection -----------------------------------------------------
# Set CLAUDE_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY to enable live calls.
# Until any of those is set we return a fallback narrative.
_PROVIDERS = ("anthropic", "openai")
_DEFAULT_MODEL = {
    "anthropic": "claude-opus-4-8",
    "openai":    "gpt-4o-mini",
}

# --- Macro indicator allowlist (only these are surfaced to the LLM) --------
HUMAN_NAMES = {
    "gdp_pc_real":               "real GDP per capita",
    "gdp_pc_real_ppp":           "real GDP per capita (PPP)",
    "cpi":                       "consumer price index",
    "inflation_rate":            "inflation rate",
    "inflation_cpi":             "CPI inflation rate",
    "fx_to_usd":                 "USD exchange rate",
    "fx_to_gbp":                 "GBP exchange rate",
    "gov_debt_gdp":              "government debt / GDP",
    "current_account_gdp":       "current account / GDP",
    "unemployment_rate":         "unemployment rate",
    "population":                "population",
    "real_wage":                 "real wage",
    "short_rate":                "short-term interest rate",
    "long_rate":                 "long-term interest rate",
    "bond_yield_lt":             "long-term bond yield",
    "gini_income":               "income Gini",
    "gini_income_wb":            "income Gini (WB)",
    "gini_wealth":               "wealth Gini",
    "trade_gdp":                 "trade / GDP",
}

# User-friendly names for engineered (lag / rolling / log-return) columns.
def _humanize_indicator(col: str) -> str:
    """Turn panel column names into the names we show the LLM."""
    # Strip suffix pieces
    for suf in ("_lag5", "_lag1", "_roll5_mean", "_delta5", "_logret5"):
        if col.endswith(suf):
            base = col[: -len(suf)]
            nice_suffix = {
                "_lag5":          " (5-year lag)",
                "_lag1":          " (1-year lag)",
                "_roll5_mean":    " (5-year rolling mean)",
                "_delta5":        " (5-year change)",
                "_logret5":       " (5-year log return)",
            }[suf]
            return (HUMAN_NAMES.get(base, base.replace("_", " ")) + nice_suffix)
    return HUMAN_NAMES.get(col, col.replace("_", " "))


# ---------------------------------------------------------------------------
# Structured input assembly
# ---------------------------------------------------------------------------
@dataclass
class ExplainInput:
    """Everything the LLM needs to write a grounded narrative."""
    iso3: str
    query_year: int
    ridge:    float
    lgbm:     float
    ensemble: float
    pi80_low: float
    pi80_high: float
    monotonic: bool                                # whether the band is well-ordered
    macro_snapshot: dict[str, float]               # raw + engineered features, current year
    analogs: list[dict[str, Any]]                  # from FAISS
    top_features: list[tuple[str, float]]          # (column, gain) sorted desc
    target_col_name: str = "5-year-ahead real GDP per-capita growth"
    conformal_applied: bool = True

    def to_prompt(self) -> tuple[str, str]:
        """Return (system_prompt, user_prompt)."""
        sys_prompt = (
            "You are an economic scenario explainer. You will be given the deterministic "
            "output of a forecasting model — DO NOT modify or invent any numbers from the "
            "model output. Compose a brief 3-5 sentence narrative explaining WHY the model "
            "made this prediction, grounded in: (a) the indicators shown, (b) the historical "
            "analogs shown, and (c) the model's track record on the test set. Be specific, "
            "cite figures directly from the input, and explain any unusual uncertainty. "
            "If a feature is missing for a country-year, say so rather than inventing. "
            "Return your answer as a JSON object with keys: summary, key_drivers, "
            "analog_notes, uncertainty_notes."
        )

        a_low = round(self.pi80_low, 4); a_high = round(self.pi80_high, 4)
        ens = round(self.ensemble, 4)
        direction = "growth" if ens > 0 else "contraction"
        band_contains_zero = self.pi80_low <= 0 <= self.pi80_high

        macro_lines = []
        for col, val in sorted(self.macro_snapshot.items(), key=lambda kv: -abs(kv[1] or 0)):
            nice = _humanize_indicator(col)
            if pd.isna(val):
                macro_lines.append(f"  - {nice}: not observed in {self.query_year}")
            else:
                macro_lines.append(f"  - {nice}: {val:,.3f}")
        macro_block = "\n".join(macro_lines[:25])  # cap to top 25

        analog_lines = []
        for i, a in enumerate(self.analogs[:5], start=1):
            analog_lines.append(
                f"  {i}. {a.get('iso3')} in {int(a.get('year'))}: "
                f"realized 5y growth={a.get('gdp_pc_growth_5y_fwd', 0):+.3f}, "
                f"cosine similarity={a.get('similarity', 0):.3f}, "
                f"features overlap={int(a.get('n_overlap', 0))}/{len(self.macro_snapshot)}"
            )
        analog_block = "\n".join(analog_lines) if analog_lines else "  (no analogs)"

        feat_lines = [
            f"  - {self._feat_name(c)} (gain={g:.1f})" for c, g in self.top_features[:10]
        ]
        feat_block = "\n".join(feat_lines)

        user_prompt = f"""Country: {self.iso3}
Query year: {self.query_year}
Target: {self.target_col_name}

MODEL OUTPUT (do not modify these):
  ridge     = {round(self.ridge, 4):+.4f}
  lgbm      = {round(self.lgbm, 4):+.4f}
  ensemble  = {ens:+.4f}
  80% band  = [{a_low:+.4f}, {a_high:+.4f}]
  band covers zero (no clear direction): {band_contains_zero}
  band is monotonically ordered (low <= high): {self.monotonic}
  conformal calibration applied: {self.conformal_applied}

KEY MACRO INDICATORS (most-influential first):
{macro_block}

TOP LGBM FEATURE GAINS:
{feat_block}

CLOSEST HISTORICAL ANALOGS (FAISS cosine similarity, query excluded):
{analog_block}

Return a JSON object with these keys:
{{
  "summary":          "<2-3 sentence explanation grounded in the numbers above>",
  "key_drivers":      ["<short driver 1>", "<short driver 2>", ...],
  "analog_notes":     "<how the historical analogs relate to {self.iso3} {self.query_year}>",
  "uncertainty_notes":"<what could push this prediction outside [{a_low:+.4f}, {a_high:+.4f}]>"
}}
"""
        return sys_prompt, user_prompt

    @staticmethod
    def _feat_name(col: str) -> str:
        return _humanize_indicator(col)


# ---------------------------------------------------------------------------
# LLM call (provider-agnostic)
# ---------------------------------------------------------------------------
def call_llm(system_prompt: str, user_prompt: str,
             provider: str | None = None,
             model: str | None = None) -> str:
    """Return the raw LLM response text. Raises RuntimeError if no provider is reachable."""
    provider = provider or _select_provider()
    model = model or _DEFAULT_MODEL[provider]

    if provider == "anthropic":
        try:
            import anthropic  # type: ignore
        except ImportError:
            raise RuntimeError("anthropic SDK not installed; `pip install anthropic`")
        client = anthropic.Anthropic()
        msg = client.messages.create(
            model=model,
            max_tokens=800,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return msg.content[0].text  # type: ignore[no-any-return]

    if provider == "openai":
        try:
            import openai  # type: ignore
        except ImportError:
            raise RuntimeError("openai SDK not installed; `pip install openai`")
        client = openai.OpenAI()
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            response_format={"type": "json_object"},
        )
        return resp.choices[0].message.content or ""

    raise RuntimeError(f"Unknown provider: {provider}")


def _select_provider() -> str:
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.environ.get("CLAUDE_API_KEY"):
        return "anthropic"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    raise RuntimeError("No LLM API key found. Set ANTHROPIC_API_KEY or OPENAI_API_KEY.")


# ---------------------------------------------------------------------------
# Output parsing + fallback
# ---------------------------------------------------------------------------
def _safe_parse_json(text: str) -> dict[str, Any] | None:
    """Extract the first JSON object from an LLM response. Tolerant of stray prose."""
    if not text:
        return None
    # Direct
    try:
        return json.loads(text)
    except Exception:
        pass
    # Find first {...} block
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None


def fallback_narrative(inp: ExplainInput) -> dict[str, Any]:
    """Deterministic stub used when no LLM is reachable. Reformats the input."""
    direction = "growth" if inp.ensemble > 0 else "contraction"
    band_span = round(inp.pi80_high - inp.pi80_low, 4)
    band_contains_zero = inp.pi80_low <= 0 <= inp.pi80_high

    drivers = []
    for col, val in inp.macro_snapshot.items():
        if pd.isna(val) or val == 0:
            continue
        nice = _humanize_indicator(col)
        drivers.append(f"{nice}={val:+.3f}")
        if len(drivers) >= 5:
            break

    analog_lines = []
    for a in inp.analogs[:3]:
        analog_lines.append(
            f"{a.get('iso3')} {int(a.get('year'))}: "
            f"5y realized {a.get('gdp_pc_growth_5y_fwd', 0):+.3f} "
            f"(sim {a.get('similarity', 0):.2f})"
        )

    return {
        "summary": (
            f"Model ensemble forecast for {inp.iso3} in {inp.query_year} is "
            f"{inp.ensemble:+.4f} (5y log-return), implying moderate {direction}. "
            f"80% band is [{inp.pi80_low:+.4f}, {inp.pi80_high:+.4f}] "
            f"(width {band_span:.3f}{', band crosses zero' if band_contains_zero else ''}). "
            "Set ANTHROPIC_API_KEY or OPENAI_API_KEY to get a model-generated narrative."
        ),
        "key_drivers": drivers,
        "analog_notes": "; ".join(analog_lines) if analog_lines else "no analogs available",
        "uncertainty_notes": (
            f"Width of the band ({band_span:.3f}) reflects residual variance after "
            f"feature conditioning; {len(inp.analogs)} historical analogs were retrieved."
        ),
        "_fallback": True,
    }


def explain(inp: ExplainInput, *, use_llm: bool = True) -> dict[str, Any]:
    """Return the structured explanation. Tries LLM first, falls back on any error."""
    sys_p, user_p = inp.to_prompt()
    if not use_llm:
        return fallback_narrative(inp)
    try:
        text = call_llm(sys_p, user_p)
        parsed = _safe_parse_json(text) or {"summary": text, "key_drivers": [], "_raw": True}
        parsed["_model_input"] = {"system": sys_p, "user": user_p}
        return parsed
    except Exception as e:
        out = fallback_narrative(inp)
        out["_llm_error"] = repr(e)
        return out


# ---------------------------------------------------------------------------
# Convenience: assemble ExplainInput from a predict_country.py output dict.
# ---------------------------------------------------------------------------
def from_predict_output(
    predict_out: dict[str, Any],
    panel_row: pd.Series,
    analogs_df: pd.DataFrame,
    top_features: list[tuple[str, float]],
    *,
    feature_cols: list[str] | None = None,
) -> ExplainInput:
    """Build an ExplainInput from the JSON output of predict_country.py."""
    fc = predict_out["forecast"]
    # `analogs_df` may come from FAISS (cols: similarity, n_overlap, distance)
    # or the L2 fallback (col: _distance). Normalize to a uniform shape.
    if "similarity" in analogs_df.columns:
        sim = analogs_df["similarity"].astype(float)
    elif "_distance" in analogs_df.columns:
        # Convert L2 distance back to a [0,1] similarity proxy: 1 / (1 + d)
        sim = 1.0 / (1.0 + analogs_df["_distance"].astype(float))
    else:
        sim = pd.Series([0.0] * len(analogs_df))
    if "n_overlap" not in analogs_df.columns:
        n_overlap = pd.Series([0] * len(analogs_df))
    else:
        n_overlap = analogs_df["n_overlap"].astype(int)
    analogs = (
        pd.DataFrame({
            "iso3":                 analogs_df["iso3"],
            "year":                 analogs_df["year"].astype(int),
            "gdp_pc_growth_5y_fwd": analogs_df["gdp_pc_growth_5y_fwd"],
            "similarity":           sim,
            "n_overlap":            n_overlap,
        })
        .head(10)
        .to_dict("records")
    )
    snapshot = (
        {c: float(panel_row[c]) for c in (feature_cols or []) if c in panel_row.index}
        if feature_cols
        else {c: float(v) for c, v in panel_row.items() if pd.api.types.is_numeric(v)}
    )
    return ExplainInput(
        iso3=predict_out["iso3"],
        query_year=int(predict_out["query_year"]),
        ridge=float(fc["ridge"]),
        lgbm=float(fc["lgbm"]),
        ensemble=float(fc["ensemble"]),
        pi80_low=float(predict_out["pi80_low"]),
        pi80_high=float(predict_out["pi80_high"]),
        monotonic=float(predict_out["pi80_low"]) <= float(predict_out["pi80_high"]),
        macro_snapshot=snapshot,
        analogs=analogs,
        top_features=top_features,
    )


if __name__ == "__main__":
    # Smoke test with a fake input
    demo = ExplainInput(
        iso3="USA",
        query_year=2018,
        ridge=-0.5,
        lgbm=0.2,
        ensemble=-0.15,
        pi80_low=-0.05,
        pi80_high=0.12,
        monotonic=True,
        macro_snapshot={
            "gdp_pc_real": 52000.0, "inflation_rate": 0.024,
            "gov_debt_gdp": 1.06, "unemployment_rate": 0.039,
        },
        analogs=[{"iso3": "CHN", "year": 2019, "gdp_pc_growth_5y_fwd": 0.27,
                  "similarity": 0.91, "n_overlap": 72}],
        top_features=[("gdp_pc_nominal_local", 59), ("fx_to_usd_delta5", 43)],
    )
    sys_p, user_p = demo.to_prompt()
    print("=== SYSTEM ===\n" + sys_p)
    print("\n=== USER ===\n" + user_p)
    print("\n=== FALLBACK ===\n", json.dumps(fallback_narrative(demo), indent=2))