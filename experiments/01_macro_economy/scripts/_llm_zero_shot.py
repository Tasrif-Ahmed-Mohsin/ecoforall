r"""Zero-shot LLM baseline for GDP-per-capita growth forecasting.

Compares a frontier LLM (DeepSeek-V4 / `deepseek-v4` alias) against the project's
v2 meta-ensemble on the same canonical test slice. The LLM is asked to predict
the *forward-h-year* growth rate for a single (iso3, year) row using only the
information the panel ML "saw" at training time — no future leakage.

Key design choices:
- **Provider**: DeepSeek via the OpenAI-compatible API (`api.deepseek.com`).
  No new dependency: the OpenAI Python SDK already in the project is reused.
  Auth comes from `$env:DEEPSEEK_API_KEY` (never read from `deepseek.txt`).
- **Slice**: holdout only (year 2023, n≈213). COVID-free by construction;
  same slice used for §6.1 / `_no_covid.py` checks.
- **Horizons**: defaults to {1, 5} per user request; `--horizon` is overridable.
- **Prompt information set**: last 5 years of realized 1y growth + 4 contemporaneous
  macro features (inflation, trade_openness, gdp_pc_real level, investment_share).
  Identical to what the panel ML's top-10 LGBM gain features include, so the LLM
  is not unfairly under-informed.
- **Determinism**: `temperature=0`. Prompts and full responses are logged so
  re-runs are auditable.
- **No-stub mode**: if `DEEPSEEK_API_KEY` is missing the script aborts loudly
  (no fake fallback) so it can never produce a misleading "result".

Outputs:
- `data/features/llm_baseline_holdout.csv` — one row per (iso3, year, horizon)
  with y_true, our meta pred, prior pred, LLM pred, raw response, latency_ms.
- `data/features/llm_baseline_{split}_h{h...}_metrics.json` — per-horizon /
  per-split MAE/RMSE/dir_acc for (a) per-country prior, (b) our v2 meta,
  (c) DeepSeek-V4 zero-shot. The default `headline` branch writes
  `llm_baseline_metrics.json` (composite); per-horizon runs (e.g. `--horizon 1
  --split val`) write `llm_baseline_val_h1_metrics.json` etc. Match the JSON
  to the CSV you actually generated — both must be from the same run.

Reproduce (PowerShell):
    $env:DEEPSEEK_API_KEY = (Get-Content E:\project_gmd\deepseek.txt).Trim() -replace '^api_key=',''
    python scripts\\_llm_zero_shot.py                                # h={1,5}, full holdout
    python scripts\\_llm_zero_shot.py --horizon 1 --max-rows 20       # smoke test
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

FEATURES_DIR = ROOT / "data" / "features"
META_DIR = FEATURES_DIR / "cross_horizon_meta"
PANEL = FEATURES_DIR / "panel_wide.parquet"
OUT_CSV = FEATURES_DIR / "llm_baseline_holdout.csv"
OUT_JSON = FEATURES_DIR / "llm_baseline_metrics.json"

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4"  # DeepSeek-V4 chat-completion alias (per API billing: V4 tokens)

# Macro features surfaced to the LLM (top of LGBM gain importance, no future leakage).
PROMPT_FEATURES = [
    ("gdp_pc_real",                "real GDP per capita (USD)"),
    ("infl_cpi",                   "annual CPI inflation (%)"),
    ("trade_openness",             "trade openness (% of GDP)"),
    ("inv_share_gdp",              "investment share of GDP (%)"),
]


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------
def _format_prompt(panel: pd.DataFrame, iso3: str,
                    year: int, horizon: int) -> tuple[str, str]:
    """Build the (system, user) prompt pair for a single (iso3, year, horizon).

    The LLM is blind to our model's predictions: it receives only the country's
    own past + contemporaneous macro features (the same info set the panel ML's
    top-10 LGBM gain features come from). This is the "zero-shot LLM" baseline,
    not a critic.

    Realized 1y growth is computed locally as pct_change of gdp_pc_real, since
    the panel parquet does not pre-store a column named *_fwd.
    """
    sub = panel[(panel["iso3"] == iso3) & (panel["year"] <= year)].sort_values("year")
    if sub.empty:
        raise ValueError(f"no panel rows for {iso3} <= {year}")

    # Compute realized 1y growth from the level (no future leakage).
    sub = sub.copy()
    sub["g1"] = sub["gdp_pc_real"].pct_change()

    # 5 most recent realized 1y growth values.
    hist = sub.dropna(subset=["g1"]).tail(5)[["year", "g1"]]
    hist_str = "\n".join(
        f"  {int(r.year)}: {float(r.g1):+.3f}"
        for r in hist.itertuples()
    ) or "  (no history)"

    # Contemporaneous macro snapshot at the most recent row <= baseline year.
    last_row = sub.iloc[-1]
    macro_str = "\n".join(
        f"  {human}: {float(last_row[code]):.3f}"
        for code, human in PROMPT_FEATURES
        if code in last_row.index and pd.notna(last_row[code])
    ) or "  (no macro features)"

    system = (
        "You are a careful macroeconomic forecaster. You receive only past data; "
        "you MUST NOT invent future values. Respond with a single JSON object "
        "exactly matching the schema: {\"forecast\": <number>, \"reasoning\": "
        "\"<1 short sentence>\"}. The forecast is the expected annualized GDP "
        "per-capita growth over the next H years (in percent per year). Numbers "
        "should be in percent, e.g. 1.7 means +1.7 %/yr."
    )

    user = (
        f"Country: {iso3}\n"
        f"Baseline year: {year}\n"
        f"Forecast horizon: {horizon} year(s)\n"
        f"Most recent realized annual GDP-per-capita growth:\n{hist_str}\n\n"
        f"Most recent macro snapshot:\n{macro_str}\n\n"
        f"Return: {{\"forecast\": <number>, \"reasoning\": \"...\"}}"
    )
    return system, user


# ---------------------------------------------------------------------------
# Provider (DeepSeek via OpenAI SDK)
# ---------------------------------------------------------------------------
def _client():
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise RuntimeError(
            "DEEPSEEK_API_KEY is not set. Set it from your secret store: "
            "$env:DEEPSEEK_API_KEY = (Get-Content E:\\project_gmd\\deepseek.txt).Trim() "
            "-replace '^api_key=',''"
        )
    try:
        from openai import OpenAI  # type: ignore
    except ImportError as e:
        raise RuntimeError("openai SDK not installed; pip install openai") from e
    return OpenAI(api_key=key, base_url=DEEPSEEK_BASE_URL)


def _call(client, system: str, user: str, model: str) -> tuple[str, float]:
    """Returns (response_text, latency_ms)."""
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        temperature=0,
        max_tokens=200,
        response_format={"type": "json_object"},
    )
    latency_ms = (time.perf_counter() - t0) * 1000
    text = resp.choices[0].message.content or ""
    return text, latency_ms


_FLOAT_RE = re.compile(r"-?\d+\.?\d*")


def _parse_forecast(text: str) -> float | None:
    """Extract the forecast number from a JSON response, in the y_true units.

    y_true is stored as a decimal (0.022 = 2.2% growth). The system prompt
    explicitly tells the LLM to respond in percent-per-year (e.g. 1.7 means
    +1.7 %/yr), so we **always** divide by 100 to convert to the decimal
    units used by the rest of the pipeline.

    Previous bug: only values with |x| > 1 were divided, so an LLM response
    of 0.5 (meaning 0.5 %) was kept as 0.5 (= 50 %), inflating MAE ~100×.
    """
    raw: float | None = None
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "forecast" in obj:
            raw = float(obj["forecast"])
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    if raw is None:
        # Fallback: scan for the first number in the text.
        m = _FLOAT_RE.search(text)
        if not m:
            return None
        raw = float(m.group(0))
    # Prompt asks for percent → always convert to decimal.
    raw = raw / 100.0
    return raw


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _load_split_targets(horizons: list[int], split: str) -> pd.DataFrame:
    """Build a (iso3, year, horizon, y_true) frame from the v2 forecasts.

    `split` is one of 'train' | 'val' | 'test' | 'holdout'. This is a more
    general form of the old `_load_holdout_targets`: that function used a
    per-horizon default (holdout for h=1, test for h>=3) and we keep the same
    behaviour when the caller passes the special name 'headline'.
    """
    if split == "headline":
        SPLITS = {1: "holdout", 3: "test", 5: "test", 10: "test"}
        frames = []
        for h in horizons:
            df = pd.read_parquet(FEATURES_DIR / f"horizon_{h}y_v2" / "forecasts.parquet")
            df = df[df["split"] == SPLITS[h]][["iso3", "year", "y_true"]].copy()
            df["horizon"] = h
            frames.append(df)
        out = pd.concat(frames, ignore_index=True)
    else:
        frames = []
        for h in horizons:
            df = pd.read_parquet(FEATURES_DIR / f"horizon_{h}y_v2" / "forecasts.parquet")
            df = df[df["split"] == split][["iso3", "year", "y_true"]].copy()
            df["horizon"] = h
            frames.append(df)
        out = pd.concat(frames, ignore_index=True)
    out = out.dropna(subset=["y_true"])
    return out


def _load_holdout_targets(horizons: list[int]) -> pd.DataFrame:
    """Backwards-compatible alias: the canonical 'headline' slice."""
    return _load_split_targets(horizons, "headline")


def _load_meta_predictions(meta_dir: Path) -> pd.DataFrame:
    """Pivot the cross_horizon_meta predictions into one row per (iso3, year).

    The cross-horizon Ridge stack (artifact produced by
    ``_cross_horizon_ensemble.py``) only emits predictions for the *test* split
    of each horizon. For horizons where the chosen comparison slice has no
    overlap (e.g. h=1 on the 2023 holdout) we fall back to the per-horizon v2
    ensemble (``y_pred_ensemble``), and record the source so the comparison
    table stays honest.
    """
    p = meta_dir / "predictions.parquet"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    df = df.dropna(subset=["pred_meta"])
    df = df.rename(columns={"pred_meta": "meta_pred"})
    df["meta_source"] = "cross_horizon_meta"
    return df[["iso3", "year", "horizon", "meta_pred", "y_true", "meta_source"]]


def _load_v2_fallback(horizon: int) -> pd.DataFrame:
    """Per-horizon v2 ensemble predictions, for slices the cross-horizon meta
    never scored (e.g. h=1 holdout).

    When ``y_pred_ensemble`` is all-NaN (holdout rows are not scored by the
    ensemble step), we reconstruct it from the component predictions using
    the ``ensemble_recipe`` stored in the parquet (typically ``lgbm+prior``).
    """
    p = FEATURES_DIR / f"horizon_{horizon}y_v2" / "forecasts.parquet"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)

    # If y_pred_ensemble is all-NaN (e.g. holdout), reconstruct from recipe.
    mask_null = df["y_pred_ensemble"].isna()
    if mask_null.any():
        recipe = df["ensemble_recipe"].dropna().iloc[0] if "ensemble_recipe" in df.columns else ""
        parts = recipe.split("+") if recipe else []
        pred_cols = [f"y_pred_{p}" for p in parts if f"y_pred_{p}" in df.columns]
        if pred_cols:
            df.loc[mask_null, "y_pred_ensemble"] = (
                df.loc[mask_null, pred_cols].mean(axis=1)
            )

    df = df.dropna(subset=["y_pred_ensemble"])
    df = df.rename(columns={"y_pred_ensemble": "meta_pred"})
    df["meta_source"] = "horizon_v2_ensemble"
    df["horizon"] = horizon
    return df[["iso3", "year", "horizon", "meta_pred", "y_true", "meta_source"]]


def _load_prior_predictions(horizons: list[int]) -> pd.DataFrame:
    """Per-country prior (last-realised training-period growth) per horizon."""
    SPLITS = {1: "holdout", 3: "test", 5: "test", 10: "test"}
    rows = []
    for h in horizons:
        df = pd.read_parquet(FEATURES_DIR / f"horizon_{h}y_v2" / "forecasts.parquet")
        split = SPLITS[h]
        df = df[df["split"] == split][["iso3", "year", "y_pred_prior"]].copy()
        df["horizon"] = h
        rows.append(df)
    return pd.concat(rows, ignore_index=True).rename(columns={"y_pred_prior": "prior_pred"})


def _metrics(y: np.ndarray, p: np.ndarray) -> dict:
    diff = y - p
    return {
        "n": int(len(y)),
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
        "dir_acc": float(np.mean(np.sign(y) == np.sign(p))),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Zero-shot LLM GDP forecaster baseline")
    ap.add_argument("--horizon", type=int, nargs="+", default=[1, 5],
                    help="horizons to score (default: 1 5)")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help=f"DeepSeek model name (default: {DEFAULT_MODEL})")
    ap.add_argument("--max-rows", type=int, default=None,
                    help="cap number of (iso3, year) cells; useful for smoke tests")
    ap.add_argument("--split", default="headline",
                    choices=["headline", "train", "val", "test", "holdout"],
                    help="which split to query the LLM on "
                         "(default: 'headline' = h=1->holdout, h>=3->test)")
    ap.add_argument("--out-csv", default=None,
                    help="output CSV path (default: derived from --split)")
    ap.add_argument("--out-json", default=None,
                    help="output JSON path (default: derived from --split, "
                         "preserves the headline file from accidental overwrite)")
    ap.add_argument("--reparse", action="store_true",
                    help="Re-parse llm_pred from cached CSV responses (no API calls)")
    ap.add_argument("--resume", action="store_true",
                    help="Resume a partial run: skip (iso3, year, horizon) keys "
                         "already present in --out-csv. Saves API tokens after a "
                         "crash. CSV is also flushed every 20 rows in any case.")
    ap.add_argument("--offset", type=int, default=0,
                    help="Skip the first N target cells before processing. "
                         "Combine with --max-rows to slice the run into batches.")
    ap.add_argument("--min-interval", type=float, default=1.5,
                    help="Minimum seconds between API calls. DeepSeek rate-limits "
                         "aggressively after the first ~50 calls in a session; "
                         "1.5s keeps the loop well below that ceiling. Lower it "
                         "for small smoke tests, raise it for sustained runs.")
    args = ap.parse_args()

    horizons = sorted(set(args.horizon))
    if not set(horizons).issubset({1, 3, 5, 10}):
        raise SystemExit(f"--horizon must be subset of {{1,3,5,10}}, got {horizons}")

    # If user didn't override --out-csv, derive a split-aware default so we
    # never silently overwrite the canonical headline CSV.
    if args.out_csv is None:
        suffix = "" if args.split == "headline" else f"_{args.split}"
        args.out_csv = str(FEATURES_DIR / f"llm_baseline{suffix}.csv")
    if args.out_json is None:
        # Default metrics file is keyed by BOTH split and horizons, so
        # running each horizon separately writes a distinct file. Headline
        # stays untouched.
        if args.split == "headline":
            # The `headline` branch writes one composite JSON for all
            # horizons in one run. Per-horizon runs (most common) write
            # `llm_baseline_{split}_h{h...}_metrics.json` instead.
            args.out_json = str(FEATURES_DIR / "llm_baseline_metrics.json")
        else:
            h_tag = "h" + "-".join(str(h) for h in horizons)
            args.out_json = str(FEATURES_DIR /
                                f"llm_baseline_{args.split}_{h_tag}_metrics.json")

    print(f"[llm] model={args.model}  horizons={horizons}  "
          f"max_rows={args.max_rows or 'all'}  reparse={args.reparse}")

    # --reparse mode: re-read cached CSV and recompute llm_pred from stored
    # responses. Avoids burning API tokens when only the parse logic changed.
    if args.reparse:
        if not Path(args.out_csv).exists():
            raise SystemExit(f"--reparse requires existing {args.out_csv}")
        out = pd.read_csv(args.out_csv)
        out = out[out["horizon"].isin(horizons)]
        if args.max_rows is not None:
            out = out.head(args.max_rows)
        n_before = out["llm_pred"].notna().sum()
        out["llm_pred"] = out["llm_response"].apply(_parse_forecast)
        n_after = out["llm_pred"].notna().sum()
        print(f"[llm] reparsed {len(out)} rows: {n_before} -> {n_after} non-null preds")
        out.to_csv(args.out_csv, index=False)
        print(f"[llm] wrote {args.out_csv} ({len(out):,} rows)")
    else:
        client = _client()

        panel = pd.read_parquet(PANEL)
        targets = _load_split_targets(horizons, args.split)
        if args.offset:
            targets = targets.iloc[args.offset:].reset_index(drop=True)
        if args.max_rows is not None:
            targets = targets.head(args.max_rows)
        print(f"[llm] target cells: {len(targets):,}  "
              f"(unique iso3={targets.iso3.nunique()})  split={args.split}  "
              f"offset={args.offset}")

        # Prior predictions are only available for the canonical headline slice
        # (holdout for h=1, test for h>=3). For other splits we still merge what
        # we have; missing prior_pred is fine (it just won't be reported).
        if args.split == "headline":
            prior = _load_prior_predictions(horizons)
            targets = targets.merge(prior, on=["iso3", "year", "horizon"], how="left")
        else:
            # Try to merge priors anyway from the per-horizon v2 forecasts.
            # For val rows these DO exist (v2 trainer scored them).
            try:
                prior = _load_prior_predictions(horizons)
                # Re-read with the actual split (override the SPLITS mapping).
                rows = []
                for h in horizons:
                    df = pd.read_parquet(FEATURES_DIR / f"horizon_{h}y_v2" / "forecasts.parquet")
                    df = df[df["split"] == args.split][["iso3", "year", "y_pred_prior"]].copy()
                    df["horizon"] = h
                    rows.append(df)
                prior_split = pd.concat(rows, ignore_index=True).rename(
                    columns={"y_pred_prior": "prior_pred"})
                targets = targets.merge(prior_split, on=["iso3", "year", "horizon"], how="left")
                print(f"[llm] merged prior_pred from v2 forecasts ({args.split}): "
                      f"{targets['prior_pred'].notna().sum()}/{len(targets)}")
            except Exception as e:
                print(f"[llm] no prior_pred available for split={args.split}: {e}")

        # Iterative call loop. We log the prompt only for the first row to keep the
        # CSV small; the user can re-run with --max-rows 1 if they want a sample.
        rows = []
        failed = 0
        t_start = time.perf_counter()
        # Optional resume: if --resume and the output CSV already exists,
        # load its (iso3, year, horizon) keys and skip those rows so we don't
        # double-spend API tokens after a crash.
        done_keys: set[tuple[str, int, int]] = set()
        if args.resume and Path(args.out_csv).exists():
            prev = pd.read_csv(args.out_csv)
            prev["year"] = pd.to_numeric(prev["year"], errors="coerce")
            prev["horizon"] = pd.to_numeric(prev["horizon"], errors="coerce").astype("Int64")
            prev = prev.dropna(subset=["iso3", "year", "horizon", "llm_pred"])
            done_keys = set(zip(prev["iso3"].astype(str),
                               prev["year"].astype(int),
                               prev["horizon"].astype(int)))
            rows = prev.to_dict("records")
            failed = int(prev["llm_response"].fillna("").str.startswith("[error]").sum())
            print(f"[llm] --resume: loaded {len(done_keys)} completed keys "
                  f"from {args.out_csv} (failed_so_far={failed})")
        # Per-call throttle: enforce a minimum gap between API calls so we don't
        # burst past DeepSeek's per-minute rate limit. On 429s we extend the gap.
        last_call_at = 0.0
        rate_limit_streak = 0
        for idx, cell in enumerate(targets.itertuples(index=False), 1):
            iso3, year, horizon, y_true, prior_pred = (
                cell.iso3, int(cell.year), int(cell.horizon), float(cell.y_true),
                float(cell.prior_pred) if pd.notna(cell.prior_pred) else float("nan"),
            )
            key = (str(iso3), int(year), int(horizon))
            if key in done_keys:
                continue
            # Enforce min interval BEFORE the call. If the previous call
            # returned a 429 we sleep extra (exponential back-off capped at
            # 60s) until the throttle streak clears.
            now = time.perf_counter()
            wait = args.min_interval - (now - last_call_at)
            extra = min(60.0, 2 ** rate_limit_streak) if rate_limit_streak else 0.0
            if wait + extra > 0:
                time.sleep(wait + extra)
            # Inner retry: 3 attempts. The first retry waits 2s; on a 429
            # response we extend the gap before re-trying and bump the
            # rate_limit_streak so subsequent calls back off too.
            text, latency_ms, llm_pred = None, -1.0, None
            for attempt in (1, 2, 3):
                try:
                    system, user = _format_prompt(panel, iso3=iso3, year=year, horizon=horizon)
                    text, latency_ms = _call(client, system, user, args.model)
                    last_call_at = time.perf_counter()
                    llm_pred = _parse_forecast(text)
                    # Success - decay the throttle streak if this call wasn't 429.
                    rate_limit_streak = max(0, rate_limit_streak - 1)
                    break
                except Exception as e:
                    msg = str(e)
                    is_429 = ("429" in msg) or ("rate" in msg.lower()) or ("limit" in msg.lower())
                    if is_429:
                        rate_limit_streak += 1
                    last_call_at = time.perf_counter()
                    if attempt < 3:
                        # Back off 2s before retry, longer if 429.
                        time.sleep(2.0 * (2 if is_429 else 1))
                        continue
                    text, latency_ms, llm_pred = f"[error] {e}", -1.0, None
                    failed += 1
            rows.append({
                "iso3": iso3, "year": year, "horizon": horizon,
                "y_true": y_true,
                "prior_pred": prior_pred,
                "llm_pred": llm_pred,
                "llm_response": text,
                "latency_ms": latency_ms,
                "model": args.model,
            })
            done_keys.add(key)
            if idx % 20 == 0 or idx == len(targets):
                elapsed = time.perf_counter() - t_start
                print(f"[llm] {idx}/{len(targets)}  failed={failed}  "
                      f"elapsed={elapsed:.1f}s  eta~{(elapsed/idx)*(len(targets)-idx):.1f}s",
                      flush=True)
                # Flush to disk every progress interval so a mid-run crash
                # loses at most 20 rows. We always rewrite the whole file from
                # `rows` rather than appending, so duplicate retries don't pile
                # up the same (iso3, year, horizon) keys. Header is written
                # only when we create the file for the first time.
                tmp_path = str(args.out_csv) + ".tmp"
                pd.DataFrame(rows).to_csv(tmp_path, index=False)
                # Atomic replace.
                os.replace(tmp_path, args.out_csv)

        out = pd.DataFrame(rows)
        print(f"[llm] wrote {args.out_csv} ({len(out):,} rows, split={args.split})")

    # Compute per-horizon metrics for prior, LLM, and our meta (which we now pull
    # from cross_horizon_meta/predictions.parquet).
    meta = _load_meta_predictions(META_DIR)
    summary = {"model": args.model, "horizons": {}}
    for h in horizons:
        sub = out[out["horizon"] == h]
        valid = sub.dropna(subset=["llm_pred", "prior_pred"])
        y = valid["y_true"].to_numpy()
        p_llm = valid["llm_pred"].to_numpy()
        p_prior = valid["prior_pred"].to_numpy()
        summary["horizons"][f"h{h}"] = {
            "n": int(len(valid)),
            "n_failed": int(sub["llm_pred"].isna().sum()),
            "llm": _metrics(y, p_llm),
            "prior": _metrics(y, p_prior),
        }
        merged = pd.DataFrame()
        source = None
        if not meta.empty:
            m_sub = meta[meta["horizon"] == h][["iso3", "year", "meta_pred", "meta_source"]]
            merged = valid.merge(m_sub, on=["iso3", "year"], how="inner")
            if not merged.empty:
                source = "cross_horizon_meta"
        if merged.empty:
            # Fallback: per-horizon v2 ensemble (handles slices the cross-horizon
            # meta never saw, e.g. h=1 on the 2023 holdout).
            v2 = _load_v2_fallback(h)
            if not v2.empty:
                m_sub = v2[["iso3", "year", "meta_pred", "meta_source"]]
                merged = valid.merge(m_sub, on=["iso3", "year"], how="inner")
                if not merged.empty:
                    source = v2["meta_source"].iloc[0]
        if not merged.empty and source is not None:
            m = _metrics(
                merged["y_true"].to_numpy(),
                merged["meta_pred"].to_numpy(),
            )
            m["source"] = source
            m["n"] = int(len(merged))
            summary["horizons"][f"h{h}"]["meta"] = m
    Path(args.out_json).write_text(json.dumps(summary, indent=2))
    print(f"[llm] wrote {args.out_json}")
    print()
    print(f"{'h':<4} {'n':>4} {'prior_mae':>11} {'prior_dir':>10} "
          f"{'llm_mae':>10} {'llm_dir':>10} {'meta_mae':>10} {'meta_dir':>10}")
    for h_key, v in summary["horizons"].items():
        prior = v.get("prior", {})
        llm = v.get("llm", {})
        meta_m = v.get("meta", {})
        print(f"{h_key:<4} {v['n']:>4} {prior.get('mae', float('nan')):>11.4f} "
              f"{prior.get('dir_acc', float('nan')):>10.3f} "
              f"{llm.get('mae', float('nan')):>10.4f} "
              f"{llm.get('dir_acc', float('nan')):>10.3f} "
              f"{meta_m.get('mae', float('nan')):>10.4f} "
              f"{meta_m.get('dir_acc', float('nan')):>10.3f}")


if __name__ == "__main__":
    main()