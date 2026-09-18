"""Orkestrira ceo pipeline sloja za merenje za jedan ili više simbola:
učitavanje -> čišćenje -> preuzorkovanje po mreži -> izgradnja panela po dizajnu
-> čuvanje u data/processed/.

Upotreba:
    uv run python -m projekat.build_panel --symbols AAPL --start 2015-01-01 --end 2016-01-01 --grids 5m --estimators bpv --jump-tests bns
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from projekat.config import PROCESSED_DIR
from projekat.data.calendar import is_half_day
from projekat.data.clean import clean_day
from projekat.data.corporate import UnadjustedDataError, fetch_corporate_actions, verify_split_adjustment
from projekat.data.vault import load_1m_range
from projekat.data.vix import fetch_vix_daily, vix_lag1_on_calendar
from projekat.measure.design import MeasurementDesign
from projekat.measure.grids import resample_day, log_returns
from projekat.measure.panel import build_symbol_design
from projekat.measure.schema import validate_panel
from projekat.qa.checks import assert_min_intraday_obs, assert_semivariance_identity


def _load_raw_bars(symbol: str, start: str, end: str) -> pd.DataFrame:
    # Chunk-aware: the vault's /export silently truncates at ~2.5M rows, so
    # long ranges are fetched in 8-year pieces (data/vault.py). load_1m_range
    # stitches whatever pieces exist back together and falls back to a single
    # whole-range file for older cache entries, so nothing downstream needs
    # to know which way a symbol was fetched.
    df = load_1m_range(symbol, start, end)
    # normalize expected column names from the vault export
    # uskladi imena kolona sa onima koje se očekuju iz izvoza vault-a
    rename_map = {}
    if "ts" in df.columns and "timestamp" not in df.columns:
        rename_map["ts"] = "timestamp"
    df = df.rename(columns=rename_map)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def _bars_by_day(raw_bars: pd.DataFrame) -> dict[str, pd.DataFrame]:
    raw_bars = raw_bars.copy()
    raw_bars["_day"] = raw_bars["timestamp"].dt.strftime("%Y-%m-%d")
    return {day: g for day, g in raw_bars.groupby("_day")}


def _daily_close(raw_by_day: dict[str, pd.DataFrame]) -> dict[str, float]:
    """Last traded 1m close per day -- used only for the split-adjustment
    hard gate (data/corporate.py), never fed into any measure/model
    computation."""
    out = {}
    for day, bars in raw_by_day.items():
        s = bars.set_index("timestamp")["close"].sort_index()
        if len(s):
            out[day] = float(s.iloc[-1])
    return out


def build_returns_for_symbol(symbol: str, start: str, end: str, grid: str) -> tuple[dict[str, "np.ndarray"], dict[str, bool], dict[str, float]]:
    print(f"[{symbol}] fetching 1m bars {start}..{end}...", file=sys.stderr)
    raw = _load_raw_bars(symbol, start, end)
    print(f"[{symbol}] {len(raw)} raw 1m bars fetched", file=sys.stderr)

    raw_by_day = _bars_by_day(raw)
    returns_by_day = {}
    half_day_flags = {}

    for day, bars in raw_by_day.items():
        cleaned = clean_day(bars, day, grid)
        if cleaned is None:
            continue
        prices = resample_day(cleaned, grid)
        r = log_returns(prices)
        if len(r) < 2:
            continue
        returns_by_day[day] = r
        half_day_flags[day] = is_half_day(day)

    print(f"[{symbol}] {len(returns_by_day)} usable trading days at grid={grid}", file=sys.stderr)
    return returns_by_day, half_day_flags, _daily_close(raw_by_day)


def build_panel_for_symbol(symbol: str, start: str, end: str, designs: list[MeasurementDesign]) -> pd.DataFrame:
    frames = []
    returns_cache: dict[str, tuple[dict, dict, dict]] = {}

    vix_lag1_by_day = None
    try:
        vix_raw = fetch_vix_daily(start, end)
        from projekat.data.calendar import sessions_between

        sessions = sessions_between(start, end)
        lag1 = vix_lag1_on_calendar(vix_raw, sessions)
        vix_lag1_by_day = {d.strftime("%Y-%m-%d"): v for d, v in lag1.items()}
    except Exception as e:
        print(f"[{symbol}] WARNING: VIX fetch failed ({e}), vix_lag1 will be NaN", file=sys.stderr)

    split_gate_checked = False

    for design in designs:
        if design.grid not in returns_cache:
            returns_cache[design.grid] = build_returns_for_symbol(symbol, start, end, design.grid)
        returns_by_day, half_day_flags, daily_close = returns_cache[design.grid]

        if not split_gate_checked:
            # Hard gate (data/corporate.py): raise loudly if the vault has
            # started serving split-unadjusted prices, instead of silently
            # feeding a corporate-action artifact into the jump tests as a
            # legitimate multi-sigma jump. Only the split fetch/check can
            # fail non-fatally here (network/vendor availability); a real
            # UnadjustedDataError must propagate and stop the build.
            try:
                splits, _dividends = fetch_corporate_actions(symbol, start, end)
                verify_split_adjustment(daily_close, splits)
            except UnadjustedDataError:
                raise
            except Exception as e:
                print(f"[{symbol}] WARNING: corporate-actions fetch failed ({e}), split gate skipped", file=sys.stderr)
            split_gate_checked = True

        print(f"[{symbol}] building panel for design={design.id}...", file=sys.stderr)
        df = build_symbol_design(symbol, design, returns_by_day, half_day_flags, vix_lag1_by_day)
        frames.append(df)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--grids", nargs="+", default=["5m"])
    parser.add_argument("--estimators", nargs="+", default=["bpv"])
    parser.add_argument("--jump-tests", nargs="+", default=["bns"])
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    designs = [
        MeasurementDesign(grid=g, estimator=e, jump_test=j)
        for g in args.grids
        for e in args.estimators
        for j in args.jump_tests
    ]
    print(f"Designs: {[d.id for d in designs]}", file=sys.stderr)

    # Per-symbol isolation: a 30-symbol build must not lose 29 completed
    # symbols because the 30th hit a network/vendor failure. A symbol that
    # fails is REPORTED and the build continues; the summary below names
    # every one, so a missing symbol can never pass unnoticed. (Raw bars
    # are cached per symbol, so re-running is cheap -- see
    # scripts/fetch_data.py, which is the resumable way to get them.)
    all_frames = []
    failed: list[tuple[str, str]] = []
    for i, symbol in enumerate(args.symbols, start=1):
        try:
            panel = build_panel_for_symbol(symbol, args.start, args.end, designs)
            all_frames.append(panel)
            print(f"[{i}/{len(args.symbols)}] {symbol}: {len(panel)} rows", file=sys.stderr)
        except KeyboardInterrupt:
            print(f"\n[SIGINT] stopped during {symbol}", file=sys.stderr)
            raise
        except Exception as exc:
            failed.append((symbol, f"{type(exc).__name__}: {exc}"))
            print(f"[{i}/{len(args.symbols)}] {symbol}: FAILED {type(exc).__name__}: {exc}", file=sys.stderr)

    full_panel = pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()
    print(f"Full panel: {len(full_panel)} rows, {full_panel['symbol'].nunique() if len(full_panel) else 0} symbols", file=sys.stderr)
    if failed:
        print(f"\n{len(failed)} symbol(s) FAILED and are absent from the panel:", file=sys.stderr)
        for symbol, err in failed:
            print(f"  {symbol}: {err[:200]}", file=sys.stderr)

    if len(full_panel):
        validate_panel(full_panel)
        # Hard QA gates (qa/checks.py) against the ACTUAL built panel, not
        # just synthetic test fixtures -- these must never fire on real
        # data; if they do, it's a genuine measurement-layer defect, not a
        # data-quality nuisance to work around.
        assert_semivariance_identity(full_panel)
        assert_min_intraday_obs(full_panel["n_obs"], context="build_panel.main")

    out_path = args.out or str(PROCESSED_DIR / "panel_mini.parquet")
    full_panel.to_parquet(out_path, index=False)
    print(f"Saved panel to {out_path}", file=sys.stderr)

    # Non-zero exit if any symbol is missing: a partial panel is a valid
    # artefact to keep (the completed symbols are real), but the caller
    # must be able to tell it apart from a complete one.
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
