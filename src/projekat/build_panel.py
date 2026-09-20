"""Orkestrira ceo pipeline sloja za merenje za jedan ili više simbola:
učitavanje -> čišćenje -> preuzorkovanje po mreži -> izgradnja panela po dizajnu
-> čuvanje u data/processed/, zajedno sa dijagnostikom panela.

Upotreba:
    uv run python -m projekat.build_panel --symbols AAPL --start 2015-01-01 --end 2016-01-01 --grids 5m --estimators bpv --jump-tests bns

Pored panela (`--out`, npr. panel.parquet) pišu se i:
    <stem>_exclusions.parquet   jedan red po izbačenom (simbol, dan) sa razlogom
    <stem>_corporate.csv        status provere korporativnih akcija po simbolu
    <stem>_diagnostics.csv      udeo dana sa skokom, Jaccard, dani po mreži,
                                preskočene provere, izbačeni dani po razlogu
                                (qa/panel_diagnostics.py)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from projekat.config import GRIDS, PROCESSED_DIR
from projekat.data.calendar import is_half_day, sessions_between
from projekat.data.clean import clean_day_with_reason
from projekat.data.corporate import UnadjustedDataError, fetch_corporate_actions, verify_split_adjustment
from projekat.data.vault import load_1m_range
from projekat.data.vix import fetch_vix_daily, vix_lag1_on_calendar
from projekat.measure.design import MeasurementDesign
from projekat.measure.grids import log_returns, resample_day
from projekat.measure.panel import build_symbol_design
from projekat.measure.schema import validate_panel
from projekat.qa.checks import assert_min_intraday_obs, assert_semivariance_identity
from projekat.qa.panel_diagnostics import build_panel_diagnostics, side_file_paths


def _load_raw_bars(symbol: str, start: str, end: str) -> pd.DataFrame:
    # Chunk-aware: the vault's /export silently truncates at ~2.5M rows, so
    # long ranges are fetched in 8-year pieces (data/vault.py). load_1m_range
    # stitches whatever pieces exist back together and falls back to a single
    # whole-range file for older cache entries, so nothing downstream needs
    # to know which way a symbol was fetched.
    df = load_1m_range(symbol, start, end)
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


def returns_by_grid_from_raw(
    symbol: str, raw_by_day: dict[str, pd.DataFrame], grids, *, start: str, end: str
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, bool], list[dict]]:
    """Očisti svaki dan JEDNOM (odluka ne zavisi od mreže -- data/clean.py),
    pa preuzorkuj zadržane svećice na svaku mrežu. Vraća
    ({mreža: {dan: prinosi}}, {dan: skraćen}, izbačeni dani sa razlogom).

    Dan ostaje samo ako ga SVE mreže zadržavaju, pa 1m/5m/15m imaju isti
    skup dana po konstrukciji."""
    returns: dict[str, dict[str, np.ndarray]] = {g: {} for g in grids}
    half_day_flags: dict[str, bool] = {}
    exclusions: list[dict] = []

    for day, bars in raw_by_day.items():
        cleaned, reason = clean_day_with_reason(bars, day)
        if cleaned is None:
            exclusions.append({"symbol": symbol, "date": day, "reason": reason})
            continue
        per_grid = {g: log_returns(resample_day(cleaned, g)) for g in grids}
        if any(len(r) < 2 for r in per_grid.values()):
            # coverage_failure već zahteva >= MIN_INTRADAY_OBS prinosa na
            # svakoj mreži; ovo je samo zaštita
            exclusions.append({"symbol": symbol, "date": day, "reason": "fewer_than_2_returns"})
            continue
        for g, r in per_grid.items():
            returns[g][day] = r
        half_day_flags[day] = is_half_day(day)

    # sesije u opsegu dobavljačevih podataka za koje nema nijedne svećice
    if raw_by_day:
        first, last = min(raw_by_day), max(raw_by_day)
        lo, hi = max(first, start), min(last, end)
        if lo <= hi:
            for session in sessions_between(lo, hi):
                day = session.strftime("%Y-%m-%d")
                if day not in raw_by_day:
                    exclusions.append({"symbol": symbol, "date": day, "reason": "no_vendor_bars"})

    return returns, half_day_flags, exclusions


def build_panel_for_symbol(
    symbol: str,
    start: str,
    end: str,
    designs: list[MeasurementDesign],
    *,
    vix_lag1_by_day: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Panel jednog simbola za sve dizajne, plus izbačeni dani i status
    provere korporativnih akcija."""
    print(f"[{symbol}] fetching 1m bars {start}..{end}...", file=sys.stderr)
    raw = _load_raw_bars(symbol, start, end)
    print(f"[{symbol}] {len(raw)} raw 1m bars", file=sys.stderr)
    raw_by_day = _bars_by_day(raw)
    del raw

    grids = sorted({d.grid for d in designs}, key=list(GRIDS).index)
    returns, half_day_flags, exclusions = returns_by_grid_from_raw(symbol, raw_by_day, grids, start=start, end=end)
    print(f"[{symbol}] {len(half_day_flags)} usable trading days (identical on {grids})", file=sys.stderr)

    # Hard gate (data/corporate.py): raise loudly if the vault has started
    # serving split-unadjusted prices. Only the fetch can fail non-fatally
    # (network/vendor); a real UnadjustedDataError propagates and stops the
    # build. Every skip is RECORDED -- it is a hole in a hard check, and the
    # thesis must be able to say how many there were.
    corporate = {"symbol": symbol, "status": "checked", "error": ""}
    try:
        splits, _dividends = fetch_corporate_actions(symbol, start, end)
        verify_split_adjustment(_daily_close(raw_by_day), splits)
    except UnadjustedDataError:
        raise
    except Exception as e:
        corporate = {"symbol": symbol, "status": "skipped_fetch_failed", "error": f"{type(e).__name__}: {e}"[:500]}
        print(f"[{symbol}] WARNING: corporate-actions fetch failed ({e}), split gate skipped", file=sys.stderr)
    del raw_by_day

    frames = []
    lm_cache: dict = {}   # LM statistike po mreži, zajedničke za sve ocenjivače
    for design in designs:
        print(f"[{symbol}] building panel for design={design.id}...", file=sys.stderr)
        frames.append(
            build_symbol_design(
                symbol, design, returns[design.grid], half_day_flags, vix_lag1_by_day, lm_cache=lm_cache
            )
        )

    panel = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return panel, pd.DataFrame(exclusions, columns=["symbol", "date", "reason"]), corporate


def _vix_lag1(start: str, end: str) -> dict[str, float] | None:
    try:
        vix_raw = fetch_vix_daily(start, end)
        lag1 = vix_lag1_on_calendar(vix_raw, sessions_between(start, end))
        return {d.strftime("%Y-%m-%d"): v for d, v in lag1.items()}
    except Exception as e:
        print(f"WARNING: VIX fetch failed ({e}), vix_lag1 will be NaN", file=sys.stderr)
        return None


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
    vix_lag1_by_day = _vix_lag1(args.start, args.end)

    # Per-symbol isolation: a symbol that fails is REPORTED and the build
    # continues; the summary below names every one.
    all_frames, all_exclusions, corporate_rows = [], [], []
    failed: list[tuple[str, str]] = []
    for i, symbol in enumerate(args.symbols, start=1):
        try:
            panel, exclusions, corporate = build_panel_for_symbol(
                symbol, args.start, args.end, designs, vix_lag1_by_day=vix_lag1_by_day
            )
            all_frames.append(panel)
            all_exclusions.append(exclusions)
            corporate_rows.append(corporate)
            print(f"[{i}/{len(args.symbols)}] {symbol}: {len(panel)} rows", file=sys.stderr)
        except KeyboardInterrupt:
            print(f"\n[SIGINT] stopped during {symbol}", file=sys.stderr)
            raise
        except Exception as exc:
            failed.append((symbol, f"{type(exc).__name__}: {exc}"))
            corporate_rows.append({"symbol": symbol, "status": "build_failed", "error": f"{type(exc).__name__}: {exc}"[:500]})
            print(f"[{i}/{len(args.symbols)}] {symbol}: FAILED {type(exc).__name__}: {exc}", file=sys.stderr)

    full_panel = pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()
    exclusions = pd.concat(all_exclusions, ignore_index=True) if all_exclusions else pd.DataFrame(
        columns=["symbol", "date", "reason"])
    corporate = pd.DataFrame(corporate_rows, columns=["symbol", "status", "error"])
    print(f"Full panel: {len(full_panel)} rows, {full_panel['symbol'].nunique() if len(full_panel) else 0} symbols", file=sys.stderr)
    if failed:
        print(f"\n{len(failed)} symbol(s) FAILED and are absent from the panel:", file=sys.stderr)
        for symbol, err in failed:
            print(f"  {symbol}: {err[:200]}", file=sys.stderr)

    if len(full_panel):
        validate_panel(full_panel)
        # Hard QA gates (qa/checks.py) against the ACTUAL built panel.
        assert_semivariance_identity(full_panel)
        assert_min_intraday_obs(full_panel["n_obs"], context="build_panel.main")

    out_path = Path(args.out or str(PROCESSED_DIR / "panel_mini.parquet"))
    # Panel se upisuje TEK OVDE, na kraju izgradnje. Ako izlazni direktorijum
    # ne postoji, to_parquet baca OSError i ceo posao (oko 20 minuta za pun
    # univerzum) propada u poslednjem koraku -- pa se direktorijum pravi ovde.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    full_panel.to_parquet(out_path, index=False)
    print(f"Saved panel to {out_path}", file=sys.stderr)

    paths = side_file_paths(out_path)
    exclusions.to_parquet(paths["exclusions"], index=False)
    corporate.to_csv(paths["corporate"], index=False)
    if len(full_panel):
        diagnostics = build_panel_diagnostics(full_panel, exclusions, corporate)
        diagnostics.to_csv(paths["diagnostics"], index=False)
        print(f"Saved diagnostics to {paths['diagnostics']}", file=sys.stderr)

    # Non-zero exit if any symbol is missing: a partial panel is a valid
    # artefact to keep, but the caller must be able to tell it apart.
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
