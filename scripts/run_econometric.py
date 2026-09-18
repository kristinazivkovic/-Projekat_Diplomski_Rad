"""Entrypoint for generating the raw forecast tables (econometric group).

    uv run python scripts/run_econometric.py --dry-run
    uv run python scripts/run_econometric.py --panel data/processed/panel.parquet

`--dry-run` is side-effect free: it creates no run directory, writes no
file, and fits no model except for timing (and that only under
`--benchmark`). It exists so the combination count and the time estimate
can be inspected BEFORE committing to a multi-hour job.

TIMING IS MEASURED ON THE LONGEST WINDOW
========================================
The windows are anchored and expanding: w01 trains on 2010-2014, w11 on
2010-2024 -- five years against fifteen. An estimate taken on w01 and
multiplied by the window count badly understates any model that grows
faster than linearly in training length (`arfima` is O(n^2) in its
fractional differencing; `lm_fdr` carries a trailing W-day window). So
`--benchmark` measures every model at BOTH w01 and w11, reports both, and
flags superlinear growth instead of assuming it away.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# allow `uv run python scripts/run_econometric.py` without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# The Windows console defaults to cp1252, which cannot encode every
# character we may print -- without this a multi-hour run would die on the
# summary line, AFTER all the work was done. errors="replace" is
# deliberate: output must never crash over a single character.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from projekat.model.runner import HORIZONS  # noqa: E402
from projekat.model.sharded import (  # noqa: E402
    MODEL_GROUPS,
    dedup_report,
    enumerate_combinations,
    parse_int_list,
    parse_window_spec,
    resolve_models,
    run_sharded,
)
from projekat.model.windows import anchored_windows, split_with_validation  # noqa: E402
from projekat.results import store  # noqa: E402
from projekat.results.store import RunManifest, create_run_dir, new_run_id, write_manifest  # noqa: E402


def _git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5,
            cwd=str(Path(__file__).resolve().parents[1]),
        )
        if out.returncode != 0:
            return None    # not a git repository -- record None, not "HEAD"
        sha = out.stdout.strip()
        return sha if len(sha) == 40 and all(c in "0123456789abcdef" for c in sha) else None
    except Exception:
        return None


def _sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _package_versions() -> dict[str, str]:
    out = {}
    for name in ("numpy", "pandas", "statsmodels", "scipy", "pyarrow"):
        try:
            out[name] = __import__(name).__version__
        except Exception:
            pass
    return out


def _filter_panel(panel: pd.DataFrame, *, symbols: str | None, grids: str | None) -> pd.DataFrame:
    out = panel
    if symbols and symbols != "all":
        wanted = [s.strip() for s in symbols.split(",") if s.strip()]
        missing = sorted(set(wanted) - set(out["symbol"].unique()))
        if missing:
            raise SystemExit(f"symbols not in the panel: {missing}")
        out = out[out["symbol"].isin(wanted)]
    if grids and grids != "all":
        wanted = [g.strip() for g in grids.split(",") if g.strip()]
        out = out[out["design_id"].str.split("__").str[0].isin(wanted)]
        if out.empty:
            raise SystemExit(f"no design on grids {wanted}")
    return out


def _benchmark(panel: pd.DataFrame, model_names: list[str], window_ids: list[int], horizon: int) -> pd.DataFrame:
    """Measure each model on the LONGEST and SHORTEST requested window.

    Both are reported so growth with training length is VISIBLE rather
    than assumed linear."""
    from projekat.model.registry import MODELS
    from projekat.model.runner import _representative_designs, run_one

    wins = anchored_windows()
    w_lo, w_hi = min(window_ids), max(window_ids)
    rows = []
    for model_name in model_names:
        design_id = _representative_designs(panel, MODELS[model_name].depends_on)["design_id"].iloc[0]
        design_df = panel[panel["design_id"] == design_id]
        measured = {}
        for label, wid in (("w01", w_lo), ("w11", w_hi)):
            window = wins[wid - 1]
            fit_train, _, _ = split_with_validation(design_df, window)
            t0 = time.monotonic()
            try:
                run_one(design_df, model_name=model_name, horizon=horizon,
                        window=window, window_id=wid)
                measured[label] = time.monotonic() - t0
            except Exception as exc:
                measured[label] = float("nan")
                print(f"  benchmark {model_name} {label} failed: {exc}", file=sys.stderr)
            measured[f"{label}_train_rows"] = len(fit_train)
        growth = (
            measured["w11"] / measured["w01"]
            if measured.get("w01") and measured["w01"] > 0 and np.isfinite(measured["w11"])
            else float("nan")
        )
        rows_ratio = (
            measured["w11_train_rows"] / measured["w01_train_rows"]
            if measured["w01_train_rows"] else float("nan")
        )
        rows.append(
            {
                "model": model_name,
                "w01_seconds": measured["w01"],
                "w11_seconds": measured["w11"],
                "w01_train_rows": measured["w01_train_rows"],
                "w11_train_rows": measured["w11_train_rows"],
                "time_growth": growth,
                "rows_growth": rows_ratio,
                # time growing markedly faster than row growth => superlinear
                "superlinear": bool(np.isfinite(growth) and np.isfinite(rows_ratio) and growth > 1.3 * rows_ratio),
            }
        )
    return pd.DataFrame(rows)


def _ensure_regime_boundary(panel: pd.DataFrame, window1) -> None:
    """Compute and persist the calm/turbulent boundary from window 1's
    training slice, unless a boundary computed from THIS panel already
    exists.

    The stored provenance (row count, symbols, date range) is what makes
    the staleness check possible: a threshold computed on a 61-row smoke
    panel is byte-identical in shape to one computed on the full 30-symbol
    panel, so without provenance it would be silently reused to label the
    entire study -- an error that raises nothing and only shows up as
    wrong regime labels in stratified_losses.csv."""
    from projekat.evaluation.stratify import (
        compute_and_persist_regime_boundary,
        regime_boundary_provenance,
    )

    train = panel[(panel["date"] >= window1.train_start) & (panel["date"] <= window1.train_end)]
    if train.empty:
        print("WARNING: window 1 has no training rows in this panel; "
              "regime boundary not computed, strata will be 'unknown'.", file=sys.stderr)
        return

    symbols_now = sorted(train["symbol"].unique().tolist())
    existing = regime_boundary_provenance()
    if existing and existing.get("symbols") == symbols_now and existing.get("n_rows") == len(train):
        print(f"regime boundary: reusing {existing['threshold']:.6g} "
              f"(from {existing['n_rows']:,} rows, {len(symbols_now)} symbols)")
        return

    if existing:
        print(f"regime boundary: RECOMPUTING -- stored one came from "
              f"{existing.get('n_rows')} rows / {len(existing.get('symbols') or [])} symbols, "
              f"this panel's window 1 has {len(train):,} rows / {len(symbols_now)} symbols",
              file=sys.stderr)

    threshold = compute_and_persist_regime_boundary(train)
    print(f"regime boundary: {threshold:.6g}  "
          f"(window 1 training: {len(train):,} rows, {len(symbols_now)} symbols)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--panel", default="data/processed/panel_mini.parquet", help="panel parquet")
    parser.add_argument("--symbols", default="all", help="comma-separated, or 'all'")
    parser.add_argument("--models", default="econometric", help=f"comma-separated, or a group: {sorted(MODEL_GROUPS)}")
    parser.add_argument("--grids", default="all", help="comma-separated, or 'all'")
    parser.add_argument("--horizons", default="all", help="comma-separated, or 'all'")
    parser.add_argument("--windows", default="all", help="e.g. '1-11' or '1,2,3', or 'all'")
    parser.add_argument("--run-dir", default=None, help="reuse an existing run directory instead of creating one")
    parser.add_argument("--label", default=None, help="suffix for a new run directory's name")
    parser.add_argument("--dry-run", action="store_true", help="enumerate and exit, with no side effects")
    parser.add_argument("--benchmark", action="store_true", help="with --dry-run: measure timing at w01 and w11")
    parser.add_argument("--resume", dest="resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--workers", type=int, default=4, help="(reserved; currently serial)")
    args = parser.parse_args()

    panel_path = Path(args.panel)
    if not panel_path.exists():
        raise SystemExit(f"panel does not exist: {panel_path}\nBuild it with: uv run python -m projekat.build_panel ...")
    panel = pd.read_parquet(panel_path)
    panel = _filter_panel(panel, symbols=args.symbols, grids=args.grids)
    if panel.empty:
        raise SystemExit("panel is empty after filtering")

    model_names = resolve_models(args.models)
    horizons = parse_int_list(args.horizons, HORIZONS)
    all_windows = anchored_windows()
    window_ids = parse_window_spec(args.windows, len(all_windows))

    combos = enumerate_combinations(
        panel, model_names=model_names, horizons=horizons,
        window_ids=window_ids, windows=all_windows,
    )

    symbols = sorted(panel["symbol"].unique())
    designs = sorted(panel["design_id"].unique())

    print(f"panel     : {panel_path}  ({len(panel):,} rows)")
    print(f"symbols   : {len(symbols)}  {symbols if len(symbols) <= 8 else symbols[:8] + ['...']}")
    print(f"designs   : {len(designs)}  {designs if len(designs) <= 6 else designs[:6] + ['...']}")
    print(f"models    : {len(model_names)}  {model_names}")
    print(f"horizons  : {horizons}")
    print(f"windows   : {window_ids[0]}..{window_ids[-1]}  ({len(window_ids)})")
    print()

    report = dedup_report(panel, model_names, len(horizons), len(window_ids))
    print("combinations per model (depends_on deduplication):")
    print(report.to_string(index=False))
    total = int(report["actual"].sum())
    naive_total = int(report["naive_product"].sum())
    print()
    if naive_total:
        print(f"TOTAL run_one calls : {total:,}   (naive product {naive_total:,}; "
              f"deduplication saves {naive_total - total:,} = {100 * (1 - total / naive_total):.0f}%)")
    print(f"tables to write     : {total * 2:,}  (test + validation per call)")
    assert total == len(combos), f"enumeration ({len(combos)}) != report ({total})"

    if args.dry_run:
        if args.benchmark:
            print("\nmeasuring timing at w01 and w11 (real data from the panel)...")
            bench = _benchmark(panel, model_names, window_ids, horizons[0])
            print(bench.to_string(index=False))
            per_call = bench["w11_seconds"].mean()
            if np.isfinite(per_call):
                est = per_call * total
                print(f"\nestimate (w11 per call x {total:,}): {est / 3600:.2f} h serial")
                if bench["superlinear"].any():
                    slow = bench.loc[bench["superlinear"], "model"].tolist()
                    print(f"WARNING: superlinear growth with training length: {slow}")
                    print("  -> estimate these per window, never by multiplying one figure")
                if est > 6 * 3600:
                    print("\nSCOPE IS LARGE (>6 h serial). Suggested reductions:")
                    print("  --windows 1-3        (3.7x fewer)")
                    print("  --horizons 1         (3x fewer)")
                    print("  --symbols <subset>   (linear)")
        else:
            print("\n(--benchmark for a measured time estimate)")
        print("\n--dry-run: nothing was written.")
        return 0

    # -- real run ---------------------------------------------------------
    if args.run_dir:
        run_dir = Path(args.run_dir)
        if not run_dir.exists():
            raise SystemExit(f"--run-dir does not exist: {run_dir}")
    else:
        run_id = new_run_id(args.label)
        run_dir = create_run_dir(run_id)
        write_manifest(
            run_dir,
            RunManifest(
                run_id=run_id,
                created_at=pd.Timestamp.utcnow().isoformat(),
                argv=sys.argv,
                git_sha=_git_sha(),
                symbols=symbols,
                models=model_names,
                grids=sorted({d.split("__")[0] for d in designs}),
                horizons=horizons,
                window_ids=window_ids,
                panel_path=str(panel_path.resolve()),
                panel_sha256=_sha256(panel_path),
                python=sys.version.split()[0],
                packages=_package_versions(),
                notes="econometric group; other groups write into the same directory",
            ),
        )
    print(f"\nrun-dir: {run_dir}\n")

    # The regime boundary must come from window 1's TRAINING data only, and
    # is then frozen for the whole study (Decision C6): letting it shift per
    # window would make a given calendar day's regime label depend on which
    # window produced the forecast being tagged. Computing it here, against
    # the panel actually being run, is what stops a boundary left over from
    # some earlier smoke panel being silently reused downstream.
    _ensure_regime_boundary(panel, all_windows[0])

    counts = run_sharded(panel, combos, run_dir, resume=args.resume)

    print()
    print(f"tables written : {counts['written']:,}")
    print(f"skipped        : {counts['skipped']:,}")
    print(f"empty          : {counts['empty']:,}")
    print(f"failed         : {counts['failed']:,}")
    index = store.read_index(run_dir)
    if not index.empty:
        print(f"index          : {len(index):,} rows, {index['n_rows'].sum():,} forecast rows")
    if counts["failed"]:
        print(f"\nfailures in {run_dir / store.FAILURES_FILENAME}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
