"""Entrypoint for generating the forecast tables (any model group).

    uv run python scripts/run_econometric.py --panel data/processed/panel.parquet --models fast --dry-run --benchmark
    uv run python scripts/run_econometric.py --panel data/processed/panel.parquet --models fast

One combination = model x design x seed = one walk-forward pass
(model/walkforward.py): a forecast every trading day from 2014 (warm-up)
through 2025, every horizon, the model re-estimated every 21 trading days
(PatchTST: yearly) on all data strictly before the refit date, corrections
from the model's own trailing out-of-sample errors.

`--dry-run` is side-effect free: no run directory, no file, no fit (except
under `--benchmark`, which times the FIRST and the LAST forecast block of
one combination per model -- smallest and largest training set -- and
extrapolates to the full pass).
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# The Windows console defaults to cp1252; output must never crash a
# multi-hour run over one character.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from projekat import config  # noqa: E402
from projekat.evaluation.anova import parse_design_id  # noqa: E402
from projekat.model.sharded import (  # noqa: E402
    MODEL_GROUPS,
    benchmark_model,
    dedup_report,
    enumerate_combinations,
    parse_int_list,
    resolve_models,
    run_sharded,
)
from projekat.results import store  # noqa: E402
from projekat.results.store import RunManifest, create_run_dir, new_run_id, write_manifest  # noqa: E402


def _git_sha() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5,
                             cwd=str(Path(__file__).resolve().parents[1]))
        sha = out.stdout.strip()
        return sha if out.returncode == 0 and len(sha) == 40 else None
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
    for name in ("numpy", "pandas", "statsmodels", "scipy", "pyarrow", "lightgbm", "xgboost", "torch"):
        try:
            out[name] = __import__(name).__version__
        except Exception:
            pass
    return out


def _scheme() -> dict:
    return {
        "sample_start": config.START_DATE,
        "first_forecast_date": config.FIRST_FORECAST_DATE,
        "eval_start": config.EVAL_START,
        "eval_end": config.EVAL_END,
        "refit_every_sessions": config.REFIT_EVERY_SESSIONS,
        "correction_window_sessions": config.CORRECTION_WINDOW_SESSIONS,
        "patchtst_seeds": list(config.PATCHTST_SEEDS),
        "regime_grid": config.REGIME_GRID,
    }


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
        out = out[out["design_id"].map(lambda d: parse_design_id(d)["grid"]).isin(wanted)]
        if out.empty:
            raise SystemExit(f"no design on grids {wanted}")
    return out


def _write_regime_labels(panel: pd.DataFrame, run_dir: Path) -> None:
    """One label per (symbol, date) from 5m RV_m against the median of the
    first training period -- written once per run, identical for every
    design (evaluation/stratify.py)."""
    import json

    from projekat.evaluation.stratify import compute_regime_labels

    labels, provenance = compute_regime_labels(panel)
    store.write_regime_labels(run_dir, labels)
    (run_dir / "regime_boundary.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(f"regime boundary: {provenance['threshold']:.6g} ({provenance['grid']} RV_m median over "
          f"{provenance['n_rows']:,} rows before {provenance['first_training_period_end']})")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--panel", default="data/processed/panel.parquet", help="panel parquet")
    parser.add_argument("--symbols", default="all", help="comma-separated, or 'all'")
    parser.add_argument("--models", default="fast", help=f"comma-separated, or a group: {sorted(MODEL_GROUPS)}")
    parser.add_argument("--grids", default="all", help="comma-separated, or 'all'")
    parser.add_argument("--horizons", default="all", help="comma-separated, or 'all'")
    parser.add_argument("--run-dir", default=None, help="reuse an existing run directory instead of creating one")
    parser.add_argument("--label", default=None, help="suffix for a new run directory's name")
    parser.add_argument("--dry-run", action="store_true", help="enumerate and exit, with no side effects")
    parser.add_argument("--benchmark", action="store_true", help="with --dry-run: time the first and last block per model")
    parser.add_argument("--resume", dest="resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--shard", default=None,
                        help="I/N: run only every N-th combination starting at I (0-based); see scripts/run_parallel.py")
    args = parser.parse_args()

    panel_path = Path(args.panel)
    if not panel_path.exists():
        raise SystemExit(f"panel does not exist: {panel_path}\nBuild it with: uv run python -m projekat.build_panel ...")
    panel = _filter_panel(pd.read_parquet(panel_path), symbols=args.symbols, grids=args.grids)
    if panel.empty:
        raise SystemExit("panel is empty after filtering")

    model_names = resolve_models(args.models)
    horizons = parse_int_list(args.horizons, config.HORIZONS)
    combos = enumerate_combinations(panel, model_names=model_names)
    symbols = sorted(panel["symbol"].unique())
    designs = sorted(panel["design_id"].unique())

    print(f"panel     : {panel_path}  ({len(panel):,} rows)")
    print(f"symbols   : {len(symbols)}  {symbols if len(symbols) <= 8 else symbols[:8] + ['...']}")
    print(f"designs   : {len(designs)}")
    print(f"models    : {len(model_names)}  {model_names}")
    print(f"horizons  : {horizons}")
    print(f"scheme    : forecasts {config.FIRST_FORECAST_DATE}..{config.EVAL_END} (warm-up until "
          f"{config.EVAL_START}), refit every {config.REFIT_EVERY_SESSIONS} sessions, "
          f"correction window {config.CORRECTION_WINDOW_SESSIONS} sessions\n")

    report = dedup_report(panel, model_names, n_horizons=len(horizons))
    print("combinations per model (depends_on deduplication):")
    print(report.to_string(index=False))
    total = int(report["combinations"].sum())
    print(f"\nTOTAL combinations (tables): {total:,}   model estimations: {int(report['fits'].sum()):,}")
    assert total == len(combos), f"enumeration ({len(combos)}) != report ({total})"

    shard_i, shard_n = 0, 1
    if args.shard:
        shard_i, shard_n = (int(x) for x in args.shard.split("/"))
        if not 0 <= shard_i < shard_n:
            raise SystemExit(f"--shard {args.shard}: need 0 <= I < N")
        combos = combos[shard_i::shard_n]
        print(f"shard {shard_i}/{shard_n}: {len(combos):,} of {total:,} combinations")

    if args.dry_run:
        if args.benchmark:
            print("\ntiming the first and last forecast block per model (real data from the panel)...")
            rows = []
            for name in model_names:
                try:
                    rows.append(benchmark_model(panel, name, horizons=horizons))
                except Exception as exc:
                    print(f"  benchmark {name} failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            bench = pd.DataFrame(rows)
            if not bench.empty:
                n_combos = report.set_index("model")["combinations"]
                bench["combinations"] = bench["model"].map(n_combos)
                bench["est_total_h"] = bench["est_combination_s"] * bench["combinations"] / 3600
                print(bench.to_string(index=False))
                print(f"\nestimate: {bench['est_total_h'].sum():.1f} h serial "
                      f"(longest single combination {bench['est_combination_s'].max() / 3600:.2f} h)")
        else:
            print("\n(--benchmark for a measured time estimate)")
        print("\n--dry-run: nothing was written.")
        return 0

    # -- real run ---------------------------------------------------------
    manifest = lambda run_id, notes: RunManifest(  # noqa: E731
        run_id=run_id, created_at=pd.Timestamp.utcnow().isoformat(), argv=sys.argv, git_sha=_git_sha(),
        symbols=symbols, models=model_names, grids=sorted({parse_design_id(d)["grid"] for d in designs}),
        horizons=horizons, scheme=_scheme(), panel_path=str(panel_path.resolve()),
        panel_sha256=_sha256(panel_path), python=sys.version.split()[0], packages=_package_versions(), notes=notes,
    )
    if args.run_dir and (Path(args.run_dir).exists() or shard_n > 1):
        run_dir = Path(args.run_dir)
        (run_dir / store.TABLES_DIRNAME).mkdir(parents=True, exist_ok=True)
        if shard_i == 0 and store.read_manifest(run_dir) is None:
            write_manifest(run_dir, manifest(run_dir.name, f"written by shard 0 of {shard_n}"))
    elif args.run_dir:
        raise SystemExit(f"--run-dir does not exist: {args.run_dir}")
    else:
        run_id = new_run_id(args.label)
        run_dir = create_run_dir(run_id)
        write_manifest(run_dir, manifest(run_id, "other model groups write into the same directory"))
    print(f"\nrun-dir: {run_dir}\n")

    # one writer for the run-level artefacts: shard 0 (or the only process)
    if shard_i == 0 and store.read_regime_labels(run_dir) is None:
        _write_regime_labels(panel, run_dir)
    if shard_i == 0 and any(n.startswith("ttm_") or n == "log_har_ttm" for n in model_names):
        # PRECONDITION for every ttm_* number, written either way
        from projekat.model.foundation.contamination import check_and_write

        report_c = check_and_write(config.EVAL_START, config.EVAL_END, out_dir=run_dir)
        print(f"TTM contamination check: {report_c.note}")

    counts = run_sharded(panel, combos, run_dir, horizons=horizons, resume=args.resume, shared_run_dir=shard_n > 1)
    print(f"\ntables written : {counts['written']:,}\nskipped        : {counts['skipped']:,}\n"
          f"empty          : {counts['empty']:,}\nfailed         : {counts['failed']:,}")
    if counts["failed"]:
        print(f"\nfailures in {run_dir / store.FAILURES_FILENAME}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
