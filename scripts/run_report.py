"""Entrypoint for the evaluation layer: forecast tables -> report CSVs.

    uv run python scripts/run_report.py --run-dir results/run_main --panel data/processed/panel.parquet

Loads the run's tables ONE HORIZON AT A TIME (pyarrow filters: only that
horizon's rows, only evaluation rows -- the 2014 warm-up is never
evaluated -- and only the columns the report reads), builds every level
1-4 result per horizon and concatenates the CSVs. Every result is computed
per horizon anyway (the ANOVA is fitted separately per horizon by design).
The one exception is the DM family: Benjamini-Hochberg runs over EVERY
reported comparison at once, so it is re-applied to the concatenated table.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from projekat.evaluation.dm_test import benjamini_hochberg_dm  # noqa: E402
from projekat.results import store  # noqa: E402
from projekat.results.report import REPORT_COLUMNS, build_all_reports, build_level4_reports  # noqa: E402


def _read_csv_or_empty(path: Path) -> pd.DataFrame:
    """An empty result (e.g. no DM-significant ranking reversal anywhere) is
    written as a CSV with no columns at all; that is a legitimate "nothing
    found", not an error."""
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _merge_horizons(parts: dict[int, Path], out_dir: Path) -> list[Path]:
    """Concatenate each per-horizon CSV; move per-horizon parquets over
    with a _h<h> suffix."""
    written = []
    names = sorted({p.name for d in parts.values() for p in d.iterdir() if p.is_file()})
    for name in names:
        files = [(h, d / name) for h, d in sorted(parts.items()) if (d / name).exists()]
        if name.endswith(".csv"):
            frames = [f for f in (_read_csv_or_empty(f) for _, f in files) if not f.empty]
            merged = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
            if name == "dm_vs_log_har.csv" and not merged.empty:
                merged = benjamini_hochberg_dm(
                    merged.drop(columns=["dm_family", "bh_significant", "bh_threshold"], errors="ignore"),
                    family="model_vs_log_har",
                )
            target = out_dir / name
            merged.to_csv(target, index=False)
            written.append(target)
        else:
            for h, f in files:
                target = out_dir / f"{Path(name).stem}_h{h}{Path(name).suffix}"
                shutil.move(str(f), target)
                written.append(target)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--panel", default="data/processed/panel.parquet", help="panel parquet, for level 4")
    parser.add_argument("--out", default=None, help="output directory (default: <run-dir>/report)")
    parser.add_argument("--models", default=None, help="comma-separated subset (default: every model in the run)")
    parser.add_argument("--exclude-symbols", default=None,
                        help="comma-separated symbols to leave out of the evaluation (e.g. a symbol excluded "
                             "during QA after the forecasts were produced); recorded in the report directory")
    parser.add_argument("--bootstrap", action="store_true", help="also run the block-bootstrap variance shares")
    parser.add_argument("--n-boot", type=int, default=200)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise SystemExit(f"--run-dir does not exist: {run_dir}")
    out_dir = Path(args.out) if args.out else run_dir / "report"
    out_dir.mkdir(parents=True, exist_ok=True)
    models = [m.strip() for m in args.models.split(",")] if args.models else None
    excluded = [s.strip() for s in args.exclude_symbols.split(",")] if args.exclude_symbols else []
    if excluded:
        (out_dir / "excluded_symbols.json").write_text(
            json.dumps({"excluded_symbols": excluded,
                        "note": "left out of the evaluation after the forecasts were produced"}, indent=2),
            encoding="utf-8")

    failures = store.read_failures(run_dir)
    if not failures.empty:
        print(f"WARNING: {len(failures)} combination(s) FAILED and are absent from every table "
              f"(see {run_dir / store.FAILURES_FILENAME}).", file=sys.stderr)

    t0 = time.monotonic()
    # shards never write the index; tables are atomic, so rebuilding is exact
    index = store.rebuild_index(run_dir)
    if index.empty:
        raise SystemExit(f"no tables in {run_dir / store.TABLES_DIRNAME}")
    manifest = store.read_manifest(run_dir)
    horizons = sorted(manifest.horizons) if manifest and manifest.horizons else [1, 5, 22]
    print(f"index   : {len(index):,} table(s), {index['model'].nunique()} model(s), horizons {horizons}")
    regime_labels = store.read_regime_labels(run_dir)
    if regime_labels is None:
        print("WARNING: no regime_labels.parquet in the run dir; regime will be 'unknown'.", file=sys.stderr)

    parts: dict[int, Path] = {}
    for horizon in horizons:
        th = time.monotonic()
        rows = store.load_forecasts(run_dir, horizon=horizon, models=models, columns=REPORT_COLUMNS)
        if rows.empty:
            continue
        if excluded:
            before = len(rows)
            rows = rows[~rows["symbol"].isin(excluded)]
            print(f"h={horizon:<3d}  : excluded {excluded} ({before - len(rows):,} rows)", flush=True)
        print(f"h={horizon:<3d}  : {len(rows):,} evaluation rows ...", flush=True)
        part = out_dir / f"_part_h{horizon}"
        shutil.rmtree(part, ignore_errors=True)
        build_all_reports(rows, regime_labels=regime_labels, out_dir=part,
                          bootstrap=args.bootstrap, n_boot=args.n_boot)
        parts[horizon] = part
        del rows
        print(f"h={horizon:<3d}  : done in {time.monotonic() - th:.0f}s", flush=True)

    written = _merge_horizons(parts, out_dir)
    for part in parts.values():
        shutil.rmtree(part, ignore_errors=True)

    panel_path = Path(args.panel)
    if panel_path.exists():
        panel = pd.read_parquet(panel_path, columns=["design_id", "symbol", "date", "jump_flag", "J_d", "RV_d"])
        written += list(build_level4_reports(panel, out_dir).values())
        del panel
    else:
        print(f"WARNING: panel {panel_path} not found; level-4 jump reports are skipped.", file=sys.stderr)

    for name in ("ttm_contamination_check.json", "regime_boundary.json"):
        if (run_dir / name).exists():
            shutil.copy(run_dir / name, out_dir / name)
            written.append(out_dir / name)

    print(f"\nwrote {len(written)} artefact(s) to {out_dir} in {time.monotonic() - t0:.0f}s:")
    for path in sorted(written):
        print(f"  {Path(path).name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
