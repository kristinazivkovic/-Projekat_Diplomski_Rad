"""WI-12 diagnostics over a panel: H5 placebo (100 permutations within
symbol, 5m, bpv, h = 1, har_j and char), re-estimation frequency, and
(optionally) the signature plot.

    uv run python scripts/run_diagnostics.py --panel data/processed/panel.parquet --out results/run_main/report --workers 8

They re-run whole walk-forwards, so they are separate from run_report.py.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from projekat.evaluation.anova import parse_design_id  # noqa: E402
from projekat.results.report import build_diagnostic_reports  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--panel", default="data/processed/panel.parquet")
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=1, help="processes for the placebo permutations")
    parser.add_argument("--n-permutations", type=int, default=None, help="default: evaluation/diagnostics.N_PERMUTATIONS")
    parser.add_argument("--parts", default="h5_placebo,reestimation_frequency",
                        help="comma-separated: h5_placebo, reestimation_frequency")
    parser.add_argument("--exclude-symbols", default=None,
                        help="comma-separated symbols to leave out (same meaning as in run_report.py)")
    args = parser.parse_args()

    panel = pd.read_parquet(args.panel)
    parsed = {d: parse_design_id(d) for d in panel["design_id"].unique()}
    keep = [d for d, f in parsed.items() if f["grid"] == "5m" and f["estimator"] == "bpv"]
    panel = panel[panel["design_id"].isin(keep)]
    if args.exclude_symbols:
        excluded = [s.strip() for s in args.exclude_symbols.split(",")]
        panel = panel[~panel["symbol"].isin(excluded)]
        print(f"excluded symbols: {excluded}")
    t0 = time.monotonic()
    written = build_diagnostic_reports(
        panel, out_dir=Path(args.out), workers=args.workers, n_permutations=args.n_permutations,
        parts=tuple(p.strip() for p in args.parts.split(",") if p.strip()),
    )
    print(f"wrote {len(written)} artefact(s) in {time.monotonic() - t0:.0f}s:")
    for name, path in written.items():
        print(f"  {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
