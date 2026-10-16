"""Combination enumeration and per-table execution for the full grid.

A combination is (model x design x seed): one walk-forward pass
(model/walkforward.py) producing daily forecasts for every horizon, written
as one table the moment it finishes (results/store.py) and then released --
peak memory is a single table, and `--resume` skips tables already on disk.

Enumeration is separated from execution (`enumerate_combinations` touches
no disk and fits nothing) so `--dry-run` can report the exact combination
count and a time estimate with no side effects.

depends_on DEDUPLICATION: the per-model design count comes from
`_representative_designs` -- a model with depends_on == {"grid"} gets 3
designs, not 36. A correctness condition, not an optimization: running the
same model 12 times per grid on byte-identical inputs would produce
duplicated, mutually dependent rows.
"""

from __future__ import annotations

import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from projekat.config import EVAL_END, FIRST_FORECAST_DATE, HORIZONS, REFIT_EVERY_SESSIONS
from projekat.model import baselines, econometric, foundation, sequence, trees  # noqa: F401  (registration)
from projekat.model.registry import MODELS
from projekat.model.runner import _representative_designs, run_one, seeds_for
from projekat.model.walkforward import refit_dates
from projekat.results import store
from projekat.results.store import TableKey

ECONOMETRIC_MODELS = ("naive", "har", "log_har", "shar", "harq", "har_iv", "char", "har_j")

MODEL_GROUPS = {
    "econometric": ECONOMETRIC_MODELS,
    "trees": ("lightgbm", "xgboost"),
    "sequence": ("patchtst",),
    "foundation": ("ttm_rv", "ttm_c", "ttm_c_plus_j", "log_har_ttm"),
    "longmemory": ("arfima", "mem"),
}
# Every model except the sequence group, so results for everything else
# exist first; PatchTST is added later into the SAME run dir.
MODEL_GROUPS["fast"] = tuple(
    m for group, models in MODEL_GROUPS.items() if group != "sequence" for m in models
)


@dataclass(frozen=True)
class Combination:
    """One walk-forward pass -> one table (all horizons)."""

    model: str
    design_id: str
    seed: int | None = None

    def key(self) -> TableKey:
        return TableKey(model=self.model, design_id=self.design_id, seed=self.seed)


def resolve_models(spec: str | None) -> list[str]:
    """'econometric' -> the whole group; 'a,b,c' -> those models; None ->
    every registered model. An unknown name is an error, not a silent skip."""
    if spec is None or spec == "all":
        return list(MODELS.keys())
    if spec in MODEL_GROUPS:
        return list(MODEL_GROUPS[spec])
    names = [s.strip() for s in spec.split(",") if s.strip()]
    unknown = [n for n in names if n not in MODELS]
    if unknown:
        raise ValueError(f"unknown models: {unknown}; registered: {sorted(MODELS)}")
    return names


def parse_int_list(spec: str | None, default: tuple[int, ...]) -> list[int]:
    if spec is None or spec == "all":
        return list(default)
    return [int(s.strip()) for s in spec.split(",") if s.strip()]


def enumerate_combinations(panel_df: pd.DataFrame, *, model_names: list[str]) -> list[Combination]:
    """Every combination, with NO side effects -- touches no disk, fits nothing."""
    out: list[Combination] = []
    for model_name in model_names:
        model = MODELS[model_name]
        for design_id in _representative_designs(panel_df, model.depends_on)["design_id"]:
            for seed in seeds_for(model_name):
                out.append(Combination(model=model_name, design_id=design_id, seed=seed))
    return out


def n_refits(model_name: str, *, first: str = FIRST_FORECAST_DATE, last: str = EVAL_END,
             every: int = REFIT_EVERY_SESSIONS) -> int:
    rule = MODELS[model_name].refit
    return 1 if rule == "never" else len(refit_dates(rule, first=first, last=last, every=every))


def dedup_report(panel_df: pd.DataFrame, model_names: list[str], *, n_horizons: int = len(HORIZONS)) -> pd.DataFrame:
    """Per model: combinations with and without depends_on deduplication,
    and the number of model estimations that implies (refits x horizons x
    combinations; zero-shot models estimate nothing)."""
    n_all_designs = panel_df["design_id"].nunique()
    rows = []
    for model_name in model_names:
        model = MODELS[model_name]
        n_designs = len(_representative_designs(panel_df, model.depends_on))
        n_seeds = len(seeds_for(model_name))
        actual = n_designs * n_seeds
        refits = n_refits(model_name)
        trained = not model_name.startswith("ttm_")
        rows.append(
            {
                "model": model_name,
                "depends_on": ",".join(sorted(model.depends_on)) or "(none)",
                "designs": n_designs,
                "seeds": n_seeds,
                "combinations": actual,
                "naive_product": n_all_designs * n_seeds,
                "deduplicated": actual < n_all_designs * n_seeds,
                "refit_rule": model.refit,
                "refits_per_combination": refits,
                "fits": actual * refits * n_horizons if trained and model.refit != "never" else 0,
            }
        )
    return pd.DataFrame(rows)


# -- benchmark ------------------------------------------------------------


def benchmark_model(panel_df: pd.DataFrame, model_name: str, *, horizons=HORIZONS) -> dict:
    """Time the FIRST and the LAST forecast block of one representative
    combination (smallest and largest training set). The expanding window
    grows linearly, so the per-block average of the two is the estimate;
    the full-combination time is n_blocks x that average."""
    model = MODELS[model_name]
    design_id = _representative_designs(panel_df, model.depends_on)["design_id"].iloc[0]
    design_df = panel_df[panel_df["design_id"] == design_id]
    seed = seeds_for(model_name)[0]
    timings: list[dict] = []
    t0 = time.perf_counter()
    run_one(design_df, model_name=model_name, horizons=horizons, seed=seed, blocks=[0, -1], timings=timings)
    wall = time.perf_counter() - t0
    blocks = len(refit_dates(model.refit, first=FIRST_FORECAST_DATE, last=EVAL_END))
    per_block = np.mean([t["fit_seconds"] + t["predict_seconds"] for t in timings]) if timings else float("nan")
    if model.refit == "never" and timings:
        # one fit, then predict-only blocks
        per_block = np.mean([t["predict_seconds"] for t in timings])
    return {
        "model": model_name,
        "design_id": design_id,
        "blocks": blocks,
        "first_block_s": timings[0]["fit_seconds"] + timings[0]["predict_seconds"] if timings else float("nan"),
        "last_block_s": timings[-1]["fit_seconds"] + timings[-1]["predict_seconds"] if timings else float("nan"),
        "first_train_rows": timings[0]["n_train"] if timings else 0,
        "last_train_rows": timings[-1]["n_train"] if timings else 0,
        "est_combination_s": per_block * blocks,
        "measured_wall_s": wall,
    }


# -- execution ------------------------------------------------------------


class _Interrupt:
    """SIGINT handler: record the interrupt rather than tearing the job
    down mid-write; the loop checks the flag BETWEEN combinations."""

    def __init__(self) -> None:
        self.requested = False
        self._previous = None

    def __enter__(self):
        def handler(signum, frame):  # noqa: ARG001
            self.requested = True
            print("\n[SIGINT] finishing the current table, then exiting cleanly...", flush=True)
        self._previous = signal.signal(signal.SIGINT, handler)
        return self

    def __exit__(self, *exc):
        if self._previous is not None:
            signal.signal(signal.SIGINT, self._previous)
        return False


def _fmt_eta(seconds: float) -> str:
    seconds = max(int(seconds), 0)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}h{m:02d}m" if h else (f"{m:d}m{s:02d}s" if m else f"{s:d}s")


def run_sharded(
    panel_df: pd.DataFrame,
    combinations: list[Combination],
    run_dir: Path,
    *,
    horizons=HORIZONS,
    resume: bool = True,
    progress: bool = True,
    shared_run_dir: bool = False,
) -> dict[str, int]:
    """Run each combination and write its table immediately. Returns
    counters; the caller derives its exit code from `failed`.

    `shared_run_dir=True` (several shard processes writing into one run
    dir): the index is never written and resume reads the table filenames
    instead; scripts/run_report.py rebuilds the index before loading."""
    if not resume:
        done = set()
    elif shared_run_dir:
        done = store.completed_keys_from_files(run_dir)
    else:
        done = store.completed_keys(run_dir)
    counts = {"written": 0, "skipped": 0, "failed": 0, "empty": 0}
    index_rows: list[dict] = []
    started = time.monotonic()
    total = len(combinations)

    with _Interrupt() as interrupt:
        for i, combo in enumerate(combinations, start=1):
            key = combo.key()
            if resume and store.key_tuple(key) in done:
                counts["skipped"] += 1
                continue

            design_df = panel_df[panel_df["design_id"] == combo.design_id]
            t0 = time.monotonic()
            try:
                frame = run_one(design_df, model_name=combo.model, horizons=horizons, seed=combo.seed)
            except BaseException as exc:  # noqa: BLE001 - every failure is recorded, none silently
                if isinstance(exc, KeyboardInterrupt):
                    raise
                store.append_failure(run_dir, key, exc)
                counts["failed"] += 1
                print(f"[{i}/{total}] FAIL {combo.model} {combo.design_id} seed={combo.seed}: "
                      f"{type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
                continue

            if frame is None or frame.empty:
                counts["empty"] += 1
            else:
                path = store.write_table(run_dir, key, frame)
                index_rows.append(store.index_row(key, path, len(frame)))
                counts["written"] += 1
                if len(index_rows) >= 20 and not shared_run_dir:
                    store.append_index(run_dir, index_rows)
                    index_rows = []

            if progress:
                elapsed = time.monotonic() - started
                rate = i / elapsed if elapsed > 0 else 0
                remaining = (total - i) / rate if rate > 0 else 0
                print(
                    f"[{i}/{total}] {combo.model} {combo.design_id}"
                    f"{'' if combo.seed is None else f' seed={combo.seed}'} "
                    f"| rows {0 if frame is None else len(frame)} | {time.monotonic() - t0:.0f}s "
                    f"| elapsed {_fmt_eta(elapsed)} | eta {_fmt_eta(remaining)}",
                    flush=True,
                )

            if interrupt.requested:
                print(f"[SIGINT] stopped at {i}/{total}; --resume continues from here.", flush=True)
                break

    if index_rows and not shared_run_dir:
        store.append_index(run_dir, index_rows)
    return counts
