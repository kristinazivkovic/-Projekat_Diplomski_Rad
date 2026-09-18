"""Combination enumeration and per-table execution.

RELATION TO runner.py::run_all
==============================
`run_all` is left untouched and remains what the tests use for a small
grid: it returns two concatenated frames in memory. This module is its
large-grid counterpart -- the same loop and the same `run_one`, but each
combination is written out as its own table immediately and released.
On the full grid `run_all` would hold ~106M rows (~21 GB with the
validation frame) before the first concat; here peak memory is the size
of a SINGLE table.

Enumeration is deliberately separated from execution
(`enumerate_combinations` touches no disk and fits nothing) so that
`--dry-run` can report the exact combination count and a time estimate
with no side effects whatsoever.

depends_on DEDUPLICATION
========================
The per-model combination count comes from `_representative_designs`,
exactly as in `run_all` -- a model with `depends_on == {"grid"}` gets 3
designs, not 36. That is not an optimization but a correctness condition:
fitting the same model 12 times per grid on byte-identical inputs would
feed duplicated, mutually dependent rows into the ANOVA and corrupt the
variance shares.
"""

from __future__ import annotations

import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from projekat.model import baselines, econometric, foundation, sequence, trees  # noqa: F401  (registration)
from projekat.model.registry import MODELS
from projekat.model.runner import HORIZONS, STOCHASTIC_SEEDS, _representative_designs, run_one
from projekat.model.windows import Window, anchored_windows
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


@dataclass(frozen=True)
class Combination:
    """One `run_one` call. Produces TWO tables (test and validation),
    because `run_one` returns both frames from a single fit -- splitting
    them apart would mean fitting the same model twice."""

    model: str
    design_id: str
    horizon: int
    window_id: int
    window: Window
    seed: int | None = None

    def keys(self) -> tuple[TableKey, TableKey]:
        base = dict(
            model=self.model, design_id=self.design_id, horizon=self.horizon,
            window_id=self.window_id, seed=self.seed,
        )
        return TableKey(split="test", **base), TableKey(split="validation", **base)


def resolve_models(spec: str | None) -> list[str]:
    """'econometric' -> the whole group; 'a,b,c' -> those models; None ->
    every registered model. An unknown name is an error, not a silent
    skip."""
    if spec is None:
        return list(MODELS.keys())
    if spec in MODEL_GROUPS:
        return list(MODEL_GROUPS[spec])
    names = [s.strip() for s in spec.split(",") if s.strip()]
    unknown = [n for n in names if n not in MODELS]
    if unknown:
        raise ValueError(f"unknown models: {unknown}; registered: {sorted(MODELS)}")
    return names


def parse_window_spec(spec: str | None, n_available: int) -> list[int]:
    """'1-11' | '1,2,3' | 'all' -> 1-based window indices."""
    if spec is None or spec == "all":
        return list(range(1, n_available + 1))
    out: list[int] = []
    for piece in spec.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if "-" in piece:
            lo, hi = piece.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(piece))
    bad = [w for w in out if not 1 <= w <= n_available]
    if bad:
        raise ValueError(f"windows out of range 1..{n_available}: {bad}")
    return sorted(set(out))


def parse_int_list(spec: str | None, default: tuple[int, ...]) -> list[int]:
    if spec is None or spec == "all":
        return list(default)
    return [int(s.strip()) for s in spec.split(",") if s.strip()]


def enumerate_combinations(
    panel_df: pd.DataFrame,
    *,
    model_names: list[str],
    horizons: list[int],
    window_ids: list[int],
    windows: list[Window] | None = None,
    seeds: tuple[int, ...] = STOCHASTIC_SEEDS,
) -> list[Combination]:
    """Every combination, with NO side effects -- touches no disk, fits
    nothing. `--dry-run` depends on this."""
    wins = windows or anchored_windows()
    out: list[Combination] = []
    for model_name in model_names:
        model = MODELS[model_name]
        rep = _representative_designs(panel_df, model.depends_on)
        model_seeds = seeds if model.stochastic else (None,)
        for design_id in rep["design_id"]:
            for horizon in horizons:
                for window_id in window_ids:
                    for seed in model_seeds:
                        out.append(
                            Combination(
                                model=model_name, design_id=design_id, horizon=horizon,
                                window_id=window_id, window=wins[window_id - 1], seed=seed,
                            )
                        )
    return out


def dedup_report(panel_df: pd.DataFrame, model_names: list[str], n_h: int, n_w: int) -> pd.DataFrame:
    """Per model: how many combinations there would be without
    `depends_on` deduplication, and how many there actually are.
    `--dry-run` prints this so it is visible whether deduplication is in
    effect and how much it saves."""
    n_all_designs = panel_df["design_id"].nunique()
    rows = []
    for model_name in model_names:
        model = MODELS[model_name]
        n_designs = len(_representative_designs(panel_df, model.depends_on))
        n_seeds = len(STOCHASTIC_SEEDS) if model.stochastic else 1
        actual = n_designs * n_h * n_w * n_seeds
        naive = n_all_designs * n_h * n_w * n_seeds
        rows.append(
            {
                "model": model_name,
                "depends_on": ",".join(sorted(model.depends_on)) or "(none)",
                "designs": n_designs,
                "designs_available": n_all_designs,
                "seeds": n_seeds,
                "naive_product": naive,
                "actual": actual,
                "deduplicated": actual < naive,
                "saved": naive - actual,
            }
        )
    return pd.DataFrame(rows)


# -- execution ------------------------------------------------------------


class _Interrupt:
    """SIGINT handler: record the interrupt rather than tearing the job
    down mid-write. The loop checks the flag BETWEEN combinations, so the
    in-flight table is always either finished or discarded whole
    (store.write_table cleans up its .tmp)."""

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
    resume: bool = True,
    progress: bool = True,
) -> dict[str, int]:
    """Fit each combination and write its two tables immediately.

    Peak memory is one table, not the whole grid. Returns counters; the
    caller derives its exit code from `failed`."""
    done = store.completed_keys(run_dir) if resume else set()
    counts = {"written": 0, "skipped": 0, "failed": 0, "empty": 0}
    index_rows: list[dict] = []
    started = time.monotonic()
    total = len(combinations)

    with _Interrupt() as interrupt:
        for i, combo in enumerate(combinations, start=1):
            test_key, val_key = combo.keys()

            if resume and store.key_tuple(test_key) in done and store.key_tuple(val_key) in done:
                counts["skipped"] += 1
                continue

            design_df = panel_df[panel_df["design_id"] == combo.design_id]
            try:
                test_rows, val_rows = run_one(
                    design_df,
                    model_name=combo.model,
                    horizon=combo.horizon,
                    window=combo.window,
                    seed=combo.seed,
                    window_id=combo.window_id,
                )
            except BaseException as exc:  # noqa: BLE001 - every failure is recorded, none silently
                if isinstance(exc, KeyboardInterrupt):
                    raise
                store.append_failure(run_dir, test_key, exc)
                counts["failed"] += 1
                print(f"[{i}/{total}] FAIL {combo.model} {combo.design_id} "
                      f"h={combo.horizon} w={combo.window_id}: {type(exc).__name__}: {exc}",
                      file=sys.stderr, flush=True)
                continue

            for key, frame in ((test_key, test_rows), (val_key, val_rows)):
                if frame is None or frame.empty:
                    counts["empty"] += 1
                    continue
                path = store.write_table(run_dir, key, frame)
                index_rows.append(store.index_row(key, path, len(frame)))
                counts["written"] += 1

            # the index is appended in batches for speed, but often enough
            # that an interrupt loses little work
            if len(index_rows) >= 50:
                store.append_index(run_dir, index_rows)
                index_rows = []

            if progress:
                elapsed = time.monotonic() - started
                rate = i / elapsed if elapsed > 0 else 0
                remaining = (total - i) / rate if rate > 0 else 0
                print(
                    f"[{i}/{total}] {combo.model} {combo.design_id} "
                    f"h={combo.horizon} w={combo.window_id}"
                    f"{'' if combo.seed is None else f' seed={combo.seed}'} "
                    f"| rows {len(test_rows)}+{len(val_rows)} "
                    f"| elapsed {_fmt_eta(elapsed)} | eta {_fmt_eta(remaining)}",
                    flush=True,
                )

            if interrupt.requested:
                print(f"[SIGINT] stopped at {i}/{total}; --resume continues from here.", flush=True)
                break

    if index_rows:
        store.append_index(run_dir, index_rows)
    return counts
