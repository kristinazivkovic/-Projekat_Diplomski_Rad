"""Persistence layer for the raw forecast tables.

One run directory; every model group writes into it; one table per
combination (model x design x seed), written the moment its walk-forward
finishes, so peak memory is one table and an interrupted run resumes.

    <run-dir>/
        manifest.json            run provenance (see RunManifest)
        index.parquet            one row per written table
        failures.jsonl           one row per failed combination
        regime_labels.parquet    one regime label per (symbol, date)
        tables/<name>.parquet    forecast rows: every horizon, every day
                                 (warm-up year included, flagged is_warmup)

There is no validation split any more: the back-transform correction and
the Mincer-Zarnowitz coefficients come from each model's own trailing
out-of-sample errors (model/backtransform.py) and are already inside the
table.

FILENAME GRAMMAR
================
    model=<m>__design=<d>__seed=<seed>.parquet

Fields are `key=value` separated by `__`, in a fixed order -- `parse_name`
is the exact inverse of `build_name` and the ONLY place a table name is
taken apart. `design` itself contains `__` (e.g. `5m__bpv__bns`), so
parsing splits on `key=` boundaries, never on a plain `split("__")`. A
missing seed is the literal `none`.

ATOMICITY
=========
Each table is written to `<name>.tmp` and then `os.replace()`d onto its
final name (atomic on Windows and POSIX), so a reader never sees a
half-written parquet.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from projekat.config import RESULTS_DIR

TABLES_DIRNAME = "tables"
INDEX_FILENAME = "index.parquet"
MANIFEST_FILENAME = "manifest.json"
FAILURES_FILENAME = "failures.jsonl"
REGIME_LABELS_FILENAME = "regime_labels.parquet"

_NONE = "none"
_VALUE_RE = re.compile(r"^[A-Za-z0-9._+-]+$")


@dataclass(frozen=True)
class TableKey:
    """One combination = one table. `seed` is None for deterministic models."""

    model: str
    design_id: str
    seed: int | None = None


def build_name(key: TableKey) -> str:
    """TableKey -> filename. Exact inverse of parse_name."""
    parts = {
        "model": key.model,
        "design": key.design_id,
        "seed": _NONE if key.seed is None else str(key.seed),
    }
    for name, value in parts.items():
        segments = value.split("__") if name == "design" else [value]
        for segment in segments:
            if not _VALUE_RE.match(segment):
                raise ValueError(f"field value {name}={value!r} is not filename-safe")
    return "__".join(f"{k}={v}" for k, v in parts.items()) + ".parquet"


def parse_name(filename: str) -> TableKey:
    """Filename -> TableKey. Exact inverse of build_name."""
    stem = filename[:-len(".parquet")] if filename.endswith(".parquet") else filename
    pattern = r"^model=(?P<model>[^=]+?)__design=(?P<design>.+?)__seed=(?P<seed>none|\d+)$"
    match = re.match(pattern, stem)
    if not match:
        raise ValueError(f"table name does not match the grammar: {filename!r}")
    seed_raw = match.group("seed")
    return TableKey(
        model=match.group("model"),
        design_id=match.group("design"),
        seed=None if seed_raw == _NONE else int(seed_raw),
    )


# -- manifest -------------------------------------------------------------


@dataclass(frozen=True)
class RunManifest:
    run_id: str
    created_at: str
    argv: list[str] = field(default_factory=list)
    git_sha: str | None = None
    symbols: list[str] = field(default_factory=list)
    models: list[str] = field(default_factory=list)
    grids: list[str] = field(default_factory=list)
    horizons: list[int] = field(default_factory=list)
    scheme: dict = field(default_factory=dict)
    panel_path: str | None = None
    panel_sha256: str | None = None
    python: str = ""
    packages: dict[str, str] = field(default_factory=dict)
    notes: str = ""


def new_run_id(label: str | None = None) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"run_{stamp}" + (f"_{label}" if label else "")


def create_run_dir(run_id: str, *, base: Path | None = None) -> Path:
    root = (Path(base) if base is not None else RESULTS_DIR) / run_id
    (root / TABLES_DIRNAME).mkdir(parents=True, exist_ok=True)
    return root


def write_manifest(run_dir: Path, manifest: RunManifest) -> Path:
    path = run_dir / MANIFEST_FILENAME
    path.write_text(json.dumps(asdict(manifest), indent=2, default=str), encoding="utf-8")
    return path


def read_manifest(run_dir: Path) -> RunManifest | None:
    path = run_dir / MANIFEST_FILENAME
    if not path.exists():
        return None
    return RunManifest(**json.loads(path.read_text(encoding="utf-8")))


# -- writing tables -------------------------------------------------------


def table_path(run_dir: Path, key: TableKey) -> Path:
    return run_dir / TABLES_DIRNAME / build_name(key)


def _atomic_parquet(frame: pd.DataFrame, final: Path) -> Path:
    final.parent.mkdir(parents=True, exist_ok=True)
    tmp = final.with_suffix(".tmp")
    try:
        frame.to_parquet(tmp, index=False)
        os.replace(tmp, final)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return final


def write_table(run_dir: Path, key: TableKey, frame: pd.DataFrame) -> Path:
    """Atomic write: `.tmp` then os.replace()."""
    return _atomic_parquet(frame, table_path(run_dir, key))


def read_table(run_dir: Path, key: TableKey) -> pd.DataFrame:
    return pd.read_parquet(table_path(run_dir, key))


# -- index ----------------------------------------------------------------

INDEX_COLUMNS = ("model", "design_id", "seed", "filename", "n_rows", "written_at")


def index_path(run_dir: Path) -> Path:
    return run_dir / INDEX_FILENAME


def index_row(key: TableKey, path: Path, n_rows: int) -> dict:
    return {
        "model": key.model,
        "design_id": key.design_id,
        "seed": -1 if key.seed is None else key.seed,   # -1 sentinel: parquet dislikes a mixed type
        "filename": path.name,
        "n_rows": int(n_rows),
        "written_at": datetime.now(timezone.utc).isoformat(),
    }


def read_index(run_dir: Path) -> pd.DataFrame:
    path = index_path(run_dir)
    if not path.exists():
        return pd.DataFrame(columns=list(INDEX_COLUMNS))
    return pd.read_parquet(path)


def append_index(run_dir: Path, rows: list[dict]) -> Path:
    """Append rows to the index, atomically. The index is derived data (it
    can be rebuilt from tables/ via parse_name)."""
    path = index_path(run_dir)
    existing = read_index(run_dir)
    combined = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True) if rows else existing
    if combined.empty:
        return path
    combined = combined.drop_duplicates(subset=["filename"], keep="last")
    return _atomic_parquet(combined, path)


def rebuild_index(run_dir: Path) -> pd.DataFrame:
    """Reconstruct the index from the filenames themselves (parse_name)."""
    import pyarrow.parquet as pq

    rows = []
    for path in sorted((run_dir / TABLES_DIRNAME).glob("*.parquet")):
        try:
            key = parse_name(path.name)
        except ValueError:
            continue
        try:
            n_rows = pq.ParquetFile(path).metadata.num_rows
        except Exception:
            n_rows = -1
        rows.append(index_row(key, path, n_rows))
    frame = pd.DataFrame(rows, columns=list(INDEX_COLUMNS))
    if not frame.empty:
        _atomic_parquet(frame, index_path(run_dir))
    return frame


def key_tuple(key: TableKey) -> tuple:
    return (key.model, key.design_id, key.seed)


def completed_keys(run_dir: Path) -> set[tuple]:
    """Combinations ALREADY written, for `--resume`. An index entry whose
    file is gone does not count, or the combination would silently go
    missing from the results."""
    index = read_index(run_dir)
    if index.empty:
        return set()
    tables = run_dir / TABLES_DIRNAME
    done = set()
    for row in index.itertuples(index=False):
        if not (tables / row.filename).exists():
            continue
        seed = None if row.seed == -1 else int(row.seed)
        done.add((row.model, row.design_id, seed))
    return done


def completed_keys_from_files(run_dir: Path) -> set[tuple]:
    """`completed_keys` without the index: parse the table filenames. Used
    when several processes share one run dir (the index has no lock)."""
    done = set()
    for path in (run_dir / TABLES_DIRNAME).glob("*.parquet"):
        try:
            done.add(key_tuple(parse_name(path.name)))
        except ValueError:
            continue
    return done


# -- failures -------------------------------------------------------------


def append_failure(run_dir: Path, key: TableKey, exc: BaseException) -> Path:
    """A failed combination must NEVER disappear silently from the results."""
    path = run_dir / FAILURES_FILENAME
    record = {
        "model": key.model,
        "design_id": key.design_id,
        "seed": key.seed,
        "error_type": type(exc).__name__,
        "error": str(exc)[:2000],
        "failed_at": datetime.now(timezone.utc).isoformat(),
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
    return path


def read_failures(run_dir: Path) -> pd.DataFrame:
    path = run_dir / FAILURES_FILENAME
    if not path.exists():
        return pd.DataFrame()
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return pd.DataFrame(records)


# -- regime labels --------------------------------------------------------


def write_regime_labels(run_dir: Path, labels: pd.DataFrame) -> Path:
    return _atomic_parquet(labels, run_dir / REGIME_LABELS_FILENAME)


def read_regime_labels(run_dir: Path) -> pd.DataFrame | None:
    path = run_dir / REGIME_LABELS_FILENAME
    return pd.read_parquet(path) if path.exists() else None


# -- reading back ---------------------------------------------------------


def load_forecasts(
    run_dir: Path,
    *,
    horizon: int | None = None,
    models: list[str] | None = None,
    columns: list[str] | None = None,
    include_warmup: bool = False,
) -> pd.DataFrame:
    """Forecast rows of the run, filtered while reading (pyarrow filters),
    so a single horizon of the full grid never materializes the others.
    The warm-up year is left out unless asked for: it is never evaluated."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    index = rebuild_index(run_dir) if not index_path(run_dir).exists() else read_index(run_dir)
    if index.empty:
        return pd.DataFrame()
    if models is not None:
        index = index[index["model"].isin(models)]
    filters = []
    if horizon is not None:
        filters.append(("horizon", "=", int(horizon)))
    if not include_warmup:
        filters.append(("is_warmup", "=", False))
    tables = []
    for filename in index["filename"]:
        path = run_dir / TABLES_DIRNAME / filename
        if not path.exists():
            continue
        table = pq.read_table(path, columns=columns, filters=filters or None)
        if table.num_rows:
            tables.append(table)
    if not tables:
        return pd.DataFrame()
    return pa.concat_tables(tables, promote_options="permissive").to_pandas()
