"""Persistence layer for the raw forecast tables.

WHY THIS EXISTS
===============
`runner.py::run_all` accumulates every forecast row in memory and
concatenates only at the very end. On the full grid that is ~106 million
rows (~10.6 GB) plus an equally large validation frame -- it does not fit
in RAM, and a crash at fit 13,000 loses everything. This module replaces
that with one table written per combination, immediately, so peak memory
is the size of a single table and an interrupted run can resume.

ONE RUN DIRECTORY, EVERY MODEL GROUP WRITES INTO IT
===================================================
The econometric models go first (`log_har` is the benchmark without which
no other model's relative QLIKE is computable), but `arfima`, `mem`, the
tree models, PatchTST and the `ttm_*` models from later sessions write
into the SAME directory and the SAME index, with no change to this file.
That is why a table name carries every field of the combination,
including `seed`, which the econometric models do not have (`none`) and
PatchTST does.

    <run-dir>/
        manifest.json            run provenance (see RunManifest)
        index.parquet            one row per written table
        failures.jsonl           one row per failed combination
        tables/<name>.parquet    raw forecast rows

FILENAME GRAMMAR
================
    model=<m>__design=<d>__h=<h>__w=<w>__split=<s>__seed=<seed>.parquet

Fields are `key=value` separated by `__`, in a fixed order, the same way
`MeasurementDesign.id` is built -- `parse_name` is the exact inverse of
`build_name`, and this is the ONLY place a table name is taken apart. A
value that does not exist is the literal `none` (never an empty string),
so the field count never varies and the round-trip stays unambiguous.

`design` itself contains `__` (e.g. `5m__bpv__bns`), so parsing splits on
`key=` boundaries, never on a plain `split("__")` -- the same trap that
makes `evaluation/anova.py::parse_design_id` the single parsing site for
`design_id`.

ATOMICITY
=========
Each table is written to `<name>.tmp` and then `os.replace()`d onto its
final name. `os.replace` is atomic within a filesystem on both Windows
and POSIX, so a reader never sees a half-written parquet: the file at the
final name is either complete or absent. An interrupt between the two
steps leaves only a `.tmp`, which `--resume` ignores and overwrites.
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

SPLITS = ("test", "validation")

# Fixed field order -- build_name and parse_name both depend on it.
_FIELDS = ("model", "design", "h", "w", "split", "seed")
_NONE = "none"

# Values must contain neither "=" nor "__", or the name is ambiguous.
_VALUE_RE = re.compile(r"^[A-Za-z0-9._+-]+$")


@dataclass(frozen=True)
class TableKey:
    """One combination = one table. `seed` is None for deterministic
    models; `window_id` is 1-based (w=1..11) to match `--windows 1-11` on
    the command line."""

    model: str
    design_id: str
    horizon: int
    window_id: int
    split: str
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {self.split!r}")


def build_name(key: TableKey) -> str:
    """TableKey -> filename. Exact inverse of parse_name."""
    parts = {
        "model": key.model,
        "design": key.design_id,
        "h": str(key.horizon),
        "w": str(key.window_id),
        "split": key.split,
        "seed": _NONE if key.seed is None else str(key.seed),
    }
    for name, value in parts.items():
        if name == "design":
            # design_id legitimately contains "__" (5m__bpv__bns); only its
            # segments are checked
            segments = value.split("__")
        else:
            segments = [value]
        for segment in segments:
            if not _VALUE_RE.match(segment):
                raise ValueError(f"field value {name}={value!r} is not filename-safe")
    return "__".join(f"{k}={v}" for k, v in parts.items()) + ".parquet"


def parse_name(filename: str) -> TableKey:
    """Filename -> TableKey. Exact inverse of build_name.

    Splits on `key=` boundaries, NOT on `__`, because `design` itself
    contains `__`. The only place in the codebase where a table name is
    taken apart."""
    stem = filename[:-len(".parquet")] if filename.endswith(".parquet") else filename

    # Only `design` may contain `__`; every other field is restricted to
    # its own narrow character set, which keeps the parse unambiguous even
    # though `design` is a non-greedy `.+?`.
    pattern = (
        r"^model=(?P<model>[^=]+?)"
        r"__design=(?P<design>.+?)"
        r"__h=(?P<h>\d+)"
        r"__w=(?P<w>\d+)"
        r"__split=(?P<split>[a-z]+)"
        r"__seed=(?P<seed>none|\d+)$"
    )
    match = re.match(pattern, stem)
    if not match:
        raise ValueError(f"table name does not match the grammar: {filename!r}")

    seed_raw = match.group("seed")
    return TableKey(
        model=match.group("model"),
        design_id=match.group("design"),
        horizon=int(match.group("h")),
        window_id=int(match.group("w")),
        split=match.group("split"),
        seed=None if seed_raw == _NONE else int(seed_raw),
    )


# -- manifest -------------------------------------------------------------


@dataclass(frozen=True)
class RunManifest:
    """Run provenance. Same shape as data/vault.py::Manifest -- a frozen
    dataclass -> asdict -> json.dumps(indent=2) -> write_text."""

    run_id: str
    created_at: str
    argv: list[str] = field(default_factory=list)
    git_sha: str | None = None
    symbols: list[str] = field(default_factory=list)
    models: list[str] = field(default_factory=list)
    grids: list[str] = field(default_factory=list)
    horizons: list[int] = field(default_factory=list)
    window_ids: list[int] = field(default_factory=list)
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
    path.write_text(json.dumps(asdict(manifest), indent=2), encoding="utf-8")
    return path


def read_manifest(run_dir: Path) -> RunManifest | None:
    path = run_dir / MANIFEST_FILENAME
    if not path.exists():
        return None
    return RunManifest(**json.loads(path.read_text(encoding="utf-8")))


# -- writing tables -------------------------------------------------------


def table_path(run_dir: Path, key: TableKey) -> Path:
    return run_dir / TABLES_DIRNAME / build_name(key)


def write_table(run_dir: Path, key: TableKey, frame: pd.DataFrame) -> Path:
    """Atomic write: `.tmp` then os.replace(). An interrupt between the two
    steps leaves only a `.tmp`, which no reader accepts as a table."""
    final = table_path(run_dir, key)
    final.parent.mkdir(parents=True, exist_ok=True)
    tmp = final.with_suffix(".tmp")
    try:
        frame.to_parquet(tmp, index=False)
        os.replace(tmp, final)
    except BaseException:
        # including KeyboardInterrupt: never leave a half-written .tmp behind
        tmp.unlink(missing_ok=True)
        raise
    return final


def read_table(run_dir: Path, key: TableKey) -> pd.DataFrame:
    return pd.read_parquet(table_path(run_dir, key))


# -- index ----------------------------------------------------------------

INDEX_COLUMNS = (
    "model", "design_id", "horizon", "window_id", "split", "seed",
    "filename", "n_rows", "written_at",
)


def index_path(run_dir: Path) -> Path:
    return run_dir / INDEX_FILENAME


def index_row(key: TableKey, path: Path, n_rows: int) -> dict:
    return {
        "model": key.model,
        "design_id": key.design_id,
        "horizon": key.horizon,
        "window_id": key.window_id,
        "split": key.split,
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
    """Append rows to the index, atomically. The index is derived data --
    it can be reconstructed from the filenames in tables/ via parse_name --
    but it is kept as parquet so `--resume` need not list a directory
    holding tens of thousands of files."""
    path = index_path(run_dir)
    existing = read_index(run_dir)
    combined = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True) if rows else existing
    if combined.empty:
        return path
    combined = combined.drop_duplicates(subset=["filename"], keep="last")
    tmp = path.with_suffix(".tmp")
    try:
        combined.to_parquet(tmp, index=False)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return path


def rebuild_index(run_dir: Path) -> pd.DataFrame:
    """Reconstruct the index from the filenames themselves (parse_name). A
    safety net for when the index is lost or falls behind tables/."""
    rows = []
    for path in sorted((run_dir / TABLES_DIRNAME).glob("*.parquet")):
        try:
            key = parse_name(path.name)
        except ValueError:
            continue
        try:
            n_rows = pd.read_parquet(path, columns=["date"]).shape[0]
        except Exception:
            n_rows = -1
        rows.append(index_row(key, path, n_rows))
    frame = pd.DataFrame(rows, columns=list(INDEX_COLUMNS))
    if not frame.empty:
        tmp = index_path(run_dir).with_suffix(".tmp")
        frame.to_parquet(tmp, index=False)
        os.replace(tmp, index_path(run_dir))
    return frame


def completed_keys(run_dir: Path) -> set[tuple]:
    """Combinations that are ALREADY written, for `--resume`. Returns a set
    of tuples aligned with `TableKey`'s fields so a caller can skip without
    re-parsing names.

    Verifies the file actually exists: an index entry without its file
    (deleted by hand, say) must not count as complete, or the combination
    would be silently skipped and go missing from the results."""
    index = read_index(run_dir)
    if index.empty:
        return set()
    tables = run_dir / TABLES_DIRNAME
    done = set()
    for row in index.itertuples(index=False):
        if not (tables / row.filename).exists():
            continue
        seed = None if row.seed == -1 else int(row.seed)
        done.add((row.model, row.design_id, int(row.horizon), int(row.window_id), row.split, seed))
    return done


def key_tuple(key: TableKey) -> tuple:
    return (key.model, key.design_id, key.horizon, key.window_id, key.split, key.seed)


# -- failures -------------------------------------------------------------


def append_failure(run_dir: Path, key: TableKey, exc: BaseException) -> Path:
    """Append a failure to failures.jsonl. A failed combination must NEVER
    disappear silently from the results -- without this record, a model
    that crashes on one design looks exactly like a model that was never
    requested for that design."""
    path = run_dir / FAILURES_FILENAME
    record = {
        "model": key.model,
        "design_id": key.design_id,
        "horizon": key.horizon,
        "window_id": key.window_id,
        "split": key.split,
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


# -- reading back ---------------------------------------------------------


def load_split(run_dir: Path, split: str, *, models: list[str] | None = None) -> pd.DataFrame:
    """Load every table of one split back into a single frame.

    This is the point where memory grows again, so use it only when the
    evaluation genuinely needs a combined frame; for a large grid, filter
    `models` or read table by table via read_index/read_table."""
    index = read_index(run_dir)
    if index.empty:
        return pd.DataFrame()
    wanted = index[index["split"] == split]
    if models is not None:
        wanted = wanted[wanted["model"].isin(models)]
    frames = []
    tables = run_dir / TABLES_DIRNAME
    for filename in wanted["filename"]:
        path = tables / filename
        if path.exists():
            frames.append(pd.read_parquet(path))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
