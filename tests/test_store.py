"""Persistence layer: filename grammar, atomic writes, index, resume.

All synthetic and seeded -- no network, no real data.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import pytest

from projekat.results import store
from projekat.results.store import TableKey, build_name, parse_name


# ── filename grammar ─────────────────────────────────────────────────────


def _all_keys():
    for model in ("log_har", "har_j", "ttm_c_plus_j", "patchtst"):
        for design in ("5m__bpv__bns", "15m__trbpv__lm_fdr__t3", "1m__medrv__naive"):
            for horizon in (1, 5, 22):
                for window_id in (1, 11):
                    for split in ("test", "validation"):
                        for seed in (None, 0, 4):
                            yield TableKey(model, design, horizon, window_id, split, seed)


def test_filename_round_trips_including_none_fields_and_seeds():
    """build_name(parse_name(x)) == x over the full combination space --
    including the trbpv `__t3` suffix (design_id itself contains `__`)
    and both the `none` and numeric seed forms."""
    keys = list(_all_keys())
    assert len(keys) == 432
    for key in keys:
        name = build_name(key)
        assert parse_name(name) == key
        assert build_name(parse_name(name)) == name


def test_design_id_containing_double_underscore_is_not_split_naively():
    key = TableKey("log_har", "15m__trbpv__lm_fdr__t3", 22, 11, "test", None)
    assert parse_name(build_name(key)).design_id == "15m__trbpv__lm_fdr__t3"


def test_seed_none_is_literal_none_not_empty():
    name = build_name(TableKey("har", "5m__bpv__bns", 1, 1, "test", None))
    assert "seed=none" in name
    assert parse_name(name).seed is None


def test_malformed_names_are_rejected():
    for bad in ("nonsense.parquet", "model=a__design=b.parquet", "model=a__design=b__h=x__w=1__split=test__seed=none.parquet"):
        with pytest.raises(ValueError):
            parse_name(bad)


def test_invalid_split_is_rejected():
    with pytest.raises(ValueError):
        TableKey("har", "5m__bpv__bns", 1, 1, "trainingfold", None)


# ── atomic writes ────────────────────────────────────────────────────────


def _frame(n=10):
    return pd.DataFrame({"date": pd.bdate_range("2020-01-01", periods=n), "y_hat": np.arange(n, dtype=float)})


def test_write_is_atomic_and_leaves_no_partial_file(tmp_path, monkeypatch):
    """Interrupting between the .tmp write and os.replace must leave no
    file a later read would accept."""
    run_dir = store.create_run_dir("run_x", base=tmp_path)
    key = TableKey("har", "5m__bpv__bns", 1, 1, "test", None)

    def boom(src, dst):
        raise KeyboardInterrupt("interrupted between tmp and replace")

    monkeypatch.setattr(store.os, "replace", boom)
    with pytest.raises(KeyboardInterrupt):
        store.write_table(run_dir, key, _frame())

    assert not store.table_path(run_dir, key).exists()
    assert list((run_dir / store.TABLES_DIRNAME).glob("*.tmp")) == []


def test_written_table_round_trips(tmp_path):
    run_dir = store.create_run_dir("run_y", base=tmp_path)
    key = TableKey("log_har", "5m__bpv__bns", 5, 3, "validation", None)
    frame = _frame(7)
    store.write_table(run_dir, key, frame)
    pd.testing.assert_frame_equal(store.read_table(run_dir, key), frame)


# ── index + resume ───────────────────────────────────────────────────────


def _write_some(run_dir, keys):
    rows = []
    for key in keys:
        path = store.write_table(run_dir, key, _frame())
        rows.append(store.index_row(key, path, 10))
    store.append_index(run_dir, rows)


def test_index_records_every_written_table(tmp_path):
    run_dir = store.create_run_dir("run_z", base=tmp_path)
    keys = [TableKey("har", "5m__bpv__bns", 1, w, s, None) for w in (1, 2) for s in ("test", "validation")]
    _write_some(run_dir, keys)

    index = store.read_index(run_dir)
    assert len(index) == 4
    assert set(index["split"]) == {"test", "validation"}
    assert store.completed_keys(run_dir) == {store.key_tuple(k) for k in keys}


def test_resume_produces_no_duplicates_and_no_gaps(tmp_path):
    """Running twice over the same combinations must give the same file
    count and the same total row count."""
    run_dir = store.create_run_dir("run_r", base=tmp_path)
    keys = [TableKey("log_har", "5m__bpv__bns", 1, w, "test", None) for w in (1, 2, 3)]

    _write_some(run_dir, keys)
    first_files = sorted(p.name for p in (run_dir / store.TABLES_DIRNAME).glob("*.parquet"))
    first_rows = store.read_index(run_dir)["n_rows"].sum()

    # second pass: everything is already complete
    done = store.completed_keys(run_dir)
    todo = [k for k in keys if store.key_tuple(k) not in done]
    assert todo == []

    _write_some(run_dir, keys)   # even re-writing must not duplicate index rows
    second_files = sorted(p.name for p in (run_dir / store.TABLES_DIRNAME).glob("*.parquet"))
    second_rows = store.read_index(run_dir)["n_rows"].sum()

    assert first_files == second_files
    assert first_rows == second_rows
    assert len(store.read_index(run_dir)) == 3


def test_index_ignores_rows_whose_file_vanished(tmp_path):
    """An index entry without its file must not count as complete, or the
    combination would be silently skipped and missing from results."""
    run_dir = store.create_run_dir("run_v", base=tmp_path)
    key = TableKey("har", "5m__bpv__bns", 1, 1, "test", None)
    _write_some(run_dir, [key])
    store.table_path(run_dir, key).unlink()
    assert store.completed_keys(run_dir) == set()


def test_rebuild_index_recovers_from_filenames(tmp_path):
    run_dir = store.create_run_dir("run_b", base=tmp_path)
    keys = [TableKey("char", "5m__bpv__bns", 1, w, "test", None) for w in (1, 2)]
    _write_some(run_dir, keys)
    store.index_path(run_dir).unlink()
    assert store.read_index(run_dir).empty

    rebuilt = store.rebuild_index(run_dir)
    assert len(rebuilt) == 2
    assert store.completed_keys(run_dir) == {store.key_tuple(k) for k in keys}


# ── failures + manifest ──────────────────────────────────────────────────


def test_failures_are_recorded_not_silently_dropped(tmp_path):
    run_dir = store.create_run_dir("run_f", base=tmp_path)
    key = TableKey("arfima", "5m__bpv__bns", 22, 11, "test", None)
    store.append_failure(run_dir, key, ValueError("singular matrix"))
    store.append_failure(run_dir, key, RuntimeError("boom"))

    failures = store.read_failures(run_dir)
    assert len(failures) == 2
    assert set(failures["error_type"]) == {"ValueError", "RuntimeError"}
    assert failures["model"].eq("arfima").all()


def test_manifest_round_trips(tmp_path):
    run_dir = store.create_run_dir("run_m", base=tmp_path)
    manifest = store.RunManifest(
        run_id="run_m", created_at="2026-09-16T00:00:00Z",
        symbols=["AAA"], models=["log_har"], horizons=[1], window_ids=[1],
    )
    store.write_manifest(run_dir, manifest)
    assert store.read_manifest(run_dir) == manifest


def test_load_split_returns_only_the_requested_split(tmp_path):
    run_dir = store.create_run_dir("run_s", base=tmp_path)
    keys = [TableKey("har", "5m__bpv__bns", 1, 1, s, None) for s in ("test", "validation")]
    _write_some(run_dir, keys)

    assert len(store.load_split(run_dir, "test")) == 10
    assert len(store.load_split(run_dir, "validation")) == 10
    assert store.load_split(run_dir, "test", models=["nope"]).empty
