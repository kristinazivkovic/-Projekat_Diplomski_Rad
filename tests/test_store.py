"""Persistence layer: filename grammar, atomic writes, index, resume, and
filtered loading. All synthetic -- no network, no real data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from projekat.results import store
from projekat.results.store import TableKey, build_name, parse_name


def _all_keys():
    for model in ("log_har", "har_j", "ttm_c_plus_j", "patchtst"):
        for design in ("5m__bpv__bns", "15m__trbpv__lm_fdr__t3", "1m__medrv__naive"):
            for seed in (None, 0, 2):
                yield TableKey(model, design, seed)


def test_filename_round_trips_including_none_seed_and_double_underscore_designs():
    keys = list(_all_keys())
    assert len(keys) == 36
    for key in keys:
        name = build_name(key)
        assert parse_name(name) == key
        assert build_name(parse_name(name)) == name
    assert "seed=none" in build_name(TableKey("har", "5m__bpv__bns", None))


def test_malformed_names_are_rejected():
    for bad in ("nonsense.parquet", "model=a__design=b.parquet", "model=a__design=b__seed=x.parquet"):
        with pytest.raises(ValueError):
            parse_name(bad)


def _frame(n=10, *, horizons=(1, 5), warm=3):
    rows = []
    for h in horizons:
        for i, d in enumerate(pd.bdate_range("2020-01-01", periods=n)):
            rows.append({"date": d, "horizon": h, "y_hat": float(i), "is_warmup": i < warm, "model": "har"})
    return pd.DataFrame(rows)


def test_write_is_atomic_and_leaves_no_partial_file(tmp_path, monkeypatch):
    run_dir = store.create_run_dir("run_x", base=tmp_path)
    key = TableKey("har", "5m__bpv__bns", None)

    def boom(src, dst):
        raise KeyboardInterrupt("interrupted between tmp and replace")

    monkeypatch.setattr(store.os, "replace", boom)
    with pytest.raises(KeyboardInterrupt):
        store.write_table(run_dir, key, _frame())
    assert not store.table_path(run_dir, key).exists()
    assert list((run_dir / store.TABLES_DIRNAME).glob("*.tmp")) == []


def test_index_resume_and_rebuild(tmp_path):
    run_dir = store.create_run_dir("run_z", base=tmp_path)
    keys = [TableKey("har", d, None) for d in ("5m__bpv__bns", "1m__bpv__bns")]
    rows = [store.index_row(k, store.write_table(run_dir, k, _frame()), 20) for k in keys]
    store.append_index(run_dir, rows)
    store.append_index(run_dir, rows)                       # re-append: no duplicates
    assert len(store.read_index(run_dir)) == 2
    assert store.completed_keys(run_dir) == {store.key_tuple(k) for k in keys}
    assert store.completed_keys_from_files(run_dir) == {store.key_tuple(k) for k in keys}

    store.table_path(run_dir, keys[0]).unlink()             # vanished file never counts as done
    assert store.completed_keys(run_dir) == {store.key_tuple(keys[1])}
    store.index_path(run_dir).unlink()
    rebuilt = store.rebuild_index(run_dir)
    assert len(rebuilt) == 1 and rebuilt["n_rows"].iloc[0] == 20


def test_load_forecasts_filters_horizon_and_drops_warm_up(tmp_path):
    run_dir = store.create_run_dir("run_l", base=tmp_path)
    for design in ("5m__bpv__bns", "1m__bpv__bns"):
        store.write_table(run_dir, TableKey("har", design, None), _frame())
    store.rebuild_index(run_dir)
    h1 = store.load_forecasts(run_dir, horizon=1)
    assert set(h1["horizon"]) == {1}
    assert len(h1) == 2 * (10 - 3) and not h1["is_warmup"].any()
    with_warm = store.load_forecasts(run_dir, horizon=5, include_warmup=True, columns=["date", "y_hat"])
    assert len(with_warm) == 20 and list(with_warm.columns) == ["date", "y_hat"]
    assert store.load_forecasts(run_dir, models=["nope"]).empty


def test_failures_manifest_and_regime_labels_round_trip(tmp_path):
    run_dir = store.create_run_dir("run_f", base=tmp_path)
    key = TableKey("arfima", "5m__bpv__bns", None)
    store.append_failure(run_dir, key, ValueError("singular matrix"))
    store.append_failure(run_dir, key, RuntimeError("boom"))
    failures = store.read_failures(run_dir)
    assert len(failures) == 2 and set(failures["error_type"]) == {"ValueError", "RuntimeError"}

    manifest = store.RunManifest(run_id="run_f", created_at="2026-09-19T00:00:00Z", symbols=["AAA"],
                                 models=["log_har"], horizons=[1], scheme={"refit_every_sessions": 21})
    store.write_manifest(run_dir, manifest)
    assert store.read_manifest(run_dir) == manifest

    labels = pd.DataFrame({"symbol": ["AAA"], "date": [pd.Timestamp("2020-01-02")], "regime": ["calm"]})
    store.write_regime_labels(run_dir, labels)
    pd.testing.assert_frame_equal(store.read_regime_labels(run_dir), labels)
    assert np.isfinite(len(labels))
