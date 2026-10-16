"""Sharded runner: enumeration, depends_on dedup, resume, failures.
Synthetic seeded fixtures only -- no network, no real data."""

from __future__ import annotations

import pandas as pd
import pytest

from projekat.model.sharded import (
    ECONOMETRIC_MODELS,
    Combination,
    dedup_report,
    enumerate_combinations,
    parse_int_list,
    resolve_models,
    run_sharded,
)
from projekat.results import store
from tests.conftest import make_panel


@pytest.fixture(scope="module")
def panel():
    """Two symbols x three designs over two grids, short evaluation span."""
    return make_panel(design_ids=("5m__bpv__bns", "5m__medrv__lm", "1m__bpv__bns"), end="2014-04-30")


def _run(panel, combos, run_dir, **kw):
    import projekat.model.sharded as sharded_mod

    real = sharded_mod.run_one

    def short(design_df, **kwargs):
        return real(design_df, first="2014-01-01", last="2014-03-31", **kwargs)

    sharded_mod.run_one = short
    try:
        return run_sharded(panel, combos, run_dir, horizons=(1, 5), progress=False, **kw)
    finally:
        sharded_mod.run_one = real


def test_model_group_resolves_and_unknown_is_an_error():
    assert resolve_models("econometric") == list(ECONOMETRIC_MODELS)
    assert resolve_models("har,log_har") == ["har", "log_har"]
    with pytest.raises(ValueError, match="unknown models"):
        resolve_models("har,not_a_model")
    assert parse_int_list("all", (1, 5, 22)) == [1, 5, 22]
    assert parse_int_list("1,5", (1, 5, 22)) == [1, 5]


def test_enumeration_is_side_effect_free(panel, tmp_path):
    before = list(tmp_path.iterdir())
    assert enumerate_combinations(panel, model_names=list(ECONOMETRIC_MODELS))
    assert list(tmp_path.iterdir()) == before


def test_one_combination_per_model_design_seed_with_depends_on_dedup(panel):
    grid_only = enumerate_combinations(panel, model_names=["log_har"])
    full = enumerate_combinations(panel, model_names=["char"])
    assert {c.design_id for c in grid_only} == {"1m__bpv__bns", "5m__bpv__bns"}   # one per grid
    assert len(full) == 3
    assert all(c.seed is None for c in grid_only)
    patch = enumerate_combinations(panel, model_names=["patchtst"])
    assert len(patch) == 3 * 3 and {c.seed for c in patch} == {0, 1, 2}


def test_dedup_report_matches_enumeration_and_counts_fits(panel):
    models = list(ECONOMETRIC_MODELS) + ["ttm_rv", "patchtst"]
    report = dedup_report(panel, models, n_horizons=3).set_index("model")
    assert int(report["combinations"].sum()) == len(enumerate_combinations(panel, model_names=models))
    assert report.loc["log_har", "deduplicated"] and not report.loc["char", "deduplicated"]
    assert report.loc["ttm_rv", "fits"] == 0                       # zero-shot: nothing is estimated
    assert report.loc["patchtst", "refits_per_combination"] == 12  # yearly, 2014..2025
    assert report.loc["log_har", "refits_per_combination"] > 140   # every 21 sessions


def test_run_writes_one_table_per_combination_with_every_horizon(panel, tmp_path):
    run_dir = store.create_run_dir("run_a", base=tmp_path)
    combos = enumerate_combinations(panel, model_names=["log_har", "har"])
    counts = _run(panel, combos, run_dir)
    assert counts["failed"] == 0 and counts["written"] == len(combos)
    table = store.read_table(run_dir, combos[0].key())
    assert set(table["horizon"]) == {1, 5}
    assert table["is_warmup"].all()                  # 2014 only: all warm-up
    assert len(store.read_index(run_dir)) == len(combos)


def test_resume_skips_completed_combinations(panel, tmp_path):
    run_dir = store.create_run_dir("run_b", base=tmp_path)
    combos = enumerate_combinations(panel, model_names=["log_har"])
    first = _run(panel, combos, run_dir)
    second = _run(panel, combos, run_dir, resume=True)
    assert second["skipped"] == len(combos) and second["written"] == 0
    assert len(store.read_index(run_dir)) == first["written"]


def test_failures_are_recorded_and_do_not_abort_the_run(panel, tmp_path, monkeypatch):
    import projekat.model.sharded as sharded_mod

    run_dir = store.create_run_dir("run_c", base=tmp_path)
    combos = enumerate_combinations(panel, model_names=["log_har"])
    calls = {"n": 0}
    real = sharded_mod.run_one

    def flaky(design_df, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("synthetic failure")
        return real(design_df, first="2014-01-01", last="2014-02-28", **kwargs)

    monkeypatch.setattr(sharded_mod, "run_one", flaky)
    counts = run_sharded(panel, combos, run_dir, horizons=(1,), progress=False)
    assert counts["failed"] == 1 and counts["written"] == len(combos) - 1
    failures = store.read_failures(run_dir)
    assert len(failures) == 1 and failures.iloc[0]["error_type"] == "RuntimeError"


def test_every_econometric_model_round_trips_by_name_alone(panel, tmp_path):
    run_dir = store.create_run_dir("run_f", base=tmp_path)
    combos = enumerate_combinations(panel[panel["design_id"] == "5m__bpv__bns"], model_names=list(ECONOMETRIC_MODELS))
    counts = _run(panel, combos, run_dir)
    assert counts["failed"] == 0
    assert set(store.read_index(run_dir)["model"]) == set(ECONOMETRIC_MODELS)


def test_combination_key_round_trips():
    combo = Combination(model="patchtst", design_id="15m__trbpv__lm_fdr", seed=2)
    assert store.parse_name(store.build_name(combo.key())) == combo.key()
