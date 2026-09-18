"""Sharded runner: enumeration, depends_on dedup, resume, invariants.

Synthetic seeded fixtures only -- no network, no real data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from projekat.model.sharded import (
    ECONOMETRIC_MODELS,
    Combination,
    dedup_report,
    enumerate_combinations,
    parse_int_list,
    parse_window_spec,
    resolve_models,
    run_sharded,
)
from projekat.model.windows import anchored_windows, split_with_validation
from projekat.results import store


@pytest.fixture
def panel():
    """Two symbols x three designs spanning two grids, 2010-2016."""
    rng = np.random.default_rng(0)
    rows = []
    for design_id in ("5m__bpv__bns", "5m__medrv__lm", "1m__bpv__bns"):
        for symbol in ("AAA", "BBB"):
            for date in pd.bdate_range("2010-01-01", "2016-12-31"):
                rv = float(rng.lognormal(-9.5, 0.4))
                j = rv * 0.3 if rng.random() < 0.1 else 0.0
                rows.append(
                    {
                        "design_id": design_id, "symbol": symbol, "date": date,
                        "RV_d": rv, "RV_w": rv, "RV_m": rv,
                        "C_d": rv - j, "C_w": rv - j, "C_m": rv - j,
                        "J_d": j, "J_w": j, "J_m": j,
                        "RS_pos": rv / 2, "RS_neg": rv / 2, "SJ": 0.0, "RQ": rv ** 2,
                        "jump_flag": j > 0, "r_d": float(rng.normal(0, rv ** 0.5)),
                        "vix_lag1": 18.0,
                    }
                )
    return pd.DataFrame(rows)


# ── argument parsing ─────────────────────────────────────────────────────


def test_model_group_resolves_to_the_econometric_set():
    assert resolve_models("econometric") == list(ECONOMETRIC_MODELS)
    assert resolve_models("har,log_har") == ["har", "log_har"]


def test_unknown_model_is_an_error_not_a_silent_skip():
    with pytest.raises(ValueError, match="unknown models"):
        resolve_models("har,not_a_model")


def test_window_spec_accepts_ranges_and_lists():
    assert parse_window_spec("1-11", 11) == list(range(1, 12))
    assert parse_window_spec("1,3,5", 11) == [1, 3, 5]
    assert parse_window_spec("all", 11) == list(range(1, 12))
    with pytest.raises(ValueError):
        parse_window_spec("0-3", 11)
    with pytest.raises(ValueError):
        parse_window_spec("12", 11)


def test_int_list_parsing():
    assert parse_int_list("all", (1, 5, 22)) == [1, 5, 22]
    assert parse_int_list("1,5", (1, 5, 22)) == [1, 5]


# ── enumeration + depends_on dedup ───────────────────────────────────────


def test_enumeration_is_side_effect_free(panel, tmp_path):
    """--dry-run depends on this: enumerating must touch no disk."""
    before = list(tmp_path.iterdir())
    combos = enumerate_combinations(
        panel, model_names=list(ECONOMETRIC_MODELS), horizons=[1, 5], window_ids=[1, 2]
    )
    assert combos
    assert list(tmp_path.iterdir()) == before


def test_grid_only_models_are_fitted_once_per_grid(panel):
    """depends_on == {"grid"} must collapse to one design per grid -- not
    one per (grid, estimator, jump_test). This is a correctness condition,
    not an optimization: duplicated non-independent rows corrupt ANOVA."""
    grid_only = enumerate_combinations(panel, model_names=["log_har"], horizons=[1], window_ids=[1])
    full = enumerate_combinations(panel, model_names=["char"], horizons=[1], window_ids=[1])

    # panel has 2 grids (5m, 1m) but 3 design_ids
    assert len({c.design_id for c in grid_only}) == 2
    assert len({c.design_id for c in full}) == 3


def test_dedup_report_states_whether_dedup_applied(panel):
    report = dedup_report(panel, list(ECONOMETRIC_MODELS), n_h=3, n_w=11)
    by_model = report.set_index("model")

    assert by_model.loc["log_har", "deduplicated"]          # grid-only: fewer
    assert not by_model.loc["char", "deduplicated"]         # full: same as naive
    assert by_model.loc["log_har", "actual"] < by_model.loc["log_har", "naive_product"]
    assert by_model.loc["char", "actual"] == by_model.loc["char", "naive_product"]


def test_enumeration_count_matches_the_dedup_report(panel):
    models, horizons, windows = list(ECONOMETRIC_MODELS), [1, 5], [1, 2, 3]
    combos = enumerate_combinations(panel, model_names=models, horizons=horizons, window_ids=windows)
    report = dedup_report(panel, models, len(horizons), len(windows))
    assert len(combos) == int(report["actual"].sum())


def test_deterministic_models_get_exactly_one_seedless_combination(panel):
    combos = enumerate_combinations(panel, model_names=["log_har"], horizons=[1], window_ids=[1])
    assert all(c.seed is None for c in combos)


# ── execution ────────────────────────────────────────────────────────────


def test_run_writes_one_table_per_split_and_indexes_them(panel, tmp_path):
    run_dir = store.create_run_dir("run_a", base=tmp_path)
    combos = enumerate_combinations(panel, model_names=["log_har"], horizons=[1], window_ids=[1])

    counts = run_sharded(panel, combos, run_dir, progress=False)

    assert counts["failed"] == 0
    assert counts["written"] == 2 * len(combos)          # test + validation
    index = store.read_index(run_dir)
    assert len(index) == 2 * len(combos)
    assert set(index["split"]) == {"test", "validation"}


def test_resume_skips_completed_combinations(panel, tmp_path):
    run_dir = store.create_run_dir("run_b", base=tmp_path)
    combos = enumerate_combinations(panel, model_names=["log_har"], horizons=[1], window_ids=[1])

    first = run_sharded(panel, combos, run_dir, progress=False)
    second = run_sharded(panel, combos, run_dir, resume=True, progress=False)

    assert second["skipped"] == len(combos)
    assert second["written"] == 0
    # same file count and same total rows -- no duplicated days
    index = store.read_index(run_dir)
    assert len(index) == first["written"]
    assert index["n_rows"].sum() > 0


def test_failures_are_recorded_and_do_not_abort_the_run(panel, tmp_path, monkeypatch):
    run_dir = store.create_run_dir("run_c", base=tmp_path)
    combos = enumerate_combinations(panel, model_names=["log_har"], horizons=[1], window_ids=[1, 2])

    import projekat.model.sharded as sharded_mod

    calls = {"n": 0}
    real = sharded_mod.run_one

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("synthetic failure")
        return real(*args, **kwargs)

    monkeypatch.setattr(sharded_mod, "run_one", flaky)
    counts = run_sharded(panel, combos, run_dir, progress=False)

    assert counts["failed"] == 1
    assert counts["written"] > 0                  # the run continued
    failures = store.read_failures(run_dir)
    assert len(failures) == 1
    assert failures.iloc[0]["error_type"] == "RuntimeError"


# ── invariants ───────────────────────────────────────────────────────────


def test_no_test_observation_appears_in_training_or_validation(panel):
    """Anchored walk-forward: training start fixed, only the end expands,
    and no test day may leak into any fitting span."""
    design_df = panel[panel["design_id"] == "5m__bpv__bns"]
    checked = 0
    for window in anchored_windows():
        fit_train, val_fold, test = split_with_validation(design_df, window)
        # disjointness must hold even where the fixture runs out of data
        train_days = set(fit_train["date"]) | set(val_fold["date"])
        assert train_days.isdisjoint(set(test["date"]))
        if test.empty:
            continue    # fixture ends 2016; later windows test beyond it
        assert fit_train["date"].max() < val_fold["date"].min()
        assert val_fold["date"].max() < test["date"].min()
        checked += 1
    assert checked >= 2, "fixture covered no window with a non-empty test set"


def test_training_start_is_fixed_while_end_expands():
    windows = anchored_windows()
    assert len({w.train_start for w in windows}) == 1
    ends = [w.train_end for w in windows]
    assert ends == sorted(ends)
    assert len(windows) == 11


def test_validation_rows_are_stored_separately_from_test(panel, tmp_path):
    """The validation fold feeds only the back-transform correction and
    Mincer-Zarnowitz -- it must never be mixed into the test split."""
    run_dir = store.create_run_dir("run_d", base=tmp_path)
    combos = enumerate_combinations(panel, model_names=["log_har"], horizons=[1], window_ids=[1])
    run_sharded(panel, combos, run_dir, progress=False)

    test = store.load_split(run_dir, "test")
    val = store.load_split(run_dir, "validation")
    assert not test.empty and not val.empty
    assert set(test["date"]).isdisjoint(set(val["date"]))


def test_levels_model_carries_no_backtransform_correction(panel, tmp_path):
    """har is NNLS-in-levels: positivity is structural, so it must have no
    correction. log_har must have one. The two mechanisms stay distinct."""
    run_dir = store.create_run_dir("run_e", base=tmp_path)
    combos = enumerate_combinations(panel, model_names=["har", "log_har"], horizons=[1], window_ids=[1])
    run_sharded(panel, combos, run_dir, progress=False)

    test = store.load_split(run_dir, "test")
    har = test[test["model"] == "har"]
    log_har = test[test["model"] == "log_har"]

    assert har["backtransform_correction"].isna().all()
    assert har["log_y_hat"].isna().all()
    assert (har["y_hat"] >= 0).all()                       # NNLS guarantees this
    assert log_har["backtransform_correction"].notna().all()


def test_every_econometric_model_round_trips_by_name_alone(panel, tmp_path):
    """The runner resolves models from the registry by string name; adding
    a model is one file with a decorator, never an edit to the resolver."""
    run_dir = store.create_run_dir("run_f", base=tmp_path)
    combos = enumerate_combinations(
        panel, model_names=list(ECONOMETRIC_MODELS), horizons=[1], window_ids=[1]
    )
    counts = run_sharded(panel, combos, run_dir, progress=False)

    assert counts["failed"] == 0
    written = set(store.read_index(run_dir)["model"])
    assert written == set(ECONOMETRIC_MODELS)


def test_separate_fit_per_horizon_no_iterated_multistep(panel):
    """Direct multi-horizon: h is a fit() argument, so each horizon is its
    own combination rather than an iterated one-step forecast."""
    combos = enumerate_combinations(panel, model_names=["log_har"], horizons=[1, 5, 22], window_ids=[1])
    assert sorted({c.horizon for c in combos}) == [1, 5, 22]
    for horizon in (1, 5, 22):
        assert sum(c.horizon == horizon for c in combos) == 2   # two grids
