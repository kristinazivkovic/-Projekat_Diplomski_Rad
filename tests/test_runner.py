"""model/runner.py orchestration: every registered model round-trips by
name alone, the positivity/correction mechanisms stay distinct, and the
primary seed-collapse path stays structurally distinct from the secondary
forecast ensemble."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from projekat.config import PATCHTST_SEEDS
from projekat.model.registry import MODELS
from projekat.model.runner import collapse_seeds, ensemble_forecast, run_all, seeds_for
from tests.conftest import make_panel

FIRST, LAST = "2014-01-01", "2014-06-30"


@pytest.fixture(autouse=True)
def fake_ttm(monkeypatch):
    from projekat.model.foundation import _ttm_base

    monkeypatch.setattr(_ttm_base, "ttm_forecast_batch",
                        lambda c: c[:, -10:].mean(axis=1, keepdims=True) + np.zeros((1, 192)))


@pytest.fixture(scope="module")
def panel():
    return make_panel(design_ids=("5m__bpv__bns", "5m__medrv__lm"), end="2014-09-30")


@pytest.mark.parametrize("model_name", sorted(set(MODELS) - {"patchtst"}))
def test_model_round_trips_through_runner_by_name(model_name, panel):
    out = run_all(panel, model_names=[model_name], horizons=(1, 5), first=FIRST, last=LAST)
    assert not out.empty
    assert (out["model"] == model_name).all()
    assert out["y_hat"].notna().mean() > 0.95
    # depends_on dedup: grid-only models run on one design per grid
    n_designs = out["design_id"].nunique()
    assert n_designs == (1 if MODELS[model_name].depends_on == {"grid"} else 2)


def test_every_model_declares_the_walk_forward_attributes():
    for name, model in MODELS.items():
        assert model.refit in ("every_21", "yearly", "never"), name
        assert model.correction in ("log", "ratio_of_sums", "none"), name
        assert not (model.correction == "none" and not model.predicts_levels), name
        assert not hasattr(model, "is_sequence_model"), name


def test_positivity_mechanisms_stay_distinct(panel):
    out = run_all(panel, model_names=["har", "mem", "log_har", "ttm_c_plus_j", "log_har_ttm"],
                  horizons=(1,), first=FIRST, last=LAST)
    by = {m: g for m, g in out.groupby("model")}
    for levels_model in ("har", "mem"):            # structural: no correction at all
        assert (by[levels_model]["correction_kind"] == "none").all()
        assert by[levels_model]["backtransform_correction"].isna().all()
        assert (by[levels_model]["y_hat"].dropna() >= 0).all()
    assert (by["log_har"]["correction_kind"] == "log").all()
    for combined in ("ttm_c_plus_j", "log_har_ttm"):  # one ratio-of-sums correction on the sum
        assert (by[combined]["correction_kind"] == "ratio_of_sums").all()
        assert by[combined]["combined_correction"].notna().all()


def test_patchtst_uses_three_seeds():
    assert seeds_for("patchtst") == PATCHTST_SEEDS == (0, 1, 2)
    assert seeds_for("log_har") == (None,)


def _seed_rows(log_y_hats, corrections):
    n = len(log_y_hats)
    return pd.DataFrame({
        "model": ["patchtst"] * n, "design_id": ["5m__bpv__bns"] * n, "horizon": [1] * n,
        "date": [pd.Timestamp("2020-01-02")] * n, "symbol": ["AAA"] * n, "seed": list(range(n)),
        "log_y_hat": log_y_hats, "backtransform_correction": corrections,
        "y_hat": np.exp(np.asarray(log_y_hats) + np.asarray(corrections)), "y_true": [1e-4] * n,
    })


def test_collapse_seeds_is_a_no_op_and_ensemble_is_a_distinct_secondary_path():
    rows = _seed_rows([0.1, 0.2, 0.3], [0.01, 0.02, 0.03])
    pd.testing.assert_frame_equal(collapse_seeds(rows), rows)
    ens = ensemble_forecast(rows)
    assert len(ens) == 1
    assert ens["y_hat_ensemble"].iloc[0] == pytest.approx(np.exp(0.2 + 0.02))
    assert ens["y_hat_ensemble"].iloc[0] != pytest.approx(rows["y_hat"].mean())


def test_ensemble_forecast_does_not_double_count_the_correction():
    rows = _seed_rows([-0.2, 0.1, 0.4], [0.0, 0.5, 1.0])
    ens = ensemble_forecast(rows)
    assert ens["y_hat_ensemble"].iloc[0] == pytest.approx(np.exp(np.mean([-0.2, 0.1, 0.4]) + 0.5))
    assert not np.allclose(rows["log_y_hat"], np.log(rows["y_hat"]))
