"""Three defects found on the first end-to-end run on real data, all in how
per-day losses are lined up before any test or table:

1. seeds: a 5-seed model reached the ANOVA/VaR/n_obs as 5 rows per day;
2. grid-only broadcast: grid-only models (log_har, the DM baseline, among
   them) carry only their representative design_id, so DM/MCS cells for the
   other designs had no baseline at all;
3. MCS alignment: loss series were positional arrays, not aligned by day.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from projekat.results.report import _iter_design_cells
from projekat.results.tables import collapse_seed_losses, mcs_membership_table

DATES = pd.date_range("2024-01-02", periods=40, freq="B")
COLS = ["date", "symbol", "model", "design_id", "horizon", "qlike", "abs_error"]


def _rows(model, design_id, qlike, *, seed=None, symbol="AAA", dates=DATES):
    n = len(dates)
    return pd.DataFrame({
        "date": dates, "symbol": symbol, "model": model, "design_id": design_id,
        "horizon": 1, "seed": seed,
        "y_true": 1.0, "y_hat": 1.0, "qlike": qlike, "abs_error": qlike, "sq_error": qlike,
        "jump_flag": False,
    }).iloc[:n]


def test_collapse_averages_per_seed_losses_to_one_row_per_day():
    seeds = [_rows("patchtst", "5m__bpv__bns", np.full(len(DATES), float(k)), seed=k) for k in range(5)]
    det = _rows("log_har", "5m__bpv__naive", np.full(len(DATES), 7.0))
    out = collapse_seed_losses(pd.concat(seeds + [det], ignore_index=True))

    pt = out[out["model"] == "patchtst"]
    assert len(pt) == len(DATES)                       # one row per day, not five
    assert np.allclose(pt["qlike"], np.mean(range(5)))  # mean of per-seed losses
    pd.testing.assert_frame_equal(
        out[out["model"] == "log_har"].reset_index(drop=True), det.reset_index(drop=True),
        check_dtype=False,
    )


def test_collapse_is_a_no_op_without_stochastic_rows():
    det = _rows("log_har", "5m__bpv__naive", np.ones(len(DATES)))
    assert collapse_seed_losses(det) is det


def test_grid_only_models_are_shared_with_every_design_of_their_grid():
    """Per-design comparison cells carry the grid-only models of the same
    grid (relabelled), built one design at a time -- never the whole 12x
    copy at once, and never into another grid."""
    daily = pd.concat([
        _rows("log_har", "5m__bpv__naive", np.ones(len(DATES))),      # grid-only, representative
        _rows("char", "5m__bpv__naive", np.ones(len(DATES))),
        _rows("char", "5m__medrv__lm", np.ones(len(DATES))),
        _rows("char", "15m__bpv__naive", np.ones(len(DATES))),
    ], ignore_index=True)
    cells = dict(_iter_design_cells(daily, COLS))

    assert set(cells) == {"5m__bpv__naive", "5m__medrv__lm", "15m__bpv__naive"}
    for design_id in ("5m__bpv__naive", "5m__medrv__lm"):
        lh = cells[design_id][cells[design_id]["model"] == "log_har"]
        assert len(lh) == len(DATES) and set(lh["design_id"]) == {design_id}
        assert (cells[design_id]["model"] == "char").sum() == len(DATES)
    assert "log_har" not in set(cells["15m__bpv__naive"]["model"])   # other grid: never shared


def test_mcs_aligns_loss_series_by_day_not_by_position():
    """Model B lacks the first 10 days (a warm-up). Positionally, B's day k
    would be paired with A's day k-10; aligned, only the shared days count."""
    a = _rows("A", "5m__bpv__bns", np.linspace(1, 2, len(DATES)))
    b = _rows("B", "5m__bpv__bns", np.linspace(1, 2, len(DATES))[10:], dates=DATES[10:])
    out = mcs_membership_table(pd.concat([a, b], ignore_index=True))
    # identical losses on every shared day -> neither can be eliminated
    assert set(out.loc[out["in_mcs"], "model"]) == {"A", "B"}
