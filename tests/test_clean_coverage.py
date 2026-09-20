"""Coverage filtering: one grid-independent decision per day.

clean_day receives 1m bars; the relative threshold is measured on 1m bars
and the absolute floor on the number of returns EVERY configured grid would
yield. Early-close sessions are dropped outright. Together this makes
1m/5m/15m share exactly the same day set by construction."""

from __future__ import annotations

import pandas as pd
import pytest

from projekat.build_panel import returns_by_grid_from_raw
from projekat.data.calendar import session_window
from projekat.data.clean import EXCLUSION_REASONS, _n_grid_returns, clean_day, clean_day_with_reason
from projekat.measure.grids import log_returns, resample_day

FULL_DAY = "2015-03-10"
FULL_DAY_2 = "2015-03-11"
HALF_DAY = "2015-11-27"  # day after Thanksgiving, 13:00 close


def _bars(day: str, *, keep_every: int = 1, drop_minutes: tuple[int, int] | None = None) -> pd.DataFrame:
    open_ts, close_ts = session_window(day)
    ts = pd.date_range(open_ts, close_ts, freq="min", inclusive="left")[::keep_every]
    if drop_minutes is not None:
        lo, hi = drop_minutes
        keep = [not (lo <= i < hi) for i in range(len(ts))]
        ts = ts[keep]
    n = len(ts)
    close = 100 + pd.Series(range(n), dtype=float) * 0.01
    return pd.DataFrame({"timestamp": ts, "open": close - 0.005, "close": close,
                         "high": close + 0.01, "low": close - 0.01, "volume": 100.0})


@pytest.mark.parametrize("grid", ["1m", "5m", "15m"])
@pytest.mark.parametrize("day", [FULL_DAY, HALF_DAY])
def test_return_count_matches_resampling(day, grid):
    bars = _bars(day)
    assert _n_grid_returns(bars, grid) == len(log_returns(resample_day(bars, grid)))


def test_half_day_is_dropped_from_every_grid_with_its_reason():
    """Early closes are dropped outright: a 15m half day gives only 13
    returns, and keeping it on 1m/5m but not 15m would give the A1 levels
    different (and systematically calmer) day sets."""
    cleaned, reason = clean_day_with_reason(_bars(HALF_DAY), HALF_DAY)
    assert cleaned is None
    assert reason == "half_day"
    assert clean_day(_bars(HALF_DAY), HALF_DAY) is None


def test_full_day_is_kept():
    cleaned, reason = clean_day_with_reason(_bars(FULL_DAY), FULL_DAY)
    assert cleaned is not None and reason is None


def test_thin_day_is_dropped_with_the_coverage_reason():
    """Half the 1m bars missing: a data hole, not a quiet day."""
    _, reason = clean_day_with_reason(_bars(FULL_DAY, keep_every=2), FULL_DAY)
    assert reason == "insufficient_1m_coverage"


def test_day_too_thin_on_one_grid_leaves_every_grid(monkeypatch):
    """If ANY grid falls below the absolute floor the day leaves ALL grids.
    With the real floor (20) and the 80% 1m rule a full 15m day always
    keeps >= 21 returns, so the floor is raised here to exercise the path:
    a 45-minute hole leaves 1m at 344 returns but 15m at 23 < 25."""
    import projekat.data.clean as clean_mod

    monkeypatch.setattr(clean_mod, "MIN_INTRADAY_OBS", 25)
    bars = _bars(FULL_DAY, drop_minutes=(100, 145))
    assert _n_grid_returns(bars, "1m") >= 25
    assert _n_grid_returns(bars, "15m") < 25
    _, reason = clean_day_with_reason(bars, FULL_DAY)
    assert reason == "below_min_intraday_obs"


def test_grids_share_one_day_set_and_every_exclusion_has_a_reason(monkeypatch):
    import projekat.data.clean as clean_mod

    monkeypatch.setattr(clean_mod, "MIN_INTRADAY_OBS", 25)
    raw_by_day = {
        FULL_DAY: _bars(FULL_DAY),
        FULL_DAY_2: _bars(FULL_DAY_2, drop_minutes=(100, 145)),   # 15m too thin -> dropped on every grid
        HALF_DAY: _bars(HALF_DAY),
    }
    returns, _flags, exclusions = returns_by_grid_from_raw(
        "AAA", raw_by_day, ["1m", "5m", "15m"], start=FULL_DAY, end=FULL_DAY_2
    )
    day_sets = {g: set(r) for g, r in returns.items()}
    assert day_sets["1m"] == day_sets["5m"] == day_sets["15m"] == {FULL_DAY}

    reasons = {e["date"]: e["reason"] for e in exclusions}
    assert reasons[HALF_DAY] == "half_day"
    assert reasons[FULL_DAY_2] == "below_min_intraday_obs"
    assert set(reasons.values()) <= set(EXCLUSION_REASONS) | {"no_vendor_bars", "fewer_than_2_returns"}


def test_missing_vendor_sessions_are_counted():
    """A session inside the vendor's own date range with no bars at all is
    an exclusion too, recorded as no_vendor_bars."""
    raw_by_day = {FULL_DAY: _bars(FULL_DAY), "2015-03-12": _bars("2015-03-12")}
    _returns, _flags, exclusions = returns_by_grid_from_raw(
        "AAA", raw_by_day, ["5m"], start="2015-03-01", end="2015-03-31"
    )
    assert {"date": FULL_DAY_2, "reason": "no_vendor_bars", "symbol": "AAA"} in exclusions
