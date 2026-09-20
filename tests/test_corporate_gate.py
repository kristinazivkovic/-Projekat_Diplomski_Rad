"""verify_split_adjustment on the three real cases from the first 30-symbol
build: two small spin-offs that were false positives, one genuinely
unadjusted rights issue (whose data is also corrupted intraday for two
years before it, 2014-03..2016-03 -- MT must stay out)."""

from __future__ import annotations

import math

import pytest

from projekat.data.corporate import SplitEvent, UnadjustedDataError, verify_split_adjustment


def _closes(day_before: str, day: str, logret: float) -> dict[str, float]:
    return {day_before: 100.0, day: 100.0 * math.exp(logret)}


@pytest.mark.parametrize("symbol, day_before, day, frm, to, logret", [
    ("PFE", "2020-11-16", "2020-11-17", 500, 527, 0.013),        # Viatris spin-off
    ("VZ", "2010-07-01", "2010-07-02", 937889, 1000000, 0.009),  # Frontier spin-off
])
def test_small_spin_off_on_an_ordinary_day_is_not_flagged(symbol, day_before, day, frm, to, logret):
    verify_split_adjustment(_closes(day_before, day, logret), [SplitEvent(symbol, day, frm, to)])


def test_rights_issue_matching_the_unadjusted_drop_is_flagged():
    with pytest.raises(UnadjustedDataError):
        verify_split_adjustment(_closes("2016-03-11", "2016-03-14", -0.221),
                                [SplitEvent("MT", "2016-03-14", 500, 629)])


def test_large_split_still_caught_when_unadjusted_and_passes_when_adjusted():
    split = [SplitEvent("AAPL", "2014-06-09", 1, 7)]
    with pytest.raises(UnadjustedDataError):
        verify_split_adjustment(_closes("2014-06-06", "2014-06-09", -math.log(7) + 0.01), split)
    verify_split_adjustment(_closes("2014-06-06", "2014-06-09", 0.016), split)
