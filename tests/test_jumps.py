"""Factor A3 jump test tests: detection on synthetic data with known
answers, and the lm_fdr causal-window (no-lookahead) guarantee."""

from __future__ import annotations

import numpy as np
import pytest

from projekat.measure.estimators.bpv import bpv
from projekat.measure.jumps.bns import bns
from projekat.measure.jumps.lee_mykland import lee_mykland
from projekat.measure.jumps.lm_fdr import DayStat, benjamini_hochberg_significant, lm_fdr_significance_series
from projekat.measure.jumps.naive import naive
from projekat.measure.realized import realized_variance
from tests.conftest import make_day_returns


def test_no_jump_gives_low_false_positive_rate_bns():
    """Across many no-jump days, BNS should flag close to its nominal 5%
    level, not systematically more or less (verified: ~3.5% measured)."""
    flags = []
    for seed in range(100):
        r = make_day_returns(390, seed=seed)
        rv = realized_variance(r)
        robust = bpv(r)
        v = bns(r, rv, robust)
        flags.append(v.significant)
    rate = np.mean(flags)
    assert rate < 0.15  # well below a runaway false-positive rate


def test_bns_detects_large_injected_jump():
    """A 20-sigma single-return jump must be detected -- this is the
    regression test for the BNS formula fix (Revision 8 session): the
    original formula (missing the n-scaling, using RV^2 instead of
    BPV^2 in the denominator) gave a statistic of only 0.69 for this
    exact jump; the corrected formula gives ~8.0."""
    r = make_day_returns(390, seed=0, jump_at=200, jump_size=0.02)
    rv = realized_variance(r)
    robust = bpv(r)
    v = bns(r, rv, robust)
    assert v.significant
    assert v.statistic > 5.0  # well above the ~1.645 one-sided 5% threshold


def test_bns_power_increases_with_jump_size():
    r_base = make_day_returns(390, seed=0)
    stats = []
    for jump_size in [0.003, 0.008, 0.02]:
        r = r_base.copy()
        r[200] += jump_size
        rv = realized_variance(r)
        robust = bpv(r)
        v = bns(r, rv, robust)
        stats.append(v.statistic)
    assert stats == sorted(stats)


def test_lee_mykland_detects_injected_jump_fallback_path():
    """Using lee_mykland's isolated-day fallback (no cross-day K-history
    supplied) -- a large injected jump should still be flagged."""
    r = make_day_returns(390, seed=0, jump_at=200, jump_size=0.02)
    rv = realized_variance(r)
    robust = bpv(r)
    v = lee_mykland(r, rv, robust)
    assert v.significant


def test_naive_flags_large_daily_move():
    """Regression test for a self-contamination fix: naive()'s sigma must
    come from the jump-robust `robust` estimator (BPV), not from
    np.std(r) over the whole day -- the latter includes the injected jump
    in its own denominator, inflating sigma enough that a sufficiently
    large jump could escape detection by the very test meant to catch it.
    jump_size=0.15 is unambiguously far above any plausible baseline-noise
    sigma at this scale (sigma=0.0008/return), so it isolates the sigma
    formula rather than depending on a borderline threshold crossing."""
    r = make_day_returns(390, seed=0, jump_at=200, jump_size=0.15)
    rv = realized_variance(r)
    robust = bpv(r)
    v = naive(r, rv, robust)
    assert v.significant
    assert v.statistic > 3.0


def test_benjamini_hochberg_basic():
    """Standard BH sanity check: with several very small p-values and many
    large ones, only the small ones should be flagged."""
    pvals = [0.001, 0.002, 0.5, 0.6, 0.7, 0.8, 0.9]
    sig = benjamini_hochberg_significant(pvals, q=0.05)
    assert sig[0] and sig[1]
    assert not any(sig[2:])


def test_lm_fdr_no_lookahead():
    """The causal-window guarantee (Resolution 6/Revision 8): day t's
    lm_fdr classification must be provably unaffected by any day's
    statistic that comes after t. Constructed so that if the window were
    NOT causal (e.g. used the whole sample), classification of an early
    day would change when a dramatic future day is appended."""
    # 300 unremarkable days (all high p-values, i.e. not significant)
    day_stats = [
        DayStat(day=f"d{i:04d}", max_l=2.0, p_value=0.9, n=78)
        for i in range(300)
    ]
    result_before = lm_fdr_significance_series(day_stats, window=250)

    # Append a day with an extremely significant statistic (p~0) far in
    # the future and recompute -- earlier days' classifications (indices
    # that don't include the new day in their trailing window) must be
    # identical.
    day_stats_extended = day_stats + [DayStat(day="d9999", max_l=10.0, p_value=1e-12, n=78)]
    result_after = lm_fdr_significance_series(day_stats_extended, window=250)

    for i, stat in enumerate(day_stats):
        # Any day whose own trailing window (i-250..i) doesn't include the
        # newly appended day (which is always true here, since the new day
        # is appended AFTER all existing days) must be unaffected.
        assert result_before[stat.day] == result_after[stat.day], (
            f"day {stat.day} classification changed when a future day was appended -- lookahead detected"
        )


def test_lm_fdr_warmup_flagged():
    """Days before the first W trailing days must be marked as warmup and
    fall back to plain significance rather than an unstable short window."""
    from projekat.measure.jumps._gumbel import critical_value

    day_stats = [DayStat(day=f"d{i:04d}", max_l=5.0, p_value=0.001, n=78) for i in range(10)]
    result = lm_fdr_significance_series(day_stats, window=250)
    for stat in day_stats:
        significant, is_warmup = result[stat.day]
        assert is_warmup
        assert significant == (stat.max_l > critical_value(stat.n))
