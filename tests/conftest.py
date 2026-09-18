"""Shared synthetic-data fixtures. No network access, no LSE/FRED calls --
everything here is constructed with a fixed seed so tests are deterministic
and fast."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest


def make_day_returns(n: int, *, sigma: float = 0.0008, seed: int = 0, jump_at: int | None = None, jump_size: float = 0.02) -> np.ndarray:
    rng = np.random.default_rng(seed)
    r = rng.normal(0, sigma, n)
    if jump_at is not None:
        r[jump_at] += jump_size
    return r


@pytest.fixture
def synthetic_no_jump_days():
    """30 days of 1m (n=390) returns, no jumps, deterministic seeds."""
    d0 = dt.date(2020, 1, 2)
    out = {}
    for i in range(30):
        day = (d0 + dt.timedelta(days=i)).isoformat()
        out[day] = make_day_returns(390, seed=100 + i)
    return out


@pytest.fixture
def synthetic_one_jump_day():
    """30 days of 1m returns with one large jump injected on a known day
    (index 15) -- used to verify jump tests localize to the exact day."""
    d0 = dt.date(2020, 1, 2)
    out = {}
    jump_day = None
    for i in range(30):
        day = (d0 + dt.timedelta(days=i)).isoformat()
        if i == 15:
            out[day] = make_day_returns(390, seed=100 + i, jump_at=200, jump_size=0.02)
            jump_day = day
        else:
            out[day] = make_day_returns(390, seed=100 + i)
    return out, jump_day
