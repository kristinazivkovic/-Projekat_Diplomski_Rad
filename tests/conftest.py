"""Shared synthetic-data fixtures. No network access, no LSE/FRED calls --
everything here is constructed with a fixed seed so tests are deterministic
and fast."""

from __future__ import annotations

import datetime as dt
import os

# torch models run on the CPU in tests: deterministic, and no contention
# with a GPU run that may be going on at the same time
os.environ.setdefault("PROJEKAT_DEVICE", "cpu")

import numpy as np  # noqa: E402
import pytest  # noqa: E402


def make_day_returns(n: int, *, sigma: float = 0.0008, seed: int = 0, jump_at: int | None = None, jump_size: float = 0.02) -> np.ndarray:
    rng = np.random.default_rng(seed)
    r = rng.normal(0, sigma, n)
    if jump_at is not None:
        r[jump_at] += jump_size
    return r


def make_panel(
    *,
    symbols=("AAA", "BBB"),
    design_ids=("5m__bpv__bns",),
    start: str = "2011-01-01",
    end: str = "2015-03-31",
    seed: int = 0,
):
    """Synthetic panel in the frozen schema's shape on REAL XNYS sessions:
    log-RV follows an AR(1), C/J come from a consistent conditional
    decomposition (J > 0 only on flagged days), HAR aggregates use the
    measurement layer's own rolling rule. RV is the same for every design of
    a grid (as in the real panel); J differs by design."""
    import pandas as pd

    from projekat.data.calendar import sessions_between
    from projekat.measure.rolling import add_har_aggregates

    sessions = pd.DatetimeIndex(sessions_between(start, end))
    if sessions.tz is not None:
        sessions = sessions.tz_localize(None)
    n = len(sessions)
    frames = []
    for s_i, symbol in enumerate(symbols):
        rng = np.random.default_rng(seed + 101 * s_i)
        x = np.empty(n)
        x[0] = -9.5
        for t in range(1, n):
            x[t] = -9.5 + 0.8 * (x[t - 1] + 9.5) + rng.normal(0, 0.35)
        rv = np.exp(x)
        base = {
            "date": sessions, "symbol": symbol, "RV_d": rv,
            "RS_pos": rv * 0.5, "RS_neg": rv * 0.5, "SJ": 0.0,
            "RQ": rv**2 * rng.uniform(1.0, 3.0, n), "r_d": rng.normal(0, np.sqrt(rv)),
            "vix_lag1": 18 + rng.normal(0, 2, n), "n_obs": 78, "is_half_day": False, "fdr_warmup": False,
        }
        for d_i, design_id in enumerate(design_ids):
            drng = np.random.default_rng(seed + 101 * s_i + 7 * d_i + 1)
            flag = drng.random(n) < 0.08
            # IV_d (ocena ocenjivačem otpornim na skokove) je izvor iz kog
            # sloj merenja izvodi J i C: J = 1{flag} * max(RV - IV, 0),
            # C = RV - J (measure/decompose.py). Fikstura ga gradi istim
            # redosledom -- prvo IV, pa J iz njega -- da bi permutacioni
            # testovi (H5 placebo) mogli da ponove tu istu izvedbu.
            iv = rv * (1.0 - drng.uniform(0.1, 0.4, n))
            j = np.where(flag, np.maximum(rv - iv, 0.0), 0.0)
            df = pd.DataFrame(base)
            df["design_id"] = design_id
            df["jump_flag"] = flag
            df["IV_d"] = iv
            df["J_d"] = j
            df["C_d"] = rv - j
            for col in ("RV", "C", "J"):
                df = add_har_aggregates(df, col)
            frames.append(df)
    return pd.concat(frames, ignore_index=True)


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
