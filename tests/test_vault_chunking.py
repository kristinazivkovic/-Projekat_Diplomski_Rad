"""The vault's /export silently truncates at ~2.5M rows.

This is a silent-corruption bug, not a crash: the parquet is valid, the
manifest is correct, and the data simply stops years early. Measured on
the real vault -- AAPL 2010-2025 came back as exactly 2,500,000 rows
ending 2023-07-06, losing 2.5 years of the test period, while INTC at
2,398,142 rows came back complete.

These tests are synthetic and offline; they never touch the vault.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from projekat.data.vault import EXPORT_ROW_CAP, _chunk_ranges, is_truncated


def _write_bars(path, *, n_rows, last_day):
    """A parquet shaped like a vault export, ending at `last_day`."""
    ts = pd.date_range(end=pd.Timestamp(last_day, tz="UTC"), periods=n_rows, freq="min")
    pd.DataFrame(
        {
            "timestamp": ts,
            "open": np.ones(n_rows), "high": np.ones(n_rows),
            "low": np.ones(n_rows), "close": np.ones(n_rows),
            "volume": np.ones(n_rows),
        }
    ).to_parquet(path, index=False)
    return path


# -- chunk planning -------------------------------------------------------


def test_long_range_is_split_into_chunks_under_the_cap():
    chunks = _chunk_ranges("2010-01-01", "2025-12-31")
    assert len(chunks) == 2
    # contiguous and covering
    assert chunks[0][0] == "2010-01-01"
    assert chunks[-1][1] == "2025-12-31"
    assert chunks[0][1] == chunks[1][0]


def test_short_ranges_stay_a_single_chunk():
    """The existing mini-panel path must not start chunking."""
    assert _chunk_ranges("2015-01-01", "2016-01-01") == [("2015-01-01", "2016-01-01")]
    assert len(_chunk_ranges("2010-01-01", "2015-01-01")) == 1


def test_chunks_are_small_enough_to_clear_the_cap():
    """A 16-year 1m history is ~2.0-2.5M rows; each chunk must leave real
    headroom under the cap rather than sitting just below it."""
    chunks = _chunk_ranges("2010-01-01", "2025-12-31")
    rows_per_chunk = 2_500_000 / len(chunks)
    assert rows_per_chunk < EXPORT_ROW_CAP / 1.5


# -- truncation detection -------------------------------------------------


def test_truncated_export_is_detected(tmp_path):
    """At the cap AND ending years early -> truncated."""
    path = _write_bars(tmp_path / "AAPL.parquet", n_rows=EXPORT_ROW_CAP, last_day="2023-07-06")
    assert is_truncated(path, expected_end="2025-12-31")


def test_complete_export_under_the_cap_is_not_flagged(tmp_path):
    """INTC's real shape: under the cap, runs to the end."""
    path = _write_bars(tmp_path / "INTC.parquet", n_rows=2_398_142, last_day="2025-12-30")
    assert not is_truncated(path, expected_end="2025-12-31")


def test_a_symbol_at_the_cap_that_still_reaches_the_end_is_not_flagged(tmp_path):
    """Hitting the cap is not by itself proof of truncation -- the data
    must also stop early. Flagging on row count alone would force
    pointless re-fetches of complete symbols."""
    path = _write_bars(tmp_path / "X.parquet", n_rows=EXPORT_ROW_CAP, last_day="2025-12-30")
    assert not is_truncated(path, expected_end="2025-12-31")


def test_small_file_is_never_flagged(tmp_path):
    path = _write_bars(tmp_path / "small.parquet", n_rows=1000, last_day="2015-03-31")
    assert not is_truncated(path, expected_end="2025-12-31")


# -- the fallback must refuse truncated data ------------------------------


def test_load_refuses_a_truncated_whole_range_file(tmp_path, monkeypatch):
    """The whole-range fallback exists for older cache entries, but it must
    never hand back silently truncated data -- that is precisely how a
    symbol vanishes from the late windows with no error."""
    import projekat.data.vault as vault

    raw = tmp_path / "1m"
    raw.mkdir(parents=True)
    monkeypatch.setattr(vault, "RAW_DIR", tmp_path)
    _write_bars(raw / "AAPL_1m_2010-01-01_2025-12-31.parquet",
                n_rows=EXPORT_ROW_CAP, last_day="2023-07-06")

    with pytest.raises(ValueError, match="odsečen|truncat"):
        vault.load_1m_range("AAPL", "2010-01-01", "2025-12-31")


def test_load_stitches_chunks_and_drops_boundary_duplicates(tmp_path, monkeypatch):
    import projekat.data.vault as vault

    raw = tmp_path / "1m"
    raw.mkdir(parents=True)
    monkeypatch.setattr(vault, "RAW_DIR", tmp_path)
    for (cs, ce), last in zip(_chunk_ranges("2010-01-01", "2025-12-31"),
                              ("2017-12-29", "2025-12-30")):
        _write_bars(raw / f"AAPL_1m_{cs}_{ce}.parquet", n_rows=500, last_day=last)

    out = vault.load_1m_range("AAPL", "2010-01-01", "2025-12-31")
    assert len(out) == 1000
    ts = out["timestamp"]
    assert ts.is_monotonic_increasing
    assert not ts.duplicated().any()
