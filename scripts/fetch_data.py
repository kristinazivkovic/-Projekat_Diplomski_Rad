"""Resumable pre-step: download 1m bars for the whole universe.

    # check what is missing, download nothing
    uv run python scripts/fetch_data.py --dry-run

    # prove the vault path works before committing hours
    uv run python scripts/fetch_data.py --symbols NVDA,AAPL

    # the real pull
    uv run python scripts/fetch_data.py

WHY THIS IS A SEPARATE SCRIPT
=============================
`build_panel.py` loops symbols with NO error handling, so one network or
vault failure at symbol 27 of 30 aborts the whole build. Fetching is also
slow (each symbol is a server-side export job, capped at 30 min), purely
network-bound, and completely idempotent -- so it belongs outside the
panel build, run once, with retries.

`fetch_1m` caches on the literal (symbol, start, end) triple baked into
the filename, so re-running is free for symbols already on disk, and a
crash costs only the symbol in flight.

EXPORT QUOTA (429) -- THE MAIN OBSTACLE
=======================================
The vault rate-limits POST /export ("too many export requests"), i.e.
job CREATION, not the download. Measured on this account: roughly four
full-history symbols succeed, then it trips, and it does NOT clear within
10 minutes -- it is a quota window, not a brief throttle.

Two consequences shape this script:
  * `--pace` puts a gap between symbols so the quota is less likely to
    trip at all. Avoiding a 429 is far cheaper than recovering from one.
  * A 429 is retried with ESCALATING backoff (10 min, then 20, 40, ...),
    not the short retry used for ordinary errors. Hammering the endpoint
    does not shorten the window.

Because every completed symbol is cached, Ctrl-C and re-running an hour
later is always a valid strategy -- and for a quota this long it is often
the fastest one. The run simply picks up where it stopped.

THE DATE RANGE IS PART OF THE CACHE KEY
=======================================
`build_panel.py::_load_raw_bars` calls `fetch_1m(symbol, start, end)` with
exactly the dates passed on ITS command line. If you fetch with one range
and build the panel with another, nothing is reused and every symbol is
downloaded a second time. Both default to config.START_DATE/END_DATE here
so they line up unless you deliberately override them.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from projekat.config import END_DATE, RAW_DIR, START_DATE, UNIVERSE  # noqa: E402
from projekat.data.vault import (  # noqa: E402
    _chunk_ranges,
    fetch_1m_chunked,
    is_truncated,
    load_manifest,
)


def _already_cached(symbol: str, start: str, end: str) -> bool:
    """A symbol counts as done only if EVERY chunk of its range has both a
    parquet and a manifest -- the same condition fetch_1m uses to skip."""
    for chunk_start, chunk_end in _chunk_ranges(start, end):
        path = RAW_DIR / "1m" / f"{symbol}_1m_{chunk_start}_{chunk_end}.parquet"
        if not path.exists() or load_manifest(symbol, chunk_start, chunk_end) is None:
            return False
    return True


def _truncated_whole_range_files(symbols: list[str], start: str, end: str) -> list[tuple[str, Path]]:
    """Whole-range files from before chunking that the vault silently cut
    off at its ~2.5M row export cap.

    These are the dangerous ones: a valid parquet with a correct manifest
    whose data simply stops years early. Left in place they would drop a
    symbol out of the late windows with no error anywhere."""
    out = []
    for symbol in symbols:
        path = RAW_DIR / "1m" / f"{symbol}_1m_{start}_{end}.parquet"
        if path.exists() and is_truncated(path, expected_end=end):
            out.append((symbol, path))
    return out


def _fmt(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else (f"{m}m{s:02d}s" if m else f"{s}s")


def _is_rate_limit(exc: BaseException) -> bool:
    """The vault returns 429 'too many export requests' from POST /export --
    job CREATION, not the download. Measured behaviour: after ~4 large
    symbols it trips and does NOT clear within 10 minutes, so this is a
    real quota window rather than a short throttle. It needs patient
    backoff (tens of minutes), not fast retries, which is why it is
    detected separately from ordinary errors."""
    text = str(exc)
    return "429" in text or "too many export requests" in text.lower()


def _is_fatal(exc: BaseException) -> bool:
    """Errors that will never succeed on retry -- a bad ticker stays bad.
    Retrying these just burns quota that working symbols need."""
    text = str(exc).lower()
    return "not in the vault catalog" in text or "pass dataset=" in text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbols", default="all", help="comma-separated, or 'all' (config.UNIVERSE)")
    parser.add_argument("--start", default=START_DATE)
    parser.add_argument("--end", default=END_DATE)
    parser.add_argument("--retries", type=int, default=3, help="attempts per symbol (ordinary errors)")
    parser.add_argument("--retry-wait", type=float, default=30.0, help="seconds between ordinary attempts")
    parser.add_argument("--pace", type=float, default=60.0,
                        help="seconds to wait between symbols, to avoid tripping the export quota")
    parser.add_argument("--rate-limit-wait", type=float, default=600.0,
                        help="first backoff after a 429, in seconds (doubles each time)")
    parser.add_argument("--rate-limit-retries", type=int, default=6,
                        help="how many times to wait out a 429 before giving up on a symbol")
    parser.add_argument("--max-wait", type=float, default=3600.0,
                        help="cap on any single backoff, in seconds")
    parser.add_argument("--force", action="store_true", help="re-download even if cached")
    parser.add_argument("--dry-run", action="store_true", help="report what is missing, download nothing")
    args = parser.parse_args()

    symbols = UNIVERSE if args.symbols == "all" else [s.strip() for s in args.symbols.split(",") if s.strip()]

    cached = [s for s in symbols if _already_cached(s, args.start, args.end)]
    missing = [s for s in symbols if s not in cached]

    chunks = _chunk_ranges(args.start, args.end)
    print(f"range   : {args.start} -> {args.end}  ({len(chunks)} chunk(s) per symbol)")
    print(f"symbols : {len(symbols)}")
    print(f"cached  : {len(cached)}  {cached if len(cached) <= 10 else cached[:10] + ['...']}")
    print(f"missing : {len(missing)}  {missing if len(missing) <= 10 else missing[:10] + ['...']}")

    # Whole-range files fetched before chunking may be silently truncated at
    # the vault's export cap. They look perfectly valid, so they must be
    # named loudly -- otherwise the affected symbol just disappears from the
    # late windows with no error.
    stale = _truncated_whole_range_files(symbols, args.start, args.end)
    if stale:
        print(f"\nWARNING: {len(stale)} whole-range file(s) are TRUNCATED at the vault's "
              f"~2.5M-row export cap:", file=sys.stderr)
        for symbol, path in stale:
            print(f"  {symbol}: {path.name}", file=sys.stderr)
        print("  These will be re-fetched in chunks. Delete the old files once the "
              "chunked fetch succeeds (they are ignored in favour of chunks).",
              file=sys.stderr)

    todo = symbols if args.force else missing
    if not todo:
        print("\nnothing to fetch; every symbol is cached.")
        return 0

    if args.dry_run:
        print(f"\n--dry-run: would fetch {len(todo)} symbol(s). Nothing downloaded.")
        print("Each symbol is a server-side export job (up to 30 min); budget hours, not minutes.")
        return 0

    print(f"\nfetching {len(todo)} symbol(s); cached symbols are skipped, so re-running is cheap.\n")
    started = time.monotonic()
    ok, failed = [], []

    for i, symbol in enumerate(todo, start=1):
        # Pace between symbols so the export quota is not tripped in the
        # first place. Cheaper than recovering from a 429, which costs tens
        # of minutes of waiting.
        if i > 1 and args.pace > 0:
            time.sleep(args.pace)

        attempt = 0
        while True:
            attempt += 1
            t0 = time.monotonic()
            try:
                paths = fetch_1m_chunked(symbol, args.start, args.end, force=args.force)
                rows = 0
                for (cs, ce) in _chunk_ranges(args.start, args.end):
                    mf = load_manifest(symbol, cs, ce)
                    rows += mf.row_count if mf else 0
                size_mb = sum(p.stat().st_size for p in paths) / 1e6
                elapsed = time.monotonic() - t0
                done = time.monotonic() - started
                rate = i / done if done > 0 else 0
                eta = (len(todo) - i) / rate if rate > 0 else 0
                print(
                    f"[{i}/{len(todo)}] {symbol:6s} OK  {rows:>10,} rows  "
                    f"{size_mb:7.1f} MB  {len(paths)} chunk(s)  in {_fmt(elapsed)}  "
                    f"| elapsed {_fmt(done)} | eta {_fmt(eta)}",
                    flush=True,
                )
                ok.append(symbol)
                break
            except KeyboardInterrupt:
                print(f"\n[SIGINT] stopped during {symbol}; cached symbols are kept. "
                      f"Re-run to continue.", flush=True)
                return 130
            except Exception as exc:
                if _is_fatal(exc):
                    print(f"[{i}/{len(todo)}] {symbol:6s} FATAL (will never succeed): "
                          f"{str(exc)[:160]}", file=sys.stderr, flush=True)
                    failed.append((symbol, f"{type(exc).__name__}: {exc}"))
                    break

                if _is_rate_limit(exc):
                    if attempt > args.rate_limit_retries:
                        failed.append((symbol, "rate limited; quota did not clear"))
                        print(f"[{i}/{len(todo)}] {symbol:6s} giving up after "
                              f"{args.rate_limit_retries} rate-limit waits",
                              file=sys.stderr, flush=True)
                        break
                    # escalating backoff: the quota window is long (measured
                    # >10 min), so waits grow rather than hammering the API
                    wait = min(args.rate_limit_wait * (2 ** (attempt - 1)), args.max_wait)
                    print(
                        f"[{i}/{len(todo)}] {symbol:6s} RATE LIMITED "
                        f"(wait {attempt}/{args.rate_limit_retries}) -- sleeping {_fmt(wait)}. "
                        f"Cached symbols are safe; Ctrl-C and re-run later is equally fine.",
                        file=sys.stderr, flush=True,
                    )
                    try:
                        time.sleep(wait)
                    except KeyboardInterrupt:
                        print("\n[SIGINT] stopped while waiting out the rate limit; "
                              "re-run to continue.", flush=True)
                        return 130
                    continue

                if attempt > args.retries:
                    failed.append((symbol, f"{type(exc).__name__}: {exc}"))
                    break
                print(
                    f"[{i}/{len(todo)}] {symbol:6s} attempt {attempt}/{args.retries} failed: "
                    f"{type(exc).__name__}: {str(exc)[:160]}",
                    file=sys.stderr, flush=True,
                )
                time.sleep(args.retry_wait)

    print()
    print(f"fetched : {len(ok)}")
    print(f"failed  : {len(failed)}")
    for symbol, err in failed:
        print(f"  {symbol}: {err[:200]}", file=sys.stderr)
    print(f"total   : {_fmt(time.monotonic() - started)}")

    if failed:
        rate_limited = [s for s, err in failed if "rate limited" in err]
        if rate_limited:
            print(f"\n{len(rate_limited)} symbol(s) hit the export quota. This is expected on a "
                  f"large pull -- wait a while (an hour is usually plenty) and re-run the same "
                  f"command; cached symbols are skipped automatically.", file=sys.stderr)
        print("\nRe-run the same command to retry only the failed symbols "
              "(successful ones are cached and skipped).", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
