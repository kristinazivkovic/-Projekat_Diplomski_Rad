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
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from projekat.config import END_DATE, RAW_DIR, START_DATE, UNIVERSE  # noqa: E402
from projekat.data.vault import (  # noqa: E402
    EXPORT_ROW_CAP,
    _chunk_ranges,
    _client,
    fetch_1m,
    is_truncated,
    load_manifest,
)


class QuotaGate:
    """Global gate in front of POST /export, shared by all workers.

    The quota limits job CREATION only, so this is the one serialised step:
    creations are spaced by `pace`, and a 429 seen by ANY worker pauses
    creation for EVERY worker (escalating backoff). Workers whose job already
    exists keep polling and downloading meanwhile -- that overlap is where the
    parallel speed-up comes from."""

    def __init__(self, pace: float, rl_wait: float, max_wait: float):
        self._pace, self._rl_wait, self._max_wait = pace, rl_wait, max_wait
        self._create_lock = threading.Lock()
        self._state = threading.Lock()
        self._next_ok = 0.0
        self._strikes = 0

    def before_create(self) -> None:
        with self._create_lock:
            while True:
                with self._state:
                    wait = self._next_ok - time.monotonic()
                    if wait <= 0:
                        self._next_ok = time.monotonic() + self._pace
                        return
                time.sleep(min(wait, 5.0))

    def rate_limited(self) -> float:
        with self._state:
            self._strikes += 1
            wait = min(self._rl_wait * (2 ** (self._strikes - 1)), self._max_wait)
            self._next_ok = max(self._next_ok, time.monotonic() + wait)
            return wait

    def succeeded(self) -> None:
        with self._state:
            self._strikes = 0


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
    parser.add_argument("--workers", type=int, default=6,
                        help="chunks processed in parallel (jobs build and download concurrently)")
    parser.add_argument("--pace", type=float, default=20.0,
                        help="minimum seconds between creating two export jobs (global, all workers)")
    parser.add_argument("--rate-limit-wait", type=float, default=600.0,
                        help="first backoff after a 429, in seconds (doubles each time)")
    # 20 waits = 10+20+40 min, then 17 x 60 min ~= 18h: enough to leave the
    # pull running overnight without it giving up on the quota.
    parser.add_argument("--rate-limit-retries", type=int, default=20,
                        help="how many times to wait out a 429 before giving up on a chunk")
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

    tasks = [(sym, cs, ce) for sym in todo for (cs, ce) in _chunk_ranges(args.start, args.end)]
    print(f"\nfetching {len(todo)} symbol(s) = {len(tasks)} chunk(s) with {args.workers} worker(s); "
          f"cached chunks are skipped, so re-running is cheap.\n", flush=True)
    started = time.monotonic()
    gate = QuotaGate(args.pace, args.rate_limit_wait, args.max_wait)
    local = threading.local()
    counter = {"done": 0}
    counter_lock = threading.Lock()

    def run_chunk(symbol: str, cs: str, ce: str) -> None:
        if getattr(local, "client", None) is None:
            local.client = _client()
        attempt = rl_attempt = 0
        tag = f"{symbol:6s} {cs}..{ce}"
        while True:
            t0 = time.monotonic()
            try:
                path = fetch_1m(symbol, cs, ce, force=args.force, client=local.client,
                                before_create=gate.before_create)
                gate.succeeded()
                mf = load_manifest(symbol, cs, ce)
                rows = mf.row_count if mf else 0
                with counter_lock:
                    counter["done"] += 1
                    done = counter["done"]
                elapsed = time.monotonic() - started
                eta = (len(tasks) - done) * elapsed / done
                warn = "  !! AT EXPORT CAP -- chunk may be truncated" if rows >= EXPORT_ROW_CAP else ""
                print(f"[{done}/{len(tasks)}] {tag} OK  {rows:>10,} rows  "
                      f"{path.stat().st_size / 1e6:7.1f} MB  in {_fmt(time.monotonic() - t0)}  "
                      f"| elapsed {_fmt(elapsed)} | eta {_fmt(eta)}{warn}", flush=True)
                return
            except Exception as exc:
                if _is_fatal(exc):
                    raise RuntimeError(f"FATAL (will never succeed): {exc}") from exc
                if _is_rate_limit(exc):
                    rl_attempt += 1
                    if rl_attempt > args.rate_limit_retries:
                        raise RuntimeError("rate limited; quota did not clear") from exc
                    wait = gate.rate_limited()
                    print(f"{tag} RATE LIMITED -- all workers pause new exports for {_fmt(wait)} "
                          f"(jobs already created keep building/downloading)",
                          file=sys.stderr, flush=True)
                    continue  # before_create blocks until the window has passed
                attempt += 1
                if attempt > args.retries:
                    raise
                print(f"{tag} attempt {attempt}/{args.retries} failed: "
                      f"{type(exc).__name__}: {str(exc)[:160]}", file=sys.stderr, flush=True)
                time.sleep(args.retry_wait)

    ok_chunks, failed = set(), []
    pool = ThreadPoolExecutor(max_workers=args.workers)
    futures = {pool.submit(run_chunk, *t): t for t in tasks}
    try:
        for fut in as_completed(futures):
            symbol, cs, ce = futures[fut]
            try:
                fut.result()
                ok_chunks.add((symbol, cs, ce))
            except Exception as exc:
                failed.append((f"{symbol} {cs}..{ce}", f"{type(exc).__name__}: {exc}"))
                print(f"{symbol:6s} {cs}..{ce} FAILED: {str(exc)[:200]}", file=sys.stderr, flush=True)
    except KeyboardInterrupt:
        # Export job ids are on disk (data/raw/1m/.jobs), so nothing is lost:
        # a re-run resumes waiting/downloading instead of creating new jobs.
        print("\n[SIGINT] stopping. Finished chunks and in-flight export job ids are kept; "
              "re-run to continue.", flush=True)
        os._exit(130)
    pool.shutdown()
    ok = sorted(s_ for s_ in todo if all((s_, cs, ce) in ok_chunks
                                         for cs, ce in _chunk_ranges(args.start, args.end)))

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
