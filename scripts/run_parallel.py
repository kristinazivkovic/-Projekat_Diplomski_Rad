"""Run scripts/run_econometric.py as N parallel shards into ONE run dir.

    uv run python scripts/run_parallel.py --panel data/processed/panel.parquet --models all
    uv run python scripts/run_parallel.py --panel data/processed/panel.parquet --models patchtst --shards 18 --gpu-shards 4

Everything after the known options is passed through to run_econometric.py
unchanged (--models, --horizons, --grids, --symbols, ...). One combination
is one walk-forward pass (model x design x seed), so the shards stride over
combinations of very different length (ARFIMA, PatchTST are the long ones).

WHY PROCESSES, ONE THREAD EACH
==============================
Measured on this machine (i7, 14 cores / 20 threads, RTX 4070) with a
PatchTST fit: more torch threads do NOT make one fit faster (1 thread 16.6s,
10 threads 17.2s) -- the model is so small that each training step (~4 ms)
is dominated by per-op overhead, not arithmetic. The GPU has the same
per-step cost for the same reason. Throughput therefore comes from running
many fits at once:

    1 process                  17   s/fit
    12 processes, CPU          2.7 s/fit
    16 processes, CPU          2.3 s/fit
    20 processes, CPU          2.2 s/fit
    4 GPU + 16 CPU processes   2.2 s/fit

So the default is 18 shards, 4 of which put PatchTST on the GPU (it
offloads the arithmetic from the CPU the other shards need; beyond ~4 the
GPU stops adding throughput). Every shard gets OMP/MKL/torch threads = 1 --
with N processes already filling the cores, extra threads only contend.

Each shard writes only its own tables (atomic) and never the shared index;
resume skips tables already on disk, so re-running the same command after
a crash or Ctrl-C continues where it stopped. scripts/run_report.py
rebuilds the index before loading.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shards", type=int, default=18, help="parallel processes")
    parser.add_argument("--gpu-shards", type=int, default=4,
                        help="how many of them may run torch models on the GPU (rest: CPU)")
    parser.add_argument("--run-dir", default=None,
                        help="shared run dir; default results/run_<utc>_parallel (reuse one to resume)")
    args, passthrough = parser.parse_known_args()

    if any(a.startswith("--run-dir") or a.startswith("--shard") for a in passthrough):
        raise SystemExit("pass --run-dir to this script, not through; --shard is set per process")

    run_dir = Path(args.run_dir) if args.run_dir else (
        ROOT / "results" / time.strftime("run_%Y%m%dT%H%M%SZ_parallel", time.gmtime())
    )
    logs = run_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    print(f"run-dir : {run_dir}")
    print(f"shards  : {args.shards}  ({min(args.gpu_shards, args.shards)} with GPU)")
    print(f"logs    : {logs}\\shard_XX.log")
    print(f"args    : {' '.join(passthrough)}\n", flush=True)

    procs = []
    for i in range(args.shards):
        env = os.environ.copy()
        env.update({
            "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
            "PYTHONUNBUFFERED": "1",
        })
        if i >= args.gpu_shards:
            env["PROJEKAT_DEVICE"] = "cpu"
        cmd = [sys.executable, str(ROOT / "scripts" / "run_econometric.py"), *passthrough,
               "--run-dir", str(run_dir), "--shard", f"{i}/{args.shards}"]
        log = open(logs / f"shard_{i:02d}.log", "a", encoding="utf-8")
        procs.append((i, subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT), log))
        # shard 0 writes the manifest and the regime boundary; give it a head start
        if i == 0:
            time.sleep(3)

    started = time.monotonic()
    try:
        while True:
            running = [i for i, p, _ in procs if p.poll() is None]
            n_tables = len(list((run_dir / "tables").glob("*.parquet")))
            print(f"\r{time.strftime('%H:%M:%S')}  running {len(running):2d}/{len(procs)}  "
                  f"tables {n_tables:,}  elapsed {int(time.monotonic() - started) // 60} min   ",
                  end="", flush=True)
            if not running:
                break
            time.sleep(30)
    except KeyboardInterrupt:
        print("\n[SIGINT] stopping all shards; finished tables are kept -- re-run with "
              f"--run-dir {run_dir} to continue.", flush=True)
        for _, p, _ in procs:
            p.terminate()
        return 130
    finally:
        for _, _, log in procs:
            log.close()

    print()
    bad = [(i, p.returncode) for i, p, _ in procs if p.returncode != 0]
    for i, code in bad:
        print(f"shard {i} exited {code} -- see {logs / f'shard_{i:02d}.log'}", file=sys.stderr)
    failures = run_dir / "failures.jsonl"
    if failures.exists() and failures.stat().st_size:
        print(f"some combinations failed: {failures}", file=sys.stderr)
    print(f"\ndone in {int(time.monotonic() - started) // 60} min. Next:\n"
          f"  uv run python scripts/run_report.py --run-dir {run_dir} --panel <panel>")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
