"""One command's seasons in several processes at once (the owner, 2026-10-21: "You can do whatever you want as long as
it's faster").

Every library runs on one thread (`cli.pin_threads`), so one `openrapm` process keeps one core of the machine busy
and leaves the rest idle: a ratings run used 13% of a 12-thread processor.  Each season is rated, and each season's
folds are cut, on its own, so the work splits by season.  `openrapm rate --jobs N` and `openrapm folds --jobs N` run
the same command N times over interleaved seasons, each child with `--jobs 1`, and then:

    rate    the children's step-3 tables, in season order, as one table; their boosters as one file
    folds   the children write into the same folder, each season's files its own; one gather at the end

Each child's numbers are the ones a single process gives for its seasons (`openrapm rate --seasons 2026` equals the
2026 rows of a thirty-season run, PIPELINE.md, step 3), so the whole equals the single-process run.

Never more than fit (the owner: "just make sure you're never just thrashing").  A process cutting folds peaks near
5.2 GB (measured 2026-10-21), so each job is counted at `GB_PER_JOB` against the memory free when the children start,
`RESERVE_GB` kept back, and at most `MAX_JOBS`, one of the six cores left to the rest of the machine
(`affordable_jobs`).  When only one fits, the command runs as it always did, in its own process.  `OPENRAPM_JOBS`
sets the default, 1 without it.

Example: `openrapm rate --seasons 1997-2026 --jobs 4` runs four children, the first rating 1997, 2001, 2005, ...
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

__all__ = ["JOBS_VARIABLE", "PART_VARIABLE", "GB_PER_JOB", "RESERVE_GB", "MAX_JOBS", "default_jobs",
           "affordable_jobs", "is_child", "season_chunks", "run_children", "part_name"]

JOBS_VARIABLE = "OPENRAPM_JOBS"
PART_VARIABLE = "OPENRAPM_PART"   # set in each child to its number: a child leaves the shared last steps to the parent
GB_PER_JOB = 5.5      # a folds process peaks near 5.2 GB
RESERVE_GB = 2.0      # kept free for everything else on the machine
MAX_JOBS = 5          # six cores, one left over


def default_jobs() -> int:
    """The `--jobs` default: `OPENRAPM_JOBS`, or 1."""
    return max(1, int(os.environ.get(JOBS_VARIABLE, "1") or "1"))


def affordable_jobs(requested: int, available_gb: float | None = None) -> int:
    """`requested`, cut to what free memory holds at `GB_PER_JOB` each after `RESERVE_GB`, and to `MAX_JOBS`; at
    least 1.  `available_gb` defaults to the memory free now.

    Example: 4 requested with 14 GB free gives 2: (14 - 2) / 5.5 holds two."""
    if available_gb is None:
        import psutil
        available_gb = psutil.virtual_memory().available / 2 ** 30
    return max(1, min(int(requested), MAX_JOBS, int((available_gb - RESERVE_GB) // GB_PER_JOB)))


def is_child() -> bool:
    """Whether this process is one of `run_children`'s: it then skips the steps that write one shared file, such as
    the folds' gather, which the parent runs once every child is done."""
    return os.environ.get(PART_VARIABLE) is not None


def season_chunks(seasons, jobs: int) -> list[list[int]]:
    """The seasons dealt out to `jobs` children in turn, so each gets early and late seasons alike.

    Example: 1997-2004 over 3 jobs gives [1997, 2000, 2003], [1998, 2001, 2004], [1999, 2002]."""
    seasons = list(seasons)
    return [chunk for chunk in (seasons[k::jobs] for k in range(max(1, jobs))) if chunk]


def part_name(name: str, k: int) -> str:
    """A child's table or boosters name: "luck88" -> "luck88__part2"."""
    return f"{name}__part{k}"


def run_children(argv: list, chunks: list, extra, log_dir: Path) -> None:
    """Run `openrapm <argv[1:]> --seasons <chunk> --jobs 1 <extra(k)>` for every chunk at once and wait for all.

    `argv` is the parent's command line ("openrapm", subcommand, options); a later option wins in argparse, so the
    child's seasons and names override the parent's.  Each child's output goes to its own log, printed whole once
    every child is done; a child that fails stops the run, naming its log."""
    log_dir.mkdir(parents=True, exist_ok=True)
    started, children = time.time(), []
    for k, chunk in enumerate(chunks):
        env = {**os.environ, JOBS_VARIABLE: "1", PART_VARIABLE: str(k)}
        command = [sys.executable, "-u", "-m", "openrapm", *argv[1:], "--seasons", ",".join(map(str, chunk)),
                   "--jobs", "1", *extra(k)]
        log = log_dir / f"part{k}.log"
        handle = open(log, "w", encoding="utf-8")
        children.append((k, chunk, log, handle, subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT,
                                                                 env=env)))
    print(f"  {len(children)} processes at once, seasons dealt out in turn", flush=True)
    failed = []
    for k, chunk, log, handle, process in children:
        process.wait()
        handle.close()
        print(f"\n  ---- process {k}: seasons {', '.join(map(str, chunk))} ----", flush=True)
        print(log.read_text(encoding="utf-8", errors="replace").rstrip(), flush=True)
        if process.returncode != 0:
            failed.append(f"process {k} ({log})")
    print(f"\n  every process done ({time.time() - started:.0f}s)", flush=True)
    if failed:
        raise SystemExit(f"{len(failed)} of {len(children)} processes failed: {'; '.join(failed)}")
