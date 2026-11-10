"""scripts/daily_lock.sh: the morning job's lock, so two runs never write the same files, and the hours a full build
waits out.  Each test runs the sourced functions in Git's own bash, as tests/test_daily_push.py does."""
import os
import subprocess
from pathlib import Path

from test_daily_push import bash

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def run(tmp_path: Path, script: str, **env) -> subprocess.CompletedProcess:
    """`script` after the lock's functions are sourced, the lock in `tmp_path`."""
    settings = {**os.environ, "DAILY_LOCK": (tmp_path / "running.lock").as_posix(), **env}
    source = f'. "{(SCRIPTS / "daily_lock.sh").as_posix()}"\n{script}\n'
    return subprocess.run([bash(), "-c", source], env=settings, capture_output=True, text=True)


def test_one_run_takes_the_lock_and_a_second_is_refused_until_it_is_given_back(tmp_path):
    first = run(tmp_path, 'take_lock "the morning job" && echo taken')
    assert first.stdout.strip() == "taken" and (tmp_path / "running.lock" / "taken").exists()
    second = run(tmp_path, 'take_lock "a full build" || echo "held by $(lock_owner)"; lock_held && echo still')
    assert second.stdout.split("\n")[:2] == ["held by the morning job", "still"]
    # the refused run wrote nothing
    assert (tmp_path / "running.lock" / "owner").read_text().strip() == "the morning job"
    after = run(tmp_path, 'release_lock; lock_held || echo free; take_lock "a full build" && lock_owner')
    assert after.stdout.split() == ["free", "a", "full", "build"]


def test_a_lock_left_by_a_run_that_died_is_taken_over_after_three_hours(tmp_path):
    """The scheduled task is stopped after 2 hours, so no live run holds a lock for 3."""
    run(tmp_path, 'take_lock "a run that died"')
    young = run(tmp_path, 'take_lock "this morning" || echo refused', LOCK_STALE_SECONDS="10800")
    assert young.stdout.strip() == "refused"
    old = run(tmp_path, 'lock_held || echo stale; take_lock "this morning" && lock_owner', LOCK_STALE_SECONDS="0")
    assert old.stdout.split("\n")[:2] == ["stale", "this morning"]
    assert not list(tmp_path.glob("running.lock.old.*"))            # the renamed lock is removed, not left beside it


def test_a_lock_without_its_record_counts_as_old(tmp_path):
    (tmp_path / "running.lock").mkdir()
    out = run(tmp_path, 'lock_held || echo stale; take_lock "this morning" && lock_owner')
    assert out.stdout.split("\n")[:2] == ["stale", "this morning"]


def test_the_morning_job_has_five_fifty_to_seven_thirty_to_itself(tmp_path):
    def inside(clock: str) -> bool:
        return run(tmp_path, "morning_window && echo yes || echo no", MORNING_NOW=clock).stdout.strip() == "yes"
    assert [inside(clock) for clock in ("0549", "0550", "0600", "0729", "0730", "2040", "0008")] == \
        [False, True, True, True, False, False, False]


def test_the_morning_job_takes_the_lock_before_its_first_step_and_a_full_build_waits_for_it():
    """The job: the lock before anything is fetched or built, given back however it ends, a dry run that finds it held
    stopped without touching the record of the last morning.  The full build: it waits while the lock is held or the
    morning's hours are on the clock."""
    job = (SCRIPTS / "daily_update.sh").read_text(encoding="utf-8")
    lines = [line.strip() for line in job.splitlines() if not line.strip().startswith("#")]
    first_step = next(k for k, line in enumerate(lines) if line.startswith("step "))
    source = next(k for k, line in enumerate(lines) if line == '. "$ROOT/scripts/daily_lock.sh"')
    taken = next(k for k, line in enumerate(lines) if "take_lock" in line)
    trap = next(k for k, line in enumerate(lines) if line == "trap release_lock EXIT")
    assert source < taken < trap < first_step
    chain = (SCRIPTS / "experiment_chain.sh").read_text(encoding="utf-8")
    assert "daily_lock.sh" in chain and "while lock_held || morning_window; do" in chain


def test_every_script_of_the_morning_job_parses():
    """A syntax error would stop the job before it could write how it ended or send a notice."""
    for script in sorted(SCRIPTS.glob("*.sh")):
        done = subprocess.run([bash(), "-n", script.as_posix()], capture_output=True, text=True)
        assert done.returncode == 0, f"{script.name}: {done.stderr}"
