"""scripts/daily_push.sh: the morning's commit reaches main through a GitHub outage and on top of someone else's push.

Each test pushes to a real bare repository on disk.  Its pre-receive hook refuses the first pushes, as GitHub's Git
service refused both of the job's on 2026-11-04.
"""
import os
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "daily_push.sh"
HOOK = """#!/bin/sh
count=$(cat pushes 2>/dev/null || echo 0)
count=$((count + 1))
echo "$count" > pushes
[ "$count" -gt {refusals} ] || {{ echo "Internal Server Error" >&2; exit 1; }}
"""


def bash() -> str:
    """Git's own bash: on Windows the `bash` on the path can be WSL's, which cannot read these paths."""
    found = shutil.which("git")
    for folder in (Path(found).resolve().parents if found else ()):
        candidate = folder / "bin" / "bash.exe"
        if candidate.exists():
            return str(candidate)
    return shutil.which("bash") or pytest.skip("no bash here")


def git(folder: Path, *args) -> str:
    who = ("-c", "user.name=Test", "-c", "user.email=test@example.com")
    done = subprocess.run(["git", "-C", str(folder), *who, *args], check=True, capture_output=True, text=True)
    return done.stdout.strip()


def commit(folder: Path, file: str, text: str, message: str) -> None:
    (folder / file).write_text(text, encoding="utf-8")
    git(folder, "add", "--", file)
    git(folder, "commit", "-q", "-m", message)


def clone(origin: Path, folder: Path) -> Path:
    subprocess.run(["git", "clone", "-q", str(origin), str(folder)], check=True, capture_output=True)
    return folder


@pytest.fixture
def repos(tmp_path):
    """(origin, site): a bare repository with one commit on main, and a clone holding the morning's commit."""
    origin = tmp_path / "origin.git"
    subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(origin)], check=True, capture_output=True)
    site = tmp_path / "site"
    subprocess.run(["git", "init", "-q", "-b", "main", str(site)], check=True, capture_output=True)
    git(site, "remote", "add", "origin", str(origin))
    git(site, "config", "user.name", "Test")            # the script's rebase needs a name, and CI's machines have none
    git(site, "config", "user.email", "test@example.com")
    commit(site, "ratings.json", "yesterday\n", "yesterday")
    git(site, "push", "-q", "origin", "HEAD:main")
    git(site, "fetch", "-q", "origin", "main")
    commit(site, "ratings.json", "today\n", "the morning's commit")
    return origin, site


def refuse(origin: Path, refusals: int) -> None:
    hook = origin / "hooks" / "pre-receive"
    hook.write_text(HOOK.format(refusals=refusals), encoding="utf-8", newline="\n")
    hook.chmod(0o755)


def push(site: Path, waits: str) -> int:
    log = site.parent / "job.log"
    env = {**os.environ, "PUSH_WAITS": waits}
    return subprocess.run([bash(), str(SCRIPT), str(site), str(log)], env=env, capture_output=True).returncode


def main_message(origin: Path) -> str:
    return git(origin, "log", "-1", "--format=%s", "main")


def test_the_first_push_needs_no_second_try(repos):
    origin, site = repos
    assert push(site, "0 0") == 0
    assert main_message(origin) == "the morning's commit"
    assert not (site.parent / "job.log").read_text(encoding="utf-8")


def test_the_push_gets_through_once_github_is_back(repos):
    origin, site = repos
    refuse(origin, refusals=3)
    assert push(site, "0 0 0") == 0
    assert main_message(origin) == "the morning's commit"
    assert (site.parent / "job.log").read_text(encoding="utf-8").count("trying again in 0 minutes") == 3


def test_an_outage_longer_than_the_tries_fails_and_keeps_the_commit(repos):
    origin, site = repos
    refuse(origin, refusals=99)
    assert push(site, "0 0") == 1
    assert main_message(origin) == "yesterday"
    assert git(site, "log", "-1", "--format=%s") == "the morning's commit"


def test_the_commit_goes_on_top_of_someone_elses_push(repos):
    origin, site = repos
    other = clone(origin, site.parent / "other")
    commit(other, "index.html", "page\n", "someone else's commit")
    git(other, "push", "-q", "origin", "HEAD:main")
    assert push(site, "0") == 0
    newest_two = git(origin, "log", "-2", "--format=%s", "main").splitlines()
    assert newest_two == ["the morning's commit", "someone else's commit"]


def test_a_conflict_with_someone_elses_push_stops_and_leaves_no_rebase_open(repos):
    origin, site = repos
    other = clone(origin, site.parent / "other")
    commit(other, "ratings.json", "theirs\n", "someone else's commit")
    git(other, "push", "-q", "origin", "HEAD:main")
    assert push(site, "0") == 2
    assert main_message(origin) == "someone else's commit"
    assert git(site, "log", "-1", "--format=%s") == "the morning's commit"
    assert not (site / ".git" / "rebase-merge").exists()
