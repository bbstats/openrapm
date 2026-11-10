"""scripts/daily_notify.sh and scripts/daily_missed.sh: a failed morning, and a morning that never ran, reach the
owner's phone, and a notice that cannot be sent never fails or holds the job.  curl is a stand-in that records what it
was asked to send."""
import os
import subprocess
from pathlib import Path

from test_daily_push import bash

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
FAKE_CURL = """#!/bin/sh
printf '%s\\n' "$@" > "{record}"
exit {status}
"""


def fake_curl(tmp_path: Path, status: int = 0) -> tuple[Path, Path]:
    """(the stand-in, the file it writes its arguments to, one a line)."""
    record, program = tmp_path / "sent.txt", tmp_path / "curl.sh"
    program.write_text(FAKE_CURL.format(record=record.as_posix(), status=status), encoding="utf-8", newline="\n")
    program.chmod(0o755)
    return program, record


def notify(tmp_path: Path, message: str, conf: str | None, status: int = 0, script: str = "daily_notify.sh",
           **env) -> tuple[subprocess.CompletedProcess, Path]:
    program, record = fake_curl(tmp_path, status)
    path = tmp_path / "notify.conf"
    if conf is not None:
        path.write_text(conf, encoding="utf-8", newline="\n")
    settings = {**os.environ, "NOTIFY_CONF": path.as_posix(), "NOTIFY_CURL": program.as_posix(), **env}
    done = subprocess.run([bash(), (SCRIPTS / script).as_posix(), message], env=settings, capture_output=True,
                          text=True)
    return done, record


def test_a_failed_morning_is_posted_to_the_topic_the_untracked_file_names(tmp_path):
    done, record = notify(tmp_path, "OpenRAPM 2026-11-10 FAILED: data stints", "ntfy https://ntfy.sh/a-private-topic\n")
    assert done.returncode == 0
    sent = record.read_text(encoding="utf-8").splitlines()
    assert sent[-1] == "https://ntfy.sh/a-private-topic"
    assert sent[sent.index("-d") + 1] == "OpenRAPM 2026-11-10 FAILED: data stints"
    assert "Title: OpenRAPM morning job" in sent and sent[sent.index("-m") + 1] == "15"   # curl gives up by itself


def test_without_the_file_nothing_is_sent(tmp_path):
    done, record = notify(tmp_path, "OpenRAPM 2026-11-10 FAILED: data stints", None)
    assert done.returncode == 0 and not record.exists() and not done.stdout


def test_a_notice_that_cannot_be_sent_still_ends_0(tmp_path):
    """The job calls it from `finish`, as it is ending FAILED: a second failure there must not change how it ends."""
    done, _ = notify(tmp_path, "OpenRAPM 2026-11-10 FAILED: data stints", "ntfy https://ntfy.sh/a-private-topic\n",
                     status=7)
    assert done.returncode == 0 and "not sent" in done.stdout
    odd, _ = notify(tmp_path, "x", "pigeon the-roof\n")
    assert odd.returncode == 0 and "names no way" in odd.stdout


def missed(tmp_path: Path, last: str | None, today: str) -> Path:
    folder = tmp_path / "daily"
    folder.mkdir()
    if last is not None:
        (folder / "last.txt").write_text(last, encoding="utf-8", newline="\n")
    _, record = notify(tmp_path, "", "ntfy https://ntfy.sh/a-private-topic\n", script="daily_missed.sh",
                       DAILY_FOLDER=folder.as_posix(), TODAY=today)
    return record


def test_a_morning_that_ended_any_way_at_all_sends_nothing_at_eight(tmp_path):
    """The job's own last line is dated today: it published, found nothing new, or failed and said so itself."""
    assert not missed(tmp_path, "2026-11-10 published 9d5b452\n", "2026-11-10").exists()


def test_a_morning_that_never_ran_is_told_at_eight_with_the_last_line_on_record(tmp_path):
    record = missed(tmp_path, "2026-11-09 published 9d5b452\n", "2026-11-10")
    sent = record.read_text(encoding="utf-8").splitlines()
    assert sent[sent.index("-d") + 1] == ("OpenRAPM 2026-11-10: the morning job has not finished. "
                                           "Last: 2026-11-09 published 9d5b452")


def test_a_pc_with_no_record_at_all_is_told_too(tmp_path):
    record = missed(tmp_path, None, "2026-11-10")
    sent = record.read_text(encoding="utf-8").splitlines()
    assert sent[sent.index("-d") + 1].endswith("Last: nothing on record")


def test_the_job_sends_a_notice_only_as_it_ends_failed_and_can_never_wait_on_it():
    job = (SCRIPTS / "daily_update.sh").read_text(encoding="utf-8")
    lines = [line.strip() for line in job.splitlines() if not line.strip().startswith("#")]
    notice = next(line for line in lines if line.startswith("notice() {"))
    assert '/usr/bin/timeout 30 "$BASH" "$ROOT/scripts/daily_notify.sh"' in notice and notice.endswith("|| true; }")
    finish = "\n".join(lines[lines.index(next(line for line in lines if line.startswith("finish() {"))):][:6])
    assert 'case "$1" in FAILED*) notice' in finish
