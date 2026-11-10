"""commands/published.py: where the morning job publishes from, a command typed by hand cannot write the published
tables' name; everywhere else, and with the name it was told to use, every command runs as before."""
from pathlib import Path

import pytest

from openrapm.commands import build, published, wins
from openrapm.settings import PRODUCT

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _cfg(tmp_path: Path, job_ran: bool = True) -> dict:
    """A config whose outputs folder the morning job has (or has not) left its record in."""
    daily = tmp_path / "outputs" / "daily"
    daily.mkdir(parents=True, exist_ok=True)
    if job_ran:
        (daily / "last.txt").write_text("2026-11-09 published 9d5b452\n", encoding="utf-8")
    return {"_root": str(tmp_path), "paths": {"outputs": "outputs", "within": "within"}, "first_season": 1997,
            "last_season": 2027}


def test_the_published_name_is_refused_where_the_morning_job_publishes_from(tmp_path, monkeypatch):
    monkeypatch.delenv(published.WRITE_PUBLISHED_ENV, raising=False)
    with pytest.raises(SystemExit) as stop:
        published.refuse_published_name(_cfg(tmp_path), "product")
    message = str(stop.value)
    assert "--name" in message and "OPENRAPM_WRITE_PUBLISHED=1" in message and str(tmp_path / "outputs") in message


def test_another_name_the_variable_or_a_folder_the_job_never_wrote_in_all_pass(tmp_path, monkeypatch):
    monkeypatch.delenv(published.WRITE_PUBLISHED_ENV, raising=False)
    published.refuse_published_name(_cfg(tmp_path), "zones139")                 # an experiment's name
    published.refuse_published_name(_cfg(tmp_path), "product139")               # a backup's name
    published.refuse_published_name(_cfg(tmp_path / "clone", job_ran=False), "product")   # a fresh clone, the tests
    monkeypatch.setenv(published.WRITE_PUBLISHED_ENV, "1")
    published.refuse_published_name(_cfg(tmp_path), "product")                  # the morning job, a full build
    monkeypatch.setenv(published.WRITE_PUBLISHED_ENV, "0")
    with pytest.raises(SystemExit):
        published.refuse_published_name(_cfg(tmp_path), "product")


def _never(*args, **kwargs):
    raise AssertionError("the command read or built something before it asked whether it may write the name")


@pytest.mark.parametrize("command", ["build", "rate", "shrink", "swap", "wins"])
def test_every_command_that_writes_a_ratings_table_asks_before_it_reads_anything(command, tmp_path, monkeypatch):
    """`openrapm rate --seasons 2026` is the worst of them: it would leave a one-season step-3 table, the one the
    morning job writes the season in progress into and reads its held prior scales from."""
    monkeypatch.delenv(published.WRITE_PUBLISHED_ENV, raising=False)
    monkeypatch.setattr(build, "require_folds", _never)
    monkeypatch.setattr(build, "rate_table", _never)
    monkeypatch.setattr(build.pd, "read_parquet", _never)
    cfg = _cfg(tmp_path)
    calls = {"build": lambda: build.run_build(cfg, PRODUCT, [2026], "product"),
             "rate": lambda: build.run_rate(cfg, PRODUCT, [2026], "product"),
             "shrink": lambda: build.run_shrink(cfg, PRODUCT, "season_ratings_product_step3", "product"),
             "swap": lambda: build.run_swap(cfg, PRODUCT, "season_ratings_product_step4", "product"),
             "wins": lambda: wins.run_wins(cfg, PRODUCT, "product")}
    with pytest.raises(SystemExit, match="the published tables'"):
        calls[command]()


def test_the_morning_job_says_it_means_the_published_tables_before_its_first_step():
    lines = [line.strip() for line in (SCRIPTS / "daily_update.sh").read_text(encoding="utf-8").splitlines()
             if not line.strip().startswith("#")]
    assert lines.index("export OPENRAPM_WRITE_PUBLISHED=1") < next(k for k, line in enumerate(lines)
                                                                    if line.startswith("step "))
