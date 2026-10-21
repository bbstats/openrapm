"""`openrapm rate --jobs N` and `openrapm folds --jobs N`: the seasons dealt out to N processes, never more than free
memory holds (src/openrapm/commands/parallel.py; the owner, 2026-10-21: "just make sure you're never just
thrashing")."""
from openrapm.commands import parallel
from openrapm.commands.cli import build_parser


def test_seasons_are_dealt_out_in_turn():
    assert parallel.season_chunks(range(1997, 2005), 3) == [[1997, 2000, 2003], [1998, 2001, 2004], [1999, 2002]]
    assert parallel.season_chunks([2026], 4) == [[2026]]
    chunks = parallel.season_chunks(range(1997, 2027), 4)
    assert sorted(season for chunk in chunks for season in chunk) == list(range(1997, 2027))


def test_never_more_processes_than_free_memory_holds():
    assert parallel.affordable_jobs(4, available_gb=14.0) == 2          # (14 - 2) / 5.5: the docstring's example
    assert parallel.affordable_jobs(4, available_gb=4.7) == 1           # one fits: the command runs on its own
    assert parallel.affordable_jobs(8, available_gb=64.0) == parallel.MAX_JOBS
    assert parallel.affordable_jobs(3, available_gb=100.0) == 3


def test_jobs_default_to_the_environment_or_one(monkeypatch):
    monkeypatch.delenv(parallel.JOBS_VARIABLE, raising=False)
    assert parallel.default_jobs() == 1
    monkeypatch.setenv(parallel.JOBS_VARIABLE, "4")
    assert parallel.default_jobs() == 4
    assert build_parser().parse_args(["rate", "--jobs", "3"]).jobs == 3
    assert build_parser().parse_args(["folds", "--models", "m", "--base", "b"]).jobs == 4


def test_a_child_knows_it_is_one(monkeypatch):
    monkeypatch.delenv(parallel.PART_VARIABLE, raising=False)
    assert not parallel.is_child()
    monkeypatch.setenv(parallel.PART_VARIABLE, "0")
    assert parallel.is_child()
    assert parallel.part_name("luck88", 2) == "luck88__part2"
