"""Docstrings and help that pointed at a section, a count or a reader that is gone (the cleanup of 2026-11-16)."""
from pathlib import Path


def edit(path: str, pairs: list) -> None:
    p = Path(path)
    raw = p.read_bytes()
    crlf = b"\r\n" in raw
    t = raw.decode("utf-8").replace("\r\n", "\n")
    for old, new in pairs:
        assert t.count(old) == 1, (path, old[:80])
        t = t.replace(old, new)
    p.write_bytes((t.replace("\n", "\r\n") if crlf else t).encode("utf-8"))
    print("edited", path)


edit("src/openrapm/commands/console.py", [
    ("(notes/HANDOFF.md, header: rebuilt on one", "(the 2026-10 cleanup: rebuilt on one")])
edit("scripts/checks/101_log_run.py", [
    ('(notes/HANDOFF.md, "The test"; scripts/experiment_chain.sh)', "(scripts/experiment_chain.sh)")])
edit("scripts/checks/63_yoy.py", [
    ("This is the test that decides experiments.  For every",
     "It guards the within-season WOWY test, check 151, which decides experiments; it decided them until 2026-10-15.  "
     "For every")])
edit("src/openrapm/commands/data.py", [
    ("covers; nothing reads them yet", "covers; the season in progress reads its hustle from them"),
    ('description="Fetches the league dashboards the ratings do not read yet (src/openrapm/data/dashboards.py) into "',
     'description="Fetches the league dashboards (src/openrapm/data/dashboards.py), whose season-to-date hustle the "'
     '\n                    "season in progress reads, into "')])
edit("src/openrapm/data/dashboards.py", [
    ('"""League dashboards from stats.nba.com that the ratings do not read yet: hustle, tracking, play types and matchups.',
     '"""League dashboards from stats.nba.com: hustle, tracking, play types and matchups; the hustle ones are inputs.')])
edit("src/openrapm/power_forecast.py", [
    ("holds them at four points of the regular season, its\nknots, from `scripts/checks/165_power_forecast_weights.py`.",
     "holds them at five points of the regular season, its\nknots, refitted by `scripts/checks/166_power_forecast_refit.py`.")])
edit("examples/tutorial.py", [
    ("    Everything else is the shipped setting: step 1's penalty of 40,000 with the defensive label un-shrunk, the\n"
     "    booster settings in params/booster_params_lgb1.json, five player folds, and step 3's penalty of 13,037.",
     "    Box Adj and the shrink are left out: the simulated league has no held-back games to fit them on.  Everything\n"
     "    else is the shipped setting but the booster's: step 1's penalty of 40,000 with the defensive label\n"
     "    un-shrunk, five player folds, step 3's penalty of 13,037, and the booster settings of\n"
     "    params/booster_params_lgb1.json, which suit the tutorial's small league.")])
