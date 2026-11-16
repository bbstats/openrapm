"""The six lines the cleanup's text edits left over 120 characters, wrapped."""
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


edit("scripts/checks/63_yoy.py", [(
    "It guards the within-season WOWY test, check 151, which decides experiments; it decided them until 2026-10-15.  "
    "For every scored season (1998-2025 by default) the previous season's",
    "It guards the within-season WOWY test, check 151, which decides experiments; it decided them until 2026-10-15.\n"
    "For every scored season (1998-2025 by default) the previous season's")])
edit("src/openrapm/commands/cli.py", [
    ('''help="what Box Adj's line reads, a key of second_prior.INPUT_SETS: context_types_team_zones_centred, "
                             "shipped, is''',
     '''help="what Box Adj's line reads, a key of second_prior.INPUT_SETS: "
                             "context_types_team_zones_centred, shipped, is'''),
    ('''"his team so no team total moves, and the spread inside teams held.  The type model trains on the within-season folds' held-back "''',
     '''"his team so no team total moves, and the spread inside teams held.  The type model trains on "
                    "the within-season folds' held-back "'''),
    ('''help="the inputs, a key of second_prior.INPUT_SETS: context_types_team_zones_centred, shipped, is "''',
     '''help="the inputs, a key of second_prior.INPUT_SETS: context_types_team_zones_centred, shipped, "
                             "is "''')])
edit("src/openrapm/power_forecast.py", [(
    "holds them at five points of the regular season, its\nknots, refitted by "
    "`scripts/checks/166_power_forecast_refit.py`.  Between two knots each number is the straight line between",
    "holds them at five points of the regular season, its\nknots, refitted by "
    "`scripts/checks/166_power_forecast_refit.py`.  Between two knots each number is the straight line\nbetween")])
edit("src/openrapm/settings.py", [(
    "    # (missing before 2017), three-point % off the offense, and on offense the owner's self-creation (notes/DECISIONS.md,\n"
    "    # experiment 110, adopted 2026-10-25, on experiments 45, 59, 76 and 77 before it); features.FEATURE_SETS says",
    "    # (missing before 2017), three-point % off the offense, and on offense the owner's self-creation\n"
    "    # (notes/DECISIONS.md, experiment 110, adopted 2026-10-25, on experiments 45, 59, 76 and 77 before it);\n"
    "    # features.FEATURE_SETS says")])
print("wrapped")
