"""One-off edit: PIPELINE.md's Bias section and the layer list, for the held-back column (game_results.py removed)."""
from pathlib import Path


def edit(path: str, pairs: list) -> None:
    p = Path(path)
    s = p.read_text(encoding="utf-8")
    for old, new in pairs:
        assert s.count(old) == 1, (path, old[:100], s.count(old))
        s = s.replace(old, new)
    p.write_text(s, encoding="utf-8")


edit("PIPELINE.md", [
    ("""  outside rating systems does, stat by stat, "vs consensus" (the owner's design, 2026-11-05), than the season's
  team-game results do, "vs game results" (the owner, 2026-11-24), and than the team-game results of the last 10
  seasons, the season's included, do, "vs game results, last 10 seasons" (the owner, 2026-11-24).""",
     """  outside rating systems does, stat by stat, "vs consensus" (the owner's design, 2026-11-05), and than games the
  ratings never saw do over the last 10 finished seasons, "vs game results, last 10 seasons" (the owner,
  2026-11-24: "Held-back, drop ’27", once the in-sample game-results columns were found to locate no error)."""),
    ("""  newest stored table, `data.public_ratings.latest`; the seasons' stored design pieces, for the team-games, about a
  second a season.""", """  newest stored table, `data.public_ratings.latest`; `params/bias_held_back.json`, the held-back column."""),
    ("""  with their weights and dates, the team-games read and the games' multipliers, the seasons' first, last and
  team-games, and one row per stat with offense, defense, total, the two leans against the game results and their
  standard errors.""", """  with their weights and dates, the held-back column's first and last season and build, and one row per stat with
  offense, defense, total, the lean against the held-back games and the standard errors."""),
    ("""- **vs game results** (`game_results.py`): no ridge. Each team-game's points minus OpenRAPM's prediction for it is
  fitted by weighted least squares on the stat of the players on the court, the attackers' and the defenders'
  shares times their z summed, with the prediction itself free per side (the width match's part) and the season's
  level and home court. The defenders' coefficient minus the attackers' is OpenRAPM minus the games, per player
  and per standard deviation, the same step as the consensus column. Its standard error is how far that number
  moves when each team's games are left out in turn: a player's miss repeats in all his games, so leaving a team out
  resamples its players as well as its games.
- **vs game results, last 10 seasons:** the same fit on the team-games of the season and the 9 before it together,
  each season against its own published ratings, with its own width match, level and home court, and each season's
  stat in standard deviations of its own shown players; one lean per stat for all ten.  A team is left out of every
  season at once for the standard error.""",
     """- **vs game results, last 10 seasons:** the incumbent build's within-season test, its held-back stints of 2017 to
  2026.  Each stint's points minus the ten players' ratings, built without its games, is fitted by weighted least
  squares on the offense five's and the defense five's summed z, the stat from the fold's own fit games in standard
  deviations of the fold's players with 60 or more offensive possessions, with team levels per fold, garbage time,
  the margin, the fives' playing time and the ratings' sums, squares and spreads free.  The defense five's
  coefficient minus the offense five's is OpenRAPM minus the games, per player and per standard deviation.  Its
  standard error is the larger of the ten seasons' spread and that of the same lean on another player's stat handed
  along his career, 200 times.  Fixed until a new build is adopted: then `scratch/2026-11-23_synergy/extract.py`,
  `scratch/2026-11-24_bias_heldback/heldback_lean.py` and `write_params.py` (notes/DECISIONS.md, "The Bias page's
  leans on held-back games")."""),
])
edit("tests/test_layer_boundary.py", [
    ("""    "game_results.py",  # the Bias page's game results: the season's team-games against the rating, stat by stat
""", ""),
])
print("edited")
