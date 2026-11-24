"""One-off edit of docs/bias.html, the CLI's help and bias.py's header: the season's in-sample game-results column off,
the last-10-seasons column from the held-back games (the owner, 2026-11-24: "Held-back, drop ’27").  The column's
header is the owner's and stays; its hover loses only the words that no longer hold."""
from pathlib import Path


def edit(path: str, pairs: list) -> None:
    p = Path(path)
    s = p.read_text(encoding="utf-8")
    for old, new in pairs:
        assert s.count(old) == 1, (path, old[:100], s.count(old))
        s = s.replace(old, new)
    p.write_text(s, encoding="utf-8")


edit("docs/bias.html", [
    ("""  // OpenRAPM vs game results. Clarify the other 2 columns as just being '27 ... last 10 seasons (inclusive of current)"
""", """  // OpenRAPM vs game results. Clarify the other 2 columns as just being '27 ... last 10 seasons (inclusive of current)",
  // then, once the in-sample leans were found not to hold on games the ratings never saw: "Held-back, drop ’27"
"""),
    ("""    ["g", m => `vs game results ${short(m.season)}`,
     m => `OpenRAPM minus the ${m.season} team-game results, per 1 standard deviation of the stat, points per 100`],
""", ""),
    ("""`OpenRAPM minus the team-game results of ${m.games_seasons.first} to ${m.games_seasons.last}, each season against its own ratings, per 1 standard deviation of the stat, points per 100`""",
     """`OpenRAPM minus the game results of ${m.games_seasons.first} to ${m.games_seasons.last}, per 1 standard deviation of the stat, points per 100`"""),
    ("""    const head = ["stat", "label", "vs_consensus", "vs_consensus_se", "vs_game_results", "vs_game_results_se",
                  "vs_game_results_last_10_seasons", "vs_game_results_last_10_seasons_se", "stat_sd"];
    const lines = [head, ...rows.map(r => [r.stat, r.label, r.t, r.t_se, r.g, r.g_se, r.gs, r.gs_se, r.sd])];""",
     """    const head = ["stat", "label", "vs_consensus", "vs_consensus_se", "vs_game_results_last_10_seasons",
                  "vs_game_results_last_10_seasons_se", "stat_sd"];
    const lines = [head, ...rows.map(r => [r.stat, r.label, r.t, r.t_se, r.gs, r.gs_se, r.sd])];"""),
])
edit("src/openrapm/commands/cli.py", [
    ("""outside systems does, their tables from `openrapm data public-ratings`, and than the season's "
                    "and the last 10 seasons' team-game results do); \"""",
     """outside systems does, their tables from `openrapm data public-ratings`, and than the games the "
                    "ratings never saw do over the last 10 finished seasons, params/bias_held_back.json); \""""),
])
edit("src/openrapm/bias.py", [
    ("""Two more columns ask the same of the team-game results (game_results.py), the season's on the same z (`stat_z`),
and the last 10 seasons' together.""",
     """A second column asks the same of games the ratings never saw, over the last 10 finished seasons
(params/bias_held_back.json, commands/site.py)."""),
])
print("edited")
