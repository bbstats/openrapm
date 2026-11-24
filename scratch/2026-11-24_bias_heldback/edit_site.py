"""One-off edit of src/openrapm/commands/site.py: the Bias page's last-10-seasons column from the held-back games, the
season's in-sample game-results column off (the owner, 2026-11-24: "Held-back, drop ’27")."""
from pathlib import Path

p = Path("src/openrapm/commands/site.py")
s = p.read_text(encoding="utf-8")


def sub(old: str, new: str) -> None:
    global s
    assert s.count(old) == 1, (old[:100], s.count(old))
    s = s.replace(old, new)


sub("""from ..game_results import lean_table, season_games
from ..game_results import multipliers as game_multipliers
""", "")
sub("""from ..data.tables import team_of_rows
""", "")
sub("""# outside systems does ("vs consensus"), and than the season's team-game results do ("vs game results"; the owner,
# 2026-11-24: "i just want like game level results or something. the ridge penalization just makes it a bit too hard
# to interpret"); and than the team-game results of the last `GAME_SEASONS` seasons, the season's included, do, each
# season against its own published ratings (the owner, 2026-11-24: "Add in 'last 10 years' just OpenRAPM vs game
# results. Clarify the other 2 columns as just being '27 ... last 10 seasons (inclusive of current)").  bias.py and
# game_results.py hold the equations; the file:
#
#   {"meta": {"season", "through", "built", "min_possessions", "players_shown", "players",
#             "multipliers": {"o", "d"}, "games": {"team_games", "multipliers": {"o", "d"}},
#             "games_seasons": {"first", "last", "team_games"}, "fields": BIAS_FIELDS},
#    "systems": [{"key", "name", "by", "link", "weight", "as_of", "players"}, ...],    the three, heaviest first
#    "rows": [[stat, label, o, d, t, o_se, d_se, t_se, sd, g, g_se, gs, gs_se], ...]}  one per stat, bias.STATS order
#
# o and d are written at two decimals and t is their sum, so every row adds up on the page; g and gs, OpenRAPM minus
# the game results of the season and of the seasons, and their standard errors at two.  players_shown is the season's
# players with 60 or more offensive possessions (`MIN_POSSESSIONS_SHOWN`), and players those of them all three systems
# list; their mean and spread of each stat are the standard deviation the season's two columns are per (bias.stat_z).
# In the seasons' column each season's stat is in standard deviations of that season's shown players.  A system's own
# `players` is how many of the shown it lists; its `as_of` is its own date, or the day it was pulled when its page
# prints none.  Only these slopes are published: the systems' ratings stay in paths.public_ratings, out of git.
BIAS_FILE = "bias.json"
BIAS_FIELDS = ["stat", "label", "o", "d", "t", "o_se", "d_se", "t_se", "sd", "g", "g_se", "gs", "gs_se"]
GAME_SEASONS = 10       # the seasons of the last game-results column, the one the page opens on and the 9 before it
""", """# outside systems does ("vs consensus"); and than the results of games the ratings never saw do, over the last ten
# finished seasons (the owner, 2026-11-24, on the in-sample game-results columns this replaced, which graded each
# season's ratings on the games they were fit on: "Held-back, drop ’27").  That column is fixed until the next build
# is adopted: `BIAS_HELD_BACK` holds it, written by scratch/2026-11-24_bias_heldback/write_params.py from the
# incumbent's within-season test (notes/DECISIONS.md, "The Bias page's leans on held-back games").  bias.py holds the
# consensus column's equations; the file:
#
#   {"meta": {"season", "through", "built", "min_possessions", "players_shown", "players",
#             "multipliers": {"o", "d"}, "games_seasons": {"first", "last", "build"}, "fields": BIAS_FIELDS},
#    "systems": [{"key", "name", "by", "link", "weight", "as_of", "players"}, ...],    the three, heaviest first
#    "rows": [[stat, label, o, d, t, o_se, d_se, t_se, sd, gs, gs_se], ...]}          one per stat, bias.STATS order
#
# o and d are written at two decimals and t is their sum, so every row adds up on the page; gs, OpenRAPM minus the
# held-back games, and its standard error, at two.  players_shown is the season's players with 60 or more offensive
# possessions (`MIN_POSSESSIONS_SHOWN`), and players those of them all three systems list; their mean and spread of
# each stat are the standard deviation the consensus column is per (bias.stat_z).  A system's own `players` is how many
# of the shown it lists; its `as_of` is its own date, or the day it was pulled when its page prints none.  Only these
# slopes are published: the systems' ratings stay in paths.public_ratings, out of git.
BIAS_FILE = "bias.json"
BIAS_FIELDS = ["stat", "label", "o", "d", "t", "o_se", "d_se", "t_se", "sd", "gs", "gs_se"]
BIAS_HELD_BACK = "params/bias_held_back.json"
""")
sub('''    """One stat's row in `BIAS_FIELDS` order: offense and defense at two decimals, total their sum, then the game
    results' leans, the season's and the seasons', and their standard errors."""''',
    '''    """One stat's row in `BIAS_FIELDS` order: offense and defense at two decimals, total their sum, then the
    held-back games' lean and its standard error."""''')
sub("""_rounded(row.sd, 3), _rounded(row.games),
            _rounded(row.games_se), _rounded(row.seasons), _rounded(row.seasons_se)]""",
    """_rounded(row.sd, 3), _rounded(row.held_back),
            _rounded(row.held_back_se)]""")
sub('''def shown_z(table: pd.DataFrame, season: int, panel: pd.DataFrame) -> pd.DataFrame:
    """bias.stat_z for one season of the seasons' column: every panel player's stat in standard deviations of the
    season's shown players (`MIN_POSSESSIONS_SHOWN` or more offensive possessions), weighted by their possessions."""
    stats = player_stats(panel)
    shown = table[(table.season == season) & (table.poss_season >= MIN_POSSESSIONS_SHOWN)]
    players = shown[["player_id", "poss_season"]].rename(columns={"poss_season": "weight"}).merge(stats, on="player_id")
    return stat_z(stats, players)


def bias_document(table: pd.DataFrame, pulls: dict, season: int, built: str, through: str, seasons: dict) -> dict:
    """bias.json's document for `season`, from the ratings table, each system's stored pull ({key:
    `public_ratings.PublicRatings`}) and `seasons`, {season: (its team-games, `game_results.season_games`, and its
    panel rows)} for the last `GAME_SEASONS` seasons, `season` among them."""
    games, panel = seasons[season]
''', '''def held_back_leans(path: Path) -> dict:
    """`BIAS_HELD_BACK` as read: {"first", "last", "build", "rows": {stat: [lean, standard error]}}, every stat of
    bias.STATS present."""
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    missing = [stat for stat in STATS if stat not in document["rows"]]
    if missing:
        raise SystemExit(f"{path} has no held-back lean for {', '.join(missing)}")
    return document


def bias_document(table: pd.DataFrame, pulls: dict, season: int, built: str, through: str, panel: pd.DataFrame,
                  held_back: dict) -> dict:
    """bias.json's document for `season`, from the ratings table, each system's stored pull ({key:
    `public_ratings.PublicRatings`}), the season's panel rows and the held-back games' leans (`held_back_leans`)."""
''')
sub("""    slopes, multipliers = bias_table(players)
    slopes = slopes.merge(lean_table([games], [stat_z(stats, players)], STATS), on="stat", how="left")
    order = sorted(seasons)
    pooled = lean_table([seasons[s][0] for s in order], [shown_z(table, s, seasons[s][1]) for s in order], STATS)
    slopes = slopes.merge(pooled.rename(columns={"games": "seasons", "games_se": "seasons_se"}), on="stat", how="left")
    fitted = game_multipliers(games)
""", """    slopes, multipliers = bias_table(players)
    leans = pd.DataFrame([(stat, *values) for stat, values in held_back["rows"].items()],
                         columns=["stat", "held_back", "held_back_se"])
    slopes = slopes.merge(leans, on="stat", how="left")
""")
sub("""                games=dict(team_games=games.team_games,
                           multipliers=dict(o=_rounded(fitted["offense"], 3), d=_rounded(fitted["defense"], 3))),
                games_seasons=dict(first=order[0], last=order[-1],
                                   team_games=sum(seasons[s][0].team_games for s in order)),
""", """                games_seasons=dict(first=held_back["first"], last=held_back["last"], build=held_back["build"]),
""")
sub('''    defense, total (vs consensus), vs game results of the season, vs game results of the seasons."""
    meta = document["meta"]
    games = meta["games"]["multipliers"]
''', '''    defense, total (vs consensus), vs the held-back games."""
    meta = document["meta"]
''')
sub("""             f"  game results on {meta['games']['team_games']} team-games, OpenRAPM multiplied by {games['o']:.3f} "
             f"on offense and {games['d']:.3f} on defense to fit them",
             f"  game results of {meta['games_seasons']['first']} to {meta['games_seasons']['last']} on "
             f"{meta['games_seasons']['team_games']} team-games"]""",
    """             f"  held-back games of {meta['games_seasons']['first']} to {meta['games_seasons']['last']}, "
             f"build {meta['games_seasons']['build']}"]""")
sub("""    columns = ["o", "d", "t", "g", "gs"]""", """    columns = ["o", "d", "t", "gs"]""")
sub("""f"   standard errors {row[field['t_se']]}, {row[field['g_se']]}, {row[field['gs_se']]}\"""",
    """f"   standard errors {row[field['t_se']]}, {row[field['gs_se']]}\"""")
sub('''def season_designs(cfg: dict, season: int) -> dict:
    """The season's design on each side's target, the build's (settings.Targets), from its stored pieces: the
    team-games the game results column is read on (game_results.season_games)."""
    cache, targets = DesignCache(cfg, keep=2), Targets()
    return {"offense": cache.season(season, targets.offence), "defense": cache.season(season, targets.defence)}


''', "")
sub("""    pull and the team-games of it and the seasons before it, `GAME_SEASONS` in all, from their stored design pieces
    (about a second a season).  A system""", """    pull, the season's panel and the held-back games' leans (`BIAS_HELD_BACK`).  A system""")
sub("""    rated = set(table.season.astype(int))
    seasons = {s: (season_games(season_designs(cfg, s), table, s, partial(team_of_rows, cfg=cfg)),
                   panel_of_season(cfg, s))
               for s in range(season - GAME_SEASONS + 1, season + 1) if s in rated}
    document = bias_document(table, pulls, season, built, through, seasons)""",
    """    held_back = held_back_leans(resolve_path(cfg, BIAS_HELD_BACK))
    document = bias_document(table, pulls, season, built, through, panel_of_season(cfg, season), held_back)""")
p.write_text(s, encoding="utf-8")
print("edited")
