"""One-off edit for the bug-fix pass of 2026-10-30, section 4 (kept for the record): the season in progress counts the
games its stints hold; the power page's teams read against the average team."""
from pathlib import Path


def edit(path: str, pairs: list) -> None:
    p = Path(path)
    s = p.read_text(encoding="utf-8")
    for old, new in pairs:
        assert s.count(old) == 1, (path, old[:70], s.count(old))
        s = s.replace(old, new)
    p.write_text(s, encoding="utf-8", newline="\n")


edit("src/openrapm/data/seasons.py", [
    ('''def share_played(season: int, cfg) -> float:
    """The share of `season`'s regular season played: 1 for a season whose Finals are over, else its regular-season
    games in the cached game log over the 1,230 scheduled.  The box-score prior is told it (box_prior.rated_rows, the
    part-season rows of experiment 107).  Example: on the morning of 2026-10-24, 2027's 16 games give 0.013."""
    if int(season) <= trainable_through(cfg):
        return 1.0
    from .ingest import raw_dir
    path = raw_dir(cfg) / "gamelog" / f"{int(season)}_RS.parquet"
    games = pd.read_parquet(path, columns=["GAME_ID"]).GAME_ID.nunique() if path.exists() else 0
    return min(1.0, games / SCHEDULED_GAMES)''',
     '''def share_played(season: int, cfg) -> float:
    """The share of `season`'s regular season played: 1 for a season whose Finals are over, else its regular-season
    games over the 1,230 scheduled.  The box-score prior is told it (box_prior.rated_rows, the part-season rows of
    experiment 107).  Example: on the morning of 2026-10-24, 2027's 16 games give 0.013.

    The games counted are the cached game log's that the season's stints hold, the games rated: a game whose
    play-by-play failed to fetch or to parse is in the game log with no stints, and `data ingest` and `data stints` go
    on without it (2026-10-30).  Such games are named.  Before the stints exist, the game log's count."""
    if int(season) <= trainable_through(cfg):
        return 1.0
    from .config import resolve
    from .ingest import raw_dir
    path = raw_dir(cfg) / "gamelog" / f"{int(season)}_RS.parquet"
    logged = set(pd.read_parquet(path, columns=["GAME_ID"]).GAME_ID.astype(str)) if path.exists() else set()
    stints = resolve(cfg, "stints") / f"{int(season)}_RS.parquet"
    if stints.exists():
        rated = set(pd.read_parquet(stints, columns=["game_id"]).game_id.astype(str))
        missing = sorted(logged - rated)
        if missing:
            print(f"  {season}: {len(missing)} games of the game log have no stints and are not rated: "
                  f"{' '.join(missing[:10])}", flush=True)
        logged &= rated
    return min(1.0, len(logged) / SCHEDULED_GAMES)'''),
])

edit("src/openrapm/commands/site.py", [
    ('''# Each team as it has played lately: every player's current-season rating times his share of the team's possessions
# over its last 10 games, summed over the team (the owner's design, 2026-10-28).  Shares add up to 500%, five on the
# floor, so a team of average players is 0 and the sums are points per 100 possessions above an average team.''',
     '''# Each team as it has played lately: every player's current-season rating times his share of the team's possessions
# over its last 10 games, summed over the team (the owner's design, 2026-10-28).  Shares add up to 500%, five on the
# floor, so the sums are points per 100 possessions above an average team: each side's sum less the mean of the 30
# teams' sums (meta "average_team").  The ratings are centred on every possession and the shares count blowouts half,
# so the plain sums put the average team at +0.60 and 17 of 30 teams above it (2026-10-30); a player's row keeps his
# rating and its share of the plain sum.'''),
    ('''    teams.sort(key=lambda team: -team["t"])
    meta = dict(season=season, through=pd.Timestamp(games.game_date.max()).strftime("%Y-%m-%d"), games=POWER_GAMES,
                recent_games=RECENT_GAMES, recent_weight=RECENT_WEIGHT, blowout_weight=BLOWOUT_WEIGHT,
                ready=min(team["g"] for team in teams) >= POWER_GAMES, built=built)''',
     '''    average = {side: float(np.mean([team[side] for team in teams])) for side in "odt"}
    for team in teams:
        team.update({side: round(team[side] - average[side], 2) for side in "odt"})
    teams.sort(key=lambda team: -team["t"])
    meta = dict(season=season, through=pd.Timestamp(games.game_date.max()).strftime("%Y-%m-%d"), games=POWER_GAMES,
                recent_games=RECENT_GAMES, recent_weight=RECENT_WEIGHT, blowout_weight=BLOWOUT_WEIGHT,
                ready=min(team["g"] for team in teams) >= POWER_GAMES, built=built,
                average_team={side: round(value, 2) for side, value in average.items()})'''),
    ('''        teams.append(dict(tm=tm, name=rows.name.iloc[0], g=int(rows.g.iloc[0]),
                          **{side: round(value, 2) for side, value in zip("odt", sums)}, players=players))''',
     '''        teams.append(dict(tm=tm, name=rows.name.iloc[0], g=int(rows.g.iloc[0]),
                          **{side: value for side, value in zip("odt", sums)}, players=players))'''),
])
edit("tests/test_site.py", [
    ('''    assert home["o"] == pytest.approx(round(4 + 80 / 190 + 3 * 110 / 190, 2))''',
     '''    plain = 4 + 80 / 190 + 3 * 110 / 190                             # the away team's ratings are all 0
    # each team against the average team, here half the home team's plain sum (2026-10-30)
    assert home["o"] == pytest.approx(round(plain / 2, 2))
    assert document["meta"]["average_team"]["o"] == pytest.approx(round(plain / 2, 2))'''),
])
TEST = '''

def test_the_share_played_counts_the_games_the_stints_hold(tmp_path, capsys):
    """2026-10-30: a game whose play-by-play failed is in the game log with no stints and is not rated; the share
    counts the games rated and names the others."""
    cfg = _write(tmp_path, 2026, _gamelog(16))
    cfg = {**cfg, "first_season": 2026, "last_season": 2027, "paths": {**cfg["paths"], "stints": "data/stints"}}
    games = [f"00226000{k:02d}" for k in range(1, 5)]
    pd.DataFrame({"GAME_ID": games}).to_parquet(tmp_path / "data" / "raw" / "gamelog" / "2027_RS.parquet")
    stints = tmp_path / "data" / "stints"
    stints.mkdir(parents=True)
    pd.DataFrame({"game_id": games[:3]}).to_parquet(stints / "2027_RS.parquet")
    assert S.share_played(2027, cfg) == pytest.approx(3 / S.SCHEDULED_GAMES)
    assert games[3] in capsys.readouterr().out
    (stints / "2027_RS.parquet").unlink()
    assert S.share_played(2027, cfg) == pytest.approx(4 / S.SCHEDULED_GAMES)
    assert S.share_played(2026, cfg) == 1.0
'''
with open("tests/test_seasons.py", "a", encoding="utf-8", newline="\n") as handle:
    handle.write(TEST)
print("ok")
