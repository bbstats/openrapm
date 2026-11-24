"""One-off edit of tests/test_site.py and tests/test_bias_site.py for the held-back last-10-seasons column."""
from pathlib import Path


def edit(path: str, pairs: list) -> None:
    p = Path(path)
    s = p.read_text(encoding="utf-8")
    for old, new in pairs:
        assert s.count(old) >= 1, (path, old[:100], s.count(old))
        s = s.replace(old, new)
    p.write_text(s, encoding="utf-8")


edit("tests/test_site.py", [
    ('''@pytest.fixture
def fake_games(monkeypatch):
    """Every season's team-games stood in for: 514 a season, the games' multipliers 1.4 and 2.1, and every stat's
    lean +0.25, standard error 0.5, for the season alone, and +0.10, standard error 0.05, for the seasons together.
    Returns the stand-in for one season's games and the calls the lean table got: (how many seasons, their z)."""
    calls = []

    def lean_table(seasons, zs, stats):
        calls.append((len(seasons), zs))
        value, error = (0.25, 0.5) if len(calls) % 2 == 1 else (0.10, 0.05)
        return pd.DataFrame({"stat": list(stats), "games": value, "games_se": error})

    monkeypatch.setattr(site, "lean_table", lean_table)
    monkeypatch.setattr(site, "game_multipliers", lambda games: {"offense": 1.4, "defense": 2.1})
    return SimpleNamespace(team_games=514), calls


def _document(fake_games, table=None, panel=None, pulls=None, extra=()):
    """`bias_document` of the made-up 2027, the seasons' column on 2027 and `extra` seasons (each the made-up 2027's
    table and panel rows under that season)."""
    world = _bias_world()
    table, panel, pulls = (world[0] if table is None else table, world[1] if panel is None else panel,
                           world[2] if pulls is None else pulls)
    seasons = {2027: (fake_games[0], panel)}
    for season in extra:
        seasons[season] = (fake_games[0], panel.assign(season=season))
        table = pd.concat([table, table[table.season == 2027].assign(season=season)], ignore_index=True)
    return site.bias_document(table, pulls, 2027, "2026-11-05", "2026-11-04", seasons), table
''', '''@pytest.fixture
def held_back():
    """The held-back games' leans as params/bias_held_back.json holds them: every stat +0.10, standard error 0.05,
    but made threes -0.034, 0.046, so a row is seen to keep its own number."""
    from openrapm.bias import STATS
    rows = {stat: [0.1, 0.05] for stat in STATS}
    rows["threes_made_per100"] = [-0.034, 0.046]
    return {"first": 2017, "last": 2026, "build": "rd143", "rows": rows}


def _document(held_back, table=None, panel=None, pulls=None):
    """`bias_document` of the made-up 2027 with `held_back`'s leans."""
    world = _bias_world()
    table, panel, pulls = (world[0] if table is None else table, world[1] if panel is None else panel,
                           world[2] if pulls is None else pulls)
    return site.bias_document(table, pulls, 2027, "2026-11-05", "2026-11-04", panel, held_back), table
'''),
    ('''def test_the_bias_document_counts_its_players_and_names_its_three_systems(fake_games):
    document, _ = _document(fake_games)''', '''def test_the_bias_document_counts_its_players_and_names_its_three_systems(held_back):
    document, _ = _document(held_back)'''),
    ('''    assert meta["games"] == {"team_games": 514, "multipliers": {"o": 1.4, "d": 2.1}}
    assert meta["games_seasons"] == {"first": 2027, "last": 2027, "team_games": 514}
''', '''    assert "games" not in meta and meta["games_seasons"] == {"first": 2017, "last": 2026, "build": "rd143"}
'''),
    ('''def test_the_seasons_game_results_column_is_the_seasons_alone_and_together(fake_games):
    """The season's column is asked of the season alone, its z in standard deviations of the 40 compared players; the
    seasons' column of every season together, each season's z in standard deviations of its own shown players (41
    here: the one xRAPM lacks counts).  Weighted by possessions, each z has mean 0 and spread 1 over those players."""
    from openrapm import bias
    document, table = _document(fake_games, extra=(2018, 2026))
    calls = fake_games[1]
    assert [count for count, _ in calls] == [1, 3]
    compared = table[(table.season == 2027) & (table.player_id <= 40)].set_index("player_id")
    shown = table[(table.season == 2027) & (table.poss_season >= 60)].set_index("player_id")
    for z, players in ((calls[0][1][0], compared), *((z, shown) for z in calls[1][1])):
        z = z.set_index("player_id")
        assert sorted(z.index) == list(range(1, 43))
        for stat in bias.STATS:
            assert bias.weighted_mean(z.loc[players.index, stat], players.poss_season) == pytest.approx(0, abs=1e-9)
            assert bias.weighted_sd(z.loc[players.index, stat], players.poss_season) == pytest.approx(1.0)
    assert document["meta"]["games_seasons"] == {"first": 2018, "last": 2027, "team_games": 3 * 514}
    for row in [dict(zip(site.BIAS_FIELDS, row)) for row in document["rows"]]:
        assert (row["g"], row["g_se"], row["gs"], row["gs_se"]) == (0.25, 0.5, 0.1, 0.05), row["label"]
''', '''def test_the_last_10_seasons_column_is_the_held_back_games_as_stored(held_back, tmp_path):
    """The column is each stat's held-back lean and its standard error at two decimals, as the params file holds
    them; a file missing a stat stops the command before anything is written."""
    document, _ = _document(held_back)
    rows = [dict(zip(site.BIAS_FIELDS, row)) for row in document["rows"]]
    assert (rows[0]["stat"], rows[0]["gs"], rows[0]["gs_se"]) == ("threes_made_per100", -0.03, 0.05)
    assert all((row["gs"], row["gs_se"]) == (0.1, 0.05) for row in rows[1:])
    path = tmp_path / "bias_held_back.json"
    path.write_text(json.dumps(held_back))
    assert site.held_back_leans(path) == held_back
    del held_back["rows"]["fouls_per100"]
    path.write_text(json.dumps(held_back))
    with pytest.raises(SystemExit, match="no held-back lean for fouls_per100"):
        site.held_back_leans(path)
'''),
    ('''def test_the_bias_rows_are_the_fifteen_stats_and_every_row_adds_up_at_two_decimals(fake_games):
    document, _ = _document(fake_games)''', '''def test_the_bias_rows_are_the_fifteen_stats_and_every_row_adds_up_at_two_decimals(held_back):
    document, _ = _document(held_back)'''),
    ('''def test_a_system_that_runs_against_openrapm_stops_the_document(fake_games):''',
     '''def test_a_system_that_runs_against_openrapm_stops_the_document(held_back):'''),
    ('''        _document(fake_games, table, panel, pulls)''', '''        _document(held_back, table, panel, pulls)'''),
    ('''def bias_site(tmp_path, monkeypatch, fake_games):
    """`run_site(bias=True)` on the made-up 2027, with the same players rated in 2016 and 2018 too: the stored pulls,
    the panel and the game dates handed in, and the seasons whose team-games were asked for recorded."""''',
     '''def bias_site(tmp_path, monkeypatch, held_back):
    """`run_site(bias=True)` on the made-up 2027, with the same players rated in 2016 and 2018 too: the stored pulls,
    the panel and the game dates handed in, and the held-back leans in the params file."""'''),
    ('''    stints = pd.DataFrame({"game_date": ["2026-10-20", "2026-11-04"]})
    asked = []
''', '''    stints = pd.DataFrame({"game_date": ["2026-10-20", "2026-11-04"]})
    (tmp_path / "params").mkdir()
    (tmp_path / site.BIAS_HELD_BACK).write_text(json.dumps(held_back), encoding="utf-8")
'''),
    ('''    monkeypatch.setattr(site, "season_designs", lambda cfg, season: None)
    monkeypatch.setattr(site, "season_games",
                        lambda designs, table, season, teams_of: asked.append(season) or fake_games[0])
    cfg = {"_root": str(tmp_path), "paths": {"outputs": str(tmp_path / "outputs")}}
    return cfg, tmp_path / "site", pulls, asked''', '''    cfg = {"_root": str(tmp_path), "paths": {"outputs": str(tmp_path / "outputs")}}
    return cfg, tmp_path / "site", pulls'''),
    ('''    """The seasons' column reads the season and the 9 before it that are rated: 2018 here, not 2016."""
    cfg, out_dir, _, asked = bias_site
    assert site.run_site(cfg, out_dir=out_dir, built="2026-11-05", bias=True) == 0
    assert asked == [2018, 2027]
''', '''    cfg, out_dir, _ = bias_site
    assert site.run_site(cfg, out_dir=out_dir, built="2026-11-05", bias=True) == 0
'''),
    ('''    assert "game results on 514 team-games, OpenRAPM multiplied by 1.400 on offense and 2.100 on defense" in out
    assert "game results of 2018 to 2027 on 1028 team-games" in out
''', '''    assert "held-back games of 2017 to 2026, build rd143" in out
'''),
    ('''    cfg, out_dir, pulls, _ = bias_site''', '''    cfg, out_dir, pulls = bias_site'''),
])
edit("tests/test_bias_site.py", [
    ('''def test_every_row_has_its_leans_against_the_game_results_and_the_file_says_what_they_were_read_on(document):
    meta = document["meta"]
    games, seasons = meta["games"], meta["games_seasons"]
    assert games["team_games"] > 0 and games["multipliers"]["o"] > 0 and games["multipliers"]["d"] > 0
    assert seasons["last"] == meta["season"] and seasons["last"] - seasons["first"] == 9
    assert seasons["team_games"] > 8 * games["team_games"]
    for row in document["rows"]:
        stat = dict(zip(BIAS_FIELDS, row))
        for key in ("g", "gs"):
            assert stat[key] is not None and stat[key + "_se"] > 0, (key, stat["label"])
            assert round(stat[key], 2) == stat[key], (key, stat["label"])
''', '''def test_every_row_has_its_lean_against_the_held_back_games_and_the_file_says_what_they_were_read_on(document):
    """Ten finished seasons, before the one the page opens on, and the build whose held-back games they are."""
    seasons = document["meta"]["games_seasons"]
    assert seasons["last"] < document["meta"]["season"] and seasons["last"] - seasons["first"] == 9
    assert seasons["build"]
    for row in document["rows"]:
        stat = dict(zip(BIAS_FIELDS, row))
        assert stat["gs"] is not None and stat["gs_se"] > 0, stat["label"]
        assert round(stat["gs"], 2) == stat["gs"], stat["label"]
'''),
    ('''def test_the_page_names_its_columns_for_the_season_and_the_last_10_seasons():
    """The owner, 2026-11-24: "the first one should say 'vs consensus' and the 2nd one should now say 'vs game
    results'", then "Clarify the other 2 columns as just being '27" and "last 10 seasons (inclusive of current)"."""
    page = (DOCS / "bias.html").read_text(encoding="utf-8")
    assert '["t", m => `vs consensus ${short(m.season)}`' in page
    assert '["g", m => `vs game results ${short(m.season)}`' in page
    assert '["gs", () => "vs game results, last 10 seasons"' in page
''', '''def test_the_page_names_its_columns_for_the_season_and_the_last_10_seasons():
    """The owner, 2026-11-24: "the first one should say 'vs consensus'", "Clarify the other 2 columns as just being
    '27", then "Held-back, drop ’27": the season's game-results column is gone."""
    page = (DOCS / "bias.html").read_text(encoding="utf-8")
    assert '["t", m => `vs consensus ${short(m.season)}`' in page
    assert '["g", ' not in page
    assert '["gs", () => "vs game results, last 10 seasons"' in page
'''),
])
print("edited")
