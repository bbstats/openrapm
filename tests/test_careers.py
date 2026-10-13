"""The careers before the data starts (src/openrapm/data/careers.py, experiment 57): who is pulled, how a season
before 1997 becomes a roles-table row, and that the career columns counted from the debut count it."""
import numpy as np
import pandas as pd
import pytest

from openrapm.data import careers
from openrapm.data.roles import career_inputs, player_season_inputs


def _career(rows):
    """A PlayerCareerStats frame: (season id, team, games, starts, minutes, age, phase) per row, player 920."""
    return pd.DataFrame([dict(PLAYER_ID=920, SEASON_ID=season, LEAGUE_ID="00", TEAM_ID=team, GP=games, GS=starts,
                              MIN=minutes, PLAYER_AGE=age, phase=phase)
                         for season, team, games, starts, minutes, age, phase in rows])


def test_a_season_before_the_first_is_a_roles_row_with_its_playoffs_and_teams():
    frame = _career([("1993-94", 1610612756, 82, 82, 2825, 30.0, "RS"),
                     ("1993-94", 1610612756, 10, 10, 300, 30.0, "PO"),
                     ("1996-97", 1610612756, 27, 20, 551, 33.0, "RS"),
                     ("1996-97", 1610612742, 56, 50, 1943, 33.0, "RS"),
                     ("1996-97", 0, 83, 70, 2494, 33.0, "RS")])            # the endpoint's TOT row
    rows = careers.rows_before(frame, first_season=1997, per_minute=4.0)
    assert list(rows.columns) == careers.ROLE_COLUMNS
    assert rows.season.tolist() == [1994], "1996-97 is the first season itself, and TOT is no team"
    row = rows.iloc[0]
    assert (row.games, row.starts, row.minutes) == (92.0, 92.0, 3125.0)
    assert row.poss_on == pytest.approx(3125.0 * 4.0) and np.isnan(row.team_poss) and row.age == 30.0
    traded = careers.rows_before(frame, first_season=1998, per_minute=4.0)
    assert traded[traded.season == 1997].team_id.tolist() == [1610612742, 1610612756]


def test_who_is_pulled():
    roles = pd.DataFrame({"player_id": [920, 977, 5, 6, 7, 8], "season": [1997, 1997, 1999, 1999, 2003, 1997],
                          "age": [33.0, 18.0, 22.0, 27.0, 30.0, 25.0], "games": [83.0, 71.0, 50.0, 40.0, 60.0, 0.0]})
    drafted = pd.Series({920: 1985.0, 977: 1996.0, 5: np.nan, 6: np.nan, 7: 1990.0, 8: np.nan})
    # 920 drafted 1985; 977 drafted 1996 (nothing before 1997); 5 an undrafted 22-year-old first seen in 1999 (a
    # rookie); 6 undrafted, first seen at 27; 7 first seen after 2002; 8 never played a game
    assert careers.players_to_pull(roles, drafted, 1997) == [6, 920]


def test_the_career_columns_count_the_seasons_before_once_they_are_added():
    """A.C. Green's 1997: the roles table alone counts no season before it; with his rows before it, eleven."""
    roles = pd.DataFrame({"player_id": [920], "season": [1997], "team_id": [1610612742], "games": [83.0],
                          "starts": [70.0], "minutes": [2494.0], "poss_on": [9000.0], "team_poss": [16000.0],
                          "age": [33.0]})
    before = careers.rows_before(_career([(f"{year - 1}-{str(year)[-2:]}", 1610612747, 82, 82, 2000, year - 1964.0,
                                            "RS") for year in range(1986, 1997)]), 1997, 4.0)
    alone = career_inputs(player_season_inputs(roles), 1997, [920], age=[33.0])
    counted = career_inputs(player_season_inputs(pd.concat([before, roles], ignore_index=True)), 1997, [920],
                            age=[33.0])
    assert alone.seasons_before[0] == 0.0 and counted.seasons_before[0] == 11.0
    assert counted.career_possessions_before_k[0] == pytest.approx(11 * 2000 * 4.0 / 1000)
    assert counted.age_at_first_season[0] == 22.0


def test_a_missing_career_names_the_command(tmp_path):
    cfg = {"_root": str(tmp_path), "first_season": 1997,
           "paths": {"raw": str(tmp_path / "raw"), "cache": str(tmp_path / "cache")}}
    (tmp_path / "raw" / "bio").mkdir(parents=True)
    pd.DataFrame({"PLAYER_ID": [920], "DRAFT_YEAR": ["1985"]}).to_parquet(tmp_path / "raw" / "bio" / "1997_RS.parquet")
    roles = pd.DataFrame({"player_id": [920], "season": [1997], "age": [33.0], "games": [83.0], "minutes": [2494.0],
                          "poss_on": [9000.0]})
    with pytest.raises(FileNotFoundError, match="openrapm data careers"):
        careers.build_careers_before(cfg, roles)


def test_offline_a_missing_career_is_refused(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENRAPM_OFFLINE", "1")
    cfg = {"_root": str(tmp_path), "paths": {"raw": str(tmp_path / "raw")}}
    with pytest.raises(RuntimeError, match="career of player 920"):
        careers.fetch_career(920, cfg)
