"""The Power page's one rating: its weights by point in the season, its last-10 stats and its rows."""

from __future__ import annotations

import pandas as pd
import pytest

from openrapm import power_forecast as forecast
from openrapm.commands import site

KNOTS = [
    {"share": 0.10, "offense_multiplier": 1.2, "defense_multiplier": 1.3, "stats": {"tov_o": -0.04, "tov_d": 0.02}},
    {"share": 0.25, "offense_multiplier": 1.1, "defense_multiplier": 1.0, "stats": {"tov_o": -0.01, "tov_d": 0.08}},
    {"share": 0.50, "offense_multiplier": 1.0, "defense_multiplier": 0.9, "stats": {"tov_o": 0.0, "tov_d": 0.0}},
]


def test_the_weights_are_the_straight_line_between_the_knots_and_hold_outside_them():
    middle = forecast.weights_at(0.175, KNOTS)
    assert middle["offense_multiplier"] == pytest.approx(1.15)
    assert middle["defense_multiplier"] == pytest.approx(1.15)
    assert middle["stats"]["tov_o"] == pytest.approx(-0.025)
    assert forecast.weights_at(0.05, KNOTS)["offense_multiplier"] == pytest.approx(1.2)
    late = forecast.weights_at(0.9, KNOTS)
    assert late["offense_multiplier"] == pytest.approx(1.0) and late["stats"] == {"tov_d": 0.0, "tov_o": 0.0}


def test_the_shares_count_each_game_once_before_a_quarter_of_the_season():
    assert forecast.flat_shares(0.12) is True
    assert forecast.flat_shares(0.25) is False


def games_for(n: int, tm: str, **counts) -> pd.DataFrame:
    """`n` games of one team, every count 0 unless given, the opponent's too."""
    base = {f"{count}_{end}": 0.0 for count in forecast.COUNTS for end in "od"}
    base.update(
        poss_o=100.0,
        poss_d=100.0,
        reb_chance_o=50.0,
        reb_chance_d=50.0,
        fta_o=20.0,
        fta_d=20.0,
        fga_rim_o=1.0,
        fga_rim_d=1.0,
        fga_smid_o=1.0,
        fga_smid_d=1.0,
        fga_lmid_o=1.0,
        fga_lmid_d=1.0,
        fga_thr_o=1.0,
        fga_thr_d=1.0,
        fga_lthr_o=1.0,
        fga_lthr_d=1.0,
    )
    base.update(counts)
    dates = pd.date_range("2026-10-20", periods=n)
    return pd.DataFrame({"game_id": [f"{tm}{i}" for i in range(n)], "game_date": dates, "tm": tm, **base})


def test_the_stats_read_only_the_last_ten_games_and_center_over_the_teams():
    old = games_for(5, "AAA", tov_o=100.0)
    new = games_for(10, "AAA", tov_o=10.0).assign(game_date=lambda f: f.game_date + pd.Timedelta(days=30))
    other = games_for(10, "BBB", tov_o=20.0)
    stats = forecast.last_ten_stats(pd.concat([old, new, other], ignore_index=True))
    assert stats.tov_o["AAA"] == pytest.approx(-5.0) and stats.tov_o["BBB"] == pytest.approx(5.0)
    assert stats.tov_o.sum() == pytest.approx(0.0)


def test_makes_over_expected_are_the_makes_less_what_the_league_rate_expects():
    good = games_for(10, "AAA", fga_rim_o=30.0, fgm_rim_o=21.0, fga_rim_d=10.0, fgm_rim_d=5.0)
    poor = games_for(10, "BBB", fga_rim_o=10.0, fgm_rim_o=5.0, fga_rim_d=30.0, fgm_rim_d=21.0)
    stats = forecast.last_ten_stats(pd.concat([good, poor], ignore_index=True))
    rate = (210.0 + 50.0 + 50.0 + 210.0) / (300.0 + 100.0 + 100.0 + 300.0)  # 0.65: both teams' makes at the rim
    expected_over = (210.0 - rate * 300.0) * 100 / 1000
    other_over = (50.0 - rate * 100.0) * 100 / 1000
    gap = (expected_over - other_over) / 2
    assert stats.makes_over_rim_o["AAA"] == pytest.approx(gap)


def test_each_end_reads_its_own_stats():
    stats = pd.DataFrame({"tov_o": [2.0, -2.0], "tov_d": [1.0, -1.0]}, index=["AAA", "BBB"])
    parts = forecast.stats_parts(stats, {"stats": {"tov_o": -0.5, "tov_d": 0.25}})
    assert parts.o["AAA"] == pytest.approx(-1.0) and parts.d["AAA"] == pytest.approx(0.25)


def document_with(parts: pd.DataFrame):
    slots = {f"{side}{i}": [10 * (side == "a") + i] * 2 for side in "ha" for i in range(1, 6)}
    stints = pd.DataFrame(
        {
            "game_id": ["g1", "g2"],
            "game_date": pd.to_datetime(["2026-10-20", "2026-10-22"]),
            "poss_h": [50, 50],
            "poss_a": [50, 50],
            "is_gt": [False, False],
            **slots,
        }
    )
    ids = [*range(1, 6), *range(11, 16)]
    teams = pd.DataFrame(
        [
            (game, pid, "HOM" if pid < 10 else "AWY", "Home" if pid < 10 else "Away")
            for game in ("g1", "g2")
            for pid in ids
        ],
        columns=["game_id", "player_id", "tm", "name"],
    )
    games = site.player_game_possessions(stints, teams)
    table = pd.DataFrame(
        {
            "season": 2027,
            "player_id": ids,
            "player_name": [f"P{i}" for i in ids],
            "rating_off": [1.0] * 5 + [-1.0] * 5,
            "rating_def": 0.5,
            "rating_total": [1.5] * 5 + [-0.5] * 5,
            "poss_season": 500,
        }
    )
    return site.power_document(
        table,
        games,
        2027,
        "2026-11-09",
        shares=site.team_shares(games, 10, recent_weight=1.0),
        forecast=dict(offense=1.2, defense=1.5, parts=parts, share=0.12, flat=True),
    )


def test_the_one_rating_is_power_at_its_multipliers_plus_the_stats_and_the_rows_add_up():
    parts = pd.DataFrame({"o": [0.4, -0.4], "d": [0.1, -0.1]}, index=["HOM", "AWY"])
    document = document_with(parts)
    home = next(team for team in document["teams"] if team["tm"] == "HOM")
    assert home["o"] == pytest.approx(1.2 * 5 + 0.4) and home["d"] == pytest.approx(0.1)
    assert home["t"] == pytest.approx(home["o"] + home["d"])
    assert home["players"][0] == ["Last 10 Performance Over Expected", None, None, 0.4, 0.1, 0.5]
    players = home["players"][1:]
    assert sum(row[2] / 100 * row[3] for row in players) + home["players"][0][3] == pytest.approx(home["o"], abs=0.03)
    assert document["meta"]["forecast"] is True and document["meta"]["recent_weight"] == 1.0


def test_with_the_stats_faded_no_stats_row_is_shown():
    document = document_with(pd.DataFrame({"o": [0.0, 0.0], "d": [0.0, 0.0]}, index=["HOM", "AWY"]))
    home = next(team for team in document["teams"] if team["tm"] == "HOM")
    assert all(row[1] is not None for row in home["players"])


def test_the_power_tab_shows_once_every_team_has_played_eight_games():
    def document_after(n: int) -> dict:
        slots = {f"{side}{i}": [10 * (side == "a") + i] * n for side in "ha" for i in range(1, 6)}
        ids = [*range(1, 6), *range(11, 16)]
        names = [f"g{k}" for k in range(n)]
        stints = pd.DataFrame({"game_id": names, "game_date": pd.date_range("2026-10-20", periods=n), "poss_h": 50,
                               "poss_a": 50, "is_gt": False, **slots})
        teams = pd.DataFrame([(game, pid, "HOM" if pid < 10 else "AWY", "Home" if pid < 10 else "Away")
                              for game in names for pid in ids], columns=["game_id", "player_id", "tm", "name"])
        table = pd.DataFrame({"season": 2027, "player_id": ids, "player_name": "P", "rating_off": 0.0,
                              "rating_def": 0.0,
                              "rating_total": 0.0, "poss_season": 500})
        return site.power_document(table, site.player_game_possessions(stints, teams), 2027, "2026-11-09")

    assert document_after(7)["meta"]["ready"] is False
    assert document_after(8)["meta"]["ready"] is True
