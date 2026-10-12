"""The design's own rules (src/openrapm/design.py), each checked on stints written out by hand.

The simulated tests reach the design only through a fit, and the tests on the scraped data skip on a clone without
it, so this file is where the rules design.py states are pinned directly: a row's response and weight, the home-court
column, which rows a target drops, the margin's clip, the garbage-time weight, the order of the rows and of the
pieces, the player columns, and the error a lineup with a player in it twice raises.

Every stint below is a version of one: game "g1" of 2026's regular season, the home five (players 1-5) against the
away five (players 11-15), 8 points on 6 possessions against 5 on 6, at a tied score with the whole game left.
"""
import numpy as np
import pandas as pd
import pytest

from openrapm.design import AWAY_SLOTS, HOME_SLOTS, assemble, pieces_from_frames

HOME_FIVE, AWAY_FIVE = [1, 2, 3, 4, 5], [11, 12, 13, 14, 15]


def stint(**changes) -> dict:
    """The stint of the module docstring, with `changes` applied."""
    row = dict(game_id="g1", season=2026, phase="RS", series_id="", period=1, poss_h=6, poss_a=6, pts_h=8, pts_a=5,
               margin_h=0, frac_rem=1.0, is_gt=False, neutral=False,
               **dict(zip(HOME_SLOTS + AWAY_SLOTS, HOME_FIVE + AWAY_FIVE)))
    return {**row, **changes}


def box_scores(stints: pd.DataFrame) -> pd.DataFrame:
    """One box-score line per player and game, one steal each: every design carries a box score, these tests do
    not read it."""
    lines = pd.concat([stints[["game_id", "season", "phase"]].assign(player_id=stints[slot])
                       for slot in HOME_SLOTS + AWAY_SLOTS])
    return lines.drop_duplicates(["game_id", "player_id"]).assign(stl=1)


def design_of(stint_rows: list, cfg: dict, target: str = "pts"):
    """The design of the stints `stint_rows`, through the one assembly."""
    stints = pd.DataFrame(stint_rows)
    return assemble(pieces_from_frames(stints, box_scores(stints), ["stl"]), cfg, target)


def context(design, name: str) -> list:
    """One context column of the design matrix, one value per row."""
    return design.X[:, design.spec.context_column(name)].toarray().ravel().tolist()


# ------------------------------------------------------------------------------------------------ one stint, two rows
def test_the_docstring_example(small_cfg):
    """8 points on 6 possessions against 5 on 6: y = 100 x 8 / 6 = 133.3 and 100 x 5 / 6 = 83.3, both weighing 6."""
    design = design_of([stint()], small_cfg())
    assert design.y == pytest.approx([100 * 8 / 6, 100 * 5 / 6])
    assert round(design.y[0], 1) == 133.3 and round(design.y[1], 1) == 83.3
    assert design.w.tolist() == [6.0, 6.0]
    assert context(design, "home") == [1.0, -1.0]
    assert design.rows.is_home_off.tolist() == [True, False]
    assert context(design, "int_2026") == [1.0, 1.0]


def test_each_row_holds_its_five_on_offence_and_its_five_on_defence(small_cfg):
    design = design_of([stint()], small_cfg())
    column = dict(zip(design.spec.players.player_id, design.spec.players.player_column))
    offence = design.X[:, design.spec.offense_columns].toarray()
    defence = design.X[:, design.spec.defense_columns].toarray()
    assert np.flatnonzero(offence[0]).tolist() == [column[player] for player in HOME_FIVE]
    assert np.flatnonzero(defence[0]).tolist() == [column[player] for player in AWAY_FIVE]
    assert np.flatnonzero(offence[1]).tolist() == [column[player] for player in AWAY_FIVE]
    assert np.flatnonzero(defence[1]).tolist() == [column[player] for player in HOME_FIVE]


def test_the_matrix_is_its_parts_side_by_side(small_cfg):
    """X = [Z_O | Z_D | C | game]: the blocks a fit reads (`parts`) are the matrix's, and the last column is the
    game number."""
    design = design_of([stint(), stint(game_id="g2"), stint(game_id="g2", pts_h=0)], small_cfg())
    matrix, n_players, n_context = design.X.toarray(), design.spec.n_players, len(design.spec.context_names)
    assert np.array_equal(matrix[:, :2 * n_players], design.parts["Z"].toarray())
    assert np.array_equal(matrix[:, 2 * n_players:2 * n_players + n_context], design.parts["F"])
    assert matrix[:, -1].tolist() == design.parts["game_idx"].tolist() == [0, 1, 1, 0, 1, 1]


def test_a_neutral_site_has_no_home_court(small_cfg):
    """Home is +1 for the home offence, -1 for the away offence and 0 for both at a neutral site; po_home is
    playoffs x home."""
    stints = [stint(neutral=True), stint(game_id="g2", phase="PO", series_id="s1"),
              stint(game_id="g3", phase="PO", series_id="s1", neutral=True)]
    design = design_of(stints, small_cfg())
    assert context(design, "home") == [0.0, 1.0, 0.0, 0.0, -1.0, 0.0]
    assert context(design, "is_po") == [0.0, 1.0, 1.0, 0.0, 1.0, 1.0]
    assert context(design, "po_home") == [0.0, 1.0, 0.0, 0.0, -1.0, 0.0]


# ------------------------------------------------------------------------------------------------ the target's rows
# Stint 1: the visitors take no shot and have no rebound chance.  Stint 2: both teams do.
SHOOTING = [dict(fgm_h=2, fg3m_h=1, fga_h=4, reb_cont_h=1, reb_chance_h=2,
                 fgm_a=0, fg3m_a=0, fga_a=0, reb_cont_a=0, reb_chance_a=0),
            dict(game_id="g2", fgm_h=3, fg3m_h=0, fga_h=5, reb_cont_h=0, reb_chance_h=3,
                 fgm_a=1, fg3m_a=1, fga_a=2, reb_cont_a=2, reb_chance_a=4)]


@pytest.mark.parametrize("target, response, weight", [
    ("efg", [100 * 2.5 / 4, 100 * 3 / 5, 100 * 1.5 / 2], [4.0, 5.0, 2.0]),     # (makes + threes / 2) / attempts
    ("oreb", [100 * 1 / 2, 0.0, 100 * 2 / 4], [2.0, 3.0, 4.0]),                # own rebounds / rebound chances
])
def test_a_row_whose_denominator_is_0_is_dropped(small_cfg, target, response, weight):
    """The visitors' row of stint 1 has no attempts and no rebound chances, so a rate target keeps three rows of
    four; points per 100 keeps all four.  Both designs have the same players and context columns."""
    stints = [stint(**counts) for counts in SHOOTING]
    design = design_of(stints, small_cfg(), target)
    assert design.rows.stint.tolist() == [0, 1, 1]
    assert design.rows.is_home_off.tolist() == [True, True, False]
    assert design.y == pytest.approx(response) and design.w.tolist() == weight
    assert design.rows.poss.tolist() == [6, 6, 6]
    assert design.counters.fga.tolist() == [4.0, 5.0, 2.0]          # one line per kept row, the offence's counts
    points = design_of(stints, small_cfg())
    assert len(points.y) == 4
    assert design.spec.players.equals(points.spec.players) and design.spec.context_names == points.spec.context_names


def test_the_margin_is_the_offences_clipped_and_times_the_share_of_the_game_left(small_cfg):
    stints = [stint(margin_h=30, frac_rem=0.5), stint(game_id="g2", margin_h=-10, frac_rem=0.25)]
    design = design_of(stints, small_cfg())                                     # margin_clip 25, the shipped value
    assert context(design, "margin") == [25.0, -10.0, -25.0, 10.0]
    assert context(design, "margin_frem") == [12.5, -2.5, -12.5, 2.5]
    tighter = design_of(stints, small_cfg(margin_clip=5))
    assert context(tighter, "margin") == [5.0, -5.0, -5.0, 5.0]
    assert context(tighter, "margin_frem") == [2.5, -1.25, -2.5, 1.25]


def test_a_garbage_time_row_weighs_gt_weight_times_its_denominator(small_cfg):
    stints = [stint(), stint(game_id="g2", is_gt=True)]
    assert design_of(stints, small_cfg()).w.tolist() == [6.0, 6.0, 6.0, 6.0]         # gt_weight 1.0, the shipped value
    design = design_of(stints, small_cfg(gt_weight=0.5))
    assert design.w.tolist() == [6.0, 3.0, 6.0, 3.0]
    assert context(design, "is_gt") == [0.0, 1.0, 0.0, 1.0]
    assert design.y == pytest.approx(100 / 6 * np.array([8, 8, 5, 5]))


def test_a_player_twice_in_one_five_is_an_error(small_cfg):
    with pytest.raises(ValueError, match="duplicate player ids inside a lineup"):
        design_of([stint(h2=1)], small_cfg())


# ------------------------------------------------------------------------------------------------ the order of rows
def test_every_home_offence_row_comes_before_every_away_offence_row(small_cfg):
    stints = [stint(game_id=game, pts_h=points, pts_a=points + 1)
              for game, points in (("g1", 6), ("g2", 9), ("g3", 12))]
    design = design_of(stints, small_cfg())
    assert design.rows.is_home_off.tolist() == [True, True, True, False, False, False]
    assert design.rows.stint.tolist() == [0, 1, 2, 0, 1, 2]
    assert design.y == pytest.approx(100 / 6 * np.array([6, 9, 12, 7, 10, 13]))


def several_pieces() -> list:
    """2025's regular season (game a1), 2026's (b1, b2) and 2026's playoffs (c1, c2, one series): three pieces.  The
    home team scores 1, 2, ... 5 points, takes 10, 11, ... 14 shots, and the visitors take 20, 21, ... 24."""
    games = [("a1", 2025, "RS", ""), ("b1", 2026, "RS", ""), ("b2", 2026, "RS", ""), ("c1", 2026, "PO", "s1"),
             ("c2", 2026, "PO", "s1")]
    return [stint(game_id=game, season=season, phase=phase, series_id=series, pts_h=number + 1, fga_h=10 + number,
                  fga_a=20 + number)
            for number, (game, season, phase, series) in enumerate(games)]


def test_pieces_are_assembled_season_by_season_regular_season_first(small_cfg):
    """Each stint keeps its place, the games are numbered across the design, and each game keeps its half: regular-
    season games alternate A, B within a season, playoff games within a series."""
    design = design_of(several_pieces(), small_cfg())
    home = design.rows.is_home_off.to_numpy()
    rows = design.rows[home]
    assert rows.stint.tolist() == rows.game_idx.tolist() == [0, 1, 2, 3, 4]
    assert design.y[home] == pytest.approx(100 / 6 * np.arange(1, 6))
    assert rows.season.tolist() == [2025, 2026, 2026, 2026, 2026]
    assert rows.phase.tolist() == ["RS", "RS", "RS", "PO", "PO"]
    assert rows.half.tolist() == ["A", "A", "B", "A", "B"]
    assert design.groups[home].tolist() == ["G:a1", "G:b1", "G:b2", "S:s1", "S:s1"]
    assert design.games.game_id.tolist() == ["a1", "b1", "b2", "c1", "c2"]
    assert np.array(context(design, "int_2025"))[home].tolist() == [1.0, 0.0, 0.0, 0.0, 0.0]
    assert design.counters.fga.tolist() == [10.0, 11.0, 12.0, 13.0, 14.0, 20.0, 21.0, 22.0, 23.0, 24.0]
    assert design.counters.pid_s1.tolist() == [1] * 5 + [11] * 5


def test_the_order_the_pieces_arrive_in_does_not_matter(small_cfg):
    """`pieces_from_frames` sorts the pieces by season and phase, so the playoffs listed first change nothing."""
    stints = several_pieces()
    in_order, playoffs_first = design_of(stints, small_cfg()), design_of(stints[3:] + stints[:3], small_cfg())
    assert np.array_equal(in_order.y, playoffs_first.y) and np.array_equal(in_order.w, playoffs_first.w)
    assert in_order.rows.equals(playoffs_first.rows)
    assert (in_order.X != playoffs_first.X).nnz == 0


# ------------------------------------------------------------------------------------------------ player columns
def test_one_column_per_player_spans_the_seasons(small_cfg):
    """Two 2025 stints (12 regular-season possessions each player), one 2026 regular-season stint (6) and two 2026
    playoff stints (12, which do not count): under player_unit "player" each player's column takes 2025, the season
    of his most regular-season possessions; under "player_season" each player-season is its own column."""
    stints = [stint(game_id="a1", season=2025), stint(game_id="a2", season=2025), stint(game_id="b1"),
              stint(game_id="c1", phase="PO", series_id="s1"), stint(game_id="c2", phase="PO", series_id="s1")]
    by_player = design_of(stints, small_cfg(player_unit="player"))
    assert by_player.spec.seasons == [2025, 2026] and by_player.spec.n_players == 10
    assert by_player.spec.season_of_player.tolist() == [0] * 10
    assert (by_player.spec.player_seasons.groupby("player_id").player_column.nunique() == 1).all()
    by_season = design_of(stints, small_cfg(player_unit="player_season"))
    assert by_season.spec.n_players == 20
    assert by_season.spec.season_of_player.tolist() == [0] * 10 + [1] * 10
