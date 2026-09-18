"""The trade set (src/eracoef/tradeset.py) against simulated seasons with a known truth.

Four things have to hold before any number this module produces means anything:

  1. pooling the stints into team-games loses nothing -- the shares are shares, the response is the
     team's actual rate, and the kept controls were constant inside a team-game to begin with;
  2. the offset goes in with the sign the design's defensive columns carry, not the sign a rating
     table is stored in;
  3. a player whose rating is deliberately wrong is corrected in the right direction, by roughly the
     right amount, out of the games his team played without him;
  4. a rating that is uniformly too wide is read as a SCALE and not as a correction to every player.
"""
import numpy as np
import pandas as pd
import pytest

from eracoef import tradeset as ts_mod
from eracoef.priorridge import MAE_SCALE
from eracoef.design import FEATURES, build_design
from eracoef.simulate import simulate

CFG = {"gt_weight": 1.0, "margin_clip": 25, "low_poss_threshold": 500, "features": FEATURES,
       "first_season": 2001, "last_season": 2003, "player_unit": "window"}


@pytest.fixture(scope="module")
def world():
    # big enough that the regression is over-determined the way the real one is: three seasons of
    # 200 games is 1,200 team-games against 240 player columns, about the five-to-one the real block
    # runs at.  At the simulator's defaults it is two-to-one and nothing is recoverable.
    sim = simulate(n_seasons=3, n_teams=10, players_per_team=12, games_per_season=200,
                   stints_per_game=(20, 30), rho=0.9, turnover=0.35, seed=17, eps_var=0.2, leak=False)
    st, box = sim["stints"], sim["box"]
    wd = build_design(st, box, FEATURES, CFG)
    return sim, wd, ts_mod.team_game_design(wd)


def test_shares_sum_to_five_on_every_row(world):
    _, _, ts = world
    n = ts.n_players
    for side, block in (("offense", ts.Z[:, :n]), ("defense", ts.Z[:, n:])):
        total = np.asarray(block.sum(axis=1)).ravel()
        assert np.allclose(total, 5.0, atol=1e-9), side
    assert ts.Z.min() >= 0.0 and ts.Z.max() <= 1.0 + 1e-9


def test_pooling_reproduces_the_team_game_totals(world):
    """The pooled response times the pooled possessions is the team's points in that game."""
    _, wd, ts = world
    rows = wd.rows
    manual = (pd.DataFrame({"g": rows.game_idx, "h": rows.is_home_off,
                            "points": wd.y * rows.den / 100.0, "poss": rows.den})
              .groupby(["g", "h"], sort=True).sum())
    order = pd.MultiIndex.from_arrays([ts.keys.game_idx, ts.keys.is_home_off])
    manual = manual.reindex(order)
    assert np.allclose(ts.w, manual.poss.to_numpy(), rtol=0, atol=1e-9)
    assert np.allclose(ts.y * ts.w / 100.0, manual.points.to_numpy(), rtol=0, atol=1e-8)


def test_a_lineup_sum_of_ratings_survives_pooling(world):
    """Pooling and taking lineup sums commute, which is why the rating can be subtracted after pooling."""
    _, wd, ts = world
    rng = np.random.default_rng(0)
    rating = rng.normal(0, 2.0, 2 * ts.n_players)
    stint_contribution = np.asarray(wd.X[:, :2 * wd.spec.n_ps] @ rating).ravel()
    den = wd.rows.den.to_numpy(float)
    pooled = (pd.DataFrame({"g": wd.rows.game_idx, "h": wd.rows.is_home_off,
                            "num": stint_contribution * den, "den": den})
              .groupby(["g", "h"], sort=True).sum())
    pooled = pooled.reindex(pd.MultiIndex.from_arrays([ts.keys.game_idx, ts.keys.is_home_off]))
    assert np.allclose(ts.Z @ rating, (pooled.num / pooled.den).to_numpy(), atol=1e-9)


def test_only_controls_constant_within_a_team_game_are_kept(world):
    _, wd, ts = world
    assert "home" in ts.control_names and any(c.startswith("int_") for c in ts.control_names)
    assert "is_gt" not in ts.control_names and "margin" not in ts.control_names
    fixed = np.asarray(wd.X[:, 2 * wd.spec.n_ps:2 * wd.spec.n_ps + len(wd.spec.f_names)].todense())
    group = pd.DataFrame({"g": wd.rows.game_idx, "h": wd.rows.is_home_off})
    for name in ts.control_names:
        column = fixed[:, wd.spec.f_names.index(name)]
        spread = pd.Series(column).groupby([group.g, group.h]).nunique()
        assert spread.max() == 1, f"{name} varies inside a team-game"


def test_offset_takes_the_raw_sign(world):
    _, _, ts = world
    ids = ts.player_ids[:3]
    ratings = pd.DataFrame({"player_id": ids, "o": [1.0, 2.0, 3.0], "d": [-1.0, -2.0, -3.0],
                            "poss": [100.0, 200.0, 300.0]})
    offset = ts_mod.offset_vector(ids, ratings)
    assert np.allclose(offset[:3], [1.0, 2.0, 3.0])
    assert np.allclose(offset[3:], [-1.0, -2.0, -3.0])
    # a player the table has no row for takes the fill, on both sides
    filled = ts_mod.offset_vector(np.append(ids, -99), ratings, fill_o=0.5, fill_d=0.25)
    assert filled[3] == 0.5 and filled[-1] == 0.25


def test_replacement_fill_matches_the_holdout_rule(world):
    ratings = pd.DataFrame({"player_id": [1, 2, 3], "o": [-2.0, -4.0, 8.0], "d": [1.0, 3.0, -9.0],
                            "poss": [100.0, 300.0, 5000.0]})
    fill_o, fill_d = ts_mod.replacement_fill(ratings, max_poss=500.0, shrink=0.25)
    assert fill_o == pytest.approx(0.25 * np.average([-2.0, -4.0], weights=[100.0, 300.0]))
    assert fill_d == pytest.approx(0.25 * np.average([1.0, 3.0], weights=[100.0, 300.0]))


def _truth_offset(sim, ts):
    """The simulator's own player impacts, in the design's column order and the model's raw sign."""
    truth = (sim["truth"]["ps"].groupby("player_id")[["impact_O", "impact_D"]].mean().reset_index()
             .rename(columns={"impact_O": "o", "impact_D": "d"}))
    return ts_mod.offset_vector(ts.player_ids, truth)


def test_a_deliberately_wrong_rating_is_corrected_in_the_right_direction(world):
    """Break every rating by a known amount; alpha must point back at the truth, not anywhere else.

    Perturbing the whole vector rather than one player is the stronger test and the robust one: a
    single player's contrast depends on how much his particular team played without him, while the
    correlation over everyone asks the only question that matters -- when the rating handed in is
    wrong by a known amount, does the correction know which way.
    """
    sim, _, ts = world
    truth = _truth_offset(sim, ts)
    n = ts.n_players
    rng = np.random.default_rng(4)
    error = rng.normal(0.0, 1.5, 2 * n)
    alpha, _ = ts_mod.fit_alpha(ts, truth + error, lam=500.0, free_scale=False)

    for side, block in (("offense", slice(0, n)), ("defense", slice(n, 2 * n))):
        wanted, got = -error[block], alpha[block]
        assert np.corrcoef(wanted, got)[0, 1] > 0.4, f"{side}: the correction does not track the error"
        slope = float(np.polyfit(wanted, got, 1)[0])
        assert 0.2 < slope < 1.2, f"{side}: the correction is {slope:.2f} of the error it should undo"
    # and it is a correction, not noise: the ratings handed in are closer to the truth afterwards
    before = float(np.mean(error ** 2))
    after = float(np.mean((error + alpha) ** 2))
    assert after < before, "adding alpha moved the ratings away from the truth"


def test_a_rating_that_is_too_wide_is_read_as_a_scale_not_as_alphas(world):
    """The whole point of the free rating columns: a uniform rescale must not become attribution."""
    sim, _, ts = world
    truth = _truth_offset(sim, ts)
    wide = 1.5 * truth
    alpha_free, free = ts_mod.fit_alpha(ts, wide, lam=2000.0, free_scale=True)
    alpha_pinned, _ = ts_mod.fit_alpha(ts, wide, lam=2000.0, free_scale=False)
    scale_off = 1.0 + free[0]
    assert 0.5 < scale_off < 0.95, f"the offensive scale did not come back below one: {scale_off:.3f}"
    # pinned, the tilt has to go somewhere, and where it goes is a correlation with the rating itself
    n = ts.n_players
    def tilt(alpha):
        return abs(float(np.corrcoef(alpha[:n], wide[:n])[0, 1]))
    assert tilt(alpha_pinned) > tilt(alpha_free), "the free scale did not take the tilt out of the alphas"


def test_exposure_counts_games_the_team_played_without_him(world):
    sim, _, ts = world
    seasons = sorted(ts.keys.season.unique())
    rated = int(seasons[len(seasons) // 2])
    truth = sim["truth"]["ps"]
    team_of_player = dict(zip(truth[truth.season == rated].player_id.astype(int),
                              truth[truth.season == rated].team.astype(int)))
    # in the simulator a game is between two teams and every player belongs to one, so the team on
    # offence is the modal team of the five with the largest shares on that side
    n = ts.n_players
    team_of = np.array([team_of_player.get(int(p), -1) for p in ts.player_ids])
    dense_off = np.asarray(ts.Z[:, :n].todense())
    dense_def = np.asarray(ts.Z[:, n:].todense())
    rows = pd.DataFrame({"team_off": [np.bincount(team_of[r > 0][team_of[r > 0] >= 0]).argmax()
                                      if (team_of[r > 0] >= 0).any() else -1 for r in dense_off],
                         "team_def": [np.bincount(team_of[r > 0][team_of[r > 0] >= 0]).argmax()
                                      if (team_of[r > 0] >= 0).any() else -1 for r in dense_def]})
    exposure = ts_mod.exposure_table(ts, rows, team_of_player, rated)

    assert set(exposure.side) == set(ts_mod.SIDES)
    assert (exposure.without_poss >= 0).all()
    assert (exposure.with_poss >= 0).all()
    offense = exposure[exposure.side == "offense"].set_index("player_id")
    # a player with no rated-season team has no trade-set evidence at all
    unknown = offense[offense.team_id < 0]
    assert (unknown.without_poss == 0).all()
    assert offense.without_poss.sum() > 0, "nobody's team ever played without them"


def test_eligibility_ignores_players_who_dressed_and_never_played():
    roles = pd.DataFrame({
        "player_id": [1, 1, 2, 3, 4],
        "season":    [2024] * 5,
        "team_id":   [10, 11, 10, 10, 11],
        "poss_on":   [900.0, 0.0, 400.0, 50.0, 2000.0],
    })
    out = ts_mod.eligibility(roles, 2024, min_poss=100.0)
    assert sorted(out.player_id) == [1, 2, 4], "a zero-possession roster row changed who is eligible"
    assert int(out[out.player_id == 1].team_id.iloc[0]) == 10
    assert 3 not in set(out.player_id), "a player under the possession threshold stayed eligible"


def test_a_mid_season_trade_is_not_eligible():
    roles = pd.DataFrame({"player_id": [1, 1], "season": [2024, 2024], "team_id": [10, 11],
                          "poss_on": [800.0, 700.0]})
    assert len(ts_mod.eligibility(roles, 2024)) == 0


def test_trade_loss_reads_only_players_with_evidence():
    table = pd.DataFrame({
        "player_id": [1, 2, 3, 4], "season": 2024, "side": "offense",
        "alpha_good": [1.0, -1.0, 5.0, 2.0], "poss_on": [2000.0, 600.0, 300.0, 1000.0],
        "with_poss": [2000.0, 600.0, 300.0, 1000.0], "without_poss": [1000.0, 500.0, 0.0, 400.0],
        "eligible": [True, True, True, False],
    })
    loss = ts_mod.trade_loss(table)
    pooled = loss[loss.tier == "all"].iloc[0]
    assert pooled.players == 2, "a player with no without-games or no eligibility was counted"
    assert pooled.missed == pytest.approx(MAE_SCALE * 1.0)
    assert pooled.points == pytest.approx(1.0 * 20.0 + 1.0 * 6.0)
    assert set(loss.tier) >= {"all", "over 1,500 possessions", "500 to 1,500 possessions"}


def _team_of_rows(sim, ts):
    """The team on offence and the team defending, per team-game row, from the simulator's truth.

    The simulator gives every player a team per season, so the team on a side is the modal team of the
    men with a share on that side.
    """
    n = ts.n_players
    truth = sim["truth"]["ps"]
    out = {}
    for side, block in (("team_off", ts.Z[:, :n]), ("team_def", ts.Z[:, n:])):
        teams = []
        dense = np.asarray(block.todense())
        for row, season in zip(dense, ts.keys.season.to_numpy()):
            here = truth[truth.season == season].set_index("player_id").team
            ids = ts.player_ids[row > 0]
            present = [int(here[i]) for i in ids if i in here.index]
            teams.append(int(pd.Series(present).mode().iloc[0]) if present else -1)
        out[side] = teams
    return pd.DataFrame(out)


def test_team_columns_name_one_team_per_side_per_row(world):
    sim, _, ts = world
    rows = _team_of_rows(sim, ts)
    per_season, names = ts_mod.team_columns(ts, rows, per_season=True)
    pooled, pooled_names = ts_mod.team_columns(ts, rows, per_season=False)
    # exactly two entries per row: the team on offence and the team defending
    assert np.allclose(np.asarray(per_season.sum(axis=1)).ravel(), 2.0)
    assert np.allclose(np.asarray(pooled.sum(axis=1)).ravel(), 2.0)
    # one column per team per side when pooled, times the number of seasons when not
    assert len(pooled_names) == 2 * rows.team_off.nunique()
    assert len(names) > len(pooled_names)
    assert all(nm.startswith("offense ") or nm.startswith("defense ") for nm in names)


def test_team_terms_take_the_team_level_away_from_the_players(world):
    """With each team-season free, the TEAM-level part of alpha goes to the team and not to its players.

    This is the control the off-court record failed.  A team's output is the sum of its players'
    columns, so with nothing standing for the team a player can be paid for his team being good.  The
    testable consequence: pool each team-season's lineup correction over its own games, and that pooled
    number must be closer to zero once the team has a column of its own.
    """
    sim, _, ts = world
    rows = _team_of_rows(sim, ts)
    truth = _truth_offset(sim, ts)
    rng = np.random.default_rng(7)
    error = rng.normal(0.0, 1.5, 2 * ts.n_players)
    extra, _ = ts_mod.team_columns(ts, rows, per_season=True)

    gram, rhs, n_free = ts_mod.normal_equations(ts, truth + error, free_scale=False, extra=extra)
    with_teams, _ = ts_mod.solve_alpha(gram, rhs, ts.n_players, n_free, lam=500.0)
    without_teams, _ = ts_mod.fit_alpha(ts, truth + error, lam=500.0, free_scale=False)
    assert not np.allclose(with_teams, without_teams, atol=1e-6), "the team terms did not bind"

    n = ts.n_players
    label = pd.Series(list(zip(rows.team_off, ts.keys.season)))

    def team_level_spread(alpha):
        lineup = ts.Z[:, :n] @ alpha[:n]
        pooled = pd.Series(lineup * ts.w).groupby(label).sum() / pd.Series(ts.w).groupby(label).sum()
        return float(pooled.std())

    assert team_level_spread(with_teams) < team_level_spread(without_teams),         "the team-level part of alpha did not move to the team columns"


def test_trade_loss_with_no_evidence_returns_the_schema_not_an_empty_frame():
    """`--block=0` fits one season alone, so nobody's team plays a neighbouring game without him.
    That is an empty answer, not a failure -- but a frame with no COLUMNS reaches the caller as
    `KeyError: 'side'`, which reads like a broken fit.  scripts/70_tradeset.py groups this by side."""
    empty = pd.DataFrame({"eligible": pd.Series(dtype=bool), "without_poss": pd.Series(dtype=float),
                          "poss_on": pd.Series(dtype=float), "with_poss": pd.Series(dtype=float),
                          "alpha_good": pd.Series(dtype=float), "season": pd.Series(dtype="int64"),
                          "side": pd.Series(dtype=object)})
    loss = ts_mod.trade_loss(empty)
    assert loss.empty
    assert list(loss.columns) == list(ts_mod.LOSS_COLUMNS)
    # the shape the caller actually asks of it
    assert loss.groupby(["side", "tier"], as_index=False).agg(seasons=("season", "nunique")).empty
