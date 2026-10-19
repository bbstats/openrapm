"""Step 5, the swap adjustment (src/openrapm/swap_adjust.py), on the synthetic league of conftest.py.

And, where the verification copies are on disk, the one-thread golden step-4 tables rebuild the one-thread golden
step-5 tables byte for byte (about twenty seconds each: every season's stints are read).
"""
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from openrapm import swap_adjust as sa
from openrapm import swap_test as sw
from openrapm.data.config import load_config, resolve
from openrapm.settings import PRODUCT, TEST

VERIFY = Path(os.environ.get("OPENRAPM_VERIFY", "A:/code/spmm_verify"))


def _season(league, seed=0, games=320):
    stints, truth, roster = league(seed, games=games)
    return sw.prepare(sw.season_rows(stints)), truth, roster


def _team_totals(c, team_poss):
    t = team_poss.assign(c=team_poss.player_id.map(c))
    return t.groupby("team").apply(lambda g: float(np.sum(g.poss * g.c)), include_groups=False)


def test_the_give_back_keeps_every_team_total_at_zero(lineup_league):
    season, truth, _ = _season(lineup_league)
    pairs = sa.residual_pairs(season, truth.assign(o=truth.o + 1.0), context="none")
    poss = sa.team_possessions(season)
    rng = np.random.default_rng(3)
    mean = pd.Series(rng.normal(0, 2, len(truth)), index=truth.player_id)
    for side in sa.SIDES:
        c = sa.give_back(*sa.possession_shares(pairs[side], poss[side]), mean)
        assert np.allclose(_team_totals(c, poss[side]).to_numpy(), 0.0, atol=1e-6), side


def test_the_give_back_is_in_proportion_to_each_players_share_of_his_teams_possessions(lineup_league):
    """The nearest point to the type prediction with every team's total at zero: each player gives back an
    amount in proportion to his share of the team's possessions, the same multiple for a whole team."""
    season, truth, _ = _season(lineup_league)
    poss = sa.team_possessions(season)["offense"]
    mean = pd.Series(np.arange(len(truth), dtype=float), index=truth.player_id)
    pairs = sa.residual_pairs(season, truth, context="none")["offense"]
    c = sa.give_back(*sa.possession_shares(pairs, poss), mean)
    t = poss.assign(m=poss.player_id.map(mean), c=poss.player_id.map(c))
    t["share"] = t.poss / t.groupby("team").poss.transform("sum")
    multiple = (t.m - t.c) / t.share
    assert np.allclose(multiple, multiple.groupby(t.team).transform("mean"))
    assert np.allclose((t.share * t.c).groupby(t.team).sum(), 0.0)


def test_the_docstring_example_of_the_give_back():
    shares = np.array([[0.6, 0.4]])
    c = sa.give_back(np.array([1, 2]), shares, pd.Series({1: 1.0, 2: 0.0}))
    assert np.allclose(c.to_numpy(), [1.0 - 0.6 * 0.6 / 0.52, -0.4 * 0.6 / 0.52])
    assert abs(float((shares @ c.to_numpy())[0])) < 1e-12
    assert np.allclose(c.round(2).to_numpy(), [0.31, -0.46])


def test_a_planted_split_inside_a_team_shows_in_the_swap_residuals(lineup_league):
    season, truth, roster = _season(lineup_league)
    good, bad = int(roster[1][0]), int(roster[1][1])
    rated = truth.set_index("player_id").copy()
    rated.loc[good, "o"] -= 2.0                       # the ratings take two points of offence from one teammate
    rated.loc[bad, "o"] += 2.0                        # and hand them to another: the team's total is unchanged
    pairs = sa.residual_pairs(season, rated.reset_index(), context="none")["offense"]

    def mean_residual(player):
        """The h-weighted mean residual of the swaps he is in, signed so positive says he beat his rating."""
        mine = pairs[(pairs.fifth_a == player) | (pairs.fifth_b == player)]
        signed = np.where(mine.fifth_a == player, mine.r, -mine.r)
        return np.average(signed, weights=mine.h), len(mine)

    (under, n_under), (over, n_over) = mean_residual(good), mean_residual(bad)
    assert n_under > 50 and n_over > 50
    assert under > 0.0 > over and under - over > 2.0        # two points off each way, four between them


def test_the_type_model_finds_a_planted_feature(lineup_league):
    season, truth, _ = _season(lineup_league)
    rng = np.random.default_rng(5)
    feats = pd.DataFrame({"x": rng.normal(0, 1, len(truth)), "noise": rng.normal(0, 1, len(truth))},
                         index=truth.player_id)
    rated = truth.set_index("player_id").copy()
    rated["o"] = rated.o - 1.5 * feats.x              # the ratings under-credit x by 1.5 a standard deviation
    pairs = sa.residual_pairs(season, rated.reset_index(), context="none")["offense"]
    fold_of = sa.player_folds(truth.player_id, 5)
    g = sa.pair_grams(pairs, feats, fold_of, 5)
    beta = sa.type_coefficients([g])
    assert abs(beta[0] - 1.5) < 0.3 and abs(beta[1]) < 0.3
    for k in range(5):                                # a fold's model is exactly the model of the other folds' pairs
        fa, fb = pairs.fifth_a.map(fold_of), pairs.fifth_b.map(fold_of)
        without = sa.pair_grams(pairs[(fa != k) & (fb != k)], feats, fold_of, 5)
        assert np.allclose(sa.type_coefficients([g], fold=k), sa.type_coefficients([without]))


def test_player_folds_are_the_id_modulo_the_number_of_folds():
    folds = sa.player_folds([2544, 2544, 201939, 1641705], 5)
    assert folds.to_dict() == {2544: 4, 201939: 4, 1641705: 0}


@pytest.mark.parametrize("exclude_near, expected", [(0, [2020, 2021, 2023, 2024]), (1, [2020, 2024])])
def test_the_training_seasons_leave_out_the_rated_season_and_in_the_test_setting_its_neighbours(exclude_near,
                                                                                               expected):
    seasons = [2020, 2021, 2022, 2023, 2024]
    assert sa.training_seasons(2022, seasons, set(seasons), exclude_near) == expected
    assert PRODUCT.swap.exclude_near == 0 and TEST.swap.exclude_near == 1


def _two_season_swaps(league):
    """{season: SeasonSwaps} for two synthetic seasons rated at their truth, and the type features."""
    swaps, features = {}, []
    for season in (2021, 2022):
        stints, truth, _ = league(season, games=200)
        lineups = sw.prepare(sw.season_rows(stints), season)
        rng = np.random.default_rng(season)
        features.append(pd.DataFrame({"player_id": truth.player_id, "season": season,
                                      **{name: rng.normal(0, 1, len(truth)) for name in ("x", "y")}}))
        swaps[season] = (lineups, truth)
    features = pd.concat(features).set_index(["player_id", "season"])
    fold_of = sa.player_folds(features.index.get_level_values("player_id"), 5)
    kept = {season: sa.season_swaps(lineups, truth, sa.season_features(features, season), fold_of, 5)
            for season, (lineups, truth) in swaps.items()}
    return kept, features, fold_of


def test_a_season_with_no_training_season_gets_no_type_prediction(lineup_league):
    swaps, features, fold_of = _two_season_swaps(lineup_league)
    one = {2022: swaps[2022]}
    predictions = sa.type_predictions(features, one, fold_of, {2021, 2022}, PRODUCT.swap)
    assert (predictions[(2022, "offense")] == 0.0).all() and (predictions[(2022, "defense")] == 0.0).all()
    both = sa.type_predictions(features, swaps, fold_of, {2021, 2022}, PRODUCT.swap)
    assert both[(2022, "offense")].abs().sum() > 0
    untrained = sa.type_predictions(features, swaps, fold_of, {2022}, PRODUCT.swap)   # 2021 still being played
    assert (untrained[(2022, "offense")] == 0.0).all()


def test_holding_the_spread_inside_teams_keeps_team_means_and_the_order_inside_each_team():
    rng = np.random.default_rng(11)
    team = np.repeat([1, 2, 3], 6).astype(float)
    weight = rng.uniform(100, 3000, len(team))
    base = rng.normal(0, 1, len(team)) + np.repeat([1.0, -0.5, 0.2], 6)
    adjusted = base + rng.normal(0, 1.5, len(team))            # wider inside each team, as the adjustment is
    held, k = sa.hold_spread_within(adjusted, base, weight, team)
    assert 0 < k < 1
    for t in (1, 2, 3):
        m = team == t
        assert np.isclose(np.average(held[m], weights=weight[m]), np.average(adjusted[m], weights=weight[m]))
        assert (np.argsort(held[m]) == np.argsort(adjusted[m])).all()

    def within(x):
        means = {t: np.average(x[team == t], weights=weight[team == t]) for t in (1, 2, 3)}
        return np.average((x - np.array([means[t] for t in team])) ** 2, weights=weight)
    assert np.isclose(within(held), within(base))


def test_ratings_the_adjustment_did_not_move_come_back_bit_for_bit():
    base = np.array([0.1, 0.2, 0.3, -0.7]) / 3.0               # values a mean-plus-distance rebuild can round
    held, k = sa.hold_spread_within(base.copy(), base, np.array([100.0, 200.0, 300.0, 50.0]),
                                    np.array([1.0, 1.0, 2.0, 2.0]))
    assert k == 1.0 and np.array_equal(held, base)


def test_a_player_with_no_main_team_keeps_his_adjusted_number():
    team = np.array([1.0, 1.0, 1.0, np.nan])
    held, _ = sa.hold_spread_within(np.array([1.0, -1.0, 0.5, 3.0]), np.array([0.5, -0.5, 0.2, 1.0]),
                                    np.array([100.0, 200.0, 300.0, 50.0]), team)
    assert held[3] == 3.0


def test_recentring_leaves_each_seasons_possession_weighted_mean_at_zero():
    rows = pd.DataFrame({"season": [2021, 2021, 2022, 2022, 2022], "c_o": [1.0, 2.0, -1.0, 0.5, 4.0],
                         "c_d": [0.0, 1.0, 1.0, 1.0, 1.0]})
    weights = {"c_o": np.array([100.0, 300.0, 50.0, 50.0, 100.0]), "c_d": np.array([1.0, 1.0, 1.0, 2.0, 3.0])}
    out = sa.recentre(rows, weights)
    for column, weight in weights.items():
        for season in (2021, 2022):
            mine = (out.season == season).to_numpy()
            assert abs(np.average(out.loc[mine, column], weights=weight[mine])) < 1e-12


def test_the_adjusted_table_moves_the_ratings_positive_good_and_leaves_the_parts():
    table = pd.DataFrame({"player_id": [1, 2], "season": [2026, 2026], "rating_off": [1.0, -1.0],
                          "rating_def": [0.5, -0.5], "rating_total": [1.5, -1.5], "offense": [1.0, -1.0],
                          "defense": [-0.5, 0.5], "prior_off": [0.8, -0.8], "u_off": [0.2, -0.2]})
    corrections = pd.DataFrame({"player_id": [1], "season": [2026], "c_o": [0.3], "c_d": [-0.2]})
    out = sa.adjusted_table(table, corrections)
    assert np.allclose(out.rating_off, [1.3, -1.0]) and np.allclose(out.rating_def, [0.7, -0.5])
    assert np.allclose(out.c_off, [0.3, 0.0]) and np.allclose(out.c_def, [0.2, 0.0])
    assert np.allclose(out.rating_total, out.rating_off + out.rating_def)
    assert np.allclose(out.defense, -out.rating_def) and np.allclose(out.prior_off, table.prior_off)
    assert list(out.columns) == [*table.columns, "c_off", "c_def"]


def _rated_table(swaps):
    """A step-4-like table for the synthetic seasons: random ratings, each player's possessions, his main team."""
    table = []
    for season, kept in swaps.items():
        poss = kept.team_possessions["offense"].groupby("player_id").poss.sum()
        table.append(pd.DataFrame({"player_id": poss.index.astype(np.int64), "season": season,
                                   "rating_off": 0.0, "rating_def": 0.0, "poss_off": poss.to_numpy(),
                                   "poss_def": poss.to_numpy()}))
    table = pd.concat(table, ignore_index=True)
    rng = np.random.default_rng(0)
    table["rating_off"] = rng.normal(0, 2, len(table))
    table["rating_def"] = rng.normal(0, 1, len(table))
    table["rating_total"] = table.rating_off + table.rating_def
    table["offense"], table["defense"] = table.rating_off, -table.rating_def
    main_team = (table.player_id // 100).rename("team_id")
    main_team.index = pd.MultiIndex.from_arrays([table.player_id, table.season])
    return table, main_team


def test_a_table_of_one_season_comes_out_of_step_5_unchanged(lineup_league):
    swaps, features, fold_of = _two_season_swaps(lineup_league)
    one = {2022: swaps[2022]}
    table, main_team = _rated_table(one)
    predictions = sa.type_predictions(features, one, fold_of, {2021, 2022}, PRODUCT.swap)
    out, factors = sa.adjust(table, sw.ratings_in_raw_sign(table), one, predictions, main_team, PRODUCT.swap)
    assert (factors.factor == 1.0).all()
    assert (out.c_off == 0.0).all() and (out.c_def == 0.0).all()
    for column in table.columns:
        assert np.array_equal(out[column].to_numpy(), table[column].to_numpy()), column


def test_adjust_end_to_end_keeps_each_team_total_and_the_spread_inside_teams(lineup_league):
    swaps, features, fold_of = _two_season_swaps(lineup_league)
    table, main_team = _rated_table(swaps)
    raw = sw.ratings_in_raw_sign(table)
    predictions = sa.type_predictions(features, swaps, fold_of, {2021, 2022},
                                      replace(PRODUCT.swap, applied_share=1.0))
    out, factors = sa.adjust(table, raw, swaps, predictions, main_team, replace(PRODUCT.swap, applied_share=1.0))
    assert len(factors) == 4 and (factors.factor > 0).all()
    assert out.c_off.abs().sum() > 0
    for season in (2021, 2022):
        mine = out[out.season == season]
        team = mine.player_id // 100
        moved = (mine.poss_off * mine.c_off).groupby(team).sum() / mine.poss_off.groupby(team).sum()
        assert np.allclose(moved, 0.0, atol=1e-9)          # every player is on one team: team totals stay put


_CFG = load_config()
# experiment 80's pair since 2026-10-19 (step 5 trained on the folds' held-back games); the weightfix pair before it,
# onethread/gate0_*, pinned step 5 as it was
_REBUILDS = ((PRODUCT, "crossfit80/product_step4.parquet", "crossfit80/product_step5.parquet"),
             (TEST, "crossfit80/test_step4.parquet", "crossfit80/test_step5.parquet"))


@pytest.mark.slow
@pytest.mark.parametrize("settings, step4, step5", _REBUILDS, ids=("product", "test"))
def test_the_golden_step4_tables_rebuild_the_one_thread_golden_step5_tables_exactly(settings, step4, step5, tmp_path):
    """The golden step-4 table through step 5 gives the one-thread golden step 5, byte for byte."""
    threadpoolctl = pytest.importorskip("threadpoolctl")
    needed = [VERIFY / "golden" / step4, VERIFY / "golden" / step5, resolve(_CFG, "panel"), resolve(_CFG, "stints")]
    if not all(path.exists() for path in needed):
        pytest.skip("needs the stints, the season panel and the verification copies (OPENRAPM_VERIFY)")
    from openrapm.commands.build import swap_seasons
    with threadpoolctl.threadpool_limits(1):
        got, _, _ = swap_seasons(_CFG, pd.read_parquet(VERIFY / "golden" / step4), settings)
    pd.testing.assert_frame_equal(got, pd.read_parquet(VERIFY / "golden" / step5), check_exact=True)
    got.to_parquet(tmp_path / "step5.parquet", index=False)
    assert (tmp_path / "step5.parquet").read_bytes() == (VERIFY / "golden" / step5).read_bytes()


# ---------------------------------------------------- the training swaps, from held-back games (experiment 80)
def test_held_back_grams_are_the_pair_grams_of_the_held_back_swaps(lineup_league):
    """A fold's share of the type model: its held-back games' swaps scored against its ratings, read on its
    fit-game features, with player folds; summing folds adds every part."""
    stints, truth, _ = lineup_league(5, games=200)
    held = sw.prepare(sw.season_rows(stints[stints.game_id >= "g0150"]), 2022)
    rng = np.random.default_rng(1)
    table = pd.DataFrame({"player_id": truth.player_id, "season": 2022, "rating_off": rng.normal(0, 2, len(truth)),
                          "rating_def": rng.normal(0, 1, len(truth))})
    features = pd.DataFrame({"x": rng.normal(0, 1, len(truth)), "y": rng.normal(0, 1, len(truth))},
                            index=truth.player_id.to_numpy())
    grams = sa.held_back_grams(held, table, features, 5)
    pairs = sa.residual_pairs(held, sw.ratings_in_raw_sign(table))
    for side in sa.SIDES:
        direct = sa.pair_grams(pairs[side], features, sa.player_folds(features.index.to_numpy(), 5), 5)
        assert np.allclose(grams[side].G, direct.G) and np.allclose(grams[side].b, direct.b)
        assert grams[side].n == direct.n > 0
    total = sa.add_grams([grams["offense"], grams["offense"]])
    assert np.allclose(total.G, 2 * grams["offense"].G) and np.allclose(total.G_fold, 2 * grams["offense"].G_fold)
    assert total.n == 2 * grams["offense"].n


def test_a_folds_features_are_read_on_the_tables_scale():
    rows = pd.DataFrame({"player_id": [1, 2, 3], "a": [1.0, 2.0, 4.0]})
    table = pd.DataFrame({"player_id": [1, 2, 3], "prior_off": [0.5, 1.0, 1.5], "prior_def": [0.0, 1.0, 2.0]})
    centre = pd.Series({"a": 2.0, "prior_off": 1.0, "prior_def": 0.0})
    scale = pd.Series({"a": 2.0, "prior_off": 0.5, "prior_def": 1.0})
    out = sa.fold_type_features(rows, table, centre, scale)
    assert list(out.columns) == ["a", "prior_off", "prior_def"]
    assert np.allclose(out.loc[3].to_numpy(), [1.0, 1.0, 2.0])


def test_the_type_features_are_from_the_rated_season_only():
    """Experiment 80, the owner, 2026-10-19: "it must ONLY come from the season in question"."""
    assert "seasons_before" not in sa.TYPE_FEATURES and len(sa.TYPE_FEATURES) == 21
