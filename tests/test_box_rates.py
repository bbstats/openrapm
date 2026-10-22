"""The padded box-score rates (src/openrapm/box_rates.py) on a simulated league: which games count, the padding toward
the possession bin, the pad k, the centring; and the two proportion helpers data/shots.py pads shooters with."""
import numpy as np
import pandas as pd
import pytest

from openrapm.box_rates import (
    K_FALLBACK,
    bin_of,
    box_rates,
    game_totals,
    lineups,
    mom_k,
    pad_k,
    pad_rates,
    shrink,
    split_half_k,
)
from openrapm.design import FEATURES
from openrapm.simulate import simulate


@pytest.fixture(scope="module")
def design(synthetic_design):
    """Two seasons of a league whose talent is redrawn every season (simulate's defaults)."""
    sim = simulate(n_seasons=2, n_teams=6, players_per_team=10, games_per_season=40, stints_per_game=(20, 30), seed=1)
    return synthetic_design(sim)


def test_shrink_limits():
    assert shrink(0.9, 100.0, 0.0, 0.5) == 0.9                     # no padding: the raw rate
    assert shrink(0.9, 0.0, 30.0, 0.5) == 0.5                      # no data: the target
    assert shrink(0.9, 0.0, 0.0, 0.5) == 0.5                       # nothing at all: still the target
    x = shrink(np.array([0.9, 0.3]), np.array([10.0, 1000.0]), 30.0, 0.5)
    assert abs(x[0] - (10 * 0.9 + 30 * 0.5) / 40) < 1e-12
    assert abs(x[1] - 0.3) < 0.01                                  # plenty of data: barely moved


def test_mom_k_recovers_a_beta_binomial():
    """Rates drawn Beta(a, b) have between-player variance mu(1-mu)/(a+b+1), so the pad k is about a + b."""
    rng = np.random.default_rng(0)
    a, b = 30.0, 10.0                                              # mean 0.75, k about 40
    p = rng.beta(a, b, 2000)
    att = rng.integers(100, 400, 2000).astype(float)
    made = rng.binomial(att.astype(int), p).astype(float)
    league, k = mom_k(made, att)
    assert abs(league - 0.75) < 0.01
    assert 0.75 * (a + b) < k < 1.25 * (a + b)


def test_mom_k_falls_back_with_too_few_players():
    league, k = mom_k(np.array([7.0, 8.0]), np.array([10.0, 10.0]))
    assert league == 0.75 and k == K_FALLBACK
    league, k = mom_k(np.array([]), np.array([]), fallback_rate=0.6)
    assert league == 0.6 and k == K_FALLBACK


def test_split_half_k_recovers_the_true_k():
    """Poisson counts at true rates mu (per 100) have per-possession noise variance 100 mu, so k = 100 E[mu] /
    Var(mu)."""
    rng = np.random.default_rng(0)
    n_players, n_games = 400, 60
    mu = rng.gamma(4.0, 2.0 / 4.0, size=(n_players, 2))
    mu[:, 1] = rng.gamma(4.0, 6.0 / 4.0, size=n_players)
    possessions = rng.integers(20, 60, size=(n_players, n_games)).astype(float)
    counts = rng.poisson(mu[:, None, :] * possessions[:, :, None] / 100.0).astype(float)
    player = np.repeat(np.arange(n_players), n_games)
    game = np.tile(np.arange(n_games), n_players)
    k = split_half_k(player, game, counts.reshape(-1, 2), possessions.ravel(), np.zeros(n_players, int), 1)
    np.testing.assert_allclose(k[0], 100 * mu.mean(0) / mu.var(0), rtol=0.2)


def test_a_design_cut_to_some_games_counts_only_their_box_scores(design):
    """`Design.subset` cuts the rows and keeps the whole box table; the rates still count only the games the
    rows come from."""
    game = design.rows["game_idx"].to_numpy()
    held = np.unique(game)[:5]
    part = design.subset(~np.isin(game, held))
    rates = box_rates(part)
    box = design.game_box[~design.game_box.game_idx.isin(held)]
    totals = np.zeros_like(rates.counts)
    np.add.at(totals, box.player_season.to_numpy(), box[FEATURES].to_numpy())
    np.testing.assert_allclose(rates.counts, totals)
    assert not np.isin(held, game_totals(part).game).any()
    assert (rates.offence.possessions < box_rates(design).offence.possessions + 1e-9).all()


def test_each_rate_is_padded_toward_its_bin_by_its_season_k(design):
    """(n x rate + k x target) / (n + k) with rate = 100 x count / n: here 100 x count + k x target over n + k."""
    rates = box_rates(design)
    season = design.spec.season_of_player
    for side in (rates.offence, rates.defence):
        k = rates.pad_k[season]
        target = rates.bin_targets[season, bin_of(rates.bin_edges, side.possessions)]
        n = side.possessions[:, None]
        np.testing.assert_allclose(side.rates, (100.0 * rates.counts + k * target) / (n + k), rtol=1e-12)
    assert np.isfinite(rates.pad_k).all() and (rates.pad_k > 0).all()


def test_defence_rates_are_per_defensive_possession(design):
    rates = box_rates(design)
    defensive = design.game_poss.groupby("player_season").poss_def.sum()
    np.testing.assert_allclose(rates.defence.possessions[defensive.index.to_numpy()], defensive.to_numpy())
    assert not np.allclose(rates.defence.rates, rates.offence.rates)


def test_the_centred_rates_of_an_average_lineup_add_to_zero(design):
    """Centred = padded minus a fifth of the row-weighted mean of a lineup's five rates summed: one shift per count,
    after which the weighted mean over the rows of a lineup's summed rates is 0."""
    rates = box_rates(design)
    weights = design.w
    for side, lineup in zip((rates.offence, rates.defence), lineups(design)):
        summed = side.centred[lineup].sum(axis=1)
        np.testing.assert_allclose((weights[:, None] * summed).sum(0) / weights.sum(), 0.0, atol=1e-9)
        shift = side.rates - side.centred
        np.testing.assert_allclose(shift, np.broadcast_to(shift[0], shift.shape), atol=1e-12)


def test_the_bins_give_each_size_of_player_his_own_target_and_the_league_target_one_level(design):
    by_bin = box_rates(design, pad_target="poss_conditional")
    assert by_bin.bin_edges is not None and by_bin.bin_targets.shape[1] == 6
    assert not np.allclose(by_bin.bin_targets[0, 0], by_bin.bin_targets[0, -1])
    league = box_rates(design, pad_target="league")
    assert league.bin_edges is None and league.bin_targets is None
    season = design.spec.season_of_player
    k, n = league.pad_k[season], league.offence.possessions[:, None]
    np.testing.assert_allclose(league.offence.rates, (100.0 * league.counts + k * league.league[season]) / (n + k),
                               rtol=1e-12)
    with pytest.raises(ValueError, match="pad_target"):
        box_rates(design, pad_target="median")


def test_the_example_and_a_player_without_possessions():
    """The module's example: 200 possessions, 2.1 steals per 100 (4.2 steals), k = 450, bin rate 1.4 -> 1.6.  With no
    possessions the rate is the target."""
    padded = pad_rates(np.array([[4.2], [3.0]]), np.array([200.0, 0.0]), np.full((2, 1), 450.0), np.full((2, 1), 1.4))
    assert abs(padded[0, 0] - (200 * 2.1 + 450 * 1.4) / 650) < 1e-12 and round(padded[0, 0], 1) == 1.6
    assert padded[1, 0] == 1.4


def test_the_body_target_reads_height_and_never_a_players_own_games():
    """Experiment 86: blocks that rise with height give tall players the higher target; and a player's target does not
    move when only his own counts change, since his fifth of the players is left out of the line he reads."""
    from openrapm.box_rates import POSSESSION_BINS, body_targets
    rng = np.random.default_rng(3)
    n = 400
    height = rng.normal(79.0, 3.5, n)
    weight = 6.0 * height - 260.0 + rng.normal(0, 15, n)
    possessions = rng.integers(200, 3000, n).astype(float)
    blocks = rng.poisson(np.maximum(0.3 * (height - 72.0), 0.1) * possessions / 100.0).astype(float)
    counts = np.column_stack([blocks, rng.poisson(2.0 * possessions / 100.0)])
    bins = rng.integers(0, POSSESSION_BINS, n)
    ids = np.arange(1000, 1000 + n)
    fallback = np.full((n, 2), 1.0)
    body = np.column_stack([height, weight])
    target = body_targets(counts, possessions, np.zeros(n, int), 1, bins, bins, body, ids, fallback)
    tall, short = height > 83, height < 75
    assert target[tall, 0].mean() > 2.0 * target[short, 0].mean()
    moved = counts.copy()
    moved[0, 0] += 500.0                                   # player 1000's blocks only
    again = body_targets(moved, possessions, np.zeros(n, int), 1, bins, bins, body, ids, fallback)
    assert again[0, 0] == target[0, 0]                     # his own target is untouched
    same_fold = (ids % 5 == ids[0] % 5)
    assert np.array_equal(again[same_fold], target[same_fold]) and not np.array_equal(again, target)


def test_box_rates_run_with_the_body_target(design):
    ids = np.unique(design.spec.players.player_id.to_numpy())
    rng = np.random.default_rng(5)
    body = pd.DataFrame({"height_in": rng.normal(79, 3, len(ids)), "weight_lb": rng.normal(220, 20, len(ids))},
                        index=ids)
    rates = box_rates(design, pad_target="body_and_possessions", body=body)
    shipped = box_rates(design)
    assert np.isfinite(rates.offence.rates).all() and np.isfinite(rates.pad_k).all()
    assert not np.allclose(rates.offence.rates, shipped.offence.rates)
    with pytest.raises(ValueError, match="height and weight"):
        box_rates(design, pad_target="body_and_possessions")


def test_a_player_without_a_height_reads_the_medians_and_breaks_no_target(design):
    ids = np.unique(design.spec.players.player_id.to_numpy())
    rng = np.random.default_rng(6)
    body = pd.DataFrame({"height_in": rng.normal(79, 3, len(ids)), "weight_lb": rng.normal(220, 20, len(ids))},
                        index=ids).iloc[1:]                       # the first player is not in the table
    rates = box_rates(design, pad_target="body_and_possessions", body=body)
    assert np.isfinite(rates.offence.rates).all() and np.isfinite(rates.defence.rates).all()


def test_the_pad_k_can_be_measured_around_the_bins_the_rates_are_padded_toward(design):
    """config pad_k_around "target": the split-half spread between players is measured around each player-season's
    possession bin, the target his rates are padded toward, not around the league; "league" is the shipped step."""
    shipped = box_rates(design)
    np.testing.assert_array_equal(box_rates(design, pad_k_around="league").pad_k, shipped.pad_k)
    spec = design.spec
    season_of, player_of = spec.season_of_player_season, spec.player_of_player_season
    target = shipped.bin_targets[season_of, bin_of(shipped.bin_edges, shipped.offence.possessions[player_of])]
    expected = pad_k(game_totals(design), season_of, spec.n_seasons, target)
    np.testing.assert_allclose(box_rates(design, pad_k_around="target").pad_k, expected)
    with pytest.raises(ValueError):
        box_rates(design, pad_k_around="median")
