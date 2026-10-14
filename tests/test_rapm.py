"""Step 1 (src/openrapm/rapm.py): the season sums, the leave-season-out solve, the un-shrunk label.

The design rests on one claim: the normal equations add up over seasons, so the fit that leaves the rated season
out is the sums of the other seasons.  If that is wrong everything downstream is wrong and nothing else would
notice, so it is tested against a direct fit on the same seasons.
"""
import numpy as np
import pandas as pd
import pytest

from openrapm import rapm
from openrapm.settings import PRODUCT

SEASONS = [2001, 2002, 2003, 2004]


@pytest.fixture(scope="module")
def world(synthetic_league, synthetic_design):
    """Four seasons of a league whose talent persists, and each season's own design."""
    sim = synthetic_league(n_seasons=4, rho=0.9, turnover=0.25, seed=6)
    return sim, {season: synthetic_design(sim, [season]) for season in SEASONS}


def _sums(designs, seasons, min_possessions=0.0, teams=None):
    sums = rapm.SeasonSums(min_possessions=min_possessions)
    for s in seasons:
        sums.add(s, designs[s], None if teams is None else teams[s])
    return sums


@pytest.fixture(scope="module")
def teams(world):
    """{season: (team_off, team_def)} per row: the simulated league moves players only between seasons, so a row's
    team is its first player's team that season."""
    sim, designs = world
    team = sim["truth"]["ps"].set_index(["season", "player_id"]).team
    out = {}
    for season, design in designs.items():
        player_ids = design.spec.players.player_id.to_numpy()
        out[season] = tuple(team.reindex(pd.MultiIndex.from_arrays(
            [np.full(len(design.y), season), player_ids[design.parts[five][:, 0]]])).to_numpy(np.int64)
            for five in ("lineup_o", "lineup_d"))
    return out


@pytest.fixture(scope="module")
def planted(world, teams):
    """The world's designs with a team-season effect added to every row, sd 4 per 100 on each side, and the effects:
    {(season, team): (attacking, defending)}."""
    from dataclasses import replace
    _, designs = world
    rng = np.random.default_rng(1)
    effects = {(season, team): tuple(rng.normal(0.0, 4.0, 2)) for season in SEASONS
               for team in np.unique(np.concatenate(teams[season]))}
    shifted = {}
    for season, design in designs.items():
        team_off, team_def = teams[season]
        bump = (np.array([effects[(season, t)][0] for t in team_off])
                + np.array([effects[(season, t)][1] for t in team_def]))
        shifted[season] = replace(design, y=design.y + bump)
    return shifted, effects


def _pen(offense, defense=None, context=0.0):
    return (offense, offense if defense is None else defense, context)


# ------------------------------------------------------------ the arithmetic
def test_the_normal_equations_are_b_transpose_w_b(world):
    _, designs = world
    d = designs[2002]
    players, context = rapm.design_blocks(d)
    sums = rapm.normal_equations(players, context, d.y, d.w)
    b = np.hstack([players.toarray(), context])
    gram = np.block([[sums.gram_players, sums.gram_players_context],
                     [sums.gram_players_context.T, sums.gram_context]])
    assert np.allclose(gram, b.T @ (b * d.w[:, None]))
    assert np.allclose(np.concatenate([sums.rhs_players, sums.rhs_context]), b.T @ (d.w * d.y))


def test_the_solve_survives_an_empty_unpenalised_column():
    """A zero context column with no penalty makes the plain solve singular; the jitter keeps the answer."""
    rng = np.random.default_rng(0)
    b = rng.normal(size=(50, 3))
    gram = np.zeros((4, 4))
    gram[:3, :3] = b.T @ b
    rhs = np.concatenate([b.T @ rng.normal(size=50), [0.0]])
    got = rapm.solve_ridge(gram, rhs, [1.0, 1.0, 0.0, 0.0])
    assert np.isfinite(got).all() and abs(got[3]) < 1e-12
    assert np.allclose(got[:3], np.linalg.solve(gram[:3, :3] + np.diag([1.0, 1.0, 0.0]), rhs[:3]))
    assert rapm.block_penalty((2, 1, 2), (5.0, 7.0, 0.0)).tolist() == [5.0, 5.0, 7.0, 0.0, 0.0]


def test_solved_ridge_returns_the_matrix_it_solved_jitter_included():
    """The plain solve's matrix is G + the penalties; after a retry the unpenalised diagonal carries the jitter."""
    rng = np.random.default_rng(0)
    b = rng.normal(size=(50, 3))
    gram = np.zeros((4, 4))
    gram[:3, :3] = b.T @ b
    rhs = np.concatenate([b.T @ rng.normal(size=50), [0.0]])
    penalty = np.array([1.0, 1.0, 0.0, 0.0])
    theta, matrix = rapm.solved_ridge(gram, rhs, penalty)
    assert np.array_equal(theta, rapm.solve_ridge(gram, rhs, penalty))
    plain = gram + np.diag(penalty)
    assert np.array_equal(matrix[:2, :2], plain[:2, :2])                             # the penalised block is untouched
    assert matrix[2, 2] > plain[2, 2] and matrix[3, 3] > plain[3, 3]                 # the two unpenalised columns
    assert np.allclose(matrix @ theta, rhs)                                          # theta solves what came back
    full = rng.normal(size=(50, 2))
    solvable = full.T @ full
    theta, matrix = rapm.solved_ridge(solvable, full.T @ rng.normal(size=50), [0.5, 0.0])
    assert np.array_equal(matrix, solvable + np.diag([0.5, 0.0]))                    # no retry, no jitter


# ------------------------------------------------------------ the claim the whole design rests on
def test_leaving_a_season_out_equals_fitting_without_it(world):
    """Leaving 2003 out of the sums must equal summing the other three seasons, or the shortcut is wrong."""
    _, designs = world
    held = _sums(designs, SEASONS).ratings([2003], _pen(2000.0)).set_index("player_id")
    direct = _sums(designs, [s for s in SEASONS if s != 2003]).ratings((), _pen(2000.0)).set_index("player_id")
    shared = held.index.intersection(direct.index)
    assert len(shared) > 50
    assert np.allclose(held.loc[shared, "offense"], direct.loc[shared, "offense"], atol=1e-8)
    assert np.allclose(held.loc[shared, "defense"], direct.loc[shared, "defense"], atol=1e-8)
    assert np.allclose(held.loc[shared, "possessions"], direct.loc[shared, "possessions"])


def test_the_order_seasons_are_added_in_changes_only_the_last_digits(world):
    _, designs = world
    forward = _sums(designs, SEASONS).ratings((), _pen(2000.0)).set_index("player_id").sort_index()
    backward = _sums(designs, SEASONS[::-1]).ratings((), _pen(2000.0)).set_index("player_id").sort_index()
    assert np.allclose(forward.offense, backward.offense, atol=1e-9)


# ------------------------------------------------------------ what the ratings mean
def test_it_recovers_the_simulated_talent(world):
    sim, designs = world
    rating = _sums(designs, SEASONS).ratings((), _pen(1000.0))
    truth = sim["truth"]["ps"].groupby("player_id")[["impact_O", "impact_D"]].mean().reset_index()
    joined = rating.merge(truth, on="player_id")
    assert len(joined) > 50
    assert np.corrcoef(joined.offense, joined.impact_O)[0, 1] > 0.6
    assert np.corrcoef(joined.defense, joined.impact_D)[0, 1] > 0.6


def test_more_seasons_beat_fewer(world):
    """The point of pooling: four seasons of evidence must track talent better than one."""
    sim, designs = world
    truth = sim["truth"]["ps"].groupby("player_id")[["impact_O"]].mean().reset_index()

    def quality(seasons):
        r = _sums(designs, seasons).ratings((), _pen(1000.0)).merge(truth, on="player_id")
        return np.corrcoef(r.offense, r.impact_O)[0, 1]

    assert quality(SEASONS) > quality([2002])


def test_a_bigger_penalty_shrinks_the_ratings(world):
    _, designs = world
    sums = _sums(designs, SEASONS)
    light, heavy = sums.ratings((), _pen(200.0)), sums.ratings((), _pen(50000.0))
    assert heavy.offense.std() < 0.5 * light.offense.std()
    assert len(light) == len(heavy)


def test_each_penalty_moves_its_own_block(world):
    _, designs = world
    sums = _sums(designs, SEASONS)
    even = sums.ratings((), _pen(5000.0, 5000.0))
    light = sums.ratings((), _pen(5000.0, 1000.0))
    assert light.defense.std() > even.defense.std()
    pinned = sums.ratings((), _pen(5000.0, 5000.0, 1e14))
    assert not np.allclose(pinned.offense.to_numpy(), even.offense.to_numpy())


def test_the_possession_minimum_drops_the_barely_seen(world):
    _, designs = world
    everyone = _sums(designs, SEASONS).ratings((), _pen(2000.0))
    assert len(_sums(designs, SEASONS, min_possessions=1e9).ratings((), _pen(2000.0))) == 0
    assert (everyone.possessions > 0).all()
    kept = _sums(designs, SEASONS, min_possessions=100.0).ratings((), _pen(2000.0))
    assert (kept.possessions >= 100.0).all() and len(kept) <= len(everyone)


def test_each_season_keeps_its_own_level(world):
    """Add 40 points per 100 to one season's scoring: its own intercept absorbs it and no rating moves."""
    from dataclasses import replace
    _, designs = world
    base = _sums(designs, SEASONS).ratings((), _pen(2000.0)).set_index("player_id")
    shifted = dict(designs)
    shifted[2002] = replace(designs[2002], y=designs[2002].y + 40.0)
    moved = _sums(shifted, SEASONS).ratings((), _pen(2000.0)).set_index("player_id")
    shared = base.index.intersection(moved.index)
    assert np.allclose(base.loc[shared, "offense"], moved.loc[shared, "offense"], atol=1e-6)


# ------------------------------------------------------------ the team-season flags (experiment 60)
def test_team_flags_on_a_hand_example():
    flags, teams = rapm.team_flags([20, 10, 30], [30, 20, 10])
    assert teams.tolist() == [10, 20, 30]
    assert flags.tolist() == [[0, 1, 0, 0, 0, 1], [1, 0, 0, 0, 1, 0], [0, 0, 1, 1, 0, 0]]


def test_without_flags_the_penalties_are_the_old_blocks_bit_for_bit(world):
    _, designs = world
    sums = _sums(designs, SEASONS)
    gram, _, possessions, n_context = sums.assemble([2003])
    old = rapm.block_penalty((possessions.size, possessions.size, n_context), [2000.0, 3000.0, 0.0])
    assert np.array_equal(sums.penalty([2003], (2000.0, 3000.0, 0.0)), old)
    assert np.array_equal(sums.penalty([2003], (2000.0, 3000.0, 0.0), (500.0, 700.0)), old)  # no flags: unread
    assert not sums.has_team_flags and sums.flag_columns().empty


def test_flags_sit_at_the_end_of_each_season_and_carry_their_own_penalty(world, teams):
    _, designs = world
    sums = _sums(designs, SEASONS, teams=teams)
    assert sums.has_team_flags
    penalty = sums.penalty((), (2000.0, 3000.0, 0.0), (500.0, 700.0))
    columns = sums.flag_columns()
    assert len(columns) == 8 * len(SEASONS)
    assert (penalty[columns.attacking] == 500.0).all() and (penalty[columns.defending] == 700.0).all()
    assert (penalty == 0.0).sum() == len(SEASONS) * len(designs[2001].spec.context_names)
    with pytest.raises(ValueError, match="above 0"):
        sums.penalty((), (2000.0, 3000.0, 0.0))


def test_a_left_out_season_drops_its_flags_and_still_equals_fitting_without_it(world, teams):
    _, designs = world
    pair = (500.0, 500.0)
    sums = _sums(designs, SEASONS, teams=teams)
    assert 2003 not in set(sums.flag_columns([2003]).season)
    held = sums.ratings([2003], _pen(2000.0), pair).set_index("player_id")
    others = [s for s in SEASONS if s != 2003]
    direct = _sums(designs, others, teams=teams).ratings((), _pen(2000.0), pair).set_index("player_id")
    shared = held.index.intersection(direct.index)
    assert len(shared) > 50
    assert np.allclose(held.loc[shared, "offense"], direct.loc[shared, "offense"], atol=1e-8)
    assert np.allclose(held.loc[shared, "defense"], direct.loc[shared, "defense"], atol=1e-8)


def test_flags_take_a_planted_team_season_effect_off_the_players(world, teams, planted):
    """Plant a team-season effect (sd 4 per 100 a side): without flags the players absorb it; with them the flags
    recover it and the players' ratings track their simulated talent better."""
    sim, _ = world
    shifted, effects = planted
    truth = sim["truth"]["ps"].groupby("player_id")[["impact_O", "impact_D"]].mean().reset_index()

    def agreement(ratings):
        joined = ratings.merge(truth, on="player_id")
        return (np.corrcoef(joined.offense, joined.impact_O)[0, 1], np.corrcoef(joined.defense, joined.impact_D)[0, 1])

    plain = agreement(_sums(shifted, SEASONS).ratings((), _pen(1000.0)))
    sums = _sums(shifted, SEASONS, teams=teams)
    flagged = agreement(sums.ratings((), _pen(1000.0), (200.0, 200.0)))
    assert flagged[0] > plain[0] + 0.03 and flagged[1] > plain[1] + 0.03
    theta, _ = sums.solve((), _pen(1000.0), (200.0, 200.0))
    columns = sums.flag_columns()
    for side, column in ((0, "attacking"), (1, "defending")):
        want = [effects[(season, team)][side] for season, team in zip(columns.season, columns.team_id)]
        assert np.corrcoef(theta[columns[column]], want)[0, 1] > 0.6


def test_career_labels_refuse_sums_built_the_other_way(world, teams):
    from dataclasses import replace
    _, designs = world
    plain = {PRODUCT.targets.offence: _sums(designs, SEASONS, 100.0),
             PRODUCT.targets.defence: _sums(designs, SEASONS, 100.0)}
    flagged = {target: _sums(designs, SEASONS, 100.0, teams) for target in plain}
    on = replace(PRODUCT.labels, team_offense_penalty=500.0, team_defense_penalty=500.0)
    with pytest.raises(ValueError, match="without each row's teams"):
        rapm.career_labels(plain, [2004], on, PRODUCT.targets)
    with pytest.raises(ValueError, match="with each row's teams"):
        rapm.career_labels(flagged, [2004], PRODUCT.labels, PRODUCT.targets)
    labels = rapm.career_labels(flagged, [2004], on, PRODUCT.targets)
    assert labels["O"].equals(flagged[PRODUCT.targets.offence].ratings([2004], (40000.0, 40000.0, 0.0),
                                                                        (500.0, 500.0)))


# ------------------------------------------------------------ the un-shrunk label
def test_unshrink_on_a_hand_example():
    """Two players in the 500-1,500 tier and one above full trust, at lambda = 40,000 and N = 4,444."""
    table = pd.DataFrame({"player_id": [1, 2, 3], "offense": [0.3, 0.2, 0.1],
                          "defense": [0.05, -0.02, 0.5], "possessions": [1000.0, 1200.0, 10000.0]})
    got = rapm.unshrink(table, "defense", 40000.0)
    s = np.array([1000 / 41000, 1200 / 41200, 10000 / 50000])
    s_cap = 4444 / 44444
    tier_level = (0.05 - 0.02) / 2 / ((s[0] + s[1]) / 2)
    assert got.defense[0] == pytest.approx(0.05 / s_cap + (1 - s[0] / s_cap) * tier_level, rel=1e-12)
    assert got.defense[1] == pytest.approx(-0.02 / s_cap + (1 - s[1] / s_cap) * tier_level, rel=1e-12)
    assert got.defense[2] == pytest.approx(0.5 / 0.2, rel=1e-12)          # above N: the raw-scale coefficient
    assert got.offense.tolist() == table.offense.tolist()                    # the other column is untouched
    assert table.defense.tolist() == [0.05, -0.02, 0.5]                      # the input is not modified


def test_career_labels_unshrink_only_the_defence(world):
    _, designs = world
    sums = {PRODUCT.targets.offence: _sums(designs, SEASONS, 100.0),
            PRODUCT.targets.defence: _sums(designs, SEASONS, 100.0)}
    labels = rapm.career_labels(sums, [2004], PRODUCT.labels, PRODUCT.targets)
    plain = sums[PRODUCT.targets.offence].ratings([2004], (40000.0, 40000.0, 0.0))
    assert labels["O"].equals(plain)
    assert labels["D"].equals(rapm.unshrink(plain, "defense", 40000.0))
    assert labels["D"].defense.std() > 3 * plain.defense.std()
