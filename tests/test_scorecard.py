"""src/eracoef/scorecard.py on a synthetic league with known ratings: the error and the slope do what they claim, the
blend weight agrees with the error, and the team-mean control shows what team-game scoring cannot see."""
import numpy as np
import pandas as pd
import scipy.sparse as sp

from eracoef import scorecard as sc


def _league(seed=0, n_seasons=8, n_teams=10, roster=10, games=240, fixed_rotation=False, noise=1.0):
    """Blocks of team-games (one per season) and the true ratings (raw sign: d adds points allowed)."""
    rng = np.random.default_rng(seed)
    n = n_teams * roster
    ids = np.arange(1, n + 1) * 7
    team = np.repeat(np.arange(n_teams), roster)
    true = pd.DataFrame({"player_id": ids, "o": rng.normal(0, 2.0, n), "d": rng.normal(0, 1.5, n)})
    minutes = rng.uniform(0.2, 1.0, n)
    fixed = {}
    for t in range(n_teams):
        members = np.flatnonzero(team == t)
        share = minutes[members] / minutes[members].sum() * 5
        share = np.minimum(share, 0.95)
        fixed[t] = (members, share / share.sum() * 5)

    def shares(t):
        members, base = fixed[t]
        if fixed_rotation:
            return members, base
        counts = np.zeros(len(members))
        poss = rng.integers(5, 20, size=8)
        for p in poss:
            g = np.log(minutes[members]) + rng.gumbel(size=len(members))
            counts[np.argpartition(-g, 5)[:5]] += p
        return members, counts / poss.sum()          # each stint puts five players on the court: shares sum to 5

    blocks = []
    for k, season in enumerate(range(2001, 2001 + n_seasons)):
        rows, ys, ws, homes = [], [], [], []
        for g in range(games):
            a, b = rng.choice(n_teams, size=2, replace=False)
            for off, de, home in ((a, b, 1.0), (b, a, -1.0)):
                z = np.zeros(2 * n)
                mo, so = shares(off)
                md, sd = shares(de)
                z[mo], z[n + md] = so, sd
                poss = rng.uniform(90, 105)
                mean = 110 + 1.5 * home + z[:n] @ true.o.to_numpy() + z[n:] @ true.d.to_numpy()
                ys.append(mean + noise * rng.normal(0, 110 / np.sqrt(poss)))
                rows.append(z)
                ws.append(poss)
                homes.append(home)
        homes = np.array(homes)
        F = np.column_stack([homes, np.ones(len(homes))])
        blocks.append(sc.Block(key=f"{season}", season=season, deal=0, Z=sp.csr_matrix(np.array(rows)),
                               y=np.array(ys), w=np.array(ws), F=F, home=homes, player_ids=ids))
    return blocks, true, pd.Series(team, index=ids), pd.Series(minutes, index=ids)


def _slopes(blocks, ratings):
    return sc.pooled([sc.side_normal(b, sc.parts(b, ratings)) for b in blocks])


def test_the_truth_has_slope_one_and_a_doubled_rating_half_of_it_exactly():
    blocks, true, _, _ = _league(n_seasons=16)
    s1 = _slopes(blocks, true)
    # unbiased: over seeds the truth averages 0.999 / 0.998 with a seed-to-seed sd of about 0.02; a jackknife over
    # few seasons is itself noisy (t with seasons - 1 degrees of freedom), so the check is on the coefficient
    assert (abs(s1.coef - 1.0) < np.maximum(0.06, 3 * s1.se)).all(), s1
    doubled = true.assign(o=2 * true.o, d=2 * true.d)
    s2 = _slopes(blocks, doubled)
    assert np.allclose(s2.coef.to_numpy(), s1.coef.to_numpy() / 2, atol=1e-10)


def test_zero_ratings_score_exactly_the_no_ratings_error():
    blocks, true, _, _ = _league(n_seasons=2)
    zero = true.assign(o=0.0, d=0.0)
    for b in blocks:
        assert abs(sc.error(b, sc.parts(b, zero).total) - sc.error(b, None)) < 1e-9


def test_the_blend_weight_is_above_one_half_exactly_when_the_candidate_has_the_lower_error():
    blocks, true, _, _ = _league(n_seasons=3)
    rng = np.random.default_rng(3)
    noisy = true.assign(o=true.o + rng.normal(0, 2.0, len(true)), d=true.d + rng.normal(0, 2.0, len(true)))
    zero = true.assign(o=0.0, d=0.0)
    for a, b in ((true, noisy), (zero, true), (noisy, true), (zero, noisy)):
        for blk in blocks:
            pa, pb = sc.parts(blk, a), sc.parts(blk, b)
            theta = sc.solve([sc.encompass_normal(blk, pa, pb, level="home", by_side=False)]).iloc[0]
            lower = sc.error(blk, pb.total) < sc.error(blk, pa.total)
            assert (theta > 0.5) == lower


def test_the_blend_weight_reads_signal_and_noise():
    blocks, true, _, _ = _league(n_seasons=16)
    rng = np.random.default_rng(5)
    noise_only = true.assign(o=true.o + rng.normal(0, 2.0, len(true)), d=true.d + rng.normal(0, 2.0, len(true)))
    zero = true.assign(o=0.0, d=0.0)
    to_truth = sc.pooled([sc.encompass_normal(b, sc.parts(b, zero), sc.parts(b, true)) for b in blocks])
    to_noise = sc.pooled([sc.encompass_normal(b, sc.parts(b, true), sc.parts(b, noise_only)) for b in blocks])
    assert (abs(to_truth.coef - 1.0) < np.maximum(0.06, 3 * to_truth.se)).all(), to_truth
    assert (abs(to_noise.coef) < np.maximum(0.06, 3 * to_noise.se)).all(), to_noise


def test_team_game_scoring_cannot_see_the_split_when_rotations_are_fixed():
    blocks, true, team, minutes = _league(fixed_rotation=True, n_seasons=2)
    weight = minutes / minutes.sum()
    # with a fixed rotation every row of a team carries the same shares, so the team's possession-share-weighted
    # mean is the only thing the rows see; the control keeps it, by the shares themselves
    share = pd.Series(np.asarray(blocks[0].Z[:, :len(true)].max(axis=0).todense()).ravel(), index=true.player_id)
    control = sc.team_mean_control(true, team, share)
    for b in blocks:
        assert abs(sc.error(b, sc.parts(b, control).total) - sc.error(b, sc.parts(b, true).total)) < 1e-8


def test_a_league_wide_shuffle_has_slope_near_zero():
    blocks, true, _, minutes = _league()
    shuffled = sc.shuffle_within_bins(true, minutes, np.random.default_rng(9), n_bins=5)
    s = _slopes(blocks, shuffled)
    assert (abs(s.coef) < 3 * s.se + 0.15).all(), s


def test_cross_fitted_multipliers_leave_out_the_season_and_its_neighbours():
    blocks, true, _, _ = _league()
    eqs = [sc.side_normal(b, sc.parts(b, true)) for b in blocks]
    m = sc.crossfit_multipliers(eqs, [2004]).set_index("season")
    assert m.at[2004, "blocks"] == len(blocks) - 3
    manual = sc.solve([e for e in eqs if e.season not in (2003, 2004, 2005)])
    assert abs(m.at[2004, "m_O"] - manual["O"]) < 1e-12


def test_variance_components_split_seasons_from_deals():
    rng = np.random.default_rng(1)
    rows = [dict(season=s, deal=r, d=season_effect + rng.normal(0, 0.01))
            for s, season_effect in zip(range(20), rng.normal(0, 1.0, 20)) for r in range(3)]
    v = sc.variance_components(pd.DataFrame(rows))
    assert v["se_unlimited_deals"] <= v["se_now"]
    assert abs(v["se_unlimited_deals"] - v["se_now"]) < 0.01


def test_moved_and_stayed_weights_split_the_error_exactly():
    blocks, true, team, _ = _league(n_seasons=1)
    b = blocks[0]
    rng = np.random.default_rng(8)
    noisy = true.assign(o=true.o + rng.normal(0, 1, len(true)), d=true.d + rng.normal(0, 1, len(true)))
    p = sc.parts(b, noisy)
    # half the league counted as "moved": their team in the rating games is a team they never play for here
    t_rating = team.copy()
    t_rating[t_rating.index[::2]] = 99
    t_row = np.array([0] * len(b.y))                       # rows' teams: the share only needs "not 99"
    share = sc.moved_share(b, t_rating, t_row, t_row)
    assert ((share >= -1e-12) & (share <= 1 + 1e-12)).all() and share.max() > 0
    q_all = sc.quadratic(b, p)
    q_m = sc.quadratic(b, p, score_weights=b.w * share)
    q_s = sc.quadratic(b, p, score_weights=b.w * (1 - share))
    total = sc.error_at(q_all) * q_all["sw"]
    assert abs(sc.error_at(q_m) * q_m["sw"] + sc.error_at(q_s) * q_s["sw"] - total) < 1e-6 * total


def test_the_traded_split_adds_back_to_the_whole_prediction():
    blocks, true, team, _ = _league(n_seasons=1)
    b = blocks[0]
    t_rating = team.astype(float).copy()
    t_rating[t_rating.index[::3]] = 99
    t_row = np.zeros(len(b.y))
    Zt = sc.traded_entries(b, t_rating, t_row, t_row)
    stay, traded = sc.split_traded(b, Zt)
    whole = sc.parts(b, true, (0.3, -0.2))
    ps, pt = sc.parts(stay, true, (0.3, -0.2)), sc.parts(traded, true, (0.3, -0.2))
    for f in ("c_o", "c_d", "s_o", "s_d"):
        assert np.allclose(getattr(ps, f) + getattr(pt, f), getattr(whole, f), atol=1e-12)
    assert np.allclose(np.asarray(Zt.sum(axis=1)).ravel() / 10.0, sc.moved_share(b, t_rating, t_row, t_row))
    # an entry is traded exactly when the player's rating-games team is not the row's team (every row's team is 0)
    n = b.n_players
    traded_cols = set(np.unique(Zt.tocoo().col % n))
    on_court = set(np.unique(b.Z.tocoo().col % n))
    other_team = {j for j in on_court if t_rating.reindex(b.player_ids[[j]]).iloc[0] != 0}
    assert traded_cols == other_team
