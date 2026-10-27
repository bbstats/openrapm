"""A ceiling for the network: a Kalman filter that knows the simulator's true structure.

It knows every setting `sim_league` used (the loadings, each stat's own spread, the drift, the offseason change,
the aging curves, the game-to-game swings) and filters each player's 19 hidden numbers game by game:

    state      his 6 factors and 13 stat-specific parts; at the start of a career they are 0 with standard
               deviations 1 and OWN_SD, as the simulator draws them
    link_j     BASE_j + offset_j + aging_j(age) + loadings_j . factors + part_j
    each game  the state drifts (standard deviation 0.02 per factor, 0.01 per part; role changes are folded in as
               extra drift on creation and volume, since a Kalman filter cannot represent a sudden jump), then his
               counts update it through their exact likelihood: a Poisson count over his possessions for a rate, a
               binomial over his attempts for a percentage, with the simulator's game-to-game swings as extra noise;
               the update is linearized twice (an iterated extended Kalman filter)
    prediction the mean rate given everything before the game: exp(mean + variance / 2) for a rate,
               a probit-corrected logistic for a percentage

It cannot be built on real data (nobody knows real loadings), and the network never sees any of this.  It shows how
much of the gap between the baseline and the truth is reachable at all from the games played.
"""
from __future__ import annotations

import numpy as np
from numba import njit

from sim_league import ATTEMPTS_OF, BASE, DECLINE, DOWN, IS_PCT, LOADINGS, OWN_SD, PEAK, UP

NF, NS = 6, 13
DIM = NF + NS


def _matrices():
    H = np.hstack([LOADINGS, np.eye(NS)])                      # link = c + H state
    Q_game = np.diag(np.r_[np.full(NF, 0.02 ** 2), np.full(NS, 0.01 ** 2)])
    # role changes: one season in five, creation += s, volume += 0.8 s + N(0, 0.15^2), s ~ N(0, 0.45^2), spread over 82
    rv = 0.2 * 0.45 ** 2 / 82.0
    Q_game[2, 2] += rv
    Q_game[3, 3] += 0.2 * (0.8 ** 2 * 0.45 ** 2 + 0.15 ** 2) / 82.0
    Q_game[2, 3] = Q_game[3, 2] = 0.2 * 0.8 * 0.45 ** 2 / 82.0
    Q_off = np.diag(np.r_[np.full(NF, 0.20 ** 2), np.full(NS, 0.06 ** 2)])
    P0 = np.diag(np.r_[np.ones(NF), OWN_SD ** 2])
    return H, Q_game, Q_off, P0


@njit(cache=True)
def _filter(player, season, age, poss, obs, H, Q_game, Q_off, P0, base, up, peak, down, decline, is_pct, att_of,
            n_iter):
    n = player.size
    pred = np.empty((n, NS))
    s = np.zeros(DIM)
    P = P0.copy()
    for i in range(n):
        if i == 0 or player[i] != player[i - 1]:
            s[:] = 0.0
            P[:, :] = P0
        elif season[i] != season[i - 1]:
            P += Q_off
        P += Q_game
        a = age[i]
        c = np.empty(NS)
        for j in range(NS):
            c[j] = base[j] + up[j] * (min(a, peak[j]) - 24.0) + down[j] * max(a - decline[j], 0.0)
        mean = c + H @ s
        var = np.empty(NS)
        HP = H @ P
        for j in range(NS):
            var[j] = HP[j] @ H[j]
            if is_pct[j]:
                pred[i, j] = 1.0 / (1.0 + np.exp(-mean[j] / np.sqrt(1.0 + np.pi * var[j] / 8.0)))
            else:
                pred[i, j] = np.exp(mean[j] + 0.5 * var[j])
        p = poss[i]
        if p <= 0:
            continue
        # which stats this game informs: every rate; a percentage only with attempts
        use = np.ones(NS, np.bool_)
        for j in range(NS):
            if is_pct[j] and obs[i, att_of[j]] <= 0:
                use[j] = False
        idx = np.flatnonzero(use)
        m = idx.size
        Hu = H[idx]
        s_lin = s.copy()
        for it in range(n_iter):                        # iterated extended Kalman update around s_lin
            ell = c[idx] + Hu @ s_lin
            z = np.empty(m)
            R = np.zeros((m, m))
            for q in range(m):
                j = idx[q]
                if is_pct[j]:
                    att = obs[i, att_of[j]]
                    pi = 1.0 / (1.0 + np.exp(-ell[q]))
                    info = att * pi * (1.0 - pi)
                    z[q] = ell[q] + (obs[i, j] - att * pi) / info
                    R[q, q] = 1.0 / info + 0.10 ** 2
                else:
                    mu = np.exp(ell[q]) * p / 100.0
                    z[q] = ell[q] + (obs[i, j] - mu) / mu
                    R[q, q] = 1.0 / mu + 2 * 0.12 ** 2
                    for r in range(m):
                        if r != q and not is_pct[idx[r]]:
                            R[q, r] = 0.12 ** 2            # the per-game swing all his counts share
            # innovation relative to the prior mean, linearized at s_lin
            y_tilde = z - c[idx] - Hu @ s
            S = Hu @ P @ Hu.T + R
            K = np.linalg.solve(S, Hu @ P).T            # P H' S^-1 (S is symmetric)
            s_lin = s + K @ y_tilde                       # posterior mean; the next pass relinearizes here
        P = P - K @ Hu @ P
        P = 0.5 * (P + P.T)
        s[:] = s_lin
    return pred


def kf_predict(d, n_iter=2):
    H, Q_game, Q_off, P0 = _matrices()
    att_of = np.zeros(NS, np.int64)
    for jp, ja in ATTEMPTS_OF.items():
        att_of[jp] = ja
    base = BASE + d["offsets"]
    return _filter(d["player"], d["season"], d["age"], d["poss"], d["obs"].astype(np.float64), H, Q_game, Q_off, P0,
                   base, UP.astype(np.float64), PEAK.astype(np.float64), DOWN.astype(np.float64),
                   DECLINE.astype(np.float64), IS_PCT, att_of, n_iter)
