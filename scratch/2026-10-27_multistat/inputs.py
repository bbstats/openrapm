"""What the network reads before each game: every stat and its exposure, summed over four time scales, plus role,
experience, and the baseline's own predictions.  Every number comes from the player's earlier games only.

Words used below
    sum set     one way of adding up a player's earlier games:
                    recent      each game's possessions p discount everything before it by exp(-p / 300)
                                (about the last 4 games of a 75-possession starter)
                    medium      the same with 1500 possessions (about the last 20 games)
                    season      this season so far
                    career      every earlier game
                Each set keeps his possessions and, per stat, his counts (makes for a percentage).
    padded rate (100 x count + 50 x league rate) / (possessions + 50): a light pad, so a set with almost no
                possessions reads as the league rate rather than as 0 or infinity; a percentage is padded with
                10 attempts of the league percentage
    input       for a rate: log(padded rate / league rate); for a percentage: logit(padded) - logit(league), so 0 is
                league average everywhere

Example: over the medium set a guard has 9 assists in 120 possessions: (900 + 50 x 4.8) / 170 = 6.7 per 100, input
log(6.7 / 4.8) = +0.33.
"""
from __future__ import annotations

import numpy as np
from numba import njit

from sim_league import ATTEMPTS_OF, IS_PCT

TAUS = np.array([300.0, 1500.0])
RATE_COLS = np.flatnonzero(~IS_PCT)
PCT_COLS = np.array(sorted(ATTEMPTS_OF))
ATT_COLS = np.array([ATTEMPTS_OF[j] for j in PCT_COLS])
K0, A0 = 50.0, 10.0          # the light pads, in possessions and attempts
ALPHA = 0.1                  # weight of the newest game in the playing-time averages


def input_names(taus=TAUS):
    names = []
    for s in [f"tau{int(t)}" for t in taus] + ["season", "career"]:
        names += [f"{s}_rate_{j}" for j in RATE_COLS] + [f"{s}_pct_{j}" for j in PCT_COLS]
        names += [f"{s}_log_poss"] + [f"{s}_log_att_{j}" for j in ATT_COLS]
    names += ["log_career_games", "log_seasons", "log_season_games", "first_game_of_season", "log_games_since_played",
              "log_last_poss", "log_avg_poss", "share_played", "season_progress"]
    names += [f"baseline_{j}" for j in range(13)]
    return names


@njit(cache=True)
def _inputs(player, season, game, poss, obs, base_pred, league, taus, rate_cols, pct_cols, att_cols, k0, a0, alpha,
            n_games):
    n = player.size
    S = taus.size + 2                                   # recent, medium, season, career
    per_set = rate_cols.size + pct_cols.size + 1 + att_cols.size
    n_in = S * per_set + 9 + 13
    X = np.empty((n, n_in), np.float32)
    E = np.zeros(S)
    C = np.zeros((S, 13))
    lr = np.log(league)
    lp = np.empty(13)
    for j in range(13):
        lp[j] = np.log(league[j]) - np.log1p(-league[j]) if j in (1, 3, 5) else 0.0
    games = 0; seasons = 0; season_games = 0; since = 0; last_poss = 0.0; avg_poss = 35.0; share = 0.8
    for i in range(n):
        new_player = i == 0 or player[i] != player[i - 1]
        if new_player:
            E[:] = 0.0; C[:, :] = 0.0
            games = 0; seasons = 1; season_games = 0; since = 0; last_poss = 0.0; avg_poss = 35.0; share = 0.8
        elif season[i] != season[i - 1]:
            E[S - 2] = 0.0; C[S - 2, :] = 0.0
            seasons += 1; season_games = 0
        c = 0
        for s in range(S):                               # the four sum sets
            for j in rate_cols:
                X[i, c] = np.log((100.0 * C[s, j] + k0 * league[j]) / (E[s] + k0)) - lr[j]; c += 1
            for q in range(pct_cols.size):
                jp, ja = pct_cols[q], att_cols[q]
                pr = (C[s, jp] + a0 * league[jp]) / (C[s, ja] + a0)
                X[i, c] = np.log(pr) - np.log1p(-pr) - lp[jp]; c += 1
            X[i, c] = np.log1p(E[s]); c += 1
            for ja in att_cols:
                X[i, c] = np.log1p(C[s, ja]); c += 1
        X[i, c] = np.log1p(games); X[i, c + 1] = np.log1p(seasons); X[i, c + 2] = np.log1p(season_games)
        X[i, c + 3] = 1.0 if season_games == 0 else 0.0
        X[i, c + 4] = np.log1p(since); X[i, c + 5] = np.log1p(last_poss); X[i, c + 6] = np.log1p(avg_poss)
        X[i, c + 7] = share; X[i, c + 8] = game[i] / n_games
        c += 9
        for j in range(13):                              # the baseline's prediction, 0 = league average
            b = base_pred[i, j]
            if j in (1, 3, 5):
                b = min(max(b, 1e-6), 1 - 1e-6)
                X[i, c + j] = np.log(b) - np.log1p(-b) - lp[j]
            else:
                X[i, c + j] = np.log(max(b, 1e-9)) - lr[j]
        p = poss[i]                                      # then this game enters the sums
        if p > 0:
            for s in range(S):
                f = np.exp(-p / taus[s]) if s < taus.size else 1.0
                E[s] = E[s] * f + p
                for j in range(13):
                    C[s, j] = C[s, j] * f + obs[i, j]
            games += 1; season_games += 1; since = 0; last_poss = p
            avg_poss = (1 - alpha) * avg_poss + alpha * p
            share = (1 - alpha) * share + alpha
        else:
            since += 1
            share = (1 - alpha) * share
    return X


def build_inputs(d, base_pred, league, n_games=82, taus=TAUS):
    return _inputs(d["player"], d["season"], d["game"], d["poss"], d["obs"].astype(np.float64),
                   base_pred.astype(np.float64), league.astype(np.float64), np.asarray(taus, np.float64), RATE_COLS,
                   PCT_COLS, ATT_COLS, K0, A0, ALPHA, float(n_games))


if __name__ == "__main__":
    import time
    from sim_league import simulate
    import baseline as B
    d = simulate()
    played = d["poss"] > 0
    train = played & (d["season"] <= 10)
    ks, x0 = B.fit(d, train)
    bp = B.predict(d, ks, x0)
    t0 = time.perf_counter(); X = build_inputs(d, bp, x0); t = time.perf_counter() - t0
    names = input_names()
    assert X.shape[1] == len(names), (X.shape, len(names))
    print(f"inputs: {X.shape[0]:,} rows x {X.shape[1]} in {t:.2f} s; finite: {np.isfinite(X).all()}")
    # leakage check: an input row must not change when the same row's own outcome changes
    d2 = {k: v.copy() for k, v in d.items()}
    i = np.flatnonzero(played)[1000]
    d2["obs"][i] += 5
    X2 = build_inputs(d2, bp, x0)
    print(f"leakage check: row's own inputs unchanged after editing its outcome: {np.array_equal(X[i], X2[i])}; "
          f"next row changed: {not np.array_equal(X[i + 1], X2[i + 1])}")
