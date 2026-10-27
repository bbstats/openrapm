"""A simulated league of player-game box scores whose true rates are known, for testing the multi-stat model.

Words used below
    stat        one of 13 outputs, in this order: 3PA, 3P%, 2PA, 2P%, FTA, FT%, ORB, DRB, AST, TOV, STL, BLK, PF.
                Ten are rates per 100 possessions; three (3P%, 2P%, FT%) are make percentages.
    exposure    a stat's denominator in one game: the player's possessions for a rate, his attempts for a percentage
                (3P% over his 3PA, 2P% over his 2PA, FT% over his FTA)
    link        the scale on which talents add up: log for a rate, logit for a percentage
    factor      one of six shared talents every player carries (size, shooting touch, creation, scoring volume,
                defensive activity, finishing), each a standard normal at the start of his career; a stat's link is
                its league level, plus loadings times factors, plus a part of its own
    true rate   a player's rate or percentage in one game before luck: what a model should predict

The league means follow openrapm's simulate.LEAGUE_MEAN, per 100 possessions: 2.4 made and 4.2 missed threes is
6.6 3PA at 36.4%, 5.6 made and 5.2 missed twos is 10.8 2PA at 51.9%, 3.6 made and 1.0 missed free throws is 4.6 FTA
at 78.3%, then 2.0 ORB, 6.6 DRB, 4.8 AST, 2.8 TOV, 1.6 STL, 1.0 BLK, 4.0 PF.

What moves a player's true rates over time
    aging        piecewise-linear in age per stat: threes rise until 30, blocks and steals fall after 28, ...
    drift        a random walk every game (standard deviation 0.02 per factor) and a change every offseason (0.20)
    role change  in one season in five, one game where creation and scoring volume jump together (a teammate's
                 injury, a trade), and his playing time with them
    injuries     stretches of games he does not play, and a chance of sitting that is higher for weaker players

What a game shows: counts drawn around true rate x possessions / 100 with game-to-game swings (a per-game multiplier
with standard deviation 0.12 shared by all his counts, and 0.12 for each count), then makes drawn from his attempts.

Example: a 7-foot rookie whose size factor is +1.5 standard deviations has a true 3PA rate of about
6.6 x exp(-0.55 x 1.5) = 2.9 per 100 and a block rate of about 1.0 x exp(0.60 x 1.5) = 2.5 per 100.
"""
from __future__ import annotations

import numpy as np

STATS = ["fg3a", "fg3_pct", "fg2a", "fg2_pct", "fta", "ft_pct", "orb", "drb", "ast", "tov", "stl", "blk", "pf"]
IS_PCT = np.array([s.endswith("_pct") for s in STATS])
ATTEMPTS_OF = {1: 0, 3: 2, 5: 4}          # percentage column -> its attempts column
LEAGUE = np.array([6.6, 2.4 / 6.6, 10.8, 5.6 / 10.8, 4.6, 3.6 / 4.6, 2.0, 6.6, 4.8, 2.8, 1.6, 1.0, 4.0])
FACTORS = ["size", "touch", "creation", "volume", "defense", "finishing"]
LOADINGS = np.array([
    #  size  touch  creat   vol    def    fin
    [-0.55, 0.35, 0.10, 0.25, 0.00, 0.00],   # 3PA
    [-0.05, 0.09, 0.00, 0.00, 0.00, 0.00],   # 3P%
    [0.10, 0.00, 0.15, 0.25, 0.00, 0.05],    # 2PA
    [0.15, 0.03, -0.03, -0.05, 0.00, 0.12],  # 2P%
    [0.15, 0.00, 0.15, 0.25, 0.00, 0.10],    # FTA
    [-0.25, 0.30, 0.05, 0.00, 0.00, 0.00],   # FT%
    [0.55, 0.00, 0.00, -0.05, 0.10, 0.00],   # ORB
    [0.35, 0.00, 0.00, 0.00, 0.15, 0.00],    # DRB
    [-0.25, 0.00, 0.45, 0.05, 0.00, 0.00],   # AST
    [-0.05, 0.00, 0.20, 0.15, 0.00, -0.05],  # TOV
    [-0.10, 0.00, 0.05, 0.00, 0.25, 0.00],   # STL
    [0.60, 0.00, 0.00, 0.00, 0.30, 0.00],    # BLK
    [0.12, 0.00, -0.03, 0.00, 0.12, 0.00],   # PF
])
OWN_SD = np.array([0.30, 0.10, 0.15, 0.08, 0.20, 0.20, 0.20, 0.15, 0.20, 0.15, 0.18, 0.30, 0.15])
UP = np.array([0.04, 0.012, 0.0, 0.010, 0.0, 0.010, 0.0, 0.010, 0.03, -0.02, 0.0, 0.0, -0.03])   # per year to PEAK
PEAK = np.array([30, 27, 24, 26, 24, 28, 24, 27, 27, 28, 24, 24, 27])
DOWN = np.array([0.0, -0.008, -0.02, -0.015, -0.03, 0.0, -0.04, -0.01, 0.0, 0.0, -0.04, -0.04, 0.0])   # per year
DECLINE = np.array([40, 31, 30, 31, 30, 40, 28, 31, 40, 40, 28, 28, 40])                            # after this age
QUALITY = np.array([0.0, 0.5, 0.5, 0.4, 0.3, 0.4])   # what earns playing time, on the factors


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def logit(p):
    return np.log(p) - np.log1p(-p)


_VAR = (LOADINGS ** 2).sum(1) + OWN_SD ** 2
_LEAGUE_LINK = np.log(LEAGUE)
_LEAGUE_LINK[IS_PCT] = logit(LEAGUE[IS_PCT])
BASE = np.where(IS_PCT, _LEAGUE_LINK, _LEAGUE_LINK - _VAR / 2)


def aging(age):
    """Link change at `age` (players x 1) relative to age 24, per stat."""
    up = UP * (np.minimum(age, PEAK) - 24.0)
    down = DOWN * np.maximum(age - DECLINE, 0.0)
    return up + down


def simulate(n_seasons=15, roster=450, games=82, seed=0, offsets=None):
    """Returns a dict of arrays, one row per player and team game (games he sat have possessions 0), sorted by
    player, season, game: player, season, game, age, poss, obs (counts for rates, makes for percentages), true.
    `offsets` shifts every stat's link; None calibrates them first (`calibrate`) so the possession-weighted league
    rates and attempt-weighted percentages match LEAGUE."""
    if offsets is None:
        offsets = calibrate(roster=roster, games=games, seed=seed + 10_000, pilot_seasons=n_seasons)
    rng = np.random.default_rng(seed)
    cap = roster * (n_seasons + 2)
    z = np.zeros((cap, 6)); u = np.zeros((cap, 13)); age = np.zeros(cap); hidden = np.zeros(cap)
    count = [0]

    def new_players(m, ages):
        idx = np.arange(count[0], count[0] + m); count[0] += m
        z[idx] = rng.normal(size=(m, 6))
        u[idx] = rng.normal(size=(m, 13)) * OWN_SD
        age[idx] = ages
        hidden[idx] = rng.normal(0.0, 0.7, m)      # what earns minutes but shows in no box count
        return idx

    def quality(ids):
        return z[ids] @ QUALITY + hidden[ids]

    active = new_players(roster, rng.uniform(19, 35, roster))
    out = {k: [] for k in ("player", "season", "game", "age", "poss", "obs", "true")}
    rate_cols = np.flatnonzero(~IS_PCT)
    for s in range(n_seasons):
        if s > 0:                                                      # the offseason
            age[active] += 1.0
            z[active] += rng.normal(0.0, 0.20, (len(active), 6))
            u[active] += rng.normal(0.0, 0.06, (len(active), 13))
            p_leave = sigmoid(-0.9 + 0.35 * (age[active] - 30.0) - 1.0 * quality(active))
            active = active[rng.random(len(active)) > p_leave]
            m = roster - len(active)
            active = np.concatenate([active, new_players(m, rng.uniform(19, 23, m))])
        P = len(active)
        q = quality(active)
        level = 6.0 + 62.0 * sigmoid(1.2 * q - 0.2)                    # possessions per game when he plays
        shift_game = np.where(rng.random(P) < 0.2, rng.integers(10, games - 10, P), -1)
        shift = rng.normal(0.0, 0.45, P)
        inj_start = np.where(rng.random(P) < 0.35, rng.integers(0, games, P), games + 1)
        inj_len = rng.geometric(1.0 / 10.0, P)
        p_sit = 0.03 + 0.20 * (1.0 - sigmoid(1.2 * q))
        for g in range(games):
            z[active] += rng.normal(0.0, 0.02, (P, 6))
            u[active] += rng.normal(0.0, 0.01, (P, 13))
            hit = shift_game == g
            if hit.any():                                              # role change: creation and volume together
                ids = active[hit]
                z[ids, 2] += shift[hit]
                z[ids, 3] += 0.8 * shift[hit] + rng.normal(0.0, 0.15, hit.sum())
                level[hit] = np.clip(level[hit] * np.exp(0.5 * shift[hit]), 3.0, 80.0)
            injured = (g >= inj_start) & (g < inj_start + inj_len)
            plays = ~injured & (rng.random(P) > p_sit)
            poss = np.where(plays, 1 + rng.poisson(level * rng.gamma(8.0, 0.125, P)), 0)
            a = age[active] + g / games
            link = BASE + offsets + z[active] @ LOADINGS.T + u[active] + aging(a[:, None])
            true = np.where(IS_PCT, sigmoid(link), np.exp(link))
            eta = rng.normal(size=(P, 1)); eps = rng.normal(size=(P, 13))
            obs = np.zeros((P, 13))
            mult = np.exp(0.12 * eta + 0.12 * eps - 0.0144)
            obs[:, rate_cols] = rng.poisson(true[:, rate_cols] * mult[:, rate_cols] * poss[:, None] / 100.0)
            for jp, ja in ATTEMPTS_OF.items():
                pi = sigmoid(link[:, jp] + 0.10 * eps[:, jp])
                obs[:, jp] = rng.binomial(obs[:, ja].astype(np.int64), pi)
            out["player"].append(active.copy()); out["season"].append(np.full(P, s)); out["game"].append(np.full(P, g))
            out["age"].append(a); out["poss"].append(poss); out["obs"].append(obs); out["true"].append(true)
    d = {k: np.concatenate(v) for k, v in out.items()}
    order = np.lexsort((d["game"], d["season"], d["player"]))
    d = {k: v[order] for k, v in d.items()}
    d["player"] = d["player"].astype(np.int64); d["season"] = d["season"].astype(np.int64)
    d["game"] = d["game"].astype(np.int64); d["poss"] = d["poss"].astype(np.float64)
    d["offsets"] = np.asarray(offsets, dtype=np.float64)
    return d


def league_rates(d):
    """Possession-weighted rates per 100 and attempt-weighted percentages over the games played."""
    played = d["poss"] > 0
    v = 100.0 * d["obs"][played].sum(0) / d["poss"][played].sum()
    for jp, ja in ATTEMPTS_OF.items():
        v[jp] = d["obs"][played, jp].sum() / d["obs"][played, ja].sum()
    return v


def calibrate(roster=450, games=82, seed=10_000, pilot_seasons=8, rounds=3):
    """Link offsets that bring a pilot league's league_rates to LEAGUE: a log ratio for a rate (exact in one round,
    since a rate scales its counts) and a logit difference for a percentage (a few rounds)."""
    off = np.zeros(13)
    for _ in range(rounds):
        r = league_rates(simulate(pilot_seasons, roster, games, seed, offsets=off))
        off = off + np.where(IS_PCT, 0.0, np.log(LEAGUE / r))
        pct = np.flatnonzero(IS_PCT)
        off[pct] += logit(LEAGUE[pct]) - logit(r[pct])
    return off


if __name__ == "__main__":
    import time
    t0 = time.perf_counter(); d = simulate(); t = time.perf_counter() - t0
    played = d["poss"] > 0
    print(f"{len(d['poss']):,} player-games ({played.sum():,} played), {d['player'].max() + 1:,} players, {t:.1f} s")
    tot_p = d["poss"][played].sum()
    print("observed league means vs targets:")
    lr = league_rates(d)
    for j, s in enumerate(STATS):
        print(f"  {s:8s} {lr[j]:7.3f}  target {LEAGUE[j]:7.3f}")
    rookies = [len(np.setdiff1d(np.unique(d["player"][d["season"] == s]), np.unique(d["player"][d["season"] < s])))
               for s in range(1, d["season"].max() + 1)]
    print(f"new players per season: mean {np.mean(rookies):.0f} of 450 ({np.mean(rookies) / 450:.0%})")
    seasons_per = np.bincount(d["player"][played], minlength=d["player"].max() + 1)
    pp = np.bincount(d["player"], weights=d["poss"]) / np.maximum(np.bincount(d["player"], weights=played), 1)
    print(f"possessions per game played: median {np.median(d['poss'][played]):.0f}, "
          f"10th-90th pct {np.percentile(d['poss'][played], 10):.0f}-{np.percentile(d['poss'][played], 90):.0f}")
    car = [len(np.unique(d["season"][d["player"] == i])) for i in range(0, d["player"].max() + 1, 7)]
    print(f"career length in seasons (sample): mean {np.mean(car):.1f}, median {np.median(car):.0f}")
    # true-talent correlation of FT% and 3P% (logit), weighted by possessions, one row per player-season
    lt = logit(d["true"][played][:, [1, 5]])
    print(f"true-talent correlation 3P% vs FT% (player-games): {np.corrcoef(lt.T)[0, 1]:.2f}")
