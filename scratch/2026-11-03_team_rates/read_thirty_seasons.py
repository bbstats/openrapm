"""Read-only: the team numbers of read_ten_seasons.py on all thirty seasons, split so team scoring can be told from the
team's stat profile (the first read's gain could have been his on-court points arriving by parts).

Sets beside today's line: the team's season rates; team ORB and TOV; team points alone; the profile without makes,
threes, twos and free throws attempted, ORB and TOV; points plus the profile; all eight rates; with him on the court, and
with him on and off, plus shuffled copies.  Each season is predicted by the line fitted without it and its neighbors.
The ten seasons of the input screens are kept apart from the other twenty in the last two columns.

Result: read_thirty_seasons.log.  notes/DECISIONS.md, experiment 128.

    .venv/Scripts/python scratch/2026-11-03_team_rates/read_thirty_seasons.py
"""
import json
import time
from dataclasses import replace

import numpy as np
import pandas as pd

from openrapm import prior_shrink, second_prior, swap_adjust, swap_test
from openrapm.commands.build import step4_folds, step4_multipliers
from openrapm.data.config import load_config, resolve
from openrapm.data.stints import load_season_stints
from openrapm.data.within_folds import fold_games
from openrapm.second_prior import _Entries, pad_toward_level
from openrapm.settings import TEST

t0 = time.time()
SEASONS = list(range(1997, 2027))
TUNE = [1998, 2001, 2004, 2007, 2010, 2013, 2016, 2019, 2022, 2025]
ERAS = {'1997-2006': range(1997, 2007), '2007-2016': range(2007, 2017), '2017-2026': range(2017, 2027)}
STATS = ['threes_made', 'threes_missed', 'twos_made', 'twos_missed', 'free_throws_made', 'free_throws_missed',
         'off_rebounds', 'turnovers', 'points', 'threes_att', 'twos_att', 'ft_att']
cfg = load_config()
base_dir = resolve(cfg, 'outputs') / 'second_prior' / 'ctx110'
run = json.loads((base_dir / 'run.json').read_text())
BASE = list(run['inputs'])
OWN = [c for c in BASE if c.endswith('_per100')]
shrink = replace(TEST.shrink, folds_tag='creation110_early,creation110_within')
folds, where = step4_folds(cfg, shrink)
mult = step4_multipliers(sorted({f.season for f in folds}), folds, shrink)
print('folds loaded', len(folds), round(time.time() - t0), flush=True)


def side_counts(t, s):
    g = lambda c: t[f'{c}_{s}'].to_numpy(float)
    return np.column_stack([g('fg3m'), g('fg3a') - g('fg3m'), g('fgm') - g('fg3m'),
                            (g('fga') - g('fg3a')) - (g('fgm') - g('fg3m')), g('ftm'), g('fta') - g('ftm'),
                            g('oreb_p'), g('tov'), g('pts'), g('fg3a'), g('fga') - g('fg3a'), g('fta')])


def season_tables(season):
    cols = list(swap_test.STINT_COLUMNS) + [f'{c}_{s}' for c in ['fg3m', 'fg3a', 'fgm', 'fga', 'ftm', 'fta', 'oreb_p',
                                                                  'tov'] for s in 'ha']
    raw = load_season_stints(season, cfg, cols)
    rows = swap_test.season_rows(raw, margin_clip=float(cfg.get('margin_clip', 25)))
    ordered = swap_test.order_stints(raw)
    counts = np.vstack([side_counts(ordered, s)[ordered[f'poss_{s}'].to_numpy(float) > 0] for s in 'ha'])
    assert len(counts) == len(rows)
    return rows, counts


def team_numbers(rows, counts):
    ids = np.unique(np.concatenate([rows[swap_test.OFF].to_numpy(dtype=np.int64).ravel(),
                                    rows[swap_test.DEF].to_numpy(dtype=np.int64).ravel()]))
    E = _Entries(rows, ids)
    n = len(rows)
    poss = rows.poss.to_numpy(float)
    out = {}
    for side, half in (('O', 0), ('D', 1)):
        num = np.zeros((2 * n, counts.shape[1]))
        den = np.zeros(2 * n)
        w2 = np.zeros(2 * n)
        lo, hi = (0, n) if half == 0 else (n, 2 * n)
        num[lo:hi] = counts
        den[lo:hi] = poss
        w2[lo:hi] = poss ** 2
        c100 = 100.0 * num
        on_d, off_d = E.on(den), np.maximum(E.bench(den), 0.0)
        on_rate = E.on(c100) / np.where(on_d > 0, on_d, 1.0)[:, None]
        off_rate = E.bench(c100) / np.where(off_d > 0, off_d, 1.0)[:, None]
        team = np.asarray(E.teams.T @ c100) / np.maximum(np.asarray(E.teams.T @ den), 1.0)[:, None]
        level = np.asarray(E.mix @ team) / np.where(E.on_weight > 0, E.on_weight, 1.0)[:, None]
        cols = {}
        for j, stat in enumerate(STATS):
            cj = c100[:, j]
            for name, total, rate, weight in (('on', E.on, on_rate, on_d), ('off', E.bench, off_rate, off_d)):
                x = rate[:, j]
                spread = total(cj * cj) - 2.0 * x * total(cj * den) + x * x * total(w2)
                sigma2 = float(np.sum(np.maximum(spread, 0.0)) / max(weight.sum(), 1e-12))
                cols[f'{name}_{stat}'] = pad_toward_level(x, weight, level[:, j], sigma2)
            cols[f'season_{stat}'] = level[:, j]
        out[side] = pd.DataFrame(cols, index=ids)
    return out


store, acc = {}, {}
rng = np.random.default_rng(0)
names = None
for season in SEASONS:
    rows, counts = season_tables(season)
    game = rows.game_id.astype(str)
    stored = pd.read_parquet(base_dir / f'inputs_{season}.parquet')
    factors = prior_shrink.season_factors(mult, season)
    for fold in [f for f in folds if f.season == season]:
        fit_ids, held_ids = fold_games(where[fold.stem], fold)
        fit = game.isin(fit_ids).to_numpy()
        numbers = team_numbers(rows[fit].reset_index(drop=True), counts[fit])
        held = swap_test.prepare(rows[game.isin(held_ids).to_numpy()].reset_index(drop=True), season)
        ratings = swap_test.ratings_in_raw_sign(prior_shrink.shrink_fold(fold.rows, season, factors))
        pairs = second_prior.swap_targets(swap_adjust.residual_pairs(held, ratings, swap_adjust.RESIDUAL_CONTEXT),
                                          ratings)
        here = stored[(stored.repeat == fold.repeat) & (stored.fold == fold.fold)]
        for long_side, side in second_prior.SIDE_KEYS.items():
            inp = here[here.side == side].set_index('player_id')[BASE].astype(float)
            cand = numbers[side].reindex(inp.index)
            cand = cand.fillna(cand.mean())
            shadow = cand[[c for c in cand.columns if not c.startswith('season_')]].copy()
            shadow.iloc[:, :] = np.column_stack([rng.permutation(shadow.iloc[:, j].to_numpy())
                                                 for j in range(shadow.shape[1])])
            shadow.columns = ['shadow_' + c for c in shadow.columns]
            x = pd.concat([inp, cand, shadow], axis=1)
            names = list(x.columns)
            a = store.setdefault((season, side), dict(G=np.zeros((len(names), len(names))), b=np.zeros(len(names)),
                                                      yy=0.0, sh=0.0))
            s = acc.setdefault(side, dict(n=0, s1=np.zeros(len(names)), s2=np.zeros(len(names))))
            s['n'] += len(x)
            s['s1'] += x.to_numpy().sum(axis=0)
            s['s2'] += (x.to_numpy() ** 2).sum(axis=0)
            sw = pairs[long_side]
            keep = sw.fifth_a.isin(x.index).to_numpy() & sw.fifth_b.isin(x.index).to_numpy()
            sw = sw[keep]
            dx = x.reindex(sw.fifth_a.to_numpy()).to_numpy(float) - x.reindex(sw.fifth_b.to_numpy()).to_numpy(float)
            h, d = sw.h.to_numpy(float), sw.r.to_numpy(float)
            a['G'] += (dx * h[:, None]).T @ dx
            a['b'] += dx.T @ (h * d)
            a['yy'] += float((h * d * d).sum())
            a['sh'] += float(h.sum())
    print(season, round(time.time() - t0), flush=True)

idx = {n: i for i, n in enumerate(names)}


def on(L):
    return [f'on_{s}' for s in L]


def off(L):
    return [f'off_{s}' for s in L]


def shuf(L):
    return ['shadow_' + c for c in L]


RATES8 = STATS[:8]
ORBTOV = ['off_rebounds', 'turnovers']
PROFILE = ['threes_att', 'twos_att', 'ft_att', 'off_rebounds', 'turnovers']
SETS = {'box prior alone': ['box_prior'], 'today': BASE,
        'team season rates, 8 stats': BASE + [f'season_{s}' for s in RATES8],
        'owner examples: team ORB and TOV, on': BASE + on(ORBTOV),
        'owner examples: team ORB and TOV, on and off': BASE + on(ORBTOV) + off(ORBTOV),
        'SHUFFLED ORB and TOV, on and off': BASE + shuf(on(ORBTOV) + off(ORBTOV)),
        'team points only, on': BASE + on(['points']),
        'team points only, on and off': BASE + on(['points']) + off(['points']),
        'SHUFFLED points, on and off': BASE + shuf(on(['points']) + off(['points'])),
        'team profile no makes, on': BASE + on(PROFILE),
        'team profile no makes, on and off': BASE + on(PROFILE) + off(PROFILE),
        'SHUFFLED profile, on and off': BASE + shuf(on(PROFILE) + off(PROFILE)),
        'team points plus profile, on and off': BASE + on(['points'] + PROFILE) + off(['points'] + PROFILE),
        'all 8 team rates, on': BASE + on(RATES8),
        'all 8 team rates, on and off': BASE + on(RATES8) + off(RATES8),
        'SHUFFLED 8 rates, on and off': BASE + shuf(on(RATES8) + off(RATES8))}


def scaled(side):
    s = acc[side]
    sd = np.sqrt(np.maximum(s['s2'] / s['n'] - (s['s1'] / s['n']) ** 2, 1e-12))
    f = np.ones(len(names))
    for n in names:
        if n not in BASE:
            f[idx[n]] = 1.0 / sd[idx[n]]
    return f


def fit(side, seasons, cols):
    f = scaled(side)
    j = [idx[c] for c in cols]
    G = sum(store[(y, side)]['G'] for y in seasons)[np.ix_(j, j)] * np.outer(f[j], f[j])
    b = sum(store[(y, side)]['b'] for y in seasons)[j] * f[j]
    return np.linalg.solve(G + 1e-4 * np.mean(np.diag(G)) * np.eye(len(j)), b)


def loss(side, y, cols, beta):
    f = scaled(side)
    j = [idx[c] for c in cols]
    a = store[(y, side)]
    G = a['G'][np.ix_(j, j)] * np.outer(f[j], f[j])
    b = a['b'][j] * f[j]
    return (a['yy'] - 2 * beta @ b + beta @ G @ beta) / a['sh']


def stat(dd):
    return dd.mean(), dd.mean() / (dd.std(ddof=1) / np.sqrt(len(dd)))


for side, label in (('O', 'OFFENSE'), ('D', 'DEFENSE')):
    print('\n=====', label, 'each season held out with its neighbors; change in loss vs today, below 0 better')
    res = {k: np.array([loss(side, y, c, fit(side, [o for o in SEASONS if abs(o - y) > 1], c)) for y in SEASONS])
           for k, c in SETS.items()}
    gain = float((res['box prior alone'] - res['today']).mean())
    print(f'what the 28 other inputs add beyond the box prior {gain:.4f}')
    print('set | change | z | better of 30 | pct of what the 28 add | 1997-2006 change z | 2007-2016 change z | '
          '2017-2026 change z | ten screen seasons z | other twenty z')
    for k in SETS:
        if k == 'today':
            continue
        dd = res[k] - res['today']
        m, z = stat(dd)
        eras = ' | '.join('%+.3f %+.2f' % stat(dd[[SEASONS.index(y) for y in ys]]) for ys in ERAS.values())
        tune = dd[[SEASONS.index(y) for y in TUNE]]
        other = dd[[SEASONS.index(y) for y in SEASONS if y not in TUNE]]
        print(f'{k} | {m:+.3f} | {z:+.2f} | {int((dd < 0).sum())} | {100 * -m / gain:+.1f}% | {eras} | '
              f'{stat(tune)[1]:+.2f} | {stat(other)[1]:+.2f}')
    f = scaled(side)
    sh = sum(store[(y, side)]['sh'] for y in SEASONS)
    Gd = np.diag(sum(store[(y, side)]['G'] for y in SEASONS)) * f * f
    show = ['today', 'owner examples: team ORB and TOV, on and off', 'team points only, on and off',
            'team profile no makes, on and off', 'all 8 team rates, on and off']
    fits = {k: dict(zip(SETS[k], fit(side, SEASONS, SETS[k]))) for k in show}
    print('\nweights of his own numbers: input | ' + ' | '.join(show))
    for c in BASE:
        print(c + ' | ' + ' | '.join(f'{fits[k][c]:+.3f}' for k in show))
    print('sum of size of the 13 own box stats | '
          + ' | '.join(f'{sum(abs(fits[k][c]) * np.sqrt(Gd[idx[c]] / sh) for c in OWN):.2f}' for k in show))
    print('\nnew inputs per sd: input | weight | size | 1997-2006 | 2007-2016 | 2017-2026')
    for k in ['team points only, on and off', 'team profile no makes, on and off',
              'owner examples: team ORB and TOV, on and off']:
        print('-- ' + k)
        per = {e: dict(zip(SETS[k], fit(side, list(ys), SETS[k]))) for e, ys in ERAS.items()}
        for c in SETS[k][len(BASE):]:
            print(f'{c} | {fits[k][c]:+.3f} | {abs(fits[k][c]) * np.sqrt(Gd[idx[c]] / sh):.3f} | '
                  + ' | '.join(f'{per[e][c]:+.3f}' for e in ERAS))
print('done', round(time.time() - t0), flush=True)
