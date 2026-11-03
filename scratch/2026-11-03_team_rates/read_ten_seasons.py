"""Read-only: does Box Adj gain from team numbers, on the held-back lineup swaps of ten seasons (the owner, 2026-11-03:
"i have a hunch that the reason certain stats are being thrown into the 'lineup swaps' ... is because they are adjusting
for the 'team share' so - I wonder if we should roll up lots more 'team share' metrics in here (like team ORB100, team
TOV 100) as well").

Each fold's inputs are the ones the shipped line ctx110 was fitted on (outputs/second_prior/ctx110/inputs_<season>.parquet),
so no booster is asked anything.  Candidates, all from the fold's fit games, all standardised:
    season_<stat>   his team's rate per 100 over the season, by his possessions with each team
    on_<stat>       his team's rate with him on the court, padded toward season_<stat>
    off_<stat>      the same with him on the bench in the games he played
    share_<count>   his padded own rate over his team's total of it, for the 13 box counts
    shadow_...      each of those shuffled among the fold's players, the noise yardstick
On offence the rates are his team's; on defence the rates of the teams facing his.  Each season is held out once and
the line refitted on the other nine; loss is the swap information's weighted mean squared miss of the line's target.

Result: read_ten_seasons.log.  notes/DECISIONS.md, experiment 128.

    .venv/Scripts/python scratch/2026-11-03_team_rates/read_ten_seasons.py
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
SEASONS = [1998, 2001, 2004, 2007, 2010, 2013, 2016, 2019, 2022, 2025]
ERAS = {'1997-2006': [1998, 2001, 2004], '2007-2016': [2007, 2010, 2013, 2016], '2017-2026': [2019, 2022, 2025]}
STATS = ['threes_made', 'threes_missed', 'twos_made', 'twos_missed', 'free_throws_made', 'free_throws_missed',
         'off_rebounds', 'turnovers']
cfg = load_config()
base_dir = resolve(cfg, 'outputs') / 'second_prior' / 'ctx110'
run = json.loads((base_dir / 'run.json').read_text())
scale = run['type_scale']
BASE = list(run['inputs'])
RATES = [c for c in BASE if c.endswith('_per100')]
panel = pd.read_parquet(resolve(cfg, 'outputs') / 'season_panel.parquet',
                        columns=['season', 'side', *RATES, *[r + '_uncentred' for r in RATES]])
panel = panel[panel.side == 'O']
league = {r: panel.groupby('season').apply(lambda g: float((g[r + '_uncentred'] - g[r]).mean())) for r in RATES}
del panel
shrink = replace(TEST.shrink, folds_tag='creation110_early,creation110_within')
folds, where = step4_folds(cfg, shrink)
mult = step4_multipliers(sorted({f.season for f in folds}), folds, shrink)
print('folds loaded', len(folds), round(time.time() - t0), flush=True)


def side_counts(t, s):
    g = lambda c: t[f'{c}_{s}'].to_numpy(float)
    return np.column_stack([g('fg3m'), g('fg3a') - g('fg3m'), g('fgm') - g('fg3m'),
                            (g('fga') - g('fg3a')) - (g('fgm') - g('fg3m')), g('ftm'), g('fta') - g('ftm'),
                            g('oreb_p'), g('tov')])


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
    return out, ids, E


def share_numbers(inp, ids, E, season):
    own = pd.DataFrame({r: inp[r] * scale[r][1] + scale[r][0] + league[r].get(season, 0.0) for r in RATES},
                       index=inp.index).clip(lower=0.0)
    arr = own.reindex(ids).fillna(0.0).to_numpy(float)
    mix = E.mix.tocsr() if hasattr(E.mix, 'tocsr') else E.mix
    per_team = 5.0 * np.asarray(mix.T @ arr) / np.maximum(np.asarray(mix.sum(axis=0)).ravel(), 1.0)[:, None]
    his = np.asarray(mix @ per_team) / np.maximum(np.asarray(mix.sum(axis=1)).ravel(), 1.0)[:, None]
    his = np.where(his > 1e-9, his, np.nan)
    return pd.DataFrame(arr / his, index=ids, columns=['share_' + r.replace('_per100', '') for r in RATES])


store = {}
rng = np.random.default_rng(0)
acc = {}
names = None
for season in SEASONS:
    rows, counts = season_tables(season)
    game = rows.game_id.astype(str)
    stored = pd.read_parquet(base_dir / f'inputs_{season}.parquet')
    factors = prior_shrink.season_factors(mult, season)
    for fold in [f for f in folds if f.season == season]:
        fit_ids, held_ids = fold_games(where[fold.stem], fold)
        fit = game.isin(fit_ids).to_numpy()
        numbers, ids, E = team_numbers(rows[fit].reset_index(drop=True), counts[fit])
        held = swap_test.prepare(rows[game.isin(held_ids).to_numpy()].reset_index(drop=True), season)
        ratings = swap_test.ratings_in_raw_sign(prior_shrink.shrink_fold(fold.rows, season, factors))
        pairs = second_prior.swap_targets(swap_adjust.residual_pairs(held, ratings, swap_adjust.RESIDUAL_CONTEXT),
                                          ratings)
        here = stored[(stored.repeat == fold.repeat) & (stored.fold == fold.fold)]
        for long_side, side in second_prior.SIDE_KEYS.items():
            inp = here[here.side == side].set_index('player_id')[BASE].astype(float)
            cand = numbers[side].reindex(inp.index).join(share_numbers(inp, ids, E, season).reindex(inp.index))
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
ONS = [f'on_{s}' for s in STATS]
OFFS = [f'off_{s}' for s in STATS]
SEAS = [f'season_{s}' for s in STATS]
SHARES = [n for n in names if n.startswith('share_')]
SETS = {'box prior alone': ['box_prior'], 'today': BASE, 'team season rate': BASE + SEAS,
        'team rate him on': BASE + ONS, 'team rate him off': BASE + OFFS, 'team rate on and off': BASE + ONS + OFFS,
        'his share of team stat': BASE + SHARES, 'on off and share': BASE + ONS + OFFS + SHARES,
        'SHUFFLED on off': BASE + ['shadow_' + c for c in ONS + OFFS],
        'SHUFFLED share': BASE + ['shadow_' + c for c in SHARES]}


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


for side, label in (('O', 'OFFENSE'), ('D', 'DEFENSE')):
    print('\n=====', label, 'held-out season loss per unit of swap information')
    res = {k: np.array([loss(side, y, c, fit(side, [o for o in SEASONS if o != y], c)) for y in SEASONS])
           for k, c in SETS.items()}
    gain = float((res['box prior alone'] - res['today']).mean())
    null = np.mean([store[(y, side)]["yy"] / store[(y, side)]["sh"] for y in SEASONS])
    print(f'null loss {null:.3f}; today {res["today"].mean():.3f}; what the 28 other inputs add beyond the box prior '
          f'{gain:.4f}')
    print('set | change vs today | se | z | seasons better of 10 | pct of what the 28 add | ' + ' | '.join(ERAS))
    for k in SETS:
        if k == 'today':
            continue
        dd = res[k] - res['today']
        z = dd.mean() / (dd.std(ddof=1) / np.sqrt(len(dd)))
        eras = ' | '.join(f'{np.mean([dd[SEASONS.index(y)] for y in ys]):+.4f}' for ys in ERAS.values())
        print(f'{k} | {dd.mean():+.4f} | {dd.std(ddof=1)/np.sqrt(len(dd)):.4f} | {z:+.2f} | {int((dd < 0).sum())} | '
              f'{100 * -dd.mean() / gain:+.1f}% | {eras}')
    print('\nweights: input | today | with on off and share | size today | size with')
    f = scaled(side)
    sh = sum(store[(y, side)]['sh'] for y in SEASONS)
    Gd = np.diag(sum(store[(y, side)]['G'] for y in SEASONS)) * f * f
    bt, ba = fit(side, SEASONS, BASE), fit(side, SEASONS, SETS['on off and share'])
    weights = dict(zip(SETS['on off and share'], ba))
    for c in BASE:
        rms = np.sqrt(Gd[idx[c]] / sh)
        print(f'{c} | {bt[BASE.index(c)]:+.3f} | {weights[c]:+.3f} | {abs(bt[BASE.index(c)]) * rms:.3f} | '
              f'{abs(weights[c]) * rms:.3f}')
    print('\nnew inputs: input | weight per sd | size | weight by era 1997-2006 | 2007-2016 | 2017-2026')
    per = {e: dict(zip(SETS['on off and share'], fit(side, ys, SETS['on off and share']))) for e, ys in ERAS.items()}
    for c in ONS + OFFS + SHARES:
        rms = np.sqrt(Gd[idx[c]] / sh)
        print(f'{c} | {weights[c]:+.3f} | {abs(weights[c]) * rms:.3f} | ' + ' | '.join(f'{per[e][c]:+.3f}' for e in ERAS))
    print('\nswap difference spread, rms, season rate columns, should be near 0: '
          + ', '.join(f'{c}={np.sqrt(Gd[idx[c]] / sh):.3f}' for c in SEAS))
print('done', round(time.time() - t0), flush=True)
