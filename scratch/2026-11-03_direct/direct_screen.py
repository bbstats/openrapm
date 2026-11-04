"""Read-only: which inputs help the rating itself, measured straight on the within-season WOWY test (the owner, 2026-11-03:
"there has to be a direct way to measure which features we can add ... we can use feature selection methods on the box adj
but then it doesn't matter for pi rapm? Don't love that").

Step 3 is a ridge, so a player's rating is a straight line in his prior.  For one fold, Z the fit stints' player columns,
C its context, w its possessions, y a target:

    S        = (the ridge's left-hand side)^-1 over [Z | C], the players penalised, C not
    r(t)     = the player part of S [Z | C]' W t: what the ridge fits to any target t
    u(p)     = r(y - Z p) = r(y) - X p,      X p = the player part of S [Z | C]' W Z p
    rating   = p + u(p) at step 3,   m p + g u(p) inside a team at step 4 (the within layer's multipliers, m and g)

so with the prior p = sum_j beta_j z_j, the prediction of a held-back stint of Z_h is

    Z_h (m p + g r(y) - g X p) = g Z_h r(y) + sum_j beta_j Z_h (m z_j - g X z_j)

Each input z_j becomes one column of the held-back stints, Z_h (m z_j - g X z_j): what the input still says about those
stints once the ridge has read the fit games.  The within-season test takes each fold's team levels, home and playoffs out
and scores the possession-weighted squared miss, so it is a weighted least-squares problem in beta, and any list of inputs
can be fitted and scored on that test itself from one season's normal equations.  An input that restates the fit games'
results loses its weight here by itself, since r(y) reads those games in the same regression.

    build   outputs/direct/<tag>/season_<season>.npz: Q, c, yy, sw over the season's folds (resumes)
    read    the tables: every list and input held out by season with its neighbors, paired against a reference

The checks printed by build: the formula against the ratings each fold stored (step 3, largest gap), and the test's own
errors at step 3 and step 4 from these sums against `151_wowy.py`'s on the same folds.

    .venv/Scripts/python scratch/2026-11-03_direct/direct_screen.py build --tag ctx110_r0 --repeats 0
    .venv/Scripts/python scratch/2026-11-03_direct/direct_screen.py read --tag ctx110_r0
"""
import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from openrapm import second_prior
from openrapm.commands.build import season_lineup_rows, step4_folds
from openrapm.data.config import load_config, resolve
from openrapm.data.designs import DesignCache
from openrapm.data.legacy_names import translate_frame
from openrapm.data.panel import fit_designs
from openrapm.data.within_folds import fold_games
from openrapm.features import BOX_RATES
from openrapm import prior_rapm
from openrapm.rapm import design_blocks, normal_equations
from openrapm.second_prior import ALL_TEAM_RATES, _Entries, pad_toward_level
from openrapm.settings import TEST
from openrapm.swap_test import DEF, OFF

SEASONS = list(range(1997, 2027))
ERAS = {"1997-2006": range(1997, 2007), "2007-2016": range(2007, 2017), "2017-2026": range(2017, 2027)}
FOLDS_TAG = "ctx110_early,ctx110_within"
PENALTY = TEST.ridge.penalty
OFF_TARGET, DEF_TARGET = TEST.targets.offence, TEST.targets.defence
# never candidates: identifiers, outcomes, other seasons (the single-season ruling), league constants, raw counts
NOT_INPUTS = {"side", "player_id", "season", "possessions", "team_id", "repeat", "fold", "test_poss_off", "test_poss_def",
              "rating_off", "rating_def", "rating_total", "prior_off", "prior_def", "poss_off", "poss_def",
              "prior_scale_off", "prior_scale_def", "u_off", "u_def", "prior_raw_off", "prior_raw_def",
              "second_prior_raw_off", "second_prior_raw_def", "seasons_before", "career_possessions_before_k",
              "age_at_first_season", "draft_pick", "seasons_with_team", "seasons_before_from_debut",
              "career_possessions_before_from_debut_k", "age_at_debut", "seasons_with_team_from_debut",
              "season_from_2017", "league_two_point_pct", "league_three_point_pct", "league_points_per_shot",
              "shot_twos_attempted", "shot_twos_made", "shot_twos_expected_made", "shot_threes_attempted",
              "shot_threes_made", "shot_threes_expected_made", "on_court_off_possessions", "on_court_def_possessions",
              "off_court_off_possessions", "off_court_def_possessions"}


def players_file(directory, stem):
    return translate_frame(pd.read_parquet(Path(directory) / f"players_{stem}.parquet"))


def panel_candidates(where, folds, today):
    """The fold-file columns a fold rebuilds from its fit games, so none has seen the held-back games: a column equal
    across two folds of one season for nearly every player is a season total and would leak (measurement trap 37)."""
    first = [f for f in folds if f.season == 2019 and f.repeat == 0][:2]
    a, b = [players_file(where[f.stem], f.stem).query("side == 'O'").set_index("player_id") for f in first]
    shared = a.index.intersection(b.index)
    out, dropped = [], []
    for name in a.columns:
        if name in NOT_INPUTS or name in today or name.endswith("_uncentred") or name.endswith("_shuffled"):
            continue
        if not pd.api.types.is_numeric_dtype(a[name]):
            continue
        x, y = a[name].reindex(shared).to_numpy(float), b[name].reindex(shared).to_numpy(float)
        seen = np.isfinite(x) & np.isfinite(y)
        same = np.mean(np.isclose(x[seen], y[seen])) if seen.any() else 1.0
        (dropped if same > 0.9 else out).append(name)
    return out, dropped


def teammate_numbers(entries, values, keep):
    """Each stat of `values` (players x 13) over his teammates on the court and his team's five when he sits, padded."""
    raw, levels = entries.numbers(values)
    out = {}
    on, bench = entries.on_weight, entries.bench_weight
    for j, stat in enumerate(BOX_RATES):
        noise = entries.noise(values[:, j], keep)
        short = stat.replace("_per100", "")
        out[f"mates_{short}_on"] = pad_toward_level(raw["teammates_{}_on"][:, j], on, levels["teammates"][:, j],
                                                    noise["teammates_{}_on"])
        out[f"mates_{short}_bench"] = pad_toward_level(raw["teammates_{}_bench"][:, j], bench, levels["teammates"][:, j],
                                                       noise["teammates_{}_bench"])
    return out


def demean_within_teams(values: np.ndarray, ids_of_values: np.ndarray, team_poss: pd.DataFrame):
    """`values` (players x columns) moved so every team's possession-weighted total is 0, second_prior.hold_team_totals
    on many columns at once: the line's part beyond the box-score prior only splits a team's credit (experiment 81)."""
    ids, player = np.unique(team_poss.player_id.to_numpy(dtype=np.int64), return_inverse=True)
    teams, team = np.unique(team_poss.team.to_numpy(dtype=np.int64), return_inverse=True)
    possessions = np.zeros((len(teams), len(ids)))
    np.add.at(possessions, (np.asarray(team).ravel(), np.asarray(player).ravel()), team_poss.poss.to_numpy(float))
    mix = (possessions / np.where(possessions.sum(axis=0) > 0, possessions.sum(axis=0), 1.0)).T
    at = pd.Series(np.arange(len(ids_of_values)), index=ids_of_values).reindex(ids)
    own = np.zeros((len(ids), values.shape[1]))
    found = at.notna().to_numpy()
    own[found] = values[at[found].to_numpy(dtype=int)]
    shift = np.linalg.solve(possessions @ mix, possessions @ own)
    return ids, own - mix @ shift


def solve_left(left, right, n_players):
    """The ridge's solve, with rapm.solved_ridge's jitter on the unpenalised diagonal when a context column is all zeros
    (the 2020 bubble's playoff home court)."""
    try:
        return np.linalg.solve(left, right)
    except np.linalg.LinAlgError:
        matrix = left.copy()
        jitter = 1e-9 * max(float(np.mean(np.abs(np.diag(matrix)))), 1.0)
        at = np.arange(n_players, matrix.shape[0])
        matrix[at, at] += jitter
        return np.linalg.solve(matrix, right)


def held_rows(full, directory, stem):
    """(row positions in the season's design, the fold's team levels): as 151_wowy.stint_fold_errors keys them."""
    test = np.load(Path(directory) / f"test_{stem}.npz", allow_pickle=True)
    key = pd.DataFrame({"game_id": test["game_id"].astype(str), "is_home_off": test["is_home_off"].astype(bool),
                        "team_off": test["team_off"], "team_def": test["team_def"]})
    games = full.games.set_index("game_idx").game_id.astype(str)
    rows = full.rows.assign(game_id=full.rows.game_idx.map(games).to_numpy())
    rows = rows.reset_index().merge(key, on=["game_id", "is_home_off"], how="inner")
    off_codes, _ = pd.factorize(rows.team_off)
    def_codes, _ = pd.factorize(rows.team_def)
    n = len(rows)
    levels = sp.hstack([
        sp.csr_matrix((np.ones(n), (np.arange(n), off_codes)), shape=(n, off_codes.max() + 1)),
        sp.csr_matrix((np.ones(n), (np.arange(n), def_codes)), shape=(n, def_codes.max() + 1))[:, 1:],
        sp.csr_matrix(rows.is_home_off.to_numpy(float)[:, None]),
        sp.csr_matrix((rows.phase == "PO").to_numpy(float)[:, None])]).tocsr()
    return rows["index"].to_numpy(), levels


class Screen:
    """What every fold of the run shares."""

    def __init__(self, cfg, args):
        self.cfg, self.args = cfg, args
        shrink = replace(TEST.shrink, folds_tag=FOLDS_TAG, extrapolate_folds_tag="")
        self.folds, self.where = step4_folds(cfg, shrink)
        # the within layer as the ctx110 chain built it: inside a team a rating is m x prior part + g x games' part
        self.within = pd.read_csv(resolve(cfg, "outputs") / "csv" / "season_ratings_ctx110_step4_multipliers.csv").set_index("season")
        self.inputs_dir = resolve(cfg, "outputs") / "second_prior" / "ctx110"
        self.today = list(json.loads((self.inputs_dir / "run.json").read_text())["inputs"])
        self.panel, self.leaky = panel_candidates(self.where, self.folds, set(self.today))
        self.mates = [f"mates_{r.replace('_per100', '')}_{w}" for w in ("on", "bench") for r in BOX_RATES]
        self.candidates = [*ALL_TEAM_RATES, *self.mates, *self.panel]
        per_side = [*self.today, *self.candidates, *[f"shuffled_{c}" for c in self.candidates]]
        self.per_side = per_side
        self.names = ["O:games", "D:games", "O:today_line", "D:today_line",
                      *[f"O:{c}" for c in per_side], *[f"D:{c}" for c in per_side]]
        self.out = resolve(cfg, "outputs") / "direct" / args.tag
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "names.json").write_text(json.dumps(dict(names=self.names, today=self.today, panel=self.panel,
                                                             mates=self.mates, leaky=self.leaky,
                                                             repeats=args.repeats), indent=0))

    def side_table(self, side, fold, stored, players, team_rates, mates, rng):
        """(player ids, players x per_side): one end's inputs for one fold, the box-score prior centred, the rest moved
        so a team's total is 0, each candidate with a copy shuffled among the fold's players."""
        inputs = stored[stored.side == side].set_index("player_id")[self.today].astype(float)
        ids = inputs.index.to_numpy(dtype=np.int64)
        rows = players[players.side == side].set_index("player_id")
        team = team_rates[side].reindex(ids)[ALL_TEAM_RATES].to_numpy(float)
        mate = mates.reindex(ids)[self.mates].to_numpy(float)
        panel = rows.reindex(ids)[self.panel].to_numpy(float)
        candidates = np.column_stack([team, mate, panel])
        means = np.nanmean(candidates, axis=0)
        means = np.where(np.isfinite(means), means, 0.0)
        candidates = np.where(np.isfinite(candidates), candidates, means)
        shadow = np.column_stack([rng.permutation(candidates[:, j]) for j in range(candidates.shape[1])])
        return ids, np.column_stack([inputs.to_numpy(float), candidates, shadow])

    def fold_sums(self, fold, whole, season_rows, stored_all, full_players, rng, check):
        season, stem, directory = fold.season, fold.stem, self.where[fold.stem]
        fit_ids, _ = fold_games(directory, fold)
        designs = fit_designs(season, self.cfg, whole, fit_ids)
        design = designs[OFF_TARGET]
        n = design.spec.n_players
        player_ids = design.spec.players["player_id"].to_numpy(dtype=np.int64)
        Z, C = design_blocks(design)
        w = np.asarray(design.w, dtype=float)
        sums = normal_equations(Z, C, np.asarray(design.y, dtype=float), w)
        y_def = np.asarray(designs[DEF_TARGET].y, dtype=float)
        weighted, weighted_context = Z.T.multiply(w), (C * w[:, None]).T
        rhs_def = np.concatenate([np.asarray(weighted @ y_def).ravel(), weighted_context @ y_def])
        rhs_off = np.concatenate([sums.rhs_players, sums.rhs_context])
        left = np.block([[sums.gram_players + PENALTY * np.eye(2 * n), sums.gram_players_context],
                         [sums.gram_players_context.T, sums.gram_context]])
        # the fold's stored prior part and today's line, raw sign, and every input on its own half of the prior
        rows_o, rows_d = fold.rows["O"].set_index("player_id"), fold.rows["D"].set_index("player_id")
        today_o = rows_o.prior_off.reindex(player_ids).fillna(0.0).to_numpy(float)
        today_d = -rows_d.prior_def.reindex(player_ids).fillna(0.0).to_numpy(float)
        stored = stored_all[(stored_all.repeat == fold.repeat) & (stored_all.fold == fold.fold)]
        players = players_file(directory, stem)
        game = season_rows.game_id.astype(str)
        fit_rows = season_rows[game.isin(fit_ids).to_numpy()].reset_index(drop=True)
        team_rates = second_prior.team_rate_numbers(fit_rows)
        everyone = np.unique(np.concatenate([fit_rows[OFF].to_numpy(dtype=np.int64).ravel(),
                                             fit_rows[DEF].to_numpy(dtype=np.int64).ravel()]))
        entries = _Entries(fit_rows, everyone)
        own = (stored[stored.side == "O"].set_index("player_id")[BOX_RATES].astype(float)
               .reindex(everyone).fillna(0.0).to_numpy())
        mates = pd.DataFrame(teammate_numbers(entries, own, np.flatnonzero(entries.on_weight > 0)), index=everyone)
        k_side = len(self.per_side)
        prior = np.zeros((2 * n, 2 + 2 * k_side))
        prior[:n, 0], prior[n:, 1] = today_o, today_d
        position = pd.Series(np.arange(n), index=player_ids)
        for half, side in enumerate(("O", "D")):
            ids, table = self.side_table(side, fold, stored, players, team_rates, mates, rng)
            box = table[:, 0].copy()
            weight = second_prior.team_possessions(fit_rows, side).groupby("player_id").poss.sum().reindex(ids).fillna(0.0)
            box_centred = box - np.average(box, weights=weight.to_numpy(float))
            moved_ids, moved = demean_within_teams(table, ids, second_prior.team_possessions(fit_rows, side))
            at = position.reindex(moved_ids).to_numpy()
            found = ~np.isnan(at)
            start = 2 + half * k_side
            prior[n * half + at[found].astype(int), start:start + k_side] = moved[found]
            box_at = position.reindex(ids).to_numpy()
            found = ~np.isnan(box_at)
            prior[:, start][:] = 0.0
            prior[n * half + box_at[found].astype(int), start] = box_centred[found]
        right = np.vstack([sums.gram_players @ prior, sums.gram_players_context.T @ prior])
        solved = solve_left(left, np.column_stack([rhs_off, rhs_def, right]), 2 * n)
        fitted_o, fitted_d = solved[:n, 0], solved[n:2 * n, 1]
        explained = solved[:2 * n, 2:]
        if check:
            # step 3 as the formula has it: the stored prior part plus r(y - Z prior), against the stored rating
            gap_o = today_o + fitted_o - explained[:n, 0] - explained[:n, 1]
            gap_d = today_d + fitted_d - explained[n:, 0] - explained[n:, 1]
            stored_o = (rows_o.prior_off + rows_o.u_off).reindex(player_ids).to_numpy(float)
            stored_d = -(rows_d.prior_def + rows_d.u_def).reindex(player_ids).to_numpy(float)
            print(f"  {stem}: the formula, one prior for both fits, against the fold's step-3 ratings, largest gap "
                  f"{np.nanmax(np.abs(gap_o - stored_o)):.1e} offense, {np.nanmax(np.abs(gap_d - stored_d)):.1e} defense",
                  flush=True)
            # the same with each target's own fitted prior scales, as step 3 runs two fits: this one must be exact
            raw_o = rows_o.second_prior_raw_off.reindex(player_ids).fillna(0.0).to_numpy(float)
            raw_d = -rows_d.second_prior_raw_def.reindex(player_ids).fillna(0.0).to_numpy(float)
            for target, half, wanted in ((OFF_TARGET, 0, stored_o), (DEF_TARGET, 1, stored_d)):
                prior_fit = prior_rapm.fit(designs[target], dict(zip(player_ids, raw_o)), dict(zip(player_ids, raw_d)), PENALTY)
                s_o, s_d = prior_fit.prior_scale
                p = np.concatenate([s_o * raw_o, s_d * raw_d])
                y_t = np.asarray(designs[target].y, dtype=float)
                aimed = y_t - Z @ p
                rhs = np.concatenate([np.asarray(weighted @ aimed).ravel(), weighted_context @ aimed])
                rating = p + solve_left(left, rhs, 2 * n)[:2 * n]
                rating = rating[:n] if half == 0 else rating[n:]
                level = np.nanmean(rating - wanted)
                print(f"    {target}: each target's own scales {s_o:.3f} {s_d:.3f}, largest gap "
                      f"{np.nanmax(np.abs(rating - wanted - level)):.1e}", flush=True)
        # the held-back stints
        full = whole["pts"]
        index, levels = held_rows(full, directory, stem)
        y, w_h = np.asarray(full.y, dtype=float)[index], np.asarray(full.w, dtype=float)[index]
        keep = w_h > 0
        rated_o, rated_d = np.isin(player_ids, rows_o.index), np.isin(player_ids, rows_d.index)
        Z_h = full_players[index[keep]]
        Z_o, Z_d = Z_h[:, :n], Z_h[:, n:]
        y, w_h, levels = y[keep], w_h[keep], levels[keep]
        offset = (np.asarray(Z_o[:, ~rated_o].sum(axis=1)).ravel() * fold.fill_o
                  + np.asarray(Z_d[:, ~rated_d].sum(axis=1)).ravel() * fold.fill_d)
        Z_o = Z_o @ sp.diags(rated_o.astype(float))
        Z_d = Z_d @ sp.diags(rated_d.astype(float))
        row = self.within.loc[season]
        m_o, g_o = float(row.within_multiplier_off), float(row.within_games_multiplier_off)
        m_d, g_d = float(row.within_multiplier_def), float(row.within_games_multiplier_def)
        columns = np.column_stack([Z_o @ fitted_o, Z_d @ fitted_d,
                                   Z_o @ (m_o * prior[:n] - g_o * explained[:n]) + Z_d @ (m_d * prior[n:] - g_d * explained[n:])])
        root = np.sqrt(w_h)
        basis = levels.multiply(root[:, None]).toarray()
        u, s, _ = np.linalg.svd(basis, full_matrices=False)
        u = u[:, s > 1e-10 * s.max()]
        matrix = columns * root[:, None]
        matrix -= u @ (u.T @ matrix)
        aim = (y - offset) * root
        aim -= u @ (u.T @ aim)
        return dict(Q=matrix.T @ matrix, c=matrix.T @ aim, yy=float(aim @ aim), sw=float(w_h.sum()),
                    step3=self._step3(Z_o, Z_d, today_o, today_d, fitted_o, fitted_d, explained, y - offset, root, u),
                    multipliers=np.array([m_o, m_d, g_o, g_d]))

    @staticmethod
    def _step3(Z_o, Z_d, today_o, today_d, fitted_o, fitted_d, explained, aim, root, u):
        """The sum of w x the WOWY miss squared at step 3, through the same arrays."""
        rating_o = today_o + fitted_o - explained[:today_o.size, 0] - explained[:today_o.size, 1]
        rating_d = today_d + fitted_d - explained[today_o.size:, 0] - explained[today_o.size:, 1]
        miss = (aim - Z_o @ rating_o - Z_d @ rating_d) * root
        miss -= u @ (u.T @ miss)
        return float(miss @ miss)


def build(args) -> int:
    cfg = load_config()
    screen = Screen(cfg, args)
    print(f"{len(screen.names)} columns; {len(screen.panel)} panel candidates; dropped as season totals: {screen.leaky}",
          flush=True)
    cache = DesignCache(cfg, keep=1)
    repeats = {int(r) for r in args.repeats.split(",")}
    seasons = SEASONS if not args.seasons else [int(s) for s in args.seasons.split(",")]
    started = time.time()
    for season in seasons:
        path = screen.out / f"season_{season}.npz"
        if path.exists():
            continue
        whole = {target: cache.season(season, target) for target in ("pts", OFF_TARGET, DEF_TARGET)}
        full_players, _ = design_blocks(whole["pts"])
        season_rows = season_lineup_rows(cfg, season)
        stored_all = pd.read_parquet(screen.inputs_dir / f"inputs_{season}.parquet")
        rng = np.random.default_rng(season)
        total = None
        step3 = 0.0
        for number, fold in enumerate(f for f in screen.folds if f.season == season and f.repeat in repeats):
            sums = screen.fold_sums(fold, whole, season_rows, stored_all, full_players, rng, number == 0)
            step3 += sums["step3"]
            if total is None:
                total = {key: sums[key] for key in ("Q", "c", "yy", "sw")}
            else:
                for key in ("Q", "c", "yy", "sw"):
                    total[key] = total[key] + sums[key]
        np.savez(path, **total, step3=step3, multipliers=sums["multipliers"])
        print(f"{season}: step-3 error {step3 / total['sw']:.4f}; step-4 at the shipped within layer "
              f"{shipped_error(total, sums['multipliers'], screen.names):.4f} ({time.time() - started:.0f}s)", flush=True)
    return 0


def shipped_error(total, multipliers, names):
    """The test's error at today's ratings through the sums: today's line at 1, the fitted parts at the within layer."""
    beta = np.zeros(len(names))
    beta[names.index("O:today_line")], beta[names.index("D:today_line")] = 1.0, 1.0
    beta[names.index("O:games")], beta[names.index("D:games")] = multipliers[2], multipliers[3]
    return float((total["yy"] - 2.0 * beta @ total["c"] + beta @ total["Q"] @ beta) / total["sw"])


class Sums:
    def __init__(self, directory):
        meta = json.loads((directory / "names.json").read_text())
        self.names, self.today, self.panel, self.mates = meta["names"], meta["today"], meta["panel"], meta["mates"]
        self.position = {name: at for at, name in enumerate(self.names)}
        self.store = {y: dict(np.load(directory / f"season_{y}.npz")) for y in SEASONS
                      if (directory / f"season_{y}.npz").exists()}
        self.seasons = sorted(self.store)
        diag = sum(np.diag(self.store[y]["Q"]) for y in self.seasons) / sum(float(self.store[y]["sw"]) for y in self.seasons)
        self.scale = np.where(diag > 1e-14, 1.0 / np.sqrt(np.where(diag > 1e-14, diag, 1.0)), 0.0)
        self.Q = {y: self.store[y]["Q"] * np.outer(self.scale, self.scale) for y in self.seasons}
        self.c = {y: self.store[y]["c"] * self.scale for y in self.seasons}
        self.yy = {y: float(self.store[y]["yy"]) for y in self.seasons}
        self.sw = {y: float(self.store[y]["sw"]) for y in self.seasons}

    def fit(self, columns, train):
        at = [self.position[c] for c in columns]
        gram = sum(self.Q[y] for y in train)[np.ix_(at, at)]
        right = sum(self.c[y] for y in train)[at]
        penalty = 1e-4 * np.mean(np.diag(gram)) * np.eye(len(at))
        return np.linalg.solve(gram + penalty, right)

    def loss(self, y, columns, weights):
        at = [self.position[c] for c in columns]
        return (self.yy[y] - 2.0 * weights @ self.c[y][at] + weights @ self.Q[y][np.ix_(at, at)] @ weights) / self.sw[y]

    def held_out(self, columns):
        return np.array([self.loss(y, columns, self.fit(columns, [o for o in self.seasons if abs(o - y) > 1]))
                         for y in self.seasons])

    def shipped(self):
        out = []
        for y in self.seasons:
            beta = np.zeros(len(self.names))
            multipliers = self.store[y]["multipliers"]
            for name, value in (("O:today_line", 1.0), ("D:today_line", 1.0), ("O:games", multipliers[2]),
                                ("D:games", multipliers[3])):
                beta[self.position[name]] = value / self.scale[self.position[name]]
            out.append((self.yy[y] - 2.0 * beta @ self.c[y] + beta @ self.Q[y] @ beta) / self.sw[y])
        return np.array(out)

    def line(self, change):
        def stat(values):
            return values.mean(), values.mean() / (values.std(ddof=1) / np.sqrt(len(values)))
        mean, z = stat(change)
        eras = []
        for ys in ERAS.values():
            mask = [self.seasons.index(y) for y in ys if y in self.seasons]
            eras.append(stat(change[mask]) if len(mask) > 2 else (float("nan"), float("nan")))
        return mean, z, int((change < 0).sum()), eras


def both(names):
    return [*[f"O:{n}" for n in names], *[f"D:{n}" for n in names]]


GAMES = ["O:games", "D:games"]


def read(args) -> int:
    cfg = load_config()
    directory = resolve(cfg, "outputs") / "direct" / args.tag
    sums = Sums(directory)
    print(f"{len(sums.seasons)} seasons, {len(sums.names)} columns; each season predicted without itself and its neighbors; "
          "below 0 is better\n")
    today_shipped = sums.shipped()
    today_free = sums.held_out([*GAMES, "O:today_line", "D:today_line"])
    games_only = sums.held_out(GAMES)
    box_only = sums.held_out([*GAMES, "O:box_prior", "D:box_prior"])
    direct = sums.held_out([*GAMES, *both(sums.today)])
    print(f"error today's ratings {today_shipped.mean():.4f}; games only {games_only.mean():.4f}")
    rows = []
    for label, values in (("no prior at all, the games' part only", games_only),
                          ("box-score prior alone, fit through the rating", box_only),
                          ("today's Box Adj, multipliers refit", today_free),
                          ("today's 29 inputs, weights fit through the rating", direct)):
        mean, z, better, eras = sums.line(values - today_shipped)
        rows.append(dict(line=label, change=mean, z=z, better=better, **{e: f"{v[0]:+.3f} {v[1]:+.2f}" for e, v in zip(ERAS, eras)}))
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    profile = list(second_prior.TEAM_PROFILE_RATES)
    lists = {"all eight team rates, on and off": list(second_prior.TEAM_RATES),
             "team attempts, ORB, TOV, on and off": profile,
             "team attempts, ORB, TOV, him on": [c for c in profile if c.endswith("_on")],
             "team attempts, ORB, TOV, him off": [c for c in profile if c.endswith("_off")],
             "teammates' box rates, on and when he sits": sums.mates,
             "on-court and off-court plus-minus": [c for c in sums.panel if c.startswith(("on_court", "off_court", "on_minus_off"))],
             "every candidate": [c[2:] for c in sums.names if c.startswith("O:") and not c.startswith("O:shuffled_")
                                 and c[2:] not in sums.today and c[2:] not in ("today_line", "games")]}
    print("\nEach list added to today's 29 inputs, weights fit through the rating, against the 29 alone")
    rows = []
    for label, names in lists.items():
        mean, z, better, eras = sums.line(sums.held_out([*GAMES, *both(sums.today), *both(names)]) - direct)
        shuffled = both([f"shuffled_{n}" for n in names])
        mean_s, z_s, _, _ = sums.line(sums.held_out([*GAMES, *both(sums.today), *shuffled]) - direct)
        rows.append(dict(list=label, inputs_a_side=len(names), change=mean, z=z, better=better,
                         **{e: f"{v[0]:+.3f} {v[1]:+.2f}" for e, v in zip(ERAS, eras)},
                         shuffled_change=mean_s, shuffled_z=z_s))
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    # one input at a time, each side, against the same inputs' shuffled copies
    candidates = lists["every candidate"]
    base = [*GAMES, *both(sums.today)]
    rows = []
    for name in candidates:
        for side, columns, null in (("offense", [f"O:{name}"], [f"O:shuffled_{name}"]),
                                    ("defense", [f"D:{name}"], [f"D:shuffled_{name}"])):
            mean, z, better, eras = sums.line(sums.held_out([*base, *columns]) - direct)
            mean_s, z_s, _, _ = sums.line(sums.held_out([*base, *null]) - direct)
            rows.append(dict(side=side, input=name, change=mean, z=z, better=better, era1=eras[0][1], era2=eras[1][1],
                             era3=eras[2][1], shuffled_change=mean_s, shuffled_z=z_s))
    table = pd.DataFrame(rows).sort_values("z")
    path = resolve(cfg, "outputs") / "csv" / f"direct_screen_{args.tag}_add_one.csv"
    table.to_csv(path, index=False)
    print(f"\nOne input added at a time, best 30 of {len(table)}; the shuffled copies' z run "
          f"{table.shuffled_z.quantile(0.025):+.2f} to {table.shuffled_z.quantile(0.975):+.2f} for the middle 95%")
    print(table.head(30).round(3).to_string(index=False))
    rows = []
    for name in sums.today:
        for side, prefix in (("offense", "O"), ("defense", "D")):
            kept = [c for c in base if c != f"{prefix}:{name}"]
            mean, z, better, eras = sums.line(sums.held_out(kept) - direct)
            rows.append(dict(side=side, input=name, change_when_dropped=mean, z=z, better=better, era1=eras[0][1],
                             era2=eras[1][1], era3=eras[2][1]))
    drop = pd.DataFrame(rows).sort_values(["side", "z"], ascending=[True, False])
    drop.to_csv(resolve(cfg, "outputs") / "csv" / f"direct_screen_{args.tag}_drop_one.csv", index=False)
    print("\nToday's 29 inputs, one dropped at a time: above 0 means the input earns its place")
    print(drop.round(3).to_string(index=False))
    print(f"\nwrote {path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("build", "read"))
    parser.add_argument("--tag", default="ctx110_r0")
    parser.add_argument("--repeats", default="0")
    parser.add_argument("--seasons", default="")
    args = parser.parse_args()
    return build(args) if args.command == "build" else read(args)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
