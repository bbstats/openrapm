"""A calibrator for the ratings' misses, trained on each season's own held-out games (scripts/97_within_season.py).

    python scripts/98_calibrator.py [--tag=within] [--out=calib] [--depth=3] [--rate=0.1] [--rounds=300]
                                    [--min_leaf=300] [--seasons=2017-2026]

The owner, 2026-10-03: "build some sort of calibrator, that takes lots of our inputs and tries to solve for the
calibration misses that we are seeing", trained so it does not shrink peak seasons -- so on the held-out games of
the rated season itself, never its neighbours.

What is fitted.  Two corrections per player, one on offence and one on defence, each a sum of small trees on that
player's inputs (box score, playing time, career, body, on-court and off-court numbers, score state, and how his
rating was built: the prior, the season's adjustment to it, the prior's scale, his possessions).  A correction
enters a held-out team-game exactly as a rating does, times his share of the team's possessions, so the trees are
fitted to the held-out team-games' points per 100 as scored:

    points per 100 of a team-game  =  level and home edge (refit per fold)
                                      + sum over the offence's five of share x (rating + offensive correction)
                                      + sum over the defence's five of share x (rating - defensive correction)

The trees start from the ratings multiplied by ONE number per side (`fit_rescale`, the yardstick), so what they
fit is what a single multiplier cannot.  Each round grows one tree per side on the per-player gradient of that loss (its split search), then sets the two
trees' leaf values TOGETHER by least squares on the team-games -- exact for the leaf values, so five teammates who
land in the same leaf are not each handed the whole team-game's residual.  Learning rate `--rate`.

Which seasons.  The calibrator for season H is trained on the folds of every season except H-1, H and H+1 -- the
seasons the year-over-year test scores H's rating on -- so neither test below has seen its own games.  The number of
rounds is chosen inside the training seasons (half of them against the other half, both ways round).

Two tests:
  within-season    H's part-season ratings, corrected, scored on H's held-out games (the year-over-year test's own
                   scoring, `predict_season` + `score`), against the same ratings uncorrected.  Paired by fold.
  year-over-year   H's WHOLE-season rating, corrected (outputs/season_ratings_<out>_raw.parquet, the 62 schema),
                   through the swap adjustment and scripts/63_yoy.py like any candidate.

Writes outputs/<out>_within.parquet (per fold: error before and after), outputs/season_ratings_<out>_raw.parquet
(the corrected whole-season ratings), outputs/<out>_corrections.parquet (every player's correction, both sides).
"""
import json
import os
import sys
import time
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.sparse as sp  # noqa: E402
from sklearn.tree import DecisionTreeRegressor  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eracoef import singleyear as sy  # noqa: E402
from eracoef.config import load_config  # noqa: E402
from eracoef.design import FEATURES  # noqa: E402
from eracoef.gbdt_prior import DERIVED, RATIOS, SHOTQ  # noqa: E402
from eracoef.holdout import Context, Ratings, predict_season, score  # noqa: E402
from eracoef.priorridge import armse  # noqa: E402
from _cli import check_flags, flag  # noqa: E402

# what a correction may read, grouped by meaning (GLOSSARY.md has the plain names)
BOX = list(FEATURES) + list(DERIVED) + list(RATIOS) + list(SHOTQ)
ROLE = ["poss_pct", "gs_pct", "age", "exp_yrs", "exp_poss", "entry_age", "height", "weight", "draft_pick",
        "tenure", "n_teams"]
ON_OFF = sy.ONC + sy.OFFC + sy.NET + ["team_net"]
CONTEXT = sy.CLOSENESS + sy.PO_SHARE
BUILT = ["rating_off", "rating_def", "prior_off", "prior_def", "u_off", "u_def", "prior_raw_off", "prior_raw_def",
         "prior_scale_off", "prior_scale_def", "poss_off", "poss_def", "poss"]
INPUTS = BOX + ROLE + ON_OFF + CONTEXT + BUILT


# ---------------------------------------------------------------------------------------------- the data
class Fold:
    """One fold of one season: its players' inputs and ratings, and its held-out team-games."""

    def __init__(self, directory: Path, stem: str):
        self.stem = stem
        info = json.loads((directory / f"fold_{stem}.json").read_text())["summary"]
        self.season, self.repeat, self.fold = int(info["season"]), int(info["repeat"]), int(info["fold"])
        self.fill_o, self.fill_d = float(info["fill_o"]), float(info["fill_d"])
        self.tg_stored = float(info["tg"])
        players = add_team_inputs(pd.read_parquet(directory / f"players_{stem}.parquet"))
        t = np.load(directory / f"test_{stem}.npz", allow_pickle=False)
        Z = sp.csr_matrix((t["Z_data"], t["Z_indices"], t["Z_indptr"]), shape=tuple(t["Z_shape"]))
        ids = t["player_ids"]
        n = len(ids)
        self.y, self.w, self.F = t["y"].astype(float), t["w"].astype(float), np.asarray(t["F"], dtype=float)
        self.game_idx = t["game_idx"]
        # the rating every player column carries: his rating where he has one, the fill where he does not --
        # exactly what the fold's own score used (scripts/97 `test_games`)
        rated = players.drop_duplicates("player_id").set_index("player_id")
        rated = rated[rated.poss_off > 0]
        o = pd.Series(ids).map(rated.rating_off).to_numpy(float)
        d = pd.Series(ids).map(-rated.rating_def).to_numpy(float)
        o, d = np.where(np.isnan(o), self.fill_o, o), np.where(np.isnan(d), self.fill_d, d)
        self.base = np.asarray(Z[:, :n] @ o).ravel() + np.asarray(Z[:, n:] @ d).ravel()
        self.rows = {}
        self.Z = {}
        for side, block in (("O", Z[:, :n]), ("D", Z[:, n:])):
            rows = players[(players.side == side) & players.player_id.isin(rated.index)].reset_index(drop=True)
            column = pd.Series(np.arange(n), index=ids).reindex(rows.player_id).to_numpy()
            assert not np.isnan(column.astype(float)).any(), "a rated player with no column in the test games"
            self.rows[side] = rows
            self.Z[side] = sp.csr_matrix(block[:, column.astype(np.int64)])


def add_team_inputs(players: pd.DataFrame) -> pd.DataFrame:
    """The team's net points per 100 over the games he played, from his on-court and off-court numbers (the
    feasibility check's `team_net`)."""
    p = players
    on_o, off_o = p.onc_poss_o.to_numpy(float), p.offc_poss_o.to_numpy(float)
    on_d, off_d = p.onc_poss_d.to_numpy(float), p.offc_poss_d.to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        scored = (p.onc_o * on_o + p.offc_o * off_o) / (on_o + off_o)
        allowed = (p.onc_d * on_d + p.offc_d * off_d) / (on_d + off_d)
    return p.assign(team_net=(scored - allowed).fillna(0.0))


def stack(folds: list) -> dict:
    """Many folds as one problem: team-games stacked, each fold's players their own rows (block diagonal)."""
    out = {"y": np.concatenate([f.y for f in folds]), "w": np.concatenate([f.w for f in folds]),
           "base": np.concatenate([f.base for f in folds]),
           "fold_of_row": np.concatenate([np.full(len(f.y), i) for i, f in enumerate(folds)]),
           "F": [f.F for f in folds]}
    for side in ("O", "D"):
        out[f"Z{side}"] = sp.block_diag([f.Z[side] for f in folds], format="csr")
        out[f"X{side}"] = np.vstack([f.rows[side][INPUTS].to_numpy(float) for f in folds])
    for side in ("O", "D"):
        out[f"fold_{side}"] = np.concatenate([np.full(len(f.rows[side]), i) for i, f in enumerate(folds)])
        out[f"wc_{side}"] = np.concatenate([f.rows[side].poss_off.to_numpy(float) for f in folds])
    # each rated row's rating in the model's raw sign (defence = points allowed), for the one-multiplier yardstick
    out["RO"] = np.concatenate([f.rows["O"].rating_off.to_numpy(float) for f in folds])
    out["RD"] = np.concatenate([-f.rows["D"].rating_def.to_numpy(float) for f in folds])
    return out


def refit_levels(data: dict, resid: np.ndarray) -> np.ndarray:
    """Per fold, the level and home edge by weighted least squares on what is left: the fitted level per row."""
    level = np.zeros_like(resid)
    start = 0
    for F in data["F"]:
        stop = start + len(F)
        w = data["w"][start:stop]
        if F.shape[1]:
            sw = np.sqrt(w)[:, None]
            beta = np.linalg.lstsq(F * sw, resid[start:stop] * sw[:, 0], rcond=None)[0]
            level[start:stop] = F @ beta
        start = stop
    return level


def centre(values: np.ndarray, group: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Each group's weighted mean taken out (columns of a 2-d array each on their own)."""
    values = np.asarray(values, dtype=float)
    n = int(group.max()) + 1 if len(group) else 0
    total = np.bincount(group, weights=weight, minlength=n)
    if values.ndim == 1:
        mean = np.bincount(group, weights=weight * values, minlength=n) / np.where(total > 0, total, 1.0)
        return values - mean[group]
    out = np.empty_like(values)
    for j in range(values.shape[1]):
        mean = np.bincount(group, weights=weight * values[:, j], minlength=n) / np.where(total > 0, total, 1.0)
        out[:, j] = values[:, j] - mean[group]
    return out


def loss(data: dict, c_o: np.ndarray, c_d: np.ndarray) -> float:
    """The possession-weighted mean squared team-game error, the level refit per fold, the corrections centred
    within each fold."""
    c_o, c_d = centre(c_o, data["fold_O"], data["wc_O"]), centre(c_d, data["fold_D"], data["wc_D"])
    rest = data["y"] - data["base"] - data["ZO"] @ c_o + data["ZD"] @ c_d
    e = rest - refit_levels(data, rest)
    return float(np.average(e ** 2, weights=data["w"]))


# ------------------------------------------------------------------------------------------ the booster
class Calibrator:
    def __init__(self, depth: int, rate: float, min_leaf: int, ridge: float = 1e-3):
        self.depth, self.rate, self.min_leaf, self.ridge = int(depth), float(rate), int(min_leaf), float(ridge)
        self.trees = []          # per round: (tree_O, values_O, tree_D, values_D)

    def fit(self, data: dict, rounds: int, valid: dict | None = None) -> list:
        """Grow `rounds` rounds; returns the validation loss after each (empty without `valid`)."""
        c_o, c_d = np.zeros(data["ZO"].shape[1]), np.zeros(data["ZD"].shape[1])
        h = {"O": np.asarray(data["ZO"].power(2).T @ data["w"]).ravel(),
             "D": np.asarray(data["ZD"].power(2).T @ data["w"]).ravel()}
        vc_o = vc_d = None
        if valid is not None:
            vc_o, vc_d = np.zeros(valid["ZO"].shape[1]), np.zeros(valid["ZD"].shape[1])
        curve = []
        for _ in range(int(rounds)):
            rest = data["y"] - data["base"] - data["ZO"] @ c_o + data["ZD"] @ c_d
            e = rest - refit_levels(data, rest)
            we = data["w"] * e
            grad = {"O": np.asarray(data["ZO"].T @ we).ravel(), "D": -np.asarray(data["ZD"].T @ we).ravel()}
            trees, leaves, columns = {}, {}, []
            for side in ("O", "D"):
                hs = h[side]
                target = np.where(hs > 0, grad[side] / np.where(hs > 0, hs, 1.0), 0.0)
                tree = DecisionTreeRegressor(max_depth=self.depth, min_samples_leaf=self.min_leaf, random_state=0)
                tree.fit(data[f"X{side}"], target, sample_weight=hs)
                leaf = tree.apply(data[f"X{side}"])
                ids, leaf_code = np.unique(leaf, return_inverse=True)
                onehot = np.zeros((len(leaf), len(ids)))
                onehot[np.arange(len(leaf)), leaf_code] = 1.0
                onehot = centre(onehot, data[f"fold_{side}"], data[f"wc_{side}"])
                sign = 1.0 if side == "O" else -1.0
                columns.append(sign * np.asarray(data[f"Z{side}"] @ onehot))
                trees[side], leaves[side] = (tree, ids), onehot
            A = np.hstack(columns)
            gram = A.T @ (A * data["w"][:, None])
            rhs = A.T @ we
            values = np.linalg.solve(gram + self.ridge * np.eye(len(rhs)) * max(np.trace(gram) / len(rhs), 1e-12),
                                     rhs) * self.rate
            n_o = len(trees["O"][1])
            v_o, v_d = values[:n_o], values[n_o:]
            c_o += leaves["O"] @ v_o
            c_d += leaves["D"] @ v_d
            self.trees.append((trees["O"], v_o, trees["D"], v_d))
            if valid is not None:
                vc_o += self._one(trees["O"], v_o, valid["XO"])
                vc_d += self._one(trees["D"], v_d, valid["XD"])
                curve.append(loss(valid, vc_o, vc_d))
        return curve

    @staticmethod
    def _one(tree_ids, values, X) -> np.ndarray:
        tree, ids = tree_ids
        position = np.searchsorted(ids, tree.apply(X))
        return values[position]

    def predict(self, X_o: np.ndarray, X_d: np.ndarray, rounds: int | None = None) -> tuple:
        c_o, c_d = np.zeros(len(X_o)), np.zeros(len(X_d))
        for tree_o, v_o, tree_d, v_d in self.trees[:rounds]:
            c_o += self._one(tree_o, v_o, X_o)
            c_d += self._one(tree_d, v_d, X_d)
        return c_o, c_d


# ------------------------------------------------------------------------------------------------- tests
def corrected_table(players: pd.DataFrame, model: Calibrator, scale: tuple = (1.0, 1.0)) -> pd.DataFrame:
    """One row per player: the rating, the two corrections, and the corrected rating (positive-good): the rating
    times the side's multiplier, plus the side's correction."""
    out = players.drop_duplicates("player_id")[["player_id", "rating_off", "rating_def", "poss_off",
                                                 "poss_def"]].set_index("player_id")
    for side, column in (("O", "corr_off"), ("D", "corr_def")):
        rows = players[players.side == side]
        X = rows[INPUTS].to_numpy(float)
        c = model.predict(X, X)[0 if side == "O" else 1]
        out[column] = pd.Series(c, index=rows.player_id.to_numpy())
    out = out.fillna({"corr_off": 0.0, "corr_def": 0.0})
    # the rankings' own centring (62's `finish`): per side, the possession-weighted mean rating is zero; the ratings
    # already are, so this takes the corrections' common level out
    weight = out.poss_off.to_numpy(float)
    for column in ("corr_off", "corr_def"):
        out[column] -= np.average(out[column], weights=weight)
    out["cal_off"] = scale[0] * out.rating_off + out.corr_off
    out["cal_def"] = scale[1] * out.rating_def + out.corr_def
    return out.reset_index()


def score_fold(ctx, season: int, test_ids: set, table: pd.DataFrame, column_off: str, column_def: str,
               fill: tuple) -> dict:
    """The year-over-year test's scoring on a fold's held-out games (scripts/97 `test_games`)."""
    full = ctx.design([season], "pts")
    games = full.games
    test_idx = games.loc[games.game_id.astype(str).isin(test_ids), "game_idx"].to_numpy()
    wd = full.subset(np.isin(full.rows["game_idx"].to_numpy(), test_idx))
    rated = table[table.poss_off > 0]
    frame = pd.DataFrame({"player_id": rated.player_id, "o": rated[column_off], "d": -rated[column_def],
                          "poss": rated.poss_off})
    return score(predict_season(Ratings(frame, fill_o=fill[0], fill_d=fill[1]), wd, level="home"))


def _seasons(text: str) -> list:
    out = []
    for part in text.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def fit_rescale(data: dict) -> tuple:
    """The simplest calibrator, the yardstick the trees must beat: ONE multiplier per side for every rated player,
    by weighted least squares on the same held-out team-games, the level and home edge free per fold."""
    blocks = sp.block_diag([sp.csr_matrix(F) if F.shape[1] else sp.csr_matrix((len(F), 0)) for F in data["F"]],
                           format="csr")
    col_o = data["ZO"] @ data["RO"]
    col_d = data["ZD"] @ data["RD"]
    M = sp.hstack([blocks, sp.csr_matrix(np.column_stack([col_o, col_d]))], format="csr")
    gram = np.asarray((M.T @ M.multiply(data["w"][:, None])).todense())
    rhs = np.asarray(M.T @ (data["w"] * (data["y"] - data["base"]))).ravel()
    coef = np.linalg.lstsq(gram, rhs, rcond=None)[0]
    return 1.0 + float(coef[-2]), 1.0 + float(coef[-1])


def with_scale(data: dict, scale_o: float, scale_d: float) -> dict:
    """The same problem with every rated player's rating multiplied first: the trees then fit what is left."""
    out = dict(data)
    out["base"] = (data["base"] + (scale_o - 1.0) * (data["ZO"] @ data["RO"])
                   + (scale_d - 1.0) * (data["ZD"] @ data["RD"]))
    return out


def main() -> None:
    check_flags()
    tag, name = flag("tag", "within"), flag("out", "calib")
    depth, rate = int(flag("depth", 3)), float(flag("rate", 0.1))
    max_rounds, min_leaf = int(flag("rounds", 300)), int(flag("min_leaf", 300))
    seasons = _seasons(flag("seasons", "2017-2026"))
    directory = ROOT / "outputs" / "within" / tag
    cache = directory / f"calib_{name}"
    cache.mkdir(exist_ok=True)
    cfg = load_config(ROOT / "config.yaml")
    ctx = Context.load(cfg)
    ctx.cache_size = 2

    stems = sorted(p.stem[len("fold_"):] for p in directory.glob("fold_*.json"))
    folds = [Fold(directory, s) for s in stems]
    by_season = {s: [f for f in folds if f.season == s] for s in seasons}
    print(f"{len(folds)} folds over {len(seasons)} seasons; {len(INPUTS)} inputs per player and side; "
          f"depth {depth}, rate {rate}, up to {max_rounds} rounds, leaves of {min_leaf}+ players", flush=True)
    dealt = {s: pd.read_parquet(directory / f"folds_{s}.parquet") for s in seasons}

    t0 = time.time()
    for season in seasons:
        done = cache / f"{season}.pkl"
        if done.exists():
            continue
        # never this season or either neighbour: the seasons the year-over-year test scores this season's rating on
        train_seasons = [s for s in seasons if abs(s - season) > 1]
        train = stack([f for s in train_seasons for f in by_season[s]])
        # -- the yardstick: one multiplier per side
        scale_o, scale_d = fit_rescale(train)
        # -- the number of rounds: half the training seasons against the other half, both ways round
        halves = [train_seasons[0::2], train_seasons[1::2]]
        curves = []
        for a, b in (halves, halves[::-1]):
            fit = stack([f for s in a for f in by_season[s]])
            check = stack([f for s in b for f in by_season[s]])
            inner = fit_rescale(fit)
            fit, check = with_scale(fit, *inner), with_scale(check, *inner)
            before = loss(check, np.zeros(check["ZO"].shape[1]), np.zeros(check["ZD"].shape[1]))
            curve = Calibrator(depth, rate, min_leaf).fit(fit, max_rounds, valid=check)
            curves.append(np.asarray(curve) - before)
        gain = np.mean(curves, axis=0)
        rounds = int(np.argmin(gain)) + 1 if gain.min() < 0 else 0
        # -- the calibrator for this season, on every training season
        model = Calibrator(depth, rate, min_leaf)
        if rounds:
            model.fit(with_scale(train, scale_o, scale_d), rounds)
        # -- within-season test: this season's part-season ratings, corrected, on their held-out games
        rows = []
        for f in by_season[season]:
            players = add_team_inputs(pd.read_parquet(directory / f"players_{f.stem}.parquet"))
            table = corrected_table(players, model, (scale_o, scale_d))
            table["res_off"], table["res_def"] = scale_o * table.rating_off, scale_d * table.rating_def
            test_ids = set(dealt[season].index[dealt[season][f"r{f.repeat}"] == f.fold])
            fill = (f.fill_o, f.fill_d)
            before = score_fold(ctx, season, test_ids, table, "rating_off", "rating_def", fill)
            assert abs(before["tg"] - f.tg_stored) < 1e-8 * max(1.0, f.tg_stored), "the fold's score did not reproduce"
            rescaled = score_fold(ctx, season, test_ids, table, "res_off", "res_def", fill)
            after = score_fold(ctx, season, test_ids, table, "cal_off", "cal_def", fill)
            rows.append(dict(season=season, repeat=f.repeat, fold=f.fold, rounds=rounds, scale_o=scale_o,
                             scale_d=scale_d, tg_before=before["tg"], tg_rescale=rescaled["tg"], tg_after=after["tg"],
                             n=before["tg_n"], scale_off_before=before["scale_off"],
                             scale_def_before=before["scale_def"], scale_off_after=after["scale_off"],
                             scale_def_after=after["scale_def"]))
        # -- the whole season, corrected: what the year-over-year test reads
        whole = add_team_inputs(pd.read_parquet(directory / f"players_{season}_all.parquet"))
        table = corrected_table(whole, model, (scale_o, scale_d)).assign(season=season, rounds=rounds,
                                                                           scale_o=scale_o, scale_d=scale_d)
        pd.to_pickle(dict(rows=rows, table=table, curve=gain, model=model), done)
        w = pd.DataFrame(rows)
        print(f"  {season}: multipliers {scale_o:.3f} / {scale_d:.3f}; {rounds} rounds (inner gain {gain.min():+.3f}); "
              f"within-season error {armse(w.tg_before.mean()):.4f}, multiplied {armse(w.tg_rescale.mean()):.4f}, "
              f"calibrated {armse(w.tg_after.mean()):.4f}; whole-season corrections sd {table.corr_off.std():.3f} / "
              f"{table.corr_def.std():.3f}  ({time.time() - t0:.0f}s)", flush=True)

    loaded = [pd.read_pickle(cache / f"{s}.pkl") for s in seasons]
    within = pd.DataFrame([r for x in loaded for r in x["rows"]])
    within.to_parquet(ROOT / "outputs" / f"{name}_within.parquet", index=False)
    corr = pd.concat([x["table"] for x in loaded], ignore_index=True)
    corr.to_parquet(ROOT / "outputs" / f"{name}_corrections.parquet", index=False)

    # the whole-season ratings, corrected, in the 62 schema (the swap adjustment and 63 read it): the calibrator's,
    # and the one-multiplier yardstick's
    base = pd.read_parquet(ROOT / "outputs" / "season_ratings_within_base.parquet")
    base = base[base.season.isin(seasons)]
    for arm in ("calib", "rescale"):
        board = base.merge(corr[["player_id", "season", "corr_off", "corr_def", "scale_o", "scale_d"]],
                           on=["player_id", "season"], how="left")
        board[["corr_off", "corr_def"]] = board[["corr_off", "corr_def"]].fillna(0.0)
        scale = board[["season"]].merge(corr.groupby("season")[["scale_o", "scale_d"]].first().reset_index(),
                                        on="season", how="left")
        board["rating_off"] = board.rating_off * scale.scale_o.to_numpy()
        board["rating_def"] = board.rating_def * scale.scale_d.to_numpy()
        if arm == "calib":
            board["rating_off"] = board.rating_off + board.corr_off
            board["rating_def"] = board.rating_def + board.corr_def
        board["rating_total"] = board.rating_off + board.rating_def
        board["offense"], board["defense"] = board.rating_off, -board.rating_def
        board["u_off"], board["u_def"] = board.rating_off - board.prior_off, board.rating_def - board.prior_def
        board["u_total"] = board.u_off + board.u_def
        out = ROOT / "outputs" / f"season_ratings_{name}{'' if arm == 'calib' else '_rescale'}_raw.parquet"
        board.drop(columns=["corr_off", "corr_def", "scale_o", "scale_d"]).to_parquet(out, index=False)
        print(f"wrote {out.relative_to(ROOT)}")

    def report(column: str, label: str, against: str = "tg_before") -> None:
        d = within[column] - within[against]
        by = within.assign(d=d).groupby("season").d.mean()
        z = by.mean() / (by.std(ddof=1) / np.sqrt(len(by))) if len(by) > 1 else np.nan
        print(f"  {label:44s} error {armse(within[against].mean()):.4f} -> {armse(within[column].mean()):.4f}; "
              f"squared error {d.mean():+.3f} per team-game, z {z:+.2f} over {len(by)} seasons, better in "
              f"{int((by < 0).sum())} of {len(by)}")
    print(f"\nwithin-season test, {len(within)} folds, each season's held-out games scored as the year-over-year test "
          f"scores a season:")
    report("tg_rescale", "one multiplier per side vs the rating")
    report("tg_after", "the calibrator vs the rating")
    report("tg_after", "the calibrator vs one multiplier per side", against="tg_rescale")

    last = max(seasons)
    names = base.drop_duplicates("player_id").set_index("player_id").player_name
    top = corr[corr.season == last].copy()
    top["name"] = top.player_id.map(names)
    top["before"], top["after"] = top.rating_off + top.rating_def, top.cal_off + top.cal_def
    top["multiplied"] = top.scale_o * top.rating_off + top.scale_d * top.rating_def
    top["rank_before"] = top.before.rank(ascending=False, method="first").astype(int)
    top["rank_after"] = top.after.rank(ascending=False, method="first").astype(int)
    show = top.sort_values("after", ascending=False).head(20)
    print(f"\n{last} top 20 after the calibrator (whole season; points per 100, positive good; the swap adjustment "
          f"not yet applied):")
    print(show[["name", "rank_before", "rank_after", "before", "multiplied", "after", "corr_off", "corr_def"]].round(2)
          .to_string(index=False))


if __name__ == "__main__":
    main()
