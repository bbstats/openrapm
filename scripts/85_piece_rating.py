"""The owner's idea: the PIECES of a season's vanilla RAPM predict a player's RAPM in the seasons before and after
better than their sum does.  A proof of concept on scripts/84_piece_panel.py's panel.

    python scripts/85_piece_rating.py --stage=coef       # what matters: the coefficient tables
    python scripts/85_piece_rating.py --stage=shootout   # every model, out of fold, resumable; Boruta
    python scripts/85_piece_rating.py --stage=score      # scores on next/previous RAPM; the rankings tables
    python scripts/85_piece_rating.py --stage=yoy        # the like-for-like pairs from 63_yoy.py's parquet
      [--boot=200] [--trials=50] [--threads=4] [--models=a,b] [--seasons=2020,2021] [--oof=...] [--tag=pieces]
      [--boruta=outputs/csv/piece_boruta_table.csv]   (--boruta= skips Boruta, for a smoke run)

**The pieces** (the Decomposition page's, per side; scripts/84_piece_panel.py): by player -- on-court rtg,
teammates, opponents, context, ridge penalty; by possession -- on court signal, off court adjustment (GP), off
court adjustment (DNP), team SOS adjustment; beside them, the actual off-court rtg (GP).  RAPM weighs every piece
1.  A model that weighs them separately contains RAPM as a special case.

**A pair** is one player's season t (the inputs) and a neighbouring season, t-1 or t+1, that he also played (the
target: his vanilla RAPM there, per side, penalty 3,000).  Both directions pooled with no direction input, as in
the year-over-year test, so aging largely cancels.  Fit weight: possessions in t x possessions in the target
season / their sum (the trade set's weight), so a 50-possession season counts for little.

**Perfect correlation** (the owner's concern).  Each side's five by-player pieces add up to its RAPM exactly, and
so do its four by-possession pieces, so RAPM plus a whole split -- or both whole splits -- are perfectly
collinear.  Linear models therefore never get a redundant set: Table 1 fits one split alone with no RAPM; Table
2 and `pieces_linear` use the EIGHT-PIECE BASIS, the five by-player pieces plus the three off-court possession
groups (on court signal is RAPM minus those three, so the eight span what all nine do).  Every linear fit
asserts full column rank first.  GBDTs keep everything (a tree splits on one column at a time); Boruta runs with
and without the RAPM total because redundant columns share importance.

**Evidence share** = possessions / (possessions + 3,000): how much of a RAPM is his own data rather than the
ridge's pull toward zero.  The linear rating for everyone scales each piece by it, so a bench player's huge,
mutually cancelling pieces fade and his rating falls back toward his RAPM.

**Folds, identical for every model.**  The rating for season t, for the players in player fold f, comes from a
fit on the pairs that touch neither t-1 nor t+1 and hold no fold-f player (five folds balanced on the label,
rloocv.BalancedGroupKFold).  That is the incumbent's `--exclude_neighbours=1` plus the out-of-player rule, so the
year-over-year test can score these ratings as it scores the incumbent.

**Two truths.**  Next season's RAPM at 3,000 is shrunk hardest for players with few possessions, and possessions
this season predict that, so a model can win by predicting the target's shrinkage.  Every direct-test verdict is
also read against the neighbour's RAPM at penalty 100; a sign change between the two is undecided.
"""
import importlib.util
import os
import sys
import time
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
os.environ.setdefault("NUMBA_NUM_THREADS", os.environ.get("OPENRAPM_NUMBA_THREADS", "4"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _cli import check_flags, flag  # noqa: E402
from eracoef.config import load_config  # noqa: E402
from eracoef.rloocv import BalancedGroupKFold  # noqa: E402

OUT = ROOT / "outputs"
PANEL = OUT / "piece_panel.parquet"
INCUMBENT = OUT / "season_ratings_unshrinkdef.parquet"
LAM = 3000.0                      # the RAPM penalty per side, and the evidence share's constant
ROTATION = 1000.0                 # possessions on a side, in both seasons, for the coefficient tables
N_FOLDS = 5
SIDES = ("off", "def")
BY_PLAYER = ["on_rtg", "teammates", "opponents", "context", "ridge"]
BY_POSSESSION = ["on_signal", "off_adj_gp", "off_adj_dnp", "team_sos"]
BASIS = BY_PLAYER + ["off_adj_gp", "off_adj_dnp", "team_sos"]      # the eight-piece basis: no exact sum left
PAGE = dict(on_rtg="on-court rtg", teammates="teammates", opponents="opponents", context="context",
            ridge="ridge penalty", on_signal="on court signal", off_adj_gp="off court adjustment (GP)",
            off_adj_dnp="off court adjustment (DNP)", team_sos="team SOS adjustment", off_rtg="off court rtg (GP)",
            rapm="RAPM itself", logposs="log possessions", s="evidence share", poss="possessions",
            poss_gp="team possessions without him (GP)", poss_dnp="team possessions in games he missed",
            inc="OpenRAPM")
L1_RATIOS = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
TIERS = [(0.0, 1000.0, "under 1,000"), (1000.0, 3000.0, "1,000-3,000"), (3000.0, np.inf, "3,000+")]
# model -> (kind, role); the like-for-like pairs isolate the pieces: same target, same possessions
MODELS = {"rapm_scaled": ("ols", "control"), "inc_scaled": ("ols", "reference: OpenRAPM x slope"),
          "rapm_linear": ("enet", "linear control"), "pieces_linear": ("enet", "linear candidate"),
          "rapm_gbdt": ("gbdt", "GBDT control"), "pieces_gbdt": ("gbdt", "GBDT candidate"),
          "pieces_gbdt_boruta": ("gbdt", "GBDT on Boruta's kept inputs"),
          "rapm_linear_stay": ("enet", "linear control, told whether he stayed"),
          "pieces_linear_stay": ("enet", "linear candidate, every input x stayed"),
          "rapm_gbdt_stay": ("gbdt", "GBDT control, told whether he stayed"),
          "pieces_gbdt_stay": ("gbdt", "GBDT candidate, told whether he stayed")}
CONTROL = {"pieces_linear": "rapm_linear", "pieces_gbdt": "rapm_gbdt", "pieces_gbdt_boruta": "rapm_gbdt",
           "pieces_linear_stay": "rapm_linear_stay", "pieces_gbdt_stay": "rapm_gbdt_stay"}
# the owner's second idea (2026-09-28): the same models told whether the player stayed on his main team between
# the rated season and the target season.  A linear model gets the flag and the flag x every input (its own
# weights for players who stayed); a GBDT gets the flag as one more input.  The flag belongs to the PAIR, so these
# models rate every player-season twice -- as if he stays (`@stay`) and as if he moves (`@move`) -- and each
# test takes the version its own seasons call for.
STAY = {"rapm_linear_stay": "rapm_linear", "pieces_linear_stay": "pieces_linear",
        "rapm_gbdt_stay": "rapm_gbdt", "pieces_gbdt_stay": "pieces_gbdt"}


def say(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------------------------------------- the data
def load_panel() -> pd.DataFrame:
    """The panel plus every derived input: evidence share, log possessions, the scaled pieces, the incumbent."""
    p = pd.read_parquet(PANEL).sort_values(["season", "player_id"]).reset_index(drop=True)
    p["poss_gp"] = (p.poss_gp_off + p.poss_gp_def) / 2
    p["poss_dnp"] = (p.poss_dnp_off + p.poss_dnp_def) / 2
    for s in SIDES:
        p[f"rapm_{s}"] = p[f"rapm_{s}"].astype(float)
        p[f"s_{s}"] = p[f"poss_{s}"] / (p[f"poss_{s}"] + LAM)
        p[f"logposs_{s}"] = np.log1p(p[f"poss_{s}"])
        p[f"off_rtg0_{s}"] = p[f"off_rtg_{s}"].fillna(0.0)
        p[f"s_rapm_{s}"] = p[f"s_{s}"] * p[f"rapm_{s}"]
        for k in BASIS + ["off_rtg0"]:
            p[f"s_{k}_{s}"] = p[f"s_{s}"] * p[f"{k}_{s}"]
    inc = pd.read_parquet(INCUMBENT, columns=["player_id", "season", "rating_off", "rating_def"])
    p = p.merge(inc.rename(columns={"rating_off": "inc_off", "rating_def": "inc_def"}),
                on=["player_id", "season"], how="left")
    p[["inc_off", "inc_def"]] = p[["inc_off", "inc_def"]].fillna(0.0)
    return p


def make_pairs(p: pd.DataFrame) -> pd.DataFrame:
    """Every (season t, neighbouring season) pair a player played both of; `row` is season t's panel row."""
    key = p[["player_id", "season"]].assign(row=np.arange(len(p)))
    tcols = ["rapm_off", "rapm_def", "rapm100_off", "rapm100_def", "poss_off", "poss_def", "team_id"]
    nb = p[["player_id", "season"] + tcols].rename(columns={"season": "target_season",
                                                             **{c: f"t_{c}" for c in tcols}})
    parts = [key.assign(target_season=key.season + d).merge(nb, on=["player_id", "target_season"])
             .assign(direction=name) for d, name in ((1, "next"), (-1, "prev"))]
    q = pd.concat(parts, ignore_index=True)
    f = p.iloc[q.row.to_numpy()]
    q["mover"] = (f.team_id.to_numpy() >= 0) & (q.t_team_id >= 0) & (f.team_id.to_numpy() != q.t_team_id)
    for s in SIDES:
        a, b = f[f"poss_{s}"].to_numpy(), q[f"t_poss_{s}"].to_numpy()
        q[f"poss_t_{s}"] = a
        q[f"w_{s}"] = a * b / (a + b)
        q[f"y_{s}"], q[f"y100_{s}"] = q[f"t_rapm_{s}"], q[f"t_rapm100_{s}"]
    q["w_net"] = (q.w_off + q.w_def) / 2
    q["y_net"], q["y100_net"] = q.y_off + q.y_def, q.y100_off + q.y100_def
    return q


def player_folds(p: pd.DataFrame, q: pd.DataFrame) -> tuple:
    """A fold per player, balanced on the label; players with no pair (never in any training set) by id."""
    fold = BalancedGroupKFold(N_FOLDS).fold_ids(q.y_net.to_numpy(), q.player_id.to_numpy(), q.w_net.to_numpy())
    of = pd.Series(fold, index=q.player_id.to_numpy()).groupby(level=0).first()
    panel_fold = p.player_id.map(of).fillna(p.player_id % N_FOLDS).astype(int).to_numpy()
    return panel_fold, q.player_id.map(of).astype(int).to_numpy()


def splits(p, q, panel_fold, pair_fold, seasons):
    """(t, f, training pair indices, panel rows to rate): pairs touching neither t-1 nor t+1, no fold-f player."""
    ps, pt = q.season.to_numpy(), q.target_season.to_numpy()
    for t in seasons:
        touch = np.isin(ps, [t - 1, t + 1]) | np.isin(pt, [t - 1, t + 1])
        for f in range(N_FOLDS):
            train = np.flatnonzero(~touch & (pair_fold != f))
            rate = np.flatnonzero((p.season.to_numpy() == t) & (panel_fold == f))
            if np.isin(ps[train], [t - 1, t + 1]).any() or np.isin(pt[train], [t - 1, t + 1]).any():
                raise SystemExit(f"a training pair touches a neighbour of {t}")
            yield t, f, train, rate


# ---------------------------------------------------------------------------------------------- the inputs
def inputs(model: str, side: str, kept: dict | None = None) -> list:
    """Column names of `model`'s inputs for `side`'s target (panel columns)."""
    both = ["rapm_off", "rapm_def"]
    poss = ["poss_off", "poss_def", "poss_gp", "poss_dnp"]
    if model in STAY:
        return inputs(STAY[model], side, kept)
    if model == "rapm_scaled":
        return [f"rapm_{side}"]
    if model == "inc_scaled":
        return [f"inc_{side}"]
    if model == "rapm_linear":
        return both + [f"logposs_{side}", f"s_{side}", f"s_rapm_{side}"]
    if model == "pieces_linear":
        return both + [f"logposs_{side}", f"s_{side}"] + [f"s_{k}_{side}" for k in BASIS + ["off_rtg0"]]
    if model == "rapm_gbdt":
        return both + poss
    if model == "pieces_gbdt":
        return both + poss + [f"{k}_{s}" for s in SIDES for k in BY_PLAYER + BY_POSSESSION + ["off_rtg"]]
    if model == "pieces_gbdt_boruta":
        return list(kept[side])
    raise KeyError(model)


def label(col: str) -> str:
    """A panel column in the page's words: 'teammates_off' -> 'teammates (off)'."""
    stem, side = col.rsplit("_", 1) if col.endswith(("_off", "_def")) else (col, "")
    scaled = stem.startswith("s_") and stem != "s"
    stem = stem[2:] if scaled else stem
    stem = {"off_rtg0": "off_rtg"}.get(stem, stem)
    words = PAGE.get(stem, stem)
    return f"{'evidence share x ' if scaled else ''}{words}{f' ({side})' if side else ''}"


# ---------------------------------------------------------------------------------------------- the fits
def wls(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted least squares with an intercept; returns [intercept, coefficients...]."""
    A = np.column_stack([np.ones(len(y)), X])
    if np.linalg.matrix_rank(A * np.sqrt(w)[:, None]) < A.shape[1]:
        raise SystemExit("a linear fit's inputs are not of full rank -- an exact sum slipped in")
    sw = np.sqrt(w)
    return np.linalg.lstsq(A * sw[:, None], y * sw, rcond=None)[0]


class OLS:
    def fit(self, X, y, w, groups=None):
        self.b = wls(X, y, w)
        return self

    def predict(self, X):
        return self.b[0] + X @ self.b[1:]


class ENet:
    """Weighted-standardised elastic net, alpha and l1_ratio chosen by inner folds over target seasons."""

    def __init__(self, alpha=None, l1_ratio=None):
        self.alpha, self.l1_ratio = alpha, l1_ratio

    def fit(self, X, y, w, groups=None, mu=None, sd=None):
        from sklearn.linear_model import ElasticNet, ElasticNetCV
        from sklearn.model_selection import GroupKFold
        self.mu = np.average(X, axis=0, weights=w) if mu is None else mu
        self.sd = np.sqrt(np.average((X - self.mu) ** 2, axis=0, weights=w)) if sd is None else sd
        Xs = (X - self.mu) / self.sd
        if np.linalg.matrix_rank(Xs * np.sqrt(w)[:, None]) < Xs.shape[1]:
            raise SystemExit("an elastic net's inputs are not of full rank -- an exact sum slipped in")
        if self.alpha is None:
            cv = list(GroupKFold(n_splits=5).split(Xs, y, groups=groups))
            m = ElasticNetCV(l1_ratio=L1_RATIOS, alphas=50, cv=cv, max_iter=50000)
            m.fit(Xs, y, sample_weight=w)
            self.alpha, self.l1_ratio = float(m.alpha_), float(m.l1_ratio_)
        else:
            m = ElasticNet(alpha=self.alpha, l1_ratio=self.l1_ratio, max_iter=50000)
            m.fit(Xs, y, sample_weight=w)
        self.coef_std, self.intercept = m.coef_.copy(), float(m.intercept_)
        self.coef = self.coef_std / self.sd                              # per point of each input
        return self

    def predict(self, X):
        return self.intercept + ((X - self.mu) / self.sd) @ self.coef_std


class GBDT:
    def __init__(self, params: dict, threads: int):
        self.params, self.threads = params, threads

    def fit(self, X, y, w, groups=None):
        from chimeraboost import ChimeraBoostRegressor
        self.m = ChimeraBoostRegressor(random_state=0, thread_count=self.threads, **self.params)
        self.m.fit(X, y, sample_weight=w, groups=groups)
        return self

    def predict(self, X):
        return self.m.predict(X)


def with_flag(X: np.ndarray, flag: np.ndarray, kind: str) -> np.ndarray:
    """The inputs plus the stayed-on-his-team flag; a linear model also gets the flag x every input, so it has
    its own weights for players who stayed.  A GBDT finds its own interactions."""
    if kind == "gbdt":
        return np.column_stack([X, flag])
    return np.column_stack([X, flag, flag[:, None] * X])


def make_model(kind: str, cfg: dict, threads: int):
    return {"ols": OLS, "enet": ENet}[kind]() if kind != "gbdt" else GBDT(dict(cfg["gbdt"]["params_def"]), threads)


def top_correlations(X: np.ndarray, names: list, w: np.ndarray, k: int = 5) -> list:
    """The k most correlated input pairs (weighted), to make a near-duplicate visible."""
    Xc = X - np.average(X, axis=0, weights=w)
    C = (Xc * w[:, None]).T @ Xc
    d = np.sqrt(np.diag(C))
    R = C / np.outer(d, d)
    iu = np.triu_indices(len(names), 1)
    order = np.argsort(-np.abs(R[iu]))[:k]
    return [(names[iu[0][j]], names[iu[1][j]], float(R[iu][j])) for j in order]


# ---------------------------------------------------------------------------------------------- scoring
def per_season_mse(err2: np.ndarray, w: np.ndarray, season: np.ndarray) -> pd.Series:
    d = pd.DataFrame({"s": season, "e": err2 * w, "w": w})
    g = d.groupby("s")[["e", "w"]].sum()
    return g.e / g.w


def per_season_order(pred: np.ndarray, y: np.ndarray, w: np.ndarray, season: np.ndarray) -> pd.Series:
    """Per target season: the weighted error after refitting one intercept and one scalar on that season -- the
    spread taken out, the order left (the year-over-year test's 'each side rescaled' row, per season)."""
    out = {}
    for s in np.unique(season):
        k = season == s
        A = np.column_stack([np.ones(k.sum()), pred[k]]) * np.sqrt(w[k])[:, None]
        b = np.linalg.lstsq(A, y[k] * np.sqrt(w[k]), rcond=None)[0]
        r = y[k] - b[0] - b[1] * pred[k]
        out[s] = float(np.sum(w[k] * r ** 2) / np.sum(w[k]))
    return pd.Series(out)


def paired(a: pd.Series, b: pd.Series) -> dict:
    """Candidate minus control per target season: mean, its standard error, the paired test statistic, wins."""
    d = (a - b).dropna()
    se = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else np.nan
    return dict(mean_diff=float(d.mean()), se=se, z=float(d.mean() / se) if se and se > 0 else np.nan,
                wins=int((d < 0).sum()), n=int(len(d)))


def rank_corr(q: pd.DataFrame, pred: np.ndarray) -> float:
    """Mean over (target season, direction) of the Spearman correlation of prediction and target, net, among
    players with 1,000+ possessions on both sides in both seasons -- every player counts once."""
    from scipy.stats import spearmanr
    keep = ((q.poss_t_off >= ROTATION) & (q.t_poss_off >= ROTATION) & (q.poss_t_def >= ROTATION)
            & (q.t_poss_def >= ROTATION)).to_numpy() & ~np.isnan(pred)
    d = pd.DataFrame({"s": q.target_season.to_numpy()[keep], "d": q.direction.to_numpy()[keep],
                      "p": pred[keep], "y": q.y_net.to_numpy()[keep]})
    rs = [spearmanr(g.p, g.y)[0] for _, g in d.groupby(["s", "d"]) if len(g) >= 20]
    return float(np.mean(rs))


# ---------------------------------------------------------------------------------------------- stage: coef
def bootstrap_counts(players: np.ndarray, rng) -> np.ndarray:
    """Per row, how many times its player is drawn in one cluster bootstrap over players."""
    keys, inv = np.unique(players, return_inverse=True)
    draw = np.bincount(rng.integers(0, keys.size, keys.size), minlength=keys.size)
    return draw[inv].astype(float)


def stage_coef(p, q, panel_fold, pair_fold, cfg, boot: int) -> None:
    rng = np.random.default_rng(0)
    t1, t2, oos = [], [], []
    for side in SIDES:
        rot = ((q[f"poss_t_{side}"] >= ROTATION) & (q[f"t_poss_{side}"] >= ROTATION)).to_numpy()
        r = q[rot]
        rows, y, w, players = r.row.to_numpy(), r[f"y_{side}"].to_numpy(), r[f"w_{side}"].to_numpy(), \
            r.player_id.to_numpy()
        say(f"\n=== {side}: {rot.sum():,} pairs with {ROTATION:,.0f}+ possessions on this side in both seasons, "
            f"{np.unique(players).size:,} players")
        # Table 1: RAPM's own slope, then each split alone -- no RAPM total, so nothing adds up exactly
        designs = {"RAPM itself": [f"rapm_{side}"], "by player": [f"{k}_{side}" for k in BY_PLAYER],
                   "by possession": [f"{k}_{side}" for k in BY_POSSESSION]}
        for split, cols in designs.items():
            X = p.loc[rows, cols].to_numpy(float)
            b = wls(X, y, w)
            draws = []
            for _ in range(boot):
                c = bootstrap_counts(players, rng)
                keep = c > 0
                draws.append(wls(X[keep], y[keep], (w * c)[keep]))
            lo, hi = np.percentile(np.array(draws), [2.5, 97.5], axis=0)
            for j, col in enumerate(["intercept"] + cols):
                t1.append(dict(side=side, split=split, input=label(col) if col != "intercept" else "intercept",
                               weight=b[j], lo=lo[j], hi=hi[j]))
            if split != "RAPM itself":
                say(f"  {split}: most correlated inputs {[(label(a), label(c), round(v, 2)) for a, c, v in top_correlations(X, cols, w, 3)]}")
        # Table 2: the elastic net on the eight-piece basis + off-court rtg + log possessions
        cols = [f"{k}_{side}" for k in BASIS] + [f"off_rtg0_{side}", f"logposs_{side}"]
        X = p.loc[rows, cols].to_numpy(float)
        say(f"  eight-piece basis: most correlated inputs "
            f"{[(label(a), label(c), round(v, 2)) for a, c, v in top_correlations(X, cols, w)]}")
        en = ENet().fit(X, y, w, groups=r.target_season.to_numpy())
        say(f"  elastic net: alpha {en.alpha:.4g}, l1_ratio {en.l1_ratio}")
        coefs = []
        for _ in range(boot):
            c = bootstrap_counts(players, rng)
            keep = c > 0
            coefs.append(ENet(en.alpha, en.l1_ratio).fit(X[keep], y[keep], (w * c)[keep], mu=en.mu, sd=en.sd).coef)
        coefs = np.array(coefs)
        lo, hi = np.percentile(coefs, [2.5, 97.5], axis=0)
        for j, col in enumerate(cols):
            t2.append(dict(side=side, input=label(col), weight=en.coef[j], lo=lo[j], hi=hi[j],
                           kept_share=float((np.abs(coefs[:, j]) > 1e-12).mean()), sd_of_input=en.sd[j]))
        # out of sample, the same players: each of those models against RAPM x slope, the shootout's folds
        rot_models = {"RAPM x slope": ("ols", [f"rapm_{side}"]),
                      "by player (OLS)": ("ols", [f"{k}_{side}" for k in BY_PLAYER]),
                      "by possession (OLS)": ("ols", [f"{k}_{side}" for k in BY_POSSESSION]),
                      "elastic net, basis + off-court rtg + log possessions": ("enet", cols)}
        rot_pair_fold = pair_fold[rot]
        preds = {m: np.full(len(p), np.nan) for m in rot_models}
        seasons = sorted(p.season.unique())
        for t, f, train, rate in splits(p, r.reset_index(drop=True), panel_fold, rot_pair_fold, seasons):
            rate = rate[p[f"poss_{side}"].to_numpy()[rate] >= ROTATION]
            if not len(rate) or not len(train):
                continue
            for m, (kind, mc) in rot_models.items():
                Xt = p.loc[rows[train], mc].to_numpy(float)
                model = (OLS() if kind == "ols" else ENet()).fit(Xt, y[train], w[train],
                                                                 groups=r.target_season.to_numpy()[train])
                preds[m][rate] = model.predict(p.loc[rate, mc].to_numpy(float))
        base = None
        for m in rot_models:
            pr = preds[m][rows]
            ok = ~np.isnan(pr)
            if not ok.all():
                say(f"  note: {m} left {int((~ok).sum())} of {len(ok)} pairs unrated; scored without them")
            for truth in ("y", "y100"):
                yy = r[f"{truth}_{side}"].to_numpy()
                mse = per_season_mse((pr - yy)[ok] ** 2, w[ok], r.target_season.to_numpy()[ok])
                if m == "RAPM x slope":
                    base = {**(base or {}), truth: mse}
                rec = dict(side=side, model=m, truth="RAPM 3,000" if truth == "y" else "RAPM 100",
                           mse=float(np.sum((w * (pr - yy) ** 2)[ok]) / np.sum(w[ok])))
                rec.update(paired(mse, base[truth]) if m != "RAPM x slope" else {})
                oos.append(rec)
    t1, t2, oos = pd.DataFrame(t1), pd.DataFrame(t2), pd.DataFrame(oos)
    (OUT / "csv").mkdir(exist_ok=True)
    t1.to_csv(OUT / "csv" / "piece_coef_table1.csv", index=False)
    t2.to_csv(OUT / "csv" / "piece_coef_table2.csv", index=False)
    oos.to_csv(OUT / "csv" / "piece_coef_oos.csv", index=False)
    with pd.option_context("display.max_rows", None, "display.width", 200):
        say("\n=== Table 1: points of neighbouring-season RAPM per point of each piece (OLS, 95% bootstrap "
            "interval over players).  RAPM itself gives every piece the weight on its 'RAPM itself' row.")
        say(t1.round(3).to_string(index=False))
        say("\n=== Table 2: elastic net on the eight-piece basis (per point of each input; kept_share = share of "
            "bootstrap draws where the weight is not zero)")
        say(t2.round(3).to_string(index=False))
        say("\n=== Out of sample, same players (weighted error, points per 100 squared; mean_diff < 0 is better "
            "than RAPM x slope; z = mean_diff / its standard error over target seasons)")
        say(oos.round(4).to_string(index=False))


# ---------------------------------------------------------------------------------------------- stage: shootout
def boruta_table(p, q, cfg, trials: int, threads: int, path: Path) -> pd.DataFrame:
    """Boruta per side, with and without the RAPM total; the full table, every input."""
    from eracoef.gbdt_prior import run_boruta
    if path.exists():
        say(f"  Boruta: reusing {path.relative_to(ROOT)}")
        return pd.read_csv(path)
    out = []
    for side in SIDES:
        for run in ("with RAPM", "without RAPM"):
            feats = [c for c in inputs("pieces_gbdt", side) if run == "with RAPM" or not c.startswith("rapm_")]
            rows = p.iloc[q.row.to_numpy()][feats].reset_index(drop=True)
            rows["target"], rows["row_weight"] = q[f"y_{side}"].to_numpy(), q[f"w_{side}"].to_numpy()
            t0 = time.time()
            res = run_boruta(rows, feats, n_trials=trials, seed=0, thread_count=threads, verbose=False,
                             **dict(cfg["gbdt"]["params_def"]))
            hist = res["history"]
            tag = run.replace(" ", "_").lower()
            hist.to_csv(path.with_name(f"{path.stem}_{side}_{tag}_history.csv"), index=False)
            imp = hist.iloc[1:].mean()          # row 0 of the history is all zeros
            verdict = {**{f: "accepted" for f in res["accepted"]}, **{f: "tentative" for f in res["tentative"]},
                       **{f: "rejected" for f in res["rejected"]}}
            for f in feats:
                out.append(dict(side=side, run=run, input=label(f), column=f, verdict=verdict.get(f, "?"),
                                importance=float(imp.get(f, np.nan))))
            out.append(dict(side=side, run=run, input="(the best shadow input: the bar to beat)",
                            column="Max_Shadow", verdict="", importance=float(imp.get("Max_Shadow", np.nan))))
            say(f"  Boruta {side}, {run}: {len(res['accepted'])} accepted, {len(res['tentative'])} tentative, "
                f"{len(res['rejected'])} rejected ({time.time() - t0:.0f}s)")
    tab = pd.DataFrame(out)
    tab.to_csv(path, index=False)
    return tab


def stage_shootout(p, q, panel_fold, pair_fold, cfg, oof_path: Path, models: list, seasons: list,
                   trials: int, threads: int, boruta_path: Path | None) -> None:
    state = dict(done=pd.read_parquet(oof_path) if oof_path.exists() else None)
    have = set() if state["done"] is None else set(zip(state["done"].model, state["done"].side,
                                                       state["done"].season))
    say(f"  {len(have)} (model, side, season) already on disk in {oof_path}")
    y_all = {s: q[f"y_{s}"].to_numpy() for s in SIDES}
    w_all = {s: q[f"w_{s}"].to_numpy() for s in SIDES}
    rows_all, groups_all = q.row.to_numpy(), q.player_id.to_numpy()
    stay_all = (~q.mover.to_numpy()).astype(float)

    def run(model: str, kept: dict | None = None) -> None:
        kind, stay = MODELS[model][0], model in STAY
        key = f"{model}@move" if stay else model
        for side in SIDES:
            cols = inputs(model, side, kept)
            Xp = p[cols].to_numpy(float)
            if kind != "gbdt" and len(cols) > 1:
                say(f"  {model} {side}: most correlated inputs "
                    f"{[(label(a), label(c), round(v, 2)) for a, c, v in top_correlations(Xp[rows_all], cols, w_all[side], 3)]}")
            slopes = []
            for t in seasons:
                if (key, side, t) in have:
                    continue
                t0, got = time.time(), []
                for _, f, train, rate in splits(p, q, panel_fold, pair_fold, [t]):
                    groups = q.target_season.to_numpy()[train] if kind == "enet" else groups_all[train]
                    Xt = Xp[rows_all[train]]
                    if stay:
                        Xt = with_flag(Xt, stay_all[train], kind)
                    m = make_model(kind, cfg, threads).fit(Xt, y_all[side][train], w_all[side][train], groups=groups)
                    if model == "rapm_scaled":
                        slopes.append(m.b[1])
                    ids = p.player_id.to_numpy()[rate]
                    if stay:          # rate every player twice: as if he stays, and as if he moves
                        for scenario, v in (("move", 0.0), ("stay", 1.0)):
                            got.append(pd.DataFrame({"model": f"{model}@{scenario}", "side": side, "season": t,
                                                     "player_id": ids, "pred": m.predict(
                                                         with_flag(Xp[rate], np.full(len(rate), v), kind))}))
                    else:
                        got.append(pd.DataFrame({"model": model, "side": side, "season": t, "player_id": ids,
                                                 "pred": m.predict(Xp[rate])}))
                state["done"] = pd.concat(([] if state["done"] is None else [state["done"]]) + got,
                                          ignore_index=True)
                state["done"].to_parquet(oof_path, index=False)
                have.add((key, side, t))
                say(f"    {model} {side} {t}: {sum(len(g) for g in got)} players ({time.time() - t0:.1f}s)")
            if slopes:
                b = wls(p[[f"rapm_{side}"]].to_numpy(float)[rows_all], y_all[side], w_all[side])[1]
                say(f"  check: rapm_scaled {side} out-of-fold slopes median {np.median(slopes):.4f} "
                    f"(range {min(slopes):.4f}-{max(slopes):.4f}); plain OLS on every pair {b:.4f}")

    for model in [m for m in MODELS if m in models and m != "pieces_gbdt_boruta"]:
        run(model)
    if boruta_path is not None and ("pieces_gbdt" in models or "pieces_gbdt_boruta" in models):
        tab = boruta_table(p, q, cfg, trials, threads, boruta_path)
        if "pieces_gbdt_boruta" in models:
            w = tab[(tab.run == "with RAPM") & (tab.column != "Max_Shadow")]
            dropped = {s: int((w[w.side == s].verdict == "rejected").sum()) for s in SIDES}
            if max(dropped.values()) < 3:
                say(f"  Boruta drops fewer than 3 inputs on either side ({dropped}): pieces_gbdt_boruta skipped")
            else:
                run("pieces_gbdt_boruta", {s: w[(w.side == s) & (w.verdict != "rejected")].column.tolist()
                                           for s in SIDES})


# ---------------------------------------------------------------------------------------------- stage: score
def centred(t: pd.DataFrame, col: str, poss: str) -> pd.Series:
    """Possession-weighted zero per season over players with 100+ possessions (the published zero point)."""
    w = t[poss].where(t[poss] >= 100, 0.0)
    m = (t[col] * w).groupby(t.season).sum() / w.groupby(t.season).sum()
    return t[col] - t.season.map(m).fillna(0.0)


def rankings_table(p: pd.DataFrame, off: np.ndarray, dfn: np.ndarray) -> pd.DataFrame:
    t = p[["player_id", "season", "player_name", "poss_off", "poss_def"]].copy()
    t["rating_off"], t["rating_def"] = off, dfn
    t["rating_off"] = centred(t, "rating_off", "poss_off")
    t["rating_def"] = centred(t, "rating_def", "poss_def")
    t["rating_total"] = t.rating_off + t.rating_def
    t["offense"], t["defense"] = t.rating_off, -t.rating_def            # the pipeline's raw sign, for old readers
    return t


def stage_score(p, q, oof_path: Path) -> None:
    oof = pd.read_parquet(oof_path)
    preds = {"vanilla": {s: p[f"rapm_{s}"].to_numpy() for s in SIDES}}
    key = pd.MultiIndex.from_arrays([p.player_id, p.season])
    for (model, side), g in oof.groupby(["model", "side"]):
        v = g.set_index(["player_id", "season"]).pred.reindex(key).to_numpy()
        preds.setdefault(model, {})[side] = v

    def has(m):
        return m in preds and all(s in preds[m] for s in SIDES)

    models = [m for m in ["vanilla"] + list(MODELS)
              if ((has(f"{m}@move") and has(f"{m}@stay")) if m in STAY else has(m))]
    missing = {m: int(np.isnan(preds[f"{m}@move" if m in STAY else m]["off"]).sum()) for m in models}
    say(f"  models scored: {models}; player-seasons without a rating: {missing}")
    rows, ts = q.row.to_numpy(), q.target_season.to_numpy()
    mover = q.mover.to_numpy()

    def pair_pred(m, side):
        """Each pair's prediction; a model told whether he stayed takes the version this pair's seasons call for."""
        if m in STAY:
            return np.where(mover, preds[f"{m}@move"][side][rows], preds[f"{m}@stay"][side][rows])
        return preds[m][side][rows]

    groups = (("all", None), ("movers", mover), ("stayers", ~mover))
    table, per_season = [], {}
    for m in models:
        for tgt in ("off", "def", "net"):
            pr = pair_pred(m, "off") + pair_pred(m, "def") if tgt == "net" else pair_pred(m, tgt)
            w, ok = q[f"w_{tgt}"].to_numpy(), ~np.isnan(pr)
            for truth in ("y", "y100"):
                yy = q[f"{truth}_{tgt}"].to_numpy()
                for grp, gm in groups:
                    k = ok if gm is None else ok & gm
                    per_season[(m, tgt, truth, "raw", grp)] = per_season_mse((pr - yy)[k] ** 2, w[k], ts[k])
                    per_season[(m, tgt, truth, "order", grp)] = per_season_order(pr[k], yy[k], w[k], ts[k])
                for metric in ("raw", "order"):
                    s = per_season[(m, tgt, truth, metric, "all")]
                    table.append(dict(model=m, target=tgt, truth="RAPM 3,000" if truth == "y" else "RAPM 100",
                                      metric=metric, mse=float(s.mean()),
                                      rank_corr=rank_corr(q, pr) if (tgt, truth, metric) == ("net", "y", "raw")
                                      else np.nan,
                                      truth_key=truth))
    table = pd.DataFrame(table)
    # paired: each model against RAPM x slope, and each candidate against its like-for-like control
    out = []
    for r in table.itertuples(index=False):
        rec = {k: v for k, v in r._asdict().items() if k != "truth_key"}
        for ref_name, ref in (("vs_rapm_scaled", "rapm_scaled"), ("vs_control", CONTROL.get(r.model))):
            if ref is None or ref == r.model or ref not in models:
                continue
            for grp, _ in groups:
                pr_ = paired(per_season[(r.model, r.target, r.truth_key, r.metric, grp)],
                             per_season[(ref, r.target, r.truth_key, r.metric, grp)])
                suffix = "" if grp == "all" else f"_{grp}"
                rec[f"{ref_name}_diff{suffix}"] = pr_["mean_diff"]
                rec[f"{ref_name}_z{suffix}"] = pr_["z"]
                if grp == "all":
                    rec[f"{ref_name}_wins"] = f"{pr_['wins']}/{pr_['n']}"
        out.append(rec)
    res = pd.DataFrame(out)
    # by possession tier (possessions in the rated season), net, the 3,000 truth
    tiers = []
    for m in models:
        pr = pair_pred(m, "off") + pair_pred(m, "def")
        for lo, hi, name in TIERS:
            g = (q.poss_t_off.to_numpy() >= lo) & (q.poss_t_off.to_numpy() < hi) & ~np.isnan(pr)
            w = q.w_net.to_numpy()[g]
            tiers.append(dict(model=m, tier=name, pairs=int(g.sum()),
                              mse=float(np.sum(w * (pr[g] - q.y_net.to_numpy()[g]) ** 2) / w.sum())))
    tiers = pd.DataFrame(tiers).pivot(index="model", columns="tier", values="mse")
    res.to_csv(OUT / "csv" / "piece_shootout.csv", index=False)
    tiers.to_csv(OUT / "csv" / "piece_shootout_tiers.csv")
    with pd.option_context("display.max_rows", None, "display.width", 250, "display.max_columns", 40):
        say("\n=== Predicting the neighbouring season's vanilla RAPM.  Error = weighted squared error in points per "
            "100, averaged over the 30 target seasons.  'raw' = the rating as it is; 'order' = after one intercept "
            "and one scalar refit on each target season (spread removed, order only).  diff < 0 = better; z = diff "
            "/ its standard error over target seasons; wins of 30.  Control = the like-for-like model without "
            "the pieces.")
        cols = ["model", "truth", "metric", "mse", "vs_rapm_scaled_diff", "vs_rapm_scaled_z", "vs_rapm_scaled_wins",
                "vs_control_diff", "vs_control_z", "vs_control_wins", "vs_control_z_movers",
                "vs_control_z_stayers", "rank_corr"]
        for tgt in ("net", "off", "def"):
            say(f"\n--- {tgt}")
            say(res[res.target == tgt][[c for c in cols if c in res.columns]].round(4).to_string(index=False))
        say("\n=== Net error by possessions in the rated season (3,000 truth, raw)")
        say(tiers.round(3).to_string())
    # the rankings tables for the year-over-year test and the trade loss.  A model told whether he stayed writes
    # four: as if everyone stays (_stay) and as if everyone moves (_move), for the rankings and the trade loss;
    # and one per direction of the year-over-year test, each player rated with the flag the scored season calls
    # for -- _fwd (same main team next season; the test's `:prev` rows) and _bwd (same team last season; `:next`)
    team = pd.Series(p.team_id.to_numpy(), index=key)

    def same_team(d):
        other = team.reindex(pd.MultiIndex.from_arrays([p.player_id, p.season + d])).to_numpy(dtype=float)
        return other == p.team_id.to_numpy()

    flags = {"fwd": same_team(1), "bwd": same_team(-1)}
    tabs, written = {}, []
    for m in models:
        if m in STAY:
            versions = {f"{m}_move": (preds[f"{m}@move"]["off"], preds[f"{m}@move"]["def"]),
                        f"{m}_stay": (preds[f"{m}@stay"]["off"], preds[f"{m}@stay"]["def"])}
            for d, f in flags.items():
                versions[f"{m}_{d}"] = tuple(np.where(f, preds[f"{m}@stay"][s], preds[f"{m}@move"][s]) for s in SIDES)
        else:
            versions = {m: (preds[m]["off"], preds[m]["def"])}
        for name, (off, dfn) in versions.items():
            if np.isnan(off).any() or np.isnan(dfn).any():
                say(f"  {name}: {int(np.isnan(off).sum())} player-seasons unrated -- no rankings table written")
                continue
            t = rankings_table(p, off, dfn)
            stem = "vanilla" if name == "vanilla" else f"piece_{name}"
            t.to_parquet(OUT / f"season_ratings_{stem}.parquet", index=False)
            written.append(f"season_ratings_{stem}")
            if not name.endswith(("_fwd", "_bwd")):
                tabs[name] = t
    say(f"  wrote rankings tables: {', '.join(written)}")
    say(f"  players on the same main team the next season: {flags['fwd'].mean():.1%} of player-seasons; "
        f"the previous season: {flags['bwd'].mean():.1%}")
    # movement against RAPM x slope, in points per 100: the amplitude part and the per-player part
    if "rapm_scaled" in tabs:
        ref = tabs["rapm_scaled"]
        mv = []
        for m, t in tabs.items():
            if m in ("rapm_scaled", "vanilla", "inc_scaled"):
                continue
            for scope, g in (("all seasons", np.ones(len(t), bool)), ("2026", (t.season == 2026).to_numpy())):
                w = t.poss_off.to_numpy()[g]
                a, b = t.rating_total.to_numpy()[g], ref.rating_total.to_numpy()[g]
                c = float(np.sum(w * a * b) / np.sum(w * b * b))       # one scalar: the amplitude part
                d, dp = np.abs(a - b), np.abs(a - c * b)
                for lo, hi, name in [(0.0, np.inf, "everyone")] + TIERS:
                    k = (t.poss_off.to_numpy()[g] >= lo) & (t.poss_off.to_numpy()[g] < hi)
                    mv.append(dict(model=m, scope=scope, tier=name, players=int(k.sum()), amplitude=c,
                                   median_move=float(np.median(d[k])), p90_move=float(np.percentile(d[k], 90)),
                                   share_over_0_1=float((d[k] > 0.1).mean()),
                                   median_move_after_amplitude=float(np.median(dp[k]))))
        mv = pd.DataFrame(mv)
        mv.to_csv(OUT / "csv" / "piece_movement.csv", index=False)
        with pd.option_context("display.max_rows", None, "display.width", 250):
            say("\n=== Movement against RAPM x slope (net rating, points per 100)")
            say(mv.round(3).to_string(index=False))


# ---------------------------------------------------------------------------------------------- stage: match
def spread(t: pd.DataFrame, col: str, poss: str) -> pd.Series:
    """Per season: the possession-weighted standard deviation over players with 100+ possessions."""
    w = t[poss].where(t[poss] >= 100, 0.0)
    g = t.assign(_w=w, _x=t[col] * w, _xx=t[col] ** 2 * w).groupby("season")[["_w", "_x", "_xx"]].sum()
    return np.sqrt(g._xx / g._w - (g._x / g._w) ** 2)


def stage_match() -> None:
    """Every model's rankings table rescaled, per season and side, to vanilla RAPM's spread, so the year-over-year
    test compares ORDER at one amplitude.  The piece models predict a shrunk target and each comes out with its
    own spread; the team-game error is not blind to a spread that far off.  A diagnostic: nothing here ships.
    Uses only the rated season's own ratings, so nothing from the scored seasons reaches it."""
    van = pd.read_parquet(OUT / "season_ratings_vanilla.parquet")
    ref = {s: spread(van, f"rating_{s}", f"poss_{s}") for s in SIDES}
    for path in sorted(OUT.glob("season_ratings_piece_*.parquet")):
        if path.stem.endswith("_sdmatch"):
            continue
        m = path.stem[len("season_ratings_piece_"):]
        t = pd.read_parquet(path)
        factors = {}
        for s in SIDES:
            k = ref[s] / spread(t, f"rating_{s}", f"poss_{s}")
            factors[s] = float(k.median())
            t[f"rating_{s}"] = t[f"rating_{s}"] * t.season.map(k).to_numpy()
        t["rating_total"] = t.rating_off + t.rating_def
        t["offense"], t["defense"] = t.rating_off, -t.rating_def
        t.to_parquet(OUT / f"season_ratings_piece_{m}_sdmatch.parquet", index=False)
        say(f"  {m}: multiplied by a median {factors['off']:.2f} on offence and {factors['def']:.2f} on defence "
            f"to match vanilla's spread")


# ---------------------------------------------------------------------------------------------- stage: yoy
def stage_yoy(tag: str, suffix: str = "") -> None:
    from eracoef.holdout import paired as h_paired, pooled
    res = pd.read_parquet(OUT / f"yoy_{tag}.parquet")
    has_dir = res.system.str.contains(":")
    res["direction"] = np.where(has_dir, res.system.str.rsplit(":", n=1).str[-1], "prev")
    res["system"] = res.system.str.rsplit(":", n=1).str[0]
    # a model told whether he stayed is scored from two tables: `_fwd` (flag = same team the next season) for the
    # rows where the rated season predicts the one after it (`:prev`), `_bwd` for the rows where it predicts the
    # one before it (`:next`).  Stitched here into one system of 56 observations.
    stitched = []
    for fwd in sorted(s for s in res.system.unique() if s.endswith(f"_fwd{suffix}")):
        base = fwd[: -len(f"_fwd{suffix}")]
        bwd = f"{base}_bwd{suffix}"
        if bwd in set(res.system):
            part = res[((res.system == fwd) & (res.direction == "prev")) | ((res.system == bwd) & (res.direction == "next"))]
            stitched.append(part.assign(system=base + suffix))
    if stitched:
        res = pd.concat([res[~res.system.str.contains(r"_(?:fwd|bwd)", regex=True)]] + stitched, ignore_index=True)
        comb = res.copy()
        comb["system"] = comb.system + ":" + comb.direction
        comb.drop(columns="direction").to_parquet(OUT / f"yoy_{tag}_stitched.parquet", index=False)
    res["held_out"] = res.held_out + res.direction.map({"prev": 0.0, "next": 0.5}).fillna(0.0)
    allr = res[res.split == "all"]
    with pd.option_context("display.max_rows", None, "display.width", 250, "display.max_columns", 30):
        say("\n=== Year-over-year test, both directions (56 observations)")
        say(pooled(allr)[["system", "seasons", "game_armse", "game_vs_no_ratings", "armse", "vs_no_ratings",
                          "calib_side", "scale_off", "scale_def", "covered"]].round(4).to_string(index=False))
        pairs = [("piece_pieces_linear", "piece_rapm_linear"), ("piece_pieces_gbdt", "piece_rapm_gbdt"),
                 ("piece_pieces_gbdt_boruta", "piece_rapm_gbdt"),
                 ("piece_pieces_linear_stay", "piece_rapm_linear_stay"), ("piece_pieces_gbdt_stay", "piece_rapm_gbdt_stay"),
                 ("piece_pieces_linear_stay", "piece_pieces_linear"), ("piece_pieces_gbdt_stay", "piece_pieces_gbdt"),
                 ("piece_rapm_scaled", "vanilla"), ("piece_pieces_linear", "vanilla"), ("piece_pieces_gbdt", "vanilla"),
                 ("piece_pieces_linear_stay", "vanilla"), ("piece_pieces_gbdt_stay", "vanilla")]
        pairs = [(a + suffix, b if b == "vanilla" else b + suffix) for a, b in pairs]
        have = set(allr.system)
        for cand, ref in pairs:
            if cand not in have or ref not in have:
                continue
            sub = allr[allr.system.isin([cand, ref])]
            for value in ("tg", "calib_side"):
                t = h_paired(sub, ref, value)
                t = t[t.system == cand]
                if not len(t):
                    continue
                say(f"  {cand} vs {ref} on {value}: " + ", ".join(
                    f"{c} {t[c].iloc[0]:.4f}" if isinstance(t[c].iloc[0], float) else f"{c} {t[c].iloc[0]}"
                    for c in ("mean_diff", "se", "z", "wins", "n_seasons")))
        mov = res[res.split == "movers"]
        if len(mov):
            say("\n=== Movers split (stint-level mse; stints labelled by how many of the ten changed team)")
            say(pooled(mov)[["system", "group", "seasons", "armse", "calib_side"]].round(4).to_string(index=False))
            for cand, ref in pairs[:7]:
                if cand in have and ref in have:
                    for value in ("mse", "calib_side"):
                        t = h_paired(mov[mov.system.isin([cand, ref])], ref, value)
                        t = t[t.system == cand]
                        say(f"  {cand} vs {ref}, stint {value} by group:")
                        say(t[["group", "mean_diff", "se", "z", "wins", "n_seasons"]].round(4).to_string(index=False))


# ---------------------------------------------------------------------------------------------- main
def main() -> None:
    check_flags()
    stage = flag("stage", "coef")
    cfg = load_config()
    tag = flag("tag", "pieces")
    if stage == "yoy":
        stage_yoy(tag, flag("suffix", ""))
        return
    if stage == "match":
        stage_match()
        return
    p = load_panel()
    q = make_pairs(p)
    panel_fold, pair_fold = player_folds(p, q)
    shift = BalancedGroupKFold(N_FOLDS).mean_shift(q.y_net.to_numpy(), q.player_id.to_numpy(), q.w_net.to_numpy())
    say(f"panel {len(p):,} player-seasons; {len(q):,} pairs ({(q.direction == 'next').sum():,} next, "
        f"{(q.direction == 'prev').sum():,} previous), {q.mover.mean():.1%} with a new main team; player folds' "
        f"label shift {np.round(shift, 4).tolist()}")
    seasons = [int(s) for s in flag("seasons", "").split(",") if s] or sorted(int(s) for s in p.season.unique())
    boot, trials, threads = int(flag("boot", 200)), int(flag("trials", 50)), int(flag("threads", 4))
    oof_path = ROOT / flag("oof", "outputs/piece_oof.parquet")
    t0 = time.time()
    if stage == "coef":
        stage_coef(p, q, panel_fold, pair_fold, cfg, boot)
    elif stage == "shootout":
        models = [m for m in flag("models", ",".join(MODELS)).split(",") if m]
        unknown = [m for m in models if m not in MODELS]
        if unknown:
            raise SystemExit(f"unknown models {unknown}; known: {list(MODELS)}")
        boruta = flag("boruta", "outputs/csv/piece_boruta_table.csv")
        stage_shootout(p, q, panel_fold, pair_fold, cfg, oof_path, models, seasons, trials, threads,
                       ROOT / boruta if boruta else None)
    elif stage == "score":
        stage_score(p, q, oof_path)
    else:
        raise SystemExit(f"unknown stage {stage}")
    say(f"stage {stage} done ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
