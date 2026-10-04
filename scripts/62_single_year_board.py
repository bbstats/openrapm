"""Build the whole board from the single-year pipeline: LOSO RAPM -> SPM -> PriorRidgeCV, season by season.

    python scripts/62_single_year_board.py [--first=1997] [--last=2026] [--out=season_ratings_sy]
                                           [--score=1] [--target_off=xpts_ft] [--target_def=x3def_w0.25]
                                           [--free_scale=1] [--buckets=low_poss:2] [--boards=2015,2024]
                                           [--features=boruta_noonc|boruta|sy_noonc|sy]
                                           [--exclude_neighbours=0] [--rows=chunks|player|season|season_capped]
                                           [--chunk_sizes=1,2,3|all] [--chunk_label=career|outside|adjacent] [--chunk_weight=career|label]
                                           [--label_threads=12] [--rate_same_team=0|1|real|both] [--same_team_diag=0|1]
                                           [--dump_rows=<name>] [--team_season_intercept=0|1] [--age_adjust_labels=0|1]
                                           [--trade_weight=<floor>] [--trade_bands=0|1] [--adjacent_penalty=3000]
                                           [--crossfit=scale|0|1]
                                           [--params_mult=l2_leaf_reg:5,min_child_weight:5] [--params_set=depth:3]
                                           [--booster_params=<name>] [--save_models=<name>]
                                           [--player_folds=0|5] [--unshrink_label=def|1|off|0] [--lambda_player=13037|cv|<value>]
                                           [--lambda_off=<value>] [--lambda_def=<value>]
                                           [--save_priors=<name>] [--priors_from=<name>] [--centre=1]
                                           [--blend_off=<x>/<k>/<a|lin>] [--blend_def=<x>/<k>/<a|lin>]

`--player_folds=5` (2026-09-14, the owner's call after the memorisation test): the SPM is fitted once
per player fold and every player's prior comes from the fit that never saw any of his rows, so it cannot
return his career number from his fingerprint.  The folds are balanced on the label (`singleyear.
stratified_player_folds`), which is the condition under which the leave-out mean shift of Austin, Pe'er
and Korem (2025) is zero; the residual shift is printed.  Five fits per season instead of one.

`--crossfit=1` (2026-09-13, the amplitude run): the free prior scale is a least-squares coefficient on the
prior summed over the five on the floor, and the prior carries the season's own on-court columns, so the
coefficient reads the season's outcomes back and comes out too big (the year-over-year test wants the
rankings multiplied by about 0.75 on both sides).  With cross-fitting each row's prior column is built from
a prior whose on-court columns were rebuilt WITHOUT that row's CV fold, so the scale is priced on games the
prior has not seen.  The final rating is still `scale * (full-season prior) + residual`.

`--rows=chunks` is the owner's design (2026-09-13): the career row per player kept exactly, PLUS rows built
from contiguous chunks of his seasons at the sizes given, with `singleyear.CHUNK_FEATURES` telling the
booster how much evidence each row rests on.  `singleyear.chunk_rows` has the weights and the reasoning.

`--rows=season` (2026-09-13) trains the prior on one row per PLAYER-SEASON -- a season's own box score,
labelled with the player's RAPM over his OTHER seasons, the rated season and that one both left out -- so a
training row is the same kind of row the prior is asked about at inference (one noisy season) and never
shares a game with its label.  `--rows=player` is the earlier shape: one row per player, his box score
averaged over every season but the rated one.  `singleyear.season_rows` / `singleyear.prior_rows`.

`--first` / `--last` are the seasons the TARGET is accumulated over and must stay the full range -- a
leave-one-season-out RAPM with one season in it has nothing left.  `--boards=` restricts which seasons a
rating is produced for, which is how a change is measured on two seasons instead of thirty.

Three more flags.  The first two are off by default and not in the shipped run; the third describes a
setting that IS shipped:

  `--trade_weight=F`   weight every training row of the prior by its player's `team_movement`
                       (the chance two possessions of his career came from different teams,
                       `1 - sum(share ** 2)`) plus F.  The owner's idea, 2026-09-16: a label is only
                       identified by team variation, so weight the rows by how much of it each player
                       has.  **F = 0 drops the one-team players outright** -- 29% of the players behind
                       a 2026 label, every single-franchise star among them -- which is what the one
                       run of this rule lost on, so sweep F above zero if it is revived.
                       `--rows=chunks` only; it has no effect on the other row shapes.
                       `singleyear.team_movement` / `reweight_by_movement`.
  `--dump_shap=NAME`   write outputs/prior_shap_NAME.parquet: per feature, how far one moves a
                       player's prior.  Diagnostic; changes no number.
  `--unshrink_label=`  which labels are put back on one scale before the SPM is trained on them.
                       **`def` is the default and is shipped** (adopted 2026-09-18, experiment 22): the
                       defensive label only, worth z -5.24 on the year-over-year test and z -7.14 on the
                       trade loss, with the consensus top five at 5 of 5 and offence moving a median
                       0.022 per 100.  `1` does BOTH sides and is rejected -- it collapses the 2026
                       offensive spread from 1.60 to 1.14 and empties the top of the list of its
                       offensive stars.  `off` is the offensive label alone, untested.  `0` is the
                       pre-adoption behaviour.  `--unshrink_floor=N` is the possession floor the
                       un-shrinking is capped at (default 4,444); see `unshrink_label`.

`--exclude_neighbours=N` also keeps the N seasons either side of the rated one out of the target and out
of the prior's training rows.  That is the setting for the year-over-year test (`scripts/63_yoy.py`),
which scores a season's rankings on the neighbouring seasons' games: with the default 0 the prior's
coefficients were learned from the very seasons being scored.  The rated season's own games are still
the evidence, exactly as in the shipped setting; only the population-level fits stop short of the
scored seasons.

For each season H:
  1. the TARGET is a RAPM over every season except H, one rating per player (looseason.LeaveSeasonOutRAPM),
     at the penalties swept on the end-to-end score (DECISIONS.md);
  2. the PRIOR is a chimeraboost mapping a player's all-other-seasons feature averages to that target,
     then asked about H's own box score -- the SPM step.  What it sees is `singleyear.feature_set()`,
     by default the BorutaShap selection with the four `onc_*` columns removed (see `singleyear` for why
     the most important-looking feature in the set is a leak);
  3. the RATING is PriorRidgeCV on H's games, shrinking toward that prior, its three penalties chosen by
     cross-validation over whole games on a closeness-weighted team-game objective.

Nothing that touches season H is ever fit on season H, and no player's own other seasons reach his H
rating except through population-level model coefficients ("single year or bust", HANDOFF ruling 2).

**Two targets, one per side.**  The offensive fit explains `xpts_ft` -- points with made free throws
replaced by the shooter's padded expectation -- and the defensive fit explains `x3def_w0.25`, which also
replaces three quarters of every opponent three-point make by the shooter's own padded 3P%.  Both are
measured variance reductions on this project's own criterion (+0.26 and -0.39 to -0.47 per 100;
DECISIONS.md 37-52), which is one to two orders of magnitude more than the plus-minus stage is currently
worth.  So the board runs two RAPMs and two ridges and takes one side from each.  `--target_off=pts
--target_def=pts` reproduces the raw-points board.

**`--score` (on by default) is the diagnostic that can see compression.**  Per season it refits both sides
on the first 75% of the games and scores the last 25% AGAINST ACTUAL POINTS -- the luck adjustment is a
fitting target, never a scoring one -- reporting `game_armse` and `scale_off` / `scale_def`, what the unseen
quarter wants each side multiplied by.  A team-game total is linear in a per-player linear map, so
`game_armse` is nearly blind to a uniform rescale of the board; the scales are not.  Above 1.0 = too narrow.

Writes outputs/<out>.parquet with one row per player-season, which scripts/52_site.py reads, and
outputs/<out>_score.parquet with the per-season diagnostic.
"""
import json
import os
import sys
import time
from pathlib import Path

# Thread pinning, BEFORE numpy and chimeraboost are imported (2026-09-14).  The booster's linear leaves
# are thousands of tiny solves; a BLAS that spins up twelve threads for each one thrashes, and numba's
# thread pool on top of it can livelock when more than one process is running: the same 1997 fit that
# took 15 s alone in the morning ran 45+ minutes with two processes up, and 111 s with everything
# pinned to one thread.  BLAS gets one thread; numba's count is OPENRAPM_NUMBA_THREADS (default 4).
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
os.environ.setdefault("NUMBA_NUM_THREADS", os.environ.get("OPENRAPM_NUMBA_THREADS", "4"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from chimeraboost import ChimeraBoostRegressor  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eracoef import pieces  # noqa: E402
from eracoef import singleyear as sy  # noqa: E402
from eracoef.config import load_config
from eracoef.seasons import drop_untrainable  # noqa: E402
from eracoef.holdout import Context, Ratings, predict_season, score  # noqa: E402
from eracoef.inseason import season_frac  # noqa: E402
from eracoef.investigate import offcourt_rates, oncourt_rates  # noqa: E402
from eracoef.looseason import LeaveSeasonOutRAPM  # noqa: E402
from eracoef.priorridge import PriorRidgeCV, armse, calibration_miss, game_folds  # noqa: E402
from eracoef.stackprior import StackedSPM  # noqa: E402
from eracoef.xshoot import DEFENSE_TARGETS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

# the RAPM penalties, the possession floor and the two targets live in `singleyear` so the Boruta run
# selects features against the same thing the board trains on
RAPM_OFFENSE_LAMBDA = sy.RAPM_OFFENSE_LAMBDA
RAPM_DEFENSE_LAMBDA = sy.RAPM_DEFENSE_LAMBDA
RAPM_CONTEXT_LAMBDA = sy.RAPM_CONTEXT_LAMBDA
MIN_POSSESSIONS = sy.MIN_POSSESSIONS

BOARD_PLAYER_LAMBDAS = np.round(np.logspace(np.log10(2000.0), np.log10(1.0e9), 8), 0)
BOARD_OFFENSE_LAMBDAS = BOARD_PLAYER_LAMBDAS       # per side, when --lambda_off / --lambda_def are given
BOARD_DEFENSE_LAMBDAS = BOARD_PLAYER_LAMBDAS
BOARD_CONTEXT_LAMBDAS = np.array([0.0, 1.0e3])

SCORE_TARGET = "pts"        # the board is always SCORED on the points that were actually scored


# The shared command line (scripts/_cli.py): `flag` records every name it is asked for so
# `check_flags` below can refuse one that was never asked for.  A misspelled flag used to be
# ignored silently, which is how a run looks right and is wrong.
from _cli import check_flags, flag as _flag, switch   # noqa: E402


def _target(name):
    """A `design.TARGETS` key, or the `xshoot.DEFENSE_TARGETS` callable of that name."""
    return DEFENSE_TARGETS[name] if name in DEFENSE_TARGETS else name


FREE_PRIOR_SCALE = True     # --free_scale=0 pins the prior at exactly the amplitude it came with
LAM_BUCKETS: dict = {}      # --buckets=low_poss:2 multiplies the bench's penalty


def _ridge(design, prior, free_prior_scale=None, lam_buckets=None, fold_prior=None, crossfit_penalty=True):
    return PriorRidgeCV(offense_lambdas=BOARD_OFFENSE_LAMBDAS, defense_lambdas=BOARD_DEFENSE_LAMBDAS,
                        context_lambdas=BOARD_CONTEXT_LAMBDAS, n_folds=5,
                        free_prior_scale=FREE_PRIOR_SCALE if free_prior_scale is None else free_prior_scale,
                        lam_buckets=LAM_BUCKETS if lam_buckets is None else lam_buckets).fit(
        design, prior["O"], prior["D"], fold_prior=fold_prior, crossfit_penalty=crossfit_penalty)


TIER_EDGES = [0.0, 500.0, 1500.0, 4444.0, np.inf]       # label possessions; the top tier is fully un-shrunk


def unshrink_label(target: pd.DataFrame, lam_o: float, lam_d: float, n_floor: float,
                   sides: tuple = ("offense", "defense")) -> pd.DataFrame:
    """Put every player's label on one scale, and shrink the thin ones toward their tier's level, not 0.

    `sides` is which columns to do it to.  The two sides are separable and they behave differently: the
    trade loss reads the defensive half as the largest per-player gain measured here (z -6.1 among
    players over 1,500 possessions) while the offensive half collapses the 2026 offensive spread from
    1.60 to 1.14 and empties the top of the list of its offensive stars (DECISIONS.md, experiment 21).

    A ridge coefficient is about s_i * raw_i with s_i = n_i / (n_i + lambda): a 54,000-possession
    player keeps 57% of his true impact, a 2,600-possession one 6%, so the label's spread grows
    fifteen-fold with career length and the SPM learns "more career possessions = bigger number" -- the
    lever that put LeBron and Curry near the top of 2026 (DECISIONS.md).  Dividing by s_i restores the
    scale, but for a thin player it multiplies noise by hundreds.  So the un-shrinking is capped at
    s_floor = n_floor / (n_floor + lambda), and what the cap leaves un-recovered is filled from the
    player's possession TIER: its mean impact, estimated stably by un-shrinking the tier's MEAN
    coefficient (hundreds of players average the noise away).  That is the owner's second point
    (2026-09-14): near-zero-possession players should sit at the replacement level of their kind,
    -2 or -3, not at 0.

        label_i = beta_i / max(s_i, s_floor)  +  (1 - s_i / max(s_i, s_floor)) * m(tier_i)
        m(tier)  = mean_tier(beta) / mean_tier(s)

    Above n_floor the second term is zero and the label is the raw-scale coefficient; below it the label
    is shrunk toward the tier level by s_i / s_floor, the usual empirical-Bayes form with an honest
    centre.  The row weight stays the label's possessions, the right precision weight for a label whose
    noise variance falls as 1 / n.
    """
    out = target.copy()
    n = out.possessions.to_numpy(float)
    tier = np.digitize(n, TIER_EDGES[1:-1])
    for column, lam in [c for c in (("offense", lam_o), ("defense", lam_d)) if c[0] in sides]:
        beta = out[column].to_numpy(float)
        s = n / (n + lam)
        s_floor = n_floor / (n_floor + lam)
        s_cap = np.maximum(s, s_floor)
        m = np.zeros(len(TIER_EDGES) - 1)
        for t in range(len(m)):
            mask = tier == t
            if mask.any():
                m[t] = beta[mask].mean() / s[mask].mean()
        out[column] = beta / s_cap + (1.0 - s / s_cap) * m[tier]
        out.attrs[f"tier_level_{column}"] = m.round(3).tolist()
    return out


def fit_with_team_season(make, X, y, w, codes, n_groups, normalise: bool = True):
    """A booster plus one random intercept per team-season, around ANY booster (experiment 27, the owner's plan).

    chimeraboost 0.34's own `random_effects=True` refuses a bagged model, and the offensive booster is a bag of
    five, so the algorithm is run by hand with chimeraboost's own solver: fit F on y; take its residuals;
    `estimate_ratio_reml` for the noise-to-group variance ratio; `solve_intercepts`; refit F from scratch on
    y - b[group].  The ratio and intercepts are then solved once more against the refit, for the log only.

    The weights are normalised to mean 1 for the solve and only there: the solver reads weights as ROW COUNTS,
    so possession weights would tell it there are millions of rows and switch the shrinkage off, while the
    booster keeps the weights it always had (its leaf penalty is not scale-free).  A ratio of infinity means no
    team-season signal: then b is exactly 0 and the refit is skipped, since it would be the same fit.

    The caller rates with the trees alone: the rated season is never a training season, so none of its
    team-seasons has an intercept -- he is rated as if on an average team.  Returns (booster, log).
    """
    from chimeraboost.random_effects import estimate_ratio_reml, solve_intercepts

    first = make().fit(X, y, sample_weight=w)
    normal = w / w.mean() if normalise else w          # `normalise=False` only to compare with chimeraboost's own
    resid = y - first.predict(X)
    ratio = float(estimate_ratio_reml(resid, codes, n_groups, normal))
    if not np.isfinite(ratio):
        return first, dict(ratio=ratio, ratio_refit=ratio, intercept_sd=0.0)
    b = solve_intercepts(resid, codes, n_groups, ratio, normal)
    second = make().fit(X, y - b[codes], sample_weight=w)
    resid = y - second.predict(X)
    ratio_refit = float(estimate_ratio_reml(resid, codes, n_groups, normal))
    b_refit = solve_intercepts(resid, codes, n_groups, ratio_refit, normal) if np.isfinite(ratio_refit) else b * 0
    per_row = b_refit[codes]
    sd = float(np.sqrt(np.average((per_row - np.average(per_row, weights=w)) ** 2, weights=w)))
    return second, dict(ratio=ratio, ratio_refit=ratio_refit, intercept_sd=sd)


def team_season_codes(train: pd.DataFrame) -> tuple:
    """0..G-1 group codes from `singleyear.TEAM_SEASON`; a row without a team-season is a group of its own."""
    labels = train[sy.TEAM_SEASON].to_numpy(np.int64).copy()
    missing = labels < 0
    labels[missing] = -1 - np.arange(int(missing.sum()))
    codes, uniques = pd.factorize(labels)
    return codes.astype(np.int64), len(uniques)


class OutOfPlayerSPM:
    """The SPM fitted once on everything and once per player fold; `predict` gives each player the fit
    that never saw his rows (`--player_folds=N`; 0 = the plain single fit).  `intercept=True` fits every one
    of those boosters with a team-season random intercept (`fit_with_team_season`) and predicts with the
    trees alone.

    `by_player=True` (experiment 32, 2026-10-03) hands chimeraboost the player ids as `groups`, so the 20% it
    holds out to choose its tree count -- and to run its linear-leaf and cross-feature races -- is whole
    players, not random rows.  A player's career row and chunks carry one label, so a row split puts him on
    both sides and the held-out error rewards remembering him; the prior is only ever asked about players
    its fit never saw.  Off, `groups=None` is chimeraboost's own default: the shipped fit, unchanged."""

    def __init__(self, params: dict, n_folds: int, intercept: bool = False, by_player: bool = False):
        self.params, self.n_folds, self.intercept = dict(params), int(n_folds), bool(intercept)
        self.by_player = bool(by_player)
        assert not (self.intercept and self.by_player), "the team-season intercept fit takes no groups"

    def _fit_one(self, X, y, w, codes, n_groups, players=None):
        def make():
            return ChimeraBoostRegressor(random_state=0, **self.params)
        if not self.intercept:
            return make().fit(X, y, sample_weight=w, groups=players if self.by_player else None), None
        return fit_with_team_season(make, X, y, w, codes, n_groups)

    def fit(self, train: pd.DataFrame, model_feats: list) -> "OutOfPlayerSPM":
        X, y, w = train[model_feats].to_numpy(float), train.target.to_numpy(float), train.weight.to_numpy(float)
        players = train.index.to_numpy()
        codes, n_groups = team_season_codes(train) if self.intercept else (None, 0)
        self.full_, self.log_ = self._fit_one(X, y, w, codes, n_groups, players)
        self.fold_models_, self.excluded_, self.fold_logs_ = [], [], []
        self.shift_ = np.zeros(0)
        if self.n_folds > 1:
            fold = sy.stratified_player_folds(train, self.n_folds)
            self.shift_ = sy.fold_mean_shift(train, fold)
            for f in range(self.n_folds):
                keep = fold != f
                model, log = self._fit_one(X[keep], y[keep], w[keep], None if codes is None else codes[keep],
                                           n_groups, players[keep])
                self.fold_models_.append(model)
                self.fold_logs_.append(log)
                self.excluded_.append(set(train.index[~keep].tolist()))
        return self

    def predict(self, frame: pd.DataFrame, model_feats: list) -> np.ndarray:
        X = frame[model_feats].to_numpy(float)
        out = self.full_.predict(X)
        ids = frame.player_id.to_numpy()
        for model, excluded in zip(self.fold_models_, self.excluded_):
            idx = np.fromiter((p in excluded for p in ids), dtype=bool, count=len(ids))
            if idx.any():
                out[idx] = model.predict(X[idx])
        return out


def prior_shap_slopes(model, train: pd.DataFrame, feats: list, side: str, season: int) -> pd.DataFrame:
    """Per feature, how far one standard deviation of it moves the PRIOR, out of player fold.

    The same quantity `scripts/72_tradeset_shap.py` reports for the trade-set correction, so the two
    can be read side by side: `slope_per_sd` is the weighted least-squares slope of the feature's
    TreeSHAP contribution on the standardised feature, in points per 100 of prior per standard
    deviation of the statistic.  Signed.  The plain mean of a contribution is ~0 by construction
    (contributions are deviations from the model's own mean), so it is reported only to show that.

    Out of fold: each player's contributions come from the fit that never saw a row of his, matching
    how his prior is actually produced.
    """
    X = train[feats].to_numpy(float)
    contribution = model.full_.shap_values(X)
    ids = train.index.to_numpy()
    for fold, excluded in zip(model.fold_models_, model.excluded_):
        index = np.fromiter((p in excluded for p in ids), dtype=bool, count=len(ids))
        if index.any():
            contribution[index] = fold.shap_values(X[index])
    weight = train.weight.to_numpy(float)
    out = []
    for j, name in enumerate(feats):
        column, part = X[:, j], contribution[:, j]
        mean_x = float(np.average(column, weights=weight))
        sd_x = float(np.sqrt(np.average((column - mean_x) ** 2, weights=weight)))
        standardised = (column - mean_x) / (sd_x if sd_x > 0 else 1.0)
        mean_shap = float(np.average(part, weights=weight))
        out.append(dict(season=season, side=side, feature=name,
                        slope_per_sd=float(np.average((part - mean_shap) * standardised, weights=weight)),
                        mean_signed=mean_shap,
                        moves_typical=float(np.average(np.abs(part), weights=weight))))
    return pd.DataFrame(out)


def _saved_fold_prior(folds: list, design):
    """The saved per-fold priors, served by matching the ridge's train mask to the fold it holds out."""
    fold = game_folds(design, 5, 0)

    def fold_prior(train_mask):
        held = ~np.asarray(train_mask, dtype=bool)
        for f, pair in enumerate(folds):
            if np.array_equal(held, fold == f):
                return pair
        raise ValueError("the ridge's fold does not match any saved fold; rebuild the priors")

    return fold_prior


def _fold_prior_builder(wd_o, wd_d, models, held_frames, model_feats, teams=None):
    """A prior for a CV fold that has NOT seen the fold's games: the season's on-court columns (`onc_*`)
    rebuilt from the training rows alone, the fitted boosters re-asked.  `--crossfit=1`.

    The panel's `onc_o` / `onc_d` are averaged over every game of the season, so a prior column built from
    them already contains the outcome of any game the ridge holds out, and the free prior scale -- a
    least-squares coefficient on that column -- is inflated by it.  `wd_o` / `wd_d` are the season's
    designs on the two targets the panel built `onc_*` from (`xpts_ft`, `x3def`), so the rebuilt columns
    differ from the panel's in the games used and nothing else (`scratch/onc_leak.py` checked that the
    full-season rebuild reproduces the panel to four decimals).

    The RAPM pieces (experiment 26) are the same kind of column -- the season's own games -- so when a model
    reads any of them they are rebuilt per fold too, through `pieces.season_pieces`, the code the panel was
    built with (scripts/86_context_panel.py checked that a fold of every row reproduces the panel's pieces to
    1e-12); `teams` is the season's (game_id, player_id) -> team lookup that needs.  `same_team` is whatever
    the held frames carry: 0, "he changed teams", unless the diagnostic asked otherwise.  Both ridges (one
    per target) ask for the same five folds, so each fold is built once.
    """
    ids_of_ps = wd_o.spec.ps_table["player_id"].to_numpy()
    # the off-court family too, when the panel carries it (scripts/65_offcourt_panel.py)
    with_offc = all(c in held_frames["O"].columns for c in sy.OFFC + sy.NET)
    with_pieces = teams is not None and any(c in model_feats[s] for s in ("O", "D") for c in sy.PIECES)
    rebuilt = sy.ONC + (sy.OFFC + sy.NET if with_offc else []) + (sy.PIECES if with_pieces else [])
    built: dict = {}

    def fold_prior(train_mask):
        key = np.packbits(np.asarray(train_mask, dtype=bool)).tobytes()
        if key in built:
            return built[key]
        fo, fd = wd_o.subset(train_mask), wd_d.subset(train_mask)
        got = oncourt_rates(fo, fd)
        onc = pd.DataFrame({"player_id": ids_of_ps, **{c: got[c].to_numpy(dtype=float) for c in sy.ONC}})
        if with_offc:
            off = offcourt_rates(fo, fd)
            for c in sy.OFFC:
                onc[c] = off[c].to_numpy(dtype=float)
            onc["net_o"], onc["net_d"] = onc.onc_o - onc.offc_o, onc.onc_d - onc.offc_d
        if with_pieces:
            got_pieces, _ = pieces.season_pieces(fo, fd, teams)
            onc = onc.merge(got_pieces, on="player_id", how="left")
        out = {}
        for side in ("O", "D"):
            h = held_frames[side].drop(columns=rebuilt).merge(onc, on="player_id", how="left")
            h[rebuilt] = h[rebuilt].fillna(0.0)
            out[side] = dict(zip(h.player_id.to_numpy(), models[side].predict(h, model_feats[side])))
        built[key] = (out["O"], out["D"])
        return built[key]

    return fold_prior


def blend_prior(prior: dict, poss: dict, x: float, k: float, a: float) -> dict:
    """The owner's replacement blend (2026-09-15): `prior * w(n) + x * (1 - w(n))`.

    Two shapes for the weight.  `a` a number gives the rational weight `w = n^a / (n^a + k^a)`, which never
    quite reaches the prior.  **`a = "lin"` gives the owner's ramp (2026-09-15, experiment 17): `w =
    min(n / k, 1)`** -- no prior at all at zero possessions, exactly the replacement level `x` there, and
    exactly the prior at `k` possessions and above, linear in between.  `k` is then the possession count at
    which the prior is trusted whole, and it is the only thing swept.

    `n` is the player's possessions ON THAT SIDE -- offensive possessions on offence, defensive on defence
    -- and `x` is in the same raw sign as the prior it is blended into, so a defensive `x` above zero means
    a player with few possessions allows more.  `k` and `a` come from `scripts/67_blend_apm.py`, fitted
    against APM and never RAPM: a ridge penalty pulls his RAPM to zero, so a blend fitted on RAPM would put
    `x` at zero and call a man nobody has seen league average.

    Applied to the prior only.  The label is untouched -- experiment 1 put the tier level INTO the label
    and scrambled the top of the list, because "few career possessions" is what a rookie looks like at
    inference and the booster handed them the tier's optimism.
    """
    if not k:
        return prior
    linear = isinstance(a, str)
    out = {}
    for pid, value in prior.items():
        n = float(poss.get(pid, 0.0))
        if linear:
            w = min(n / k, 1.0) if n > 0 else 0.0
        else:
            w = 0.0 if n <= 0 else n ** a / (n ** a + k ** a)
        out[pid] = value * w + x * (1.0 - w)
    return out


def side_possessions(design) -> tuple[dict, dict]:
    """{player_id: offensive possessions}, {player_id: defensive possessions} for one season's design.

    The two differ by about nine possessions in twenty-five hundred, but the owner asked for the right
    count on each side and `design.game_poss` already carries both.

    The board's `poss_def` column comes from here (2026-09-18).  It used to be a copy of `poss_off`,
    which made `--match_spread` and any per-side exposure correction read the wrong count on
    defence; a parquet written before that date still carries the copy, and
    scripts/68_exposure_slope.py says so when it sees one.
    """
    table = (design.game_poss.groupby("psx_idx")[["poss_off", "poss_def"]].sum()
             .reindex(range(design.spec.n_psx)).fillna(0.0))
    psx = design.spec.psx_table
    ids = psx["player_id"].to_numpy()
    off = table.poss_off.to_numpy()[psx["psx_idx"].to_numpy()]
    dfe = table.poss_def.to_numpy()[psx["psx_idx"].to_numpy()]
    return (pd.Series(off).groupby(ids).sum().to_dict(),
            pd.Series(dfe).groupby(ids).sum().to_dict())


def _first_75(design):
    position = season_frac(design.games)[design.rows["game_idx"].to_numpy()]
    return design.subset(position < 0.75), design.subset(position >= 0.75)


def main():
    check_flags()      # refuse a flag this script does not understand (scripts/_cli.py)
    cfg = load_config(ROOT / "config.yaml")
    ctx = Context.load(cfg)
    # one row per player, season and team, for the movement weighting; loaded here because the model
    # layer may not open a file
    roles_played = None
    if _flag("trade_weight") is not None:
        _roles = pd.read_parquet(ROOT / "data/cache/roles_RSPO.parquet")
        roles_played = _roles[_roles.poss_on > 0][["player_id", "season", "team_id", "poss_on"]].copy()
        # franchises (2026-09-30): Charlotte's 1997-2002 id is New Orleans's, or the players who moved with
        # the team read as movers
        roles_played["team_id"] = sy.franchise(roles_played.team_id.to_numpy(), roles_played.season.to_numpy())
    first, last = int(_flag("first", 1997)), int(_flag("last", 2026))
    out = ROOT / "outputs" / f"{_flag('out', 'season_ratings_sy')}.parquet"
    out_stayed = out.with_name(out.stem + "_stayed.parquet")   # `--rate_same_team=both`
    do_score = _flag("score", "1") not in ("0", "no", "false")
    global FREE_PRIOR_SCALE, LAM_BUCKETS
    FREE_PRIOR_SCALE = _flag("free_scale", "1") not in ("0", "no", "false")
    LAM_BUCKETS = {k: float(v) for k, v in
                   (part.split(":") for part in _flag("buckets", "").split(",") if part)}
    names = {"O": _flag("target_off", sy.OFFENSE_TARGET), "D": _flag("target_def", sy.DEFENSE_TARGET)}
    seasons = list(range(first, last + 1))
    boards = [int(x) for x in _flag("boards", "").split(",") if x] or seasons
    assert set(boards) <= set(seasons), f"--boards outside [{first}, {last}]"
    exclude_neighbours = int(_flag("exclude_neighbours", 0))
    assert exclude_neighbours >= 0
    # the incumbent since 2026-09-14 (DECISIONS.md, experiments 3 and 4b): the career row plus 1-, 2- and
    # 3-season chunks, and the free prior scale priced on cross-fitted columns
    row_shape = _flag("rows", "chunks")
    assert row_shape in ("player", "season", "season_capped", "chunks"), "--rows=player|season|season_capped|chunks"
    chunk_flag = _flag("chunk_sizes", "1,2,3")        # "all" = every contiguous window of a career
    chunk_sizes = "all" if chunk_flag == "all" else tuple(int(x) for x in chunk_flag.split(",") if x)
    # experiment 25 (the owner, 2026-09-28): what a chunk row is labelled with.  `career` (the incumbent)
    # gives every chunk row his career label, the chunk's own seasons included, so the label is constant
    # across his rows and nothing that differs between his chunks can be learned from it (experiment 15).
    # `outside` labels each chunk with his RAPM over the seasons OUTSIDE it -- one solve per distinct set
    # of chunk seasons, about 0.7 s each, built per rated season -- and keeps the career row's label.
    chunk_label = _flag("chunk_label", "career")
    assert chunk_label in ("career", "outside", "adjacent"), "--chunk_label=career|outside|adjacent"
    # experiment 30 (the owner, 2026-09-30): `adjacent` replaces the chunks by windows of two, four or six
    # seasons split in the middle, each half labelled by a RAPM fit on the other half alone
    # (`singleyear.adjacent_rows`), at a lighter penalty so one season of label is not shrunk to nothing
    adjacent_penalty = float(_flag("adjacent_penalty", 3000))
    # the BLAS threads the outside labels may use while they are solved (the rest of the run stays at one)
    label_threads = int(_flag("label_threads", os.cpu_count() or 1))
    # the owner, 2026-09-29, after experiment 25: what a chunk row's weight is built on.  `career` (the
    # incumbent) uses the career label's possessions; `label` uses the possessions its OWN label rests on,
    # which for an outside label is the career minus the chunk -- noisier labels, less weight.
    chunk_weight = _flag("chunk_weight", "career")
    assert chunk_weight in ("career", "label"), "--chunk_weight=career|label"
    # experiment 26 (the owner, 2026-09-28): the soft same-team measure on every training row
    # (`singleyear.chunk_rows(shares=...)`), and the value it takes at rating time: 0, "he changed teams".  That is
    # also the only legal value -- the real one needs his other seasons' teams, which ruling 2 forbids --
    # so `--rate_same_team=real` exists for measurement only, and `--same_team_diag=1` writes both priors side
    # by side (outputs/<out>_same_team_diag.parquet) to say how far the choice moves each player.
    rate_same_team = _flag("rate_same_team", "0")
    # `both` (the owner, 2026-10-01: "are you testing the trade flag by turning it on for players who are
    # traded?"): the same models rate every player twice, as if he changed teams (the table this run
    # writes) and as if he stayed (same_team 1, outputs/<out>_stayed.parquet), so the year-over-year test
    # can hand each player the one that matches what he did next (scripts/89_stitch_by_move.py).  That
    # stitched list reads the next season's teams and is a test of the flag only, never a ranking.
    # `1` writes the stayed list as the table.
    assert rate_same_team in ("0", "1", "real", "both"), "--rate_same_team=0|1|real|both"
    same_team_diag = _flag("same_team_diag", "0") not in ("0", "no", "false")
    diag_rows: list = []
    # --dump_rows=NAME writes each side's actual training rows, every column, for every rated season:
    # outputs/prior_rows_NAME_<season>_<side>.parquet (what the Boruta run of experiment 26 reads)
    dump_rows = _flag("dump_rows")
    # --save_models=NAME (the owner, 2026-10-03: the within-season calibrator) pickles each rated season's two
    # fitted priors to outputs/prior_models_NAME.pkl, rewritten after every season, so scripts/97_within_season.py
    # can ask the SAME boosters about a season's box score rebuilt from part of its games.  Plain dicts of the
    # boosters, not the `OutOfPlayerSPM` objects: those are defined in this script, which runs as __main__, and a
    # pickled __main__ class cannot be read back by another script.  Changes no number.
    save_models = _flag("save_models")
    saved_models: dict = {}
    # experiment 27 (the owner, 2026-09-28): one random intercept per team-season around every booster fit
    # (`fit_with_team_season`), the rating from the trees alone
    team_intercept = _flag("team_season_intercept", "0") not in ("0", "no", "false")
    assert not team_intercept or row_shape == "chunks", "--team_season_intercept needs --rows=chunks"
    # the owner, 2026-10-02: the prior as a STACK (`stackprior.StackedSPM`) -- the plus-minus columns
    # (`singleyear.ONC`) into an elastic net, every other column into chimeraboost at `--stack_quality` (5 = a bag
    # of eight), blended on their out-of-player-fold predictions.  Pair with `--features=stack`.
    stack = _flag("stack", "0") not in ("0", "no", "false")
    stack_quality = int(_flag("stack_quality", 5))
    assert not (stack and team_intercept), "--stack and --team_season_intercept do not combine"
    assert not (stack and _flag("dump_shap")), "--dump_shap reads one booster; the stack has two models"
    # the owner, 2026-09-30, after the split by player showed experiments 26-27 carried an age tilt: move each
    # chunk's label to the chunk's own age with an aging curve (`singleyear.AgingCurve`), fit per rated season on
    # the season-to-season changes of single-season RAPM at penalty 100 (outputs/piece_panel.parquet, raw
    # points), never on the seasons the table will be scored on
    age_adjust = _flag("age_adjust_labels", "0") not in ("0", "no", "false")
    assert not age_adjust or row_shape == "chunks", "--age_adjust_labels needs --rows=chunks"
    aging = None
    if age_adjust:
        _pieces = pd.read_parquet(ROOT / "outputs/piece_panel.parquet",
                                  columns=["player_id", "season", "rapm100_off", "rapm100_def", "poss_off", "poss_def"])
        # each side in the sign its label uses: defence as points allowed, so the curve moves the label directly
        _pieces["raw_def"] = -_pieces.rapm100_def
    team_shares = None
    if chunk_label in ("outside", "adjacent") or dump_rows or team_intercept:
        # franchises, not team ids, or everyone who stayed through Charlotte's move looks like a mover
        _roles = pd.read_parquet(ROOT / "data/cache/roles_RSPO.parquet")
        team_shares = _roles[_roles.poss_on > 0][["player_id", "season", "team_id", "poss_on"]].copy()
        team_shares["team_id"] = sy.franchise(team_shares.team_id.to_numpy(), team_shares.season.to_numpy())
    crossfit = _flag("crossfit", "scale")            # 0 | 1 (scale and penalty) | scale (the scale only)
    # adopted 2026-09-14 (the owner: "adopt"): every player's prior from the fit without his rows
    player_folds = int(_flag("player_folds", 5))     # 0 or 1: the plain single fit
    # experiment 1 (2026-09-14): un-shrink the label.  A ridge shrinks a player by n / (n + lambda), so the
    # label's spread grows fifteen-fold from short careers to long ones and the SPM learns "more career
    # possessions = bigger number".  Multiplying by (n + lambda) / n puts every player's label on one scale.
    # `=def` is the SHIPPED setting (adopted 2026-09-18, experiment 22): the defensive label is un-shrunk
    # and the offensive one is not, because the gain and the damage sit on opposite sides.  `=1` does both
    # and is the version rejected on the eye test; `=0` restores the pre-adoption behaviour.
    _unshrink = str(_flag("unshrink_label", "def")).lower()
    unshrink_sides = {"0": (), "no": (), "false": (),
                      "1": ("offense", "defense"), "yes": ("offense", "defense"),
                      "true": ("offense", "defense"), "both": ("offense", "defense"),
                      "off": ("offense",), "offense": ("offense",),
                      "def": ("defense",), "defense": ("defense",)}.get(_unshrink)
    if unshrink_sides is None:
        raise SystemExit(f"--unshrink_label={_unshrink}: write 0, 1, off or def.")
    unshrink = bool(unshrink_sides)
    unshrink_floor = float(_flag("unshrink_floor", 4444))       # see unshrink_label()
    # the owner's rule (2026-09-16): weight each player's training rows by how much TEAM VARIATION sits
    # behind his label, 1 minus the share of his possessions on his most-played team.  The label is one
    # career-pooled RAPM, so its identification rests on the whole career's movement; a player who never
    # left one team teaches the map from the most teammate-collinear evidence there is.  The value given
    # is a FLOOR added to the movement: 0 is the rule exactly and drops one-team players, 0.25 keeps them
    # at a quarter weight.  Absent = off, the shipped behaviour.  `--rows=chunks` only.
    trade_weight = None if _flag("trade_weight") is None else float(_flag("trade_weight"))
    # the owner, 2026-09-30: `--trade_bands=1` rebalances the movement weight within career-length bands, so it
    # moves weight only between players of similar career length (`singleyear.reweight_by_movement(bands=)`)
    trade_bands = _flag("trade_bands", "0") not in ("0", "no", "false")
    # --dump_shap=<name> writes outputs/prior_shap_<name>.parquet: per feature, how far one
    # standard deviation moves the prior, signed, out of player fold.  Comparable with
    # scripts/72_tradeset_shap.py, which reports the same quantity for the correction.
    dump_shap = _flag("dump_shap")
    shap_rows: list = []
    # experiment 2: the ridge's player penalty fixed at one value for every season instead of chosen by
    # cross-validation inside the season (which cannot validate a per-player residual and switches the
    # games off in 2024-2026).  Chosen on the year-over-year test, once.
    # adopted 2026-09-15 (the owner: "adopt"): 13,037 on both sides, every season, from the sweep of
    # 2,000 to 1e9 on the year-over-year test (DECISIONS.md).  --lambda_player=cv restores the per-season CV.
    lambda_player = _flag("lambda_player", "13037")
    lambda_player = None if str(lambda_player).lower() == "cv" else lambda_player
    # the priors are the slow stage (five booster fits a season); save them once, sweep the ridge on them
    save_priors, priors_from = _flag("save_priors"), _flag("priors_from")
    # experiment 12 (2026-09-15, the owner's design): blend the PRIOR toward a replacement level before the
    # ridge sees it -- `prior * w(n) + x * (1 - w(n))`, `w = n^a / (n^a + k^a)`, `n` the possessions on that
    # side.  `--blend_off=x/k/a` and `--blend_def=x/k/a`, each in the raw sign of its own prior; the pooled
    # APM fit of scripts/67_blend_apm.py reads -10.8/64.2/1 on offence and +16.3/12.1/1 on defence.
    def _blend_flag(name):
        raw = _flag(name)
        if not raw:
            return None
        x, k, a = raw.split("/")
        # `a = lin` is the owner's linear ramp, `w = min(n / k, 1)`; anything else is the rational weight
        return float(x), float(k), ("lin" if a.strip().lower() == "lin" else float(a))
    blend = {"O": _blend_flag("blend_off"), "D": _blend_flag("blend_def")}
    # experiment 7: the booster's settings, the same change on both sides.  --params_mult=l2_leaf_reg:5,...
    # multiplies a numeric setting; --params_set=depth:3,... sets one.
    params_mult = {k: float(v) for k, v in (p.split(":") for p in _flag("params_mult", "").split(",") if p)}
    params_set = {k: float(v) for k, v in (p.split(":") for p in _flag("params_set", "").split(",") if p)}

    def tuned(params: dict) -> dict:
        out = dict(params)
        for k, m in params_mult.items():
            out[k] = type(params[k])(params[k] * m)
        for k, v in params_set.items():
            out[k] = type(params[k])(v)
        return out
    assert crossfit in ("0", "1", "scale"), "--crossfit=0|1|scale"
    # experiment 32 (the owner, 2026-10-03: chimeraboost tuning, item 2 of their list): each side's booster
    # settings from outputs/booster_params_<name>.json -- `{"O": {"params": {...}, "early_stop_split": "rows" or
    # "players"}, "D": ...}`, which scripts/94_tune_booster.py writes -- in place of config.yaml's, and whether
    # its early-stopping split holds out whole players (`OutOfPlayerSPM(by_player=)`).  A side the file does not
    # name keeps the shipped settings.  `--params_mult` / `--params_set` still apply on top.
    booster_name = _flag("booster_params")
    booster = (json.loads((ROOT / "outputs" / f"booster_params_{booster_name}.json").read_text())
               if booster_name else {})
    side_params = {"O": booster.get("O", {}).get("params", cfg["gbdt"]["params"]),
                   "D": booster.get("D", {}).get("params", cfg["gbdt"]["params_def"])}
    by_player = {s: booster.get(s, {}).get("early_stop_split", "rows") == "players" for s in ("O", "D")}
    assert not (booster_name and stack), "--booster_params sets the single booster; the stack has its own"
    if booster_name:
        for s in ("O", "D"):
            print(f"booster {s} from booster_params_{booster_name}.json: {side_params[s]}; early-stopping split "
                  f"by {'players' if by_player[s] else 'rows'}", flush=True)

    # The trust boundary (src/eracoef/seasons.py): rows whose unit reaches into a season still
    # being played may not reach a fit.  Gating at the read is what keeps every fit below honest;
    # it drops nothing while no season is in progress, and says so when it does.
    panel, _ = drop_untrainable(pd.read_parquet(ROOT / "outputs/role_panel_season.parquet"),
                                cfg, what="the season panel")
    panel = panel[panel.poss > 0].reset_index(drop=True)
    if age_adjust:
        # the gated panel's ages (its offensive rows; both sides carry the same age) beside the RAPMs
        _ages = panel[panel.side == "O"][["player_id", "season", "age"]]
        aging = _pieces.merge(_ages, on=["player_id", "season"], how="inner")
    feature_set = _flag("features", "boruta")
    features = sy.feature_set(feature_set)
    for side, names_ in features.items():
        assert not any(c.startswith("past_") for c in names_), "single year or bust"
    print(f"targets: offense {names['O']}, defense {names['D']}; features {feature_set} "
          f"(O {len(features['O'])}, D {len(features['D'])}); "
          f"free_prior_scale {FREE_PRIOR_SCALE}; lam_buckets {LAM_BUCKETS or '{}'}; "
          f"exclude_neighbours {exclude_neighbours}; rows {row_shape} (sizes {chunk_sizes}, "
          f"chunk label {chunk_label}, chunk weight {chunk_weight}); crossfit {crossfit}; "
          f"params_mult {params_mult or '{}'}; params_set {params_set or '{}'}; "
          f"player_folds {player_folds}; unshrink_label {'+'.join(unshrink_sides) or 0}; lambda_player {lambda_player or 'CV'}; "
          f"priors_from {priors_from or '-'}; save_priors {save_priors or '-'}; "
          f"blend_off {blend['O'] or '-'}; blend_def {blend['D'] or '-'}", flush=True)
    if lambda_player is not None:
        global BOARD_PLAYER_LAMBDAS, BOARD_CONTEXT_LAMBDAS, BOARD_OFFENSE_LAMBDAS, BOARD_DEFENSE_LAMBDAS
        BOARD_PLAYER_LAMBDAS = np.array([float(lambda_player)])
        BOARD_CONTEXT_LAMBDAS = np.array([0.0])
        # the two sides' penalties separately (2026-09-15, the owner: "same penalty on both sides though?");
        # each defaults to --lambda_player
        BOARD_OFFENSE_LAMBDAS = np.array([float(_flag("lambda_off", lambda_player))])
        BOARD_DEFENSE_LAMBDAS = np.array([float(_flag("lambda_def", lambda_player))])
    saved = pd.read_pickle(ROOT / "outputs" / f"priors_{priors_from}.pkl") if priors_from else None
    to_save: dict = {}

    t0 = time.time()
    # One accumulator per TARGET, because the two sides explain different things.  The designs are NOT
    # held: thirty of them is several gigabytes and the run pages, while `LeaveSeasonOutRAPM` only needs
    # each season's normal equations (a 1,000 x 1,000 gram, ~8 MB).  `ctx.design` caches the last four,
    # and a rebuild is a couple of seconds off the disk cache.
    def design_for(name, season):
        return ctx.design([season], _target(name))

    rapm = {}
    for name in dict.fromkeys(names.values()):
        if saved is not None:
            break                                    # the priors are on disk; no labels, no boosters
        rapm[name] = LeaveSeasonOutRAPM(min_possessions=MIN_POSSESSIONS)
        for s in seasons:
            rapm[name].add_season(s, design_for(name, s))
        print(f"  accumulated {len(seasons)} seasons on {name}, "
              f"{len(rapm[name].player_ids)} players ({time.time() - t0:.0f}s)", flush=True)

    rows, diagnostics = [], []
    rows_of = {"": rows, "stayed": []}          # `--rate_same_team=both`: the second list's rows
    for season in boards:
        prior, table, lam = {}, {}, {}
        # the seasons nothing population-level may be fit on: the rated one, plus its neighbours when
        # those are the seasons this table is going to be scored on
        unseen = [s for s in range(season - exclude_neighbours, season + exclude_neighbours + 1)]
        labels_by_target: dict = {}
        models, held_frames, model_feats_of = {}, {}, {}
        prior_stayed, stayed_frames = {}, {}       # `--rate_same_team=both`: the second list
        for side, params in (("O", tuned(side_params["O"])), ("D", tuned(side_params["D"]))):
            column = "offense" if side == "O" else "defense"
            # `same_team` belongs to a (row, label) pair, not to the panel: it is built by `chunk_rows` and
            # appended to the model's inputs the way the chunk features are
            use_same_team = sy.SAME_TEAM in features[side]
            feats = [f for f in features[side] if f != sy.SAME_TEAM]
            assert not use_same_team or (row_shape == "chunks" and chunk_label in ("outside", "adjacent")), \
                "same_team needs --rows=chunks --chunk_label=outside: a career label cannot vary with the team"
            assert rate_same_team in ("0", "real") or use_same_team, \
                "--rate_same_team=1|both needs same_team among the features"
            if saved is not None:
                prior[side] = saved[season][side]
                continue
            training = panel[(panel.side == side) & ~panel.season.isin(unseen)]

            def label(exclude, method="solve"):
                out = rapm[names[side]].ratings(
                    held_out_season=exclude, offense_lambda=RAPM_OFFENSE_LAMBDA,
                    defense_lambda=RAPM_DEFENSE_LAMBDA, context_lambda=RAPM_CONTEXT_LAMBDA, method=method)
                if column in unshrink_sides:
                    out = unshrink_label(out, RAPM_OFFENSE_LAMBDA, RAPM_DEFENSE_LAMBDA, unshrink_floor,
                                         sides=(column,))
                return out

            model_feats = list(feats)
            if column in unshrink_sides and season == boards[0]:
                lab = label(unseen)
                print(f"  prior {side}: un-shrunk label, tier levels {lab.attrs.get(f'tier_level_{column}')} "
                      f"per 100 for tiers {TIER_EDGES[:-1]}+ label possessions; label sd {lab[column].std():.3f}",
                      flush=True)
            if row_shape == "player":
                train = sy.prior_rows(label(unseen), training, column, feats)
            elif row_shape == "chunks":
                # the owner's design: the career row plus contiguous chunks of his seasons, with two
                # features saying how much evidence each row rests on
                sizes = (range(1, int(training.groupby("player_id").season.nunique().max()) + 1)
                         if chunk_sizes == "all" else chunk_sizes)
                outside = None
                if chunk_label == "outside":
                    # one label per distinct set of chunk seasons, the rated season(s) left out as well;
                    # the same RAPM --rows=season fits for a season, so a 1-season chunk of season s carries
                    # that shape's label for s.  Per side: the defensive label is un-shrunk.  About 300 of
                    # them a season and side, not the 84 first guessed: a player who skipped seasons makes
                    # chunk sets nobody else has.  So they are solved by Cholesky with BLAS allowed
                    # `--label_threads` cores for the duration (0.6 s a solve, against 5.0 s on the one
                    # pinned thread); nothing else runs meanwhile, and the limit is restored before the
                    # booster.  The career label above is untouched: same solver, one thread, as shipped.
                    t_label = time.time()
                    keys = list(dict.fromkeys(tuple(sorted(set(unseen) | set(chunk)))
                                              for chunk in sy.chunk_season_sets(training, sizes)))
                    with threadpool_limits(limits=label_threads, user_api="blas"):
                        outside = {key: label(list(key), method="cholesky") for key in keys}
                    if season == boards[0] or season == boards[-1]:
                        print(f"  prior {side}: {len(outside)} outside labels, one per set of chunk seasons "
                              f"({time.time() - t_label:.0f}s on {label_threads} BLAS threads)", flush=True)
                    if season == boards[0]:
                        # the fast solver against the shipped one, on one of these very labels
                        slow = label(list(keys[0])).set_index("player_id")[column]
                        fast = outside[keys[0]].set_index("player_id")[column]
                        print(f"  prior {side}: outside label by Cholesky vs the shipped solver, largest "
                              f"difference {float((fast - slow.reindex(fast.index)).abs().max()):.1e} per 100",
                              flush=True)
                # the seasons the career label is fit on, and no others: what `same_team` compares against
                shares = None if team_shares is None else team_shares[~team_shares.season.isin(unseen)]
                curve = scale_of = None
                if age_adjust:
                    value, poss_col = ("rapm100_off", "poss_off") if side == "O" else ("raw_def", "poss_def")
                    curve = sy.AgingCurve(aging, value, poss=poss_col, exclude=unseen, degree=2)
                    lam_side = RAPM_OFFENSE_LAMBDA if side == "O" else RAPM_DEFENSE_LAMBDA
                    floor_s = unshrink_floor / (unshrink_floor + lam_side)

                    def scale_of(n, _lam=lam_side, _floor=floor_s, _un=column in unshrink_sides):
                        # the label's own scale: a ridge label is his impact x n / (n + penalty); an un-shrunk
                        # one is that divided back out, down to the floor it is capped at
                        s = np.asarray(n, dtype=float) / (np.asarray(n, dtype=float) + _lam)
                        return np.minimum(1.0, s / _floor) if _un else s
                    if season == boards[0] or season == boards[-1]:
                        at = [20, 23, 25, 27, 30, 33, 36]
                        print(f"  prior {side}: aging curve from {curve.pairs:,} season-to-season pairs, points per 100 "
                              f"against age 27 in the label's sign: "
                              + ", ".join(f"{a} {float(curve(a)):+.2f}" for a in at), flush=True)
                if chunk_label == "adjacent":
                    # experiment 30: each half of a window labelled by a RAPM fit on the other half's seasons
                    # alone, at `--adjacent_penalty`; the defensive side un-shrunk as the career label is
                    def window_label(half, _side=side, _column=column):
                        out = rapm[names[_side]].ratings(
                            held_out_season=[s for s in seasons if s not in half], offense_lambda=adjacent_penalty,
                            defense_lambda=adjacent_penalty, context_lambda=RAPM_CONTEXT_LAMBDA, method="cholesky")
                        if _column in unshrink_sides:
                            out = unshrink_label(out, adjacent_penalty, adjacent_penalty, unshrink_floor,
                                                 sides=(_column,))
                        return out
                    t_label = time.time()
                    halves = sy.adjacent_season_sets(training, sizes)
                    with threadpool_limits(limits=label_threads, user_api="blas"):
                        beside = {half: window_label(half) for half in halves}
                    train = sy.adjacent_rows(label(unseen), training, column, beside, feats, sizes=sizes,
                                             unseen=unseen, shares=shares)
                    if season == boards[0] or season == boards[-1]:
                        win = train[train.row_kind == "window"]
                        traded = float((win[sy.SAME_TEAM] < 0.5).mean()) if sy.SAME_TEAM in win else float("nan")
                        print(f"  prior {side}: {len(beside)} labels fit on one half of a window each, penalty "
                              f"{adjacent_penalty:,.0f} ({time.time() - t_label:.0f}s); {len(win):,} window rows, "
                              f"{traded:.1%} of them traded examples (same team under 0.5); label sd "
                              f"{win.target.std():.3f} against the career rows' "
                              f"{train[train.row_kind == 'career'].target.std():.3f}", flush=True)
                else:
                    train = sy.chunk_rows(label(unseen), training, column, feats, sizes=sizes,
                                          labels=outside, unseen=unseen, weight_by=chunk_weight, shares=shares,
                                          age_curve=curve, label_scale=scale_of)
                if age_adjust and season == boards[0]:
                    moved = train[train.label_key != sy._key_text(sy._season_key(unseen))]
                    bins = pd.cut(moved.age, [0, 24, 27, 30, 33, 99], right=False,
                                  labels=["under 24", "24-26", "27-29", "30-32", "33+"])
                    by_age = moved.groupby(bins, observed=True).age_shift.mean().round(3).to_dict()
                    print(f"  prior {side}: mean move of a chunk's label by the chunk's age {by_age}; "
                          f"median size {moved.age_shift.abs().median():.3f} per 100", flush=True)
                model_feats = feats + sy.CHUNK_FEATURES + ([sy.SAME_TEAM] if use_same_team else [])
                if season == boards[0]:
                    whole = train.label_key.to_numpy() == sy._key_text(sy._season_key(unseen))
                    print(f"  prior {side}: training weight on career rows {train.weight[whole].sum():,.0f}, "
                          f"on chunk rows {train.weight[~whole].sum():,.0f} (chunk weight: {chunk_weight}); "
                          f"median label possessions, career rows {np.median(train.label_possessions[whole]):,.0f}, "
                          f"chunk rows {np.median(train.label_possessions[~whole]):,.0f}", flush=True)
                    # career labels make every row's label key the career one, so `whole` is every row and
                    # there is nothing to describe (`--dump_rows` with career labels crashed here)
                    if sy.SAME_TEAM in train.columns and (~whole).any():
                        st = train[sy.SAME_TEAM].to_numpy()[~whole]
                        print(f"  prior {side}: same_team on the {st.size:,} chunk rows: exactly 0 {np.mean(st == 0):.1%}, "
                              f"exactly 1 {np.mean(st == 1):.1%}, quartiles "
                              f"{np.round(np.quantile(st, [0.25, 0.5, 0.75]), 3).tolist()}; "
                              f"career rows all 1: {bool((train[sy.SAME_TEAM].to_numpy()[whole] == 1).all())}",
                              flush=True)
                if trade_weight is not None:
                    # the owner's rule (2026-09-16): weight a player's rows by the chance that two
                    # possessions of his career came from different teams, over the seasons his label was
                    # fitted on.  The label is one career-pooled RAPM, so how well it is identified
                    # depends on the team variation behind all of it, and this is that quantity.
                    played = roles_played[~roles_played.season.isin(unseen)]
                    per_team = played.groupby(["player_id", "team_id"], as_index=False).poss_on.sum()
                    moved = sy.team_movement(per_team, min_poss=MIN_POSSESSIONS)
                    before = train.weight.to_numpy(float).copy()
                    train = sy.reweight_by_movement(train, moved, floor=trade_weight,
                                                    bands=True if trade_bands else None)
                    if season == boards[0]:
                        kept = (moved.reindex(train.index.unique()).fillna(0.0) + trade_weight) > 0
                        m_row = moved.reindex(train.index).fillna(0.0).to_numpy()
                        band = np.digitize(train.possessions.to_numpy(float), sy.CAREER_BANDS[1:-1])
                        shares = [f"{a:,.0f}+: {before[band == b].sum() / before.sum():.3f} -> "
                                  f"{train.weight.to_numpy()[band == b].sum() / before.sum():.3f}"
                                  for b, a in enumerate(sy.CAREER_BANDS[:-1])]
                        print(f"  prior {side}: weighted by team movement, floor {trade_weight:g}"
                              f"{', rebalanced within career-length bands' if trade_bands else ''}; "
                              f"mean movement {moved.mean():.3f}; {int((~kept).sum())} of {len(kept)} players "
                              f"left at zero weight; one-team players' share of the weight "
                              f"{before[m_row == 0].sum() / before.sum():.3f} -> "
                              f"{train.weight.to_numpy()[m_row == 0].sum() / before.sum():.3f}; "
                              f"by label possessions {shares}", flush=True)
            else:
                # one label per TRAINING season: the RAPM with the rated season(s) and that season out, so
                # a row's box score and its label share no game.  One solve each, ~0.7 s, cached per target
                if names[side] not in labels_by_target:
                    labels_by_target[names[side]] = {s: label(unseen + [s])
                                                     for s in seasons if s not in unseen}
                train = sy.season_rows(labels_by_target[names[side]], training, column, feats,
                                       cap_per_player=(row_shape == "season_capped"))
            if season == boards[0]:
                print(f"  prior {side}: {len(train):,} training rows ({row_shape})", flush=True)
            if dump_rows:
                path = ROOT / "outputs" / f"prior_rows_{dump_rows}_{season}_{side}.parquet"
                train.reset_index().to_parquet(path, index=False)
                print(f"  prior {side}: wrote {path.name} ({len(train):,} rows, {train.shape[1]} columns)", flush=True)
            if stack:
                model = StackedSPM(params, player_folds, quality=stack_quality).fit(train, model_feats)
                print(f"  prior {side} {season}: {model.describe()}", flush=True)
            else:
                model = OutOfPlayerSPM(params, player_folds, intercept=team_intercept,
                                       by_player=by_player[side]).fit(train, model_feats)
            if team_intercept and (season == boards[0] or season == boards[-1]):
                log = model.log_
                print(f"  prior {side}: team-season intercept over {len(np.unique(train[sy.TEAM_SEASON]))} "
                      f"team-seasons: variance ratio {log['ratio']:.3g} (after the refit {log['ratio_refit']:.3g}; "
                      f"infinity = no team-season signal), intercepts' sd {log['intercept_sd']:.3f} per 100; "
                      f"the fold fits' ratios {[round(g['ratio'], 3) for g in model.fold_logs_]}", flush=True)
            if dump_shap:
                shap_rows.append(prior_shap_slopes(model, train, model_feats, side, season))
            if player_folds > 1 and season == boards[0]:
                print(f"  prior {side}: {player_folds} player folds balanced on the label; training-mean "
                      f"shift per held-out fold {np.round(model.shift_, 4).tolist()} per 100 "
                      f"(label sd {train.target.std():.3f})", flush=True)
            held = sy.season_frame(panel[(panel.side == side) & (panel.season == season)], feats)
            held = held.assign(chunk_poss=held.poss.to_numpy(float), chunk_seasons=1.0)
            if use_same_team:
                # "rate him as if he changed teams": 0 for everyone.  The real value -- his rated season's
                # teams against those of the seasons his career label is fit on -- is ruled out for the rating
                # (ruling 2) and computed only to measure what the choice does
                real = None
                if rate_same_team == "real" or same_team_diag:
                    his = team_shares[team_shares.season == season].rename(columns={"player_id": "key"})
                    rest = team_shares[~team_shares.season.isin(unseen)].rename(columns={"player_id": "key"})
                    real = (sy.harmonic_overlap(his, rest).reindex(held.player_id.to_numpy())
                            .fillna(0.0).to_numpy(float))
                held[sy.SAME_TEAM] = (real if rate_same_team == "real" else
                                     1.0 if rate_same_team == "1" else 0.0)
                if rate_same_team == "both":
                    stayed_frames[side] = held.assign(**{sy.SAME_TEAM: 1.0})
                if same_team_diag:
                    as_zero = model.predict(held.assign(**{sy.SAME_TEAM: 0.0}), model_feats)
                    as_real = model.predict(held.assign(**{sy.SAME_TEAM: real}), model_feats)
                    diag_rows.append(pd.DataFrame({"season": season, "side": side,
                                                   "player_id": held.player_id.to_numpy(),
                                                   "poss": held.poss.to_numpy(float), "same_team_real": real,
                                                   "prior_changed_teams": as_zero, "prior_real_teams": as_real}))
            prior[side] = dict(zip(held.player_id.to_numpy(), model.predict(held, model_feats)))
            models[side], held_frames[side], model_feats_of[side] = model, held, model_feats
            if save_models:
                assert isinstance(model, OutOfPlayerSPM), "--save_models saves the single booster, not the stack"
                saved_models.setdefault(season, {})[side] = dict(
                    full=model.full_, folds=model.fold_models_, excluded=model.excluded_, params=model.params,
                    n_folds=model.n_folds, intercept=model.intercept, by_player=model.by_player,
                    feats=list(feats), model_feats=list(model_feats), unseen=list(unseen),
                    # the prior this run handed the ridge, for the reader to check its rebuild against
                    prior=dict(prior[side]))
            if side in stayed_frames:
                frame = stayed_frames[side]
                prior_stayed[side] = dict(zip(frame.player_id.to_numpy(), model.predict(frame, model_feats)))

        # one ridge per list of priors: the table this run writes and, with `--rate_same_team=both`, the same
        # models' priors at same_team 1 ("he stayed") ridged on the same games into the second list
        variants = {"": (prior, held_frames)}
        if rate_same_team == "both":
            assert saved is None, "--rate_same_team=both needs the boosters, not --priors_from"
            variants["stayed"] = (prior_stayed, stayed_frames)
        for variant, (prior_v, frames_v) in variants.items():
            # the blend, before anything downstream sees the prior.  The cross-fitted fold priors get the
            # SAME transform below: the free scale is priced on those columns, so a fold prior that skipped
            # the blend would price the scale on a different prior from the one that ships.
            # Always, not only when a blend asks: the board carries a defensive possession count and it
            # has to be the defensive one.  The design is already built and cached, so this is a groupby.
            off_poss, def_poss = side_possessions(design_for(names["O"], season))
            poss_side = {"O": off_poss, "D": def_poss}
            if any(blend.values()):
                for side in ("O", "D"):
                    if blend[side]:
                        prior_v[side] = blend_prior(prior_v[side], poss_side[side], *blend[side])

            fold_prior = None
            if saved is not None and crossfit != "0":
                fold_prior = _saved_fold_prior(saved[season]["folds"], design_for(names["O"], season))
            elif crossfit != "0":
                wd_o, wd_d = design_for("xpts_ft", season), design_for("x3def", season)
                # the pieces are rebuilt per fold when a model reads them, which needs each row's two teams
                reads_pieces = any(c in model_feats_of[s] for s in model_feats_of for c in sy.PIECES)
                fold_prior = _fold_prior_builder(wd_o, wd_d, models, frames_v, model_feats_of,
                                                 teams=_box_teams(season) if reads_pieces else None)
            if fold_prior is not None and any(blend.values()):
                inner = fold_prior

                def fold_prior(train_mask, _inner=inner, _poss=poss_side):
                    po, pd_ = _inner(train_mask)
                    if blend["O"]:
                        po = blend_prior(po, _poss["O"], *blend["O"])
                    if blend["D"]:
                        pd_ = blend_prior(pd_, _poss["D"], *blend["D"])
                    return po, pd_

            if save_priors and not variant:
                # the full priors and, per game fold the ridge will use, the priors rebuilt without that fold
                fold = game_folds(design_for(names["O"], season), 5, 0)
                folds = [] if fold is None or fold_prior is None else [fold_prior(fold != f) for f in range(5)]
                to_save[season] = {"O": prior_v["O"], "D": prior_v["D"], "folds": folds}
                pd.to_pickle(to_save, ROOT / "outputs" / f"priors_{save_priors}.pkl")

            # one ridge per side's target; each contributes only its own half of the board
            scale = {}
            # one fit per DISTINCT target: when both sides explain the same thing (--target_off=pts
            # --target_def=pts) the second fit would be the first one over again
            fits = {name: _ridge(design_for(name, season), prior_v, fold_prior=fold_prior,
                                 crossfit_penalty=(crossfit == "1"))
                    for name in dict.fromkeys(names.values())}
            for side in ("O", "D"):
                ridge = fits[names[side]]
                table[side] = ridge.ratings_
                lam[side] = (ridge.offense_lambda_, ridge.defense_lambda_, ridge.context_lambda_)
                # how far the season's own games decided to trust the prior on this side.  Above 1 means the
                # prior was compressed and the games stretched it; 1.0 exactly means the lever is off.
                scale[side] = (ridge.prior_scale_[0 if side == "O" else 1]
                               if ridge.prior_scale_ is not None else 1.0)
            merged = (table["O"][["player_id", "offense", "prior_offense", "possessions"]]
                      .merge(table["D"][["player_id", "defense", "prior_defense"]], on="player_id"))
            # the DEFENSIVE possession count, not a copy of the offensive one (side_possessions)
            merged["poss_def"] = merged.player_id.map(def_poss).fillna(0.0).to_numpy(float)
            # the team-season intercept's own numbers, per season and side, when it is on (experiment 27)
            logs = {side: getattr(models.get(side), "log_", None) for side in ("O", "D")}
            extra = {}
            if logs["O"] is not None and logs["D"] is not None:
                extra = dict(group_ratio_off=logs["O"]["ratio"], group_ratio_def=logs["D"]["ratio"],
                             intercept_sd_off=logs["O"]["intercept_sd"], intercept_sd_def=logs["D"]["intercept_sd"])
            rows_of[variant].append(merged.assign(season=season, offense_lambda=lam["O"][0],
                                      defense_lambda=lam["D"][1], context_lambda=lam["O"][2],
                                      prior_scale_off=scale["O"], prior_scale_def=scale["D"], **extra))
            print(f"  {season}{f' ({variant})' if variant else ''}: {len(merged)} players, lambdas O {lam['O'][0]:,.0f} / "
                  f"D {lam['D'][1]:,.0f} / ctx {lam['O'][2]:,.0f},  prior_scale "
                  f"{scale['O']:.2f} / {scale['D']:.2f}"
                  + (f",  team-season ratio {extra['group_ratio_off']:.3g} / {extra['group_ratio_def']:.3g}" if extra else "")
                  + f"  ({time.time() - t0:.0f}s)", flush=True)

        if do_score:
            diagnostics.append(_diagnose(season, design_for, names, prior))
            d = diagnostics[-1]
            print(f"      75/25 on points: game_armse {d['game_armse']:.4f} (base {d['base_armse']:.4f})"
                  f"  scale {d['scale_off']:.2f} / {d['scale_def']:.2f}  miss {d['miss']:.3f}", flush=True)
        pd.concat(rows, ignore_index=True).to_parquet(out, index=False)
        if rows_of["stayed"]:
            pd.concat(rows_of["stayed"], ignore_index=True).to_parquet(out_stayed, index=False)
        if save_models:
            saved_models["meta"] = dict(argv=list(sys.argv), exclude_neighbours=exclude_neighbours,
                                        features=feature_set, rows=row_shape, chunk_label=chunk_label,
                                        unshrink=list(unshrink_sides), player_folds=player_folds,
                                        booster_params=booster_name, crossfit=crossfit, out=out.name)
            pd.to_pickle(saved_models, ROOT / "outputs" / f"prior_models_{save_models}.pkl")

    if dump_shap and shap_rows:
        out_shap = ROOT / "outputs" / f"prior_shap_{dump_shap}.parquet"
        pd.concat(shap_rows, ignore_index=True).to_parquet(out_shap, index=False)
        print(f"wrote {out_shap.name}: prior feature slopes, {len(shap_rows)} fits")
    if diag_rows:
        diag = pd.concat(diag_rows, ignore_index=True)
        out_diag = out.with_name(out.stem + "_same_team_diag.parquet")
        diag.to_parquet(out_diag, index=False)
        moved = (diag.prior_real_teams - diag.prior_changed_teams).abs()
        print(f"wrote {out_diag.name}: the SPM prior rated as if every player changed teams against the same "
              f"prior at his real same-team value, before the ridge: median move {moved.median():.3f} per 100, "
              f"ninth decile {moved.quantile(0.9):.3f}, {np.mean(moved > 0.1):.1%} past 0.1")

    def finish(rows_, out_):
        """One list of ridge rows to the published table: centred, positive-good, named, written."""
        board = pd.concat(rows_, ignore_index=True)
        # The owner, 2026-09-14: "players' values are not centered well ... guys are positive that should be
        # pushed down".  The SPM prior's possession-weighted mean is not zero (the label's mean is positive for
        # the heavy-minute players who dominate a season's possessions, and the free scale multiplies it:
        # +1.47 on offence in 2026), and nothing downstream re-centres it.  So: per season and side, the
        # possession-weighted mean of the rating is zero -- the average possession is played by a zero
        # player, the RAPM convention.  A level shift only: the year-over-year test refits the level and the
        # consensus checks are rank- and spread-based, so neither moves.  --centre=0 keeps the raw level.
        if _flag("centre", "1") not in ("0", "no", "false"):
            for side in ("offense", "defense"):
                level = (board.groupby("season").apply(
                    lambda g, c=side: np.average(g[c], weights=g.possessions), include_groups=False)
                         .reindex(board.season).to_numpy())
                board[side] = board[side] - level
                board[f"prior_{side}"] = board[f"prior_{side}"] - level
        # 52_site.py's schema: raw sign in, positive-good out, one row per player-season
        board = board.rename(columns={"possessions": "poss_off"})
        board["poss_season"] = board.poss_off
        board["rating_off"] = board.offense
        board["rating_def"] = -board.defense                 # positive good on both ends
        board["rating_total"] = board.rating_off + board.rating_def
        board["prior_off"] = board.prior_offense
        board["prior_def"] = -board.prior_defense
        board["prior_total"] = board.prior_off + board.prior_def
        board["u_off"] = board.rating_off - board.prior_off
        board["u_def"] = board.rating_def - board.prior_def
        board["u_total"] = board.u_off + board.u_def
        # `player_name`, because both scripts/52_site.py and tests/test_vs_consensus.py key on it and a board
        # of bare player_ids silently fails them both
        board = board.merge(_names(), on="player_id", how="left")
        board.to_parquet(out_, index=False)
        print(f"\nwrote {out_}: {len(board)} rows, {board.season.nunique()} seasons "
              f"({time.time() - t0:.0f}s)")
        return board

    board = finish(rows, out)
    if rows_of["stayed"]:
        stayed = finish(rows_of["stayed"], out_stayed)
        top = stayed[stayed.season == boards[-1]].sort_values("rating_total", ascending=False)
        print(f"\n=== {boards[-1]}, top 20 of the second list, every player rated as if he stayed")
        print(top[["player_name", "rating_off", "rating_def", "rating_total",
                   "poss_off"]].head(20).to_string(index=False))

    pd.set_option("display.width", 200, "display.max_columns", 20, "display.precision", 3)
    if diagnostics:
        D = pd.DataFrame(diagnostics)
        D.to_parquet(out.with_name(out.stem + "_score.parquet"), index=False)
        print("\n=== 75/25 within-season diagnostic, scored on actual points "
              "(game_armse decides; miss breaks ties)")
        print(D.round(4).to_string(index=False))
        print(f"\npooled: game_armse {D.game_armse.mean():.4f}  base {D.base_armse.mean():.4f}  "
              f"scale_off {D.scale_off.mean():.3f}  scale_def {D.scale_def.mean():.3f}  "
              f"miss {D['miss'].mean():.3f}")

    wide = board[board.poss_off >= 1000]
    print("\n=== board spread, players with 1,000+ possessions (the shipped board reads 1.317 / 1.434)")
    print(f"  rating_off sd {wide.rating_off.std():.3f}   rating_def sd {wide.rating_def.std():.3f}")
    print(f"  prior_off  sd {wide.prior_off.std():.3f}   prior_def  sd {wide.prior_def.std():.3f}")
    print(f"  u_off      sd {wide.u_off.std():.3f}   u_def      sd {wide.u_def.std():.3f}"
          "        <- what the season's own games added")
    if "prior_scale_off" in board:
        by_season = board.groupby("season")[["prior_scale_off", "prior_scale_def"]].first()
        print(f"  prior_scale: offense {by_season.prior_scale_off.mean():.2f} "
              f"(min {by_season.prior_scale_off.min():.2f}, max {by_season.prior_scale_off.max():.2f}), "
              f"defense {by_season.prior_scale_def.mean():.2f} "
              f"(min {by_season.prior_scale_def.min():.2f}, max {by_season.prior_scale_def.max():.2f})")

    # the owner reads the top 20 of the latest season for EVERY experiment (2026-09-14): the eye test
    for season in dict.fromkeys([boards[-1], 2015 if 2015 in boards else boards[0]]):
        top = board[board.season == season].sort_values("rating_total", ascending=False)
        print(f"\n=== {season}, top 20 (points per 100 possessions, positive good on both ends)")
        print(top[["player_name", "rating_off", "rating_def", "rating_total",
                   "poss_off"]].head(20).to_string(index=False))


_D83 = None


def _box_teams(season: int) -> pd.Series:
    """(game_id, player_id) -> team id for a season, from the box scores: scripts/83_decompose_site.py's lookup,
    the one the season panel's pieces were built with (scripts/86_context_panel.py)."""
    global _D83
    if _D83 is None:
        import importlib.util
        spec = importlib.util.spec_from_file_location("_borrowed_83", ROOT / "scripts" / "83_decompose_site.py")
        _D83 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_D83)
    return _D83.box_teams(season)


def _names() -> pd.DataFrame:
    return pd.read_parquet(ROOT / "artifacts/season_ratings.parquet",
                           columns=["player_id", "player_name"]).drop_duplicates("player_id")


def _diagnose(season, design_for, names, prior) -> dict:
    """Refit each side on the first 75% of its own target, score the last 25% on ACTUAL POINTS.

    The two sides come from two fits, so the board scored here is assembled the way the shipped one is:
    offense out of the offensive target's ridge, defense out of the defensive one's.
    """
    fits = {}
    for name in dict.fromkeys(names.values()):
        fit_games, _ = _first_75(design_for(name, season))
        fits[name] = _ridge(fit_games, prior).as_ratings_frame()
    frame = (fits[names["O"]][["player_id", "o", "poss", "prior_o"]]
             .merge(fits[names["D"]][["player_id", "d", "prior_d"]], on="player_id"))
    _, score_games = _first_75(design_for(SCORE_TARGET, season))
    result = score(predict_season(Ratings(frame), score_games, level="home"))
    return dict(season=season, game_armse=armse(result["tg"]), base_armse=armse(result["tg_base"]),
                scale_off=result["scale_off"], scale_def=result["scale_def"],
                calib_side=result["calib_side"],
                miss=calibration_miss(result["scale_off"], result["scale_def"]),
                sd_o=float(frame.o.std()), sd_d=float(frame.d.std()))


if __name__ == "__main__":
    main()
