"""Build the whole board from the single-year pipeline: LOSO RAPM -> SPM -> PriorRidgeCV, season by season.

    python scripts/62_single_year_board.py [--first=1997] [--last=2026] [--out=season_ratings_sy]
                                           [--score=1] [--target_off=xpts_ft] [--target_def=x3def_w0.25]
                                           [--free_scale=1] [--buckets=low_poss:2] [--boards=2015,2024]
                                           [--features=boruta_noonc|boruta|sy_noonc|sy]
                                           [--exclude_neighbours=0] [--rows=chunks|player|season|season_capped]
                                           [--chunk_sizes=1,2,3|all] [--crossfit=scale|0|1]
                                           [--params_mult=l2_leaf_reg:5,min_child_weight:5] [--params_set=depth:3]
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eracoef import singleyear as sy  # noqa: E402
from eracoef.config import load_config
from eracoef.seasons import drop_untrainable  # noqa: E402
from eracoef.holdout import Context, Ratings, predict_season, score  # noqa: E402
from eracoef.inseason import season_frac  # noqa: E402
from eracoef.investigate import offcourt_rates, oncourt_rates  # noqa: E402
from eracoef.looseason import LeaveSeasonOutRAPM  # noqa: E402
from eracoef.priorridge import PriorRidgeCV, armse, calibration_miss, game_folds  # noqa: E402
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


class OutOfPlayerSPM:
    """The SPM fitted once on everything and once per player fold; `predict` gives each player the fit
    that never saw his rows (`--player_folds=N`; 0 = the plain single fit)."""

    def __init__(self, params: dict, n_folds: int):
        self.params, self.n_folds = dict(params), int(n_folds)

    def fit(self, train: pd.DataFrame, model_feats: list) -> "OutOfPlayerSPM":
        X, y, w = train[model_feats].to_numpy(float), train.target.to_numpy(float), train.weight.to_numpy(float)
        self.full_ = ChimeraBoostRegressor(random_state=0, **self.params).fit(X, y, sample_weight=w)
        self.fold_models_, self.excluded_ = [], []
        self.shift_ = np.zeros(0)
        if self.n_folds > 1:
            fold = sy.stratified_player_folds(train, self.n_folds)
            self.shift_ = sy.fold_mean_shift(train, fold)
            for f in range(self.n_folds):
                keep = fold != f
                self.fold_models_.append(ChimeraBoostRegressor(random_state=0, **self.params).fit(
                    X[keep], y[keep], sample_weight=w[keep]))
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


def _fold_prior_builder(wd_o, wd_d, models, held_frames, model_feats):
    """A prior for a CV fold that has NOT seen the fold's games: the season's on-court columns (`onc_*`)
    rebuilt from the training rows alone, the fitted boosters re-asked.  `--crossfit=1`.

    The panel's `onc_o` / `onc_d` are averaged over every game of the season, so a prior column built from
    them already contains the outcome of any game the ridge holds out, and the free prior scale -- a
    least-squares coefficient on that column -- is inflated by it.  `wd_o` / `wd_d` are the season's
    designs on the two targets the panel built `onc_*` from (`xpts_ft`, `x3def`), so the rebuilt columns
    differ from the panel's in the games used and nothing else (`scratch/onc_leak.py` checked that the
    full-season rebuild reproduces the panel to four decimals).
    """
    ids_of_ps = wd_o.spec.ps_table["player_id"].to_numpy()
    # the off-court family too, when the panel carries it (scripts/65_offcourt_panel.py)
    with_offc = all(c in held_frames["O"].columns for c in sy.OFFC + sy.NET)
    rebuilt = sy.ONC + (sy.OFFC + sy.NET if with_offc else [])

    def fold_prior(train_mask):
        fo, fd = wd_o.subset(train_mask), wd_d.subset(train_mask)
        got = oncourt_rates(fo, fd)
        onc = pd.DataFrame({"player_id": ids_of_ps, **{c: got[c].to_numpy(dtype=float) for c in sy.ONC}})
        if with_offc:
            off = offcourt_rates(fo, fd)
            for c in sy.OFFC:
                onc[c] = off[c].to_numpy(dtype=float)
            onc["net_o"], onc["net_d"] = onc.onc_o - onc.offc_o, onc.onc_d - onc.offc_d
        out = {}
        for side in ("O", "D"):
            h = held_frames[side].drop(columns=rebuilt).merge(onc, on="player_id", how="left")
            h[rebuilt] = h[rebuilt].fillna(0.0)
            out[side] = dict(zip(h.player_id.to_numpy(), models[side].predict(h, model_feats[side])))
        return out["O"], out["D"]

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
        roles_played = _roles[_roles.poss_on > 0][["player_id", "season", "team_id", "poss_on"]]
    first, last = int(_flag("first", 1997)), int(_flag("last", 2026))
    out = ROOT / "outputs" / f"{_flag('out', 'season_ratings_sy')}.parquet"
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

    # The trust boundary (src/eracoef/seasons.py): rows whose unit reaches into a season still
    # being played may not reach a fit.  Gating at the read is what keeps every fit below honest;
    # it drops nothing while no season is in progress, and says so when it does.
    panel, _ = drop_untrainable(pd.read_parquet(ROOT / "outputs/role_panel_season.parquet"),
                                cfg, what="the season panel")
    panel = panel[panel.poss > 0].reset_index(drop=True)
    feature_set = _flag("features", "boruta")
    features = sy.feature_set(feature_set)
    for side, names_ in features.items():
        assert not any(c.startswith("past_") for c in names_), "single year or bust"
    print(f"targets: offense {names['O']}, defense {names['D']}; features {feature_set} "
          f"(O {len(features['O'])}, D {len(features['D'])}); "
          f"free_prior_scale {FREE_PRIOR_SCALE}; lam_buckets {LAM_BUCKETS or '{}'}; "
          f"exclude_neighbours {exclude_neighbours}; rows {row_shape} (sizes {chunk_sizes}); crossfit {crossfit}; "
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
    for season in boards:
        prior, table, lam = {}, {}, {}
        # the seasons nothing population-level may be fit on: the rated one, plus its neighbours when
        # those are the seasons this table is going to be scored on
        unseen = [s for s in range(season - exclude_neighbours, season + exclude_neighbours + 1)]
        labels_by_target: dict = {}
        models, held_frames, model_feats_of = {}, {}, {}
        for side, params in (("O", tuned(cfg["gbdt"]["params"])), ("D", tuned(cfg["gbdt"]["params_def"]))):
            column = "offense" if side == "O" else "defense"
            feats = features[side]
            if saved is not None:
                prior[side] = saved[season][side]
                continue
            training = panel[(panel.side == side) & ~panel.season.isin(unseen)]

            def label(exclude):
                out = rapm[names[side]].ratings(
                    held_out_season=exclude, offense_lambda=RAPM_OFFENSE_LAMBDA,
                    defense_lambda=RAPM_DEFENSE_LAMBDA, context_lambda=RAPM_CONTEXT_LAMBDA)
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
                train = sy.chunk_rows(label(unseen), training, column, feats, sizes=sizes)
                model_feats = feats + sy.CHUNK_FEATURES
                if trade_weight is not None:
                    # the owner's rule (2026-09-16): weight a player's rows by the chance that two
                    # possessions of his career came from different teams, over the seasons his label was
                    # fitted on.  The label is one career-pooled RAPM, so how well it is identified
                    # depends on the team variation behind all of it, and this is that quantity.
                    played = roles_played[~roles_played.season.isin(unseen)]
                    per_team = played.groupby(["player_id", "team_id"], as_index=False).poss_on.sum()
                    moved = sy.team_movement(per_team, min_poss=MIN_POSSESSIONS)
                    train = sy.reweight_by_movement(train, moved, floor=trade_weight)
                    if season == boards[0]:
                        kept = (moved.reindex(train.index.unique()).fillna(0.0) + trade_weight) > 0
                        print(f"  prior {side}: weighted by team movement, floor {trade_weight:g}; "
                              f"mean movement {moved.mean():.3f}, "
                              f"{int((~kept).sum())} of {len(kept)} players left at zero weight",
                              flush=True)
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
            model = OutOfPlayerSPM(params, player_folds).fit(train, model_feats)
            if dump_shap:
                shap_rows.append(prior_shap_slopes(model, train, model_feats, side, season))
            if player_folds > 1 and season == boards[0]:
                print(f"  prior {side}: {player_folds} player folds balanced on the label; training-mean "
                      f"shift per held-out fold {np.round(model.shift_, 4).tolist()} per 100 "
                      f"(label sd {train.target.std():.3f})", flush=True)
            held = sy.season_frame(panel[(panel.side == side) & (panel.season == season)], feats)
            held = held.assign(chunk_poss=held.poss.to_numpy(float), chunk_seasons=1.0)
            prior[side] = dict(zip(held.player_id.to_numpy(), model.predict(held, model_feats)))
            models[side], held_frames[side], model_feats_of[side] = model, held, model_feats

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
                    prior[side] = blend_prior(prior[side], poss_side[side], *blend[side])

        fold_prior = None
        if saved is not None and crossfit != "0":
            fold_prior = _saved_fold_prior(saved[season]["folds"], design_for(names["O"], season))
        elif crossfit != "0":
            wd_o, wd_d = design_for("xpts_ft", season), design_for("x3def", season)
            fold_prior = _fold_prior_builder(wd_o, wd_d, models, held_frames, model_feats_of)
        if fold_prior is not None and any(blend.values()):
            inner = fold_prior

            def fold_prior(train_mask, _inner=inner, _poss=poss_side):
                po, pd_ = _inner(train_mask)
                if blend["O"]:
                    po = blend_prior(po, _poss["O"], *blend["O"])
                if blend["D"]:
                    pd_ = blend_prior(pd_, _poss["D"], *blend["D"])
                return po, pd_

        if save_priors:
            # the full priors and, per game fold the ridge will use, the priors rebuilt without that fold
            fold = game_folds(design_for(names["O"], season), 5, 0)
            folds = [] if fold is None or fold_prior is None else [fold_prior(fold != f) for f in range(5)]
            to_save[season] = {"O": prior["O"], "D": prior["D"], "folds": folds}
            pd.to_pickle(to_save, ROOT / "outputs" / f"priors_{save_priors}.pkl")

        # one ridge per side's target; each contributes only its own half of the board
        scale = {}
        # one fit per DISTINCT target: when both sides explain the same thing (--target_off=pts
        # --target_def=pts) the second fit would be the first one over again
        fits = {name: _ridge(design_for(name, season), prior, fold_prior=fold_prior,
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
        rows.append(merged.assign(season=season, offense_lambda=lam["O"][0],
                                  defense_lambda=lam["D"][1], context_lambda=lam["O"][2],
                                  prior_scale_off=scale["O"], prior_scale_def=scale["D"]))
        print(f"  {season}: {len(merged)} players, lambdas O {lam['O'][0]:,.0f} / "
              f"D {lam['D'][1]:,.0f} / ctx {lam['O'][2]:,.0f},  prior_scale "
              f"{scale['O']:.2f} / {scale['D']:.2f}  ({time.time() - t0:.0f}s)", flush=True)

        if do_score:
            diagnostics.append(_diagnose(season, design_for, names, prior))
            d = diagnostics[-1]
            print(f"      75/25 on points: game_armse {d['game_armse']:.4f} (base {d['base_armse']:.4f})"
                  f"  scale {d['scale_off']:.2f} / {d['scale_def']:.2f}  miss {d['miss']:.3f}", flush=True)
        pd.concat(rows, ignore_index=True).to_parquet(out, index=False)

    if dump_shap and shap_rows:
        out_shap = ROOT / "outputs" / f"prior_shap_{dump_shap}.parquet"
        pd.concat(shap_rows, ignore_index=True).to_parquet(out_shap, index=False)
        print(f"wrote {out_shap.name}: prior feature slopes, {len(shap_rows)} fits")

    board = pd.concat(rows, ignore_index=True)
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
    board.to_parquet(out, index=False)
    print(f"\nwrote {out}: {len(board)} rows, {board.season.nunique()} seasons "
          f"({time.time() - t0:.0f}s)")

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
