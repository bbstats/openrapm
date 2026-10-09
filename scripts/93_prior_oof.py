"""How well each candidate prior predicts players it never saw, once the luck it shares with its label is out.

    python scripts/93_prior_oof.py [--rows=oofcheck] [--season=2026] [--sides=O,D] [--quality=3] [--players=0]

HANDOFF.md's open thread, option (a) (the owner, 2026-10-02: "A").  Experiment 31's stack logged an out-of-fold
error far below either of its halves (offence 0.545 against 0.637 and 0.636; defence 0.953 against 1.035 and
1.407), on rows whose inputs and label come from the same possessions: a chunk row's on-court plus-minus and its
career label share the chunk's games, so luck on twos, turnovers and rebounds sits in both.  And the stack was
never set against the booster that ships.

Four priors, each fitted the way scripts/62 fits one -- five player folds balanced on the label, every row
predicted by the fit that never saw its player:

  linear     ElasticNetCV on `stackprior.PLUS_MINUS`, on- and off-court (8 columns)       experiment 31's half
  booster    chimeraboost `quality=3`, no bag, on `STACK_BOOSTER_<side>` + the chunk features
                                                                                        experiment 31's half
  stack      the two blended by a non-negative line, the blend itself cross-fitted on the same folds
  incumbent  chimeraboost at the shipped settings (the offence bag of five) on `BORUTA_<side>` + the chunk
             features

on two training sets cut from ONE dump, so the rows and weights are the same in both:

  as built   every row labelled with his career RAPM, the label the incumbent trains on.  Its "all rows"
             error is the number experiment 31 logged, up to the chunks the dump drops (below).
  clean      the chunk rows alone, each labelled with his RAPM over the seasons OUTSIDE the chunk, so no
             row's inputs share a game with its label.  The career row goes: `--chunk_label=outside` keeps
             the career label on it.

The dump is `62 --chunk_label=outside --features=stack --exclude_neighbours=1 --boards=<season>
--dump_rows=<rows>`.  Outside labels drop the chunks of a player with no evidence outside them, so "as built"
here is experiment 31's training set less those rows.

Error is the weighted rmse against the row's own label (the training weights), on all rows, chunk rows and
one-season chunk rows -- the shape of the rated season's own row.  On "as built" the chunk rows are also scored
against their OUTSIDE label ("vs outside label"): the priors trained the way the build trains them, judged on a
label that shares no game with the row's inputs.  `label sd` is the error of predicting every row at the
weighted mean.  Each prior is set against the incumbent player by player: z is the mean of the
per-player difference in weighted squared error over its standard error, players taken as independent.
"stack, blend in sample" is the blend experiment 31 logged: fitted once on every out-of-fold prediction and
scored on the same.

Writes outputs/csv/prior_oof_<rows>_<season>.csv and every row's out-of-fold prediction to
outputs/prior_oof_<rows>_<season>_preds.parquet.  Reads only; builds no rankings.
"""
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
from eracoef import singleyear as sy  # noqa: E402
from eracoef.config import load_config  # noqa: E402
from eracoef.stackprior import PLUS_MINUS, _Linear  # noqa: E402

N_FOLDS = 5
L1_RATIOS = (0.1, 0.5, 0.9, 1.0)                 # StackedSPM's
MODELS = ["linear", "booster", "stack", "incumbent"]
INSAMPLE = "stack, blend in sample"


def _row_weight(frame: pd.DataFrame) -> np.ndarray:
    """Each training row's sample weight.  Dumps written before 2026-10-08 carry it in `weight`, over the body-weight
    input (DECISIONS.md, "The Robustness pass"): in them that column IS the sample weight, and body weight is lost."""
    column = sy.ROW_WEIGHT if sy.ROW_WEIGHT in frame.columns else "weight"
    return frame[column].to_numpy(float)


def training_sets(frame: pd.DataFrame) -> dict:
    """{"as built": ..., "clean": ...}, each indexed by player_id with `target`, `weight` and `career_row`.

    `chunk_rows` puts the career rows first, one per player, so a player's first row is his career row; with
    outside labels it is also the only row whose label key is the career one, which is checked.
    """
    career = ~frame.player_id.duplicated().to_numpy()
    keys = frame.label_key.to_numpy()
    if len(set(keys)) == 1:
        raise SystemExit("every row carries the career label: dump with --chunk_label=outside")
    career_key = set(keys[career])
    assert len(career_key) == 1 and not np.isin(keys[~career], list(career_key)).any(), \
        "the career rows are not the first row of each player"
    frame = frame.assign(career_row=career).set_index("player_id")
    label = frame.target[career]
    # the dumped label stays beside the career one, so the priors as built can be scored without shared games
    built = frame.assign(target=label.reindex(frame.index).to_numpy(float), outside_target=frame.target.to_numpy(float))
    return {"as built": built, "clean": frame[~career]}


def fit_out_of_fold(train: pd.DataFrame, side: str, cfg: dict, quality: int) -> tuple:
    """Every row's prediction from the fits that never saw its player, for the four priors."""
    from chimeraboost import ChimeraBoostRegressor
    from sklearn.linear_model import LinearRegression

    shipped = dict(cfg["gbdt"]["params" if side == "O" else "params_def"])
    # StackedSPM's booster: the shipped settings with the bag replaced by `quality`, never forking
    stacked = {k: v for k, v in shipped.items() if k != "n_ensembles"} | {"quality": quality, "ensemble_n_jobs": 1}
    cols = {"linear": list(PLUS_MINUS),
            "booster": (sy.STACK_BOOSTER_O if side == "O" else sy.STACK_BOOSTER_D) + sy.CHUNK_FEATURES,
            "incumbent": (sy.BORUTA_O if side == "O" else sy.BORUTA_D) + sy.CHUNK_FEATURES}
    X = {k: train[v].to_numpy(float) for k, v in cols.items()}
    y, w, groups = train.target.to_numpy(float), _row_weight(train), train.index.to_numpy()
    fold = sy.stratified_player_folds(train, N_FOLDS)
    pred = {k: np.full(len(train), np.nan) for k in MODELS}
    for f in range(N_FOLDS):
        fit, held = fold != f, fold == f
        t = time.time()
        pred["linear"][held] = (_Linear(L1_RATIOS).fit(X["linear"][fit], y[fit], w[fit], groups[fit])
                                .predict(X["linear"][held]))
        pred["booster"][held] = (ChimeraBoostRegressor(random_state=0, **stacked)
                                 .fit(X["booster"][fit], y[fit], sample_weight=w[fit]).predict(X["booster"][held]))
        pred["incumbent"][held] = (ChimeraBoostRegressor(random_state=0, **shipped)
                                   .fit(X["incumbent"][fit], y[fit], sample_weight=w[fit])
                                   .predict(X["incumbent"][held]))
        print(f"    fold {f + 1}/{N_FOLDS} ({time.time() - t:.0f}s)", flush=True)
    halves = np.column_stack([pred["linear"], pred["booster"]])
    for f in range(N_FOLDS):
        fit, held = fold != f, fold == f
        pred["stack"][held] = (LinearRegression(positive=True).fit(halves[fit], y[fit], sample_weight=w[fit])
                               .predict(halves[held]))
    whole = LinearRegression(positive=True).fit(halves, y, sample_weight=w)
    pred[INSAMPLE] = whole.predict(halves)
    return pred, whole


def score(train: pd.DataFrame, pred: dict) -> list:
    """Weighted rmse per prior and subset, and each prior against the incumbent player by player.

    On "as built" the chunk rows are scored twice: against the career label they were trained on, and against
    their outside label ("vs outside label"), which shares no game with the row's inputs."""
    y, w = train.target.to_numpy(float), _row_weight(train)
    career = train.career_row.to_numpy(bool)
    one = ~career & (train.chunk_seasons.to_numpy(float) == 1)
    subsets = {"all rows": (y, np.ones(len(train), bool))} if career.any() else {}
    subsets |= {"chunk rows": (y, ~career), "one-season chunk rows": (y, one)}
    if "outside_target" in train:
        outside = train.outside_target.to_numpy(float)
        subsets |= {"chunk rows vs outside label": (outside, ~career),
                    "one-season chunk rows vs outside label": (outside, one)}
    players = train.index.to_numpy()
    out = []
    for subset, (label, m) in subsets.items():
        yy, ww = label[m], w[m]
        mean = np.average(yy, weights=ww)
        label_sd = float(np.sqrt(np.average((yy - mean) ** 2, weights=ww)))
        e2_ref = (yy - pred["incumbent"][m]) ** 2
        for name, p in pred.items():
            e2 = (yy - p[m]) ** 2
            per_player = pd.Series(ww * (e2 - e2_ref)).groupby(players[m]).sum().to_numpy()
            se = per_player.std(ddof=1) / np.sqrt(per_player.size)
            out.append(dict(subset=subset, model=name, rmse=float(np.sqrt(np.average(e2, weights=ww))),
                            label_sd=label_sd, rows=int(m.sum()), players=int(per_player.size),
                            z_vs_incumbent=float(per_player.mean() / se) if name != "incumbent" and se > 0 else np.nan))
    return out


def main() -> None:
    check_flags()
    tag, season = flag("rows", "oofcheck"), int(flag("season", 2026))
    sides = flag("sides", "O,D").split(",")
    quality = int(flag("quality", 3))
    n_players = int(flag("players", 0))          # a random sample of players: a quick check of the script
    cfg = load_config(ROOT / "config.yaml")
    results, preds, started = [], [], time.time()
    for side in sides:
        frame = pd.read_parquet(ROOT / "outputs" / f"prior_rows_{tag}_{season}_{side}.parquet")
        if n_players:
            keep = pd.Series(frame.player_id.unique()).sample(n_players, random_state=0)
            frame = frame[frame.player_id.isin(keep)].reset_index(drop=True)
        for name, train in training_sets(frame).items():
            print(f"=== {side}, {name}: {len(train):,} rows, {train.index.nunique():,} players "
                  f"({time.time() - started:.0f}s)", flush=True)
            pred, blend = fit_out_of_fold(train, side, cfg, quality)
            a, (w_lin, w_boost) = blend.intercept_, blend.coef_
            print(f"  blend on every out-of-fold prediction: {a:+.3f} + {w_lin:.3f} x linear + {w_boost:.3f} x "
                  f"booster", flush=True)
            part = pd.DataFrame(score(train, pred)).assign(side=side, training_set=name)
            results.append(part)
            keep = ["career_row", "chunk_seasons", "chunk_poss", "target", "weight"] + (
                ["outside_target"] if "outside_target" in train else [])
            preds.append(train[keep].reset_index().assign(side=side, training_set=name,
                                                          **{f"pred_{k}": v for k, v in pred.items()}))
            with pd.option_context("display.width", 200):
                print("  weighted rmse against the row's label (label sd = predicting everyone at the mean):")
                rmse = part.pivot(index="model", columns="subset", values="rmse")
                rmse.loc["label sd"] = part.groupby("subset").label_sd.first()
                print(rmse.reindex(MODELS + [INSAMPLE, "label sd"], columns=part.subset.unique())
                      .round(3).to_string())
                print("  z against the incumbent, player by player (below zero = smaller error than the incumbent):")
                z = part.pivot(index="model", columns="subset", values="z_vs_incumbent")
                print(z.reindex([m for m in MODELS + [INSAMPLE] if m != "incumbent"], columns=part.subset.unique())
                      .round(1).to_string(), flush=True)
    out = pd.concat(results, ignore_index=True)
    stem = f"prior_oof_{tag}_{season}{f'_sample{n_players}' if n_players else ''}"
    path = ROOT / "outputs" / "csv" / f"{stem}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    # every row's out-of-fold prediction, so a new way of scoring them needs no refit
    pd.concat(preds, ignore_index=True).to_parquet(ROOT / "outputs" / f"{stem}_preds.parquet", index=False)
    print(f"wrote {path.relative_to(ROOT)} and outputs/{stem}_preds.parquet ({time.time() - started:.0f}s)")


if __name__ == "__main__":
    main()
