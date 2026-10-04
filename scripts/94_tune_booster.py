"""Tune the shipped booster's settings, then its bag size, on the prior screen -- no rankings built.

    python scripts/94_tune_booster.py [--rows=oofcheck] [--season=2026] [--sides=O,D] [--trials=60] [--tag=tune1]
                                      [--split=both|players|rows] [--players=0]

`--split=players` fixes the early-stopping split to whole players on every trial.  **The owner's ruling, 2026-10-03:
one holdout method on both sides -- "we are not going to use different holdout methods for statistical reasons that
are 100% identical".**  tune1 searched the split (`both`) and came out rows on offence, players on defence; tune2
fixes it to players and adds the patience (`early_stopping_rounds`) to the search, the learning rate down to 0.01.

The owner, 2026-10-03 ("a"): item 2 of their list, chimeraboost hyperparameter tuning, then item 4, bagging --
every setting screened before any build, and only what beats the shipped booster built.

**The screen** is scripts/93's, on the same dump (`62 --chunk_label=outside --features=stack --exclude_neighbours=1
--boards=2026 --dump_rows=oofcheck`): the booster on the SHIPPED lists (`BORUTA_<side>` + the chunk features),
trained the way the build trains it (every row on its career label), five player folds balanced on the label, every
row predicted by the fit that never saw its player; scored on the one-season rows -- the shape of the rated season's
own row -- against their OUTSIDE label, which shares no game with the row's inputs.

**The score is the error after rescaling.**  The build fits the prior's scale on the season's games (`scale x prior
+ residual`, then centred), so a prior is judged by its order: each prediction is first mapped onto the label by its
own best line (weighted least squares, intercept and slope), and the score is the weighted rmse left over.  Raw rmse
would reward a narrow prior, and the outside label is narrower than the career label it was trained on (it rests on
fewer possessions, so it is shrunk harder).  The raw rmse and the correlation are printed beside it.

**What is searched** (Optuna TPE, `--trials` per side, the shipped settings enqueued first): depth, learning rate,
the two leaf penalties, row and column sampling, histogram bins, linear leaves and their penalty, cross features, and
how the early-stopping split is drawn.  That last one is the only setting here that is not a number.  Without
`groups`, chimeraboost holds out a random 20% of ROWS to pick its tree count (and to run its linear-leaf and
cross-feature races), so a player's career row and chunks -- one career label -- sit on both sides of the split and
the held-out error rewards remembering players; the prior is only ever asked about players its fit never saw.
`early_stop_split=players` passes `groups=player_id`, which holds out whole players.  The search fits single models;
the bag is item 4, below.

**Then, on two fold splits** (the search's, and a second one balanced the same way but dealt differently, so a
setting cannot win on one lucky split), five candidates against the shipped booster as shipped (offence's bag of
five, defence single, rows split):

  shipped, players split    the shipped settings with only the early-stopping split changed -- one change
  tuned                     the search's best single model
  tuned, bag of 5 / 8       the same, bagged (with the players split, each member draws whole players)

z is the per-player paired difference in the rescaled squared error over its standard error.

Writes outputs/csv/tune_<tag>_trials.csv (every trial), outputs/csv/tune_<tag>_final.csv (the candidates, both
splits) and outputs/booster_params_<tag>.json (the tuned settings per side, for a build).
"""
import importlib
import json
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

screen93 = importlib.import_module("93_prior_oof")          # its training sets, so the rows are the same

N_FOLDS = 5
SEARCHED = ["depth", "learning_rate", "l2_leaf_reg", "min_child_weight", "subsample", "colsample", "max_bins",
            "linear_leaves", "linear_lambda", "cross_features", "early_stopping_rounds"]


def second_folds(train: pd.DataFrame, n_folds: int, seed: int) -> np.ndarray:
    """A second player split, balanced on the label like `stratified_player_folds`: players sorted by their weighted
    mean label and each run of `n_folds` neighbours dealt to the folds in a random order, not the snake."""
    y, w, g = train.target.to_numpy(float), train.weight.to_numpy(float), train.index.to_numpy()
    keys, inv = np.unique(g, return_inverse=True)
    mean = np.bincount(inv, w * y, keys.size) / np.maximum(np.bincount(inv, w, keys.size), 1e-12)
    order = np.argsort(mean, kind="stable")
    rng = np.random.default_rng(seed)
    fold_of = np.empty(keys.size, dtype=int)
    for start in range(0, keys.size, n_folds):
        block = order[start:start + n_folds]
        fold_of[block] = rng.permutation(n_folds)[:block.size]
    return fold_of[inv]


class Screen:
    """One side's training rows: out-of-player-fold predictions, scored on the one-season rows."""

    def __init__(self, train: pd.DataFrame, feats: list):
        self.X = train[feats].to_numpy(float)
        self.y = train.target.to_numpy(float)                 # career labels: what the build trains on
        self.w = train.weight.to_numpy(float)
        self.players = train.index.to_numpy()
        self.rows = ~train.career_row.to_numpy(bool) & (train.chunk_seasons.to_numpy(float) == 1)
        self.label = train.outside_target.to_numpy(float)[self.rows]
        self.weight = self.w[self.rows]

    def predict(self, params: dict, by_player: bool, fold: np.ndarray) -> np.ndarray:
        from chimeraboost import ChimeraBoostRegressor
        pred = np.full(len(self.y), np.nan)
        for f in range(N_FOLDS):
            fit, held = fold != f, fold == f
            model = ChimeraBoostRegressor(random_state=0, **params)
            model.fit(self.X[fit], self.y[fit], sample_weight=self.w[fit],
                      groups=self.players[fit] if by_player else None)
            pred[held] = model.predict(self.X[held])
        return pred

    def residual(self, pred: np.ndarray) -> np.ndarray:
        """The one-season rows' label minus the prediction mapped onto it by its own best line."""
        x, y, w = pred[self.rows], self.label, self.weight
        design = np.column_stack([np.ones_like(x), x]) * np.sqrt(w)[:, None]
        coef = np.linalg.lstsq(design, y * np.sqrt(w), rcond=None)[0]
        return y - (coef[0] + coef[1] * x)

    def score(self, pred: np.ndarray) -> dict:
        x, y, w = pred[self.rows], self.label, self.weight
        xm, ym = np.average(x, weights=w), np.average(y, weights=w)
        corr = np.average((x - xm) * (y - ym), weights=w) / np.sqrt(
            np.average((x - xm) ** 2, weights=w) * np.average((y - ym) ** 2, weights=w))
        return dict(rescaled=float(np.sqrt(np.average(self.residual(pred) ** 2, weights=w))),
                    raw=float(np.sqrt(np.average((y - x) ** 2, weights=w))), corr=float(corr))

    def z_against(self, pred: np.ndarray, ref: np.ndarray) -> float:
        d = pd.Series(self.weight * (self.residual(pred) ** 2 - self.residual(ref) ** 2)).groupby(
            self.players[self.rows]).sum()
        return float(d.mean() / (d.std(ddof=1) / np.sqrt(d.size)))


def suggest(trial, split: str) -> tuple:
    """One trial's settings.  `split` "rows" or "players" fixes the early-stopping split (the owner, 2026-10-03:
    one holdout method on both sides, whole players); "both" searches it."""
    params = dict(depth=trial.suggest_int("depth", 3, 8),
                  learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                  l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 0.3, 100.0, log=True),
                  min_child_weight=trial.suggest_float("min_child_weight", 0.3, 100.0, log=True),
                  subsample=trial.suggest_float("subsample", 0.5, 1.0),
                  colsample=trial.suggest_float("colsample", 0.3, 1.0),
                  max_bins=trial.suggest_categorical("max_bins", [32, 64, 128]),
                  linear_leaves=trial.suggest_categorical("linear_leaves", [True, False]),
                  linear_lambda=trial.suggest_float("linear_lambda", 0.1, 100.0, log=True),
                  cross_features=trial.suggest_categorical("cross_features", [True, False]),
                  # patience: rounds without improvement before stopping (chimeraboost's default 50).  Added for
                  # tune2: a whole-player holdout is noisier than a row one and may stop the fit too early
                  early_stopping_rounds=trial.suggest_categorical("early_stopping_rounds", [50, 100, 200, 400]))
    if split != "both":
        return params, split == "players"
    return params, trial.suggest_categorical("early_stop_split", ["rows", "players"]) == "players"


def single(shipped: dict) -> dict:
    """The shipped settings as one model: the bag off, `linear_lambda` and `early_stopping_rounds` at
    chimeraboost's defaults spelled out."""
    out = {k: v for k, v in shipped.items() if k not in ("n_ensembles", "ensemble_n_jobs")}
    return {"linear_lambda": 1.0, "early_stopping_rounds": 50, **out}


def main() -> None:
    check_flags()
    tag, season = flag("tag", "tune1"), int(flag("season", 2026))
    rows_tag, sides = flag("rows", "oofcheck"), flag("sides", "O,D").split(",")
    n_trials, n_players = int(flag("trials", 60)), int(flag("players", 0))
    split = flag("split", "both")                  # the early-stopping split: rows | players | both (searched)
    assert split in ("rows", "players", "both"), "--split=rows|players|both"
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    cfg = load_config(ROOT / "config.yaml")
    trials_out, final_out, best_out, started = [], [], {}, time.time()
    for side in sides:
        frame = pd.read_parquet(ROOT / "outputs" / f"prior_rows_{rows_tag}_{season}_{side}.parquet")
        if n_players:
            keep = pd.Series(frame.player_id.unique()).sample(n_players, random_state=0)
            frame = frame[frame.player_id.isin(keep)].reset_index(drop=True)
        train = screen93.training_sets(frame)["as built"]
        feats = (sy.BORUTA_O if side == "O" else sy.BORUTA_D) + sy.CHUNK_FEATURES
        scr = Screen(train, feats)
        folds = {"search split": sy.stratified_player_folds(train, N_FOLDS), "second split": second_folds(train, N_FOLDS, 1)}
        shipped = dict(cfg["gbdt"]["params" if side == "O" else "params_def"])
        print(f"=== {side}: {len(train):,} rows, {train.index.nunique():,} players, {int(scr.rows.sum()):,} one-season "
              f"rows scored; shipped {shipped} ({time.time() - started:.0f}s)", flush=True)
        t = time.time()
        reference = scr.predict(shipped, False, folds["search split"])
        ref_score = scr.score(reference)
        print(f"  shipped as shipped: rescaled {ref_score['rescaled']:.4f}, raw {ref_score['raw']:.4f}, correlation "
              f"{ref_score['corr']:.4f} ({time.time() - t:.0f}s)", flush=True)

        preds = {}

        def objective(trial):
            params, by_player = suggest(trial, split)
            t0 = time.time()
            pred = scr.predict(params, by_player, folds["search split"])
            preds[trial.number] = pred
            s = scr.score(pred)
            for k, v in s.items():
                trial.set_user_attr(k, v)
            trial.set_user_attr("seconds", time.time() - t0)
            print(f"  trial {trial.number:3d}: rescaled {s['rescaled']:.4f} (shipped {ref_score['rescaled']:.4f}), "
                  f"raw {s['raw']:.4f}, corr {s['corr']:.4f}, {time.time() - t0:.0f}s; "
                  f"{'players' if by_player else 'rows'} split, depth {params['depth']}, lr {params['learning_rate']:.3f}, "
                  f"l2 {params['l2_leaf_reg']:.2g}, mcw {params['min_child_weight']:.2g}, linear "
                  f"{params['linear_leaves']}, cross {params['cross_features']}", flush=True)
            return s["rescaled"]

        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=0))
        base = single(shipped)
        if split == "both":
            for how in ("rows", "players"):
                study.enqueue_trial({**{k: base[k] for k in SEARCHED}, "early_stop_split": how})
        else:
            study.enqueue_trial({k: base[k] for k in SEARCHED})
        study.optimize(objective, n_trials=n_trials, catch=(Exception,))
        table = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs", "state"))
        table.columns = [c.replace("params_", "").replace("user_attrs_", "") for c in table.columns]
        trials_out.append(table.assign(side=side))
        done = table[table.state == "COMPLETE"].sort_values("value")
        with pd.option_context("display.width", 250, "display.max_columns", 30):
            print(f"\n  {side}: the ten best of {len(done)} trials (value = rescaled error; shipped as shipped "
                  f"{ref_score['rescaled']:.4f}):")
            print(done.head(10).drop(columns=["state"]).round(4).to_string(index=False), flush=True)
        try:
            importance = optuna.importance.get_param_importances(study)
            print("  what moved the score (Optuna's fANOVA importance): "
                  + ", ".join(f"{k} {v:.2f}" for k, v in importance.items()), flush=True)
        except Exception as exc:                          # fANOVA needs enough completed trials
            print(f"  importance not computed: {exc}", flush=True)

        best = study.best_trial
        tuned = {k: best.params[k] for k in SEARCHED}
        how = best.params.get("early_stop_split", split)
        by_player = how == "players"
        best_out[side] = {"params": tuned, "early_stop_split": how,
                          "screen_rescaled": best.value, "shipped_rescaled": ref_score["rescaled"]}
        candidates = {"shipped": (shipped, False),
                      "shipped, players split": (shipped, True),
                      "tuned": (tuned, by_player),
                      "tuned, bag of 5": ({**tuned, "n_ensembles": 5, "ensemble_n_jobs": 1}, by_player),
                      "tuned, bag of 8": ({**tuned, "n_ensembles": 8, "ensemble_n_jobs": 1}, by_player)}
        print(f"\n  {side}: the candidates on both splits (best trial {best.number}, "
              f"{'players' if by_player else 'rows'} split)", flush=True)
        for split_name, fold in folds.items():
            got = {}
            for name, (params, grouped) in candidates.items():
                t0 = time.time()
                if split_name == "search split" and name == "shipped":
                    got[name] = reference
                elif split_name == "search split" and name == "tuned":
                    got[name] = preds[best.number]
                else:
                    got[name] = scr.predict(params, grouped, fold)
                s = scr.score(got[name])
                z = np.nan if name == "shipped" else scr.z_against(got[name], got["shipped"])
                final_out.append(dict(side=side, split=split_name, candidate=name, **s, z_vs_shipped=z,
                                      seconds=time.time() - t0))
                print(f"    {split_name:12s} {name:24s} rescaled {s['rescaled']:.4f}, raw {s['raw']:.4f}, corr "
                      f"{s['corr']:.4f}, z vs shipped {z:+.1f} ({time.time() - t0:.0f}s)", flush=True)
    out_csv = ROOT / "outputs" / "csv"
    out_csv.mkdir(parents=True, exist_ok=True)
    suffix = f"_sample{n_players}" if n_players else ""
    pd.concat(trials_out, ignore_index=True).to_csv(out_csv / f"tune_{tag}{suffix}_trials.csv", index=False)
    pd.DataFrame(final_out).to_csv(out_csv / f"tune_{tag}{suffix}_final.csv", index=False)
    (ROOT / "outputs" / f"booster_params_{tag}{suffix}.json").write_text(json.dumps(best_out, indent=2))
    print(f"wrote outputs/csv/tune_{tag}{suffix}_trials.csv, tune_{tag}{suffix}_final.csv and "
          f"outputs/booster_params_{tag}{suffix}.json ({time.time() - started:.0f}s)")


if __name__ == "__main__":
    main()
