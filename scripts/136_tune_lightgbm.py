"""Tune the deterministic LightGBM prior (`eracoef.lgbprior`) on the prior screen, against the shipped chimeraboost.

    python scripts/136_tune_lightgbm.py [--rows=oofcheck] [--seasons=2026,2015] [--sides=O,D] [--trials=150]
                                        [--jobs=10] [--tag=lgb1] [--linear=search|on|off] [--wide=0|1]

The owner, 2026-10-08 ("go! (lightgbm mode)"): LightGBM in place of chimeraboost, then leave one player out.  This is
step 2, the settings, screened before any build.

**The screen is scripts/94's** (its `Screen`, on the same dumps: `62 --chunk_label=outside --features=stack
--exclude_neighbours=1 --boards=<season> --dump_rows=oofcheck`): every row on its career label as the build trains
it, five player folds balanced on the label, every row predicted by the fit that never saw its player, scored on the
one-season rows against their OUTSIDE label (no shared games), after mapping each prediction onto the label by its
own best line (the build fits the prior's scale, so a prior is judged by its order).  Two seasons instead of 94's
one, so a setting cannot win on one season's quirks: the objective is the mean over seasons of the rescaled error
divided by the shipped chimeraboost's on the same folds (below 1 = better than shipped).

**What is searched** (Optuna TPE, `--trials` per side): learning rate, tree count, leaves, depth, minimum rows per
leaf, the two leaf penalties, histogram bins, path smoothing, linear leaves and their penalty.  Never searched: row
or column subsampling, which `lgbprior.FIXED` switches off so the model is deterministic.  Each trial's ten fits
(two seasons x five folds) run in `--jobs` processes.

**Then** the best trial against the shipped chimeraboost on both fold splits of both seasons (the search's, and a
second one balanced the same way but dealt differently), with z = the per-player paired difference in the rescaled
squared error over its standard error.

Writes outputs/csv/tune_<tag>_trials.csv, tune_<tag>_final.csv and outputs/booster_params_<tag>.json, the file
`62 --booster_params=<tag>` reads (`"learner": "lightgbm"` per side).
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
from eracoef import lgbprior  # noqa: E402
from eracoef import singleyear as sy  # noqa: E402
from eracoef.config import load_config  # noqa: E402

screen93 = importlib.import_module("93_prior_oof")
tune94 = importlib.import_module("94_tune_booster")

N_FOLDS = 5


def _fold_fit(params: dict, X, y, w, fold, f):
    fit = fold != f
    model = lgbprior.fit(params, X[fit], y[fit], w[fit])
    return f, model.predict(X[fold == f])


def lgb_predictions(jobs: dict, params: dict, n_jobs: int) -> dict:
    """{(season, split): out-of-player-fold predictions}, every fold of every job fitted in parallel."""
    from joblib import Parallel, delayed
    tasks = [(key, f) for key in jobs for f in range(N_FOLDS)]
    got = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_fold_fit)(params, jobs[key][0].X, jobs[key][0].y, jobs[key][0].w, jobs[key][1], f)
        for key, f in tasks)
    out = {key: np.full(len(jobs[key][0].y), np.nan) for key in jobs}
    for (key, _), (f, pred) in zip(tasks, got):
        out[key][jobs[key][1] == f] = pred
    return out


def suggest(trial, linear: str = "search", wide: bool = False) -> dict:
    """One trial's settings.  `linear`: search | off | on.  `wide` (the rematch, 2026-10-08: lgb1 settled on the
    edge of two ranges, 250-300 rows a leaf and 31 bins) reaches 1,000 rows a leaf and 15 bins."""
    params = dict(learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                  n_estimators=trial.suggest_int("n_estimators", 100, 2000, log=True),
                  num_leaves=trial.suggest_int("num_leaves", 4, 64, log=True),
                  max_depth=trial.suggest_categorical("max_depth", [-1, 3, 4, 5, 6, 8]),
                  min_child_samples=trial.suggest_int("min_child_samples", 5, 1000 if wide else 300, log=True),
                  reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 100.0, log=True),
                  reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
                  max_bin=trial.suggest_categorical("max_bin", [15, 31, 63, 127, 255] if wide else [31, 63, 127, 255]),
                  path_smooth=trial.suggest_float("path_smooth", 1e-3, 100.0, log=True),
                  linear_tree=(trial.suggest_categorical("linear_tree", [False, True]) if linear == "search"
                               else linear == "on"))
    if params["linear_tree"]:
        params["linear_lambda"] = trial.suggest_float("linear_lambda", 1e-3, 100.0, log=True)
    return params


def main() -> None:
    check_flags()
    tag, rows_tag = flag("tag", "lgb1"), flag("rows", "oofcheck")
    seasons = [int(s) for s in flag("seasons", "2026,2015").split(",") if s]
    sides, n_trials, n_jobs = flag("sides", "O,D").split(","), int(flag("trials", 150)), int(flag("jobs", 10))
    # the rematch (the owner, 2026-10-08): linear leaves off -- with them, a 590-possession rookie's defensive prior
    # doubled (Zach Edey 2.2 -> 4.1); without them 2.7 -- and the two ranges lgb1 hit the edge of widened
    linear = flag("linear", "search")
    assert linear in ("search", "on", "off"), "--linear=search|on|off"
    wide = flag("wide", "0") not in ("0", "no", "false")
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    cfg = load_config(ROOT / "config.yaml")
    trials_out, final_out, best_out, started = [], [], {}, time.time()
    for side in sides:
        feats = (sy.BORUTA_O if side == "O" else sy.BORUTA_D) + sy.CHUNK_FEATURES
        shipped = dict(cfg["gbdt"]["params" if side == "O" else "params_def"])
        screens, folds = {}, {}
        for season in seasons:
            frame = pd.read_parquet(ROOT / "outputs" / f"prior_rows_{rows_tag}_{season}_{side}.parquet")
            train = screen93.training_sets(frame)["as built"]
            screens[season] = tune94.Screen(train, feats)
            folds[(season, "search split")] = sy.stratified_player_folds(train, N_FOLDS)
            folds[(season, "second split")] = tune94.second_folds(train, N_FOLDS, 1)
            print(f"=== {side} {season}: {len(train):,} rows, {train.index.nunique():,} players, "
                  f"{int(screens[season].rows.sum()):,} one-season rows scored", flush=True)
        # the shipped chimeraboost on every split, sequentially (its numba pool must not be run in parallel)
        reference = {}
        for (season, split), fold in folds.items():
            t = time.time()
            reference[(season, split)] = screens[season].predict(shipped, False, fold)
            s = screens[season].score(reference[(season, split)])
            print(f"  shipped chimeraboost, {season} {split}: rescaled {s['rescaled']:.4f}, raw {s['raw']:.4f}, "
                  f"corr {s['corr']:.4f} ({time.time() - t:.0f}s)", flush=True)
        search = {season: (screens[season], folds[(season, "search split")]) for season in seasons}
        ref_rescaled = {season: screens[season].score(reference[(season, "search split")])["rescaled"]
                        for season in seasons}

        def objective(trial):
            params = suggest(trial, linear, wide)
            t0 = time.time()
            preds = lgb_predictions(search, params, n_jobs)
            ratios = {}
            for season in seasons:
                s = screens[season].score(preds[season])
                ratios[season] = s["rescaled"] / ref_rescaled[season]
                for k, v in s.items():
                    trial.set_user_attr(f"{k}_{season}", v)
            value = float(np.mean(list(ratios.values())))
            trial.set_user_attr("seconds", time.time() - t0)
            print(f"  trial {trial.number:3d}: {value:.4f} of shipped ("
                  + ", ".join(f"{s} {r:.4f}" for s, r in ratios.items()) + f"), {time.time() - t0:.1f}s; "
                  f"lr {params['learning_rate']:.3f}, trees {params['n_estimators']}, leaves {params['num_leaves']}, "
                  f"depth {params['max_depth']}, min rows {params['min_child_samples']}, linear {params['linear_tree']}",
                  flush=True)
            return value

        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=0))
        start = dict(learning_rate=0.05, n_estimators=400, num_leaves=15, max_depth=4, min_child_samples=20,
                     reg_lambda=10.0, reg_alpha=0.001, max_bin=127, path_smooth=0.001)
        study.enqueue_trial({**start, "linear_tree": False} if linear == "search" else start)
        study.optimize(objective, n_trials=n_trials, catch=(Exception,))
        table = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs", "state"))
        table.columns = [c.replace("params_", "").replace("user_attrs_", "") for c in table.columns]
        trials_out.append(table.assign(side=side))
        done = table[table.state == "COMPLETE"].sort_values("value")
        with pd.option_context("display.width", 250, "display.max_columns", 40):
            print(f"\n  {side}: the ten best of {len(done)} trials (value = rescaled error as a share of shipped):")
            print(done.head(10).drop(columns=["state"]).round(4).to_string(index=False), flush=True)
        try:
            importance = optuna.importance.get_param_importances(study)
            print("  what moved the score: " + ", ".join(f"{k} {v:.2f}" for k, v in importance.items()), flush=True)
        except Exception as exc:
            print(f"  importance not computed: {exc}", flush=True)

        best = study.best_trial
        tuned = dict(best.params)
        best_out[side] = {"learner": "lightgbm", "params": tuned, "screen_value": best.value,
                          "screen_seasons": seasons}
        print(f"\n  {side}: tuned LightGBM (trial {best.number}) against shipped chimeraboost, every split", flush=True)
        jobs = {key: (screens[key[0]], fold) for key, fold in folds.items()}
        got = lgb_predictions(jobs, tuned, n_jobs)
        for key in folds:
            scr = screens[key[0]]
            s, r = scr.score(got[key]), scr.score(reference[key])
            z = scr.z_against(got[key], reference[key])
            final_out.append(dict(side=side, season=key[0], split=key[1], lgb_rescaled=s["rescaled"],
                                  shipped_rescaled=r["rescaled"], lgb_corr=s["corr"], shipped_corr=r["corr"],
                                  z_vs_shipped=z))
            print(f"    {key[0]} {key[1]:12s} LightGBM rescaled {s['rescaled']:.4f} vs shipped {r['rescaled']:.4f} "
                  f"(corr {s['corr']:.4f} vs {r['corr']:.4f}), z {z:+.1f}", flush=True)
    out_csv = ROOT / "outputs" / "csv"
    out_csv.mkdir(parents=True, exist_ok=True)
    pd.concat(trials_out, ignore_index=True).to_csv(out_csv / f"tune_{tag}_trials.csv", index=False)
    pd.DataFrame(final_out).to_csv(out_csv / f"tune_{tag}_final.csv", index=False)
    (ROOT / "outputs" / f"booster_params_{tag}.json").write_text(json.dumps(best_out, indent=2))
    print(f"wrote outputs/csv/tune_{tag}_trials.csv, tune_{tag}_final.csv and outputs/booster_params_{tag}.json "
          f"({time.time() - started:.0f}s)")


if __name__ == "__main__":
    main()
