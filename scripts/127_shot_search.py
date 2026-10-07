"""The shot-quality search: trials of every model family on the SEARCH blocks, scored by SALL (eracoef.shotsearch).

    python scripts/127_shot_search.py --family=placebo                      # the current model on other row samples
    python scripts/127_shot_search.py --family=glm --trials=60              # the regression's options (optuna)
    python scripts/127_shot_search.py --family=feat --blocks_add=late,ato   # add-one feature arms on the current model
    python scripts/127_shot_search.py --family=xgb --trials=40 --base=L7    # boosters on top of a base regression
    python scripts/127_shot_search.py --family=one --params='{...}' --name=G1  # one named configuration

Every trial: for each search block, fit on the block's allowed training seasons (the first `--rows` of each season's
fixed permutation, closeness-weighted), price the block's regular-season attempts (heaves out) with the shooter and
arena terms zeroed, relevel from the season (shotsearch.relevel_fast) and score SALL.  The objective is the mean
per-attempt SALL minus trial 0's (the current model on the same rows) over the 15 search seasons; optuna studies live
in outputs/shotsearch/optuna.db and resume.  Per trial: outputs/shotsearch/trials.parquet (season x sub-model rows:
SALL, raw log loss, n, params, family, trial) and data/shotsearch/<family>/<trial>.parquet (the logits, for stacking).
No confirm season is ever loaded for scoring (shotsearch.assert_phase).
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "10")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "10")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _cli import check_flags, flag  # noqa: E402
from eracoef import shotmodel as sm  # noqa: E402
from eracoef import shotsearch as ss  # noqa: E402
from eracoef.config import load_config  # noqa: E402

FULL = ["spot", "start", "putback", "clock", "context", "fatigue", "prev"]
L7 = dict(blocks=FULL, ridge=1.0, lam_shooter=50.0, lam_arena=50.0, rounds=4)
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "shotsearch"
STORE = ROOT / "data" / "shotsearch"
_TABLES: dict = {}


def table(season: int) -> pd.DataFrame:
    if season not in _TABLES:
        t = pd.read_parquet(ROOT / "data" / "shotq" / "_table" / f"{season}.parquet")
        from eracoef import shotfeatures
        # on the WHOLE season, before heaves go: the previous attempt of a possession is another row
        t = shotfeatures.prepare(t, require_blocked=True)
        _TABLES[season] = t[~t["heave"]].reset_index(drop=True)
    return _TABLES[season]


def block_data(block, rows: int, seed: int, hl_past=None, hl_future=None, half_life=5.0, phase: str = "search"):
    """(train, weights, holdout mask, score) for one block of the given phase (search, or confirm for scripts/131).
    With either half-life given the weights are 1: shotmodel.fit applies its own past/future closeness for the block
    (never twice)."""
    rated = list(range(block[0], block[1] + 1))
    ss.assert_phase(rated, phase)
    allowed = sm.train_seasons(rated, list(range(1997, 2027)))
    per = ss.block_rows(len(allowed), rows)
    parts = []
    for s in allowed:
        t = table(s)
        idx = ss.row_perm(len(t), s, seed)[:per]
        parts.append(t.iloc[np.sort(idx)])
    train = pd.concat(parts, ignore_index=True)
    sw = train["season"].to_numpy()
    if hl_past is not None or hl_future is not None:
        w = np.ones(len(train))
    else:
        w = sm.closeness_weights(sw, rated, half_life=half_life)
    near = sorted(allowed, key=lambda s: min(abs(s - block[0]), abs(s - block[1])))[:2]
    holdout = np.isin(sw, near)
    score = pd.concat([table(s) for s in rated], ignore_index=True)
    score = score[score["phase"] == "RS"].reset_index(drop=True)
    return train, w, holdout, score


def evaluate(eta_raw: np.ndarray, score: pd.DataFrame) -> pd.DataFrame:
    rs = np.ones(len(score), bool)
    lev = ss.relevel_fast(eta_raw, score["season"].to_numpy(), ss.band_codes(score), score["made"].to_numpy(), rs)
    return ss.sall_table(lev, score)


PENALTIES = ("ridge", "lam_shooter", "lam_arena", "ridge_int", "era_lambda")


def fit_glm(train, w, params, block=None):
    """shotmodel.fit with the penalties entered per 1.5M rows and scaled to the rows actually fitted, so a setting
    found on the search's 750k rows means the same prior strength in a 1.5M build."""
    import inspect
    p = {k: v for k, v in params.items() if k in inspect.signature(sm.fit).parameters and k not in ("train", "weights")}
    scale = len(train) / ss.FULL_ROWS
    for k in PENALTIES:
        if k in p and p[k] is not None:
            p[k] = p[k] * scale
    blocks = p.pop("blocks", FULL)
    if block is not None and "block" in inspect.signature(sm.fit).parameters:
        p["block"] = tuple(block)
    return sm.fit(train, blocks, weights=w, **p)


def base_model(name: str, block, train, w, params, rows: int = 0, seed: int = 0):
    """The base regression of a block, cached on disk (boosters reuse it trial after trial), keyed on its rows."""
    path = STORE / "_base" / f"{name}_{block[0]}_rows{rows}_seed{seed}.pkl"
    if path.exists():
        with open(path, "rb") as fh:
            return pickle.load(fh)
    m = fit_glm(train, w, params, block=range(block[0], block[1] + 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(m, fh)
    return m


def run_trial(family: str, params: dict, rows: int, seed: int = 0, base_name: str = "L7", base_params=None,
              report=None, phase: str = "search"):
    """Fit and score one configuration on every block of the phase; returns (table, logits frame)."""
    tabs, logits = [], []
    for i, block in enumerate(ss.SEARCH_BLOCKS if phase == "search" else ss.CONFIRM_BLOCKS):
        train, w, holdout, score = block_data(block, rows, seed, params.get("half_life_past"),
                                              params.get("half_life_future"), phase=phase)
        if family in ("glm", "placebo", "feat", "one"):
            model = fit_glm(train, w, {**L7, **params}, block=range(block[0], block[1] + 1))
            eta = model.predict_raw(score)
        else:
            from eracoef import shotlearners as sl
            bp = base_params or L7
            # the base regression gets weights of 1 when its own half-lives are set (shotmodel.fit then applies the
            # closeness itself); otherwise the driver's 5-season closeness, as the current model was fitted
            w_base = np.ones(len(train)) if (bp.get("half_life_past") is not None or bp.get("half_life_future") is not None)                 else sm.closeness_weights(train["season"].to_numpy(), range(block[0], block[1] + 1))
            base = base_model(base_name, block, train, w_base, bp, rows, seed)
            fitted = sl.fit_booster(family, train, w, params, base, holdout)
            eta = fitted.predict_raw(score)
        t = evaluate(eta, score)
        tabs.append(t)
        logits.append(pd.DataFrame(dict(game_id=score["game_id"], action_number=score["action_number"],
                                        season=score["season"], eta_raw=eta.astype(np.float32))))
        if report is not None:
            report(i, pd.concat(tabs, ignore_index=True))
    return pd.concat(tabs, ignore_index=True), pd.concat(logits, ignore_index=True)


def mean_sall(t: pd.DataFrame) -> float:
    return float(np.average(t["sall"], weights=t["n"]))


def log_trial(family, trial, params, tab, logits, seconds):
    OUT.mkdir(parents=True, exist_ok=True)
    rows = tab.assign(family=family, trial=str(trial), params=json.dumps(params, sort_keys=True, default=str),
                      seconds=seconds)
    path = OUT / "trials.parquet"
    if path.exists():
        old = pd.read_parquet(path)
        old = old[~((old["family"] == family) & (old["trial"] == str(trial)))]
        rows = pd.concat([old, rows], ignore_index=True)
    rows.to_parquet(path, index=False)
    d = STORE / family
    d.mkdir(parents=True, exist_ok=True)
    logits.to_parquet(d / f"{trial}.parquet", index=False)


def baseline(rows: int, seed: int = 0) -> pd.DataFrame:
    path = OUT / f"trial0_rows{rows}_seed{seed}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    t0 = time.time()
    tab, logits = run_trial("one", {}, rows, seed)
    OUT.mkdir(parents=True, exist_ok=True)
    tab.to_parquet(path, index=False)
    log_trial("baseline", f"L7_seed{seed}", L7, tab, logits, time.time() - t0)
    return tab


def compare(tab: pd.DataFrame, base: pd.DataFrame) -> str:
    out = []
    for sub in (None, "rim", "mid", "three"):
        p = ss.paired(tab, base, sub=sub)
        out.append(f"{sub or 'all'} {1000 * p['diff']:+.3f}/1000 z {p['z']:+.1f} ({p['won']}/{p['n']})")
    return " | ".join(out)


def main():
    check_flags()
    load_config()
    family = flag("family", "placebo")
    rows = int(flag("rows", str(ss.SEARCH_ROWS)))
    base = baseline(rows)
    print(f"trial 0 (the current model, {rows:,} rows): SALL {mean_sall(base):.5f}", flush=True)
    if family == "placebo":
        for seed in (1, 2, 3):
            t0 = time.time()
            tab, logits = run_trial("one", {}, rows, seed)
            log_trial("placebo", f"seed{seed}", L7, tab, logits, time.time() - t0)
            print(f"placebo seed {seed}: {compare(tab, base)}  ({time.time() - t0:.0f}s)", flush=True)
        return
    if family == "one":
        params = json.loads(flag("params", "{}"))
        name = flag("name", "one")
        t0 = time.time()
        tab, logits = run_trial("one", params, rows)
        log_trial("one", name, params, tab, logits, time.time() - t0)
        print(f"{name}: {compare(tab, base)}  ({time.time() - t0:.0f}s)")
        return
    if family == "feat":
        adds = [b for b in flag("blocks_add", "").split(",") if b]
        for b in adds:
            t0 = time.time()
            params = dict(blocks=FULL + [b])
            tab, logits = run_trial("feat", params, rows)
            log_trial("feat", f"add_{b}", params, tab, logits, time.time() - t0)
            print(f"+ {b:10s}: {compare(tab, base)}  ({time.time() - t0:.0f}s)", flush=True)
        return
    if family == "fixed":
        # one booster configuration, scored like a trial: --of=<family> --params='<json>' --name=<label> and the base
        # flags (the registered chimeraboost quality 3 and 5 rows are this, on the family's best parameters)
        of, params, name = flag("of"), json.loads(flag("params", "{}")), flag("name", "fixed")
        bp = {**L7, **json.loads(flag("base_params", "{}"))}
        t0 = time.time()
        tab, logits = run_trial(of, params, rows, base_name=flag("base", "L7"), base_params=bp)
        log_trial(f"{of}_fixed", name, params, tab, logits, time.time() - t0)
        print(f"{of} {name}: {compare(tab, base)}  ({time.time() - t0:.0f}s)", flush=True)
        return
    import optuna
    n_trials = int(flag("trials", "40"))
    # the regression the boosters correct: --base=<name> --base_params='<json>' (default: the current model)
    base_params = {**L7, **json.loads(flag("base_params", "{}"))}
    study_name = flag("study", family)
    storage = f"sqlite:///{(OUT / 'optuna.db').as_posix()}"
    sampler = optuna.samplers.TPESampler(seed=0, multivariate=True, group=True, n_startup_trials=max(n_trials // 3, 8))
    study = optuna.create_study(study_name=study_name, storage=storage, direction="minimize", sampler=sampler,
                                load_if_exists=True, pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=2))
    if family == "glm" and len(study.trials) == 0:
        study.enqueue_trial(TRIAL0_GLM)                 # the current model's settings, scored like every other trial
    base_by_season = base.groupby("season").apply(lambda g: np.average(g["sall"], weights=g["n"]), include_groups=False)

    def suggest(trial):
        if family == "glm":
            return glm_space(trial)                     # this script's space: the registered one, with the feature toggles
        from eracoef import shotlearners as sl
        return sl.suggest(trial, family)

    def objective(trial):
        params = suggest(trial)
        t0 = time.time()

        def report(i, partial):
            cur = partial.groupby("season").apply(lambda g: np.average(g["sall"], weights=g["n"]), include_groups=False)
            trial.report(float((cur - base_by_season.reindex(cur.index)).mean()), i)
            if trial.should_prune():
                raise optuna.TrialPruned()

        tab, logits = run_trial(family, params, rows, base_name=flag("base", "L7"), base_params=base_params,
                                report=report)
        log_trial(family, trial.number, params, tab, logits, time.time() - t0)
        cur = tab.groupby("season").apply(lambda g: np.average(g["sall"], weights=g["n"]), include_groups=False)
        val = float((cur - base_by_season.reindex(cur.index)).mean())
        print(f"{family} #{trial.number}: {compare(tab, base)}  ({time.time() - t0:.0f}s)  {json.dumps(params, default=str)[:160]}", flush=True)
        return val

    study.optimize(objective, n_trials=n_trials, gc_after_trial=True)
    print("best:", study.best_trial.number, study.best_value, study.best_trial.params)


def glm_space(trial) -> dict:
    """The regression's options (spec 2(a)); every penalty in Hessian units per 1.5M rows."""
    p = dict(
        ridge=trial.suggest_float("ridge", 1e-2, 100.0, log=True),
        lam_shooter=trial.suggest_float("lam_shooter", 5.0, 500.0, log=True),
        lam_arena=trial.suggest_float("lam_arena", 5.0, 500.0, log=True),
        rounds=trial.suggest_categorical("rounds", [2, 4, 6]),
        half_life_past=trial.suggest_float("half_life_past", 1.0, 30.0, log=True),
        half_life_future=trial.suggest_float("half_life_future", 1.0, 30.0, log=True),
    )
    import inspect
    sig = inspect.signature(sm.fit).parameters
    if "standardise" in sig:
        p["standardise"] = trial.suggest_categorical("standardise", [False, True])
    if "ridge_int" in sig:
        inter = [b for b in ("spot2d", "start_x_time", "clock_x_dist", "putback_x_dist", "context_x")
                 if trial.suggest_categorical(f"inter_{b}", [False, True])]
        p["inter"] = tuple(inter)
        p["ridge_int"] = trial.suggest_float("ridge_int", 1e2, 1e7, log=True)   # the review: 1e3 barely shrinks
    if "era_lambda" in sig and trial.suggest_categorical("era", [False, True]):
        p["era_lambda"] = trial.suggest_float("era_lambda", 1.0, 1e4, log=True)
    if "knots" in sig:
        p["knots"] = dict(dist=trial.suggest_categorical("k_dist", [4, 6, 8]),
                          tposs=trial.suggest_categorical("k_tposs", [3, 5, 7]),
                          on=trial.suggest_categorical("k_on", [3, 5]))
    # the features that passed step 1 (DECISIONS.md, the search's feature step), each a toggle
    p["blocks"] = FULL + [b for b in FEATURE_TOGGLES if trial.suggest_categorical(f"feat_{b}", [False, True])]
    return p


FEATURE_TOGGLES = ("late", "line")   # the feature step on the coded spot (DECISIONS.md, the location leak)
TRIAL0_GLM = dict(ridge=1.0, lam_shooter=50.0, lam_arena=50.0, rounds=4, half_life_past=5.0, half_life_future=5.0,
                  standardise=False, ridge_int=1e4, era=False, k_dist=6, k_tposs=5, k_on=5,
                  **{f"inter_{b}": False for b in ("spot2d", "start_x_time", "clock_x_dist", "putback_x_dist", "context_x")},
                  **{f"feat_{b}": False for b in FEATURE_TOGGLES})


if __name__ == "__main__":
    main()
