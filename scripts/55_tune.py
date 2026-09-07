"""Hyperparameter search over the WHOLE estimator, against the criterion, with a held-back half.

    python scripts/55_tune.py [--trials=200] [--workers=5] [--study=all] [--k=3] [--resume]

Why this exists: every knob in FINDINGS 21 and 22 was moved one at a time by hand, and 21.25 and 22.6 both
found that the parts INTERACT -- a knob's sign depends on what else is on.  One-at-a-time cannot find that,
and the criterion is only ~50 fit-seconds per evaluation, so a search can.

What is searched (`SPACE`): the ridge (`lam` as a multiple of the shipped one, `lam_ratio`), the prior's
pooling (`win_decay` per side, the offensive target's blend weight), and the booster per side (depth,
learning rate, l2, bins, subsample, colsample, min_child_weight, linear leaves, cross features, the bag).
The two sides get separate booster spaces because 21.26 established they want different priors.

**The search/confirm split is the point.**  Selecting on the criterion is selecting on the test set, and 200
trials will overfit it.  The objective is scored on the SEARCH seasons only -- every other held-out season,
so both halves span all 28 years and every era -- and the CONFIRM seasons are never seen by the optimizer.
At the end the top candidates are re-scored on the confirm half and on all 28, and the gap between search and
confirm is printed.  A candidate that wins the search by more than it wins the confirm half is fitted noise.

Parallelism is over TRIALS, not seasons: each worker fits all of one trial's seasons in one process, so the
design cache, the scoring frames and the shot tables are loaded once per worker and stay warm.  Results
stream to outputs/tune_<study>.csv after every trial, so the run can be killed and resumed.
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eracoef.config import load_config  # noqa: E402

ROOT = Path(load_config()["_root"])
OUT = ROOT / "outputs"
MAP = "linear+log2&xlog&prior&tshare:linear+log2&xlog"     # the shipping map family
THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS")


def flag(name, default=None):
    hit = [a for a in sys.argv[1:] if a.startswith(f"--{name}=")]
    return hit[0].split("=", 1)[1] if hit else default


# --------------------------------------------------------------------------- the space
def suggest(t) -> dict:
    """One trial's parameters.  Kept flat and JSON-ish so a row of the log rebuilds the system exactly."""
    p = dict(
        lam_mult=t.suggest_float("lam_mult", 0.3, 2.0, log=True),
        lam_ratio=t.suggest_float("lam_ratio", 0.4, 1.6, log=True),
        blend=t.suggest_float("blend", 0.4, 1.0),                  # offensive target: blend x APM + (1-x) RAPM_1
        win_decay=t.suggest_float("win_decay", 0.2, 1.0),
        win_decay_d=t.suggest_float("win_decay_d", 0.2, 1.0),
    )
    for side in ("o", "d"):
        p[f"{side}_depth"] = t.suggest_int(f"{side}_depth", 3, 8)
        p[f"{side}_lr"] = t.suggest_float(f"{side}_lr", 0.03, 0.3, log=True)
        p[f"{side}_l2"] = t.suggest_float(f"{side}_l2", 0.3, 30.0, log=True)
        p[f"{side}_bins"] = t.suggest_categorical(f"{side}_bins", [32, 64, 128, 254])
        p[f"{side}_subsample"] = t.suggest_float(f"{side}_subsample", 0.5, 1.0)
        p[f"{side}_colsample"] = t.suggest_float(f"{side}_colsample", 0.5, 1.0)
        p[f"{side}_mcw"] = t.suggest_float(f"{side}_mcw", 1.0, 40.0, log=True)
        p[f"{side}_ll"] = t.suggest_categorical(f"{side}_ll", [True, False])
        p[f"{side}_cf"] = t.suggest_categorical(f"{side}_cf", [True, False])
        p[f"{side}_bag"] = t.suggest_categorical(f"{side}_bag", [1, 3, 5])
    return p


def build(p: dict, cfg: dict):
    """A trial's parameters -> the MspiFast the criterion will score."""
    from eracoef.fastfit import MspiFast
    from eracoef.gbdt_prior import FULL_FEATURES, SHOT_FEATURES, SHOTQ

    def booster(side):
        b = dict(depth=int(p[f"{side}_depth"]), learning_rate=float(p[f"{side}_lr"]),
                 l2_leaf_reg=float(p[f"{side}_l2"]), max_bins=int(p[f"{side}_bins"]),
                 subsample=float(p[f"{side}_subsample"]), colsample=float(p[f"{side}_colsample"]),
                 min_child_weight=float(p[f"{side}_mcw"]), linear_leaves=bool(p[f"{side}_ll"]),
                 cross_features=bool(p[f"{side}_cf"]))
        n = int(p[f"{side}_bag"])
        if n > 1:
            b.update(n_ensembles=n, ensemble_n_jobs=1)          # never fork inside a worker (the 323 s trap)
        return b

    w = float(p["blend"])
    target = "apm" if w >= 0.999 else f"blend{round(w, 3)}"
    return MspiFast("tune", target=target, target_d="rapm1",
                    lam=float(cfg["lam_plugin"]) * float(p["lam_mult"]),
                    lam_ratio=float(p["lam_ratio"]),
                    win_decay=float(p["win_decay"]), win_decay_d=float(p["win_decay_d"]),
                    gbdt_params=booster("o"), gbdt_params_d=booster("d"),
                    gbdt_features={"O": list(SHOT_FEATURES), "D": [*FULL_FEATURES, *SHOTQ]})


# --------------------------------------------------------------------------- the worker
_W: dict = {}


def _warm(k: int, seasons=None):
    """Per-process state: the config, the Context and the scoring frames, loaded once and kept.

    Only the seasons this worker actually scores are loaded -- a worker in the search phase holds 14 frames,
    not 28.  The box has a 48 GB commit limit and the handoff has been bitten by oversubscribing it once."""
    if not _W:
        from eracoef.holdout import Context, Holdout
        cfg = load_config()
        _W.update(cfg=cfg, ho=Holdout.from_config(cfg, ks=[k]), ctx=Context.load(cfg), k=k, frames={})
    want = [s for s in (seasons or []) if s not in _W["frames"]]
    if want:
        from eracoef.calmap import load_frames
        _W["frames"].update(load_frames(_W["ctx"], want, level=_W["ho"].level, verbose=False))
    return _W


def evaluate(p: dict, held: list, k: int, system: str | None = None) -> dict:
    """Fit the system for each held-out season IN THIS PROCESS, fit the map leave-one-season-out on the
    result, return the pooled mapped team-game error over `held`."""
    from eracoef.calmap import SideMap, dump_systems, evaluate as cal_evaluate
    from eracoef.holdout import pooled
    w = _warm(k, held)
    cfg, ho, ctx = w["cfg"], w["ho"], w["ctx"]
    if system:                                   # a registry system, for the reference line
        from eracoef.systems import registry
        obj = registry(cfg)[system]
        obj = __import__("dataclasses").replace(obj, name="tune")
    else:
        obj = build(p, cfg)
    R = dump_systems(ho, [obj], ctx, held=held, verbose=False)
    ctx.current_h = None
    secs = float(R.groupby("held_out").seconds.first().sum())
    fo, fd = MAP.split(":")
    frames = {s: f for s, f in w["frames"].items() if s in set(held)}
    r, _ = cal_evaluate(R, frames, "tune", k, SideMap.parse(fo), SideMap.parse(fd), "tune_mapped")
    P = pooled(r).set_index("system")
    return dict(game=float(P.loc["tune_mapped"].game), seconds=secs, n=len(held))


def _job(args):
    idx, p, held, k = args[:4]
    system = args[4] if len(args) > 4 else None
    os.environ.update({v: "2" for v in THREAD_VARS})
    try:
        r = evaluate(p, held, k, system)
        return idx, r, None
    except Exception as e:                       # a bad corner of the space must not kill the study
        import traceback
        return idx, None, traceback.format_exc(limit=3) or str(e)


# --------------------------------------------------------------------------- the study
def main():
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    import optuna

    from eracoef.holdout import Holdout

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    cfg = load_config()
    k = int(flag("k", 3))
    n_trials = int(flag("trials", 200))
    workers = int(flag("workers", 5))
    study_name = flag("study", "all")
    log = OUT / f"tune_{study_name}.csv"

    seasons = Holdout.from_config(cfg, ks=[k]).seasons()
    search = seasons[::2]              # alternating, so both halves span every era
    confirm = seasons[1::2]
    print(f"K = {k}, {len(seasons)} held-out seasons: {len(search)} SEARCH, {len(confirm)} CONFIRM (untouched)")
    print(f"  search  {search}")
    print(f"  confirm {confirm}")
    print(f"{n_trials} trials, {workers} workers, log -> {log.relative_to(ROOT)}\n", flush=True)

    base_name = flag("baseline", "ship_shot7d")
    base = {}
    if base_name and base_name != "none":
        with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context("spawn")) as ex:
            fs = {half: ex.submit(_job, (-1, {}, sea, k, base_name))
                  for half, sea in (("search", search), ("confirm", confirm))}
            for half, f in fs.items():
                _, r, err = f.result()
                base[half] = np.nan if err else r["game"]
        print(f"reference line `{base_name}`: search {base['search']:.4f}   confirm {base['confirm']:.4f}")
        print("  (the two halves sit at different levels; every number below is read against THIS line)",
              flush=True)

    sampler = optuna.samplers.TPESampler(seed=0, n_startup_trials=max(30, workers * 4), multivariate=True)
    # sqlite storage so a two-hour run survives being killed: re-run the same command and it continues,
    # sampler state and all.  The csv beside it is for reading, not for resuming.
    db = f"sqlite:///{(OUT / f'tune_{study_name}.db').as_posix()}"
    study = optuna.create_study(direction="minimize", sampler=sampler, study_name=study_name,
                                storage=db, load_if_exists=True)
    prior = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if prior:
        print(f"resumed {len(prior)} completed trials from {Path(db).name}, "
              f"best so far {min(t.value for t in prior):.4f}", flush=True)
    n_trials = max(0, n_trials - len(prior))

    done, t0 = [], time.time()
    if n_trials == 0:
        print("nothing left to run; raise --trials")
    ctxm = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctxm) as ex:
        pending, asked = {}, 0
        while len(done) < n_trials:
            while len(pending) < workers and asked < n_trials:
                t = study.ask()
                p = suggest(t)
                pending[ex.submit(_job, (t.number, p, search, k))] = (t, p)
                asked += 1
            from concurrent.futures import FIRST_COMPLETED, wait
            fut = next(iter(wait(list(pending), return_when=FIRST_COMPLETED).done))
            t, p = pending.pop(fut)
            idx, r, err = fut.result()
            if err:
                study.tell(t, state=optuna.trial.TrialState.FAIL)
                print(f"  trial {idx:4d} FAILED: {err.splitlines()[-1][:110]}", flush=True)
                done.append(dict(trial=idx, game=np.nan, **p))
                continue
            study.tell(t, r["game"])
            row = dict(trial=idx, game=r["game"], seconds=r["seconds"], n=r["n"], **p)
            done.append(row)
            pd.DataFrame(done).to_csv(log, index=False)
            best = min(d["game"] for d in done if not np.isnan(d.get("game", np.nan)))
            print(f"  trial {idx:4d}  search {r['game']:9.4f}   best {best:9.4f}   "
                  f"{r['seconds']:5.1f}s fit   [{len(done)}/{n_trials}, {(time.time() - t0) / 60:.1f} min]",
                  flush=True)

    D = pd.DataFrame(done).sort_values("game")
    D.to_csv(log, index=False)
    print(f"\n=== search done, {len(D)} trials in {(time.time() - t0) / 60:.1f} min.  "
          f"Top {min(8, len(D))} on the SEARCH half:\n")
    cols = ["trial", "game", "lam_mult", "lam_ratio", "blend", "win_decay", "win_decay_d",
            "o_depth", "o_ll", "o_cf", "o_bag", "d_depth", "d_ll", "d_cf", "d_bag"]
    print(D.head(8)[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # ---------------------------------------------------------------- the honest part
    top = D.head(int(flag("confirm_n", 6))).to_dict("records")
    print(f"\n=== the {len(top)} best re-scored on the CONFIRM half, which the optimizer never saw\n")
    print(f"  {'trial':>6s} {'search':>9s} {'vs base':>8s} {'confirm':>9s} {'vs base':>8s}")
    conf = []
    with ProcessPoolExecutor(max_workers=min(workers, len(top)), mp_context=ctxm) as ex:
        futs = {ex.submit(_job, (int(r["trial"]), {kk: r[kk] for kk in r if kk not in
                                                   ("trial", "game", "seconds", "n")}, confirm, k)): r
                for r in top}
        for f in futs:
            r = futs[f]
            idx, res, err = f.result()
            if err:
                print(f"  {int(r['trial']):6d}  confirm FAILED")
                continue
            conf.append(dict(trial=int(r["trial"]), search=r["game"], confirm=res["game"]))
    bs, bc = base.get("search", np.nan), base.get("confirm", np.nan)
    for c in sorted(conf, key=lambda x: x["confirm"]):
        c["d_search"], c["d_confirm"] = c["search"] - bs, c["confirm"] - bc
        flagstr = "" if not (c["d_confirm"] > 0 > c["d_search"]) else "   <- search-only, discard"
        print(f"  {c['trial']:6d} {c['search']:9.4f} {c['d_search']:+8.4f} {c['confirm']:9.4f} "
              f"{c['d_confirm']:+8.4f}{flagstr}")
    pd.DataFrame(conf).to_csv(OUT / f"tune_{study_name}_confirm.csv", index=False)
    print(f"\nwrote {log.name} and tune_{study_name}_confirm.csv")
    print("The winner is the best CONFIRM score, not the best search score.  Re-run it through "
          "scripts/54_track.py on all 28 seasons and read the consensus floors before shipping anything.")


if __name__ == "__main__":
    main()
