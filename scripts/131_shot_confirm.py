"""The search's confirm read: the registered shortlist, refitted once on the CONFIRM blocks.

    python scripts/131_shot_confirm.py --candidates=glm:48,xgb:13 [--seeds=0,1,2] [--rows=1500000] [--report=1]

Each candidate is a trial already scored on the search half by scripts/127 (its parameters are read from
outputs/shotsearch/trials.parquet; a booster sits on the best regression, outputs/shotsearch/G_best_params.json).  It
and the current model (L7) are fitted at the shipped size (1.5M rows) on every confirm block, on three row samples, and
scored by SALL on the confirm seasons, which no trial, feature decision or stack weight has read.  Resumable: a
(candidate, seed) already in outputs/shotsearch/confirm.parquet is not refitted.

--report=1 prints, per candidate and sub-model (DECISIONS.md, the registered decision rule):
  the SALL difference against L7 on the same rows, per 1000 attempts, season-paired z over the 15 confirm seasons;
  Holm-corrected one-sided p across the shortlist; blocks better (of 5); retention = confirm gain / search gain; the
  sd of the all-attempts gain across the row samples.  Written to outputs/shotsearch/confirm.csv.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _cli import check_flags, flag  # noqa: E402
from eracoef import shotsearch as ss  # noqa: E402
from eracoef.config import load_config  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "shotsearch"
STORE = ROOT / "data" / "shotsearch" / "confirm"
_spec = importlib.util.spec_from_file_location("shot_search", Path(__file__).resolve().parent / "127_shot_search.py")
S = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S)
SUBS = ("rim", "mid", "three")


def candidate(name: str):
    """(family, params, base params) of a shortlisted trial; 'L7' is the current model."""
    if name == "L7":
        return "one", {}, None
    fam, trial = name.split(":", 1)
    t = pd.read_parquet(OUT / "trials.parquet")
    row = t[(t["family"] == fam) & (t["trial"] == trial)]
    if not len(row):
        raise SystemExit(f"{name}: not in trials.parquet")
    params = json.loads(row["params"].iloc[0])
    if fam == "glm":
        return "glm", params, None
    base = {**S.L7, **json.loads((OUT / "G_best_params.json").read_text(encoding="utf-8"))}
    return fam.replace("_fixed", ""), params, base


def equal_stack(name: str, seed: int) -> pd.DataFrame:
    """'eq:a+b+c': the equal blend of members' offset-zeroed logits on the confirm seasons (each fitted here first),
    relevelled and scored block by block like any candidate.  Equal weights: the fitted ones moved more than 0.2
    between search blocks (DECISIONS.md, the stack)."""
    members = name[3:].split("+")
    L = [pd.read_parquet(STORE / f"{m.replace(':', '_')}_seed{seed}.parquet") for m in members]
    for x in L[1:]:
        if not (x["game_id"].equals(L[0]["game_id"]) and x["action_number"].equals(L[0]["action_number"])):
            raise SystemExit(f"{name}: members' confirm rows differ")
    eta = np.mean([x["eta_raw"].to_numpy(float) for x in L], axis=0)
    tabs, i = [], 0
    for a, b in ss.CONFIRM_BLOCKS:
        rated = list(range(a, b + 1))
        ss.assert_phase(rated, "confirm")
        score = pd.concat([S.table(s) for s in rated], ignore_index=True)
        score = score[score["phase"] == "RS"].reset_index(drop=True)
        assert (score["game_id"].to_numpy() == L[0]["game_id"].to_numpy()[i:i + len(score)]).all()
        tabs.append(S.evaluate(eta[i:i + len(score)], score))
        i += len(score)
    return pd.concat(tabs, ignore_index=True)


def run(names, seeds, rows):
    path = OUT / "confirm.parquet"
    done = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=["cand", "seed"])
    stacks = [n for n in names if n.startswith("eq:")]
    names = names + [m for s in stacks for m in s[3:].split("+") if m not in names]
    for name in ["L7"] + [n for n in names if n != "L7" and not n.startswith("eq:")]:
        fam, params, base = candidate(name)
        for seed in seeds:
            if ((done["cand"] == name) & (done["seed"] == seed)).any():
                continue
            t0 = time.time()
            kw = dict(base_name="G", base_params=base) if base is not None else {}
            tab, logits = S.run_trial(fam, params, rows, seed, phase="confirm", **kw)
            tab = tab.assign(cand=name, seed=seed, seconds=time.time() - t0)
            done = pd.concat([done, tab], ignore_index=True) if len(done) else tab
            done.to_parquet(path, index=False)
            STORE.mkdir(parents=True, exist_ok=True)
            logits.to_parquet(STORE / f"{name.replace(':', '_')}_seed{seed}.parquet", index=False)
            print(f"  {name} seed {seed}: {time.time() - t0:.0f}s", flush=True)
    for name in stacks:
        for seed in seeds:
            if ((done["cand"] == name) & (done["seed"] == seed)).any():
                continue
            tab = equal_stack(name, seed).assign(cand=name, seed=seed, seconds=0.0)
            done = pd.concat([done, tab], ignore_index=True)
            done.to_parquet(path, index=False)
            print(f"  {name} seed {seed}: blended", flush=True)
    return done


def block_of(season: int) -> str:
    for a, b in ss.CONFIRM_BLOCKS:
        if a <= season <= b:
            return f"{a}-{b}"
    return "?"


def search_gain(name: str, sub) -> float:
    """The candidate's search-half gain against trial 0 (per 1000), for the retention ratio (a stack: its equal
    blend as scripts/130 scored it, logged under family 'stack', trial = the --stack_name given)."""
    if name == "L7":
        return 0.0
    if name.startswith("eq:"):
        fam, trial = "stack", flag("stack_name", "stack3_equal")
    else:
        fam, trial = name.split(":", 1)
    t = pd.read_parquet(OUT / "trials.parquet")
    a = t[(t["family"] == fam) & (t["trial"] == trial)].drop(columns=["family", "trial", "params", "seconds"])
    if not len(a):
        return np.nan
    b = pd.read_parquet(OUT / f"trial0_rows{ss.SEARCH_ROWS}_seed0.parquet")
    return 1000 * ss.paired(a, b, sub=sub)["diff"]


def report(done: pd.DataFrame, names):
    rows = []
    for name in names:
        for sub in (None,) + SUBS:
            per_seed, z0, blocks, n_seasons = [], None, None, 0
            for seed in sorted(done["seed"].unique()):
                a = done[(done["cand"] == name) & (done["seed"] == seed)]
                b = done[(done["cand"] == "L7") & (done["seed"] == seed)]
                if not len(a) or not len(b):
                    continue
                cols = ["season", "sub", "n", "sall", "raw"]
                p = ss.paired(a[cols], b[cols], sub=sub)
                per_seed.append(1000 * p["diff"])
                if seed == 0:
                    z0 = p["z"]
                    n_seasons = p["n"]
                    aa, bb = a[cols], b[cols]
                    if sub is not None:
                        aa, bb = aa[aa["sub"] == sub], bb[bb["sub"] == sub]
                    sa = aa.groupby("season").apply(lambda g: np.average(g["sall"], weights=g["n"]), include_groups=False)
                    sb = bb.groupby("season").apply(lambda g: np.average(g["sall"], weights=g["n"]), include_groups=False)
                    d = (sa - sb).groupby(lambda s: block_of(int(s))).mean()
                    blocks = int((d < 0).sum())
            if not per_seed:
                continue
            sg = search_gain(name, sub)
            rows.append(dict(candidate=name, sub=sub or "all", confirm_per_1000=round(per_seed[0], 3),
                             z=round(z0, 2) if z0 is not None else np.nan, blocks_better=blocks,
                             search_per_1000=round(sg, 3), retention=round(per_seed[0] / sg, 2) if sg < 0 else np.nan,
                             seed_mean=round(float(np.mean(per_seed)), 3),
                             seed_sd=round(float(np.std(per_seed, ddof=1)), 3) if len(per_seed) > 1 else np.nan,
                             n_seeds=len(per_seed), n_seasons=n_seasons))
    r = pd.DataFrame(rows)
    # Holm across the shortlist, per sub-model, one-sided (better = negative z)
    r["p_one_sided"] = norm.cdf(r["z"])
    r["p_holm"] = np.nan
    for sub, g in r[r["candidate"] != "L7"].groupby("sub"):
        order = g["p_one_sided"].sort_values().index
        m = len(order)
        running = 0.0
        for i, ix in enumerate(order):
            running = max(running, min(1.0, (m - i) * r.loc[ix, "p_one_sided"]))
            r.loc[ix, "p_holm"] = running
    r = r[r["candidate"] != "L7"]
    r.to_csv(OUT / "confirm.csv", index=False)
    with pd.option_context("display.width", 250):
        print(r.drop(columns=["p_one_sided"]).to_string(index=False))


def main():
    check_flags()
    load_config()
    names = [n for n in flag("candidates", "").split(",") if n]
    seeds = [int(s) for s in flag("seeds", "0,1,2").split(",")]
    rows = int(flag("rows", str(ss.FULL_ROWS)))
    if flag("report", "0") == "1":
        report(pd.read_parquet(OUT / "confirm.parquet"), names)
        return
    t0 = time.time()
    run(names, seeds, rows)
    print(f"confirm fits done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
