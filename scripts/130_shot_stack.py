"""Stacking the search's members: a convex blend of their logits per sub-model, weights fitted by SALL out of block.

    python scripts/130_shot_stack.py --members=baseline:L7_seed0,glm:12,xgb:7 [--name=stack1]

Members are trials already scored by scripts/127 (their logits on the search seasons in data/shotsearch/<family>/
<trial>.parquet).  For each sub-model (rim, other twos, threes) the blend is eta = sum_k w_k eta_k with w >= 0 summing
to 1 (softmax parametrisation), relevelled from the season and scored by SALL like every trial.  A search block's
weights are fitted on the OTHER search blocks without that block's neighbours, so the stack is scored out of block,
like the members.  Variants printed side by side: each member alone, equal weights, the fitted blend; and the weights
per block (if they move by more than 0.2 between blocks, the protocol ships equal weights).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _cli import check_flags, flag  # noqa: E402
from eracoef import shotmodel as sm  # noqa: E402
from eracoef import shotsearch as ss  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
STORE = ROOT / "data" / "shotsearch"


def load(members):
    frames = []
    for s in sorted(ss.SEARCH_SEASONS):
        t = pd.read_parquet(ROOT / "data" / "shotq" / "_table" / f"{s}.parquet")
        frames.append(t[(t.phase == "RS") & ~t.heave])
    F = pd.concat(frames, ignore_index=True)
    for name in members:
        fam, trial = name.split(":", 1)
        e = pd.read_parquet(STORE / fam / f"{trial}.parquet", columns=["game_id", "action_number", "eta_raw"])
        F = F.merge(e.rename(columns={"eta_raw": name}), on=["game_id", "action_number"], how="inner")
    return F.reset_index(drop=True)


def blend_sall(w, E, F, band, sub_mask):
    eta = E @ w
    lev = ss.relevel_fast(eta, F["season"].to_numpy(), band, F["made"].to_numpy(), np.ones(len(F), bool))
    return ss.sall_table(lev, F)


def fit_weights(E, F, band, rows):
    k = E.shape[1]

    # The weights are fitted on season-relevelled log loss, not SALL: SALL refits the shooter ridge on every call
    # (seconds a call, hours a stack).  The members are all shooter-neutral with the same inputs, so the shooter
    # adjustment cannot favour one; the blend is still SCORED by SALL below.
    Er, Fr, br = E[rows], F[rows].reset_index(drop=True), band[rows]
    season, made = Fr["season"].to_numpy(), Fr["made"].to_numpy()
    rs = (Fr["phase"] == "RS").to_numpy() if "phase" in Fr.columns else np.ones(len(Fr), bool)
    y = made.astype(float)

    def loss(theta):
        w = np.exp(theta - theta.max()); w /= w.sum()
        lev = ss.relevel_fast(Er @ w, season, br, made, rs)
        return 1000.0 * float(np.mean(np.logaddexp(0.0, lev) - y * lev))   # log loss per 1000 attempts

    # Unit first steps: from all-zero logits scipy's default simplex steps by 0.00025, a weight change of about 1e-4
    # that moves the loss by ~1e-9 -- under any tolerance, so it stopped at equal weights (the first run, 2026-10-06).
    simplex = np.vstack([np.zeros(k), np.eye(k)])
    r = minimize(loss, np.zeros(k), method="Nelder-Mead",
                 options=dict(maxiter=300 * k, xatol=1e-3, fatol=1e-5, initial_simplex=simplex))
    w = np.exp(r.x - r.x.max())
    return w / w.sum()


def main():
    check_flags()
    members = [m for m in flag("members", "").split(",") if m]
    name = flag("name", "stack")
    F = load(members)
    band = ss.band_codes(F)
    sub = sm.submodel_of(F)
    out_eta = {v: np.zeros(len(F)) for v in ("equal", "fitted")}
    weights = []
    for block in ss.SEARCH_BLOCKS:
        rated = set(range(block[0], block[1] + 1))
        near = rated | {block[0] - 1, block[1] + 1}
        test = F["season"].isin(rated).to_numpy()
        fit_rows = ~F["season"].isin(near).to_numpy()
        for s in ("rim", "mid", "three"):
            m = sub == s
            E = F.loc[m, members].to_numpy(float)
            w = fit_weights(E, F[m].reset_index(drop=True), band[m], fit_rows[m])
            weights.append(dict(block=f"{block[0]}-{block[1]}", sub=s, **dict(zip(members, np.round(w, 3)))))
            idx = np.flatnonzero(m & test)
            out_eta["fitted"][idx] = F.loc[idx, members].to_numpy(float) @ w
            out_eta["equal"][idx] = F.loc[idx, members].to_numpy(float).mean(axis=1)
    W = pd.DataFrame(weights)
    print("fitted weights by block (out of block):")
    print(W.to_string(index=False))
    spread = W.groupby("sub")[members].agg(lambda x: x.max() - x.min()).max(axis=1)
    print("largest move of a weight between blocks, by sub-model:", spread.round(3).to_dict())
    rs = np.ones(len(F), bool)
    tabs = {}
    for col in members:
        tabs[col] = ss.sall_table(ss.relevel_fast(F[col].to_numpy(float), F["season"].to_numpy(), band, F["made"].to_numpy(), rs), F)
    for v, eta in out_eta.items():
        tabs[f"{name}_{v}"] = ss.sall_table(ss.relevel_fast(eta, F["season"].to_numpy(), band, F["made"].to_numpy(), rs), F)
    ref = members[0]
    print(f"\nSALL per 1000 attempts, minus {ref} (search seasons, out of block):")
    for k, t in tabs.items():
        line = f"  {k:24s}"
        for s in (None, "rim", "mid", "three"):
            p = ss.paired(t, tabs[ref], sub=s)
            line += f" | {s or 'all'} {1000 * p['diff']:+.3f} z {p['z']:+.1f}"
        print(line)
    d = STORE / "stack"
    d.mkdir(parents=True, exist_ok=True)
    for v, eta in out_eta.items():
        pd.DataFrame(dict(game_id=F["game_id"], action_number=F["action_number"], season=F["season"],
                          eta_raw=eta.astype(np.float32))).to_parquet(d / f"{name}_{v}.parquet", index=False)
    W.to_csv(ROOT / "outputs" / "shotsearch" / f"{name}_weights.csv", index=False)
    (ROOT / "outputs" / "shotsearch" / f"{name}_members.json").write_text(json.dumps(members), encoding="utf-8")


if __name__ == "__main__":
    main()
