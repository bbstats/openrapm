"""Price a search candidate on every season, in scripts/120's format, for the final battery (117, 118, 119, 121).

    python scripts/132_shot_price.py --candidate=glm:48 --name=anch_G48
    python scripts/132_shot_price.py --candidate=L7 --name=anch_L7
    python scripts/132_shot_price.py --candidate=eq:anch_xgb13+anch_lgbm25+anch_chim26 --name=anch_stack3
    python scripts/132_shot_price.py --spec=<file.json or inline JSON> --name=c_nctx_xgb   # a configuration not in trials

--spec (see spec_candidate): {"family": "glm" or a booster, "params": {...} or "trial": "<family>:<trial>",
"base_name" / "base_params" (a booster's base regression: 127's cache name, params on top of L7), "drop" (tree inputs
left out of shotfeatures.tree_features), "cache" (a glm spec: fit through 127's base-model cache under that name, so a
booster spec with that base_name reuses the very same fit)}.

For each of the ten three-season blocks (config `windows`): the candidate is fitted exactly as scripts/127 and 131 fit
it -- the block's allowed training seasons (never the block, nor the season either side, nor 2026), the fixed row
sample at the shipped 1.5M rows (seed 0), the base regression `G` under a booster -- and every attempt of the block's
seasons is priced, regular season and playoffs, offsets zeroed, level from the whole season (shotmodel.relevel);
heaves get the training seasons' heave rate.  So on a confirm block this is the confirm read's seed-0 model.
`eq:` blends the named, already priced directories' raw logits with equal weights, then relevels.

Writes data/shotq/<name>/<season>.parquet: game_id, action_number, phase, q, eta_raw.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _cli import check_flags, flag  # noqa: E402
from eracoef import shotfeatures, shotmodel as sm, shotsearch as ss  # noqa: E402
from eracoef.config import load_config  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "shotsearch"
SHOTQ = ROOT / "data" / "shotq"
_spec = importlib.util.spec_from_file_location("shot_search", Path(__file__).resolve().parent / "127_shot_search.py")
S = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S)
ALL = list(range(1997, 2027))


def full_table(season: int) -> pd.DataFrame:
    """The season's whole table, heaves and playoffs kept, with the search's derived fields."""
    t = pd.read_parquet(SHOTQ / "_table" / f"{season}.parquet")
    return shotfeatures.prepare(t, require_blocked=True)


def candidate(name: str):
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


def spec_candidate(text: str):
    """(family, params, base params, base name, features, cache name) of a --spec configuration."""
    p = Path(text)
    spec = json.loads(p.read_text(encoding="utf-8") if text.endswith(".json") and p.exists() else text)
    unknown = set(spec) - {"family", "params", "trial", "base_name", "base_params", "drop", "cache"}
    if unknown:
        raise SystemExit(f"--spec: unknown keys {sorted(unknown)}")
    fam = spec["family"]
    if "trial" in spec:
        _, params, _ = candidate(spec["trial"])
    else:
        params = dict(spec.get("params", {}))
    base = {**S.L7, **spec.get("base_params", {})} if fam != "glm" else None
    drop = list(spec.get("drop", []))
    features = None
    if drop:
        def features(f, _drop=tuple(drop)):
            X = shotfeatures.tree_features(f)
            missing = [c for c in _drop if c not in X.columns]
            if missing:
                raise ValueError(f"--spec drop: {missing} are not tree inputs")
            return X.drop(columns=list(_drop))
    return fam, params, base, spec.get("base_name", "G"), features, spec.get("cache")


def train_block(block, rows: int, seed: int, params: dict):
    """127's block_data without the phase guard (the battery prices all ten blocks) and without the score frame."""
    rated = list(range(block[0], block[1] + 1))
    allowed = sm.train_seasons(rated, ALL)
    per = ss.block_rows(len(allowed), rows)
    parts = []
    for s in allowed:
        t = S.table(s)
        parts.append(t.iloc[np.sort(ss.row_perm(len(t), s, seed)[:per])])
    train = pd.concat(parts, ignore_index=True)
    sw = train["season"].to_numpy()
    if params.get("half_life_past") is not None or params.get("half_life_future") is not None:
        w = np.ones(len(train))
    else:
        w = sm.closeness_weights(sw, rated, half_life=5.0)
    near = sorted(allowed, key=lambda s: min(abs(s - block[0]), abs(s - block[1])))[:2]
    return train, w, np.isin(sw, near), allowed


def write(name: str, season: int, f: pd.DataFrame, eta: np.ndarray, heave_rate: float):
    heave = f["heave"].to_numpy()
    lev, _ = sm.relevel(eta[~heave], f[~heave].reset_index(drop=True))
    q = np.full(len(f), heave_rate)
    q[~heave] = sm.probability(lev)
    d = SHOTQ / name
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(dict(game_id=f["game_id"], action_number=f["action_number"], phase=f["phase"], q=q,
                      eta_raw=eta)).to_parquet(d / f"{season}.parquet", index=False)


def heave_rate_of(seasons) -> float:
    h = pd.concat([pd.read_parquet(SHOTQ / "_table" / f"{s}.parquet", columns=["heave", "made"]) for s in seasons])
    return float(h.loc[h["heave"], "made"].mean())


def main():
    check_flags()
    cfg = load_config()
    cand, name, spec = flag("candidate", ""), flag("name"), flag("spec", "")
    rows, seed = int(flag("rows", str(ss.FULL_ROWS))), int(flag("seed", "0"))
    blocks = [tuple(w) for w in cfg["windows"]]
    only = flag("block", "")                         # e.g. --block=2015: just the block holding that season
    if only:
        blocks = [b for b in blocks if b[0] <= int(only) <= b[1]]
    t_all = time.time()
    if not spec and cand.startswith("eq:"):
        members = cand[3:].split("+")
        for block in blocks:
            allowed = sm.train_seasons(list(range(block[0], block[1] + 1)), ALL)
            hr = heave_rate_of(allowed)
            for s in range(block[0], block[1] + 1):
                f = full_table(s)
                E = [pd.read_parquet(SHOTQ / m / f"{s}.parquet") for m in members]
                for e in E:
                    assert (e["game_id"].to_numpy() == f["game_id"].to_numpy()).all() and \
                           (e["action_number"].to_numpy() == f["action_number"].to_numpy()).all(), f"{s}: rows differ"
                write(name, s, f, np.mean([e["eta_raw"].to_numpy(float) for e in E], axis=0), hr)
            print(f"  {block[0]}-{block[1]}: blended {members}", flush=True)
        print(f"done in {time.time() - t_all:.0f}s")
        return
    if spec:
        fam, params, base_params, base_name, features, cache = spec_candidate(spec)
    else:
        fam, params, base_params = candidate(cand)
        base_name, features, cache = "G", None, None
    from eracoef import shotlearners as sl
    for block in blocks:
        t0 = time.time()
        train, w, holdout, allowed = train_block(block, rows, seed, params)
        rated = range(block[0], block[1] + 1)
        if fam in ("glm", "one") and cache:
            model = S.base_model(cache, block, train, w, {**S.L7, **params}, rows, seed)
        elif fam in ("glm", "one"):
            model = S.fit_glm(train, w, {**S.L7, **params}, block=rated)
        else:
            bp = base_params
            w_base = np.ones(len(train)) if (bp.get("half_life_past") is not None or bp.get("half_life_future") is not None) \
                else sm.closeness_weights(train["season"].to_numpy(), rated)
            base = S.base_model(base_name, block, train, w_base, bp, rows, seed)
            model = sl.fit_booster(fam, train, w, params, base, holdout, features=features)
        hr = heave_rate_of(allowed)
        for s in rated:
            f = full_table(s)
            heave = f["heave"].to_numpy()
            eta = np.zeros(len(f))
            eta[~heave] = model.predict_raw(f[~heave].reset_index(drop=True))
            write(name, s, f, eta, hr)
        print(f"  {block[0]}-{block[1]}: {len(train):,} training attempts, {time.time() - t0:.0f}s", flush=True)
    print(f"done in {time.time() - t_all:.0f}s")


if __name__ == "__main__":
    main()
