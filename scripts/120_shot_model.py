"""Train a shot-quality model block by block and price every attempt 1997-2026 (eracoef.shotmodel).

    python scripts/120_shot_model.py --name=pbp --blocks=spot,start,putback,clock,context,fatigue,prev
                                     [--offsets=1] [--max_rows=1500000] [--ridge=1] [--first=1997] [--last=2026]
    python scripts/120_shot_model.py --table=1        # (re)build the per-season model tables only
    ... --soft=teacher:0.8      the tracking-taught student: on tracked attempts the label is 0.2 x made + 0.8 x the
                                teacher's out-of-fold quality (data/shotq/teacher/labels.parquet, scripts/121)
    ... --soft_weight=4         and each tracked attempt counts this many times (default 1); tracked attempts are
                                never subsampled
    ... --after=1               also write q_after, the owner's update (quality given the result), with the update
                                sizes v of outputs/shottest/info_gap_v.csv (scripts/119)

The model table (data/shotq/_table/<season>.parquet, built once): the shot frame's regular season and playoff
attempts with shotframe.derive's fields, near-rim depth mapped per arena-season (data/shotframe/rim_map.parquet), and the
rebuilt shot clock (shotclock.rebuild with the lags in data/shotq/clock_lags.json).

For each block of three rated seasons (config `windows`): train on shotmodel.train_seasons (never the block,
nor the season either side, nor 2026), weighted by closeness (half-life 5 seasons), at most `--max_rows`
attempts sampled evenly across those seasons; price the block's attempts with the offsets zeroed; set the
level from the other half (shotmodel.relevel).  Heaves (36 ft or more with 2 s or less left) are priced at the
training seasons' heave rate and never fitted.

Writes data/shotq/<name>/<season>.parquet: game_id, action_number, phase, q (the quality), eta_raw (before the
level), and the model's coefficients to data/shotq/<name>/coef.csv.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _cli import check_flags, flag  # noqa: E402
from eracoef import shotmodel as sm  # noqa: E402
from eracoef.config import load_config  # noqa: E402
from eracoef.shotclock import ClockRules, rebuild  # noqa: E402
from eracoef.shotframe import LOGGED, anchored, code_location, derive, frame_dir, load_frame  # noqa: E402

TABLE_COLS = ["season", "phase", "game_id", "action_number", "half", "arena", "neutral", "team", "opp", "home",
              "shooter", "period", "clock", "value", "made", "dist", "x", "y", "lp", "margin", "poss_start",
              "secs_into_poss", "secs_since_oreb", "shooter_secs_on", "start_prev_value", "start_prev_dist",
              "start_prev_made", "start_prev_blocked", "noloc", "xc", "yc", "dist_xy", "angle", "corner3", "heave",
              "sc", "sc_eff", "reset_kind", "since_reset", "clock_off",
              # the search's extra possession fields (2026-10-06): inputs to shotfeatures, never raw model inputs
              "n_oreb", "att_no", "fga_no", "poss_no", "secs_since_timeout", "poss_start_clock", "start_prev_x",
              "start_prev_y", "start_prev_clock", "events",
              # whether THIS attempt was blocked: never an input for this attempt (shotfeatures.FORBIDDEN bans post_*),
              # only the 'previous attempt was blocked' fact of the NEXT attempt in the possession
              "post_blocked", "anchor_clock", "anchor_kind", "anchor_nev"]


def table_path(cfg, season: int) -> Path:
    return Path(cfg["_root"]) / "data" / "shotq" / "_table" / f"{season}.parquet"


STAMP_COLS = ("clock", "secs_into_poss", "secs_since_oreb", "shooter_secs_on", "secs_since_timeout", "sc", "sc_eff",
              "reset_kind", "since_reset", "clock_off")


def build_table(cfg, season: int, rmap: pd.DataFrame, rules: ClockRules) -> pd.DataFrame:
    f = derive(load_frame([season], cfg, phases=("RS", "PO")), rmap=rmap)
    stamp = pd.concat([f, rebuild(f, rules)], axis=1)
    g = code_location(anchored(f))
    r = rebuild(g, rules)
    t = pd.concat([g, r], axis=1)[TABLE_COLS + [f"{c}_logged" for c in LOGGED]]
    # the stamp-based timing, kept for diagnostics only (shotfeatures.FORBIDDEN bans *_stamp as an input)
    for c in STAMP_COLS:
        t[f"{c}_stamp"] = stamp[c].to_numpy()
    p = table_path(cfg, season)
    p.parent.mkdir(parents=True, exist_ok=True)
    t.to_parquet(p, index=False)
    return t


def load_table(cfg, seasons) -> pd.DataFrame:
    return pd.concat([pd.read_parquet(table_path(cfg, s)) for s in seasons], ignore_index=True)


after_of = sm.after_of          # moved to shotmodel (scripts/133 uses it too)


def main():
    check_flags()
    cfg = load_config()
    first, last = int(flag("first", "1997")), int(flag("last", "2026"))
    all_seasons = list(range(cfg["first_season"], cfg["last_season"] + 1))
    root = Path(cfg["_root"]) / "data" / "shotq"
    if flag("table", "0") == "1":
        rmap = pd.read_parquet(frame_dir(cfg) / "rim_map.parquet")
        lags = json.loads((root / "clock_lags.json").read_text(encoding="utf-8"))
        rules = ClockRules(**lags["rules"], lag=lags["lags"])
        only = [int(x) for x in flag("seasons", "").split(",") if x]
        for s in (only or all_seasons):
            t0 = time.time()
            t = build_table(cfg, s, rmap, rules)
            print(f"  table {s}: {len(t):,} attempts, {time.time() - t0:.0f}s", flush=True)
        return
    name = flag("name")
    blocks = [b for b in flag("blocks", "spot").split(",") if b]
    bad = (set(blocks) - set(sm.BLOCKS) - set(sm.EXTRA_BLOCKS)) | ({"tracking"} & set(blocks))   # no tracking inputs: play-by-play only
    if bad:
        raise SystemExit(f"unknown blocks {sorted(bad)}; known: {sm.BLOCKS} (+ clockfine as an ablation)")
    use_off = flag("offsets", "1") == "1"
    soft = flag("soft", "")
    labels, lam = None, 0.0
    if soft:
        tdir, lam = soft.split(":")
        lam = float(lam)
        # the soft label is the teacher's prediction for THIS shooter (offsets kept): the student fits its own
        # shooter offsets, and a label with the shooter taken out would read a great shooter's makes as luck
        labels = pd.read_parquet(root / tdir / "labels.parquet", columns=["game_id", "action_number", "q_teacher_full"])
        labels = labels.rename(columns={"q_teacher_full": "q_teacher"})
    soft_w = float(flag("soft_weight", "1"))
    vtab = None
    if flag("after", "0") == "1":
        vtab = pd.read_csv(Path(cfg["_root"]) / "outputs" / "shottest" / "info_gap_v.csv")
    max_rows = int(flag("max_rows", "1500000"))
    ridge = float(flag("ridge", "1"))
    out = root / name
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    coef_rows, level_rows = [], []
    t_all = time.time()
    for block in [tuple(w) for w in cfg["windows"]]:
        rated = [s for s in range(block[0], block[1] + 1) if first <= s <= last]
        if not rated:
            continue
        train_s = sm.train_seasons(range(block[0], block[1] + 1), all_seasons)
        per = max_rows // len(train_s)
        parts = []
        for s in train_s:
            t = pd.read_parquet(table_path(cfg, s))
            t = t[~t["heave"]]
            if len(t) > per:
                keep = np.zeros(len(t), dtype=bool)
                keep[rng.choice(len(t), per, replace=False)] = True
                if labels is not None:                       # tracked attempts are all kept
                    keep |= t.set_index(["game_id", "action_number"]).index.isin(
                        labels.set_index(["game_id", "action_number"]).index)
                t = t[keep]
            parts.append(t)
        train = pd.concat(parts, ignore_index=True)
        if labels is not None:
            train = train.merge(labels, on=["game_id", "action_number"], how="left")
            tracked = train["q_teacher"].notna().to_numpy()
            y = train["made"].to_numpy(float)
            y[tracked] = (1.0 - lam) * y[tracked] + lam * train.loc[tracked, "q_teacher"].to_numpy(float)
            train["made"] = y
        w = sm.closeness_weights(train["season"].to_numpy(), range(block[0], block[1] + 1))
        if labels is not None:
            w = w * np.where(train["q_teacher"].notna().to_numpy(), soft_w, 1.0)
        t0 = time.time()
        model = sm.fit(train, blocks, weights=w, ridge=ridge, offsets=use_off)
        heave_rate = float(pd.concat([pd.read_parquet(table_path(cfg, s), columns=["heave", "made"]) for s in train_s])
                           .query("heave")["made"].mean())
        for sub, (names, beta) in model.coef.items():
            coef_rows += [dict(block=f"{block[0]}-{block[1]}", sub=sub, name=n, beta=float(b)) for n, b in zip(names, beta)]
        for s in rated:
            f = pd.read_parquet(table_path(cfg, s))
            heave = f["heave"].to_numpy()
            eta = np.zeros(len(f))
            eta[~heave] = model.predict_raw(f[~heave])
            lev, shifts = sm.relevel(eta[~heave], f[~heave].reset_index(drop=True))
            q = np.full(len(f), heave_rate)
            q[~heave] = sm.probability(lev)
            res = dict(game_id=f["game_id"], action_number=f["action_number"], phase=f["phase"], q=q, eta_raw=eta)
            if vtab is not None:
                res["q_after"] = after_of(f, q, vtab)
            pd.DataFrame(res).to_parquet(out / f"{s}.parquet", index=False)
            level_rows.append(shifts)
        print(f"  {block[0]}-{block[1]}: trained on {len(train):,} attempts from {len(train_s)} seasons in "
              f"{time.time() - t0:.0f}s; priced {rated}", flush=True)
    pd.DataFrame(coef_rows).to_csv(out / "coef.csv", index=False)
    pd.concat(level_rows, ignore_index=True).to_csv(out / "levels.csv", index=False)
    (out / "spec.json").write_text(json.dumps(dict(blocks=blocks, offsets=use_off, max_rows=max_rows, ridge=ridge,
                                                   soft=soft, soft_weight=soft_w),
                                              indent=1), encoding="utf-8")
    print(f"done {name} in {time.time() - t_all:.0f}s")


if __name__ == "__main__":
    main()
