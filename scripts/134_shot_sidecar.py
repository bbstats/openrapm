"""Side tables that carry a per-attempt shot quality into the stint designs (experiment 42), without rebuilding stints.

    python scripts/134_shot_sidecar.py --q=q42_L7 [--seasons=1997,2010]

For one quality set (data/shotq/<q>/<season>.parquet from scripts/133: q_before, q_after per attempt) and every
season-phase, two tables:

  data/shotq/<q>/sidecar/<season>_<phase>.parquet   one row per row of data/stints/<season>_<phase>.parquet, in its
      order: xq3a_s{1-5,x}_{h,a} and xq2a_s{1-5,x}_{h,a}, the sums of q_AFTER over the threes / twos each lineup slot
      took in that stint (the same slots stints() files fg3a_s* under), and fg3a_s*_{h,a} / fg2a_s*_{h,a} counted
      from the same attempts.  Each attempt is placed by the shot frame's stint_no / stint_side / stint_slot
      (FRAME_VERSION 5, stints.stint_slots: exact, from the possession record the attempt closed into).
  data/shotq/<q>/shooters/<season>_<phase>.parquet  per game x shooter: xq2, xq3 = the sums of q_BEFORE over his twos /
      threes (his padded ratio is computed on quality before the result), with fg2a / fg3a / fg2m / fg3m.

Gates (a failure raises): every frame attempt has a quality; every placed attempt's stint row exists; the counted
fg3a_s* / fg2a_s* equal the cached stints' columns on EVERY row and slot; the shooter table's attempts and makes equal
data/stints/<season>_<phase>_shots.parquet for every game x shooter.  A stamp of the stints file (mtime, size) is
written beside each side table; xshoot refuses a side table whose stints file has changed since.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _cli import check_flags, flag  # noqa: E402
from eracoef.config import load_config, resolve  # noqa: E402
from eracoef.shotframe import FRAME_VERSION, frame_path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SLOTS6 = ("1", "2", "3", "4", "5", "x")
SIDES = ("h", "a")


def cols(prefix: str) -> list:
    return [f"{prefix}_s{s}_{side}" for side in SIDES for s in SLOTS6]


def side_table(st: pd.DataFrame, fr: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """The stints-aligned sums for one season-phase.  `fr`: the frame's attempts with q_after attached."""
    first = pd.Series(np.arange(len(st)), index=st.index).groupby(st["game_id"].to_numpy()).min()
    n_game = st.groupby("game_id").size()
    placed = fr[fr["stint_no"].to_numpy() >= 0]
    if (placed["stint_no"].to_numpy() >= n_game.reindex(placed["game_id"]).to_numpy()).any():
        raise SystemExit("an attempt's stint_no is past its game's stints")
    row = first.reindex(placed["game_id"]).to_numpy() + placed["stint_no"].to_numpy()
    key = placed["stint_side"].to_numpy(object)
    key = np.char.add(np.char.add(placed["stint_slot"].to_numpy(str), "_"), key.astype(str))
    out = {}
    for v, kind in ((3, "3"), (2, "2")):
        m = placed["value"].to_numpy() == v
        for side in SIDES:
            for s in SLOTS6:
                k = m & (key == f"{s}_{side}")
                cnt = np.bincount(row[k], minlength=len(st)).astype(float)
                qa = np.bincount(row[k], weights=placed["q_after"].to_numpy(float)[k], minlength=len(st))
                out[f"fg{kind}a_s{s}_{side}"] = cnt
                out[f"xq{kind}a_s{s}_{side}"] = qa
    T = pd.DataFrame(out)
    bad = {}
    for kind in ("3", "2"):
        for c in cols(f"fg{kind}a"):
            d = np.flatnonzero(T[c].to_numpy() != st[c].to_numpy(float))
            if len(d):
                bad[c] = len(d)
    return T, bad


def shooter_table(fr: pd.DataFrame, shots: pd.DataFrame) -> pd.DataFrame:
    g = fr.assign(xq2=np.where(fr["value"] == 2, fr["q_before"], 0.0), xq3=np.where(fr["value"] == 3, fr["q_before"], 0.0),
                  fg2a=(fr["value"] == 2).astype(int), fg3a=(fr["value"] == 3).astype(int),
                  fg2m=((fr["value"] == 2) & (fr["made"] == 1)).astype(int), fg3m=((fr["value"] == 3) & (fr["made"] == 1)).astype(int))
    T = g.groupby(["game_id", "shooter"], sort=False)[["xq2", "xq3", "fg2a", "fg3a", "fg2m", "fg3m"]].sum().reset_index()
    T = T.rename(columns={"shooter": "player_id"})
    chk = shots[["game_id", "player_id", "fg2a", "fg3a", "fg2m", "fg3m"]].merge(
        T[["game_id", "player_id", "fg2a", "fg3a", "fg2m", "fg3m"]], on=["game_id", "player_id"], how="outer",
        suffixes=("_st", "_q"), indicator=True)
    att = chk[["fg2a_st", "fg3a_st", "fg2m_st", "fg3m_st"]].fillna(0).to_numpy() != \
        chk[["fg2a_q", "fg3a_q", "fg2m_q", "fg3m_q"]].fillna(0).to_numpy()
    if att.any():
        raise SystemExit(f"shooter table: {int(att.any(axis=1).sum())} game x shooter rows differ from the stints' shots")
    return T


def main():
    check_flags()
    cfg = load_config()
    q = flag("q")
    seasons = [int(s) for s in flag("seasons", ",".join(str(s) for s in range(1997, 2027))).split(",") if s]
    qdir = ROOT / "data" / "shotq" / q
    (qdir / "sidecar").mkdir(parents=True, exist_ok=True)
    (qdir / "shooters").mkdir(parents=True, exist_ok=True)
    sdir = resolve(cfg, "stints")
    t0 = time.time()
    for s in seasons:
        qp = qdir / f"{s}.parquet"
        Q = pd.read_parquet(qp, columns=["game_id", "action_number", "phase", "value", "made", "q_before", "q_after"])
        Q = Q.rename(columns={"value": "value_q", "made": "made_q"})
        for ph in ("RS", "PO"):
            sp = sdir / f"{s}_{ph}.parquet"
            if not sp.exists():
                continue
            st = pd.read_parquet(sp)
            fp = frame_path(s, ph, cfg)
            fv = pd.read_parquet(fp, columns=["frame_version"])["frame_version"].unique()
            if list(fv) != [FRAME_VERSION]:
                raise SystemExit(f"{fp}: frame version {list(fv)}, need {FRAME_VERSION} (scripts/114)")
            fr = pd.read_parquet(fp, columns=["game_id", "action_number", "shooter", "value", "made", "stint_no",
                                              "stint_side", "stint_slot"])
            fr = fr.merge(Q[Q["phase"] == ph].drop(columns="phase"), on=["game_id", "action_number"], how="left",
                          suffixes=("", "_q"))
            if fr["q_after"].isna().any():
                raise SystemExit(f"{s} {ph}: {int(fr['q_after'].isna().sum())} attempts without a quality")
            # the quality's update read the table's result: the frame must agree on every attempt's value and result
            if ((fr["made"] != fr["made_q"]) | (fr["value"] != fr["value_q"])).any():
                raise SystemExit(f"{s} {ph}: the frame and {q} disagree on an attempt's value or result")
            T, bad = side_table(st, fr)
            if bad:
                raise SystemExit(f"{s} {ph}: slot counts differ from the cached stints: {bad}")
            T.to_parquet(qdir / "sidecar" / f"{s}_{ph}.parquet", index=False)
            stat, qstat, fstat = sp.stat(), qp.stat(), fp.stat()
            (qdir / "sidecar" / f"{s}_{ph}.stamp.json").write_text(
                json.dumps(dict(stints=sp.name, mtime_ns=stat.st_mtime_ns, size=stat.st_size, rows=len(st),
                                quality=qp.name, q_mtime_ns=qstat.st_mtime_ns, q_size=qstat.st_size,
                                frame=fp.name, frame_mtime_ns=fstat.st_mtime_ns, frame_version=FRAME_VERSION)),
                encoding="utf-8")
            shots = pd.read_parquet(sdir / f"{s}_{ph}_shots.parquet")
            shooter_table(fr, shots).to_parquet(qdir / "shooters" / f"{s}_{ph}.parquet", index=False)
            placed = float((fr["stint_no"] >= 0).mean())
            print(f"  {q} {s} {ph}: {len(st):,} stints, {len(fr):,} attempts ({placed:.4%} in a stint), "
                  f"slot counts and shooter totals exact", flush=True)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
