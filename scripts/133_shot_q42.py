"""Per-attempt shot quality for experiment 42: q_before and q_after, one parquet per season, for one priced arm.

    python scripts/133_shot_q42.py --arm=c_L7       --name=q42_L7
    python scripts/133_shot_q42.py --arm=c_L7_xgb38 --name=q42_LXr --alpha3=0.87

q_before  the arm's priced quality (data/shotq/<arm>/<season>.parquet, column q; scripts/132).
          --alpha3: the arm's THREES are recalibrated toward the current model, eta = eta_L7 + alpha3 * (eta_arm -
          eta_L7) on the raw logits, then relevelled per (season, distance band) from the season's regular-season makes
          as shotmodel.relevel (mode "season") does; twos and heaves keep the arm's q.  0.87 is the SALL optimum on the
          15 search seasons' threes (the trees spread recent threes too wide: DECISIONS.md, the verdict corrected).
          Unlocated threes (x = y = 0; 615 in all, 457 of them one arena in 1997, where the scorer left MISSED threes
          unlocated) get their season's mean quality of located regular-season threes: their missing spot carries the
          result, so they are priced as a three of unknown spot.
q_after   the owner's update (shotmodel.after_of): q + (made - q) * min(v / (q (1 - q)), 0.15), v by (sub-model,
          update band, clock class) from --vtab (default outputs/shottest/info_gap_v_clean_L7.csv, scripts/119 with the
          current model as base; with the trees as base v is 0.99-1.03 times it, so one table serves both arms);
          heaves keep q.

Writes data/shotq/<name>/<season>.parquet: game_id, action_number, phase, value, made, noloc, q_before, q_after (every
regular-season and playoff attempt, in the model table's order) and data/shotq/<name>/spec.json.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _cli import check_flags, flag  # noqa: E402
from eracoef import shotmodel as sm  # noqa: E402
from eracoef import shotsearch as ss  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SHOTQ = ROOT / "data" / "shotq"
TCOLS = ["season", "phase", "game_id", "action_number", "value", "made", "dist_xy", "corner3", "noloc", "heave", "sc_eff"]


def aligned(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    return bool((a["game_id"].to_numpy() == b["game_id"].to_numpy()).all()
                and (a["action_number"].to_numpy() == b["action_number"].to_numpy()).all())


def quality(t: pd.DataFrame, p: pd.DataFrame, base: pd.DataFrame | None, alpha3: float | None) -> np.ndarray:
    """q_before for one season: the arm's q; its threes relevelled per (season, distance band) from the LOCATED
    regular-season threes only (the unlocated ones are result-coded misses in 1997: left in, they pull the level of
    every located three in their band down, -1.3 points in 1997), recalibrated toward `base` first when alpha3 is
    given; then unlocated threes at the season's located-three mean."""
    q = p["q"].to_numpy(float).copy()
    three = (t["value"].to_numpy() == 3) & ~t["heave"].to_numpy(bool)
    noloc = t["noloc"].to_numpy(bool)
    eX = p["eta_raw"].to_numpy(float)
    eta = eX if alpha3 is None else base["eta_raw"].to_numpy(float) + alpha3 * (eX - base["eta_raw"].to_numpy(float))
    g = t[three]
    fit = ((g["phase"] == "RS") & ~g["noloc"]).to_numpy()
    lev = ss.relevel_fast(eta[three], g["season"].to_numpy(), ss.band_codes(g), g["made"].to_numpy(), fit)
    q[three] = sm.probability(lev)
    located = three & ~noloc & (t["phase"] == "RS").to_numpy()
    q[three & noloc] = float(q[located].mean())
    return q


def main():
    check_flags()
    arm, name = flag("arm"), flag("name")
    alpha3 = float(flag("alpha3")) if flag("alpha3", "") else None
    vpath = Path(flag("vtab", str(ROOT / "outputs" / "shottest" / "info_gap_v_clean_L7.csv")))
    vtab = pd.read_csv(vpath)
    out = SHOTQ / name
    out.mkdir(parents=True, exist_ok=True)
    t0, rows, patched = time.time(), {}, {}
    for s in range(1997, 2027):
        t = pd.read_parquet(SHOTQ / "_table" / f"{s}.parquet", columns=TCOLS)
        p = pd.read_parquet(SHOTQ / arm / f"{s}.parquet")
        base = pd.read_parquet(SHOTQ / "c_L7" / f"{s}.parquet") if alpha3 is not None else None
        if not aligned(p, t) or (base is not None and not aligned(base, t)):
            raise SystemExit(f"{arm} {s}: rows not aligned with the model table")
        q = quality(t, p, base, alpha3)
        qa = sm.after_of(t, q, vtab)
        res = pd.DataFrame(dict(game_id=t["game_id"], action_number=t["action_number"], phase=t["phase"],
                                value=t["value"].astype("int8"), made=t["made"].astype("int8"),
                                noloc=t["noloc"].astype(bool), q_before=q, q_after=qa))
        if res[["q_before", "q_after"]].isna().any().any() or not ((res.q_after > 0) & (res.q_after < 1)).all():
            raise SystemExit(f"{s}: bad quality values")
        res.to_parquet(out / f"{s}.parquet", index=False)
        rows[s] = len(res)
        patched[s] = int(((t["value"] == 3) & t["noloc"] & ~t["heave"]).sum())
    (out / "spec.json").write_text(json.dumps(dict(
        arm=arm, alpha3=alpha3, vtab=str(vpath), vtab_sha256=hashlib.sha256(vpath.read_bytes()).hexdigest(),
        cap=0.15, unlocated_threes_at_season_mean=patched, rows=rows, created=time.strftime("%Y-%m-%d %H:%M")),
        indent=1), encoding="utf-8")
    print(f"{name}: {sum(rows.values()):,} attempts, {sum(patched.values())} unlocated threes at their season mean, "
          f"{time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
