"""Does a playoff run tell you anything the regular season did not?

    python scratch/playoff_chain.py [--blocks=2018-2020,2021-2023,2024-2026] [--mults=0.5,1,2,4,8] [--csv]

The owner, 2026-09-09: *"SPM is the prior for reg season PI RAPM, then PI RAPM is the playoff prior for
playoff PI'' RAPM."*  `playoffs.playoff_system` builds that chain -- the regular-season rating becomes the
offset for a fit that sees only playoff possessions.  This asks whether the third stage is worth having.

**The split.**  Playoff games alternate A / B WITHIN each series (`design._order_games`), so fitting on one
half and scoring on the other holds the teams, the series and the lineups fixed and varies only the games.
That is a far cleaner comparison than anything available across seasons, where rosters turn over.

**The comparison.**  On the held-out half, predict each team-game's points from the ten players on the
floor, and score the same rows twice:

    regular season only    the RS rating, which is what the board publishes today
    + playoff update       the same rating moved by what the OTHER half of each series said

Only the level is refit in both, as the criterion always does.  Both directions are run (fit A score B,
fit B score A) and paired by block, so a block contributes two numbers built from disjoint games.

**What to expect.**  A three-season block holds about 47,700 playoff possessions against 734,000 regular
season -- 6.5% -- and half of that is 3%.  If the answer is "no", that is a real finding and not a null
result: it says playoff performance is regular-season performance plus noise, which is worth knowing before
anyone publishes a playoff board.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dataclasses import replace                                   # noqa: E402

from eracoef.config import load_config                            # noqa: E402
from eracoef.holdout import Context, predict_season, team_game_mse  # noqa: E402
from eracoef.investigate import attributable                       # noqa: E402
from eracoef.playoffs import playoff_system                       # noqa: E402
from eracoef.systems import registry                              # noqa: E402


def flag(name, default=None):
    hit = [a for a in sys.argv[1:] if a.startswith(f"--{name}=")]
    return hit[-1].split("=", 1)[1] if hit else default


def score_on(rat, wd, mask, slope: bool = False) -> tuple[float, float]:
    """Team-game MSE of a rating on the masked rows, level refit there (the criterion's convention).

    `slope=True` also refits ONE COEFFICIENT PER SIDE on the held-out rows.  That is deliberately generous
    to both arms and it is the check FINDINGS 36.5 demands: a per-side calibration curve is what the board
    ships with, and it absorbs any difference that is only amplitude.  If the playoff update still wins
    with the slopes free, what it is contributing is RANKING, not scale.
    """
    sub = wd.subset(mask)
    p = predict_season(rat, sub)
    if not slope:
        return team_game_mse(p.y, p.pred, p.poss, p.game_idx, p.is_home_off)
    X = np.column_stack([p.A, p.c_off, p.c_def])
    sw = np.sqrt(p.w)
    beta = np.linalg.lstsq(X * sw[:, None], p.y * sw, rcond=None)[0]
    return team_game_mse(p.y, X @ beta, p.poss, p.game_idx, p.is_home_off)


def credit_on(rat, wd, mask, lam: float = 2000.0) -> float:
    """The investigator's score on the held-out half: how much of the residual a player ridge can still put
    on named players.  Lower is better -- a rating that has already credited the right people leaves less
    behind.  The criterion is nearly blind to this (FINDINGS 26.5), so a candidate needs both."""
    sub = wd.subset(mask)
    p = predict_season(rat, sub)
    m = sub.spec.n_ps
    X = sub.X
    return float(attributable(X[:, :m], X[:, m:2 * m], p.y - p.pred, p.w, lam=lam)["player"])


def main():
    cfg = load_config()
    S = registry(cfg)
    ctx = Context.load(cfg)
    base = S[flag("base", "tune501_b7_pasto_pOD")]
    blocks = [[int(x) for x in b.split("-")] for b in
              (flag("blocks") or "2015-2017,2018-2020,2021-2023,2024-2026").split(",")]
    blocks = [list(range(b[0], b[-1] + 1)) for b in blocks]
    which = flag("held", "all")           # the 22.7 protocol: choose the penalty on one half of the blocks
    if which == "search":                 # and read it on the other
        blocks = blocks[::2]
    elif which == "confirm":
        blocks = blocks[1::2]
    mults = [float(m) for m in (flag("mults") or "0.5,1,2,4,8").split(",")]
    pd.set_option("display.width", 200)

    rows, t0 = [], time.time()
    for seasons in blocks:
        lab = f"{seasons[0]}-{seasons[-1]}"
        wd = ctx.design(seasons, base.off_target if base.off_target in ("pts",) else "pts", ("PO",))
        gh = wd.game_half[wd.rows["game_idx"].to_numpy()]
        for fit_half, score_half in (("A", "B"), ("B", "A")):
            mask = gh == score_half
            if not mask.any():
                continue
            # the regular-season rating alone: the board's own number, no playoff information at all
            rs_sys = playoff_system(base, f"po_{lab}", lam_po=base.lam)
            rs = rs_sys.prior_from.ratings(seasons, ctx)
            mse_rs, poss = score_on(rs, wd, mask)
            mse_rs_s, _ = score_on(rs, wd, mask, slope=True)
            cr_rs = credit_on(rs, wd, mask)
            for m in mults:
                sysm = replace(playoff_system(base, f"po_{lab}_x{m:g}", lam_po=float(base.lam) * m),
                               half=fit_half)
                sysm.prior_from._cache = rs_sys.prior_from._cache      # one regular-season fit per block
                po = sysm.fit(seasons, ctx)
                mse_po, _ = score_on(po, wd, mask)
                mse_po_s, _ = score_on(po, wd, mask, slope=True)
                rows.append(dict(block=lab, fit=fit_half, mult=m, poss=poss,
                                 mse_rs=mse_rs, mse_po=mse_po, diff=mse_po - mse_rs,
                                 mse_rs_s=mse_rs_s, mse_po_s=mse_po_s, diff_s=mse_po_s - mse_rs_s,
                                 cr_rs=cr_rs, cr_po=(cr_po := credit_on(po, wd, mask)),
                                 diff_cr=cr_po - cr_rs))
            print(f"  {lab} fit {fit_half} score {score_half}  ({time.time() - t0:.0f}s)", flush=True)
    D = pd.DataFrame(rows)

    print("\n=== the playoff update against the regular-season rating alone, on held-out playoff games")
    print("    negative = the playoff update predicts the other half of each series better\n")
    print(f"  {'penalty':>9} {'RS only':>10} {'+ playoff':>10} {'diff':>9} {'z':>7} {'won':>8}"
          f"   |{'slopes free':>12} {'z':>7}   |{'attribution':>12} {'z':>7}")
    out = []
    for m, g in D.groupby("mult"):
        w = g["poss"].to_numpy(float)
        rs = float(np.average(g.mse_rs, weights=w))
        po = float(np.average(g.mse_po, weights=w))
        d = g["diff"].to_numpy(float)
        z = float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))) if len(d) > 1 else float("nan")
        ds = g["diff_s"].to_numpy(float)
        zs = float(ds.mean() / (ds.std(ddof=1) / np.sqrt(len(ds)))) if len(ds) > 1 else float("nan")
        dif_s = float(np.average(g.mse_po_s, weights=w) - np.average(g.mse_rs_s, weights=w))
        dc = g["diff_cr"].to_numpy(float)
        zc = float(dc.mean() / (dc.std(ddof=1) / np.sqrt(len(dc)))) if len(dc) > 1 else float("nan")
        print(f"  x{m:<8g} {rs:10.3f} {po:10.3f} {po - rs:+9.3f} {z:7.2f} {int((d < 0).sum()):>3}/{len(d):<4}"
              f"   |{dif_s:+12.3f} {zs:7.2f}   |{float(np.mean(dc)):+12.3f} {zc:7.2f}")
        out.append(dict(mult=m, mse_rs=rs, mse_po=po, diff=po - rs, z=z, won=int((d < 0).sum()), n=len(d),
                        diff_slope=dif_s, z_slope=zs, won_slope=int((ds < 0).sum()),
                        diff_credit=float(np.mean(dc)), z_credit=zc))
    if "--csv" in sys.argv:
        p = Path(cfg["_root"]) / "outputs" / "csv"
        D.to_csv(p / "playoff_chain_rows.csv", index=False)
        pd.DataFrame(out).to_csv(p / "playoff_chain.csv", index=False)
        print(f"\nwrote {p / 'playoff_chain.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
