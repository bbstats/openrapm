"""Does a single-season prior-informed RAPM's ridge just pick the box score?

The owner, 2026-09-09: *"the goal here is to get single year PI rapm to actually give us a lambda that
doesn't pick one or the other (i think it usually just picks box score)"*.

This is the direct measurement of that.  In this estimator the prior is an OFFSET: the ridge fits the
residual of `y - X @ prior` and the rating is `prior + residual`.  So one number says whether the ridge
is doing anything --

    share = var(rating - prior) / var(rating)          possession-weighted, over players above a floor

-- and it runs from 0 (the rating IS the box prior, lambda has switched the plus-minus off) to 1 (no
prior at all).  Swept over the ridge penalty and over the TARGET, on single-season fits, it says whether
a less noisy target buys the on-court evidence any more weight.

    python scratch/lamshare.py [--seasons=2014,2019,2024] [--targets=xpts_ft,pts,tlxft3o_O]
        [--mults=0.03,0.0625,0.125,0.25,0.5,1,2,4] [--floor=500] [--kernel=00|52] [--csv]

`--targets` takes an offense/defense pair as `off:def`, or one name for both; the bare names below are
expanded.  Nothing here is a criterion -- it is a decomposition, and it is fast (a fit is under a
second).  Read it beside `scratch/inseason_run.py`, which says which lambda actually PREDICTS best.
"""
from __future__ import annotations

import sys
from dataclasses import replace as _replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eracoef.config import load_config          # noqa: E402
from eracoef.holdout import Context             # noqa: E402
from eracoef.systems import registry            # noqa: E402

# the named target pairs worth comparing: the shipped one, the raw one, and the luck-adjusted rungs
PAIRS = {
    "ship": ("xpts_ft", "x3def"),          # what the board uses
    "pts": ("pts", "pts"),                 # no luck adjustment at all
    "xft": ("xpts_ft", "xpts_ft"),         # free-throw adjustment on both sides
    "tl3": ("tlxft3o", "x3def"),           # + the team-game three-point adjustment on offense (FINDINGS 33)
    "tl3b": ("tlxft3o", "tlxft3o"),        # the same on both sides
}


def _flag(name, default=None):
    hit = [a for a in sys.argv[1:] if a.startswith(f"--{name}=")]
    return hit[-1].split("=", 1)[1] if hit else default


def _wsd(x, w) -> float:
    x, w = np.asarray(x, float), np.asarray(w, float)
    if w.sum() <= 0:
        return float("nan")
    m = np.average(x, weights=w)
    return float(np.sqrt(np.average((x - m) ** 2, weights=w)))


def train_for(system, season: int, cfg) -> list:
    """The seasons this system's kernel actually wants, anchored at `season`.

    A KERNEL system weights the anchor's neighbours (ks52 = {0: 1, -1: 0.5, -2: 0.25}), so handing it
    `[season]` alone silently turns it into the one-season kernel -- which is how this script first
    reported ks52 and ks00 as identical.  Build the list the kernel names.
    """
    kern = getattr(system, "kernel", None) or {0: 1.0}
    first = int(cfg["first_season"])
    return sorted(int(season) + int(o) for o in kern if int(season) + int(o) >= first)


def decompose(system, season: int, ctx, floor: float, train=None) -> dict:
    """One fit anchored at `season`, split into the prior it was handed and the residual the ridge kept."""
    r = system.fit(train or [int(season)], ctx).df
    r = r[r.poss >= floor]
    out = dict(season=int(season), n=len(r))
    for side in ("o", "d"):
        pc = f"prior_{side}"
        prior = r[pc].to_numpy(float) if pc in r.columns else np.zeros(len(r))
        rating = r[side].to_numpy(float)
        w = r["poss"].to_numpy(float)
        sd_p, sd_r = _wsd(prior, w), _wsd(rating - prior, w)
        sd_t = _wsd(rating, w)
        out[f"sd_prior_{side}"] = sd_p
        out[f"sd_resid_{side}"] = sd_r
        out[f"sd_rating_{side}"] = sd_t
        # the share of the RATING's variance that the on-court evidence put there.  Not sd_r^2/sd_t^2 by
        # itself: prior and residual are correlated, so the identity that closes is
        #   var(rating) = var(prior) + 2 cov(prior, resid) + var(resid)
        # and the residual's share is reported as var(resid)/var(rating) with the covariance printed
        # beside it, so a share above 1 (the ridge fighting the prior) is visible rather than hidden.
        out[f"share_{side}"] = (sd_r / sd_t) ** 2 if sd_t > 0 else float("nan")
        out[f"corr_{side}"] = (float(np.corrcoef(prior, rating)[0, 1])
                               if sd_p > 0 and sd_t > 0 else float("nan"))
        out[f"corr_pr_{side}"] = (float(np.corrcoef(prior, rating - prior)[0, 1])
                                  if sd_p > 0 and sd_r > 0 else float("nan"))
    return out


def main():
    cfg = load_config()
    S = registry(cfg)
    ctx = Context.load(cfg)
    seasons = [int(s) for s in _flag("seasons", "2014,2019,2024").split(",")]
    # a mult is either "0.5" (both sides) or "0.5:0.25" (offense:defense).  Offense and defense have
    # ALWAYS had separate penalties here -- `lam` is the offensive one and `lam_ratio` multiplies it for
    # defense -- so this sweeps the two independently instead of dragging them together.
    mults = []
    for tok in _flag("mults", "0.03,0.0625,0.125,0.25,0.5,1,2,4").split(","):
        a, _, b = tok.partition(":")
        mults.append((float(a), float(b) if b else float(a)))
    floor = float(_flag("floor", "500"))
    kern = _flag("kernel", "00")
    base = S[f"ks{kern}"]
    lam0 = float(base.lam)
    names = _flag("targets", "ship,pts,tl3").split(",")

    pd.set_option("display.width", 260)
    rows = []
    for nm in names:
        off, dfn = PAIRS[nm] if nm in PAIRS else (nm.split(":") * 2)[:2]
        for a, b in mults:
            s = _replace(base, name=f"{nm}_x{a:g}_{b:g}", lam=lam0 * a, lam_ratio=b / a,
                         off_target=off, def_target=dfn)
            for season in seasons:
                tr = train_for(s, season, cfg)
                rows.append(dict(target=nm, off=off, dfn=dfn, mult=a, mult_d=b,
                                 lam=lam0 * a, lam_d=lam0 * b, n_train=len(tr),
                                 **decompose(s, season, ctx, floor, train=tr)))
                print(f"  {nm:6s} O x{a:<6g} D x{b:<6g} {season}  done", flush=True)
    D = pd.DataFrame(rows)

    print(f"\n=== single-season fits, kernel ks{kern}, {len(seasons)} seasons, players with {floor:.0f}+ possessions")
    print("    share = var(rating - prior) / var(rating): 0 = the rating IS the box prior, 1 = no prior")
    print("    corr_pr = correlation of the prior with the residual (negative = the ridge undoing the prior)\n")
    g = D.groupby(["target", "mult", "mult_d"]).agg(
        lam=("lam", "first"), lam_d=("lam_d", "first"), n=("n", "mean"),
        sd_prior_o=("sd_prior_o", "mean"), sd_resid_o=("sd_resid_o", "mean"), share_o=("share_o", "mean"),
        corr_pr_o=("corr_pr_o", "mean"),
        sd_prior_d=("sd_prior_d", "mean"), sd_resid_d=("sd_resid_d", "mean"), share_d=("share_d", "mean"),
        corr_pr_d=("corr_pr_d", "mean")).reset_index()
    print(g.round(3).to_string(index=False))

    print("\n=== the shipped operating point (mult = 1 is what ks<k> uses; the board's block lambda)")
    ship = g[g["mult"] == 1.0]
    for _, r in ship.iterrows():
        print(f"  {r['target']:6s} lambda {r['lam']:9.0f}   offense: prior sd {r['sd_prior_o']:.2f}, "
              f"residual sd {r['sd_resid_o']:.2f}, share {r['share_o'] * 100:4.1f}%   "
              f"defense: prior sd {r['sd_prior_d']:.2f}, residual sd {r['sd_resid_d']:.2f}, "
              f"share {r['share_d'] * 100:4.1f}%")

    if _flag("csv") is not None or "--csv" in sys.argv:
        out = Path(cfg["_root"]) / "outputs" / "csv" / "lamshare.csv"
        D.to_csv(out, index=False)
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
