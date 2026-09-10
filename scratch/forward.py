"""Does the luck adjustment actually predict better? The direct test, with no ratings pipeline in it.

    python scratch/forward.py [--first=1997] [--last=2026] [--kpool=other|own] [--csv]

The owner, 2026-09-09, on why the last three attempts were hard to read: they all went straight to the
ratings criterion, which is noisy and mixes up two different questions.  This asks the narrow one.

For every team-season: build its offensive and defensive profile from the FIRST half of its games, then
predict each of its SECOND-half games' actual points per 100.  About 900 team-seasons and nothing but
arithmetic in between.  Two arms:

    raw         the team's realised first-half rating, straight
    adjusted    the same rebuilt through `teamloo.possession_points` with EVERY component shrunk by its
                own estimated constant -- turnovers, offensive rebounds, free-throw rate and percentage,
                the shot mix, and the make rate at the rim, on long twos and from three

**The scoring is the part that makes this a real test.**  Shrinking any noisy predictor toward the mean
lowers its forward error, so a naive MSE comparison would hand the win to the adjusted arm for a reason
that has nothing to do with components.  Instead each arm is scored by a leave-one-SEASON-out regression
of the held-out season's second-half results on that arm alone.  A regression absorbs any global rescaling,
so the raw arm and a globally shrunk raw arm score IDENTICALLY, and the only thing the adjusted arm can win
on is the component structure -- which is the claim being tested.  The third table puts both arms in one
regression and asks whether the adjusted one still carries anything once raw is already in.

`--kpool=other` (the default) estimates each constant from the OTHER seasons, `own` from this season's
first half alone -- the fully self-contained version, which is what a one-season product would have.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eracoef import teamloo as tl                 # noqa: E402
from eracoef.config import load_config            # noqa: E402

SIDES = {"off": "team_id", "def": "opp_id"}


def flag(name, default=None):
    hit = [a for a in sys.argv[1:] if a.startswith(f"--{name}=")]
    return hit[-1].split("=", 1)[1] if hit else default


def halves(tg: pd.DataFrame, key: str) -> tuple[np.ndarray, np.ndarray]:
    """A boolean mask for each team's first half of its own games, and for its second."""
    r = tg.groupby(key)["rank"].rank(method="first").to_numpy()
    g = tg.groupby(key)["rank"].transform("count").to_numpy()
    first = r <= np.ceil(g / 2.0)
    return first, ~first


def totals(tg: pd.DataFrame, key: str, mask: np.ndarray) -> pd.DataFrame:
    """Summed counters per team over the masked rows, in the (m_, n_) shape the model wants."""
    cols = [f"{a}_{c}" for c in tl.MODEL_RATES for a in ("m", "n")]
    d = tg.loc[mask, [key, *cols, "pts", "poss"]].groupby(key).sum()
    return d


def season_constants(tg: pd.DataFrame) -> pd.DataFrame:
    """One season's variance pieces per component and side.  Estimated PER SEASON and pooled afterwards,
    never by concatenating seasons: a team's true rate moves year to year, so grouping its 1997 and 2024
    games together measures roster churn as if it were noise (and the leave-one-out step then builds an
    82 x 82 table per team-season instead of one per team, which is what made the first version crawl)."""
    loo = tl.loo_rates(tg, rates=tl.MODEL_RATES)
    return tl.k_table(tg, loo, rates=tl.MODEL_RATES)


def pool_constants(kts: list, side: str) -> dict:
    """Attempt-weighted pool of `tau2_half` and `within` across seasons, then k = within / tau2."""
    K = pd.concat(kts, ignore_index=True)
    K = K[K.side == side]
    out = {}
    for c, g in K.groupby("component"):
        w = g["att"].to_numpy(float)
        tau2 = float(np.average(g["tau2_half"].to_numpy(float), weights=w))
        within = float(np.average(g["within"].to_numpy(float), weights=w))
        out[c] = within / tau2 if tau2 > 1e-9 else tl.K_CAP
    return out


def league_of(tg: pd.DataFrame) -> dict:
    return {c: float(tg[f"m_{c}"].sum() / max(tg[f"n_{c}"].sum(), 1.0)) for c in tl.MODEL_RATES}


def build(seasons, cfg, kpool: str) -> pd.DataFrame:
    """One row per (season, team, side): the two first-half predictors and the second-half truth."""
    frames = {s: tl.team_games(s, cfg) for s in seasons}
    kts = {s: season_constants(frames[s]) for s in seasons}
    own = {}
    if kpool == "own":
        for s in seasons:
            tg = frames[s]
            first, _ = halves(tg, "team_id")          # the same split both sides, so one frame serves
            own[s] = season_constants(tg[first].reset_index(drop=True))
    ks = {}
    for s in seasons:
        for side in SIDES:
            src = [own[s]] if kpool == "own" else [kts[t] for t in seasons if t != s]
            ks[(s, side)] = (pool_constants(src, side), league_of(frames[s]))
    rows = []
    for s in seasons:
        tg = frames[s]
        for side, key in SIDES.items():
            first, second = halves(tg, key)
            A, B = totals(tg, key, first), totals(tg, key, second)
            k, lg = ks[(s, side)]
            tot = {c: A[c].to_numpy() for c in A.columns}
            raw = tl.possession_points(tl.rates_from_totals(tot))
            adj = tl.possession_points(tl.rates_from_totals(tot, shrink_k=k, league=lg))
            # and each component ON ITS OWN, everything else left as it happened: the only way to see
            # which parts of a possession are luck worth removing and which are skill worth keeping
            # the principled arm: shrink the OUTCOME rates, leave the STYLE ones as they happened.  The
            # split is a priori (a team chooses its shot mix and does not choose whether shots drop),
            # not read off the results below.
            ksel = {x: (k[x] if x in tl.OUTCOME_RATES else 0.0) for x in tl.MODEL_RATES}
            sel = tl.possession_points(tl.rates_from_totals(tot, shrink_k=ksel, league=lg))
            one = {"selective": sel["pts100"]}
            for c in tl.MODEL_RATES:
                k1 = {x: (k[x] if x == c else 0.0) for x in tl.MODEL_RATES}
                one[f"adj_{c}"] = tl.possession_points(tl.rates_from_totals(tot, shrink_k=k1, league=lg))["pts100"]
            j = B.reindex(A.index)
            rows.append(pd.DataFrame(dict(
                season=s, side=side, team=A.index.to_numpy(),
                raw=raw["pts100"], adjusted=adj["pts100"], **one,
                first_actual=100.0 * A["pts"].to_numpy() / np.maximum(A["poss"].to_numpy(), 1.0),
                truth=100.0 * j["pts"].to_numpy() / np.maximum(j["poss"].to_numpy(), 1.0),
                w=j["poss"].to_numpy())))
    return pd.concat(rows, ignore_index=True).dropna(subset=["truth"])


def loso_mse(D: pd.DataFrame, cols: list) -> tuple[float, np.ndarray]:
    """Held-out weighted MSE of a leave-one-season-out regression of `truth` on `cols` (+ intercept),
    and the per-season error so the two arms can be paired."""
    per, tot_e, tot_w = {}, 0.0, 0.0
    for s in sorted(D.season.unique()):
        tr, te = D[D.season != s], D[D.season == s]
        X = np.column_stack([np.ones(len(tr))] + [tr[c].to_numpy(float) for c in cols])
        sw = np.sqrt(tr.w.to_numpy(float))
        beta = np.linalg.lstsq(X * sw[:, None], tr.truth.to_numpy(float) * sw, rcond=None)[0]
        Xt = np.column_stack([np.ones(len(te))] + [te[c].to_numpy(float) for c in cols])
        e = (te.truth.to_numpy(float) - Xt @ beta) ** 2
        w = te.w.to_numpy(float)
        per[s] = float((e * w).sum() / w.sum())
        tot_e += float((e * w).sum())
        tot_w += float(w.sum())
    return tot_e / tot_w, pd.Series(per)


def main():
    cfg = load_config()
    first = int(flag("first", cfg["first_season"]))
    last = int(flag("last", cfg["last_season"]))
    kpool = flag("kpool", "other")
    seasons = list(range(first, last + 1))
    pd.set_option("display.width", 220)

    D = build(seasons, cfg, kpool)
    print(f"\n=== forward test: first half predicts second half, {len(D)} team-seasons, "
          f"constants pooled from the {kpool} seasons")
    print("    scored by leave-one-season-out regression, so a global rescale cannot win it\n")
    print(f"  {'side':<5} {'arm':<20} {'held-out MSE':>13} {'vs raw':>9} {'z':>7} {'seasons won':>12}")
    out = []
    for side in ("off", "def"):
        S = D[D.side == side]
        mse_r, per_r = loso_mse(S, ["raw"])
        mse_a, per_a = loso_mse(S, ["adjusted"])
        d = per_a - per_r
        z = float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))) if len(d) > 1 else float("nan")
        print(f"  {side:<5} {'raw':<20} {mse_r:13.4f} {'':>9} {'':>7} {'':>12}")
        print(f"  {side:<5} {'adjusted':<20} {mse_a:13.4f} {mse_a - mse_r:+9.4f} {z:7.2f} "
              f"{int((d < 0).sum()):>5}/{len(d):<6}")
        both, per_b = loso_mse(S, ["raw", "adjusted"])
        db = per_b - per_r
        zb = float(db.mean() / (db.std(ddof=1) / np.sqrt(len(db)))) if len(db) > 1 else float("nan")
        print(f"  {side:<5} {'both together':<20} {both:13.4f} {both - mse_r:+9.4f} {zb:7.2f} "
              f"{int((db < 0).sum()):>5}/{len(db):<6}")
        for nm in ("selective",):
            ms, ps = loso_mse(S, [nm])
            ds = ps - per_r
            zs = float(ds.mean() / (ds.std(ddof=1) / np.sqrt(len(ds))))
            print(f"  {side:<5} {nm + ' (outcomes only)':<20} {ms:13.4f} {ms - mse_r:+9.4f} {zs:7.2f} "
                  f"{int((ds < 0).sum()):>5}/{len(ds):<6}")
            mb, pb = loso_mse(S, ["raw", nm])
            dbs = pb - per_r
            zbs = float(dbs.mean() / (dbs.std(ddof=1) / np.sqrt(len(dbs))))
            print(f"  {side:<5} {'  + raw beside it':<20} {mb:13.4f} {mb - mse_r:+9.4f} {zbs:7.2f} "
                  f"{int((dbs < 0).sum()):>5}/{len(dbs):<6}")
        for c in tl.MODEL_RATES:
            m1, p1 = loso_mse(S, [f"adj_{c}"])
            d1 = p1 - per_r
            z1 = float(d1.mean() / (d1.std(ddof=1) / np.sqrt(len(d1)))) if len(d1) > 1 else float("nan")
            print(f"  {side:<5} {'  only ' + c:<20} {m1:13.4f} {m1 - mse_r:+9.4f} {z1:7.2f} "
                  f"{int((d1 < 0).sum()):>5}/{len(d1):<6}")
        # what the pooled fit makes of the two arms side by side
        X = np.column_stack([np.ones(len(S)), S.raw.to_numpy(float), S.adjusted.to_numpy(float)])
        sw = np.sqrt(S.w.to_numpy(float))
        b = np.linalg.lstsq(X * sw[:, None], S.truth.to_numpy(float) * sw, rcond=None)[0]
        print(f"  {side:<5} {'  pooled coefficients':<20} raw {b[1]:+.3f}   adjusted {b[2]:+.3f}")
        out.append(dict(side=side, mse_raw=mse_r, mse_adj=mse_a, mse_both=both, z=z,
                        wins=int((d < 0).sum()), n=len(d), coef_raw=b[1], coef_adj=b[2]))
        print()
    if "--csv" in sys.argv:
        p = Path(cfg["_root"]) / "outputs" / "csv"
        D.to_csv(p / f"forward_rows_{kpool}.csv", index=False)
        pd.DataFrame(out).to_csv(p / f"forward_{kpool}.csv", index=False)
        print(f"wrote {p / f'forward_{kpool}.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
