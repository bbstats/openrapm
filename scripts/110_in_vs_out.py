"""Experiment 38: what makes a rating better IN-SEASON versus better for OTHER seasons.

    python scripts/110_in_vs_out.py [--parts=decisions,stats,stability]

The owner (2026-10-05): "I would like to find out what stats / model decisions make one model better at in-season vs
better at 'outside seasons'."  Development seasons only (heldoutprior.LOCKBOX untouched).  Three parts:

  decisions  each model decision's effect on within-season error (3/4 of a season rated, the rest scored) and on
             year-over-year error, paired by season: the ridge penalty (vanilla RAPM), the prior's weight and penalty
             (RAPM on the linear box prior), and the scorecard's decisions (adding a prior, boosted vs linear prior,
             OpenRAPM's prior shrink, the swap adjustment)
  stats      every one of the 65 inputs added ALONE to the linear-box-prior baseline (B3), its shape fit on
             within-season held-out games and its scale year over year (scripts/108's procedure, nested by season)
  stability  each input's within-season split-half reliability (two disjoint halves of a season, the `within2`
             folds) and its year-to-year correlation (season S vs S+1), and whether "reliable within a season but
             unstable across seasons" predicts "helps in-season, not out of season"

Writes outputs/heldout/in_vs_out_{decisions,stats}.csv.
"""
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _cli import check_flags, flag  # noqa: E402
from eracoef import boxspm as bx  # noqa: E402
from eracoef import heldoutprior as hp  # noqa: E402
from eracoef import scorecard as sc  # noqa: E402
from eracoef.boxtable import season_box  # noqa: E402
from eracoef.config import load_config  # noqa: E402
from eracoef.exposure import K_MAX  # noqa: E402

OUT = ROOT / "outputs" / "heldout"
SEASONS = list(range(1997, 2027))
DEV = [s for s in SEASONS if s not in hp.LOCKBOX]
MIN_HALF_POSS, MIN_SEASON_POSS = 300.0, 600.0


def _borrow(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def armse(mse):
    return float(np.sqrt(mse) * np.sqrt(2 / np.pi))


def paired(per_season: pd.DataFrame, a: str, b: str) -> dict:
    """b minus a, per season, then the paired t over seasons."""
    d = (per_season[b] - per_season[a]).dropna()
    se = d.std(ddof=1) / np.sqrt(len(d))
    return dict(diff=float(d.mean()), t=float(d.mean() / se) if se > 0 else np.nan, n=len(d), better=int((d < 0).sum()))


# ------------------------------------------------------------------------------------------ part 1: decisions
def decisions() -> pd.DataFrame:
    rows = []
    # the penalty and the prior weight, from 103's stored quadratics (development seasons; yoy pairs inside them)
    w = pd.read_parquet(ROOT / "outputs" / "within" / "within" / "ridge_grid.parquet")
    w = w[w.season.isin(DEV)]
    y = pd.read_parquet(ROOT / "outputs" / "csv" / "ridge_grid_yoy.parquet")
    y = y[y.rated.isin(DEV) & y.season.isin(DEV)]
    curves = []
    for (lam, ratio), gw in w.groupby(["lam_o", "ratio"]):
        gy = y[(y.lam_o == lam) & np.isclose(y.ratio, ratio)]
        sw_w, sw_y = gw.sw.sum(), gy.sw.sum()
        b2_w, b2_y = gw.g00.sum() / sw_w, gy.g00.sum() / sw_y
        # B3 at this penalty pair: the multipliers best for each target, and each set scored on the other target
        def best(g):
            s = g[["g0o", "g0d", "goo", "god", "gdd"]].sum()
            return np.linalg.solve([[s.goo, s.god], [s.god, s.gdd]], [s.g0o, s.g0d])

        def err(g, m):
            s = g[["g00", "g0o", "g0d", "goo", "god", "gdd"]].sum()
            return (s.g00 - 2 * (m[0] * s.g0o + m[1] * s.g0d) + m[0] ** 2 * s.goo + 2 * m[0] * m[1] * s.god
                    + m[1] ** 2 * s.gdd) / g.sw.sum()
        mw, my = best(gw), best(gy)
        curves.append(dict(lam_o=lam, ratio=ratio, b2_within=armse(b2_w), b2_yoy=armse(b2_y),
                           b3_within=armse(err(gw, mw)), b3_yoy=armse(err(gy, my)), m_within_o=mw[0], m_within_d=mw[1],
                           m_yoy_o=my[0], m_yoy_d=my[1], b3_yoy_at_within_m=armse(err(gy, mw)),
                           b3_within_at_yoy_m=armse(err(gw, my))))
    curves = pd.DataFrame(curves)
    curves.to_csv(OUT / "in_vs_out_penalty_curves.csv", index=False)
    for label, col in (("vanilla RAPM", "b2"), ("RAPM on the linear box prior", "b3")):
        iw, iy = curves[f"{col}_within"].idxmin(), curves[f"{col}_yoy"].idxmin()
        rows.append(dict(decision=f"{label}: best penalty (offense / defense ratio)",
                         in_season=f"{curves.lam_o[iw]:,.0f} / {curves.ratio[iw]}",
                         out_of_season=f"{curves.lam_o[iy]:,.0f} / {curves.ratio[iy]}",
                         cost_in_season_of_out_choice=curves[f"{col}_within"][iy] - curves[f"{col}_within"][iw],
                         cost_out_of_season_of_in_choice=curves[f"{col}_yoy"][iw] - curves[f"{col}_yoy"][iy]))
    iw = curves.b3_within.idxmin()
    iy = curves.b3_yoy.idxmin()
    rows.append(dict(decision="RAPM on the linear box prior: prior multiplier offense / defense",
                     in_season=f"{curves.m_within_o[iw]:.2f} / {curves.m_within_d[iw]:.2f}",
                     out_of_season=f"{curves.m_yoy_o[iy]:.2f} / {curves.m_yoy_d[iy]:.2f}"))
    # the scorecard's decisions, paired by development season
    q = pd.read_parquet(ROOT / "outputs" / "scorecard" / "quadratics.parquet")
    q["mse"] = sc.error_at(q, 1.0, 1.0)
    win = q[(q.test == "within") & q.season.isin(DEV)].groupby(["season", "system"]).mse.mean().unstack()
    yy = q[q.test == "yoy"].copy()
    yy["rated"] = yy.season + np.where(yy.key.str.endswith("prev"), -1, 1)
    yy = yy[yy.season.isin(DEV) & yy.rated.isin(DEV)].groupby(["season", "system"]).mse.mean().unstack()
    for label, a, b in (("add a linear box-score prior to vanilla RAPM", "B2", "B3"),
                        ("boosted (OpenRAPM) prior instead of the linear one", "B3", "B4"),
                        ("OpenRAPM's prior shrink", "O13", "O"),
                        ("the swap adjustment (OpenRAPM, x0.5)", "O", "O+swap0.5")):
        pw = paired(win, a, b)
        b_y = "incumbent" if (b == "O+swap0.5") else b
        a_y = "O" if (a == "O" and b == "O+swap0.5") else a
        py = paired(yy, a_y, b_y) if {a_y, b_y} <= set(yy.columns) else dict(diff=np.nan, t=np.nan, n=0, better=0)
        rows.append(dict(decision=label, in_season=f"{pw['diff']:+.3f} (t {pw['t']:+.1f}, {pw['better']}/{pw['n']})",
                         out_of_season=f"{py['diff']:+.3f} (t {py['t']:+.1f}, {py['better']}/{py['n']})"))
    return pd.DataFrame(rows), curves


# ------------------------------------------------------------------------------------------ part 2: stats
def stats() -> pd.DataFrame:
    S108 = _borrow("_select108_for_110", "108_heldout_prior_select.py")
    D = S108.Data()
    t0 = time.time()

    def run(cfg):
        rows = []
        for H in D.dev:
            scored = [t for t in (H - 1, H + 1) if t in D.dev]
            rows.append(S108.evaluate(D, cfg, H, scored))
        r = pd.DataFrame(rows)
        r["within"] = r[[f"{t}_home" for t in S108.FIT_SIZES]].mean(axis=1)
        r["yoy"] = r[[c for c in ("yoy_prev", "yoy_next") if c in r.columns]].mean(axis=1)
        return r.set_index("season")

    ref = run(S108.Config("B3 (anchor only)", []))
    out = []
    for col in [c for c in D.columns if c != "b1"]:
        r = run(S108.Config(f"+{col}", columns=[col]))
        row = dict(stat=col, group=next(g for g, cs in hp.GROUPS.items() if col in cs))
        for metric in ("within", "yoy"):
            d = (r[metric] - ref[metric]).dropna()
            se = d.std(ddof=1) / np.sqrt(len(d))
            row[f"{metric}_diff"] = float(d.mean())
            row[f"{metric}_t"] = float(d.mean() / se) if se > 0 else np.nan
            row[f"{metric}_better"] = int((d < 0).sum())
            row[f"{metric}_n"] = len(d)
        out.append(row)
    print(f"  {len(out)} stats evaluated ({time.time() - t0:.0f}s)", flush=True)
    return pd.DataFrame(out)


# ------------------------------------------------------------------------------------------ part 3: stability
def stat_frame(players: pd.DataFrame, b1: pd.DataFrame, bxin: pd.DataFrame, columns) -> pd.DataFrame:
    rows = players[players.side == "O"].drop_duplicates("player_id").set_index("player_id")
    f = rows[[c for c in columns if c in rows.columns]].astype(float).copy()
    f["booster"] = rows.prior_raw_off.astype(float)
    f["b1"] = b1.drop_duplicates("player_id").set_index("player_id").o
    x = bxin.set_index("player_id")
    for c in bx.INPUTS:
        f["bx_" + (c[2:] if c.startswith("r_") else c)] = x[c]
    f["_poss"] = rows.poss_off.astype(float)
    return f


def corr(a: pd.DataFrame, b: pd.DataFrame, cols, min_poss) -> dict:
    j = a.join(b, lsuffix="_a", rsuffix="_b", how="inner")
    j = j[(j["_poss_a"] >= min_poss) & (j["_poss_b"] >= min_poss)]
    out = {}
    for c in cols:
        x, y = j[f"{c}_a"], j[f"{c}_b"]
        ok = x.notna() & y.notna()
        out[c] = float(np.corrcoef(x[ok], y[ok])[0, 1]) if ok.sum() > 20 and x[ok].std() > 0 and y[ok].std() > 0 else np.nan
    return out


def stability() -> pd.DataFrame:
    S102 = _borrow("_box102_for_110", "102_box_spm.py")
    cfg = load_config(ROOT / "config.yaml")
    cols = [c for c in hp.columns_of(list(hp.GROUPS))]
    box = {s: bx.with_possessions(season_box([s], ("RS", "PO"), cfg)) for s in SEASONS}
    k_season = {s: bx.padding_k(box[s]) for s in SEASONS}

    def k_for(H):
        k = np.nanmean(np.array([k_season[s] for s in SEASONS if abs(s - H) > 1]), axis=0)
        return np.where(np.isfinite(k), k, K_MAX)

    t0 = time.time()
    within = []
    d2 = ROOT / "outputs" / "within" / "within2"
    b1f = pd.read_parquet(d2 / "baseline_b1.parquet")
    for S in DEV:
        assignment = pd.read_parquet(d2 / f"folds_{S}.parquet")
        assignment.index = assignment.index.astype(str)
        kS = k_for(S)
        for r in range(5):
            halves = []
            for f in (0, 1):
                stem = f"{S}_r{r}_f{f}"
                if not (d2 / f"players_{stem}.parquet").exists():
                    break
                fit_ids, _, _ = S102.fold_fit_games(d2, stem, assignment)
                bxin = bx.player_inputs(box[S][box[S].game_id.isin(fit_ids)], kS)
                halves.append(stat_frame(pd.read_parquet(d2 / f"players_{stem}.parquet"), b1f[b1f.key == stem],
                                         bxin, cols))
            if len(halves) == 2:
                within.append(corr(halves[0], halves[1], cols, MIN_HALF_POSS))
    print(f"  within-season split halves: {len(within)} deals ({time.time() - t0:.0f}s)", flush=True)
    b1s = pd.read_parquet(ROOT / "outputs" / "season_ratings_base_b1.parquet")
    b1s = b1s.assign(o=b1s.rating_off)
    whole = {}
    for S in DEV:
        whole[S] = stat_frame(pd.read_parquet(ROOT / "outputs" / "within" / "within" / f"players_{S}_all.parquet"),
                              b1s[b1s.season == S], bx.player_inputs(box[S], k_for(S)), cols)
    across = [corr(whole[S], whole[S + 1], cols, MIN_SEASON_POSS) for S in DEV if S + 1 in whole]
    w, a = pd.DataFrame(within).mean(), pd.DataFrame(across).mean()
    return pd.DataFrame({"stat": cols, "within_season_reliability": [w.get(c) for c in cols],
                         "year_to_year_correlation": [a.get(c) for c in cols]})


def main():
    check_flags()
    parts = flag("parts", "decisions,stats,stability").split(",")
    pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_rows", 200)
    if "decisions" in parts:
        dec, curves = decisions()
        dec.to_csv(OUT / "in_vs_out_decisions.csv", index=False)
        print("\n=== model decisions: in-season vs out of season")
        print(dec.to_string(index=False))
        print("\n=== penalty curves (typical miss, pts/100; development seasons)")
        print(curves.round(4).to_string(index=False))
    if "stats" in parts:
        st = stats()
        st.to_csv(OUT / "in_vs_out_stats.csv", index=False)
    if "stability" in parts:
        sb = stability()
        st = pd.read_csv(OUT / "in_vs_out_stats.csv").merge(sb, on="stat", how="left")
        st["specificity"] = st.within_season_reliability - st.year_to_year_correlation
        st["transfer_gap"] = st.yoy_diff - st.within_diff        # > 0: helps in-season more than out of season
        st.to_csv(OUT / "in_vs_out_stats.csv", index=False)
        ok = st.dropna(subset=["specificity", "transfer_gap"])
        X = np.column_stack([np.ones(len(ok)), ok.specificity])
        beta = np.linalg.lstsq(X, ok.transfer_gap, rcond=None)[0]
        r = np.corrcoef(ok.specificity, ok.transfer_gap)[0, 1]
        print(f"\n=== across {len(ok)} stats: transfer gap (yoy diff minus within diff, squared pts/100) on specificity "
              f"(within-season reliability minus year-to-year correlation): slope {beta[1]:+.3f}, correlation {r:+.2f}")
        cols = ["stat", "group", "within_diff", "within_t", "yoy_diff", "yoy_t", "within_season_reliability",
                "year_to_year_correlation", "specificity"]
        print(st.sort_values("transfer_gap", ascending=False)[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
