"""The team-game leave-one-out report: the shrinkage constants, the matchup regression, the sanity gates.

    python scripts/62_teamloo.py [first] [last] [--rebalance=label|rate|none] [--cut=q] [--csv]

For each season it prints

  * the CONSTANTS table, one row per component and side: the between-team variance four ways
    (rebalanced LOO, plain LOO, split-half, method of moments), the mechanical bias of identity (A),
    the resulting k in attempts, and `shrink` -- the share of a game's OWN rate that survives the
    blend.  Read `shrink` first: it says how much of this is a luck adjustment and how much is
    replacement.
  * the MATCHUP regression: points per 100 of a team-game on the two teams' leave-one-out composites
    and home, possession weighted, with the plain-LOO fit beside it so the attenuation the paper
    describes is visible rather than asserted.
  * the GATES, which are the sanity checks that must pass before any ladder rung is worth running:
    the offensive constants in a plausible range, defensive three-point and free-throw variance near
    zero (a defence does not control whether an open three drops -- Nylon Calculus 2018, and the
    reason `x3def` exists), and the rebalanced fit de-attenuated against the plain one.

`--csv` writes `outputs/csv/teamloo_report.csv` (the constants, every season) and
`outputs/csv/teamloo_rating_<season>.csv` (per team-game: the LOO rates, the fitted points per 100,
and the predictive net rating -- the offence's fitted value minus the opponent's).

See `src/eracoef/teamloo.py` for what the estimator is and FINDINGS section 33 for what came of it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eracoef import teamloo as tl                                        # noqa: E402
from eracoef.config import load_config                                   # noqa: E402

CONST_COLS = ["season", "side", "component", "n_tg", "att", "p", "tau2", "tau2_plain", "tau2_half",
              "tau2_mom", "bias_exact", "k_loo", "k_half", "k_mom", "k", "shrink_mean"]


def _fmt(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in ("tau2", "tau2_plain", "tau2_half", "tau2_mom", "bias_exact"):
        out[c] = out[c].map(lambda v: f"{v: .2e}")
    for c in ("k_loo", "k_half", "k_mom", "k"):
        out[c] = out[c].map(lambda v: f"{v:9.0f}")
    out["p"] = out["p"].round(4)
    out["shrink_mean"] = out["shrink_mean"].round(4)
    out["att"] = out["att"].astype(int)
    return out


def season_report(season: int, cfg, rebalance: str, cut: float | None, quiet: bool = False):
    keep = None
    if cut is not None:
        tg_all = tl.team_games(season, cfg)
        ids = tg_all.drop_duplicates("game_id").sort_values(["game_date", "game_id"])["game_id"].to_numpy()
        keep = ids[: int(round(float(cut) * len(ids)))]
    tg, loo, kt = tl.season_table(season, cfg, keep=keep, rebalance=rebalance)
    kt = kt.assign(season=season)
    reb = tl.matchup_regression(tg, loo, design="composite")
    pln = tl.matchup_regression(tg, tl.loo_rates(tg, rebalance="none"), design="composite")
    raw = tl.matchup_regression(tg, loo, design="raw")
    ident = tl.matchup_regression(tg, loo, design="realised")
    if not quiet:
        print(f"\n=== {season} RS: {len(tg)} team-games, {tg.game_id.nunique()} games"
              + (f", cut at {cut} ({len(keep)} games)" if keep is not None else "") + " ===")
        print(_fmt(kt[CONST_COLS]).to_string(index=False))
        print(f"\n  matchup regression, possession weighted   R2 {reb['r2']:.4f} (plain LOO {pln['r2']:.4f}, "
              f"raw rates {raw['r2']:.4f}, realised identity {ident['r2']:.6f})")
        print(f"  {'coefficient':<12} {'rebalanced':>11} {'plain LOO':>11}   (a composite is points per "
              f"possession, so 100 is unattenuated)")
        for k in reb["coef"]:
            print(f"  {k:<12} {reb['coef'][k]:11.2f} {pln['coef'][k]:11.2f}")
    return tg, loo, kt, reb, pln, ident


def gates(kt: pd.DataFrame, reb: dict, pln: dict, ident: dict) -> pd.DataFrame:
    """What must hold before a ladder rung is worth running.

    A between-team variance estimated from ONE season of 30 teams is noisy: its relative sampling error
    is about 26% before anything else, and in the low-volume three-point eras the split-half reference
    comes out negative outright (1997-2009).  So the bands here are wide, a gate whose reference is not
    estimable is skipped rather than failed, and the verdict that matters is the pooled one across
    seasons that `main` prints underneath.  A single-season MISS is a reading, not a defect.
    """
    o = kt[kt.side == "off"].set_index("component")
    d = kt[kt.side == "def"].set_index("component")
    comps = ("off_fg3", "off_fg2", "off_ft")
    est = o["tau2_half"] > 0                       # components whose reference is estimable this season
    rows = [
        ("offensive k in a plausible range (100-5000 attempts)",
         bool(o["k"].between(100, 5000).all()), f"fg3 {o.loc['fg3','k']:.0f} fg2 {o.loc['fg2','k']:.0f} "
                                                f"ft {o.loc['ft','k']:.0f}"),
        ("the three estimates of the between-team variance agree within 40%",
         bool((abs(o.loc[est, "tau2"] - o.loc[est, "tau2_half"]) < 0.40 * o.loc[est, "tau2_half"]).all()),
         " ".join(f"{c} {o.loc[c, 'tau2'] / o.loc[c, 'tau2_half']:.2f}x" if est[c] else f"{c} n/a"
                  for c in tl.COMPONENTS)),
        ("the mechanical bias of identity (A) is negative and rebalancing raises tau2",
         bool((o["bias_exact"] < 0).all() and (o["tau2"] > o["tau2_plain"]).all()),
         f"gap {float((o['tau2'] / o['tau2_plain'] - 1).mean()) * 100:+.1f}% on average"),
        ("a defence controls opponent threes far less than an offence controls its own",
         bool(d.loc["fg3", "tau2_half"] < 0.5 * o.loc["fg3", "tau2_half"]) if est["fg3"] else True,
         f"def {d.loc['fg3','tau2_half']:.2e} vs off {o.loc['fg3','tau2_half']:.2e}"
         + ("" if est["fg3"] else "  (offensive reference not estimable this season)")),
        ("a defence controls opponent free throws not at all",
         bool(d.loc["ft", "tau2_half"] < 0.35 * o.loc["ft", "tau2_half"]) if est["ft"] else True,
         f"def {d.loc['ft','tau2_half']:.2e} vs off {o.loc['ft','tau2_half']:.2e}"),
        ("rebalancing de-attenuates every offensive coefficient",
         bool(all(reb["coef"][c] > pln["coef"][c] for c in comps)),
         " ".join(f"{c.split('_')[1]} {pln['coef'][c]:.0f}->{reb['coef'][c]:.0f}" for c in comps)),
        ("the realised design is an identity (R2 == 1)", bool(ident["r2"] > 1.0 - 1e-9), f"{ident['r2']:.8f}"),
        ("HOW MUCH of a game's own shooting survives the blend (a reading, not a pass/fail)",
         True, " ".join(f"{c} {o.loc[c, 'shrink_mean'] * 100:.1f}%" for c in tl.COMPONENTS)),
    ]
    return pd.DataFrame(rows, columns=["gate", "ok", "reading"])


def main(argv):
    args = [a for a in argv if not a.startswith("--")]
    flags = {a.split("=")[0]: (a.split("=")[1] if "=" in a else True) for a in argv if a.startswith("--")}
    cfg = load_config()
    first = int(args[0]) if args else int(cfg["last_season"])
    last = int(args[1]) if len(args) > 1 else first
    rebalance = str(flags.get("--rebalance", "label"))
    cut = float(flags["--cut"]) if "--cut" in flags else None
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 60)

    consts, gate_rows, ratings = [], [], {}
    for season in range(first, last + 1):
        try:
            tg, loo, kt, reb, pln, ident = season_report(season, cfg, rebalance, cut)
        except (FileNotFoundError, KeyError) as e:
            print(f"\n=== {season}: skipped ({type(e).__name__}: {e}) ===")
            continue
        consts.append(kt)
        g = gates(kt, reb, pln, ident)
        gate_rows.append(g.assign(season=season))
        print()
        for _, r in g.iterrows():
            print(f"  [{'ok ' if r['ok'] else 'MISS'}] {r['gate']:<62} {r['reading']}")
        out = tg[["game_id", "season", "is_home_off", "team_id", "opp_id", "home", "poss", "pts100"]].copy()
        for side in ("off", "def"):
            for c in tl.COMPONENTS:
                out[f"{side}_p_{c}"] = loo[f"{side}_p_{c}"].to_numpy()
        out["pred"] = reb["pred"]
        # the predictive net rating: what this team is expected to score minus what its opponent is,
        # in the same game, so the schedule is differenced out
        flip = out.set_index(["game_id", "is_home_off"])["pred"]
        other = flip.reindex(pd.MultiIndex.from_arrays([out["game_id"], ~out["is_home_off"]])).to_numpy()
        out["net"] = out["pred"].to_numpy() - other
        ratings[season] = out

    if not consts:
        return 1
    allc = pd.concat(consts, ignore_index=True)
    allg = pd.concat(gate_rows, ignore_index=True)
    if last > first:
        print("\n=== across seasons: the constants, mean and spread ===")
        s = allc.groupby(["side", "component"]).agg(
            k_mean=("k", "mean"), k_sd=("k", "std"), shrink=("shrink_mean", "mean"),
            tau2=("tau2", "mean"), tau2_half=("tau2_half", "mean")).round(6)
        print(s.to_string())
        print("\n=== gates, seasons passing ===")
        print(allg.groupby("gate")["ok"].agg(["sum", "count"]).to_string())
        # The verdict that decides whether the ladder is worth running: the same comparisons on the
        # POOLED (attempt-weighted) variances, where one season's 26% sampling error has averaged out.
        def pooled(side, c, col):
            m = (allc.side == side) & (allc.component == c)
            return float(np.average(allc.loc[m, col], weights=allc.loc[m, "att"]))
        print("\n=== pooled over the seasons run: the three estimates, and what a defence controls ===")
        print(f"  {'':<5} {'rebalanced':>11} {'plainLOO':>10} {'splithalf':>10} {'mom':>10}   "
              f"{'reb/half':>8} {'def/off':>8}   {'k':>7} {'shrink':>7}")
        for side in ("off", "def"):
            for c in tl.COMPONENTS:
                h = pooled(side, c, "tau2_half")
                oh = pooled("off", c, "tau2_half")
                print(f"  {side}{c:<4} {pooled(side, c, 'tau2'):11.2e} {pooled(side, c, 'tau2_plain'):10.2e} "
                      f"{h:10.2e} {pooled(side, c, 'tau2_mom'):10.2e}   "
                      f"{pooled(side, c, 'tau2') / h if h > 0 else float('nan'):8.2f} "
                      f"{h / oh if oh > 0 else float('nan'):8.2f}   "
                      f"{pooled(side, c, 'k'):7.0f} {pooled(side, c, 'shrink_mean') * 100:6.1f}%")

    if flags.get("--csv"):
        d = Path(cfg["_root"]) / "outputs" / "csv"
        d.mkdir(parents=True, exist_ok=True)
        allc[CONST_COLS].to_csv(d / "teamloo_report.csv", index=False)
        for season, out in ratings.items():
            out.to_csv(d / f"teamloo_rating_{season}.csv", index=False)
        print(f"\nwrote {d / 'teamloo_report.csv'} and {len(ratings)} per-season rating tables")
    return 0 if bool(allg["ok"].all()) else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
