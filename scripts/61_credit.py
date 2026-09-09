"""Credit against forecast: two shares on the same held-out rows, and the players the board is missing.

A rating is asked to do two different things and this reports them apart, on the same residual, in the same
units, both as a share of what there was to get:

  FORECAST   1 - tg / tg_base.  The share of held-out team-game error the ratings remove, against the same
             prediction with no ratings at all (the season's level and home edge only).  This is the
             criterion (scripts/45_holdout.py) turned into a percentage.
  CREDIT     1 - player_board / player_zero.  Run a fresh player ridge on the residual and ask how much of it
             it can still put on named players (investigate.attributable).  `player_zero` is that quantity
             with no ratings -- all the player signal in the season -- and `player_board` is what the board
             left behind.  So this is the share of the attributable player signal the board has captured.

They are not the same number and they do not have to move together: a rating can predict a team's points
while splitting the credit for them wrongly among five men (FINDINGS 26.5, 27).

Then: who.  Per player and side, the ridge miss pooled across seasons with a z -- points per 100 the board
should ADD to that side (positive = under-rated) -- and the same by rating decile, by exposure and for
players in their first season.

    python scripts/61_credit.py [--dump=outputs/ratings_insea_ship_q75.parquet]
        [--systems=ks52_lam05_q75,ks52_q75,ks11_q75,blk_q75] [--who=ks52_lam05_q75] [--k=3] [--top=15]
        [--lam=2000] [--min-poss=1000]

Writes outputs/credit_<who>.csv (per player) and outputs/csv/credit_summary.csv (the two shares).
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eracoef.calmap import (build_design, cut_of, fit_theta, load_frames, mapped_ratings, parse_maps,  # noqa: E402
                            ratings_for, train_of)
from eracoef.config import load_config  # noqa: E402
from eracoef.holdout import Context, Holdout, predict_season, team_game_mse  # noqa: E402
from eracoef.investigate import attributable, pooled, residual_ridge, season_table  # noqa: E402

pd.set_option("display.width", 250, "display.max_columns", 40, "display.precision", 3)
cfg = load_config()
OUT = Path(cfg["_root"]) / "outputs"
FAM = "linear+log2&xlog&prior&tshare:linear+log2&xlog"


def flag(name, default=None):
    hit = [a for a in sys.argv[1:] if a.startswith(f"--{name}=")]
    return hit[0].split("=", 1)[1] if hit else default


def names_of() -> dict:
    out = {}
    for p in (OUT / "season_ratings.parquet", OUT / "player_ratings.parquet"):
        if p.exists():
            t = pd.read_parquet(p, columns=["player_id", "player_name"]).dropna().drop_duplicates("player_id")
            out.update(dict(zip(t.player_id.astype(int), t.player_name)))
    return out


def main():
    dump_path = Path(cfg["_root"]) / flag("dump", "outputs/ratings_insea_ship_q75.parquet")
    if not dump_path.exists():
        raise SystemExit(f"{dump_path} is missing; run scratch/inseason_run.py --tag=ship --held=all first")
    dump = pd.read_parquet(dump_path)
    systems = (flag("systems") or ",".join(sorted(dump.system.unique()))).split(",")
    who = flag("who", systems[0])
    k = int(flag("k", 3))
    lam = float(flag("lam", 2000.0))
    top = int(flag("top", 15))
    min_poss = float(flag("min-poss", 1000.0))
    ho = Holdout.from_config(cfg, ks=[k])
    ctx = Context.load(cfg)
    map_o, map_d, bend = parse_maps(FAM)
    cut = cut_of(dump, systems[0])
    t0 = time.time()
    frames = load_frames(ctx, ho.seasons(), level=ho.level, verbose=False, cut=cut)
    print(f"{len(frames)} held-out seasons, cut {cut}, K = {k}, ridge {lam:g} ({time.time() - t0:.0f}s)", flush=True)

    zero: dict = {}                      # the player signal in each season with NO ratings, computed once
    rows, players = [], []
    for s in systems:
        have = set(dump.loc[(dump.system == s) & (dump.k == k), "held_out"].astype(int))
        fr = {h: f for h, f in frames.items() if h in have}
        D = build_design(dump, fr, s, k, map_o, map_d)
        for h, f in fr.items():
            th = fit_theta(D, exclude_h=h, map_o=map_o, map_d=map_d)
            rat = mapped_ratings(ratings_for(dump, s, k, h), th, map_o, map_d, D.scale_o, D.scale_d,
                                 extra=f.covariates(k, train_of(dump, s, k, h)))
            p = predict_season(rat, f.wd, level=ho.level)
            r = p.y - p.pred
            if h not in zero:            # the same two questions asked of a board with no ratings at all
                a0 = attributable(f.Zo, f.Zd, p.y - p.base_pred, p.w, lam)
                tg0, _ = team_game_mse(p.y, p.base_pred, p.poss, p.game_idx, p.is_home_off)
                zero[h] = (a0, tg0)
            a0, tg0 = zero[h]
            a = attributable(f.Zo, f.Zd, r, p.w, lam)
            tg, _ = team_game_mse(p.y, p.pred, p.poss, p.game_idx, p.is_home_off)
            rows.append(dict(system=s, held_out=int(h), w=float(p.w.sum()), tg=tg, tg_base=tg0,
                             player=a["player"], player_zero=a0["player"], total=a["total"],
                             ss_o=a["ss_o"], ss_d=a["ss_d"]))
            if s == who:
                players.append(season_table(h, f.ids, f.Zo, f.Zd, r, p.w, rat=rat.df, lam=lam))
        print(f"  {s} done ({time.time() - t0:.0f}s)", flush=True)

    R = pd.DataFrame(rows)
    g = R.groupby("system", sort=False).apply(lambda d: pd.Series({
        "forecast_pct": 100.0 * (1 - d.tg.mean() / d.tg_base.mean()),
        "credit_pct": 100.0 * (1 - d.player.mean() / d.player_zero.mean()),
        "tg": d.tg.mean(), "tg_base": d.tg_base.mean(),
        "left_o": np.sqrt(d.ss_o.mean()), "left_d": np.sqrt(d.ss_d.mean()),
        "seasons": len(d)}), include_groups=False).reset_index()
    g.to_csv(OUT / "csv" / "credit_summary.csv", index=False)
    print("\n=== the two shares, on the same held-out rows\n")
    print("  forecast %  = share of team-game error the ratings remove (higher better)")
    print("  credit %    = share of the attributable player signal the board has captured (higher better)")
    print("  left o / d  = what the board still misses per player, points per 100, possession-weighted rms\n")
    print(f"  {'system':18s} {'forecast %':>10s} {'credit %':>9s} {'left o':>7s} {'left d':>7s} "
          f"{'tg':>8s} {'tg base':>8s} {'seasons':>7s}")
    for r in g.itertuples(index=False):
        print(f"  {r.system:18s} {r.forecast_pct:10.2f} {r.credit_pct:9.2f} {r.left_o:7.2f} {r.left_d:7.2f} "
              f"{r.tg:8.2f} {r.tg_base:8.2f} {int(r.seasons):7d}")

    P = pd.concat(players, ignore_index=True)
    nm = names_of()
    P["name"] = P.player_id.map(nm).fillna("")
    P.to_parquet(OUT / f"credit_{who}.parquet", index=False)
    Q = pooled(P, min_poss=min_poss)
    Q["name"] = Q.player_id.map(nm).fillna("")
    Q.to_csv(OUT / f"credit_{who}.csv", index=False)

    print(f"\n=== who {who} is missing: the ridge miss pooled over seasons, {min_poss:.0f}+ possessions each.")
    print("    miss = points per 100 the board should ADD to that side.  Positive = under-rated.\n")
    for side, label in (("O", "offense"), ("D", "defense")):
        q = Q[Q.side == side].sort_values("z")
        print(f"--- most UNDER-rated on {label}")
        print(q.tail(top).iloc[::-1][["name", "miss", "se", "z", "seasons", "poss"]].to_string(index=False))
        print(f"\n--- most OVER-rated on {label}")
        print(q.head(top)[["name", "miss", "se", "z", "seasons", "poss"]].to_string(index=False))
        print()

    # where the miss lives: by rating decile, by exposure, and for players the fit had no past for
    print("=== where the miss lives (possession-weighted mean miss, points per 100)\n")
    for side in ("o", "d"):
        d = P[P[f"poss_{side}"] >= min_poss].copy()
        d["dec"] = d.groupby("held_out")[f"rating_{side}"].transform(
            lambda v: pd.qcut(v.rank(method="first"), 10, labels=False) + 1)
        t = d.groupby("dec").apply(lambda x: pd.Series({
            "miss": np.average(x[f"miss_{side}"], weights=x[f"poss_{side}"]),
            "rating": np.average(x[f"rating_{side}"], weights=x[f"poss_{side}"]),
            "n": len(x)}), include_groups=False)
        print(f"--- {side.upper()} by rating decile (1 = worst rated, 10 = best)")
        print(t.T.round(2).to_string())
        print()
    edges = [0, 500, 1500, 4000, 1e9]
    for side in ("o", "d"):
        d = P.copy()
        d["bin"] = pd.cut(d[f"poss_{side}"], edges, labels=["under 500", "500-1499", "1500-3999", "4000+"])
        t = d.groupby("bin", observed=True).apply(lambda x: pd.Series({
            "miss": np.average(x[f"miss_{side}"], weights=np.maximum(x[f"poss_{side}"], 1e-9)),
            "n": len(x)}), include_groups=False)
        print(f"--- {side.upper()} by the possessions he played in the scored games")
        print(t.T.round(2).to_string())
        print()
    print(f"wrote outputs/credit_{who}.csv and outputs/csv/credit_summary.csv ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
