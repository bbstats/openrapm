"""The investigator: which lineups and which players the shipped board gets wrong OUT OF SEASON, and by how much.

    python scripts/57_investigate.py [--system=<tracked system>] [--k=3] [--lam=2000] [--min-poss=1000]
                                     [--top=20] [--seasons=1998,1999,...] [--level=home]

Needs the system's tracker dump (outputs/ratings_track_<system>.parquet, from scripts/54_track.py) and its
tracker map table; both exist for anything the tracker has measured.  Default system: the shipped one
(config ratings_prior.cal_map.base), under the shipping map family.

For every held-out season H: the dump's ratings for H under the map fitted without H predict every stint row
of H exactly as the criterion does; the residual is what the board did not know (src/eracoef/investigate.py).
Prints the players the board is most wrong about, pooled across seasons (the residual ridge, z-scored), the
worst single player-seasons, and the five-man units with the largest miss; writes
outputs/investigate_<system>.parquet (player-seasons), outputs/investigate_pooled_<system>.csv and
outputs/investigate_lineups_<system>.csv.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eracoef.calmap import build_design, fit_theta, load_frames, mapped_ratings, parse_maps, ratings_for  # noqa: E402
from eracoef.config import load_config  # noqa: E402
from eracoef.holdout import Context, Holdout, predict_season  # noqa: E402
from eracoef.investigate import lineups, pooled, season_table  # noqa: E402

pd.set_option("display.width", 250, "display.max_columns", 40, "display.precision", 2)
cfg = load_config()
OUT = Path(cfg["_root"]) / "outputs"


def flag(name, default=None):
    hit = [a for a in sys.argv[1:] if a.startswith(f"--{name}=")]
    return hit[0].split("=", 1)[1] if hit else default


def names_of(cfg) -> dict:
    """player_id -> name, from the shipped board's table (any window)."""
    p = OUT / "player_ratings.parquet"
    if not p.exists():
        return {}
    t = pd.read_parquet(p, columns=["player_id", "player_name"]).dropna().drop_duplicates("player_id")
    return dict(zip(t.player_id.astype(int), t.player_name))


def main():
    cm = cfg.get("ratings_prior", {}).get("cal_map") or {}
    system = flag("system", cm.get("base"))
    k = int(flag("k", cm.get("k", 3)))
    lam = float(flag("lam", 2000.0))
    min_poss = float(flag("min-poss", 1000.0))
    top = int(flag("top", 20))
    level = flag("level", cfg.get("holdout", {}).get("level", "home"))
    fam = cm.get("system", "").replace(f"{system}_", "", 1) if cm.get("base") == system else "linear+log2&xlog&prior&tshare_linear+log2&xlog"
    fam = fam.replace("_linear", ":linear", 1) if ":" not in fam else fam
    dump_path = OUT / f"ratings_track_{system}.parquet"
    if not dump_path.exists():
        raise SystemExit(f"{dump_path} is missing; run scripts/54_track.py --systems={system} first")
    dump = pd.read_parquet(dump_path)
    ho = Holdout.from_config(cfg, ks=[k])
    seasons = [int(s) for s in (flag("seasons") or "").split(",") if s] or ho.seasons()
    ctx = Context.load(cfg)
    t0 = time.time()
    frames = load_frames(ctx, seasons, level=level, verbose=True)
    map_o, map_d, bend = parse_maps(fam)
    D = build_design(dump, frames, system, k, map_o, map_d)
    names = names_of(cfg)
    players, units = [], []
    for h, f in frames.items():
        th = fit_theta(D, exclude_h=h, map_o=map_o, map_d=map_d)
        rat = mapped_ratings(ratings_for(dump, system, k, h), th, map_o, map_d, D.scale_o, D.scale_d, extra=f.covariates(k))
        p = predict_season(rat, f.wd, level=level)
        r = p.y - p.pred
        players.append(season_table(h, f.ids, f.Zo, f.Zd, r, p.w, rat=rat.df, lam=lam))
        u = lineups(f.Zo, f.Zd, f.ids, r, p.w)
        u.insert(0, "held_out", int(h))
        units.append(u)
        print(f"  {h}: {len(f.ids)} players, weighted residual sd {np.sqrt(np.average(r ** 2, weights=p.w)):.2f} "
              f"({time.time() - t0:.0f}s)", flush=True)
    P = pd.concat(players, ignore_index=True)
    U = pd.concat(units, ignore_index=True)
    P["name"] = P.player_id.map(names).fillna("")
    U["names"] = U.key.map(lambda ks: ", ".join(names.get(int(q), str(q)) for q in ks))
    P.to_parquet(OUT / f"investigate_{system}.parquet", index=False)
    Q = pooled(P, min_poss=min_poss)
    Q["name"] = Q.player_id.map(names).fillna("")
    Q.to_csv(OUT / f"investigate_pooled_{system}.csv", index=False)
    U.drop(columns="key").to_csv(OUT / f"investigate_lineups_{system}.csv", index=False)

    print(f"\n=== {system} under {fam}, K = {k}, ridge {lam:g}, {len(seasons)} held-out seasons; miss = points per 100 the "
          f"board should ADD to that side (positive = under-rated)")
    for side in ("O", "D"):
        q = Q[Q.side == side].sort_values("z")
        print(f"\n--- {side}: pooled across seasons, {min_poss:.0f}+ possessions a season; the most UNDER-rated")
        print(q.tail(top).iloc[::-1][["name", "miss", "se", "z", "seasons", "poss"]].to_string(index=False))
        print(f"--- {side}: the most OVER-rated")
        print(q.head(top)[["name", "miss", "se", "z", "seasons", "poss"]].to_string(index=False))
    for side in ("o", "d"):
        d = P[P[f"poss_{side}"] >= min_poss].copy()
        d["z"] = d[f"miss_{side}"] / d[f"se_{side}"]
        d = d.sort_values("z", key=np.abs, ascending=False).head(top)
        print(f"\n--- {side.upper()}: the worst single player-seasons")
        print(d[["held_out", "name", f"miss_{side}", f"se_{side}", "z", f"oncourt_{side}", f"poss_{side}",
                 f"rating_{side}", f"prior_{side}"]].to_string(index=False))
    print(f"\n--- five-man units, 200+ possessions, the largest miss (positive = better than its five ratings say)")
    print(U.sort_values("miss", key=np.abs, ascending=False).head(top)[["held_out", "side", "miss", "poss", "rows", "names"]].to_string(index=False))
    print(f"\nwrote outputs/investigate_{system}.parquet, investigate_pooled_{system}.csv, investigate_lineups_{system}.csv "
          f"({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
