"""One row per field-goal attempt: data/shotframe/<season>_<phase>.parquet (src/eracoef/shotframe.py).

    python scripts/114_shot_frame.py [first] [last] [phases=RS,PO] [--check=1] [--recentre=1]

--check=1   also asserts the stints the logger parsed alongside equal the cached stints exactly (every parser
            column), and that per shooter-game FGA / FGM / 3PA / 3PM equal data/stints/<s>_<ph>_shots.parquet.
            The stage-1 gate of the shot-quality plan runs this on 1997, 2010 and 2026.
--recentre=1  writes data/shotframe/rim_map.parquet (each arena-season's near-rim depth mapped onto the league's,
            from the visiting teams' x/y alone) over every frame on disk, and prints the arena spread of the 0-3 ft
            share before and after.

Reads only cached play-by-play and box scores; never scrapes.  About 1.5 min a regular season.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _cli import check_flags, flag  # noqa: E402
from eracoef.config import load_config, resolve  # noqa: E402
from eracoef.shotframe import derive, frame_dir, load_frame, player_game_totals, rim_map, write_season  # noqa: E402

pd.set_option("display.width", 250, "display.max_columns", 40, "display.precision", 3)


def reconcile(season, phase, cfg) -> dict:
    """Per shooter-game counts in the frame against the stints' per-game shooter table."""
    f = load_frame([season], cfg, phases=(phase,))
    mine = player_game_totals(f).rename(columns={"shooter": "player_id"})
    t = pd.read_parquet(resolve(cfg, "stints") / f"{season}_{phase}_shots.parquet")
    t = t.assign(fga=t.fg2a + t.fg3a, fgm=t.fg2m + t.fg3m)[["game_id", "player_id", "fga", "fgm", "fg3a", "fg3m"]]
    m = mine.merge(t, on=["game_id", "player_id"], how="outer", suffixes=("", "_st"), indicator=True)
    both = m["_merge"] == "both"
    same = both & (m.fga == m.fga_st) & (m.fgm == m.fgm_st) & (m.fg3a == m.fg3a_st) & (m.fg3m == m.fg3m_st)
    out = dict(player_games=len(m), match=int(same.sum()), only_frame=int((m["_merge"] == "left_only").sum()),
               only_stints=int((m["_merge"] == "right_only").sum()))
    if out["match"] != out["player_games"]:
        raise AssertionError(f"{season} {phase}: shooter-game counts differ from the stints' table: {out}")
    thr = f[f.value == 3]
    out["threes_without_xy"] = int(thr.x.isna().sum())
    return out


def recentre_all(cfg):
    """data/shotframe/rim_map.parquet over every regular season on disk, and the arena spread of the 0-3 ft share
    of twos before and after (the stage-1 gate: at most 4 points in 2021-26)."""
    paths = [p for p in sorted(frame_dir(cfg).glob("*_RS.parquet")) if not p.name.endswith("_aux.parquet")]
    f = pd.concat([pd.read_parquet(p, columns=["season", "arena", "team", "value", "x", "y"]) for p in paths],
                  ignore_index=True)
    rmap = rim_map(f)
    rmap.to_parquet(frame_dir(cfg) / "rim_map.parquet", index=False)
    base = f.assign(events="", poss_start_clock=0.0, clock=0.0, prev_event_clock=0.0)
    rows = []
    for name, m in (("raw", None), ("mapped", rmap)):
        d = derive(base, rmap=m)
        two = d[d.value == 2]                      # a (0, 0) two counts as a rim shot (99% are, by the research)
        for who, g in (("all", two), ("visitors", two[two.team != two.arena]), ("home", two[two.team == two.arena])):
            share = g.assign(rim=(g.dist_xy <= 3.0) | g.noloc).groupby(["season", "arena"]).rim.mean()
            rows.append(share.groupby(level=0).std().rename(f"{name}_{who}"))
    print("arena sd of the 0-3 ft share of twos ((0, 0) counted as rim shots)")
    print(pd.concat(rows, axis=1).round(3).to_string())


def main():
    check_flags()
    cfg = load_config()
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    check = flag("check", "0") == "1"
    if flag("recentre", "0") == "1":
        recentre_all(cfg)
        return
    first = int(args[0]) if args else cfg["first_season"]
    last = int(args[1]) if len(args) > 1 else first
    phases = tuple(args[2].split(",")) if len(args) > 2 else ("RS", "PO")
    rows = []
    for season in range(first, last + 1):
        for phase in phases:
            t0 = time.time()
            rep = write_season(season, phase, cfg, check=check, verbose=True)
            rep.update(reconcile(season, phase, cfg))
            rep["wall_seconds"] = round(time.time() - t0, 1)
            rows.append(rep)
            print(pd.Series(rep).to_string(), flush=True)
            print()
    print(pd.DataFrame(rows).drop(columns=["path"]).to_string(index=False))


if __name__ == "__main__":
    main()
