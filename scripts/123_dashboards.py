"""Per-date tracking dashboards, 2013-14 to 2016-17 only (the owner's approval, 2026-10-06; the STATS era).

    python scripts/123_dashboards.py --seasons=2016,2015,2014,2017 [--families=def,clock,dribble,touch,general,total]
                                     [--from=2015-10-27] [--to=2016-01-23] [--sleep=3]

One stats.nba.com call (leaguedashplayerptshot) per game date and per bin returns every player who shot that
date: FGA / FGM / 2s / 3s in that bin.  A date is one game per player, so these are per player-game counts by
closest-defender distance, shot clock, dribbles, touch time, and catch-and-shoot / pull-up / under 10 ft --
the check on the 2015-16 movement parse (stage 3b) and test 2(c) of the shot test.  Regular season only.

Every response is cached at data/tracking/dashboards/<season>/<date>/<family>__<bin>.parquet (an empty table
for a bin with no shots), so the pull resumes where it stopped.  One thread, `--sleep` seconds apart, retried
by ingest._retry.  Never run beside a ratings build.  About 22 calls a date, 165 dates a season.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _cli import check_flags, flag  # noqa: E402
from eracoef.config import load_config  # noqa: E402
from eracoef.ingest import _retry, load_gamelog, season_str  # noqa: E402

APPROVED = {2014, 2015, 2016, 2017}
FAMILIES = {
    "def": ("close_def_dist_range_nullable", ["0-2 Feet - Very Tight", "2-4 Feet - Tight", "4-6 Feet - Open",
                                              "6+ Feet - Wide Open"]),
    "clock": ("shot_clock_range_nullable", ["24-22", "22-18 Very Early", "18-15 Early", "15-7 Average", "7-4 Late",
                                            "4-0 Very Late", "ShotClock Off"]),
    "dribble": ("dribble_range_nullable", ["0 Dribbles", "1 Dribble", "2 Dribbles", "3-6 Dribbles", "7+ Dribbles"]),
    "touch": ("touch_time_range_nullable", ["Touch < 2 Seconds", "Touch 2-6 Seconds", "Touch 6+ Seconds"]),
    "general": ("general_range_nullable", ["Catch and Shoot", "Pullups", "Less Than 10 ft"]),
    "total": (None, ["all"]),
}


def safe(s: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in s)


def pull(season: int, date: str, family: str, value: str, timeout: int = 60) -> pd.DataFrame:
    from nba_api.stats.endpoints import leaguedashplayerptshot
    kw = dict(season=season_str(season), season_type_all_star="Regular Season", per_mode_simple="Totals",
              date_from_nullable=date, date_to_nullable=date, timeout=timeout)
    param = FAMILIES[family][0]
    if param:
        kw[param] = value
    frames = leaguedashplayerptshot.LeagueDashPlayerPtShot(**kw).get_data_frames()
    return frames[0] if frames else pd.DataFrame()


def main():
    check_flags()
    cfg = load_config()
    seasons = [int(s) for s in flag("seasons", "2016").split(",")]
    bad = set(seasons) - APPROVED
    if bad:
        raise SystemExit(f"seasons {sorted(bad)} are not approved: the owner approved 2013-14 to 2016-17 only")
    fams = flag("families", "def,clock,dribble,touch,general,total").split(",")
    sleep = float(flag("sleep", "3"))
    d_from, d_to = flag("from", None), flag("to", None)
    root = Path(cfg["_root"]) / "data" / "tracking" / "dashboards"
    n_calls = 0
    t0 = time.time()
    for season in seasons:
        gl = load_gamelog(season, "RS", cfg)
        dates = sorted(pd.to_datetime(gl["GAME_DATE"]).dt.strftime("%m/%d/%Y").unique(),
                       key=lambda d: pd.to_datetime(d))
        if d_from:
            dates = [d for d in dates if pd.to_datetime(d) >= pd.to_datetime(d_from)]
        if d_to:
            dates = [d for d in dates if pd.to_datetime(d) <= pd.to_datetime(d_to)]
        print(f"{season}: {len(dates)} dates x {sum(len(FAMILIES[f][1]) for f in fams)} bins", flush=True)
        for date in dates:
            ddir = root / str(season) / pd.to_datetime(date).strftime("%Y-%m-%d")
            for fam in fams:
                for value in FAMILIES[fam][1]:
                    path = ddir / f"{fam}__{safe(value)}.parquet"
                    if path.exists():
                        continue
                    df = _retry(lambda: pull(season, date, fam, value), what=f"{season} {date} {fam} {value}")
                    df = df.assign(season=season, date=pd.to_datetime(date).strftime("%Y-%m-%d"), family=fam, bin=value)
                    ddir.mkdir(parents=True, exist_ok=True)
                    df.to_parquet(path, index=False)
                    n_calls += 1
                    time.sleep(sleep)
            if n_calls and n_calls % 200 < 22:
                print(f"  {season} {date}: {n_calls} calls, {time.time() - t0:.0f}s", flush=True)
    print(f"done: {n_calls} calls in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
