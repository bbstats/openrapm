"""2015-16 tracking labels from the movement data, calibrated on the tracking dashboards (stage 3b).

    python scripts/124_movement_labels.py --calibrate=1 [--games=40]     # grid the release rules on early games
    python scripts/124_movement_labels.py [--release_back=3 --hold_ft=3.5 --bounce_ft=2.5]   # label every game

Calibration: takes `--games` archives from dates whose dashboards are pulled (data/tracking/dashboards/2016,
scripts/123), half of them from even-numbered dates and half from odd, keeps their frame timelines in memory, and
for a grid of release offsets, possession radii and bounce heights compares our per player-date counts with the
dashboards' in four families: closest-defender distance, shot clock, dribbles, touch time.  Agreement is read on
the player-dates where we found every attempt.  The grid sees only the even dates; the chosen rules are checked
on the odd dates, which it never saw.

The full run labels every archive (one at a time; each unpacks to ~100 MB of JSON, deleted after parsing)
and writes data/tracking/movement_shots_2016.parquet: game_id, action_number and the tracking fields.
Stage-3b gate: defender bins >= 85% and clock bins >= 90% on the held-out games.
"""
from __future__ import annotations

import sys
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _cli import check_flags, flag  # noqa: E402
from eracoef.config import load_config  # noqa: E402
from eracoef.movement import ReleaseRules, archives, game_id_of, parse_archive, shot_rows, timeline  # noqa: E402
from eracoef.shotframe import load_frame  # noqa: E402
from eracoef.tracking import bin_agreement, dashboard_counts, to_bins  # noqa: E402

pd.set_option("display.width", 220, "display.max_columns", 30)
FAMS = {"def": "def_dist", "clock": "mv_shot_clock", "dribble": "dribbles", "touch": "touch_time"}


def frames_by_game(cfg) -> tuple[dict, pd.DataFrame]:
    f = load_frame([2016], cfg, phases=("RS",))
    f = f[f["at_heave"] == 0] if "at_heave" in f.columns else f
    meta = f.drop_duplicates("game_id").set_index("game_id")["game_date"]
    return {g: d[["action_number", "period", "clock", "shooter", "value"]] for g, d in f.groupby("game_id")}, meta


def score(rows: pd.DataFrame, shooters: pd.DataFrame, dates: pd.Series, dash: dict, totals: pd.Series) -> dict:
    """Agreement per family for one set of labelled rows."""
    r = rows.merge(shooters, on=["game_id", "action_number"], how="left")
    r["date"] = pd.to_datetime(r["game_id"].map(dates)).dt.strftime("%Y-%m-%d")
    out = {}
    for fam, col in FAMS.items():
        b = to_bins(r[col].to_numpy(float), fam)
        ours = r.assign(bin=b).pivot_table(index=["date", "shooter"], columns="bin", values="action_number",
                                           aggfunc="size", fill_value=0)
        ours.index.names = ["date", "PLAYER_ID"]
        d = dash[fam]
        d = d[d.index.get_level_values("date").isin(set(r["date"]))]
        out[fam] = bin_agreement(ours, d, totals)
    return out


def main():
    check_flags()
    cfg = load_config()
    fg, dates = frames_by_game(cfg)
    shooters = pd.concat([d[["action_number", "shooter"]].assign(game_id=g) for g, d in fg.items()], ignore_index=True)
    tmp = Path(cfg["_root"]) / "data" / "tracking" / "movement_tmp"
    rules = ReleaseRules(release_back=int(flag("release_back", "3")), hold_ft=float(flag("hold_ft", "3.5")),
                         bounce_ft=float(flag("bounce_ft", "2.5")), hold_gap=int(flag("hold_gap", "4")))
    if flag("calibrate", "0") == "1":
        dash = {fam: dashboard_counts(cfg, 2016, fam) for fam in FAMS}
        tot = dashboard_counts(cfg, 2016, "total")
        totals = tot.sum(axis=1)
        n = int(flag("games", "40"))
        pulled = set(totals.index.get_level_values("date"))
        arcs = [p for p in archives(cfg)
                if pd.to_datetime(p.name[:10], format="%m.%d.%Y").strftime("%Y-%m-%d") in pulled]
        day = {p: pd.to_datetime(p.name[:10], format="%m.%d.%Y").dayofyear for p in arcs}
        early = [p for p in arcs if day[p] % 2 == 0][:n]
        late = [p for p in arcs if day[p] % 2 == 1][:n]
        T_early, T_late = [], []
        for group, sink in ((early, T_early), (late, T_late)):
            for a in group:
                js = game_id_of(a, tmp)
                if js:
                    T = timeline(js)
                    if T["gameid"] in fg:
                        sink.append(T)
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)
        print(f"calibration games {len(T_early)} (even dates), held-out games {len(T_late)} (odd dates)")
        grid = list(product([0, 2, 3, 4, 6], [3.0, 3.5, 4.5], [2.0, 2.5, 3.5], [2, 4]))
        res = []
        t0 = time.time()
        for back, hold, bounce, gap in grid:
            rr = ReleaseRules(release_back=back, hold_ft=hold, bounce_ft=bounce, hold_gap=gap)
            rows = pd.concat([shot_rows(T, fg[T["gameid"]], rr).assign(game_id=T["gameid"]) for T in T_early], ignore_index=True)
            sc = score(rows, shooters, dates, dash, totals)
            res.append(dict(release_back=back, hold_ft=hold, bounce_ft=bounce, hold_gap=gap,
                            found=len(rows), **{f: sc[f]["agree_complete"] for f in FAMS}))
        R = pd.DataFrame(res)
        print(f"grid of {len(grid)} in {time.time() - t0:.0f}s")
        # the release offset moves the defender and clock bins; the radius, gap and bounce move dribbles and touch
        print(R.groupby("release_back")[["def", "clock"]].max().round(3).to_string())
        best_back = int(R.groupby("release_back")["def"].max().idxmax())
        sub = R[R.release_back == best_back]
        best = sub.loc[(sub["dribble"] + sub["touch"]).idxmax()]
        print("\nchosen:", best.to_dict())
        rr = ReleaseRules(release_back=best_back, hold_ft=float(best.hold_ft), bounce_ft=float(best.bounce_ft),
                          hold_gap=int(best.hold_gap))
        rows = pd.concat([shot_rows(T, fg[T["gameid"]], rr).assign(game_id=T["gameid"]) for T in T_late], ignore_index=True)
        held = score(rows, shooters, dates, dash, totals)
        print("\nheld-out games:")
        for fam, v in held.items():
            print(f"  {fam:8s} agreement {v['agree_complete']:.3f} (all attempts {v['agree_all']:.3f}) on "
                  f"{v['complete_player_dates']} complete player-dates\n           ours {v['share_ours']}\n           dash {v['share_dash']}")
        out = Path(cfg["_root"]) / "outputs" / "shotclock"
        out.mkdir(parents=True, exist_ok=True)
        R.to_csv(out / "movement_calibration.csv", index=False)
        return
    rows, t0 = [], time.time()
    arcs = archives(cfg)
    for k, a in enumerate(arcs):
        gid, r = parse_archive(a, fg, tmp, rules)
        if len(r):
            rows.append(r)
        if (k + 1) % 25 == 0:
            print(f"  {k + 1}/{len(arcs)} games, {sum(len(x) for x in rows)} attempts, {time.time() - t0:.0f}s", flush=True)
    M = pd.concat(rows, ignore_index=True)
    M.attrs["rules"] = str(rules)
    p = Path(cfg["_root"]) / "data" / "tracking" / "movement_shots_2016.parquet"
    M.assign(release_back=rules.release_back, hold_ft=rules.hold_ft, bounce_ft=rules.bounce_ft,
             hold_gap=rules.hold_gap).to_parquet(p, index=False)
    n_frame = sum(len(fg[g]) for g in M["game_id"].unique() if g in fg)
    print(f"wrote {p}: {len(M)} attempts from {M.game_id.nunique()} games ({len(M) / max(n_frame, 1):.3f} of their attempts)")


if __name__ == "__main__":
    main()
