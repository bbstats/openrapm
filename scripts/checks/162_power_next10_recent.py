"""Check 162: check 159's last-10 stats with the latest 5 games counted double, as the Power page counts them.

The owner, 2026-11-08: "are zones/orb/tov/ft done w/ same 'newer 5 weighted more' dealio? even if worse performance,
i want to apply the same weight possibly?".  notes/DECISIONS.md, "Check 162", holds the reading, written before any
result.

Words used below
    plain      a team's last 10 games, every game counting once, as checks 159 to 161 ran
    weighted   the same games, each of the latest 5 counting twice (site.RECENT_GAMES, site.RECENT_WEIGHT).  The weight
               goes on each game's counts and possessions before any rate is formed.  Example, three-point attempts per
               100 from latest-5 totals of 170 attempts in 500 possessions and earlier-5 totals of 150 in 490: plain is
               100 x 320 / 990 = 32.3, weighted is 100 x (2 x 170 + 150) / (2 x 500 + 490) = 32.9

    python scripts/checks/162_power_next10_recent.py [--out outputs/csv/check162]
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def load_161():
    """Check 161 as a module; checks 160 and 159 come with it."""
    spec = importlib.util.spec_from_file_location("check161", ROOT / "scripts" / "checks" / "161_power_next10_audit.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["check161"] = module
    spec.loader.exec_module(module)
    return module


C161 = load_161()
C160, C159 = C161.C160, C161.C159
from openrapm.commands.site import RECENT_GAMES, RECENT_WEIGHT  # noqa: E402
from openrapm.data.config import load_config  # noqa: E402

PLAIN, WEIGHTED, POWER = "plain last 10", "latest 5 counted double", "Power at its own scale"


def check_weighting(table: pd.DataFrame) -> float:
    """The largest gap, over teams, between a hand-built weighted three-attempt rate and `last_games_stats`', before its
    centering is added back: 0 when the stats are weighted as the module says."""
    C159.STATS_RECENT_WEIGHT = RECENT_WEIGHT
    built = C159.last_games_stats(table).attempts_thr_o
    gaps = {}
    for tm, rows in table.sort_values(["tm", "game_date", "game_id"], kind="stable").groupby("tm"):
        last = rows.tail(C159.LAST_GAMES)
        recent = np.arange(len(last)) >= len(last) - RECENT_GAMES
        weight = np.where(recent, RECENT_WEIGHT, 1.0)
        gaps[tm] = 100 * float((weight * last.fga_thr_o).sum() / (weight * last.poss_o).sum())
    by_hand = pd.Series(gaps)
    return float((built - (by_hand - by_hand.mean())).abs().max())


def stats_rows(cut, ratings, tables, stints, teams, weight) -> tuple:
    """Check 159's rows of one day with the stats at `weight`."""
    C159.STATS_RECENT_WEIGHT = weight
    return C159.day_rows(cut, ratings, tables, stints, teams)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="outputs/csv/check162")
    args = parser.parse_args()
    started = time.time()
    cfg = load_config()
    within = ROOT / "outputs" / "within"
    wowy = C159.load_wowy()
    tables = {season: C159.season_game_table(season, cfg) for season in C159.SEASONS}
    stints = {season: C159.season_stints_ids(season, cfg) for season in C159.SEASONS}
    teams = {season: C159.regular_game_teams(season, cfg) for season in C159.SEASONS}
    print(f"hand-built weighting against the code, largest gap: {check_weighting(tables[2026]):.2e}", flush=True)

    against_power, against_plain, walks, keep = [], [], [], {}
    for cut, folder in C159.CUTS.items():
        ratings = C159.day_ratings(wowy, within, folder)
        days = {}
        for name, weight in ((PLAIN, 1.0), (WEIGHTED, RECENT_WEIGHT)):
            team_rows, game_rows = stats_rows(cut, ratings, tables, stints, teams, weight)
            days[name] = C160.Day(team_rows, game_rows)
        base, _ = C160.predictions(days[PLAIN], lambda test, train: [])
        frames = [C159.score_frame(POWER, days[PLAIN].game_rows, days[PLAIN].y, base)]
        for name, day in days.items():
            made, _ = C160.predictions(day, lambda test, train: C160.ALL_IN)
            frames.append(C159.score_frame(name, day.game_rows, day.y, made))
            walks.append(C161.walk_forward(day).assign(cut=cut, model=name))
        scores = pd.concat(frames, ignore_index=True)
        against_power.append(C159.compared(scores, POWER).assign(cut=cut))
        against_plain.append(C159.compared(scores, PLAIN).assign(cut=cut))
        keep[cut] = days
        print(f"{cut} ({time.time() - started:.0f}s)", flush=True)

    power, plain, walk = (pd.concat(frames, ignore_index=True) for frames in (against_power, against_plain, walks))
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    power.to_csv(f"{out}_against_power.csv", index=False)
    plain.to_csv(f"{out}_against_plain.csv", index=False)
    walk.to_csv(f"{out}_walk.csv", index=False)
    shown = ["cut", "era", "game_rmse", "team_rmse", "game_change", "game_z", "game_better", "team_change", "team_z"]
    print("\nweighted minus plain", flush=True)
    print(plain[plain.model == WEIGHTED][shown].round(3).to_string(index=False), flush=True)
    print("\neach against Power at its own scale", flush=True)
    picked = power[power.model.isin([PLAIN, WEIGHTED])]
    print(picked[["model", *shown]].round(3).to_string(index=False), flush=True)
    print("\neach season predicted only from the seasons before it, against Power fitted the same way", flush=True)
    print(
        walk[["cut", "model", "era", "change", "se", "z", "seasons_better", "seasons"]].round(3).to_string(index=False),
        flush=True,
    )

    day = keep[C159.TODAY_CUT][WEIGHTED]
    columns = C160.columns_of(C160.ALL_IN)
    names = ["home", "power_o", "power_d", *C160.COLUMNS]
    weights = pd.Series(C160.fitted(day, list(day.seasons), columns) / day.spread[columns], index=names)
    plain_day = keep[C159.TODAY_CUT][PLAIN]
    plain_weights = pd.Series(
        C160.fitted(plain_day, list(plain_day.seasons), columns) / plain_day.spread[columns], index=names
    )
    pd.DataFrame({"weighted": weights, "plain": plain_weights}).to_csv(f"{out}_weights.csv")
    print("\nweights, weighted and plain", flush=True)
    print(pd.DataFrame({"weighted": weights, "plain": plain_weights}).round(4).to_string(), flush=True)

    C159.STATS_RECENT_WEIGHT = RECENT_WEIGHT
    now = C159.today_teams(cfg)
    stats = [*C159.ADDED, "power_o", "power_d"]
    centered = now[stats] - now[stats].mean()
    parts = pd.DataFrame({"power_part": centered.power_o * weights.power_o + centered.power_d * weights.power_d})
    for group, group_columns in C159.GROUPS.items():
        parts[group] = (centered[group_columns] * weights[group_columns]).sum(axis=1)
    parts["new_total"] = parts.sum(axis=1)
    parts["power"] = now.power_o + now.power_d
    parts = parts.sort_values("new_total", ascending=False)
    parts.to_csv(f"{out}_today.csv")
    C159.STATS_RECENT_WEIGHT = 1.0
    plain_now = C159.today_teams(cfg)
    plain_stats = plain_now[stats] - plain_now[stats].mean()
    parts["plain_total"] = (plain_stats * plain_weights[stats]).sum(axis=1)
    print("\n2027 through the latest games, latest 5 counted double", flush=True)
    print(parts.round(2).to_string(), flush=True)
    print(f"done in {time.time() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
