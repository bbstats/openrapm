"""Check 166: the Power table's weights refitted on the ratings the page multiplies, with a knot at three quarters.

Check 165 fitted the weights on season-so-far ratings whose shrink was fitted on those same season-so-far folds, with
the weight on the box-score prior held in full at half a season.  The Power page multiplies the published ratings of
the season in progress: the whole-season shrink, and the prior's weight held in full through a quarter of the season
and then on a straight line to the season's fitted one at three quarters.  notes/DECISIONS.md, "Check 166", holds the
method and the rule, written before any result.

Words used below
    knot       a point of the regular season with fitted multipliers and weights: the 7.15%, 10% and 25% days with flat
               shares and the 14 stats, the 50% and 75% days with the page's doubled latest 5 and no stats
    line       the multipliers and weights on any other day: the straight line between the two knots around it
    held       a folder of season-so-far ratings with the prior's weight held in full, outputs/within/<build>_cut*_held:
               what the published season in progress has through a quarter of its season
    blended    the same from a quarter on, the prior's weight on the line to its fitted value at three quarters
               (outputs/within/<build>_cut*_line, `openrapm folds --blend-through 0.75 --blend-from 0.25`)

    python scripts/checks/166_power_forecast_refit.py [--write]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("POWER_NEXT10_FAMILY", "zones139")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def load_165():
    """Check 165 as a module; checks 163, 161, 160 and 159 come with it."""
    spec = importlib.util.spec_from_file_location(
        "check165", ROOT / "scripts" / "checks" / "165_power_forecast_weights.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["check165"] = module
    spec.loader.exec_module(module)
    return module


C165 = load_165()
C163, C160, C159 = C165.C163, C165.C160, C165.C159
from openrapm.data.config import load_config  # noqa: E402
from openrapm.power_forecast import WEIGHTS_FILE  # noqa: E402

FAMILY = C159.FAMILY
# each knot's folder: held through a quarter of the season, blended from there
KNOTS = {
    "7%": f"{FAMILY}_cut0715_held",
    "10%": f"{FAMILY}_cut10_held",
    "25%": f"{FAMILY}_cut25_line",
    "50%": f"{FAMILY}_cut50_line",
    "75%": f"{FAMILY}_cut75_line",
}
NO_STATS = ("50%", "75%")  # Power alone on the doubled latest 5: the stats add nothing from half a season (check 164)
# the two days that feed no knot, each between two knots: its folder, and whether it is fitted with the stats
TESTS = {"17.5%": (f"{FAMILY}_cut175_held", True), "62.5%": (f"{FAMILY}_cut625_line", False)}
NAMES = C165.NAMES
FLAT, DOUBLED = C165.FLAT, C165.DOUBLED
POWER_ONLY = C160.columns_of([])


def with_unlogged_players(teams: pd.DataFrame, stints: pd.DataFrame) -> pd.DataFrame:
    """`teams` (game_id, player_id, tm, name) with a row for each player the stints put on the floor in a game whose
    log does not list him: his team is that of the players on his side of the floor.

    `site.player_game_possessions` stops on such a player-game, rightly for the season in progress.  The finished
    seasons hold two, one possession each, 0029600896 of 1997 and 0029700714 of 1998, on days later in a season than
    checks 159 to 165 read."""
    on_floor = pd.concat(
        [stints[["game_id", f"{side}{i}"]].set_axis(["game_id", "player_id"], axis=1).assign(side=side)
         for side in "ha" for i in range(1, 6)], ignore_index=True).drop_duplicates()
    on_floor["player_id"] = on_floor.player_id.astype("int64")
    known = on_floor.merge(teams, on=["game_id", "player_id"], how="left")
    unlogged = known[known.tm.isna()][["game_id", "player_id", "side"]]
    if unlogged.empty:
        return teams
    side_team = (known.dropna(subset=["tm"]).groupby(["game_id", "side"])[["tm", "name"]]
                 .agg(lambda values: values.mode().iloc[0]).reset_index())
    added = unlogged.merge(side_team, on=["game_id", "side"], how="left")[["game_id", "player_id", "tm", "name"]]
    return pd.concat([teams, added], ignore_index=True)


def knot_days(cut: str, within: Path, wowy, tables: dict, stints: dict, teams: dict, folder: str) -> dict:
    """{pair: Day} of a day from the ratings as the published season in progress is shrunk."""
    ratings = C159.day_ratings(wowy, within, folder, step4="published")
    stats, power, game_rows = C163.day_tables(cut, ratings, tables, stints, teams, False)
    return {pair: C163.make_day(stats[pair[0]], power[pair[1]], game_rows) for pair in (FLAT, DOUBLED)}


def raw_weights(cut: str, days: dict, train: list) -> pd.Series:
    """The home edge, the two multipliers and the stats weights of a knot, per unit, fitted on `train`."""
    if cut in NO_STATS:
        day = days[DOUBLED]
        out = pd.Series(0.0, index=NAMES)
        out.iloc[POWER_ONLY] = C160.fitted(day, train, POWER_ONLY) / day.spread[POWER_ONLY]
        return out
    day = days[FLAT]
    return pd.Series(C160.fitted(day, train, C163.ALL_STATS) / day.spread[C163.ALL_STATS], index=NAMES)


def published_knots() -> tuple[dict, dict]:
    """({day: its weights per unit}, {day: its share}) of params/power_forecast.json as it stands."""
    document = json.loads((ROOT / WEIGHTS_FILE).read_text(encoding="utf-8"))
    frames, shares = {}, {}
    for knot in document["knots"]:
        weights = pd.Series(0.0, index=NAMES)
        weights["home"], weights["power_o"] = knot["home_edge"], knot["offense_multiplier"]
        weights["power_d"] = knot["defense_multiplier"]
        for name, value in knot["stats"].items():
            weights[name] = value
        frames[knot["day"]], shares[knot["day"]] = weights, knot["share"]
    return frames, shares


def test_day(name: str, days: dict, knots_of, mean_shares: dict, own_share: dict, published: tuple) -> pd.DataFrame:
    """One test day's readings, each season predicted from the other 29: the line between the refitted knots against a
    direct fit at the day, against Power alone at its own scale (a day fitted with the stats) or plain Power (one
    without), and against the line between the published knots."""
    with_stats = TESTS[name][1]
    test = days[FLAT] if with_stats else days[DOUBLED]
    keep = C163.ALL_STATS if with_stats else POWER_ONLY
    raw_x = test.x * test.spread
    plain = pd.Series(0.0, index=NAMES)
    plain["power_o"], plain["power_d"] = 1.0, 1.0
    arms = {arm: np.full(len(test.y), np.nan)
            for arm in ("line", "direct fit", "Power alone at its own scale", "plain Power", "published weights")}
    for season in test.seasons:
        train = [s for s in test.seasons if s != season]
        mask = test.masks[season]
        share = own_share[int(season)]
        arms["line"][mask] = raw_x[mask] @ C165.line(knots_of(train), mean_shares, share)[NAMES].to_numpy(float)
        arms["direct fit"][mask] = test.x[mask][:, keep] @ C160.fitted(test, train, keep)
        arms["Power alone at its own scale"][mask] = test.x[mask][:, POWER_ONLY] @ C160.fitted(test, train, POWER_ONLY)
        arms["plain Power"][mask] = raw_x[mask] @ plain.to_numpy(float)
        arms["published weights"][mask] = raw_x[mask] @ C165.line(*published, share)[NAMES].to_numpy(float)
    rows = []
    for against in ("direct fit", "Power alone at its own scale", "plain Power", "published weights"):
        rows += [dict(day=name, line_against=against, **row)
                 for row in C165.paired(test.game_rows, test.y, arms["line"], arms[against])]
    return pd.DataFrame(rows)


def accepted(name: str, table: pd.DataFrame) -> bool:
    """The registered rule at one test day, on all seasons: not worse than the direct fit at z +1; and, where the day
    is fitted with the stats, lower than Power alone at its own scale on the mean."""
    every = table[table.era == "all seasons"].set_index("line_against")
    passed = bool(every.z["direct fit"] <= 1.0)
    if TESTS[name][1]:
        passed = passed and bool(every.change["Power alone at its own scale"] < 0.0)
    return passed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--write", action="store_true",
                        help="write the knots to params/power_forecast.json when both test days pass the rule")
    args = parser.parse_args()
    started = time.time()
    cfg = load_config()
    within = ROOT / "outputs" / "within"
    wowy = C159.load_wowy()
    tables = {season: C159.season_game_table(season, cfg) for season in C159.SEASONS}
    stints = {season: C159.season_stints_ids(season, cfg) for season in C159.SEASONS}
    teams = {season: with_unlogged_players(C159.regular_game_teams(season, cfg), stints[season])
             for season in C159.SEASONS}

    days, mean_shares = {}, {}
    for cut, folder in KNOTS.items():
        days[cut] = knot_days(cut, within, wowy, tables, stints, teams, folder)
        mean_shares[cut] = float(np.mean(list(C165.season_shares(within, folder).values())))
    print(f"read ({time.time() - started:.0f}s); the knots' mean shares "
          f"{ {cut: round(share, 4) for cut, share in mean_shares.items()} }", flush=True)

    def knots_of(train: list) -> dict:
        return {cut: raw_weights(cut, days[cut], train) for cut in KNOTS}

    published = published_knots()
    results, passed = [], {}
    for name, (folder, _) in TESTS.items():
        test_days = knot_days(name, within, wowy, tables, stints, teams, folder)
        own_share = C165.season_shares(within, folder)
        table = test_day(name, test_days, knots_of, mean_shares, own_share, published)
        passed[name] = accepted(name, table)
        results.append(table)
        print(f"\nthe line at the {name} day, mean share {np.mean(list(own_share.values())):.4f}, squared error per "
              f"game; the rule {'holds' if passed[name] else 'FAILS'}", flush=True)
        print(table.drop(columns="day").round(3).to_string(index=False), flush=True)
    out = ROOT / "outputs" / "csv"
    out.mkdir(parents=True, exist_ok=True)
    pd.concat(results, ignore_index=True).to_csv(out / "check166_line.csv", index=False)

    knots = knots_of(list(C159.SEASONS))
    old_frames, old_shares = published
    document = {
        "built_from": f"{FAMILY}, the published shrink and prior scales (check 166)",
        "knots": [
            {
                "share": round(mean_shares[cut], 4),
                "day": cut,
                "offense_multiplier": float(knots[cut].power_o),
                "defense_multiplier": float(knots[cut].power_d),
                "home_edge": float(knots[cut].home),
                "stats": {name: float(knots[cut][name]) for name in C160.COLUMNS},
            }
            for cut in KNOTS
        ],
    }
    compare = []
    for knot in document["knots"]:
        old = C165.line(old_frames, old_shares, knot["share"])
        compare.append(dict(day=knot["day"], share=knot["share"],
                            offense_multiplier=knot["offense_multiplier"], published_offense=float(old.power_o),
                            defense_multiplier=knot["defense_multiplier"], published_defense=float(old.power_d),
                            largest_stats_weight=max(abs(v) for v in knot["stats"].values()),
                            published_largest=float(old[C160.COLUMNS].abs().max())))
    compare = pd.DataFrame(compare)
    compare.to_csv(out / "check166_knots.csv", index=False)
    print("\nthe refitted knots beside the published weights read at the same share", flush=True)
    print(compare.round(3).to_string(index=False), flush=True)
    if args.write and all(passed.values()):
        path = ROOT / WEIGHTS_FILE
        path.write_text(json.dumps(document, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {path}", flush=True)
    elif args.write:
        print(f"NOT written: the rule fails at {[name for name, ok in passed.items() if not ok]}", flush=True)
    print(f"done in {time.time() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
