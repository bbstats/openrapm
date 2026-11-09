"""Check 165: the Power table's multipliers and stats weights by point in the season, and a test of the line between.

The owner, 2026-11-09: "One rating, stats fade".  notes/DECISIONS.md, "Check 165", holds the reading, written before any
result.  This writes params/power_forecast.json.

Words used below
    knot     a point of the regular season with fitted multipliers and weights: the 7.15%, 10% and 25% days with flat
             shares and the 14 stats, and the 50% day with the page's doubled latest 5 and no stats
    line     the multipliers and weights at any other point: the straight line between the two knots around it

    POWER_NEXT10_FAMILY=zones139 python scripts/checks/165_power_forecast_weights.py [--write]
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


def load_163():
    """Check 163 as a module; checks 161, 160 and 159 come with it."""
    spec = importlib.util.spec_from_file_location(
        "check163", ROOT / "scripts" / "checks" / "163_power_next10_recency.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["check163"] = module
    spec.loader.exec_module(module)
    return module


C163 = load_163()
C160, C159 = C163.C160, C163.C159
C163.DECAYS[:] = [1.0]
from openrapm.data.config import load_config  # noqa: E402
from openrapm.power_forecast import WEIGHTS_FILE  # noqa: E402

FAMILY = C159.FAMILY
KNOTS = {"7%": "cut0715", "10%": "cut10", "25%": "cut25", "50%": "cut50"}
TEST_FOLDER = f"{FAMILY}_cut175_held"
NAMES = ["home", "power_o", "power_d", *C160.COLUMNS]
FLAT, DOUBLED = (1.0, 1.0), (1.0, C163.STEP)
LAST_KNOT = "50%"


def folder_of(cut: str) -> str:
    return f"{FAMILY}_{KNOTS[cut]}_held"


def season_shares(within: Path, folder: str) -> dict:
    """{season: the share of its regular season its games hold}, from the folder's folds."""
    return {
        season: json.loads((within / folder / f"fold_{season}_r0_f0.json").read_text())["summary"]["season_share"]
        for season in C159.SEASONS
    }


def knot_days(cut: str, within: Path, wowy, tables: dict, stints: dict, teams: dict, folder: str) -> dict:
    """{pair: Day} of a day, flat stats with flat or doubled shares."""
    stats, power, game_rows = C163.day_tables(cut, C159.day_ratings(wowy, within, folder), tables, stints, teams, False)
    return {pair: C163.make_day(stats[pair[0]], power[pair[1]], game_rows) for pair in (FLAT, DOUBLED)}


def raw_weights(cut: str, days: dict, train: list) -> pd.Series:
    """The home edge, the two multipliers and the stats weights of a knot, per unit, fitted on `train`."""
    if cut == LAST_KNOT:
        day = days[DOUBLED]
        keep = C160.columns_of([])
        out = pd.Series(0.0, index=NAMES)
        out.iloc[keep] = C160.fitted(day, train, keep) / day.spread[keep]
        return out
    day = days[FLAT]
    return pd.Series(C160.fitted(day, train, C163.ALL_STATS) / day.spread[C163.ALL_STATS], index=NAMES)


def line(frames: dict, shares: dict, share: float) -> pd.Series:
    """The weights at `share` by the straight line between the knots."""
    cuts = sorted(frames, key=lambda cut: shares[cut])
    x = [shares[cut] for cut in cuts]
    table = pd.DataFrame({cut: frames[cut] for cut in cuts})[cuts]
    return pd.Series({name: float(np.interp(share, x, table.loc[name].to_numpy(float))) for name in table.index})


def paired(game_rows, y, new, old) -> list:
    """`new` minus `old`, squared error per game, by era."""
    a, b = (
        C159.score_frame("a", game_rows, y, new).set_index("season"),
        C159.score_frame("b", game_rows, y, old).set_index("season"),
    )
    rows = []
    for era, (first, last) in {"all seasons": (1997, 2026), **C159.ERAS}.items():
        diff = (a.game - b.game).loc[first:last]
        se = float(diff.std(ddof=1) / np.sqrt(len(diff)))
        rows.append(
            dict(
                era=era,
                change=float(diff.mean()),
                se=se,
                z=float(diff.mean() / se),
                better=int((diff < 0).sum()),
                seasons=len(diff),
                rmse=float(np.sqrt(a.game.loc[first:last].mean())),
            )
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--write", action="store_true", help="write the knots to params/power_forecast.json")
    args = parser.parse_args()
    started = time.time()
    cfg = load_config()
    within = ROOT / "outputs" / "within"
    wowy = C159.load_wowy()
    tables = {season: C159.season_game_table(season, cfg) for season in C159.SEASONS}
    stints = {season: C159.season_stints_ids(season, cfg) for season in C159.SEASONS}
    teams = {season: C159.regular_game_teams(season, cfg) for season in C159.SEASONS}

    days, mean_shares = {}, {}
    for cut in KNOTS:
        days[cut] = knot_days(cut, within, wowy, tables, stints, teams, folder_of(cut))
        mean_shares[cut] = float(np.mean(list(season_shares(within, folder_of(cut)).values())))
    test_days = knot_days("17.5%", within, wowy, tables, stints, teams, TEST_FOLDER)
    print(f"read ({time.time() - started:.0f}s); the knots' mean shares {mean_shares}", flush=True)

    test = test_days[FLAT]
    own_share = season_shares(within, TEST_FOLDER)
    print(f"the 17.5% day's mean share {np.mean(list(own_share.values())):.4f}", flush=True)
    raw_x = test.x * test.spread
    arms = {name: np.full(len(test.y), np.nan) for name in ("line", "direct fit", "Power alone, flat shares")}
    power_keep = C160.columns_of([])
    for season in test.seasons:
        train = [s for s in test.seasons if s != season]
        mask = test.masks[season]
        knots = {cut: raw_weights(cut, days[cut], train) for cut in KNOTS}
        weights = line(knots, mean_shares, own_share[int(season)])
        arms["line"][mask] = raw_x[mask] @ weights[NAMES].to_numpy(float)
        arms["direct fit"][mask] = test.x[mask] @ C160.fitted(test, train, C163.ALL_STATS)
        arms["Power alone, flat shares"][mask] = test.x[mask][:, power_keep] @ C160.fitted(test, train, power_keep)
    results = []
    for against in ("direct fit", "Power alone, flat shares"):
        results += [dict(against=against, **row) for row in paired(test.game_rows, test.y, arms["line"], arms[against])]
    table = pd.DataFrame(results)
    print("\nthe line at the 17.5% day, squared error per game", flush=True)
    print(table.round(3).to_string(index=False), flush=True)

    every = list(test.seasons)
    knots = {cut: raw_weights(cut, days[cut], every) for cut in KNOTS}
    document = {
        "built_from": FAMILY,
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
    for knot in document["knots"]:
        print(
            f"{knot['day']}: share {knot['share']}, offense multiplier {knot['offense_multiplier']:.3f}, defense "
            f"{knot['defense_multiplier']:.3f}, home {knot['home_edge']:.2f}, stats weights' largest "
            f"{max(abs(v) for v in knot['stats'].values()):.3f}",
            flush=True,
        )
    if args.write:
        path = ROOT / WEIGHTS_FILE
        path.write_text(json.dumps(document, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {path}", flush=True)
    print(f"done in {time.time() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
