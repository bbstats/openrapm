"""Check 161: an audit of checks 159 and 160, the forecast of a team's next 10 games.

The owner, 2026-11-08: "be patient, be solid, be certain."  Three questions, none of which changes a rating.

    1. Could a past day's Power or last-10 stats have seen the games they predict?  Every day of every season: the games
       read are the fold's own, none is a game predicted, and each fold's recorded leak checks passed.  The
       shrink's multipliers for a season must come from other seasons.
    2. Each season predicted only from the seasons before it, as a forecast would really be made.  Checks 159 and 160
       predict a season from the 29 others, later ones included.
    3. All 14 stats dealt to the wrong teams many times over: how large a gain can a shuffle produce?  Each shuffle
       moves a team's whole line of stats to another team of the same season, as check 159's control does once.

    python scripts/checks/161_power_next10_audit.py [--shuffles 500]
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def load_160():
    """Check 160 as a module; check 159 comes with it, as C159."""
    spec = importlib.util.spec_from_file_location(
        "check160", ROOT / "scripts" / "checks" / "160_power_next10_select.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["check160"] = module
    spec.loader.exec_module(module)
    return module


C160 = load_160()
C159 = C160.C159
from openrapm.data.config import load_config  # noqa: E402
from openrapm.prior_shrink import fold_seasons_used  # noqa: E402

SEED = 161
WALK_FIRST = 2007  # the first season predicted from earlier seasons alone: it has ten seasons to learn from
NEAREST = 8
ERA_SPANS = {"all seasons": (1997, 2026), **C159.ERAS}


def audit_day(cut: str, folder: str, tables: dict, within: Path, ratings: dict) -> dict:
    """The leak audit of one day, every season: counts of what failed, each 0 when the day is clean."""
    out = dict(
        cut=cut,
        seasons=0,
        games_read=0,
        fit_games_mismatch=0,
        read_not_regular=0,
        games_after_not_later=0,
        predicted_also_read=0,
        recorded_checks=0,
        recorded_checks_failed=0,
        own_season_in_shrink=0,
    )
    seasons = sorted(ratings)
    for season in seasons:
        _, played_ids = ratings[season]
        table = C159.season_game_table(season, load_config()) if season not in tables else tables[season]
        played = table[table.game_id.isin(played_ids)]
        record = json.loads((within / folder / f"fold_{season}_r0_f0.json").read_text())
        out["seasons"] += 1
        out["games_read"] += len(played_ids)
        out["fit_games_mismatch"] += int(record["summary"]["fit_games"] != len(played_ids))
        out["read_not_regular"] += int(played.game_id.nunique() != len(played_ids))
        day = played.game_date.max()
        later = table[table.game_date > day].sort_values(["tm", "game_date", "game_id"], kind="stable")
        next_games = later.groupby("tm").head(C159.NEXT_GAMES)
        out["games_after_not_later"] += int(len(next_games) and next_games.game_date.min() <= day)
        out["predicted_also_read"] += int(next_games.game_id.isin(played_ids).sum())
        checks = [value for value in record["checks"].values() if isinstance(value, bool)]
        out["recorded_checks"] += len(checks)
        out["recorded_checks_failed"] += sum(not value for value in checks)
        used = fold_seasons_used(season, seasons, "test", NEAREST)
        out["own_season_in_shrink"] += int(season in used)
    return out


def mean_change_by_era(day, predicted: np.ndarray, base: np.ndarray, seasons=None) -> dict:
    """The mean change in squared error per game, `predicted` minus `base`, by era, each season counting once."""
    frame = pd.DataFrame({"season": day.season, "new": (day.y - predicted) ** 2, "old": (day.y - base) ** 2})
    if seasons is not None:
        frame = frame[frame.season.isin(seasons)]
    by_season = frame.groupby("season")[["new", "old"]].mean()
    change = by_season.new - by_season.old
    return {era: change.loc[first:last] for era, (first, last) in ERA_SPANS.items() if len(change.loc[first:last])}


def walk_forward(day) -> pd.DataFrame:
    """Each season from `WALK_FIRST` predicted from the seasons before it alone, against Power fitted the same way."""
    made = {"all": np.full(len(day.y), np.nan), "base": np.full(len(day.y), np.nan)}
    for test in day.seasons[day.seasons >= WALK_FIRST]:
        train = [s for s in day.seasons if s < test]
        for name, units in (("all", C160.ALL_IN), ("base", [])):
            keep = C160.columns_of(units)
            made[name][day.masks[test]] = day.x[day.masks[test]][:, keep] @ C160.fitted(day, train, keep)
    tested = day.seasons[day.seasons >= WALK_FIRST]
    rows = []
    for era, change in mean_change_by_era(day, made["all"], made["base"], tested).items():
        se = float(change.std(ddof=1) / np.sqrt(len(change)))
        rows.append(
            dict(
                era=era,
                change=float(change.mean()),
                se=se,
                z=float(change.mean() / se),
                seasons_better=int((change < 0).sum()),
                seasons=len(change),
            )
        )
    return pd.DataFrame(rows)


def shuffled(day, rng):
    """`day` with each season's teams' lines of stats dealt to random teams of that season."""
    order = np.arange(len(day.stats))
    for group in day.team_groups:
        order[group] = rng.permutation(group)
    out = copy.copy(day)
    stats = day.stats[order]
    out.x = day.x.copy()
    out.x[:, C160.BASE :] = (stats[day.me] - stats[day.opp]) / day.spread[C160.BASE :]
    out.xtx, out.xty, out.yy, out.rows = C160.season_sums(out.x, day.y, day.season, day.seasons)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shuffles", type=int, default=500)
    parser.add_argument("--out", default="outputs/csv/check161")
    args = parser.parse_args()
    started = time.time()
    cfg = load_config()
    within = ROOT / "outputs" / "within"
    wowy = C159.load_wowy()
    tables = {season: C159.season_game_table(season, cfg) for season in C159.SEASONS}
    stints = {season: C159.season_stints_ids(season, cfg) for season in C159.SEASONS}
    teams = {season: C159.regular_game_teams(season, cfg) for season in C159.SEASONS}
    audits, walks, today = [], [], None
    for cut, folder in C159.CUTS.items():
        ratings = C159.day_ratings(wowy, within, folder)
        audits.append(audit_day(cut, folder, tables, within, ratings))
        team_rows, game_rows = C159.day_rows(cut, ratings, tables, stints, teams)
        day = C160.Day(team_rows, game_rows)
        walks.append(walk_forward(day).assign(cut=cut))
        if cut == C159.TODAY_CUT:
            today = day
        print(f"{cut} ({time.time() - started:.0f}s)", flush=True)
    audit = pd.DataFrame(audits)
    print("\nthe leak audit: every count but seasons, games_read and recorded_checks must be 0", flush=True)
    print(audit.T.to_string(header=False), flush=True)
    walk = pd.concat(walks, ignore_index=True)
    print(
        "\neach season predicted only from the seasons before it, all 14 stats minus Power, squared error per game",
        flush=True,
    )
    print(
        walk[["cut", "era", "change", "se", "z", "seasons_better", "seasons"]].round(3).to_string(index=False),
        flush=True,
    )

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(f"{out}_leaks.csv", index=False)
    walk.to_csv(f"{out}_walk.csv", index=False)

    base, _ = C160.predictions(today, lambda test, train: [])
    real_made, _ = C160.predictions(today, lambda test, train: C160.ALL_IN)
    real = {era: float(change.mean()) for era, change in mean_change_by_era(today, real_made, base).items()}
    rng = np.random.default_rng(SEED)
    draws: dict = {era: [] for era in real}
    for k in range(args.shuffles):
        wrong = shuffled(today, rng)
        made, _ = C160.predictions(wrong, lambda test, train: C160.ALL_IN)
        for era, change in mean_change_by_era(today, made, base).items():
            draws[era].append(float(change.mean()))
        if (k + 1) % 50 == 0:
            print(f"  {k + 1} shuffles ({time.time() - started:.0f}s)", flush=True)
    rows = []
    for era, values in draws.items():
        values = np.array(values)
        rows.append(
            dict(
                era=era,
                real_change=real[era],
                shuffled_mean=values.mean(),
                shuffled_sd=values.std(ddof=1),
                shuffled_lowest=values.min(),
                shuffled_at_or_below_real=int((values <= real[era]).sum()),
                shuffles=len(values),
            )
        )
    table = pd.DataFrame(rows)
    table.to_csv(f"{out}_shuffles.csv", index=False)
    print("\nall 14 stats on the wrong teams, 10% day: change in squared error per game against Power", flush=True)
    print(table.round(3).to_string(index=False), flush=True)
    print(f"done in {time.time() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
