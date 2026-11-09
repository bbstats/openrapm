"""Check 164: are the last-10 stats a stand-in for an easier upcoming schedule?

The owner, 2026-11-09: "are we sure that the stats we are adding aren't just projecting whether or not a team is
going to be playing easier competition?".  notes/DECISIONS.md, "Check 164", holds the reading, written before any
result.

Words used below
    hindsight strength  a team's margin per 100 possessions over its whole regular season with one game taken out.  It
                        reads the games after the day, so no forecast could know it: here it is the strongest control of
                        who the opponent really was
    stats part          what the 14 last-10 stats add to a team's forecast, from a fit on the other seasons

    POWER_NEXT10_FAMILY=zones139 python scripts/checks/164_power_next10_schedule.py [--out outputs/csv/check164]
"""

from __future__ import annotations

import argparse
import importlib.util
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
C163.DECAYS[:] = [1.0]  # flat, the owner's choice: no other weighting is read here
from openrapm.data.config import load_config  # noqa: E402

ERA_SPANS = {"all seasons": (1997, 2026), **C159.ERAS}


def hindsight(tables: dict) -> dict:
    """{season: frame of game_id, tm, opp_strength}: each opponent's hindsight strength for the game."""
    out = {}
    for season, table in tables.items():
        rows = table[["game_id", "tm", "opp"]].copy()
        rows["points"] = (table.pts_o - table.pts_d).to_numpy(float)
        rows["poss"] = ((table.poss_o + table.poss_d) / 2).to_numpy(float)
        totals = rows.groupby("tm")[["points", "poss"]].sum()
        rows["without"] = 100 * (rows.tm.map(totals.points) - rows.points) / (rows.tm.map(totals.poss) - rows.poss)
        strength = rows.set_index(["game_id", "tm"]).without
        keys = pd.MultiIndex.from_arrays([rows.game_id, rows.opp])
        out[season] = pd.DataFrame(
            {
                "game_id": rows.game_id.to_numpy(),
                "tm": rows.tm.to_numpy(),
                "opp_strength": strength.reindex(keys).to_numpy(),
            }
        )
    return out


def past_schedule(tables: dict, strength: dict, ratings: dict) -> pd.Series:
    """The mean hindsight strength of each team's last 10 opponents through the day, indexed by (season, tm)."""
    parts = []
    for season, (_, played_ids) in ratings.items():
        table = tables[season]
        played = table[table.game_id.isin(played_ids)].sort_values(["tm", "game_date", "game_id"], kind="stable")
        last = (
            played.groupby("tm").tail(C159.LAST_GAMES)[["game_id", "tm"]].merge(strength[season], on=["game_id", "tm"])
        )
        parts.append(last.groupby("tm").opp_strength.mean().rename("past_schedule").to_frame().assign(season=season))
    joined = pd.concat(parts).reset_index().set_index(["season", "tm"]).past_schedule
    return joined - joined.groupby(level=0).transform("mean")


def predictions(x: np.ndarray, free: int, y: np.ndarray, season: np.ndarray) -> np.ndarray:
    """Each season predicted from the others, the first `free` columns unpenalized and the penalty picked inside."""
    penalized = np.array([False] * free + [True] * (x.shape[1] - free))
    made, _ = C159.fit_season_out(x / x.std(axis=0), y, season, penalized)
    return made


def paired(game_rows: pd.DataFrame, y: np.ndarray, base: np.ndarray, full: np.ndarray) -> list:
    """Stats minus no stats, by era: the change in squared error per game and in a team's 10-game average."""
    plain = C159.score_frame("base", game_rows, y, base).set_index("season")
    stats = C159.score_frame("stats", game_rows, y, full).set_index("season")
    rows = []
    for era, (first, last) in ERA_SPANS.items():
        row = dict(era=era, seasons=int(last - first + 1))
        for unit in ("game", "team"):
            diff = (stats[unit] - plain[unit]).loc[first:last]
            se = float(diff.std(ddof=1) / np.sqrt(len(diff)))
            row.update(
                {
                    f"{unit}_change": float(diff.mean()),
                    f"{unit}_se": se,
                    f"{unit}_z": float(diff.mean() / se),
                    f"{unit}_better": int((diff < 0).sum()),
                }
            )
        rows.append(row)
    return rows


def run_day(
    cut: str, ratings: dict, tables: dict, stints: dict, teams: dict, strength: dict
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(the four designs' gains, the stats part's correlations) of one day."""
    stats, power, game_rows = C163.day_tables(cut, ratings, tables, stints, teams, False)
    day = C163.make_day(stats[1.0], power[1.0], game_rows)
    rows = pd.concat([stats[1.0][C159.ADDED], power[1.0]], axis=1).reset_index()
    rows.columns = ["season", "tm", *rows.columns[2:]]
    team_power = rows[["power_o", "power_d"]].to_numpy(float)
    joined = game_rows.merge(pd.concat(strength.values(), ignore_index=True), on=["game_id", "tm"], how="left")
    opp_strength = joined.opp_strength.to_numpy(float)[:, None]
    if np.isnan(opp_strength).any():
        raise SystemExit(f"{cut}: a game has no hindsight strength")
    past = past_schedule(tables, strength, ratings).reindex(pd.MultiIndex.from_frame(rows[["season", "tm"]])).to_numpy()
    past_me = past[day.me][:, None]
    y, season, home = day.y, day.season, game_rows.home.to_numpy(float)[:, None]
    own, theirs = day.stats[day.me], day.stats[day.opp]
    power_diff = team_power[day.me] - team_power[day.opp]
    designs = {
        "1: as it stands": (np.hstack([home, power_diff]), own - theirs),
        "2: plus hindsight opponent strength": (np.hstack([home, power_diff, opp_strength]), own - theirs),
        "3: team side only, hindsight opponent": (np.hstack([home, team_power[day.me], opp_strength]), own),
        "4: as 3, plus the team's past schedule": (np.hstack([home, team_power[day.me], opp_strength, past_me]), own),
        "5: the team's own stats alone": (np.hstack([home, power_diff]), own),
        "6: the opponents' stats alone": (np.hstack([home, power_diff]), theirs),
    }
    gains = []
    for name, (base, added) in designs.items():
        without = predictions(base, base.shape[1], y, season)
        with_stats = predictions(np.hstack([base, added]), base.shape[1], y, season)
        gains += [dict(cut=cut, design=name, **row) for row in paired(game_rows, y, without, with_stats)]

    # the stats part of each team, from a fit on the other seasons, against its schedule
    links = []
    team_season = rows.season.to_numpy()
    next_games = joined.groupby(["season", "tm"]).agg(
        next_schedule=("opp_strength", "mean"), home_share=("home", lambda h: (h > 0).mean())
    )
    for test in day.seasons:
        train = [s for s in day.seasons if s != test]
        beta = C160.fitted(day, train, C163.ALL_STATS)
        here = np.flatnonzero(team_season == test)
        part = (day.stats[here] / day.spread[C160.BASE :]) @ beta[C160.BASE :]
        power_part = team_power[here] @ (beta[1 : C160.BASE] / day.spread[1 : C160.BASE])
        frame = (
            next_games.loc[test]
            .reindex(rows.tm.to_numpy()[here])
            .assign(past_schedule=past[here], stats_part=part, power_part=power_part)
        )
        frame = frame.dropna()
        spread = frame.stats_part.var(ddof=1)
        links.append(
            dict(
                cut=cut,
                season=int(test),
                next_corr=float(frame.stats_part.corr(frame.next_schedule)),
                past_corr=float(frame.stats_part.corr(frame.past_schedule)),
                home_corr=float(frame.stats_part.corr(frame.home_share)),
                power_next_corr=float(frame.power_part.corr(frame.next_schedule)),
                next_slope=float(frame.stats_part.cov(frame.next_schedule) / spread),
            )
        )
    return pd.DataFrame(gains), pd.DataFrame(links)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="outputs/csv/check164")
    args = parser.parse_args()
    started = time.time()
    cfg = load_config()
    within = ROOT / "outputs" / "within"
    wowy = C159.load_wowy()
    tables = {season: C159.season_game_table(season, cfg) for season in C159.SEASONS}
    stints = {season: C159.season_stints_ids(season, cfg) for season in C159.SEASONS}
    teams = {season: C159.regular_game_teams(season, cfg) for season in C159.SEASONS}
    strength = hindsight(tables)
    print(f"ratings: {C159.FAMILY}", flush=True)
    all_gains, all_links = [], []
    for cut, folder in C159.CUTS.items():
        gains, links = run_day(cut, C159.day_ratings(wowy, within, folder), tables, stints, teams, strength)
        all_gains.append(gains)
        all_links.append(links)
        print(f"{cut} ({time.time() - started:.0f}s)", flush=True)
    gains, links = pd.concat(all_gains, ignore_index=True), pd.concat(all_links, ignore_index=True)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    gains.to_csv(f"{out}_gains.csv", index=False)
    links.to_csv(f"{out}_links.csv", index=False)
    print("\nstats minus no stats, squared error per game and a team's 10-game average", flush=True)
    columns = ["cut", "design", "era", "game_change", "game_se", "game_z", "game_better", "team_change", "team_z"]
    print(gains[columns].round(3).to_string(index=False), flush=True)
    print("\nthe stats part's correlation inside a season, mean of 30 seasons, standard error, z", flush=True)
    rows = []
    for cut, part in links.groupby("cut", sort=False):
        for name in ("next_corr", "past_corr", "home_corr", "power_next_corr", "next_slope"):
            values = part[name].to_numpy(float)
            se = float(values.std(ddof=1) / np.sqrt(len(values)))
            rows.append(dict(cut=cut, reading=name, mean=float(values.mean()), se=se, z=float(values.mean() / se)))
    table = pd.DataFrame(rows)
    table.to_csv(f"{out}_link_summary.csv", index=False)
    print(table.round(3).to_string(index=False), flush=True)
    print(f"done in {time.time() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
