"""`openrapm wins`: the wins over starter average, written into a ratings table.

    openrapm wins --name NAME    the columns wins.WINS_COLUMNS in outputs/season_ratings_<NAME>.parquet, every season,
                                 and outputs/csv/wins_<NAME>_seasons.csv: each season's bar and win model
    openrapm wins --name NAME --rebuild-regular --jobs 4
                                 first each finished season's rating from its regular season alone, about 4 minutes a
                                 season a process, in outputs/season_ratings_<NAME>_regular.parquet; a stopped run goes
                                 on from the seasons it finished

`openrapm build` calls the same `add_wins` on the table it has just made, so a built table always carries them; this
command is for a table built before them.

Every season is computed from its own regular-season stints, except its win model when the season is still in
progress: nothing is fitted on a season whose Finals are not over (data/seasons.py), so it reads the latest finished
season's.  Its bar and its pace effects are its own, like its ratings: they describe the season, they train nothing.

The lock (the owner, 2026-11-02: "ideally we just lock WOSA after reg season").  The wins count the regular season, and
the published rating holds the playoffs too, so a season with playoff games reads a rating from its regular season
alone, and playoff games cannot move it.  A season in progress has no such rating to build: until its first playoff game
its own rows are the regular season, each morning stores them, and after that game the last rows stored stand.  So WOSA
stops changing the morning after the last regular-season game.

Example: `openrapm wins --name product --seasons 2027` writes 2027's columns over yesterday's and leaves every other
season's as they were.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from .. import swap_test, wins
from ..data.config import resolve
from ..data.seasons import last_finished, share_played
from ..data.stints import has_playoff_stints, load_season_stints
from .published import refuse_published_name

RATING_COLUMNS = ["player_id", "rating_off", "rating_def", "rating_total"]
# what the regular-season ratings file keeps of a ratings table
REGULAR_COLUMNS = ["season", "player_id", "player_name", "rating_off", "rating_def", "rating_total", "prior_off",
                   "prior_def", "u_off", "u_def", "poss_off", "poss_def", "poss_season"]


def season_stints(cfg: dict, season: int, phases: tuple) -> tuple[pd.DataFrame, np.ndarray]:
    """(the season's stints in time order within each game, each with its team ids; each stint's seconds)."""
    stints = swap_test.order_stints(load_season_stints(season, cfg, wins.STINT_COLUMNS, phases=phases))
    return stints, swap_test.stint_durations(stints)


def win_model_for(cfg: dict, season: int, stints: pd.DataFrame, seconds: np.ndarray, phases: tuple,
                  kept: dict) -> wins.WinModel:
    """The season's own win model when its Finals are over, else the latest finished season's, fitted once a run and
    kept in `kept`: it costs a second, and a csv of them could go stale."""
    finished = last_finished(cfg)
    if season <= finished:
        return wins.fit_win_model(stints, seconds, season)
    if finished not in kept:
        kept[finished] = wins.fit_win_model(*season_stints(cfg, finished, phases), finished)
    return kept[finished]


def season_summary(cfg: dict, stints: pd.DataFrame, seconds: np.ndarray, rows: pd.DataFrame, model: wins.WinModel,
                   bar: wins.Bar, season: int) -> dict:
    """One line of the summary: what this season's wins read, and the standings' own numbers for a finished season."""
    results = wins.game_results(stints)
    weight = wins.leverage_inside_games(stints, seconds, model)
    margin = wins.expected_margins(stints, rows)[0]
    finished = season <= last_finished(cfg)
    return dict(season=season, share_of_season_played=float(share_played(season, cfg)), win_model_season=model.season,
                home_edge=model.home_edge, strength_slope=model.strength_slope,
                lead_kept_at_half=float(wins.lead_kept(0.5, model)), spread_tipoff=float(wins.spread_at(1.0, model)),
                games_margin_zero=model.n_ties, bar_off=bar.off, bar_def=bar.defense, bar_total=bar.total,
                bar_pace=bar.pace, bar_players=bar.n, bar_fewest_starts=bar.fewest_starts,
                net_points_per_win=wins.net_points_per_win(results) if finished else np.nan,
                rating_points_per_win=wins.rating_points_per_win(stints, weight, margin) if finished else np.nan)


def regular_path(cfg: dict, name: str) -> Path:
    return resolve(cfg, "outputs") / f"season_ratings_{name}_regular.parquet"


def load_regular(cfg: dict, name: str) -> pd.DataFrame | None:
    """The regular-season ratings stored under `name`, or None."""
    path = regular_path(cfg, name)
    return pd.read_parquet(path) if path.exists() else None


def write_regular(regular: pd.DataFrame | None, cfg: dict, name: str) -> None:
    if regular is not None:
        regular.to_parquet(regular_path(cfg, name), index=False)


def with_seasons_so_far(cfg: dict, table: pd.DataFrame, regular: pd.DataFrame | None) -> pd.DataFrame | None:
    """`regular` with each season of `table` that has no playoff game yet in place of its stored rows: its own rows are
    the regular season so far, and the last rows stored before its first playoff game are the lock."""
    current = [int(season) for season in table.season.unique() if not has_playoff_stints(int(season), cfg)]
    if not current:
        return regular
    fresh = table[table.season.isin(current)][REGULAR_COLUMNS]
    if regular is None:
        return fresh.reset_index(drop=True)
    kept = regular[~regular.season.isin(current)]
    return pd.concat([kept, fresh], ignore_index=True).sort_values("season", kind="stable").reset_index(drop=True)


def rating_rows(cfg: dict, season: int, rows: pd.DataFrame, regular: pd.DataFrame | None) -> pd.DataFrame:
    """The rating a season's wins read: its own rows until it has a playoff game, then its regular-season rows."""
    if not has_playoff_stints(season, cfg):
        return rows
    locked = None if regular is None else regular[regular.season == season]
    if locked is None or locked.empty:
        raise SystemExit(f"{season} has playoff games and no rating from its regular season alone: run `openrapm wins "
                         "--name NAME --rebuild-regular`")
    return locked


def season_columns(cfg: dict, rows: pd.DataFrame, settings, kept: dict,
                   regular: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict]:
    """(one season's rows of a ratings table with the wins columns, its summary line).  The rows keep their published
    ratings; the wins read `rating_rows`."""
    season = int(rows.season.iloc[0])
    choices = settings.wins
    stints, seconds = season_stints(cfg, season, choices.phases)
    model = win_model_for(cfg, season, stints, seconds, choices.phases, kept)
    pace = wins.pace_effects(stints, seconds, choices.pace_penalty, choices.pace_min_seconds)
    rated = rating_rows(cfg, season, rows, regular)[RATING_COLUMNS]
    bar = wins.starter_bar(rated, wins.starts_and_minutes(stints, seconds), pace, choices.starters)
    weight = wins.leverage_inside_games(stints, seconds, model)
    computed = wins.season_wins(stints, seconds, rated, weight, pace, bar, choices.points_per_win)
    return wins.wins_columns(rows, computed), season_summary(cfg, stints, seconds, rated, model, bar, season)


def add_wins(cfg: dict, table: pd.DataFrame, settings, seasons=None,
             regular: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(`table` with the wins columns, the summary of the seasons computed).  `seasons` are computed, every season of
    the table when not given; any other season must already carry the columns and keeps them.  The rows stay in the
    table's order within each season and the seasons in order."""
    started, kept = time.time(), {}
    known = sorted(int(season) for season in table.season.unique())
    chosen = known if seasons is None else [season for season in known if season in set(seasons)]
    missing = [column for column in wins.WINS_COLUMNS if column not in table.columns]
    if missing and set(chosen) != set(known):
        raise SystemExit(f"the table lacks {missing}: compute every season first, `openrapm wins` without --seasons")
    parts, lines = {}, []
    for season in chosen:
        parts[season], line = season_columns(cfg, table[table.season == season], settings, kept, regular)
        lines.append(line)
    columns = [*[column for column in table.columns if column not in wins.WINS_COLUMNS], *wins.WINS_COLUMNS]
    rows = [parts[season][columns] if season in parts else table[table.season == season][columns] for season in known]
    print(f"  wins over starter average: {len(chosen)} seasons ({time.time() - started:.0f}s)", flush=True)
    return pd.concat(rows, ignore_index=True), pd.DataFrame(lines)


def write_wins_summary(summary: pd.DataFrame, cfg: dict, name: str) -> None:
    """outputs/csv/wins_<name>_seasons.csv; a season not in `summary` keeps its line, so a morning that computes one
    season leaves the rest."""
    path = resolve(cfg, "outputs") / "csv" / f"wins_{name}_seasons.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        before = pd.read_csv(path)
        summary = pd.concat([before[~before.season.isin(summary.season)], summary], ignore_index=True)
    summary.sort_values("season").to_csv(path, index=False)


def run_wins(cfg: dict, settings, name: str, seasons=None, rebuild_regular: bool = False, jobs: int = 1) -> int:
    """`openrapm wins`: the columns written into outputs/season_ratings_<name>.parquet, in place; with
    `rebuild_regular`, each finished season's rating from its regular season alone first, in `jobs` processes.  A child
    of those processes rates its seasons and stops; its parent does the rest."""
    refuse_published_name(cfg, name)
    from . import parallel
    if rebuild_regular and parallel.is_child():
        from .build import rate_regular_seasons, regular_parts_folder
        rate_regular_seasons(cfg, settings, seasons, regular_parts_folder(cfg, name))
        return 0
    path = resolve(cfg, "outputs") / f"season_ratings_{name}.parquet"
    if not path.exists():
        raise SystemExit(f"{path} does not exist: `openrapm build --name {name}` writes it")
    table, regular = pd.read_parquet(path), load_regular(cfg, name)
    if rebuild_regular:
        from .build import regular_ratings
        asked = table if seasons is None else table[table.season.isin(seasons)]
        regular = regular_ratings(cfg, settings, asked, regular, name, jobs)
        write_regular(regular, cfg, name)
    table, summary = add_wins(cfg, table, settings, seasons, regular)
    table.to_parquet(path, index=False)
    write_regular(with_seasons_so_far(cfg, table, regular), cfg, name)
    write_wins_summary(summary, cfg, name)
    print(f"wrote {path} ({len(table)} rows)", flush=True)
    latest = int(table.season.max())
    top = table[table.season == latest].nlargest(5, "wosa")
    print(f"{latest}: " + ", ".join(f"{row.player_name} {row.wosa:+.1f}" for row in top.itertuples()), flush=True)
    return 0
