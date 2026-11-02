"""`openrapm wins`: the wins over starter average, written into a ratings table.

    openrapm wins --name NAME    the columns wins.WINS_COLUMNS in outputs/season_ratings_<NAME>.parquet, every season,
                                 and outputs/csv/wins_<NAME>_seasons.csv: each season's bar and win model

`openrapm build` calls the same `add_wins` on the table it has just made, so a built table always carries them; this
command is for a table built before them.

Every season is computed from its own regular-season stints, except its win model when the season is still in
progress: nothing is fitted on a season whose Finals are not over (data/seasons.py), so it reads the latest finished
season's.  Its bar and its pace effects are its own, like its ratings: they describe the season, they train nothing.

Example: `openrapm wins --name product --seasons 2027` writes 2027's columns over yesterday's and leaves every other
season's as they were.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .. import swap_test, wins
from ..data.config import resolve
from ..data.seasons import last_finished, share_played
from ..data.stints import load_season_stints

RATING_COLUMNS = ["player_id", "rating_off", "rating_def", "rating_total"]


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


def season_columns(cfg: dict, rows: pd.DataFrame, settings, kept: dict) -> tuple[pd.DataFrame, dict]:
    """(one season's rows of a ratings table with the wins columns, its summary line)."""
    season = int(rows.season.iloc[0])
    choices = settings.wins
    stints, seconds = season_stints(cfg, season, choices.phases)
    model = win_model_for(cfg, season, stints, seconds, choices.phases, kept)
    pace = wins.pace_effects(stints, seconds, choices.pace_penalty, choices.pace_min_seconds)
    rated = rows[RATING_COLUMNS]
    bar = wins.starter_bar(rated, wins.starts_and_minutes(stints, seconds), pace, choices.starters)
    weight = wins.leverage_inside_games(stints, seconds, model)
    computed = wins.season_wins(stints, seconds, rated, weight, pace, bar, choices.points_per_win)
    return wins.wins_columns(rows, computed), season_summary(cfg, stints, seconds, rated, model, bar, season)


def add_wins(cfg: dict, table: pd.DataFrame, settings, seasons=None) -> tuple[pd.DataFrame, pd.DataFrame]:
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
        parts[season], line = season_columns(cfg, table[table.season == season], settings, kept)
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


def run_wins(cfg: dict, settings, name: str, seasons=None) -> int:
    """`openrapm wins`: the columns written into outputs/season_ratings_<name>.parquet, in place."""
    path = resolve(cfg, "outputs") / f"season_ratings_{name}.parquet"
    if not path.exists():
        raise SystemExit(f"{path} does not exist: `openrapm build --name {name}` writes it")
    table, summary = add_wins(cfg, pd.read_parquet(path), settings, seasons)
    table.to_parquet(path, index=False)
    write_wins_summary(summary, cfg, name)
    print(f"wrote {path} ({len(table)} rows)", flush=True)
    latest = int(table.season.max())
    top = table[table.season == latest].nlargest(5, "wosa")
    print(f"{latest}: " + ", ".join(f"{row.player_name} {row.wosa:+.1f}" for row in top.itertuples()), flush=True)
    return 0
