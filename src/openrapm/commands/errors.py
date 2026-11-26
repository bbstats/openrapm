"""`openrapm errors`: every published rating's standard error (rating_error.py), from the folds that hold back a
tenth of a season's games.

Reads, for each season asked for:
    outputs/within/<TAG>/                                     the season's folds, the first of --folds that holds it
    outputs/csv/season_ratings_<NAME>_step4_multipliers.csv   step 4's multipliers of the season, as published
Writes outputs/rating_errors_<NAME>.parquet: player_id, season, se_off, se_def, se_total, repeats.  The seasons
asked for go in place of their rows there; every other season's rows stay.

Each fold's ratings are shrunk with the season's published multipliers (prior_shrink.shrink_fold), so a fold's
table is the published table made from nine tenths of the games.  The multipliers are not refitted per fold: they
are fitted on the 8 seasons nearest the rated one, and one season's tenth barely moves them.

A finished season's folds are rated by the test setting's boosters, which leave out the seasons either side of it
as well; the published ratings' boosters leave out the season alone.  The spread of the folds is read the same.  The
season in progress has its own folds, rated as its build rates it (`openrapm folds --in-progress`).

    openrapm errors                                    1997-2026 from rd143_k10
    openrapm errors --seasons 2027 --folds daily_k10   the season in progress, each morning
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from .. import prior_shrink, rating_error
from ..data import within_folds
from ..data.config import resolve
from .published import refuse_published_name

COLUMNS = ["player_id", "season", *rating_error.ERRORS, "repeats"]


def errors_path(cfg: dict, name: str) -> Path:
    """outputs/rating_errors_<name>.parquet, beside the ratings table of the same name."""
    return resolve(cfg, "outputs") / f"rating_errors_{name}.parquet"


def season_folds(cfg: dict, tags: list, season: int) -> tuple[list, str]:
    """(the season's folds, the tag they came from): the first of `tags` that holds any fold of the season."""
    for tag in tags:
        directory = resolve(cfg, "within") / tag
        if any(directory.glob(f"fold_{season}_*.json")):
            return within_folds.load_folds(directory, (season, season)), tag
    raise SystemExit(f"no folds for {season} in outputs/within/ under {', '.join(tags)}; `openrapm folds "
                     "--folds-per-repeat 10` writes them")


def season_errors(folds: list, season: int, factors: tuple) -> pd.DataFrame:
    """One season's standard errors, `COLUMNS`: each fold's ratings after step 4 at `factors`
    (prior_shrink.season_factors), then the jackknife over them."""
    ratings = pd.concat([prior_shrink.shrink_fold(fold.rows, season, factors)[["player_id", *rating_error.RATINGS]]
                         .assign(repeat=fold.repeat, fold=fold.fold) for fold in folds], ignore_index=True)
    return rating_error.jackknife(ratings).assign(season=season)[COLUMNS]


def run_errors(cfg: dict, name: str, seasons, tags: list) -> int:
    """`openrapm errors`: the module docstring."""
    refuse_published_name(cfg, name)
    started = time.time()
    path = resolve(cfg, "outputs") / "csv" / f"season_ratings_{name}_step4_multipliers.csv"
    if not path.exists():
        raise SystemExit(f"{path} does not exist; `openrapm build --name {name}` writes it")
    multipliers = pd.read_csv(path)
    tables = []
    for season in seasons:
        folds, tag = season_folds(cfg, tags, int(season))
        table = season_errors(folds, int(season), prior_shrink.season_factors(multipliers, int(season)))
        regulars = table[table.repeats > 0]
        print(f"  {season}: {len(folds)} folds of {tag}; {len(regulars)} of {len(table)} players with a standard "
              f"error, median {regulars.se_off.median():.2f} offense, {regulars.se_def.median():.2f} defense, "
              f"{regulars.se_total.median():.2f} total", flush=True)
        tables.append(table)
    fresh = pd.concat(tables, ignore_index=True)
    out = errors_path(cfg, name)
    if out.exists():
        kept = pd.read_parquet(out)
        fresh = pd.concat([kept[~kept.season.isin(set(fresh.season))], fresh], ignore_index=True)
    fresh = fresh.sort_values(["season", "player_id"]).reset_index(drop=True)
    fresh.to_parquet(out, index=False)
    print(f"wrote {out} ({len(fresh)} rows, {fresh.season.nunique()} seasons, {time.time() - started:.0f}s)",
          flush=True)
    return 0
