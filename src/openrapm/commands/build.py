"""`openrapm rate`: steps 1-3 for every season asked for, in one run, the table written once at the end.

Step 1 adds each trainable season's normal equations to one `rapm.SeasonSums` per target.  Then, season by
season: the labels leave the rated season out (`rapm.career_labels`), the box-score prior learns them and answers
for the rated season's own box score (`box_prior.season_priors`), and the ridge pulls the season's RAPM toward
that answer (`prior_rapm.rate_season`).  The table is outputs/season_ratings_<name>_step3.parquet, in the
published schema (`prior_rapm.published_table`).

Words used below
    trainable season   a season whose Finals are over (data/seasons.py).  Only those reach the labels, and the
                       season panel the prior trains on is read through `data.panel.trainable_panel`
    unseen seasons     the rated season, plus the seasons either side of it in the test setting: no label and
                       no training row may come from them

Each season is rated on its own, so one run over thirty seasons writes the same bytes as three runs of ten
seasons stitched together (checked when the cleanup began, 2026-10-09).

Example: `openrapm rate --seasons 2026 --name check` writes the 2026 rows alone to
outputs/season_ratings_check_step3.parquet; their values equal the 2026 rows of a thirty-season run.
"""
from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from .. import box_prior, prior_rapm, rapm
from ..data.config import booster_params, resolve
from ..data.designs import DesignCache
from ..data.panel import trainable_panel
from ..data.seasons import all_seasons, trainable
from ..data.tables import player_names

SIDES = ("O", "D")


def check_seasons(seasons, cfg: dict) -> list:
    """The seasons to rate, ascending: every season in config.yaml when none are given."""
    known = all_seasons(cfg)
    chosen = sorted(seasons) if seasons else list(known)
    outside = [season for season in chosen if season not in known]
    if outside:
        raise SystemExit(f"seasons {outside} are outside config.yaml's {known[0]}-{known[-1]}")
    return chosen


def describe(settings, seasons) -> str:
    """One line naming what this run uses."""
    targets, prior = settings.targets, settings.prior
    return (f"rate {seasons[0]}-{seasons[-1]} ({len(seasons)} seasons): offence target {targets.offence}, defence "
            f"target {targets.defence}; inputs {prior.feature_set}; booster {prior.booster_file}; {prior.n_folds} "
            f"player folds, assignment {prior.fold_assignment}; neighbours left out {settings.exclude_neighbours}")


def accumulate_rapm(designs: DesignCache, seasons, settings) -> dict:
    """Step 1: {target: rapm.SeasonSums} over `seasons`, one per distinct target of the two sides.

    The seasons are added in ascending order: a player's column is fixed by the first season that brings him,
    and the published ratings were built with that column order.
    """
    started = time.time()
    sums = {}
    for target in settings.targets.distinct():
        sums[target] = rapm.SeasonSums(settings.labels.min_possessions)
        for season in sorted(seasons):
            sums[target].add(season, designs.season(season, target))
        print(f"  step 1: {len(seasons)} seasons on {target}, {len(sums[target].player_ids)} players "
              f"({time.time() - started:.0f}s)", flush=True)
    return sums


def unseen_seasons(season: int, settings) -> list:
    """The rated season, and in the test setting the season either side of it.

    Example: rating 2026 gives [2026] in the product setting and [2025, 2026, 2027] in the test setting.
    """
    reach = int(settings.exclude_neighbours)
    return list(range(season - reach, season + reach + 1))


def saved_side(prior: box_prior.SidePrior, params: dict, settings, unseen) -> dict:
    """One side's boosters in the layout the published build wrote (scripts/62_single_year_board.py --save_models
    at the tag archive/research-2026-10), so `openrapm folds` reads either file."""
    model = prior.model
    return dict(full=model.full, folds=model.fold_models, excluded=model.held_out_players, params=dict(params),
                n_folds=settings.prior.n_folds, groupings=1, intercept=False, by_player=False,
                feats=list(prior.feature_names), model_feats=list(model.inputs), unseen=list(unseen),
                prior=dict(prior.values))


def rate_one_season(season: int, sums: dict, panel: pd.DataFrame, designs: DesignCache, settings, params: dict):
    """The end of step 1 (labels), step 2 (priors) and step 3 (ridge) for one season.

    Returns (the season's centred table in raw sign, {"O", "D"} its SidePriors, its unseen seasons).
    """
    unseen = unseen_seasons(season, settings)
    labels = rapm.career_labels(sums, unseen, settings.labels, settings.targets)
    priors = box_prior.season_priors(labels, panel, season, unseen, settings, params)
    season_designs = {target: designs.season(season, target) for target in settings.targets.distinct()}
    values = {side: priors[side].values for side in SIDES}
    return prior_rapm.rate_season(season_designs, values, settings, season), priors, unseen


def rate_seasons(seasons, sums: dict, panel: pd.DataFrame, designs: DesignCache, settings, params: dict,
                 keep_models: bool = False) -> tuple[list, dict]:
    """`rate_one_season` for each season: ([one centred table per season], {season: {side: saved boosters}}).

    The boosters are kept only when `keep_models`; thirty seasons of them are about a gigabyte.
    """
    started = time.time()
    tables, models = [], {}
    for season in seasons:
        table, priors, unseen = rate_one_season(season, sums, panel, designs, settings, params)
        tables.append(table)
        if keep_models:
            models[season] = {side: saved_side(priors[side], params[side], settings, unseen) for side in SIDES}
        print(f"  {season}: {len(table)} players, prior scale {table.prior_scale_off.iloc[0]:.2f} / "
              f"{table.prior_scale_def.iloc[0]:.2f}  ({time.time() - started:.0f}s)", flush=True)
    return tables, models


def booster_name(settings) -> str:
    """params/booster_params_lgb1.json -> "lgb1"."""
    return Path(settings.prior.booster_file).stem.removeprefix("booster_params_")


def models_meta(settings, name: str, argv) -> dict:
    """What the saved boosters were built with.

    schema 2 adds the settings themselves; the other keys are the ones scripts/62 wrote, with the values it wrote
    for these settings.  `crossfit` "scale" was its shipped cross-fitted scale: the cross-fitted fold priors
    equalled the full prior value for value, so dropping the cross-fit changed no number (checked 2026-10-09),
    and the earlier within-season script reads these files too.
    """
    return dict(schema=2, settings=asdict(settings), argv=list(argv),
                exclude_neighbours=settings.exclude_neighbours, features=settings.prior.feature_set,
                rows="chunks", chunk_label="career", unshrink=list(settings.labels.unshrink_columns),
                player_folds=settings.prior.n_folds, booster_params=booster_name(settings), crossfit="scale",
                pad_once=False, out=f"season_ratings_{name}_step3.parquet")


def summary(table: pd.DataFrame) -> str:
    """Rows and seasons, the spread among players with 1,000+ possessions, and the latest season's top 20."""
    regulars = table[table.poss_off >= 1000]
    scale = table.groupby("season")[["prior_scale_off", "prior_scale_def"]].first()
    latest = int(table.season.max())
    top = table[table.season == latest].sort_values("rating_total", ascending=False).head(20)
    lines = [f"{len(table)} rows, {table.season.nunique()} seasons",
             "standard deviation among players with 1,000+ possessions, points per 100:",
             f"  rating   offence {regulars.rating_off.std():.3f}   defence {regulars.rating_def.std():.3f}",
             f"  prior    offence {regulars.prior_off.std():.3f}   defence {regulars.prior_def.std():.3f}",
             f"  u        offence {regulars.u_off.std():.3f}   defence {regulars.u_def.std():.3f}"
             "   (u = rating - prior: what the season's own games added)",
             f"prior scale, mean over seasons: offence {scale.prior_scale_off.mean():.2f} "
             f"(from {scale.prior_scale_off.min():.2f} to {scale.prior_scale_off.max():.2f}), defence "
             f"{scale.prior_scale_def.mean():.2f} (from {scale.prior_scale_def.min():.2f} to "
             f"{scale.prior_scale_def.max():.2f})",
             "", f"{latest}, top 20 (points per 100 possessions, positive good on both ends)"]
    columns = ["player_name", "rating_off", "rating_def", "rating_total", "poss_off"]
    with pd.option_context("display.width", 200, "display.max_columns", 20, "display.precision", 3):
        lines.append(top[columns].to_string(index=False))
    return "\n".join(lines)


def run_rate(cfg: dict, settings, seasons, name: str, save_models: str | None = None, argv=()) -> int:
    """`openrapm rate`: rate `seasons`, write the step-3 table (and the boosters when asked), print a summary."""
    started = time.time()
    seasons = check_seasons(seasons, cfg)
    print(describe(settings, seasons), flush=True)
    designs = DesignCache(cfg)
    sums = accumulate_rapm(designs, trainable(cfg).seasons, settings)
    panel = trainable_panel(cfg)
    panel = panel[panel.poss > 0].reset_index(drop=True)
    params = booster_params(cfg, settings.prior.booster_file)
    tables, models = rate_seasons(seasons, sums, panel, designs, settings, params, keep_models=bool(save_models))
    table = prior_rapm.published_table(pd.concat(tables, ignore_index=True), player_names(cfg))
    folder = resolve(cfg, "outputs")
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"season_ratings_{name}_step3.parquet"
    table.to_parquet(path, index=False)
    print(f"\nwrote {path} ({time.time() - started:.0f}s)", flush=True)
    if save_models:
        models["meta"] = models_meta(settings, name, argv)
        pd.to_pickle(models, folder / f"prior_models_{save_models}.pkl")
        print(f"wrote {folder / f'prior_models_{save_models}.pkl'}: {len(models) - 1} seasons of boosters", flush=True)
    print(summary(table), flush=True)
    return 0
