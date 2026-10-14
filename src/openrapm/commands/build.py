"""`openrapm rate`, `shrink`, `swap` and `build`: the five steps, each table written once at the end of its command.

    openrapm rate     steps 1-3   ->  outputs/season_ratings_<name>_step3.parquet
    openrapm shrink   step 4      ->  outputs/season_ratings_<name>_step4.parquet, and its multipliers in
                                      outputs/csv/season_ratings_<name>_step4_multipliers.csv
    openrapm swap     step 5      ->  outputs/season_ratings_<name>.parquet
    openrapm build    all five in one process, writing the three tables and the multipliers above

Step 1 adds each trainable season's normal equations to one `rapm.SeasonSums` per target.  Then, season by
season: the labels leave the rated season out (`rapm.career_labels`), the box-score prior learns them and answers
for the rated season's own box score (`box_prior.season_priors`), and the ridge pulls the season's RAPM toward
that answer (`prior_rapm.rate_season`).  The step-3 table is in the published schema
(`prior_rapm.published_table`).  Step 4 multiplies each season's prior part by the multipliers its within-season
folds give (`prior_shrink`); step 5 re-splits each team's credit by its lineup swaps (`swap_adjust`).

Words used below
    trainable season   a season whose Finals are over (data/seasons.py).  Only those reach the labels and the swap
                       type model, and the season panel the prior trains on is read through
                       `data.panel.trainable_panel`
    unseen seasons     the rated season, plus the seasons either side of it in the test setting: no label and
                       no training row may come from them

Each season is rated on its own, so one run over thirty seasons writes the same bytes as three runs of ten
seasons stitched together (checked when the cleanup began, 2026-10-09).  Step 5 is not like that: a season's type
model trains on the other seasons of the same table, so a table of one season gets no swap adjustment, and a
season's final row equals the published one only when the table holds all thirty seasons.

Example: `openrapm build` writes outputs/season_ratings_product.parquet, the published table, with its step-3 and
step-4 tables beside it; `openrapm rate --seasons 2026 --name check` writes the 2026 rows alone to
outputs/season_ratings_check_step3.parquet, whose values equal the 2026 rows of a thirty-season run.
"""
from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from .. import box_prior, prior_rapm, prior_shrink, rapm, swap_adjust, swap_test
from ..data.config import booster_params, resolve
from ..data.designs import DesignCache
from ..data.panel import panel_and_trainable_seasons, trainable_panel
from ..data.roles import main_teams
from ..data.seasons import all_seasons, trainable
from ..data.stints import load_season_stints
from ..data.tables import design_row_teams, player_names
from ..data.within_folds import check_fold_files, load_folds

SIDES = ("O", "D")
SIDE_WORDS = {"offense": "offence", "defense": "defence"}     # the tables' side keys as printed, offence first


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
    era = (f"; era half-life {prior.era_halflife:g} seasons" if prior.era_halflife > 0 else "")
    labels = settings.labels
    flags = (f"; team-season flags {labels.team_offense_penalty:g} / {labels.team_defense_penalty:g}"
             if labels.team_flags else "")
    return (f"rate {seasons[0]}-{seasons[-1]} ({len(seasons)} seasons): offence target {targets.offence}, defence "
            f"target {targets.defence}; inputs {prior.feature_set}; booster {prior.booster_file}; {prior.n_folds} "
            f"player folds, assignment {prior.fold_assignment}; neighbours left out {settings.exclude_neighbours}"
            f"{era}{flags}")


def accumulate_rapm(designs: DesignCache, seasons, settings) -> dict:
    """Step 1: {target: rapm.SeasonSums} over `seasons`, one per distinct target of the two sides.

    The seasons are added in ascending order: a player's column is fixed by the first season that brings him,
    and the published ratings were built with that column order.  With the team-season flags on
    (`settings.labels.team_flags`), each season is added with its rows' teams.
    """
    started = time.time()
    flags = settings.labels.team_flags
    sums = {}
    for target in settings.targets.distinct():
        sums[target] = rapm.SeasonSums(settings.labels.min_possessions)
        for season in sorted(seasons):
            design = designs.season(season, target)
            sums[target].add(season, design, design_row_teams(design, designs.cfg) if flags else None)
        print(f"  step 1: {len(seasons)} seasons on {target}, {len(sums[target].player_ids)} players"
              f"{', team-season flags' if flags else ''} ({time.time() - started:.0f}s)", flush=True)
    return sums


def unseen_seasons(season: int, settings) -> list:
    """The rated season, and in the test setting the season either side of it.

    Example: rating 2026 gives [2026] in the product setting and [2025, 2026, 2027] in the test setting.
    """
    reach = int(settings.exclude_neighbours)
    return list(range(season - reach, season + reach + 1))


def saved_side(prior: box_prior.SidePrior, params: dict, settings, unseen) -> dict:
    """One side's boosters in the layout of the saved-booster files written before the 2026-10 cleanup (the code that
    wrote them is at the tag archive/research-2026-10), so `openrapm folds` reads a file of either age."""
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

    schema 2 adds the settings themselves; schema 3 names every input, input list and target by its plain name
    (2026-10-11; `openrapm folds` reads the older files through data/legacy_names.py).  The other keys are the
    ones the files written before the 2026-10 cleanup carry, with the values those files hold for these settings.
    `crossfit` "scale" was the shipped cross-fitted scale then: the cross-fitted fold priors equalled the full
    prior value for value, so dropping the cross-fit changed no number (checked 2026-10-09).
    """
    return dict(schema=3, settings=asdict(settings), argv=list(argv),
                exclude_neighbours=settings.exclude_neighbours, features=settings.prior.feature_set,
                rows="chunks", chunk_label="career", unshrink=list(settings.labels.unshrink_columns),
                player_folds=settings.prior.n_folds, booster_params=booster_name(settings), crossfit="scale",
                pad_once=False, out=f"season_ratings_{name}_step3.parquet")


def spread_lines(regulars: pd.DataFrame) -> list:
    """The standard deviation of each part of the rating, offence and defence, one line per part.

    After step 5 a rating is prior + u + c, so the swap adjustment c gets a line of its own.
    """
    parts = [("rating", ""), ("prior", ""),
             ("u", "   (u: how far the season's own games moved the rating from the prior part)")]
    if "c_off" in regulars.columns:
        parts.append(("c", "   (c: step 5's swap adjustment; rating = prior + u + c)"))
    return [f"  {part:<8} offence {regulars[f'{part}_off'].std():.3f}   defence {regulars[f'{part}_def'].std():.3f}"
            f"{note}" for part, note in parts]


def summary(table: pd.DataFrame) -> str:
    """Rows and seasons, the spread among players with 1,000+ possessions, and the latest season's top 20."""
    regulars = table[table.poss_off >= 1000]
    scale = table.groupby("season")[["prior_scale_off", "prior_scale_def"]].first()
    latest = int(table.season.max())
    top = table[table.season == latest].sort_values("rating_total", ascending=False).head(20)
    lines = [f"{len(table)} rows, {table.season.nunique()} seasons",
             "standard deviation among players with 1,000+ possessions, points per 100:",
             *spread_lines(regulars),
             f"prior scale, mean over seasons: offence {scale.prior_scale_off.mean():.2f} "
             f"(from {scale.prior_scale_off.min():.2f} to {scale.prior_scale_off.max():.2f}), defence "
             f"{scale.prior_scale_def.mean():.2f} (from {scale.prior_scale_def.min():.2f} to "
             f"{scale.prior_scale_def.max():.2f})",
             "", f"{latest}, top 20 (points per 100 possessions, positive good on both ends)"]
    columns = ["player_name", "rating_off", "rating_def", "rating_total", "poss_off"]
    with pd.option_context("display.width", 200, "display.max_columns", 20, "display.precision", 3):
        lines.append(top[columns].to_string(index=False))
    return "\n".join(lines)


def rate_table(cfg: dict, settings, seasons, keep_models: bool = False) -> tuple[pd.DataFrame, dict]:
    """Steps 1-3 for `seasons`: (the step-3 table in the published schema, {season: saved boosters} or {})."""
    print(describe(settings, seasons), flush=True)
    designs = DesignCache(cfg)
    sums = accumulate_rapm(designs, trainable(cfg).seasons, settings)
    panel = trainable_panel(cfg)
    panel = panel[panel.possessions > 0].reset_index(drop=True)
    params = booster_params(cfg, settings.prior.booster_file)
    tables, models = rate_seasons(seasons, sums, panel, designs, settings, params, keep_models=keep_models)
    return prior_rapm.published_table(pd.concat(tables, ignore_index=True), player_names(cfg)), models


def write_table(table: pd.DataFrame, path: Path, started: float) -> None:
    """Write one table and say where."""
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(path, index=False)
    print(f"wrote {path} ({len(table)} rows, {time.time() - started:.0f}s)", flush=True)


def write_models(models: dict, settings, name: str, save_models: str, argv, cfg: dict) -> None:
    """outputs/prior_models_<save_models>.pkl: every rated season's boosters and what built them."""
    models["meta"] = models_meta(settings, name, argv)
    path = resolve(cfg, "outputs") / f"prior_models_{save_models}.pkl"
    pd.to_pickle(models, path)
    print(f"wrote {path}: {len(models) - 1} seasons of boosters", flush=True)


def run_rate(cfg: dict, settings, seasons, name: str, save_models: str | None = None, argv=()) -> int:
    """`openrapm rate`: rate `seasons`, write the step-3 table (and the boosters when asked), print a summary."""
    started = time.time()
    seasons = check_seasons(seasons, cfg)
    table, models = rate_table(cfg, settings, seasons, keep_models=bool(save_models))
    print()
    write_table(table, resolve(cfg, "outputs") / f"season_ratings_{name}_step3.parquet", started)
    if save_models:
        write_models(models, settings, name, save_models, argv, cfg)
    print(summary(table), flush=True)
    return 0


# ------------------------------------------------------------------------------------------- step 4
def require_folds(cfg: dict, shrink) -> None:
    """Stop at once, naming the folder and the two commands that make it, when step 4's within-season folds are
    missing or incomplete.

    `openrapm build` and `openrapm shrink` call this first: on fresh data `build` would otherwise run steps 1-3,
    about fifteen minutes, before step 4 found no folds.  Only the folds' small json files are read.
    """
    for tag in fold_tags(shrink):
        directory = resolve(cfg, "within") / tag
        try:
            check_fold_files(directory, shrink.fold_seasons)
        except (OSError, ValueError) as problem:
            raise SystemExit(folds_missing_message(directory, shrink, problem)) from None


def fold_tags(shrink) -> list:
    """The fold folders step 4 reads: settings.shrink.folds_tag, split at commas."""
    return [tag for tag in shrink.folds_tag.split(",") if tag]


def folds_missing_message(directory: Path, shrink, problem: Exception) -> str:
    """What `require_folds` prints: the problem, the folds folder, and the two commands that make the folds.

    Example, on a fresh clone with the shipped settings: "step 4 needs the within-season folds in
    .../outputs/within/weightfix_within for 2017-2026, so nothing was run: no folds in ..."
    """
    span = f"{shrink.fold_seasons[0]}-{shrink.fold_seasons[1]}"
    return (f"step 4 needs the within-season folds in {directory} for {span}, so nothing was run: {problem}.\n"
            f"Make the folds with these two commands (NAME is any name), then run this again:\n"
            f"    openrapm rate --setting test --seasons {span} --save-models NAME\n"
            f"    openrapm folds --models NAME --base season_ratings_test_step3 --seasons {span} "
            f"--tag {directory.name}\n"
            f"or name a folder that holds them with --folds TAG (a folder under {directory.parent}).")


def shrink_seasons(cfg: dict, table: pd.DataFrame, settings) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Step 4 on a step-3 table: (the shrunk table, the multipliers of every season).

    The folds are paths.within/<settings.shrink.folds_tag>, their seasons pinned to settings.shrink.fold_seasons.
    The multipliers table also names the fold folder and the first and last fold season read, so a step-4 table
    can be traced to the folds that shrank it.
    """
    shrink = settings.shrink
    folds = [fold for tag in fold_tags(shrink) for fold in load_folds(resolve(cfg, "within") / tag,
                                                                       shrink.fold_seasons)]
    fold_seasons = sorted({fold.season for fold in folds})
    multipliers = prior_shrink.multipliers_by_season(table.season.unique(), folds, shrink.rule,
                                                     shrink.nearest).assign(
        fold_tag=shrink.folds_tag, fold_first=fold_seasons[0], fold_last=fold_seasons[-1])
    nearest = f", the {shrink.nearest} nearest fold seasons" if shrink.nearest else ""
    print(f"  step 4: {len(folds)} folds of {shrink.folds_tag} ({shrink.fold_seasons[0]}-{shrink.fold_seasons[1]}), "
          f"{shrink.rule} rule{nearest}; the prior part multiplied by {multipliers.multiplier_off.min():.3f} to "
          f"{multipliers.multiplier_off.max():.3f} on offence, {multipliers.multiplier_def.min():.3f} to "
          f"{multipliers.multiplier_def.max():.3f} on defence", flush=True)
    return prior_shrink.shrink(table, multipliers), multipliers


def write_multipliers(multipliers: pd.DataFrame, cfg: dict, name: str) -> None:
    """outputs/csv/season_ratings_<name>_step4_multipliers.csv: each season's two multipliers and their folds."""
    path = resolve(cfg, "outputs") / "csv" / f"season_ratings_{name}_step4_multipliers.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    multipliers.to_csv(path, index=False)
    print(f"wrote {path}", flush=True)


def run_shrink(cfg: dict, settings, base: str, name: str) -> int:
    """`openrapm shrink`: step 4 on outputs/<base>.parquet, written as outputs/season_ratings_<name>_step4.parquet,
    with its multipliers beside it (`write_multipliers`)."""
    started = time.time()
    require_folds(cfg, settings.shrink)
    outputs = resolve(cfg, "outputs")
    shrunk, multipliers = shrink_seasons(cfg, pd.read_parquet(outputs / f"{base}.parquet"), settings)
    write_table(shrunk, outputs / f"season_ratings_{name}_step4.parquet", started)
    write_multipliers(multipliers, cfg, name)
    return 0


# ------------------------------------------------------------------------------------------- step 5
def collect_swaps(cfg: dict, raw: pd.DataFrame, features: pd.DataFrame, fold_of: pd.Series, n_folds: int) -> dict:
    """{season: swap_adjust.SeasonSwaps} for every season of `raw` (the table in raw sign), ascending."""
    started = time.time()
    margin_clip = float(cfg.get("margin_clip", 25))
    swaps = {}
    for season in sorted(int(season) for season in raw.season.unique()):
        stints = load_season_stints(season, cfg, swap_test.STINT_COLUMNS)
        lineups = swap_test.prepare(swap_test.season_rows(stints, margin_clip=margin_clip), season)
        swaps[season] = swap_adjust.season_swaps(lineups, raw[raw.season == season],
                                                 swap_adjust.season_features(features, season), fold_of, n_folds)
    pairs = sum(len(kept.pairs["offense"]) for kept in swaps.values())
    print(f"  step 5: {pairs:,} offensive lineup swaps over {len(swaps)} seasons ({time.time() - started:.0f}s)",
          flush=True)
    return swaps


def swap_seasons(cfg: dict, table: pd.DataFrame, settings) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Step 5 on a step-4 table: (the adjusted table, the spread factors, the type model fitted on every season)."""
    panel, trainable_seasons = panel_and_trainable_seasons(cfg)
    raw = swap_test.ratings_in_raw_sign(table)
    features = swap_adjust.type_features(panel, table)
    fold_of = swap_adjust.player_folds(raw.player_id, settings.swap.n_folds)
    swaps = collect_swaps(cfg, raw, features, fold_of, settings.swap.n_folds)
    predictions = swap_adjust.type_predictions(features, swaps, fold_of, trainable_seasons, settings.swap)
    main_team = main_teams(cfg).set_index(["player_id", "season"]).team_id
    adjusted, factors = swap_adjust.adjust(table, raw, swaps, predictions, main_team, settings.swap)
    types = swap_adjust.type_model_table(swaps, trainable_seasons, settings.swap.type_model_penalty)
    spread = factors.groupby("side").factor.agg(["mean", "min", "max"]).reindex(list(SIDE_WORDS))
    print("  step 5: the spread inside teams held, factor per season " + "; ".join(
        f"{SIDE_WORDS[side]} {row['mean']:.3f} ({row['min']:.3f} to {row['max']:.3f})"
        for side, row in spread.iterrows()), flush=True)
    return adjusted, factors, types


def swap_movement(table: pd.DataFrame) -> str:
    """How far step 5 moved each player's total in the latest season, points per 100."""
    latest = table[table.season == table.season.max()]
    move = (latest.c_off + latest.c_def).abs()
    return (f"{int(latest.season.iloc[0])}: step 5 moved a player's total a median {move.median():.3f} per 100, "
            f"the 90th percentile {move.quantile(0.9):.3f}, the largest {move.max():.3f}")


def write_types(types: pd.DataFrame, cfg: dict, name: str) -> None:
    """outputs/<name>_types.csv: the type model fitted on every trainable season, for reading."""
    path = resolve(cfg, "outputs") / f"{name}_types.csv"
    types.to_csv(path, index=False)
    print(f"wrote {path}", flush=True)


def run_swap(cfg: dict, settings, base: str, name: str, with_types: bool = False) -> int:
    """`openrapm swap`: step 5 on outputs/<base>.parquet, written as outputs/season_ratings_<name>.parquet."""
    started = time.time()
    outputs = resolve(cfg, "outputs")
    source, path = outputs / f"{base}.parquet", outputs / f"season_ratings_{name}.parquet"
    if source.resolve() == path.resolve():
        raise SystemExit(f"--name {name} would overwrite the table it reads; give another --name")
    adjusted, _, types = swap_seasons(cfg, pd.read_parquet(source), settings)
    write_table(adjusted, path, started)
    if with_types:
        write_types(types, cfg, name)
    print(swap_movement(adjusted), flush=True)
    return 0


# ------------------------------------------------------------------------------------------- all five
def run_build(cfg: dict, settings, seasons, name: str, save_models: str | None = None, argv=(),
              with_types: bool = False) -> int:
    """`openrapm build`: steps 1-5 in one process, the three tables and step 4's multipliers written at the end.
    Step 4's folds are checked first (`require_folds`), so missing folds stop the run in seconds."""
    started = time.time()
    seasons = check_seasons(seasons, cfg)
    require_folds(cfg, settings.shrink)
    step3, models = rate_table(cfg, settings, seasons, keep_models=bool(save_models))
    step4, multipliers = shrink_seasons(cfg, step3, settings)
    final, _, types = swap_seasons(cfg, step4, settings)
    outputs = resolve(cfg, "outputs")
    print()
    for table, suffix in ((step3, "_step3"), (step4, "_step4"), (final, "")):
        write_table(table, outputs / f"season_ratings_{name}{suffix}.parquet", started)
    write_multipliers(multipliers, cfg, name)
    if save_models:
        write_models(models, settings, name, save_models, argv, cfg)
    if with_types:
        write_types(types, cfg, name)
    print(summary(final), flush=True)
    print(swap_movement(final), flush=True)
    return 0
