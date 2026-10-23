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
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from .. import box_prior, prior_rapm, prior_shrink, rapm, second_prior, swap_adjust, swap_test
from ..data.config import booster_params, resolve
from ..data.designs import DesignCache
from ..data.panel import panel_and_trainable_seasons, trainable_panel
from ..data.roles import main_teams
from ..data.seasons import all_seasons, trainable
from ..data.second_prior_files import line_setup, load_grams
from ..data.stints import load_season_stints
from ..data.tables import design_row_teams, player_names
from ..data.within_folds import check_fold_files, fold_games, load_folds
from ..settings import describe_factors
from . import parallel

SIDES = ("O", "D")
NEAREST_UNSHRINK = ("offense", "defense")     # experiment 83's labels from the seasons next to a row, both un-shrunk
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
    second = (f"; step 3 reads the second prior, its line from outputs/second_prior/{prior.second_prior} on the "
              f"inputs {prior.second_prior_inputs}, team totals {prior.second_prior_team_totals}"
              if prior.second_prior else "")
    return (f"rate {seasons[0]}-{seasons[-1]} ({len(seasons)} seasons): offence target {targets.offence}, defence "
            f"target {targets.defence}; inputs {prior.feature_set}; booster {prior.booster_file}; {prior.n_folds} "
            f"player folds, assignment {prior.fold_assignment}; neighbours left out {settings.exclude_neighbours}"
            f"{era}{flags}{second}")


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


def saved_side(prior: box_prior.SidePrior, params: dict, settings, unseen, second=None, side: str = "O") -> dict:
    """One side's boosters in the layout of the saved-booster files written before the 2026-10 cleanup (the code that
    wrote them is at the tag archive/research-2026-10), so `openrapm folds` reads a file of either age.  With the second
    prior on (`second`, a SeasonSecondPrior), also its values (`second_prior`) and the season's line (`line`, rows as
    second_prior.season_coefficients gives them), which `openrapm folds` checks its whole-season rebuild against."""
    model = prior.model
    members = model.members if isinstance(model, box_prior.BaggedPrior) else [model]
    out = dict(full=members[0].full, folds=members[0].fold_models, excluded=members[0].held_out_players,
               params=dict(params), n_folds=settings.prior.n_folds, groupings=1, intercept=False, by_player=False,
               feats=list(prior.feature_names), model_feats=list(model.inputs), unseen=list(unseen),
               prior=dict(prior.values))
    if isinstance(model, box_prior.BaggedPrior):
        # every bag, the first also in the old layout's keys; `openrapm folds` reads the bags when they are there
        out["bags"] = [dict(full=member.full, folds=member.fold_models, excluded=member.held_out_players)
                       for member in members]
    if second is not None:
        out["second_prior"] = dict(second.values[side])
        out["line"] = np.asarray(second.coefficients[side]).tolist()
    return out


def outside_row_labels(sums: dict, panel: pd.DataFrame, unseen, settings) -> dict:
    """rapm.outside_labels for every set of seasons a chunk row of either side averages (experiment 79)."""
    training = panel[~panel.season.isin(unseen)]
    season_sets = set().union(*(box_prior.chunk_season_sets(training[training.side == side], settings.prior.chunk_sizes)
                                for side in SIDES))
    return rapm.outside_labels(sums, unseen, sorted(season_sets, key=sorted), settings.labels, settings.targets)


@dataclass
class SecondPriorLines:
    """What every rated season's line is fitted from: {season: {side: PairGrams}} (data/second_prior_files.py) and
    the seasons a fit may train on."""
    grams: dict
    trainable: set
    type_scale: dict | None = None          # with the line on step 5's numbers too (second_prior.type_scale)


@dataclass
class SeasonSecondPrior:
    """One rated season's second prior: {"O", "D"} each of its values, the line it came from and that line's inputs."""
    values: dict
    coefficients: dict
    inputs: dict


def rated_second_prior(cfg: dict, season: int, panel: pd.DataFrame, priors: dict, lines: SecondPriorLines,
                       settings) -> SeasonSecondPrior:
    """The rated season's second prior (second_prior.py): its context numbers from the whole season's stint rows, every
    prior in them from the season's own boosters that never saw the player, and the line of the seasons that may train
    it (more than `exclude_neighbours` away)."""
    prior = settings.prior
    coefficients = second_prior.season_coefficients(lines.grams, season, lines.trainable, settings.exclude_neighbours,
                                                    prior.second_prior_folds, prior.second_prior_penalty,
                                                    prior.second_prior_nearest)
    if coefficients is None:
        raise SystemExit(f"{season}: no season of outputs/second_prior/{prior.second_prior} may train its line")
    answers = {side: second_prior.booster_answers(
        priors[side].model,
        box_prior.rated_rows(panel[(panel.side == side) & (panel.season == season)], priors[side].feature_names))
        for side in SIDES}
    types = (None if lines.type_scale is None else
             second_prior.standardised_types(panel[(panel.side == "O") & (panel.season == season)], lines.type_scale))
    values, inputs = second_prior.season_second_priors(season_lineup_rows(cfg, season), answers,
                                                       {side: priors[side].values for side in SIDES}, coefficients,
                                                       prior.second_prior_folds, prior.second_prior_team_totals,
                                                       types, prior.second_prior_inputs)
    return SeasonSecondPrior(values=values, coefficients=coefficients, inputs=inputs)


def line_words(coefficients: dict, names) -> str:
    """The line fitted on every swap, both sides, one weight per input (`names`), for the log."""
    return "; ".join(f"{side} " + ", ".join(f"{name} {weight:+.3f}" for name, weight
                                             in zip(names, np.asarray(coefficients[side])[-1]))
                     for side in SIDES)


def row_label_tables(sums: dict, panel: pd.DataFrame, unseen, settings, cache: dict | None = None) -> dict | None:
    """The tables the prior's chunk rows are labelled from, by `settings.prior.row_labels`: None for "career" (every
    row carries the career label), the seasons outside each row for "outside" (experiment 79), the seasons right next
    to it for "nearest" (experiment 83, `cache` shared across the run's rated seasons)."""
    if settings.prior.row_labels == "outside":
        return outside_row_labels(sums, panel, unseen, settings)
    if settings.prior.row_labels == "nearest":
        training = panel[~panel.season.isin(unseen)]
        season_sets = set().union(*(box_prior.chunk_season_sets(training[training.side == side],
                                                                settings.prior.chunk_sizes) for side in SIDES))
        # both sides un-shrunk: a label over one or two seasons at step 1's penalty keeps 5-13% of a player's impact,
        # by how much he played next to the row, which the booster cannot see (notes/DECISIONS.md, experiment 83)
        labels = replace(settings.labels, unshrink_columns=NEAREST_UNSHRINK)
        return rapm.nearest_labels(sums, unseen, season_sets, labels, settings.targets, cache)
    return None


def rate_one_season(season: int, sums: dict, panel: pd.DataFrame, designs: DesignCache, settings, params: dict,
                    lines: SecondPriorLines | None = None, label_cache: dict | None = None):
    """The end of step 1 (labels), step 2 (priors) and step 3 (ridge) for one season.

    Returns (the season's centred table in raw sign, {"O", "D"} its SidePriors, its unseen seasons, its
    SeasonSecondPrior or None).  With `lines` (the second prior on, settings.PriorSettings.second_prior) step 3 reads
    the second prior instead of the box-score prior.
    """
    unseen = unseen_seasons(season, settings)
    labels = rapm.career_labels(sums, unseen, settings.labels, settings.targets)
    outside = row_label_tables(sums, panel, unseen, settings, label_cache)
    priors = box_prior.season_priors(labels, panel, season, unseen, settings, params, outside)
    season_designs = {target: designs.season(season, target) for target in settings.targets.distinct()}
    values = {side: priors[side].values for side in SIDES}
    second = None
    if lines is not None:
        second = rated_second_prior(designs.cfg, season, panel, priors, lines, settings)
        values = second.values
    return prior_rapm.rate_season(season_designs, values, settings, season), priors, unseen, second


def rate_seasons(seasons, sums: dict, panel: pd.DataFrame, designs: DesignCache, settings, params: dict,
                 keep_models: bool = False, lines: SecondPriorLines | None = None) -> tuple[list, dict]:
    """`rate_one_season` for each season: ([one centred table per season], {season: {side: saved boosters}}).

    The boosters are kept only when `keep_models`; thirty seasons of them are about a gigabyte.
    """
    started = time.time()
    tables, models, label_cache = [], {}, {}
    for season in seasons:
        table, priors, unseen, second = rate_one_season(season, sums, panel, designs, settings, params, lines,
                                                        label_cache)
        tables.append(table)
        if keep_models:
            models[season] = {side: saved_side(priors[side], params[side], settings, unseen, second, side)
                              for side in SIDES}
        print(f"  {season}: {len(table)} players, prior scale {table.prior_scale_off.iloc[0]:.2f} / "
              f"{table.prior_scale_def.iloc[0]:.2f}  ({time.time() - started:.0f}s)", flush=True)
        if second is not None:
            print(f"    the line: {line_words(second.coefficients, second.inputs['O'].columns)}", flush=True)
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
                rows="chunks", chunk_label=settings.prior.row_labels, unshrink=list(settings.labels.unshrink_columns),
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
    lines = None
    if settings.prior.second_prior:
        record = line_setup(cfg, settings.prior.second_prior,
                            second_prior.INPUT_SETS[settings.prior.second_prior_inputs])
        lines = SecondPriorLines(load_grams(cfg, settings.prior.second_prior), set(trainable(cfg).seasons),
                                 record.get("type_scale"))
    tables, models = rate_seasons(seasons, sums, panel, designs, settings, params, keep_models=keep_models,
                                  lines=lines)
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


def rate_in_parts(cfg: dict, seasons, name: str, save_models: str | None, argv, jobs: int) -> tuple[pd.DataFrame, dict]:
    """`rate_table` over `parallel.season_chunks`, one `openrapm rate` child per chunk: (the step-3 table, the
    boosters by season or {}).  Each child writes season_ratings_<name>__part<k>_step3.parquet and, when boosters are
    kept, prior_models_<save_models>__part<k>.pkl; both are read back here, in season order, and removed."""
    outputs = resolve(cfg, "outputs")
    chunks = parallel.season_chunks(seasons, jobs)

    def names(k):
        kept = ["--save-models", parallel.part_name(save_models, k)] if save_models else []
        return ["--name", parallel.part_name(name, k), *kept]

    parallel.run_children(list(argv), chunks, names, outputs / "parallel" / name)
    tables, models = [], {}
    for k in range(len(chunks)):
        path = outputs / f"season_ratings_{parallel.part_name(name, k)}_step3.parquet"
        tables.append(pd.read_parquet(path))
        path.unlink()
        if save_models:
            part = outputs / f"prior_models_{parallel.part_name(save_models, k)}.pkl"
            saved = pd.read_pickle(part)
            saved.pop("meta", None)
            models.update(saved)
            part.unlink()
    table = pd.concat(tables, ignore_index=True).sort_values("season", kind="mergesort").reset_index(drop=True)
    return table, ({season: models[season] for season in seasons} if save_models else {})


def run_rate(cfg: dict, settings, seasons, name: str, save_models: str | None = None, argv=(), jobs: int = 1) -> int:
    """`openrapm rate`: rate `seasons`, write the step-3 table (and the boosters when asked), print a summary.  With
    `jobs` above 1, the seasons are rated in that many processes at once, as many as free memory holds
    (`parallel.affordable_jobs`), with the same numbers."""
    started = time.time()
    seasons = check_seasons(seasons, cfg)
    fits = parallel.affordable_jobs(jobs) if jobs > 1 and len(seasons) > 1 else 1
    if fits < jobs:
        print(f"{jobs} processes asked for, free memory holds {fits}", flush=True)
    if fits > 1:
        table, models = rate_in_parts(cfg, seasons, name, save_models, argv, fits)
    else:
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
    for tag in fold_tags(shrink) + ([shrink.extrapolate_folds_tag] if shrink.extrapolate_folds_tag else []):
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


def step4_folds(cfg: dict, shrink) -> tuple[list, dict]:
    """(every fold step 4 reads, {stem: its folder}): paths.within/<settings.shrink.folds_tag>, their seasons pinned to
    settings.shrink.fold_seasons.  Step 5 reads the same folds (`held_back_type_grams`)."""
    folds, where = [], {}
    for tag in fold_tags(shrink):
        directory = resolve(cfg, "within") / tag
        for fold in load_folds(directory, shrink.fold_seasons):
            folds.append(fold)
            where[fold.stem] = directory
    return folds, where


def extrapolation_folds(cfg: dict, shrink) -> list:
    """The second fold set step 4 reads its multipliers to a full season with, paths.within/<extrapolate_folds_tag>,
    its seasons pinned as step 4's; [] when the tag is empty."""
    if not shrink.extrapolate_folds_tag:
        return []
    return load_folds(resolve(cfg, "within") / shrink.extrapolate_folds_tag, shrink.fold_seasons)


def step4_multipliers(seasons, folds: list, shrink, second_folds: list | None = None) -> pd.DataFrame:
    """`prior_shrink.multipliers_by_season` for `seasons`, read to a full season with `second_folds` when given
    (`prior_shrink.full_season_multipliers`), the defence factor applied, with the fold folders and the first and
    last fold season read, so a step-4 table can be traced to the folds that shrank it."""
    fold_seasons = sorted({fold.season for fold in folds})
    if second_folds:
        multipliers = prior_shrink.full_season_multipliers(seasons, folds, second_folds, shrink.rule, shrink.nearest,
                                                           shrink.free_games_part)
        tag = f"{shrink.folds_tag}+{shrink.extrapolate_folds_tag}"
    else:
        multipliers = prior_shrink.multipliers_by_season(seasons, folds, shrink.rule, shrink.nearest,
                                                         shrink.free_games_part)
        tag = shrink.folds_tag
    multipliers = multipliers.assign(fold_tag=tag, fold_first=fold_seasons[0], fold_last=fold_seasons[-1])
    factor = multipliers.season.map(shrink.defence_factor_for).to_numpy(float)
    multipliers["multiplier_def"] *= factor
    if shrink.free_games_part:
        multipliers["games_multiplier_def"] *= factor
    return multipliers


def shrink_seasons(cfg: dict, table: pd.DataFrame, settings,
                   folds: list | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Step 4 on a step-3 table: (the shrunk table, the multipliers of every season).  `folds` defaults to
    `step4_folds`'s."""
    shrink = settings.shrink
    folds = step4_folds(cfg, shrink)[0] if folds is None else folds
    second = extrapolation_folds(cfg, shrink)
    multipliers = step4_multipliers(table.season.unique(), folds, shrink, second)
    if second:
        print(f"  step 4: each multiplier read to a full season from the folds holding back "
              f"{prior_shrink.held_back_share(folds):.0%} and {prior_shrink.held_back_share(second):.0%} of the "
              f"games ({shrink.extrapolate_folds_tag})", flush=True)
    if shrink.defence_factors != ((0, 1.0),):
        print(f"  step 4: both defensive multipliers times {describe_factors(shrink.defence_factors)}", flush=True)
    nearest = f", the {shrink.nearest} nearest fold seasons" if shrink.nearest else ""
    print(f"  step 4: {len(folds)} folds of {shrink.folds_tag} ({shrink.fold_seasons[0]}-{shrink.fold_seasons[1]}), "
          f"{shrink.rule} rule{nearest}; the prior part multiplied by {multipliers.multiplier_off.min():.3f} to "
          f"{multipliers.multiplier_off.max():.3f} on offence, {multipliers.multiplier_def.min():.3f} to "
          f"{multipliers.multiplier_def.max():.3f} on defence", flush=True)
    if shrink.free_games_part:
        print(f"  step 4: the games' part multiplied by {multipliers.games_multiplier_off.min():.3f} to "
              f"{multipliers.games_multiplier_off.max():.3f} on offence, {multipliers.games_multiplier_def.min():.3f} "
              f"to {multipliers.games_multiplier_def.max():.3f} on defence", flush=True)
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
def season_lineup_rows(cfg: dict, season: int) -> pd.DataFrame:
    """One season's stints as swap_test.season_rows, the rows every lineup of step 5 is built from."""
    stints = load_season_stints(season, cfg, swap_test.STINT_COLUMNS)
    return swap_test.season_rows(stints, margin_clip=float(cfg.get("margin_clip", 25)))


def collect_swaps(cfg: dict, raw: pd.DataFrame, features: pd.DataFrame, fold_of: pd.Series, n_folds: int) -> dict:
    """{season: swap_adjust.SeasonSwaps} for every season of `raw` (the table in raw sign), ascending: each season's
    whole-season swaps and each player's possessions for each team, which the give-back reads."""
    started = time.time()
    swaps = {}
    for season in sorted(int(season) for season in raw.season.unique()):
        lineups = swap_test.prepare(season_lineup_rows(cfg, season), season)
        swaps[season] = swap_adjust.season_swaps(lineups, raw[raw.season == season],
                                                 swap_adjust.season_features(features, season), fold_of, n_folds)
    pairs = sum(len(kept.pairs["offense"]) for kept in swaps.values())
    print(f"  step 5: {pairs:,} offensive lineup swaps over {len(swaps)} seasons ({time.time() - started:.0f}s)",
          flush=True)
    return swaps


def season_held_back_grams(cfg: dict, season: int, season_folds: list, where: dict, factors: tuple, scale: tuple,
                           n_folds: int) -> dict:
    """{side: PairGrams}: one season's share of the type model, summed over its folds' held-back games."""
    rows = season_lineup_rows(cfg, season)
    game = rows.game_id.astype(str)
    parts = []
    for fold in season_folds:
        _, held_ids = fold_games(where[fold.stem], fold)
        held = swap_test.prepare(rows[game.isin(held_ids).to_numpy()].reset_index(drop=True), season)
        table = prior_shrink.shrink_fold(fold.rows, season, factors)
        features = swap_adjust.fold_type_features(fold.rows["O"], table, *scale)
        parts.append(swap_adjust.held_back_grams(held, table, features, n_folds))
    return {side: swap_adjust.add_grams([part[side] for part in parts]) for side in swap_adjust.SIDES}


def held_back_type_grams(cfg: dict, table: pd.DataFrame, panel: pd.DataFrame, folds: list, where: dict,
                         multipliers: pd.DataFrame, settings) -> dict:
    """{season: {side: PairGrams}}: the type model's training equations, each season's from the held-back games of
    its within-season folds (swap_adjust, "Since experiment 80"), each fold scored at its season's step-4
    multipliers.  The type features are read on `table`'s scale."""
    started = time.time()
    scale = swap_adjust.feature_scale(panel, table)
    by_season: dict = {}
    for fold in folds:
        by_season.setdefault(fold.season, []).append(fold)
    grams = {season: season_held_back_grams(cfg, season, by_season[season], where,
                                            prior_shrink.season_factors(multipliers, season), scale,
                                            settings.swap.n_folds)
             for season in sorted(by_season)}
    print(f"  step 5: the type model's swaps from the held-back games of {len(folds)} folds over {len(grams)} "
          f"seasons ({time.time() - started:.0f}s)", flush=True)
    return grams


def no_grams(n_features: int, n_folds: int) -> dict:
    """{side: zero PairGrams}: a season without folds trains nothing."""
    zero = swap_adjust.PairGrams(G=np.zeros((n_features, n_features)), b=np.zeros(n_features),
                                 G_fold=np.zeros((n_folds, n_features, n_features)),
                                 b_fold=np.zeros((n_folds, n_features)), n=0)
    return {side: zero for side in swap_adjust.SIDES}


def swap_seasons(cfg: dict, table: pd.DataFrame, settings, folds: list | None = None, where: dict | None = None,
                 multipliers: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Step 5 on a step-4 table: (the adjusted table, the spread factors, the type model fitted on every season).

    The type model trains on the within-season folds' held-back games (`held_back_type_grams`); `folds`, `where` and
    `multipliers` default to step 4's, read again from the folds when not handed in.

    With step 5 off, an applied share of 0 as shipped since experiment 82, nothing is fitted: the table comes back as
    step 4 left it with step 5's two columns at 0, which is what the full step gives at that share, and a build of
    one season, whose step 4 holds no other season's multipliers, needs none."""
    if settings.swap.applied_share == 0:
        print("  step 5: off, applied share 0; step 4's ratings kept as they are", flush=True)
        return (table.assign(c_off=0.0, c_def=0.0), pd.DataFrame(columns=["season", "side", "factor"]),
                pd.DataFrame())
    panel, trainable_seasons = panel_and_trainable_seasons(cfg)
    if folds is None:
        folds, where = step4_folds(cfg, settings.shrink)
        multipliers = step4_multipliers(table.season.unique(), folds, settings.shrink,
                                        extrapolation_folds(cfg, settings.shrink))
    raw = swap_test.ratings_in_raw_sign(table)
    features = swap_adjust.type_features(panel, table)
    fold_of = swap_adjust.player_folds(raw.player_id, settings.swap.n_folds)
    swaps = collect_swaps(cfg, raw, features, fold_of, settings.swap.n_folds)
    grams = held_back_type_grams(cfg, table, panel, folds, where, multipliers, settings)
    empty = no_grams(features.shape[1], settings.swap.n_folds)
    swaps = {season: replace(kept, grams=grams.get(season, empty)) for season, kept in swaps.items()}
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
    require_folds(cfg, settings.shrink)
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
    folds, where = step4_folds(cfg, settings.shrink)
    step4, multipliers = shrink_seasons(cfg, step3, settings, folds)
    final, _, types = swap_seasons(cfg, step4, settings, folds, where, multipliers)
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
