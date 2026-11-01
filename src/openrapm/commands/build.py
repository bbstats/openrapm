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
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd

from .. import box_prior, game_prior, game_rates, prior_rapm, prior_shrink, rapm, second_prior, swap_adjust, swap_test
from ..data.config import booster_params, resolve
from ..data.designs import DesignCache
from ..data.game_rate_files import load_network
from ..data.panel import in_progress_panel, panel_and_trainable_seasons, trainable_panel, trainable_panel_so_far
from ..data.player_games import trainable_player_games
from ..data.roles import main_teams
from ..data.seasons import all_seasons, share_played, trainable, trainable_through
from ..data.second_prior_files import line_setup, load_grams
from ..data.stints import load_season_stints
from ..data.tables import design_row_teams, player_names
from ..data.within_folds import check_fold_files, fold_games, held_back_stints, load_folds
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
    bags = "".join([f"; {prior.bags} bags predicted out of bag" if prior.prediction == "out_of_bag" else "",
                    "; each bag its own split" if prior.split_per_bag else "",
                    f"; bag draw {prior.bag_draw}" if prior.bag_draw else ""])
    return (f"rate {seasons[0]}-{seasons[-1]} ({len(seasons)} seasons): offence target {targets.offence}, defence "
            f"target {targets.defence}; inputs {prior.feature_set}; booster {prior.booster_file}; {prior.n_folds} "
            f"player folds, assignment {prior.fold_assignment}; neighbours left out {settings.exclude_neighbours}"
            f"{bags}{era}{flags}{second}")


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


def booster_layout(model) -> dict:
    """A box-score prior's boosters under the saved-booster files' keys: `full`, `folds` and `excluded`, the first
    bag's in the layout of the files written before the 2026-10 cleanup; every bag under `bags`, which `openrapm folds`
    reads when they are there; an out-of-bag prior's boosters under `out_of_bag`, each with the players its resample
    drew, its first booster as `full` and no folds (within_season.saved_booster reads every kind back)."""
    if isinstance(model, box_prior.OutOfBagPrior):
        return dict(full=model.models[0], folds=[], excluded=[],
                    out_of_bag=[dict(model=booster, drawn=drawn)
                                for booster, drawn in zip(model.models, model.drawn_players)])
    members = model.members if isinstance(model, box_prior.BaggedPrior) else [model]
    out = dict(full=members[0].full, folds=members[0].fold_models, excluded=members[0].held_out_players)
    if isinstance(model, box_prior.BaggedPrior):
        out["bags"] = [dict(full=member.full, folds=member.fold_models, excluded=member.held_out_players)
                       for member in members]
    return out


def saved_side(prior: box_prior.SidePrior, params: dict, settings, unseen, second=None, side: str = "O") -> dict:
    """One side's boosters in the layout of the saved-booster files written before the 2026-10 cleanup (the code that
    wrote them is at the tag archive/research-2026-10), so `openrapm folds` reads a file of either age.  With the second
    prior on (`second`, a SeasonSecondPrior), also its values (`second_prior`) and the season's line (`line`, rows as
    second_prior.season_coefficients gives them), which `openrapm folds` checks its whole-season rebuild against."""
    model = prior.model
    layout = booster_layout(model)
    extra = {key: layout.pop(key) for key in ("bags", "out_of_bag") if key in layout}
    out = dict(**layout, params=dict(params), n_folds=settings.prior.n_folds, groupings=1, intercept=False,
               by_player=False, feats=list(prior.feature_names), model_feats=list(model.inputs), unseen=list(unseen),
               prior=dict(prior.values))
    out.update(extra)
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
                       settings, share: float = 1.0, box: dict | None = None,
                       types_panel: pd.DataFrame | None = None) -> SeasonSecondPrior:
    """The rated season's second prior (second_prior.py): its context numbers from the whole season's stint rows, every
    prior in them from the season's own boosters that never saw the player, and the line of the seasons that may train
    it (more than `exclude_neighbours` away).  `panel` holds the rated season's rows; `share` is the share of its
    regular season played (box_prior.rated_rows).  With the box-score prior game by game, `box` is each player's
    season average ({"O", "D"}: {player_id: value}) and `types_panel` the season panel the line's step-5 numbers are
    read from, as without it."""
    prior = settings.prior
    coefficients = second_prior.season_coefficients(lines.grams, season, lines.trainable, settings.exclude_neighbours,
                                                    prior.second_prior_folds, prior.second_prior_penalty,
                                                    prior.second_prior_nearest)
    if coefficients is None:
        raise SystemExit(f"{season}: no season of outputs/second_prior/{prior.second_prior} may train its line")
    answers = {side: second_prior.booster_answers(
        priors[side].model,
        box_prior.rated_rows(panel[(panel.side == side) & (panel.season == season)], priors[side].feature_names, share,
                             box_prior.least_share(settings)))
        for side in SIDES}
    typed = panel if types_panel is None else types_panel
    types = (None if lines.type_scale is None else
             second_prior.standardised_types(typed[(typed.side == "O") & (typed.season == season)], lines.type_scale))
    box = {side: priors[side].values for side in SIDES} if box is None else box
    values, inputs = second_prior.season_second_priors(season_lineup_rows(cfg, season), answers, box, coefficients,
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


@dataclass
class SeasonsInProgress:
    """The seasons a build rates that no fit may train on (data/seasons.py): their season panel rows, which the
    trainable panel lacks, {season: the share of its regular season played} (data.seasons.share_played), and
    {season: (s_O, s_D)} the held prior scales of a season still short of settings.RidgeSettings.held_scale_through
    (prior_rapm.held_prior_scale, experiment 109), which step 3 blends with the season's own fitted ones by the share
    played (prior_rapm.blend_weight, the owner, 2026-10-31)."""
    rows: pd.DataFrame
    shares: dict
    held_scales: dict = field(default_factory=dict)
    deferred: set = field(default_factory=set)   # the seasons whose held scales wait for the run's finished seasons

    @classmethod
    def read(cls, cfg: dict, seasons, settings=None) -> "SeasonsInProgress":
        rows = in_progress_panel(cfg)
        rows = rows[rows.possessions > 0].reset_index(drop=True)
        shares = {int(season): share_played(int(season), cfg) for season in seasons
                  if int(season) in set(rows.season.astype(int))}
        # a run that rates finished seasons too holds the season at their fresh scales, once they are rated
        # (`hold`, from rate_seasons); the morning's run of the season alone reads the published table now
        finished_in_run = any(int(season) not in shares for season in seasons)
        progress = cls(rows, shares)
        for season, share in shares.items():
            print(f"  {season} is in progress: rated from its {share:.1%} of the regular season so far, nothing "
                  f"trained on it", flush=True)
            if settings is not None and share < settings.ridge.held_scale_through:
                if finished_in_run:
                    progress.deferred.add(season)
                else:
                    progress.hold(cfg, settings, season)
                if settings.ridge.held_scale_blend:
                    weight = prior_rapm.blend_weight(share, settings.ridge.held_scale_through,
                                                     settings.ridge.held_scale_blend_from)
                    print(f"  {season}: its prior scales {weight:.0%} of the way from those to its own fitted ones, "
                          f"all the way at {settings.ridge.held_scale_through:.0%} of the season", flush=True)
        return progress

    def hold(self, cfg: dict, settings, season: int, fresh: pd.DataFrame | None = None) -> None:
        """The season's held scales (`held_scales`) from `fresh`, the step-3 tables this run has rated so far, and
        the published table."""
        self.held_scales[season] = held_scales(cfg, settings.ridge.held_scale_table, season, fresh)
        self.deferred.discard(season)


SCALE_COLUMNS = ["season", "prior_scale_off", "prior_scale_def"]


def held_scales(cfg: dict, table_name: str, season: int, fresh: pd.DataFrame | None = None) -> tuple | None:
    """(s_O, s_D) the season in progress holds: the mean of the 8 finished seasons nearest it
    (prior_rapm.held_prior_scale), each season's scales from `fresh`, the step-3 rows of the seasons this run has
    rated, or else from outputs/<table_name>.parquet; said once.  A full build after a change would otherwise hold
    the season at the previous build's scales until the next morning's run read the new table (2026-11-01).  None,
    the scales fitted on its games as before experiment 109, when neither holds a finished season."""
    path = resolve(cfg, "outputs") / f"{table_name}.parquet"
    stored = pd.read_parquet(path, columns=SCALE_COLUMNS) if path.exists() else pd.DataFrame(columns=SCALE_COLUMNS)
    fresh = pd.DataFrame(columns=SCALE_COLUMNS) if fresh is None else fresh[SCALE_COLUMNS]
    finished = trainable_through(cfg)
    stored = stored[~stored.season.isin(set(fresh.season)) & (stored.season.astype(int) <= finished)]
    fresh = fresh[fresh.season.astype(int) <= finished]
    parts = [part for part in (stored, fresh) if len(part)]
    if not parts:
        print(f"  {season}: no finished season in {path.name} or this run to hold its prior scales at; fitted on its "
              "games", flush=True)
        return None
    held = prior_rapm.held_prior_scale(pd.concat(parts, ignore_index=True), season)
    where = " and ".join(name for name, part in ((path.name, stored), ("this run's seasons", fresh)) if len(part))
    print(f"  {season}: prior scales held at {held[0]:.3f} / {held[1]:.3f}, the mean of the 8 finished seasons "
          f"nearest it in {where} (experiment 109)", flush=True)
    return held


def rate_one_season(season: int, sums: dict, panel: pd.DataFrame, designs: DesignCache, settings, params: dict,
                    lines: SecondPriorLines | None = None, label_cache: dict | None = None,
                    part_seasons: pd.DataFrame | None = None, progress: SeasonsInProgress | None = None):
    """The end of step 1 (labels), step 2 (priors) and step 3 (ridge) for one season.

    Returns (the season's centred table in raw sign, {"O", "D"} its SidePriors, its unseen seasons, its
    SeasonSecondPrior or None).  With `lines` (the second prior on, settings.PriorSettings.second_prior) step 3 reads
    the second prior instead of the box-score prior; `part_seasons` (`part_season_panel`) adds the box-score prior's
    part-season rows.  A season in `progress` is rated from its own rows there, the priors told the share played.
    """
    unseen = unseen_seasons(season, settings)
    labels = rapm.career_labels(sums, unseen, settings.labels, settings.targets)
    outside = row_label_tables(sums, panel, unseen, settings, label_cache)
    rated, share, scale = None, 1.0, None
    if progress is not None and season in progress.shares:
        rated, share, scale = progress.rows, progress.shares[season], progress.held_scales.get(season)
    priors = box_prior.season_priors(labels, panel, season, unseen, settings, params, outside, part_seasons, rated,
                                     share)
    season_designs = {target: designs.season(season, target) for target in settings.targets.distinct()}
    values = {side: priors[side].values for side in SIDES}
    second = None
    if lines is not None:
        second = rated_second_prior(designs.cfg, season, panel if rated is None else rated, priors, lines, settings,
                                    share)
        values = second.values
    ridge = settings.ridge
    weight = (1.0 if scale is None else prior_rapm.blend_weight(share, ridge.held_scale_through,
                                                                ridge.held_scale_blend_from)
              if ridge.held_scale_blend else 0.0)
    return (prior_rapm.rate_season_blended(season_designs, values, settings, season, scale, weight), priors, unseen,
            second)


# what a game-by-game build keeps of each game row: who, when, the box-score prior's answer and the prior step 3 read
GAME_PRIOR_COLUMNS = ["season", "player_id", "side", "game_id", "direction", "possessions", "season_progress",
                      "poss_off_game", "poss_def_game", "box_game", "prior_game"]


def network_rows(season: int, network, games: dict, unseen, base: pd.DataFrame, shares, backward=None,
                 per_game: bool = True) -> tuple:
    """(whole-season rows of every season but the unseen ones, and of the rated one; part-season rows of the seasons
    the box-score prior may train on; the rated season's game rows), each from the network that never saw the rated
    season (game_prior.py).  `games` maps each season to its game rows (data/player_games.py), `base` is the season
    panel the columns the network does not make come from.  With `backward`, the backward network of the rated season,
    the first half of each player's games take their rows from the backward pass (game_prior.splice).  Without
    `per_game`, one prior a season (PriorSettings.game_prior_whole_season), no game rows: None in their place."""
    training = [s for s in sorted(games) if s not in set(unseen)]
    trained, trained_states = game_prior.season_states(pd.concat([games[s] for s in training], ignore_index=True),
                                                       network)
    whole, parts = game_prior.network_panels(trained, trained_states, base, shares)
    rated, rated_states = game_prior.season_states(games[season], network)
    rated_whole, _ = game_prior.network_panels(rated, rated_states, base, ())
    if not per_game:
        return pd.concat([whole, rated_whole], ignore_index=True), parts, None
    rows = game_prior.game_rows(rated, rated_states, base)
    if backward is not None:
        rows = game_prior.splice(rows, game_prior.backward_game_rows(games[season], backward, base))
    else:
        rows = rows.assign(direction="forward")
    return pd.concat([whole, rated_whole], ignore_index=True), parts, rows


def season_network(cfg: dict, tag: str, season: int, unseen, reverse: bool):
    """The rated season's network of `tag` (None for no tag), refused when it trained on an unseen season or reads the
    season the other way round from `reverse`."""
    if not tag:
        return None
    network = load_network(cfg, tag, season, game_rates.input_names(), game_rates.NAMES)
    if set(int(s) for s in network.seasons) & set(unseen):
        raise SystemExit(f"{season}: the network of {tag} trained on {sorted(set(unseen))}")
    if bool(network.reverse) != bool(reverse):
        raise SystemExit(f"{tag}: its networks read the season {'backward' if network.reverse else 'forward'}; this "
                         f"setting reads it {'backward' if reverse else 'forward'}")
    return network


def game_box_priors(rows: pd.DataFrame, priors: dict, least_share: float) -> np.ndarray:
    """The box-score prior's answer for every game row, each side from its boosters, each row from the ones that
    never saw the player (box_prior.predict_chosen), its possessions and season share held at least at the training
    rows' smallest (box_prior.game_rated_rows)."""
    out = np.empty(len(rows))
    for side in SIDES:
        mask = (rows.side == side).to_numpy()
        rated = box_prior.game_rated_rows(rows[mask], priors[side].feature_names, 1.0, least_share)
        out[mask] = box_prior.predict_chosen(priors[side].model, rated)
    return out


def line_weights(coefficients: dict, n_folds: int, rows: pd.DataFrame) -> dict:
    """{"O", "D"}: {player_id: b}, the line's coefficient on his box-score prior, from the line his player fold reads
    (second_prior.season_second_priors holds team totals at b times it)."""
    ids = np.unique(rows.player_id.to_numpy(dtype=np.int64))
    return {side: dict(zip(ids, np.asarray(coefficients[side])[ids % int(n_folds), 0])) for side in SIDES}


def least_share(settings) -> float:
    """The smallest share of a season a part-season training row covers, which a game row's season share is held at;
    1 without part-season rows (box_prior.least_share)."""
    return box_prior.least_share(settings)


def rate_one_season_by_game(season: int, sums: dict, panel: pd.DataFrame, designs: DesignCache, settings,
                            params: dict, games: dict, lines: SecondPriorLines | None = None,
                            label_cache: dict | None = None):
    """`rate_one_season` with the box-score prior game by game (settings.PriorSettings.game_prior; game_prior.py):

    1. the labels, as `rate_one_season`;
    2. the box-score prior trained on the network's whole-season and part-season rows of the seasons it may train on;
    3. its answer before every game of the rated season, and each player's season average, by his possessions;
    4. with the second prior on, the line applied to the season average as today, and each game's prior the line's
       box-score coefficient times that game's answer plus the player's correction (game_prior.per_game_second);
    5. step 3 with each row pulled toward its own game's priors, the reported prior the scale times the average.

    Returns `rate_one_season`'s four, each SidePrior's values the season averages, and the rated season's game rows
    (`GAME_PRIOR_COLUMNS`).

    With PriorSettings.game_prior_whole_season, one prior a season in place of 3-5: the box-score prior's answer for
    the network's state after each player's last game is his prior, the second prior reads it as it reads the season
    panel's, and step 3 is today's; no game rows (None)."""
    unseen = unseen_seasons(season, settings)
    network = season_network(designs.cfg, settings.prior.game_prior, season, unseen, reverse=False)
    backward = season_network(designs.cfg, settings.prior.game_prior_backward, season, unseen, reverse=True)
    labels = rapm.career_labels(sums, unseen, settings.labels, settings.targets)
    outside = row_label_tables(sums, panel, unseen, settings, label_cache)
    whole_season = bool(settings.prior.game_prior_whole_season)
    whole, parts, rows = network_rows(season, network, games, unseen, panel, settings.prior.part_season_shares,
                                      backward, per_game=not whole_season)
    whole = whole[whole.possessions > 0].reset_index(drop=True)
    parts = None if parts is None else parts[parts.possessions > 0].reset_index(drop=True)
    priors = box_prior.season_priors(labels, whole, season, unseen, settings, params, outside, parts)
    if whole_season:
        values, second = {side: priors[side].values for side in SIDES}, None
        if lines is not None:
            second = rated_second_prior(designs.cfg, season, whole, priors, lines, settings, types_panel=panel)
            values = second.values
        season_designs = {target: designs.season(season, target) for target in settings.targets.distinct()}
        return prior_rapm.rate_season(season_designs, values, settings, season), priors, unseen, second, None
    rows = rows.assign(box_game=game_box_priors(rows, priors, least_share(settings)))
    box_average = game_prior.season_average(rows, "box_game")
    values, second = box_average, None
    if lines is not None:
        second = rated_second_prior(designs.cfg, season, whole, priors, lines, settings, box=box_average,
                                    types_panel=panel)
        weights = line_weights(second.coefficients, settings.prior.second_prior_folds, rows)
        rows["prior_game"] = game_prior.per_game_second(rows, "box_game", second.values, box_average, weights)
        values = second.values
    else:
        rows["prior_game"] = rows["box_game"]
    priors = {side: replace(priors[side], values=box_average[side]) for side in SIDES}
    season_designs = {target: designs.season(season, target) for target in settings.targets.distinct()}
    tables = {target: game_prior.game_tables(design, rows, "prior_game", values)
              for target, design in season_designs.items()}
    table = prior_rapm.rate_season(season_designs, values, settings, season, game_tables=tables)
    return table, priors, unseen, second, rows[GAME_PRIOR_COLUMNS]


def rate_seasons(seasons, sums: dict, panel: pd.DataFrame, designs: DesignCache, settings, params: dict,
                 keep_models: bool = False, lines: SecondPriorLines | None = None,
                 part_seasons: pd.DataFrame | None = None,
                 progress: SeasonsInProgress | None = None, games: dict | None = None) -> tuple[list, dict, list]:
    """`rate_one_season` for each season, or `rate_one_season_by_game` with the box-score prior game by game (`games`,
    each season's game rows): ([one centred table per season], {season: {side: saved boosters}}, [each season's game
    rows, when game by game]).

    The boosters are kept only when `keep_models`; thirty seasons of them are about a gigabyte.
    """
    started = time.time()
    rated, models, label_cache, game_rows = {}, {}, {}, []
    # the seasons in progress last: their held prior scales are the mean of the finished seasons' fresh ones
    ordered = sorted(seasons, key=lambda s: (progress is not None and int(s) in progress.shares, int(s)))
    for season in ordered:
        if progress is not None and season in progress.deferred:
            progress.hold(designs.cfg, settings, season, pd.concat(list(rated.values()), ignore_index=True)
                          if rated else None)
        if settings.prior.game_prior:
            table, priors, unseen, second, rows = rate_one_season_by_game(season, sums, panel, designs, settings,
                                                                          params, games, lines, label_cache)
            if rows is not None:
                game_rows.append(rows)
        else:
            table, priors, unseen, second = rate_one_season(season, sums, panel, designs, settings, params, lines,
                                                            label_cache, part_seasons, progress)
        rated[season] = table
        if keep_models:
            models[season] = {side: saved_side(priors[side], params[side], settings, unseen, second, side)
                              for side in SIDES}
        print(f"  {season}: {len(table)} players, prior scale {table.prior_scale_off.iloc[0]:.2f} / "
              f"{table.prior_scale_def.iloc[0]:.2f}  ({time.time() - started:.0f}s)", flush=True)
        if second is not None:
            print(f"    the line: {line_words(second.coefficients, second.inputs['O'].columns)}", flush=True)
    return [rated[season] for season in seasons], models, game_rows


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


def part_season_panel(cfg: dict, settings) -> pd.DataFrame | None:
    """The rows of each finished season so far at the shares settings.PriorSettings.part_season_shares asks, with
    possessions (data.panel.trainable_panel_so_far); None when it asks for none, as shipped."""
    shares = tuple(settings.prior.part_season_shares)
    if not shares:
        return None
    panel = trainable_panel_so_far(cfg)
    panel = panel[(panel.possessions > 0) & panel.share.isin(shares)].reset_index(drop=True)
    missing = sorted(set(shares) - set(panel.share.unique()))
    if missing:
        raise SystemExit(f"the season-so-far panel holds no rows at the shares {missing}; build it with `openrapm data "
                         f"panel --so-far {','.join(f'{share:g}' for share in shares)}`")
    print(f"part-season rows: each training season so far at {', '.join(f'{share:g}' for share in shares)} of its "
          f"regular season, {len(panel)} rows", flush=True)
    return panel


def season_game_rows(cfg: dict, settings, seasons) -> dict | None:
    """{season: its game rows} for a game-by-game build (settings.PriorSettings.game_prior): every season a fit may
    train on, read through the gate (data/player_games.py); None for a build that is not game by game.  A season in
    progress cannot be rated game by game yet: its whole-season rows would read as finished seasons."""
    if not settings.prior.game_prior:
        return None
    every = trainable(cfg).seasons
    late = sorted(set(int(s) for s in seasons) - set(every))
    if late:
        raise SystemExit(f"the box-score prior game by game rates finished seasons only; {late} is in progress")
    rows = trainable_player_games(cfg, every)
    print(f"game rows: {len(rows):,} over {rows.season.nunique()} seasons, the networks of "
          f"outputs/game_rates/{settings.prior.game_prior}/", flush=True)
    return {int(season): part.reset_index(drop=True) for season, part in rows.groupby("season")}


def rate_table(cfg: dict, settings, seasons, keep_models: bool = False) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Steps 1-3 for `seasons`: (the step-3 table in the published schema, {season: saved boosters} or {}, every
    season's game rows with the box-score prior game by game, or None)."""
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
    games = season_game_rows(cfg, settings, seasons)
    tables, models, game_rows = rate_seasons(seasons, sums, panel, designs, settings, params, keep_models=keep_models,
                                             lines=lines,
                                             part_seasons=None if games else part_season_panel(cfg, settings),
                                             progress=SeasonsInProgress.read(cfg, seasons, settings), games=games)
    table = prior_rapm.published_table(pd.concat(tables, ignore_index=True), player_names(cfg))
    return table, models, (pd.concat(game_rows, ignore_index=True) if game_rows else None)


def write_game_rows(rows: pd.DataFrame | None, cfg: dict, name: str) -> None:
    """outputs/game_priors_<name>.parquet: every game row's box-score prior and the prior step 3 read, so any
    player's prior can be followed across his season; nothing is written for a build that is not game by game."""
    if rows is None:
        return
    path = resolve(cfg, "outputs") / f"game_priors_{name}.parquet"
    rows.to_parquet(path, index=False)
    print(f"wrote {path}: {len(rows):,} game rows", flush=True)


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


def rate_in_parts(cfg: dict, seasons, name: str, save_models: str | None, argv, jobs: int) -> tuple:
    """`rate_table` over `parallel.season_chunks`, one `openrapm rate` child per chunk: (the step-3 table, the
    boosters by season or {}, the game rows or None).  Each child writes season_ratings_<name>__part<k>_step3.parquet,
    when boosters are kept prior_models_<save_models>__part<k>.pkl, and game by game
    game_priors_<name>__part<k>.parquet; all are read back here, in season order, and removed."""
    outputs = resolve(cfg, "outputs")
    chunks = parallel.season_chunks(seasons, jobs)

    def names(k):
        kept = ["--save-models", parallel.part_name(save_models, k)] if save_models else []
        return ["--name", parallel.part_name(name, k), *kept]

    parallel.run_children(list(argv), chunks, names, outputs / "parallel" / name)
    tables, models, game_rows = [], {}, []
    for k in range(len(chunks)):
        path = outputs / f"season_ratings_{parallel.part_name(name, k)}_step3.parquet"
        tables.append(pd.read_parquet(path))
        path.unlink()
        games = outputs / f"game_priors_{parallel.part_name(name, k)}.parquet"
        if games.exists():
            game_rows.append(pd.read_parquet(games))
            games.unlink()
        if save_models:
            part = outputs / f"prior_models_{parallel.part_name(save_models, k)}.pkl"
            saved = pd.read_pickle(part)
            saved.pop("meta", None)
            models.update(saved)
            part.unlink()
    table = pd.concat(tables, ignore_index=True).sort_values("season", kind="mergesort").reset_index(drop=True)
    rows = pd.concat(game_rows, ignore_index=True).sort_values("season", kind="mergesort") if game_rows else None
    return table, ({season: models[season] for season in seasons} if save_models else {}), rows


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
        table, models, game_rows = rate_in_parts(cfg, seasons, name, save_models, argv, fits)
    else:
        table, models, game_rows = rate_table(cfg, settings, seasons, keep_models=bool(save_models))
    print()
    write_table(table, resolve(cfg, "outputs") / f"season_ratings_{name}_step3.parquet", started)
    write_game_rows(game_rows, cfg, name)
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


def held_back_stint_equations(cfg: dict, folds: list, where: dict) -> dict:
    """{fold season: (A, b)}: the within layer's normal equations (prior_shrink.stint_equations) summed over each
    season's folds, each fold's held-back stints read from its season's points design."""
    started = time.time()
    designs = DesignCache(cfg, keep=1)
    equations: dict = {}
    for fold in sorted(folds, key=lambda f: (f.season, f.repeat, f.fold)):
        stints = held_back_stints(designs.season(fold.season, "pts"), fold_games(where[fold.stem], fold)[1])
        a, b = prior_shrink.stint_equations(fold, stints)
        slot = equations.get(fold.season)
        equations[fold.season] = (a, b) if slot is None else (slot[0] + a, slot[1] + b)
    print(f"  step 4: the within layer's equations from {len(folds)} folds' held-back stints "
          f"({time.time() - started:.0f}s)", flush=True)
    return equations


def within_layer_multipliers(cfg: dict, seasons, folds: list, where: dict, shrink,
                             second_folds: list | None = None) -> pd.DataFrame:
    """season and the four prior_shrink.WITHIN_COLUMNS: the within layer's multipliers on `folds`' held-back stints,
    each season's from the fold seasons `shrink.within_nearest` names (the team layer's `nearest` when None), read to
    a full season with `second_folds` as the team layer's are, then times settings.shrink.within_factors (offence,
    defence) and the defensive two times the defence factor."""
    nearest = shrink.nearest if shrink.within_nearest is None else int(shrink.within_nearest)
    multipliers = prior_shrink.within_multipliers_by_season(seasons, held_back_stint_equations(cfg, folds, where),
                                                            shrink.rule, nearest)
    if second_folds:
        directory = resolve(cfg, "within") / shrink.extrapolate_folds_tag
        second = prior_shrink.within_multipliers_by_season(
            seasons, held_back_stint_equations(cfg, second_folds, {fold.stem: directory for fold in second_folds}),
            shrink.rule, nearest)
        multipliers = prior_shrink.read_to_full_season(multipliers, second, prior_shrink.held_back_share(folds),
                                                       prior_shrink.held_back_share(second_folds))
    offence, defence = shrink.within_factors
    factor = multipliers.season.map(shrink.defence_factor_for).to_numpy(float)
    off_columns, def_columns = prior_shrink.WITHIN_COLUMNS[:2], prior_shrink.WITHIN_COLUMNS[2:]
    multipliers[list(off_columns)] = multipliers[list(off_columns)] * offence
    multipliers[list(def_columns)] = multipliers[list(def_columns)].mul(defence * factor, axis=0)
    return multipliers


def shrink_seasons(cfg: dict, table: pd.DataFrame, settings, folds: list | None = None,
                   where: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Step 4 on a step-3 table: (the shrunk table, the multipliers of every season).  `folds` and `where` default to
    `step4_folds`'s."""
    shrink = settings.shrink
    if folds is None or where is None:
        folds, where = step4_folds(cfg, shrink)
    second = extrapolation_folds(cfg, shrink)
    multipliers = step4_multipliers(table.season.unique(), folds, shrink, second)
    teams = None
    if shrink.within_team_layer:
        within = within_layer_multipliers(cfg, table.season.unique(), folds, where, shrink, second)
        multipliers = multipliers.merge(within, on="season")
        teams = main_teams(cfg)
        print(f"  step 4: two layers; each player's deviation from his team's mean multiplied by "
              f"{multipliers.within_multiplier_off.min():.3f} to {multipliers.within_multiplier_off.max():.3f} "
              f"(prior part) and {multipliers.within_games_multiplier_off.min():.3f} to "
              f"{multipliers.within_games_multiplier_off.max():.3f} (games' part) on offence, "
              f"{multipliers.within_multiplier_def.min():.3f} to {multipliers.within_multiplier_def.max():.3f} and "
              f"{multipliers.within_games_multiplier_def.min():.3f} to "
              f"{multipliers.within_games_multiplier_def.max():.3f} on defence, the within factors "
              f"{shrink.within_factors[0]:g} and {shrink.within_factors[1]:g} in", flush=True)
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
    return prior_shrink.shrink(table, multipliers, teams), multipliers


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
def replace_season_rows(old: pd.DataFrame, new: pd.DataFrame, what: str) -> pd.DataFrame:
    """`old` with the rows of `new`'s seasons replaced by `new`'s, in season order, every other row as it was.

    Example: the published table holds 1997-2026 and yesterday's 2027; with this morning's 2027 it holds 1997-2026 as
    they were and this morning's 2027."""
    if list(old.columns) != list(new.columns):
        raise SystemExit(f"{what} has the columns {list(old.columns)}, the build {list(new.columns)}: build every "
                         "season again instead of replacing some")
    kept = old[~old.season.isin(set(new.season.unique()))]
    return pd.concat([kept, new], ignore_index=True).sort_values("season", kind="stable").reset_index(drop=True)


def stored_tables(outputs: Path, name: str, cfg: dict) -> dict:
    """{suffix: the table written under `name` before}, and "multipliers": step 4's, for --replace-seasons."""
    paths = {suffix: outputs / f"season_ratings_{name}{suffix}.parquet" for suffix in ("_step3", "_step4", "")}
    paths["multipliers"] = resolve(cfg, "outputs") / "csv" / f"season_ratings_{name}_step4_multipliers.csv"
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise SystemExit(f"--replace-seasons needs the tables of {name} on disk; missing {missing}")
    return {key: pd.read_csv(path) if key == "multipliers" else pd.read_parquet(path) for key, path in paths.items()}


def run_build(cfg: dict, settings, seasons, name: str, save_models: str | None = None, argv=(),
              with_types: bool = False, replace_seasons: bool = False) -> int:
    """`openrapm build`: steps 1-5 in one process, the three tables and step 4's multipliers written at the end.
    Step 4's folds are checked first (`require_folds`), so missing folds stop the run in seconds.  With
    `replace_seasons`, the seasons built go into the tables already under `name`, in place of their rows there."""
    started = time.time()
    seasons = check_seasons(seasons, cfg)
    require_folds(cfg, settings.shrink)
    outputs = resolve(cfg, "outputs")
    stored = stored_tables(outputs, name, cfg) if replace_seasons else None
    step3, models, game_rows = rate_table(cfg, settings, seasons, keep_models=bool(save_models))
    write_game_rows(game_rows, cfg, name)
    folds, where = step4_folds(cfg, settings.shrink)
    step4, multipliers = shrink_seasons(cfg, step3, settings, folds, where)
    final, _, types = swap_seasons(cfg, step4, settings, folds, where, multipliers)
    if stored is not None:
        step3, step4, final = (replace_season_rows(stored[suffix], table, f"season_ratings_{name}{suffix}")
                               for suffix, table in (("_step3", step3), ("_step4", step4), ("", final)))
        multipliers = replace_season_rows(stored["multipliers"], multipliers, f"{name}'s step 4 multipliers")
        print(f"  seasons {', '.join(str(s) for s in seasons)} written in place of their rows in {name}'s tables, "
              f"{final.season.nunique()} seasons in all", flush=True)
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
