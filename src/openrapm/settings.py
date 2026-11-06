"""The shipped settings: every number the five steps use, in one place, with where it was decided.

Two settings exist.  PRODUCT builds the published ratings.  TEST builds the table the year-over-year test
(defined below) scores: it rates each season from labels and training rows that have seen neither the season
before it nor the season after it, because those are the seasons the test predicts (notes/DECISIONS.md, "The
year-over-year test", 2026-09-13).

Words used below
    label       a player's RAPM over every season except the rated one (and, in TEST, its two neighbours):
                the number the box-score prior learns to predict (step 1)
    penalty     a ridge's lambda: a player's coefficient keeps about n / (n + lambda) of his raw impact (the
                coefficient an unpenalised fit would give him), n = his possessions
    booster     LightGBM's gradient-boosted trees, the model the box-score prior is (step 2)
    player fold one of five groups of players; each player's prior comes from the booster fitted without
                his group, so it cannot simply return his own label (step 2)
    within-season fold   one season's games split into a fitted three quarters and a held-back quarter;
                step 4's factors are fitted on the held-back quarters
    year-over-year test   the deciding score (scripts/checks/63_yoy.py): each season's ratings predict the games of
                the season before and the season after from the ten players' ratings alone (only the scored
                season's intercept and home edge are refitted), scored against the points actually scored

Example: with the label penalty of 40,000, a player with 2,600 possessions keeps 2,600 / 42,600 = 6% of his
raw impact, which is why the defensive label is un-shrunk before the prior learns from it.

Every value names the experiment that set it.  "notes/DECISIONS.md, experiment 2" means the section of that name in
notes/DECISIONS.md.  The booster's own settings are a file, params/booster_params_lgb1_tune2.json, read by
`openrapm.data.config.booster_params` (this module does no I/O).
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

__all__ = ["Targets", "LabelSettings", "UNSHRINK_LABELS", "PriorSettings", "ROW_LABELS", "TEAM_TOTALS", "PREDICTIONS",
           "SECOND_PRIOR_INPUTS", "SECOND_PRIOR_TARGETS",
           "RidgeSettings", "ShrinkSettings", "factor_for", "describe_factors",
           "SwapSettings", "WinsSettings", "Settings", "PRODUCT", "TEST"]


def factor_for(factors: tuple, season: int) -> float:
    """The factor of `season` from (first season, factor) pairs in season order, each pair holding from its first
    season until the next pair's; a season before the first pair takes the first pair's.

    Example: with ((0, 0.89), (2017, 0.84)), 2016 gives 0.89, and 2017 and 2027 give 0.84.
    """
    chosen = factors[0][1]
    for first, factor in factors:
        if int(season) >= first:
            chosen = factor
    return float(chosen)


def describe_factors(factors: tuple) -> str:
    """The pairs in words: "0.866 every season", or "0.89 before 2017, 0.84 from 2017"."""
    if len(factors) == 1:
        return f"{factors[0][1]:g} every season"
    return ", ".join([f"{factors[0][1]:g} before {factors[1][0]}"]
                     + [f"{factor:g} from {first}" for first, factor in factors[1:]])


@dataclass(frozen=True)
class Targets:
    """What each side's ridge explains, per stint, in points per 100 possessions (rapm.py defines both).

    Before 2026-10-11 the two were called xpts_ft and x3def_w0.25 (data/legacy_names.py still reads those).
    """
    # 100 x (points - made free throws + expected free throws) / possessions: each free throw counted at the
    # shooter's padded free-throw %.  The only luck adjustment that has improved the ratings: 0.26 per 100 on
    # the year-over-year test against plain points (notes/DECISIONS.md, "What was settled": the free-throw adjustment)
    offence: str = "points_with_expected_free_throws"
    # the offence target with 3 x 0.75 x (threes made - threes expected) also taken off: three quarters of the
    # shooters' three-point luck removed, a quarter kept (data/shots.py, OPPONENT_THREE_LUCK_KEPT).  Keeping the
    # quarter moved the score by -0.007 per 100 at z = -0.34 (z: the paired t-statistic over the scored seasons),
    # nothing measurable (notes/DECISIONS.md, "What was settled": x3def).  Since experiment 95, adopted 2026-10-22
    # (the bug-fix pass), the threes of a period's last two seconds count as they happened, not at the shooter's
    # percentage: heaves go in 15-16% of the time, against 36% for every three (data/shots.py, late_as_scored)
    defence: str = "points_with_expected_opponent_threes_late_as_scored"

    def distinct(self) -> list:
        """The targets to fit, each once: [offence, defence], or one name when both sides share a target."""
        return list(dict.fromkeys([self.offence, self.defence]))


@dataclass(frozen=True)
class LabelSettings:
    """Step 1: the leave-season-out RAPM that becomes the prior's label."""
    # swept on the end-to-end score, not on the RAPM's own accuracy; defence decisive, offence a plateau
    # (notes/DECISIONS.md, "Choose a training target's penalty on the end-to-end score", 2026-09-12)
    offense_penalty: float = 40000.0
    defense_penalty: float = 40000.0
    context_penalty: float = 0.0
    # a player needs this many possessions, counted on offence only (rapm.player_possessions), to get a label
    # at all; set with the label when it was introduced (the same 2026-09-12 section)
    min_possessions: float = 100.0
    # only the defensive label is un-shrunk: the offensive un-shrink emptied the top of the offensive
    # ratings of its stars (notes/DECISIONS.md, experiment 22, 2026-09-18)
    unshrink_columns: tuple = ("defense",)
    # above this many label possessions the label is the raw-scale coefficient; below it, shrunk toward
    # its possession tier's level (notes/DECISIONS.md, "Experiments 1 and 2 of 2026-09-14 evening": experiment 1)
    full_trust_possessions: float = 4444.0
    tier_edges: tuple = (0.0, 500.0, 1500.0, 4444.0)
    # the team-season flags' penalties, attacking and defending (rapm.py, "team-season flags"); 0 and 0 = no flags,
    # the shipped setting.  Experiment 60 (notes/DECISIONS.md) sets them on the command line
    team_offense_penalty: float = 0.0
    team_defense_penalty: float = 0.0

    @property
    def team_flags(self) -> bool:
        """Whether step 1's fit carries the team-season flags."""
        return max(self.team_offense_penalty, self.team_defense_penalty) > 0.0


@dataclass(frozen=True)
class PriorSettings:
    """Step 2: the box-score prior."""
    # "hustle_no_three_pct_self_creation" (notes/DECISIONS.md, experiment 110, adopted 2026-10-25: "the rating changes
    # are good - we can adopt and publish"): the offensive list with the owner's self-creation appended.  Before it:
    # "hustle_no_three_pct" (notes/DECISIONS.md, experiment 77, adopted 2026-10-17: "Adopt and publish"): hustle with
    # three-point % off the offensive booster, the one input the cull on the decider found worse than its own
    # shuffled copy (check 152 `cull`).  Before it:
    # "hustle" (notes/DECISIONS.md, experiment 76, adopted 2026-10-16: "Hell yeah. Publish"): no_career_counts plus the
    # hustle box score's counts, each summed over the games a rating is built from, missing before 2017.  Before it:
    # "no_career_counts" (notes/DECISIONS.md, experiment 59, adopted 2026-10-14: "adopt anyway"): box_score_and_bio
    # without the career counts (career possessions before, seasons played before, seasons with his current team), so
    # the prior reads only the rated season's box score, role and body.  Before it:
    # "career_from_debut" (notes/DECISIONS.md, experiment 57, adopted 2026-10-13): box_score_and_bio with seasons played
    # before, career possessions before and seasons with his current team counted from each player's debut, his
    # seasons before 1997 included (data/careers.py), each in the place of the 1997-counted column.
    # "box_score_and_bio": the inputs Boruta selected (a feature-selection method: an input is kept when its importance
    # beats that of shuffled copies of the inputs), without the on-court columns (features.SHIPPED; experiment 45,
    # adopted 2026-10-08, where it is called boruta_noonc)
    feature_set: str = "hustle_no_three_pct_self_creation"
    # the career row plus every run of 1, 2 and 3 consecutive seasons (notes/DECISIONS.md, experiment 3)
    chunk_sizes: tuple = (1, 2, 3)
    # the seasons so far whose rows join each training season's: its first quarter, half and three quarters, each a
    # chunk row of its own with the share it covers as its row_seasons (box_prior.training_rows; notes/DECISIONS.md,
    # experiment 107, the owner, 2026-10-23: "it would be good to add in partial seasons - 1/4, 1/2, 3/4 as well - to
    # better support early season data", adopted the same day: "Sure adopt").  () adds none, as before it
    part_season_shares: tuple = (0.25, 0.5, 0.75)
    # each player's prior from the booster fitted without his fold (notes/DECISIONS.md, "The SPM memorises
    # long-career stars, and the out-of-player prior removes it", 2026-09-14)
    n_folds: int = 5
    # 0: players sorted by label and assigned to folds 0,1,2,3,4,4,3,2,1,0,...  A positive number k is the
    # noise control, which measures how much the ratings move when nothing changes but which players share a
    # fold: each group of five players, in label order, gets the folds 0-4 in a random order drawn with seed k.
    # k = 1 is the run notes/DECISIONS.md calls noise_regroup2 ("The noise control, the swap step on every fold, and
    # the body-weight preview")
    fold_assignment: int = 0
    # LightGBM's settings, chosen by the with-or-without-you decider itself: check 152's search with Optuna's
    # Gaussian-process sampler, the next-season check a hard limit (notes/DECISIONS.md, experiment 76, adopted
    # 2026-10-16).  Before it lgb1_cols06: tuned on the prior's error against career labels on 2026 and 2015
    # (experiments 32, 44 and 63).  Read from params/ only, so an untracked copy in outputs/ can never stand in for it
    booster_file: str = "params/booster_params_lgb1_tune2.json"
    # 0: every training season counts the same.  A positive H: a training row counts half as much for every H
    # seasons between its seasons and the rated one, beyond the nearest season the fit may use (box_prior.era_kernel;
    # notes/DECISIONS.md, "Era drift" and experiment 56: the defensive prices drift between eras)
    era_halflife: float = 0.0
    # 0: one booster per side.  K: K boosters, each on a bootstrap of the players, their answers averaged
    # (box_prior.BaggedPrior; notes/DECISIONS.md, experiment 78, adopted 2026-10-17: "Adopt and publish"; one
    # booster before it)
    bags: int = 10
    # how a player's prior avoids the boosters that saw him (box_prior's module docstring; notes/DECISIONS.md, "The
    # noise control read closely", 2026-10-26).  "folds": every bag's players split into the n_folds player folds, each
    # answered by the booster fitted without his fold (shipped).  "out_of_bag": one booster per bag and no folds, each
    # player answered by the mean of the bags whose resample did not draw him; it needs enough bags that every player
    # misses one, 60 for the cost of the shipped 10
    prediction: str = "folds"
    # False: every bag's players split by fold_assignment (shipped).  True: each bag splits its players its own random
    # way (box_prior.bag_split), so the prior averages over as many splits as bags
    split_per_bag: bool = False
    # 0: the shipped bags.  A positive d: bag k drawn and fitted at k + 100 d (box_prior.BAG_DRAW_STEP), every bag's
    # resample and LightGBM seed another; a noise control that redraws the resamples, where fold_assignment redraws
    # only the split
    bag_draw: int = 0
    # what each training row is labelled with.  "career": every row of a player carries his RAPM over every training
    # season, its own included (shipped).  "outside": each chunk row carries his RAPM over the seasons outside its
    # own and the career row is dropped, so no label shares a game with its row's box score (box_prior.training_rows;
    # notes/DECISIONS.md, experiment 79, the owner, 2026-10-19: "try it on the prior first").  "nearest": each chunk
    # row carries his RAPM over only the seasons right next to it, the one before and the one after, so the label is
    # the player at about the row's age (rapm.nearest_labels; notes/DECISIONS.md, experiment 83)
    row_labels: str = "career"
    # "": step 3 reads the box-score prior itself, as it did until experiment 82.  A tag: step 3 reads the second
    # prior instead, shipped as ctx82 since experiment 82 was adopted 2026-10-20, "adopt for now", its inputs centred
    # since experiment 93 (`second_prior_inputs`); ctx110 since experiment 126, adopted 2026-11-01, the same line
    # refitted on creation110's boosters and folds (ctx82's came from experiment 78's, before the parser fixes, the
    # body listing, the part-season rows and self-creation); experiment 127, adopted the same day, was tested and
    # adopted on this line, so it stayed until a refit passed a test of its own: rating134 since experiment 134, adopted
    # 2026-11-04 (the owner: "Adopt, then show me"), the weights fitted on the within-season test through the ridge
    # (`openrapm second-prior --target rating`) instead of on the swaps.  It is the
    # box-score prior corrected for who a player played with and against, by a line fitted on the swaps of the
    # within-season folds' held-back games written under outputs/second_prior/<tag>/ (second_prior.py;
    # notes/DECISIONS.md, experiment 81, the owner, 2026-10-19: "take the handoff idea ... in as inputs to the prior")
    second_prior: str = "rating134"
    # the line's player folds (player_id % 5) and its ridge penalty relative to the mean diagonal of its Gram matrix:
    # step 5's type model's, the method the line borrows
    second_prior_folds: int = 5
    second_prior_penalty: float = 1e-4
    # each team's total: "held" at the line's box-score part's, b x box-score prior, the context re-splitting credit
    # inside the team only, or "free", the line as it is.  A swap cannot see team strength, and with totals free the
    # step-3 ratings predicted the held-back team-games worse in 44 of 48 folds of four seasons (second_prior.py;
    # notes/DECISIONS.md, experiment 81).  "spread_held": totals held and each season's spread inside teams held to
    # the box-score part's too, as step 5 held it (experiment 90)
    second_prior_team_totals: str = "held"
    # what the line reads: "context", the box-score prior and the eight context numbers (experiment 81), or
    # "context_and_types", also the other end's box-score prior and step 5's box counts, role and body, so the swaps'
    # lesson enters before the ridge (experiment 82, the owner, 2026-10-19: "wouldnt it stand to reason to put the
    # swap adjustments into the priors before RAPMizing?"), or "context_and_types_centred", the same centred on the
    # season's possession-weighted average, so no player fold's line carries its own level (experiment 93)
    # shipped: "context_types_team_rates_centred", experiment 134, adopted 2026-11-04, today's 29 and the team's eight
    # box rates with the player on and off court ("context_and_types_centred", experiment 93, from 2026-10-21;
    # experiment 82's "context_and_types" from 2026-10-20)
    second_prior_inputs: str = "context_types_team_rates_centred"
    # 0: each season's line is fitted on every season the setting allows.  K: on only the K nearest of them, as step 4
    # fits its multipliers (experiment 85: experiment 82 won 1997-2016 and lost 2017-2026 with one line for all eras)
    second_prior_nearest: int = 0
    # how the Detail view divides the prior; no rating reads this.  False: the Box score column is the line's own weight
    # on the box-score prior.  True: it is his own stats' weights learned alone, and Box Adj is the rest of the same
    # prior (rating_line.first_stage; the owner's choice of 2026-11-05, published the same day)
    box_first: bool = True
    # "": one box-score prior per player and season, from the season panel (shipped).  A tag: the box-score prior game
    # by game (game_prior.py; notes/DECISIONS.md, experiment 114, the owner, 2026-10-27): its rows are the multi-stat
    # networks' forecasts, outputs/game_rates/<tag>/, each player's state before each game of the season, the
    # part-season rows the states on the days of `part_season_shares`; the ridge pulls each stint toward its own game's
    # prior and the season's prior part is the possession-weighted average
    game_prior: str = ""
    # "": every game's prior from the games before it.  A tag of networks trained on reversed seasons: the first half of
    # each player's games take their prior from the games after them instead (game_prior.splice; the owner, 2026-10-27:
    # "run inference in reverse, to smooth out the weird beginning")
    game_prior_backward: str = ""
    # False: with `game_prior`, a prior before every game.  True: one prior per player and season, the box-score
    # prior's answer for the networks' state after his last game, every game of the season read at once (the owner,
    # 2026-10-27: "build the whole season version"): the networks' cleaner rates in place of the season panel's, and no
    # game-by-game averaging
    game_prior_whole_season: bool = False


ROW_LABELS = ("career", "outside", "nearest")
# LabelSettings.unshrink_columns by name (`openrapm rate --unshrink-labels`): the defensive label alone, shipped since
# experiment 22 (2026-09-18), or both labels, experiment 102 (notes/DECISIONS.md; the owner, 2026-10-23: "let's try un
# shrinking the offensive label"): the offensive prior is then priced on the same scale as the defensive one
UNSHRINK_LABELS = {"defense": ("defense",), "both": ("offense", "defense")}
TEAM_TOTALS = ("held", "free", "spread_held")    # PriorSettings.second_prior_team_totals
PREDICTIONS = ("folds", "out_of_bag")              # PriorSettings.prediction
# PriorSettings.second_prior_inputs
SECOND_PRIOR_INPUTS = ("context", "context_and_types", "context_and_types_centred", "own_teammates_opponents",
                       "context_types_team_rates_centred", "context_types_team_profile_centred",
                       "context_types_team_rates_zones_centred")
# what the line's target adds back to a swap's residual, `openrapm second-prior --target` (second_prior.py, experiment
# 130)
SECOND_PRIOR_TARGETS = ("whole_rating", "prior_part", "rating")


@dataclass(frozen=True)
class RidgeSettings:
    """Step 3: the ridge around the prior."""
    # one penalty for every season and both sides, chosen once on the year-over-year test
    # (notes/DECISIONS.md, experiment 2 of 2026-09-14 evening)
    penalty: float = 13037.0
    # the season in progress holds its prior scales, the ridge's unpenalised weight on each side's prior, at the mean
    # of the 8 nearest finished seasons' until this share of its regular season is played: fitted on a season's first
    # games they swing (2027 on 16 games, 2.24 on offense; 2026 on its whole season, 1.17), and held beat fitted on the
    # rest of every past season at 2%, 5%, 10% and 25% of it, in every era, and at half of it in every era by a little
    # (z -2.1 over all seasons); at three quarters the two tie, so from three quarters on the scales are fitted, as a
    # finished season's are (notes/DECISIONS.md, experiment 109 and its continuation, adopted 2026-10-30; 0.50, a
    # switch at midseason, until the line below was adopted 2026-10-31).  0 fits them on the season's games throughout
    held_scale_through: float = 0.75
    # False: held in full until `held_scale_through`, then fitted, as adopted 2026-10-30; True: held in full until
    # `held_scale_blend_from`, then a straight line to fitted at `held_scale_through` (prior_rapm.blend_weight; the
    # owner, 2026-10-31: "If it's just two different shrinkages we should probably just blend between them").  From
    # the first game the line lost to holding at a tenth and a quarter of a season (experiment 109 blended); from a
    # quarter it ties holding at half a season and beats fitting there, z -2.7 (adopted 2026-10-31)
    held_scale_blend: bool = True
    held_scale_blend_from: float = 0.25
    # the step-3 table whose finished seasons' scales the season in progress holds: the published build's own
    held_scale_table: str = "season_ratings_product_step3"


@dataclass(frozen=True)
class ShrinkSettings:
    """Step 4: the prior's part of each rating multiplied down by factors fitted on within-season held-out games.

    A within-season fold rates a season from three quarters of its games and keeps the other quarter back;
    the factors are a weighted least-squares fit on those held-back games (notes/DECISIONS.md, experiment 33b).
    """
    # "product" fits the factors without the rated season; "test" also without its two neighbours
    rule: str = "product"
    # the first and last season whose within-season folds are read; pinned, because the folds on disk grew
    # once and silently changed the factors (notes/DECISIONS.md, experiment 34's correction, 2026-10-05).  Every
    # season since experiment 55 (adopted 2026-10-12); 2017-2026 before
    fold_seasons: tuple = (1997, 2026)
    # the within-season folds the factors are fitted on, comma-separated folders: experiment 78's own folds, 1997-2016
    # and 2017-2026 (notes/DECISIONS.md, "Experiment 78", adopted 2026-10-17; before it experiment 77's,
    # nothreepct_early,nothreepct_within, experiment 76's, tune2_early,tune2_within, and experiment 63's,
    # cols06_early,cols06_within); the method is experiment 33b's, adopted 2026-10-03
    # since experiment 82, adopted 2026-10-20, its own folds, ctx82_early,ctx82_within, built with the second prior;
    # since experiment 93, adopted 2026-10-21, its own, ctx82c_early,ctx82c_within, built with the centred inputs;
    # since experiment 94, adopted 2026-10-22 (the bug-fix pass's parser fixes), its own on the rebuilt stints,
    # parser94_early,parser94_within; since experiment 95, adopted the same day (heaves counted as they happened), its
    # own, heave95_early,heave95_within; since experiment 99, adopted the same day (each season's own body listing, the
    # pad k around the bins), its own, inputs99_early,inputs99_within; since experiment 107, adopted 2026-10-23 (the
    # box-score prior's part-season rows), its own; since experiment 110, adopted 2026-10-25 (self-creation in the
    # offensive prior), its own; since experiment 126, adopted 2026-11-01 (the swap part of the prior refitted), its
    # own; since experiment 127, adopted the same day (the possession bins' targets padded toward the league rate),
    # its own; since experiment 134, adopted 2026-11-04 (the line fitted on the rating, with team rates), its own
    folds_tag: str = "rating134_early,rating134_within"
    # each season's factors from only the K fold seasons nearest it (prior_shrink.fold_seasons_used): the defensive
    # prior part holds up less in older seasons, 0.79 in 1997-2006 against 0.95 in 2017-2026 (notes/DECISIONS.md,
    # "Era drift" and experiment 55, adopted 2026-10-12: "yes. build."); 0 would use every fold season the rule allows
    nearest: int = 8
    # a second fold set, a tenth of each season's games held back ten times a repeat: each multiplier is read off the
    # straight line through the two holdout sizes at nothing held back, as the full-season ratings are built
    # (Richardson extrapolation, prior_shrink.full_season_multipliers; notes/DECISIONS.md, experiment 101, adopted
    # 2026-10-22: "we are going to do it"); "" fits on folds_tag alone, as the experiment chain does.  inputs99_k10
    # until experiment 107 was adopted; its own since, parts107_k10; experiment 110's own since 2026-10-25; experiment
    # 126's own since 2026-11-01, then experiment 127's the same day; experiment 134's own since 2026-11-04
    extrapolate_folds_tag: str = "rating134_k10"
    # a multiplier on the games' part too, fitted beside the prior's (prior_shrink, "The games' part freed";
    # notes/DECISIONS.md, experiment 65, adopted 2026-10-15: "Adopt one multiplier as the new incumbent")
    free_games_part: bool = True
    # one more factor on both defensive multipliers, by era: (first season, factor) pairs, each holding from its first
    # season until the next pair's (`factor_for`).  0.866 in every season from experiment 76 (adopted 2026-10-16,
    # check 152's guarded search: the held-back games set the defence's width within a season, and the next-season
    # check wants it narrower); since experiment 98 (adopted 2026-10-22) 0.89 before 2017 and 0.84 from 2017: the
    # within-season test wants the older seasons' defence wider in all 20 of them, and narrowing the newest era keeps
    # the year-over-year test even (notes/DECISIONS.md, "The defense factor by era"); since experiment 136, adopted
    # 2026-11-04, 0.8682 before 2017 and 0.8704 from 2017: halfway from those to what check 156 found for experiment
    # 134's line, 0.8464 and 0.9007, which went past the year-over-year limit.  ((0, 1.0),) is the old step
    defence_factors: tuple = ((0, 0.8682), (2017, 0.8704))
    # two layers (prior_shrink, "Since experiment 106"): each part split into his team's mean, multiplied as above, and
    # his deviation from it, with four multipliers of its own fitted on the held-back stints within games
    # (notes/DECISIONS.md, experiment 106, adopted 2026-10-23: "please adopt, yes - D"); False is one layer
    within_team_layer: bool = True
    # the within layer's multipliers times these, offence then defence: (0.75, 1.0), version D of experiment 106, until
    # experiment 136, adopted 2026-11-04: halfway from those to check 156's (0.7826, 0.9935).  Rounding them is not
    # free: (0.77, 1.0) with a defense factor of 0.87 moves the year-over-year test from z +1.46 to z +1.83 (experiment
    # 137)
    within_factors: tuple = (0.7663, 0.9968)
    # the within layer's fold seasons: None the team layer's `nearest`, as shipped; K its K nearest; 0 every fold season
    # the rule allows.  Experiment 123 (registered 2026-10-30): fitted on one season alone the within layer's
    # offensive games' multiplier runs from 0.23 (2020) to 2.50 (2011), so the 8-season window jumps as a season
    # enters or leaves it (1.293 in 2010, 0.976 in 2011)
    within_nearest: int | None = None

    def defence_factor_for(self, season: int) -> float:
        """The factor on `season`'s two defensive multipliers."""
        return factor_for(self.defence_factors, season)


@dataclass(frozen=True)
class SwapSettings:
    """Step 5: each team's credit re-split by its same-four lineup swaps, team totals held.

    A same-four swap is a pair of lineups of one team that share four players; the type model is a ridge that
    predicts what a swap says about the two swapped players from the difference of their box-score profiles.
    Two rules have no setting, because only one of each ever shipped: the give-back takes a team's total back
    from its players in proportion to their share of its possessions (notes/DECISIONS.md, experiment 41), and the
    spread is held inside teams, one factor per season and side (notes/DECISIONS.md, "The swap adjustment with the
    spread held: it passes").  Nor does where the type model learns: since experiment 80 (adopted 2026-10-19) from
    the within-season folds' held-back games, its inputs from the fit games, so a box count never meets the results
    it was counted in; the folds are step 4's (`ShrinkSettings.folds_tag`).
    """
    # the share of the type model's prediction applied: half (kappa in notes/DECISIONS.md, "The swap adjustment with
    # the spread held: it passes", adopted 2026-10-01: "Team version.")
    # 0 since experiment 82, adopted 2026-10-20: the swaps' lesson moved into the second prior, so step 5 leaves
    # step 4's ratings exactly as they are; 0.5 before it
    applied_share: float = 0.0
    # the type model's ridge penalty, relative to the mean diagonal of its Gram matrix, and the number of player
    # folds it is fitted on (the swap adjustment's first run, 2026-10-01)
    type_model_penalty: float = 1e-4
    n_folds: int = 5
    # the type model trains on seasons more than this far from the rated one: 0 in the product setting (every
    # other season: model coefficients may be learned from other seasons, the owner's second ruling in
    # notes/HANDOFF.md, "The rulings that bind"), 1 in the test setting, so neither season the test scores trains it
    exclude_near: int = 0


@dataclass(frozen=True)
class WinsSettings:
    """Wins over starter average (wins.py): no rating moves, and nothing here is scored by the within-season test.
    The owner's decisions of 2026-11-02 and check 157's readings are in notes/DECISIONS.md, "WOSA"."""
    # the bar: the mean of the players with the most regular-season starts, minutes breaking ties.  150 is five starters
    # for each of 30 teams; before 2005 the league had 29
    starters: int = 150
    # the pace ridge's penalty, in seconds of weight: the least held-back error of check 157 on half-A and half-B games
    # of six seasons across the eras, least in five of them and both neighbours worse (10,000 by 0.04 in 2001)
    pace_penalty: float = 30000.0
    # the pace ridge leaves out stints shorter than this: under 15 seconds a stint is 0.5% of the time and 43% of the
    # weighted squared error (2025), a substitution at a free throw rather than a pace
    pace_min_seconds: float = 15.0
    # rating points, weighted by leverage inside each game, for one win: check 157's slope of teams' records on the
    # regular-season ratings, 28.7 / 28.9 / 30.4 by era (a season's standard error about 1.5).  One number for every
    # season: a season's own slope would add its noise to every player, about 0.6 wins on a star
    points_per_win: float = 30.0
    # what the wins count and everything the module fits reads; the rating itself stays the season's
    phases: tuple = ("RS",)


@dataclass(frozen=True)
class Settings:
    """Everything one build reads, step by step."""
    targets: Targets = field(default_factory=Targets)
    labels: LabelSettings = field(default_factory=LabelSettings)
    prior: PriorSettings = field(default_factory=PriorSettings)
    ridge: RidgeSettings = field(default_factory=RidgeSettings)
    shrink: ShrinkSettings = field(default_factory=ShrinkSettings)
    swap: SwapSettings = field(default_factory=SwapSettings)
    wins: WinsSettings = field(default_factory=WinsSettings)
    # 0: the labels and training rows leave out the rated season only; 1: its two neighbours as well
    exclude_neighbours: int = 0


PRODUCT = Settings()
TEST = replace(PRODUCT, exclude_neighbours=1,
               shrink=replace(PRODUCT.shrink, rule="test"),
               swap=replace(PRODUCT.swap, exclude_near=1))
