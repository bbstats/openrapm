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

__all__ = ["Targets", "LabelSettings", "PriorSettings", "ROW_LABELS", "TEAM_TOTALS", "SECOND_PRIOR_INPUTS",
           "RidgeSettings", "ShrinkSettings",
           "SwapSettings", "Settings", "PRODUCT", "TEST"]


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
    feature_set: str = "hustle_no_three_pct"
    # the career row plus every run of 1, 2 and 3 consecutive seasons (notes/DECISIONS.md, experiment 3)
    chunk_sizes: tuple = (1, 2, 3)
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
    # what each training row is labelled with.  "career": every row of a player carries his RAPM over every training
    # season, its own included (shipped).  "outside": each chunk row carries his RAPM over the seasons outside its
    # own and the career row is dropped, so no label shares a game with its row's box score (box_prior.training_rows;
    # notes/DECISIONS.md, experiment 79, the owner, 2026-10-19: "try it on the prior first").  "nearest": each chunk
    # row carries his RAPM over only the seasons right next to it, the one before and the one after, so the label is
    # the player at about the row's age (rapm.nearest_labels; notes/DECISIONS.md, experiment 83)
    row_labels: str = "career"
    # "": step 3 reads the box-score prior itself, as it did until experiment 82.  A tag: step 3 reads the second
    # prior instead, shipped as ctx82 since experiment 82 was adopted 2026-10-20, "adopt for now", its inputs centred
    # since experiment 93 (`second_prior_inputs`).  It is the
    # box-score prior corrected for who a player played with and against, by a line fitted on the swaps of the
    # within-season folds' held-back games written under outputs/second_prior/<tag>/ (second_prior.py;
    # notes/DECISIONS.md, experiment 81, the owner, 2026-10-19: "take the handoff idea ... in as inputs to the prior")
    second_prior: str = "ctx82"
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
    # shipped: "context_and_types_centred", experiment 93, adopted 2026-10-21 (experiment 82's "context_and_types"
    # from 2026-10-20)
    second_prior_inputs: str = "context_and_types_centred"
    # 0: each season's line is fitted on every season the setting allows.  K: on only the K nearest of them, as step 4
    # fits its multipliers (experiment 85: experiment 82 won 1997-2016 and lost 2017-2026 with one line for all eras)
    second_prior_nearest: int = 0


ROW_LABELS = ("career", "outside", "nearest")
TEAM_TOTALS = ("held", "free", "spread_held")    # PriorSettings.second_prior_team_totals
# PriorSettings.second_prior_inputs
SECOND_PRIOR_INPUTS = ("context", "context_and_types", "context_and_types_centred", "own_teammates_opponents")


@dataclass(frozen=True)
class RidgeSettings:
    """Step 3: the ridge around the prior."""
    # one penalty for every season and both sides, chosen once on the year-over-year test
    # (notes/DECISIONS.md, experiment 2 of 2026-09-14 evening)
    penalty: float = 13037.0


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
    # own
    folds_tag: str = "heave95_early,heave95_within"
    # each season's factors from only the K fold seasons nearest it (prior_shrink.fold_seasons_used): the defensive
    # prior part holds up less in older seasons, 0.79 in 1997-2006 against 0.95 in 2017-2026 (notes/DECISIONS.md,
    # "Era drift" and experiment 55, adopted 2026-10-12: "yes. build."); 0 would use every fold season the rule allows
    nearest: int = 8
    # a multiplier on the games' part too, fitted beside the prior's (prior_shrink, "The games' part freed";
    # notes/DECISIONS.md, experiment 65, adopted 2026-10-15: "Adopt one multiplier as the new incumbent")
    free_games_part: bool = True
    # one more factor on both defensive multipliers, shared by every season, chosen with the booster's settings by
    # check 152's guarded search (notes/DECISIONS.md, experiment 76, adopted 2026-10-16): the held-back games set the
    # defence's width within a season, and the next-season check wants it narrower on the tuned prior; 1 is the old step
    defence_factor: float = 0.866


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
class Settings:
    """Everything one build reads, step by step."""
    targets: Targets = field(default_factory=Targets)
    labels: LabelSettings = field(default_factory=LabelSettings)
    prior: PriorSettings = field(default_factory=PriorSettings)
    ridge: RidgeSettings = field(default_factory=RidgeSettings)
    shrink: ShrinkSettings = field(default_factory=ShrinkSettings)
    swap: SwapSettings = field(default_factory=SwapSettings)
    # 0: the labels and training rows leave out the rated season only; 1: its two neighbours as well
    exclude_neighbours: int = 0


PRODUCT = Settings()
TEST = replace(PRODUCT, exclude_neighbours=1,
               shrink=replace(PRODUCT.shrink, rule="test"),
               swap=replace(PRODUCT.swap, exclude_near=1))
