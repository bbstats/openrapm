"""The shipped settings: every number the five steps use, in one place, with where it was decided.

Two settings exist.  PRODUCT builds the published ratings.  TEST builds the table the year-over-year test
(defined below) scores: it rates each season from labels and training rows that have seen neither the season
before it nor the season after it, because those are the seasons the test predicts (DECISIONS.md, "The
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

Every value names the experiment that set it.  "DECISIONS.md, experiment 2" means the section of that name in
DECISIONS.md.  The booster's own settings are a file, params/booster_params_lgb1.json, read by
`openrapm.data.config.booster_params` (this module does no I/O).
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

__all__ = ["Targets", "LabelSettings", "PriorSettings", "RidgeSettings", "ShrinkSettings", "SwapSettings",
           "Settings", "PRODUCT", "TEST"]


@dataclass(frozen=True)
class Targets:
    """What each side's ridge explains, per stint, in points per 100 possessions (rapm.py defines both)."""
    # 100 x (points - made free throws + expected free throws) / possessions: each free throw counted at the
    # shooter's padded free-throw %.  The only luck adjustment that has improved the ratings: 0.26 per 100 on
    # the year-over-year test against plain points (DECISIONS.md, "What was settled": the free-throw adjustment)
    offence: str = "xpts_ft"
    # the offence target with 3 x 0.75 x (threes made - threes expected) also taken off: three quarters of the
    # shooters' three-point luck removed.  Keeping the other quarter moved the score by -0.007 per 100 at
    # z = -0.34 (z: the paired t-statistic over the scored seasons), nothing measurable (DECISIONS.md, "What was
    # settled": x3def)
    defence: str = "x3def_w0.25"

    def distinct(self) -> list:
        """The targets to fit, each once: [offence, defence], or one name when both sides share a target."""
        return list(dict.fromkeys([self.offence, self.defence]))


@dataclass(frozen=True)
class LabelSettings:
    """Step 1: the leave-season-out RAPM that becomes the prior's label."""
    # swept on the end-to-end score, not on the RAPM's own accuracy; defence decisive, offence a plateau
    # (DECISIONS.md, "Choose a training target's penalty on the end-to-end score", 2026-09-12)
    offense_penalty: float = 40000.0
    defense_penalty: float = 40000.0
    context_penalty: float = 0.0
    # a player needs this many possessions, counted on offence only (rapm.player_possessions), to get a label
    # at all; set with the label when it was introduced (the same 2026-09-12 section)
    min_possessions: float = 100.0
    # only the defensive label is un-shrunk: the offensive un-shrink emptied the top of the offensive
    # ratings of its stars (DECISIONS.md, experiment 22, 2026-09-18)
    unshrink_columns: tuple = ("defense",)
    # above this many label possessions the label is the raw-scale coefficient; below it, shrunk toward
    # its possession tier's level (DECISIONS.md, experiments 1 and 2 of 2026-09-14 evening)
    full_trust_possessions: float = 4444.0
    tier_edges: tuple = (0.0, 500.0, 1500.0, 4444.0)


@dataclass(frozen=True)
class PriorSettings:
    """Step 2: the box-score prior."""
    # "boruta_noonc": the inputs Boruta selected, without the on-court columns (features.SHIPPED; DECISIONS.md,
    # experiment 45, adopted 2026-10-08)
    feature_set: str = "boruta_noonc"
    # the career row plus every run of 1, 2 and 3 consecutive seasons (DECISIONS.md, experiment 3)
    chunk_sizes: tuple = (1, 2, 3)
    # each player's prior from the booster fitted without his fold (DECISIONS.md, "The SPM memorises
    # long-career stars, and the out-of-player prior removes it", 2026-09-14)
    n_folds: int = 5
    # 0: players sorted by label and assigned to folds 0,1,2,3,4,4,3,2,1,0,...  A positive number k is the
    # noise control, which measures how much the ratings move when nothing changes but which players share a
    # fold: each group of five players, in label order, gets the folds 0-4 in a random order drawn with seed k.
    # k = 1 is the run DECISIONS.md calls noise_regroup2 ("The noise control, the swap step on every fold, and
    # the body-weight preview")
    fold_assignment: int = 0
    # LightGBM's settings, tuned by scoring each candidate prior on 2026's and 2015's one-season rows against
    # labels from the other seasons (DECISIONS.md, experiments 32 and 44); read from params/ only, so an
    # untracked copy in outputs/ can never stand in for it
    booster_file: str = "params/booster_params_lgb1.json"


@dataclass(frozen=True)
class RidgeSettings:
    """Step 3: the ridge around the prior."""
    # one penalty for every season and both sides, chosen once on the year-over-year test
    # (DECISIONS.md, experiment 2 of 2026-09-14 evening)
    penalty: float = 13037.0


@dataclass(frozen=True)
class ShrinkSettings:
    """Step 4: the prior's part of each rating multiplied down by factors fitted on within-season held-out games.

    A within-season fold rates a season from three quarters of its games and keeps the other quarter back;
    the factors are a weighted least-squares fit on those held-back games (DECISIONS.md, experiment 33b).
    """
    # "product" fits the factors without the rated season; "test" also without its two neighbours
    rule: str = "product"
    # the first and last season whose within-season folds are read; pinned, because the folds on disk grew
    # once and silently changed the factors (DECISIONS.md, experiment 34's correction, 2026-10-05)
    fold_seasons: tuple = (2017, 2026)
    # the within-season folds the shipped factors were fitted on (adopted 2026-10-03, DECISIONS.md, experiment 33)
    folds_tag: str = "weightfix_within"


@dataclass(frozen=True)
class SwapSettings:
    """Step 5: each team's credit re-split by its same-four lineup swaps, team totals held.

    A same-four swap is a pair of lineups of one team that share four players; the type model is a ridge that
    predicts what a swap says about the two swapped players from the difference of their box-score profiles.
    Two rules have no setting, because only one of each ever shipped: the give-back takes a team's total back
    from its players in proportion to their share of its possessions (DECISIONS.md, experiment 41), and the
    spread is held inside teams, one factor per season and side (DECISIONS.md, "The swap adjustment with the
    spread held: it passes").
    """
    # the share of the type model's prediction applied: half (kappa in DECISIONS.md, "The swap adjustment with
    # the spread held: it passes", adopted 2026-10-01: "Team version.")
    applied_share: float = 0.5
    # the type model's ridge penalty, relative to the mean diagonal of its Gram matrix, and the number of player
    # folds it is fitted on (the swap adjustment's first run, 2026-10-01)
    type_model_penalty: float = 1e-4
    n_folds: int = 5
    # the type model trains on seasons more than this far from the rated one: 0 in the product setting (every
    # other season: model coefficients may be learned from other seasons, the owner's second ruling in
    # HANDOFF.md, "The rulings that bind"), 1 in the test setting, so neither season the test scores trains it
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
