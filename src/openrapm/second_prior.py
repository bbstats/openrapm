"""Between steps 2 and 3: the second prior, a player's box-score prior corrected for who he played with and against.

Step 2's booster reads a player's own box score.  A box score also depends on the other nine players on the court:
beside good shooters a passer's assists come easier, and against bench units a bench player's points come easier.
This module fits by how much, on games the box scores were not counted in, and step 3 pulls the season's RAPM toward
the corrected number instead of the box-score prior (the owner, 2026-10-19: "take the handoff idea ... in as inputs to
the prior, to hopefully learn a correction factor"; notes/DECISIONS.md, experiment 81).

Words used below
    stint row         one stint seen from one side (swap_test.season_rows): the offence's five, the defence's five,
                      both teams, the game and the possessions.  A stint gives two rows, one per end
    box-score prior   step 2's answer for one player, raw sign: on offence points scored per 100, on defence points
                      allowed per 100
    on the court      the stint rows he played, on either end
    on the bench      the stint rows of the games he played in where his team was on the court without him
    never saw him     every number in a player's inputs comes from boosters that never trained on his rows: in each
                      bag, the fold booster that left him out, or the bag's full booster when no fold did
                      (box_prior.predict's rule).  A teammate's prior inside his inputs is that same booster's answer
                      for the teammate, not the teammate's own prior, so his own other seasons cannot reach him
                      through a teammate's booster
    swap              two lineups of one team in one season that share four players (swap_test.py)
    line              the second prior: one weight on his box-score prior and one on each of his eight context
                      numbers, per side

His eight context numbers, each a possession-weighted average of box-score priors:

    teammates_offence_on       his 4 teammates' offensive priors, over his rows on the court
    teammates_defence_on       their defensive priors, over the same rows
    opponents_offence_on       the 5 opponents' offensive priors, over the same rows
    opponents_defence_on       their defensive priors
    teammates_offence_bench    his team's 5 players' offensive priors, over his rows on the bench
    teammates_defence_bench    their defensive priors
    opponents_offence_bench    the 5 opponents' offensive priors, over his rows on the bench
    opponents_defence_bench    their defensive priors

All eight keep their level: a player on a strong team carries his team's strength in them (the owner, 2026-10-19: "i
dont think you actually want to cancel out team strength here, that's a helpful data point").

Example, before padding: one game, player 7's team.  For 10 possessions he plays beside teammates whose offensive
priors are +1.0, 0.0, 0.0 and -1.0; for 5 more his team's five without him average +0.4.  His teammates_offence_on is
(1 + 0 + 0 - 1) / 4 = 0.0 and his teammates_offence_bench is +0.4.

Each number is then padded toward his team's level (the owner, 2026-10-19: "let's use padded stats yea?"), the method of
moments of on_court.py:

    padded = (n x number + k x level) / (n + k)
    tau^2  = Var_n(number - level) - sigma^2 E_n[1 / n]        k = sigma^2 / tau^2

    n        the possessions behind the number: his possessions on the court, or on the bench
    level    his team's season average of the same kind of number: the average prior of the five it put on the court,
             or of the five it faced, over all its stint rows; a player of two teams takes each by his possessions
             with it
    sigma^2  the noise of one possession, with each stint row counted as one draw because its possessions share one
             five: sum over players and their rows of w^2 (row value - his number)^2, over sum of his w
    tau^2    the spread of the numbers between players beyond that noise, one per number and season

so a bench number resting on a few possessions is mostly his team's level, and a player who never sat gets the level
itself.  The padding is the same when the line is fitted (on a fold's fit games) and when it is applied (on the rated
season), so the line's weights are read on inputs of the kind they were fitted on.  sigma^2 and tau^2 are read off the
average of all the boosters' answers; each player's number and level off the boosters that never saw him.

The line, one per side, is fitted on the swaps in the held-back games of the within-season folds
(commands/second_prior.py writes their normal equations):

    minimise over beta   sum over swaps  h (d - (x_a - x_b)' beta)^2  +  lambda |beta|^2

    x_a, x_b   the two swapped players' inputs, all from the fold's fit games: his box-score prior on this side, then
               his eight context numbers (`INPUTS`)
    d          the two lineups' points difference per 100 in the held-back games, the opponents they faced and the game
               context taken out with the fold's step-4 ratings held fixed:
                   d = r + (rating_a - rating_b)
               r is the swap's residual against those ratings (swap_adjust.residual_pairs).  The four kept players
               cancel in r, so adding back the two swapped players' own ratings leaves the observed difference minus
               the opponents' and the context's part of it
    h          the swap's information, P_a P_b / (P_a + P_b), P each lineup's possessions
    lambda     `penalty` x the mean diagonal of the Gram matrix, as step 5's type model

    second prior = x' beta, from the line fitted without the swaps of his player fold (player_id % n_folds)

A swap compares two players of one team, so the line's weights are learned inside teams; the intercept cancels in every
swap and is 0 (step 3 centres each season and refits its own scale on the second prior).  Applied to the raw levels
above, a weight learned inside teams would also move whole teams, which no swap tested.  So by default each team's total
stays the line's box-score part's, b x box-score prior, b the line's own weight on it (`TEAM_TOTALS` "held"), every
player of a team moved by the same amount:

    deviation    = line - b x box-score prior: the context part, per player
    c            = deviation - M s,   s = (W M)^-1 W deviation        every team's possession-weighted total of c is 0
    second prior = b x box-score prior + c

    W[t, i]   player i's possessions for team t on this end
    M[i, t]   the share of player i's possessions on this end that were for team t

so for a player of one team, c = deviation - his team's possession-weighted mean deviation, and the line's gaps inside
the team are kept exactly; a player of two teams moves by his mix of their two amounts.  The anchor is b x box-score
prior and not the box-score prior itself, so team totals and the gaps inside teams are on one scale and step 3's one
prior scale fits both: held at the box-score prior's own scale, about a third of the line's, step 3's scale came out
near 1.3 and stretched every team's gaps by as much (notes/DECISIONS.md, experiment 81, the first run).

Team strength still splits the credit inside a team, through the raw numbers; the team's total is the box-score
prior's.  On the held-back team-games of 48 folds of 2002, 2012, 2019 and 2025 (notes/DECISIONS.md, experiment 81), the
second prior with team totals free was worse than the incumbent in 44 of 48, and its split inside teams with the
incumbent's totals better in 44 of 48.  "free" applies the line as it is.

"spread_held" also holds the spread inside teams, as step 5 held it (notes/DECISIONS.md, "The swap adjustment with the
spread held: it passes", and experiment 90): after the team totals, one factor per season and side pulls every player
toward his main team's possession-weighted mean until the spread inside teams equals the box-score part's,
b x box-score prior's, so the line orders the players inside each team and adds no width (`hold_spread`).  Without
it, experiment 82's ratings came out wider than the incumbent's, defence 14-22% by era, as step 5's did without its
hold on 2026-10-01: 12% on offence, 27% on defence.

With step 5's inputs (`INPUT_SETS` "context_and_types", experiment 82; the owner, 2026-10-19: "wouldnt it stand to
reason to put the swap adjustments into the priors before RAPMizing? instead of doing it after RAPM?"), the line also
reads the other end's box-score prior and step 5's 19 box-score, role and body numbers (swap_adjust.PANEL_FEATURES),
each standardised with one mean and standard deviation over every player-season of the panel (`type_scale`), all from
the same games as the rest of his inputs.  Fitted on the same held-back swaps, it learns before the ridge what step 5
learns after it, and step 3 weighs it against the season's games.

The three-input prior (`INPUT_SETS` "own_teammates_opponents", experiment 92; the owner, 2026-10-21: "[stats] ->
prelim prior; [prelim prior, teammate prelim (same side of off/def), opponent prelim (other side of off/def)] -> final
prior") reads three numbers per side and nothing else:

    box_prior                  his box-score prior on this side
    teammates_same_side_on     his 4 teammates' priors on this side, summed, over his rows on the court
    opponents_other_side_on    the 5 opponents' priors on the other side, summed, over the same rows

the sums four and five times experiment 81's padded averages.  All three passed the second stage's Boruta over every
season on both sides (notes/DECISIONS.md, "Boruta on the second stage").  Its inputs are applied centred on the
season's possession-weighted average (`centred`): the line is fitted on swaps, which see only differences between
players, so it has no intercept, and each player fold's line then carried its own level, w_f x the average inputs;
in experiment 82 a player's defensive prior moved by up to 0.8 per 100 with the fold his id falls in.  Centred, every
fold's line reads 0 at the average player.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp

from . import box_prior, swap_adjust
from .box_rates import shrink
from .on_court import NO_SPREAD_K
from .settings import TEAM_TOTALS
from .swap_test import DEF, OFF, Season

__all__ = ["CONTEXT", "BOX", "INPUTS", "OTHER_BOX", "TYPES", "INPUT_SETS", "type_scale", "standardised_types",
           "SIDE_KEYS", "BoosterAnswers", "booster_answers", "own_priors", "KINDS", "pad_toward_level",
           "context_numbers", "line_inputs", "side_inputs", "swap_targets", "held_back_grams", "season_coefficients",
           "second_priors", "TEAM_TOTALS", "team_possessions", "hold_team_totals", "hold_spread",
           "season_second_priors", "SAME_TEAMMATES", "OTHER_OPPONENTS", "CENTRED_SETS", "three_inputs", "centred"]

CONTEXT = ["teammates_offence_on", "teammates_defence_on", "opponents_offence_on", "opponents_defence_on",
           "teammates_offence_bench", "teammates_defence_bench", "opponents_offence_bench", "opponents_defence_bench"]
BOX = "box_prior"
INPUTS = [BOX, *CONTEXT]              # the line's inputs, in the order of its weights
OTHER_BOX = "box_prior_other_end"     # his box-score prior on the other end, raw sign
TYPES = list(swap_adjust.PANEL_FEATURES)    # step 5's box counts, role and body, standardised (`type_scale`)
SAME_TEAMMATES = "teammates_same_side_on"     # the three-input prior's teammates, this side, summed (module docstring)
OTHER_OPPONENTS = "opponents_other_side_on"   # its opponents, the other side, summed
# the inputs each version of the line reads (settings.PriorSettings.second_prior_inputs)
INPUT_SETS = {"context": INPUTS, "context_and_types": [*INPUTS, OTHER_BOX, *TYPES],
              "own_teammates_opponents": [BOX, SAME_TEAMMATES, OTHER_OPPONENTS]}
CENTRED_SETS = {"own_teammates_opponents"}    # applied on inputs centred on the season's average (`centred`)
SIDE_KEYS = {"offense": "O", "defense": "D"}      # swap_adjust's side names -> the priors' side keys


# ------------------------------------------------------------------------------- every booster's answer
@dataclass
class BoosterAnswers:
    """Every booster's answer for every player of one side, and, per bag, which answer never saw each player.

    player_ids   (n,) the players, in the order of the rows the boosters were asked about
    values       (n, M): column m is booster m's answer for every player
    choice       (n, B): for each of the B bags, the column of `values` whose booster never saw that row's player
    """
    player_ids: np.ndarray
    values: np.ndarray
    choice: np.ndarray


def booster_answers(prior, rows: pd.DataFrame) -> BoosterAnswers:
    """Every booster of `prior` (box_prior.OutOfPlayerPrior or BaggedPrior) asked about every row of `rows`.

    Each bag gives 1 + n_folds columns: its full booster, then its fold boosters in order.  A player held out of a
    fold is answered for by that fold's booster, any other player by the bag's full booster, exactly as
    box_prior.predict chooses; so the mean over bags of each player's chosen answers is his box-score prior.
    """
    members = prior.members if isinstance(prior, box_prior.BaggedPrior) else [prior]
    matrix = rows[prior.inputs].to_numpy(float)
    ids = rows.player_id.to_numpy()
    columns, choice = [], []
    for member in members:
        full = len(columns)
        columns.append(member.full.predict(matrix))
        chosen = np.full(len(ids), full)
        for k, (model, held_out) in enumerate(zip(member.fold_models, member.held_out_players)):
            columns.append(model.predict(matrix) + 0.0)
            chosen[np.fromiter((pid in held_out for pid in ids), dtype=bool, count=len(ids))] = full + 1 + k
        choice.append(chosen)
    return BoosterAnswers(player_ids=ids, values=np.column_stack(columns), choice=np.column_stack(choice))


def own_priors(answers: BoosterAnswers) -> pd.Series:
    """player_id -> his box-score prior: the mean over bags of the answers that never saw him."""
    rows = np.arange(len(answers.player_ids))[:, None]
    return pd.Series(answers.values[rows, answers.choice].mean(axis=1), index=answers.player_ids)


# ------------------------------------------------------------------------------- the context numbers
def _positions(ids: np.ndarray, players: np.ndarray) -> np.ndarray:
    return np.searchsorted(ids, players)


def _aligned(ids: np.ndarray, answers: BoosterAnswers) -> np.ndarray:
    """`answers.values` with one row per id of `ids`, 0 for a player the boosters were not asked about (step 3 gives
    such a player a prior of 0 too: prior_rapm.aligned_prior)."""
    out = np.zeros((len(ids), answers.values.shape[1]))
    out[_positions(ids, answers.player_ids)] = answers.values
    return out


def _chosen(numbers: np.ndarray, ids: np.ndarray, answers: BoosterAnswers) -> np.ndarray:
    """Each answered player's number averaged over bags, every bag's read on the column that never saw him."""
    rows = _positions(ids, answers.player_ids)[:, None]
    return numbers[rows, answers.choice].mean(axis=1)


KINDS = ("teammates_{}_on", "opponents_{}_on", "teammates_{}_bench", "opponents_{}_bench")


def pad_toward_level(number, possessions, level, sigma2: float) -> np.ndarray:
    """The pad of the module docstring: (n x number + k x level) / (n + k), k = sigma^2 / tau^2 by the method of
    moments over the players with possessions.  Everyone is at his level when the numbers show no spread beyond the
    noise; a player without possessions is at his level always.

    Example: two players with 100 possessions each, numbers +1 and -1 around a level of 0, sigma^2 50: tau^2 = 1 -
    50 x 0.01 = 0.5, k = 100, so both are padded halfway, to +0.5 and -0.5.
    """
    number, n, level = (np.asarray(value, dtype=float) for value in (number, possessions, level))
    played = n > 0
    if not played.any():
        return level.copy()
    tau2 = float(np.average((number[played] - level[played]) ** 2, weights=n[played])
                 - sigma2 * np.average(1.0 / n[played], weights=n[played]))
    k = sigma2 / tau2 if tau2 > 1e-12 else NO_SPREAD_K
    return shrink(number, n, k, level)


class _Entries:
    """The stint rows as the context numbers read them: each row twice, once for each of its fives (an entry: one team's
    five on the court, on either end, and the five it faced), and the sums a player's numbers are made of."""

    def __init__(self, stints: pd.DataFrame, ids: np.ndarray):
        off = stints[OFF].to_numpy(dtype=np.int64)
        de = stints[DEF].to_numpy(dtype=np.int64)
        n_entries = 2 * len(stints)
        entries = np.arange(n_entries)

        def incidence(columns, n_columns):
            return sp.csr_matrix((np.ones(columns.size), (np.repeat(entries, columns.size // n_entries),
                                                          columns.ravel())), shape=(n_entries, n_columns))

        self.weight = np.tile(stints.poss.to_numpy(dtype=float), 2)
        self.five = incidence(np.searchsorted(ids, np.vstack([off, de])), len(ids))
        self.faced = incidence(np.searchsorted(ids, np.vstack([de, off])), len(ids))
        team = np.concatenate([stints.team_off.to_numpy(dtype=np.int64), stints.team_def.to_numpy(dtype=np.int64)])
        games = np.tile(stints.game_id.astype(str).to_numpy(), 2)
        group, _ = pd.factorize(pd.Series(games) + ":" + pd.Series(team).astype(str))
        self.groups = incidence(group, int(group.max()) + 1)
        # his team's games: every (team, game) he was on the court in
        self.played = ((self.five.T @ self.groups) > 0).astype(float)
        team_code, _ = pd.factorize(team)
        self.teams = incidence(team_code, int(team_code.max()) + 1)
        self.team_weight = np.asarray(self.teams.T @ self.weight).ravel()
        self.mix = self.five.T @ sp.diags(self.weight) @ self.teams         # his possessions with each team
        self.on_weight = self.on(self.weight)
        self.bench_weight = np.maximum(self.bench(self.weight), 0.0)

    def on(self, values):
        """Each player's sum of `values` (one per entry, or entries x columns) over his entries on the court."""
        return np.asarray(self.five.T @ values)

    def bench(self, values):
        """The same over his team's entries in his games without him on the court."""
        return np.asarray(self.played @ (self.groups.T @ values)) - self.on(values)

    def level(self, per_entry):
        """Each player's team level of an entry's number (entries x columns): his teams' possession-weighted averages
        over all their entries, each team by his possessions with it."""
        team_average = (np.asarray(self.teams.T @ (self.weight[:, None] * per_entry))
                        / np.where(self.team_weight > 0, self.team_weight, 1.0)[:, None])
        return np.asarray(self.mix @ team_average) / np.where(self.on_weight > 0, self.on_weight, 1.0)[:, None]

    def numbers(self, values):
        """{kind: players x columns}: the four kinds of number of the module docstring, unpadded, for every column of
        `values` (players x columns of priors), and {"teammates", "opponents"}: their team levels."""
        lineup, against = self.five @ values, self.faced @ values          # an entry's five's priors, summed
        w = self.weight[:, None]
        on = np.where(self.on_weight > 0, self.on_weight, 1.0)[:, None]
        bench = np.where(self.bench_weight > 0, self.bench_weight, 1.0)[:, None]
        raw = {"teammates_{}_on": (self.on(w * lineup) - self.on_weight[:, None] * values) / (4.0 * on),
               "opponents_{}_on": self.on(w * against) / (5.0 * on),
               "teammates_{}_bench": self.bench(w * lineup) / (5.0 * bench),
               "opponents_{}_bench": self.bench(w * against) / (5.0 * bench)}
        return raw, {"teammates": self.level(lineup / 5.0), "opponents": self.level(against / 5.0)}

    def noise(self, prior: np.ndarray, keep: np.ndarray) -> dict:
        """{kind: sigma^2}, the noise of one possession of each kind of number (module docstring), on one prior per
        player, over the players at the positions `keep`.

        A player's number is a possession-weighted mean of its rows' values v; counting each row as one draw, its
        variance is sum w^2 (v - number)^2 / (sum w)^2, so sigma^2 = sum w^2 (v - number)^2 / sum w over everyone.
        With v = u - shift, sum w^2 (v - x)^2 = sum w^2 u^2 - 2 c sum w^2 u + c^2 sum w^2 for c = x + shift.
        """
        raw, _ = self.numbers(prior[:, None])
        lineup, against = self.five @ prior, self.faced @ prior
        w2 = self.weight ** 2
        parts = {"teammates_{}_on": (self.on, lineup / 4.0, prior / 4.0, self.on_weight),
                 "opponents_{}_on": (self.on, against / 5.0, 0.0, self.on_weight),
                 "teammates_{}_bench": (self.bench, lineup / 5.0, 0.0, self.bench_weight),
                 "opponents_{}_bench": (self.bench, against / 5.0, 0.0, self.bench_weight)}
        out = {}
        for kind, (total, u, shift, n) in parts.items():
            c = raw[kind][:, 0] + shift
            spread = total(w2 * u * u) - 2.0 * c * total(w2 * u) + c * c * total(w2)
            out[kind] = float(np.sum(spread[keep]) / max(np.sum(n[keep]), 1e-12))
        return out


def context_numbers(stints: pd.DataFrame, offence: BoosterAnswers, defence: BoosterAnswers,
                    pad: bool = True) -> pd.DataFrame:
    """player_id -> his eight context numbers (`CONTEXT`), padded toward his team's level (module docstring), for every
    player answered on both sides who played in `stints`.

    `stints` is swap_test.season_rows' columns game_id, team_off, team_def, poss, o1..o5, d1..d5, for the games the
    numbers may read.  `offence` and `defence` are `booster_answers` for the offensive and defensive boosters.  Each
    stint row counts once for the offence's five and once for the defence's five, by its possessions.  `pad` False
    gives the averages unpadded (the docstring example), a number without possessions at its level.
    """
    ids = np.unique(np.concatenate([stints[OFF].to_numpy(dtype=np.int64).ravel(),
                                    stints[DEF].to_numpy(dtype=np.int64).ravel(),
                                    offence.player_ids.astype(np.int64), defence.player_ids.astype(np.int64)]))
    rows = _Entries(stints, ids)
    keep = np.intersect1d(np.intersect1d(offence.player_ids, defence.player_ids), ids[rows.on_weight > 0])
    at = np.searchsorted(ids, keep)
    out = {}
    for prior, answers in (("offence", offence), ("defence", defence)):
        values = _aligned(ids, answers)
        raw, levels = rows.numbers(values)
        noise = rows.noise(values.mean(axis=1), at)
        for kind, number in raw.items():
            who = "teammates" if kind.startswith("teammates") else "opponents"
            n = (rows.on_weight if kind.endswith("_on") else rows.bench_weight)[at]
            chosen = pd.Series(_chosen(number, ids, answers), index=answers.player_ids).reindex(keep).to_numpy()
            level = pd.Series(_chosen(levels[who], ids, answers), index=answers.player_ids).reindex(keep).to_numpy()
            out[kind.format(prior)] = (pad_toward_level(chosen, n, level, noise[kind]) if pad
                                       else np.where(n > 0, chosen, level))
    frame = pd.DataFrame({name: out[name] for name in CONTEXT}, index=pd.Index(keep, name="player_id"))
    return frame


def type_scale(rows: pd.DataFrame) -> dict:
    """{name: [mean, population standard deviation]} of each of `TYPES` over `rows`, one row per player-season (the
    panel's offence rows: its box counts, role and body are the same on both ends)."""
    values = rows[TYPES].astype(float)
    return {name: [float(values[name].mean()), float(values[name].std(ddof=0))] for name in TYPES}


def standardised_types(rows: pd.DataFrame, scale: dict) -> pd.DataFrame:
    """player_id -> `TYPES`, each (value - mean) / standard deviation with `scale`'s numbers; `rows` one per player."""
    frame = rows.drop_duplicates("player_id").set_index("player_id")[TYPES].astype(float)
    centre = pd.Series({name: scale[name][0] for name in TYPES})
    spread = pd.Series({name: scale[name][1] for name in TYPES}).replace(0.0, 1.0)
    return (frame - centre) / spread


def line_inputs(box: pd.Series, context: pd.DataFrame, other_box: pd.Series | None = None,
                types: pd.DataFrame | None = None) -> pd.DataFrame:
    """player_id -> the line's inputs for one side: his box-score prior on that side (raw sign), then his eight context
    numbers (`INPUTS`); with `types`, also his box-score prior on the other end and step 5's numbers
    (`INPUT_SETS["context_and_types"]`).  A player missing a number gets its mean over the players who have it (0
    for a standardised one)."""
    frame = pd.DataFrame({BOX: box.astype(float)}).join(context, how="left")
    frame[CONTEXT] = frame[CONTEXT].fillna(context.mean())
    frame.index.name = "player_id"
    if types is None:
        return frame[INPUTS]
    other = other_box.astype(float).reindex(frame.index)
    frame[OTHER_BOX] = other.fillna(other_box.astype(float).mean())
    frame = frame.join(types, how="left")
    frame[TYPES] = frame[TYPES].fillna(0.0)
    return frame[INPUT_SETS["context_and_types"]]


def three_inputs(box: pd.Series, context: pd.DataFrame, side: str) -> pd.DataFrame:
    """player_id -> the three-input prior's inputs for one side (`INPUT_SETS["own_teammates_opponents"]`): his
    box-score prior on that side, his teammates' priors on that side summed over the four, his opponents' priors on the
    other side summed over the five, both with him on the court.  A missing number is filled as `line_inputs` fills it.

    Example: on defence, teammates whose defensive priors average +0.5 sum to +2.0; opponents whose offensive priors
    average +0.2 sum to +1.0."""
    word, other = {"O": "offence", "D": "defence"}, {"O": "D", "D": "O"}[side]
    frame = pd.DataFrame({BOX: box.astype(float)}).join(context, how="left")
    frame[CONTEXT] = frame[CONTEXT].fillna(context.mean())
    out = pd.DataFrame({BOX: frame[BOX], SAME_TEAMMATES: 4.0 * frame[f"teammates_{word[side]}_on"],
                        OTHER_OPPONENTS: 5.0 * frame[f"opponents_{word[other]}_on"]}, index=frame.index)
    out.index.name = "player_id"
    return out


def centred(frame: pd.DataFrame, possessions: pd.Series) -> pd.DataFrame:
    """`frame` less each column's possession-weighted mean; `possessions` player_id -> his possessions on this end,
    0 for a player without them.

    Example: two players of 300 and 100 possessions with values 1.0 and 5.0 average 2.0 and become -1.0 and +3.0."""
    weight = possessions.reindex(frame.index).fillna(0.0).to_numpy(float)
    mean = (frame.to_numpy(float) * weight[:, None]).sum(axis=0) / weight.sum()
    return frame - mean


def side_inputs(box: dict, context: pd.DataFrame, types: pd.DataFrame | None = None,
                input_set: str | None = None) -> dict:
    """{"O", "D"}: `line_inputs` for each side, `box` {"O", "D"}: player_id -> box-score prior (raw sign); with
    `input_set` "own_teammates_opponents", `three_inputs` instead."""
    if input_set == "own_teammates_opponents":
        return {side: three_inputs(pd.Series(box[side], dtype=float), context, side) for side in ("O", "D")}
    other = {"O": "D", "D": "O"}
    return {side: line_inputs(pd.Series(box[side], dtype=float), context, pd.Series(box[other[side]], dtype=float),
                              types) for side in ("O", "D")}


# ------------------------------------------------------------------------------- the line, on swaps
def swap_targets(pairs: dict, ratings: pd.DataFrame) -> dict:
    """{side: `pairs[side]` with r replaced by d = r + (rating_a - rating_b)}, the line's target (module docstring).

    `pairs` is swap_adjust.residual_pairs on `ratings` (player_id, o, d in raw sign); a player the ratings lack was
    rated 0 there and is 0 here too.
    """
    by_player = ratings.set_index("player_id")
    out = {}
    for side, column in (("offense", "o"), ("defense", "d")):
        kept = pairs[side]
        rating = by_player[column]
        gap = (rating.reindex(kept.fifth_a.to_numpy()).fillna(0.0).to_numpy()
               - rating.reindex(kept.fifth_b.to_numpy()).fillna(0.0).to_numpy())
        out[side] = kept.assign(r=kept.r.to_numpy(dtype=float) + gap)
    return out


def held_back_grams(held: Season, ratings: pd.DataFrame, inputs: dict, n_folds: int) -> dict:
    """{side: swap_adjust.PairGrams}: one fold's share of the line's normal equations.

    `held` is the fold's held-back games (swap_test.prepare), `ratings` the fold's step-4 ratings in raw sign
    (player_id, o, d), `inputs` {"O", "D"}: `line_inputs` from the fit games.  A swap counts only if both players
    have inputs, and the swaps of each player fold are kept apart so his own swaps stay out of his own line.
    """
    pairs = swap_targets(swap_adjust.residual_pairs(held, ratings, swap_adjust.RESIDUAL_CONTEXT), ratings)
    fold_of = swap_adjust.player_folds(np.union1d(inputs["O"].index.to_numpy(), inputs["D"].index.to_numpy()),
                                       n_folds)
    return {side: swap_adjust.pair_grams(pairs[side], inputs[key], fold_of, n_folds)
            for side, key in SIDE_KEYS.items()}


def season_coefficients(grams: dict, season: int, trainable: set, exclude_near: int, n_folds: int,
                        penalty: float, nearest: int = 0) -> dict | None:
    """{"O", "D"}: the rated season's line, an array (n_folds + 1, len(INPUTS)): row k fitted without player fold k's
    swaps, the last row on every swap.  None when no season may train it.

    `grams` is {season: {side: PairGrams}}; the line trains on the seasons more than `exclude_near` from the rated
    one that are trainable (swap_adjust.training_seasons: 0 in the product setting, 1 in the test setting).  With
    `nearest` = K, only the K of them closest to the rated season, the nearer first and on an equal distance the
    earlier, as step 4 picks its fold seasons (prior_shrink.fold_seasons_used; notes/DECISIONS.md, experiment 85).

    Example: rating 2020 in the test setting with nearest 8 trains on 2015-2018 and 2022-2025.
    """
    train = swap_adjust.training_seasons(season, sorted(grams), trainable, exclude_near)
    if nearest:
        train = sorted(sorted(train, key=lambda other: (abs(other - season), other))[:int(nearest)])
    if not train:
        return None
    return {key: np.vstack([swap_adjust.type_coefficients([grams[other][side] for other in train], fold, penalty)
                            for fold in [*range(n_folds), None]])
            for side, key in SIDE_KEYS.items()}


def second_priors(inputs: pd.DataFrame, coefficients: np.ndarray, n_folds: int) -> pd.Series:
    """player_id -> x' beta, each player's line the one fitted without his player fold (player_id % n_folds)."""
    fold = inputs.index.to_numpy(dtype=np.int64) % int(n_folds)
    return pd.Series(np.einsum("ij,ij->i", inputs.to_numpy(float), coefficients[fold]), index=inputs.index)


def team_possessions(stints: pd.DataFrame, side: str) -> pd.DataFrame:
    """(player_id, team, poss): each player's possessions for each team on one end of `stints`, "O" on offence and
    "D" on defence (swap_adjust.team_possessions' layout)."""
    five, team = (OFF, "team_off") if side == "O" else (DEF, "team_def")
    long = pd.DataFrame({"player_id": stints[five].to_numpy(dtype=np.int64).ravel(),
                         "team": np.repeat(stints[team].to_numpy(dtype=np.int64), 5),
                         "poss": np.repeat(stints.poss.to_numpy(dtype=float), 5)})
    return long.groupby(["player_id", "team"], as_index=False).poss.sum()


def hold_team_totals(line: pd.Series, anchor: pd.Series, team_poss: pd.DataFrame) -> pd.Series:
    """player_id -> the anchor plus the line's deviation from it, every team's players moved by one amount so each
    team's possession-weighted total is the anchor's (module docstring; the anchor is b x box-score prior).  A player
    without possessions on this end keeps the line.

    Example: one team, two players of 600 and 400 possessions whose line is +1.0 and 0.0 above their anchor: the
    team's mean deviation is 0.6, so they end +0.4 and -0.6 above it, 0.6 x 0.4 - 0.4 x 0.6 = 0, the gap of 1.0 kept.
    """
    anchor = anchor.reindex(line.index).astype(float)
    deviation = line - anchor
    ids, player = np.unique(team_poss.player_id.to_numpy(dtype=np.int64), return_inverse=True)
    teams, team = np.unique(team_poss.team.to_numpy(dtype=np.int64), return_inverse=True)
    possessions = np.zeros((len(teams), len(ids)))
    np.add.at(possessions, (np.asarray(team).ravel(), np.asarray(player).ravel()), team_poss.poss.to_numpy(float))
    mix = (possessions / np.where(possessions.sum(axis=0) > 0, possessions.sum(axis=0), 1.0)).T
    own = deviation.reindex(ids).fillna(0.0).to_numpy(float)
    shift = np.linalg.solve(possessions @ mix, possessions @ own)
    moved = pd.Series(own - mix @ shift, index=ids).reindex(line.index)
    return anchor + moved.fillna(deviation)


def hold_spread(line: pd.Series, anchor: pd.Series, team_poss: pd.DataFrame) -> tuple[pd.Series, float]:
    """(player_id -> the line with its spread inside teams held to the anchor's, the factor): step 5's
    `swap_adjust.hold_spread_within` on one season and side, each player on his main team, the team he had the most
    possessions for on this end, weighted by his possessions on this end.  Every main team's possession-weighted mean
    stays, and so does the order inside it; a player without possessions on this end keeps the line.

    Example: one team, two players of equal possessions whose anchor is +1 and -1 and whose line is +3 and -1: the
    team's mean is +1, the anchor's spread about its mean 1 and the line's 4, so the factor is 1/2: they end +2 and 0.
    """
    order = team_poss.sort_values(["player_id", "poss", "team"], ascending=[True, False, True])
    main = order.drop_duplicates("player_id").set_index("player_id").team
    possessions = team_poss.groupby("player_id").poss.sum()
    ids = line.index
    team = main.reindex(ids).to_numpy(dtype=float)
    weight = possessions.reindex(ids).fillna(0.0).to_numpy(float)
    held, factor = swap_adjust.hold_spread_within(line.to_numpy(float), anchor.reindex(ids).to_numpy(float), weight,
                                                  team)
    return pd.Series(held, index=ids), factor


def season_second_priors(stints: pd.DataFrame, answers: dict, box: dict, coefficients: dict,
                         n_folds: int, team_totals: str = "held",
                         types: pd.DataFrame | None = None, input_set: str | None = None) -> tuple[dict, dict]:
    """({"O", "D"}: {player_id: second prior}, {"O", "D"}: the inputs it came from) for one rated season.

    `answers` is {"O", "D"}: `booster_answers` for the players the context reads; `box` {"O", "D"}: {player_id:
    box-score prior} for the players step 3 rates, the step-2 values themselves; `coefficients`
    `season_coefficients`; `team_totals` one of `TEAM_TOTALS`, each team's total read on `stints`' possessions;
    `types`, with the line on `INPUT_SETS["context_and_types"]`, `standardised_types` of the same games; `input_set`
    the line's inputs, read from `types` when not given; a set of `CENTRED_SETS` is centred on the season (`centred`).
    """
    if team_totals not in TEAM_TOTALS:
        raise ValueError(f"team totals are one of {TEAM_TOTALS}, not {team_totals!r}")
    context = context_numbers(stints, answers["O"], answers["D"])
    values = {}
    inputs = side_inputs(box, context, types, input_set)
    if input_set in CENTRED_SETS:
        inputs = {side: centred(frame, team_possessions(stints, side).groupby("player_id").poss.sum())
                  for side, frame in inputs.items()}
    for side in ("O", "D"):
        line = second_priors(inputs[side], coefficients[side], n_folds)
        if team_totals in ("held", "spread_held"):
            # the line's own box-score part, b x box-score prior, b from the line his player fold reads
            weight = coefficients[side][inputs[side].index.to_numpy(dtype=np.int64) % int(n_folds), 0]
            anchor = pd.Series(weight * inputs[side][BOX].to_numpy(float), index=inputs[side].index)
            team_poss = team_possessions(stints, side)
            line = hold_team_totals(line, anchor, team_poss)
            if team_totals == "spread_held":
                line, _ = hold_spread(line, anchor, team_poss)
        values[side] = line.to_dict()
    return values, inputs
