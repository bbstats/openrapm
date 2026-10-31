"""Step 3 of 5: RAPM pulled toward the box-score prior, with the prior's weight set by the season's games.

Plain RAPM (step 1) pulls every player toward 0, the league average.  Here a player is pulled toward a
multiple of his prior instead (step 2), and the season's own games decide the multiple, one per side: a
player with few possessions stays close to s x prior, and a starter's games can move him away from it.  With
lambda = 13,037, a 500-possession player keeps about 500 / 13,537 = 4% of what his games say beyond the prior;
a 2,500-possession starter keeps about 2,500 / 15,537 = 16%.

Words used below
    prior           step 2's prediction for the player, one number per side
    lineup prior    the sum of the five priors on the court for a row: one column per side
    prior scale     s, the coefficient on a side's lineup prior; unpenalised, so the games set it
    deviation       r, how far the season's games move a player from s x prior: one coefficient per player per
                    side (a coefficient, not a regression residual), penalised like a plain RAPM coefficient
    centring        a shift per season and side so that the average possession is played by a 0 player

The fit, for one season and one target:

    minimise   sum over rows  w (y - Z r - C gamma - s_O P_O - s_D P_D)^2  +  lambda (|r_O|^2 + |r_D|^2)

    y, w      the row's target (adjusted points per 100) and weight (its possessions), as in step 1
    Z, C      the player columns and the context columns (step 1); gamma the context coefficients (unpenalised)
    P_O, P_D  the lineup priors: the offence's five offensive priors summed, the defence's five defensive
              priors summed
    s_O, s_D  the prior scales (unpenalised); r_O, r_D the deviations, offence and defence (penalised)
    lambda    13,037 (settings.RidgeSettings)

    rating = s x prior + r,   reported prior = s x prior,   u = rating - reported prior = r   (per side)

Two fits run per season: the offence half of the table comes from the fit on the offence target, the defence
half from the fit on the defence target (step 1 defines both, "points_with_expected_free_throws" and
"points_with_expected_opponent_threes").

Example: 2026's offensive prior scale is 2.35 (product setting).  A player whose booster prior is +0.5 per 100
starts at 2.35 x 0.5 = +1.18.  If his 2,500 possessions say he was 2.5 per 100 better than that, he keeps about
16% of it, r = +0.4, for a rating of +1.58 before centring.

Why s is well above 1 on offence: the offensive label is a ridge coefficient that is not un-shrunk (step 1), so it
keeps only n / (n + 40,000) of a player's raw impact, and the prior learned from it is on that shrunk scale; s puts
it back on the season's scale.  The defensive label is un-shrunk, and its scale stays near 1.  Over the thirty
published seasons s runs from 2.32 to 4.47 on offence and from 0.93 to 1.39 on defence; with both labels un-shrunk
(notes/DECISIONS.md, experiment 1 of 2026-09-14 evening) the scales read 0.84 and 0.92.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .rapm import design_blocks, normal_equations, player_possessions, solve_ridge

__all__ = ["aligned_prior", "lineup_prior_sums", "game_lineup_sums", "game_averages", "PriorRapmFit", "RidgeSystem",
           "ridge_system", "fit", "defence_possessions", "season_table", "centre", "rate_season", "held_prior_scale",
           "blend_weight", "blended_scale", "rate_season_blended", "PUBLISHED_COLUMNS", "published_table"]


def aligned_prior(prior: dict, player_ids) -> np.ndarray:
    """A {player_id: prior} as a vector in the design's player order; a player with no prior gets 0."""
    return pd.Series(prior).reindex(player_ids).fillna(0.0).to_numpy(dtype=float)


def lineup_prior_sums(players, prior: np.ndarray, n_players: int):
    """(P_O, P_D) per row: the offence's five priors summed and the defence's five priors summed.

    `prior` is [offence priors | defence priors] in the design's player order.
    """
    return players[:, :n_players] @ prior[:n_players], players[:, n_players:] @ prior[n_players:]


def game_lineup_sums(players, game: np.ndarray, tables: tuple, n_players: int):
    """(P_O, P_D) per row from priors that change game by game (game_prior.game_tables: per side, a games x players
    table): the five attackers' priors at the row's game summed, and the five defenders'.  The product Z x prior done
    on the nonzeros of each side's player columns, each gathered from the table at its row's game."""
    out = []
    for block, table in ((players[:, :n_players], tables[0]), (players[:, n_players:], tables[1])):
        cells = block.tocoo()
        values = cells.data * table[game[cells.row], cells.col]
        out.append(np.bincount(cells.row, weights=values, minlength=block.shape[0]))
    return out[0], out[1]


def game_averages(players, game: np.ndarray, weights: np.ndarray, tables: tuple, n_players: int) -> np.ndarray:
    """[offence | defence] in the design's player order: each player's game priors averaged over his rows on that side
    by the rows' weights, his possessions; the prior his season value reports (game_prior.py), since each of his
    rows credits him the prior scale times that game's prior plus his deviation.  A player with no rows on a side
    reads his table's mean over the games."""
    out = []
    for block, table in ((players[:, :n_players], tables[0]), (players[:, n_players:], tables[1])):
        cells = block.tocoo()
        weight = weights[cells.row] * cells.data
        total = np.bincount(cells.col, weights=weight * table[game[cells.row], cells.col], minlength=n_players)
        mass = np.bincount(cells.col, weights=weight, minlength=n_players)
        out.append(np.where(mass > 0, total / np.where(mass > 0, mass, 1.0), table.mean(axis=0)))
    return np.concatenate(out)


@dataclass
class PriorRapmFit:
    """One season's fit on one target."""
    ratings: pd.DataFrame        # player_id, offense, defense, prior_offense, prior_defense, possessions
    prior_scale: np.ndarray      # (s_O, s_D)
    deviation: np.ndarray        # r, [offence | defence]
    unpenalised: np.ndarray      # gamma (the context coefficients), then s_O, s_D
    penalty: float


@dataclass
class RidgeSystem:
    """One season's step-3 ridge as a linear system, before it is solved.

    The columns are [players | context | P_O | P_D]; `players` is Z (rows x 2 n_players, sparse) and `unpenalised`
    holds the rest, [C | P_O | P_D] (rows x (context + 2)).  The solve is (gram + diag(penalties)) theta = rhs.
    """
    players: sp.csr_matrix
    unpenalised: np.ndarray
    prior: np.ndarray            # [offence priors | defence priors], in the design's player order
    gram: np.ndarray
    rhs: np.ndarray
    penalties: np.ndarray


def ridge_system(design, prior_offense: dict, prior_defense: dict, penalty: float = 13037.0,
                 scale=None, game_tables: tuple | None = None) -> RidgeSystem:
    """The linear system of the fit below: the lineup priors join the context as two more unpenalised columns, so
    the solve is a plain ridge on [players | context | P_O | P_D] with penalty lambda on the players and 0 on
    everything else.  With `scale`, (s_O, s_D) held at those values instead of fitted (`held_prior_scale`): the two
    columns times their scales come off the target, and the system is [players | context].

    With `game_tables`, the priors change game by game (game_prior.py): each row's lineup prior reads its own game's
    (`game_lineup_sums`), `prior_offense` and `prior_defense` are not read, and the prior the system carries is each
    player's average over his rows (`game_averages`)."""
    n_players = design.spec.n_players
    player_ids = design.spec.players["player_id"].to_numpy()
    players, context = design_blocks(design)
    if game_tables is None:
        prior = np.concatenate([aligned_prior(prior_offense, player_ids), aligned_prior(prior_defense, player_ids)])
        sums_o, sums_d = lineup_prior_sums(players, prior, n_players)
    else:
        game = design.rows["game_idx"].to_numpy()
        sums_o, sums_d = game_lineup_sums(players, game, game_tables, n_players)
        prior = game_averages(players, game, np.asarray(design.w, dtype=float), game_tables, n_players)
    target = design.y
    if scale is None:
        context = np.column_stack([context, sums_o, sums_d])
    else:
        target = design.y - float(scale[0]) * sums_o - float(scale[1]) * sums_d
    sums = normal_equations(players, context, target, design.w)
    gram = np.block([[sums.gram_players, sums.gram_players_context],
                     [sums.gram_players_context.T, sums.gram_context]])
    rhs = np.concatenate([sums.rhs_players, sums.rhs_context])
    penalties = np.concatenate([np.full(n_players, penalty), np.full(n_players, penalty),
                                np.zeros(context.shape[1])])
    return RidgeSystem(players=players, unpenalised=context, prior=prior, gram=gram, rhs=rhs, penalties=penalties)


def fit(design, prior_offense: dict, prior_defense: dict, penalty: float = 13037.0, scale=None,
        game_tables: tuple | None = None) -> PriorRapmFit:
    """The prior-informed ridge on one season's design (the equation in the module docstring), the system of
    `ridge_system` solved; with `scale`, the prior scales held at (s_O, s_D) rather than fitted; with `game_tables`,
    priors that change game by game, the reported prior the scale times each player's average."""
    n_players = design.spec.n_players
    player_ids = design.spec.players["player_id"].to_numpy()
    system = ridge_system(design, prior_offense, prior_defense, penalty, scale, game_tables)
    players, prior = system.players, system.prior
    coefficients = solve_ridge(system.gram, system.rhs, system.penalties)
    deviation, unpenalised = coefficients[:2 * n_players], coefficients[2 * n_players:]
    if scale is None:
        scale = unpenalised[-2:].copy()
    else:
        scale = np.asarray(scale, dtype=float)
        unpenalised = np.concatenate([unpenalised, scale])
    scaled_prior = np.concatenate([scale[0] * prior[:n_players], scale[1] * prior[n_players:]])
    ratings = pd.DataFrame({
        "player_id": player_ids,
        "offense": scaled_prior[:n_players] + deviation[:n_players],
        "defense": scaled_prior[n_players:] + deviation[n_players:],
        "prior_offense": scaled_prior[:n_players],
        "prior_defense": scaled_prior[n_players:],
        "possessions": player_possessions(players, design.rows["poss"].to_numpy(), n_players),
    })
    return PriorRapmFit(ratings=ratings, prior_scale=scale, deviation=deviation, unpenalised=unpenalised,
                        penalty=float(penalty))


def defence_possessions(design) -> dict:
    """{player_id: defensive possessions} for one season's design: the table's `poss_def`.

    `poss_off` comes from the stint rows (`rapm.player_possessions`), `poss_def` from the design's per-game
    possession table (`design.game_poss`).  The two counts differ by about nine possessions in 2,500 (2026: a
    mean gap of 9.5 among players with 2,000 to 3,000), and the owner asked for the right count on each side
    (2026-09-18; before then `poss_def` was a copy of `poss_off`).  The game table holds one row per player-season
    and game (design.py's `player_season`: a player traded mid-season still has one player-season); the rows are
    summed over games here, and over seasons when a design spans several.
    """
    per_entry = (design.game_poss.groupby("player_season")["poss_def"].sum()
                 .reindex(range(design.spec.n_player_seasons)).fillna(0.0))
    player_seasons = design.spec.player_seasons
    defence = per_entry.to_numpy()[player_seasons["player_season"].to_numpy()]
    return pd.Series(defence).groupby(player_seasons["player_id"].to_numpy()).sum().to_dict()


def season_table(offence_fit: PriorRapmFit, defence_fit: PriorRapmFit, poss_def: dict,
                 season: int) -> pd.DataFrame:
    """One season's rows in raw sign (defence = points allowed): offence from its fit, defence from its own.

    `poss_def` is `defence_possessions` of the season's design.  Columns, in order: player_id, offense,
    prior_offense, possessions, defense, prior_defense, poss_def, season, offense_lambda, defense_lambda,
    context_lambda, prior_scale_off, prior_scale_def.
    """
    table = (offence_fit.ratings[["player_id", "offense", "prior_offense", "possessions"]]
             .merge(defence_fit.ratings[["player_id", "defense", "prior_defense"]], on="player_id"))
    table["poss_def"] = table.player_id.map(poss_def).fillna(0.0).to_numpy(float)
    return table.assign(season=season, offense_lambda=offence_fit.penalty, defense_lambda=defence_fit.penalty,
                        context_lambda=0.0, prior_scale_off=offence_fit.prior_scale[0],
                        prior_scale_def=defence_fit.prior_scale[1])


def centre(table: pd.DataFrame) -> pd.DataFrame:
    """Shift each season's ratings, per side, so their mean weighted by possessions is 0; the prior moves with them.

        shift  = sum(possessions x rating) / sum(possessions),  over the season's players
        rating = rating - shift,   prior = prior - shift

    `possessions` is the offensive count (step 1's `player_possessions`) on both sides.

    The prior's weighted mean is not 0 on its own (the prior scale multiplies a label mean that heavy-minute
    players pull positive), and an average possession should be played by a 0 player (the owner, 2026-09-14).
    A level shift only: u = rating - prior does not move.

    Example: if a season's possession-weighted mean offensive rating is +1.2, every player's offensive rating
    and offensive prior drop by 1.2 that season.
    """
    table = table.copy()
    for side in ("offense", "defense"):
        level = (table.groupby("season").apply(
            lambda season_rows, column=side: np.average(season_rows[column], weights=season_rows.possessions),
            include_groups=False)
            .reindex(table.season).to_numpy())
        table[side] = table[side] - level
        table[f"prior_{side}"] = table[f"prior_{side}"] - level
    return table


def rate_season(designs: dict, priors: dict, settings, season: int, scale=None,
                game_tables: dict | None = None) -> pd.DataFrame:
    """Step 3 for one season: both fits, the season's table, centred.

    `designs` maps each target name to the season's design on it; `priors` is {"O": {player_id: prior},
    "D": {...}} from step 2.  One fit per distinct target.  `scale`, (s_O, s_D), holds the prior scales in both fits
    (`held_prior_scale`); None fits them on the season's games, as shipped.  `game_tables`, {target: (table_O,
    table_D)} (game_prior.game_tables on that target's design), pulls each row toward its own game's priors.
    """
    targets = settings.targets
    fits = {name: fit(designs[name], priors["O"], priors["D"], settings.ridge.penalty, scale,
                      None if game_tables is None else game_tables[name])
            for name in targets.distinct()}
    defence = defence_possessions(designs[targets.offence])
    return centre(season_table(fits[targets.offence], fits[targets.defence], defence, season))


HELD_SCALE_SEASONS = 8      # the seasons a held prior scale is the mean of, as step 4's multipliers take 8


def held_prior_scale(table: pd.DataFrame, season: int, nearest: int = HELD_SCALE_SEASONS) -> tuple:
    """(s_O, s_D): the mean prior scales of the `nearest` seasons of `table` closest to `season`, never `season`
    itself (the nearer first; at an equal distance, the earlier), each season's scales as its whole-season fit set
    them.  Experiment 109: a season's first days hold the prior scale at this value, since a scale fitted on a few
    games swings (2027 on 16 games, 2.24 on offense; 2026 on its whole season, 1.17).

    Example: for 2010, the mean of 2006-2009 and 2011-2014."""
    scales = table.groupby("season")[["prior_scale_off", "prior_scale_def"]].first()
    others = sorted((int(s) for s in scales.index if int(s) != int(season)),
                    key=lambda s: (abs(s - int(season)), s))[:nearest]
    if not others:
        raise ValueError(f"no season besides {season} to hold its prior scale at")
    chosen = scales.loc[others]
    return float(chosen.prior_scale_off.mean()), float(chosen.prior_scale_def.mean())


def blend_weight(share: float, through: float, start: float = 0.0) -> float:
    """The weight on a season so far's own fitted prior scales against the held ones (`held_prior_scale`): 0 until
    `start` of its regular season, 1 from `through` on, a straight line between (the owner, 2026-10-31: "If it's just
    two different shrinkages we should probably just blend between them").  `through` at or below `start` is a switch.

    Example: start 0.25 and through 0.75, half the season played weighs the fitted scales a half.  From the first game
    (start 0) the line lost to holding at a tenth and a quarter of a season in every era (experiment 109 blended)."""
    if through <= start:
        return 1.0 if float(share) >= through else 0.0
    return min(max((float(share) - float(start)) / (float(through) - float(start)), 0.0), 1.0)


def blended_scale(held, fitted, weight: float) -> tuple:
    """(s_O, s_D) `weight` of the way from `held` to `fitted`."""
    return tuple((1.0 - float(weight)) * float(h) + float(weight) * float(f) for h, f in zip(held, fitted))


def rate_season_blended(designs: dict, priors: dict, settings, season: int, held=None, weight: float = 0.0,
                        game_tables: dict | None = None) -> pd.DataFrame:
    """`rate_season` with the prior scales `weight` of the way from `held` to the season's own fitted ones: the scales
    fitted first, then held at the blend (`blended_scale`).  Weight 0 is the held fit, weight 1 or no `held` the plain
    one, each through `rate_season` alone."""
    if held is None or weight >= 1.0:
        return rate_season(designs, priors, settings, season, None, game_tables)
    if weight <= 0.0:
        return rate_season(designs, priors, settings, season, held, game_tables)
    free = rate_season(designs, priors, settings, season, None, game_tables)
    fitted = (float(free.prior_scale_off.iloc[0]), float(free.prior_scale_def.iloc[0]))
    return rate_season(designs, priors, settings, season, blended_scale(held, fitted, weight), game_tables)


# the published schema, positive good on both ends (`openrapm site` and the checks read it)
PUBLISHED_COLUMNS = ["player_id", "offense", "prior_offense", "poss_off", "defense", "prior_defense", "poss_def",
                     "season", "offense_lambda", "defense_lambda", "context_lambda", "prior_scale_off",
                     "prior_scale_def", "poss_season", "rating_off", "rating_def", "rating_total", "prior_off",
                     "prior_def", "prior_total", "u_off", "u_def", "u_total", "player_name"]


def published_table(table: pd.DataFrame, names: pd.DataFrame) -> pd.DataFrame:
    """The centred seasons in the published schema: ratings positive-good on both ends, plus player names.

        rating_def = -defense      prior_def = -prior_defense      u = rating - prior,  per side
        *_total    = offence + defence

    `names` is (player_id, player_name), one row per player.  The raw-sign columns stay beside the published
    ones because later steps read them.
    """
    out = table.rename(columns={"possessions": "poss_off"})
    out["poss_season"] = out.poss_off
    out["rating_off"] = out.offense
    out["rating_def"] = -out.defense
    out["rating_total"] = out.rating_off + out.rating_def
    out["prior_off"] = out.prior_offense
    out["prior_def"] = -out.prior_defense
    out["prior_total"] = out.prior_off + out.prior_def
    out["u_off"] = out.rating_off - out.prior_off
    out["u_def"] = out.rating_def - out.prior_def
    out["u_total"] = out.u_off + out.u_def
    return out.merge(names, on="player_id", how="left")[PUBLISHED_COLUMNS]
