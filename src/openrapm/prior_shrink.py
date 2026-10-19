"""Step 4 of 5: the two parts of every rating multiplied by how much each holds up on unseen games.

Step 3 splits each rating into two parts: the prior part, s x prior (the box-score prior times the season's prior
scale), and the games' part, u (how far the season's own games moved the player from it).  Scored on games the
rating never saw, the two parts hold up differently.  Fit those games' points per 100 by weighted least squares on
the ten players' summed prior parts and summed games' parts, both coefficients free (1 = the part holds up
exactly): the prior part gets 0.70 on offence (0.64 for players with 2,500+ possessions) and 0.91 on defence, the
games' part 1.21 and 1.30 (notes/DECISIONS.md, experiment 33b).

From 2026-10-03 the shipped step multiplied the prior part only, one multiplier per side and season, fitted the same
way with the games' part held at 1: multiplying the games' part as well tied on the year-over-year test (8.648 either
way) and added nothing on the team-game held-out test (z -1.1), so the simpler version shipped.

Since experiment 65 (adopted 2026-10-15, the owner: "Adopt one multiplier as the new incumbent") both parts are
multiplied (`free_games=True`, settings.ShrinkSettings.free_games_part): the games' part's multipliers g_O and g_D
are fitted in the same least squares with U_O and U_D as free columns and t_g = y_g - S_g.  The with-or-without-you
test (scripts/checks/151_wowy.py), which scores how a team's results move as its lineup changes, read it at -0.648,
z -17.5, better in 30 of 30 seasons, with g about 1.33 on offence and 1.48 on defence; the year-over-year test a tie.
Then u <- g x u before the re-centring, which still moves the prior part only.  `free_games=False` is the step as it
was, and the module's defaults keep it so.

Example, from before experiment 65: rating 2026 in the product setting, m_O = 0.704 and m_D = 0.929.  Jokic's
offensive prior part of 5.394 becomes 0.704 x 5.394 = 3.80, less the season's shift of 0.02 = 3.78; his games' part
stays 0.97, so his offensive rating goes from 6.37 to 4.75.  With g_O = 1.33 his games' part would be 1.29.

Words used below
    within-season fold
                    one season rated from three quarters of its games (the fit games), and scored on the
                    other quarter (the test games), which the rating never saw.  Each season's games are
                    assigned at random to four folds, three times over: twelve folds a season (`openrapm folds`,
                    data/within_folds.py)
    test team-game  one team's offence in one test game: its points per 100 possessions as scored, its
                    possessions, and each player's share of the possessions on both ends
    rated player    a player with offensive possessions in the fold's fit games, so with a rating of his own
    stand-in rating what every other player in the test games carries: a quarter of the possession-weighted
                    mean rating of the fold's players with under 500 possessions (data/within_folds.py)
    prior part      s x prior from the fold's own rating (`prior_off`, `prior_def`)
    games' part     u from the fold's own rating (`u_off`, `u_def`)
    fold seasons    the seasons whose folds may be used: 2017-2026, pinned (settings.ShrinkSettings), because
                    the fold folder grew to 1997-2026 on 2026-10-04 and an unpinned rerun shrank harder than
                    what ships (notes/DECISIONS.md, experiment 34's correction)
    multiplier      m_O, m_D: what the prior part is multiplied by, offence and defence

The multipliers are one weighted least-squares fit over the test team-games of every fold used:

    minimise   sum over test team-games g   w_g (t_g - F_g gamma_f - m_O P_O,g - m_D P_D,g)^2
    where      t_g = y_g - S_g - U_O,g - U_D,g

    y_g, w_g      the team-game's points per 100 possessions as scored, and its possessions
    P_O,g         sum over the offence's rated players of share x prior part (positive good)
    P_D,g         sum over the defence's rated players of share x (-prior_def): the prior part in points allowed
    U_O,g, U_D,g  the same two sums of the games' parts, held at their own values (a multiplier of 1)
    S_g           what the players without a rating in the fold add, at the fold's stand-in rating
    F_g, gamma_f  the fold's controls (home, the season's intercept, playoffs, playoff home) and their
                  coefficients, free for each fold f: every fold has its own level and home edge

Which folds a season's multipliers are fitted on:

    product rule   every fold season except the rated one, so a season's own games never set its multipliers.
                   Seasons 1997-2016 have no folds, so each borrows the fit on all ten fold seasons
    test rule      every fold season except the rated one and its two neighbours, because the year-over-year
                   test scores a season's ratings on its neighbours' games (1997-2015 borrow all ten; 2016 uses
                   2018-2026)

Then, for each season and side (`shrink`):

    prior <- m x prior,   u <- g x u   (g = 1 when the games' part is held)
    shift  = sum(poss_off x (prior + u)) / sum(poss_off),   prior <- prior - shift,   rating = prior + u

The weight is the offensive possession count on both sides, as in step 3's centring (prior_rapm.centre).  The
re-centring keeps the possession-weighted mean rating at zero, and it moves the prior part only.

Example: rating 2026 in the product setting, m_O = 0.704 and m_D = 0.929.  Jokic's offensive prior part of 5.394
becomes 0.704 x 5.394 = 3.80, less the season's shift of 0.02 = 3.78; his games' part stays 0.97, so his
offensive rating goes from 6.37 to 4.75.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

__all__ = ["RULES", "unseen_fold_seasons", "fold_seasons_used", "fold_terms", "fit_multipliers",
           "multipliers_by_season", "shrink", "season_factors", "shrink_fold"]

RULES = ("product", "test")


def unseen_fold_seasons(season: int, rule: str) -> set:
    """The seasons whose folds may not fit `season`'s multipliers.

    Example: 2020 gives {2020} under the product rule and {2019, 2020, 2021} under the test rule.
    """
    if rule not in RULES:
        raise ValueError(f"rule must be one of {RULES}, got {rule!r}")
    return {season} if rule == "product" else {season - 1, season, season + 1}


def fold_seasons_used(season: int, fold_seasons, rule: str, nearest: int = 0) -> tuple:
    """The fold seasons that fit `season`'s multipliers, ascending: every fold season the rule allows, or with
    `nearest` = K only the K of them closest to the rated season (the nearer first; an equal distance, the earlier).

    `nearest` is the era-drift candidate (notes/DECISIONS.md, "Era drift" and experiment 55): the defensive prior part
    holds up less in older seasons (0.79 in 1997-2006, 0.95 in 2017-2026), so a season's multipliers come from the
    seasons around it rather than from 2017-2026 for all.

    Example: with fold seasons 2017-2026, 2010 uses all ten under either rule, and 2016 under the test rule
    uses 2018-2026.  With fold seasons 1997-2026, the test rule and nearest = 8, 2010 uses 2005-2008 and 2012-2015.
    """
    unseen = unseen_fold_seasons(int(season), rule)
    allowed = [fold_season for fold_season in sorted(fold_seasons) if fold_season not in unseen]
    if nearest:
        allowed = sorted(allowed, key=lambda fold_season: (abs(fold_season - int(season)), fold_season))[:nearest]
    return tuple(sorted(allowed))


def fold_terms(fold, free_games: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """One fold's (t, [P_O | P_D]): the points left for the prior parts to explain, and the two prior sums.  With
    `free_games`, (t, [P_O | P_D | U_O | U_D]): the games' parts are columns of their own, not taken off t.

    `fold` is a `data.within_folds.Fold`.  The defence's parts enter in points allowed, so they are negated.
    """
    offence, defence = fold.rows["O"], fold.rows["D"]
    prior_offence = fold.Z["O"] @ offence.prior_off.to_numpy(float)
    games_offence = fold.Z["O"] @ offence.u_off.to_numpy(float)
    prior_defence = fold.Z["D"] @ (-defence.prior_def.to_numpy(float))
    games_defence = fold.Z["D"] @ (-defence.u_def.to_numpy(float))
    stand_in = fold.base - (prior_offence + games_offence) - (prior_defence + games_defence)
    if free_games:
        return fold.y - stand_in, np.column_stack([prior_offence, prior_defence, games_offence, games_defence])
    return fold.y - stand_in - games_offence - games_defence, np.column_stack([prior_offence, prior_defence])


def fit_multipliers(folds: list, free_games: bool = False) -> tuple:
    """(m_O, m_D): the weighted least squares in the module docstring over the test team-games of `folds`; with
    `free_games`, (m_O, m_D, g_O, g_D), the games' parts' multipliers fitted beside them.

    The design is [one block of controls per fold, block diagonal | P_O | P_D]; the normal equations are solved
    by least squares, because a fold without playoff games has an empty playoff column.
    """
    targets, prior_parts = zip(*(fold_terms(fold, free_games) for fold in folds))
    controls = sp.block_diag([sp.csr_matrix(fold.F) for fold in folds], format="csr")
    design = sp.hstack([controls, sp.csr_matrix(np.vstack(prior_parts))], format="csr")
    target = np.concatenate(targets)
    weight = np.concatenate([fold.w for fold in folds])
    gram = np.asarray((design.T @ design.multiply(weight[:, None])).todense())
    right_hand_side = np.asarray(design.T @ (weight * target)).ravel()
    multipliers = np.linalg.lstsq(gram, right_hand_side, rcond=None)[0][controls.shape[1]:]
    return tuple(float(value) for value in multipliers)


def multipliers_by_season(seasons, folds: list, rule: str, nearest: int = 0, free_games: bool = False) -> pd.DataFrame:
    """One row per season: season, multiplier_off, multiplier_def, fold_seasons (how many fold seasons fitted it);
    with `free_games` also games_multiplier_off and games_multiplier_def.

    A fit depends only on the folds it is handed, so it runs once per distinct set of fold seasons and is reused:
    under the product rule the twenty seasons 1997-2016 share one fit.  Reusing it gives the same bits as twenty
    refits, because each refit would hand the same folds, in the same order, to the same arithmetic.
    """
    fold_seasons = sorted({fold.season for fold in folds})
    fits: dict = {}
    rows = []
    for season in sorted(seasons):
        used = fold_seasons_used(season, fold_seasons, rule, nearest)
        if used not in fits:
            chosen = [fold for fold in folds if fold.season in used]
            fits[used] = fit_multipliers(chosen, free_games=True) if free_games else fit_multipliers(chosen)
        row = dict(season=int(season), multiplier_off=fits[used][0], multiplier_def=fits[used][1])
        if free_games:
            row.update(games_multiplier_off=fits[used][2], games_multiplier_def=fits[used][3])
        rows.append(dict(row, fold_seasons=len(used)))
    return pd.DataFrame(rows)


def _shrink_season(rows: pd.DataFrame, multiplier_off: float, multiplier_def: float, games_off: float = 1.0,
                   games_def: float = 1.0) -> pd.DataFrame:
    """One season's rows with the prior parts multiplied (and the games' parts, when freed) and the ratings
    re-centred (the module docstring)."""
    rows = rows.copy()
    rows["prior_off"] = multiplier_off * rows.prior_off
    rows["prior_def"] = multiplier_def * rows.prior_def
    if games_off != 1.0 or games_def != 1.0:
        rows["u_off"] = games_off * rows.u_off
        rows["u_def"] = games_def * rows.u_def
    weight = rows.poss_off.to_numpy(float)
    for side in ("off", "def"):
        shift = np.average(rows[f"prior_{side}"] + rows[f"u_{side}"], weights=weight)
        rows[f"prior_{side}"] -= shift
        rows[f"rating_{side}"] = rows[f"prior_{side}"] + rows[f"u_{side}"]
    rows["rating_total"] = rows.rating_off + rows.rating_def
    rows["prior_total"], rows["u_total"] = rows.prior_off + rows.prior_def, rows.u_off + rows.u_def
    rows["offense"], rows["defense"] = rows.rating_off, -rows.rating_def
    rows["prior_offense"], rows["prior_defense"] = rows.prior_off, -rows.prior_def
    return rows


def shrink(table: pd.DataFrame, multipliers: pd.DataFrame) -> pd.DataFrame:
    """Step 4 on a step-3 table: each season's prior parts times its multipliers, re-centred; u unchanged unless
    `multipliers` carries games_multiplier_off and games_multiplier_def (the games' part freed).

    `table` is in the published schema (prior_rapm.PUBLISHED_COLUMNS); `multipliers` is `multipliers_by_season`.
    The raw-sign columns (offense, defense, prior_offense, prior_defense) follow the published ones, and the
    columns keep the table's order.  Rows come out sorted by season, each season's rows in their own order.
    """
    by_season = multipliers.set_index("season")
    freed = "games_multiplier_off" in by_season.columns

    def games(season, side):
        return by_season.at[season, f"games_multiplier_{side}"] if freed else 1.0
    seasons = [_shrink_season(rows, by_season.at[season, "multiplier_off"], by_season.at[season, "multiplier_def"],
                              games(season, "off"), games(season, "def"))
               for season, rows in table.groupby("season", sort=True)]
    return pd.concat(seasons, ignore_index=True)[list(table.columns)]


def season_factors(multipliers: pd.DataFrame, season: int) -> tuple:
    """(m_O, m_D, g_O, g_D): one season's row of `multipliers_by_season`, as the build applies it (the defence factor
    already in it), 1 for a games' part the table does not free."""
    row = multipliers.set_index("season").loc[int(season)]
    return (float(row.multiplier_off), float(row.multiplier_def), float(row.get("games_multiplier_off", 1.0)),
            float(row.get("games_multiplier_def", 1.0)))


def _shrink_side(prior: np.ndarray, games: np.ndarray, multiplier: float, games_multiplier: float,
                 weight: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """(prior part, games' part, the level taken off) after step 4 on one side of a fold."""
    games = games_multiplier * games
    level = np.average(multiplier * prior + games, weights=weight)
    return multiplier * prior - level, games, float(level)


def shrink_fold(rows: dict, season: int, factors: tuple) -> pd.DataFrame:
    """Step 4 on one within-season fold: one row per rated player, positive good, in the published schema's rating
    columns (player_id, season, prior_off, u_off, prior_def, u_def, rating_off, rating_def, rating_total, offense,
    defense, poss_off, poss_def).

    `rows` is the fold's {"O", "D"} rated players (data/within_folds.Fold.rows) and `factors` `season_factors`.  Each
    side is re-centred over its own rows, weighted by offensive possessions, as the with-or-without-you test applies
    step 4 to a fold (scripts/checks/151_wowy.py).  A player with offensive possessions and none on defence has no
    defence row (2015, repeat 0, fold 2: one possession); his defence parts come from his offence row, shrunk as
    the defence rows were, and his defensive possessions are 0, so step 5 weighs him at nothing there.
    """
    offence, defence = rows["O"], rows["D"]
    prior_off, u_off, _ = _shrink_side(offence.prior_off.to_numpy(float), offence.u_off.to_numpy(float), factors[0],
                                       factors[2], offence.poss_off.to_numpy(float))
    prior_def, u_def, level = _shrink_side(defence.prior_def.to_numpy(float), defence.u_def.to_numpy(float),
                                           factors[1], factors[3], defence.poss_off.to_numpy(float))
    table = pd.DataFrame({"player_id": offence.player_id.to_numpy(np.int64), "season": np.int64(season),
                          "prior_off": prior_off, "u_off": u_off, "poss_off": offence.poss_off.to_numpy(float),
                          "poss_def": offence.poss_def.to_numpy(float)})
    table = table.join(pd.DataFrame({"prior_def": prior_def, "u_def": u_def},
                                    index=defence.player_id.to_numpy(np.int64)), on="player_id")
    alone = table.prior_def.isna().to_numpy()
    table.loc[alone, "prior_def"] = factors[1] * offence.prior_def.to_numpy(float)[alone] - level
    table.loc[alone, "u_def"] = factors[3] * offence.u_def.to_numpy(float)[alone]
    table["rating_off"] = table.prior_off + table.u_off
    table["rating_def"] = table.prior_def + table.u_def
    table["rating_total"] = table.rating_off + table.rating_def
    table["offense"], table["defense"] = table.rating_off, -table.rating_def
    return table
