"""Step 4 of 5: the box-score prior part of every rating multiplied down, by how much it holds up on unseen games.

Step 3 splits each rating into two parts: the prior part, s x prior (the box-score prior times the season's prior
scale), and the games' part, u (how far the season's own games moved the player from it).  Scored on games the
rating never saw, the two parts hold up differently.  Fit those games' points per 100 by weighted least squares on
the ten players' summed prior parts and summed games' parts, both coefficients free (1 = the part holds up
exactly): the prior part gets 0.70 on offence (0.64 for players with 2,500+ possessions) and 0.91 on defence, the
games' part 1.21 and 1.30 (notes/DECISIONS.md, experiment 33b).

The shipped step multiplies the prior part only, one multiplier per side and season, fitted the same way with the
games' part held at 1.  That gives the published multipliers, 0.699 to 0.715 on offence and 0.929 to 0.976 on
defence.  Multiplying the games' part as well tied on the year-over-year test (8.648 either way) and added nothing
within the season (z -1.1), so the simpler version ships.  The owner adopted it on 2026-10-03: the year-over-year
test's error fell from 8.666 to 8.600 points per 100, lower in 53 of its 56 predictions (each season 1998-2025
predicted from the season before and from the season after).

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

    prior <- m x prior
    shift  = sum(poss_off x (prior + u)) / sum(poss_off),   prior <- prior - shift,   rating = prior + u

The weight is the offensive possession count on both sides, as in step 3's centring (prior_rapm.centre).  The
re-centring keeps the possession-weighted mean rating at zero, and it moves the prior part only: u is untouched.

Example: rating 2026 in the product setting, m_O = 0.704 and m_D = 0.929.  Jokic's offensive prior part of 5.394
becomes 0.704 x 5.394 = 3.80, less the season's shift of 0.02 = 3.78; his games' part stays 0.97, so his
offensive rating goes from 6.37 to 4.75.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

__all__ = ["RULES", "unseen_fold_seasons", "fold_seasons_used", "fold_terms", "fit_multipliers",
           "multipliers_by_season", "shrink"]

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


def fold_terms(fold) -> tuple[np.ndarray, np.ndarray]:
    """One fold's (t, [P_O | P_D]): the points left for the prior parts to explain, and the two prior sums.

    `fold` is a `data.within_folds.Fold`.  The defence's parts enter in points allowed, so they are negated.
    """
    offence, defence = fold.rows["O"], fold.rows["D"]
    prior_offence = fold.Z["O"] @ offence.prior_off.to_numpy(float)
    games_offence = fold.Z["O"] @ offence.u_off.to_numpy(float)
    prior_defence = fold.Z["D"] @ (-defence.prior_def.to_numpy(float))
    games_defence = fold.Z["D"] @ (-defence.u_def.to_numpy(float))
    stand_in = fold.base - (prior_offence + games_offence) - (prior_defence + games_defence)
    return fold.y - stand_in - games_offence - games_defence, np.column_stack([prior_offence, prior_defence])


def fit_multipliers(folds: list) -> tuple[float, float]:
    """(m_O, m_D): the weighted least squares in the module docstring over the test team-games of `folds`.

    The design is [one block of controls per fold, block diagonal | P_O | P_D]; the normal equations are solved
    by least squares, because a fold without playoff games has an empty playoff column.
    """
    targets, prior_parts = zip(*(fold_terms(fold) for fold in folds))
    controls = sp.block_diag([sp.csr_matrix(fold.F) for fold in folds], format="csr")
    design = sp.hstack([controls, sp.csr_matrix(np.vstack(prior_parts))], format="csr")
    target = np.concatenate(targets)
    weight = np.concatenate([fold.w for fold in folds])
    gram = np.asarray((design.T @ design.multiply(weight[:, None])).todense())
    right_hand_side = np.asarray(design.T @ (weight * target)).ravel()
    multipliers = np.linalg.lstsq(gram, right_hand_side, rcond=None)[0][controls.shape[1]:]
    return float(multipliers[0]), float(multipliers[1])


def multipliers_by_season(seasons, folds: list, rule: str, nearest: int = 0) -> pd.DataFrame:
    """One row per season: season, multiplier_off, multiplier_def, fold_seasons (how many fold seasons fitted it).

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
            fits[used] = fit_multipliers([fold for fold in folds if fold.season in used])
        multiplier_off, multiplier_def = fits[used]
        rows.append(dict(season=int(season), multiplier_off=multiplier_off, multiplier_def=multiplier_def,
                         fold_seasons=len(used)))
    return pd.DataFrame(rows)


def _shrink_season(rows: pd.DataFrame, multiplier_off: float, multiplier_def: float) -> pd.DataFrame:
    """One season's rows with the prior parts multiplied and the ratings re-centred (the module docstring)."""
    rows = rows.copy()
    rows["prior_off"] = multiplier_off * rows.prior_off
    rows["prior_def"] = multiplier_def * rows.prior_def
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
    """Step 4 on a step-3 table: each season's prior parts times its multipliers, re-centred; u unchanged.

    `table` is in the published schema (prior_rapm.PUBLISHED_COLUMNS); `multipliers` is `multipliers_by_season`.
    The raw-sign columns (offense, defense, prior_offense, prior_defense) follow the published ones, and the
    columns keep the table's order.  Rows come out sorted by season, each season's rows in their own order.
    """
    by_season = multipliers.set_index("season")
    seasons = [_shrink_season(rows, by_season.at[season, "multiplier_off"], by_season.at[season, "multiplier_def"])
               for season, rows in table.groupby("season", sort=True)]
    return pd.concat(seasons, ignore_index=True)[list(table.columns)]
