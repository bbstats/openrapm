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

Since experiment 106 (adopted 2026-10-23, the owner: "please adopt, yes - D") the step has two layers
(settings.ShrinkSettings.within_team_layer).  Each of the four parts (prior and games' part, offence and defence) is
split into his team's possession-weighted mean and his deviation from it (`team_split`: the team is his main team,
the one he played the most possessions for; the weight his possessions on that side).  The team means keep the
multipliers above, fitted on held-back team-games.  The deviations get four multipliers of their own, fitted on the
held-back STINTS with a free level per team-game (`stint_equations`), so only how a team's lineups differ inside a
game sets them: on team-games, minutes that react to how a lineup is doing pull the within-team split down
(notes/DECISIONS.md, "Why the shrink cuts the offense: the unit it is fitted on").  The within layer is then
multiplied by settings.ShrinkSettings.within_factors, 0.75 on offence and 1.0 on defence (version D), and by the
defence factor as the team layer is.

Example, the within layer, 2017-2026 means before any factor: the deviation of the offensive prior part kept 0.97
(the team layer 0.66), of the offensive games' part 1.32 (1.38); defence 0.84 and 2.06 (0.93 and 1.56).  A star whose
offensive prior part is +5.0 on a team whose mean is +0.4 gets 0.66 x 0.4 + 0.75 x 0.97 x 4.6 = 3.61 before the
re-centring, against 0.66 x 5.0 = 3.30 with one multiplier.

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
           "multipliers_by_season", "held_back_share", "read_to_full_season", "full_season_multipliers",
           "team_split", "stint_equations", "within_multipliers_by_season", "shrink", "season_factors",
           "layered_fold_parts", "shrink_fold"]

RULES = ("product", "test")
# the within layer's four multipliers (experiment 106), in this order: offence prior part, offence games' part,
# defence prior part, defence games' part
WITHIN_COLUMNS = ("within_multiplier_off", "within_games_multiplier_off", "within_multiplier_def",
                  "within_games_multiplier_def")
MULTIPLIER_COLUMNS = ("multiplier_off", "multiplier_def", "games_multiplier_off", "games_multiplier_def",
                      *WITHIN_COLUMNS)
# each part of the within layer: its column, the possessions that weigh its team mean, and its sign in a team-game
# (the defence in points allowed)
LAYER_PARTS = (("prior_off", "poss_off", 1.0), ("u_off", "poss_off", 1.0), ("prior_def", "poss_def", -1.0),
               ("u_def", "poss_def", -1.0))


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


def held_back_share(folds: list) -> float:
    """1 / K, the share of its season's games each fold holds back, K the folds of a repeat; one K for every repeat of
    `folds` (a ValueError when they differ).

    Example: step 4's folds, four a repeat, give 0.25; `openrapm folds --folds-per-repeat 10` gives 0.10.
    """
    repeats: dict = {}
    for fold in folds:
        repeats.setdefault((int(fold.season), int(fold.repeat)), set()).add(int(fold.fold))
    sizes = {len(held) for held in repeats.values()}
    if len(sizes) != 1:
        raise ValueError(f"the folds hold back different shares of their seasons: {sorted(sizes)} folds a repeat")
    return 1.0 / sizes.pop()


def read_to_full_season(first: pd.DataFrame, second: pd.DataFrame, first_share: float,
                        second_share: float) -> pd.DataFrame:
    """Each season's multipliers off the straight line through two holdout sizes, read at nothing held back, as the
    full-season ratings are built (Richardson extrapolation, notes/DECISIONS.md, experiment 101).  `first` and
    `second` are `multipliers_by_season` tables of fold sets holding back `first_share` and `second_share`.

    Example: a games' part of 1.41 with a quarter held back and 1.35 with a tenth: the line falls 0.4 per whole season
    held back, and reads 1.41 - 0.25 x 0.4 = 1.31 at nothing held back.
    """
    if first_share == second_share:
        raise ValueError("the two fold sets hold back the same share; the line needs two")
    columns = [column for column in MULTIPLIER_COLUMNS if column in first.columns]
    a = first.set_index("season")
    b = second.set_index("season").loc[a.index]
    slope = (a[columns] - b[columns]) / (first_share - second_share)
    out = a.copy()
    out[columns] = a[columns] - first_share * slope
    return out.reset_index()


def full_season_multipliers(seasons, folds: list, second_folds: list, rule: str, nearest: int = 0,
                            free_games: bool = False) -> pd.DataFrame:
    """`multipliers_by_season` on two fold sets of different holdout sizes, read to a full season
    (`read_to_full_season`): step 4's ratings are built from every game, its folds' from part of them, and the
    multipliers that fit a fold's held-back games move with the part held back (notes/DECISIONS.md, "2026 at six fold
    sizes" and experiment 101)."""
    return read_to_full_season(multipliers_by_season(seasons, folds, rule, nearest, free_games),
                               multipliers_by_season(seasons, second_folds, rule, nearest, free_games),
                               held_back_share(folds), held_back_share(second_folds))


# ------------------------------------------------------------------------------------- the within layer
def team_split(team, values, weight) -> tuple[np.ndarray, np.ndarray]:
    """(each row's team mean, its deviation from it): the possession-weighted mean of `values` over the rows of the same
    team; a team whose rows weigh nothing has mean 0.

    Example: two players of one team, +4.0 on 3,000 possessions and -1.0 on 1,000: the mean is (12,000 - 1,000) /
    4,000 = 2.75, the deviations +1.25 and -3.75.
    """
    codes, _ = pd.factorize(np.asarray(team))
    values = np.asarray(values, dtype=float)
    weight = np.maximum(np.asarray(weight, dtype=float), 0.0)
    total = np.bincount(codes, weights=weight)
    sums = np.bincount(codes, weights=weight * values)
    mean = np.divide(sums, total, out=np.zeros_like(sums), where=total > 0)[codes]
    return mean, values - mean


def _within_games(codes: np.ndarray, values: np.ndarray, w: np.ndarray) -> np.ndarray:
    """sqrt(w) x (values less their team-game's possession-weighted mean): a free level per team-game, taken out."""
    values = np.atleast_2d(values.T).T
    total = np.bincount(codes, weights=w)
    means = np.column_stack([np.bincount(codes, weights=w * values[:, k]) / total for k in range(values.shape[1])])
    return (values - means[codes]) * np.sqrt(w)[:, None]


def stint_equations(fold, stints) -> tuple[np.ndarray, np.ndarray]:
    """(A, b): one fold's share of the within layer's normal equations, A m = b with m the four WITHIN_COLUMNS.

    `stints` is a `data.within_folds.HeldStints`: the fold's held-back stints.  Each stint's points per 100, less its
    stand-ins and less the ten players' team means (each at 1), are fitted on the ten players' summed deviations of
    the four parts, with a free level per team-game, weighted by possessions:

        y_s - S_s - sum of team means  =  level of the team-game + sum over k of m_k x D_k,s

    D_k,s the offence's five's summed deviations of part k (the defence's negated, points allowed).  A player without
    a rating in the fold carries the fold's stand-in rating, as check 151 scores him.
    """
    columns, means, rated = [], np.zeros(len(stints.y)), {}
    for column, weight_column, sign in LAYER_PARTS:
        side = "O" if column.endswith("off") else "D"
        rows = fold.rows[side]
        mean, deviation = team_split(rows.team_id.to_numpy(), rows[column].to_numpy(float),
                                     rows[weight_column].to_numpy(float))
        lineup = stints.lineup_o if side == "O" else stints.lineup_d
        index = pd.Index(rows.player_id.to_numpy(np.int64)).get_indexer(lineup.ravel()).reshape(lineup.shape)
        rated[side] = index >= 0
        safe = np.where(rated[side], index, 0)
        columns.append(sign * np.where(rated[side], deviation[safe], 0.0).sum(axis=1))
        means += sign * np.where(rated[side], mean[safe], 0.0).sum(axis=1)
    fills = (5 - rated["O"].sum(axis=1)) * fold.fill_o + (5 - rated["D"].sum(axis=1)) * fold.fill_d
    x = _within_games(stints.team_game, np.column_stack(columns), stints.w)
    t = _within_games(stints.team_game, stints.y - fills - means, stints.w)[:, 0]
    return x.T @ x, x.T @ t


def within_multipliers_by_season(seasons, equations: dict, rule: str, nearest: int = 0) -> pd.DataFrame:
    """One row per season: season and the four WITHIN_COLUMNS, each season's from the summed `stint_equations` of the
    fold seasons the rule allows (`fold_seasons_used`); `equations` is {fold season: (A, b)}, summed over its folds."""
    fold_seasons = sorted(equations)
    fits, rows = {}, []
    for season in sorted(seasons):
        used = fold_seasons_used(season, fold_seasons, rule, nearest)
        if used not in fits:
            fits[used] = np.linalg.solve(sum(equations[s][0] for s in used), sum(equations[s][1] for s in used))
        rows.append(dict(season=int(season), **{name: float(value) for name, value in zip(WITHIN_COLUMNS,
                                                                                           fits[used])}))
    return pd.DataFrame(rows)


def _layered(rows: pd.DataFrame, team, team_multipliers: tuple, within: tuple) -> dict:
    """{part column: the part after both layers}: team multiplier x team mean + within multiplier x deviation, each
    part's team mean weighted by that side's possessions.  `team_multipliers` and `within` follow LAYER_PARTS."""
    out = {}
    for (column, weight_column, _), team_m, within_m in zip(LAYER_PARTS, team_multipliers, within):
        mean, deviation = team_split(team, rows[column].to_numpy(float), rows[weight_column].to_numpy(float))
        out[column] = team_m * mean + within_m * deviation
    return out


def _shrink_season(rows: pd.DataFrame, multiplier_off: float, multiplier_def: float, games_off: float = 1.0,
                   games_def: float = 1.0, within: tuple | None = None, team=None) -> pd.DataFrame:
    """One season's rows with the prior parts multiplied (and the games' parts, when freed) and the ratings
    re-centred (the module docstring).  With `within`, the four within-layer multipliers in LAYER_PARTS order, and
    `team`, each row's main team, every part is split into team mean and deviation first (the two layers)."""
    rows = rows.copy()
    if within is not None:
        for column, values in _layered(rows, team, (multiplier_off, games_off, multiplier_def, games_def),
                                       within).items():
            rows[column] = values
    else:
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


def shrink(table: pd.DataFrame, multipliers: pd.DataFrame, teams: pd.DataFrame | None = None) -> pd.DataFrame:
    """Step 4 on a step-3 table: each season's prior parts times its multipliers, re-centred; u unchanged unless
    `multipliers` carries games_multiplier_off and games_multiplier_def (the games' part freed).  When `multipliers`
    carries the four WITHIN_COLUMNS, the two layers: `teams` (player_id, season, team_id: each player's main team,
    data.roles.main_teams) says whose mean each part is split around; a player it lacks is a team of his own.

    `table` is in the published schema (prior_rapm.PUBLISHED_COLUMNS); `multipliers` is `multipliers_by_season`.
    The raw-sign columns (offense, defense, prior_offense, prior_defense) follow the published ones, and the
    columns keep the table's order.  Rows come out sorted by season, each season's rows in their own order.
    """
    by_season = multipliers.set_index("season")
    freed = "games_multiplier_off" in by_season.columns
    layered = all(column in by_season.columns for column in WITHIN_COLUMNS)
    if layered and teams is None:
        raise ValueError("the multipliers carry the within layer, so `teams` must say each player's main team")
    team_of = teams.set_index(["player_id", "season"]).team_id if layered else None

    def games(season, side):
        return by_season.at[season, f"games_multiplier_{side}"] if freed else 1.0
    seasons = []
    for season, rows in table.groupby("season", sort=True):
        within, team = None, None
        if layered:
            within = tuple(float(by_season.at[season, column]) for column in WITHIN_COLUMNS)
            key = pd.MultiIndex.from_arrays([rows.player_id.to_numpy(np.int64), np.full(len(rows), int(season))])
            team = np.array(team_of.reindex(key).to_numpy(float), dtype=float)
            alone = np.isnan(team)
            team[alone] = -1.0 - np.arange(int(alone.sum()))          # a team of his own
        seasons.append(_shrink_season(rows, by_season.at[season, "multiplier_off"],
                                      by_season.at[season, "multiplier_def"], games(season, "off"),
                                      games(season, "def"), within, team))
    return pd.concat(seasons, ignore_index=True)[list(table.columns)]


def season_factors(multipliers: pd.DataFrame, season: int) -> tuple:
    """(m_O, m_D, g_O, g_D): one season's row of `multipliers_by_season`, as the build applies it (the defence factor
    already in it), 1 for a games' part the table does not free; with the within layer, its four WITHIN_COLUMNS after
    them, eight in all."""
    row = multipliers.set_index("season").loc[int(season)]
    factors = (float(row.multiplier_off), float(row.multiplier_def), float(row.get("games_multiplier_off", 1.0)),
               float(row.get("games_multiplier_def", 1.0)))
    if all(column in row.index for column in WITHIN_COLUMNS):
        factors += tuple(float(row[column]) for column in WITHIN_COLUMNS)
    return factors


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
    if len(factors) == 8:
        return _shrink_fold_layered(rows, season, factors)
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


def layered_fold_parts(rows: dict, factors: tuple) -> dict:
    """{"O", "D"}: (prior part, games' part) of a fold's rated players after both layers, each side re-centred over its
    own rows by offensive possessions, as step 4 re-centres; `factors` is `season_factors`'s eight.  The offence's
    parts are split over the offence rows' teams, the defence's over the defence rows'."""
    team_m = (factors[0], factors[2], factors[1], factors[3])
    out = {}
    for side, prior, games in (("O", "prior_off", "u_off"), ("D", "prior_def", "u_def")):
        side_rows = rows[side]
        parts = _layered(side_rows, side_rows.team_id.to_numpy(), team_m, factors[4:])
        level = np.average(parts[prior] + parts[games], weights=side_rows.poss_off.to_numpy(float))
        out[side] = (parts[prior] - level, parts[games])
    return out


def _shrink_fold_layered(rows: dict, season: int, factors: tuple) -> pd.DataFrame:
    """`shrink_fold` with the two layers.  A player with no defence row takes his defence parts from his offence row,
    split around his team's defensive mean as the defence rows are (0 when his team has no defence row)."""
    offence, defence = rows["O"], rows["D"]
    parts = layered_fold_parts(rows, factors)
    table = pd.DataFrame({"player_id": offence.player_id.to_numpy(np.int64), "season": np.int64(season),
                          "prior_off": parts["O"][0], "u_off": parts["O"][1],
                          "poss_off": offence.poss_off.to_numpy(float), "poss_def": offence.poss_def.to_numpy(float)})
    table = table.join(pd.DataFrame({"prior_def": parts["D"][0], "u_def": parts["D"][1]},
                                    index=defence.player_id.to_numpy(np.int64)), on="player_id")
    alone = table.prior_def.isna().to_numpy()
    if alone.any():
        raw = _layered(defence, defence.team_id.to_numpy(), (factors[0], factors[2], factors[1], factors[3]),
                       factors[4:])
        level = np.average(raw["prior_def"] + raw["u_def"], weights=defence.poss_off.to_numpy(float))
        for column, team_m, within_m in (("prior_def", factors[1], factors[6]), ("u_def", factors[3], factors[7])):
            mean, _ = team_split(defence.team_id.to_numpy(), defence[column].to_numpy(float),
                                 defence.poss_def.to_numpy(float))
            team_mean = pd.Series(mean, index=defence.team_id.to_numpy()).groupby(level=0).first()
            mine = team_mean.reindex(offence.team_id.to_numpy()[alone]).fillna(0.0).to_numpy()
            value = offence[column].to_numpy(float)[alone]
            shift = level if column == "prior_def" else 0.0
            table.loc[alone, column] = team_m * mine + within_m * (value - mine) - shift
    table["rating_off"] = table.prior_off + table.u_off
    table["rating_def"] = table.prior_def + table.u_def
    table["rating_total"] = table.rating_off + table.rating_def
    table["offense"], table["defense"] = table.rating_off, -table.rating_def
    return table
