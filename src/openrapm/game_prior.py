"""The box-score prior game by game: the multi-stat network's forecasts (game_rates.py) as the box-score prior's rows,
one before every game, and what the RAPM fit reads from them.

The owner's design (2026-10-27): the prior changes over the season, each game's from the player's games before it; the
ridge pulls every stint toward the prior of its game ("align the prior / do multiplication in a smart/sparse way"); and
a player's season value is his +/- coefficient plus his prior averaged over the season:

    rating = s x (sum over his games g of  w_g x prior_g) / (sum of w_g)  +  u

    prior_g   his prior before game g: the box-score prior answering for the network's forecast then, with the swap
              part of the prior added (`per_game_second`)
    w_g       his possessions on that side in game g, the weight his stint rows carry in the fit
    s, u      the prior scale and his +/- (prior_rapm.py)

That is the fit's own average credit to him: each of his stint rows credits him s x prior_g + u.

Words used below
    state       the network's forecast at a point of a player's season: before one of his games, after his last one
                (the end row of game_rates.Games), or on a cut day
    cut         the day by which a share of the season's regular-season games had been played, as
                data.panel.season_to_date picks it (`cut_days`); the part-season rows read the state on that day
    panel rows  rows in the season panel's layout (features.SEASON_PANEL_COLUMNS) that the box-score prior reads: the
                12 count rates and their `_uncentred` twins and the 9 hustle inputs from the network's forecasts; the
                shares of team possessions and of games started and his possessions from his games so far; every
                other column from the season panel's row of the same player, season and side (`base`), which the
                shipped inputs do not read but for height and weight
    whole-season row   the state after his last game, with the whole season's shares and possessions

Example: before Jokić's 41st game the network's forecast is a row with his 40 games' possessions, his shares so far and
row_seasons the share of the regular season played that morning; the box-score prior answers for it, 1.9 per 100 on
offence say, and every stint of game 41 is pulled toward 1.9 times the season's prior scale.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import game_rates as G
from .features import BOX_RATES, UNCENTRED_RATES

__all__ = ["BOX_COUNTS", "HUSTLE_INPUTS", "SHARE_PAD_GAMES", "season_states", "cut_days", "network_panels",
           "game_rows", "season_average", "per_game_second", "game_tables"]

# the season panel's 13 rates by the per-game counts they are built from (config.yaml box_counts), and how each
# count comes out of the network's forecasts: attempts times the make share, or one forecast as it is
BOX_COUNTS = ["fg3m", "fg3_miss", "fg2m", "fg2_miss", "ftm", "ft_miss", "orb", "drb", "ast", "tov", "stl", "blk", "pf"]
_FROM_SHOTS = {"fg3m": ("three_point_attempts", "three_point_pct", True),
               "fg3_miss": ("three_point_attempts", "three_point_pct", False),
               "fg2m": ("two_point_attempts", "two_point_pct", True),
               "fg2_miss": ("two_point_attempts", "two_point_pct", False),
               "ftm": ("free_throw_attempts", "free_throw_pct", True),
               "ft_miss": ("free_throw_attempts", "free_throw_pct", False)}
_FROM_ONE = {"orb": "off_rebounds", "drb": "def_rebounds", "ast": "assists", "tov": "turnovers", "stl": "steals",
             "blk": "blocks"}
# the panel's hustle inputs (features.HUSTLE_RATES) by the network's stat each is read from
HUSTLE_INPUTS = {"deflections_vs_league_avg": "deflections", "contested_twos_vs_league_avg": "contested_twos",
                 "contested_threes_vs_league_avg": "contested_threes", "charges_drawn_vs_league_avg": "charges_drawn",
                 "loose_balls_recovered_vs_league_avg": "loose_balls_recovered",
                 "def_box_outs_vs_league_avg": "def_box_outs", "off_box_outs_vs_league_avg": "off_box_outs",
                 "screen_assists_vs_league_avg": "screen_assists",
                 "screen_assist_points_vs_league_avg": "screen_assist_points"}
# the shares of team possessions and of games started are padded by two games toward the season's average, so a
# player's first game reads a typical role rather than 0 of 0 (the review of 2026-10-27)
SHARE_PAD_GAMES = 2.0
SHARE_CAP = 0.9               # config.yaml roles.share_cap
_TEAM_SHARE, _STARTS_SHARE = "team_possession_share", "games_started_share"


# ------------------------------------------------------------------------------------------ the forecasts
def season_states(rows: pd.DataFrame, network) -> tuple:
    """(game_rates.Games, (n, 21) forecasts at a neutral game against an average opponent): every state of the rows'
    player-seasons, the end rows included.  `network` carries weights, mean, sd, ks and opponent_pads (a saved
    network, data/game_rate_files.py).  The league levels and the opponent factors come from `rows` alone, so a
    caller that cuts them to a rating's fit games cuts everything with them."""
    games = G.games_from_rows(rows)
    league_row = G.league_by_row(games, G.league_levels(games))
    factors = G.opponent_factors(G.opponent_sums(games, league_row), network.opponent_pads)
    padding = G.padding_forecasts(games, league_row, factors, network.ks)
    inputs = G.network_inputs(games, league_row, factors, padding)
    scaling = G.Standardised(mean=np.asarray(network.mean, np.float32), sd=np.asarray(network.sd, np.float32))
    return games, G.forecast(network.weights, scaling, inputs, padding)


def cut_days(games: G.Games, shares) -> pd.DataFrame:
    """season, share, day, season_share: for each season of `games` and each share, the day by which that share of
    its regular-season games had been played, and the share of them played on or before it: the rule of
    data.panel.season_to_date, on the games handed in.

    Example: 1,230 games and a share of 0.25: the 308th game's day, and 308 / 1,230 if no other game shares it."""
    frame = games.frame
    regular = frame[(~frame.end) & (frame.phase == "RS")].drop_duplicates("game_id")
    out = []
    for season, part in regular.groupby("season"):
        days = part.sort_values("game_idx").game_date.to_numpy()
        for share in shares:
            reached = int(np.ceil(float(share) * len(days)))
            day = days[max(reached, 1) - 1]
            out.append(dict(season=int(season), share=float(share), day=pd.Timestamp(day),
                            season_share=float((days <= day).sum() / len(days))))
    return pd.DataFrame(out)


# ------------------------------------------------------------------------------------------ the panel rows
def _so_far(games: G.Games, queries: pd.DataFrame) -> pd.DataFrame:
    """For each query (season, player_id, team_id, before: a day), his games before that day and his team's:
    possessions at each end, starts, games, his team's possessions at both ends.  The panel's role inputs on that
    day (data/roles.py), from the games handed in."""
    frame = games.frame[~games.frame.end].sort_values("game_date", kind="stable")
    own = frame[["season", "player_id", "game_date", "poss_off", "poss_def", "started"]].copy()
    grouped = own.groupby(["season", "player_id"], sort=False)
    for column in ("poss_off", "poss_def", "started"):
        own[column] = grouped[column].cumsum()
    own["games"] = grouped.cumcount() + 1.0
    team = (frame.drop_duplicates(["season", "team_id", "game_id"])
            .assign(team_poss=lambda f: f.team_poss_off + f.team_poss_def)
            [["season", "team_id", "game_date", "team_poss"]].copy())
    team["team_poss"] = team.groupby(["season", "team_id"], sort=False).team_poss.cumsum()
    ordered = queries.reset_index().rename(columns={"index": "query"}).sort_values("before", kind="stable")
    mine = pd.merge_asof(ordered, own.rename(columns={"game_date": "before"}), on="before",
                         by=["season", "player_id"], allow_exact_matches=False, direction="backward")
    mine = pd.merge_asof(mine.sort_values("before", kind="stable"), team.rename(columns={"game_date": "before"}),
                         on="before", by=["season", "team_id"], allow_exact_matches=False, direction="backward")
    mine = mine.set_index("query").reindex(queries.index)
    return mine[["poss_off", "poss_def", "started", "games", "team_poss"]].fillna(0.0)


def _role_shares(so_far: pd.DataFrame, season: np.ndarray, typical: pd.DataFrame) -> pd.DataFrame:
    """The two role inputs from `_so_far`, each padded by `SHARE_PAD_GAMES` games toward the season's typical value
    (`typical`, indexed by season: share, starts, team_poss_per_game)."""
    level = typical.reindex(season)
    per_game = level.team_poss_per_game.to_numpy()
    own = (so_far.poss_off + so_far.poss_def).to_numpy()
    team_share = ((own + SHARE_PAD_GAMES * level.share.to_numpy() * per_game)
                  / (so_far.team_poss.to_numpy() + SHARE_PAD_GAMES * per_game))
    starts = ((so_far.started.to_numpy() + SHARE_PAD_GAMES * level.starts.to_numpy())
              / (so_far.games.to_numpy() + SHARE_PAD_GAMES))
    return pd.DataFrame({_TEAM_SHARE: np.clip(team_share, 0.0, SHARE_CAP), _STARTS_SHARE: starts}, index=so_far.index)


def _typical(games: G.Games) -> pd.DataFrame:
    """Per season: the average player-season's share of team possessions, the share of games played that were
    starts, and a team's possessions at both ends per game, over the games handed in."""
    frame = games.frame[~games.frame.end]
    team_games = frame.drop_duplicates(["season", "team_id", "game_id"])
    per_game = (team_games.team_poss_off + team_games.team_poss_def).groupby(team_games.season).mean()
    own = (frame.poss_off + frame.poss_def).groupby([frame.season, frame.player_id, frame.team_id]).sum()
    team_total = (team_games.team_poss_off + team_games.team_poss_def).groupby(
        [team_games.season, team_games.team_id]).sum()
    share = (own / team_total.reindex(own.index.droplevel("player_id")).to_numpy()).groupby(level="season").mean()
    starts = frame.started.groupby(frame.season).mean()
    return pd.DataFrame({"share": share, "starts": starts, "team_poss_per_game": per_game})


def _levels(games: G.Games) -> dict:
    """{"O", "D"}: each season's league level of the 13 counts per 100 of that side's possessions (the panel's centring
    level, box_rates.py), and of the 9 hustle counts (data/dashboard_rates.py's league rate), over the games handed
    in."""
    frame = games.frame[~games.frame.end]
    hustle_counts = {stat.name: stat.counts[0] for stat in G.STATS[G.HUSTLE_STATS[0]:]}
    out = {}
    for side, possessions in (("O", "poss_off"), ("D", "poss_def")):
        rows = {}
        for season, part in frame.groupby("season"):
            total = part[possessions].sum()
            values = {count: 100.0 * part[count].sum() / total for count in BOX_COUNTS}
            for stat, column in hustle_counts.items():
                # a count nobody recorded all season is missing, as in the season panel: the 2017 hustle box scores
                # list no box-outs at all (the network's 2017 box-out forecasts are therefore never read)
                counted = part[column].notna()
                made = part.loc[counted, column].sum()
                values[stat] = (100.0 * made / part.loc[counted, possessions].sum() if made > 0 else np.nan)
            rows[int(season)] = values
        out[side] = pd.DataFrame.from_dict(rows, orient="index")
    return out


def _side_rates(forecasts: np.ndarray, poss_off: np.ndarray, poss_def: np.ndarray, side: str) -> pd.DataFrame:
    """The 13 counts and 9 hustle counts per 100 possessions of `side`, from forecasts per 100 of each stat's own
    exposure: a defensive stat on an offence row is scaled by his defensive over offensive possessions so far, and
    so on (1 before his first game)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        to_off = np.where(poss_off > 0, poss_def / poss_off, 1.0)
        to_def = np.where(poss_def > 0, poss_off / poss_def, 1.0)
        both_to_side = np.where((poss_off + poss_def) > 0,
                                0.5 * (poss_off + poss_def) / np.where(side == "O", poss_off, poss_def), 1.0)
    both_to_side = np.where(np.isfinite(both_to_side), both_to_side, 1.0)
    scale = {"off": np.ones(len(poss_off)) if side == "O" else to_def,
             "def": to_off if side == "O" else np.ones(len(poss_off)), "both": both_to_side}
    value = {name: forecasts[:, j] for j, name in enumerate(G.NAMES)}
    exposure = {stat.name: stat.exposure for stat in G.STATS}
    out = {}
    for count, (attempts, share, made) in _FROM_SHOTS.items():
        rate = value[attempts] * scale[exposure[attempts]]
        out[count] = rate * (value[share] if made else 1.0 - value[share])
    for count, stat in _FROM_ONE.items():
        out[count] = value[stat] * scale[exposure[stat]]
    out["pf"] = np.full(len(poss_off), np.nan)
    for stat in G.NAMES[G.HUSTLE_STATS[0]:]:
        out[stat] = value[stat] * scale[exposure[stat]]
    return pd.DataFrame(out)


def _panel_side(states: pd.DataFrame, forecasts: np.ndarray, so_far: pd.DataFrame, shares: pd.DataFrame,
                levels: pd.DataFrame, base: pd.DataFrame, side: str) -> pd.DataFrame:
    """One side's panel rows for the states (season, player_id, plus any columns to keep): the network's rates
    centred on the season's level, the hustle inputs against the league's, the role shares, the possessions, every
    other column from `base`'s row of the player, season and side."""
    rates = _side_rates(forecasts, so_far.poss_off.to_numpy(), so_far.poss_def.to_numpy(), side)
    level = levels.reindex(states.season.to_numpy()).reset_index(drop=True)
    out = states.reset_index(drop=True).assign(side=side)
    out["possessions"] = (so_far.poss_off if side == "O" else so_far.poss_def).to_numpy()
    for rate, uncentred, count in zip(BOX_RATES, UNCENTRED_RATES, BOX_COUNTS):
        out[uncentred] = rates[count].to_numpy()
        out[rate] = rates[count].to_numpy() - level[count].to_numpy()
    for name, stat in HUSTLE_INPUTS.items():
        with np.errstate(invalid="ignore", divide="ignore"):
            against = (rates[stat].to_numpy() - level[stat].to_numpy()) / level[stat].to_numpy()
        out[name] = np.where(np.isfinite(against), against, np.nan)
    out[_TEAM_SHARE] = shares[_TEAM_SHARE].to_numpy()
    out[_STARTS_SHARE] = shares[_STARTS_SHARE].to_numpy()
    finite = [*BOX_RATES[:-1], *UNCENTRED_RATES[:-1], _TEAM_SHARE, _STARTS_SHARE, "possessions"]
    broken = [column for column in finite if not np.isfinite(out[column].to_numpy(dtype=float)).all()]
    if broken:
        # a booster with linear leaves fitted on one infinite cell answers NaN for every row that reaches the leaf
        raise ValueError(f"the network's panel rows hold values that are not finite in {broken}")
    rebuilt = set(out.columns)
    keep = [column for column in base.columns if column not in rebuilt and column not in ("side", "player_id",
                                                                                             "season")]
    carried = base[base.side == side].drop_duplicates(["player_id", "season"]).set_index(["player_id", "season"])[keep]
    joined = carried.reindex(pd.MultiIndex.from_arrays([out.player_id, out.season])).reset_index(drop=True)
    return pd.concat([out, joined], axis=1).copy()


def _panel_rows(games: G.Games, forecasts: np.ndarray, states: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    """Both sides' panel rows for `states` (one row each: row, season, player_id, team_id, before: the day the state
    stands on, plus any columns to keep), offence first."""
    so_far = _so_far(games, states[["season", "player_id", "team_id", "before"]])
    shares = _role_shares(so_far, states.season.to_numpy(), _typical(games))
    levels = _levels(games)
    picked = forecasts[states.row.to_numpy()]
    keys = states.drop(columns=["row", "team_id", "before"])
    return pd.concat([_panel_side(keys, picked, so_far.reset_index(drop=True), shares.reset_index(drop=True),
                                  levels[side], base, side) for side in ("O", "D")], ignore_index=True)


def network_panels(games: G.Games, forecasts: np.ndarray, base: pd.DataFrame, shares) -> tuple:
    """(whole-season rows, part-season rows) in the season panel's layout for every player-season of `games`: the
    box-score prior's training rows (box_prior.training_rows' `rows` and `part_seasons`), and the rated season's
    whole-season rows.  A part-season row carries `share` and `season_share` and exists only when he had played by
    the cut day; `base` is the season panel the other columns come from."""
    frame = games.frame
    end = np.flatnonzero(frame.end.to_numpy())
    whole = pd.DataFrame({"row": end, "season": frame.season.to_numpy()[end],
                          "player_id": frame.player_id.to_numpy()[end], "team_id": frame.team_id.to_numpy()[end],
                          "before": frame.game_date.to_numpy()[end]})
    whole_rows = _panel_rows(games, forecasts, whole, base)
    if not len(shares):
        return whole_rows, None
    cuts = cut_days(games, shares)
    parts = []
    for cut in cuts.itertuples():
        mask = (frame.season == cut.season).to_numpy() & (frame.game_date > cut.day).to_numpy()
        first_after = (frame[mask].groupby("player_id", sort=False).head(1))
        first_after = first_after[games.position[first_after.index.to_numpy()] > 0]
        parts.append(pd.DataFrame({"row": first_after.index.to_numpy(), "season": cut.season,
                                   "player_id": first_after.player_id.to_numpy(),
                                   "team_id": first_after.team_id.to_numpy(),
                                   "before": cut.day + pd.Timedelta(days=1), "share": cut.share,
                                   "season_share": cut.season_share}))
    part_rows = _panel_rows(games, forecasts, pd.concat(parts, ignore_index=True), base)
    return whole_rows, part_rows


def game_rows(games: G.Games, forecasts: np.ndarray, base: pd.DataFrame) -> pd.DataFrame:
    """One panel row per side and game played: the state before the game, with game_id and `season_progress`, the
    share of the season's regular-season games played on days before it (1 in the playoffs): what the box-score
    prior answers for game by game (box_prior.game_rated_rows)."""
    frame = games.frame
    real = np.flatnonzero(~frame.end.to_numpy())
    progress = G.season_progress(frame)
    states = pd.DataFrame({"row": real, "season": frame.season.to_numpy()[real],
                           "player_id": frame.player_id.to_numpy()[real], "team_id": frame.team_id.to_numpy()[real],
                           "before": frame.game_date.to_numpy()[real], "game_id": frame.game_id.to_numpy()[real],
                           "poss_off_game": frame.poss_off.to_numpy()[real],
                           "poss_def_game": frame.poss_def.to_numpy()[real], "season_progress": progress[real]})
    return _panel_rows(games, forecasts, states, base)


# ------------------------------------------------------------------------------------------ season averages
def season_average(rows: pd.DataFrame, value: str) -> dict:
    """{"O", "D"}: {player_id: his `value` averaged over his game rows of that side, weighted by his possessions on
    that side in each game}: the weights his stint rows carry in the fit, so this is the prior his season value
    reports.  A player with no possessions on a side is averaged with equal weights."""
    out = {}
    for side, weight in (("O", "poss_off_game"), ("D", "poss_def_game")):
        part = rows[rows.side == side]
        w = part[weight].to_numpy(dtype=float)
        frame = pd.DataFrame({"player_id": part.player_id.to_numpy(), "wv": w * part[value].to_numpy(dtype=float),
                              "w": w, "v": part[value].to_numpy(dtype=float)})
        sums = frame.groupby("player_id")[["wv", "w", "v"]].sum()
        counts = frame.groupby("player_id").size()
        mean = np.where(sums.w > 0, sums.wv / sums.w.where(sums.w > 0, 1.0), sums.v / counts)
        out[side] = dict(zip(sums.index.to_numpy(), mean))
    return out


def per_game_second(rows: pd.DataFrame, box: str, second: dict, box_average: dict, weight: dict) -> np.ndarray:
    """Each game row's prior with the swap part of the prior (second_prior.py) added:

        prior_g = b x box_g + (second - b x box average)

    `second` and `box_average` {"O", "D"}: {player_id: value}; `weight` {"O", "D"}: {player_id: b, the line's
    coefficient on his box-score prior}.  Averaged over his games with `season_average`'s weights it is his second
    prior exactly, and every team's possession-weighted total stays where second_prior.hold_team_totals held it."""
    out = np.empty(len(rows))
    for side in ("O", "D"):
        mask = (rows.side == side).to_numpy()
        ids = rows.player_id.to_numpy()[mask]
        b = pd.Series(weight[side]).reindex(ids).fillna(1.0).to_numpy()
        level = (pd.Series(second[side]).reindex(ids).to_numpy()
                 - b * pd.Series(box_average[side]).reindex(ids).to_numpy())
        out[mask] = b * rows[box].to_numpy(dtype=float)[mask] + level
    return out


def game_tables(design, rows: pd.DataFrame, value: str, fallback: dict) -> tuple:
    """(table_O, table_D): `value` of each game row as a (design games x design players) array per side, the layout
    prior_rapm.ridge_system reads, the game by game_id and the player by player_id.  A cell the rows lack (a player
    on the court in a game with no game row) takes his `fallback` {"O", "D"}: {player_id: value}."""
    games = design.games
    game_of = pd.Series(games.game_idx.to_numpy(), index=games.game_id.astype(str).to_numpy())
    players = design.spec.players.player_id.to_numpy()
    column_of = pd.Series(np.arange(len(players)), index=players)
    out = []
    for side in ("O", "D"):
        table = np.tile(pd.Series(fallback[side]).reindex(players).fillna(0.0).to_numpy(dtype=float),
                        (int(games.game_idx.max()) + 1, 1))
        part = rows[rows.side == side]
        game = game_of.reindex(part.game_id.astype(str).to_numpy()).to_numpy()
        column = column_of.reindex(part.player_id.to_numpy()).to_numpy()
        ok = ~(pd.isna(game) | pd.isna(column))
        table[game[ok].astype(np.int64), column[ok].astype(np.int64)] = part[value].to_numpy(dtype=float)[ok]
        out.append(table)
    return tuple(out)
