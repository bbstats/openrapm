"""Who guarded whom, game by game: a season's matchup pairs, and the season panel's matchup inputs built from them.

Why (the owner, 2026-10-28): "the matchup data, e.g. who is matched up with whom.  we can use this to build all sorts
of things like - position! (average height of who you are defending, average height of who defended you are wonderful
proxies for this on D/O).  we can also use it to build features like '% of time matched up with the #1 self creator'
and '% of time matched up with the #1 overall creator', and 'average self-creation of who you are defending' and
'average total creation of who you are defending'".  Experiment 62 read the season totals (`LeagueSeasonMatchups`,
data/dashboard_rates.py), which a fold cannot cut to its fit games; the per-game matchup box score
(`BoxScoreMatchupsV3`, `openrapm data matchup-games`, 2017-18 on, playoffs too) can, so every input here comes from
the fit games alone.

Words used below
    attacker             the offensive player of a pair
    defender             the player the league's tracking says guarded him
    matchup possessions  the pair's partial possessions: each of the attacker's possessions split among the defenders
                         by the time each guarded him, so a team's attackers sum to about five times its possessions
    fit games            the games the inputs are built from: all of a season's for the panel, three quarters of them
                         in a within-season fold, the first part of a season for the season so far
    total creation       self-creation (features.CREATION: the owner's self-created shot attempts per 100) plus the
                         owner's estimated potential assists per 100 (features.DERIVED: the passes to a shot his
                         assists by zone imply), both at their level: the shots he creates for himself and for others.
                         The tracked potential assists the matchup box score names are 0 in every game
    top creator          of the attackers a team used in a game, the one with the most created shots per game over
                         the fit games, his rate per 100 times his offensive possessions per game, so a starter who
                         creates a lot outranks a bench scorer with a higher rate; one by self-creation, one by total
    shared floor         the possessions a defender and the opposing top creator were on the court together, the top
                         creator's team on offence, from the stints; times the game's matchup possessions of that team
                         over five times its stint possessions, so the two counts are in one unit

The inputs (features.MATCHUP_GAME_INPUTS), one value per player and season, the same on both of his rows; w is a
pair's matchup possessions in one game:

    attackers_guarded_height_in             sum of w x the attacker's height / sum of w, over his pairs as defender
    defenders_faced_height_in               the same over his pairs as attacker, with the defender's height
    attackers_guarded_self_creation_per100  as the height, with the attacker's self-creation
    attackers_guarded_total_creation_per100 as the height, with the attacker's total creation
    top_self_creator_guarded_share          sum of w on the attacking team's top creator in that game / sum of w
    top_total_creator_guarded_share         the same, the top creator by total creation
    ..._on_shared_floor                     sum of w on the top creator / the shared floor with him

Each is padded like a box rate (box_rates.py): toward the average of the season's players in his bin of the same
denominator (box_rates.possession_bins), by a pad k from his odd against his even games (box_rates.split_half_k).
Missing for a player without pairs in the fit games, and for every season before 2018.

Example: a defender who spent 300 matchup possessions on 6-foot-3 guards and 100 on 6-foot-9 forwards reads
(300 x 75 + 100 x 81) / 400 = 76.5 inches; padded by k = 30 toward his bin's 78.4, (400 x 76.5 + 30 x 78.4) / 430 =
76.6.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from ..box_rates import MIN_BIN_PLAYERS, bin_of, possession_bins, shrink, split_half_k
from ..design import AWAY_SLOTS, HOME_SLOTS
from ..features import MATCHUP_GAME_INPUTS
from .dashboards import MATCHUP_GAMES_FIRST
from .ingest import raw_dir
from .seasons import in_progress

__all__ = ["PAIR_COLUMNS", "FALLBACK_PAD", "matchup_folder", "matchup_pairs", "top_creators", "shared_floor",
           "padded_average", "matchup_inputs"]

# the matchup box score's columns read here, under the names used here; the game id comes from the file's name
PAIR_COLUMNS = {"teamId": "attacker_team_id", "personIdOff": "attacker_id", "personIdDef": "defender_id",
                "partialPossessions": "matchup_possessions"}
# the pad k, in matchup possessions, of a season too short for split_half_k (fewer than three players with 20 in each
# half: a season so far of a few games); the light pad of data/dashboard_rates.FIXED_PAD
FALLBACK_PAD = 50.0
(HEIGHT, DEFENDERS_HEIGHT, SELF_CREATION, TOTAL_CREATION, TOP_SELF, TOP_TOTAL, TOP_SELF_SHARED,
 TOP_TOTAL_SHARED) = MATCHUP_GAME_INPUTS
IDS = ("attacker_team_id", "attacker_id", "defender_id")


# ------------------------------------------------------------------------------------------------------ the pairs
def matchup_folder(cfg: dict, season: int) -> Path:
    """paths.raw/dashboards/matchups/games/<season>: one file a game, as `openrapm data matchup-games` writes it."""
    return raw_dir(cfg) / "dashboards" / "matchups" / "games" / str(int(season))


@lru_cache(maxsize=2)
def _season_pairs(folder: str) -> pd.DataFrame:
    """Every cached game of one season's folder stacked: game_id (the file's name), `PAIR_COLUMNS` under their names
    here, ids as integers, pairs without matchup possessions left out.  A season's folds read it once."""
    frames = [pd.read_parquet(path, columns=list(PAIR_COLUMNS)).assign(game_id=path.stem)
              for path in sorted(Path(folder).glob("*.parquet"))]
    frames = [frame for frame in frames if len(frame)]
    raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=[*PAIR_COLUMNS, "game_id"])
    out = raw.rename(columns=PAIR_COLUMNS)[["game_id", *PAIR_COLUMNS.values()]]
    out = out.assign(**{column: pd.to_numeric(out[column], errors="coerce") for column in PAIR_COLUMNS.values()})
    keep = out[list(IDS)].notna().all(axis=1) & (out["matchup_possessions"] > 0)
    return out[keep].astype({**dict.fromkeys(IDS, np.int64), "game_id": str}).reset_index(drop=True)


def matchup_pairs(cfg: dict, season: int, game_ids) -> tuple:
    """(the pairs of the games `game_ids`: game_id, attacker_team_id, attacker_id, defender_id, matchup_possessions;
    the game ids read), or (None, set()) for a season before `MATCHUP_GAMES_FIRST` or without per-game files.

    A within-season fold hands its fit games, so the held-back games never enter (notes/WITHIN_SEASON_LEDGER.md).  A
    game of `game_ids` with no file stops the run, naming the command that fetches it; a game the league answered with
    no pairs is read and adds nothing.  The season in progress is the exception: the league may not have processed
    last night's games by the morning, so a game not on disk yet is left out, and the daily job asks again tomorrow.

    Example: the 980 fit games of a 2024 fold give about 180 pairs a game.
    """
    folder = matchup_folder(cfg, season)
    if int(season) < MATCHUP_GAMES_FIRST or not folder.exists():
        return None, set()
    wanted = {str(game) for game in game_ids}
    on_disk = {path.stem for path in folder.glob("*.parquet")}
    missing = sorted(wanted - on_disk)
    if missing and int(season) not in in_progress(cfg):
        raise FileNotFoundError(f"{len(missing)} games of {season} have no matchup box score (first {missing[0]}); "
                                f"`openrapm data matchup-games` fetches them")
    read = wanted & on_disk
    pairs = _season_pairs(str(folder))
    return pairs[pairs["game_id"].isin(read)].reset_index(drop=True), read


# ------------------------------------------------------------------------------------ the top creators and the floor
def top_creators(pairs: pd.DataFrame, volume: pd.Series) -> pd.DataFrame:
    """game_id, attacker_team_id, top_id: each team's top creator in each game, the attacker it used with the most
    created shots per game (`volume`, indexed by player id).  The lower id breaks a tie; an attacker without a volume
    cannot be the top.

    Example: a team that used attackers with 9.1, 4.0 and 12.5 created shots per game: the third.
    """
    used = pairs[["game_id", "attacker_team_id", "attacker_id"]].drop_duplicates()
    used = used.assign(volume=used["attacker_id"].map(volume).to_numpy(dtype=float))
    used = used[np.isfinite(used["volume"])].sort_values(["game_id", "attacker_team_id", "volume", "attacker_id"],
                                                          ascending=[True, True, False, True])
    top = used.drop_duplicates(["game_id", "attacker_team_id"])
    return top[["game_id", "attacker_team_id", "attacker_id"]].rename(columns={"attacker_id": "top_id"})


def _on_top(pairs: pd.DataFrame, tops: pd.DataFrame) -> np.ndarray:
    """One per pair: 1 when the attacker is his team's top creator in that game, 0 when not, NaN when his team has no
    top creator that game."""
    joined = pairs[["game_id", "attacker_team_id", "attacker_id"]].merge(tops, on=["game_id", "attacker_team_id"],
                                                                         how="left")
    on = (joined["attacker_id"].to_numpy() == joined["top_id"].to_numpy()).astype(float)
    return np.where(joined["top_id"].isna().to_numpy(), np.nan, on)


def shared_floor(pairs: pd.DataFrame, tops: pd.DataFrame, stints: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """game_id, defender_id, shared: each defender's shared floor with the opposing top creator (`tops`) in each game,
    in matchup possessions (the module docstring).

    `stints` are the fit games' stints (game_id, h1-h5, a1-a5, poss_h, poss_a), `games` each game's two teams
    (game_id, home_team_id, away_team_id).

    Example: a top creator on the floor with a defender for 40 of his team's possessions, in a game where his team's
    attackers have 470 matchup possessions on 100 stint possessions: 40 x 470 / 500 = 37.6.
    """
    stints = stints.merge(games[["game_id", "home_team_id", "away_team_id"]], on="game_id")
    shared, offence = [], []
    for attack, defend, possessions, team in ((HOME_SLOTS, AWAY_SLOTS, "poss_h", "home_team_id"),
                                              (AWAY_SLOTS, HOME_SLOTS, "poss_a", "away_team_id")):
        offence.append(stints.groupby(["game_id", team], as_index=False)[possessions].sum()
                       .set_axis(["game_id", "attacker_team_id", "stint_possessions"], axis=1))
        part = stints.merge(tops, left_on=["game_id", team], right_on=["game_id", "attacker_team_id"])
        on = (part[attack].to_numpy() == part["top_id"].to_numpy()[:, None]).any(axis=1)
        part = part[on]
        shared += [pd.DataFrame({"game_id": part["game_id"].to_numpy(), "defender_id": part[slot].to_numpy(),
                                 "attacker_team_id": part["attacker_team_id"].to_numpy(),
                                 "shared": part[possessions].to_numpy(dtype=float)}) for slot in defend]
    floor = (pd.concat(shared, ignore_index=True)
             .groupby(["game_id", "attacker_team_id", "defender_id"], as_index=False)["shared"].sum())
    scale = (pairs.groupby(["game_id", "attacker_team_id"], as_index=False)["matchup_possessions"].sum()
             .merge(pd.concat(offence, ignore_index=True), on=["game_id", "attacker_team_id"]))
    scale = scale[scale["stint_possessions"] > 0]
    scale["scale"] = scale["matchup_possessions"] / (5.0 * scale["stint_possessions"])
    floor = floor.merge(scale[["game_id", "attacker_team_id", "scale"]], on=["game_id", "attacker_team_id"])
    return floor.assign(shared=floor["shared"] * floor["scale"])[["game_id", "defender_id", "shared"]]


# --------------------------------------------------------------------------------------------------- the padding
def _bin_targets(count_sum: np.ndarray, weight_sum: np.ndarray, present: np.ndarray) -> np.ndarray:
    """Each player's target: the average of the players in his bin of weight (box_rates.possession_bins) when it has
    `MIN_BIN_PLAYERS`, else the season's."""
    target = np.full(len(count_sum), count_sum[present].sum() / weight_sum[present].sum())
    edges = possession_bins(weight_sum, weight_sum)
    if edges is None:
        return target
    bins = bin_of(edges, weight_sum)
    for one in range(len(edges) - 1):
        members = present & (bins == one)
        if members.sum() >= MIN_BIN_PLAYERS:
            target[bins == one] = count_sum[members].sum() / weight_sum[members].sum()
    return target


def padded_average(player: np.ndarray, game: np.ndarray, counts: np.ndarray, weights: np.ndarray,
                   n_players: int, pad: float | None = None) -> np.ndarray:
    """Each player's sum of `counts` over his sum of `weights`, padded (the module docstring); NaN for a player without
    weight.  One row per player and game: `player` runs 0 to n_players - 1, `game` numbers the games in the order they
    were played.  `pad` fixes the pad k instead of estimating it.

    Example: with the pad k at 30 and his bin at 78.4, a player with 400 weight at 76.5 reads 76.6.
    """
    count_sum = np.bincount(player, weights=counts, minlength=n_players)
    weight_sum = np.bincount(player, weights=weights, minlength=n_players)
    present = weight_sum > 0
    if not present.any():
        return np.full(n_players, np.nan)
    k = pad if pad is not None else split_half_k(player, game, counts[:, None], weights,
                                                 np.zeros(n_players, dtype=int), 1)[0, 0]
    k = FALLBACK_PAD if np.isnan(k) else k
    raw = np.where(present, count_sum / np.where(present, weight_sum, 1.0), 0.0)
    padded = shrink(raw, weight_sum, k, _bin_targets(count_sum, weight_sum, present))
    return np.where(present, padded, np.nan)


def _padded_by_player(rows: pd.DataFrame, ids: np.ndarray, pad: float | None) -> np.ndarray:
    """`padded_average` of rows (player, game_id, counts, weight; several a game allowed), one per player of the
    sorted `ids`."""
    rows = rows[np.isin(rows["player"].to_numpy(), ids) & (rows["weight"].to_numpy(dtype=float) > 0)]
    per_game = rows.groupby(["player", "game_id"], as_index=False)[["counts", "weight"]].sum()
    return padded_average(np.searchsorted(ids, per_game["player"].to_numpy()),
                          pd.factorize(per_game["game_id"], sort=True)[0], per_game["counts"].to_numpy(dtype=float),
                          per_game["weight"].to_numpy(dtype=float), len(ids), pad)


def _average(pairs: pd.DataFrame, who: str, value, ids: np.ndarray, pad: float | None) -> np.ndarray:
    """The padded average of `value` (one per pair) over the pairs of each player in column `who`, weighted by matchup
    possessions; a pair whose value is missing is left out."""
    value = np.asarray(value, dtype=float)
    keep = np.isfinite(value)
    weight = pairs["matchup_possessions"].to_numpy(dtype=float)[keep]
    return _padded_by_player(pd.DataFrame({"player": pairs[who].to_numpy()[keep],
                                           "game_id": pairs["game_id"].to_numpy()[keep],
                                           "counts": weight * value[keep], "weight": weight}), ids, pad)


def _shared_floor_share(pairs: pd.DataFrame, tops: pd.DataFrame, on_top: np.ndarray, stints: pd.DataFrame,
                        games: pd.DataFrame, ids: np.ndarray, pad: float | None) -> np.ndarray:
    """The padded share of the shared floor with the top creator that each defender spent guarding him."""
    floor = shared_floor(pairs, tops, stints, games)
    guarded = (pd.DataFrame({"game_id": pairs["game_id"].to_numpy(), "defender_id": pairs["defender_id"].to_numpy(),
                             "on": pairs["matchup_possessions"].to_numpy(dtype=float) * np.nan_to_num(on_top)})
               .groupby(["game_id", "defender_id"], as_index=False)["on"].sum())
    rows = floor.merge(guarded, on=["game_id", "defender_id"], how="outer").fillna({"shared": 0.0, "on": 0.0})
    # where the tracking has him on the top creator for longer than the stints have them on the floor together, the
    # larger of the two, so one game's share stays at most 1
    rows["weight"] = np.maximum(rows["shared"], rows["on"])
    return _padded_by_player(rows.rename(columns={"defender_id": "player", "on": "counts"}), ids, pad)


# ------------------------------------------------------------------------------------------------------ the inputs
def matchup_inputs(pairs: pd.DataFrame, partners: pd.DataFrame, stints: pd.DataFrame | None = None,
                   games: pd.DataFrame | None = None, pad: float | None = None) -> pd.DataFrame:
    """The `MATCHUP_GAME_INPUTS` columns, one row per player with pairs, indexed by player id (the module docstring).

    `pairs` are `matchup_pairs`' rows of the fit games.  `partners`, indexed by player id: height_in,
    self_creation_per100, total_creation_per100 and possessions (his offensive possessions in the fit games).  With
    `stints` and `games` (data/panel.SeasonTables' columns, cut to the fit games) the two shared-floor shares are
    built; without them they are missing.  `pad` fixes every pad k instead of estimating it (0: no padding).

    Example: tests/test_matchups.py works three games of two teams through every column.
    """
    ids = np.union1d(pairs["attacker_id"].to_numpy(), pairs["defender_id"].to_numpy()).astype(np.int64)
    out = pd.DataFrame(np.nan, index=pd.Index(ids, name="player_id"), columns=MATCHUP_GAME_INPUTS)
    if not len(pairs):
        return out
    attackers, defenders = pairs["attacker_id"], pairs["defender_id"]
    out[HEIGHT] = _average(pairs, "defender_id", attackers.map(partners["height_in"]), ids, pad)
    out[DEFENDERS_HEIGHT] = _average(pairs, "attacker_id", defenders.map(partners["height_in"]), ids, pad)
    out[SELF_CREATION] = _average(pairs, "defender_id", attackers.map(partners["self_creation_per100"]), ids, pad)
    out[TOTAL_CREATION] = _average(pairs, "defender_id", attackers.map(partners["total_creation_per100"]), ids, pad)
    games_played = pairs.groupby("attacker_id")["game_id"].nunique().reindex(partners.index)
    per_game = partners["possessions"] / games_played / 100.0
    for share, on_floor, rate in ((TOP_SELF, TOP_SELF_SHARED, "self_creation_per100"),
                                  (TOP_TOTAL, TOP_TOTAL_SHARED, "total_creation_per100")):
        tops = top_creators(pairs, partners[rate] * per_game)
        on_top = _on_top(pairs, tops)
        out[share] = _average(pairs, "defender_id", on_top, ids, pad)
        if stints is not None and games is not None:
            out[on_floor] = _shared_floor_share(pairs, tops, on_top, stints, games, ids, pad)
    return out
