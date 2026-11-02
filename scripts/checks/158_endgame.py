"""Check 158: do players really play better or worse than their rating at the end of close games?

The owner, 2026-11-02: leverage-adjust, "only adjusting for the leverage, not how they played there UNLESS it's
statistically significant", "smart about sample sizes".  The wins over starter average (src/openrapm/wins.py) weight
these possessions by their leverage at each player's normal rating.  This check decides whether a player's own play
there should enter at all.

Words used below
    end of game   the swap test's clutch stints: the last five minutes of the fourth or overtime, margin within five.
                  About 3% of possessions, about 150 a season for a starter
    miss          a stint's actual margin for the home team less its expected margin, per 100 possessions a team.  The
                  expected points are read at each end from that end's own possessions: the season's points per 100
                  plus the five on offence's rating_off minus the five on defence's rating_def.  Late in a game a coach
                  subs a shooter in for one trip and a stopper for the next, so many stints have the ball at one end
                  only; read as a margin per possession those show a shooter as a star and a stopper as a disaster, and
                  a first version that did read z +8.1 between half A and half B, Korver and Lou Williams against Bowen
                  and Ben Wallace, with no end-of-game skill in it
    rest          the same ridge as below on the stints of the same games that are not end of game, taken off the
                  misses first: the published ratings are shrunk, so a good player's misses are positive on any stints
                  at all, which is how far his rating is from his season, not how he plays at the end of a game.  The
                  rest of half A's games is taken off half A's end-of-game stints and half B's off half B's, so no noise
                  is shared between the two
    deviation     a player's coefficient in a ridge regression of the misses less the rest, on the end-of-game stints,
                  on the ten players, +1 at home and -1 away, weighted by possessions, plus a free home edge, pulled to
                  0 by `PENALTY`
    standard error  the coefficient's standard error as a statistic of the sample,
                  sqrt(sigma^2 [M^-1 G M^-1]_jj): with no real deviations about 5% of players sit beyond 2, whatever the
                  penalty.  Not the posterior standard deviation, which is wider by the penalty and would put almost
                  nobody beyond 2
    shuffled      the same fit on as many stints of each game, drawn at random: every game's count of stints is kept,
                  and no stint is special
    flipped       the same fit on the real end-of-game stints with each stint's miss given a random sign: every lineup
                  and possession is kept, and no player can have a real deviation
    repeat        the correlation across players, weighted by possessions, of deviations from two sets of games that
                  share none: half A with half B of one season, and one season with the next.  Noise cannot repeat

The rule.  A player's own end-of-game play enters only if all three hold: over 30 seasons more players sit beyond 2
standard errors than in both controls, by more than 3 of the controls' own standard deviations; half A repeats in half B
at z above 2 and above the shuffled stints' repeat; and a season repeats in the next at z above 2.  Otherwise it stays
out, and every player is valued at his rating.  The same three were set before the first reading; the possessions at
each end, the rest and the shuffled repeat were added after it, because of what it showed (notes/DECISIONS.md, "WOSA").

Usage: .venv/Scripts/python scripts/checks/158_endgame.py --tag first
writes outputs/csv/endgame_<tag>_seasons.csv, _players.csv, _controls.csv and _verdict.csv.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from openrapm import swap_test, wins
from openrapm.commands.console import one_thread, use_utf8
from openrapm.data.config import load_config, resolve
from openrapm.data.seasons import last_finished
from openrapm.data.stints import load_season_stints
from openrapm.settings import PRODUCT

# A light penalty, as many possessions' worth: a starter with 150 end-of-game possessions keeps 60% of his raw
# deviation, so the standard error measures the sample (about 10 per 100) and not the penalty.  A penalty near the
# ratings' own 13,037 would leave everyone within 2 standard errors by construction.
PENALTY = 100.0
OTHER_PENALTIES = (30.0, 300.0)
MIN_POSSESSIONS = 50.0            # a player enters the counts with this many end-of-game possessions
MIN_REPEAT_POSSESSIONS = 25.0     # and the repeats with this many in each of the two sets of games
DRAWS = 10                        # draws of each control per season
Z_LINES = (2.0, 3.0)


def end_of_game(stints: pd.DataFrame) -> np.ndarray:
    """The swap test's clutch rule on a season's stints."""
    return ((stints.period.to_numpy() >= swap_test.CLUTCH_PERIOD)
            & (stints.start_clock.to_numpy(dtype=float) <= swap_test.CLUTCH_SECONDS)
            & (np.abs(stints.margin_h.to_numpy(dtype=float)) <= swap_test.CLUTCH_MARGIN))


def stint_misses(stints: pd.DataFrame, ratings: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(each stint's miss, its weight: the possessions per team).  A stint without possessions weighs nothing.

    Example: a season at 110 per 100, ten average players, and one home trip that scores 2: expected 1.1 at that end and
    nothing at the other, a miss of +0.9 points on half a possession a team, +180 per 100 on a weight of 0.5."""
    player_ids, home, away = wins.player_index(stints)
    aligned = ratings.set_index("player_id").reindex(player_ids).fillna(0.0)
    offense, defense = aligned.rating_off.to_numpy(dtype=float), aligned.rating_def.to_numpy(dtype=float)
    poss_h, poss_a = stints.poss_h.to_numpy(dtype=float), stints.poss_a.to_numpy(dtype=float)
    pts_h, pts_a = stints.pts_h.to_numpy(dtype=float), stints.pts_a.to_numpy(dtype=float)
    level = 100.0 * (pts_h.sum() + pts_a.sum()) / (poss_h.sum() + poss_a.sum())
    home_points = (level + np.asarray(home @ offense) - np.asarray(away @ defense)) * poss_h / 100.0
    away_points = (level + np.asarray(away @ offense) - np.asarray(home @ defense)) * poss_a / 100.0
    possessions = (poss_h + poss_a) / 2.0
    miss = np.divide(100.0 * ((pts_h - home_points) - (pts_a - away_points)), possessions, out=np.zeros(len(stints)),
                     where=possessions > 0.0)
    return miss, possessions


def deviations(signed, at_home: np.ndarray, miss: np.ndarray, weight: np.ndarray, chosen: np.ndarray,
               penalty: float) -> pd.DataFrame:
    """Per player: deviation, standard_error and possessions on the `chosen` stints.  `signed` is stints x players, +1
    for the home five and -1 for the away five, and the free home edge is the last coefficient."""
    players = signed[chosen]
    home = at_home[chosen].astype(float)[:, None]
    weights = weight[chosen]
    weighted = players.T.multiply(weights)
    gram = np.block([[np.asarray((weighted @ players).todense()), np.asarray(weighted @ home)],
                     [np.asarray(weighted @ home).T, np.array([[float((home[:, 0] ** 2 * weights).sum())]])]])
    right_hand_side = np.concatenate([np.asarray(weighted @ miss[chosen]).ravel(),
                                      [float((home[:, 0] * weights * miss[chosen]).sum())]])
    n_players = players.shape[1]
    matrix = gram + np.diag(np.concatenate([np.full(n_players, float(penalty)), [1e-9]]))
    inverse = np.linalg.inv(matrix)
    theta = inverse @ right_hand_side
    fitted = np.asarray(players @ theta[:n_players]).ravel() + home[:, 0] * theta[n_players]
    sigma_squared = float(np.sum(weights * (miss[chosen] - fitted) ** 2) / max(int(chosen.sum()), 1))
    possessions = np.asarray(abs(players).T @ weights).ravel()
    variance = np.clip(np.einsum("ij,jk,ki->i", inverse, gram, inverse)[:n_players], 0.0, None)
    return pd.DataFrame({"deviation": theta[:n_players], "standard_error": np.sqrt(sigma_squared * variance),
                         "possessions": possessions})


def shuffled_flag(games: np.ndarray, flag: np.ndarray, rng) -> np.ndarray:
    """As many stints of each game as `flag` marks there, drawn at random."""
    order = rng.permutation(len(flag))
    frame = pd.DataFrame({"game": games[order], "flag": flag[order]})
    wanted = frame.groupby("game").flag.transform("sum").to_numpy()
    rank = frame.groupby("game").cumcount().to_numpy()
    drawn = np.zeros(len(flag), dtype=bool)
    drawn[order] = rank < wanted
    return drawn


def beyond(table: pd.DataFrame, minimum: float = MIN_POSSESSIONS) -> dict:
    """How many players with `minimum` possessions sit beyond each line, and how many there are."""
    shown = table[table.possessions >= minimum]
    z = (shown.deviation / shown.standard_error).abs()
    return {"players": len(shown), **{f"beyond_{line:g}": int((z > line).sum()) for line in Z_LINES},
            "mean_z_squared": float((z ** 2).mean()) if len(shown) else np.nan}


def repeat_table(first: pd.DataFrame, second: pd.DataFrame, minimum: float = MIN_REPEAT_POSSESSIONS) -> pd.DataFrame:
    """The players with `minimum` possessions in each set: both deviations and a weight, the harmonic possessions."""
    both = first.join(second, how="inner", lsuffix="_one", rsuffix="_two")
    both = both[(both.possessions_one >= minimum) & (both.possessions_two >= minimum)]
    return pd.DataFrame({"one": both.deviation_one, "two": both.deviation_two,
                         "weight": both.possessions_one * both.possessions_two
                         / (both.possessions_one + both.possessions_two)})


def repeat_reading(pairs: pd.DataFrame) -> dict:
    """The weighted correlation of a table of `repeat_table` rows, its z (Fisher, on the effective number of players),
    and the real part's standard deviation in points per 100 at this penalty: the root of the positive covariance."""
    weight = pairs.weight.to_numpy()
    one, two = pairs.one.to_numpy(), pairs.two.to_numpy()
    one, two = one - np.average(one, weights=weight), two - np.average(two, weights=weight)
    covariance = float(np.average(one * two, weights=weight))
    spreads = np.average(one ** 2, weights=weight) * np.average(two ** 2, weights=weight)
    correlation = covariance / float(np.sqrt(spreads))
    effective = float(weight.sum() ** 2 / (weight ** 2).sum())
    return dict(pairs=len(pairs), correlation=correlation,
                z=float(np.arctanh(correlation) * np.sqrt(max(effective - 3.0, 1.0))),
                real_part_sd=float(np.sqrt(max(covariance, 0.0))))


def season_parts(cfg: dict, season: int, ratings: pd.DataFrame) -> dict:
    """What every fit of a season reads."""
    stints = swap_test.order_stints(load_season_stints(season, cfg, wins.STINT_COLUMNS, phases=PRODUCT.wins.phases))
    player_ids, home, away = wins.player_index(stints)
    miss, weight = stint_misses(stints, ratings)
    return dict(stints=stints, player_ids=player_ids, signed=(home - away).tocsr(), miss=miss, weight=weight,
                at_home=~stints.neutral.fillna(False).to_numpy(dtype=bool), flag=end_of_game(stints),
                games=stints.game_id.to_numpy(), half=stints.half.to_numpy())


def fit_on(parts: dict, chosen: np.ndarray, penalty: float = PENALTY, miss=None) -> pd.DataFrame:
    table = deviations(parts["signed"], parts["at_home"], parts["miss"] if miss is None else miss, parts["weight"],
                       chosen, penalty)
    return table.set_index(parts["player_ids"])


def beyond_the_rest(parts: dict, chosen: np.ndarray, games: np.ndarray, penalty: float = PENALTY) -> np.ndarray:
    """The misses less each lineup's deviations on the stints of `games` that are not `chosen`."""
    rest = fit_on(parts, games & ~chosen, penalty).deviation.to_numpy()
    return parts["miss"] - np.asarray(parts["signed"] @ rest).ravel()


def fit_beyond_the_rest(parts: dict, chosen: np.ndarray, games: np.ndarray, penalty: float = PENALTY) -> pd.DataFrame:
    """The deviations on the `chosen` stints of `games`, beyond what those games' other stints say of each player."""
    return fit_on(parts, chosen & games, penalty, beyond_the_rest(parts, chosen, games, penalty))


def half_pairs(parts: dict, chosen: np.ndarray, minimum: float = MIN_REPEAT_POSSESSIONS) -> pd.DataFrame:
    """Half A's deviations beside half B's, each beyond the rest of its own games."""
    half_a, half_b = parts["half"] == "A", parts["half"] == "B"
    return repeat_table(fit_beyond_the_rest(parts, chosen, half_a), fit_beyond_the_rest(parts, chosen, half_b), minimum)


def plain_half_pairs(parts: dict, chosen: np.ndarray) -> pd.DataFrame:
    """The same with nothing taken off: the first version's reading, kept to show what it measured."""
    half_a, half_b = parts["half"] == "A", parts["half"] == "B"
    return repeat_table(fit_on(parts, chosen & half_a), fit_on(parts, chosen & half_b))


def control_counts(parts: dict, rng) -> list:
    """`DRAWS` draws of each control, one row of counts a draw."""
    rows, every_game = [], np.ones(len(parts["miss"]), dtype=bool)
    misses = beyond_the_rest(parts, parts["flag"], every_game)
    for draw in range(DRAWS):
        drawn = shuffled_flag(parts["games"], parts["flag"], rng)
        rows.append(dict(control="shuffled", draw=draw, **beyond(fit_beyond_the_rest(parts, drawn, every_game))))
        signs = rng.choice([-1.0, 1.0], len(parts["miss"]))
        flipped = fit_on(parts, parts["flag"], miss=misses * signs)
        rows.append(dict(control="flipped", draw=draw, **beyond(flipped)))
    return rows


def season_reading(parts: dict, season: int, rng) -> dict:
    """One season: its line, its players, its controls' counts, half A against half B on the end-of-game stints and on
    shuffled ones, each with the rest taken off and with nothing taken off, and its deviations to set beside the next
    season's."""
    every_game = np.ones(len(parts["miss"]), dtype=bool)
    table = fit_beyond_the_rest(parts, parts["flag"], every_game)
    line = dict(season=season, end_of_game_share=float(parts["weight"][parts["flag"]].sum() / parts["weight"].sum()),
                **beyond(table))
    for penalty in OTHER_PENALTIES:
        line[f"beyond_2_at_penalty_{penalty:g}"] = beyond(fit_beyond_the_rest(parts, parts["flag"], every_game,
                                                                               penalty))["beyond_2"]
    shuffled = shuffled_flag(parts["games"], parts["flag"], rng)
    return dict(line=line, players=table.reset_index(names="player_id").assign(season=season),
                controls=[dict(season=season, **row) for row in control_counts(parts, rng)],
                halves=half_pairs(parts, parts["flag"]), shuffled_halves=half_pairs(parts, shuffled),
                plain_halves=plain_half_pairs(parts, parts["flag"]),
                plain_shuffled_halves=plain_half_pairs(parts, shuffled), whole=table)


def verdict(lines: pd.DataFrame, controls: pd.DataFrame, halves: pd.DataFrame, nexts: pd.DataFrame,
            shuffled_halves: pd.DataFrame) -> dict:
    """The rule at the top of this file on the 30 seasons' readings."""
    totals = controls.groupby(["control", "draw"]).beyond_2.sum().unstack("control")
    real = int(lines.beyond_2.sum())
    excess = {name: float((real - totals[name].mean()) / totals[name].std()) for name in totals.columns}
    half_reading, next_reading = repeat_reading(halves), repeat_reading(nexts)
    shuffled_reading = repeat_reading(shuffled_halves)
    passes = (min(excess.values()) > 3.0 and half_reading["z"] > max(2.0, shuffled_reading["z"])
              and next_reading["z"] > 2.0)
    return dict(shuffled_half_correlation=shuffled_reading["correlation"], shuffled_half_z=shuffled_reading["z"],
                real_beyond_2=real, real_beyond_3=int(lines.beyond_3.sum()), players=int(lines.players.sum()),
                shuffled_beyond_2=float(totals.shuffled.mean()), flipped_beyond_2=float(totals.flipped.mean()),
                sds_above_shuffled=excess["shuffled"], sds_above_flipped=excess["flipped"],
                half_correlation=half_reading["correlation"], half_z=half_reading["z"],
                half_real_part_sd=half_reading["real_part_sd"], next_correlation=next_reading["correlation"],
                next_z=next_reading["z"], next_real_part_sd=next_reading["real_part_sd"], passes=passes)


def run(cfg: dict, seasons: list, table: pd.DataFrame, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    readings, nexts, before = [], [], None
    for season in seasons:
        reading = season_reading(season_parts(cfg, season, table[table.season == season]), season, rng)
        readings.append(reading)
        if before is not None:
            nexts.append(repeat_table(before, reading["whole"], MIN_POSSESSIONS))
        before = reading["whole"]
        line = reading["line"]
        print(f"  {season}: {line['players']} players with {MIN_POSSESSIONS:g}+, beyond 2 standard errors "
              f"{line['beyond_2']}, beyond 3 {line['beyond_3']}, mean z squared {line['mean_z_squared']:.2f}",
              flush=True)
    joined = {key: pd.concat([reading[key] for reading in readings], ignore_index=True)
              for key in ("players", "halves", "shuffled_halves", "plain_halves", "plain_shuffled_halves")}
    return dict(lines=pd.DataFrame([reading["line"] for reading in readings]),
                controls=pd.DataFrame([row for reading in readings for row in reading["controls"]]),
                nexts=pd.concat(nexts, ignore_index=True), **joined)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Do players play differently from their rating late in close games?")
    parser.add_argument("--tag", required=True, help="the outputs are outputs/csv/endgame_<TAG>_*.csv")
    parser.add_argument("--table", default="season_ratings_product", help="the ratings table the misses read")
    parser.add_argument("--first", type=int, default=1997)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    use_utf8()
    args = parse_args(argv)
    cfg = load_config()
    table = pd.read_parquet(resolve(cfg, "outputs") / f"{args.table}.parquet",
                            columns=["season", "player_id", "player_name", "rating_off", "rating_def"])
    with one_thread():
        found = run(cfg, list(range(args.first, last_finished(cfg) + 1)), table)
    folder = resolve(cfg, "outputs") / "csv"
    folder.mkdir(parents=True, exist_ok=True)
    stem = folder / f"endgame_{args.tag}"
    reading = verdict(found["lines"], found["controls"], found["halves"], found["nexts"], found["shuffled_halves"])
    plain, plain_shuffled = repeat_reading(found["plain_halves"]), repeat_reading(found["plain_shuffled_halves"])
    reading.update(plain_half_correlation=plain["correlation"], plain_half_z=plain["z"],
                   plain_shuffled_half_correlation=plain_shuffled["correlation"],
                   plain_shuffled_half_z=plain_shuffled["z"])
    found["lines"].to_csv(f"{stem}_seasons.csv", index=False)
    names = table[["season", "player_id", "player_name"]]
    found["players"].merge(names, on=["season", "player_id"], how="left").to_csv(f"{stem}_players.csv", index=False)
    found["controls"].to_csv(f"{stem}_controls.csv", index=False)
    pd.Series(reading).to_csv(f"{stem}_verdict.csv", header=["value"])
    print(pd.Series(reading).to_string(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
