"""Read-only: which of Box Adj's inputs moved 2026's players under experiment 141's fade, and by how much.

Box Adj's line for a rated season is a weight per input, 57 an end.  With a half-life of 8 the weights for 2026 lean on
the recent seasons, so each weight moves, and a player's prior moves by the sum over inputs of (his input as the line
applies it) x (the weight's move).  This prints

    1. the inputs whose weights moved most, per one standard deviation of the input among 2026's regulars, beside the
       weight the early seasons alone and the recent seasons alone would give;
    2. how much of each rating's move the player's own prior's move accounts for;
    3. for named players and for bigs and guards, the move split by groups of inputs.

A player's inputs are his folds' inputs averaged (outputs/second_prior/zones139/inputs_2026.parquet), moved as the line
moves them (`rating_line.prior_columns`) on the season's possessions.

    .venv/Scripts/python scratch/2026-11-09_improvements/fade_what_moved.py
"""
import json
import sys

import numpy as np
import pandas as pd

from openrapm import rating_line
from openrapm import second_prior as sp
from openrapm.commands.build import season_lineup_rows
from openrapm.data.config import load_config
from openrapm.data.seasons import trainable
from openrapm.data.second_prior_files import load_grams

sys.stdout.reconfigure(encoding="utf-8")
SEASON, HALF_LIFE, PENALTY = 2026, 8.0, 1e-4
NAMES = sp.INPUT_SETS["context_types_team_zones_centred"]
K = len(NAMES)
BOX_STATS = ["threes_made_per100", "threes_missed_per100", "twos_made_per100", "twos_missed_per100",
             "free_throws_made_per100", "free_throws_missed_per100", "off_rebounds_per100", "def_rebounds_per100",
             "assists_per100", "turnovers_per100", "steals_per100", "blocks_per100", "fouls_per100"]
GROUPS = {
    "his box-score estimate": ["box_prior"],
    "his other end's estimate": ["box_prior_other_end"],
    "teammates' and opponents' estimates": [name for name in NAMES if name.startswith(("teammates_", "opponents_"))],
    "his own box stats": BOX_STATS,
    "role: starts, share of possessions, garbage time": ["team_possession_share", "games_started_share",
                                                          "garbage_time_share"],
    "age, height, weight": ["age", "height_in", "weight_lb"],
    "team shooting with him on court": [name for name in NAMES if name.startswith("team_") and name.endswith("_on")],
    "team shooting with him off court": [name for name in NAMES if name.startswith("team_") and name.endswith("_off")],
}
assert sorted(sum(GROUPS.values(), [])) == sorted(NAMES)
PLAYERS = ["Wendell Carter Jr.", "Mark Williams", "Neemias Queta", "Jarrett Allen", "Moussa Diabaté",
           "Luka Dončić", "Anthony Edwards", "Nolan Traore", "Jared McCain", "Keon Ellis", "Victor Wembanyama"]


def line_weights(grams: dict, finished: set, half_life: float, seasons=None) -> dict:
    """{"O", "D"}: 2026's weights, one per input; from `seasons` only when given."""
    use = grams if seasons is None else {season: grams[season] for season in seasons}
    line = sp.season_coefficients(use, SEASON, finished, 1, 5, PENALTY, half_life=half_life)
    return {side: line[side][-1] for side in ("O", "D")}


def main() -> None:
    cfg = load_config()
    grams, finished = load_grams(cfg, "zones139_line"), set(trainable(cfg).seasons)
    plain, faded = line_weights(grams, finished, 0.0), line_weights(grams, finished, HALF_LIFE)
    early = line_weights(grams, finished, 0.0, range(1997, 2011))
    recent = line_weights(grams, finished, 0.0, range(2011, 2025))

    saved = pd.read_parquet(f"outputs/second_prior/zones139/inputs_{SEASON}.parquet")
    inputs = {side: saved[saved.side == side].groupby("player_id")[NAMES].mean() for side in ("O", "D")}
    stints = season_lineup_rows(cfg, SEASON)
    team_poss = {side: sp.team_possessions(stints, side) for side in ("O", "D")}
    ids = np.array(sorted(set(inputs["O"].index) | set(inputs["D"].index)), dtype=np.int64)
    applied = rating_line.prior_columns(inputs, ids, team_poss)
    n = len(ids)
    part = {"O": applied[:n, :K], "D": applied[n:, K:]}

    today = pd.read_parquet("outputs/season_ratings_zones139.parquet")
    new = pd.read_parquet("outputs/season_ratings_fade141.parquet")
    key = ["season", "player_id"]
    both = today[today.season == SEASON].merge(new[new.season == SEASON], on=key, suffixes=("", "_new"))
    both = both.set_index("player_id").reindex(ids)
    regular = (both.poss_season >= 2000).to_numpy()
    # the saved inputs hold height standardised: back to inches with the run's own mean and spread
    with open("outputs/second_prior/zones139_line/run.json", encoding="utf-8") as handle:
        middle, width = json.load(handle)["type_scale"]["height_in"]
    height = inputs["O"].height_in.reindex(ids).to_numpy() * width + middle

    # 1. the weights that moved most, in points of prior per standard deviation of the input among regulars
    rows = []
    for side, label in (("O", "offense"), ("D", "defense")):
        spread = part[side][regular].std(axis=0)
        for k, name in enumerate(NAMES):
            rows.append(dict(end=label, input=name, today=plain[side][k] * spread[k], faded=faded[side][k] * spread[k],
                             move=(faded[side][k] - plain[side][k]) * spread[k],
                             seasons_1997_2010=early[side][k] * spread[k],
                             seasons_2011_2024=recent[side][k] * spread[k]))
    weights = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print("the 12 weights that moved most, per one standard deviation of the input among 2026's regulars, each end")
    for label in ("offense", "defense"):
        table = weights[weights.end == label]
        most = table.reindex(table.move.abs().sort_values(ascending=False).index).head(12)
        print(most.round(3).to_string(index=False))

    # 2. each player's prior's move in the line's units, against his rating's move
    change = {side: part[side] * (faded[side] - plain[side]) for side in ("O", "D")}
    prior_move = {side: change[side].sum(axis=1) for side in ("O", "D")}
    passed = {}
    for side, column in (("O", "rating_off"), ("D", "rating_def")):
        moved = (both[f"{column}_new"] - both[column]).to_numpy()
        sign = 1.0                                          # both tables carry defense as positive good
        slope, level = np.polyfit(prior_move[side][regular], sign * moved[regular], 1)
        r = float(np.corrcoef(prior_move[side][regular], moved[regular])[0, 1])
        passed[side] = slope
        print(f"\n{column}: a regular's rating moves {slope:+.3f} per unit his own prior moves; correlation {r:+.3f}; "
              f"level {level:+.3f}")

    # 3. the move by groups of inputs, in rating points through that slope
    at = {name: [NAMES.index(member) for member in members] for name, members in GROUPS.items()}

    def split(mask: np.ndarray) -> pd.DataFrame:
        out = {}
        for name, columns in at.items():
            out[name] = (passed["O"] * change["O"][mask][:, columns].sum(axis=1).mean()
                         + passed["D"] * change["D"][mask][:, columns].sum(axis=1).mean())
        return pd.Series(out)

    names = both.player_name.to_numpy()
    table = {}
    for player in PLAYERS:
        mask = names == player
        if mask.any():
            series = split(mask)
            series["sum of the groups"] = series.sum()
            series["his rating's move"] = float((both.rating_total_new - both.rating_total).to_numpy()[mask][0])
            table[player] = series
    print("\nnamed players: the move by group of inputs, rating points, offense and defense together")
    print(pd.DataFrame(table).T.round(2).to_string())
    print()
    print("named players: the five single inputs that moved each most, rating points, by end")
    moved_off = (both.rating_off_new - both.rating_off).to_numpy()
    moved_def = (both.rating_def_new - both.rating_def).to_numpy()
    for player in PLAYERS:
        mask = names == player
        if not mask.any():
            continue
        one = pd.concat([pd.Series(passed["O"] * change["O"][mask][0], index=[f"offense: {name}" for name in NAMES]),
                         pd.Series(passed["D"] * change["D"][mask][0], index=[f"defense: {name}" for name in NAMES])])
        top = one.reindex(one.abs().sort_values(ascending=False).index).head(5)
        print(f"  {player}: " + "; ".join(f"{name} {value:+.2f}" for name, value in top.items()))
        print(f"      his own inputs say offense {passed['O'] * prior_move['O'][mask][0]:+.2f}, defense "
              f"{passed['D'] * prior_move['D'][mask][0]:+.2f}; the tables moved offense {moved_off[mask][0]:+.2f}, "
              f"defense {moved_def[mask][0]:+.2f}")

    groups = {"regulars 6 ft 10 and taller": regular & (height >= 82),
              "regulars 6 ft 7 to 6 ft 9": regular & (height >= 79) & (height < 82),
              "regulars 6 ft 4 to 6 ft 6": regular & (height >= 76) & (height < 79),
              "regulars under 6 ft 4": regular & (height < 76)}
    table = {}
    for name, mask in groups.items():
        series = split(mask)
        series["sum of the groups"] = series.sum()
        series["mean rating move"] = float((both.rating_total_new - both.rating_total).to_numpy()[mask].mean())
        series["players"] = int(mask.sum())
        table[name] = series
    print("\nby height: the mean move by group of inputs, rating points")
    print(pd.DataFrame(table).T.round(3).to_string())

    # the same moves by four drivers, and what the player's own inputs leave: his teammates' priors moved too, and
    # the ridge splits a lineup's results between them again
    drivers = {"rebounds": ["off_rebounds_per100", "def_rebounds_per100"],
               "threes taken": ["threes_missed_per100", "threes_made_per100"],
               "the other end's estimate": ["box_prior_other_end"],
               "assists and twos made": ["assists_per100", "twos_made_per100"]}
    both_ends = passed["O"] * change["O"] + passed["D"] * change["D"]
    total_move = (both.rating_total_new - both.rating_total).to_numpy()

    def driven(mask: np.ndarray) -> dict:
        out = {name: float(both_ends[mask][:, [NAMES.index(member) for member in members]].sum(axis=1).mean())
               for name, members in drivers.items()}
        own = float(both_ends[mask].sum(axis=1).mean())
        out["his other inputs"] = own - sum(out.values())
        out["not his own inputs"] = float(total_move[mask].mean()) - own
        out["rating move"] = float(total_move[mask].mean())
        return out

    print("\nthe move by driver, rating points")
    table = {player: driven(names == player) for player in PLAYERS if (names == player).any()}
    table.update({name: driven(mask) for name, mask in groups.items()})
    print(pd.DataFrame(table).T.round(2).to_string())

    # the single inputs behind the tall players' move
    tall = regular & (height >= 82)
    small = regular & (height < 76)
    single = pd.DataFrame({
        "tall": passed["O"] * change["O"][tall].mean(axis=0) + passed["D"] * change["D"][tall].mean(axis=0),
        "small": passed["O"] * change["O"][small].mean(axis=0) + passed["D"] * change["D"][small].mean(axis=0),
        "tall, offense": passed["O"] * change["O"][tall].mean(axis=0),
        "tall, defense": passed["D"] * change["D"][tall].mean(axis=0)}, index=NAMES)
    single["gap"] = single.tall - single.small
    print("\nthe 14 inputs that separate tall regulars from small ones most, mean move in rating points")
    print(single.reindex(single.gap.abs().sort_values(ascending=False).index).head(14).round(3).to_string())


if __name__ == "__main__":
    main()
