"""Per player-season counts of play-by-play events not yet in the panel, their rates per 100 possessions, how
stable each rate is from one season to the next, and how it tracks blocks.  Scratch check for a proposal."""
import glob
import sys

import numpy as np
import pandas as pd

SEASONS = [int(s) for s in sys.argv[1:]] or [2022, 2023, 2024, 2025]
KINDS = {
    "def_goaltends": ("Violation", ["Defensive Goaltending"]),
    "off_goaltends": ("Turnover", ["Offensive Goaltending"]),
    "shooting_fouls": ("Foul", ["Shooting", "Shooting Block"]),
    "loose_ball_fouls": ("Foul", ["Loose Ball"]),
    "take_fouls": ("Foul", ["Personal Take", "Transition Take"]),
    "def_three_seconds": ("Foul", ["Defense 3 Second"]),
    "kicked_balls": ("Violation", ["Kicked Ball"]),
    "charges_committed": ("Foul", ["Offensive Charge"]),
    "travels": ("Turnover", ["Traveling"]),
    "off_three_seconds": ("Turnover", ["3 Second Violation"]),
    "fouled_on_threes": ("Free Throw", ["Free Throw 1 of 3"]),
}
COLS = ["personId", "actionType", "subType", "description"]

rows = []
for season in SEASONS:
    files = sorted(glob.glob(f"data/raw/pbp/{season}/002*.parquet"))
    counts = {}
    for f in files:
        d = pd.read_parquet(f, columns=COLS)
        for name, (atype, subs) in KINDS.items():
            hit = d.loc[(d.actionType == atype) & d.subType.isin(subs), "personId"]
            for pid, n in hit.value_counts().items():
                counts[(pid, name)] = counts.get((pid, name), 0) + n
        # the shooter of a miss whose next row is a BLOCK row
        nxt = d.description.shift(-1).fillna("")
        hit = d.loc[(d.actionType == "Missed Shot") & nxt.str.contains("BLOCK"), "personId"]
        for pid, n in hit.value_counts().items():
            counts[(pid, "shots_blocked")] = counts.get((pid, "shots_blocked"), 0) + n
    s = pd.Series(counts).rename("n").reset_index()
    s.columns = ["player_id", "kind", "n"]
    s["season"] = season
    rows.append(s)
    print(season, len(files), "games", file=sys.stderr)

c = pd.concat(rows).pivot_table(index=["player_id", "season"], columns="kind", values="n", fill_value=0).reset_index()
panel = pd.read_parquet("outputs/season_panel.parquet")
panel = panel[(panel.side == "D") & panel.season.isin(SEASONS)][
    ["player_id", "season", "possessions", "blocks_per100_uncentred", "fouls_per100_uncentred"]]
m = panel.merge(c, on=["player_id", "season"], how="left").fillna(0)
kinds = [*KINDS, "shots_blocked"]
for k in kinds:
    m[k + "_per100"] = 100 * m[k] / m.possessions

big = m[m.possessions >= 1500]
nxt = big.assign(season=big.season - 1)
pair = big.merge(nxt, on=["player_id", "season"], suffixes=("", "_next"))
out = []
for k in kinds:
    r = k + "_per100"
    league_per_season = m.groupby("season")[k].sum().mean()
    top = m[m.season == SEASONS[-1]].nlargest(1, k)
    out.append({"count": k, "league_a_season": round(league_per_season),
                "per100_typical": round(big[r].median(), 3), "per100_90th": round(big[r].quantile(0.9), 3),
                "next_season_corr": round(pair[r].corr(pair[r + "_next"]), 2),
                "corr_with_blocks": round(big[r].corr(big.blocks_per100_uncentred), 2),
                "leader_last_season": f"{int(top.player_id.iloc[0])}:{int(top[k].iloc[0])}"})
pd.set_option("display.width", 200)
print(pd.DataFrame(out).to_string(index=False))
print("pairs", len(pair), "blocks next-season corr",
      round(pair.blocks_per100_uncentred.corr(pair.blocks_per100_uncentred_next), 2),
      "fouls", round(pair.fouls_per100_uncentred.corr(pair.fouls_per100_uncentred_next), 2))
m.to_parquet(sys.argv[0].replace(".py", ".parquet"))
