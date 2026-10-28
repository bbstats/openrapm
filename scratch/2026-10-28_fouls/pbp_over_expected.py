"""Each proxy as J minus the J expected from K, per 100 possessions, the expectation a possession-weighted line
fitted per season over every player.  Reads how stable each is from one season to the next among players with
1,500+ possessions, and how it lines up with the games' part of the published rating, the part the box-score prior
missed.  Scratch check for a proposal."""
import sys

import numpy as np
import pandas as pd

S = sys.argv[0].rsplit("/", 1)[0].rsplit("\\", 1)[0]
m = pd.read_parquet(f"{S}/pbp_candidate_counts.parquet")
panel = pd.read_parquet("outputs/season_panel.parquet")
off = panel[panel.side == "O"][["player_id", "season", "possessions", "assists_per100_uncentred",
                                "turnovers_per100_uncentred", "twos_made_per100_uncentred",
                                "twos_missed_per100_uncentred", "threes_made_per100_uncentred",
                                "threes_missed_per100_uncentred", "free_throws_made_per100_uncentred",
                                "free_throws_missed_per100_uncentred"]].rename(columns={"possessions": "poss_o"})
dfn = panel[panel.side == "D"][["player_id", "season", "steals_per100_uncentred"]]
m = m.merge(off, on=["player_id", "season"]).merge(dfn, on=["player_id", "season"])
m["twos_attempted"] = m.twos_made_per100_uncentred + m.twos_missed_per100_uncentred
m["threes_attempted"] = m.threes_made_per100_uncentred + m.threes_missed_per100_uncentred
m["true_shooting_attempts"] = m.twos_attempted + m.threes_attempted + 0.44 * (
    m.free_throws_made_per100_uncentred + m.free_throws_missed_per100_uncentred)
# the event counts were per 100 defensive possessions; offensive ones go per 100 offensive possessions
for k in ["shots_blocked", "fouled_on_threes"]:
    m[k + "_per100"] = 100 * m[k] / m.poss_o
r = pd.read_parquet("outputs/season_ratings_creation110.parquet")[["player_id", "season", "u_off", "u_def"]]
m = m.merge(r, on=["player_id", "season"], how="left")

B, F, ST = "blocks_per100_uncentred", "fouls_per100_uncentred", "steals_per100_uncentred"
CANDIDATES = [  # name, side, J, K list
    ("goaltends", "D", "def_goaltends_per100", []),
    ("goaltends over expected from blocks", "D", "def_goaltends_per100", [B]),
    ("blocks over expected from fouls", "D", B, [F]),
    ("steals over expected from fouls", "D", ST, [F]),
    ("shooting fouls", "D", "shooting_fouls_per100", []),
    ("loose-ball fouls", "D", "loose_ball_fouls_per100", []),
    ("shots blocked over expected from attempts", "O", "shots_blocked_per100", ["twos_attempted", "threes_attempted"]),
    ("fouled on threes over expected from attempts", "O", "fouled_on_threes_per100", ["threes_attempted"]),
    ("assists over expected from turnovers", "O", "assists_per100_uncentred", ["turnovers_per100_uncentred"]),
    ("turnovers over expected from true-shooting attempts and assists", "O", "turnovers_per100_uncentred",
     ["true_shooting_attempts", "assists_per100_uncentred"]),
    ("turnovers, plain", "O", "turnovers_per100_uncentred", []),
]


def over_expected(g, j, ks, w):
    if not ks:
        return g[j] - np.average(g[j], weights=w)
    X = np.column_stack([np.ones(len(g))] + [g[k] for k in ks])
    sw = np.sqrt(w.to_numpy())
    b = np.linalg.lstsq(X * sw[:, None], g[j].to_numpy() * sw, rcond=None)[0]
    return g[j] - X @ b


out = []
for name, side, j, ks in CANDIDATES:
    w = m.possessions if side == "D" else m.poss_o
    m["x"] = 0.0
    for _, g in m.groupby("season"):
        m.loc[g.index, "x"] = over_expected(g, j, ks, w.loc[g.index])
    big = m[(m.possessions >= 1500)]
    nx = big[["player_id", "season", "x"]].assign(season=lambda d: d.season - 1)
    pr = big.merge(nx, on=["player_id", "season"], suffixes=("", "_next"))
    u = "u_def" if side == "D" else "u_off"
    out.append({"input": name, "side": side, "next_season": round(pr.x.corr(pr.x_next), 2),
                "with_games_part": round(big.x.corr(big[u]), 3), "players": len(big)})
pd.set_option("display.width", 200)
print(pd.DataFrame(out).to_string(index=False))
