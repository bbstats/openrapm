"""The top-fifth lines of the count tables in 2026 and 2006, with players just above each line, for the report."""
import numpy as np
import pandas as pd
from openrapm.data.config import load_config, resolve

cfg = load_config()
names = pd.read_parquet(resolve(cfg, "outputs") / "season_ratings_product.parquet")[["season", "player_id", "player_name"]]
pick = {"O": ["self_creation_per100", "used_possessions_per100", "blocks_per100_uncentred"],
        "D": ["height_in", "off_rebounds_per100_uncentred", "transition_attempts_per100_uncentred"]}
for season in (2006, 2026):
    p = pd.read_parquet(f"outputs/synergy/rd143/players_{season}.parquet")
    p = p[(p.repeat == 0) & (p.fold == 0)]
    for side, traits in pick.items():
        rows = p[p.side == side]
        poss = rows["poss_off" if side == "O" else "poss_def"].to_numpy(float)
        rows, poss = rows[poss > 0], poss[poss > 0]
        for t in traits:
            v = rows[t].to_numpy(float)
            order = np.argsort(v)
            mid = (np.cumsum(poss[order]) - 0.5 * poss[order]) / poss.sum()
            line = float(np.interp(0.8, mid, v[order]))
            near = rows.assign(v=v, poss=poss)
            near = near[(near.v >= line) & (near.poss >= 1500)].sort_values("v").head(4)
            near = near.merge(names[names.season == season], on="player_id", how="left")
            print(season, side, t, round(line, 2), "| just above:", ", ".join(f"{n} {x:.1f}" for n, x in zip(near.player_name, near.v)))
