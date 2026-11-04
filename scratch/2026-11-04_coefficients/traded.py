"""Read-only: how much of Box Adj's team rates acts through a traded player's mix of team rates rather than on/off."""
import sys

import numpy as np
import pandas as pd

from openrapm import second_prior as sp
from openrapm.commands.build import season_lineup_rows
from openrapm.data.config import load_config
from openrapm.data.seasons import trainable
from openrapm.data.second_prior_files import load_grams
from openrapm.settings import PRODUCT

sys.stdout.reconfigure(encoding="utf-8")
cfg, settings = load_config(), PRODUCT.prior
grams = load_grams(cfg, settings.second_prior)
finished = set(trainable(cfg).seasons)
names = sp.INPUT_SETS[settings.second_prior_inputs]
full = pd.read_parquet("outputs/season_ratings_product.parquet")
mult = pd.read_csv("outputs/csv/season_ratings_product_step4_multipliers.csv").set_index("season")
SEASONS = [int(x) for x in sys.argv[1].split(",")]
rows_all = []
for season in SEASONS:
    stints = season_lineup_rows(cfg, season)
    rates = sp.team_rate_numbers(stints, pad=True)
    line = sp.season_coefficients(grams, season, finished, 0, settings.second_prior_folds, settings.second_prior_penalty, settings.second_prior_nearest)
    table = full[full.season == season].set_index("player_id")
    for side, poss, scale_col, within, sign in (("O", "poss_off", "prior_scale_off", "within_multiplier_off", 1.0), ("D", "poss_def", "prior_scale_def", "within_multiplier_def", -1.0)):
        page = float(table[scale_col].iloc[0] * mult.loc[season, within])
        weights = dict(zip(names, line[side][-1]))
        team_poss = sp.team_possessions(stints, side)
        team_col = [c for c in team_poss.columns if c not in ("player_id", "poss")][0]
        n_teams = team_poss[team_poss.poss > 0].groupby("player_id")[team_col].nunique()
        main = team_poss.sort_values("poss").groupby("player_id").tail(1).set_index("player_id")
        frame = rates[side][sp.TEAM_RATES]
        contribution = pd.DataFrame({n: sign * page * weights[n] * frame[n] for n in sp.TEAM_RATES})
        teams = main[team_col].reindex(frame.index)
        p = main.poss.reindex(frame.index).fillna(0.0)
        inside = contribution.sub((contribution.mul(p, axis=0)).groupby(teams).transform("sum").div(p.groupby(teams).transform("sum"), axis=0), axis=0)
        total = inside.sum(axis=1)
        one = inside["team_off_rebounds_off"]
        d = pd.DataFrame({"season": season, "side": side, "total": total, "orb_off": one, "teams": n_teams.reindex(frame.index).fillna(1).astype(int),
                          "poss": table[poss].reindex(frame.index), "name": table.player_name.reindex(frame.index), "box_adj": table[f"box_adj_{'off' if side == 'O' else 'def'}"].reindex(frame.index)})
        rows_all.append(d[d.poss >= 500].reset_index())
d = pd.concat(rows_all, ignore_index=True)
d["traded"] = d.teams > 1
summary = d.groupby(["side", "traded"]).agg(players=("total", "size"), team_rates_sd=("total", "std"), team_rates_abs_90th=("total", lambda x: x.abs().quantile(.9)), orb_off_sd=("orb_off", "std"), box_adj_sd=("box_adj", "std")).round(3)
print(summary.to_string())
for season in SEASONS[-1:]:
    for side in ("D", "O"):
        s = d[(d.season == season) & (d.side == side)].copy()
        s["a"] = s.total.abs()
        print(f"\n{season} {side}: largest pull from the 16 team rates, among teammates; column orb_off is opponents' or team's offensive rebounds, player on bench")
        print(s.nlargest(12, "a")[["name", "teams", "poss", "total", "orb_off", "box_adj"]].round(2).to_string(index=False))
