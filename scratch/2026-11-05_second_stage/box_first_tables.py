"""Read-only on the published tables: writes season_ratings_NAME_boxfirst{,_step3,_step4}.parquet, the same tables with only
box_adj_off and box_adj_def changed to the split learned first, without a rebuild.

Today's table already holds what the split needs.  Box part = prior - Box Adj = scale x b x box-score prior, on both ends, so
both box-score priors come back from the step-3 table up to a constant, and a team-held split ignores a constant; the box rates,
role and body come from the panel.  What moves between the columns is hold_team_totals(first stage's line, anchor) - anchor.

    python box_first_tables.py [NAME]
"""
import sys

import numpy as np
import pandas as pd

from openrapm import prior_shrink, second_prior as sp
from openrapm.commands.build import season_lineup_rows
from openrapm.data.config import load_config
from openrapm.data.roles import main_teams
from openrapm.data.seasons import trainable
from openrapm.data.second_prior_files import line_setup, load_grams
from openrapm.settings import PRODUCT

sys.stdout.reconfigure(encoding="utf-8")
NAME = sys.argv[1] if len(sys.argv) > 1 else "product"
cfg, settings = load_config(), PRODUCT
prior = settings.prior
names = sp.INPUT_SETS[prior.second_prior_inputs]
TAG = prior.second_prior
type_scale = line_setup(cfg, TAG, names)["type_scale"]
grams = load_grams(cfg, TAG)
finished = set(trainable(cfg).seasons)
step3 = pd.read_parquet(f"outputs/season_ratings_{NAME}_step3.parquet")
step4 = pd.read_parquet(f"outputs/season_ratings_{NAME}_step4.parquet")
final = pd.read_parquet(f"outputs/season_ratings_{NAME}.parquet")
multipliers = pd.read_csv(f"outputs/csv/season_ratings_{NAME}_step4_multipliers.csv").set_index("season")
teams = main_teams(cfg).set_index(["player_id", "season"]).team_id
panel = pd.read_parquet("outputs/season_panel.parquet")
OWN = [names[at] for at in sp.box_score_positions(prior.second_prior_inputs)]
out3, out4, worst = [], [], dict(old_step4=0.0, team_total=0.0)
for season, rows in step3.groupby("season", sort=True):
    rows = rows.reset_index(drop=True)
    ids = rows.player_id.to_numpy(np.int64)
    args = (grams, int(season), finished, settings.exclude_neighbours, prior.second_prior_folds, prior.second_prior_penalty)
    line = sp.season_coefficients(*args, prior.second_prior_nearest)
    first = sp.season_first_stage(*args, prior.second_prior_nearest, prior.second_prior_inputs)
    s = {"O": float(rows.prior_scale_off.iloc[0]), "D": float(rows.prior_scale_def.iloc[0])}
    b = {side: float(line[side][-1, 0]) for side in "OD"}
    box = {"O": pd.Series((rows.prior_off - rows.box_adj_off).to_numpy() / (s["O"] * b["O"]), index=ids),
           "D": pd.Series(-(rows.prior_def - rows.box_adj_def).to_numpy() / (s["D"] * b["D"]), index=ids)}
    stints = season_lineup_rows(cfg, int(season))
    types = sp.standardised_types(panel[(panel.season == season) & (panel.side == "O")], type_scale)
    moved = {}
    for side, other in (("O", "D"), ("D", "O")):
        frame = pd.DataFrame({sp.BOX: box[side], sp.OTHER_BOX: box[other]}, index=ids).join(types, how="left")
        frame[sp.TYPES] = frame[sp.TYPES].fillna(0.0)
        weights = first[side][-1][sp.box_score_positions(prior.second_prior_inputs)]
        own_line = pd.Series(frame[OWN].to_numpy(float) @ weights, index=ids)
        anchor = pd.Series(b[side] * box[side].to_numpy(), index=ids)
        team_poss = sp.team_possessions(stints, side)
        moved[side] = sp.hold_team_totals(own_line, anchor, team_poss) - anchor
        held = moved[side].reindex(team_poss.player_id).to_numpy() * team_poss.poss.to_numpy()
        totals = pd.Series(held).groupby(team_poss.team.to_numpy()).sum().abs().max() / team_poss.groupby("team").poss.sum().max()
        worst["team_total"] = max(worst["team_total"], float(totals))
    new = rows.copy()
    new["box_adj_off"] = rows.box_adj_off - s["O"] * moved["O"].to_numpy()
    new["box_adj_def"] = rows.box_adj_def + s["D"] * moved["D"].to_numpy()
    out3.append(new)
    # step 4's layers on Box Adj, checked first on today's column
    m = multipliers.loc[season]
    key = pd.MultiIndex.from_arrays([ids, np.full(len(ids), int(season))])
    team = np.array(teams.reindex(key).to_numpy(float), dtype=float)
    alone = np.isnan(team)
    team[alone] = -1.0 - np.arange(int(alone.sum()))

    def layered(column, source, weight, team_m, within_m):
        mean, deviation = prior_shrink.team_split(team, source[column].to_numpy(float), source[weight].to_numpy(float))
        return pd.Series(team_m * mean + within_m * deviation, index=ids)

    shrunk = step4[step4.season == season].set_index("player_id").reindex(ids)
    for column, weight, team_m, within_m in (("box_adj_off", "poss_off", m.multiplier_off, m.within_multiplier_off), ("box_adj_def", "poss_def", m.multiplier_def, m.within_multiplier_def)):
        gap = (layered(column, rows, weight, team_m, within_m) - shrunk[column]).abs().max()
        worst["old_step4"] = max(worst["old_step4"], float(gap))
        shrunk[column] = layered(column, new, weight, team_m, within_m)
    out4.append(shrunk.reset_index())
    print(season, {k: f"{v:.1e}" for k, v in worst.items()}, flush=True)

new3, new4 = pd.concat(out3, ignore_index=True)[list(step3.columns)], pd.concat(out4, ignore_index=True)[list(step4.columns)]
lookup = new4.set_index(["season", "player_id"])
new_final = final.copy()
key = pd.MultiIndex.from_arrays([final.season, final.player_id])
new_final["box_adj_off"], new_final["box_adj_def"] = lookup.box_adj_off.reindex(key).to_numpy(), lookup.box_adj_def.reindex(key).to_numpy()
print("final's Box Adj against step 4's, largest gap on today's columns:", float((final.box_adj_off - step4.set_index(["season", "player_id"]).box_adj_off.reindex(key).to_numpy()).abs().max()))
for label, frame, old in (("_step3", new3, step3), ("_step4", new4, step4), ("", new_final, final)):
    frame.to_parquet(f"outputs/season_ratings_{NAME}_boxfirst{label}.parquet", index=False)
    other = [c for c in old.columns if c not in ("box_adj_off", "box_adj_def")]
    a, b_ = frame.sort_values(["season", "player_id"]).reset_index(drop=True), old.sort_values(["season", "player_id"]).reset_index(drop=True)
    numeric = [c for c in other if pd.api.types.is_numeric_dtype(old[c])]
    print(f"{label or '_final'}: rows {len(frame)}, largest change in any other number {float((a[numeric] - b_[numeric]).abs().max().max())}")
print("wrote", NAME, "boxfirst tables; published tables untouched")
