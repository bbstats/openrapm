"""Read-only: how hard Box Adj's rates are padded in 2026, with real examples."""
import sys

import numpy as np
import pandas as pd

from openrapm import second_prior as sp
from openrapm.commands.build import season_lineup_rows
from openrapm.data.config import load_config

sys.stdout.reconfigure(encoding="utf-8")
cfg = load_config()
SEASON = 2026
stints = season_lineup_rows(cfg, SEASON)
ids = np.unique(np.concatenate([stints[sp.OFF].to_numpy(dtype=np.int64).ravel(), stints[sp.DEF].to_numpy(dtype=np.int64).ravel()]))
rows = sp._Entries(stints, ids)
table = pd.read_parquet("outputs/season_ratings_product.parquet")
table = table[table.season == SEASON].set_index("player_id")
panel = pd.read_parquet("outputs/season_panel.parquet")
panel = panel[(panel.season == SEASON) & (panel.side == "O")].set_index("player_id")
regular = table.index[table.poss_off >= 1000]
index = pd.Index(ids)
at = {int(p): i for i, p in enumerate(ids)}
n_rows = len(stints)
counts = (*sp.TEAM_COUNTS, *sp.TEAM_TAKEN)
out, examples = [], {}
for side in ("O", "D"):
    counted = np.zeros((2 * n_rows, len(counts)))
    weight = np.zeros(2 * n_rows)
    entries = slice(0, n_rows) if side == "O" else slice(n_rows, 2 * n_rows)
    counted[entries] = 100.0 * sp._counts_read(stints)
    weight[entries] = stints.poss.to_numpy(dtype=float)
    team_weight = np.asarray(rows.teams.T @ weight).ravel()
    team_rate = np.asarray(rows.teams.T @ counted) / np.where(team_weight > 0, team_weight, 1.0)[:, None]
    level = np.asarray(rows.mix @ team_rate) / np.where(rows.on_weight > 0, rows.on_weight, 1.0)[:, None]
    for where, total in (("on", rows.on), ("off", rows.bench)):
        behind = np.maximum(total(weight), 0.0)
        rate = total(counted) / np.where(behind > 0, behind, 1.0)[:, None]
        spread = (total(counted * counted) - 2.0 * rate * total(counted * weight[:, None]) + rate * rate * total(weight * weight)[:, None])
        noise = np.maximum(spread, 0.0).sum(axis=0) / max(float(behind.sum()), 1e-12)
        played = behind > 0
        for j, count in enumerate(counts):
            if count in sp.TEAM_TAKEN:
                continue
            tau2 = float(np.average((rate[played, j] - level[played, j]) ** 2, weights=behind[played]) - noise[j] * np.average(1.0 / behind[played], weights=behind[played]))
            k = noise[j] / tau2 if tau2 > 1e-12 else np.inf
            reg = np.array([at[int(p)] for p in regular if int(p) in at])
            kept = behind[reg] / (behind[reg] + k) if np.isfinite(k) else np.zeros(len(reg))
            out.append(dict(side=side, where=where, count=count, k=k, median_possessions=float(np.median(behind[reg])), kept_median=float(np.median(kept))))
            examples[(side, where, count)] = (rate[:, j], level[:, j], behind, k)
t = pd.DataFrame(out)
pd.set_option("display.width", 200)
print(t.pivot_table(index="count", columns=["side", "where"], values="kept_median").round(3).to_string())
print(t.pivot_table(index="count", columns=["side", "where"], values="k").round(0).to_string())
print(t.groupby(["side", "where"]).median_possessions.first().round(0).to_string())
name = table.player_name
wemby = name[name == "Victor Wembanyama"].index[0]
for key in (("D", "off", "off_rebounds"), ("D", "on", "off_rebounds"), ("O", "on", "threes_made")):
    r, l, b, k = examples[key]
    i = at[int(wemby)]
    pad = (b[i] * r[i] + k * l[i]) / (b[i] + k)
    print("Wembanyama", key, f"possessions {b[i]:.0f}, raw {r[i]:.2f}, team's season rate {l[i]:.2f}, padded {pad:.2f}, keeps {b[i] / (b[i] + k):.3f}")
log = pd.concat([pd.read_parquet(f"data/raw/gamelog/{SEASON}_{phase}.parquet") for phase in ("RS", "PO")], ignore_index=True)
for who, stat, column, rate_name in (("Victor Wembanyama", "BLK", "blocks", "blocks_per100_uncentred"), ("Shai Gilgeous-Alexander", "TOV", "turnovers", "turnovers_per100_uncentred"), ("Matisse Thybulle", "STL", "steals", "steals_per100_uncentred")):
    pid = name[name == who].index[0]
    raw = 100.0 * log[log.PLAYER_ID == pid][stat].sum() / panel.possessions[pid]
    print(who, column, f"possessions {panel.possessions[pid]:.0f}, raw {raw:.2f}, padded {panel[rate_name][pid]:.2f}")
