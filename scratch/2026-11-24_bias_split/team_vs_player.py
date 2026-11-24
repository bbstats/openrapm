"""Read-only: the ten-season game-results lean with and without a free level per team-season, split by side.

With team-season levels free, a lean is identified only by who was on the court within a team's season: player
credit.  What disappears was between teams: the games say the team missed, not which of its players did.
"""
import dataclasses
import sys
from functools import partial

import numpy as np
import pandas as pd
from scipy.linalg import block_diag

from openrapm import bias, game_results, trade_set
from openrapm.commands.site import panel_of_season, season_designs, shown_z
from openrapm.data.config import load_config, resolve
from openrapm.data.tables import team_of_rows

cfg = load_config()
table = pd.read_parquet(resolve(cfg, "outputs") / "season_ratings_product.parquet")
SEASONS = list(range(2018, 2028))
NO_GARBAGE = len(sys.argv) > 1 and sys.argv[1] == 'no_garbage'


def without_garbage(design):
    """The design's rows outside garbage time only."""
    keep = ~design.rows.is_gt.to_numpy().astype(bool)
    Z, F = trade_set._player_block(design), trade_set._fixed_block(design)
    return dataclasses.replace(design, rows=design.rows[keep].reset_index(drop=True), y=design.y[keep],
                               w=design.w[keep], parts={'Z': Z[keep], 'F': F[keep]})


def designs_of(s):
    designs = season_designs(cfg, s)
    return {k: without_garbage(v) for k, v in designs.items()} if NO_GARBAGE else designs


games = {s: game_results.season_games(designs_of(s), table, s, partial(team_of_rows, cfg=cfg)) for s in SEASONS}
print('team-games', sum(g.team_games for g in games.values()), flush=True)
zs = {s: shown_z(table, s, panel_of_season(cfg, s)) for s in SEASONS}
print("loaded", flush=True)


def stacked(side, stat, team_levels):
    shared, own, miss, w, toff, tdef = [], [], [], [], [], []
    for s in SEASONS:
        g = games[s]
        ts = g.games[side]
        n = ts.n_players
        z = pd.DataFrame({"player_id": g.player_ids}).merge(zs[s], on="player_id", how="left").fillna(0.0)[stat]
        z = z.to_numpy(float)
        shared.append(np.column_stack([ts.Z[:, :n] @ z, ts.Z[:, n:] @ z]))
        blocks = [trade_set.rating_columns(ts, g.offset), ts.F[:, np.any(ts.F != 0.0, axis=0)]]
        if team_levels:
            for column in ("team_off", "team_def"):
                ids = g.teams[column].to_numpy()
                levels = np.unique(ids)[1:]          # one team per side left as the season's own level
                blocks.append((ids[:, None] == levels[None, :]).astype(float))
        own.append(np.column_stack(blocks))
        miss.append(ts.y - ts.Z @ g.offset)
        w.append(ts.w)
        toff.append(g.teams.team_off.to_numpy())
        tdef.append(g.teams.team_def.to_numpy())
    X = np.column_stack([np.vstack(shared), block_diag(*own)])
    return X, np.concatenate(miss), np.concatenate(w), np.concatenate(toff), np.concatenate(tdef)


def fit(X, y, w, keep=None):
    if keep is not None:
        X, y, w = X[keep], y[keep], w[keep]
    used = np.any(X != 0.0, axis=0)
    used[:2] = True
    Xu = X[:, used]
    weighted = Xu * w[:, None]
    coef = np.linalg.lstsq(Xu.T @ weighted, weighted.T @ y, rcond=None)[0]
    return coef[:2]


def jackknife(values):
    values = np.asarray(values)
    k = len(values)
    return float(np.sqrt((k - 1) / k * np.sum((values - values.mean(axis=0)) ** 2, axis=0)).item()) \
        if values.ndim == 1 else np.sqrt((k - 1) / k * np.sum((values - values.mean(axis=0)) ** 2, axis=0))


rows = []
for stat in bias.STATS:
    out = {"stat": stat}
    for label, team_levels in (("pooled", False), ("within", True)):
        Xo, yo, wo, toff, tdef = stacked("offense", stat, team_levels)
        Xd, yd, wd, _, _ = stacked("defense", stat, team_levels)
        b_o, b_d = fit(Xo, yo, wo)[0], fit(Xd, yd, wd)[1]
        parts = []
        for team in np.unique(np.concatenate([toff, tdef])):
            keep = (toff != team) & (tdef != team)
            parts.append([-fit(Xo, yo, wo, keep)[0], fit(Xd, yd, wd, keep)[1]])
        parts = np.array(parts)
        se = jackknife(parts)
        se_total = jackknife(parts.sum(axis=1))
        out[f"{label}_off"], out[f"{label}_off_se"] = -b_o, se[0]
        out[f"{label}_def"], out[f"{label}_def_se"] = b_d, se[1]
        out[f"{label}_total"], out[f"{label}_total_se"] = b_d - b_o, se_total
    rows.append(out)
    print(stat, {k: round(v, 2) for k, v in out.items() if k != "stat"}, flush=True)
frame = pd.DataFrame(rows)
frame.to_csv("outputs/"
             f"bias_split_team_vs_player{'_no_garbage' if NO_GARBAGE else ''}.csv", index=False)
with pd.option_context("display.width", 250, "display.max_columns", 30):
    print(frame.round(2).to_string())
