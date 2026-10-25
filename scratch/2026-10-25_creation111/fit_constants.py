"""Experiment 111: the constants of data/creation_counts.py, fitted on its own shot tables, and the two checks again.

    CELL_PRIOR    each cell's assisted share of made shots, 1997-2026 pooled
    MISS_FACTOR   from 2016 tracking: a make model on the cell, the dribbles and the touch time scores every miss;
                  each cell's logit shift from the tracked makes' share to that mean, pulled toward its spot's shift
                  by 300 misses
    FT_FACTOR     the trips off a pass with a free throw made, summed over 2014-2017's regular seasons, set equal to
                  the NBA's free-throw assists
The player check (2016, cross-fitted by halves of the tracked games) and the team check (the NBA's counts,
2014-2026) are printed as miss_factor.py and team_check.py print them; the constants go to
outputs/creation111/constants.json for pasting into the module.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from openrapm.data.creation_counts import CONTEXTS, SPOT_GROUPS, SPOTS, cell_shares, expit, logit

SHRINK_MISSES = 300.0
LEAN_PAD = 10.0
SEASONS = range(1997, 2027)
FT_FIT = range(2014, 2018)


def shots(season):
    return pd.read_parquet(f"data/event_counts/{season}_creation_shots.parquet")


def trips(season):
    return pd.read_parquet(f"data/event_counts/{season}_creation_trips.parquet")


def handling(frame):
    dribbles = pd.cut(frame.dribbles, [-1, 0, 1, 2, 5, 99], labels=["d0", "d1", "d2", "d3_5", "d6"]).astype(str)
    touch = pd.cut(frame.touch_time, [-1, 1, 2, 4, 8, 99], labels=["t1", "t2", "t4", "t8", "t9"]).astype(str)
    group = frame.spot.map(SPOT_GROUPS)
    return pd.get_dummies(pd.DataFrame({"cell": frame.cell, "drib": dribbles, "touch": touch,
                                        "g_drib": group + dribbles, "g_touch": group + touch}), dtype=float)


def fit_factors(train):
    x = handling(train)
    makes = train.made.to_numpy() == 1
    model = LogisticRegression(C=1.0, max_iter=3000).fit(x[makes], train.assisted[makes])
    scored = train.assign(p=model.predict_proba(x)[:, 1])
    rate = scored[makes].groupby("cell").assisted.mean()
    miss = scored[~makes]
    by = miss.groupby("cell").agg(target=("p", "mean"), n=("p", "size"), spot=("spot", "first"))
    by["shift"] = logit(by.target) - logit(rate.reindex(by.index))
    spot_target = miss.groupby("spot").p.mean()
    spot_rate = scored[makes].groupby("spot").assisted.mean().reindex(spot_target.index)
    spot_shift = pd.Series(logit(spot_target) - logit(spot_rate), index=spot_target.index)
    w = by.n / (by.n + SHRINK_MISSES)
    by["factor"] = w * by["shift"] + (1 - w) * by.spot.map(spot_shift)
    return model, list(x.columns), by.factor.to_dict()


def season_table(season, prior, factor):
    """One row per player, team and phase: makes and assisted makes by group, misses off a pass with his lean."""
    s = shots(season)
    share = cell_shares(s, prior)
    s["p_cell"] = s.cell.map(share)
    s["p_miss"] = expit(logit(s.p_cell) + s.cell.map(factor).fillna(0.0))
    made, miss = s[s.made == 1], s[s.made == 0].copy()
    lean = made.groupby(["shooter", "group"]).agg(a=("assisted", "sum"), e=("p_cell", "sum"))
    lean = (lean.a + LEAN_PAD) / (lean.e + LEAN_PAD)
    miss["lean"] = lean.reindex(pd.MultiIndex.from_arrays([miss.shooter, miss.group])).fillna(1.0).to_numpy()
    miss["passed"] = (miss.p_miss * miss.lean).clip(upper=1)
    by = ["shooter", "team", "phase"]
    out = made.groupby(by).agg(makes=("made", "size"), assisted_makes=("assisted", "sum"))
    out = out.join(miss.groupby(by).agg(misses=("made", "size"), passed_misses=("passed", "sum"),
                                        cells_misses=("p_miss", "sum")), how="outer")
    t = trips(season).rename(columns={"player_id": "shooter"})
    t["other_trips"] = (t.trips - t.and_one_trips).clip(lower=0)
    out = out.join(t.groupby(by)[["trips", "other_trips"]].sum(), how="outer").fillna(0.0).reset_index()
    out["lean_inside"] = lean.xs("inside", level="group").reindex(out.shooter).fillna(1.0).to_numpy()
    out["rim_base"] = float(s.loc[(s.made == 0) & (s.spot == "rim"), "p_miss"].mean())
    out["season"] = season
    return out


def main():
    pooled = pd.concat([shots(season).query("made == 1").groupby("cell").assisted.agg(["sum", "size"])
                        for season in SEASONS]).groupby(level=0).sum()
    prior = (pooled["sum"] / pooled["size"]).to_dict()

    tracked = pd.read_parquet("data/tracking/movement_shots_2016.parquet")[["game_id", "action_number", "dribbles",
                                                                            "touch_time"]]
    t = shots(2016).query("phase == 'RS'").merge(tracked, on=["game_id", "action_number"])
    print(f"tracked shots matched {len(t)} of {len(tracked)}")
    _, _, factor = fit_factors(t)

    # the player check, cross-fitted by halves of the tracked games
    t["half"] = t.game_id.str[-1].astype(int) % 2
    scored = []
    for half in (0, 1):
        model, columns, f = fit_factors(t[t.half != half])
        test = t[t.half == half]
        x = handling(test).reindex(columns=columns, fill_value=0.0)
        scored.append(test.assign(reference=model.predict_proba(x)[:, 1], factor=test.cell.map(f).fillna(0.0)))
    t = pd.concat(scored)
    season = shots(2016)
    share = cell_shares(season, prior)
    made = season[season.made == 1].assign(p_cell=lambda d: d.cell.map(share))
    lean = made.groupby(["shooter", "group"]).agg(a=("assisted", "sum"), e=("p_cell", "sum"))
    lean = (lean.a + LEAN_PAD) / (lean.e + LEAN_PAD)
    own = made.groupby("shooter").agg(a=("assisted", "sum"), n=("made", "size"))
    miss = t[t.made == 0].copy()
    miss["p_miss"] = expit(logit(miss.cell.map(share)) + miss.factor)
    miss["lean"] = lean.reindex(pd.MultiIndex.from_arrays([miss.shooter, miss.group])).fillna(1.0).to_numpy()
    miss["estimate"] = (miss.p_miss * miss.lean).clip(upper=1)
    per = miss.groupby("shooter").agg(n=("made", "size"), reference=("reference", "sum"), estimate=("estimate", "sum"),
                                      cells=("p_miss", "sum"))
    per = per[per.n >= 150]
    panel = pd.read_parquet("outputs/season_panel.parquet")
    o = panel[(panel.side == "O") & (panel.season == 2016)].set_index("player_id")
    owner = np.clip(0.73 - 0.03 * o.assists_per100_uncentred
                    - 0.01 * (o.twos_made_per100_uncentred + o.threes_made_per100_uncentred), 0, 1)
    truth = per.reference / per.n
    rows = []
    for name, share_est in (("owner's formula", owner.reindex(per.index)),
                            ("his makes' assisted share", (own.a / own.n).reindex(per.index)),
                            ("cells alone", per.cells / per.n), ("the estimate", per.estimate / per.n)):
        err = share_est - truth
        rows.append({"estimate": name, "players": len(per), "error_sd": np.sqrt(np.average(err ** 2, weights=per.n)),
                     "mean_error": np.average(err, weights=per.n), "correlation": np.corrcoef(share_est, truth)[0, 1]})
    print("\nthe player check, 2016 tracking, share of his misses off a pass")
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    table = pd.concat([season_table(s, prior, factor) for s in SEASONS], ignore_index=True)
    panel_o = panel[panel.side == "O"].set_index(["player_id", "season"])
    ft_pct = panel_o.free_throws_made_per100_uncentred / (panel_o.free_throws_made_per100_uncentred
                                                         + panel_o.free_throws_missed_per100_uncentred)
    key = pd.MultiIndex.from_arrays([table.shooter, table.season])
    table["one_made"] = 1 - (1 - ft_pct.reindex(key).fillna(0.75).to_numpy()) ** 2
    table["owner_share"] = np.clip(0.73 - 0.03 * panel_o.assists_per100_uncentred
                                   - 0.01 * (panel_o.twos_made_per100_uncentred + panel_o.threes_made_per100_uncentred),
                                   0, 1).reindex(key).to_numpy()
    nba = []
    for s in range(2014, 2027):
        p = pd.read_parquet(f"data/raw/dashboards/tracking/pt_passing/{s}_RS.parquet")
        nba.append(p.groupby("TEAM_ID")[["AST", "POTENTIAL_AST", "FT_AST"]].sum().assign(season=s))
    nba = pd.concat(nba).reset_index().rename(columns={"TEAM_ID": "team"})
    nba["misses"], nba["trips"] = nba.POTENTIAL_AST - nba.AST, nba.FT_AST
    base = table.other_trips * table.lean_inside * table.rim_base * table.one_made
    fit = table.season.isin(FT_FIT) & (table.phase == "RS")
    ft_factor = float(nba[nba.season.isin(FT_FIT)].FT_AST.sum() / base[fit].sum())
    print(f"\nfree-throw factor {ft_factor:.3f}")

    rs = table[(table.phase == "RS") & (table.season >= 2014)].copy()
    whole = table.groupby(["shooter", "season"])[["assisted_makes", "makes"]].sum()
    rs["makes_share"] = (whole.assisted_makes / whole.makes.clip(lower=1)).reindex(
        pd.MultiIndex.from_arrays([rs.shooter, rs.season])).to_numpy()
    rs["owner_share"] = rs.owner_share.fillna(rs.makes_share)
    chance = (rs.lean_inside * rs.rim_base * ft_factor).clip(upper=1)
    estimates = {"misses": {"owner's formula": rs.owner_share * rs.misses,
                            "his makes' assisted share": rs.makes_share * rs.misses,
                            "the estimate": rs.passed_misses},
                 "trips": {"owner's formula": rs.owner_share * rs.other_trips * rs.one_made,
                           "his makes' assisted share": rs.makes_share * rs.other_trips * rs.one_made,
                           "the estimate": rs.other_trips * chance * rs.one_made}}
    rows = []
    for kind, named in estimates.items():
        for name, value in named.items():
            team = rs.assign(v=value).groupby(["season", "team"]).v.sum().rename("estimate").reset_index()
            both = team.merge(nba[["season", "team", kind]], on=["season", "team"])
            for era, seasons in (("2014-2017", range(2014, 2018)),
                                 ("2018-2026", [s for s in range(2018, 2027) if not (kind == "trips" and s == 2025)])):
                part = both[both.season.isin(seasons)]
                r = part.groupby("season").apply(lambda g: np.corrcoef(g.estimate, g[kind])[0, 1],
                                                 include_groups=False)
                rows.append({"count": kind, "estimate": name, "seasons": era, "mean_correlation": r.mean(),
                             "lowest_season": r.min()})
    print("\nthe team check, correlation across teams within each season with the NBA's count")
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    cells = [f"{spot}|{context}" for spot in SPOTS for context in CONTEXTS]
    constants = {"CELL_PRIOR": {c: round(float(prior[c]), 4) for c in cells if c in prior},
                 "MISS_FACTOR": {c: round(float(factor[c]), 4) for c in cells if c in factor},
                 "FT_FACTOR": round(ft_factor, 4)}
    with open("outputs/creation111/constants.json", "w") as handle:
        json.dump(constants, handle, indent=1)
    print("\nwrote outputs/creation111/constants.json:", len(constants["CELL_PRIOR"]), "cells")


if __name__ == "__main__":
    main()
