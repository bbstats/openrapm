"""Experiment 111: the miss factor from 2016 tracking, and the player-level check on the same games.

On 2016's tracked shots (631 regular-season games), a make model that knows the cell, the dribbles before the shot
and the touch time is fitted on made shots and scores every miss: its chance that the miss came off a pass that would
have been an assist.  Each cell's miss factor is the logit shift from the tracked makes' assisted share to that
model's mean over the cell's misses, pulled toward its spot's shift by 300 misses.  Everything is cross-fitted by
games: the halves of the tracked games, odd and even, each scored with the other half's model and factors.

The player check: each player's tracked misses scored by the tracking model, summed, is the reference; against it
five estimates that use only the play-by-play: the owner's formula, his makes' own assisted share, the cells alone,
the cells times his lean, the cells times his lean by spot group.  His lean = (assisted makes + k) / (expected
assisted makes + k), expected from the season's cell shares, k = LEAN_PAD makes.
Writes outputs/creation111/miss_factor.parquet (cell, factor) and prints the check.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from cells import SPOTS, cell_rates, expit, load_shots, logit

SHRINK_MISSES = 300.0
LEAN_PADS = (0.0, 5.0, 10.0, 20.0, 40.0)
GROUP = {"rim": "inside", "paint": "inside", "mid": "two", "long_two": "two", "corner_three": "three",
         "above_break_three": "three"}


def handling(frame: pd.DataFrame) -> pd.DataFrame:
    dribbles = pd.cut(frame.dribbles, [-1, 0, 1, 2, 5, 99], labels=["d0", "d1", "d2", "d3_5", "d6"]).astype(str)
    touch = pd.cut(frame.touch_time, [-1, 1, 2, 4, 8, 99], labels=["t1", "t2", "t4", "t8", "t9"]).astype(str)
    group = frame.spot.map(GROUP)
    return pd.get_dummies(pd.DataFrame({"cell": frame.cell, "drib": dribbles, "touch": touch,
                                        "g_drib": group + dribbles, "g_touch": group + touch}), dtype=float)


def fit_factors(train: pd.DataFrame) -> tuple:
    """(the make model, its columns, {cell: logit shift}) from one half's tracked shots."""
    x = handling(train)
    makes = train.made.to_numpy() == 1
    model = LogisticRegression(C=1.0, max_iter=2000).fit(x[makes], train.post_assisted[makes])
    scored = train.assign(p_handling=model.predict_proba(x)[:, 1])
    rate = cell_rates(scored[makes])
    misses = scored[~makes]
    by_cell = misses.groupby("cell").agg(target=("p_handling", "mean"), n=("p_handling", "size"), spot=("spot", "first"))
    by_cell["shift"] = logit(by_cell.target) - logit(rate.reindex(by_cell.index))
    spot_target = misses.groupby("spot").p_handling.mean()
    spot_rate = scored[makes].groupby("spot").post_assisted.mean()
    spot_shift = pd.Series(logit(spot_target) - logit(spot_rate.reindex(spot_target.index)), index=spot_target.index)
    weight = by_cell.n / (by_cell.n + SHRINK_MISSES)
    by_cell["factor"] = weight * by_cell["shift"] + (1 - weight) * by_cell.spot.map(spot_shift)
    return model, list(x.columns), by_cell.factor.to_dict()


def main() -> None:
    shots = load_shots(2016, ("RS",))
    tracked = pd.read_parquet("data/tracking/movement_shots_2016.parquet")[["game_id", "action_number", "dribbles",
                                                                            "touch_time"]]
    t = shots.merge(tracked, on=["game_id", "action_number"])
    t["half"] = t.game_id.str[-1].astype(int) % 2
    print(f"tracked shots {len(t)}, games {t.game_id.nunique()}, makes {int(t.made.sum())}")

    scored, factors = [], {}
    for half in (0, 1):
        model, columns, factor = fit_factors(t[t.half != half])
        test = t[t.half == half]
        x = handling(test).reindex(columns=columns, fill_value=0.0)
        scored.append(test.assign(p_handling=model.predict_proba(x)[:, 1],
                                  factor=test.cell.map(factor).fillna(np.mean(list(factor.values())))))
        factors[half] = factor
    t = pd.concat(scored, ignore_index=True)
    makes, misses = t[t.made == 1], t[t.made == 0]
    print(f"handling model on held-out makes: mean {makes.p_handling.mean():.3f} against assisted "
          f"{makes.post_assisted.mean():.3f}; misses scored {misses.p_handling.mean():.3f}")

    # the factors written are fitted on every tracked game
    _, _, every = fit_factors(t.drop(columns=["p_handling", "factor"]))
    table = pd.DataFrame({"cell": list(every), "factor": list(every.values())})
    table.to_parquet("outputs/creation111/miss_factor.parquet", index=False)
    show = table.assign(spot=table.cell.str.split("|").str[0], context=table.cell.str.split("|").str[1])
    print("\nmiss factor, logit shift by cell; below 0 = misses come off a pass less often than makes")
    print(show.pivot(index="context", columns="spot", values="factor").reindex(columns=SPOTS).round(2).to_string())

    # the player check, every 2016 make of the season behind his lean and the cell shares
    season = load_shots(2016, ("RS", "PO"))
    rate = cell_rates(season[season.made == 1])
    season["p_cell"] = season.cell.map(rate)
    made = season[season.made == 1]
    group = made.assign(group=made.spot.map(GROUP))
    lean_all = made.groupby("shooter").agg(assisted=("post_assisted", "sum"), expected=("p_cell", "sum"),
                                           makes=("made", "size"))
    lean_group = group.groupby(["shooter", "group"]).agg(assisted=("post_assisted", "sum"), expected=("p_cell", "sum"))
    misses = misses.assign(p_cell=misses.cell.map(rate))
    misses["p_miss"] = expit(logit(misses.p_cell) + misses.factor)
    misses["group"] = misses.spot.map(GROUP)

    panel = pd.read_parquet("outputs/season_panel.parquet")
    o = panel[(panel.side == "O") & (panel.season == 2016)].set_index("player_id")
    fgm = o.twos_made_per100_uncentred + o.threes_made_per100_uncentred
    owner = np.clip(0.73 - 0.03 * o.assists_per100_uncentred - 0.01 * fgm, 0, 1)

    per = misses.groupby("shooter").agg(n=("made", "size"), reference=("p_handling", "sum"),
                                        cells=("p_miss", "sum"))
    per = per[per.n >= 150]
    rows = []
    for pad in LEAN_PADS:
        lean = (lean_all.assisted + pad) / (lean_all.expected + pad)
        lg = (lean_group.assisted + pad) / (lean_group.expected + pad)
        by_group = misses.assign(lean=[lg.get((s, g), np.nan) for s, g in zip(misses.shooter, misses.group)])
        by_group["lean"] = by_group.lean.fillna(by_group.shooter.map(lean)).fillna(1.0)
        est_group = (by_group.p_miss * by_group.lean).clip(upper=1).groupby(by_group.shooter).sum()
        est = {"owner's formula": owner.reindex(per.index) * per.n,
               "his makes' assisted share": (lean_all.assisted / lean_all.makes).reindex(per.index) * per.n,
               "cells alone": per.cells,
               "cells x his lean": (misses.p_miss * misses.shooter.map(lean).fillna(1.0)).clip(upper=1)
               .groupby(misses.shooter).sum().reindex(per.index),
               "cells x his lean by spot group": est_group.reindex(per.index)}
        truth = per.reference / per.n
        for name, value in est.items():
            if pad != LEAN_PADS[0] and "lean" not in name:
                continue
            share = value / per.n
            ok = share.notna()
            err = share[ok] - truth[ok]
            rows.append({"estimate": name, "lean_pad": pad if "lean" in name else np.nan, "players": int(ok.sum()),
                         "rmse_share": float(np.sqrt(np.average(err ** 2, weights=per.n[ok]))),
                         "mean_error": float(np.average(err, weights=per.n[ok])),
                         "correlation": float(np.corrcoef(share[ok], truth[ok])[0, 1])})
    print(f"\nplayers with 150+ tracked misses: {len(per)}; reference share of misses off a pass "
          f"{(per.reference.sum() / per.n.sum()):.3f}")
    print(pd.DataFrame(rows).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
