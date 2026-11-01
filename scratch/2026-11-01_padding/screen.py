"""Experiment 127's quick read, before any build: does padding each possession bin's target toward the league rate by
the bin's own evidence (box_rates.group_k, config pad_bin_target league) reach the plain league target's gain?

Every finished season is cut to a share of its regular season (0.0715 is 2027's 88 games); the padded rates from
those games alone are compared with each player's rate over the rest of the regular season, for all 13 box counts,
players weighted by their later possessions (at least MIN_LATER).  Per arm: the line of later rate on padded rate
(1.00 = padded right) and the error.  Arms: today's padding; the league rate as the target with today's k; the
padded bin target with today's k; the padded bin target as the code composes it (the k measured around it); a hedge,
the bin target padded by a k_group pooled over the OTHER finished seasons' whole-season readings; the raw rate.

Writes scratch/2026-11-01_padding/screen.csv and prints the summary (the log beside it).  Read-only otherwise.
"""
import sys
import time

import numpy as np
import pandas as pd

from openrapm.box_rates import K_MAX, bin_of, bin_possessions, box_rates, padded_bin_targets, shrink
from openrapm.data.config import load_config
from openrapm.data.designs import DesignCache
from openrapm.data.panel import cut_design, season_to_date

SHARES = (0.0715, 0.10, 0.25, 0.50)
MIN_LATER = 100.0
ARMS = ("shipped", "league target", "padded target, shipped k", "padded target, its k", "pooled k_group", "raw")


def centre(values, weight):
    return values - np.average(values, weights=weight)


def slope(x, y, w):
    x, y = centre(x, w), centre(y, w)
    return float((w * x * y).sum() / (w * x * x).sum())


def error(x, y, w):
    return float(np.average((centre(x, w) - centre(y, w)) ** 2, weights=w))


def whole_season_group_k(cache, seasons, kw):
    """{season: its whole-season k_group per count} and the per-count pooled value each season reads from the others:
    the mean of the others' log k_group, so one K_MAX season does not swamp the rest."""
    whole = {}
    for season in seasons:
        whole[season] = box_rates(cache.season(season, "pts"), **kw, pad_bin_target="league").group_k[0]
        cache.clear()
    logs = np.log(np.array([whole[season] for season in seasons]))
    pooled = {}
    for i, season in enumerate(seasons):
        others = np.delete(logs, i, axis=0)
        pooled[season] = np.exp(others.mean(0))
    return whole, pooled


def readings(whole, fit_idx, kw, counts, pooled_k):
    regular = whole.games.loc[whole.games.phase == "RS", "game_idx"].to_numpy()
    cut = cut_design(whole, fit_idx)
    shipped = box_rates(cut, **kw)
    on = box_rates(cut, **kw, pad_bin_target="league")
    later = box_rates(cut_design(whole, np.setdiff1d(regular, fit_idx)), **kw)
    edges = shipped.bin_edges
    n_o = shipped.offence.possessions
    bins_train = bin_of(edges, n_o)
    held = bin_possessions(n_o, np.zeros(len(n_o), dtype=np.int64), 1, bins_train)
    hedge_targets = padded_bin_targets(shipped.bin_targets, held, shipped.league, pooled_k[None, :])
    rows = []
    for side_name, side, side_on, side_later in (("D", shipped.defence, on.defence, later.defence),
                                                 ("O", shipped.offence, on.offence, later.offence)):
        n, n_later = side.possessions, side_later.possessions
        bins = bin_of(edges, n)
        keep = (n > 0) & (n_later >= MIN_LATER)
        w = n_later[keep]
        for c, name in enumerate(counts):
            k = shipped.pad_k[0, c]
            raw = 100.0 * shipped.counts[:, c] / np.where(n > 0, n, 1.0)
            future = 100.0 * later.counts[:, c] / np.where(n_later > 0, n_later, 1.0)
            league = shipped.league[0, c]
            arms = {
                "shipped": side.rates[:, c],
                "league target": shrink(raw, n, k, league),
                "padded target, shipped k": shrink(raw, n, k, on.bin_targets[0, bins, c]),
                "padded target, its k": side_on.rates[:, c],
                "pooled k_group": shrink(raw, n, k, hedge_targets[0, bins, c]),
                "raw": raw,
            }
            row = {"side": side_name, "count": name, "players": int(keep.sum()), "k": k, "k_on": on.pad_k[0, c],
                   "k_group": on.group_k[0, c], "k_group_pooled": pooled_k[c],
                   "target_move": float(np.average(np.abs(on.bin_targets[0, bins, c] - shipped.bin_targets[0, bins, c])
                                                   [keep] / np.maximum(np.abs(league - shipped.bin_targets[0, bins, c])
                                                                       [keep], 1e-9), weights=w))}
            for arm, values in arms.items():
                row[f"line: {arm}"] = slope(values[keep], future[keep], w)
                row[f"error: {arm}"] = error(values[keep], future[keep], w)
            rows.append(row)
    return rows


def summary(table):
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    seasons = table.season.nunique()
    for side in ("D", "O"):
        print(f"\n===== side {side}: later rate per extra padded unit (1.00 = padded right), mean over seasons")
        part = table[table.side == side]
        lines = part.groupby(["share", "count"])[[f"line: {arm}" for arm in ARMS]].mean()
        print(lines.round(3).to_string())
        print(f"\n===== side {side}: error change against today's padding, percent; and seasons better of {seasons}")
        out = []
        for (share, count), group in part.groupby(["share", "count"]):
            row = {"share": share, "count": count}
            for arm in ARMS[1:]:
                row[f"{arm} %"] = 100.0 * (group[f"error: {arm}"].mean() / group["error: shipped"].mean() - 1.0)
                row[f"{arm} better"] = int((group[f"error: {arm}"] < group["error: shipped"]).sum())
            out.append(row)
        print(pd.DataFrame(out).set_index(["share", "count"]).round(1).to_string())
    print("\n===== k_group per share (defense rows), median over seasons, and the share of seasons at K_MAX")
    part = table[table.side == "D"]
    print(part.groupby(["share", "count"]).k_group.median().unstack("count").round(0).to_string())
    print((part.assign(at_max=part.k_group >= K_MAX).groupby(["share", "count"]).at_max.mean()
           .unstack("count").round(2)).to_string())
    print("\n===== how far the targets moved toward the league (1 = all the way), players weighted by later possessions")
    print(part.groupby(["share", "count"]).target_move.mean().unstack("count").round(2).to_string())
    print("\n===== steals by era, defense side: the line per arm")
    steals = part[part["count"] == "stl"].copy()
    steals["era"] = pd.cut(steals.season, [1996, 2006, 2016, 2026], labels=["1997-2006", "2007-2016", "2017-2026"])
    print(steals.groupby(["share", "era"], observed=True)[[f"line: {arm}" for arm in ARMS[:-1]]].mean().round(3)
          .to_string())


def main():
    cfg = load_config()
    cache = DesignCache(cfg)
    counts = list(cfg["box_counts"])
    kw = dict(pad_target=cfg["pad_target"], pad_k_around=cfg["pad_k_around"])
    seasons = list(range(int(cfg["first_season"]), 2027))
    started = time.time()
    whole_k, pooled = whole_season_group_k(cache, seasons, kw)
    print("whole-season k_group per count, median over seasons:",
          dict(zip(counts, np.round(np.median([whole_k[s] for s in seasons], axis=0), 0))), flush=True)
    rows = []
    for season in seasons:
        whole = cache.season(season, "pts")
        for share in SHARES:
            fit, season_share = season_to_date(whole.games, share)
            fit_idx = whole.games.loc[whole.games.game_id.astype(str).isin(fit), "game_idx"].to_numpy()
            for row in readings(whole, fit_idx, kw, counts, pooled[season]):
                rows.append({"season": season, "share": share, "season_share": season_share, **row})
        cache.clear()
        print(f"{season} done, {time.time() - started:.0f}s", flush=True)
    table = pd.DataFrame(rows)
    table.to_csv("scratch/2026-11-01_padding/screen.csv", index=False)
    summary(table)


if __name__ == "__main__":
    sys.exit(main())
