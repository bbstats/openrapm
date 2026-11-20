"""Early-season defense, read 1 (notes/DECISIONS.md, "Early-season defense, read 1"): does padding the hustle stats the
way experiment 127 pads the box stats predict the rest of the season better from its first games?

Each season 2017-2026 cut to a share of its regular season; each player's nine hustle counts from those games alone
(dashboards.hustle_game_totals) over his possessions in them, padded each way, against his rate over the rest of the
regular season, players with MIN_LATER or more later possessions weighted by them, both sides.  Arms:
    shipped                 today's: k by moment_k around the league, toward his possession bin's raw rate
    league target           the league rate as the target, today's k
    padded target           each bin's rate padded toward the league by its own possessions, k_group (experiment 127),
                            today's k
    padded target, its k    the same with the k measured around the padded target (the method of moments, Poisson
                            noise as moment_k)
    raw                     no padding
Per arm the line of later rate on padded rate (1.00 = padded right) and the error.

Writes scratch/2026-11-20_early_defense/hustle_screen.csv and prints the summary.  Read-only otherwise.
    .venv/Scripts/python scratch/2026-11-20_early_defense/hustle_screen.py
"""
import sys
import time

import numpy as np
import pandas as pd

from openrapm.box_rates import K_MAX, MIN_BIN_PLAYERS, POSSESSION_BINS, bin_of, box_rates, possession_bins, shrink
from openrapm.data.config import load_config
from openrapm.data.dashboard_rates import COUNTS, MIN_POSSESSIONS, moment_k
from openrapm.data.dashboards import hustle_game_totals
from openrapm.data.designs import DesignCache
from openrapm.data.panel import cut_design, season_to_date

sys.stdout.reconfigure(encoding="utf-8")
SHARES = (0.0715, 0.10, 0.25, 0.50)
MIN_LATER = 100.0
SEASONS = range(2017, 2027)
ARMS = ("shipped", "league target", "padded target", "padded target, its k", "raw")
HUSTLE = [spec for spec in COUNTS if spec.source == "hustle"]
DEFENCE_READ = ["deflections_vs_league_avg", "contested_twos_vs_league_avg", "contested_threes_vs_league_avg",
                "charges_drawn_vs_league_avg", "def_box_outs_vs_league_avg", "loose_balls_recovered_vs_league_avg"]


def centre(values, weight):
    return values - np.average(values, weights=weight)


def slope(x, y, w):
    x, y = centre(x, w), centre(y, w)
    return float((w * x * y).sum() / (w * x * x).sum())


def error(x, y, w):
    return float(np.average((centre(x, w) - centre(y, w)) ** 2, weights=w))


def bins_and_targets(counts, n, present, league):
    """(each player's bin, -1 without bins; the bins' raw rates; the bins' possessions), as _bin_targets builds them."""
    bins = np.full(len(n), -1)
    raw = np.full(POSSESSION_BINS, league)
    held = np.zeros(POSSESSION_BINS)
    edges = possession_bins(n[present], n[present])
    if edges is None:
        return bins, raw, held
    bins = bin_of(edges, n)
    for b in range(POSSESSION_BINS):
        inside = present & (bins == b)
        if inside.sum() >= MIN_BIN_PLAYERS and n[inside].sum() > 0:
            raw[b] = 100.0 * counts[inside].sum() / n[inside].sum()
            held[b] = n[inside].sum()
    return bins, raw, held


def group_k(raw, held, league):
    """experiment 127's k_group for one count: Poisson noise 100 x league over the true spread of the bin rates."""
    noise = 100.0 * league
    total = held.sum()
    if total <= 0:
        return K_MAX
    spread = (held * (raw - league) ** 2).sum() / total
    chance = noise * (held > 0).sum() / total
    between = spread - chance
    return float(min(noise / between, K_MAX)) if between > 0 else K_MAX


def k_around(counts, n, target):
    """moment_k with the spread measured around each player's own target."""
    keep = np.isfinite(counts) & (n >= MIN_POSSESSIONS)
    league = 100.0 * counts[np.isfinite(counts) & (n > 0)].sum() / n[np.isfinite(counts) & (n > 0)].sum()
    if keep.sum() < 20 or league <= 0:
        return K_MAX
    rate = 100.0 * counts[keep] / n[keep]
    observed = np.average((rate - target[keep]) ** 2, weights=n[keep])
    noise = np.average(100.0 * league / n[keep], weights=n[keep])
    return float(min(100.0 * league / max(observed - noise, 1e-6), K_MAX))


def side_arms(counts, n):
    """{arm: rate per 100} for one count on one side, every player (0 where he has no possessions)."""
    present = np.isfinite(counts) & (n > 0)
    safe = np.where(present, counts, 0.0)
    league = 100.0 * safe[present].sum() / n[present].sum()
    raw = np.where(present, 100.0 * safe / np.where(present, n, 1.0), 0.0)
    k = moment_k(np.where(present, counts, np.nan), n)
    bins, bin_raw, held = bins_and_targets(safe, n, present, league)
    shipped_target = np.where(bins >= 0, bin_raw[np.clip(bins, 0, None)], league)
    kg = group_k(bin_raw, held, league)
    padded_bins = shrink(bin_raw, held, kg, league)
    padded_target = np.where(bins >= 0, padded_bins[np.clip(bins, 0, None)], league)
    k_on = k_around(np.where(present, counts, np.nan), n, padded_target)
    return {"shipped": shrink(raw, n, k, shipped_target), "league target": shrink(raw, n, k, league),
            "padded target": shrink(raw, n, k, padded_target),
            "padded target, its k": shrink(raw, n, k_on, padded_target), "raw": raw}, \
        dict(k=k, k_on=k_on, k_group=kg)


def player_totals(cfg, season, design, game_ids):
    """(player ids, {"O", "D"}: possessions in `design`, the hustle counts frame over `game_ids`)."""
    rates = box_rates(design)
    ids = design.spec.players["player_id"].to_numpy(dtype=np.int64)
    totals, _ = hustle_game_totals(cfg, season, game_ids)
    return ids, {"O": rates.offence.possessions, "D": rates.defence.possessions}, totals


def readings(cfg, season, whole, fit_idx):
    regular = whole.games.loc[whole.games.phase == "RS"]
    games = whole.games.set_index("game_idx").game_id.astype(str)
    later_idx = np.setdiff1d(regular.game_idx.to_numpy(), fit_idx)
    ids, n_fit, fit = player_totals(cfg, season, cut_design(whole, fit_idx), games.loc[fit_idx].tolist())
    ids_l, n_later, later = player_totals(cfg, season, cut_design(whole, later_idx), games.loc[later_idx].tolist())
    later_n = {side: pd.Series(n_later[side], index=ids_l).reindex(ids).fillna(0.0).to_numpy() for side in "OD"}
    rows = []
    for side in ("D", "O"):
        n, nl = n_fit[side], later_n[side]
        for spec in HUSTLE:
            counts = fit[spec.column].reindex(ids).to_numpy(float) if spec.column in fit.columns else None
            if counts is None:
                continue
            future_counts = later[spec.column].reindex(ids).to_numpy(float)
            arms, ks = side_arms(counts, n)
            future = np.where(nl > 0, 100.0 * np.nan_to_num(future_counts) / np.where(nl > 0, nl, 1.0), 0.0)
            keep = (n > 0) & (nl >= MIN_LATER) & np.isfinite(counts) & np.isfinite(future_counts)
            w = nl[keep]
            row = {"side": side, "count": spec.name.replace("_vs_league_avg", ""), "players": int(keep.sum()), **ks}
            for arm, values in arms.items():
                row[f"line: {arm}"] = slope(values[keep], future[keep], w)
                row[f"error: {arm}"] = error(values[keep], future[keep], w)
            rows.append(row)
    return rows


def summary(table):
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    for side in ("D", "O"):
        part = table[table.side == side]
        print(f"\n===== side {side}: later rate per extra padded unit (1.00 = padded right), mean over seasons")
        print(part.groupby(["share", "count"])[[f"line: {arm}" for arm in ARMS]].mean().round(3).to_string())
        print(f"\n===== side {side}: error change against today's padding, percent, its standard error over seasons, "
              "and seasons better of 10")
        out = []
        for (share, count), group in part.groupby(["share", "count"]):
            row = {"share": share, "count": count}
            for arm in ARMS[1:]:
                change = 100.0 * (group[f"error: {arm}"] / group["error: shipped"] - 1.0)
                row[f"{arm} %"] = 100.0 * (group[f"error: {arm}"].mean() / group["error: shipped"].mean() - 1.0)
                row[f"{arm} se"] = change.std(ddof=1) / np.sqrt(len(change))
                row[f"{arm} better"] = int((group[f"error: {arm}"] < group["error: shipped"]).sum())
            out.append(row)
        print(pd.DataFrame(out).set_index(["share", "count"]).round(1).to_string())
    print("\n===== k_group (defense rows), median over seasons")
    print(table[table.side == "D"].groupby(["share", "count"]).k_group.median().unstack("count").round(0).to_string())


def main():
    cfg = load_config()
    cache = DesignCache(cfg)
    started = time.time()
    rows = []
    for season in SEASONS:
        whole = cache.season(season, "pts")
        for share in SHARES:
            fit, season_share = season_to_date(whole.games, share)
            fit_idx = whole.games.loc[whole.games.game_id.astype(str).isin(fit), "game_idx"].to_numpy()
            for row in readings(cfg, season, whole, fit_idx):
                rows.append({"season": season, "share": share, "season_share": season_share, **row})
        cache.clear()
        print(f"{season} done, {time.time() - started:.0f}s", flush=True)
    table = pd.DataFrame(rows)
    table.to_csv("scratch/2026-11-20_early_defense/hustle_screen.csv", index=False)
    summary(table)


if __name__ == "__main__":
    sys.exit(main())
