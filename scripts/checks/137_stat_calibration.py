"""Check 137, calibration curves: is the rating over- or underrated as a function of each input stat?

**Superseded for decisions by check 138, the held-out audit (2026-10-08, the Robustness pass).**  The miss here comes
from the neighbouring seasons (which teaches "peak seasons regress"), on luck-adjusted targets, and the permutation z
below shuffles inputs within a season, which inflates the z of anything a player carries from season to season three
to four times (body weight 16.9 -> about 4 once whole players are shuffled).  Kept as a picture of the
neighbouring-season reading.

The owner, 2026-10-08, after the fifths of check 95 showed steals overvalued on defence: "I almost would rather do a
calibration plot style rather than binning ... with overrating / underrating we should see y = 0 ... based on steals
per 100", and "more like kbins or tree based model to get any non-monotonicity".

Words used below
    miss            a player-season's trade-set correction (check 70's `alpha_good`), points per 100: positive means
                    his team's games with and without him say he is better than rated, i.e. underrated.  Read on the
                    rows the trade loss reads: one-team players whose team played neighbouring games without them
    input           a box-prior input exactly as the prior reads it: the season panel through `features.add_features`,
                    the feature set "box_score_and_bio"; steals and blocks also on their uncentred padded rates per 100,
                    for a readable axis
    bin             one of --bins equal-count groups of player-seasons by the input (at least 30 rows each): its mean
                    miss, with the standard error of that mean clustered by player (a player's seasons share errors)
    curve           a one-input LightGBM fit of the miss: 150 shallow trees of 6 leaves, at least 400 rows a leaf, so
                    it bends where the data do and nowhere else.  Its predictions come from five folds by player, each
                    player predicted by the fit that never saw him
    curve size      the standard deviation of those out-of-fold predictions, points per 100
    permutation z   (curve size - mean null size) / standard deviation of the null sizes; a null size is the same fit
                    with the input shuffled among each season's player-seasons, --perms times

A well-calibrated rating has a miss curve flat at zero against every input.  z above 3 is a stat the ratings
mis-price as a whole.

Reads the alpha table (--alpha) and the season panel (paths.panel, without any season still being played).  Writes
outputs/csv/stat_calibration_<tag>.csv (one row per input and side), outputs/csv/stat_calibration_<tag>_bins.csv
(every bin) and outputs/stat_calibration_<tag>.png (steals, blocks and the four most mis-priced inputs per side).

How to read a line, from experiment 45's defence (--tag=lgb_noonc_rs, 20 permutations): "steals per 100 (axis)  curve
0.030 (null 0.015), z +11.0; bottom bin +0.067 +/- 0.018, top bin -0.050 +/- 0.019" says the miss moves with steals
twice as much as with a shuffled input, and the players with the most steals are overrated by about 0.05 points per
100 while those with the fewest are underrated by about 0.07.

    python scripts/checks/137_stat_calibration.py --alpha=outputs/tradeset_weightfix_alpha.parquet --tag=weightfix
"""
import argparse
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from openrapm.commands.console import one_thread, use_utf8
from openrapm.data.config import load_config, resolve, resolve_path
from openrapm.data.seasons import drop_untrainable
from openrapm.features import FEATURE_SETS, add_features, meaning

CURVE = dict(n_estimators=150, learning_rate=0.05, num_leaves=6, min_child_samples=400, reg_lambda=10.0,
             subsample=1.0, colsample_bytree=1.0, n_jobs=1, verbose=-1, random_state=0)
FEATURE_SET = "box_score_and_bio"
SIDE = {"offense": "O", "defense": "D"}
# The axis inputs: (the panel column, the label).  Steals and blocks are always drawn.  Their own names, so they
# never overwrite the centred steals_per100 and blocks_per100 a prior reads.
AXIS = {"steals_axis": ("steals_per100_uncentred", "steals per 100 (axis)"),
        "blocks_axis": ("blocks_per100_uncentred", "blocks per 100 (axis)")}
MIN_BIN_ROWS = 30
FOLDS = 5
SHOWN_PER_SIDE = 4
INK, MUTED, GRID, BLUE = "#0b0b0b", "#52514e", "#e1e0d9", "#2a78d6"
SUMMARY = ("side", "plain", "curve_size", "null_size", "z", "bottom_bin_miss", "top_bin_miss")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibration curves of the rating miss against every input.",
                                     allow_abbrev=False)
    parser.add_argument("--alpha", default="outputs/tradeset_lgb_noonc_rs_alpha.parquet",
                        help="a scripts/checks/70_tradeset.py <out>_alpha.parquet (default %(default)s)")
    parser.add_argument("--tag", default="lgb_noonc_rs", help="names the outputs (default %(default)s)")
    parser.add_argument("--perms", default=20, type=int, help="permutations for the z (default %(default)s)")
    parser.add_argument("--bins", default=20, type=int, help="equal-count bins per input (default %(default)s)")
    return parser.parse_args(argv)


# ------------------------------------------------------------------------------------------------ the inputs
def read_misses(path: Path) -> pd.DataFrame:
    """The alpha table's rows the trade loss reads: eligible, with at least one possession of absence."""
    alpha = pd.read_parquet(path)
    return alpha[alpha.eligible & (alpha.without_poss >= 1)]


def read_panel(cfg: dict) -> pd.DataFrame:
    """The season panel's player-seasons with possessions, without any season still being played."""
    panel, _ = drop_untrainable(pd.read_parquet(resolve(cfg, "panel")), cfg, what="the season panel")
    return panel[panel.possessions > 0]


def side_rows(panel: pd.DataFrame, alpha: pd.DataFrame, side_name: str, inputs: list) -> pd.DataFrame:
    """One side's inputs as the prior reads them, plus the axis rates, joined to the misses."""
    rows = panel[panel.side == SIDE[side_name]]
    frame = add_features(rows.copy(), inputs)
    frame = frame.assign(**{name: rows[column].to_numpy(float) for name, (column, _) in AXIS.items()})
    misses = alpha[alpha.side == side_name][["player_id", "season", "alpha_good"]]
    return frame.merge(misses, on=["player_id", "season"], how="inner")


# ------------------------------------------------------------------------------------------------- one curve
def player_folds(players: np.ndarray, k: int = FOLDS) -> np.ndarray:
    """Each row's fold: players are numbered 0, 1, ..., k-1, 0, 1, ... in id order, then the numbers are shuffled
    (seed 0), so every season of a player lands in the same fold."""
    keys = np.unique(players)
    rng = np.random.default_rng(0)
    fold_of = dict(zip(keys, rng.permutation(np.arange(keys.size) % k)))
    return np.array([fold_of[p] for p in players])


def oof_curve(x: np.ndarray, y: np.ndarray, fold: np.ndarray) -> np.ndarray:
    """The curve's prediction for every row from the fit on the other folds."""
    out = np.empty_like(y)
    for f in np.unique(fold):
        train = fold != f
        model = lgb.LGBMRegressor(**CURVE).fit(x[train, None], y[train])
        out[~train] = model.predict(x[~train, None])
    return out


def curve_size(pred: np.ndarray) -> float:
    """A curve's size: the standard deviation of its predictions over the rows (0 = flat, calibrated)."""
    return float(np.std(pred))


def shuffled_within_seasons(x: np.ndarray, seasons: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """The input shuffled among the player-seasons of each season, which keeps each season's spread."""
    shuffled = x.copy()
    for season in np.unique(seasons):
        rows = np.flatnonzero(seasons == season)
        shuffled[rows] = shuffled[rng.permutation(rows)]
    return shuffled


def null_sizes(x, y, fold, seasons, rng, n_perms: int) -> np.ndarray:
    """The curve size with the input shuffled within seasons, `n_perms` times."""
    return np.array([curve_size(oof_curve(shuffled_within_seasons(x, seasons, rng), y, fold))
                     for _ in range(n_perms)])


def clustered_bins(x, y, players, n_bins: int) -> pd.DataFrame:
    """Equal-count bins of `x`: the median input, the rows, the mean miss and its standard error clustered by player,
    sqrt(sum over players of (sum of his deviations from the bin mean)^2) / rows.  Bins under 30 rows are dropped."""
    edges = np.unique(np.quantile(x, np.linspace(0, 1, n_bins + 1)))
    which = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, len(edges) - 2)
    rows = []
    for b in range(len(edges) - 1):
        m = which == b
        if m.sum() < MIN_BIN_ROWS:
            continue
        mean = float(y[m].mean())
        dev = pd.Series(y[m] - mean).groupby(players[m]).sum().to_numpy()
        se = float(np.sqrt((dev ** 2).sum()) / m.sum())
        rows.append(dict(bin=b, x_mid=float(np.median(x[m])), n=int(m.sum()), mean_miss=mean, se=se))
    return pd.DataFrame(rows)


def calibrate_input(name: str, x, y, players, seasons, fold, rng, n_perms: int, n_bins: int) -> tuple:
    """(the input's summary, its bins, what the picture draws) for one input on one side."""
    size = curve_size(oof_curve(x, y, fold))
    null = null_sizes(x, y, fold, seasons, rng, n_perms)
    z = float((size - null.mean()) / null.std(ddof=1)) if null.std(ddof=1) > 0 else np.nan
    bins = clustered_bins(x, y, players, n_bins)
    bottom, top = bins.iloc[0], bins.iloc[-1]
    label = AXIS[name][1] if name in AXIS else meaning(name)
    summary = dict(plain=label, curve_size=size, null_size=float(null.mean()), z=z, bottom_bin_miss=bottom.mean_miss,
                   bottom_bin_se=bottom.se, top_bin_miss=top.mean_miss, top_bin_se=top.se)
    # for the picture: one fit on every row, read on a grid (the out-of-fold curve above sets the size and z)
    grid = np.linspace(*np.quantile(x, [0.005, 0.995]), 200)
    shown = lgb.LGBMRegressor(**CURVE).fit(x[:, None], y).predict(grid[:, None])
    print(f"  {label:42s} curve {size:.3f} (null {null.mean():.3f}), z {z:+.1f}; bottom bin {bottom.mean_miss:+.3f}"
          f" +/- {bottom.se:.3f}, top bin {top.mean_miss:+.3f} +/- {top.se:.3f}", flush=True)
    return summary, bins, (bins, grid, shown, label)


def calibrate_side(panel: pd.DataFrame, alpha: pd.DataFrame, side_name: str, n_perms: int, n_bins: int) -> tuple:
    """(summary rows, bins, curves) for every input of one side that varies and has no missing value."""
    inputs = list(FEATURE_SETS[FEATURE_SET][SIDE[side_name]])
    merged = side_rows(panel, alpha, side_name, inputs)
    y, players, seasons = merged.alpha_good.to_numpy(float), merged.player_id.to_numpy(), merged.season.to_numpy()
    fold = player_folds(players)
    rng = np.random.default_rng(1)
    print(f"{side_name}: {len(merged):,} player-seasons with a with/without correction; mean miss {y.mean():+.3f}",
          flush=True)
    rows, bins_out, curves = [], [], {}
    for name in inputs + list(AXIS):
        x = merged[name].to_numpy(float)
        if np.nanstd(x) == 0 or np.isnan(x).any():
            continue
        summary, bins, curve = calibrate_input(name, x, y, players, seasons, fold, rng, n_perms, n_bins)
        curves[(side_name, name)] = curve
        rows.append(dict(side=side_name, input=name, **summary, rows=len(merged)))
        bins_out.append(bins.assign(side=side_name, input=name))
    return rows, bins_out, curves


# ----------------------------------------------------------------------------------------------- the picture
def plot_picks(table: pd.DataFrame) -> list:
    """(side, input) to draw: steals and blocks, then the side's four highest z."""
    picks = []
    for side in ("defense", "offense"):
        t = table[table.side == side]
        forced = [n for n in AXIS if n in set(t.input)]
        ranked = t.sort_values("z", ascending=False).input
        top = [n for n in ranked if n not in forced + ["stl", "blk"]][:SHOWN_PER_SIDE]
        picks += [(side, n) for n in forced + top]
    return picks


def draw_curve(ax, key: tuple, curve: tuple, z: float) -> None:
    """One input: the bins with 95% bars, the curve fitted on every row, and the zero line."""
    bins, xs, ys, label = curve
    ax.axhline(0, color=MUTED, lw=1)
    ax.errorbar(bins.x_mid, bins.mean_miss, yerr=1.96 * bins.se, fmt="o", ms=3.5, color=BLUE, ecolor=BLUE, elinewidth=1,
                alpha=0.85)
    ax.plot(xs, ys, color=INK, lw=2)
    ax.set_title(f"{key[0]}: {label}  (z {z:+.1f})", fontsize=10, color=INK, loc="left")
    ax.grid(axis="y", color=GRID, lw=0.8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.set_ylabel("miss (pts/100, + = underrated)", fontsize=8, color=MUTED)


def plot(table: pd.DataFrame, curves: dict, path: Path) -> None:
    """One panel per picked input, three to a row: the binned misses and the tree curve; written as a PNG."""
    picks = plot_picks(table)
    cols = 3
    rows_n = int(np.ceil(len(picks) / cols))
    fig = Figure(figsize=(cols * 4.6, rows_n * 3.3))
    axes = fig.subplots(rows_n, cols, squeeze=False)
    for ax, key in zip(axes.flat, picks):
        draw_curve(ax, key, curves[key], float(table[(table.side == key[0]) & (table.input == key[1])].z.iloc[0]))
    for ax in list(axes.flat)[len(picks):]:
        ax.axis("off")
    fig.suptitle("Rating miss against each input: dots = 20 bins (95% bars), line = tree curve on all rows (its size "
                 "and z: out of fold); flat at 0 = calibrated", fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=110)
    print(f"wrote {path}")


def calibrate(args: argparse.Namespace) -> None:
    """Every input on both sides: write the summary, the bins and the picture, then print the summary."""
    cfg = load_config()
    alpha = read_misses(resolve_path(cfg, args.alpha))
    panel = read_panel(cfg)
    rows, bins, curves = [], [], {}
    for side_name in SIDE:
        side_summary, side_bins, side_curves = calibrate_side(panel, alpha, side_name, args.perms, args.bins)
        rows += side_summary
        bins += side_bins
        curves |= side_curves
    table = pd.DataFrame(rows).sort_values(["side", "z"], ascending=[True, False])
    outputs = resolve(cfg, "outputs")
    (outputs / "csv").mkdir(parents=True, exist_ok=True)
    table.to_csv(outputs / "csv" / f"stat_calibration_{args.tag}.csv", index=False)
    pd.concat(bins, ignore_index=True).to_csv(outputs / "csv" / f"stat_calibration_{args.tag}_bins.csv", index=False)
    plot(table, curves, outputs / f"stat_calibration_{args.tag}.png")
    with pd.option_context("display.width", 220):
        print(table[list(SUMMARY)].round(3).to_string(index=False))


def main(argv=None) -> None:
    use_utf8()
    args = parse_args(argv)
    with one_thread():
        calibrate(args)


if __name__ == "__main__":
    main()
