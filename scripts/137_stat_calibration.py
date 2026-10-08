"""Calibration curves: is the rating over- or underrated as a function of each input stat?

    python scripts/137_stat_calibration.py [--alpha=outputs/tradeset_lgb_noonc_rs_alpha.parquet] [--tag=lgb_noonc_rs]
                                           [--perms=20] [--bins=20]

The owner, 2026-10-08, after the fifths of `95_miss_by_group.py` showed steals overvalued on defence: "I almost would
rather do a calibration plot style rather than binning ... with overrating / underrating we should see y = 0 ... based
on steals per 100", and "more like kbins or tree based model to get any non-monotonicity".

**The miss** is 95's: each player-season's with/without correction (the trade set's alpha behind the trade loss, in
points per 100, positive = his team's games with and without him say he is better than rated), on the rows the trade
loss reads (one-team players whose team played neighbouring games without them).  A well-calibrated rating has a miss
curve flat at zero against every input: no stat whose high values are systematically over- or underrated.

**Per input and side**, against the input exactly as the prior reads it (the season panel through
`singleyear.season_frame`; steals and blocks also on their uncentred padded rates per 100 for the axis):
  * twenty equal-count bins: the mean miss and its standard error, clustered by player (a player's seasons share his
    errors);
  * a one-input LightGBM curve (shallow, many rows a leaf, so it bends where the data do and nowhere else), its
    predictions out of player fold, so the curve's size is honest;
  * the curve's size (the spread of the out-of-fold curve, points per 100) and a permutation z: the same fit with the
    input shuffled among the player-seasons of each season, `--perms` times.  z above 3 is a stat the ratings
    mis-price as a whole.

Writes outputs/csv/stat_calibration_<tag>.csv (one row per input and side), outputs/csv/stat_calibration_<tag>_bins.csv
(every bin and curve point) and outputs/stat_calibration_<tag>.png (the curves for steals, blocks and the most
mis-priced inputs).  Reads only; changes nothing.
"""
import os
import sys
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _cli import check_flags, flag  # noqa: E402
from eracoef import singleyear as sy  # noqa: E402
from eracoef.config import load_config  # noqa: E402
from eracoef.glossary import plain  # noqa: E402
from eracoef.seasons import drop_untrainable  # noqa: E402

CURVE = dict(n_estimators=150, learning_rate=0.05, num_leaves=6, min_child_samples=400, reg_lambda=10.0,
             subsample=1.0, colsample_bytree=1.0, n_jobs=1, verbose=-1, random_state=0)
SIDE = {"offense": "O", "defense": "D"}


def player_folds(players: np.ndarray, k: int = 5) -> np.ndarray:
    keys = np.unique(players)
    rng = np.random.default_rng(0)
    fold_of = dict(zip(keys, rng.permutation(np.arange(keys.size) % k)))
    return np.array([fold_of[p] for p in players])


def oof_curve(x: np.ndarray, y: np.ndarray, fold: np.ndarray) -> np.ndarray:
    import lightgbm as lgb
    out = np.empty_like(y)
    for f in np.unique(fold):
        train = fold != f
        model = lgb.LGBMRegressor(**CURVE).fit(x[train, None], y[train])
        out[~train] = model.predict(x[~train, None])
    return out


def curve_size(pred: np.ndarray) -> float:
    return float(np.std(pred))


def clustered_bins(x, y, players, n_bins):
    edges = np.unique(np.quantile(x, np.linspace(0, 1, n_bins + 1)))
    which = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, len(edges) - 2)
    rows = []
    for b in range(len(edges) - 1):
        m = which == b
        if m.sum() < 30:
            continue
        mean = float(y[m].mean())
        dev = pd.Series(y[m] - mean).groupby(players[m]).sum().to_numpy()
        se = float(np.sqrt((dev ** 2).sum()) / m.sum())
        rows.append(dict(bin=b, x_mid=float(np.median(x[m])), n=int(m.sum()), mean_miss=mean, se=se))
    return pd.DataFrame(rows)


def main() -> None:
    check_flags()
    alpha_path = ROOT / flag("alpha", "outputs/tradeset_lgb_noonc_rs_alpha.parquet")
    tag, n_perms, n_bins = flag("tag", "lgb_noonc_rs"), int(flag("perms", 20)), int(flag("bins", 20))
    cfg = load_config(ROOT / "config.yaml")
    alpha = pd.read_parquet(alpha_path)
    alpha = alpha[alpha.eligible & (alpha.without_poss >= 1)]
    panel, _ = drop_untrainable(pd.read_parquet(ROOT / "outputs/role_panel_season.parquet"), cfg, what="the season panel")
    panel = panel[panel.poss > 0]
    feats = sy.feature_set("boruta_noonc")
    rows, bins_out, keep_curves = [], [], {}
    for side_name, side in SIDE.items():
        inputs = [f for f in feats[side] if f != sy.SAME_TEAM]
        frame = sy.season_frame(panel[panel.side == side], inputs)
        frame = frame.assign(stl_per100=panel.loc[panel.side == side, "raw_stl"].to_numpy(float),
                             blk_per100=panel.loc[panel.side == side, "raw_blk"].to_numpy(float))
        a = alpha[alpha.side == side_name][["player_id", "season", "alpha_good"]]
        m = frame.merge(a, on=["player_id", "season"], how="inner")
        y, players, seasons = m.alpha_good.to_numpy(float), m.player_id.to_numpy(), m.season.to_numpy()
        fold = player_folds(players)
        rng = np.random.default_rng(1)
        print(f"{side_name}: {len(m):,} player-seasons with a with/without correction; mean miss {y.mean():+.3f}",
              flush=True)
        for name in inputs + ["stl_per100", "blk_per100"]:
            x = m[name].to_numpy(float)
            if np.nanstd(x) == 0 or np.isnan(x).any():
                continue
            pred = oof_curve(x, y, fold)
            size = curve_size(pred)
            null = []
            for _ in range(n_perms):
                xs = x.copy()
                for s in np.unique(seasons):          # shuffle within season: keeps each season's spread
                    idx = np.flatnonzero(seasons == s)
                    xs[idx] = xs[rng.permutation(idx)]
                null.append(curve_size(oof_curve(xs, y, fold)))
            null = np.array(null)
            z = float((size - null.mean()) / null.std(ddof=1)) if null.std(ddof=1) > 0 else np.nan
            b = clustered_bins(x, y, players, n_bins)
            lo, hi = b.iloc[0], b.iloc[-1]
            label = {"stl_per100": "steals per 100 (axis)", "blk_per100": "blocks per 100 (axis)"}.get(name, plain(name))
            rows.append(dict(side=side_name, input=name, plain=label, curve_size=size, null_size=float(null.mean()),
                             z=z, bottom_bin_miss=lo.mean_miss, bottom_bin_se=lo.se, top_bin_miss=hi.mean_miss,
                             top_bin_se=hi.se, rows=len(m)))
            # for the picture: one fit on every row, read on a grid (the out-of-fold curve above sets the size and z)
            import lightgbm as lgb
            grid = np.linspace(*np.quantile(x, [0.005, 0.995]), 200)
            shown = lgb.LGBMRegressor(**CURVE).fit(x[:, None], y).predict(grid[:, None])
            keep_curves[(side_name, name)] = (b, grid, shown, label)
            bins_out.append(b.assign(side=side_name, input=name))
            print(f"  {label:42s} curve {size:.3f} (null {null.mean():.3f}), z {z:+.1f}; bottom bin {lo.mean_miss:+.3f}"
                  f" +/- {lo.se:.3f}, top bin {hi.mean_miss:+.3f} +/- {hi.se:.3f}", flush=True)
    table = pd.DataFrame(rows).sort_values(["side", "z"], ascending=[True, False])
    out_csv = ROOT / "outputs" / "csv"
    out_csv.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_csv / f"stat_calibration_{tag}.csv", index=False)
    pd.concat(bins_out, ignore_index=True).to_csv(out_csv / f"stat_calibration_{tag}_bins.csv", index=False)
    plot(table, keep_curves, ROOT / "outputs" / f"stat_calibration_{tag}.png")
    with pd.option_context("display.width", 220):
        print(table[["side", "plain", "curve_size", "null_size", "z", "bottom_bin_miss", "top_bin_miss"]]
              .round(3).to_string(index=False))


def plot(table, curves, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    picks = []
    for side in ("defense", "offense"):
        t = table[table.side == side]
        forced = [n for n in ("stl_per100", "blk_per100") if n in set(t.input)]
        top = [n for n in t.sort_values("z", ascending=False).input if n not in forced + ["stl", "blk"]][:4]
        picks += [(side, n) for n in forced + top]
    cols = 3
    rows_n = int(np.ceil(len(picks) / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(cols * 4.6, rows_n * 3.3), squeeze=False)
    ink, muted, grid, blue = "#0b0b0b", "#52514e", "#e1e0d9", "#2a78d6"
    for ax, key in zip(axes.flat, picks):
        b, xs, ys, label = curves[key]
        z = float(table[(table.side == key[0]) & (table.input == key[1])].z.iloc[0])
        ax.axhline(0, color=muted, lw=1)
        ax.errorbar(b.x_mid, b.mean_miss, yerr=1.96 * b.se, fmt="o", ms=3.5, color=blue, ecolor=blue, elinewidth=1,
                    alpha=0.85)
        ax.plot(xs, ys, color=ink, lw=2)
        ax.set_title(f"{key[0]}: {label}  (z {z:+.1f})", fontsize=10, color=ink, loc="left")
        ax.grid(axis="y", color=grid, lw=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=muted, labelsize=8)
        ax.set_ylabel("miss (pts/100, + = underrated)", fontsize=8, color=muted)
    for ax in list(axes.flat)[len(picks):]:
        ax.axis("off")
    fig.suptitle("Rating miss against each input: dots = 20 bins (95% bars), line = tree curve on all rows (its size and z: out of fold); flat at 0 "
                 "= calibrated", fontsize=11, color=ink, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=110)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
