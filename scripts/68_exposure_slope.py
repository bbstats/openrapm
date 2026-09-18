"""One slope per side on log possessions: the parsimonious exposure correction to the rankings.

    python scripts/68_exposure_slope.py [--rankings=season_ratings_sy_lam13037] [--out=season_ratings_slope]
                                        [--apm=apm_by_player_season] [--exclude_neighbours=1] [--quad=0] [--mult=1.0] [--match_spread=0]

**What this corrects.**  The box prior is FLAT in exposure where the truth is steep.  Centred at
possession-weighted zero, over 30 seasons, the prior hands -0.76 to a man with 0-50 possessions and -0.81
to a man with 1,000-2,000 -- a difference of 0.05, standard error 0.03 -- while APM runs from -9.94 to
-2.15 across the same range, monotone, z 7 to 15.  The ridge's free prior scale stretches the prior and
fixes about 40% of that compression; the rest is still there in the finished rankings.

**Why the APM of the season with few possessions is not the target.**  A man with 0-50 possessions reads -9.25 that season but
-5.15 the season before and -3.64 the season after, and in those seasons he plays about 700 possessions
so he is properly measured.  Only **48%** of the deficit from the season with few possessions persists; by 200 possessions it is
86% and by 1,000 it is essentially all of it.  Both directions are measured because forward alone cannot
separate persistence from ageing -- the deficit is about the same looking back as looking forward, so it
is a property of the man, not a trajectory.  The target here is therefore the mean of his neighbouring
seasons' APM, which is a forecasting target and needs no discounting afterwards.

**Why one slope on log10, and not the blend it replaces.**  Every earlier version -- a scalar level
(experiment 12), then a level modelled on eleven covariates (experiment 13) -- used a weight `1 - w(n)`
that goes to a hard constant as possessions fall, and that constant was unbounded: the loss weights each
player by his own information, so a four-possession man costs it nothing and the curve ran to -22.41 per
100.  `log` has no constant to run to and grows so slowly that extrapolation is gentle.  It asks for -4.6
offence at four possessions where the measurement says -3.75 -- an overshoot of 0.9, not of 16.

    rating_off += slope_off * (log10(poss_off) - L0)
    rating_def += slope_def * (log10(poss_def) - L0)      (both positive-good)

Fitted by regressing (the mean of the neighbouring seasons' APM, minus the rating) on log10 of that
side's possessions, one row per player-season, equal weight per player -- equal weight because weighting
by information is exactly what blinded the earlier fits to the men being corrected.  `L0` is the mean of
log10(possessions); it is arbitrary, since the board re-centres at possession-weighted zero afterwards.

**`--exclude_neighbours=1` (the default) refits the slope for every scored season without that season or
either neighbour**, so the two coefficients have not seen the games they are scored on.  This matters:
they are fitted against APM from the neighbouring seasons, which is exactly what `63_yoy.py` scores.

Applied to the rating and to the prior columns alike, so `u_off` / `u_def` -- what the season's own games
added -- are unchanged by it: this is a population-level exposure correction, not evidence.

Reads a rankings table and outputs/<apm>.parquet (scratch/bucket_persist.py builds the latter).  Writes
outputs/<out>.parquet in the same schema, so 63_yoy.py, 66_compare.py and the consensus report all read it.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SIDES = {"O": "off", "D": "def"}


# The shared command line (scripts/_cli.py): `flag` records every name it is asked for so
# `check_flags` below can refuse one that was never asked for.  A misspelled flag used to be
# ignored silently, which is how a run looks right and is wrong.
from _cli import check_flags, flag as _flag, switch   # noqa: E402


def neighbour_apm(apm: pd.DataFrame, side: str) -> pd.DataFrame:
    """Per player-season: the mean of his APM in the seasons either side, and that side's possessions.

    Both neighbours, because forward alone cannot tell a persistent deficit from a declining player.
    """
    d = apm[apm.side == side][["season", "player_id", "apm", "poss"]]
    nxt = d.assign(season=d.season - 1).rename(columns={"apm": "apm_next"})
    prv = d.assign(season=d.season + 1).rename(columns={"apm": "apm_prev"})
    j = (d.merge(nxt[["season", "player_id", "apm_next"]], on=["season", "player_id"], how="left")
          .merge(prv[["season", "player_id", "apm_prev"]], on=["season", "player_id"], how="left"))
    j["target"] = j[["apm_prev", "apm_next"]].mean(axis=1)
    return j.dropna(subset=["target"]).loc[lambda f: f.poss > 0]


def fit_slope(frame: pd.DataFrame, l0: float, quad: bool) -> np.ndarray:
    """Least squares of `target - rating` on (log10(poss) - L0), one row per player, equal weight.

    Equal weight per PLAYER, not per possession: weighting by information is what blinded the earlier
    fits to the very men being corrected.
    """
    x = np.log10(frame.poss.to_numpy(dtype=float)) - l0
    columns = [np.ones(x.size), x] + ([x ** 2] if quad else [])
    beta, *_ = np.linalg.lstsq(np.column_stack(columns), frame.gap.to_numpy(), rcond=None)
    return beta


def main():
    check_flags()      # refuse a flag this script does not understand (scripts/_cli.py)
    rankings = _flag("rankings", "season_ratings_sy_lam13037")
    board = pd.read_parquet(ROOT / "outputs" / f"{rankings}.parquet")
    apm = pd.read_parquet(ROOT / "outputs" / f"{_flag('apm', 'apm_by_player_season')}.parquet")
    exclude = int(_flag("exclude_neighbours", 1))
    quad = _flag("quad", "0") not in ("0", "no", "false")
    # `--mult` scales the fitted slope.  The APM fit gives the SAME-SEASON attribution gradient; the
    # year-over-year test asks how much of it helps FORECASTING, and those need not be the same number.
    mult = float(_flag("mult", 1.0))
    match_spread = _flag("match_spread", "0") not in ("0", "no", "false")
    out = ROOT / "outputs" / f"{_flag('out', 'season_ratings_slope')}.parquet"

    if "poss_def" not in board or bool((board.poss_off == board.poss_def).all()):
        # scripts/62_single_year_board.py copies poss_off into poss_def; the real per-side counts differ
        # by about nine possessions in twenty-five hundred, so this is a rounding matter here, but the
        # slope is defined per side and should read the side it is correcting
        print("note: the rankings table has poss_def == poss_off, so both sides use the same count")

    print(f"rankings {rankings}: {len(board):,} rows, {board.season.nunique()} seasons; "
          f"exclude_neighbours {exclude}; quadratic {quad}; slope multiplier {mult}")

    fitted, pieces = [], []
    for side, suffix in SIDES.items():
        nb = neighbour_apm(apm, side)
        # the rating in the SAME raw sign as this side's APM: offence positive-good, defence points allowed
        rating = board[["player_id", "season", "rating_off", "rating_def", "poss_off", "poss_def"]].copy()
        rating["rating"] = rating.rating_off if side == "O" else -rating.rating_def
        j = nb.merge(rating[["player_id", "season", "rating"]], on=["player_id", "season"], how="inner")
        j["gap"] = j.target - j.rating
        l0 = float(np.log10(j.poss.to_numpy(dtype=float)).mean())

        for season in sorted(board.season.unique()):
            blocked = set(range(season - exclude, season + exclude + 1))
            train = j[~j.season.isin(blocked)]
            beta = fit_slope(train, l0, quad)
            fitted.append(dict(side=side, season=season, n_train=len(train), l0=l0,
                               level=float(beta[0]), slope=float(beta[1]),
                               curve=float(beta[2]) if quad else 0.0))
        # one all-seasons fit, reported so the leave-out spread is readable against it
        beta_all = fit_slope(j, l0, quad)
        per = pd.DataFrame([f for f in fitted if f["side"] == side])
        print(f"\n--- {'offence' if side == 'O' else 'defence (raw sign)'}: {len(j):,} player-seasons")
        print(f"    all seasons together: slope {beta_all[1]:+.4f} per decade"
              + (f", curve {beta_all[2]:+.4f}" if quad else "")
              + f"   (L0 {l0:.4f} = {10 ** l0:,.0f} possessions)")
        print(f"    leave-{exclude}-neighbours-out: slope mean {per.slope.mean():+.4f}, "
              f"sd {per.slope.std():.4f}, min {per.slope.min():+.4f}, max {per.slope.max():+.4f}")
        pieces.append(per)

    F = pd.concat(pieces, ignore_index=True)
    F.to_parquet(out.with_name(out.stem + "_slopes.parquet"), index=False)

    # ---------------------------------------------------------------- apply, then re-centre
    new = board.copy()
    for side, suffix in SIDES.items():
        coef = F[F.side == side].set_index("season")
        n = np.maximum(new[f"poss_{suffix}"].to_numpy(dtype=float), 1.0)
        x = np.log10(n) - coef.l0.reindex(new.season).to_numpy()
        slope = coef.slope.reindex(new.season).to_numpy()
        curve = coef.curve.reindex(new.season).to_numpy()
        # the fit is in each side's RAW sign; the table's rating_def is positive-good, so flip on defence
        delta = (slope * x + curve * x ** 2) * (1.0 if side == "O" else -1.0) * mult
        for column in (f"rating_{suffix}", f"prior_{suffix}"):
            new[column] = new[column].to_numpy() + delta
    new["rating_total"] = new.rating_off + new.rating_def
    new["prior_total"] = new.prior_off + new.prior_def
    # the RAPM convention, same as the board: each season centred at possession-weighted zero per side
    for column in ("rating_off", "rating_def", "prior_off", "prior_def"):
        level = (new.groupby("season").apply(
            lambda g, c=column: np.average(g[c], weights=g.poss_off), include_groups=False)
                 .reindex(new.season).to_numpy())
        new[column] = new[column] - level
    if match_spread:
        # The rankings are ALREADY about 26% too wide on the year-over-year test (scale_off 0.796), so any
        # correction that adds spread is punished on the team-game score whether or not the ORDER improved.
        # The shipped pipeline fits a per-side calibration scale, which absorbs exactly that; DECISIONS.md
        # records the same absorption swallowing 99% of the offensive three-point target.  So hold each
        # side's spread at the incumbent's, per season, and let the test read the order alone.
        for column in ("rating_off", "rating_def"):
            for season, group in new.groupby("season"):
                idx = group.index
                before = board.loc[idx, column].std()
                after = new.loc[idx, column].std()
                if after > 0:
                    new.loc[idx, column] = new.loc[idx, column] * (before / after)
    new["rating_total"] = new.rating_off + new.rating_def
    new["prior_total"] = new.prior_off + new.prior_def
    new["u_off"] = new.rating_off - new.prior_off
    new["u_def"] = new.rating_def - new.prior_def
    new["u_total"] = new.u_off + new.u_def
    new.to_parquet(out, index=False)

    pd.set_option("display.width", 200, "display.max_columns", 20, "display.precision", 2)
    tier = pd.cut(new.poss_off, [0, 50, 100, 200, 500, 1000, 2000, 4000, 1e9])
    move = pd.DataFrame({"tier": tier, "old": board.rating_total.to_numpy(),
                         "new": new.rating_total.to_numpy()})
    move["change"] = move.new - move.old
    print("\n=== what moved, all seasons, points per 100")
    print(move.groupby("tier", observed=True).agg(
        players=("change", "size"), mean_change=("change", "mean"),
        new_mean=("new", "mean"), worst=("new", "min")).round(2).to_string())
    print(f"\n  players below -10 per 100: {int((new.rating_total < -10).sum())} "
          f"(was {int((board.rating_total < -10).sum())});  below -6: "
          f"{int((new.rating_total < -6).sum())} (was {int((board.rating_total < -6).sum())})")

    latest = new[new.season == new.season.max()].nlargest(20, "rating_total")
    print(f"\n=== {int(new.season.max())}, top 20 after the correction")
    print(latest[["player_name", "rating_off", "rating_def", "rating_total", "poss_off"]]
          .to_string(index=False))
    print(f"\nwrote {out} and {out.with_name(out.stem + '_slopes.parquet')}")


if __name__ == "__main__":
    main()
