"""Write each player-season's box COUNTS and padding constants beside the season panel, so a multi-season training row
can be padded once on its summed counts (the Robustness pass, experiment 48: `singleyear.aggregate(pad_once=True)`).

    python scripts/139_count_panel.py [--panel=outputs/role_panel_season.parquet] [--out=outputs/role_panel_counts.parquet]

The panel stores each season's 13 box rates already padded: (n * rate + k * target) / (n + k), with n his possessions
on the side, k the season's padding constant for the stat (`exposure.split_half_k`) and the target the season's mean
for players in his possession bin (`pad_target: poss_conditional`).  Averaging those padded rates over several seasons
-- what a career or chunk row does today -- shrinks a ten-season row as hard as a single season, so the booster is
trained on rows narrower than their evidence and learns too steep a price for a noisy stat (steals most: k = 440
possessions against 46-269 for the others).  Padding once needs what the panel does not keep: the counts, and each
season's k and target per player.

For every season: the offensive design (`xpts_ft`, the one 49 rated the panel's offence on; the counts and possessions
are the same in the defensive one), its `BoxExposure` refitted exactly as 49's `apm_fit` fits it (mode full, the
configured padding target), and per player and side:

  cnt_<stat>    his count of the stat over the season's games, regular season and playoffs (the panel's phases)
  padk_<stat>   the padding constant his rate was padded with on this side (possessions)
  padt_<stat>   the padding target (per 100, uncentred)

Checks, before anything is written: rebuilding every rate from these columns gives the panel's `raw_*` on both sides
(largest difference must be under 1e-9), and the possessions are the panel's `poss`.  The panel itself is not touched.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eracoef import singleyear as sy  # noqa: E402
from eracoef.config import load_config  # noqa: E402
from eracoef.cv import make_exposure  # noqa: E402
from eracoef.design import FEATURES  # noqa: E402
from eracoef.holdout import Context  # noqa: E402
from eracoef.pad import shrink  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
from _cli import check_flags, flag as _flag   # noqa: E402


def season_counts(ctx, cfg, season: int) -> pd.DataFrame:
    """One season's counts, constants and targets, one row per (player, side)."""
    wd = ctx.design([season], "xpts_ft")
    exp = make_exposure(wd, mode="full", pad_target=cfg["pad_target"])
    exp.fit(wd.X, wd.y, sample_weight=wd.w)
    ids = wd.spec.ps_table["player_id"].to_numpy()
    counts = pd.DataFrame(np.asarray(exp.totals_, dtype=float), columns=sy.PAD_COUNTS)
    parts = []
    for side, poss, n_eff, k, target in (("O", exp.poss_off_, exp.poss_off_eff_, exp.pad_k_ps_, exp.target_ps_),
                                         ("D", exp.poss_def_, exp.poss_def_eff_, exp.pad_k_ps_d_, exp.target_ps_d_)):
        # the panel pads on true possessions (no game weights), so the effective count is the count itself
        assert np.allclose(n_eff, poss, rtol=0, atol=1e-9), f"{season} {side}: weighted games in the panel's padding"
        frame = pd.concat([pd.DataFrame({"season": season, "side": side, "player_id": ids,
                                         "pad_poss": np.asarray(poss, dtype=float)}),
                           counts,
                           pd.DataFrame(np.asarray(k, dtype=float), columns=sy.PAD_K),
                           pd.DataFrame(np.asarray(target, dtype=float), columns=sy.PAD_TARGET)], axis=1)
        parts.append(frame)
    return pd.concat(parts, ignore_index=True)


def rebuilt_rates(frame: pd.DataFrame) -> np.ndarray:
    """The padded per-100 rates the columns imply: what `singleyear.aggregate(pad_once=True)` makes of one row."""
    poss = frame.pad_poss.to_numpy(float)
    C = frame[sy.PAD_COUNTS].to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = np.where(poss[:, None] > 0, 100.0 * C / np.where(poss > 0, poss, 1.0)[:, None], 0.0)
    return shrink(rate, poss[:, None], frame[sy.PAD_K].to_numpy(float), frame[sy.PAD_TARGET].to_numpy(float))


def main():
    check_flags()      # refuse a flag this script does not understand (scripts/_cli.py)
    path = ROOT / _flag("panel", "outputs/role_panel_season.parquet")
    out = ROOT / _flag("out", "outputs/role_panel_counts.parquet")
    cfg = load_config(ROOT / "config.yaml")
    ctx = Context.load(cfg)
    panel = pd.read_parquet(path)
    t0 = time.time()
    parts = []
    for season in sorted(int(s) for s in panel.season.unique()):
        parts.append(season_counts(ctx, cfg, season))
        print(f"  {season}: {parts[-1].player_id.nunique()} players ({time.time() - t0:.0f}s)", flush=True)
    new = pd.concat(parts, ignore_index=True)

    # every rate rebuilt from the new columns must be the panel's own, on both sides, or a pad-once row would mix two
    # paddings: the one-season row it reproduces and the panel row the rated season is read from
    keys = ["season", "side", "player_id"]
    check = panel[keys + ["poss"] + [f"raw_{c}" for c in FEATURES]].merge(new, on=keys, how="left", indicator=True)
    missing = int((check["_merge"] != "both").sum())
    assert missing == 0, f"{missing} panel rows have no counts"
    gap_rate = float(np.nanmax(np.abs(rebuilt_rates(check) - check[[f"raw_{c}" for c in FEATURES]].to_numpy(float))))
    gap_poss = float(np.max(np.abs(check.pad_poss.to_numpy(float) - check.poss.to_numpy(float))))
    print(f"rates rebuilt from the counts against the panel's raw_*: largest difference {gap_rate:.2e}; "
          f"possessions against the panel's poss {gap_poss:.2e}")
    assert gap_rate < 1e-9 and gap_poss < 1e-9, "the counts do not reproduce the panel's padded rates"
    new = new.merge(panel[keys], on=keys, how="inner")
    new.to_parquet(out, index=False)
    k = new[new.side == "O"]
    print(f"wrote {out.relative_to(ROOT)}: {len(new)} rows; padding constant by stat, offence, median over seasons: "
          + ", ".join(f"{c} {k.groupby('season')[f'padk_{c}'].median().median():.0f}" for c in FEATURES))


if __name__ == "__main__":
    main()
