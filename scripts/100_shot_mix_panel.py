"""Write each player-season's SHOT MIX into the season panel: how good his teammates' shot locations are with him
on the floor against without him.

    python scripts/100_shot_mix_panel.py [--panel=outputs/role_panel_season.parquet] [--first=1997] [--last=2026]

The owner's idea (2026-10-04): "teammate shot quality while on court".  For every pair of teammates X and T who
shared the floor in a season (regular season and playoffs, `design.SEASON_PHASES`), T's expected points per
field-goal attempt with X on the floor minus T's without him, where a shot's expected points are the LEAGUE's
make rate at that distance that season times its value (`shotcurve.py`, carried per lineup slot in the stints as
`xl2_s*` / `xl3_s*`) -- the shot's location, not who took it or whether it went in.  X's raw value is the
average of those differences over his teammates, each weighted 1 / (1/on + 1/off) in T's attempts, the
inverse of a difference's sampling variance up to a constant; `mix_n` is the sum of those weights.

Padding (the owner's rule): `mix_lift = pad.shrink(raw, mix_n, k, target)`, target the season's weighted mean.
`k` is chosen ONCE, pooled over every season, as the value at which a player's half-A number best predicts his
half-B number and the reverse (the stints' own game halves), weighted by the predicted half's `mix_n`.

The "makes" twin (teammates' makes over their own season rates, him on) was measured on 2025 at a split-half
correlation of -0.37 and is not built.  The panel is backed up to `<panel>_pre_mix.parquet` first.
"""
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eracoef.design import SEASON_PHASES  # noqa: E402
from eracoef.pad import shrink  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

from _cli import check_flags, flag as _flag   # noqa: E402

K_GRID = np.unique(np.round(np.geomspace(5, 20000, 120)))


def pair_totals(st: pd.DataFrame) -> pd.DataFrame:
    """Per (x, t, half): T's attempts and expected points while X shared the floor, plus T's totals per half."""
    parts = []
    for side in ("h", "a"):
        P = st[[f"{side}{k}" for k in range(1, 6)]].to_numpy(dtype=np.int64)
        fga = np.stack([(st[f"fg2a_s{k}_{side}"] + st[f"fg3a_s{k}_{side}"]).to_numpy(float) for k in range(1, 6)], 1)
        xp = np.stack([(2 * st[f"xl2_s{k}_{side}"] + 3 * st[f"xl3_s{k}_{side}"]).to_numpy(float)
                       for k in range(1, 6)], 1)
        half = st["half"].to_numpy()
        for i in range(5):
            for j in range(5):
                if i != j:
                    parts.append(pd.DataFrame({"x": P[:, i], "t": P[:, j], "half": half,
                                               "fga": fga[:, j], "xp": xp[:, j]}))
    d = pd.concat(parts, ignore_index=True)
    d = d[(d.x > 0) & (d.t > 0)]
    return d.groupby(["x", "t", "half"], as_index=False)[["fga", "xp"]].sum()


def lift(pairs: pd.DataFrame) -> pd.DataFrame:
    """Per X: the weighted mean over teammates of (T's xPPS with X on) - (T's xPPS with X off), and its weight."""
    on = pairs.groupby(["x", "t"], as_index=False)[["fga", "xp"]].sum()
    # T's totals: each of T's shots appears once per teammate on the floor (four), so divide by four
    tot = on.groupby("t")[["fga", "xp"]].sum() / 4.0
    on = on.join(tot, on="t", rsuffix="_tot")
    off_fga = on.fga_tot - on.fga
    off_xp = on.xp_tot - on.xp
    ok = (on.fga > 0) & (off_fga > 0)
    on, off_fga, off_xp = on[ok], off_fga[ok], off_xp[ok]
    delta = on.xp / on.fga - off_xp / off_fga
    w = 1.0 / (1.0 / on.fga + 1.0 / off_fga)
    g = pd.DataFrame({"x": on.x, "wd": w * delta, "w": w}).groupby("x").sum()
    return pd.DataFrame({"raw": g.wd / g.w, "n": g.w})


def season_tables(season: int, src: Path) -> tuple:
    st = pd.concat([pd.read_parquet(src / f"{season}_{ph}.parquet") for ph in SEASON_PHASES
                    if (src / f"{season}_{ph}.parquet").exists()], ignore_index=True)
    # the halves are per game and per phase, so they stay balanced within each phase
    pairs = pair_totals(st)
    whole = lift(pairs)
    a = lift(pairs[pairs.half == "A"])
    b = lift(pairs[pairs.half == "B"])
    return whole, a.join(b, lsuffix="_a", rsuffix="_b", how="inner")


def main():
    check_flags()
    path = ROOT / _flag("panel", "outputs/role_panel_season.parquet")
    first, last = int(_flag("first", "1997")), int(_flag("last", "2026"))
    src = ROOT / "data/stints"
    wholes, halves = {}, {}
    t0 = time.time()
    for s in range(first, last + 1):
        wholes[s], halves[s] = season_tables(s, src)
        print(f"{s}: {len(wholes[s])} players, {time.time() - t0:.0f}s", flush=True)

    # the season target: the weighted mean of the raw values (on minus off need not average to zero exactly)
    targets = {s: float(np.average(w.raw, weights=w.n)) for s, w in wholes.items()}
    hv = pd.concat([h.assign(season=s, tgt=targets[s]) for s, h in halves.items()], ignore_index=True)

    def loss(k):
        pa = shrink(hv.raw_a, hv.n_a, k, hv.tgt)
        pb = shrink(hv.raw_b, hv.n_b, k, hv.tgt)
        return float(np.sum(hv.n_b * (hv.raw_b - pa) ** 2) + np.sum(hv.n_a * (hv.raw_a - pb) ** 2))

    losses = np.array([loss(k) for k in K_GRID])
    k = float(K_GRID[losses.argmin()])
    r = np.corrcoef(hv.raw_a, hv.raw_b)[0, 1]
    big = hv[(hv.n_a > np.median(hv.n_a)) & (hv.n_b > np.median(hv.n_b))]
    print(f"\nk = {k:.0f} (weight units); half-season raw correlation {r:.3f} over {len(hv)} player-seasons, "
          f"{np.corrcoef(big.raw_a, big.raw_b)[0, 1]:.3f} for the heavier half")
    if k in (K_GRID[0], K_GRID[-1]):
        raise SystemExit(f"k at the edge of its grid ({k}); widen K_GRID")

    rows = []
    for s, w in wholes.items():
        rows.append(pd.DataFrame({"player_id": w.index.to_numpy(), "season": s,
                                  "mix_lift": shrink(w.raw, w.n, k, targets[s]) - targets[s],
                                  "mix_n": w.n.to_numpy()}))
    mix = pd.concat(rows, ignore_index=True)
    trust = mix.mix_n / (mix.mix_n + k)
    print(f"share of a season's own number kept after padding (median, players above the median weight): "
          f"{trust[mix.mix_n > mix.mix_n.median()].median():.2f}; spread 10th-90th "
          f"{mix.mix_lift.quantile(.1):+.4f} to {mix.mix_lift.quantile(.9):+.4f} points per teammate shot")

    panel = pd.read_parquet(path)
    backup = path.with_name(path.stem + "_pre_mix.parquet")
    if not backup.exists():
        shutil.copy(path, backup)
    panel = panel.drop(columns=[c for c in ("mix_lift", "mix_n") if c in panel.columns])
    panel = panel.merge(mix, on=["player_id", "season"], how="left")
    missing = panel.mix_lift.isna()
    print(f"{int(missing.sum())} panel rows with no shared-floor teammate shots -> 0 (the season mean); "
          f"their possessions: {panel.poss[missing].sum():.0f} of {panel.poss.sum():.0f}")
    panel["mix_lift"] = panel.mix_lift.fillna(0.0)
    panel["mix_n"] = panel.mix_n.fillna(0.0)
    panel.to_parquet(path, index=False)
    print(f"wrote {path} ({len(panel)} rows); backup {backup.name}")


if __name__ == "__main__":
    main()
