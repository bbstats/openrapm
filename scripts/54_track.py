"""The iterate-and-improve tracker: one system, the K=3 criterion, one row in docs/progress.csv, one chart.

    python scripts/54_track.py --system=mspi_linear+sat --calmap=outputs/calmap_chain.parquet --label="what changed"
            [--k=3] [--workers=4] [--when=2026-09-05T12:00] [--dry]

Loss  = the pooled team-game MSE of the held-out seasons (holdout.pooled, column `game`), K = 3 -- the
        shipped block length.  Lower is better.
Time  = the sum over the 28 held-out fits of the fit's wall seconds (the RESULT_COLUMNS `seconds`: design
        build, prior, ridge; not the scoring, not the tuning of any map).
True loss = (loss / loss_0) * (time / time_0), both relative to the FIRST row of docs/progress.csv.
The chart is docs/progress.png: the loss over time and the true loss over time.
"""
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eracoef.config import load_config  # noqa: E402
from eracoef.holdout import Holdout, pooled, run_parallel  # noqa: E402

cfg = load_config()
ROOT = Path(cfg["_root"])
OUT = ROOT / "outputs"
DOCS = ROOT / "docs"
LOG = DOCS / "progress.csv"
PNG = DOCS / "progress.png"

BLUE, ORANGE, TEXT, TEXT2, SURFACE, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#fcfcfb", "#e6e5e1"


def _flag(name, default=None):
    hit = [a for a in sys.argv[1:] if a.startswith(f"--{name}=")]
    return hit[0].split("=", 1)[1] if hit else default


def measure(system: str, k: int, calmap=None, workers: int = 4) -> dict:
    ho = Holdout.from_config(cfg, ks=[k])
    t0 = time.time()
    res, _, _ = run_parallel(ho, [system], out=OUT / f"holdout_track_{system}.parquet", verbose=False,
                             workers=workers, calmap=calmap)
    wall = time.time() - t0
    r = res[(res.split == "all") & (res.group == "all")]
    P = pooled(r).iloc[0]
    return dict(game=float(P.game), stint=float(P.mse), scale_off=float(P.scale_off), scale_def=float(P.scale_def),
                seconds=float(r.seconds.sum()), wall=float(wall), seasons=int(r.held_out.nunique()))


def append(row: dict) -> pd.DataFrame:
    log = pd.read_csv(LOG) if LOG.exists() else pd.DataFrame()
    log = pd.concat([log, pd.DataFrame([row])], ignore_index=True)
    log["when"] = pd.to_datetime(log["when"], format="mixed")
    log = log.sort_values("when").reset_index(drop=True)
    ref = log[log.label.str.startswith("baseline")]
    ref = ref.iloc[0] if len(ref) else log.iloc[0]
    g0, t0 = float(ref.game), float(ref.seconds)
    log["loss_frac"] = log.game / g0
    log["time_frac"] = log.seconds / t0
    log["true_loss"] = log.loss_frac * log.time_frac
    log.to_csv(LOG, index=False)
    return log


def chart(log: pd.DataFrame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(10, 7.2), sharex=True, facecolor=SURFACE)
    x = log["when"]
    panels = [(axes[0], log.game, "Held-out team-game error (points per 100, squared), K = 3, 28 seasons", BLUE, "{:.2f}"),
              (axes[1], log.true_loss, "True loss = (error / first) x (fit time / first)", ORANGE, "{:.3f}")]
    for ax, y, title, col, fmt in panels:
        ax.set_facecolor(SURFACE)
        ax.plot(x, y, color=col, lw=2, marker="o", ms=6, mec=SURFACE, mew=1.5, zorder=3)
        ax.axhline(float(y[log.label.str.startswith("baseline")].iloc[0]) if log.label.str.startswith("baseline").any() else float(y.iloc[0]), color=TEXT2, lw=1, ls=(0, (4, 4)), zorder=1)
        ax.text(x.iloc[-1], y.iloc[-1], "  " + fmt.format(float(y.iloc[-1])), color=TEXT, va="center", fontsize=10)
        best = int(np.argmin(y.to_numpy()))
        if best != len(y) - 1:
            ax.text(x.iloc[best], y.iloc[best], "  " + fmt.format(float(y.iloc[best])), color=TEXT2, va="center", fontsize=9)
        ax.set_title(title, loc="left", color=TEXT, fontsize=11)
        ax.grid(True, axis="y", color=GRID, lw=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=TEXT2, labelsize=9)
    axes[1].set_yscale("log")
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d %H:%M"))
    pad = pd.Timedelta(hours=1) if len(x) < 2 else (x.iloc[-1] - x.iloc[0]) * 0.12 + pd.Timedelta(minutes=30)
    axes[1].set_xlim(x.iloc[0] - pad, x.iloc[-1] + pad)
    last = log.iloc[-1]
    fig.text(0.01, 0.01, f"latest: {last.label}  |  {last.system}  |  fit time {last.seconds:.0f}s for 28 fits  |  "
             f"{pd.Timestamp(last.when):%Y-%m-%d %H:%M}", color=TEXT2, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(PNG, dpi=130, facecolor=SURFACE)


def main():
    system = _flag("system", "mspi_linear+sat")
    k = int(_flag("k", 3))
    label = _flag("label", system)
    when = pd.Timestamp(_flag("when") or datetime.now()).strftime("%Y-%m-%d %H:%M:%S")
    if "--dry" in sys.argv:
        chart(pd.read_csv(LOG, parse_dates=["when"]))
        return
    m = measure(system, k, calmap=_flag("calmap"), workers=int(_flag("workers", 4)))
    row = dict(when=when, label=label, system=system, k=k, **m)
    log = append(row)
    chart(log)
    show = log[["when", "label", "system", "game", "seconds", "wall", "loss_frac", "time_frac", "true_loss"]].copy()
    print(show.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    if "--push" in sys.argv:
        subprocess.run(["git", "add", str(LOG), str(PNG)], cwd=ROOT, check=True)
        subprocess.run(["git", "commit", "-q", "-m", f"Progress: {label} ({m['game']:.3f}, {m['seconds']:.0f}s)"], cwd=ROOT, check=True)
        subprocess.run(["git", "push", "-q"], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
