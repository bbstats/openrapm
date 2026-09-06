"""The iterate-and-improve tracker: one system, the K=3 criterion, one row in docs/progress.csv, one chart.

    python scripts/54_track.py --system=mspi --label="what changed" [--maps=linear+sat] [--k=3] [--workers=4]
            [--when=2026-09-05T12:00] [--dry] [--push]

The system is the UNMAPPED one; the calibration map (calmap.py, `--maps`, default linear+sat on both sides) is
fitted leave-one-season-out on the dumped ratings and the mapped score is what is logged (`game`), next to the
unmapped one (`game_unmapped`).

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


def measure(system: str, k: int, workers: int = 4, maps=("linear+sat",)) -> dict:
    """Dump the system's ratings once per held-out season (timed), fit the calibration map leave-one-season-out
    on the dump, score with the criterion's scorer.  Loss = the MAPPED system's pooled team-game error; time = the
    dump's fit seconds summed over the held-out seasons."""
    from eracoef.calmap import SideMap, dump_ratings, evaluate, load_frames, unmapped_rows
    from eracoef.holdout import Context
    ho = Holdout.from_config(cfg, ks=[k])
    t0 = time.time()
    R = dump_ratings(ho, [system], OUT / f"ratings_track_{system}.parquet", workers=workers, verbose=False)
    wall = time.time() - t0
    secs = float(R.groupby("held_out").seconds.first().sum())
    ctx = Context.load(cfg)
    frames = load_frames(ctx, ho.seasons(), level=ho.level, verbose=False)
    res, params = [unmapped_rows(R, frames, system, k)], []
    for fam in maps:
        fo, fd = (fam.split(":") + [None])[:2]
        r, p = evaluate(R, frames, system, k, SideMap.parse(fo), SideMap.parse(fd or fo), f"{system}_{fam.replace(':', '_')}")
        res.append(r)
        params.append(p)
    res = pd.concat(res, ignore_index=True)
    res.to_parquet(OUT / f"holdout_track_{system}.parquet", index=False)
    pd.concat(params, ignore_index=True).to_parquet(OUT / f"calmap_track_{system}.parquet", index=False)
    P = pooled(res).set_index("system")
    mapped = P.loc[f"{system}_{maps[0].replace(':', '_')}"]
    return dict(game=float(mapped.game), game_unmapped=float(P.loc[system].game), stint=float(mapped.mse),
                scale_off=float(mapped.scale_off), scale_def=float(mapped.scale_def),
                seconds=secs, wall=float(wall), seasons=int(res.held_out.nunique()))


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
    m = measure(system, k, workers=int(_flag("workers", 4)), maps=tuple((_flag("maps") or "linear+sat").split(",")))
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
