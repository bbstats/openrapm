"""Experiment 35: the lineup-swap test on each within-season fold's HELD-OUT games -- the check that sees how credit
is split among teammates, which team-game error and slope cannot.

    python scripts/105_held_out_swaps.py [--tags=q1of4,q1of3,within2,q2of3,within,within10]

Per fold: a swap-test season (src/eracoef/swaptest.py) built from the held-out games' stints only; every pair of
lineups there that shares four players; each system's rating gap between the two swapped players (rated from the
rating games) against the pair's result, the opponents and context taken out by a plain RAPM (penalty 3,000) fit on
the held-out games -- the same adjustment for every system (`swaptest.score_season`).  A pair counts only if both
swapped players are rated by every system.  Systems and controls are the scorecard's (scripts/104_scorecard.py).

    order  higher is better, 0 = no idea, spread-free;  slope  below 1 = the within-team gaps are too wide

Writes outputs/scorecard/held_out_swaps.parquet and prints, per fold size, each system's order and slope against O.
"""
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _cli import check_flags, flag  # noqa: E402
from eracoef import swaptest as sw  # noqa: E402
from eracoef.config import load_config  # noqa: E402


def _borrow(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


S104 = _borrow("_scorecard104_for_105", "104_scorecard.py")
T90 = _borrow("_swap90_for_105", "90_swap_test.py")
PLAIN_LAMBDA = 3000.0


def summarise(res: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (tag, side), g in res.groupby(["tag", "side"]):
        if "O" not in set(g.ranking):
            continue
        piv = g.pivot_table(index=["season", "key"], columns="ranking", values="order")
        slope = g.groupby("ranking").apply(lambda x: np.average(x.slope.dropna(), weights=x.information[x.slope.notna()]))
        for ranking in piv.columns:
            d = (piv[ranking] - piv["O"]).groupby(level="season").mean()
            se = d.std(ddof=1) / np.sqrt(len(d))
            rows.append(dict(tag=tag, side=side, ranking=ranking, order=float(piv[ranking].mean()),
                             vs_O=float(d.mean()), z=float(d.mean() / se) if se > 0 else np.nan,
                             seasons_better=int((d > 0).sum()), slope=float(slope.get(ranking, np.nan))))
    return pd.DataFrame(rows)


def main():
    check_flags()
    tags = [t for t in flag("tags", ",".join(S104.SIZES)).split(",") if t]
    cfg = load_config(ROOT / "config.yaml")
    margin_clip = float(cfg.get("margin_clip", 25))
    rng = np.random.default_rng(3503)
    out = ROOT / "outputs" / "scorecard"
    out.mkdir(exist_ok=True)
    rows_of, results, t0 = {}, [], time.time()
    for tag in tags:
        directory = ROOT / "outputs" / "within" / tag
        tables = {}
        for name, file in (("B1", "baseline_b1"), ("B2", "baseline_b2"), ("B2g", "baseline_b2g"), ("B3", "baseline_b3"),
                           ("B4", "baseline_b4")):
            if (directory / f"{file}.parquet").exists():
                tables[name] = pd.read_parquet(directory / f"{file}.parquet")
        if (directory / "openrapm_shipped.parquet").exists():
            for variant, g in pd.read_parquet(directory / "openrapm_shipped.parquet").groupby("variant"):
                tables["O" if variant == "shrunk" else f"O+{variant}"] = g
        for stem in S104.fold_stems(directory):
            H = int(stem[:4])
            if H not in rows_of:
                rows_of = {H: sw.season_rows(T90.load_stints(H, cfg), margin_clip=margin_clip)}
            test_ids = set(map(str, np.load(directory / f"test_{stem}.npz", allow_pickle=True)["game_id"]))
            rows = rows_of[H][rows_of[H].game_id.astype(str).isin(test_ids)].reset_index(drop=True)
            season = sw.prepare(rows, H)
            f = S104.load_fold(directory, stem)
            systems, _, _ = S104.within_systems(stem, f, tables, rng)
            rated = set.intersection(*[set(r.player_id.astype(np.int64)) for r in systems.values()])
            common = sw.plain_rapm(season, PLAIN_LAMBDA)
            scores, _ = sw.score_season(season, systems, rated, common, contexts=("nofatigue",))
            results.append(scores.assign(tag=tag, key=stem, season=H))
        print(f"  {tag}: {len(S104.fold_stems(directory))} folds ({time.time() - t0:.0f}s)", flush=True)
    res = pd.concat(results, ignore_index=True)
    res = res[res.group == "all"]
    res.to_parquet(out / "held_out_swaps.parquet", index=False)
    summary = summarise(res)
    summary.to_csv(out / "held_out_swaps_summary.csv", index=False)
    pd.set_option("display.width", 220)
    for tag, g in summary.groupby("tag", sort=False):
        print(f"\n=== {tag}: held-out lineup swaps, order (higher = better) and within-team slope, vs O")
        print(g.pivot_table(index="ranking", columns="side", values=["order", "z", "slope"])
              .to_string(float_format=lambda v: f"{v:+.3f}"))


if __name__ == "__main__":
    main()
