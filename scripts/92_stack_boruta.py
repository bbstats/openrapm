"""BorutaShap for the booster half of the stacked prior: every prior feature EXCEPT the plus-minus columns.

    python scripts/92_stack_boruta.py [--rows=stackpool] [--season=2026] [--trials=50] [--sides=O,D] [--explain_rows=1000]

The owner, 2026-10-02: "a stacking regressor of the +/- features into an elastic net, and all other features into
boruta -> chimeraboost quality=5".  The plus-minus columns are `singleyear.ONC` (his own season's on-court points
per 100 on each side, padded, and the possessions behind them); they go to the elastic net and are NOT candidates
here.  Every other name in `singleyear.PRIOR_FEATURES` is (50), so the booster's list is chosen without the
plus-minus columns in the room -- the shipped lists were chosen with them, which ranked them first by a factor of
six and may have pushed out the box-score columns that now have to carry the booster alone.

Reads outputs/prior_rows_<rows>_<season>_<side>.parquet, the rows `scripts/62_single_year_board.py
--features=sy --dump_rows=<rows>` trained on for that rated season (the incumbent's labels and weights).  The two
chunk features ride along because the booster always has them, and are kept whatever their verdict.  The booster
is the cheap, unbagged shape scripts/50_boruta.py and 87 use, 50 trials.  BORUTA PRUNES, IT DOES NOT DECIDE.

Writes outputs/csv/boruta_stack_<side>.csv (the importance history) and outputs/csv/boruta_stack_table.csv (every
candidate, each side: verdict, mean importance, whether it is on the shipped list), prints the table in full and
the kept lists (accepted + tentative) for `singleyear.STACK_BOOSTER_O` / `STACK_BOOSTER_D`.
"""
import os
import sys
import time
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
os.environ.setdefault("NUMBA_NUM_THREADS", os.environ.get("OPENRAPM_NUMBA_THREADS", "4"))

import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _cli import check_flags, flag  # noqa: E402
from eracoef import singleyear as sy  # noqa: E402
from eracoef.gbdt_prior import run_boruta  # noqa: E402

CHEAP = {"linear_leaves": True, "cross_features": False}      # scripts/50_boruta.py's "cheap"
POOL = [f for f in sy.PRIOR_FEATURES if f not in sy.ONC]
SHIPPED = {"O": sy.BORUTA_O, "D": sy.BORUTA_D}


def main() -> None:
    check_flags()
    tag, season = flag("rows", "stackpool"), int(flag("season", 2026))
    trials, sides = int(flag("trials", 50)), flag("sides", "O,D").split(",")
    explain_rows = int(flag("explain_rows", 1000)) or None
    out_csv = ROOT / "outputs" / "csv"
    out_csv.mkdir(parents=True, exist_ok=True)
    table, kept, started = [], {}, time.time()
    for side in sides:
        rows = pd.read_parquet(ROOT / "outputs" / f"prior_rows_{tag}_{season}_{side}.parquet")
        candidates = list(dict.fromkeys(POOL + sy.CHUNK_FEATURES))
        missing = [c for c in candidates if c not in rows.columns]
        if missing:
            raise SystemExit(f"{side}: the dumped rows lack {missing}; re-dump with --features=sy")
        print(f"=== side {side}: {len(rows):,} training rows, {len(candidates)} candidates, {trials} trials",
              flush=True)
        res = run_boruta(rows, candidates, n_trials=trials, seed=0, thread_count=4, verbose=False,
                         explain_rows=explain_rows, **CHEAP)
        history = res["history"]
        history.to_csv(out_csv / f"boruta_stack_{side}.csv", index=False)
        importance = history.iloc[1:].mean()        # BorutaShap's first history row is its zero initial value
        verdict = {**{f: "accepted" for f in res["accepted"]}, **{f: "tentative" for f in res["tentative"]},
                   **{f: "rejected" for f in res["rejected"]}}
        for f in candidates:
            table.append(dict(side=side, feature=f,
                              verdict=verdict.get(f, "kept by design" if f in sy.CHUNK_FEATURES else "?"),
                              importance=float(importance.get(f, float("nan"))),
                              on_shipped_list=f in SHIPPED[side]))
        shadow = float(importance.get("Max_Shadow", float("nan")))
        kept[side] = [f for f in candidates if verdict.get(f) in ("accepted", "tentative")
                      and f not in sy.CHUNK_FEATURES]
        print(f"  best shadow's mean importance {shadow:.3f}; accepted {len(res['accepted'])}, tentative "
              f"{len(res['tentative'])}, rejected {len(res['rejected'])} ({time.time() - started:.0f}s)", flush=True)
    frame = pd.DataFrame(table).sort_values(["side", "importance"], ascending=[True, False])
    frame.to_csv(out_csv / "boruta_stack_table.csv", index=False)
    with pd.option_context("display.max_rows", None, "display.width", 200):
        print("\n=== every candidate, each side (importance: mean over trials, z-scored; the best shadow is the bar)")
        print(frame.round(3).to_string(index=False))
    print("\n=== kept (accepted + tentative) for singleyear.STACK_BOOSTER_O / STACK_BOOSTER_D:")
    for side in sides:
        print(f"  {side} ({len(kept[side])}): {sorted(kept[side])}")
    print(f"wrote outputs/csv/boruta_stack_table.csv ({time.time() - started:.0f}s)")


if __name__ == "__main__":
    main()
