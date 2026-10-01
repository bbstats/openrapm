"""BorutaShap on the single-year prior's own training rows, with team and game context among the candidates.

    python scripts/87_context_boruta.py [--rows=context] [--season=2026] [--trials=50] [--sides=O,D] [--explain_rows=1000]

Experiment 26 (the owner, 2026-09-28).  Reads outputs/prior_rows_<rows>_<season>_<side>.parquet, the rows
`scripts/62_single_year_board.py --dump_rows=<rows>` actually trained on for that rated season (outside labels,
the incumbent's settings), so the selection is made against the booster's own target, weights and row shapes.

Candidates per side:
  - that side's shipped Boruta list (`singleyear.BORUTA_O`, 21 names, or `BORUTA_D`, 17);
  - every RAPM piece, both sides (`singleyear.PIECES`, 20 names), as `onc_o` and `onc_d` are on both lists;
  - `same_team`, the soft same-team measure;
  - the three closeness columns and `po_share`, the game difficulty not already in the pieces.
The two chunk features ride along because the booster always has them -- without them `same_team` could win
merely by saying which row is the career row -- and are kept whatever their verdict.

The booster is the cheap, unbagged shape scripts/50_boruta.py uses (Boruta fits two models a trial), 50 trials.
BORUTA PRUNES, IT DOES NOT DECIDE: an acceptance says "not noise against this target", and the year-over-year
test decides.

Writes outputs/csv/boruta_context_<side>.csv (the importance history per trial) and
outputs/csv/boruta_context_table.csv (every candidate, each side: verdict, mean importance, whether it is on the
shipped list), prints the table in full, and prints the kept lists (accepted + tentative) for
`singleyear.FEATURE_SETS["boruta_context"]`.
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
CONTEXT = sy.PIECES + [sy.SAME_TEAM] + sy.CLOSENESS + sy.PO_SHARE
SHIPPED = {"O": sy.BORUTA_O, "D": sy.BORUTA_D}


def main() -> None:
    check_flags()
    tag, season = flag("rows", "context"), int(flag("season", 2026))
    trials, sides = int(flag("trials", 50)), flag("sides", "O,D").split(",")
    # each trial's mean |SHAP| from a fresh random 1,000 rows, not all 33,000: chimeraboost 0.34's exact
    # interventional SHAP made a whole-row trial take 30 minutes (`gbdt_prior.make_boruta`)
    explain_rows = int(flag("explain_rows", 1000)) or None
    out_csv = ROOT / "outputs" / "csv"
    out_csv.mkdir(parents=True, exist_ok=True)
    table, kept, started = [], {}, time.time()
    for side in sides:
        rows = pd.read_parquet(ROOT / "outputs" / f"prior_rows_{tag}_{season}_{side}.parquet")
        candidates = list(dict.fromkeys(SHIPPED[side] + CONTEXT + sy.CHUNK_FEATURES))
        missing = [c for c in candidates if c not in rows.columns]
        if missing:
            raise SystemExit(f"{side}: the dumped rows lack {missing}; rebuild the panel (scripts/86) and re-dump")
        print(f"=== side {side}: {len(rows):,} training rows, {len(candidates)} candidates, {trials} trials",
              flush=True)
        res = run_boruta(rows, candidates, n_trials=trials, seed=0, thread_count=4, verbose=False,
                         explain_rows=explain_rows, **CHEAP)
        history = res["history"]
        history.to_csv(out_csv / f"boruta_context_{side}.csv", index=False)
        importance = history.iloc[1:].mean()        # BorutaShap's first history row is its zero initial value
        verdict = {**{f: "accepted" for f in res["accepted"]}, **{f: "tentative" for f in res["tentative"]},
                   **{f: "rejected" for f in res["rejected"]}}
        for f in candidates:
            table.append(dict(side=side, feature=f, verdict=verdict.get(f, "kept by design" if f in sy.CHUNK_FEATURES
                                                                        else "?"),
                              importance=float(importance.get(f, float("nan"))),
                              on_shipped_list=f in SHIPPED[side],
                              family=("piece" if f in sy.PIECES else "same team" if f == sy.SAME_TEAM
                                      else "closeness" if f in sy.CLOSENESS else "playoffs" if f in sy.PO_SHARE
                                      else "chunk" if f in sy.CHUNK_FEATURES else "shipped list")))
        shadow = float(importance.get("Max_Shadow", float("nan")))
        kept[side] = [f for f in candidates if verdict.get(f) in ("accepted", "tentative")
                      and f not in sy.CHUNK_FEATURES]
        print(f"  best shadow's mean importance {shadow:.3f}; accepted {len(res['accepted'])}, tentative "
              f"{len(res['tentative'])}, rejected {len(res['rejected'])} ({time.time() - started:.0f}s)", flush=True)
    frame = pd.DataFrame(table).sort_values(["side", "importance"], ascending=[True, False])
    frame.to_csv(out_csv / "boruta_context_table.csv", index=False)
    with pd.option_context("display.max_rows", None, "display.width", 200):
        print("\n=== every candidate, each side (importance: mean over trials, z-scored; the best shadow is the bar)")
        print(frame.round(3).to_string(index=False))
    rejected_both = all(frame[(frame.side == s) & (frame.feature == sy.SAME_TEAM)].verdict.iloc[0] == "rejected"
                        for s in sides)
    if rejected_both:
        print("\nsame_team is REJECTED on every side: rating 'as if he changed teams' has nothing to act on.")
    print("\n=== kept (accepted + tentative) for singleyear.FEATURE_SETS['boruta_context']:")
    for side in sides:
        print(f"  {side} ({len(kept[side])}): {kept[side]}")
    print(f"wrote outputs/csv/boruta_context_table.csv ({time.time() - started:.0f}s)")


if __name__ == "__main__":
    main()
