"""The consensus numbers for several boards side by side, reported and never selected on.

It lived in the gitignored `scratch/` until 2026-09-17, which meant the workflow in HANDOFF.md
named a file no clone had.

`tests/test_vs_consensus.py` only prints a figure when a floor FAILS, which makes it useless for
comparing candidates -- a board that passes tells you nothing about by how much.  This rebuilds that
module's own join (its `_norm`, `_pooled`, its consensus file, its 1,000-possession cut) and reports the
same quantities for every board named, so the passing ones carry numbers too.

    python scripts/64_consensus_report.py outputs/a.parquet outputs/b.parquet ...

The owner's standing rule (DECISIONS.md): the consensus is a sanity check, never a fitting target.  A
marginal miss is not a veto; a gross one still is.  Nothing here chooses anything.
"""
import importlib.util
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from _cli import check_flags   # noqa: E402

check_flags()      # this script takes board paths and no flags at all

spec = importlib.util.spec_from_file_location("tvc", ROOT / "tests" / "test_vs_consensus.py")
tvc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tvc)

con = pd.read_csv(tvc.CONSENSUS)[["player_name", "team", "adj_offense", "adj_defense", "adj_overall"]]
def team_r2(frame: pd.DataFrame, column: str) -> float:
    """Share of `column`'s variance explained by which team the player is on (one-way ANOVA R-squared)."""
    y = frame[column].to_numpy(float)
    within = frame.groupby("team")[column].transform("mean").to_numpy(float)
    return float(1.0 - ((y - within) ** 2).sum() / ((y - y.mean()) ** 2).sum())


con = con.dropna(subset=["player_name", "adj_overall"])
con["key"] = con.player_name.map(tvc._norm)
con = con.drop_duplicates("key")

rows = []
for path in [Path(a) for a in sys.argv[1:]] or [ROOT / "outputs" / "season_ratings_sy.parquet"]:
    ours = tvc._pooled(pd.read_parquet(path))
    ours["key"] = ours.player_name.map(tvc._norm)
    ours = ours.sort_values("poss_off").drop_duplicates("key", keep="last")
    b = ours.merge(con, on="key", how="inner")
    b = b[b.poss_off >= tvc.MIN_POSS].copy()
    for side, col in (("total", "adj_overall"), ("off", "adj_offense"), ("def", "adj_defense")):
        b[f"rk_ours_{side}"] = b[f"rating_{side}"].rank(ascending=False)
        b[f"rk_con_{side}"] = b[col].rank(ascending=False)
    rows.append(dict(
        rankings=path.stem, n=len(b),
        rho_off=spearmanr(b.rating_off, b.adj_offense).statistic,
        rho_def=spearmanr(b.rating_def, b.adj_defense).statistic,
        rho_total=spearmanr(b.rating_total, b.adj_overall).statistic,
        spread_off=b.rating_off.std() / b.adj_offense.std(),
        spread_def=b.rating_def.std() / b.adj_defense.std(),
        top5=len(set(b.nsmallest(5, "rk_ours_total").key) & set(b.nsmallest(5, "rk_con_total").key)),
        def_share=b.rating_def.var() / (b.rating_off.var() + b.rating_def.var()),
        team_r2_def=team_r2(b, "rating_def"), team_r2_off=team_r2(b, "rating_off")))

if rows:
    rows.append(dict(rankings="consensus itself", n=len(b), rho_off=1, rho_def=1, rho_total=1, spread_off=1,
                     spread_def=1, top5=5,
                     def_share=b.adj_defense.var() / (b.adj_offense.var() + b.adj_defense.var()),
                     team_r2_def=team_r2(b, "adj_defense"), team_r2_off=team_r2(b, "adj_offense")))

R = pd.DataFrame(rows)
pd.set_option("display.width", 200, "display.max_columns", 20, "display.precision", 3)
print("\n=== against data/external/consensus.csv, 2024-26, 1,000+ possessions")
print("    tests: rho_* >= 0.75, spread_off in [0.55, 1.30], spread_def >= 0.70, top5 >= 3")
print("    def_share: defence's share of off+def rating variance.  team_r2_*: how much of a player's rating")
print("    you can guess from his team alone (one-way ANOVA R-squared on the consensus's team column).")
print(R.to_string(index=False))
