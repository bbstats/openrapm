"""Print the scan's single-statistic synergy table: all seasons and each era."""
import sys
import pandas as pd
data = sys.argv[1] if len(sys.argv) > 1 else "outputs/synergy/rd143"
s = pd.read_csv(f"{data}/scan_summary.csv")
pd.set_option("display.width", 250)
for analysis in ("single", "pair", "joint"):
    for term in ("synergy per lineup sd", "lean per player"):
        part = s[(s.analysis == analysis) & (s.term == term)]
        if part.empty:
            continue
        allp = part[part.era == "all"].set_index(["end", "trait"])[["mean", "z", "z_newey_west", "seasons_with_its_sign"]]
        eras = part[part.era != "all"].pivot_table(index=["end", "trait"], columns="era", values="mean")
        table = allp.join(eras).sort_values(["end", "z"])
        print(f"\n== {analysis}: {term}")
        print(table.round(3).to_string())
