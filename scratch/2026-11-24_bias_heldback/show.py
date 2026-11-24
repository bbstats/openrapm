"""Print the held-back read's tables and apply the registered rule (notes/DECISIONS.md, "The Bias page's leans on
held-back games").

    .venv/Scripts/python scratch/2026-11-24_bias_heldback/show.py
"""
from pathlib import Path

import pandas as pd

OUT = Path("outputs/bias_heldback")
NAMES = {
    "threes_made_per100": "made threes", "threes_missed_per100": "missed threes", "twos_made_per100": "made twos",
    "twos_missed_per100": "missed twos", "free_throws_made_per100": "made free throws",
    "free_throws_missed_per100": "missed free throws", "off_rebounds_per100": "offensive rebounds",
    "def_rebounds_per100": "defensive rebounds", "assists_per100": "assists", "turnovers_per100": "turnovers",
    "steals_per100": "steals", "blocks_per100": "blocks", "fouls_per100": "personal fouls",
}
ERAS = ("1997-2006", "2007-2016", "2017-2026")


def passes(table: pd.DataFrame) -> pd.Series:
    """The registered rule, per (end, stat): conditions 1, 2 and 4 in both deciding runs, the same sign in both."""
    deciding = table[table.run.isin(["width free", "own width"])]
    ok = deciding["same sign in every era"] & deciding["beyond 2"] & deciding["null centred"]
    grouped = deciding.assign(ok=ok, sign=deciding["all mean"].gt(0)).groupby(["end", "stat"])
    return grouped.ok.all() & (grouped.sign.nunique() == 1)


def main() -> None:
    table = pd.read_csv(OUT / "summary.csv")
    verdict = passes(table)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    for end in ("total", "offense", "defense"):
        rows = []
        for stat, name in NAMES.items():
            row = {"stat": name}
            for run, short in (("width free", "free"), ("own width", "own"), ("width free, no playing time", "no time")):
                r = table[(table.run == run) & (table.end == end) & (table.stat == stat)].iloc[0]
                row[f"{short}"] = round(r["all mean"], 3)
                row[f"{short} z"] = round(r["z"], 1)
                row[f"{short} z null"] = round(r["z against null"], 1)
                if run == "width free":
                    for era in ERAS:
                        row[f"free {era}"] = round(r[f"{era} mean"], 3)
                    row["free null spread"] = round(r["all null spread"], 3)
                    row["free null mean"] = round(r["all null mean"], 3)
                if run == "own width":
                    for era in ERAS:
                        row[f"own {era}"] = round(r[f"{era} mean"], 3)
            row["passes"] = bool(verdict[(end, stat)])
            rows.append(row)
        print(f"\n=== {end} ===")
        print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
