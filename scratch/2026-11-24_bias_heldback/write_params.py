"""params/bias_held_back.json, the Bias page's "vs game results, last 10 seasons" column (the owner, 2026-11-24:
"Held-back, drop ’27"), from heldback_lean.py's summary: each of the page's 15 stats, total, in the page's sign
(OpenRAPM minus the games), with each fold's width free as the page's in-sample columns had it, over the last ten
finished seasons; its standard error the larger of the across-season one and the career null's spread over the same
seasons.  Rerun after a new build is adopted, once extract.py and heldback_lean.py have read its within-season test.

    .venv/Scripts/python scratch/2026-11-24_bias_heldback/write_params.py --summary=outputs/bias_heldback15/summary.csv
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from openrapm.bias import STATS


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--summary", default="outputs/bias_heldback15/summary.csv")
    parser.add_argument("--build", default="rd143")
    parser.add_argument("--first", type=int, default=2017)
    parser.add_argument("--last", type=int, default=2026)
    parser.add_argument("--out", default="params/bias_held_back.json")
    args = parser.parse_args(argv)
    era = f"{args.first}-{args.last}"
    table = pd.read_csv(args.summary)
    table = table[(table.run == "width free") & (table.end == "total")].set_index("stat")
    missing = [stat for stat in STATS if stat not in table.index]
    if missing:
        raise SystemExit(f"{args.summary} has no {', '.join(missing)}")
    rows = {}
    for stat in STATS:
        row = table.loc[stat]
        error = max(float(row[f"{era} standard error"]), float(row[f"{era} null spread"]))
        rows[stat] = [round(float(row[f"{era} mean"]), 4), round(error, 4)]
    document = {"first": args.first, "last": args.last, "build": args.build,
                "source": "scratch/2026-11-24_bias_heldback/heldback_lean.py, notes/DECISIONS.md, \"The Bias page's "
                          "leans on held-back games\"",
                "rows": rows}
    Path(args.out).write_text(json.dumps(document, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    for stat, (lean, error) in rows.items():
        print(f"  {stat:<32}{lean:+.3f}  {error:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
