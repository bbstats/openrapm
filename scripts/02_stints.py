"""Build stints per season and phase from cached play-by-play; print per-season diagnostics.
usage: python scripts/02_stints.py [first_season] [last_season] [phases=RS,PO] [--force]
"""
import argparse
import time

import pandas as pd

from openrapm.commands.console import use_utf8
from openrapm.data.config import load_config, resolve
from openrapm.data.stints import build_season, season_diagnostics


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build stints from cached play-by-play.", allow_abbrev=False)
    parser.add_argument("first_season", nargs="?", type=int, help="default config.yaml's first_season")
    parser.add_argument("last_season", nargs="?", type=int, help="default the first season")
    parser.add_argument("phases", nargs="?", default="RS,PO", help="default %(default)s")
    parser.add_argument("--force", action="store_true", help="rebuild stints that are already on disk")
    return parser.parse_args(argv)


def main() -> None:
    use_utf8()
    args = parse_args()
    pd.set_option("display.width", 250, "display.max_columns", 40, "display.precision", 3)
    cfg = load_config()
    first = args.first_season if args.first_season is not None else cfg["first_season"]
    last = args.last_season if args.last_season is not None else first
    rows = []
    for season in range(first, last + 1):
        for phase in tuple(args.phases.split(",")):
            t0 = time.time()
            build_season(season, phase, cfg, force=args.force)
            d = season_diagnostics(season, phase, cfg)
            d["wall_seconds"] = time.time() - t0
            rows.append(d)
            print(pd.Series(d).to_string())
            print()
    out = pd.DataFrame(rows)
    out_path = resolve(cfg, "stints") / "diagnostics_summary.parquet"
    if out_path.exists():
        old = pd.read_parquet(out_path)
        old = old[~(old.season.astype(str) + old.phase).isin(out.season.astype(str) + out.phase)]
        out = pd.concat([old, out], ignore_index=True)
    out.to_parquet(out_path, index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
