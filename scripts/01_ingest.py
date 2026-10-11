"""Pull game logs, play-by-play (V3) and box scores for seasons; cached, idempotent.
usage: python scripts/01_ingest.py [first_season] [last_season] [phases=RS,PO] [workers=2] [delay=0.6]
"""
import argparse

from openrapm.commands.console import use_utf8
from openrapm.data.config import load_config
from openrapm.data.ingest import ingest_season


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pull game logs, play-by-play and box scores; cached.",
                                     allow_abbrev=False)
    parser.add_argument("first_season", nargs="?", type=int, help="default config.yaml's first_season")
    parser.add_argument("last_season", nargs="?", type=int, help="default the first season")
    parser.add_argument("phases", nargs="?", default="RS,PO", help="default %(default)s")
    parser.add_argument("workers", nargs="?", type=int, default=2, help="default %(default)s")
    parser.add_argument("delay", nargs="?", type=float, default=0.6,
                        help="seconds between requests (default %(default)s)")
    return parser.parse_args(argv)


def main() -> None:
    use_utf8()
    args = parse_args()
    cfg = load_config()
    first = args.first_season if args.first_season is not None else cfg["first_season"]
    last = args.last_season if args.last_season is not None else first
    for season in range(first, last + 1):
        ingest_season(season, tuple(args.phases.split(",")), cfg, workers=args.workers, delay=args.delay)


if __name__ == "__main__":
    main()
