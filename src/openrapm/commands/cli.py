"""The command line: `python -m openrapm <command>`, or `openrapm <command>` once the package is installed.

    openrapm rate      steps 1-3 for the seasons asked for: the ratings table before the prior shrink
    openrapm folds     the within-season folds: each season rated from three quarters of its games
    openrapm compare   two parquet tables, value by value

`--config FILE`, given before the command, reads another config.yaml; without it the file named by the
environment variable OPENRAPM_CONFIG is read, else the repository's own.  `openrapm <command> --help` lists each
command's options.

Every linear-algebra library (OpenMP, MKL, OpenBLAS) is held to one thread before numpy loads, so the same inputs
give the same bits on any machine: a multi-threaded library adds in another order and moves the last digits
(DECISIONS.md, the cleanup's thread rule, 2026-10-09).  numpy is imported only inside the commands, after that.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import replace
from pathlib import Path

from ..settings import PRODUCT, TEST, Settings

THREAD_VARIABLES = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
BOOSTER_FILE = "params/booster_params_{}.json"
SHIPPED_BOOSTER = Path(PRODUCT.prior.booster_file).stem.removeprefix("booster_params_")
SEASON_SPAN = re.compile(r"(\d+)(?:-(\d+))?")          # "2026" or "2017-2026"


def pin_threads() -> None:
    """One thread for every linear-algebra library, unless the environment already names a number."""
    for variable in THREAD_VARIABLES:
        os.environ.setdefault(variable, "1")


pin_threads()


def utf8_output() -> None:
    """Write UTF-8: player names carry accents (Schröder, Ginóbili), and a Windows console defaults to cp1252."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass


def parse_seasons(text: str) -> list[int]:
    """Seasons as typed, sorted and without repeats.

    Example: "2026" gives [2026], "2017-2019" gives [2017, 2018, 2019], "2026,2017-2018" gives [2017, 2018, 2026].
    """
    seasons: set[int] = set()
    for part in filter(None, (piece.strip() for piece in str(text).split(","))):
        span = SEASON_SPAN.fullmatch(part)
        if span is None:
            raise argparse.ArgumentTypeError(f"{text!r}: write seasons as 2026, 2017-2026 or 2017,2018")
        low, high = int(span[1]), int(span[2] or span[1])
        if low > high:
            raise argparse.ArgumentTypeError(f"{text!r}: {low} comes after {high}")
        seasons.update(range(low, high + 1))
    if not seasons:
        raise argparse.ArgumentTypeError("no season given")
    return sorted(seasons)


def chosen_settings(args: argparse.Namespace) -> Settings:
    """The setting named by --setting, with the defensive target, input list, booster file and fold assignment
    given on the command line (each defaults to the shipped value, so a plain run is the setting itself)."""
    base = {"product": PRODUCT, "test": TEST}[args.setting]
    if args.fold_assignment < 0:
        raise SystemExit("--fold-assignment is 0 (the shipped order) or a positive seed")
    return replace(base,
                   targets=replace(base.targets, defence=args.defensive_target),
                   prior=replace(base.prior, feature_set=args.features,
                                 booster_file=BOOSTER_FILE.format(args.booster_params),
                                 fold_assignment=args.fold_assignment))


def _run_rate(args: argparse.Namespace, cfg: dict) -> int:
    from .build import run_rate
    return run_rate(cfg, chosen_settings(args), args.seasons, args.name or args.setting, args.save_models,
                    argv=args.argv)


def _run_folds(args: argparse.Namespace, cfg: dict) -> int:
    from .within_season import run_folds
    return run_folds(cfg, args.models, args.base, args.seasons, args.tag, args.reproduce_only)


def _run_compare(args: argparse.Namespace, cfg: dict) -> int:
    from .compare import run_compare
    return run_compare(args.a, args.b, season=args.season, ignore=args.ignore, key=args.key)


def _add_rate(commands) -> None:
    rate = commands.add_parser(
        "rate", help="steps 1-3: the ratings table before the prior shrink",
        description="Steps 1-3 for every season asked for, in one run: the leave-season-out RAPM label, the "
                    "box-score prior, and the ridge that pulls each season's RAPM toward the prior.  Writes "
                    "outputs/season_ratings_<NAME>_step3.parquet once, at the end.")
    rate.add_argument("--setting", choices=("product", "test"), default="product",
                      help="product: the published ratings.  test: the labels and the prior's training rows also "
                           "leave out the seasons either side of the rated one, as the year-over-year test needs "
                           "(default %(default)s)")
    rate.add_argument("--seasons", type=parse_seasons, default=None,
                      help="the seasons to rate: 2026, 2017-2026 or 2017,2018 (default every season in config.yaml)")
    rate.add_argument("--name", help="the table is outputs/season_ratings_<NAME>_step3.parquet (default the "
                                     "setting's name)")
    rate.add_argument("--save-models", metavar="NAME",
                      help="also write every rated season's boosters to outputs/prior_models_<NAME>.pkl, which "
                           "`openrapm folds` reads")
    rate.add_argument("--defensive-target", default=PRODUCT.targets.defence,
                      help="what the defence ridge explains, per 100 possessions (default %(default)s)")
    rate.add_argument("--features", default=PRODUCT.prior.feature_set,
                      help="the prior's input list, a key of features.FEATURE_SETS (default %(default)s)")
    rate.add_argument("--booster-params", metavar="NAME", default=SHIPPED_BOOSTER,
                      help="LightGBM settings from params/booster_params_<NAME>.json (default %(default)s)")
    rate.add_argument("--fold-assignment", type=int, default=PRODUCT.prior.fold_assignment,
                      help="0: players sorted by label and assigned to folds 0,1,2,3,4,4,3,2,1,0,...  A positive "
                           "number assigns each block of five in an order drawn with that seed; 1 is the noise "
                           "control (default %(default)s)")
    rate.set_defaults(run=_run_rate)


def _add_folds(commands) -> None:
    folds = commands.add_parser(
        "folds", help="the within-season folds: each season rated from three quarters of its games",
        description="Each season's games are assigned to four folds at random, three times over; the season is "
                    "rated from three folds with the saved boosters and scored on the fourth.  Writes "
                    "outputs/within/<TAG>/.")
    folds.add_argument("--models", required=True, metavar="NAME",
                       help="outputs/prior_models_<NAME>.pkl, from `openrapm rate --setting test --save-models`")
    folds.add_argument("--base", required=True, metavar="NAME",
                       help="outputs/<NAME>.parquet, the table the same run wrote")
    folds.add_argument("--seasons", type=parse_seasons, default=parse_seasons("2017-2026"),
                       help="the seasons to fold: 2026, 2017-2026 or 2017,2018 (default 2017-2026)")
    folds.add_argument("--tag", default="within", help="the folder under outputs/within/ (default %(default)s)")
    folds.add_argument("--reproduce-only", action="store_true",
                       help="only rebuild each season from all of its games and check it equals the base table")
    folds.set_defaults(run=_run_folds)


def _add_compare(commands) -> None:
    compare = commands.add_parser(
        "compare", help="two parquet tables, value by value",
        description="Prints IDENTICAL-BYTES, IDENTICAL-VALUES or DIFFERENT, then each difference: columns, rows, "
                    "row order, data types and the largest absolute difference per numeric column.  Exits 1 "
                    "when the tables differ.")
    compare.add_argument("a", help="the first parquet file")
    compare.add_argument("b", help="the second parquet file")
    compare.add_argument("--season", type=int, help="compare one season of each table")
    compare.add_argument("--ignore", default="", help="columns to leave out, comma-separated")
    compare.add_argument("--key", default="player_id,season",
                         help="the columns rows are aligned on when their order differs (default %(default)s)")
    compare.set_defaults(run=_run_compare)


def build_parser() -> argparse.ArgumentParser:
    """The `openrapm` parser: the global --config, then one subcommand."""
    parser = argparse.ArgumentParser(prog="openrapm", description="OpenRAPM: one rating per NBA player per season.")
    parser.add_argument("--config", help="the config.yaml to read (default $OPENRAPM_CONFIG, else the repository's)")
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")
    _add_rate(commands)
    _add_folds(commands)
    _add_compare(commands)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse the command line, read the config and run the command; returns the exit status."""
    utf8_output()
    args = build_parser().parse_args(argv)
    args.argv = ["openrapm", *(sys.argv[1:] if argv is None else argv)]
    from ..data.config import load_config
    return int(args.run(args, load_config(args.config)) or 0)
