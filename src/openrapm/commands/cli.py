"""The command line: `python -m openrapm <command>`, or `openrapm <command>` once the package is installed.

    openrapm build     all five steps in one run: the ratings tables after steps 3, 4 and 5
    openrapm rate      steps 1-3 for the seasons asked for: the ratings table before the prior shrink
    openrapm shrink    step 4 on a step-3 table: the prior part multiplied down
    openrapm swap      step 5 on a step-4 table: each team's credit re-split by its lineup swaps
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
SETTINGS = {"product": PRODUCT, "test": TEST}
TABLE_PREFIX = "season_ratings_"


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
    base = SETTINGS[args.setting]
    if args.fold_assignment < 0:
        raise SystemExit("--fold-assignment is 0 (the shipped order) or a positive seed")
    return replace(base,
                   targets=replace(base.targets, defence=args.defensive_target),
                   prior=replace(base.prior, feature_set=args.features,
                                 booster_file=BOOSTER_FILE.format(args.booster_params),
                                 fold_assignment=args.fold_assignment))


def table_stem(text: str) -> str:
    """A table named on the command line, without a trailing .parquet: "x.parquet" and "x" both give "x"."""
    return text.removesuffix(".parquet")


def output_name(base: str, step: str) -> str:
    """The name a step gives its output when --name is not given, from the table it reads.

    Example: "season_ratings_trial_step3" with step "_step3" gives "trial", so `shrink` writes
    season_ratings_trial_step4.parquet and `swap` on that writes season_ratings_trial.parquet.
    """
    return table_stem(base).removeprefix(TABLE_PREFIX).removesuffix(step)


def _run_build(args: argparse.Namespace, cfg: dict) -> int:
    from .build import run_build
    settings = chosen_settings(args)
    settings = replace(settings, shrink=replace(settings.shrink, folds_tag=args.folds))
    return run_build(cfg, settings, args.seasons, args.name or args.setting, args.save_models, argv=args.argv,
                     with_types=args.types)


def _run_rate(args: argparse.Namespace, cfg: dict) -> int:
    from .build import run_rate
    return run_rate(cfg, chosen_settings(args), args.seasons, args.name or args.setting, args.save_models,
                    argv=args.argv)


def _run_shrink(args: argparse.Namespace, cfg: dict) -> int:
    from .build import run_shrink
    base = SETTINGS[args.setting]
    settings = replace(base, shrink=replace(base.shrink, folds_tag=args.folds))
    return run_shrink(cfg, settings, table_stem(args.base), args.name or output_name(args.base, "_step3"))


def _run_swap(args: argparse.Namespace, cfg: dict) -> int:
    from .build import run_swap
    return run_swap(cfg, SETTINGS[args.setting], table_stem(args.base), args.name or output_name(args.base, "_step4"),
                    with_types=args.types)


def _run_folds(args: argparse.Namespace, cfg: dict) -> int:
    from .within_season import run_folds
    return run_folds(cfg, args.models, args.base, args.seasons, args.tag, args.reproduce_only)


def _run_compare(args: argparse.Namespace, cfg: dict) -> int:
    from .compare import run_compare
    return run_compare(args.a, args.b, season=args.season, ignore=args.ignore, key=args.key)


def _add_setting_options(parser) -> None:
    """The options `rate` and `build` share: which setting, which seasons, the name, and the experiment hooks."""
    parser.add_argument("--setting", choices=("product", "test"), default="product",
                        help="product: the published ratings.  test: what the year-over-year test scores, so "
                             "nothing fitted for a season comes from the seasons either side of it (the labels and "
                             "the prior's training rows; in build also step 4's multipliers and step 5's type "
                             "model) (default %(default)s)")
    parser.add_argument("--seasons", type=parse_seasons, default=None,
                        help="the seasons to rate: 2026, 2017-2026 or 2017,2018 (default every season in "
                             "config.yaml)")
    parser.add_argument("--name", help="the tables are outputs/season_ratings_<NAME>_step3.parquet (rate), and also "
                                       "_step4.parquet and season_ratings_<NAME>.parquet (build); default the "
                                       "setting's name")
    parser.add_argument("--save-models", metavar="NAME",
                        help="also write every rated season's boosters to outputs/prior_models_<NAME>.pkl, which "
                             "`openrapm folds` reads")
    parser.add_argument("--defensive-target", default=PRODUCT.targets.defence,
                        help="what the defence ridge explains, per 100 possessions (default %(default)s)")
    parser.add_argument("--features", default=PRODUCT.prior.feature_set,
                        help="the prior's input list, a key of features.FEATURE_SETS (default %(default)s)")
    parser.add_argument("--booster-params", metavar="NAME", default=SHIPPED_BOOSTER,
                        help="LightGBM settings from params/booster_params_<NAME>.json (default %(default)s)")
    parser.add_argument("--fold-assignment", type=int, default=PRODUCT.prior.fold_assignment,
                        help="0: players sorted by label and assigned to folds 0,1,2,3,4,4,3,2,1,0,...  A positive "
                             "number assigns each block of five in an order drawn with that seed; 1 is the noise "
                             "control (default %(default)s)")


def _add_folds_tag(parser) -> None:
    parser.add_argument("--folds", metavar="TAG", default=PRODUCT.shrink.folds_tag,
                        help="the within-season folds the multipliers are fitted on, outputs/within/<TAG>/ "
                             "(default %(default)s)")


def _add_types(parser) -> None:
    parser.add_argument("--types", action="store_true",
                        help="also write outputs/<NAME>_types.csv: the swap type model fitted on every season")


def _add_build(commands) -> None:
    build = commands.add_parser(
        "build", help="all five steps in one run: the ratings tables after steps 3, 4 and 5",
        description="Steps 1-5 for every season asked for, in one run: rate (steps 1-3), the prior shrink (step "
                    "4) and the swap adjustment (step 5).  Writes outputs/season_ratings_<NAME>_step3.parquet, "
                    "_step4.parquet and outputs/season_ratings_<NAME>.parquet once, at the end, with step 4's "
                    "multipliers in outputs/csv/season_ratings_<NAME>_step4_multipliers.csv.  The product "
                    "setting's default name is product, so a plain `openrapm build` writes the published "
                    "table, outputs/season_ratings_product.parquet.  Step 5 trains on the other seasons of the "
                    "same table: a table of one season gets no swap adjustment.")
    _add_setting_options(build)
    _add_folds_tag(build)
    _add_types(build)
    build.set_defaults(run=_run_build)


def _add_rate(commands) -> None:
    rate = commands.add_parser(
        "rate", help="steps 1-3: the ratings table before the prior shrink",
        description="Steps 1-3 for every season asked for, in one run: the leave-season-out RAPM label, the "
                    "box-score prior, and the ridge that pulls each season's RAPM toward the prior.  Writes "
                    "outputs/season_ratings_<NAME>_step3.parquet once, at the end.")
    _add_setting_options(rate)
    rate.set_defaults(run=_run_rate)


def _add_shrink(commands) -> None:
    shrink = commands.add_parser(
        "shrink", help="step 4: the prior part of every rating multiplied down",
        description="Step 4 on a step-3 table: each season's box-score prior part times two multipliers (offence, "
                    "defence) fitted on the within-season folds' held-out games, the ratings re-centred, the "
                    "games' part unchanged.  Writes outputs/season_ratings_<NAME>_step4.parquet, and the "
                    "multipliers of every season to outputs/csv/season_ratings_<NAME>_step4_multipliers.csv.")
    shrink.add_argument("--base", required=True, metavar="NAME",
                        help="outputs/<NAME>.parquet, a step-3 table (season_ratings_<name>_step3)")
    shrink.add_argument("--setting", choices=("product", "test"), default="product",
                        help="product: a season's multipliers come from every fold season but itself.  test: also "
                             "not from the seasons either side of it (default %(default)s)")
    _add_folds_tag(shrink)
    shrink.add_argument("--name", help="the table is outputs/season_ratings_<NAME>_step4.parquet (default the "
                                       "base's name: season_ratings_X_step3 gives X)")
    shrink.set_defaults(run=_run_shrink)


def _add_swap(commands) -> None:
    swap_parser = commands.add_parser(
        "swap", help="step 5: each team's credit re-split by its lineup swaps",
        description="Step 5 on a step-4 table: half of what a type model of other seasons' lineup swaps predicts "
                    "each player is under- or over-credited, given back inside his team so no team total moves, "
                    "and the spread inside teams held.  Writes outputs/season_ratings_<NAME>.parquet.")
    swap_parser.add_argument("--base", required=True, metavar="NAME",
                             help="outputs/<NAME>.parquet, a step-4 table (season_ratings_<name>_step4)")
    swap_parser.add_argument("--setting", choices=("product", "test"), default="product",
                             help="product: the type model trains on every other season.  test: only on seasons "
                                  "more than one away (default %(default)s)")
    swap_parser.add_argument("--name", help="the table is outputs/season_ratings_<NAME>.parquet (default the "
                                            "base's name: season_ratings_X_step4 gives X)")
    _add_types(swap_parser)
    swap_parser.set_defaults(run=_run_swap)


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
    _add_build(commands)
    _add_rate(commands)
    _add_shrink(commands)
    _add_swap(commands)
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
