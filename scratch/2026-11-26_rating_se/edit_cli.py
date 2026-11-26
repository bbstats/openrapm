"""One-off: `openrapm errors` on the command line (run once)."""
from edit_folds import edit

CLI = [
    ('''    openrapm folds     the within-season folds: each season rated from three quarters of its games
''',
     '''    openrapm folds     the within-season folds: each season rated from three quarters of its games
    openrapm errors    every rating's standard error, from the folds that hold back a tenth of a season's games
'''),
    ('''def _add_second_prior(commands) -> None:
''',
     '''def _run_errors(args: argparse.Namespace, cfg: dict) -> int:
    from .errors import run_errors
    return run_errors(cfg, args.name, args.seasons, [tag for tag in args.folds.split(",") if tag])


def _add_errors(commands) -> None:
    errors = commands.add_parser(
        "errors", help="every rating's standard error, from the folds that hold back a tenth of a season's games",
        description="How far each rating would land from the one shown had the season's games come out as another "
                    "draw of the same schedule: the jackknife over the folds of `openrapm folds --folds-per-repeat "
                    "10`, each fold's ratings shrunk with the season's published multipliers.  Writes "
                    "outputs/rating_errors_<NAME>.parquet, the seasons asked for in place of their rows there.")
    first, last = PRODUCT.shrink.fold_seasons
    errors.add_argument("--name", default="product",
                        help="the build whose multipliers are read and whose errors are written (default %(default)s)")
    errors.add_argument("--seasons", type=parse_seasons, default=parse_seasons(f"{first}-{last}"),
                        help="the seasons: 2026, 2017-2026 or 2017,2018 (default %(default)s)")
    errors.add_argument("--folds", metavar="TAG", default=PRODUCT.shrink.extrapolate_folds_tag,
                        help="the folds, outputs/within/<TAG>/; several with commas, each season read from the first "
                             "that holds it (default %(default)s)")
    errors.set_defaults(run=_run_errors)


def _add_second_prior(commands) -> None:
'''),
    ('''    _add_folds(commands)
    _add_second_prior(commands)''',
     '''    _add_folds(commands)
    _add_errors(commands)
    _add_second_prior(commands)'''),
]

if __name__ == "__main__":
    edit("src/openrapm/commands/cli.py", CLI)
