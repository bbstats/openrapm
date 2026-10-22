"""The year-over-year test: how well does one season's ratings table predict the games of the seasons either side?

This is the test that decides experiments.  For every scored season (1998-2025 by default) the previous season's
ratings and the next season's ratings are each asked to predict its games from the ten players' offensive and
defensive ratings alone.  Only the season's level (intercept and home edge) is refitted on the scored season;
nothing else is fitted here, so what is measured is exactly the table handed in.  The runner is
`openrapm.commands.year_over_year`; how a prediction is made and scored is in `openrapm.scoring`.

Reads
    each table named in `--rankings`: player_id, season, rating_off, rating_def (both positive-good), poss_off
    (`--columns=prior` reads prior_off and prior_def instead, to score the box-score prior alone)
    the points design of every scored season (`openrapm.data.designs.DesignCache`)
Writes
    outputs/yoy_<tag>.parquet: one row per scored season x direction x table x split x group, rewritten after
    every season so a run can be read before it ends

How to read the printed numbers (lower is better everywhere)
    game_armse   what a typical team-game misses by, in points per 100 possessions: the root mean squared miss
                 times sqrt(2 / pi).  A team-game is one team's offence in one game
    armse        the same per stint (a stretch of play with the same ten players on the court)
    scale_off, scale_def   what the scored season wants each side multiplied by: 1 = calibrated, below 1 = the
                 ratings are too spread out
    calib_side   the stint error left once each side is multiplied by its scale: the part that is ranking.  A
                 diagnostic, never a score
    paired block each table against `--ref`, season by season: mean_diff below zero = better than the reference;
                 se = standard error of the mean difference; z = mean_diff / se; wins = seasons better
"prev" rows are the PREVIOUS season's ratings predicting this one (they look forward); "next" rows are the next
season's ratings looking back; "both" counts the two as separate observations, 56 over 28 scored seasons.
Example: over both directions the incumbent (outputs/season_ratings_weightfix.parquet) reads game_armse 8.5857 and
the noise control (the same build with the box prior's players assigned to other folds) 8.5853; paired, that is
mean_diff -0.0108, z -0.24, 29 wins of 56: no difference, which is what a noise control should show.

Example (one command; the incumbent is named second and paired against):

    python scripts/checks/63_yoy.py --ref=incumbent --tag=new --splits= \\
        --rankings=new=outputs/season_ratings_new.parquet,incumbent=outputs/season_ratings_weightfix.parquet

Why this test and not the within-season 75/25 split.  The 75/25 split holds a quarter of a season's games out of
the ridge but not out of the prior's on-court inputs, which are averaged over the whole season (notes/DECISIONS.md,
2026-09-13: the within-season split leaks).  Here the scored games are a different season from the rated one,
so nothing in the rated season's prior or evidence has seen them.  And a player who changed teams carries his
rating to new teammates, which is where a rating that is really his old team's credit stops predicting: the
`movers` split reads that group on its own.

What a table must satisfy to be tested honestly.  A season's ratings may use that season's games and model
coefficients learned from OTHER seasons, but not from the two seasons being scored.  `openrapm rate --setting
test` builds a table that way.  A table whose prior was trained on every other season (the shipped product
setting) has seen the scored season's outcomes through the prior's coefficients and reads optimistically here.

`--fill` gives a table a replacement level for the players it has no row for.  A season's table covers only the
players who played that season, so a rookie of the scored season (10% of its possessions going forward, 6% going
back) is scored as the average player, 0, which is the one thing he is not.  `--fill=name:500` gives that table
the possession-weighted mean rating of its own players under 500 possessions, read off the rated season alone
(`openrapm.scoring.replacement_level`); `500x0.25` takes a quarter of that level, the depth the test settled on
(notes/DECISIONS.md, experiment 11); `name:-1.6/-0.5` sets the two levels by hand, positive-good, offence first.  A
table not listed keeps the 0 fill, so the same parquet can be handed in twice under two names and paired.

`--offset=N` asks the ratings N seasons away instead of the adjacent one.  The trade set
(scripts/checks/70_tradeset.py) is fitted on the games one season either side of the rated one, so it is scored
at offset 2, where the scored games are outside its window.  Both arms of a comparison must use the same offset.

`--rankings=name=a.parquet|b.parquet` scores the first file in the "prev" direction and the second in the "next"
(the trade-flag test's stitched list, from the retired scripts/89_stitch_by_move.py, depended on the direction).

`--splits` names any of `movers` (how many of the ten on the court changed team) and `game_bench_share` (the share
of each team-game's possessions played by players with fewer than 500 possessions in the rated season);
`--splits=` scores the whole season only.
"""
import argparse
import time

from openrapm.commands import year_over_year as yoy
from openrapm.commands.console import use_utf8
from openrapm.data.config import load_config, resolve, resolve_path
from openrapm.data.designs import DesignCache
from openrapm.data.tables import load_ratings


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="The year-over-year test: one season's ratings predict the "
                                                 "games of the seasons either side of it.", allow_abbrev=False)
    parser.add_argument("--rankings", required=True,
                        help="name=path[,name=path...]; name=a.parquet|b.parquet gives one file per direction")
    parser.add_argument("--ref", help="the table the others are paired with (default the first named)")
    parser.add_argument("--splits", default="movers",
                        help=f"comma-separated, any of {', '.join(yoy.SPLITS)}; empty for none (default %(default)s)")
    parser.add_argument("--tag", default="yoy", help="writes outputs/yoy_<TAG>.parquet (default %(default)s)")
    parser.add_argument("--first", type=int, help="first scored season (default the second configured season)")
    parser.add_argument("--last", type=int, help="last scored season (default the second-last configured season)")
    parser.add_argument("--columns", default="rating", choices=("rating", "prior"),
                        help="rating: the finished ratings; prior: the box prior alone (default %(default)s)")
    parser.add_argument("--fill", default="",
                        help="name:500, name:500x0.25 or name:-1.6/-0.5 (positive-good), comma-separated")
    parser.add_argument("--offset", type=int, default=1,
                        help="the rated season's distance from the scored one (default %(default)s)")
    return parser.parse_args(argv)


def read_tables(spec: str, columns: str, cfg: dict) -> dict:
    """{name: {"prev": table, "next": table}} from `--rankings`, each table in raw sign with its possessions.

    Example: "new=a.parquet|b.parquet" reads a.parquet for the "prev" direction and b.parquet for "next".
    """
    tables = {}
    for part in spec.split(","):
        name, path = part.split("=", 1)
        paths = path.split("|")
        if len(paths) not in (1, 2):
            raise SystemExit(f"--rankings {name}: one path, or two joined by |")
        loaded = [load_ratings(resolve_path(cfg, p), columns, possessions=True) for p in paths]
        tables[name] = {"prev": loaded[0], "next": loaded[-1]}
    return tables


def read_fills(spec: str, names) -> dict:
    """{name: fill spec} from `--fill`; every name must be one of the tables'.

    Example: "new:500x0.25,old:-1.6/-0.5" gives {"new": "500x0.25", "old": "-1.6/-0.5"}.
    """
    fills = dict(part.rsplit(":", 1) for part in spec.split(",") if part)
    unknown = set(fills) - set(names)
    if unknown:
        raise SystemExit(f"--fill names not in --rankings: {sorted(unknown)}")
    return fills


def describe_fill(spec: str) -> str:
    """The fill in words, as the run log prints it."""
    return spec if "/" in spec else f"mean rating of the table players under {spec} possessions"


def main() -> None:
    use_utf8()
    args = parse_args()
    cfg = load_config()
    tables = read_tables(args.rankings, args.columns, cfg)
    ref = next(iter(tables)) if args.ref is None else args.ref
    if ref not in tables:
        # a misspelled reference used to print the pooled scores and no paired comparison, without a word
        raise SystemExit(f"--ref {ref}: not one of the --rankings names {list(tables)}")
    fills = read_fills(args.fill, tables)
    if args.offset != 1:
        print(f"offset {args.offset}: each season is predicted by the rankings {args.offset} seasons before and "
              "after it")
    systems = yoy.neighbour_tables(tables, args.offset, fills)
    for name, spec in fills.items():
        print(f"fill {name}: {describe_fill(spec)}")
    started = time.time()
    res = yoy.run(systems, DesignCache(cfg), yoy.scored_seasons(cfg, args.first, args.last),
                  splits=[s for s in args.splits.split(",") if s],
                  out=resolve(cfg, "outputs") / f"yoy_{args.tag}.parquet")
    print(f"scored {res.held_out.nunique()} seasons in {time.time() - started:.0f}s\n")
    print(yoy.report(res, ref, compare=len(tables) > 1))


if __name__ == "__main__":
    main()
