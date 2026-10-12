r"""The swap test: does a season's ranking order TEAMMATES the way the games of the seasons either side do?

Take two lineups of one team in one season that share four players: Podziemski in one, Spencer in the other, the
same four teammates in both.  Everything about the team that stays the same cancels, so the difference between the
two lineups' results is evidence about Podziemski against Spencer.  This check asks whether a ranking, fitted on
OTHER seasons, orders such pairs the way their games did (src/openrapm/swap_test.py; notes/DECISIONS.md, "The swap
test").

Words used below
    swap pair       two lineups of one team in one season that share four players
    scored season   the season whose swap pairs are the evidence
    rated season    the season whose ranking is graded: the season before the scored one ("forward", a rating
                    looking ahead) or the season after it ("back").  A season's own swaps never grade its own
                    ranking, because the ranking was fitted on them
    observation     one scored season graded by one rated season.  1998-2025 gives 56 of them, the
                    year-over-year test's convention
    information     a pair's weight, h = n_a n_b / (n_a + n_b) possessions, with n_a and n_b the two lineups'
                    possessions: a swap seen for 2,000 possessions one way and 20 the other knows about as much as
                    20 possessions
    plain RAPM      single-season RAPM on actual points, one ridge penalty (`--plain_rapm`) on every player.  It is
                    fitted here from the stints once per season and kept in
                    outputs/season_ratings_plainrapm<penalty>.parquet.  It is graded like any other ranking, and
                    the SCORED season's own plain RAPM takes the opponents and the game context out of each swap
                    difference before `order` reads it: one adjustment, the same for every ranking
    context         the level refitted on the scored season with the ratings held fixed (`--contexts`; the
                    columns of each are swap_test.CONTEXTS).  "home" is the intercept and home court; "full" adds
                    playoffs, garbage time, score margin, minutes into the period, its last two minutes, the
                    clutch minutes and each five's minutes on the court without a break; "nofatigue" is "full"
                    without those minutes, and is what the experiment chain uses

How to read the tables, one per side (offense, defense, and net = scored minus allowed), each number averaged over
the observations:
    order          the information-weighted mean of the adjusted swap difference, signed by the ranking's rating
                   gap between the two swapped players, in points per 100.  Higher is better, 0 = no idea, and a
                   ranking's spread cannot change it
    gaps           the information-weighted squared error of the ranking's predicted swap difference.  Lower is
                   better; it sees spread as well as order
    slope          what the swaps ask the ranking's gaps inside a team to be multiplied by: below 1, too wide
    order_vs_ref   the ranking's mean difference from the reference over the observations, its paired
    gaps_vs_ref    t-statistic ("z") and the number of observations it wins ("x of n")

`--checks=1` adds three copies of the reference that say whether the test sees what it claims to:
    shuffled    the reference's ratings reassigned at random among each rated-season team's players (offence and
                defence kept together): the order inside every team destroyed.  It must lose clearly
    team_now    one random constant per team of the SCORED season (normal, standard deviation `--shift_sd`, each
                side) added to every player who played for it: what the test claims to be blind to.  `order`
                must tie, apart from the few players who played for two teams
    team_then   the same with the teams of the RATED season.  Rosters change between seasons, so this must tie on
                pairs who were teammates then too, and lose on pairs who were not: it measures how much of the
                test is a player's credit travelling with him to a new team

Reads   each `--rankings` table, given as name=path[:rating|prior] (`rating`, the default, is the finished
        ratings; `prior` the box prior alone, the prior_off / prior_def columns of the same table); the stints of
        every season the scored seasons touch; the roles table (each player's main team); the plain RAPM file.
Writes  outputs/swaptest_<tag>.parquet (scored season x rated season x ranking x context x side x pair group) and
        outputs/swaptest_<tag>_levels.parquet (the coefficients of the level refit on each scored season), and
        adds any season the plain RAPM file lacks to it.

Example, the experiment chain's call:

    python scripts/checks/90_swap_test.py --contexts=nofatigue --checks=0 --tag=mine \
        --rankings=incumbent=outputs/season_ratings_weightfix.parquet,mine=outputs/season_ratings_mine.parquet
"""
import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from openrapm import swap_test as sw
from openrapm.commands.console import one_thread, use_utf8
from openrapm.data.config import load_config, resolve, resolve_path
from openrapm.data.roles import main_teams
from openrapm.data.stints import load_season_stints
from openrapm.data.tables import load_ratings

REPORT_SIDES = ("net", "offense", "defense")
DIRECTIONS = (("forward", "a ranking predicting the season AFTER it"),
              ("back", "a ranking predicting the season BEFORE it"))
LEVEL_TERMS = ("fatigue_off", "fatigue_def", "period_elapsed", "period_end", "clutch", "is_gt", "home")
LEGEND = ("\n    order: the swap difference with opponents and context taken out, signed by the ranking's gap between",
          "    the two swapped players, information-weighted, points per 100 -- HIGHER is better, 0 = no idea, "
          "spread-free.",
          "    gaps: squared error of the ranking's predicted swap difference -- LOWER is better.  slope: what "
          "the swaps",
          "    ask the ranking's gaps inside a team to be multiplied by (below 1 = too wide).  'x of n' = observations "
          "won.")


@dataclass(frozen=True)
class Grading:
    """What every scored season is graded with.

    names      the `--rankings` names, in order
    tables     name -> (player_id, season, o, d) in raw sign (o = points scored, d = points allowed): the
               `--rankings`, then "shuffled" when the checks are on
    ref        the reference ranking every other is paired against (`--ref`, else the first named)
    contexts   the level refits to run, each a key of swap_test.CONTEXTS (`--contexts`)
    team_of    season -> {player_id: his main team that season}
    checks     whether the shuffled and team copies are graded too (`--checks`)
    seed       seeds the shuffle and the team copies' constants (`--seed`)
    shift_sd   the standard deviation of the team copies' constants (`--shift_sd`)
    """
    names: tuple
    tables: dict
    ref: str
    contexts: list
    team_of: dict
    checks: bool
    seed: int
    shift_sd: float


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="The swap test: does a season's ranking order teammates the way "
                                                 "the neighbouring seasons' games do?", allow_abbrev=False)
    parser.add_argument("--rankings", default="", help="name=path[:rating|prior],... (required)")
    parser.add_argument("--ref", default="", help="the reference ranking (default: the first named)")
    parser.add_argument("--plain_rapm", type=float, default=3000.0,
                        help="plain RAPM's ridge penalty, positive (default %(default)s)")
    parser.add_argument("--checks", default="1",
                        help="0, no or false: leave out the shuffled and team copies (default %(default)s)")
    parser.add_argument("--seed", type=int, default=0, help="seeds the checks' random draws (default %(default)s)")
    parser.add_argument("--shift_sd", type=float, default=2.0,
                        help="standard deviation of the team copies' constants (default %(default)s)")
    parser.add_argument("--contexts", default="full,home", help="comma-separated (default %(default)s)")
    parser.add_argument("--first", type=int, default=1998, help="first scored season (default %(default)s)")
    parser.add_argument("--last", type=int, default=2025, help="last scored season (default %(default)s)")
    parser.add_argument("--tag", default="swap", help="outputs/swaptest_<TAG>.parquet (default %(default)s)")
    return parser.parse_args(argv)


# ------------------------------------------------------------------------------------------- the rankings
def parse_ranking(part: str) -> tuple[str, str, str]:
    """(name, path, columns) from "name=path[:rating|prior]".  Example: "base=outputs/t.parquet:prior"."""
    name, _, path = part.partition("=")
    columns = "rating"
    if path.endswith(":prior") or path.endswith(":rating"):
        path, columns = path.rsplit(":", 1)
    return name, path, columns


def ranking_tables(spec: str, cfg: dict) -> dict:
    """{name: (player_id, season, o, d) in raw sign}, one per comma-separated entry of `--rankings`, in order.

    A relative path is read from the repository root.
    """
    tables = {}
    for part in spec.replace("\n", "").split(","):
        if part.strip():
            name, path, columns = parse_ranking(part.strip())
            tables[name] = load_ratings(resolve_path(cfg, path), columns)
    return tables


def check_arguments(tables: dict, ref: str, penalty: float, contexts: list) -> None:
    """Stop before any work on a reference that is not a ranking, a penalty that is not positive, or an unknown
    context."""
    if ref not in tables:
        raise SystemExit(f"--ref={ref} is not one of {list(tables)}")
    if not penalty > 0:
        raise SystemExit("--plain_rapm must be positive: the scored season's plain RAPM is the swap adjustment")
    unknown = [context for context in contexts if context not in sw.CONTEXTS]
    if unknown:
        raise SystemExit(f"--contexts: {unknown} are not among {list(sw.CONTEXTS)}")


def shuffled_within_team(table: pd.DataFrame, teams: pd.DataFrame, seed: int) -> pd.DataFrame:
    """The same ratings reassigned at random among each team-season's players, a player's (o, d) kept together."""
    rng = np.random.default_rng(seed)
    out = table.merge(teams, on=["player_id", "season"], how="left").reset_index(drop=True)
    for _, rows in out[out.team_id.notna()].groupby(["season", "team_id"]).groups.items():
        rows = np.asarray(rows)
        out.loc[rows, ["o", "d"]] = out.loc[rng.permutation(rows), ["o", "d"]].to_numpy()
    return out.drop(columns="team_id")


def shifted_by_team(table: pd.DataFrame, team_of: dict, seed: int, sd: float) -> pd.DataFrame:
    """One random constant per team and side added to every player `team_of` puts on that team.

    Teams are drawn in ascending id order, offence then defence, so a seed always gives the same constants.
    """
    rng = np.random.default_rng(seed)
    out = table.copy()
    team = out.player_id.map(team_of)
    for team_id in sorted(team.dropna().unique()):
        on_team = (team == team_id).to_numpy()
        out.loc[on_team, "o"] += rng.normal(0.0, sd)
        out.loc[on_team, "d"] += rng.normal(0.0, sd)
    return out


# ------------------------------------------------------------------------------------------- plain RAPM
def plain_rapm_path(cfg: dict, penalty: float) -> Path:
    """The plain RAPM file for one penalty: outputs/season_ratings_plainrapm<penalty>.parquet."""
    return resolve(cfg, "outputs") / f"season_ratings_plainrapm{penalty:.0f}.parquet"


def plain_rapm_of(lineups: sw.Season, season: int, stored: pd.DataFrame | None,
                  penalty: float) -> tuple[pd.DataFrame, bool]:
    """(one season's plain RAPM, whether it was fitted here): read from the file's rows when it has the season
    (player_id, season, o, d), else fitted on the season's lineups (the same columns plus `poss`)."""
    if stored is not None and season in set(stored.season.unique()):
        return stored[stored.season == season], False
    return sw.plain_rapm(lineups, penalty).assign(season=season), True


def write_plain_rapm(fitted: list, path: Path) -> None:
    """Add the seasons fitted here to the plain RAPM file, in the published sign (rating_def positive = good)."""
    new = pd.concat(fitted, ignore_index=True)
    out = pd.DataFrame({"player_id": new.player_id.astype(np.int64), "season": new.season.astype(int),
                        "rating_off": new.o, "rating_def": -new.d, "poss_off": new.poss})
    out["rating_total"] = out.rating_off + out.rating_def
    if path.exists():
        old = pd.read_parquet(path)
        out = pd.concat([old[~old.season.isin(out.season.unique())], out], ignore_index=True)
    out.sort_values(["season", "player_id"]).to_parquet(path, index=False)
    print(f"wrote outputs/{path.name}")


# ------------------------------------------------------------------------------------------- grading
def season_lineups(season: int, cfg: dict, margin_clip: float) -> sw.Season:
    """A season's stints made into its lineups and swap pairs on each end."""
    stints = load_season_stints(season, cfg, sw.STINT_COLUMNS)
    return sw.prepare(sw.season_rows(stints, margin_clip=margin_clip), season)


def rated_tables(grading: Grading, plain: dict, scored: int, rated: int) -> dict:
    """Every ranking to grade on `scored`'s swaps, as its `rated`-season rows: the `--rankings` (and "shuffled"),
    plain RAPM, then the two team copies when the checks are on."""
    tables = {name: table[table.season == rated] for name, table in grading.tables.items()}
    tables["plain_rapm"] = plain[rated]
    if grading.checks:
        ref = tables[grading.ref]
        tables["team_now"] = shifted_by_team(ref, grading.team_of.get(scored, {}), grading.seed + 1000 * scored,
                                             grading.shift_sd)
        tables["team_then"] = shifted_by_team(ref, grading.team_of.get(rated, {}), grading.seed + 1000 * rated + 7,
                                              grading.shift_sd)
    return tables


def grade_season(grading: Grading, lineups: sw.Season, plain: dict, scored: int, seasons: range) -> tuple:
    """([scores], [levels]) for one scored season, graded by the season before it and the season after it.

    Only pairs whose two swapped players every ranking rates are scored, so every ranking is graded on the same
    pairs.
    """
    scores, levels = [], []
    for rated, direction in ((scored - 1, "forward"), (scored + 1, "back")):
        if rated not in seasons:
            continue
        tables = rated_tables(grading, plain, scored, rated)
        rated_players = set.intersection(*(set(table.player_id.astype(np.int64)) for table in tables.values()))
        season_scores, season_levels = sw.score_season(lineups, tables, rated_players, plain[scored],
                                                       contexts=grading.contexts,
                                                       team_then=grading.team_of.get(rated, {}))
        scores.append(season_scores.assign(scored=scored, rated=rated, direction=direction))
        levels.append(season_levels.assign(scored=scored, rated=rated, direction=direction))
    return scores, levels


def run(grading: Grading, cfg: dict, first: int, last: int, penalty: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Grade every ranking on the swaps of every scored season in [first, last], both directions.

    Seasons are read in order and each is kept until the season after it is read, so at most two seasons' lineups
    are in memory.  Returns (scores, levels) in swap_test.score_season's shape plus scored, rated and direction.
    """
    first_season, last_season = int(cfg["first_season"]), int(cfg["last_season"])
    seasons = range(first_season, last_season + 1)
    margin_clip = float(cfg.get("margin_clip", 25))
    path = plain_rapm_path(cfg, penalty)
    stored = load_ratings(path) if path.exists() else None
    plain, fitted, scores, levels = {}, [], [], []
    previous = None
    started = time.time()
    for season in range(max(first_season, first - 1), min(last_season, last + 1) + 1):
        lineups = season_lineups(season, cfg, margin_clip)
        fit, fitted_here = plain_rapm_of(lineups, season, stored, penalty)
        if fitted_here:
            fitted.append(fit)
        plain[season] = fit[["player_id", "season", "o", "d"]]
        scored = season - 1                                     # every neighbour of `scored` is now rated
        if previous is not None and first <= scored <= last:
            season_scores, season_levels = grade_season(grading, previous, plain, scored, seasons)
            scores += season_scores
            levels += season_levels
            print(f"  scored {scored} ({time.time() - started:.0f}s)", flush=True)
        previous = lineups
    if fitted:
        write_plain_rapm(fitted, path)
    return pd.concat(scores, ignore_index=True), pd.concat(levels, ignore_index=True)


# ------------------------------------------------------------------------------------------- the report
def paired_line(part: pd.DataFrame, ranking: str, ref: str, value: str, better: str) -> str:
    """The mean difference ranking - ref over the observations, its paired t-statistic, and how many it wins."""
    by_observation = part.pivot_table(index=["scored", "rated"], columns="ranking", values=value)
    if ranking not in by_observation or ref not in by_observation:
        return "--"
    difference = (by_observation[ranking] - by_observation[ref]).dropna().to_numpy()
    se = difference.std(ddof=1) / np.sqrt(len(difference)) if len(difference) > 1 else np.nan
    z = difference.mean() / se if se and se > 0 else np.nan
    wins = int((difference > 0).sum() if better == "higher" else (difference < 0).sum())
    return f"{difference.mean():+.4f} z {z:+.1f}, {wins} of {len(difference)}"


def table_line(part: pd.DataFrame, name: str, ref: str) -> dict:
    """One ranking's row: its mean order, gaps and slope, and its order and gaps against the reference."""
    mine = part[part.ranking == name]
    return dict(ranking=name, order=mine.order.mean(), gaps=mine.gaps.mean(), slope=mine.slope.mean(),
                order_vs_ref="" if name == ref else paired_line(part, name, ref, "order", "higher"),
                gaps_vs_ref="" if name == ref else paired_line(part, name, ref, "gaps", "lower"))


def print_group(part: pd.DataFrame, names: list, ref: str, side: str, group: str) -> None:
    """The table of one context, side and pair group."""
    if part.empty:
        return
    print(f"\n--- {side}, {group} pairs: {part.pairs.mean():,.0f} swap pairs and {part.information.mean():,.0f} "
          f"possessions of information per observation, {part.groupby(['scored', 'rated']).ngroups} observations")
    lines = [table_line(part, name, ref) for name in names]
    print(pd.DataFrame(lines).to_string(index=False, float_format=lambda value: f"{value:.4f}"))


def print_directions(net: pd.DataFrame, names: list, ref: str) -> None:
    """Net order against the reference, the forward and the back observations apart."""
    for direction, label in DIRECTIONS:
        part = net[net.direction == direction]
        text = "; ".join(f"{name} {paired_line(part, name, ref, 'order', 'higher')}" for name in names if name != ref)
        print(f"    net order vs {ref}, {label}: {text}")


def print_levels(levels: pd.DataFrame, ref: str) -> None:
    """The reference's level refit in the full context: each term's mean and standard deviation."""
    refit = levels[(levels.ranking == ref) & (levels.context == "full")]
    if not len(refit):
        return
    print(f"\n    the level refit on the scored seasons ({ref}, full context), mean and spread over observations:")
    for term in LEVEL_TERMS:
        unit = " per minute on the court" if term.startswith("fatigue") else ""
        print(f"      {term:12s} {refit[term].mean():+.3f}  (sd {refit[term].std():.3f})  points per 100{unit}")


def report(res: pd.DataFrame, levels: pd.DataFrame, ref: str) -> None:
    """Every context's tables (the pair groups apart only for net in the full context), then the legend."""
    names = list(dict.fromkeys(res.ranking))
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        for context in dict.fromkeys(res.context):
            print(f"\n================ context: {context} ({', '.join(sw.CONTEXTS[context])})")
            this = res[res.context == context]
            for side in REPORT_SIDES:
                groups = sw.PAIR_GROUPS if side == "net" and context == "full" else ("all",)
                for group in groups:
                    print_group(this[(this.side == side) & (this.group == group)], names, ref, side, group)
            print_directions(this[(this.side == "net") & (this.group == "all")], names, ref)
    print("\n".join(LEGEND))
    print_levels(levels, ref)


# ------------------------------------------------------------------------------------------- main
def grading_from(args: argparse.Namespace, cfg: dict) -> Grading:
    """The rankings and the checks' settings, the arguments checked; adds "shuffled" when the checks are on."""
    tables = ranking_tables(args.rankings, cfg)
    names = tuple(tables)
    ref = args.ref or names[0]
    contexts = [context for context in args.contexts.split(",") if context]
    check_arguments(tables, ref, args.plain_rapm, contexts)
    checks = args.checks not in ("0", "no", "false")
    teams = main_teams(cfg)
    if checks:
        tables["shuffled"] = shuffled_within_team(tables[ref], teams, args.seed)
    team_of = {season: dict(zip(group.player_id, group.team_id)) for season, group in teams.groupby("season")}
    return Grading(names=names, tables=tables, ref=ref, contexts=contexts, team_of=team_of, checks=checks,
                   seed=args.seed, shift_sd=args.shift_sd)


def swap_test(args: argparse.Namespace) -> None:
    """Grade the rankings, write both tables, print the report."""
    cfg = load_config()
    grading = grading_from(args, cfg)
    copies = ", shuffled, team_now, team_then" if grading.checks else ""
    print(f"rankings: {', '.join(grading.names)}, plain_rapm{copies}; reference {grading.ref}; "
          f"scored seasons {args.first}-{args.last}; contexts {grading.contexts}")
    started = time.time()
    res, levels = run(grading, cfg, args.first, args.last, args.plain_rapm)
    outputs = resolve(cfg, "outputs")
    res.to_parquet(outputs / f"swaptest_{args.tag}.parquet", index=False)
    levels.to_parquet(outputs / f"swaptest_{args.tag}_levels.parquet", index=False)
    print(f"wrote outputs/swaptest_{args.tag}.parquet and _levels ({time.time() - started:.0f}s)")
    report(res, levels, grading.ref)


def main() -> None:
    use_utf8()
    args = parse_args()
    if not args.rankings:
        raise SystemExit(__doc__)
    with one_thread():                     # the level refits are least-squares solves
        swap_test(args)


if __name__ == "__main__":
    main()
