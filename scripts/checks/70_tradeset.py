"""Check 70, the trade set: what do a player's comings and goings say that his rating did not already know?

Inside one season the same five men keep playing together, so how a lineup's credit is split among them is barely
identified.  A player being gone identifies it: a trade, an injury, a rotation change.  For every rated season this
check pools that season's games and the games of the season either side into team-games, subtracts the rated
season's rating from every team-game, and ridges what is left back onto the players (`openrapm.trade_set`).  The
coefficient is alpha, and the check writes it for every player the rated season's table carries.

Words used below
    team-game      one team's offence in one game: its points per 100 possessions on the target, weighted by its
                   possessions
    share          a player's share of his team's possessions in that game: 0 when he did not play, and the five on
                   the court sum to 5
    block          the rated season and --block seasons either side of it, clipped to the seasons that exist.  A
                   block is one-sided when it has fewer than three seasons (the first and the last season)
    alpha          what a player's comings and goings say his rating did not already know, points per 100.  Written
                   positive-good as `alpha_good` (a defensive alpha of -1.0 raw is +1.0 good)
    rating scale   with --free_scale=1 each side's summed rating is one more free column, so a rating that is too
                   wide for the neighbouring seasons is read as one multiplier per side, not written into every alpha
    penalty        the ridge's lambda on the alphas, chosen from --lambdas
    fill           what a player the rated season's table has no row for is worth: --fill=500x0.25 is a quarter of
                   the possession-weighted mean rating of the table's players under 500 possessions

The fit, per side, for one rated season:

    minimise   sum over team-games  w (y - Z r - s (Z r) - F c - Z alpha)^2  +  lambda |alpha|^2

    y      the team-game's points per 100 possessions on the target (--target_off, --target_def); w its possessions
    Z      the players' possession shares; r the rated season's rating in raw sign (defence = points allowed)
    s      the rating scale minus one (free, one per side; absent with --free_scale=0)
    F, c   the fixed columns, unpenalised: each season's level, home court, playoffs, and with --team_effects one
           level per team (team) or per team and season (team_season)

**Alpha never becomes a published rating.**  It reads the neighbouring seasons' games, and ruling 1 is that a
season's rating comes from that season's games alone.  It is a per-player loss that counts a bench player as much
as a starter (check 73 reads it), and a diagnostic of the box prior.

**Choosing the penalty.**  Each penalty is scored on the games two seasons away (H-2 and H+2), which are outside the
three seasons alpha was fitted on.  Three kinds of arm are scored: the rating alone; the rating times its rating
scale ("rescale only"); and that plus alpha ("rescale + alpha").  Without the middle arm a rescale and a discovery
about players cannot be told apart.  The best penalty must be inside the grid: a best value at either end has chosen
nothing.  Both arms read optimistically, because the incumbent's own prior was only held off H's immediate
neighbours, so only the paired difference between arms means anything.

**Only players the rated season's table already carries get a row.**  The regression has a column for everyone who
played in the three seasons, but publishing alpha for a player the rated season has no rating for would change who
is covered as well as how well they are rated.

Reads the ratings table (--rankings), the stints through the design cache, and the roles table.  Writes, in outputs/:
    <out>_alpha.parquet       one row per player, season and side at the chosen penalty
    <out>_lambda.parquet      the penalty grid's scores (with --score=1)
    <out>_loss.parquet        the trade loss by season, side and possession tier (`trade_set.trade_loss`)
    <out>_yoy_input.parquet   the corrected rating, scale x rating + alpha, in the schema check 63 reads
    <out>_rankings.parquet    the same in the full ratings schema, for check 66

How to read the penalty table: game_armse is what a typical team-game misses by, in points per 100 (root mean
square x sqrt(2 / pi)); vs_rating_alone is the paired mean difference in team-game mean square against the rating
alone, negative = better; z is that difference over its standard error; wins counts the observations better.
Example (the incumbent's 2014, 2015 and 2026, --team_effects=team): at a penalty of 10,000 the corrected rating misses
a typical team-game by 8.7083 points per 100 against 8.7489 for the rating alone, better in all 5 observations; the
rescale alone removes -0.0104 of that (it makes things worse) and the per-player part removes 0.0510.

    python scripts/checks/70_tradeset.py --rankings=outputs/season_ratings_weightfix.parquet --team_effects=team \\
                                         --out=tradeset_weightfix

and the corrected rating's year-over-year score, at offset 2 so the scored games are outside alpha's block:

    python scripts/checks/63_yoy.py --offset=2 --ref=incumbent --tag=tradeset \\
        --fill=alpha:500x0.25,incumbent:500x0.25 \\
        --rankings=alpha=outputs/tradeset_weightfix_yoy_input.parquet,incumbent=outputs/season_ratings_weightfix.parquet
"""
import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from openrapm import trade_set
from openrapm.commands.console import one_thread, use_utf8
from openrapm.data.config import load_config, resolve, resolve_path
from openrapm.data.designs import DesignCache
from openrapm.data.legacy_names import target_name
from openrapm.data.roles import roles_path
from openrapm.data.tables import load_ratings, team_of_rows
from openrapm.scoring import Ratings, armse, predict_season, score

NO_ALPHA = "rating alone"
OFF = ("0", "no", "false")
# Under per-season team levels each team-season column already carries its season's level, so the season
# intercepts come out rather than sit there exactly collinear with them.
TEAM_SEASON_CONTROLS = ("home", "is_po", "po_home")


def on_off(text: str) -> bool:
    """A switch as typed: 0, no or false is off, anything else on."""
    return str(text) not in OFF


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="The trade set: each season's with-and-without-you correction.",
                                     allow_abbrev=False)
    parser.add_argument("--rankings", default="outputs/season_ratings_sy_lam13037.parquet",
                        help="the ratings table, relative to the repository (default %(default)s)")
    parser.add_argument("--lambdas", default="1000,3000,10000,30000,100000,300000",
                        help="the penalties swept (default %(default)s)")
    parser.add_argument("--fill", default="500x0.25", help="the replacement level (default %(default)s)")
    # an old target name (xpts_ft, x3def_w0.25, x3def) is read as its plain one (openrapm.data.legacy_names)
    parser.add_argument("--target_off", default="points_with_expected_free_throws", type=target_name,
                        help="the offence's target (default %(default)s)")
    parser.add_argument("--target_def", default="points_with_expected_opponent_threes", type=target_name,
                        help="the defence's target (default %(default)s)")
    parser.add_argument("--out", default="tradeset", help="outputs/<OUT>_*.parquet (default %(default)s)")
    parser.add_argument("--score", default="1", type=on_off, help="0: skip scoring the penalties (default %(default)s)")
    parser.add_argument("--block", default=1, type=int,
                        help="seasons either side of the rated one (default %(default)s)")
    parser.add_argument("--team_effects", default="none", choices=("none", "team", "team_season"),
                        help="free team levels (default %(default)s)")
    parser.add_argument("--free_scale", default="1", type=on_off,
                        help="0: the rating enters at its own amplitude (default %(default)s)")
    parser.add_argument("--seasons", help="comma-separated rated seasons (default --first to --last)")
    parser.add_argument("--first", type=int, help="first rated season (default config.yaml's first_season)")
    parser.add_argument("--last", type=int, help="last rated season (default config.yaml's last_season)")
    return parser.parse_args(argv)


@dataclass(frozen=True)
class Run:
    """One run's settings: the command line read against config.yaml."""
    rankings: Path
    outputs: Path
    lambdas: tuple
    max_poss: float
    shrink: float
    target: dict
    tag: str
    score: bool
    width: int
    team_effects: str
    free_scale: bool
    seasons: tuple
    first: int          # the first season with data (config.yaml), which every block is clipped to
    last: int


def settings(args: argparse.Namespace, cfg: dict) -> Run:
    """The run's settings.  --fill=500x0.25 is (500 possessions, a quarter); --fill=500 is the whole level."""
    first, last = int(cfg["first_season"]), int(cfg["last_season"])
    max_poss, _, shrink = args.fill.partition("x")
    seasons = ([int(s) for s in args.seasons.split(",")] if args.seasons
               else list(range(args.first or first, (args.last or last) + 1)))
    return Run(rankings=resolve_path(cfg, args.rankings), outputs=resolve(cfg, "outputs"),
               lambdas=tuple(float(x) for x in args.lambdas.split(",")), max_poss=float(max_poss),
               shrink=float(shrink or 1.0), target={"offense": args.target_off, "defense": args.target_def},
               tag=args.out, score=args.score, width=args.block, team_effects=args.team_effects,
               free_scale=args.free_scale, seasons=tuple(seasons), first=first, last=last)


def print_header(run: Run, table: pd.DataFrame) -> None:
    """The run's settings in words: the table, the penalties, the fill, the team terms and the free scale."""
    print(f"rankings {run.rankings.name}: {len(table)} player-seasons, {table.season.nunique()} seasons")
    print(f"penalties {', '.join(f'{lam:,.0f}' for lam in run.lambdas)}"
          f"   fill {run.shrink:g} x the mean rating of players under {run.max_poss:,.0f} possessions")
    if run.team_effects != "none":
        print(f"team terms: {run.team_effects}. Each team's own level is free, per side, so a player is")
        print("identified by variation in who played inside his team, not by his team being good")
    print("the rating enters as two free columns per fit, one per side, so a pure rescale is read as a"
          if run.free_scale else "the rating enters at exactly the amplitude it came with (--free_scale=0):")
    print("scale and not written into every player's alpha (--free_scale=0 turns that off)"
          if run.free_scale else "any uniform tilt will appear in the alphas")


# ------------------------------------------------------------------------------------------ one rated season
def block_of(season: int, first: int, last: int, width: int = 1) -> list:
    """The rated season and `width` seasons either side, clipped to the seasons that exist.

    Clipped rather than merely skipped: handing the loader a season with no stints on disk makes it scrape, the one
    failure here that costs an hour instead of raising.

    `width=0` is the control that matters.  It leaves the rated season on its own, so there is no with-and-without
    contrast across seasons and alpha can only be a team-game refit of the games the rating already saw.  Whatever
    that arm scores owes nothing to a player being absent; the trade set's claim is only what it wins on top.

    Example: block_of(2026, 1997, 2026) is [2025, 2026], a one-sided block.
    """
    return [s for s in range(season - int(width), season + int(width) + 1) if first <= s <= last]


@dataclass
class SeasonFit:
    """One rated season's trade set: who the columns are, how much absence each has, and the fit at every penalty.

    `alpha` maps a penalty to {"offense": vector, "defense": vector} in raw sign, one entry per player column;
    `scale` maps a penalty to the (offence, defence) rating multipliers, and is empty with --free_scale=0.
    """
    season: int
    block: list
    player_ids: np.ndarray
    teams: pd.DataFrame
    exposure: pd.DataFrame
    normal: dict
    n_team_games: int = 0
    n_columns: int = 0
    alpha: dict | None = None
    scale: dict | None = None


def first_side(ts: trade_set.TradeSet, roles: pd.DataFrame, season: int, block: list, cfg: dict) -> SeasonFit:
    """What the first target's team-games fix for both sides: the player columns, each row's teams, the exposure."""
    teams = team_of_rows(ts.keys, cfg)
    eligible = trade_set.eligibility(roles, season)
    exposure = trade_set.exposure_table(
        ts, teams, dict(zip(eligible.player_id.astype(int), eligible.team_id.astype(int))), season)
    return SeasonFit(season=season, block=block, player_ids=ts.player_ids, teams=teams, exposure=exposure, normal={})


def team_terms(ts: trade_set.TradeSet, teams: pd.DataFrame, team_effects: str) -> tuple:
    """(the free team columns or None, how many there are) for --team_effects."""
    if team_effects == "none":
        return None, 0
    extra, names = trade_set.team_columns(ts, teams, per_season=(team_effects == "team_season"))
    return extra, len(names)


def count_size(fit: SeasonFit, ts: trade_set.TradeSet, n_team_columns: int) -> None:
    """Record the fit's team-games and columns, after checking every team-game's offensive shares sum to five."""
    shares = np.asarray(ts.Z[:, :ts.n_players].sum(axis=1)).ravel()
    if not np.allclose(shares, 5.0, atol=1e-6):
        raise ValueError("a team-game's shares do not sum to five")
    fit.n_team_games = ts.Z.shape[0]
    fit.n_columns = ts.Z.shape[1] + len(ts.control_names) + n_team_columns


def fit_season(designs: DesignCache, run: Run, season: int, block: list, ratings: pd.DataFrame, fill: tuple,
               roles: pd.DataFrame, cfg: dict) -> SeasonFit:
    """Both sides' normal equations on the block's team-games, then the alphas at every penalty."""
    fit = None
    controls = TEAM_SEASON_CONTROLS if run.team_effects == "team_season" else trade_set.CONTROL_PREFIXES
    for side in trade_set.SIDES:
        ts = trade_set.team_game_design(designs.block(block, run.target[side]), control_prefixes=controls)
        if fit is None:
            fit = first_side(ts, roles, season, block, cfg)
        elif not np.array_equal(fit.player_ids, ts.player_ids):
            raise SystemExit("the two targets built different player columns")
        extra, n_team_columns = team_terms(ts, fit.teams, run.team_effects)
        offset = trade_set.offset_vector(ts.player_ids, ratings, *fill)
        fit.normal[side] = (*trade_set.normal_equations(ts, offset, free_scale=run.free_scale, extra=extra),
                            ts.n_players)
        if side == "offense":
            count_size(fit, ts, n_team_columns)
    designs.clear()
    fit.alpha, fit.scale = fit_penalties(fit.normal, run.lambdas, run.free_scale)
    return fit


def side_alpha(alpha: np.ndarray, n_players: int, side: str) -> np.ndarray:
    """The half of a fit's coefficients that side takes, raw sign."""
    return alpha[:n_players] if side == "offense" else alpha[n_players:]


def fit_penalties(normal: dict, lambdas: tuple, free_scale: bool) -> tuple:
    """({penalty: {side: alpha}}, {penalty: (offence scale, defence scale)}): one solve per side and penalty.

    The scale is what each side's rating would have to be multiplied by for the three seasons to agree with it;
    read it before the alphas, because a rescale is not attribution.
    """
    alpha, scale = {}, {}
    for lam in lambdas:
        fits = {side: trade_set.solve_alpha(gram, rhs, n_players, n_free, lam)
                for side, (gram, rhs, n_free, n_players) in normal.items()}
        alpha[lam] = {side: side_alpha(fit[0], normal[side][3], side) for side, fit in fits.items()}
        if free_scale:
            scale[lam] = (1.0 + fits["offense"][1][0], 1.0 + fits["defense"][1][1])
    return alpha, scale


def print_season(fit: SeasonFit, run: Run, started: float) -> None:
    """One line per rated season: its block, its size, how many players have games without them, its scale."""
    moved = fit.exposure[(fit.exposure.side == "offense") & (fit.exposure.without_poss > 0)]
    first_scale = fit.scale.get(run.lambdas[0])
    scales = f", rating scale {first_scale[0]:.2f} / {first_scale[1]:.2f}" if run.free_scale else ""
    print(f"  {fit.season}: block {fit.block[0]}-{fit.block[-1]}, {fit.n_team_games:,} team-games, "
          f"{fit.n_columns:,} columns, {len(moved):,} players with games their team played without them{scales}"
          f"  ({time.time() - started:.0f}s)", flush=True)


# ---------------------------------------------------------------------------------- scoring the penalties
def corrected(ratings_h: pd.DataFrame, player_ids, alpha_off, alpha_def, scale=(1.0, 1.0)) -> pd.DataFrame:
    """`scale x rating + alpha` for the rated season's own players, raw sign, ready for `Ratings`.

    The scale belongs here.  The fit explains a team-game by the rating times one free number per side PLUS the
    per-player correction, so the corrected rating is the two together; adding alpha to the unscaled rating would
    be reading half of a fit and reporting it as the whole.

    Example: rating +2.0, scale 0.9 and alpha +0.5 give 0.9 x 2.0 + 0.5 = +2.3.
    """
    add = pd.DataFrame({"player_id": np.asarray(player_ids, dtype=np.int64),
                        "alpha_o": alpha_off, "alpha_d": alpha_def})
    out = ratings_h.merge(add, on="player_id", how="left").fillna({"alpha_o": 0.0, "alpha_d": 0.0})
    return pd.DataFrame({"player_id": out.player_id, "o": scale[0] * out.o + out.alpha_o,
                         "d": scale[1] * out.d + out.alpha_d, "poss": out.poss})


def penalty_arms(fit: SeasonFit, lambdas: tuple) -> dict:
    """{arm: (offence alpha, defence alpha, scale)}: the rating alone, then per penalty the rescale only and the
    rescale plus alpha."""
    zero = np.zeros(len(fit.player_ids))
    arms = {NO_ALPHA: (zero, zero, (1.0, 1.0))}
    for lam in lambdas:
        scale = fit.scale.get(lam, (1.0, 1.0))
        arms[f"{lam:,.0f} rescale only"] = (zero, zero, scale)
        arms[f"{lam:,.0f} rescale + alpha"] = (fit.alpha[lam]["offense"], fit.alpha[lam]["defense"], scale)
    return arms


def score_neighbours(designs: DesignCache, fit: SeasonFit, ratings: pd.DataFrame, fill: tuple, run: Run) -> list:
    """Every arm's score on the games two seasons either side of the rated one: one row per scored season and arm."""
    rows = []
    for neighbour in (fit.season - 2, fit.season + 2):
        if not (run.first <= neighbour <= run.last):
            continue
        design = designs.season(neighbour, "pts")
        for arm, (alpha_off, alpha_def, scale) in penalty_arms(fit, run.lambdas).items():
            table = corrected(ratings, fit.player_ids, alpha_off, alpha_def, scale)
            result = score(predict_season(Ratings(table, fill_o=fill[0], fill_d=fill[1]), design))
            rows.append(dict(season=fit.season, scored=neighbour,
                             direction="prev" if neighbour < fit.season else "next",
                             arm=arm, tg=result["tg"], tg_n=result["tg_n"], mse=result["mse"],
                             scale_off=result["scale_off"], scale_def=result["scale_def"]))
    designs.clear()
    return rows


def fit_seasons(run: Run, table: pd.DataFrame, roles: pd.DataFrame, designs: DesignCache, cfg: dict) -> tuple:
    """(every rated season's fit, the penalty grid's rows, the one-sided seasons), season by season."""
    fits, grid_rows, one_sided = [], [], []
    started = time.time()
    for season in run.seasons:
        block = block_of(season, run.first, run.last, run.width)
        if len(block) < 3:
            one_sided.append(season)
        ratings = table[table.season == season]
        if len(ratings) == 0:
            print(f"  {season}: no rows in the rankings, skipped")
            continue
        fill = trade_set.replacement_fill(ratings, max_poss=run.max_poss, shrink=run.shrink)
        fit = fit_season(designs, run, season, block, ratings, fill, roles, cfg)
        print_season(fit, run, started)
        fits.append(fit)
        if run.score:
            grid_rows += score_neighbours(designs, fit, ratings, fill, run)
    return fits, grid_rows, one_sided


# --------------------------------------------------------------------------------------------- the penalty
def arm_summary(arm: str, part: pd.DataFrame, base: pd.Series) -> dict:
    """One arm pooled over the scored seasons, and paired against the rating alone on the same observations."""
    pooled = float(np.average(part.tg, weights=part.tg_n))
    diff = (part.set_index(["season", "scored"]).tg - base).dropna()
    se = float(diff.std(ddof=1) / np.sqrt(len(diff))) if len(diff) > 1 else np.nan
    return dict(penalty=arm, observations=len(part), game_armse=armse(pooled),
                vs_rating_alone=float(diff.mean()), z=float(diff.mean() / se) if se else np.nan,
                wins=int((diff < 0).sum()), scale_off=part.scale_off.mean(), scale_def=part.scale_def.mean())


def paired_arms(grid: pd.DataFrame, arm: str, reference: str) -> tuple:
    """(mean difference, paired statistic, wins, observations) of one arm against another, same rows."""
    key = ["season", "scored"]
    difference = (grid[grid.arm == arm].set_index(key).tg
                  - grid[grid.arm == reference].set_index(key).tg).dropna()
    se = float(difference.std(ddof=1) / np.sqrt(len(difference))) if len(difference) > 1 else np.nan
    return (float(difference.mean()), float(difference.mean() / se) if se else np.nan,
            int((difference < 0).sum()), len(difference))


def print_split(grid: pd.DataFrame, frame: pd.DataFrame, chosen: float, best: int) -> None:
    """The improvement split in two: the part one number per side buys, and the per-player part on top of it.

    A rescale says nothing about any individual player, so the two halves are separated before the number is
    quoted anywhere.
    """
    swept = frame[frame.penalty.str.endswith("rescale + alpha")].reset_index(drop=True)
    rescale = frame[frame.penalty.str.endswith("rescale only")].reset_index(drop=True)
    baseline = float(frame[frame.penalty == NO_ALPHA].game_armse.iloc[0])
    only = float(rescale.game_armse.iloc[best])
    beyond = paired_arms(grid, f"{chosen:,.0f} rescale + alpha", f"{chosen:,.0f} rescale only")
    print("\n  Splitting that improvement in two, in points per 100 of game_armse REMOVED "
          "(bigger is better here):")
    print(f"    {baseline - only:.4f} removed by the RESCALE alone, one number per side, "
          "no player told apart from another")
    print(f"    {only - float(swept.game_armse.iloc[best]):.4f} removed by the per-player correction "
          "on top of it")
    print(f"    paired on the per-player part only: {beyond[0]:+.4f} mse, paired statistic {beyond[1]:+.2f}, "
          f"{beyond[2]} of {beyond[3]} observations better")
    if beyond[0] >= 0:
        print("    The per-player correction adds nothing beyond the rescale.  That is the finding;")
        print("    the alpha table below is still written, as the diagnostic it is.")
    print("\n  Read this as a forecast and not as proof of attribution: the corrected rating carries three")
    print("  seasons of games where the plain one carries a single season, so part of any gap is simply")
    print("  the larger sample.  What the trade set is FOR is the per-player loss and the feature table.")


def report_grid(grid: pd.DataFrame, lambdas: tuple) -> float:
    """The penalty grid, scored two seasons out, with the paired difference against the rating alone; returns the
    penalty with the lowest game_armse."""
    base = grid[grid.arm == NO_ALPHA].set_index(["season", "scored"]).tg
    frame = pd.DataFrame([arm_summary(arm, part, base) for arm, part in grid.groupby("arm", sort=False)])
    print("\n=== the penalty, scored on the games TWO seasons away, LOWER IS BETTER")
    print("    game_armse: what a typical team-game misses by, points per 100.  Both arms read optimistically")
    print("    here, so only vs_rating_alone means anything; a negative difference is an improvement.")
    print(frame.round(4).to_string(index=False))

    swept = frame[frame.penalty.str.endswith("rescale + alpha")].reset_index(drop=True)
    best = int(swept.game_armse.idxmin())
    chosen = lambdas[best]
    interior = 0 < best < len(swept) - 1
    baseline = float(frame[frame.penalty == NO_ALPHA].game_armse.iloc[0])
    print(f"\n  best penalty {chosen:,.0f} ({'interior' if interior else 'AT THE EDGE OF THE GRID'}), "
          f"{swept.game_armse.iloc[best]:.4f} against {baseline:.4f} for the rating alone")
    if not interior:
        print("  An argmin on a boundary has chosen nothing: widen --lambdas before believing it.")
    print_split(grid, frame, chosen, best)
    return chosen


def choose_penalty(grid_rows: list, run: Run) -> float:
    """The penalty the alpha table is written at: the grid's best when it was scored, else the first."""
    if not run.score:
        return run.lambdas[0]
    if not grid_rows:
        print("\nnothing was scored")
        return run.lambdas[0]
    grid = pd.DataFrame(grid_rows)
    grid.to_parquet(run.outputs / f"{run.tag}_lambda.parquet", index=False)
    with pd.option_context("display.width", 220, "display.max_columns", 30, "display.precision", 4):
        return report_grid(grid, run.lambdas)


# ------------------------------------------------------------------------------------------ the alpha table
def side_frame(fit: SeasonFit, side: str, chosen: float, raw_season: pd.DataFrame) -> pd.DataFrame:
    """One season and side at the chosen penalty, for the players the season's table carries."""
    column = "rating_off" if side == "offense" else "rating_def"
    sign = 1.0 if side == "offense" else -1.0     # positive-good, from the model's raw sign
    alpha = fit.alpha[chosen][side]
    frame = pd.DataFrame({"player_id": fit.player_ids, "season": fit.season, "side": side,
                          "alpha_raw": alpha, "alpha_good": sign * alpha,
                          "rating_scale": fit.scale.get(chosen, (1.0, 1.0))[0 if side == "offense" else 1]})
    return (frame.merge(raw_season[["player_id", column, "poss_off"]], on="player_id", how="inner")
            .rename(columns={column: "rating"}))


def alpha_table(fits: list, chosen: float, raw: pd.DataFrame, roles: pd.DataFrame, one_sided: list,
                team_effects: str, names: pd.DataFrame) -> pd.DataFrame:
    """One row per player, season and side: alpha, the rating, the exposure, eligibility and the corrected rating."""
    table = pd.concat([side_frame(fit, side, chosen, raw[raw.season == fit.season])
                       for fit in fits for side in trade_set.SIDES], ignore_index=True)
    exposure = pd.concat([fit.exposure.assign(season=fit.season) for fit in fits], ignore_index=True)
    table = table.merge(exposure, on=["player_id", "season", "side"], how="left")
    eligible = pd.concat([trade_set.eligibility(roles, fit.season).assign(season=fit.season) for fit in fits],
                         ignore_index=True)
    table = table.merge(eligible[["player_id", "season", "poss_on"]], on=["player_id", "season"], how="left")
    table["eligible"] = table.poss_on.notna()
    table["poss_on"] = table.poss_on.fillna(table.poss_off)
    table["one_sided"] = table.season.isin(one_sided)
    table["penalty"] = chosen
    # Which arm produced this: without it nothing downstream can tell an uncontrolled table from a team-controlled one.
    table["team_effects"] = team_effects
    # the corrected rating the fit implies: the amplitude and the per-player part together
    table["corrected"] = table.rating_scale * table.rating + table.alpha_good
    return table.merge(names, on="player_id", how="left")


def to_wide(alpha: pd.DataFrame) -> pd.DataFrame:
    """The corrected rating per player-season, positive-good, in the schema check 63 reads."""
    wide = alpha.pivot_table(index=["player_id", "season"], columns="side",
                             values=["rating", "alpha_good", "corrected", "poss_off"]).reset_index()
    wide.columns = [c[0] if not c[1] else f"{c[0]}_{c[1][:3]}" for c in wide.columns]
    return pd.DataFrame({"player_id": wide.player_id.astype(np.int64), "season": wide.season.astype(int),
                         "rating_off": wide.corrected_off, "rating_def": wide.corrected_def,
                         "poss_off": wide.poss_off_off})


def write_tables(alpha: pd.DataFrame, names: pd.DataFrame, run: Run, chosen: float) -> None:
    """The alpha table, the corrected rating for check 63, and the same in the full ratings schema for check 66."""
    alpha.to_parquet(run.outputs / f"{run.tag}_alpha.parquet", index=False)
    print(f"\nwrote outputs/{run.tag}_alpha.parquet: {len(alpha):,} rows at penalty {chosen:,.0f}")
    wide = to_wide(alpha)
    wide.to_parquet(run.outputs / f"{run.tag}_yoy_input.parquet", index=False)
    print(f"wrote outputs/{run.tag}_yoy_input.parquet: the corrected rating, the schema scripts/checks/63_yoy.py reads")
    (wide.assign(rating_total=wide.rating_off + wide.rating_def).merge(names, on="player_id", how="left")
     .to_parquet(run.outputs / f"{run.tag}_rankings.parquet", index=False))
    print(f"wrote outputs/{run.tag}_rankings.parquet: the same, in the schema scripts/checks/66_compare.py reads")


def print_rating_scale(alpha: pd.DataFrame) -> None:
    """The free scale each side's fits asked for: its mean, minimum and maximum over the rated seasons."""
    by_season = alpha.drop_duplicates(["season", "side"])
    print()
    print("=== the rating amplitude the three seasons ask for (1.00 = the rating as it stands)")
    for side, group in by_season.groupby("side").rating_scale:
        print(f"    {side}: {group.mean():.3f} (min {group.min():.3f}, max {group.max():.3f})")
    print("    Below 1 means the rating is too wide for the neighbouring seasons to bear out.")


# --------------------------------------------------------------------------------------------- the report
def flat_table(alpha: pd.DataFrame) -> pd.DataFrame:
    """One row per player-season, both sides side by side, with the totals."""
    flat = (alpha.pivot_table(index=["player_id", "player_name", "season", "eligible"], columns="side",
                              values=["rating", "alpha_good", "corrected", "with_poss", "without_poss", "poss_off"])
            .reset_index())
    flat.columns = [c[0] if not c[1] else f"{c[0]}_{c[1][:3]}" for c in flat.columns]
    flat["total"] = flat.rating_off + flat.rating_def
    flat["alpha_total"] = flat.alpha_good_off + flat.alpha_good_def
    flat["corrected"] = flat.corrected_off + flat.corrected_def
    return flat


def print_season_tables(flat: pd.DataFrame, season: int) -> None:
    """A season's top 20 by the rating with alpha beside it, and its 20 largest corrections."""
    part = flat[flat.season == season].sort_values("total", ascending=False)
    cols = ["player_name", "rating_off", "rating_def", "total", "alpha_good_off", "alpha_good_def",
            "alpha_total", "corrected", "poss_off_off", "without_poss_off"]
    print(f"\n=== {season}, top 20 by the rating, with the trade set's correction beside it "
          "(points per 100, positive good)")
    print(part[cols].head(20).to_string(index=False))
    # The same filter the loss uses: one team all season, and games his team played without him.  A player whose
    # team never took the court without him has an alpha the other columns implied, not one his own absence earned.
    seen = part[part.eligible & (part.without_poss_off > 0) & (part.without_poss_def > 0)]
    big = seen.reindex(seen.alpha_total.abs().sort_values(ascending=False).index)
    print(f"\n=== {season}, the 20 largest corrections among players whose team played without them")
    print(big[cols].head(20).to_string(index=False))


def print_loss(loss: pd.DataFrame, one_sided: list) -> None:
    """The trade loss pooled over seasons, by side and possession tier."""
    print("\n=== the trade loss: how far a season's ratings are from what the absences say, "
          "every player counted once")
    if loss.empty:
        print("    no player has a with-and-without contrast in this run, so there is no loss")
        print("    to read.  That is what --block=0 means: one season on its own has no")
        print("    neighbouring games in which a player's team played without him.")
        return
    print("    missed: the typical player's miss in points per 100.  missed_eff: the same, weighting a")
    print("    player by how much with-and-without evidence he has.  points: the season's total misprice.")
    pooled = loss.groupby(["side", "tier"], as_index=False).agg(
        seasons=("season", "nunique"), players=("players", "mean"), missed=("missed", "mean"),
        missed_eff=("missed_eff", "mean"), points=("points", "mean"))
    print(pooled.round(3).to_string(index=False))
    if one_sided:
        kept = loss[~loss.season.isin(one_sided)]
        both = kept.groupby(["side", "tier"], as_index=False).missed.mean()
        print(f"\n    dropping the {len(one_sided)} one-sided seasons: "
              + ", ".join(f"{r.side} {r.tier} {r.missed:.3f}" for r in both.itertuples() if r.tier == "all"))


def report_tables(alpha: pd.DataFrame, loss: pd.DataFrame, one_sided: list) -> None:
    """The latest season's and 2015's tables with alpha beside the rating, then the trade loss."""
    with pd.option_context("display.width", 220, "display.max_columns", 30, "display.precision", 3):
        flat = flat_table(alpha)
        seasons = [s for s in (flat.season.max(), 2015) if s in set(flat.season)]
        for season in dict.fromkeys(seasons):
            print_season_tables(flat, season)
        print_loss(loss, one_sided)


def trade_set_run(args: argparse.Namespace) -> None:
    """Fit every rated season's block, choose the penalty, write the tables, print the report."""
    cfg = load_config()
    run = settings(args, cfg)
    raw = pd.read_parquet(run.rankings)
    names = raw[["player_id", "player_name"]].drop_duplicates("player_id")
    table = load_ratings(run.rankings, "rating", possessions=True)
    roles = pd.read_parquet(roles_path(cfg))
    print_header(run, table)
    fits, grid_rows, one_sided = fit_seasons(run, table, roles, DesignCache(cfg, keep=2), cfg)
    if one_sided:
        print(f"\none-sided blocks (no season on one side): {', '.join(str(s) for s in one_sided)}")
    chosen = choose_penalty(grid_rows, run)
    alpha = alpha_table(fits, chosen, raw, roles, one_sided, run.team_effects, names)
    write_tables(alpha, names, run, chosen)
    if run.free_scale:
        print_rating_scale(alpha)
    loss = trade_set.trade_loss(alpha)
    loss.to_parquet(run.outputs / f"{run.tag}_loss.parquet", index=False)
    report_tables(alpha, loss, one_sided)


def main(argv=None) -> None:
    use_utf8()
    args = parse_args(argv)
    # one thread also because twelve threads per small solve thrash
    with one_thread():
        trade_set_run(args)


if __name__ == "__main__":
    main()
