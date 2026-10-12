"""Check 73, the trade loss: does one ratings table leave smaller trade-set corrections than another, same players?

Check 70 (the trade set) gives every player-season a correction, alpha: what his team's games with and without him,
in the seasons either side, say his rating did not already know, in points per 100 (positive = better than rated).
This check compares two or more tables by the size of those corrections:

    trade loss = sqrt(2 / pi) x sqrt(mean over players of alpha^2)

in points per 100, every player counted once (`trade_set.missed`).  Smaller means the rating already knew more.

**Why it exists.**  The year-over-year test (check 63) is an error per team-game, so a 200-possession bench player is
a rounding error in it.  The trade loss counts him the same as a starter.  It is an instrument and never a fitting
target: alpha reads the seasons either side of the rated one, so nothing scored here can become a published rating
(ruling 1).

Words used below
    arm         one table's corrections: a check 70 <out>_alpha.parquet, named on the command line
    eligible    a player-season the loss reads: one team all season, and at least --min_without possessions of that
                team playing without him in the neighbouring seasons (`trade_set.eligible_corrections`)
    tier        a group of players read on its own.  By default the rated season's possessions (under 500, 500 to
                1,500, over 1,500).  With --quality, the player's rank by total rating within his season in that
                table (top 30, 31-90, 91-150, 151-300, 301+; "no rating" when the table has no row for him).  With
                --movers=1, whether his main team (most box-score minutes, as a franchise) changed between the rated
                season and the season before or after ("changed teams", "same team", "no neighbour season")
    paired      per season, the candidate's loss minus the reference's, on the players both arms have

**The two things the check gets right.**
1. The same players.  Eligibility depends on the ratings covering the player, so two arms can differ in who has a row
   as well as in how well they are rated.  Every comparison is on the player-season-sides both arms have, and the
   number dropped from each arm is printed.
2. Paired by season.  Thirty seasons per side, candidate minus reference, with the standard error of the mean
   difference.  The level of one arm alone means nothing: each arm's penalty is chosen by its own grid, and the
   corrections of neighbouring seasons share games.

What it cannot tell you: alpha is a three-season average deviation from a one-season rating, so it mixes "the rating
missed something" with "he was different the next year".  That confound is the same for both arms and cancels in the
difference; it does not cancel in the level.

Reads the --alphas tables (and the --quality table, or the box scores for --movers=1).  Writes nothing.

How to read the table: cand and ref are each arm's loss, averaged over seasons; mean_diff = cand - ref, below zero
means the candidate leaves a SMALLER correction and is better; se is its standard error; z = mean_diff / se; wins
counts the seasons the candidate is better.  Example: corrections of +1.0 and -3.0 points per 100 give a loss of
0.798 x sqrt((1 + 9) / 2) = 1.78.  The noise control noise_regroup2 (the incumbent with its prior's training players
regrouped at random) reads, on offence, cand 0.2942 against ref 0.2952, mean_diff -0.0010, z -2.21, 21 of 30 seasons:
a regrouping alone can reach z -2, so read a candidate beside its noise control (check 135).

    python scripts/checks/73_tradeloss.py \\
        --alphas=incumbent=outputs/tradeset_weightfix_alpha.parquet,new=outputs/tradeset_new_alpha.parquet \\
        --ref=incumbent --quality=outputs/season_ratings_weightfix.parquet --tier=each
"""
import argparse

import numpy as np
import pandas as pd

from openrapm.commands.console import use_utf8
from openrapm.data.box_scores import main_team_by_minutes
from openrapm.data.config import load_config, resolve_path
from openrapm.data.roles import franchise
from openrapm.scoring import QUALITY_LABELS, quality_tiers
from openrapm.trade_set import SIDES, eligible_corrections, missed

MOVER_LABELS = ("changed teams", "same team", "no neighbour season")
NO_RATING = "no rating"
OFF = ("0", "no", "false")


def on_off(text: str) -> bool:
    """A switch as typed: 0, no or false is off, anything else on."""
    return str(text) not in OFF


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare rankings by the trade loss, paired by season.",
                                     allow_abbrev=False)
    parser.add_argument("--alphas", default="",
                        help="name=path,... : each a scripts/checks/70_tradeset.py <out>_alpha.parquet")
    parser.add_argument("--min_without", default=1.0, type=float,
                        help="possessions of absence a correction needs (default %(default)s)")
    parser.add_argument("--tier", default="all", help="a tier's name, all, or each (default %(default)s)")
    parser.add_argument("--quality", help="rankings whose per-season rank sets the quality tiers")
    parser.add_argument("--movers", default="0", type=on_off,
                        help="1: tiers are changed teams / same team (default %(default)s)")
    parser.add_argument("--ref", help="the reference arm (default the first named)")
    return parser.parse_args(argv)


# ------------------------------------------------------------------------------------------------ the tiers
def franchise_by_player(season: int, cfg: dict) -> dict:
    """player_id -> the franchise of his main team in `season` (most box-score minutes); {} for a season the box
    scores cannot supply."""
    try:
        main_team = main_team_by_minutes(season, cfg)
    except Exception:          # a season outside the data: its players have no team, which is "no neighbour season"
        main_team = {}
    players = np.fromiter(main_team.keys(), dtype=np.int64, count=len(main_team))
    franchises = franchise(np.fromiter(main_team.values(), dtype=np.int64, count=len(main_team)),
                           np.full(len(main_team), season))
    return dict(zip(players.tolist(), franchises.tolist()))


def mover_label(now: int, neighbours: list) -> str:
    """"changed teams" if a neighbouring season's franchise differs from `now`, "same team" if none does, and "no
    neighbour season" if he played in neither neighbouring season."""
    if not neighbours:
        return MOVER_LABELS[2]
    return MOVER_LABELS[0] if any(other != now for other in neighbours) else MOVER_LABELS[1]


def mover_tiers(seasons, cfg: dict) -> pd.Series:
    """(player_id, season) -> whether he changed teams around the rated season.

    The trade set's correction reads exactly the rated season and its two neighbours, so the owner (2026-10-01) asked
    for every result split into traded and not traded.
    """
    rated = sorted(set(int(s) for s in seasons))
    span = sorted(set(rated) | {s - 1 for s in rated} | {s + 1 for s in rated})
    teams = {season: franchise_by_player(season, cfg) for season in span}
    keys, labels = [], []
    for season in rated:
        for player, now in teams[season].items():
            neighbours = [teams[n][player] for n in (season - 1, season + 1) if player in teams.get(n, {})]
            keys.append((player, season))
            labels.append(mover_label(now, neighbours))
    return pd.Series(labels, index=pd.MultiIndex.from_tuples(keys))


def player_tiers(args: argparse.Namespace, cfg: dict) -> pd.Series | None:
    """(player_id, season) -> tier from --quality or --movers=1; None keeps the possession tiers."""
    if args.movers and args.quality:
        raise SystemExit("--movers=1 and --quality= each set the tiers; pick one")
    if args.movers:
        return mover_tiers(range(int(cfg["first_season"]), int(cfg["last_season"]) + 1), cfg)
    if args.quality:
        return quality_tiers(pd.read_parquet(resolve_path(cfg, args.quality),
                                             columns=["player_id", "season", "rating_total"]))
    return None


def tier_order(args: argparse.Namespace, quality: pd.Series | None, rows: pd.DataFrame) -> list:
    """The order --tier=each reads the tiers in."""
    if args.movers:
        return [*MOVER_LABELS, NO_RATING]
    if quality is not None:
        return [*QUALITY_LABELS, NO_RATING]
    return sorted(set(rows.tier))


# ------------------------------------------------------------------------------------------- the comparison
def read_arms(spec: str, cfg: dict, min_without: float, quality: pd.Series | None) -> dict:
    """{name: that arm's eligible corrections} from --alphas=name=path,..."""
    arms = {}
    for part in spec.replace("\n", "").split(","):
        if not part.strip():
            continue
        name, _, path = part.strip().partition("=")
        if not path:
            raise SystemExit(f"--alphas wants name=path pairs; got {part!r}")
        arms[name] = eligible_corrections(pd.read_parquet(resolve_path(cfg, path)), min_without, quality)
    return arms


def season_losses(cand: pd.DataFrame, ref: pd.DataFrame) -> tuple:
    """(one row per season: its loss for each arm on the players both have, the player-seasons compared)."""
    rows, n_players = [], 0
    for season in sorted(set(cand.index.get_level_values("season"))):
        cand_season, ref_season = cand.xs(season, level="season"), ref.xs(season, level="season")
        both = cand_season.index.intersection(ref_season.index)
        if len(both) == 0:
            continue
        n_players += len(both)
        rows.append((season, missed(cand_season.loc[both].alpha_good.to_numpy()),
                     missed(ref_season.loc[both].alpha_good.to_numpy())))
    return pd.DataFrame(rows, columns=["season", "cand", "ref"]), n_players


def paired_side(cand: pd.DataFrame, ref: pd.DataFrame, side: str, tier: str) -> dict:
    """One side: both losses, the paired per-season difference, its standard error and z, and the seasons won."""
    cand, ref = cand.xs(side, level="side"), ref.xs(side, level="side")
    if tier != "all":
        cand, ref = cand[cand.tier == tier], ref[ref.tier == tier]
    frame, n_players = season_losses(cand, ref)
    diff = (frame.cand - frame.ref).to_numpy()
    se = float(np.std(diff, ddof=1) / np.sqrt(len(diff))) if len(diff) > 1 else np.nan
    return {"side": side, "seasons": len(frame), "players": n_players,
            "cand": float(frame.cand.mean()), "ref": float(frame.ref.mean()),
            "mean_diff": float(np.mean(diff)), "se": se,
            "z": float(np.mean(diff) / se) if se and np.isfinite(se) and se > 0 else np.nan,
            "wins": int((diff < 0).sum())}


def paired(cand: pd.DataFrame, ref: pd.DataFrame, tier: str) -> pd.DataFrame:
    """Per side: both losses, the paired per-season difference, its z, and the seasons won."""
    return pd.DataFrame([paired_side(cand, ref, side, tier) for side in SIDES])


def four_places(value: float) -> str:
    """A number to four decimal places, as the tables print it."""
    return f"{value:.4f}"


def print_arms(arms: dict, tier: str, min_without: float) -> None:
    """The header: the tier and the minimum absence, then each arm's eligible rows, seasons and alpha penalties."""
    print(f"the trade loss, points per 100, every player counting once, tier {tier!r}, "
          f"corrections on at least {min_without:g} possession(s) of absence")
    for name, rows in arms.items():
        penalties = sorted(rows.penalty.unique())
        print(f"  {name}: {len(rows):,} eligible player-season-sides, "
              f"{rows.index.get_level_values('season').nunique()} seasons, "
              f"alpha penalty {'/'.join(f'{p:,.0f}' for p in penalties[:6])}"
              f"{' ...' if len(penalties) > 6 else ''}")


def compare(name: str, rows: pd.DataFrame, ref_name: str, ref: pd.DataFrame, tier: str, order: list) -> None:
    """One arm against the reference on the player-season-sides both have, in one tier or in each."""
    shared = rows.index.intersection(ref.index)
    print(f"\n=== {name} against {ref_name} ===")
    print(f"  {len(shared):,} player-season-sides in both; dropped {len(rows) - len(shared):,} "
          f"from {name} and {len(ref) - len(shared):,} from {ref_name}")
    for each in ([t for t in order if t in set(rows.tier)] if tier == "each" else [tier]):
        table = paired(rows.loc[shared], ref.loc[shared], each)
        if tier == "each":
            print(f"  -- tier {each}")
        print(table.to_string(index=False, float_format=four_places))
    print("  mean_diff below zero means the rating leaves a SMALLER correction, so it is better; "
          "z is that difference over its own standard error, wins is out of `seasons`")


def main(argv=None) -> None:
    use_utf8()
    args = parse_args(argv)
    if not args.alphas:
        raise SystemExit(__doc__)
    cfg = load_config()
    quality = player_tiers(args, cfg)
    arms = read_arms(args.alphas, cfg, args.min_without, quality)
    ref_name = args.ref or next(iter(arms))
    if ref_name not in arms:
        raise SystemExit(f"--ref={ref_name} is not one of {', '.join(arms)}")
    print_arms(arms, args.tier, args.min_without)
    for name, rows in arms.items():
        if name != ref_name:
            compare(name, rows, ref_name, arms[ref_name], args.tier, tier_order(args, quality, rows))


if __name__ == "__main__":
    main()
