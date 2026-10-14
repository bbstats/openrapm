"""Check 144, the team-season flags' penalties and how far the flags move step 1's labels (experiment 60).

The owner, 2026-10-14: team-season flags for offence and defence in the RAPM fits, placed in step 1 (the labels the
box-score prior learns from), where players changing teams between seasons tell a flag apart from its players.
This check runs before the build: it sets the two penalties and says whether the labels move enough for the build to
be worth running.

Words used below
    flag            one column per team-season and side in step 1's fit (rapm.py, "team-season flags"): the attacking
                    flag is 1 on the rows that team is on offence, the defending flag 1 on the rows it is on defence
    penalty         one per flag block, estimated by empirical Bayes on every trainable season at once and never tuned
                    on the year-over-year test.  The ridge is the posterior mean of the model
                    y_row ~ N(x theta, s2 / w_row),  coefficient ~ N(0, s2 / its block's penalty),  so EM alternates
                        s2       = the rows' weighted squared misses / the number of rows
                        var      = mean over the block's columns of (theta^2 + s2 x its diagonal of (G + Lambda)^-1)
                        penalty  = s2 / var
                    until the penalties move less than 0.1%.  The attacking penalty is read off the offence target's
                    fit, the defending one off the defence target's (each fit estimates both blocks; all printed).
                    Two rules (--rule):
                      flags   EM on the two flag blocks only, the players held at 40,000.  The first run (2026-10-14)
                              gave 1,530 / 1,360: the players' 40,000 allows them a spread of 0.55 per 100, far less
                              than they have, so the flags took the stars' credit (Nash's offensive label -4.5)
                      joint   EM on all four blocks (offence players, defence players, both flag blocks); each flag
                              penalty is then its estimate x 40,000 / its side's player estimate, which keeps the
                              split between flags and players the joint estimate found, at the shipped player penalty
    label move      a player's label with the flags minus without, at the estimated penalties, for one rated season
                    (default 2026, product setting: every other trainable season in the labels).  "In rating points" is
                    the move times that season's prior scale in the product step-3 table, the most of it the prior
                    could pass on; positive = better on both sides

Reads the design pieces (data/cache), the stint diagnostics (each row's teams), the game logs (team abbreviations),
outputs/season_ratings_product_step3.parquet (the prior scales).  Writes outputs/team_season_penalties.json,
outputs/csv/team_season_flags.csv (every team-season's two flags, positive = better) and
outputs/csv/team_season_label_moves_<season>.csv (every labelled player's move), and prints the summary.

    python scripts/checks/144_team_season_labels.py                (about 15-25 minutes)
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from scipy.linalg.lapack import dpotri as potri
from threadpoolctl import threadpool_limits

from openrapm import rapm
from openrapm.commands.console import use_utf8
from openrapm.data.config import load_config, resolve
from openrapm.data.designs import DesignCache
from openrapm.data.ingest import raw_dir
from openrapm.data.seasons import trainable
from openrapm.data.tables import design_row_teams, player_names
from openrapm.settings import PRODUCT

START = 40000.0                    # each block's first penalty: the players'
TOLERANCE = 1e-3
MAX_ROUNDS = 60
ONE_TEAM = {1495: "Tim Duncan", 1717: "Dirk Nowitzki", 201939: "Stephen Curry", 977: "Kobe Bryant",
            397: "Reggie Miller", 1938: "Manu Ginobili", 2225: "Tony Parker", 203110: "Draymond Green",
            2617: "Udonis Haslem", 203497: "Rudy Gobert"}
SIDES = (("O", "offense", "offence"), ("D", "defense", "defence"))


# ------------------------------------------------------------------------------------------ step 1, both ways
def season_sums(cfg: dict, seasons: list) -> tuple[dict, dict, dict]:
    """({target: plain sums}, {target: sums with flags}, {target: (sum of w y^2, rows)}), one pass over the designs."""
    designs = DesignCache(cfg)
    plain, flagged, moments = {}, {}, {}
    started = time.time()
    for target in PRODUCT.targets.distinct():
        plain[target] = rapm.SeasonSums(PRODUCT.labels.min_possessions)
        flagged[target] = rapm.SeasonSums(PRODUCT.labels.min_possessions)
        squares, rows = 0.0, 0
        for season in sorted(seasons):
            design = designs.season(season, target)
            plain[target].add(season, design)
            flagged[target].add(season, design, design_row_teams(design, cfg))
            squares += float(np.sum(design.w * design.y ** 2))
            rows += len(design.y)
        moments[target] = (squares, rows)
        print(f"  step 1 on {target}: {len(seasons)} seasons, {len(plain[target].player_ids)} players "
              f"({time.time() - started:.0f}s)", flush=True)
    return plain, flagged, moments


# ------------------------------------------------------------------------------------------ the penalties
BLOCKS = ("offence players", "defence players", "attacking flags", "defending flags")


def block_columns(sums: rapm.SeasonSums) -> dict:
    """{block: its columns in `assemble(())`}."""
    n_players = len(sums.player_ids)
    flags = sums.flag_columns()
    return {"offence players": np.arange(n_players), "defence players": n_players + np.arange(n_players),
            "attacking flags": flags.attacking.to_numpy(), "defending flags": flags.defending.to_numpy()}


def block_variances(sums: rapm.SeasonSums, penalties: dict, moments: tuple) -> tuple:
    """(theta, s2, {block: variance}) at one penalty per block, on every season: each block's mean of
    theta^2 + s2 x its diagonal of (G + Lambda)^-1, the inverse from the Cholesky factor (LAPACK potri)."""
    gram, rhs, _, _ = sums.assemble(())
    penalty = sums.penalty((), (penalties["offence players"], penalties["defence players"], 0.0),
                           (penalties["attacking flags"], penalties["defending flags"]))
    live = np.flatnonzero((np.diag(gram) > 0.0) | (penalty > 0.0))      # an empty column (2020's playoff home court)
    factor, lower = cho_factor(gram[np.ix_(live, live)] + np.diag(penalty[live]), lower=True, check_finite=False)
    theta = np.zeros(len(rhs))
    theta[live] = cho_solve((factor, lower), rhs[live], check_finite=False)
    inverse, info = potri(factor, lower=True)
    if info != 0:
        raise SystemExit(f"the inverse failed (LAPACK info {info})")
    diagonal = np.zeros(len(rhs))
    diagonal[live] = np.diag(inverse)
    squares, rows = moments
    s2 = float((squares - 2.0 * theta @ rhs + theta @ gram @ theta) / rows)
    variances = {name: float(np.mean(theta[columns] ** 2 + s2 * diagonal[columns]))
                 for name, columns in block_columns(sums).items()}
    return theta, s2, variances


def estimate_penalties(sums: rapm.SeasonSums, moments: tuple, target: str, free: tuple) -> tuple:
    """({block: penalty}, theta at them, s2): EM to a fixed point on the `free` blocks, the others held at START."""
    penalties = dict.fromkeys(BLOCKS, START)
    for round_number in range(1, MAX_ROUNDS + 1):
        theta, s2, variances = block_variances(sums, penalties, moments)
        updated = {name: (s2 / variances[name] if name in free else START) for name in BLOCKS}
        print(f"  {target} round {round_number}: s2 {s2:,.0f}; "
              + "; ".join(f"{name} {updated[name]:,.0f} (sd {np.sqrt(s2 / updated[name]):.2f})" for name in free),
              flush=True)
        done = max(abs(updated[name] / penalties[name] - 1.0) for name in free) < TOLERANCE
        penalties = updated
        if done:
            break
    theta, s2, _ = block_variances(sums, penalties, moments)
    return penalties, theta, s2


FREE = {"flags": ("attacking flags", "defending flags"), "joint": BLOCKS}


def scaled_to_shipped(penalties: dict, flag_block: str, player_block: str) -> float:
    """A flag penalty with the joint estimate's ratio to its side's players kept, the players at the shipped 40,000."""
    return penalties[flag_block] / penalties[player_block] * START


def rounded(value: float) -> float:
    """Three significant figures: the setting typed on the command line."""
    return float(f"{value:.3g}")


# ------------------------------------------------------------------------------------------ the readout
def team_names(cfg: dict, seasons: list) -> pd.Series:
    """(season, team_id) -> abbreviation, from each season's regular-season game log."""
    frames = []
    for season in seasons:
        log = pd.read_parquet(raw_dir(cfg) / "gamelog" / f"{season}_RS.parquet",
                              columns=["TEAM_ID", "TEAM_ABBREVIATION"])
        frames.append(log.drop_duplicates("TEAM_ID").assign(season=season))
    table = pd.concat(frames, ignore_index=True)
    return table.set_index(["season", "TEAM_ID"]).TEAM_ABBREVIATION


def flag_table(sums: dict, thetas: dict, names: pd.Series) -> pd.DataFrame:
    """Every team-season's attacking flag (offence fit) and defending flag (defence fit), positive = better."""
    targets = PRODUCT.targets
    columns = sums[targets.offence].flag_columns()
    defending = sums[targets.defence].flag_columns()
    if not columns[["season", "team_id"]].equals(defending[["season", "team_id"]]):
        raise SystemExit("the two targets' team-seasons differ")
    table = columns[["season", "team_id"]].copy()
    table["team"] = names.reindex(pd.MultiIndex.from_frame(table[["season", "team_id"]])).to_numpy()
    table["offence_flag"] = thetas[targets.offence][columns.attacking]
    table["defence_flag"] = -thetas[targets.defence][defending.defending]
    return table


def label_moves(plain: dict, flagged: dict, penalties: tuple, season: int, scales: pd.Series,
                names: pd.DataFrame) -> pd.DataFrame:
    """Every labelled player's two labels and their moves for one rated season, product setting."""
    on = replace(PRODUCT.labels, team_offense_penalty=penalties[0], team_defense_penalty=penalties[1])
    before = rapm.career_labels(plain, [season], PRODUCT.labels, PRODUCT.targets)
    after = rapm.career_labels(flagged, [season], on, PRODUCT.targets)
    frames = []
    for side, column, word in SIDES:
        sign = 1.0 if side == "O" else -1.0                       # positive = better
        joined = before[side][["player_id", column, "possessions"]].merge(
            after[side][["player_id", column]], on="player_id", suffixes=("_before", "_after"))
        frames.append(pd.DataFrame({"player_id": joined.player_id, "side": word,
                                    "label_possessions": joined.possessions,
                                    "before": sign * joined[f"{column}_before"],
                                    "after": sign * joined[f"{column}_after"]}))
    table = pd.concat(frames, ignore_index=True)
    table["move"] = table.after - table.before
    table["move_in_rating_points"] = table.move * table.side.map(scales).to_numpy()
    return table.merge(names, on="player_id", how="left")


def print_readout(estimates: dict, chosen: tuple, rule: str, flags: pd.DataFrame, moves: pd.DataFrame,
                  season: int) -> None:
    print(f"\n=== penalties, rule {rule!r}: each target's estimate (the sd per 100 each penalty allows)")
    for target, (penalties, s2) in estimates.items():
        print(f"  {target} (s2 {s2:,.0f}): " + "; ".join(
            f"{name} {value:,.0f} ({np.sqrt(s2 / value):.2f})" for name, value in penalties.items()))
    print(f"  chosen, the players at 40,000: attacking {chosen[0]:,.0f}, defending {chosen[1]:,.0f}")
    print("\n=== team-season flags, points per 100, positive = better")
    for column in ("offence_flag", "defence_flag"):
        print(f"  {column}: sd {flags[column].std():.2f}; best and worst five")
        ordered = flags.sort_values(column, ascending=False)
        for _, row in pd.concat([ordered.head(5), ordered.tail(5)]).iterrows():
            print(f"    {row.season} {row.team}: {row[column]:+.2f}")
    print(f"\n=== labels for rating {season} (product setting), with flags minus without, positive = better")
    for word in ("offence", "defence"):
        side = moves[moves.side == word]
        regulars = side[side.label_possessions >= 5000]
        weights = side.label_possessions
        print(f"  {word}: {len(side)} players; correlation {np.corrcoef(side.before, side.after)[0, 1]:.4f}; "
              f"move sd {side.move.std():.3f} label points, {side.move_in_rating_points.std():.3f} rating points "
              f"(possession-weighted sd {np.sqrt(np.average(side.move_in_rating_points ** 2, weights=weights)):.3f}); "
              f"share of 5,000+ possession players moving 0.1+ rating points "
              f"{(regulars.move_in_rating_points.abs() >= 0.1).mean():.0%}")
        top = regulars.reindex(regulars.move_in_rating_points.abs().sort_values(ascending=False).index).head(10)
        for _, row in top.iterrows():
            print(f"    {row.player_name}: {row.before:+.2f} -> {row.after:+.2f} label, "
                  f"{row.move_in_rating_points:+.2f} rating points ({row.label_possessions:,.0f} possessions)")
    print("\n=== one-team careers, rating points")
    for player_id, name in ONE_TEAM.items():
        rows = moves[moves.player_id == player_id].set_index("side").move_in_rating_points
        if len(rows):
            print(f"  {name}: offence {rows.get('offence', np.nan):+.2f}, defence {rows.get('defence', np.nan):+.2f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--season", type=int, default=2026, help="the rated season of the label readout")
    parser.add_argument("--rule", choices=tuple(FREE), default="joint",
                        help="flags: EM on the flags alone, the players held at 40,000.  joint: EM on the players and "
                             "the flags together, each flag penalty then scaled by 40,000 / its side's player estimate "
                             "(default %(default)s)")
    parser.add_argument("--threads", type=int, default=4, help="linear-algebra threads (default %(default)s)")
    args = parser.parse_args()
    use_utf8()
    cfg = load_config()
    outputs = resolve(cfg, "outputs")
    seasons = trainable(cfg).seasons
    with threadpool_limits(limits=args.threads):
        plain, flagged, moments = season_sums(cfg, seasons)
        estimates = {}
        for target in PRODUCT.targets.distinct():
            penalties, _, s2 = estimate_penalties(flagged[target], moments[target], target, FREE[args.rule])
            estimates[target] = (penalties, s2)
        offence, defence = estimates[PRODUCT.targets.offence][0], estimates[PRODUCT.targets.defence][0]
        if args.rule == "joint":
            chosen = (rounded(scaled_to_shipped(offence, "attacking flags", "offence players")),
                      rounded(scaled_to_shipped(defence, "defending flags", "defence players")))
        else:
            chosen = (rounded(offence["attacking flags"]), rounded(defence["defending flags"]))
        labels = PRODUCT.labels
        thetas = {target: flagged[target].solve((), (labels.offense_penalty, labels.defense_penalty, 0.0), chosen)[0]
                  for target in PRODUCT.targets.distinct()}
        product = pd.read_parquet(outputs / "season_ratings_product_step3.parquet",
                                  columns=["season", "prior_scale_off", "prior_scale_def"])
        scale = product[product.season == args.season].iloc[0]
        scales = pd.Series({"offence": scale.prior_scale_off, "defence": scale.prior_scale_def})
        moves = label_moves(plain, flagged, chosen, args.season, scales, player_names(cfg))
    flags = flag_table(flagged, thetas, team_names(cfg, seasons))
    print_readout(estimates, chosen, args.rule, flags, moves, args.season)
    print(f"\nthe build's flag: --team-season-penalties {chosen[0]:g},{chosen[1]:g}")
    (outputs / "csv").mkdir(parents=True, exist_ok=True)
    flags.to_csv(outputs / "csv" / f"team_season_flags_{args.rule}.csv", index=False)
    moves.to_csv(outputs / "csv" / f"team_season_label_moves_{args.rule}_{args.season}.csv", index=False)
    summary = {"rule": args.rule, "chosen": {"attacking": chosen[0], "defending": chosen[1]},
               "estimates": {target: {**penalties, "s2": s2} for target, (penalties, s2) in estimates.items()},
               "seasons": [int(season) for season in seasons]}
    path = outputs / f"team_season_penalties_{args.rule}.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {path} and the two tables in {outputs / 'csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
