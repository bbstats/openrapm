"""How closely does each ratings table agree with the consensus of public all-in-one metrics?  Reported, never
selected on.

tests/test_vs_consensus.py holds the shipped ratings to minimums and prints a figure only when one FAILS, so it
cannot compare two tables that both pass.  This check runs the same join (`openrapm.consensus`: the name key, the
three-season pooling, the 1,000-possession minimum) and prints the same numbers for every table named, the
passing ones included.  The owner's standing rule (notes/DECISIONS.md): the consensus is a sanity check, never a fitting
target; a marginal miss is not a veto, a gross one still is.  Nothing here chooses anything.

Reads
    each table named on the command line (default outputs/season_ratings_product.parquet, the published table):
    player_id, season, player_name, rating_off, rating_def, rating_total, poss_off; paths are relative to the
    current folder
    data/external/consensus.csv (`openrapm.data.tables.load_consensus`)
Writes
    nothing; prints one row per table and a last row for the consensus itself

How to read the numbers
    n            players on both sides with 1,000+ offensive possessions over 2024-26
    rho_*        Spearman's rank correlation between the table's rating and the consensus's, per side
    spread_*     the table's standard deviation over the consensus's: 1 = the same width
    top5         players in both top fives by total rating
    def_share    defence's share of the offence-plus-defence rating variance
    team_r2_*    how much of a player's rating his team alone explains (one-way ANOVA R-squared)
Example: the incumbent (outputs/season_ratings_weightfix.parquet) reads n 475, rho_total 0.805 and spread_off
0.778: it orders the 475 matched players much as the consensus does, with an offensive spread 22% narrower.

    python scripts/checks/64_consensus_report.py \\
        outputs/season_ratings_new.parquet outputs/season_ratings_weightfix.parquet
"""
import argparse
from pathlib import Path

import pandas as pd

from openrapm.commands.console import use_utf8
from openrapm.consensus import (
    MAX_DEFENCE_SPREAD,
    MIN_RANK_AGREEMENT,
    MIN_TOP_FIVE_SHARED,
    OFFENCE_SPREAD,
    agreement,
    matched,
    team_r2,
    with_ranks,
)
from openrapm.data.config import load_config, resolve
from openrapm.data.tables import load_consensus

HEADER = (
    "\n=== against data/external/consensus.csv, 2024-26, 1,000+ possessions",
    f"    tests: rho_* >= {MIN_RANK_AGREEMENT}, spread_off in [{OFFENCE_SPREAD[0]}, {OFFENCE_SPREAD[1]}], "
    f"spread_def <= {MAX_DEFENCE_SPREAD}, top5 >= {MIN_TOP_FIVE_SHARED}",
    "    def_share: defence's share of off+def rating variance.  team_r2_*: how much of a player's rating",
    "    you can guess from his team alone (one-way ANOVA R-squared on the consensus's team column).")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="The consensus numbers for several ratings tables side by side.",
                                     allow_abbrev=False)
    parser.add_argument("tables", nargs="*", type=Path,
                        help="ratings tables (default outputs/season_ratings_product.parquet)")
    return parser.parse_args(argv)


def consensus_row(joined: pd.DataFrame) -> dict:
    """The consensus itself on the last table's matched players: every agreement 1, its own shares."""
    return dict(rankings="consensus itself", n=len(joined), rho_off=1, rho_def=1, rho_total=1, spread_off=1,
                spread_def=1, top5=5,
                def_share=joined.adj_defense.var() / (joined.adj_offense.var() + joined.adj_defense.var()),
                team_r2_def=team_r2(joined, "adj_defense"), team_r2_off=team_r2(joined, "adj_offense"))


def report_rows(paths: list, consensus: pd.DataFrame) -> list[dict]:
    """One row of consensus numbers per table, named by its file, then the consensus's own row."""
    rows, joined = [], None
    for path in paths:
        joined = with_ranks(matched(pd.read_parquet(path), consensus))
        rows.append(dict(rankings=path.stem, **agreement(joined)))
    rows.append(consensus_row(joined))
    return rows


def main() -> None:
    use_utf8()
    cfg = load_config()
    paths = parse_args().tables or [resolve(cfg, "outputs") / "season_ratings_product.parquet"]
    rows = report_rows(paths, load_consensus(cfg))
    with pd.option_context("display.width", 200, "display.max_columns", 20, "display.precision", 3):
        print("\n".join(HEADER))
        print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
