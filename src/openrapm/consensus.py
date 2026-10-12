"""The consensus check: the ratings against a consensus of modern all-in-one metrics, read and never fitted to.

The consensus (data/external/consensus.csv, read by `data.tables.load_consensus`) blends several public metrics
for 2024-26.  It shares none of the ratings' margin data, so it catches an error the ratings' own year-over-year
test cannot: a defect both seasons of a pair share.  The owner's standing rule (notes/DECISIONS.md): the consensus is a
sanity check, never a fitting target; a marginal miss is not a veto, a gross one still is.
tests/test_vs_consensus.py holds the ratings to the minimums below; scripts/checks/64_consensus_report.py prints the
same numbers for several tables side by side, and scripts/checks/135_robust_review.py bootstraps their change.

Words used below
    match key     a player's name as the join reads it (`match_key`): accents dropped, lower case, no
                  punctuation, no suffix.  Neither side carries a player id, so the join is by name
    pooled        each player's ratings over the three consensus seasons, averaged with his offensive
                  possessions as weights, because the minimums were set on three-season ratings
    matched       the players on both sides with at least MIN_POSSESSIONS pooled possessions; a name two players
                  share goes to the one with more possessions
    rank agreement  Spearman's rank correlation between the ratings and the consensus, per side
    spread        the ratings' standard deviation over the consensus's, per side: 1 = the same width

Example: a player rated +3.0 on offence over 2,000 possessions in 2025 and +1.0 over 1,000 in 2026 is pooled at
(2,000 x 3.0 + 1,000 x 1.0) / 3,000 = +2.33.
"""
from __future__ import annotations

import re
import unicodedata

import pandas as pd
from scipy.stats import spearmanr

__all__ = ["SEASONS", "MIN_POSSESSIONS", "CONSENSUS_COLUMNS", "SIDES", "MIN_RANK_AGREEMENT", "OFFENCE_SPREAD",
           "MAX_DEFENCE_SPREAD", "MIN_TOP_FIVE_SHARED", "match_key", "pooled_over_seasons", "consensus_by_key",
           "matched", "with_ranks", "team_r2", "agreement"]

SEASONS = (2024, 2025, 2026)        # what the consensus snapshot covers
MIN_POSSESSIONS = 1000              # pooled offensive possessions a player needs to be compared
CONSENSUS_COLUMNS = ("player_name", "team", "adj_offense", "adj_defense", "adj_overall")
SIDES = (("total", "adj_overall"), ("off", "adj_offense"), ("def", "adj_defense"))   # rating_<side> -> consensus
SUFFIX = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b")

# The minimums tests/test_vs_consensus.py holds the shipped ratings to; the reason for each is in that file.
MIN_RANK_AGREEMENT = 0.75           # Spearman, offence, defence and total
OFFENCE_SPREAD = (0.55, 1.3)        # the ratings' offensive standard deviation over the consensus's
MAX_DEFENCE_SPREAD = 1.4            # the same on defence
MIN_TOP_FIVE_SHARED = 3             # players in both top fives by total


def match_key(name) -> str:
    """A name as the consensus join reads it.

    Example: "Nikola Jokić" gives "nikola jokic", "Gary Trent Jr." gives "gary trent".
    """
    text = unicodedata.normalize("NFKD", str(name))
    text = "".join(c for c in text if not unicodedata.combining(c)).lower()
    text = text.replace("&", " ").replace("-", " ").replace("'", "").replace(".", " ")
    return re.sub(r"\s+", " ", SUFFIX.sub(" ", text)).strip()


def pooled_over_seasons(ratings: pd.DataFrame) -> pd.DataFrame:
    """One row per named player over SEASONS: each rating's possession-weighted mean, his possessions
    (`poss_off`) and the name he went by in the latest of those seasons."""
    rows = ratings[ratings.season.isin(SEASONS) & ratings.player_name.notna()].copy()
    weights = rows["poss_off"].to_numpy(dtype=float)
    out = {}
    for column in ("rating_total", "rating_off", "rating_def"):
        rows["_wv"] = rows[column].to_numpy(dtype=float) * weights
        out[column] = rows.groupby("player_id")["_wv"].sum() / rows.groupby("player_id")["poss_off"].sum()
    pooled = pd.DataFrame(out)
    pooled["poss_off"] = rows.groupby("player_id")["poss_off"].sum()
    latest = rows.sort_values("season").drop_duplicates("player_id", keep="last").set_index("player_id")
    pooled["player_name"] = latest["player_name"]
    return pooled.reset_index()


def consensus_by_key(consensus: pd.DataFrame) -> pd.DataFrame:
    """The consensus's CONSENSUS_COLUMNS with a `key`, one row per match key (the first of a repeated name)."""
    con = consensus[list(CONSENSUS_COLUMNS)].dropna(subset=["player_name", "adj_overall"])
    con["key"] = con.player_name.map(match_key)
    return con.drop_duplicates("key")


def matched(ratings: pd.DataFrame, consensus: pd.DataFrame, min_possessions: float = MIN_POSSESSIONS) -> pd.DataFrame:
    """The pooled ratings joined to the consensus by match key, players with `min_possessions` or more.

    `ratings` is a ratings table (player_id, season, player_name, rating_off, rating_def, rating_total,
    poss_off); `consensus` is the file as `data.tables.load_consensus` reads it.
    """
    pooled = pooled_over_seasons(ratings)
    pooled["key"] = pooled.player_name.map(match_key)
    pooled = pooled.sort_values("poss_off").drop_duplicates("key", keep="last")
    joined = pooled.merge(consensus_by_key(consensus), on="key", how="inner")
    return joined[joined.poss_off >= min_possessions].copy()


def with_ranks(joined: pd.DataFrame) -> pd.DataFrame:
    """The `matched` players with, per side, the ratings' rank and the consensus's (1 = best) and the gap between
    the two standardised ratings: rk_ours_<side>, rk_con_<side>, gap_<side> for side total, off and def."""
    joined = joined.copy()
    for side, consensus_column in SIDES:
        rating = joined[f"rating_{side}"]
        consensus = joined[consensus_column]
        joined[f"rk_ours_{side}"] = rating.rank(ascending=False)
        joined[f"rk_con_{side}"] = consensus.rank(ascending=False)
        rating_standardised = (rating - rating.mean()) / rating.std()
        consensus_standardised = (consensus - consensus.mean()) / consensus.std()
        joined[f"gap_{side}"] = rating_standardised - consensus_standardised
    return joined


def team_r2(frame: pd.DataFrame, column: str) -> float:
    """Share of `column`'s variance explained by which team the player is on (one-way ANOVA R-squared), on the
    consensus's `team` column."""
    values = frame[column].to_numpy(float)
    team_mean = frame.groupby("team")[column].transform("mean").to_numpy(float)
    return float(1.0 - ((values - team_mean) ** 2).sum() / ((values - values.mean()) ** 2).sum())


def agreement(ranked: pd.DataFrame) -> dict:
    """The consensus numbers of one table's matched, ranked players (`with_ranks(matched(...))`).

      n               players matched
      rho_*           rank agreement per side
      spread_*        the ratings' standard deviation over the consensus's, offence and defence
      top5            players in both top fives by total
      def_share       defence's share of the ratings' offence-plus-defence variance
      team_r2_*       how much of a player's rating his team alone explains (`team_r2`)
    """
    return dict(
        n=len(ranked),
        rho_off=spearmanr(ranked.rating_off, ranked.adj_offense).statistic,
        rho_def=spearmanr(ranked.rating_def, ranked.adj_defense).statistic,
        rho_total=spearmanr(ranked.rating_total, ranked.adj_overall).statistic,
        spread_off=ranked.rating_off.std() / ranked.adj_offense.std(),
        spread_def=ranked.rating_def.std() / ranked.adj_defense.std(),
        top5=len(set(ranked.nsmallest(5, "rk_ours_total").key) & set(ranked.nsmallest(5, "rk_con_total").key)),
        def_share=ranked.rating_def.var() / (ranked.rating_off.var() + ranked.rating_def.var()),
        team_r2_def=team_r2(ranked, "rating_def"), team_r2_off=team_r2(ranked, "rating_off"))
