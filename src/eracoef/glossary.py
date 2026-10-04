"""Plain names for the prior's columns, the same as GLOSSARY.md: reports to the owner use these, never the codes.

The owner, 2026-10-03: "A big problem right now is that I don't understand the names of the features you're
sharing with me."  `plain(code)` returns the name a report should print; an unknown code comes back unchanged,
so a missing entry shows up as a code in the output rather than as a wrong name.
"""
from __future__ import annotations

__all__ = ["PLAIN", "plain"]

PLAIN = {
    # box-score rates, per 100 possessions
    "fg3m": "made threes", "fg3_miss": "missed threes", "fg2m": "made twos", "fg2_miss": "missed twos",
    "ftm": "made free throws", "ft_miss": "missed free throws", "orb": "offensive rebounds",
    "drb": "defensive rebounds", "ast": "assists", "tov": "turnovers", "stl": "steals", "blk": "blocks",
    "pf": "personal fouls",
    # combinations, per 100
    "pts": "points", "fga": "shot attempts", "fta": "free-throw attempts", "fg3a": "three-point attempts",
    "usage": "possessions he finishes", "reb": "rebounds", "stocks": "steals plus blocks",
    "creation": "assists minus turnovers", "shotmix": "threes minus twos taken",
    "bigness": "'plays like a centre' score",
    # percentages and shares
    "efg": "effective FG %", "ts": "true shooting %", "fg3p": "three-point %", "fg2p": "two-point %",
    "ftp": "free-throw %", "p3r": "share of shots that are threes", "ftr": "free throws per shot",
    "astr": "assists per possession used", "tovr": "turnovers per possession used",
    "orbsh": "offensive-rebound share of his rebounds",
    # shot quality
    "q2": "difficulty of his twos", "q3": "difficulty of his threes", "m2": "two-point shot-making",
    "m3": "three-point shot-making", "xps": "expected points per shot", "mpts": "points per shot above expected",
    # role, body, career
    "poss_pct": "share of team possessions played", "gs_pct": "share of games started", "age": "age",
    "height": "height", "weight": "weight", "draft_pick": "draft position",
    "exp_yrs": "seasons played before", "exp_poss": "career possessions before",
    "entry_age": "age entering the league", "tenure": "seasons with current team", "n_teams": "teams played for",
    # his team's results with him on and off the court
    "onc_o": "team points scored per 100, him on court", "onc_d": "team points allowed per 100, him on court",
    "onc_poss_o": "offensive possessions on court", "onc_poss_d": "defensive possessions on court",
    "offc_o": "team points scored per 100, him off court", "offc_d": "team points allowed per 100, him off court",
    "offc_poss_o": "offensive possessions off court", "offc_poss_d": "defensive possessions off court",
    "net_o": "on/off, points scored", "net_d": "on/off, points allowed",
    # score state and playoffs
    "gt_share": "share of possessions in garbage time", "closeness": "how close his games were",
    "abs_margin": "average score gap while he played", "po_share": "share of possessions in the playoffs",
    # how much evidence a training row rests on
    "chunk_poss": "possessions behind the row", "chunk_seasons": "seasons the row covers",
}


def plain(code: str) -> str:
    """The plain name for a column code, or the code itself when there is none."""
    return PLAIN.get(code, code)
