"""The box-score prior's inputs: what each column means, how the derived ones are built, and which ones ship.

The season panel (one row per player, season and side) carries a player's padded box-score rates and his role,
career, body and on-court numbers.  A tree splits one column at a time, so it can never form a ratio such as
true shooting from the two columns it is made of; the derived columns hand those over ready-made.

Words used below
    tree        one decision tree of the box-score prior's model (LightGBM, gradient-boosted trees; step 2)
    rate        a count per 100 possessions, padded before it is used: pulled toward the average of players
                with a similar number of possessions (one of six possession bins) by k possessions' worth, k
                estimated per rate and season from odd- versus even-game halves (exposure.py).  With k = 450
                and a bin average of 1.4 (illustrative values), a 200-possession bench player's 2.1 steals per
                100 become (200 x 2.1 + 450 x 1.4) / (200 + 450) = 1.6
    centred     the possession-weighted league mean subtracted, so 0 is a league-average rate; "level"
                columns keep their units
    uncentred   the same padded rate with the level kept (the `raw_` columns); a ratio needs the level back
    combination a fixed linear sum of the centred rates
    ratio       an efficiency built from the uncentred rates, its denominator padded

The two derived families:

    combination   = sum over its rates of  weight x centred rate
    ratio         = (numerator + pad x typical value) / (denominator + pad)

    numerator, denominator   weighted sums of the uncentred rates (`RATIOS` lists the weights)
    pad                      how many denominator units of the typical value are added, so a bench player's
                             ratio is not a ratio of two roundings
    typical value            roughly the league's value of the ratio (0.54 for true shooting)

Example: a player with 6.0 made twos, 2.0 made threes and 3.0 made free throws per 100 (uncentred) and the league
at 5.0, 1.5 and 2.5 has `pts` = 2 x 1.0 + 3 x 0.5 + 0.5 = 4.0 points per 100 above the league.

The codes are the OLD panel names (phase 8 of the cleanup gives every column a plain name).  The order of the
terms and the constants below are load-bearing: the shipped booster reads its inputs by position, and the
sums are written in the order the published ratings were built with.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .design import FEATURES as BOX_RATES

__all__ = ["Feature", "BOX_RATES", "RAW_RATES", "ROLE", "CLOSENESS", "SHOT_TOTALS", "SHOT_LEAGUE", "CAREER", "BIO",
           "ONC", "OFFC", "NET", "PO_SHARE", "DERIVED", "RATIOS", "SHOTQ", "SHOTQ_PADS", "PRIOR_FEATURES",
           "PANEL_INPUTS", "ROW_FEATURES", "SHIPPED", "FEATURE_SETS", "CATALOGUE", "add_features", "add_shotq",
           "glossary_markdown"]


@dataclass(frozen=True)
class Feature:
    """One column the prior can read: its code, what it means, how it is computed, its unit, centred or level."""
    name: str
    meaning: str
    formula: str
    unit: str
    scale: str          # "centred" or "level"
    group: str


# ------------------------------------------------------------------------------ the panel's own columns
RAW_RATES = [f"raw_{c}" for c in BOX_RATES]
ROLE = ["poss_pct", "gs_pct", "age"]
# the score state a player's statistics were compiled in (scripts/69_closeness_panel.py)
CLOSENESS = ["gt_share", "closeness", "abs_margin"]
# per shooter, summed over the season: attempts, makes, and the league's expected makes from his spots
SHOT_TOTALS = ["shot_fg2a", "shot_fg2m", "shot_xl2", "shot_fg3a", "shot_fg3m", "shot_xl3"]
SHOT_LEAGUE = ["shot_lg2", "shot_lg3", "shot_lgpps"]
# counted BEFORE the season: a 25-year-old rookie and a 25-year-old in year seven are different players
CAREER = ["exp_yrs", "exp_poss", "entry_age"]
BIO = ["height", "weight", "draft_pick", "tenure", "n_teams"]
ONC = ["onc_o", "onc_d", "onc_poss_o", "onc_poss_d"]          # his team's record with him on the court
OFFC = ["offc_o", "offc_d", "offc_poss_o", "offc_poss_d"]     # ... and with him off it, same games
NET = ["net_o", "net_d"]                                      # on-court minus off-court, per side
PO_SHARE = ["po_share"]

# ------------------------------------------------------------------------------ the derived columns
# Linear sums of the centred rates.  A tree splits on one column at a time (x_j < c), so it can only approximate
# a sum such as points = 2 x made twos + 3 x made threes + made free throws with many splits; the sum is handed
# over as its own column.  {code: {rate: weight}}, terms in the order they are summed.
DERIVED = {
    "pts":      {"fg2m": 2.0, "fg3m": 3.0, "ftm": 1.0},                       # points per 100
    "fga":      {"fg2m": 1.0, "fg2_miss": 1.0, "fg3m": 1.0, "fg3_miss": 1.0},
    "fta":      {"ftm": 1.0, "ft_miss": 1.0},
    "fg3a":     {"fg3m": 1.0, "fg3_miss": 1.0},
    "usage":    {"fg2m": 1.0, "fg2_miss": 1.0, "fg3m": 1.0, "fg3_miss": 1.0,   # possessions he finishes
                 "ftm": 0.44, "ft_miss": 0.44, "tov": 1.0},
    "bigness":  {"orb": 1.0, "blk": 1.0, "drb": 0.3, "ast": -0.5, "fg3m": -0.4},
    "reb":      {"orb": 1.0, "drb": 1.0},
    "stocks":   {"stl": 1.0, "blk": 1.0},
    "creation": {"ast": 1.0, "tov": -1.0},
    "shotmix":  {"fg3m": 1.0, "fg3_miss": 1.0, "fg2m": -1.0, "fg2_miss": -1.0},    # threes minus twos taken
}
# Efficiencies from the uncentred rates (a ratio of centred rates means nothing), each denominator padded so a
# 200-possession player is not a ratio of two roundings.  {code: (numerator, denominator, pad, typical value)}.
RATIOS = {
    "efg":   ({"fg2m": 1.0, "fg3m": 1.5}, {"fg2m": 1, "fg2_miss": 1, "fg3m": 1, "fg3_miss": 1}, 3.0, 0.50),
    "ts":    ({"fg2m": 2.0, "fg3m": 3.0, "ftm": 1.0},
              {"fg2m": 2, "fg2_miss": 2, "fg3m": 2, "fg3_miss": 2, "ftm": 0.88, "ft_miss": 0.88}, 6.0, 0.54),
    "p3r":   ({"fg3m": 1, "fg3_miss": 1}, {"fg2m": 1, "fg2_miss": 1, "fg3m": 1, "fg3_miss": 1}, 3.0, 0.25),
    "ftr":   ({"ftm": 1, "ft_miss": 1}, {"fg2m": 1, "fg2_miss": 1, "fg3m": 1, "fg3_miss": 1}, 3.0, 0.25),
    "fg3p":  ({"fg3m": 1}, {"fg3m": 1, "fg3_miss": 1}, 3.0, 0.34),
    "fg2p":  ({"fg2m": 1}, {"fg2m": 1, "fg2_miss": 1}, 3.0, 0.48),
    "ftp":   ({"ftm": 1}, {"ftm": 1, "ft_miss": 1}, 2.0, 0.75),
    "astr":  ({"ast": 1}, {"fg2m": 1, "fg2_miss": 1, "fg3m": 1, "fg3_miss": 1, "ftm": 0.44, "ft_miss": 0.44,
                           "tov": 1}, 3.0, 0.18),
    "tovr":  ({"tov": 1}, {"fg2m": 1, "fg2_miss": 1, "fg3m": 1, "fg3_miss": 1, "ftm": 0.44, "ft_miss": 0.44,
                           "tov": 1}, 3.0, 0.13),
    "orbsh": ({"orb": 1}, {"orb": 1, "drb": 1}, 3.0, 0.22),
}
# Shot quality from where his attempts came from: difficulty (the league's make probability from his spots) and
# shot-making (how far he beats a league shooter from those spots), padded in attempts toward the season's own
# league level.  The pads are attempts: threes need about 450 before a shooter's own rate is trusted.
SHOTQ_PADS = {"q2": 50.0, "q3": 50.0, "m2": 250.0, "m3": 450.0, "xps": 100.0, "mpts": 300.0}
SHOTQ = ["q2", "q3", "m2", "m3", "xps", "mpts"]

# Every input the prior could read (54): the rates, the role, every derived column, career, body, on-court.
# scripts/95_miss_by_group.py and commands/within_season.py (its PLAYER_COLUMNS) report on all of them.
PRIOR_FEATURES = [*BOX_RATES, *ROLE, *DERIVED, *RATIOS, *SHOTQ, *CAREER, *BIO, *ONC]
# The panel columns averaged into a multi-season training row, before anything is derived from them.
PANEL_INPUTS = [*BOX_RATES, *ROLE, *CLOSENESS, *RAW_RATES, *SHOT_TOTALS, *SHOT_LEAGUE, *CAREER, *BIO, *ONC, *OFFC,
                *NET, *PO_SHARE]
# How much evidence a training row rests on; the rated season's own row reads its possessions and 1.
ROW_FEATURES = ["chunk_poss", "chunk_seasons"]

# The shipped inputs, each side, in the order the booster reads them: the columns Boruta (a feature-selection
# method that keeps a column only when its importance beats that of randomly shuffled copies of every column)
# accepted or left undecided, sorted alphabetically, on-court columns removed (DECISIONS.md, experiment 45).  The
# booster reads its inputs by position, so the positions are load-bearing.  In plain words:
# offence: assists, assists minus turnovers, effective field-goal %, career possessions before this season
#   (thousands), seasons before this one, three-point %, shot attempts, free-throw attempts, offensive rebounds,
#   offensive-rebound share, share of shots that are threes, share of team possessions played, points, steals,
#   seasons with his current team, true shooting %, body weight
# defence: blocks, defensive rebounds, career possessions before this season, seasons before this one, shot
#   attempts, share of games started, height, share of team possessions played, points, steals, steals plus
#   blocks, true shooting %, possessions he finishes, body weight
SHIPPED = {
    "O": ["ast", "creation", "efg", "exp_poss", "exp_yrs", "fg3p", "fga", "fta", "orb", "orbsh", "p3r", "poss_pct",
          "pts", "stl", "tenure", "ts", "weight"],
    "D": ["blk", "drb", "exp_poss", "exp_yrs", "fga", "gs_pct", "height", "poss_pct", "pts", "stl", "stocks", "ts",
          "usage", "weight"],
}
FEATURE_SETS = {"boruta_noonc": SHIPPED}


# ------------------------------------------------------------------------------ building them
def _add_combinations(frame: pd.DataFrame) -> None:
    """Each `DERIVED` column whose rates the frame carries, summed term by term in the listed order.

    Example: `creation` = 1 x ast + (-1) x tov, so 3.0 assists and 1.0 turnovers above the league give 2.0.
    """
    for name, weights in DERIVED.items():
        if name in frame.columns or not all(rate in frame.columns for rate in weights):
            continue
        frame[name] = sum(weight * frame[rate].to_numpy(dtype=float) for rate, weight in weights.items())


def _add_ratios(frame: pd.DataFrame) -> None:
    """Each `RATIOS` column on the uncentred (`raw_`) rates:

        ratio = (numerator + pad x typical value) / (denominator + pad)

    Example: offensive-rebound share with 3.0 offensive and 9.0 defensive rebounds per 100, pad 3 and typical
    value 0.22: (3.0 + 3 x 0.22) / (3.0 + 9.0 + 3) = 0.244.
    """
    for name, (numerator, denominator, pad, typical) in RATIOS.items():
        rates = set(numerator) | set(denominator)
        if name in frame.columns or not all(f"raw_{rate}" in frame.columns for rate in rates):
            continue
        top = sum(weight * frame[f"raw_{rate}"].to_numpy(dtype=float) for rate, weight in numerator.items())
        bottom = sum(weight * frame[f"raw_{rate}"].to_numpy(dtype=float) for rate, weight in denominator.items())
        frame[name] = (top + pad * typical) / (bottom + pad)


def add_shotq(frame: pd.DataFrame) -> pd.DataFrame:
    """The `SHOTQ` columns, when the frame carries the shot totals and the league levels.

        q2   = (expected two-point makes + 50 x league 2P%) / (two-point attempts + 50)       difficulty
        m2   = (two-point makes - expected) / (two-point attempts + 250)                       shot-making
        xps  = (2 x expected 2s + 3 x expected 3s + 100 x league points per shot) / (attempts + 100)
        mpts = (2 x (makes - expected) on twos + 3 x the same on threes) / (attempts + 300)
    and q3, m3 the same on threes (pads 50 and 450).  "Expected" makes are the league's make probability from
    each of his shot locations, summed over his attempts; "attempts" counts twos and threes together.

    Example: 300 three-point attempts, 111 made and 105 expected give m3 = (111 - 105) / (300 + 450) = 0.008.
    """
    if "q2" in frame.columns or not all(c in frame.columns for c in (*SHOT_TOTALS, *SHOT_LEAGUE)):
        return frame
    column = {name: frame[name].to_numpy(dtype=float) for name in (*SHOT_TOTALS, *SHOT_LEAGUE)}
    attempts2, makes2, expected2 = column["shot_fg2a"], column["shot_fg2m"], column["shot_xl2"]
    attempts3, makes3, expected3 = column["shot_fg3a"], column["shot_fg3m"], column["shot_xl3"]
    league2, league3, league_points = column["shot_lg2"], column["shot_lg3"], column["shot_lgpps"]
    pad, attempts = SHOTQ_PADS, attempts2 + attempts3
    frame["q2"] = (expected2 + pad["q2"] * league2) / (attempts2 + pad["q2"])
    frame["q3"] = (expected3 + pad["q3"] * league3) / (attempts3 + pad["q3"])
    # padded toward 0, not toward the league: a league-average shooter beats his expected makes by nothing
    frame["m2"] = (makes2 - expected2) / (attempts2 + pad["m2"])
    frame["m3"] = (makes3 - expected3) / (attempts3 + pad["m3"])
    frame["xps"] = (2.0 * expected2 + 3.0 * expected3 + pad["xps"] * league_points) / (attempts + pad["xps"])
    frame["mpts"] = (2.0 * (makes2 - expected2) + 3.0 * (makes3 - expected3)) / (attempts + pad["mpts"])
    return frame


def add_features(frame: pd.DataFrame, names) -> pd.DataFrame:
    """Add every combination, ratio and shot-quality column the frame can make (in place), then check `names`.

    A family whose inputs are absent is skipped, so the check is what stops a prior from training on a
    quietly shorter list: every name asked for must exist afterwards, or this raises.
    """
    _add_combinations(frame)
    _add_ratios(frame)
    add_shotq(frame)
    missing = [name for name in names if name not in frame.columns]
    if missing:
        raise KeyError(f"cannot build {missing}: the rows lack their inputs. "
                       f"Needed: {[column for column in PANEL_INPUTS if column not in frame.columns]}")
    return frame


# ------------------------------------------------------------------------------ the catalogue
def _linear(weights: dict) -> str:
    """{"fg2m": 2.0, "fg3m": 3.0, "ftm": 1.0} -> "2 fg2m + 3 fg3m + ftm"."""
    out = ""
    for position, (code, weight) in enumerate(weights.items()):
        sign = ("-" if weight < 0 else "") if position == 0 else (" - " if weight < 0 else " + ")
        out += f"{sign}{'' if abs(weight) == 1 else f'{abs(weight):g} '}{code}"
    return out


def _family(codes, meanings, formula, unit, scale, group) -> list:
    """One `Feature` per code, all sharing a formula, unit, scale and group."""
    return [Feature(code, meaning, formula, unit, scale, group) for code, meaning in zip(codes, meanings)]


_RATE_MEANINGS = ["made threes", "missed threes", "made twos", "missed twos", "made free throws",
                  "missed free throws", "offensive rebounds", "defensive rebounds", "assists", "turnovers", "steals",
                  "blocks", "personal fouls"]
_DERIVED_MEANINGS = {"pts": "points", "fga": "shot attempts", "fta": "free-throw attempts",
                     "fg3a": "three-point attempts", "usage": "possessions he finishes", "reb": "rebounds",
                     "stocks": "steals plus blocks", "creation": "assists minus turnovers",
                     "shotmix": "threes minus twos taken", "bigness": "'plays like a centre' score"}
_RATIO_MEANINGS = {"efg": "effective field-goal %", "ts": "true shooting %", "p3r": "share of shots that are threes",
                   "ftr": "free throws per shot", "fg3p": "three-point %", "fg2p": "two-point %",
                   "ftp": "free-throw %", "astr": "assists per possession used",
                   "tovr": "turnovers per possession used", "orbsh": "offensive-rebound share of his rebounds"}
_SHOTQ_ROWS = [
    # code, meaning, formula ("attempts" = shot_fg2a + shot_fg3a; {pad} is the code's SHOTQ_PADS entry), unit
    ("q2", "difficulty of his twos", "(shot_xl2 + {pad} shot_lg2) / (shot_fg2a + {pad})", "fraction"),
    ("q3", "difficulty of his threes", "(shot_xl3 + {pad} shot_lg3) / (shot_fg3a + {pad})", "fraction"),
    ("m2", "two-point shot-making", "(shot_fg2m - shot_xl2) / (shot_fg2a + {pad})", "makes per attempt"),
    ("m3", "three-point shot-making", "(shot_fg3m - shot_xl3) / (shot_fg3a + {pad})", "makes per attempt"),
    ("xps", "expected points per shot", "(2 shot_xl2 + 3 shot_xl3 + {pad} shot_lgpps) / (attempts + {pad})", "points"),
    ("mpts", "points per shot above expected",
     "(2 (shot_fg2m - shot_xl2) + 3 (shot_fg3m - shot_xl3)) / (attempts + {pad})", "points"),
]
_PANEL_ROWS = [
    # code, meaning, unit, scale, group
    ("poss_pct", "share of team possessions played", "share", "level", "role"),
    ("gs_pct", "share of games started", "share", "level", "role"),
    ("age", "age", "years", "level", "role"),
    ("gt_share", "share of possessions in garbage time", "share", "level", "score state"),
    ("closeness", "how close his games were: mean of 1 / max(|margin|, 1)", "1 / points", "level", "score state"),
    ("abs_margin", "average score gap while he played", "points", "level", "score state"),
    ("shot_fg2a", "two-point attempts", "attempts", "level", "shot totals"),
    ("shot_fg2m", "two-point makes", "makes", "level", "shot totals"),
    ("shot_xl2", "expected two-point makes, league rates from his spots", "makes", "level", "shot totals"),
    ("shot_fg3a", "three-point attempts", "attempts", "level", "shot totals"),
    ("shot_fg3m", "three-point makes", "makes", "level", "shot totals"),
    ("shot_xl3", "expected three-point makes, league rates from his spots", "makes", "level", "shot totals"),
    ("shot_lg2", "league two-point %", "fraction", "level", "shot totals"),
    ("shot_lg3", "league three-point %", "fraction", "level", "shot totals"),
    ("shot_lgpps", "league points per shot", "points", "level", "shot totals"),
    ("exp_yrs", "seasons played before this one", "seasons", "level", "career"),
    ("exp_poss", "career possessions before this season", "thousands", "level", "career"),
    ("entry_age", "age entering the league", "years", "level", "career"),
    ("height", "height", "inches", "level", "body"),
    ("weight", "weight", "pounds", "level", "body"),
    ("draft_pick", "draft position", "pick", "level", "body"),
    ("tenure", "seasons with his current team", "seasons", "level", "career"),
    ("n_teams", "teams he played for this season", "teams", "level", "career"),
    ("onc_o", "team points scored per 100, him on court", "per 100 possessions", "centred", "on court"),
    ("onc_d", "team points allowed per 100, him on court", "per 100 possessions", "centred", "on court"),
    ("onc_poss_o", "offensive possessions on court", "possessions", "level", "on court"),
    ("onc_poss_d", "defensive possessions on court", "possessions", "level", "on court"),
    ("offc_o", "team points scored per 100, him off court", "per 100 possessions", "centred", "off court"),
    ("offc_d", "team points allowed per 100, him off court", "per 100 possessions", "centred", "off court"),
    ("offc_poss_o", "offensive possessions off court", "possessions", "level", "off court"),
    ("offc_poss_d", "defensive possessions off court", "possessions", "level", "off court"),
    ("net_o", "on court minus off court, points scored", "per 100 possessions", "centred", "on/off"),
    ("net_d", "on court minus off court, points allowed", "per 100 possessions", "centred", "on/off"),
    ("po_share", "share of possessions in the playoffs", "share", "level", "playoffs"),
    ("chunk_poss", "possessions behind the row", "possessions", "level", "row evidence"),
    ("chunk_seasons", "seasons the row covers", "seasons", "level", "row evidence"),
]


def _ratio_formula(numerator: dict, denominator: dict, pad: float, typical: float) -> str:
    """A `RATIOS` entry written out: "(fg3m + 3 x 0.34) / (fg3m + fg3_miss + 3), uncentred rates"."""
    return f"({_linear(numerator)} + {pad:g} x {typical:g}) / ({_linear(denominator)} + {pad:g}), uncentred rates"


def _catalogue() -> tuple:
    """Every column the prior can read, family by family, in the order the panel and the derivations list them."""
    rows = _family(BOX_RATES, _RATE_MEANINGS, "padded rate minus the league level", "per 100 possessions",
                   "centred", "box rate")
    rows += _family(RAW_RATES, [f"{meaning}, league level kept" for meaning in _RATE_MEANINGS], "padded rate",
                    "per 100 possessions", "level", "box rate, uncentred")
    rows += [Feature(code, _DERIVED_MEANINGS[code], _linear(weights), "per 100 possessions", "centred",
                     "combination") for code, weights in DERIVED.items()]
    rows += [Feature(code, _RATIO_MEANINGS[code], _ratio_formula(*spec), "fraction", "level", "ratio")
             for code, spec in RATIOS.items()]
    rows += [Feature(code, meaning, formula.format(pad=f"{SHOTQ_PADS[code]:g}"), unit, "level", "shot quality")
             for code, meaning, formula, unit in _SHOTQ_ROWS]
    rows += [Feature(code, meaning, "panel column", unit, scale, group)
             for code, meaning, unit, scale, group in _PANEL_ROWS]
    return tuple(rows)


CATALOGUE = _catalogue()


def glossary_markdown() -> str:
    """The catalogue as a markdown table, one row per code, for GLOSSARY.md."""
    def cell(text: str) -> str:
        return text.replace("|", "\\|")          # a literal |margin| would end the cell

    lines = ["| code | meaning | formula | unit | scale |", "|---|---|---|---|---|"]
    lines += [f"| `{feature.name}` | {cell(feature.meaning)} | {cell(feature.formula)} | {feature.unit} | "
              f"{feature.scale} |" for feature in CATALOGUE]
    return "\n".join(lines) + "\n"
