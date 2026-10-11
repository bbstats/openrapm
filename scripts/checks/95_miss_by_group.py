"""Where the ratings miss most: each player's miss, adjusted for his rating level and his evidence, cut every way.

The owner, 2026-10-03: "find the players for whom we struggle with the most ... not bias like the bias chart, this
is like variance ... there are specific types and subsets of players and maybe even stats that cause us to
struggle", and then, on the first version: "did you test a whole bunch of cuts and stats?? We should do the latter."
DECISIONS.md, "Where the ratings miss most", has the first results.

Words used below
    miss            the trade set's correction for a player-season (scripts/checks/70_tradeset.py, the alpha behind
                    the trade loss): how far his team's games over the rated season and the two either side say his
                    rating should move, in points per 100, after the season's whole list is rescaled.  Read only
                    where the trade loss reads it: one-team players whose team played some neighbouring games
                    without him.  Positive = the games say he is better than rated.  It includes real change
                    between seasons as well as the rating's own error, and it is a ridge estimate, so true misses
                    run larger
    evidence        what the correction rests on: his on-court possessions over the three seasons (`with_poss`)
                    and his team's neighbouring possessions without him (`without_poss`)
    expected miss   the squared miss a player rated the same, with the same evidence, is expected to have: a
                    gradient-boosted Poisson regression of the squared miss on the rating, both evidence counts and
                    the season, fitted with every player's seasons held out together (`adjusted_misses`).  The
                    owner: "variance will probably be higher for better players or better rated players, so you
                    would need to figure out how to do an adjustment for the magnitude"
    miss ratio      squared miss / expected miss.  A group's mean ratio is its miss against expectation: 1.00 =
                    missed as much as players rated the same with the same evidence, 1.40 = 40% more squared miss
    lean            miss / sqrt(expected miss), signed: a group can be missed mostly one way (the bias page's
                    question), and "of which lean" (the mean lean squared) is the share of its miss that is
                    one-directional
    z               a group's mean against 1 (the ratio) or 0 (the lean) over its standard error, with all of a
                    player's seasons in the group as one cluster

The cuts, every one reported:
  1. statistics: fifths of every statistic in GLOSSARY.md, cut within each season, plus the off-court and on/off
     ratings, score state and playoff share;
  2. player types: a Bayesian Gaussian mixture (`player_types.fit_mixture`, up to eight types) on the thirteen box
     rates per 100, standardised within each season so a type is a role relative to its era, fitted on
     player-seasons with 500+ possessions and used to place every one;
  3. context: age, years in the league, era, his team's quality that season, and whether he changed teams the
     season before and the season after;
  4. how the rating was built: fifths of the prior, of how far his own games moved him from it (signed and in
     size), of the swap adjustment's move, and of his rating on the other side of the ball;
  5. combinations: a three-level regression tree on everything above, fitted on half the players and read on the
     other half, so every subset it reports is measured on players it never saw; and the share of the miss ratio
     a gradient-boosted model on everything can predict for players it never saw (0% = the misses carry no
     pattern beyond what the expected miss already accounts for).

How to read it: the run first prints its own check, the mean miss ratio by rating level, evidence level and share
of team possessions, which should read about 1.00 everywhere.  With several hundred cuts, about one in 370 passes
|z| 3 by chance; the run prints how many pass against how many chance alone would give.

Reads   `--alpha` (a 70_tradeset.py <out>_alpha.parquet), `--ratings` (the ratings table it was run on), the
        season panel (paths.panel, gated: a season still being played reaches no fit) and the roles table.
Writes  outputs/miss_by_group_<tag>.html (every cut, plain words), outputs/csv/miss_by_group_<tag>.csv (every cut),
        outputs/csv/miss_subsets_<tag>.csv (the tree's subsets) and outputs/csv/miss_players_<tag>.csv (every
        player-season's adjusted miss).

Example:

    python scripts/checks/95_miss_by_group.py --alpha=outputs/tradeset_weightfix_alpha.parquet \\
        --ratings=outputs/season_ratings_weightfix.parquet --tag=weightfix
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import d2_tweedie_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.tree import DecisionTreeRegressor

from openrapm.commands.console import use_utf8
from openrapm.data.config import load_config, resolve, resolve_path
from openrapm.data.roles import franchise, roles_path
from openrapm.data.seasons import drop_untrainable
from openrapm.features import CLOSENESS, NET, OFFC, PO_SHARE, PRIOR_FEATURES, add_features
from openrapm.glossary import plain
from openrapm.player_types import cluster_proba, fit_mixture

N_CELLS = 6                        # levels for the adjustment's own check
N_GROUPS = 5                       # fifths
MIN_ROWS = 50                      # a cut with fewer player-seasons is not reported
TYPE_POSSESSIONS = 500             # the player types are fitted on player-seasons with this many possessions or more
STATS = tuple(dict.fromkeys(PRIOR_FEATURES + OFFC + NET + CLOSENESS + PO_SHARE))
TYPE_RATES = ("fg2m", "fg2_miss", "fg3m", "fg3_miss", "ftm", "ft_miss", "orb", "drb", "ast", "tov", "stl", "blk",
              "pf")
SIDES = {"offense": ("O", "off", "def"), "defense": ("D", "def", "off")}
FIFTH = {1: "bottom fifth", 2: "second fifth", 3: "middle fifth", 4: "fourth fifth", 5: "top fifth"}
BUILT = {"team_net": "his team's quality that season (net points per 100)",
         "prior": "his prior (the box-score estimate)",
         "games": "how far his own games moved him from the prior (signed)",
         "games_size": "how far his own games moved him from the prior (size)",
         "swap": "the swap adjustment's move",
         "other_side": "his rating on the other side of the ball"}
CHANCE_Z = 3.0
CHANCE_RATE = 0.0027               # two-sided |z| >= 3
RECENT_SEASON = 2022               # the printed list of the player-seasons missed most starts here
PEOPLE_COLUMNS = ("side", "player_id", "player_name", "season", "player_type", "rated", "alpha_good", "evidence",
                  "miss_ratio", "lean")
STYLE = ("<style>body{font-family:system-ui,sans-serif;margin:24px;max-width:1100px}"
         "table{border-collapse:collapse;font-size:13px;margin:8px 0 24px}"
         "th,td{border-bottom:1px solid #ddd;padding:3px 10px;text-align:left}"
         "th{background:#f3f3f3;position:sticky;top:0}</style>")
HTML_INTRO = ("<h1>Where the ratings miss most, against expectation</h1>",
              "<p>A player's miss: how far his team's games in the rated season and the seasons either side say his "
              "rating is off. Each miss is compared with what players rated the same, with the same evidence, miss: "
              "+20% means 20% more squared miss than expected. <b>Lean</b> = missed mostly one way; <b>of which "
              "lean</b> = the part of the miss that is one-directional. Every cut is listed, sorted by z.</p>")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Where the ratings miss, by player group.", allow_abbrev=False)
    parser.add_argument("--tag", default="incumbent", help="names the outputs (default %(default)s)")
    parser.add_argument("--alpha", default="outputs/tradeset_swapadj_within_alpha.parquet",
                        help="a scripts/checks/70_tradeset.py <out>_alpha.parquet (default %(default)s)")
    parser.add_argument("--ratings", default="outputs/season_ratings_swapadj_within.parquet",
                        help="the ratings table (default %(default)s)")
    return parser.parse_args(argv)


# ------------------------------------------------------------------------------------------- small helpers
def name(code: str) -> str:
    """A column's plain name."""
    return BUILT.get(code) or plain(code)


def levels(x: pd.Series, n: int) -> np.ndarray:
    """0..n-1 equal-count levels of `x` (ties broken by order, so every level is the same size)."""
    return pd.qcut(pd.Series(x).rank(method="first"), n, labels=False).to_numpy()


def cluster_mean(values: np.ndarray, players: np.ndarray, centre: float) -> tuple:
    """The mean of `values`, and its z against `centre` with each player's rows as one cluster.

        se = sqrt(sum over players (sum of his values - mean)^2) / rows
    """
    mean = float(np.mean(values))
    by_player = pd.Series(values - mean).groupby(players).sum().to_numpy()
    se = float(np.sqrt(np.sum(by_player ** 2))) / len(values)
    return mean, (mean - centre) / se if se > 0 else np.nan


def poisson_booster(max_iter: int) -> HistGradientBoostingRegressor:
    """The gradient-boosted Poisson regression both the expected miss and the predictable share use."""
    return HistGradientBoostingRegressor(loss="poisson", max_depth=3, learning_rate=0.05, max_iter=max_iter,
                                         min_samples_leaf=200, random_state=0)


# ------------------------------------------------------------------------------------------- the inputs
def season_stats(panel: pd.DataFrame, short: str) -> pd.DataFrame:
    """One side's panel rows ("O" or "D") with the prior's derived statistics, one row per player-season."""
    return add_features(panel[panel.side == short].copy(), PRIOR_FEATURES).drop_duplicates(["player_id", "season"])


def side_stats(panel: pd.DataFrame, short: str) -> pd.DataFrame:
    """The statistics the cuts read for one side, and his team's quality that season (`team_net`).

    team_net = (on-court and off-court points scored per 100, possession-weighted together) minus the same for
    points allowed.
    """
    stats = season_stats(panel, short)
    wanted = [*STATS, *(f"raw_{rate}" for rate in TYPE_RATES)]
    keep = ["player_id", "season", "poss"] + [c for c in wanted if c in stats.columns and c not in ("poss",)]
    stats = stats[list(dict.fromkeys(keep))]
    team_o = ((stats.onc_o * stats.onc_poss_o + stats.offc_o * stats.offc_poss_o)
              / (stats.onc_poss_o + stats.offc_poss_o))
    team_d = ((stats.onc_d * stats.onc_poss_d + stats.offc_d * stats.offc_poss_d)
              / (stats.onc_poss_d + stats.offc_poss_d))
    stats["team_net"] = team_o - team_d
    return stats


def built_columns(ratings: pd.DataFrame, own: str, other: str) -> pd.DataFrame:
    """How each rating was built, for one side: prior, games (u), swap (c), other_side and games_size."""
    built = ratings[["player_id", "season", f"prior_{own}", f"u_{own}", f"c_{own}", f"rating_{other}"]].rename(
        columns={f"prior_{own}": "prior", f"u_{own}": "games", f"c_{own}": "swap", f"rating_{other}": "other_side"})
    built["games_size"] = built.games.abs()
    return built


def player_types(stats: pd.DataFrame, k: int = 8) -> tuple:
    """(each player-season's type, {type: its plain description}), the biggest type first.

    The thirteen box rates per 100 (uncentred, padded), standardised within each season so a type is a role
    relative to its era rather than to 1997 or 2026; a Bayesian Gaussian mixture fitted on the player-seasons with
    500+ possessions (a 100-possession profile is mostly noise) and used to place every player-season.  A type is
    described by the two rates it has most of and the two it has least of.
    """
    columns = [f"raw_{rate}" for rate in TYPE_RATES]
    standardised = (stats[columns].groupby(stats.season.to_numpy())
                    .transform(lambda column: (column - column.mean()) / column.std(ddof=0)).to_numpy(float))
    fitted = stats.poss.to_numpy(float) >= TYPE_POSSESSIONS
    model, mean, sd = fit_mixture(standardised[fitted], k=k, seed=0, n_init=3)
    label = cluster_proba(model, mean, sd, standardised).argmax(1)
    names = {}
    for rank, kind in enumerate(pd.Series(label).value_counts().index, start=1):
        profile = pd.Series(standardised[label == kind].mean(0), index=list(TYPE_RATES))
        high = profile.sort_values(ascending=False).index[:2]
        low = profile.sort_values().index[:2]
        names[kind] = (f"type {rank}: most {plain(high[0])} and {plain(high[1])}, "
                       f"least {plain(low[0])} and {plain(low[1])}")
    return pd.Series(label, index=stats.index).map(names), names


def team_moves(cfg: dict) -> pd.DataFrame:
    """Per player-season: whether he was on the same franchise the season before and the season after.

    His team is the franchise he played the most possessions for that season (`roles.franchise` maps a relocated
    team to its franchise, so a move of city is not a change of team).
    """
    roles = pd.read_parquet(roles_path(cfg))
    played = roles[roles.poss_on > 0][["player_id", "season", "team_id", "poss_on"]].copy()
    played["team_id"] = franchise(played.team_id.to_numpy(), played.season.to_numpy())
    main = (played.groupby(["player_id", "season", "team_id"], as_index=False).poss_on.sum()
            .sort_values("poss_on").drop_duplicates(["player_id", "season"], keep="last")
            [["player_id", "season", "team_id"]])
    first, last = int(main.season.min()), int(main.season.max())
    following = main.assign(season=main.season - 1).rename(columns={"team_id": "team_next"})
    preceding = main.assign(season=main.season + 1).rename(columns={"team_id": "team_prev"})
    out = (main.merge(following, on=["player_id", "season"], how="left")
           .merge(preceding, on=["player_id", "season"], how="left"))
    out["next_season"] = np.select(
        [out.season == last, out.team_next.isna(), out.team_next == out.team_id],
        ["no next season yet", "out of the league next season", "same team next season"],
        "new team next season")
    out["previous_season"] = np.select(
        [out.season == first, out.team_prev.isna(), out.team_prev == out.team_id],
        ["no earlier season in the data", "new to the league (or back from a gap)", "same team as the season before"],
        "changed teams since the season before")
    return out[["player_id", "season", "next_season", "previous_season"]]


def load_inputs(args: argparse.Namespace, cfg: dict) -> tuple:
    """(the eligible corrections, the ratings table, the season panel's played rows)."""
    alpha = pd.read_parquet(resolve_path(cfg, args.alpha))
    alpha = alpha[alpha.eligible & (alpha.without_poss >= 1.0)]
    ratings = pd.read_parquet(resolve_path(cfg, args.ratings))
    # the trust boundary (src/openrapm/data/seasons.py): a season still being played reaches no fit
    panel, _ = drop_untrainable(pd.read_parquet(resolve(cfg, "panel")), cfg, what="the season panel")
    return alpha, ratings, panel[panel.poss > 0]


def side_rows(alpha: pd.DataFrame, stats: pd.DataFrame, types: pd.DataFrame, built: pd.DataFrame,
              moves: pd.DataFrame, side: str) -> pd.DataFrame:
    """One side's corrections joined with everything the cuts read, one row per player-season."""
    return (alpha[alpha.side == side]
            .merge(stats, on=["player_id", "season"], how="inner")
            .merge(types, on=["player_id", "season"], how="left")
            .merge(built, on=["player_id", "season"], how="left")
            .merge(moves, on=["player_id", "season"], how="left"))


# ------------------------------------------------------------------------------------------- the adjustment
def adjusted_misses(rows: pd.DataFrame) -> pd.DataFrame:
    """Each row's squared miss against the expected squared miss for its rating level and its evidence.

    The expectation is a gradient-boosted Poisson model of the squared miss on the rating level, his on-court
    possessions over the three seasons (`with_poss`), his team's neighbouring games without him (`without_poss`)
    and the season, cross-fitted over players (five folds), so no player's own miss sets his own expectation, and
    rescaled so the mean ratio is exactly 1.  The two possession counts go in separately because the correction is
    identified by both.  A first pass (2026-10-03) expected the miss from 6 x 6 cells of the rating level and ONE
    combined evidence number, `with x without / (with + without)`; it left starters, heavy-minute and high-usage
    players 15-40% above expectation and bench players 20-30% below, and the minutes check read far from 1.  With
    the two counts separate the minutes check is flat (0.97-1.06) and those groups vanish.  Since `with_poss` is
    close to minutes, "starters are harder" cannot be told apart from "the correction sees starters more clearly"
    with this measure.
    """
    rows = rows.reset_index(drop=True).copy()
    rows["rated"] = rows.rating * rows.rating_scale
    rows["evidence"] = rows.with_poss * rows.without_poss / (rows.with_poss + rows.without_poss)
    rows["rating_level"] = levels(rows.rated, N_CELLS)
    rows["evidence_level"] = levels(rows.evidence, N_CELLS)
    inputs = np.column_stack([rows.rated, np.log1p(rows.with_poss), np.log1p(rows.without_poss), rows.season])
    squared = rows.alpha_good.to_numpy(float) ** 2
    expected = np.empty(len(rows))
    for fit_rows, held_rows in GroupKFold(n_splits=5).split(inputs, squared, rows.player_id):
        expected[held_rows] = poisson_booster(300).fit(inputs[fit_rows], squared[fit_rows]).predict(inputs[held_rows])
    expected *= np.mean(squared / expected)                      # the average ratio exactly 1
    rows["expected"] = expected
    rows["miss_ratio"] = squared / expected
    rows["lean"] = rows.alpha_good / np.sqrt(expected)          # signed, in expected misses
    return rows


# ------------------------------------------------------------------------------------------- the cuts
def group_row(rows: pd.DataFrame, mask: np.ndarray, players: np.ndarray, side: str, kind: str, label: str) -> dict:
    """One cut's line: its size, its mean miss ratio and lean, and their z."""
    extra, z = cluster_mean(rows.miss_ratio.to_numpy()[mask], players[mask], 1.0)
    lean, lean_z = cluster_mean(rows.lean.to_numpy()[mask], players[mask], 0.0)
    return dict(side=side, kind=kind, group=label, rows=int(mask.sum()), miss_vs_expected=extra, z=z,
                lean=lean, lean_z=lean_z)


def statistic_cuts(rows: pd.DataFrame, players: np.ndarray, side: str) -> list:
    """Fifths of every statistic and of every part of how the rating was built, each cut within its season."""
    out = []
    continuous = ([(code, "statistic") for code in STATS if code in rows.columns]
                  + [(code, "how the rating was built") for code in BUILT])
    for code, kind in continuous:
        if rows[code].nunique() < N_GROUPS:
            continue
        kind = "context" if code == "team_net" else kind
        share = rows.groupby("season")[code].rank(pct=True, method="average")
        fifth = np.clip(np.ceil(share * N_GROUPS), 1, N_GROUPS).astype(int).to_numpy()
        for group in range(1, N_GROUPS + 1):
            mask = fifth == group
            if mask.sum() >= MIN_ROWS:
                out.append(group_row(rows, mask, players, side, kind, f"{FIFTH[group]} of {name(code)}"))
    return out


def type_cuts(rows: pd.DataFrame, players: np.ndarray, side: str) -> list:
    """One cut per player type."""
    out = []
    for label in sorted(rows.player_type.dropna().unique()):
        mask = (rows.player_type == label).to_numpy()
        if mask.sum() >= MIN_ROWS:
            out.append(group_row(rows, mask, players, side, "player type", label))
    return out


def context_values(rows: pd.DataFrame) -> list:
    """Age, years in the league, era and the two team moves, one label per row each."""
    return [
        pd.cut(rows.age, [0, 23, 27, 31, 99], right=False,
               labels=["age under 23", "age 23-26", "age 27-30", "age 31 and over"]),
        pd.cut(rows.exp_yrs, [-1, 0.5, 3.5, 8.5, 99],
               labels=["rookie", "1-3 years in the league", "4-8 years in the league", "9+ years in the league"]),
        pd.cut(rows.season, [1996, 2005, 2015, 2026], labels=["1997-2005", "2006-2015", "2016-2026"]),
        rows.next_season,
        rows.previous_season,
    ]


def context_cuts(rows: pd.DataFrame, players: np.ndarray, side: str) -> list:
    """One cut per value of each context."""
    out = []
    for values in context_values(rows):
        values = pd.Series(values)
        for label in values.dropna().unique():
            mask = (values == label).to_numpy()
            if mask.sum() >= MIN_ROWS:
                out.append(group_row(rows, mask, players, side, "context", str(label)))
    return out


def all_cuts(rows: pd.DataFrame, side: str) -> pd.DataFrame:
    """Every cut of one side: statistics and how the rating was built, player types, then context."""
    players = rows.player_id.to_numpy()
    return pd.DataFrame(statistic_cuts(rows, players, side) + type_cuts(rows, players, side)
                        + context_cuts(rows, players, side))


# ------------------------------------------------------------------------------------------- combinations
def tree_parents(tree) -> dict:
    """{node: (its parent, "<=" or ">")} for a fitted sklearn tree's `tree_`."""
    parent = {}
    for node in range(tree.node_count):
        if tree.children_left[node] >= 0:
            parent[tree.children_left[node]] = (node, "<=")
            parent[tree.children_right[node]] = (node, ">")
    return parent


def rule_text(label: str, how: str, cut: float) -> str:
    """One split in plain words: "is type 3" / "is not type 3", or "<label> <= 2,730"."""
    if label.startswith("is type"):
        return label if how == ">" else label.replace("is ", "is not ")
    # plain numbers: 2,730 not 2.73e+03 (the owner read the scientific form as a typo)
    return f"{label} {how} {cut:,.0f}" if abs(cut) >= 1000 else f"{label} {how} {cut:.3g}"


def leaf_rule(tree, labels: list, parent: dict, leaf: int) -> str:
    """The path from the root to `leaf`, every split in plain words, joined with "and"."""
    rules, node = [], leaf
    while node in parent:
        up, how = parent[node]
        rules.append(rule_text(labels[tree.feature[up]], how, tree.threshold[up]))
        node = up
    return " and ".join(reversed(rules))


def subset_inputs(rows: pd.DataFrame, features: list) -> tuple:
    """(inputs, their plain labels): every feature (missing values set to the median) and one column per type."""
    types = pd.get_dummies(rows.player_type, prefix="", prefix_sep="").astype(float)
    inputs = pd.concat([rows[features].astype(float), types], axis=1)
    inputs = inputs.fillna(inputs.median())
    labels = [name(c) for c in features] + [f"is {kind.split(':')[0]}" for kind in types.columns]
    return inputs, labels


def held_out_leaves(rows: pd.DataFrame, inputs: pd.DataFrame, labels: list, held_rows: np.ndarray,
                    tree_model: DecisionTreeRegressor) -> pd.DataFrame:
    """Each leaf of the tree fitted on one half, measured on the other half, `held_rows` (leaves under 50 rows
    skipped)."""
    tree = tree_model.tree_
    parent = tree_parents(tree)
    leaf_held = tree_model.apply(inputs.iloc[held_rows])
    rows_held = rows.iloc[held_rows].reset_index(drop=True)
    leaves = []
    for leaf in np.unique(leaf_held):
        mask = leaf_held == leaf
        if mask.sum() < MIN_ROWS:
            continue
        line = group_row(rows_held, mask, rows_held.player_id.to_numpy(), "", "subset",
                         leaf_rule(tree, labels, parent, leaf))
        line["miss_in_fitting_half"] = float(tree.value[leaf].ravel()[0])
        leaves.append(line)
    return pd.DataFrame(leaves)


def predictable_share(inputs: pd.DataFrame, ratio: np.ndarray, players: np.ndarray) -> float:
    """The share of the miss ratio a gradient-boosted model predicts for players it never saw (Poisson D^2)."""
    expected = np.empty(len(ratio))
    for fit_rows, held_rows in GroupKFold(n_splits=5).split(inputs, ratio, players):
        expected[held_rows] = poisson_booster(200).fit(inputs.iloc[fit_rows], ratio[fit_rows]).predict(
            inputs.iloc[held_rows])
    return float(d2_tweedie_score(ratio, expected, power=1))


def subset_search(rows: pd.DataFrame, features: list) -> tuple:
    """(a three-level tree's leaves fitted on half the players and measured on the other half, the held-out share
    of the miss ratio a gradient-boosted model on everything can predict)."""
    inputs, labels = subset_inputs(rows, features)
    ratio, players = rows.miss_ratio.to_numpy(float), rows.player_id.to_numpy()
    fit_rows, held_rows = next(GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=0).split(inputs, ratio,
                                                                                                   players))
    tree_model = DecisionTreeRegressor(max_depth=3, min_samples_leaf=300, random_state=0).fit(
        inputs.iloc[fit_rows], ratio[fit_rows])
    return held_out_leaves(rows, inputs, labels, held_rows, tree_model), predictable_share(inputs, ratio, players)


# ------------------------------------------------------------------------------------------- the report
def lean_words(row) -> str:
    """A cut's lean in words: none unless |lean z| is 3 or more."""
    if not np.isfinite(row.lean_z) or abs(row.lean_z) < 3:
        return "no clear lean"
    return "rated too low" if row.lean > 0 else "rated too high"


def describe(frame: pd.DataFrame) -> pd.DataFrame:
    """The plain-words table: who, how much more miss, which way it leans."""
    return pd.DataFrame({
        "players": frame.group.to_numpy(),
        "extra miss": [f"{100 * (v - 1):+.0f}%" for v in frame.miss_vs_expected],
        "z": frame.z.round(1).to_numpy(),
        "lean": [lean_words(r) for r in frame.itertuples()],
        "of which lean": [f"{100 * v ** 2:.0f}%" for v in frame.lean],
        "player-seasons": frame.rows.to_numpy(),
    })


def print_types(stats_o: pd.DataFrame, type_names: dict) -> None:
    """Each player type's name and how many player-seasons it holds."""
    print("player types (fitted on player-seasons with 500+ possessions, rates standardised within each season):")
    counts = stats_o.player_type.value_counts()
    for label in type_names.values():
        print(f"  {label} ({counts.get(label, 0):,} player-seasons)")


def print_adjustment_check(rows: pd.DataFrame, side: str) -> None:
    """The typical miss by rating level, and the mean miss ratio by rating, evidence and possession-share level."""
    print(f"\n=== {side.upper()}: {len(rows):,} player-seasons, {rows.player_id.nunique():,} players "
          f"(one-team players whose team played some neighbouring games without them)")
    typical = rows.groupby("rating_level").alpha_good.apply(lambda a: float(np.sqrt(np.mean(a ** 2))))
    print("  typical miss (points per 100) by rating level, worst sixth to best: "
          + ", ".join(f"{v:.2f}" for v in typical))
    for label, level in (("rating level", rows.rating_level), ("evidence level", rows.evidence_level),
                         ("share of team possessions played", levels(rows.poss_pct, N_CELLS))):
        print(f"  after the adjustment, miss against expected by {label}: "
              + ", ".join(f"{v:.2f}" for v in rows.miss_ratio.groupby(level).mean()))


def print_passes(table: pd.DataFrame) -> int:
    """How many cuts pass |z| 3, against chance; returns the count."""
    passes = int((table.z.abs() >= CHANCE_Z).sum())
    print(f"  {len(table)} cuts; {passes} pass |z| 3 against {CHANCE_RATE * len(table):.1f} expected by chance "
          f"({int((table.z >= CHANCE_Z).sum())} miss more, {int((table.z <= -CHANCE_Z).sum())} miss less)")
    return passes


def print_findings(table: pd.DataFrame, leaves: pd.DataFrame, share: float) -> None:
    """The cuts that pass |z| 3 either way, the tree's subsets, and the predictable share."""
    worst = table[table.z >= CHANCE_Z].sort_values("miss_vs_expected", ascending=False)
    best = table[table.z <= -CHANCE_Z].sort_values("miss_vs_expected")
    with pd.option_context("display.width", 250, "display.max_colwidth", 120):
        print("\n  where we miss MORE than players rated the same with the same evidence (|z| 3 or more):")
        print(describe(worst).to_string(index=False) if len(worst) else "  none")
        print("\n  where we miss LESS:")
        print(describe(best).to_string(index=False) if len(best) else "  none")
        print("\n  combinations: a three-level tree fitted on half the players, measured on the other half:")
        print(describe(leaves.sort_values("miss_vs_expected", ascending=False)).to_string(index=False))
        print(f"\n  how much of the adjusted miss can be predicted for players never seen, from everything above: "
              f"{100 * share:.1f}%")


def print_recent(rows: pd.DataFrame) -> None:
    """The twelve player-seasons since RECENT_SEASON with the largest miss ratio."""
    recent = rows[rows.season >= RECENT_SEASON].sort_values("miss_ratio", ascending=False).head(12)
    print(f"\n  the player-seasons we miss most since {RECENT_SEASON}, against expectation:")
    print(pd.DataFrame({
        "player": recent.player_name, "season": recent.season, "rated": recent.rated.round(2),
        "games say": [f"{v:+.2f} {'better' if v > 0 else 'worse'}" for v in recent.alpha_good],
        "times the expected miss": np.sqrt(recent.miss_ratio).round(1)}).to_string(index=False))


def analyse_side(rows: pd.DataFrame, side: str) -> tuple:
    """Adjust one side's misses, cut them every way and print what stands out.

    Returns (every cut, the tree's subsets, every player-season's adjusted miss, the side's part of the page).
    """
    rows = adjusted_misses(rows)
    print_adjustment_check(rows, side)
    table = all_cuts(rows, side)
    passes = print_passes(table)
    features = [c for c in STATS if c in rows.columns] + list(BUILT) + ["age", "exp_yrs"]
    leaves, share = subset_search(rows, list(dict.fromkeys(features)))
    leaves["side"] = side
    print_findings(table, leaves, share)
    rows["side"] = side
    print_recent(rows)
    page = (side, len(table), passes, CHANCE_RATE * len(table), share,
            describe(table.sort_values("z", ascending=False)),
            describe(leaves.sort_values("miss_vs_expected", ascending=False)))
    return table, leaves, rows[list(PEOPLE_COLUMNS)], page


# ------------------------------------------------------------------------------------------- the files
def write_html(path: Path, parts: list, type_names: dict) -> None:
    """The page: the introduction, the player types, then each side's subsets and every cut."""
    body = [*HTML_INTRO,
            "<h2>Player types</h2><ul>" + "".join(f"<li>{v}</li>" for v in type_names.values()) + "</ul>"]
    for side, cuts, passes, chance, share, table, leaves in parts:
        body.append(f"<h2>{side.capitalize()}</h2><p>{cuts} cuts; {passes} pass |z| 3 against {chance:.1f} expected by "
                    f"chance. Share of the adjusted miss predictable for unseen players from everything here: "
                    f"{100 * share:.1f}%.</p>")
        body.append("<h3>Combinations (fitted on half the players, measured on the other half)</h3>")
        body.append(leaves.to_html(index=False, border=0))
        body.append("<h3>Every cut</h3>")
        body.append(table.to_html(index=False, border=0))
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("<html><head><meta charset='utf-8'>" + STYLE + "</head><body>" + "".join(body)
                     + "</body></html>")


def write_outputs(outputs: Path, tag: str, results: list, type_names: dict) -> None:
    """The three csv files and the page; `results` holds analyse_side's four parts per side."""
    tables, subsets, people, pages = zip(*results)
    folder = outputs / "csv"
    folder.mkdir(parents=True, exist_ok=True)
    pd.concat(tables, ignore_index=True).to_csv(folder / f"miss_by_group_{tag}.csv", index=False)
    pd.concat(subsets, ignore_index=True).to_csv(folder / f"miss_subsets_{tag}.csv", index=False)
    pd.concat(people, ignore_index=True).to_csv(folder / f"miss_players_{tag}.csv", index=False)
    write_html(outputs / f"miss_by_group_{tag}.html", list(pages), type_names)
    print(f"\nwrote outputs/miss_by_group_{tag}.html, outputs/csv/miss_by_group_{tag}.csv, miss_subsets_{tag}.csv "
          f"and miss_players_{tag}.csv")


def main() -> None:
    use_utf8()
    args = parse_args()
    cfg = load_config()
    alpha, ratings, panel = load_inputs(args, cfg)
    moves = team_moves(cfg)
    stats_o = season_stats(panel, "O")
    stats_o["player_type"], type_names = player_types(stats_o)
    print_types(stats_o, type_names)
    types = stats_o[["player_id", "season", "player_type"]]
    results = []
    for side, (short, own, other) in SIDES.items():
        rows = side_rows(alpha, side_stats(panel, short), types, built_columns(ratings, own, other), moves, side)
        results.append(analyse_side(rows, side))
    write_outputs(resolve(cfg, "outputs"), args.tag, results, type_names)


if __name__ == "__main__":
    main()
