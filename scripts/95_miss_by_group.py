"""Where the ratings miss most: each player's miss, adjusted for his rating level and his evidence, cut every way.

    python scripts/95_miss_by_group.py [--alpha=outputs/tradeset_swapadj_within_alpha.parquet]
                                       [--ratings=outputs/season_ratings_swapadj_within.parquet] [--tag=incumbent]

The owner, 2026-10-03: "find the players for whom we struggle with the most ... not bias like the bias chart, this
is like variance ... there are specific types and subsets of players and maybe even stats that cause us to
struggle", and then, on the first version: "did you test a whole bunch of cuts and stats?? We should do the latter."

**A player's miss** is the trade set's correction (`scripts/70_tradeset.py`, the alpha behind the trade loss): how
far his team's games over the rated season and the two either side say his rating should move, in points per 100,
after the season's whole list is rescaled.  Read only where the trade loss reads it: one-team players whose team
played some neighbouring games without them.  Positive = the games say he is better than rated.  It includes real
change between seasons as well as the rating's own error, and it is a ridge estimate, so true misses run larger.

**The magnitude adjustment** (the owner: "variance will probably be higher for better players or better rated
players, so you would need to figure out how to do an adjustment for the magnitude").  Two things set how big a
miss looks before any player type enters: how good the rating says he is, and how much evidence the correction
rests on (his on-floor possessions over the three seasons, and his team's neighbouring games without him).  So
each squared miss is divided by its expectation given those (`adjusted_misses`), and a group's average of that
ratio is its miss against expectation: 1.00 = missed as much as players rated the same with the same evidence,
1.40 = 40% more squared miss.  The run prints its own check: after the adjustment the ratio should read ~1.00 at
every rating level, evidence level and minutes level.  A group can also LEAN -- be missed mostly one way, the bias
page's question -- so the lean is printed beside the extra miss, with the share of the miss it accounts for.

**The cuts**, every one reported (outputs/miss_by_group_<tag>.html and .csv):
  1. statistics: fifths of every statistic in GLOSSARY.md, cut within each season, plus the off-court and on/off
     ratings, score state and playoff share;
  2. player types: a Bayesian Gaussian mixture (`archetype.fit_mixture`, up to eight types) on the thirteen box
     rates per 100, standardised within each season so a type is a role relative to its era, fitted on
     player-seasons with 500+ possessions and used to place every one;
  3. context: age, years in the league, era, his team's quality that season, and whether he changed teams the
     season before and the season after;
  4. how the rating was built: fifths of the prior, of how far his own games moved him from it (signed and in
     size), of the swap adjustment's move, and of his rating on the other side of the ball;
  5. combinations: a three-level regression tree on everything above, fitted on half the players and read on the
     other half, so every subset it reports is measured on players it never saw; and the share of the adjusted
     miss a gradient-boosted model on everything can predict for players it never saw (0% = the misses carry no
     pattern beyond what is adjusted for).

With several hundred cuts about one in 370 passes |z| 3 by chance; the run prints how many pass against how many
chance alone would give.  z: players as the unit (all of a player's seasons in a group are one cluster).

Writes outputs/miss_by_group_<tag>.html (every cut, plain words), outputs/csv/miss_by_group_<tag>.csv and
outputs/csv/miss_players_<tag>.csv (every player-season's adjusted miss).  Reads only.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _cli import check_flags, flag  # noqa: E402
from eracoef import singleyear as sy  # noqa: E402
from eracoef.archetype import cluster_proba, fit_mixture  # noqa: E402
from eracoef.config import load_config  # noqa: E402
from eracoef.glossary import plain  # noqa: E402
from eracoef.seasons import drop_untrainable  # noqa: E402

N_CELLS = 6                        # levels for the adjustment's own check
N_GROUPS = 5                       # fifths
STATS = list(dict.fromkeys(sy.PRIOR_FEATURES + sy.OFFC + sy.NET + sy.CLOSENESS + sy.PO_SHARE))
TYPE_RATES = ["fg2m", "fg2_miss", "fg3m", "fg3_miss", "ftm", "ft_miss", "orb", "drb", "ast", "tov", "stl", "blk",
              "pf"]
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


def name(code: str) -> str:
    return BUILT.get(code) or plain(code)


def levels(x: pd.Series, n: int) -> np.ndarray:
    """0..n-1 equal-count levels of `x` (ties broken by order, so every level is the same size)."""
    return pd.qcut(pd.Series(x).rank(method="first"), n, labels=False).to_numpy()


def cluster_mean(values: np.ndarray, players: np.ndarray, centre: float) -> tuple:
    """The mean of `values`, and its z against `centre` with each player's rows as one cluster."""
    mean = float(np.mean(values))
    by_player = pd.Series(values - mean).groupby(players).sum().to_numpy()
    se = float(np.sqrt(np.sum(by_player ** 2))) / len(values)
    return mean, (mean - centre) / se if se > 0 else np.nan


def adjusted_misses(rows: pd.DataFrame) -> pd.DataFrame:
    """Each row's squared miss against the expected squared miss for its rating level and its evidence.

    The expectation is a gradient-boosted Poisson model of the squared miss on the rating level, his on-floor
    possessions over the three seasons (`with_poss`), his team's neighbouring games without him (`without_poss`)
    and the season -- cross-fitted over players, so no player's own miss sets his own expectation.  The two
    possession counts go in separately because the correction is identified by both.  A first pass (2026-10-03)
    expected the miss from 6 x 6 cells of the rating level and ONE combined evidence number, `with x without /
    (with + without)`; it left starters, heavy-minute and high-usage players 15-40% above expectation and
    bench players 20-30% below, and the minutes check read far from 1.  With the two counts separate the minutes
    check is flat (0.97-1.06) and those groups vanish.  Since `with_poss` is close to minutes, "starters are
    harder" cannot be told apart from "the correction sees starters more clearly" with this measure."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import GroupKFold

    rows = rows.reset_index(drop=True).copy()
    rows["rated"] = rows.rating * rows.rating_scale
    rows["evidence"] = rows.with_poss * rows.without_poss / (rows.with_poss + rows.without_poss)
    rows["rating_level"] = levels(rows.rated, N_CELLS)
    rows["evidence_level"] = levels(rows.evidence, N_CELLS)
    X = np.column_stack([rows.rated, np.log1p(rows.with_poss), np.log1p(rows.without_poss), rows.season])
    squared = rows.alpha_good.to_numpy(float) ** 2
    expected = np.empty(len(rows))
    for fit, held in GroupKFold(n_splits=5).split(X, squared, rows.player_id):
        model = HistGradientBoostingRegressor(loss="poisson", max_depth=3, learning_rate=0.05, max_iter=300,
                                              min_samples_leaf=200, random_state=0)
        expected[held] = model.fit(X[fit], squared[fit]).predict(X[held])
    expected *= np.mean(squared / expected)                      # the average ratio exactly 1
    rows["expected"] = expected
    rows["miss_ratio"] = squared / expected
    rows["lean"] = rows.alpha_good / np.sqrt(expected)          # signed, in expected misses
    return rows


def player_types(stats: pd.DataFrame, k: int = 8) -> tuple:
    """Each player-season's type and a plain description of every type.

    The thirteen box rates per 100 (uncentred, padded), standardised within each season so a type is a role
    relative to its era rather than to 1997 or 2026; a Bayesian Gaussian mixture fitted on the player-seasons
    with 500+ possessions (a 100-possession profile is mostly noise) and used to place every player-season."""
    cols = [f"raw_{c}" for c in TYPE_RATES]
    z = stats[cols].groupby(stats.season.to_numpy()).transform(lambda c: (c - c.mean()) / c.std(ddof=0))
    Z = z.to_numpy(float)
    fit = stats.poss.to_numpy(float) >= 500
    model, mean, sd = fit_mixture(Z[fit], k=k, seed=0, n_init=3)
    label = cluster_proba(model, mean, sd, Z).argmax(1)
    names, order = {}, pd.Series(label).value_counts().index           # biggest type first
    for rank, t in enumerate(order, start=1):
        profile = pd.Series(Z[label == t].mean(0), index=TYPE_RATES)
        high = profile.sort_values(ascending=False).index[:2]
        low = profile.sort_values().index[:2]
        names[t] = (f"type {rank}: most {plain(high[0])} and {plain(high[1])}, "
                    f"least {plain(low[0])} and {plain(low[1])}")
    return pd.Series(label, index=stats.index).map(names), names


def team_moves() -> pd.DataFrame:
    """Per player-season: whether he was on the same franchise the season before and the season after."""
    roles = pd.read_parquet(ROOT / "data" / "cache" / "roles_RSPO.parquet")
    played = roles[roles.poss_on > 0][["player_id", "season", "team_id", "poss_on"]].copy()
    played["team_id"] = sy.franchise(played.team_id.to_numpy(), played.season.to_numpy())
    main = (played.groupby(["player_id", "season", "team_id"], as_index=False).poss_on.sum()
            .sort_values("poss_on").drop_duplicates(["player_id", "season"], keep="last")
            [["player_id", "season", "team_id"]])
    first, last = int(main.season.min()), int(main.season.max())
    nxt = main.assign(season=main.season - 1).rename(columns={"team_id": "team_next"})
    prv = main.assign(season=main.season + 1).rename(columns={"team_id": "team_prev"})
    out = main.merge(nxt, on=["player_id", "season"], how="left").merge(prv, on=["player_id", "season"], how="left")
    out["next_season"] = np.select(
        [out.season == last, out.team_next.isna(), out.team_next == out.team_id],
        ["no next season yet", "out of the league next season", "same team next season"],
        "new team next season")
    out["previous_season"] = np.select(
        [out.season == first, out.team_prev.isna(), out.team_prev == out.team_id],
        ["no earlier season in the data", "new to the league (or back from a gap)", "same team as the season before"],
        "changed teams since the season before")
    return out[["player_id", "season", "next_season", "previous_season"]]


def group_row(rows, mask, players, side, kind, label) -> dict:
    extra, z = cluster_mean(rows.miss_ratio.to_numpy()[mask], players[mask], 1.0)
    lean, lean_z = cluster_mean(rows.lean.to_numpy()[mask], players[mask], 0.0)
    return dict(side=side, kind=kind, group=label, rows=int(mask.sum()), miss_vs_expected=extra, z=z,
                lean=lean, lean_z=lean_z)


def all_cuts(rows: pd.DataFrame, side: str) -> pd.DataFrame:
    players = rows.player_id.to_numpy()
    out = []
    continuous = [(c, "statistic") for c in STATS if c in rows.columns] + [(c, "how the rating was built")
                                                                          for c in BUILT]
    for code, kind in continuous:
        if rows[code].nunique() < N_GROUPS:
            continue
        kind = "context" if code == "team_net" else kind
        share = rows.groupby("season")[code].rank(pct=True, method="average")
        group = np.clip(np.ceil(share * N_GROUPS), 1, N_GROUPS).astype(int).to_numpy()
        for g in range(1, N_GROUPS + 1):
            m = group == g
            if m.sum() >= 50:
                out.append(group_row(rows, m, players, side, kind, f"{FIFTH[g]} of {name(code)}"))
    for label in sorted(rows.player_type.dropna().unique()):
        m = (rows.player_type == label).to_numpy()
        if m.sum() >= 50:
            out.append(group_row(rows, m, players, side, "player type", label))
    context = {
        "age": pd.cut(rows.age, [0, 23, 27, 31, 99], right=False,
                      labels=["age under 23", "age 23-26", "age 27-30", "age 31 and over"]),
        "years": pd.cut(rows.exp_yrs, [-1, 0.5, 3.5, 8.5, 99],
                        labels=["rookie", "1-3 years in the league", "4-8 years in the league", "9+ years in the league"]),
        "era": pd.cut(rows.season, [1996, 2005, 2015, 2026], labels=["1997-2005", "2006-2015", "2016-2026"]),
        "next": rows.next_season, "previous": rows.previous_season,
    }
    for values in context.values():
        for label in [v for v in pd.Series(values).dropna().unique()]:
            m = (pd.Series(values) == label).to_numpy()
            if m.sum() >= 50:
                out.append(group_row(rows, m, players, side, "context", str(label)))
    return pd.DataFrame(out)


def subset_search(rows: pd.DataFrame, features: list) -> tuple:
    """A three-level tree on half the players, its leaves measured on the other half; and the held-out share of
    the adjusted miss a gradient-boosted model on everything can predict."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.metrics import d2_tweedie_score
    from sklearn.model_selection import GroupKFold, GroupShuffleSplit
    from sklearn.tree import DecisionTreeRegressor

    types = pd.get_dummies(rows.player_type, prefix="", prefix_sep="").astype(float)
    X = pd.concat([rows[features].astype(float), types], axis=1)
    X = X.fillna(X.median())
    labels = [name(c) for c in features] + [f"is {t.split(':')[0]}" for t in types.columns]
    y, players = rows.miss_ratio.to_numpy(float), rows.player_id.to_numpy()
    fit, held = next(GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=0).split(X, y, players))
    tree = DecisionTreeRegressor(max_depth=3, min_samples_leaf=300, random_state=0).fit(X.iloc[fit], y[fit])
    t = tree.tree_

    def path_to(leaf):
        rules, node = [], 0
        parent = {}
        for n in range(t.node_count):
            if t.children_left[n] >= 0:
                parent[t.children_left[n]] = (n, "<=")
                parent[t.children_right[n]] = (n, ">")
        node = leaf
        while node in parent:
            up, how = parent[node]
            label = labels[t.feature[up]]
            if label.startswith("is type"):
                rules.append(label if how == ">" else label.replace("is ", "is not "))
            else:
                cut = t.threshold[up]
                # plain numbers: 2,730 not 2.73e+03 (the owner read the scientific form as a typo)
                rules.append(f"{label} {how} {cut:,.0f}" if abs(cut) >= 1000 else f"{label} {how} {cut:.3g}")
            node = up
        return " and ".join(reversed(rules))

    leaf_held = tree.apply(X.iloc[held])
    leaves = []
    rows_held = rows.iloc[held].reset_index(drop=True)
    for leaf in np.unique(leaf_held):
        m = leaf_held == leaf
        if m.sum() < 50:
            continue
        r = group_row(rows_held, m, rows_held.player_id.to_numpy(), "", "subset", path_to(leaf))
        r["miss_in_fitting_half"] = float(t.value[leaf].ravel()[0])
        leaves.append(r)
    expected = np.empty(len(rows))
    for a, b in GroupKFold(n_splits=5).split(X, y, players):
        model = HistGradientBoostingRegressor(loss="poisson", max_depth=3, learning_rate=0.05, max_iter=200,
                                              min_samples_leaf=200, random_state=0)
        expected[b] = model.fit(X.iloc[a], y[a]).predict(X.iloc[b])
    return pd.DataFrame(leaves), float(d2_tweedie_score(y, expected, power=1))


def describe(frame: pd.DataFrame) -> pd.DataFrame:
    """The plain-words table: who, how much more miss, which way it leans."""
    def lean_words(row):
        if not np.isfinite(row.lean_z) or abs(row.lean_z) < 3:
            return "no clear lean"
        return "rated too low" if row.lean > 0 else "rated too high"
    return pd.DataFrame({
        "players": frame.group.to_numpy(),
        "extra miss": [f"{100 * (v - 1):+.0f}%" for v in frame.miss_vs_expected],
        "z": frame.z.round(1).to_numpy(),
        "lean": [lean_words(r) for r in frame.itertuples()],
        "of which lean": [f"{100 * v ** 2:.0f}%" for v in frame.lean],
        "player-seasons": frame.rows.to_numpy(),
    })


def main() -> None:
    check_flags()
    tag = flag("tag", "incumbent")
    alpha = pd.read_parquet(ROOT / flag("alpha", "outputs/tradeset_swapadj_within_alpha.parquet"))
    alpha = alpha[alpha.eligible & (alpha.without_poss >= 1.0)]
    ratings = pd.read_parquet(ROOT / flag("ratings", "outputs/season_ratings_swapadj_within.parquet"))
    # the trust boundary (src/eracoef/seasons.py): a season still being played reaches no fit
    panel, _ = drop_untrainable(pd.read_parquet(ROOT / "outputs" / "role_panel_season.parquet"),
                                load_config(ROOT / "config.yaml"), what="the season panel")
    panel = panel[panel.poss > 0]
    moves = team_moves()
    stats_o = sy.season_frame(panel[panel.side == "O"], sy.PRIOR_FEATURES).drop_duplicates(["player_id", "season"])
    stats_o["player_type"], type_names = player_types(stats_o)
    print("player types (fitted on player-seasons with 500+ possessions, rates standardised within each season):")
    counts = stats_o.player_type.value_counts()
    for label in type_names.values():
        print(f"  {label} ({counts.get(label, 0):,} player-seasons)")
    groups, people, subsets, html = [], [], [], []
    for side, (short, own, other) in SIDES.items():
        stats = sy.season_frame(panel[panel.side == short], sy.PRIOR_FEATURES).drop_duplicates(["player_id", "season"])
        keep = ["player_id", "season", "poss"] + [c for c in STATS + [f"raw_{c}" for c in TYPE_RATES]
                                                  if c in stats.columns and c not in ("poss",)]
        stats = stats[list(dict.fromkeys(keep))]
        # his team's quality that season: on-court and off-court records together, each side per 100
        team_o = ((stats.onc_o * stats.onc_poss_o + stats.offc_o * stats.offc_poss_o)
                  / (stats.onc_poss_o + stats.offc_poss_o))
        team_d = ((stats.onc_d * stats.onc_poss_d + stats.offc_d * stats.offc_poss_d)
                  / (stats.onc_poss_d + stats.offc_poss_d))
        stats["team_net"] = team_o - team_d
        built = ratings[["player_id", "season", f"prior_{own}", f"u_{own}", f"c_{own}", f"rating_{other}"]].rename(
            columns={f"prior_{own}": "prior", f"u_{own}": "games", f"c_{own}": "swap", f"rating_{other}": "other_side"})
        built["games_size"] = built.games.abs()
        rows = (alpha[alpha.side == side]
                .merge(stats, on=["player_id", "season"], how="inner")
                .merge(stats_o[["player_id", "season", "player_type"]], on=["player_id", "season"], how="left")
                .merge(built, on=["player_id", "season"], how="left")
                .merge(moves, on=["player_id", "season"], how="left"))
        rows = adjusted_misses(rows)
        print(f"\n=== {side.upper()}: {len(rows):,} player-seasons, {rows.player_id.nunique():,} players "
              f"(one-team players whose team played some neighbouring games without them)")
        typical = rows.groupby("rating_level").alpha_good.apply(lambda a: float(np.sqrt(np.mean(a ** 2))))
        print("  typical miss (points per 100) by rating level, worst sixth to best: "
              + ", ".join(f"{v:.2f}" for v in typical))
        for label, level in (("rating level", rows.rating_level), ("evidence level", rows.evidence_level),
                             ("share of team possessions played", levels(rows.poss_pct, N_CELLS))):
            print(f"  after the adjustment, miss against expected by {label}: "
                  + ", ".join(f"{v:.2f}" for v in rows.miss_ratio.groupby(level).mean()))
        table = all_cuts(rows, side)
        groups.append(table)
        passes = int((table.z.abs() >= CHANCE_Z).sum())
        print(f"  {len(table)} cuts; {passes} pass |z| 3 against {CHANCE_RATE * len(table):.1f} expected by chance "
              f"({int((table.z >= CHANCE_Z).sum())} miss more, {int((table.z <= -CHANCE_Z).sum())} miss less)")
        worst = table[table.z >= CHANCE_Z].sort_values("miss_vs_expected", ascending=False)
        best = table[table.z <= -CHANCE_Z].sort_values("miss_vs_expected")
        features = [c for c in STATS if c in rows.columns] + list(BUILT) + ["age", "exp_yrs"]
        leaves, d2 = subset_search(rows, list(dict.fromkeys(features)))
        leaves["side"] = side
        subsets.append(leaves)
        with pd.option_context("display.width", 250, "display.max_colwidth", 120):
            print("\n  where we miss MORE than players rated the same with the same evidence (|z| 3 or more):")
            print(describe(worst).to_string(index=False) if len(worst) else "  none")
            print("\n  where we miss LESS:")
            print(describe(best).to_string(index=False) if len(best) else "  none")
            print("\n  combinations: a three-level tree fitted on half the players, measured on the other half:")
            print(describe(leaves.sort_values("miss_vs_expected", ascending=False)).to_string(index=False))
            print(f"\n  how much of the adjusted miss can be predicted for players never seen, from everything above: "
                  f"{100 * d2:.1f}%")
        rows["side"] = side
        people.append(rows[["side", "player_id", "player_name", "season", "player_type", "rated", "alpha_good",
                            "evidence", "miss_ratio", "lean"]])
        recent = rows[rows.season >= 2022].sort_values("miss_ratio", ascending=False).head(12)
        print("\n  the player-seasons we miss most since 2022, against expectation:")
        print(pd.DataFrame({
            "player": recent.player_name, "season": recent.season, "rated": recent.rated.round(2),
            "games say": [f"{v:+.2f} {'better' if v > 0 else 'worse'}" for v in recent.alpha_good],
            "times the expected miss": np.sqrt(recent.miss_ratio).round(1)}).to_string(index=False))
        html.append((side, len(table), passes, CHANCE_RATE * len(table), d2,
                     describe(table.sort_values("z", ascending=False)),
                     describe(leaves.sort_values("miss_vs_expected", ascending=False))))
    out = ROOT / "outputs" / "csv"
    out.mkdir(parents=True, exist_ok=True)
    pd.concat(groups, ignore_index=True).to_csv(out / f"miss_by_group_{tag}.csv", index=False)
    pd.concat(subsets, ignore_index=True).to_csv(out / f"miss_subsets_{tag}.csv", index=False)
    pd.concat(people, ignore_index=True).to_csv(out / f"miss_players_{tag}.csv", index=False)
    write_html(ROOT / "outputs" / f"miss_by_group_{tag}.html", html, type_names)
    print(f"\nwrote outputs/miss_by_group_{tag}.html, outputs/csv/miss_by_group_{tag}.csv, miss_subsets_{tag}.csv "
          f"and miss_players_{tag}.csv")


def write_html(path: Path, parts: list, type_names: dict) -> None:
    style = ("<style>body{font-family:system-ui,sans-serif;margin:24px;max-width:1100px}"
             "table{border-collapse:collapse;font-size:13px;margin:8px 0 24px}"
             "th,td{border-bottom:1px solid #ddd;padding:3px 10px;text-align:left}"
             "th{background:#f3f3f3;position:sticky;top:0}</style>")
    body = ["<h1>Where the ratings miss most, against expectation</h1>",
            "<p>A player's miss: how far his team's games in the rated season and the seasons either side say his "
            "rating is off. Each miss is compared with what players rated the same, with the same evidence, miss: "
            "+20% means 20% more squared miss than expected. <b>Lean</b> = missed mostly one way; <b>of which "
            "lean</b> = the part of the miss that is one-directional. Every cut is listed, sorted by z.</p>",
            "<h2>Player types</h2><ul>" + "".join(f"<li>{v}</li>" for v in type_names.values()) + "</ul>"]
    for side, n, passes, chance, d2, table, leaves in parts:
        body.append(f"<h2>{side.capitalize()}</h2><p>{n} cuts; {passes} pass |z| 3 against {chance:.1f} expected by "
                    f"chance. Share of the adjusted miss predictable for unseen players from everything here: "
                    f"{100 * d2:.1f}%.</p>")
        body.append("<h3>Combinations (fitted on half the players, measured on the other half)</h3>")
        body.append(leaves.to_html(index=False, border=0))
        body.append("<h3>Every cut</h3>")
        body.append(table.to_html(index=False, border=0))
    path.write_text("<html><head><meta charset='utf-8'>" + style + "</head><body>" + "".join(body)
                    + "</body></html>", encoding="utf-8")


if __name__ == "__main__":
    main()
