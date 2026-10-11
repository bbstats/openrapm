"""Player types: a Bayesian Gaussian mixture over each player's per-36 box-score profile, and how far the ratings
sit from another source, type by type.

The site's bias page (`openrapm site --bias`) and tests/test_vs_consensus.py read these.  The types are fitted,
not hand-made: a formula such as "offensive rebounds + blocks - assists" is a guess about what a type is, and it
answers only the question it was shaped for.

Words used below
    profile     a player's per-36-minute rates (RATES): points, threes made, threes missed, free throws made,
                assists, offensive and defensive rebounds, steals, blocks, turnovers, fouls
    type        one component of the mixture; each player belongs to the component with the highest probability.
                Bayesian, so `k` is an upper bound: a component the data does not support has its weight driven
                toward zero rather than splitting a real type in two
    z-score     (value - mean) / standard deviation, with the mean and standard deviation of the seasons the
                mixture was fitted on
    signature   the four rates a type's average z-score is furthest from zero on, strongest first
    gap         a player's z-score on our rating minus his z-score on the other source's
    spread      the standard deviation, across types with at least ten players, of the mean gap on the total
                rating.  The total and not each side: a type whose offensive and defensive gaps are large and
                opposite is a disagreement about which end a player's value comes from, and only the total says
                a rating is wrong

Example (2026, the bias page): the type of Rudy Gobert, Donovan Clingan and Deandre Ayton averages z-scores of
+2.25 in offensive rebounds, -1.67 in threes missed, +1.60 in blocks and -1.55 in threes made, so its signature is
"+orb -fg3_miss +blk -fg3m", which the page names "Rim-protecting centers".

Model layer: everything here takes frames.  The caller loads the box scores.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# The per-36 profile the mixture clusters on: volume, shot diet, playmaking, the glass, defensive events.
# Every column is in data.box_scores.season_box's schema, so the caller hands that frame straight in.
RATES = ["pts", "fg3m", "fg3_miss", "ftm", "ast", "orb", "drb", "stl", "blk", "tov", "pf"]


def per36(box: pd.DataFrame, rates: list | None = None) -> pd.DataFrame:
    """One row per player: his per-36 rates over whatever rows `box` holds.  `minutes` is kept."""
    rates = list(rates or RATES)
    g = box.groupby("player_id", as_index=False)[["minutes", *rates]].sum()
    for c in rates:
        g[c] = g[c] / g.minutes.clip(lower=1) * 36.0
    return g


def fit_mixture(X: np.ndarray, k: int = 8, seed: int = 0, n_init: int = 5) -> tuple:
    """Fit a Bayesian Gaussian mixture and return `(model, mean, sd)` -- the standardisation included.

    Bayesian, not plain EM, so `k` is an upper bound rather than a choice: a component the data does not
    support has its weight driven toward zero instead of splitting a real cluster in two.

    **The mean and sd come back because a season scored by this fit has to be standardised by the
    TRAINING seasons' constants.**  Standardising the scored season by its own mean and sd would slide
    every player toward the middle of his own year and read a league-wide shift -- pace, the rise of the
    three -- as every player changing type.
    """
    from sklearn.mixture import BayesianGaussianMixture

    Z = np.asarray(X, dtype=float)
    mean, sd = Z.mean(0), np.where(Z.std(0) > 0, Z.std(0), 1.0)
    bgm = BayesianGaussianMixture(n_components=int(k), covariance_type="full",
                                  weight_concentration_prior=1.0 / int(k), max_iter=1000,
                                  n_init=int(n_init), random_state=int(seed)).fit((Z - mean) / sd)
    return bgm, mean, sd


def cluster_proba(model, mean: np.ndarray, sd: np.ndarray, X: np.ndarray) -> np.ndarray:
    """One row per player, one column per component: how much he belongs to each type.

    Standardised by the fit's own `mean` and `sd`, never by `X`'s, so a season the mixture never saw is
    placed against the seasons it learned from.
    """
    return model.predict_proba((np.asarray(X, dtype=float) - mean) / sd)


def fit_clusters(X: np.ndarray, k: int = 8, seed: int = 0, n_init: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Standardise, fit a Bayesian Gaussian mixture, return (label per row, weight per component).

    In-sample: fits and labels the same rows.  Use `fit_mixture` plus `cluster_proba` to fit on one set
    of seasons and place another."""
    bgm, mean, sd = fit_mixture(X, k=k, seed=seed, n_init=n_init)
    Z = (np.asarray(X, dtype=float) - mean) / sd
    return bgm.predict(Z), bgm.weights_


def cluster_gaps(labels: np.ndarray, gaps: pd.DataFrame, extra: pd.DataFrame | None = None) -> pd.DataFrame:
    """Mean gap per cluster.  `gaps`: columns `gap_off`, `gap_def`, `gap_total`, one row per player, in
    `labels` order.  `extra`: any per-player columns to average beside them (the per-36 rates, a name)."""
    d = gaps.reset_index(drop=True).assign(cluster=np.asarray(labels))
    if extra is not None:
        d = pd.concat([d, extra.reset_index(drop=True)], axis=1)
    num = [c for c in d.columns if c != "cluster" and pd.api.types.is_numeric_dtype(d[c])]
    out = d.groupby("cluster", as_index=False)[num].mean()
    out.insert(1, "n", d.groupby("cluster").size().to_numpy())
    return out.sort_values("gap_total").reset_index(drop=True)


def spread(table: pd.DataFrame, col: str = "gap_total", min_n: int = 10) -> float:
    """The statistic: sd across clusters of the mean gap, over clusters with at least `min_n` players.

    Clusters below `min_n` are dropped rather than weighted -- a five-player component's mean gap is noise,
    and the mixture produces one whenever `k` is set above what the data supports."""
    v = table.loc[table.n >= int(min_n), col].to_numpy(dtype=float)
    return float(np.std(v, ddof=1)) if len(v) > 1 else 0.0


def gap_columns(ours: pd.DataFrame, theirs: pd.DataFrame, pairs=(("rating_off", "adj_offense"),
                                                                 ("rating_def", "adj_defense"),
                                                                 ("rating_total", "adj_overall"))) -> pd.DataFrame:
    """`gap_<side>` = our z minus their z, side by side.  Both are standardised because the two sources are
    on different scales: the ratings' offensive spread is 1.19 per 100 against the blend's 1.96, and an
    unstandardised difference would read that as disagreement about every player."""
    def z(v):
        v = pd.Series(np.asarray(v, dtype=float))
        return (v - v.mean()) / v.std()

    return pd.DataFrame({f"gap_{side}": z(ours[mine]) - z(theirs[yours])
                         for (mine, yours), side in zip(pairs, ("off", "def", "total"))})


# ------------------------------------------------------------------------------- one table row per type
MIN_TYPE_PLAYERS = 10      # a five-player component's mean is noise, not a type
SHOWN_MEMBERS = 6          # the most-played and the most typical members listed per type
SHOWN_OUTLIERS = 3         # the least typical members listed per type


def signature(z_profile: pd.Series, n: int = 4) -> str:
    """The `n` rates a type's average z-score is furthest from zero on, strongest first, each with its sign.

    Four and not two because the two biggest scoring types (Jokic's and Brunson's) are both "+ftm +pts" and
    separate only on the fourth rate, rebounds against assists.

    Example: z-scores pts -0.40, orb +2.25, blk +1.60, fg3m -1.55, fg3_miss -1.67 (the rest nearer zero) give
    "+orb -fg3_miss +blk -fg3m".
    """
    order = z_profile.abs().sort_values(ascending=False).index[:n]
    return " ".join(f"{'+' if z_profile[c] > 0 else '-'}{c}" for c in order)


def members(group: pd.DataFrame, z: pd.DataFrame, centre: pd.Series) -> dict:
    """Who to show so a type's name can be judged: the most-played members, and the most and least typical.

    Typical = the smallest distance sqrt(sum over RATES of (z - the type's average z)^2).
    """
    distance = np.sqrt(((z.loc[group.index, RATES] - centre) ** 2).sum(axis=1))
    typical = group.assign(d=distance).sort_values("d")
    return {"most_played": group.sort_values("poss", ascending=False).player_name.head(SHOWN_MEMBERS).tolist(),
            "most_typical": typical.player_name.head(SHOWN_MEMBERS).tolist(),
            "least_typical": typical.player_name.tail(SHOWN_OUTLIERS).tolist()[::-1]}


def describe_types(players: pd.DataFrame, z: pd.DataFrame, names: dict) -> dict:
    """{type: its signature, average z-score profile, name and members}, from every player of the type.

    `players` has a `group` column (the type), `poss` and `player_name`; `z` holds the players' z-scores on
    RATES, on the same index.  `names` maps a signature to a hand-chosen name; a signature it lacks reads
    "UNNAMED" instead of borrowing a neighbour's name.  Computed once from the whole type and shared by every
    table, so a type keeps its name when another table covers fewer of its players.
    """
    described = {}
    for group, rows in players.groupby("group"):
        centre = z.loc[rows.index, RATES].mean()
        sig = signature(centre)
        described[int(group)] = {"signature": sig, "profile": centre.round(2).to_dict(),
                                 "name": names.get(sig, "UNNAMED"), **members(rows, z, centre)}
    return described


def bias_by_type(players: pd.DataFrame, described: dict, ours: str, theirs: str, bias: str) -> list:
    """One row per type: the two levels and the bias, possession-weighted, largest |bias| x possessions first.

        bias(type) = sum over its players of poss x bias / sum of poss

    Ordered by |bias| x possessions, how much of the league a type's bias applies to, because a 0.6 miss over a
    quarter of the possessions matters more than a 1.1 miss over a twentieth.  Players missing `ours` or
    `theirs` are left out, and a type left with fewer than MIN_TYPE_PLAYERS players gets no row.
    """
    rows = []
    for group, members_of in players.groupby("group"):
        members_of = members_of.dropna(subset=[ours, theirs])
        if len(members_of) < MIN_TYPE_PLAYERS:
            continue
        weights = members_of.poss.to_numpy(float)
        rows.append({"group": int(group), "players": len(members_of), "possessions": float(weights.sum()),
                     "bias": float(np.average(members_of[bias], weights=weights)),
                     "ours": float(np.average(members_of[ours], weights=weights)),
                     "theirs": float(np.average(members_of[theirs], weights=weights)),
                     **described[int(group)]})
    for row in rows:
        row["cost"] = abs(row["bias"]) * row["possessions"]
    rows.sort(key=lambda row: -row["cost"])
    return rows
