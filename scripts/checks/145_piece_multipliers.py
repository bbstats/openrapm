"""Check 145, how much of each piece of a rating holds up on games the rating never saw (the screen for experiment 61).

Step 4 (prior_shrink.py) multiplies the box-score part of a rating by how much of it holds up on the season's own
held-out games (0.70 on offence, 0.93 on defence) and leaves the games' part, u, at 1.  u has only ever been
multiplied whole, which tied (notes/DECISIONS.md, experiment 33b).  This check splits it into pieces and gives each
piece its own multiplier, to see whether the pieces hold up differently.  It changes no rating.

Words used below
    piece         a part of a player's rating that adds up with its siblings to the whole: u split by the rows it came
                  from (on court; sat, his team's rows without him in games he played; missed, his team's rows in games
                  he missed; team SOS, every other row) or by the points the rows are made of (twos, threes, free
                  throws), or the box-score part split by the player's body or role (`openrapm folds --pieces` writes
                  them, decomposition.split_pieces)
    family        a set of pieces that add up to u (or to the prior part): `source`, `outcome`, `size`, `certainty`,
                  `height`, `role`; `step4` is the shipped step (prior part free, u held at 1) and `u_whole` is u
                  with one multiplier
    multiplier    what a piece is multiplied by so the ratings predict the season's held-out team-games best: 1 means
                  the piece holds up exactly, 0.5 means half of it does
    fold          a season rated from three quarters of its games and scored on the rest (`openrapm folds`), twelve a
                  season
    size groups   the games' part's |u| / posterior standard deviation, grouped by possessions in each fold and side:
                  the lower half of the possessions, the next 40% and the top 10%.  (The first plan cut it at 1 and 2;
                  under the ridge's penalty |u| / sd is under 1 for 97% of possessions and never over 2, so no season
                  would have filled a bin)
    certainty     the posterior standard deviation of u in thirds, by possessions: small for players with many
                  possessions
    height, role  the box-score part split by thirds, by possessions, of height and of share of team possessions

How.  Each fold's test team-games are fitted exactly as step 4 fits them (prior_shrink.fit_multipliers): the points per
100 possessions as scored, minus what the unrated players add, on the sums over the ten players of share x piece, with
the fold's own level, home edge, playoff and playoff-home columns free.  The defence's pieces enter in points allowed.
Here every piece's multiplier is free in one fit, per season, over its twelve folds.  The season is the unit: the
mean over the 30 seasons of a multiplier, its standard error over the seasons, and for a pair of pieces the paired
difference's t.  The `step4` family is the shipped fit and is held to `prior_shrink.fit_multipliers` season by
season, to 1e-8, as the check runs.

The held-out error. A season's multipliers fitted on the other seasons (the 8 nearest fold seasons, the test rule: not
the season or its neighbours, as step 4's are) and scored on the season's own held-out games, against the shipped
step 4 and against `u_whole`, in the units of the year-over-year test (game_armse; below 0 is better).

The gate, registered 2026-10-14 before the first run.  A family passes on a side when some pair of its pieces has
multipliers that differ by at least 0.2 (mean over the seasons), at |t| >= 3 over the seasons, and the difference has
the same sign at |t| >= 2 in each half (1997-2011 and 2012-2026).  A family passes when it passes on both sides: the
statistical reason is the same on both, so one method (the owner's rule, same reason same method); a family that
passes on one side is reported and left to the owner.  The held-out error is reported beside the gate and is not part
of it.

Writes outputs/csv/piece_multipliers_<tag>.csv (every piece's multiplier, mean and standard error over seasons, and by
half), _pairs.csv (every pair), _seasons.csv (every season's multipliers), _heldout.csv (each family's held-out error
against the shipped step 4 and against u_whole) and outputs/piece_multipliers_<tag>_decision.json.

    python scripts/checks/145_piece_multipliers.py --folds nocareer_pieces            (a few minutes)
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from openrapm import decomposition, prior_shrink
from openrapm.commands.console import one_thread, use_utf8
from openrapm.data.within_folds import check_fold_files, read_fold

SIDES = ("O", "D")
ROW_PIECES = list(decomposition.ROW_PIECES)
OUTCOME_PIECES = list(decomposition.OUTCOME_PIECES)
HALVES = {"early": (1997, 2011), "late": (2012, 2026)}
ARMSE = float(np.sqrt(2.0 / np.pi))
MIN_DIFFERENCE, MIN_T, MIN_HALF_T = 0.2, 3.0, 2.0
NEAREST = 8                         # step 4's fold seasons (settings.ShrinkSettings.nearest)
REFERENCE_TOLERANCE = 1e-8          # `step4` against prior_shrink.fit_multipliers
SIZE_EDGES, SIZE_NAMES = (0.5, 0.9), ("low", "mid", "top")
THIRD_EDGES, THIRD_NAMES = (1 / 3, 2 / 3), ("low", "mid", "high")

# family -> (the columns in its fit, per side; "response": the fit's response, with or without u held at 1)
FAMILIES = {
    "step4": dict(columns=["prior"], held=True),
    "u_whole": dict(columns=["prior", "u"]),
    "source": dict(columns=["prior", *ROW_PIECES]),
    "outcome": dict(columns=["prior", *OUTCOME_PIECES]),
    "size": dict(columns=["prior", *[f"u_size_{name}" for name in SIZE_NAMES]]),
    "certainty": dict(columns=["prior", *[f"u_sd_{name}" for name in THIRD_NAMES]]),
    "height": dict(columns=[*[f"prior_height_{name}" for name in THIRD_NAMES], "u"]),
    "role": dict(columns=[*[f"prior_role_{name}" for name in THIRD_NAMES], "u"]),
}
# the pieces of each family that are compared pairwise
PAIRED = {
    "source": ROW_PIECES, "outcome": OUTCOME_PIECES,
    "size": [f"u_size_{name}" for name in SIZE_NAMES], "certainty": [f"u_sd_{name}" for name in THIRD_NAMES],
    "height": [f"prior_height_{name}" for name in THIRD_NAMES], "role": [f"prior_role_{name}" for name in THIRD_NAMES]}


# ------------------------------------------------------------------------------------------------ one fold
def weighted_groups(values, weights, edges) -> np.ndarray:
    """Each value's group, 0, 1, 2, ...: by the share of the weight below the value's middle (ties in the order given),
    cut at `edges`, which ascend inside (0, 1).  With no weight at all everyone is in group 0.

    Example: values 1, 2, 3, 4 with weights 1, 1, 1, 1 and edges (0.5,): shares 0.125, 0.375, 0.625, 0.875, so groups
    0, 0, 1, 1.
    """
    values, weights = np.asarray(values, dtype=float), np.asarray(weights, dtype=float)
    if weights.sum() <= 0:
        return np.zeros(len(values), dtype=int)
    order = np.argsort(values, kind="stable")
    share = (np.cumsum(weights[order]) - weights[order] / 2.0) / weights.sum()
    groups = np.empty(len(values), dtype=int)
    groups[order] = np.searchsorted(np.asarray(edges), share, side="right")
    return groups


def side_values(fold, pieces: pd.DataFrame, side: str) -> dict:
    """{piece: one value per rated player of the side, in the fold's Z column order, positive good}: the prior part, u,
    its pieces and the grouped versions the families use."""
    rows = fold.rows[side]
    got = pieces[pieces.side == side].set_index("player_id").reindex(rows.player_id.to_numpy())
    needed = ["u", *ROW_PIECES, *OUTCOME_PIECES, "sd_u"]
    if got[needed].isna().any().any():
        raise SystemExit(f"{fold.stem} {side}: a rated player has no pieces")
    prior, u = (rows.prior_off, rows.u_off) if side == "O" else (rows.prior_def, rows.u_def)
    prior, u = prior.to_numpy(float), u.to_numpy(float)
    if np.abs(got.u.to_numpy(float) - u).max() > 1e-9:
        raise SystemExit(f"{fold.stem} {side}: the pieces' games' part is not the fold's")
    weight = (rows.poss_off if side == "O" else rows.poss_def).to_numpy(float)
    values = {"prior": prior, "u": u}
    values.update({name: got[name].to_numpy(float) for name in (*ROW_PIECES, *OUTCOME_PIECES)})
    sd = got.sd_u.to_numpy(float)
    height, role = rows.height_in.to_numpy(float), rows.team_possession_share.to_numpy(float)
    for label, groups, names, source in (
            ("u_size", weighted_groups(np.abs(u) / sd, weight, SIZE_EDGES), SIZE_NAMES, u),
            ("u_sd", weighted_groups(sd, weight, THIRD_EDGES), THIRD_NAMES, u),
            ("prior_height", weighted_groups(height, weight, THIRD_EDGES), THIRD_NAMES, prior),
            ("prior_role", weighted_groups(role, weight, THIRD_EDGES), THIRD_NAMES, prior)):
        for k, name in enumerate(names):
            values[f"{label}_{name}"] = np.where(groups == k, source, 0.0)
    return values


def family_ratings(fold, pieces: pd.DataFrame, family: str, multipliers: np.ndarray) -> dict:
    """{"O", "D"}: each rated player's rating with the family's pieces times their multipliers, positive good, in the
    fold's row order, re-centred so its possession-weighted mean is zero as step 4 re-centres (check 151 scores it).

    `multipliers` is one family's solved vector, offence's pieces then defence's (`FAMILIES[family]["columns"]`); a
    family whose games' part is held (`step4`) adds u at 1.  NaN multipliers, pieces nobody had, count as 0.
    """
    spec, out = FAMILIES[family], {}
    columns, multipliers = spec["columns"], np.nan_to_num(np.asarray(multipliers, dtype=float))
    for side_index, side in enumerate(SIDES):
        values = side_values(fold, pieces, side)
        rating = sum(multipliers[side_index * len(columns) + k] * values[piece] for k, piece in enumerate(columns))
        if spec.get("held"):
            rating = rating + values["u"]
        weight = fold.rows[side].poss_off.to_numpy(float)
        out[side] = rating - np.average(rating, weights=weight)
    return out


def fold_columns(fold, pieces: pd.DataFrame) -> tuple[np.ndarray, list, np.ndarray]:
    """(the matrix of every piece's team-game sums, its column names "<piece>|<side>", the response before u).

    A column is sum over the ten players of share x piece, the defence's in points allowed (negated), as
    prior_shrink.fold_terms builds its sums.  The response is the team-games' points per 100 as scored minus what the
    players without a rating add: step 4's response before its games' part is taken off.
    """
    names, columns = [], []
    for side, sign in (("O", 1.0), ("D", -1.0)):
        for name, value in side_values(fold, pieces, side).items():
            names.append(f"{name}|{side}")
            columns.append(sign * np.asarray(fold.Z[side] @ value).ravel())
    matrix = np.column_stack(columns)
    index = {name: k for k, name in enumerate(names)}
    rated = sum(matrix[:, index[f"{name}|{side}"]] for name in ("prior", "u") for side in SIDES)
    return matrix, names, fold.y - (fold.base - rated)


def residual(matrix: np.ndarray, controls: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """`matrix` with the controls' weighted least-squares fit taken out, column by column (the fold's own level, home
    edge and playoff columns, as step 4 fits them; a fold without playoff games has an empty column, which the least
    squares step leaves out)."""
    root = np.sqrt(weight)[:, None]
    coefficients = np.linalg.lstsq(controls * root, matrix * root, rcond=None)[0]
    return matrix - controls @ coefficients


class Season:
    """One season's normal equations for every family, summed over its folds."""

    def __init__(self):
        self.sums = {name: dict(gram=0.0, rhs=0.0, response=0.0) for name in FAMILIES}
        self.weight = 0.0

    def add(self, fold, pieces: pd.DataFrame) -> None:
        matrix, names, response = fold_columns(fold, pieces)
        index = {name: k for k, name in enumerate(names)}
        both = np.column_stack([matrix, response])
        cleaned = residual(both, fold.F, fold.w)
        cleaned_matrix, cleaned_response = cleaned[:, :-1], cleaned[:, -1]
        upper = cleaned_matrix[:, [index["u|O"], index["u|D"]]].sum(axis=1)
        for name, spec in FAMILIES.items():
            wanted = [index[f"{piece}|{side}"] for side in SIDES for piece in spec["columns"]]
            x = cleaned_matrix[:, wanted]
            y = cleaned_response - upper if spec.get("held") else cleaned_response
            weighted = fold.w[:, None] * x
            self.sums[name]["gram"] = self.sums[name]["gram"] + x.T @ weighted
            self.sums[name]["rhs"] = self.sums[name]["rhs"] + weighted.T @ y
            self.sums[name]["response"] += float(np.sum(fold.w * y * y))
        self.weight += float(fold.w.sum())


# --------------------------------------------------------------------------------------------- over the seasons
def solve(gram: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """The multipliers; a piece with nothing in it (a zero diagonal) is NaN, not 0."""
    coefficients = np.linalg.lstsq(gram, rhs, rcond=None)[0]
    coefficients[np.diag(gram) <= 0] = np.nan
    return coefficients


def paired_t(differences) -> tuple[float, float, int]:
    """(mean, t, n) of the per-season differences: the mean over its standard error, seasons as the unit."""
    values = np.asarray(differences, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) < 3:
        return float("nan"), float("nan"), len(values)
    mean, spread = float(values.mean()), float(values.std(ddof=1))
    return mean, (mean / (spread / np.sqrt(len(values))) if spread > 0 else float("inf")), len(values)


def half_of(seasons: np.ndarray, half: str) -> np.ndarray:
    first, last = HALVES[half]
    return (seasons >= first) & (seasons <= last)


def multiplier_tables(per_season: dict, seasons: np.ndarray) -> pd.DataFrame:
    """Every piece's multiplier: mean, standard error and median over the seasons, and the mean in each half."""
    rows = []
    for family, spec in FAMILIES.items():
        for side_index, side in enumerate(SIDES):
            for k, piece in enumerate(spec["columns"]):
                values = np.array([per_season[family][int(season)][side_index * len(spec["columns"]) + k]
                                   for season in seasons])
                kept = values[~np.isnan(values)]
                row = dict(family=family, side=side, piece=piece, seasons=len(kept), multiplier=float(np.mean(kept)),
                           se=float(kept.std(ddof=1) / np.sqrt(len(kept))), median=float(np.median(kept)))
                for half in HALVES:
                    row[half] = float(np.nanmean(values[half_of(seasons, half)]))
                rows.append(row)
    return pd.DataFrame(rows)


def passes_gate(mean: float, t: float, half_t: dict) -> bool:
    """The gate for one pair (the module docstring): a difference of at least 0.2 at |t| >= 3, the same sign at
    |t| >= 2 in each half."""
    if not (abs(mean) >= MIN_DIFFERENCE and abs(t) >= MIN_T):
        return False
    return all(np.sign(value) == np.sign(mean) and abs(value) >= MIN_HALF_T for value in half_t.values())


def pair_tables(per_season: dict, seasons: np.ndarray) -> pd.DataFrame:
    """Every pair of pieces in a family, each side: the mean difference of their multipliers, t over the seasons, t in
    each half, and whether the pair passes the gate."""
    rows = []
    for family, pieces in PAIRED.items():
        columns = FAMILIES[family]["columns"]
        for side_index, side in enumerate(SIDES):
            for first, second in itertools.combinations(pieces, 2):
                a, b = (side_index * len(columns) + columns.index(piece) for piece in (first, second))
                differences = np.array([per_season[family][int(season)][a] - per_season[family][int(season)][b]
                                        for season in seasons])
                mean, t, n = paired_t(differences)
                half_t = {half: paired_t(differences[half_of(seasons, half)])[1] for half in HALVES}
                rows.append(dict(family=family, side=side, first=first, second=second, seasons=n, difference=mean,
                                 t=t, **{f"t_{half}": value for half, value in half_t.items()},
                                 passes=passes_gate(mean, t, half_t)))
    return pd.DataFrame(rows)


def held_out(accumulated: dict, seasons: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(each family against the references, each family's error by season).

    A family's held-out error in a season: its multipliers fitted on the 8 nearest other seasons (the test rule),
    scored on the season's own held-out games, as game_armse.  The change is the family's error minus the
    reference's, season by season; below 0 is better.
    """
    errors = {}
    for family in FAMILIES:
        for season in seasons:
            used = prior_shrink.fold_seasons_used(int(season), [int(s) for s in seasons], "test", NEAREST)
            if not used:
                errors[(family, int(season))] = float("nan")
                continue
            gram = sum(accumulated[s].sums[family]["gram"] for s in used)
            rhs = sum(accumulated[s].sums[family]["rhs"] for s in used)
            multipliers = np.nan_to_num(solve(gram, rhs))
            sums = accumulated[int(season)].sums[family]
            squared = sums["response"] - 2.0 * multipliers @ sums["rhs"] + multipliers @ sums["gram"] @ multipliers
            errors[(family, int(season))] = ARMSE * np.sqrt(max(squared, 0.0) / accumulated[int(season)].weight)
    rows = []
    for family in FAMILIES:
        for reference in ("step4", "u_whole"):
            if family == reference:
                continue
            change = np.array([errors[(family, int(s))] - errors[(reference, int(s))] for s in seasons])
            mean, t, n = paired_t(change)
            rows.append(dict(family=family, against=reference, mean_change=mean, t=t, seasons=n,
                             seasons_better=int((change < 0).sum()),
                             early=float(change[half_of(seasons, "early")].mean()),
                             late=float(change[half_of(seasons, "late")].mean())))
    return pd.DataFrame(rows), pd.Series(errors).unstack(0)


def decide(pairs: pd.DataFrame) -> dict:
    """{family: {passes, sides, pairs}}: a family passes when it passes on both sides (the module docstring)."""
    out = {}
    for family in PAIRED:
        found = pairs[(pairs.family == family) & pairs.passes]
        sides = sorted(found.side.unique())
        out[family] = dict(passes=sides == sorted(SIDES), sides=sides,
                           pairs=[f"{r.side}: {r.first} - {r.second} {r.difference:+.2f} (t {r.t:+.1f})"
                                  for r in found.itertuples()])
    return out


def run_season(directory: Path, stems: list, tolerance_report: list) -> Season:
    """One season's folds into its normal equations, after checking `step4` against prior_shrink.fit_multipliers."""
    season, folds = Season(), []
    for stem in stems:
        fold = read_fold(directory, stem)
        season.add(fold, pd.read_parquet(directory / f"pieces_{stem}.parquet"))
        folds.append(fold)
    shipped = np.array(prior_shrink.fit_multipliers(folds))
    mine = solve(season.sums["step4"]["gram"], season.sums["step4"]["rhs"])
    tolerance_report.append(float(np.abs(shipped - mine).max()))
    return season


def report(multipliers: pd.DataFrame, pairs: pd.DataFrame, heldout: pd.DataFrame, decision: dict) -> None:
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    print("\n=== multipliers: mean over the 30 seasons (standard error) [median]; 1 = holds up exactly ===", flush=True)
    for family in FAMILIES:
        table = multipliers[multipliers.family == family].copy()
        table["multiplier (se) [median]"] = [f"{m:.2f} ({s:.2f}) [{d:.2f}]"
                                              for m, s, d in zip(table.multiplier, table.se, table["median"])]
        print(f"\n{family}", flush=True)
        print(table.pivot(index="piece", columns="side", values="multiplier (se) [median]").reindex(
            FAMILIES[family]["columns"]).to_string(), flush=True)
    print("\n=== pairs that pass the gate ===", flush=True)
    passing = pairs[pairs.passes]
    print(passing.round(3).to_string(index=False) if len(passing) else "  none", flush=True)
    print("\n=== the largest differences, passing or not ===", flush=True)
    order = pairs.reindex(pairs.difference.abs().sort_values(ascending=False).index).head(12)
    print(order.round(3).to_string(index=False), flush=True)
    print("\n=== held-out error, family minus reference (game_armse; below 0 is better) ===", flush=True)
    print(heldout.round(4).to_string(index=False), flush=True)
    print("\n=== the gate ===", flush=True)
    for family, verdict in decision.items():
        print(f"  {family}: {'passes' if verdict['passes'] else 'fails'} (sides: {verdict['sides'] or 'none'}) "
              f"{verdict['pairs']}", flush=True)


def run(args: argparse.Namespace) -> int:
    from openrapm.data.config import load_config, resolve
    cfg = load_config()
    outputs, directory = resolve(cfg, "outputs"), resolve(cfg, "within") / args.folds
    started = time.time()
    stems = check_fold_files(directory, (args.first, args.last))
    by_season: dict = {}
    for stem in stems:
        by_season.setdefault(int(stem[:4]), []).append(stem)
    seasons = np.array(sorted(by_season))
    print(f"{len(stems)} folds, seasons {seasons[0]}-{seasons[-1]} ({time.time() - started:.0f}s)", flush=True)
    accumulated, misses = {}, []
    for season in seasons:
        accumulated[int(season)] = run_season(directory, by_season[season], misses)
        print(f"  {season}: step 4 reproduced to {misses[-1]:.1e} ({time.time() - started:.0f}s)", flush=True)
    if max(misses) > REFERENCE_TOLERANCE:
        raise SystemExit(f"the `step4` fit does not reproduce prior_shrink.fit_multipliers: {max(misses):.1e}")
    per_season = {family: {int(season): solve(accumulated[int(season)].sums[family]["gram"],
                                              accumulated[int(season)].sums[family]["rhs"]) for season in seasons}
                  for family in FAMILIES}
    multipliers, pairs = multiplier_tables(per_season, seasons), pair_tables(per_season, seasons)
    heldout, errors = held_out(accumulated, seasons)
    decision = decide(pairs)
    csv = outputs / "csv"
    csv.mkdir(exist_ok=True)
    stem = f"piece_multipliers_{args.folds}"
    multipliers.to_csv(csv / f"{stem}.csv", index=False)
    pairs.to_csv(csv / f"{stem}_pairs.csv", index=False)
    heldout.to_csv(csv / f"{stem}_heldout.csv", index=False)
    long = [dict(family=family, season=int(season), side=side, piece=piece,
                 multiplier=per_season[family][int(season)][side_index * len(spec["columns"]) + k])
            for family, spec in FAMILIES.items() for season in seasons
            for side_index, side in enumerate(SIDES) for k, piece in enumerate(spec["columns"])]
    pd.DataFrame(long).to_csv(csv / f"{stem}_seasons.csv", index=False)
    (outputs / f"{stem}_decision.json").write_text(json.dumps(
        dict(decision=decision, reference_miss=max(misses), seasons=[int(s) for s in seasons]), indent=1),
        encoding="utf-8")
    report(multipliers, pairs, heldout, decision)
    print(f"done in {time.time() - started:.0f}s", flush=True)
    return 0


def main(argv=None) -> int:
    use_utf8()
    parser = argparse.ArgumentParser(description="How much of each piece of a rating holds up on held-out games.")
    parser.add_argument("--folds", required=True, metavar="TAG",
                        help="the folder under outputs/within/ that `openrapm folds --pieces` wrote")
    parser.add_argument("--first", type=int, default=1997, help="first season (default %(default)s)")
    parser.add_argument("--last", type=int, default=2026, help="last season (default %(default)s)")
    args = parser.parse_args(argv)
    with one_thread():
        return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
