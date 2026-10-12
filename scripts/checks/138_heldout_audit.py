"""Check 138, the held-out audit: on games the ratings never saw, are players who are high on some input rated too
low or too high?

A rating system can lean on an input: give big men too little credit on defence, say, or steals too much.  If it
does, the games those players play miss the ratings' prediction in one direction.  This check measures that for
every input a box-score prior could read, and for every group of inputs that move together, on each season's own
held-back games.  Every Robustness-pass build was judged by it (notes/DECISIONS.md, "The Robustness pass").

Words used below
    fold          one season rated from three quarters of its games, the other quarter held back
                  (`openrapm folds`; read back by openrapm.data.within_folds)
    team-game     one team's offence in one held-back game: its points per 100 possessions as scored, weighted
                  by its possessions
    prediction    each rated player's share of the team-game's possessions times his rating, a stand-in rating
                  for players without one (held at the fold's own value), and the level and home edge, fitted
                  freely in each fold
    miss          a team-game's points per 100 minus its prediction
    axis          an input turned into a possession-weighted percentile within the fold and side (tied values
                  share the midpoint of their block), and that percentile's normal score (its standard normal
                  quantile)
    correction    an amount per player that depends only on his place on one axis; it enters a team-game the
                  way a rating does, share of the possessions times the correction
    lean          the correction the misses ask for: + means players high on the input are under-rated, - that
                  they are over-rated, in points per 100
    group         inputs that move together: average linkage on 1 - |rank correlation| over the incumbent's
                  whole-season rated rows with 500 or more possessions, cut at 0.4, 0.5 and 0.6.  A group's axis
                  is the first principal component of its members' normal scores, ranked again.  The groups and
                  their loadings are frozen in params/audit_groups.json (`--freeze=1` writes it, once, before the
                  first build; notes/DECISIONS.md, measurement trap 22)

The fit, per side, pooled over every fold (weighted least squares, weight = possessions):

    miss_g  =  level_f + home_f  +  sign x  sum over rated players i of  share_gi x correction(axis_i)

    g a team-game and f its fold; share_gi player i's share of g's possessions (each side's shares sum to 5);
    sign +1 on offence and -1 on defence (a correction raises a rating, and on defence that lowers points allowed)

Three shapes of correction, each read per side on its own:
    line      slope x normal score: points per 100 per standard deviation of the axis
    bend      a line plus the square of the normal score: the square's coefficient (a hump is negative, a U positive)
    tenths    one level per possession-weighted tenth of the axis: the top tenth minus the bottom tenth, points
              per 100

Example: on the weightfix folds (2017-2026) the defence row for the group "rebounds, defensive rebounds, height
(+1 more)" reads +1.77 with line z +8.2.  In held-back games the players in that group's top tenth are under-rated
on defence by 1.77 points per 100 against its bottom tenth, and the line sits 8.2 null standard deviations from its
null mean, past the threshold of 3.64, so (with the other checks below) the lean COUNTS.

The yardstick.  The z and the threshold come from a null that moves whole careers: every player is handed another
player's inputs (all of them, so the inputs keep their correlations), season by season along the donor's career (so
a player's inputs persist across his seasons as much as real ones do).  Each test's line and bend are recomputed for
every reassignment.  A test's z is its estimate minus its null mean over its null standard deviation, and the
threshold is the 95th percentile, over reassignments, of the largest |z| among the tests that can count (the
studentised max-T of Westfall and Young).  A season jackknife is printed beside it but does not decide: a player's
misfit carries over between seasons, so the jackknife's standard error is too small for anything a player carries
with him.  Every summary row carries the smallest line it could have detected: the threshold times the line's null
standard deviation.  `--draws=0 --threshold_from=<an earlier --out>` reads the threshold and each test's null from
that earlier audit instead of drawing again, so every candidate is judged on the same frozen yardstick.

The stages, the scale pinned to what ships:
    prior     the prior part alone, times step 4's multiplier (openrapm.prior_shrink, fitted without the fold's
              season); the season's games' part left out
    raw       the rating before step 4 (prior part + games' part, as the fold built it)
    shrunk    the rating as it ships before the swap step: the decision stage
At the shrunk stage, three more versions: `prior_free` (step 4's two multipliers refitted together with the lean),
`rating_free` (one free scale per side on the whole rating) and `nuisance` (pinned, plus log possessions and the
team-game shares of rated players and of players with under 500 possessions).

When a lean COUNTS: at the shrunk stage its line or bend z passes the threshold; its tenths reading is at least 0.1
per 100 from bottom to top; its line keeps its sign (and its size within a factor of two) under all three versions;
once the folds span more than one era (1997-2006, 2007-2016, 2017-2026) it keeps its sign in every era; a
`--deadline` folder's line does not contradict it at |z| 2 or more; and the trade set's line (below) does not
contradict it at |z| 2 or more.  Team context, playing time and score state are measured but never count, and are
left out of the threshold.

Why this and not check 137.  137 reads the with/without correction, which comes from the NEIGHBOURING seasons (it
teaches "peak seasons regress", and the owner's 2026-10-03 rule forbids fitting on it), scores luck-adjusted targets,
and its null shuffles inputs within a season, which inflates the z of anything a player carries from season to
season three to four times (body weight: 16.9 under that null, about 4 under this one).  Here: actual points on
the season's own games; each fold's REBUILT inputs on the axis (the full-season panel holds the held-out games' own
steals, a leak that turns the steal lean's z from -3.1 into -0.8); a null that moves whole careers; and one
threshold for all the tests together.

The trade set, a veto only.  Each decision test's line of the neighbouring-season miss (the trade set's alpha on
actual points, eligible rows; `--tradeset`, by default outputs/tradeset_lgb_noonc_rs_pts_alpha.parquet when it
exists) on the input's within-season normal score, with standard errors clustered by player.  It never fits anything.

The paired mode.  `--reference=<tag>[:<mult tag>]` reads the incumbent's folds on the same seasons and test games, and
`--control=<tag>[:<mult tag>]` the noise control's (a rebuild of the incumbent that changes only which players share
a fold when the box-score prior is fitted, so its movement is noise): each lean's change, with a season jackknife of
the paired difference; how far the ratings themselves moved along each axis; and the paired held-out error per fold.

Inputs from retired scripts, still read when present.  A fold folder holding openrapm_shipped.parquet (written by the
retired script 106) adds two stages, `shrunk106` and `swap`; the swap step's own lean is their difference.
`--correction=<name>` reads outputs/csv/<name>_coefficients_product.csv (a group correction from the retired script
140) and folds its lines into the prior part.  `--shares=2017,2021,2026` writes each group's share of the prior:
Shapley values with every group of correlated inputs as one player, against a background of the season's rated rows,
on the incumbent's saved prior models (LightGBM cannot give SHAP values for linear leaves); about a minute a season.

Reads   outputs/within/<tag>/ (each fold's json, players parquet and test npz), params/audit_groups.json, the season
        panel (gated by seasons.drop_untrainable), the trade set's alpha table, and with --draws=0 the earlier
        audit's outputs/<name>_thresholds.json and outputs/csv/<name>_null.csv.
Writes  outputs/csv/<out>.csv (every reading), <out>_tenths.csv (each tenth's level), <out>_summary.csv (one line
        per test and side at the decision stage, then offence plus defence); with a new null, <out>_null.csv and
        outputs/<out>_thresholds.json; with --reference, <out>_paired.csv and outputs/<out>_paired_error.json; with
        --shares, <out>_shares.csv.

The console: per side, the 30 largest tenths readings (+ = under-rated), each with its line z, bend z and joint z
(all groups' lines fitted together), marked COUNTS when the rule above holds; then offence plus defence (is the lean
real, or credit landing on the wrong side?); then, paired, the change in held-out team-game error (negative: the
candidate predicts better) and the 20 largest changes in a lean.

    python scripts/checks/138_heldout_audit.py --random=weightfix_within --mult_tag=weightfix_within \\
        --reference=weightfix_within --control=noise_regroup2_within --draws=0 \\
        --threshold_from=audit_lgb_noonc_2017 --out=audit_weightfix
"""
from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import norm

from openrapm.commands.console import one_thread, use_utf8
from openrapm.data.config import load_config, resolve
from openrapm.data.legacy_names import column_name, saved_models, translate_frame
from openrapm.data.seasons import drop_untrainable
from openrapm.data.within_folds import load_folds, read_fold
from openrapm.features import BOX_RATES, add_features, meaning
from openrapm.prior_shrink import fit_multipliers, multipliers_by_season

# ------------------------------------------------------------------------------------------- what is audited
# Every input a fold's player rows carry that a prior could read, by meaning (features.py says what each is).  The
# order is the frozen groups' order (params/audit_groups.json lists the codes in it).
CLASSES = {
    "box score": (*BOX_RATES, "points_per100", "shots_per100", "free_throw_attempts_per100",
                  "three_point_attempts_per100", "used_possessions_per100", "rebounds_per100",
                  "steals_plus_blocks_per100", "assists_minus_turnovers_per100", "threes_minus_twos_attempted_per100"),
    "shooting and shares": ("effective_fg_pct", "true_shooting_pct", "three_point_pct", "two_point_pct",
                            "free_throw_pct", "three_point_attempt_share", "free_throws_per_shot",
                            "assists_per_used_possession", "turnovers_per_used_possession", "off_rebound_share"),
    "shot quality": ("two_point_expected_pct", "three_point_expected_pct", "two_point_pct_over_expected",
                     "three_point_pct_over_expected", "expected_points_per_shot", "points_per_shot_over_expected"),
    "body": ("height_in", "weight_lb"),
    "age and career": ("age", "seasons_before", "career_possessions_before_k", "age_at_first_season",
                       "seasons_with_team", "teams_this_season", "draft_pick"),
    "playing time": ("team_possession_share", "games_started_share"),
    "score state": ("garbage_time_share", "closeness", "mean_abs_margin", "playoff_share"),
    "team context": ("on_court_scored_per100", "on_court_allowed_per100", "off_court_scored_per100",
                     "off_court_allowed_per100", "on_minus_off_scored_per100", "on_minus_off_allowed_per100",
                     "team_net_per100"),
}
MEASURED_ONLY = frozenset({"team context", "playing time", "score state"})   # measured, never counted
GROUPED = tuple(code for name, codes in CLASSES.items() if name != "team context" for code in codes)
CLASS_OF = {code: name for name, codes in CLASSES.items() for code in codes}
SIDES = ("O", "D")
SIGN = {"O": 1.0, "D": -1.0}           # a correction raises a player's rating; on defence that lowers points allowed
CUTS = (0.4, 0.5, 0.6)
GROUP_MIN_POSS = 500.0
BENCH_POSS = 500                        # the nuisance version's bench: players under 500 possessions on the side
TENTHS = 10
REFERENCE_TENTH = 4                     # the dropped level; the top-minus-bottom reading does not depend on it
NCOL = 2 + TENTHS - 1                   # each test's columns: z, z squared, the nine tenths kept
KIND_COLUMNS = {"line": (0,), "bend": (0, 1), "tenths": tuple(range(2, NCOL))}
STAGES = ("prior", "raw", "shrunk")
SWAP_STAGES = ("shrunk106", "swap")     # only when a folder holds the retired script 106's ratings
VERSIONS = ("prior_free", "rating_free", "nuisance")
ERAS = ((1997, 2006), (2007, 2016), (2017, 2026))
GROUPS_FILE = Path("params") / "audit_groups.json"
DEFAULT_TRADESET = "outputs/tradeset_lgb_noonc_rs_pts_alpha.parquet"
NULL_SEED = 20261008
NOT_FROZEN = ("0", "no", "false")


@dataclass(frozen=True)
class Paths:
    """Where the audit reads and writes: config.yaml's paths, and the repository root for params/."""
    root: Path
    outputs: Path
    within: Path
    panel: Path

    @classmethod
    def from_config(cls, cfg: dict) -> Paths:
        return cls(Path(cfg["_root"]), resolve(cfg, "outputs"), resolve(cfg, "within"), resolve(cfg, "panel"))

    @property
    def csv(self) -> Path:
        return self.outputs / "csv"


def era_of(season: int) -> str:
    """The era a season falls in, as "first-last": 1997-2006, 2007-2016 or 2017-2026."""
    return next(f"{first}-{last}" for first, last in ERAS if first <= season <= last)


def poss_of(rows: pd.DataFrame, side: str) -> np.ndarray:
    """Each rated player's possessions on one side ("O" offence, "D" defence)."""
    return rows[("poss_off" if side == "O" else "poss_def")].to_numpy(float)


def decision_test(test: dict) -> bool:
    """A test whose lean can count: neither team context, playing time nor score state."""
    return test["cls"] not in MEASURED_ONLY


def jackknife_se(drops: np.ndarray) -> np.ndarray:
    """The season jackknife's standard error: sqrt((n - 1) / n x sum of (drop - mean drop)^2), one drop per season
    left out (per column when `drops` has two axes)."""
    count = len(drops)
    return np.sqrt((count - 1) / count * ((drops - drops.mean(axis=0)) ** 2).sum(axis=0))


def write_json(path: Path, payload: dict) -> None:
    """Write `payload` as JSON indented one space per level."""
    with open(path, "w") as handle:
        handle.write(json.dumps(payload, indent=1))


# ---------------------------------------------------------------------------------------------- the axes
def percentile(values: np.ndarray, weights: np.ndarray, broken: bool = False) -> np.ndarray:
    """Possession-weighted percentile in (0, 1); a missing value takes the median.  Tied values share the midpoint
    of their block, unless `broken`, which breaks ties by possessions so an input with a large tied block (one team,
    no draft slot) still fills its tenths."""
    values = np.asarray(values, dtype=float)
    if np.isnan(values).any():
        values = np.where(np.isnan(values), np.nanmedian(values) if np.isfinite(values).any() else 0.0, values)
    if broken:
        return _percentile_ties_broken(values, weights)
    return _percentile_ties_shared(values, weights)


def _percentile_ties_broken(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    order = np.lexsort((weights, values))
    cumulative = np.cumsum(weights[order])
    out = np.empty_like(values)
    out[order] = (cumulative - weights[order] / 2.0) / cumulative[-1]
    return out


def _percentile_ties_shared(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values, sorted_weights = values[order], weights[order]
    _, start = np.unique(sorted_values, return_index=True)
    block_weight = np.add.reduceat(sorted_weights, start)
    before = np.concatenate([[0.0], np.cumsum(block_weight)[:-1]])
    midpoint = (before + block_weight / 2.0) / block_weight.sum()
    out = np.empty_like(values)
    out[order] = np.repeat(midpoint, np.diff(np.append(start, len(sorted_values))))
    return out


def normal_score(p: np.ndarray) -> np.ndarray:
    """The standard normal quantile of a percentile, clipped to [0.0001, 0.9999] so both ends stay finite."""
    return norm.ppf(np.clip(p, 1e-4, 1.0 - 1e-4))


def centre_columns(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Each column's possession-weighted mean taken out (a correction is centred within its fold and side)."""
    values = np.asarray(values, dtype=float)
    return values - (weights @ values) / weights.sum()


def axis_scores(rows: pd.DataFrame, weights: np.ndarray, tests: list, broken: bool = False) -> np.ndarray:
    """Each test's percentile for the rated rows, one column per test in the order of `tests`.  A group's axis is
    its loadings times its members' normal scores, ranked again.  `broken`: ties broken by possessions (for the
    tenths); otherwise tied values share a percentile (for the line and the bend)."""
    member_scores = {}
    out = np.empty((len(rows), len(tests)))
    for column, test in enumerate(tests):
        if test["kind"] == "input":
            out[:, column] = percentile(rows[test["code"]].to_numpy(float), weights, broken)
            continue
        for member in test["members"]:
            if member not in member_scores:
                member_scores[member] = normal_score(percentile(rows[member].to_numpy(float), weights))
        combined = sum(loading * member_scores[member] for member, loading in zip(test["members"], test["loadings"]))
        out[:, column] = percentile(combined, weights, broken)
    return out


def basis_block(shared: np.ndarray, broken: np.ndarray) -> np.ndarray:
    """[z, z squared, the nine kept tenths] for one test: z, the normal score, from the shared-ties percentile, the
    tenths from the percentile with ties broken."""
    z = normal_score(shared)
    tenth = np.minimum((broken * TENTHS).astype(int), TENTHS - 1)
    return np.column_stack([z, z ** 2, np.delete(np.eye(TENTHS)[tenth], REFERENCE_TENTH, axis=1)])


# --------------------------------------------------------------------------------------------- the groups
def freeze_groups(directory: Path, root: Path) -> dict:
    """The groups of correlated inputs, per side, from the incumbent's whole-season rated rows with 500 or more
    possessions: the clusters at each cut with two or more members, and their axis loadings."""
    files = sorted(directory.glob("players_*_all.parquet"))
    assert files, f"no whole-season rows in {directory}"
    frame = pd.concat([translate_frame(pd.read_parquet(path)) for path in files], ignore_index=True)
    return {"source": str(directory.relative_to(root)).replace("\\", "/"),
            "seasons": sorted(int(season) for season in frame.season.unique()), "min_possessions": GROUP_MIN_POSS,
            "cuts": list(CUTS), "codes": list(GROUPED), "sides": {side: side_groups(frame, side) for side in SIDES}}


def side_groups(frame: pd.DataFrame, side: str) -> dict:
    """Average linkage on 1 - |rank correlation| between the inputs' within-season normal scores, cut at CUTS."""
    rows = frame[(frame.side == side) & (poss_of(frame, side) >= GROUP_MIN_POSS)].reset_index(drop=True)
    by_season = rows.groupby("season")
    scores = np.column_stack([norm.ppf((by_season[code].rank(method="average") - 0.5)
                                       / by_season[code].transform("count")) for code in GROUPED])
    distance = 1.0 - np.abs(np.corrcoef(scores.T))
    np.fill_diagonal(distance, 0.0)
    tree = linkage(squareform(np.clip(distance, 0.0, None), checks=False), method="average")
    return dict(linkage=tree.tolist(), groups=groups_at_cuts(tree, scores), rows=int(len(rows)))


def groups_at_cuts(tree: np.ndarray, scores: np.ndarray) -> list:
    """Every cluster of two or more inputs at any cut, coarsest cut first; a cluster found at several cuts is one
    group that lists them all."""
    labels = {cut: fcluster(tree, t=cut, criterion="distance") for cut in CUTS}
    groups, seen = [], {}
    for cut in sorted(CUTS, reverse=True):
        for label in np.unique(labels[cut]):
            members = np.where(labels[cut] == label)[0]
            if len(members) < 2:
                continue
            key = tuple(sorted(GROUPED[member] for member in members))
            if key in seen:
                seen[key]["cuts"].append(cut)
                continue
            seen[key] = principal_axis(scores, members, cut)
            groups.append(seen[key])
    for group in groups:
        group["cuts"] = sorted(group["cuts"])
    return groups


def principal_axis(scores: np.ndarray, members: np.ndarray, cut: float) -> dict:
    """A group's members, sorted by the size of their loading on the first principal component of their
    correlation matrix (signed so the largest loading is positive), and the share of variance it explains."""
    eigenvalues, eigenvectors = np.linalg.eigh(np.corrcoef(scores[:, members].T))
    first = eigenvectors[:, -1]
    loadings = first * np.sign(eigenvectors[np.argmax(np.abs(first)), -1])
    order = np.argsort(-np.abs(loadings))
    return dict(members=[GROUPED[members[rank]] for rank in order], loadings=[float(loadings[rank]) for rank in order],
                cuts=[cut], explained=float(eigenvalues[-1] / eigenvalues.sum()))


def group_name(members: list) -> str:
    """A group's name on the console: its first three members' plain names, then "(+N more)"."""
    head = ", ".join(meaning(member) for member in members[:3])
    return head + (f" (+{len(members) - 3} more)" if len(members) > 3 else "")


def audit_groups(freeze: str, random_tags: list, paths: Paths) -> dict:
    """The frozen groups; `--freeze=1` writes them first, from the first random folder, and only when no frozen
    file exists."""
    groups_file = paths.root / GROUPS_FILE
    if freeze not in NOT_FROZEN:
        assert not groups_file.exists(), f"{GROUPS_FILE} is frozen; delete it to refreeze"
        groups = freeze_groups(paths.within / random_tags[0], paths.root)
        write_json(groups_file, groups)
        print(f"froze {GROUPS_FILE}: " + "; ".join(
            f"{side} {len(found['groups'])} groups over {found['rows']} rows"
            for side, found in groups["sides"].items()), flush=True)
    return json.loads(groups_file.read_text())


def build_tests(groups: dict) -> list:
    """One test per input, then one per group (a group found on both sides is one test)."""
    tests = [dict(id=code, kind="input", code=code, cls=CLASS_OF[code], name=meaning(code))
             for codes in CLASSES.values() for code in codes]
    for side in SIDES:
        for group in groups["sides"][side]["groups"]:
            test_id = "group:" + "+".join(sorted(group["members"]))
            if not any(test["id"] == test_id for test in tests):
                tests.append(group_test(test_id, group))
    return tests


def group_test(test_id: str, group: dict) -> dict:
    """The test for one group.  A group made only of inputs that are measured but never count (team context,
    playing time, score state) is filed under "playing time", so it never counts either."""
    measured_only = all(CLASS_OF[member] in MEASURED_ONLY for member in group["members"])
    return dict(id=test_id, kind="group", members=group["members"], loadings=group["loadings"],
                cls="playing time" if measured_only else "group", name=group_name(group["members"]),
                cuts=group["cuts"])


# ---------------------------------------------------------------------------------------------- the folds
def add_team_net(rows: pd.DataFrame) -> pd.DataFrame:
    """The team's net points per 100 over the games he played, from his on-court and off-court numbers (the
    feasibility check's `team_net_per100`; the fold reader, openrapm.data.within_folds, does not add it)."""
    on_o, off_o = rows.on_court_off_possessions.to_numpy(float), rows.off_court_off_possessions.to_numpy(float)
    on_d, off_d = rows.on_court_def_possessions.to_numpy(float), rows.off_court_def_possessions.to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        scored = (rows.on_court_scored_per100 * on_o + rows.off_court_scored_per100 * off_o) / (on_o + off_o)
        allowed = (rows.on_court_allowed_per100 * on_d + rows.off_court_allowed_per100 * off_d) / (on_d + off_d)
    return rows.assign(team_net_per100=(scored - allowed).fillna(0.0))


def read_audit_fold(directory: Path, stem: str):
    """One fold (openrapm.data.within_folds.read_fold) with `team_net_per100` on its rated rows."""
    fold = read_fold(directory, stem)
    fold.rows = {side: add_team_net(rows) for side, rows in fold.rows.items()}
    return fold


def level_and_home_projection(fold) -> Callable:
    """The weighted projection that takes the fold's controls (level, home edge, playoffs) out of team-game
    vectors: v -> sqrt(w) v - Q Q' sqrt(w) v, Q an orthonormal basis of sqrt(w) F."""
    root_weight = np.sqrt(fold.w)
    left, singular, _ = np.linalg.svd(fold.F * root_weight[:, None], full_matrices=False)
    return partial(_project, root_weight, left[:, singular > 1e-9 * singular.max()])


def _project(root_weight: np.ndarray, basis: np.ndarray, values) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    weighted = (root_weight * values.T).T if values.ndim == 2 else root_weight * values
    return weighted - basis @ (basis.T @ weighted)


@dataclass
class RatingParts:
    """A fold's ratings split into parts, per rated player and summed into each test team-game.

    Defence in the team-game sums is in points allowed: share x (-rating_def).  `stand_in` is what the players
    without a rating add to each team-game (the fold's base prediction minus the rated players' four parts).
    """
    prior_off: np.ndarray
    u_off: np.ndarray
    prior_def: np.ndarray
    u_def: np.ndarray
    team_prior_off: np.ndarray
    team_games_off: np.ndarray
    team_prior_def: np.ndarray
    team_games_def: np.ndarray
    stand_in: np.ndarray

    @classmethod
    def of(cls, fold) -> RatingParts:
        offence, defence = fold.rows["O"], fold.rows["D"]
        prior_off, u_off = offence.prior_off.to_numpy(float), offence.u_off.to_numpy(float)
        prior_def, u_def = defence.prior_def.to_numpy(float), defence.u_def.to_numpy(float)
        team_prior_off, team_games_off = fold.Z["O"] @ prior_off, fold.Z["O"] @ u_off
        team_prior_def, team_games_def = fold.Z["D"] @ (-prior_def), fold.Z["D"] @ (-u_def)
        stand_in = fold.base - team_prior_off - team_games_off - team_prior_def - team_games_def
        return cls(prior_off, u_off, prior_def, u_def, team_prior_off, team_games_off, team_prior_def,
                   team_games_def, stand_in)


def correction_parts(fold, lines: list | None) -> dict:
    """`--correction` (the retired script 140's group correction): per side, each line times the player's axis
    score on this fold's own inputs, centred; zero without one."""
    add = {side: np.zeros(len(fold.rows[side])) for side in SIDES}
    for side in SIDES:
        axes = [(test, slope) for test, line_side, slope in (lines or []) if line_side == side]
        if axes:
            weights = np.maximum(poss_of(fold.rows[side], side), 1e-9)
            scores = normal_score(axis_scores(fold.rows[side], weights, [test for test, _ in axes]))
            add[side] = centre_columns(scores, weights) @ np.array([slope for _, slope in axes], dtype=float)
    return add


def shrunk_prior(prior: np.ndarray, games: np.ndarray, multiplier: float, correction: np.ndarray,
                 weights: np.ndarray) -> np.ndarray:
    """Step 4's shrink on this fold: the prior part times its season's multiplier, re-centred so the rating's
    possession-weighted mean is zero."""
    level = np.average(multiplier * prior + correction + games, weights=weights)
    return multiplier * prior + correction - level


def stage_misses(fold, parts: RatingParts, shrunk_prior_off: np.ndarray, shrunk_prior_def: np.ndarray,
                 shrunk_rating: dict) -> dict:
    """Each stage's miss per test team-game: the prior part alone, the rating as the fold built it, the rating as
    it ships."""
    return {"prior": fold.y - parts.stand_in - fold.Z["O"] @ shrunk_prior_off - fold.Z["D"] @ (-shrunk_prior_def),
            "raw": fold.y - fold.base,
            "shrunk": fold.y - parts.stand_in - fold.Z["O"] @ shrunk_rating["O"]
            - fold.Z["D"] @ (-shrunk_rating["D"])}


def swap_misses(fold, parts: RatingParts, swap: pd.DataFrame) -> dict:
    """The retired script 106's ratings on this fold, read as two more stages: its own shrink (step 4's test rule)
    and the swap step on top of it, so the swap step's own lean is the difference, multipliers held equal."""
    misses = {}
    for stage, variant in (("shrunk106", "shrunk"), ("swap", "swap0.5")):
        ratings = swap[swap.variant == variant].set_index("player_id")
        offence = ratings.o.reindex(fold.rows["O"].player_id.to_numpy()).to_numpy(float)
        defence = ratings.d.reindex(fold.rows["D"].player_id.to_numpy()).to_numpy(float)
        assert np.isfinite(offence).all() and np.isfinite(defence).all(), f"{fold.stem}: 106 lacks rated players"
        misses[stage] = fold.y - parts.stand_in - fold.Z["O"] @ offence - fold.Z["D"] @ defence
    return misses


def free_columns(fold, parts: RatingParts, shrunk_miss: np.ndarray, weights: dict) -> dict:
    """Each version's (miss, free columns): the columns fitted together with the lean.

    prior_free   the two prior parts' team-game sums (step 4's multipliers refitted)
    rating_free  the two whole ratings' team-game sums (one free scale per side)
    nuisance     log possessions per side, the team-game share of rated players, of bench players
    """
    log_possessions = {side: centre_columns(np.log1p(poss_of(fold.rows[side], side)), weights[side])
                       for side in SIDES}
    rated = np.asarray(fold.Z["O"].sum(axis=1)).ravel() + np.asarray(fold.Z["D"].sum(axis=1)).ravel()
    bench = sum(np.asarray(fold.Z[side][:, np.where(poss_of(fold.rows[side], side) < BENCH_POSS)[0]]
                           .sum(axis=1)).ravel() for side in SIDES)
    nuisance = np.column_stack([SIGN[side] * (fold.Z[side] @ log_possessions[side]) for side in SIDES]
                               + [rated, bench])
    return {"prior_free": (fold.y - parts.stand_in - parts.team_games_off - parts.team_games_def,
                           np.column_stack([parts.team_prior_off, parts.team_prior_def])),
            "rating_free": (fold.y - parts.stand_in,
                            np.column_stack([parts.team_prior_off + parts.team_games_off,
                                             parts.team_prior_def + parts.team_games_def])),
            "nuisance": (shrunk_miss, nuisance)}


@dataclass
class AuditFold:
    """One fold, reduced to the small matrices every regression is read from.

    With the level and home edge projected out (weighted), let Z~ be a side's possession shares, projected, and B a
    correction basis (one row per rated player of the side, centred).  The lean's columns are X = sign Z~ B, so

        X'X = B' H B,   X'r = sign B' g,   X'N = sign B' M,   with  H = Z~'Z~,  g = Z~' r,  M = Z~' N

    for any miss r and any pooled free columns N, and no test touches the team-games again.
    """
    stem: str
    season: int
    setting: str
    rows: dict                 # side -> the rated players' rows
    weights: dict              # side -> each rated player's possessions on that side (at least 1e-9)
    shares: dict               # side -> Z, test team-games x rated players
    project: Callable          # the level-and-home projection
    game_idx: np.ndarray
    total_weight: float
    shrunk_rating: dict        # side -> the shipped rating before the swap step, positive = good
    shrunk_miss: np.ndarray    # the decision stage's miss, projected
    free_gram: dict            # version -> N'N
    free_miss: dict            # version -> N'r for that version's miss
    gram: dict                 # side -> H (float32)
    miss_cross: dict           # side -> stage -> g
    free_miss_cross: dict      # side -> version -> Z~'r for that version's miss
    free_cross: dict           # side -> version -> M


def audit_fold(fold, multiplier_off: float, multiplier_def: float, setting: str, swap: pd.DataFrame | None = None,
               lines: list | None = None) -> AuditFold:
    """One fold with its season's step-4 multipliers (and, with `lines`, the retired script 140's correction)."""
    project = level_and_home_projection(fold)
    parts = RatingParts.of(fold)
    correction = correction_parts(fold, lines)
    prior_off = shrunk_prior(parts.prior_off, parts.u_off, multiplier_off, correction["O"],
                             fold.rows["O"].poss_off.to_numpy(float))
    prior_def = shrunk_prior(parts.prior_def, parts.u_def, multiplier_def, correction["D"],
                             fold.rows["D"].poss_off.to_numpy(float))
    shrunk_rating = {"O": prior_off + parts.u_off, "D": prior_def + parts.u_def}
    misses = stage_misses(fold, parts, prior_off, prior_def, shrunk_rating)
    if swap is not None:
        misses.update(swap_misses(fold, parts, swap))
    weights = {side: np.maximum(poss_of(fold.rows[side], side), 1e-9) for side in SIDES}
    free = free_columns(fold, parts, misses["shrunk"], weights)
    projected = {stage: project(miss) for stage, miss in misses.items()}
    free_misses = {version: project(miss) for version, (miss, _) in free.items()}
    free_sums = {version: project(columns) for version, (_, columns) in free.items()}
    side_terms = {side: _side_terms(project, fold.Z[side], projected, free_misses, free_sums) for side in SIDES}
    return AuditFold(
        stem=fold.stem, season=fold.season, setting=setting, rows=fold.rows, weights=weights, shares=fold.Z,
        project=project, game_idx=np.asarray(fold.game_idx), total_weight=float(fold.w.sum()),
        shrunk_rating=shrunk_rating, shrunk_miss=projected["shrunk"],
        free_gram={version: free_sums[version].T @ free_sums[version] for version in free},
        free_miss={version: free_sums[version].T @ free_misses[version] for version in free},
        gram={side: terms[0] for side, terms in side_terms.items()},
        miss_cross={side: terms[1] for side, terms in side_terms.items()},
        free_miss_cross={side: terms[2] for side, terms in side_terms.items()},
        free_cross={side: terms[3] for side, terms in side_terms.items()})


def _side_terms(project: Callable, shares, misses: dict, free_misses: dict, free_sums: dict) -> tuple:
    """(H, g per stage, Z~'r per version, M per version) for one side."""
    projected = project(shares.toarray())
    return ((projected.T @ projected).astype(np.float32),
            {stage: projected.T @ miss for stage, miss in misses.items()},
            {version: projected.T @ miss for version, miss in free_misses.items()},
            {version: projected.T @ columns for version, columns in free_sums.items()})


def swap_ratings(directory: Path) -> dict | None:
    """The retired script 106's ratings per fold stem, when it ran on the folder."""
    path = directory / "openrapm_shipped.parquet"
    if not path.exists():
        return None
    ratings = pd.read_parquet(path)
    ratings = ratings[ratings.variant.isin(["shrunk", "swap0.5"])]
    return dict(tuple(ratings.groupby("key")))


def check_leaks(path: Path, tag: str, stem: str) -> None:
    """Stop on a fold whose json records a failed leak check (`openrapm folds` checks, for example, that the fit
    and test games are disjoint and that the prior models never trained on the season or its neighbours)."""
    info = json.loads(path.read_text())
    failed = [name for name, passed in info["checks"].items() if isinstance(passed, bool) and not passed]
    assert not failed, f"{tag} fold {stem}: leak checks failed {failed}"


def load_setting(tags: list, setting: str, multipliers: pd.DataFrame, within: Path,
                 lines_of: Callable | None = None) -> list:
    """Every fold of the named folders (any size or split), with its season's multipliers (and with
    `lines_of(season)`, the retired script 140's group correction)."""
    by_season = multipliers.set_index("season")
    folds = []
    for tag in tags:
        directory = within / tag
        swap = swap_ratings(directory)
        for path in sorted(directory.glob("fold_*.json")):
            stem = path.stem[len("fold_"):]
            check_leaks(path, tag, stem)
            fold = read_audit_fold(directory, stem)
            folds.append(audit_fold(fold, float(by_season.at[fold.season, "prior_off"]),
                                    float(by_season.at[fold.season, "prior_def"]), setting,
                                    swap.get(stem) if swap is not None else None,
                                    lines_of(fold.season) if lines_of is not None else None))
    return folds


def shipped_multipliers(seasons, mult_tag: str, within: Path) -> tuple:
    """The product rule's multipliers for every season (step 4 as it ships: the folds of `mult_tag` pinned to
    2017-2026, the rated season's own folds left out; other seasons borrow the pooled value), and those folds.  A
    candidate is audited on its own multipliers, fitted on its own folds: what would ship with it."""
    folds = load_folds(within / mult_tag, (2017, 2026))
    multipliers = multipliers_by_season(seasons, folds, "product")
    return multipliers.rename(columns={"multiplier_off": "prior_off", "multiplier_def": "prior_def"}), folds


# ------------------------------------------------------------------------------------------- the regressions
def solve_block(gram: np.ndarray, right: np.ndarray, columns, extra_gram=None, extra_cross=None,
                extra_right=None) -> np.ndarray:
    """The coefficients of `columns` (and any pooled free columns) by least squares from the accumulated sums."""
    columns = list(columns)
    block_gram, block_right = gram[np.ix_(columns, columns)], right[columns]
    if extra_gram is not None:
        cross = extra_cross[columns, :]
        block_gram = np.block([[block_gram, cross], [cross.T, extra_gram]])
        block_right = np.concatenate([block_right, extra_right])
    return np.linalg.lstsq(block_gram, block_right, rcond=None)[0][:len(columns)]


def statistic(kind: str, beta: np.ndarray) -> float:
    """The reading of a fitted correction: the line's slope, the bend's square, or the top tenth minus the
    bottom."""
    if kind == "line":
        return float(beta[0])
    if kind == "bend":
        return float(beta[1])
    full = np.insert(beta, REFERENCE_TENTH, 0.0)
    return float(full[-1] - full[0])


def centred_basis(fold: AuditFold, side: str, tests: list) -> np.ndarray:
    """Players x NCOL x tests: every test's basis block, centred within the fold and side."""
    rows, weights = fold.rows[side], fold.weights[side]
    shared = axis_scores(rows, weights, tests)
    broken = axis_scores(rows, weights, tests, broken=True)
    basis = np.concatenate([basis_block(shared[:, column], broken[:, column])[:, :, None]
                            for column in range(len(tests))], axis=2)
    return basis - np.einsum("i,ijk->jk", weights, basis)[None] / weights.sum()


def fold_side_sums(fold: AuditFold, side: str, tests: list, stages: list) -> dict:
    """One fold and side's contribution to every test's normal equations: G = X'X, h = X'r per stage, and per
    version X'r and X'N."""
    basis = centred_basis(fold, side, tests)
    flat = basis.reshape(len(basis), -1)
    gram_basis = (fold.gram[side] @ flat.astype(np.float32)).astype(float).reshape(basis.shape)
    sums = {"G": np.einsum("ijk,ilk->kjl", basis, gram_basis)}                 # tests x NCOL x NCOL
    for stage in stages:
        sums[f"h_{stage}"] = SIGN[side] * np.einsum("ijk,i->kj", basis, fold.miss_cross[side][stage])
    for version in VERSIONS:
        sums[f"hv_{version}"] = SIGN[side] * np.einsum("ijk,i->kj", basis, fold.free_miss_cross[side][version])
        sums[f"cv_{version}"] = SIGN[side] * np.einsum("ijk,il->kjl", basis, fold.free_cross[side][version])
    return sums


def empty_slot(stages: list) -> dict:
    """One season's sums before any fold is added, every term zero: G, h per stage, and per version the free
    columns' four terms."""
    keys = ["G", *(f"h_{stage}" for stage in stages),
            *(f"{part}_{version}" for part in ("hv", "cv", "NN", "Nr") for version in VERSIONS)]
    return dict.fromkeys(keys, 0.0)


def season_sums(folds: list, tests: list, stages: list, by_era: bool) -> dict:
    """{(side, era): {season: the summed normal equations}}, kept per season so a season can be dropped.  The
    pooled free columns' terms are added once per fold, on offence, and shared with defence."""
    sums = {}
    for fold in folds:
        for side in SIDES:
            parts = fold_side_sums(fold, side, tests, stages)
            for era in ("all",) + ((era_of(fold.season),) if by_era else ()):
                slot = sums.setdefault((side, era), {}).setdefault(fold.season, empty_slot(stages))
                for key, value in parts.items():
                    slot[key] = slot[key] + value
                if side == "O":
                    _add_pooled_terms(slot, fold)
    _share_pooled_terms(sums)
    return sums


def _add_pooled_terms(slot: dict, fold: AuditFold) -> None:
    for version in VERSIONS:
        slot[f"NN_{version}"] = slot[f"NN_{version}"] + fold.free_gram[version]
        slot[f"Nr_{version}"] = slot[f"Nr_{version}"] + fold.free_miss[version]


def _share_pooled_terms(sums: dict) -> None:
    for (side, era), seasons in sums.items():
        if side != "D":
            continue
        for season, slot in seasons.items():
            for version in VERSIONS:
                slot[f"NN_{version}"] = sums[("O", era)][season][f"NN_{version}"]
                slot[f"Nr_{version}"] = sums[("O", era)][season][f"Nr_{version}"]


def reading_estimate(sums: dict, test_index: int, version: str, stage: str, kind: str) -> float:
    """One test's reading from summed normal equations: pinned, or with a version's free columns fitted too."""
    columns = KIND_COLUMNS[kind]
    gram = sums["G"][test_index]
    if version == "pinned":
        return statistic(kind, solve_block(gram, sums[f"h_{stage}"][test_index], columns))
    return statistic(kind, solve_block(gram, sums[f"hv_{version}"][test_index], columns, sums[f"NN_{version}"],
                                       sums[f"cv_{version}"][test_index], sums[f"Nr_{version}"]))


def read_side_era(side: str, era: str, seasons: dict, tests: list, stages: list) -> tuple:
    """Every test x stage (pinned) and, over all eras, every version at the shrunk stage, for one side and era:
    (reading rows, tenth rows, {key: (seasons, the season-dropped estimates)})."""
    keys = sorted(seasons)
    total = {key: sum(seasons[season][key] for season in keys) for key in seasons[keys[0]]}
    left_out = ({season: {key: total[key] - seasons[season][key] for key in total} for season in keys}
                if len(keys) > 2 else {})
    variants = [("pinned", stage) for stage in stages] + ([(version, "shrunk") for version in VERSIONS]
                                                          if era == "all" else [])
    rows, tenth_rows, dropped = [], [], {}
    for test_index, test in enumerate(tests):
        for version, stage in variants:
            for kind in KIND_COLUMNS:
                estimate = reading_estimate(total, test_index, version, stage, kind)
                drops = (np.array([reading_estimate(left_out[season], test_index, version, stage, kind)
                                   for season in keys]) if left_out else None)
                se = float(jackknife_se(drops)) if drops is not None else float("nan")
                rows.append(dict(test=test["id"], side=side, stage=stage, version=version, statistic=kind, era=era,
                                 estimate=estimate, se=se, z=estimate / se if se > 0 else float("nan"),
                                 seasons=len(keys)))
                dropped[(test["id"], side, stage, version, kind, era)] = (keys, drops)
            if version == "pinned" and era == "all":
                tenth_rows.extend(tenth_levels(total, test_index, test, side, stage))
    return rows, tenth_rows, dropped


def tenth_levels(total: dict, test_index: int, test: dict, side: str, stage: str) -> list:
    """Each tenth's level, centred on the ten levels' mean."""
    levels = np.insert(solve_block(total["G"][test_index], total[f"h_{stage}"][test_index], KIND_COLUMNS["tenths"]),
                       REFERENCE_TENTH, 0.0)
    return [dict(test=test["id"], side=side, stage=stage, tenth=tenth + 1, level=float(level))
            for tenth, level in enumerate(levels - levels.mean())]


def run_tests(folds: list, tests: list, by_era: bool) -> tuple:
    """Every test x side x stage (pinned) and every version at the shrunk stage: estimate, jackknife standard
    error, z; plus the tenths' levels and each reading's season-dropped estimates."""
    stages = [stage for stage in STAGES + SWAP_STAGES if all(stage in fold.miss_cross["O"] for fold in folds)]
    rows, tenth_rows, dropped = [], [], {}
    for (side, era), seasons in season_sums(folds, tests, stages, by_era).items():
        side_rows, side_tenths, side_dropped = read_side_era(side, era, seasons, tests, stages)
        rows.extend(side_rows)
        tenth_rows.extend(side_tenths)
        dropped.update(side_dropped)
    return pd.DataFrame(rows), pd.DataFrame(tenth_rows), dropped


def run_joint(folds: list, tests: list) -> pd.DataFrame:
    """All groups' lines together, both sides, at the decision stage: each group's lean holding the others fixed."""
    groups = [test for test in tests if test["kind"] == "group"]
    gram, right = {}, {}
    for fold in folds:
        columns = []
        for side in SIDES:
            scores = centre_columns(normal_score(axis_scores(fold.rows[side], fold.weights[side], groups)),
                                    fold.weights[side])
            columns.append(SIGN[side] * fold.project(np.asarray(fold.shares[side] @ scores)))
        design = np.column_stack(columns)
        gram[fold.season] = gram.get(fold.season, 0.0) + design.T @ design
        right[fold.season] = right.get(fold.season, 0.0) + design.T @ fold.shrunk_miss
    keys = sorted(gram)
    total_gram, total_right = sum(gram.values()), sum(right.values())
    beta = np.linalg.lstsq(total_gram, total_right, rcond=None)[0]
    drops = np.array([np.linalg.lstsq(total_gram - gram[season], total_right - right[season], rcond=None)[0]
                      for season in keys])
    se = jackknife_se(drops)
    labels = [(side, test) for side in SIDES for test in groups]
    return pd.DataFrame([dict(test=test["id"], side=side, stage="shrunk", version="joint", statistic="line", era="all",
                              estimate=float(beta[column]), se=float(se[column]), z=float(beta[column] / se[column]),
                              seasons=len(keys))
                         for column, (side, test) in enumerate(labels)])


# ---------------------------------------------------------------------------------- the null: whole careers
@dataclass
class CareerIndex:
    """Every decision test's normal score for every rated player, one reference fold per season, and where each
    player's rows sit."""
    scores: dict          # (season key, side) -> players x tests
    ids: dict             # (season key, side) -> the players' ids, in row order
    position: dict        # player id -> his position
    seasons_of: dict      # position -> his season keys (offence rows), in season order
    row_of: dict          # (season key, side) -> {position: row}
    players: np.ndarray   # the positions that have a season key, sorted


def career_index(folds: list, tests: list) -> CareerIndex:
    """Every player's axis scores per season and side, and his seasons in career order: what the career null
    hands from one player to another."""
    scores, ids = {}, {}
    for fold in folds:
        key = (fold.setting, fold.season)
        if (key, SIDES[0]) in ids:
            continue
        for side in SIDES:
            rows = fold.rows[side]
            scores[(key, side)] = normal_score(axis_scores(rows, fold.weights[side], tests))
            ids[(key, side)] = rows.player_id.to_numpy()
    everyone = np.unique(np.concatenate([side_ids for side_ids in ids.values()]))
    position = {player: place for place, player in enumerate(everyone)}
    seasons_of = {}
    for (key, side), side_ids in ids.items():
        if side == "O":
            for player in side_ids:
                seasons_of.setdefault(position[player], []).append(key)
    for place in seasons_of:
        seasons_of[place] = sorted(seasons_of[place], key=lambda season_key: season_key[1])
    row_of = {(key, side): {position[player]: row for row, player in enumerate(side_ids)}
              for (key, side), side_ids in ids.items()}
    return CareerIndex(scores, ids, position, seasons_of, row_of, np.array(sorted(seasons_of)))


def donor_scores(index: CareerIndex, key: tuple, side: str, player, donor: dict):
    """The scores a player takes in one season: his donor's, from the donor's season at the same point of his
    career (or the donor's last), or None when the donor has no row on that side then."""
    place = index.position[player]
    career_step = index.seasons_of[place].index(key) if key in index.seasons_of[place] else 0
    giver = donor.get(place, place)
    giver_key = index.seasons_of[giver][min(career_step, len(index.seasons_of[giver]) - 1)]
    row = index.row_of[(giver_key, side)].get(giver)
    return None if row is None else index.scores[(giver_key, side)][row]


def donor_block(index: CareerIndex, key: tuple, side: str, donor: dict, n_tests: int) -> tuple:
    """({player id: row}, the reassigned scores) for one season key and side; zero where nothing was handed over."""
    ids = index.ids[(key, side)]
    block = np.zeros((len(ids), n_tests))
    for row, player in enumerate(ids):
        scores = donor_scores(index, key, side, player, donor)
        if scores is not None:
            block[row] = scores
    return dict(zip(ids, range(len(ids)))), block


def fold_null_parts(fold: AuditFold, side: str, where: dict, block: np.ndarray, n_tests: int) -> np.ndarray:
    """One fold and side's sums for every test's line and bend under one reassignment:
    [z'Hz, z'Hz^2, z^2'Hz^2, sign z'g, sign z^2'g], one column per test."""
    block_row = np.array([where.get(player, -1) for player in fold.rows[side].player_id.to_numpy()])
    scores = np.where(block_row[:, None] >= 0, block[np.maximum(block_row, 0)], 0.0)
    weights = fold.weights[side]
    linear = centre_columns(scores, weights)
    square = centre_columns(scores ** 2, weights)
    gram_scores = (fold.gram[side] @ np.hstack([linear, square]).astype(np.float32)).astype(float)
    miss = fold.miss_cross[side]["shrunk"]
    return np.stack([np.einsum("ij,ij->j", linear, gram_scores[:, :n_tests]),
                     np.einsum("ij,ij->j", linear, gram_scores[:, n_tests:]),
                     np.einsum("ij,ij->j", square, gram_scores[:, n_tests:]),
                     SIGN[side] * (linear.T @ miss), SIGN[side] * (square.T @ miss)])


def null_draw_sums(folds: list, index: CareerIndex, donor: dict, n_tests: int) -> dict:
    """{(season, side): the null sums} for one reassignment of whole careers."""
    sums, blocks = {}, {}
    for fold in folds:
        key = (fold.setting, fold.season)
        for side in SIDES:
            if (key, side) not in blocks:
                blocks[(key, side)] = donor_block(index, key, side, donor, n_tests)
            where, block = blocks[(key, side)]
            season_side = (fold.season, side)
            sums[season_side] = sums.get(season_side, 0.0) + fold_null_parts(fold, side, where, block, n_tests)
    return sums


def line_slope(parts: np.ndarray) -> np.ndarray:
    """The line's slope from `fold_null_parts`' sums: sign z'g over z'Hz."""
    return parts[3] / parts[0]


def bend_square(parts: np.ndarray) -> np.ndarray:
    """The square's coefficient of a line-plus-square fit, from the 2 x 2 normal equations."""
    return (parts[0] * parts[4] - parts[1] * parts[3]) / (parts[0] * parts[2] - parts[1] ** 2)


def side_null_estimates(sums: dict, side: str) -> tuple:
    """(line, bend, [the line's and the bend's largest season-jackknife |z|]) for one side under one draw."""
    seasons = sorted(season for (season, sums_side) in sums if sums_side == side)
    per_season = np.array([sums[(season, side)] for season in seasons])
    total = per_season.sum(axis=0)
    left_out = (total[None] - per_season).transpose(1, 0, 2)
    line, bend = line_slope(total), bend_square(total)
    largest = [float(np.nanmax(np.abs(estimate / jackknife_se(drops))))
               for estimate, drops in ((line, line_slope(left_out)), (bend, bend_square(left_out)))]
    return line, bend, largest


def career_null(folds: list, tests: list, draws: int, seed: int = NULL_SEED) -> dict:
    """Every decision test's line and bend estimates (both sides) for `draws` reassignments of whole careers: each
    player takes a donor's inputs, season by season along the donor's career.  Returns {"line": draws x sides x
    tests, "bend": the same, "jack_max": the largest season-jackknife |z| of each draw, "ids": the tests}.

    The season jackknife treats seasons as independent, but a player's misfit carries over from season to season,
    so an input fixed per player reaches |z| near 4 by chance in one test of twenty: each test is therefore scaled
    by its own spread under this null (the studentised max-T of Westfall and Young), not by its jackknife."""
    tests = [test for test in tests if decision_test(test)]
    n_tests = len(tests)
    index = career_index(folds, tests)
    rng = np.random.default_rng(seed)
    maxima = np.empty(draws)
    line_draws, bend_draws = np.empty((draws, len(SIDES), n_tests)), np.empty((draws, len(SIDES), n_tests))
    started = time.time()
    for draw in range(draws):
        donor = dict(zip(index.players, rng.permutation(index.players)))
        sums = null_draw_sums(folds, index, donor, n_tests)
        largest = 0.0
        for side_index, side in enumerate(SIDES):
            line_draws[draw, side_index], bend_draws[draw, side_index], side_largest = side_null_estimates(sums, side)
            for value in side_largest:
                largest = max(largest, value)
        maxima[draw] = largest
        if (draw + 1) % 100 == 0:
            print(f"    career null: {draw + 1} of {draws} draws ({time.time() - started:.0f}s)", flush=True)
    return {"line": line_draws, "bend": bend_draws, "jack_max": maxima, "ids": [test["id"] for test in tests]}


def studentized_threshold(null: dict, level: float = 0.95) -> tuple:
    """Each test's null mean and standard deviation, and the `level` quantile of the largest standardised
    |estimate| over all tests, sides and both statistics in each draw."""
    centre, spread, standardized = {}, {}, []
    for kind in ("line", "bend"):
        estimates = null[kind]                                        # draws x sides x tests
        mean, sd = estimates.mean(axis=0), estimates.std(axis=0, ddof=1)
        centre[kind], spread[kind] = mean, sd
        standardized.append(np.abs((estimates - mean[None]) / sd[None]).reshape(len(estimates), -1))
    maxima = np.concatenate(standardized, axis=1).max(axis=1)
    return float(np.quantile(maxima, level)), centre, spread, maxima


@dataclass
class Yardstick:
    """The threshold and each decision test's null mean and standard deviation (sides x tests per statistic),
    drawn in this run (`null`, `maxima`) or read from an earlier audit (`source`)."""
    threshold: float
    centre: dict
    spread: dict
    ids: list
    null: dict | None = None
    maxima: np.ndarray | None = None
    source: str = ""

    @property
    def position(self) -> dict:
        return {test_id: place for place, test_id in enumerate(self.ids)}


def plain_test_id(test_id: str) -> str:
    """A test id written before 2026-10-11 under the plain names: "ts" -> "true_shooting_pct", and a group's
    "group:ast+astr+creation" -> its members renamed and sorted again, the way `build_tests` names a group."""
    if not test_id.startswith("group:"):
        return column_name(test_id)
    return "group:" + "+".join(sorted(column_name(member) for member in test_id[len("group:"):].split("+")))


def null_from_file(name: str, paths: Paths) -> Yardstick:
    """An earlier audit's threshold and each test's null mean and standard deviation (a candidate is judged on the
    frozen yardstick, not on a null drawn again).  An audit written before 2026-10-11 names its tests the old way;
    each id is read under the plain names (`plain_test_id`)."""
    saved = json.loads((paths.outputs / f"{name}_thresholds.json").read_text())
    table = pd.read_csv(paths.csv / f"{name}_null.csv")
    table["test"] = table.test.map(plain_test_id)
    ids = list(dict.fromkeys(table.test))
    position = {test_id: place for place, test_id in enumerate(ids)}
    centre = {kind: np.full((len(SIDES), len(ids)), np.nan) for kind in ("line", "bend")}
    spread = {kind: np.full((len(SIDES), len(ids)), np.nan) for kind in ("line", "bend")}
    for row in table.itertuples():
        place, side_index = position[row.test], SIDES.index(row.side)
        centre[row.statistic][side_index, place], spread[row.statistic][side_index, place] = row.centre, row.spread
    return Yardstick(float(saved["threshold"]), centre, spread, ids, source=name)


def yardstick(draws: int, threshold_from: str, folds: list, tests: list, paths: Paths) -> Yardstick:
    """The career null drawn now (`--draws`), or the frozen one of an earlier audit (`--threshold_from`)."""
    if draws > 0:
        print(f"the career null ({draws} draws)", flush=True)
        null = career_null(folds, tests, draws)
        threshold, centre, spread, maxima = studentized_threshold(null)
        return Yardstick(threshold, centre, spread, null["ids"], null, maxima)
    assert threshold_from, "--draws=0 needs --threshold_from=<an earlier audit's --out>"
    frozen = null_from_file(threshold_from, paths)
    print(f"threshold {frozen.threshold:.2f} and each test's null spread read from {threshold_from} (no new null)",
          flush=True)
    return frozen


# ------------------------------------------------------------------------------------- each group's share
def models_for(season: int) -> str:
    """The incumbent's saved prior models for a season (one file per ten-season chunk)."""
    return ("lgb_noonc_early_1997" if season <= 2006 else "lgb_noonc_early_2007" if season <= 2016
            else "lgb_noonc_within")


def trainable_panel(paths: Paths, cfg: dict) -> pd.DataFrame:
    """The season panel, gated like every reader of it: a season still being played never enters a reading."""
    panel, _ = drop_untrainable(pd.read_parquet(paths.panel), cfg, what="the season panel")
    return panel


def share_groups(groups: dict, side: str, model_feats: list) -> tuple:
    """(names, the model's input columns of each): the 0.5-cut groups as one player each, an input in no group on
    its own."""
    member_of = {}
    for group in groups["sides"][side]["groups"]:
        if 0.5 in group["cuts"]:
            for member in group["members"]:
                member_of.setdefault(member, group_name(group["members"]))
    labels = [member_of.get(feature, meaning(feature)) for feature in model_feats]
    names = list(dict.fromkeys(labels))
    return names, [np.array([place for place, label in enumerate(labels) if label == name]) for name in names]


def shapley_values(model, inputs: np.ndarray, columns: list, rng, orderings: int, background: int,
                   threads: int) -> np.ndarray:
    """Interventional Shapley values per row and group, by sampled orderings of the groups: each group in turn
    swaps its columns from the background rows to the row's own, each coalition's prediction made in one batch."""
    count, width = inputs.shape
    rows = inputs[rng.choice(count, size=min(background, count), replace=False)]

    def mean_prediction(current: np.ndarray) -> np.ndarray:
        return model.predict(current.reshape(-1, width), num_threads=threads).reshape(count, len(rows)).mean(axis=1)

    values = np.zeros((count, len(columns)))
    for _ in range(orderings):
        current = np.repeat(rows[None], count, axis=0)
        before = mean_prediction(current)
        for group in rng.permutation(len(columns)):
            current[:, :, columns[group]] = inputs[:, None, columns[group]]
            after = mean_prediction(current)
            values[:, group] += after - before
            before = after
    values /= orderings
    return values


def side_shares(models: dict, panel: pd.DataFrame, season: int, side: str, groups: dict, rng,
                settings: dict) -> list:
    """One season and side's group shares: a group's mean |Shapley value| over the rated rows over the sum of all
    groups'."""
    held = add_features(panel[(panel.side == side) & (panel.season == season)].copy(), models["feats"])
    held = held.assign(row_possessions=held.possessions.to_numpy(float), row_seasons=1.0)
    inputs = held[models["model_feats"]].to_numpy(float)
    names, columns = share_groups(groups, side, models["model_feats"])
    size = np.abs(shapley_values(models["full"], inputs, columns, rng, **settings)).mean(axis=0)
    rows = [dict(season=season, side=side, group=name, share=float(share), mean_abs=float(mean_abs))
            for name, share, mean_abs in zip(names, size / size.sum(), size)]
    top = np.argsort(-size)[:3]
    print(f"  group shares {season} {side}: "
          + "; ".join(f"{names[rank]} {size[rank] / size.sum():.2f}" for rank in top), flush=True)
    return rows


def group_shapley_shares(seasons: list, groups: dict, out: Path, paths: Paths, cfg: dict, orderings: int = 32,
                         background: int = 30, seed: int = 0, threads: int = 4) -> pd.DataFrame:
    """Each group's share of the prior: Shapley values with every group of correlated inputs as ONE player (the
    audit's 0.5-cut groups; an input in no group stands alone), interventional against a background of the season's
    own rated rows.  LightGBM cannot give SHAP values for linear leaves (4.7.0: "not implemented for linear trees"),
    and shap's model-agnostic Partition explainer took 35 minutes a season and side; this is the same group-level
    quantity in a few minutes (linear leaves predict slowly, so `threads` cores; run it only when no build is
    running)."""
    panel = trainable_panel(paths, cfg)
    settings = dict(orderings=orderings, background=background, threads=threads)
    loaded, rows = {}, []
    rng = np.random.default_rng(seed)
    for season in seasons:
        name = models_for(season)
        if name not in loaded:
            # boosters saved before 2026-10-11 name their inputs the old way (openrapm.data.legacy_names)
            loaded[name] = saved_models(pd.read_pickle(paths.outputs / f"prior_models_{name}.pkl"))[0]
        for side in SIDES:
            rows.extend(side_shares(loaded[name][season][side], panel, season, side, groups, rng, settings))
    shares = pd.DataFrame(rows)
    shares.to_csv(out, index=False)
    return shares


# ---------------------------------------------------------------------------------------- the trade set (a veto)
def season_percentiles(values: np.ndarray, weights: np.ndarray, per_season: list) -> np.ndarray:
    """Possession-weighted percentiles taken within each season; `per_season` lists each season's rows."""
    out = np.empty(len(values))
    for rows in per_season:
        out[rows] = percentile(values[rows], weights[rows])
    return out


def tradeset_axis(frame: pd.DataFrame, weights: np.ndarray, per_season: list, test: dict, member_scores: dict):
    """A test's within-season percentile on the trade set's rows."""
    if test["kind"] == "input":
        return season_percentiles(frame[test["code"]].to_numpy(float), weights, per_season)
    for member in test["members"]:
        if member not in member_scores:
            member_scores[member] = normal_score(season_percentiles(frame[member].to_numpy(float), weights,
                                                                    per_season))
    combined = sum(loading * member_scores[member] for member, loading in zip(test["members"], test["loadings"]))
    return season_percentiles(combined, weights, per_season)


def clustered_line(z: np.ndarray, y: np.ndarray, player_ids: np.ndarray) -> tuple:
    """The least-squares slope of y on z (both centred) and its standard error clustered by player."""
    z_centred, y_centred = z - z.mean(), y - y.mean()
    slope = float(z_centred @ y_centred / (z_centred @ z_centred))
    score = pd.Series(z_centred * (y_centred - slope * z_centred)).groupby(player_ids).sum().to_numpy()
    se = float(np.sqrt((score ** 2).sum()) / (z_centred @ z_centred))
    return slope, se


def tradeset_leans(tests: list, alpha_path: Path, paths: Paths, cfg: dict) -> pd.DataFrame:
    """Each decision test's line of the neighbouring-season miss (the trade set's alpha on actual points, eligible
    rows) on the input's within-season normal score, standard errors clustered by player.  It never fits anything:
    a lean the season's own games find and this contradicts at |z| 2 does not count."""
    alpha = pd.read_parquet(alpha_path)
    alpha = alpha[alpha.eligible]
    panel = trainable_panel(paths, cfg)
    decision = [test for test in tests if decision_test(test)]
    codes = sorted({code for test in decision for code in ([test["code"]] if test["kind"] == "input"
                                                            else test["members"])})
    rows = []
    for side, name in (("O", "offense"), ("D", "defense")):
        frame = add_features(panel[panel.side == side].copy(), codes)
        frame = frame.merge(alpha[alpha.side == name][["player_id", "season", "alpha_good"]],
                            on=["player_id", "season"], how="inner").reset_index(drop=True)
        weights = np.maximum(frame["possessions"].to_numpy(float), 1e-9)   # the panel's possessions on this side
        per_season = list(frame.groupby("season").indices.values())
        member_scores = {}
        for test in decision:
            z = normal_score(tradeset_axis(frame, weights, per_season, test, member_scores))
            slope, se = clustered_line(z, frame.alpha_good.to_numpy(float), frame.player_id.to_numpy())
            rows.append(dict(test=test["id"], side=side, tradeset_line=slope, tradeset_z=slope / se))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------- one fold setting
@dataclass
class Audit:
    """One set of fold folders, read: its folds, every reading, the tenths, and each reading's season drops."""
    folds: list
    readings: pd.DataFrame
    tenths: pd.DataFrame
    drops: dict
    estimates: dict
    by_era: bool
    pooled: tuple             # step 4's multipliers with every fold of the multiplier folder pooled


def correction_lines(coefficients: pd.DataFrame, axes: list, season: int) -> list:
    """[(test, side, line)]: the retired script 140's line on each axis in one season."""
    return [(test, side, float(coefficients.at[season, "line_" + side + "_" + test["id"]])) for test, side in axes]


def correction_multipliers(correction: str, tests: list, seasons, csv_dir: Path) -> tuple:
    """The retired script 140's multipliers and a reader of its lines, both fitted outside each fold's season."""
    coefficients = pd.read_csv(csv_dir / f"{correction}_coefficients_product.csv").set_index("season")
    missing = sorted(set(seasons) - set(coefficients.index))
    assert not missing, f"{correction} has no coefficients for {missing}"
    by_id = {test["id"]: test for test in tests}
    axes = [(by_id[test_id], side) for test_id, side in json.loads(coefficients["axes"].iloc[0])]
    multipliers = coefficients.loc[sorted(seasons), ["prior_off", "prior_def"]].reset_index()
    return multipliers, partial(correction_lines, coefficients, axes)


def audit_setting(tags: list, mult_tag: str, setting: str, tests: list, paths: Paths, seasons=None,
                  correction: str = "") -> Audit:
    """Load a set of fold folders with its own multipliers and read every test on it.  `correction` names a
    retired script 140 output: its multipliers and lines, fitted outside each fold's season (the product rule),
    replace step 4's."""
    if seasons is None:
        seasons = sorted({int(path.stem.split("_")[1]) for tag in tags
                          for path in (paths.within / tag).glob("fold_*.json")})
    multipliers, mult_folds = shipped_multipliers(seasons, mult_tag, paths.within)
    lines_of = None
    if correction:
        multipliers, lines_of = correction_multipliers(correction, tests, seasons, paths.csv)
    folds = load_setting(tags, setting, multipliers, paths.within, lines_of)
    by_era = len({era_of(fold.season) for fold in folds}) > 1
    readings, tenths, drops = run_tests(folds, tests, by_era)
    estimates = {(row.test, row.side, row.stage, row.version, row.statistic, row.era): row.estimate
                 for row in readings.itertuples()}
    return Audit(folds, readings, tenths, drops, estimates, by_era, fit_multipliers(mult_folds))


# ------------------------------------------------------------------------------------------------ the paired mode
def lean_change(key: tuple, candidate: Audit, reference: Audit) -> dict:
    """One reading's change from the reference to the candidate, with a season jackknife of the paired
    difference when both have the same seasons."""
    test_id, side, stage, _, kind, _ = key
    keys_candidate, drops_candidate = candidate.drops[key]
    keys_reference, drops_reference = reference.drops[key]
    change = candidate.estimates[key] - reference.estimates[key]
    se = float("nan")
    if drops_candidate is not None and drops_reference is not None and list(keys_candidate) == list(keys_reference):
        se = float(jackknife_se(drops_candidate - drops_reference))
    return dict(test=test_id, side=side, stage=stage, statistic=kind, reference=reference.estimates[key],
                candidate=candidate.estimates[key], change=change, change_se=se,
                change_z=change / se if se > 0 else float("nan"))


def paired_leans(candidate: Audit, reference: Audit) -> pd.DataFrame:
    """Every lean's change from the reference to the candidate: the pinned version, all eras, each stage and
    statistic both audits read."""
    return pd.DataFrame([lean_change(key, candidate, reference) for key in candidate.drops
                         if key[3] == "pinned" and key[5] == "all" and key in reference.drops])


def side_moves(candidate: AuditFold, reference: AuditFold, side: str, tests: list) -> list:
    """How far the candidate's shrunk ratings moved from the reference's along each axis, same players: the
    possession-weighted slope of the move on the normal score, and the top tenth's mean move minus the bottom's."""
    ids = reference.rows[side].player_id.to_numpy()
    moved = pd.Series(candidate.shrunk_rating[side], index=candidate.rows[side].player_id.to_numpy()).reindex(ids)
    delta = moved.to_numpy() - reference.shrunk_rating[side]
    ok = np.isfinite(delta)
    weights = reference.weights[side][ok]
    shared = axis_scores(reference.rows[side], reference.weights[side], tests)[ok]
    broken = axis_scores(reference.rows[side], reference.weights[side], tests, broken=True)[ok]
    delta = delta[ok]
    out = []
    for column, test in enumerate(tests):
        z = normal_score(shared[:, column])
        z_centred = z - np.average(z, weights=weights)
        delta_centred = delta - np.average(delta, weights=weights)
        top, bottom = broken[:, column] >= 0.9, broken[:, column] < 0.1
        out.append(((test["id"], side),
                    (float(np.sum(weights * z_centred * delta_centred) / np.sum(weights * z_centred * z_centred)),
                     float(np.average(delta[top], weights=weights[top])
                           - np.average(delta[bottom], weights=weights[bottom])))))
    return out


def rating_moves(common: list, by_stem: dict, tests: list) -> pd.DataFrame:
    """`side_moves` averaged over the folds both audits share: one row per test and side."""
    moves = {}
    for fold in common:
        for side in SIDES:
            for key, move in side_moves(fold, by_stem[fold.stem], side, tests):
                moves.setdefault(key, []).append(move)
    return pd.DataFrame([dict(test=key[0], side=key[1], move_line=float(np.mean([move[0] for move in values])),
                              move_tenths=float(np.mean([move[1] for move in values])))
                         for key, values in moves.items()])


def paired_error(common: list, by_stem: dict) -> dict:
    """The paired held-out error per fold at the decision stage (team-game error, level and home edge refitted),
    averaged per season, with a season jackknife."""
    error = pd.DataFrame([dict(stem=fold.stem, season=fold.season,
                               candidate=float(np.sum(fold.shrunk_miss ** 2) / fold.total_weight),
                               reference=float(np.sum(by_stem[fold.stem].shrunk_miss ** 2)
                                               / by_stem[fold.stem].total_weight))
                          for fold in common])
    error["change"] = error.candidate - error.reference
    per_season = error.groupby("season").change.mean()
    se = float(jackknife_se(np.array([per_season.drop(season).mean() for season in per_season.index])))
    return dict(change=float(per_season.mean()), se=se,
                z=float(per_season.mean() / se) if se > 0 else float("nan"),
                better_seasons=int((per_season < 0).sum()), seasons=int(len(per_season)), folds=int(len(error)))


def paired(candidate: Audit, reference: Audit, tests: list) -> tuple:
    """A candidate against a reference on identical test games: the change in every lean (a jackknife over seasons
    of the paired difference), how far the ratings themselves moved along each axis, and the paired held-out
    error."""
    by_stem = {fold.stem: fold for fold in reference.folds}
    common = [fold for fold in candidate.folds if fold.stem in by_stem]
    assert common, "the two folders share no fold"
    for fold in common:
        assert np.array_equal(fold.game_idx, by_stem[fold.stem].game_idx), \
            f"{fold.stem}: the two folders' test games differ"
    return paired_leans(candidate, reference), rating_moves(common, by_stem, tests), paired_error(common, by_stem)


def parse_folder(spec: str) -> tuple:
    """`tag[:mult_tag]`: a fold folder and the folder its multipliers are fitted on (default: itself)."""
    tag, _, mult = spec.partition(":")
    return tag, (mult or tag)


def with_control(out: pd.DataFrame, error: dict, control: str, reference: Audit, tests: list, paths: Paths,
                 seasons: list) -> pd.DataFrame:
    """The noise control's own changes against the same reference, beside the candidate's (`error` gains them)."""
    tag, mult = parse_folder(control)
    audit = audit_setting([tag], mult, "control", tests, paths, seasons=seasons)
    leans, moves, control_error = paired(audit, reference, tests)
    out = out.merge(leans[["test", "side", "stage", "statistic", "change"]].rename(
        columns={"change": "control_change"}), on=["test", "side", "stage", "statistic"], how="left")
    out = out.merge(moves.rename(columns={"move_line": "control_move_line", "move_tenths": "control_move_tenths"}),
                    on=["test", "side"], how="left")
    error["control_change"], error["control_z"] = control_error["change"], control_error["z"]
    return out


def print_paired(out: pd.DataFrame, error: dict, reference_tag: str, control: bool) -> None:
    """The paired console: the change in held-out error, then the 20 largest changes in a tenths reading at the
    shrunk stage."""
    print(f"\npaired against {reference_tag}: held-out team-game error {error['change']:+.4f} (z {error['z']:+.1f}, "
          f"{error['better_seasons']} of {error['seasons']} seasons better)"
          + (f"; the noise control {error['control_change']:+.4f}" if control else ""), flush=True)
    focus = out[(out.stage == "shrunk") & (out.statistic == "tenths")]
    focus = focus.reindex(focus.change.abs().sort_values(ascending=False).index).head(20)
    for row in focus.itertuples():
        control_text = f"  control {row.control_change:+.2f}, moved {row.control_move_tenths:+.2f}" if control else ""
        print(f"  {row.side} {row.name[:50]:50s} lean {row.reference:+.2f} -> {row.candidate:+.2f} ({row.change:+.2f}, "
              f"z {row.change_z:+.1f}); ratings moved {row.move_tenths:+.2f}{control_text}")


def paired_mode(options: Options, candidate: Audit, tests: list, paths: Paths) -> None:
    """The candidate against the incumbent's folds, beside the noise control's own movement."""
    seasons = sorted({fold.season for fold in candidate.folds})
    reference_tag, reference_mult = parse_folder(options.reference)
    reference = audit_setting([reference_tag], reference_mult, "reference", tests, paths, seasons=seasons)
    leans, moves, error = paired(candidate, reference, tests)
    out = leans.merge(moves, on=["test", "side"], how="left")
    if options.control:
        out = with_control(out, error, options.control, reference, tests, paths, seasons)
    out.insert(2, "name", out.test.map({test["id"]: test["name"] for test in tests}))
    out.to_csv(paths.csv / f"{options.out}_paired.csv", index=False)
    write_json(paths.outputs / f"{options.out}_paired_error.json", error)
    print_paired(out, error, reference_tag, bool(options.control))


# ------------------------------------------------------------------------------------------ the summary table
def pick(readings: pd.DataFrame, stage: str, version: str, kind: str, era: str = "all",
         setting: str = "random") -> pd.DataFrame:
    """The readings of one stage, version, statistic, era and setting, indexed by (test, side)."""
    chosen = readings[(readings.stage == stage) & (readings.version == version) & (readings.statistic == kind)
                      & (readings.era == era) & (readings.setting == setting)]
    return chosen.set_index(["test", "side"])


@dataclass
class DecisionReadings:
    """The readings the summary is built from, each indexed by (test, side)."""
    line: pd.DataFrame
    bend: pd.DataFrame
    tenths: pd.DataFrame
    prior_free: pd.DataFrame
    rating_free: pd.DataFrame
    nuisance: pd.DataFrame
    joint: pd.DataFrame
    prior_tenths: pd.DataFrame
    raw_tenths: pd.DataFrame
    shrunk106_tenths: pd.DataFrame | None
    swap_tenths: pd.DataFrame | None
    era_line: dict             # era -> the line in that era's folds alone (only when the folds span eras)
    deadline_line: pd.DataFrame | None

    @classmethod
    def of(cls, readings: pd.DataFrame, by_era: bool, with_deadline: bool) -> DecisionReadings:
        has_swap = bool((readings.stage == "swap").any())
        era_line = ({f"{first}-{last}": pick(readings, "shrunk", "pinned", "line", era=f"{first}-{last}")
                     for first, last in ERAS} if by_era else {})
        return cls(line=pick(readings, "shrunk", "pinned", "line"), bend=pick(readings, "shrunk", "pinned", "bend"),
                   tenths=pick(readings, "shrunk", "pinned", "tenths"),
                   prior_free=pick(readings, "shrunk", "prior_free", "line"),
                   rating_free=pick(readings, "shrunk", "rating_free", "line"),
                   nuisance=pick(readings, "shrunk", "nuisance", "line"),
                   joint=readings[readings.version == "joint"].set_index(["test", "side"]),
                   prior_tenths=pick(readings, "prior", "pinned", "tenths"),
                   raw_tenths=pick(readings, "raw", "pinned", "tenths"),
                   shrunk106_tenths=pick(readings, "shrunk106", "pinned", "tenths") if has_swap else None,
                   swap_tenths=pick(readings, "swap", "pinned", "tenths") if has_swap else None,
                   era_line={era: line for era, line in era_line.items() if len(line)},
                   deadline_line=pick(readings, "shrunk", "pinned", "line", setting="deadline")
                   if with_deadline else None)


def same_sign_all_scales(line: float, prior_free: float, rating_free: float) -> bool:
    """The line keeps its sign, and its size within a factor of two, when the scale is refitted."""
    return bool(line != 0 and all(np.sign(refitted) == np.sign(line) and 0.5 <= refitted / line <= 2.0
                                  for refitted in (prior_free, rating_free)))


def null_z(stick: Yardstick, test_id: str, side: str, line: float, bend: float) -> tuple:
    """(line z, bend z, the line's null standard deviation) under the career null; NaN for a measured-only test."""
    if test_id not in stick.position:
        return float("nan"), float("nan"), float("nan")
    place, side_index = stick.position[test_id], SIDES.index(side)
    return ((line - stick.centre["line"][side_index, place]) / stick.spread["line"][side_index, place],
            (bend - stick.centre["bend"][side_index, place]) / stick.spread["bend"][side_index, place],
            float(stick.spread["line"][side_index, place]))


def vetoes(readings: DecisionReadings, tradeset: pd.DataFrame | None, key: tuple, line: float) -> dict:
    """The sign checks that can stop a lean from counting; None where a check cannot be read.  A veto with no
    spread (the trade set's eligible players are one-team players) vetoes nothing."""
    era_ok = (all(np.sign(by_era.loc[key].estimate) == np.sign(line) for by_era in readings.era_line.values())
              if readings.era_line else None)
    deadline = readings.deadline_line
    deadline_ok = (bool(deadline.loc[key].z * np.sign(line) > -2)
                   if deadline is not None and np.isfinite(deadline.loc[key].z) else None)
    in_tradeset = tradeset is not None and key in tradeset.index and np.isfinite(tradeset.loc[key].tradeset_z)
    tradeset_ok = bool(tradeset.loc[key].tradeset_z * np.sign(line) > -2) if in_tradeset else None
    return dict(in_tradeset=in_tradeset, era=era_ok, deadline=deadline_ok, tradeset=tradeset_ok)


def stage_columns(readings: DecisionReadings, key: tuple) -> dict:
    """The joint fit, the three versions, the earlier stages' tenths and the swap step's, for one test and side."""
    joint, nuisance = readings.joint, readings.nuisance.loc[key]
    has_swap = readings.swap_tenths is not None
    return dict(
        joint_line=joint.loc[key].estimate if key in joint.index else np.nan,
        joint_z=joint.loc[key].z if key in joint.index else np.nan,
        prior_free_line=readings.prior_free.loc[key].estimate, rating_free_line=readings.rating_free.loc[key].estimate,
        nuisance_line=nuisance.estimate, nuisance_z=nuisance.z,
        prior_stage_tenths=readings.prior_tenths.loc[key].estimate,
        raw_stage_tenths=readings.raw_tenths.loc[key].estimate,
        # the swap step's own lean: 106's swapped ratings against 106's shrunk ones (the same multipliers)
        swap_stage_tenths=readings.swap_tenths.loc[key].estimate if has_swap else np.nan,
        swap_step_tenths=(readings.swap_tenths.loc[key].estimate - readings.shrunk106_tenths.loc[key].estimate)
        if has_swap else np.nan)


def summary_row(test: dict, side: str, readings: DecisionReadings, stick: Yardstick,
                tradeset: pd.DataFrame | None) -> dict:
    """One test and side at the decision stage, with every check of the counting rule."""
    key = (test["id"], side)
    line, bend, tenths = readings.line.loc[key], readings.bend.loc[key], readings.tenths.loc[key]
    nuisance = readings.nuisance.loc[key].estimate
    same = same_sign_all_scales(line.estimate, readings.prior_free.loc[key].estimate,
                                readings.rating_free.loc[key].estimate)
    z_line, z_bend, spread_line = null_z(stick, test["id"], side, line.estimate, bend.estimate)
    passes = bool(np.isfinite(z_line) and max(abs(z_line), abs(z_bend)) >= stick.threshold)
    veto = vetoes(readings, tradeset, key, line.estimate)
    large_enough = bool(abs(tenths.estimate) >= 0.1)
    counts = all((decision_test(test), passes, large_enough, same, veto["era"] in (None, True),
                  veto["deadline"] in (None, True), veto["tradeset"] in (None, True)))
    return dict(
        test=test["id"], kind=test["kind"], cls=test["cls"], name=test["name"], side=side,
        tenths_bottom_to_top=tenths.estimate, tenths_se=tenths.se, line_per_sd=line.estimate,
        line_null_sd=spread_line, line_z=z_line, bend=bend.estimate, bend_z=z_bend, line_jackknife_z=line.z,
        bend_jackknife_z=bend.z, smallest_detectable_line=stick.threshold * spread_line,
        **stage_columns(readings, key),
        **{f"line_{era}": by_era.loc[key].estimate for era, by_era in readings.era_line.items()},
        deadline_line_z=readings.deadline_line.loc[key].z if readings.deadline_line is not None else np.nan,
        tradeset_line=tradeset.loc[key].tradeset_line if veto["in_tradeset"] else np.nan,
        tradeset_z=tradeset.loc[key].tradeset_z if veto["in_tradeset"] else np.nan,
        passes_threshold=passes, at_least_0_1=large_enough, same_sign_all_scales=same,
        same_sign_every_era=veto["era"], deadline_not_against=veto["deadline"],
        tradeset_not_against=veto["tradeset"], counts=counts,
        # information, not part of the rule: the line loses its sign or half its size once playing time (log
        # possessions, the bench's share of the team-game) is held -- the closed playing-time topic
        playing_time_explains=bool(np.sign(nuisance) != np.sign(line.estimate)
                                   or abs(nuisance) < 0.5 * abs(line.estimate)))


def total_row(test: dict, readings: DecisionReadings, stick: Yardstick) -> dict:
    """Offence plus defence: does the lean survive on the total, or is it credit landing on the wrong side?"""
    offence, defence = (test["id"], "O"), (test["id"], "D")
    total_line = readings.line.loc[offence].estimate + readings.line.loc[defence].estimate
    z_total = float("nan")
    if test["id"] in stick.position and stick.null is not None:
        place = stick.position[test["id"]]
        draws_total = stick.null["line"][:, 0, place] + stick.null["line"][:, 1, place]
        z_total = (total_line - draws_total.mean()) / draws_total.std(ddof=1)
    return dict(test=test["id"], kind=test["kind"], cls=test["cls"], name=test["name"], side="total",
                tenths_bottom_to_top=readings.tenths.loc[offence].estimate + readings.tenths.loc[defence].estimate,
                line_per_sd=total_line, line_z=z_total, counts=False)


def summary_table(tests: list, readings: DecisionReadings, stick: Yardstick,
                  tradeset: pd.DataFrame | None) -> pd.DataFrame:
    """One summary row per test and side, then one per test for offence plus defence."""
    rows = [summary_row(test, side, readings, stick, tradeset) for test in tests for side in SIDES]
    return pd.DataFrame(rows + [total_row(test, readings, stick) for test in tests])


# --------------------------------------------------------------------------------------------------- the run
@dataclass(frozen=True)
class Options:
    """The command line, read: tag lists split, the trade set's default filled in."""
    random: list
    mult_tag: str
    deadline: list
    reference: str
    control: str
    out: str
    draws: int
    threshold_from: str
    tradeset: str
    shares: list
    freeze: str
    correction: str

    @classmethod
    def of(cls, args: argparse.Namespace, root: Path) -> Options:
        tradeset = args.tradeset
        if tradeset is None:
            tradeset = DEFAULT_TRADESET if (root / DEFAULT_TRADESET).exists() else ""
        return cls(random=tag_list(args.random), mult_tag=args.mult_tag, deadline=tag_list(args.deadline),
                   reference=args.reference, control=args.control, out=args.out, draws=args.draws,
                   threshold_from=args.threshold_from, tradeset=tradeset,
                   shares=[int(season) for season in tag_list(args.shares)], freeze=args.freeze,
                   correction=args.correction)


def tag_list(text: str) -> list:
    """Comma-separated tags as a list, empty entries dropped."""
    return [tag for tag in text.split(",") if tag]


def random_readings(options: Options, tests: list, paths: Paths, started: float) -> tuple:
    """The random folds' audit, and every reading to write: theirs, the joint fit, and the deadline folds' shrunk
    pinned readings."""
    audit = audit_setting(options.random, options.mult_tag, "random", tests, paths, correction=options.correction)
    if options.correction:
        print(f"the random folds carry the group correction {options.correction} (its multipliers and lines, fitted "
              f"outside each fold's season)", flush=True)
    folds = audit.folds
    print(f"step 4's multipliers on {options.mult_tag} 2017-2026, all folds pooled: offence {audit.pooled[0]:.4f}, "
          f"defence {audit.pooled[1]:.4f}", flush=True)
    print(f"{len(folds)} random folds from {options.random}, seasons {min(f.season for f in folds)}-"
          f"{max(f.season for f in folds)}; {len(tests)} tests ({sum(t['kind'] == 'group' for t in tests)} groups) "
          f"({time.time() - started:.0f}s)", flush=True)
    parts = [audit.readings.assign(setting="random"), run_joint(folds, tests).assign(setting="random")]
    if options.deadline:
        deadline = audit_setting(options.deadline, options.mult_tag, "deadline", tests, paths,
                                 correction=options.correction)
        readings = deadline.readings
        parts.append(readings[(readings.stage == "shrunk") & (readings.version == "pinned")].assign(setting="deadline"))
        print(f"{len(deadline.folds)} deadline folds read ({time.time() - started:.0f}s)", flush=True)
    return audit, pd.concat(parts, ignore_index=True)


def write_null(options: Options, stick: Yardstick, audit: Audit, tests: list, paths: Paths) -> None:
    """A new null's per-test mean and standard deviation, and the threshold with its provenance."""
    pd.DataFrame([dict(test=test_id, side=side, statistic=kind, centre=float(stick.centre[kind][side_index, place]),
                       spread=float(stick.spread[kind][side_index, place]))
                  for place, test_id in enumerate(stick.ids) for side_index, side in enumerate(SIDES)
                  for kind in ("line", "bend")]).to_csv(paths.csv / f"{options.out}_null.csv", index=False)
    write_json(paths.outputs / f"{options.out}_thresholds.json", dict(
        threshold=stick.threshold, draws=options.draws,
        null_maxima_quantiles={level: float(np.quantile(stick.maxima, level)) for level in (0.5, 0.9, 0.95, 0.99)},
        jackknife_z_maxima_quantiles={level: float(np.quantile(stick.null["jack_max"], level))
                                      for level in (0.5, 0.95)},
        decision_tests=sum(decision_test(test) for test in tests), random=options.random, mult_tag=options.mult_tag,
        deadline=options.deadline, tradeset=options.tradeset, pooled_multipliers=list(audit.pooled),
        folds=len(audit.folds), seasons=sorted({fold.season for fold in audit.folds})))


def print_leans(summary: pd.DataFrame, stick: Yardstick, tests: list, draws: int) -> None:
    """The threshold, then per side the 30 largest tenths readings, then offence plus defence."""
    source = (f" ({draws} draws; the season jackknife's own z would have needed "
              f"{np.quantile(stick.null['jack_max'], 0.95):.2f})" if stick.null is not None
              else f" (from {stick.source})")
    print(f"\nthreshold |z| {stick.threshold:.2f}: the 95th percentile of the largest standardised estimate over "
          f"{2 * 2 * sum(decision_test(test) for test in tests)} line and bend tests, whole careers reassigned"
          + source + f"; {int(summary.counts.sum())} leans count", flush=True)
    show = summary.reindex(summary.tenths_bottom_to_top.abs().sort_values(ascending=False).index)
    for side in SIDES:
        print(f"\n{'offence' if side == 'O' else 'defence'}: bottom tenth to top, points per 100 (+ = underrated); "
              f"line z; bend z; joint z")
        for row in show[show.side == side].head(30).itertuples():
            print(f"  {row.name[:56]:56s} {row.tenths_bottom_to_top:+.2f}  line z {row.line_z:+5.1f}  bend z "
                  f"{row.bend_z:+5.1f}  joint z {row.joint_z:+5.1f}  {row.cls[:12]:12s} "
                  f"{'COUNTS' if row.counts else ''}")
    print("\noffence plus defence: bottom tenth to top, points per 100; line z")
    for row in show[show.side == "total"].head(15).itertuples():
        print(f"  {row.name[:56]:56s} {row.tenths_bottom_to_top:+.2f}  line z {row.line_z:+5.1f}")


def run_audit(args: argparse.Namespace, cfg: dict) -> None:
    """The whole audit: the groups and tests, the readings, the yardstick, the summary, then the paired mode and
    the group shares when asked."""
    paths = Paths.from_config(cfg)
    options = Options.of(args, paths.root)
    started = time.time()
    groups = audit_groups(options.freeze, options.random, paths)
    tests = build_tests(groups)
    audit, readings = random_readings(options, tests, paths, started)
    stick = yardstick(options.draws, options.threshold_from, audit.folds, tests, paths)
    tradeset = (tradeset_leans(tests, paths.root / options.tradeset, paths, cfg).set_index(["test", "side"])
                if options.tradeset else None)
    summary = summary_table(tests, DecisionReadings.of(readings, audit.by_era, bool(options.deadline)), stick,
                            tradeset)
    paths.csv.mkdir(exist_ok=True)
    readings.to_csv(paths.csv / f"{options.out}.csv", index=False)
    audit.tenths.to_csv(paths.csv / f"{options.out}_tenths.csv", index=False)
    summary.to_csv(paths.csv / f"{options.out}_summary.csv", index=False)
    if stick.null is not None:
        write_null(options, stick, audit, tests, paths)
    print_leans(summary, stick, tests, options.draws)
    if options.reference:
        paired_mode(options, audit, tests, paths)
    if options.shares:
        print("\neach group's share of the prior (group-level Shapley values)", flush=True)
        group_shapley_shares(options.shares, groups, paths.csv / f"{options.out}_shares.csv", paths, cfg)
    print(f"\ndone in {time.time() - started:.0f}s: outputs/csv/{options.out}.csv, _summary.csv, _tenths.csv",
          flush=True)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="The held-out audit of the within-season folds.", allow_abbrev=False)
    parser.add_argument("--random", default="lgb_noonc_within", help="fold tags, comma-separated (default %(default)s)")
    parser.add_argument("--mult_tag", default="lgb_noonc_within", help="step 4's folds (default %(default)s)")
    parser.add_argument("--deadline", default="", help="deadline fold tags, comma-separated")
    parser.add_argument("--reference", default="", help="the incumbent's fold tag[:its multiplier tag]")
    parser.add_argument("--control", default="", help="the noise control's fold tag[:its multiplier tag]")
    parser.add_argument("--out", default="audit_lgb_noonc", help="names the outputs (default %(default)s)")
    parser.add_argument("--draws", type=int, default=400, help="career-null draws (default %(default)s)")
    parser.add_argument("--threshold_from", default="", help="read the threshold from this earlier audit's --out")
    parser.add_argument("--tradeset", help="a trade-set alpha table (default the actual-points one when on disk)")
    parser.add_argument("--shares", default="", help="seasons for the group Shapley shares, comma-separated")
    parser.add_argument("--freeze", default="0", help="1: freeze the groups (default %(default)s)")
    parser.add_argument("--correction", default="", help="a group correction the random folds carry")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    use_utf8()
    args = parse_args(argv)
    with one_thread():
        run_audit(args, load_config())


if __name__ == "__main__":
    main()
