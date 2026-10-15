"""scripts/checks/145_piece_multipliers.py (the screen for experiment 61): the pieces' multipliers on made-up folds
whose truth is known, the shipped step 4 reproduced, the groups and the gate; and the script starts."""
import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from openrapm import prior_shrink
from openrapm.data.within_folds import Fold

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "checks" / "145_piece_multipliers.py"
N_PLAYERS, N_TEAM_GAMES = 14, 80


@pytest.fixture(scope="module")
def check():
    spec = importlib.util.spec_from_file_location("piece_multipliers", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_help_lists_its_flag():
    result = subprocess.run([sys.executable, str(SCRIPT), "--help"], capture_output=True, text=True,
                            encoding="utf-8", cwd=ROOT, timeout=300)
    assert result.returncode == 0, result.stderr
    assert "--folds" in result.stdout


# ------------------------------------------------------------------------------------------- the groups and gate
def test_the_groups_cut_by_the_share_of_the_weight_below_a_value(check):
    assert check.weighted_groups([1, 2, 3, 4], [1, 1, 1, 1], (0.5,)).tolist() == [0, 0, 1, 1]
    assert check.weighted_groups([4, 1, 3, 2], [1, 1, 1, 1], (0.5,)).tolist() == [1, 0, 1, 0]      # any order
    # ten equal players, cut at half and nine tenths of the possessions: five, four, one
    assert check.weighted_groups(np.arange(10), np.ones(10), (0.5, 0.9)).tolist() == [0] * 5 + [1] * 4 + [2]
    # a player is placed by the middle of his own weight: one holding 8 of 10 sits at 0.6
    assert check.weighted_groups([1, 2, 3], [1, 1, 8], (0.5, 0.9)).tolist() == [0, 0, 1]
    assert check.weighted_groups([1, 2], [0, 0], (0.5,)).tolist() == [0, 0]


def test_the_gate_wants_size_t_and_both_halves(check):
    agree = dict(early=2.5, late=3.1)
    assert check.passes_gate(0.3, 3.5, agree)
    assert not check.passes_gate(0.19, 3.5, agree)                                     # too small a difference
    assert not check.passes_gate(0.3, 2.9, agree)                                      # not significant over seasons
    assert not check.passes_gate(0.3, 3.5, dict(early=2.5, late=1.9))                  # one half unconvinced
    assert not check.passes_gate(0.3, 3.5, dict(early=2.5, late=-2.5))                 # the halves disagree
    assert check.passes_gate(-0.3, -3.5, dict(early=-2.5, late=-3.1))
    assert not check.passes_gate(0.3, 3.5, dict(early=float("nan"), late=3.0))


def test_the_paired_t_is_the_mean_over_its_standard_error_with_nan_left_out(check):
    mean, t, n = check.paired_t([1.0, 2.0, 3.0, np.nan])
    assert (mean, n) == (2.0, 3) and t == pytest.approx(2.0 / (1.0 / np.sqrt(3.0)))
    assert np.isnan(check.paired_t([1.0, 2.0])[1])


# ---------------------------------------------------------------------------------------------- made-up folds
TRUE = {"prior": dict(O=0.7, D=0.9), "on_court": dict(O=1.5, D=1.2), "sat": dict(O=0.5, D=0.8),
        "missed": dict(O=1.0, D=2.0), "team_sos": dict(O=2.0, D=0.4)}


def _split(rng, total, names):
    """`total` as the sum of random pieces with these names, the last one the remainder."""
    parts = [rng.normal(0.0, 0.5, len(total)) for _ in names[:-1]]
    parts.append(total - sum(parts))
    return dict(zip(names, parts))


def make_fold(seed: int, season: int = 2020):
    """A fold whose points are exactly its rated players' pieces times the true multipliers, plus a level and a home
    edge of its own: (the Fold, its pieces table).  The players' games' parts u are the sums of their pieces."""
    rng = np.random.default_rng(seed)
    ids = np.arange(100, 100 + N_PLAYERS)
    shares = {side: sp.csr_matrix(5.0 * rng.dirichlet(np.ones(N_PLAYERS), N_TEAM_GAMES)) for side in "OD"}
    rows, pieces = {}, []
    for side, (prior_name, u_name, poss_name) in dict(O=("prior_off", "u_off", "poss_off"),
                                                       D=("prior_def", "u_def", "poss_def")).items():
        u = rng.normal(0.0, 1.0, N_PLAYERS)
        rows[side] = pd.DataFrame({"player_id": ids, prior_name: rng.normal(0.0, 1.5, N_PLAYERS), u_name: u,
                                   poss_name: rng.uniform(100, 3000, N_PLAYERS),
                                   "height_in": rng.normal(78, 3, N_PLAYERS),
                                   "team_possession_share": rng.uniform(0.05, 0.3, N_PLAYERS)})
        frame = pd.DataFrame({"player_id": ids, "side": side, "u": u, "sd_u": rng.uniform(0.5, 1.0, N_PLAYERS)})
        for name, value in {**_split(rng, u, ["on_court", "sat", "missed", "team_sos"]),
                            **_split(rng, u, ["twos", "threes", "free_throws"])}.items():
            frame[name] = value
        pieces.append(frame)
    controls = np.column_stack([rng.choice([-1.0, 1.0], N_TEAM_GAMES), np.ones(N_TEAM_GAMES),
                                np.zeros(N_TEAM_GAMES), np.zeros(N_TEAM_GAMES)])          # no playoff games: rank 2
    points = controls @ np.array([0.7, 108.0, 0.0, 0.0])
    for side, sign in (("O", 1.0), ("D", -1.0)):
        frame, name = pieces[0 if side == "O" else 1], dict(O="prior_off", D="prior_def")[side]
        parts = dict(prior=rows[side][name].to_numpy())
        parts.update({piece: frame[piece].to_numpy() for piece in ("on_court", "sat", "missed", "team_sos")})
        points = points + sign * sum(TRUE[piece][side] * (shares[side] @ value) for piece, value in parts.items())
    rated = sum(sign * (shares[side] @ (rows[side][p].to_numpy() + rows[side][u].to_numpy()))
                for side, sign, p, u in (("O", 1.0, "prior_off", "u_off"), ("D", -1.0, "prior_def", "u_def")))
    fold = Fold(stem=f"{season}_r0_f{seed}", season=season, repeat=0, fold=seed, fill_o=0.0, fill_d=0.0, y=points,
                w=rng.uniform(80, 120, N_TEAM_GAMES), F=controls, game_idx=np.arange(N_TEAM_GAMES), base=rated,
                rows=rows, Z=shares)
    return fold, pd.concat(pieces, ignore_index=True)


@pytest.fixture(scope="module")
def season(check):
    folds = [make_fold(seed) for seed in range(4)]
    accumulated = check.Season()
    for fold, pieces in folds:
        accumulated.add(fold, pieces)
    return accumulated, [fold for fold, _ in folds]


def test_each_pieces_multiplier_is_recovered_when_the_points_are_made_of_them(check, season):
    accumulated, _ = season
    got = check.solve(accumulated.sums["source"]["gram"], accumulated.sums["source"]["rhs"])
    columns = check.FAMILIES["source"]["columns"]
    for side_index, side in enumerate("OD"):
        for k, piece in enumerate(columns):
            assert got[side_index * len(columns) + k] == pytest.approx(TRUE[piece][side], abs=1e-8), (piece, side)


def test_the_shipped_step_is_the_prior_part_fitted_with_the_games_part_held_at_one(check, season):
    accumulated, folds = season
    mine = check.solve(accumulated.sums["step4"]["gram"], accumulated.sums["step4"]["rhs"])
    assert mine == pytest.approx(np.array(prior_shrink.fit_multipliers(folds)), abs=1e-9)


def test_a_piece_nobody_has_is_nan_rather_than_zero(check):
    gram, rhs = np.diag([2.0, 0.0, 3.0]), np.array([1.0, 0.0, 3.0])
    got = check.solve(gram, rhs)
    assert got[0] == pytest.approx(0.5) and np.isnan(got[1]) and got[2] == pytest.approx(1.0)


def test_the_size_groups_and_the_thirds_add_up_to_the_whole(check, season):
    _, folds = season
    fold, pieces = make_fold(0)
    for side in "OD":
        values = check.side_values(fold, pieces, side)
        for label in ("u_size", "u_sd"):
            assert sum(values[k] for k in values if k.startswith(label)) == pytest.approx(values["u"])
        for label in ("prior_height", "prior_role"):
            assert sum(values[k] for k in values if k.startswith(label)) == pytest.approx(values["prior"])


def test_the_fold_response_takes_off_what_unrated_players_add(check):
    fold, pieces = make_fold(1)
    shifted = Fold(**{**fold.__dict__, "base": fold.base + 3.0})            # a stand-in player adds 3 points everywhere
    _, _, response = check.fold_columns(fold, pieces)
    _, _, moved = check.fold_columns(shifted, pieces)
    assert response - moved == pytest.approx(np.full(N_TEAM_GAMES, 3.0))


def test_the_held_out_error_of_a_family_that_is_true_is_zero(check):
    seasons = np.arange(2010, 2022)
    accumulated = {int(s): check.Season() for s in seasons}
    for s in seasons:
        for seed in range(2):
            fold, pieces = make_fold(seed + 7 * int(s), int(s))
            accumulated[int(s)].add(fold, pieces)
    heldout, errors = check.held_out(accumulated, seasons)
    assert errors["source"].abs().max() < 1e-6                          # the points are exactly its pieces
    assert errors["step4"].min() > 0.1                                  # the shipped step cannot make them
    against = heldout[(heldout.family == "source") & (heldout.against == "step4")].iloc[0]
    assert against.mean_change < -0.1 and against.seasons_better == len(seasons)
