"""The Detail view's split with his own stats learned first (rating_line.first_stage, second_prior.adjustments): the
fit is the one regression's, and only the division of the prior changes."""
import numpy as np
import pandas as pd

from openrapm import rating_line as rl
from openrapm import second_prior as sp
from openrapm.settings import PRODUCT


def _sums(seed=0, n_rows=4000):
    """Two fitted parts and two inputs an end, the first input of each end correlated with the second."""
    rng = np.random.default_rng(seed)
    columns = rng.normal(size=(n_rows, 6))
    columns[:, 3] = 0.8 * columns[:, 2] + 0.6 * columns[:, 3]
    columns[:, 5] = 0.5 * columns[:, 4] + 0.9 * columns[:, 5]
    aim = columns @ np.array([1.0, 1.0, 1.0, 1.0, 0.5, 0.5]) + rng.normal(size=n_rows)
    return rl.RatingSums(Q=columns.T @ columns, c=columns.T @ aim, yy=float(aim @ aim), sw=float(n_rows))


def test_the_first_stage_is_its_inputs_solved_alone_and_zero_on_the_rest():
    sums = _sums()
    got = rl.first_stage([sums], 5, 1e-4, [0])
    assert got["O"].shape == (6, 2) and np.allclose(got["O"], got["O"][0])
    assert got["O"][0, 1] == 0.0 and got["D"][0, 1] == 0.0
    # the correlated input left out, its weight goes to the one kept: 1.0 + 0.8 on offense, 0.5 + 0.25 on defense
    assert abs(got["O"][0, 0] - 1.8) < 0.05 and abs(got["D"][0, 0] - 0.75) < 0.05


def test_the_rest_of_the_fit_is_the_other_inputs_with_the_first_stages_columns_taken_out():
    # Frisch-Waugh-Lovell: stage one, then the others with stage one's columns taken out of them, is the one regression
    sums = _sums()
    first, others = [0, 1, 2, 4], [3, 5]
    A = np.linalg.solve(sums.Q[np.ix_(first, first)], sums.Q[np.ix_(first, others)])
    one = np.linalg.solve(sums.Q[np.ix_(first, first)], sums.c[first])
    rest = sums.Q[np.ix_(others, others)] - sums.Q[np.ix_(others, first)] @ A
    gamma = np.linalg.solve(rest, sums.c[others] - sums.Q[np.ix_(others, first)] @ one)
    together = np.zeros(6)
    together[others] = gamma
    together[first] = one - A @ gamma
    assert np.allclose(together, np.linalg.solve(sums.Q, sums.c), atol=1e-10)


def test_the_box_score_positions_leave_out_only_teammates_opponents_and_team_rates():
    names = sp.INPUT_SETS["context_types_team_rates_centred"]
    read = [names[at] for at in sp.box_score_positions("context_types_team_rates_centred")]
    assert len(names) == 45 and len(read) == 21
    assert {sp.BOX, sp.OTHER_BOX, "team_possession_share", "threes_made_per100", "age"} <= set(read)
    assert not [name for name in read if name.startswith(("teammates_", "opponents_", "team_threes", "team_twos"))]


def _split():
    frame = pd.DataFrame({sp.BOX: [1.0, -1.0, 2.0, 0.0], "own": [0.5, 0.5, -0.5, -0.5], "other": [1.0, -2.0, 0.5, 0.0]},
                         index=[10, 11, 12, 13])
    line = np.tile([2.0, 1.0, 0.5], (3, 1))
    first = np.tile([3.0, 0.5, 0.0], (3, 1))
    team_poss = pd.DataFrame({"player_id": [10, 11, 12, 13], "team": [1, 1, 2, 2],
                              "poss": [100.0, 300.0, 200.0, 200.0]})
    anchor = pd.Series(2.0 * frame[sp.BOX].to_numpy(), index=frame.index)
    values = sp.hold_team_totals(sp.second_priors(frame, line, 2), anchor, team_poss)
    return frame, line, first, team_poss, anchor, values


def test_learned_first_box_adj_still_cancels_on_a_team():
    frame, line, first, team_poss, _, values = _split()
    got = pd.Series(sp.adjustments({"O": values.to_dict()}, {"O": frame}, {"O": line}, 2, {"O": first},
                                   {"O": team_poss})["O"])
    assert abs(got[10] * 100 + got[11] * 300) < 1e-9 and abs(got[12] * 200 + got[13] * 200) < 1e-9


def test_only_the_split_changes_each_teams_box_score_total_stays():
    frame, line, first, team_poss, anchor, values = _split()
    today = pd.Series(sp.adjustments({"O": values.to_dict()}, {"O": frame}, {"O": line}, 2)["O"])
    new = pd.Series(sp.adjustments({"O": values.to_dict()}, {"O": frame}, {"O": line}, 2, {"O": first},
                                   {"O": team_poss})["O"])
    assert not np.allclose(today, new)
    box_today, box_new = values - today, values - new
    weight = team_poss.set_index("player_id").poss
    for ids in ([10, 11], [12, 13]):
        assert abs((box_today[ids] * weight[ids]).sum() - (box_new[ids] * weight[ids]).sum()) < 1e-9
    assert np.allclose(box_today, anchor)


def test_the_published_ratings_use_the_split_learned_first():
    assert PRODUCT.prior.box_first is True
