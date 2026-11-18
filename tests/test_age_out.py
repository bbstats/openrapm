"""Box Adj's line without age (experiment 142; the owner, 2026-11-18: "I want age removed!")."""
import numpy as np
import pandas as pd
import pytest

from openrapm import second_prior as sp
from openrapm.settings import SECOND_PRIOR_INPUTS

WITH_AGE = "context_types_team_zones_centred"
NO_AGE = "context_types_no_age_team_zones_centred"


def _both_lists(seed: int = 142) -> tuple[dict, dict]:
    """`side_inputs` on the shipped list and on the one without age, from the same made-up players."""
    rng = np.random.default_rng(seed)
    index = pd.Index([3, 5, 8, 13], name="player_id")
    context = pd.DataFrame(rng.normal(size=(4, len(sp.CONTEXT))), index=index, columns=sp.CONTEXT)
    types = pd.DataFrame(rng.normal(size=(4, len(sp.TYPES))), index=index, columns=sp.TYPES)
    box = {side: dict(zip(index, rng.normal(size=4))) for side in ("O", "D")}
    names = [*sp.ALL_TEAM_RATES, *sp.ZONE_RATES]
    rates = {side: pd.DataFrame(rng.normal(size=(4, len(names))), index=index, columns=names) for side in ("O", "D")}
    return sp.side_inputs(box, context, types, WITH_AGE, rates), sp.side_inputs(box, context, types, NO_AGE, rates)


def test_the_list_is_the_shipped_one_without_age():
    shipped, without = sp.INPUT_SETS[WITH_AGE], sp.INPUT_SETS[NO_AGE]
    assert "age" in shipped and "age" not in without
    assert without == [name for name in shipped if name != "age"] and len(without) == 56
    assert without[0] == sp.BOX                      # the Detail view's split reads the box-score prior first
    assert NO_AGE in sp.CENTRED_SETS and NO_AGE in SECOND_PRIOR_INPUTS


def test_it_still_reads_his_other_numbers_and_his_teams_rates():
    assert sp.reads_types(NO_AGE) and sp.reads_team_rates(NO_AGE)
    # his own stats: the shipped 21 less age
    assert len(sp.box_score_positions(WITH_AGE)) == 21 and len(sp.box_score_positions(NO_AGE)) == 20


def test_a_list_that_reads_none_of_step_fives_numbers_still_says_so():
    assert not sp.reads_types("context") and not sp.reads_types("own_teammates_opponents") and not sp.reads_types(None)
    assert all(sp.reads_types(name) for name in sp.INPUT_SETS if name not in ("context", "own_teammates_opponents"))


def test_every_other_input_is_the_number_the_shipped_list_reads():
    # so the line's sums without age are the shipped line's with age's two columns dropped
    shipped, without = _both_lists()
    for side in ("O", "D"):
        assert list(without[side].columns) == sp.INPUT_SETS[NO_AGE]
        assert without[side].equals(shipped[side][sp.INPUT_SETS[NO_AGE]])


def test_a_line_without_age_still_refuses_to_run_without_his_numbers():
    index = pd.Index([1, 2], name="player_id")
    context = pd.DataFrame(0.0, index=index, columns=sp.CONTEXT)
    box = {"O": {1: 0.5, 2: -0.5}, "D": {1: -1.0, 2: 1.0}}
    with pytest.raises(ValueError, match="standardisation"):
        sp.side_inputs(box, context, None, NO_AGE)
