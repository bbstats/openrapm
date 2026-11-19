"""Box Adj's two inputs on the defense end alone (experiment 143; the owner, 2026-11-19: "Two inputs first")."""
import numpy as np
import pandas as pd
import pytest

from openrapm import second_prior as sp
from openrapm.settings import SECOND_PRIOR_INPUTS
from openrapm.swap_test import DEF, OFF

NO_AGE = "context_types_no_age_team_zones_centred"
TWO = "context_types_no_age_team_zones_defence_only_centred"


def _stints(rows: list) -> pd.DataFrame:
    """Stint rows from (game, team_off, team_def, offense five, defense five, possessions, close game)."""
    out = []
    for game, team_off, team_def, five_o, five_d, poss, close in rows:
        out.append({"game_id": game, "team_off": team_off, "team_def": team_def, "poss": float(poss),
                    "clutch": float(close), **dict(zip(OFF, five_o)), **dict(zip(DEF, five_d))})
    return pd.DataFrame(out)


def test_the_list_is_the_shipped_one_and_the_two():
    assert sp.INPUT_SETS[TWO] == [*sp.INPUT_SETS[NO_AGE], "off_rebound_share", "close_game_share_over_share"]
    assert TWO in sp.CENTRED_SETS and TWO in SECOND_PRIOR_INPUTS
    assert sp.reads_defence_only(TWO) and not sp.reads_defence_only(NO_AGE) and not sp.reads_defence_only(None)
    assert sp.INPUT_SETS[TWO][0] == sp.BOX


def test_close_game_share_over_share():
    # team 1 defends; player 1 is on for every possession, player 6 sits the close stint: 100 possessions, 20 close
    on, bench = (1, 2, 3, 4, 5), (6, 2, 3, 4, 5)
    attackers = (11, 12, 13, 14, 15)
    stints = _stints([("g", 9, 1, attackers, on, 60, 0), ("g", 9, 1, attackers, bench, 20, 0),
                      ("g", 9, 1, attackers, on, 20, 1)])
    shares = sp.close_game_shares(stints, "D")
    # player 1: all of the close ones, 80 of 100 overall: (1.0 - 0.8) x 20 / (20 + 50)
    assert shares[1] == pytest.approx(0.2 * 20 / 70)
    # player 6: none of the close ones, 20 of 100 overall
    assert shares[6] == pytest.approx(-0.2 * 20 / 70)
    assert shares[2] == pytest.approx(0.0)
    # the docstring's example: 30% of 150 close possessions, 20% of all
    assert (0.30 - 0.20) * 150 / (150 + sp.CLOSE_GAME_PAD) == pytest.approx(0.075)


def test_no_close_games_reads_nothing():
    five, attackers = (1, 2, 3, 4, 5), (11, 12, 13, 14, 15)
    shares = sp.close_game_shares(_stints([("g", 9, 1, attackers, five, 50, 0)]), "D")
    assert (shares == 0.0).all()


def test_offense_end_reads_zero_and_defense_end_the_numbers():
    rng = np.random.default_rng(143)
    index = pd.Index([3, 5, 8, 13], name="player_id")
    context = pd.DataFrame(rng.normal(size=(4, len(sp.CONTEXT))), index=index, columns=sp.CONTEXT)
    types = pd.DataFrame(rng.normal(size=(4, len(sp.TYPES))), index=index, columns=sp.TYPES)
    box = {side: dict(zip(index, rng.normal(size=4))) for side in ("O", "D")}
    names = [*sp.ALL_TEAM_RATES, *sp.ZONE_RATES]
    rates = {side: pd.DataFrame(rng.normal(size=(4, len(names))), index=index, columns=names) for side in ("O", "D")}
    numbers = pd.DataFrame({"off_rebound_share": [0.10, 0.02, 0.05], "close_game_share_over_share": [0.03, -0.01, 0.0]},
                           index=pd.Index([3, 5, 8], name="player_id"))
    two = sp.side_inputs(box, context, types, TWO, rates, numbers)
    shipped = sp.side_inputs(box, context, types, NO_AGE, rates)
    for side in ("O", "D"):
        assert list(two[side].columns) == sp.INPUT_SETS[TWO]
        assert two[side][sp.INPUT_SETS[NO_AGE]].equals(shipped[side])      # every other input as shipped
    assert (two["O"][sp.DEFENCE_ONLY] == 0.0).all().all()
    assert two["D"].loc[3, "off_rebound_share"] == pytest.approx(0.10)
    # player 13 has none: the mean of the three who do
    assert two["D"].loc[13, "off_rebound_share"] == pytest.approx(np.mean([0.10, 0.02, 0.05]))
    with pytest.raises(ValueError, match="none were given"):
        sp.side_inputs(box, context, types, TWO, rates)


def test_the_share_is_built_from_the_raw_rates_when_the_rows_lack_it():
    # the features docstring's example: 3.0 offensive and 9.0 defensive rebounds per 100 give 0.244
    five, attackers = (1, 2, 3, 4, 5), (11, 12, 13, 14, 15)
    stints = _stints([("g", 9, 1, attackers, five, 50, 0)])
    rows = pd.DataFrame({"player_id": [1], "off_rebounds_per100_uncentred": [3.0],
                         "def_rebounds_per100_uncentred": [9.0]})
    numbers = sp.defence_only_numbers(stints, rows)
    assert numbers.loc[1, "off_rebound_share"] == pytest.approx((3.0 + 3 * 0.22) / (3.0 + 9.0 + 3))
    built = rows.assign(off_rebound_share=0.5)
    assert sp.defence_only_numbers(stints, built).loc[1, "off_rebound_share"] == pytest.approx(0.5)
