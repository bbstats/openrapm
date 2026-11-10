"""The shots by zone (experiment 139): the stints' counts, the offence's made and missed by zone, and the team rates
with the placebo beside them."""
import numpy as np
import pandas as pd

from openrapm import second_prior as sp
from openrapm import swap_test as st


def _stints():
    counts = {"fga_rim": 3, "fgm_rim": 2, "fga_smid": 2, "fgm_smid": 1, "fga_lmid": 1, "fgm_lmid": 0,
              "fga_thr": 4, "fgm_thr": 1, "fga_lthr": 1, "fgm_lthr": 0}
    return pd.DataFrame({f"{name}_{side}": [float(value)] for name, value in counts.items() for side in ("h", "a")})


def test_a_three_is_every_three_less_the_long_ones():
    got = st.zone_counts(_stints(), "h")
    assert list(got) == st.ZONE_COUNTS
    assert got["zone_three_made"][0] == 1.0 and got["zone_three_missed"][0] == 2.0
    assert got["long_three_made"][0] == 0.0 and got["long_three_missed"][0] == 1.0
    assert got["rim_made"][0] == 2.0 and got["rim_missed"][0] == 1.0
    assert got["short_mid_missed"][0] == 1.0 and got["long_mid_missed"][0] == 1.0


def test_the_zones_add_up_to_the_threes_and_twos():
    got = st.zone_counts(_stints(), "h")
    threes = sum(got[f"{zone}_{kind}"][0] for zone in ("zone_three", "long_three") for kind in ("made", "missed"))
    twos = sum(got[f"{zone}_{kind}"][0] for zone in ("rim", "short_mid", "long_mid") for kind in ("made", "missed"))
    assert threes == 4.0 and twos == 6.0


def test_the_screens_set_holds_todays_rates_the_zones_and_their_placebo():
    names = sp.INPUT_SETS["context_types_team_rates_zones_centred"]
    assert names[:45] == sp.INPUT_SETS["context_types_team_rates_centred"]
    assert names[45:65] == sp.ZONE_RATES and names[65:] == sp.SHUFFLED_ZONE_RATES
    assert len(sp.box_score_positions("context_types_team_rates_zones_centred")) == 21


def test_the_placebo_is_the_same_numbers_handed_to_other_players():
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(rng.normal(size=(30, len(sp.ZONE_RATES))), columns=sp.ZONE_RATES)
    order = np.random.default_rng(sp.SHUFFLE_DRAW).permutation(30)
    shuffled = frame.to_numpy()[order]
    assert np.allclose(np.sort(shuffled, axis=0), np.sort(frame.to_numpy(), axis=0))
    assert not np.allclose(shuffled, frame.to_numpy())


def test_the_builds_set_counts_shots_by_zone_in_place_of_threes_and_twos():
    names = sp.INPUT_SETS["context_types_team_zones_centred"]
    assert names[:29] == sp.INPUT_SETS["context_and_types_centred"] and len(names) == 29 + 28
    assert not [name for name in names[29:] if "threes" in name or "twos" in name]
    assert {"team_rim_made_on", "team_long_three_missed_off", "team_free_throws_made_on",
            "team_turnovers_off"} <= set(names)
    assert len(sp.box_score_positions("context_types_team_zones_centred")) == 21
    assert sp.reads_team_rates("context_types_team_zones_centred")
