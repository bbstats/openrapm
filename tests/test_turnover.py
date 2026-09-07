"""Teammate turnover (src/eracoef/turnover.py) on hand-built stints."""
import numpy as np
import pandas as pd

from eracoef.turnover import familiar_share, season_turnover, teammates_from_stints


def _stint(h, a, poss_h, poss_a):
    d = {"poss_h": poss_h, "poss_a": poss_a}
    d.update({f"h{i + 1}": p for i, p in enumerate(h)})
    d.update({f"a{i + 1}": p for i, p in enumerate(a)})
    return d


def test_teammates_count_shared_possessions_at_both_ends_and_never_pair_opponents():
    st = pd.DataFrame([_stint([1, 2, 3, 4, 5], [11, 12, 13, 14, 15], 10, 8)])
    t = teammates_from_stints(st)
    assert len(t) == 40                                     # 20 ordered home pairs, 20 away
    assert set(t.loc[t.player_id == 1, "teammate_id"]) == {2, 3, 4, 5}
    assert (t.shared == 18.0).all()                         # both ends of the floor
    assert not ((t.player_id <= 5) & (t.teammate_id >= 11)).any()


def test_zero_possession_stints_add_nothing():
    st = pd.DataFrame([_stint([1, 2, 3, 4, 5], [11, 12, 13, 14, 15], 0, 0)])
    assert len(teammates_from_stints(st)) == 0


def _table():
    # season 1: player 1 plays with 2,3,4,5; season 2 he plays with 2,3 (kept) and 6,7 (new), half the time each
    s1 = teammates_from_stints(pd.DataFrame([_stint([1, 2, 3, 4, 5], [11, 12, 13, 14, 15], 100, 100)])).assign(season=1)
    s2 = teammates_from_stints(pd.DataFrame([_stint([1, 2, 3, 6, 7], [11, 12, 13, 14, 15], 50, 50)])).assign(season=2)
    return pd.concat([s1, s2], ignore_index=True)


def test_familiar_share_is_the_possession_share_with_known_teammates():
    f = familiar_share(_table(), [1], [2], min_shared=100.0).set_index("player_id")
    assert abs(f.loc[1, "familiar"] - 0.5) < 1e-12           # 2 and 3 known, 6 and 7 new, equal possessions
    assert abs(f.loc[1, "turnover"] - 0.5) < 1e-12
    assert abs(f.loc[11, "turnover"] - 0.0) < 1e-12          # the away five never changed
    assert abs(f.loc[6, "turnover"] - 1.0) < 1e-12           # a player with no season-1 possessions at all


def test_min_shared_threshold_and_player_ids_alignment():
    tm = _table()
    f = familiar_share(tm, [1], [2], min_shared=1000.0).set_index("player_id")     # nobody reaches the threshold
    assert abs(f.loc[1, "turnover"] - 1.0) < 1e-12
    f2 = familiar_share(tm, [1], [2], min_shared=100.0, player_ids=[99, 1])
    assert list(f2.player_id) == [99, 1]
    assert np.isnan(f2.turnover.iloc[0]) and abs(f2.turnover.iloc[1] - 0.5) < 1e-12


def test_season_turnover_keeps_only_players_seen_the_season_before():
    st = season_turnover(_table(), min_shared=100.0)
    assert set(st.season) == {2}
    assert 6 not in set(st.player_id) and 1 in set(st.player_id)
