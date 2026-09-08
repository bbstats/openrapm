"""Plus-minus as an input (gbdt_prior.PAST): his record before the window, leak-free on pair rows."""
import numpy as np
import pandas as pd
import pytest

from eracoef.gbdt_prior import PAST, PAST_DECAY, PAST_OWN, pair_rows, past_features, past_inputs, training_rows

WINS = ["2000-2002", "2003-2005", "2006-2008", "2009-2011"]


def _panel():
    rows = []
    # player 1 in all four windows, APM 2, 4, 6, 8 with 1000, 2000, 3000, 4000 possessions; player 2 only in the last two
    for side in ("O", "D"):
        for i, w in enumerate(WINS):
            rows.append(dict(window=w, side=side, player_id=1, poss=1000.0 * (i + 1), apm=2.0 * (i + 1), rapm1=1.0 * (i + 1),
                             fg3m=0.1 * i, season=2001 + 3 * i, blend=0.0))
            if i >= 2:
                rows.append(dict(window=w, side=side, player_id=2, poss=500.0, apm=-1.0, rapm1=-0.5,
                                 fg3m=0.0, season=2001 + 3 * i, blend=0.0))
    return pd.DataFrame(rows)


def test_past_features_discount_exclude_and_zero_without_a_past():
    p = _panel()
    keys = pd.DataFrame({"player_id": [1, 1, 1, 2, 2], "window": [WINS[0], WINS[2], WINS[3], WINS[2], WINS[3]]})
    f = past_features(p[p.side == "O"], WINS, keys)
    assert list(f.columns) == PAST_OWN
    assert f.past_apm[0] == 0 and f.past_poss[0] == 0                           # nothing before the first window
    d = PAST_DECAY
    w0, w1 = 1000 * d ** 2, 2000 * d                                             # window 2: windows 0 and 1 before it
    assert abs(f.past_apm[1] - (w0 * 2 + w1 * 4) / (w0 + w1)) < 1e-9 and abs(f.past_poss[1] - (w0 + w1) / 1000) < 1e-9
    assert abs(f.past_rapm[1] - (w0 * 1 + w1 * 2) / (w0 + w1)) < 1e-9
    assert f.past_apm[3] == 0 and abs(f.past_apm[4] - (-1.0)) < 1e-9             # player 2's past starts at window 2
    # the pair's target window is left out of the past, and so is the exclusion set
    keys2 = pd.DataFrame({"player_id": [1], "window": [WINS[3]], "window_to": [WINS[1]]})
    g = past_features(p[p.side == "O"], WINS, keys2, exclude={WINS[0]})
    assert abs(g.past_apm[0] - 6.0) < 1e-9 and abs(g.past_poss[0] - 3000 * d / 1000) < 1e-9   # only window 2 is left


def test_pair_rows_carry_past_and_pooled_rows_refuse_it():
    p = _panel()
    feats = ["fg3m", "season", *PAST]
    out = pair_rows(p, "O", (), feats, target_col="apm")
    r = out[(out.player_id == 1) & (out.window == WINS[2]) & (out.window_to == WINS[1])].iloc[0]
    assert abs(r.past_apm - 2.0) < 1e-9                                          # window 1 is the target: only window 0 is past
    r2 = out[(out.player_id == 1) & (out.window == WINS[2]) & (out.window_to == WINS[3])].iloc[0]
    d = PAST_DECAY
    assert abs(r2.past_apm - (1000 * d ** 2 * 2 + 2000 * d * 4) / (1000 * d ** 2 + 2000 * d)) < 1e-9
    assert set(PAST) <= set(out.columns) and "fg3m" in out.columns
    with pytest.raises(ValueError):
        training_rows(p, "O", (), feats, target_col="apm")


def test_past_inputs_use_only_the_windows_before_the_block():
    p = _panel()
    f = past_inputs(p, "O", {WINS[2], WINS[3]}, [1, 2, 99])
    d = PAST_DECAY
    w0, w1 = 1000 * d ** 2, 2000 * d
    assert abs(f.past_apm[0] - (w0 * 2 + w1 * 4) / (w0 + w1)) < 1e-9
    assert f.past_apm[1] == 0 and f.past_poss[1] == 0                            # player 2 has nothing before the block
    assert f.past_apm[2] == 0 and len(f) == 3                                    # never seen at all
    g = past_inputs(p, "D", set(), [1])                                          # nothing excluded: everything is past
    assert g.past_poss[0] > f.past_poss[0]


def test_cross_side_past_is_the_other_side_record():
    from eracoef.gbdt_prior import past_all
    p = _panel()
    p.loc[p.side == "D", "apm"] = -3.0                                          # defense reads -3 everywhere
    keys = pd.DataFrame({"player_id": [1], "window": [WINS[3]]})
    f = past_all(p, "O", WINS, keys)
    assert set(PAST) <= set(f.columns)
    assert abs(f.past_apm[0] - f.past_apm_o[0]) < 1e-12 and abs(f.past_apm_d[0] + 3.0) < 1e-9
    assert abs(f.past_poss_o[0] - f.past_poss_d[0]) < 1e-12                     # the same windows behind both
    g = past_all(p, "D", WINS, keys)
    assert abs(g.past_apm[0] + 3.0) < 1e-9 and abs(g.past_apm_o[0] - f.past_apm_o[0]) < 1e-12
