"""The per-season role panel (HANDOFF 3.5): everything downstream reads the granularity off the labels.

`scripts/49_role_panel.py --season` writes one row per player per SEASON with labels like "2004-2004".
Nothing else in the chain is told about it: the exclusion set, the pair rows' turnover windows and the
PAST discount all come from the panel's own labels, and on a 3-season panel they must be what they were.
"""
import numpy as np
import pandas as pd
import pytest

from eracoef.gbdt_prior import PAST_DECAY, pair_rows, past_decay_for, past_inputs
from eracoef.holdout import Context
from eracoef.turnover import window_pair_turnover
from eracoef.windows import label_seasons, label_step, labels_covering

BLOCK = ["1998-2000", "2001-2003", "2004-2006"]
SEASON = [f"{s}-{s}" for s in range(1998, 2007)]

CFG = {"_root": ".", "windows": [[1998, 2000], [2001, 2003], [2004, 2006]], "first_season": 1998,
       "last_season": 2006}


def test_label_helpers_read_the_granularity():
    assert label_seasons("1998-2000") == [1998, 1999, 2000]
    assert label_seasons("2004-2004") == [2004]
    assert label_step(BLOCK) == 3.0 and label_step(SEASON) == 1.0
    # the block panel's covering label is the season's own window; the season panel's is the season
    assert labels_covering(BLOCK, [2002, 2003]) == {"2001-2003"}
    assert labels_covering(SEASON, [2002, 2003]) == {"2002-2002", "2003-2003"}
    assert labels_covering(BLOCK, [2000, 2001]) == {"1998-2000", "2001-2003"}


def test_labels_agree_with_the_configured_windows_on_a_block_panel():
    ctx = Context(cfg=CFG)
    ctx.current_h = 2002
    block = pd.DataFrame({"window": BLOCK})
    season = pd.DataFrame({"window": SEASON})
    assert ctx.labels([2001, 2003]) == ctx.labels([2001, 2003], block) == {"2001-2003"}
    # the per-season panel excludes exactly the training seasons and H, not their neighbours
    assert ctx.labels([2001, 2003], season) == {"2001-2001", "2002-2002", "2003-2003"}
    ctx.current_h = None


def test_past_decay_holds_the_reach_in_years():
    assert past_decay_for(BLOCK) == PAST_DECAY                       # the shipped panel is untouched
    assert past_decay_for(SEASON) == pytest.approx(PAST_DECAY ** (1 / 3))
    assert past_decay_for(SEASON, decay=0.5) == 0.5                  # an explicit decay still wins


def _season_panel():
    """One player over four seasons, APM 2 / 4 / 6 / 8 at 1000 possessions each, on both sides."""
    rows = []
    for side in ("O", "D"):
        for i, s in enumerate(range(2000, 2004)):
            rows.append(dict(window=f"{s}-{s}", side=side, player_id=1, poss=1000.0, apm=2.0 * (i + 1),
                             rapm1=1.0 * (i + 1), fg3m=0.1 * i, season=s))
    return pd.DataFrame(rows)


def test_past_on_a_season_panel_discounts_per_season():
    p = _season_panel()
    f = past_inputs(p, "O", exclude={"2003-2003"}, player_ids=[1])
    d = PAST_DECAY ** (1 / 3)
    # his record before 2003: APM 2, 4, 6 at one, two and three seasons back
    w = np.array([d ** 3, d ** 2, d]) * 1000.0
    assert f.past_apm[0] == pytest.approx(float(np.average([2.0, 4.0, 6.0], weights=w)))
    assert f.past_poss[0] == pytest.approx(w.sum() / 1000.0)


def test_pair_rows_on_a_season_panel_are_season_pairs():
    p = _season_panel()
    r = pair_rows(p, "O", features=["fg3m", "season"], target_col="apm", win_decay=0.8)
    assert len(r) == 4 * 3                                            # every ordered pair of his four seasons
    assert set(r.window) == {f"{s}-{s}" for s in range(2000, 2004)}
    # the pair 2000 -> 2001 is one season apart, so its weight is the target's possessions x win_decay
    one = r[(r.window == "2000-2000") & (r.window_to == "2001-2001")]
    assert float(one.weight.iloc[0]) == pytest.approx(1000.0 * 0.8)
    assert float(one.target.iloc[0]) == 4.0


def test_turnover_windows_come_from_the_panel_labels():
    # two seasons, one stayer (player 1 keeps teammate 2) and one mover (player 3 changes every teammate)
    tm = pd.DataFrame({"season": [2000, 2000, 2001, 2001],
                       "player_id": [1, 3, 1, 3], "teammate_id": [2, 4, 2, 5],
                       "shared": [400.0, 400.0, 400.0, 400.0]})
    wins = [(lab, label_seasons(lab)) for lab in ("2000-2000", "2001-2001")]
    t = window_pair_turnover(tm, wins)
    fwd = t[(t.window == "2000-2000") & (t.window_to == "2001-2001")].set_index("player_id")
    assert fwd.turnover[1] == pytest.approx(0.0) and fwd.turnover[3] == pytest.approx(1.0)
