"""The shot clock rebuilt from a possession's start and its clock events (eracoef.shotclock)."""
import numpy as np
import pandas as pd

from eracoef.shotclock import ClockRules, accuracy, fit_lags, rebuild


def _rows(*rows):
    return pd.DataFrame([dict(season=s, clock=c, poss_start=k, poss_start_clock=t0, events=e)
                         for s, c, k, t0, e in rows])


def test_a_possession_counts_down_from_24():
    r = rebuild(_rows((2015, 690.0, "dreb", 700.0, "")))
    assert r.loc[0, "sc"] == 14.0 and r.loc[0, "reset_kind"] == "dreb" and r.loc[0, "since_reset"] == 10.0


def test_an_offensive_rebound_resets_to_24_then_14():
    old = rebuild(_rows((2015, 680.0, "dreb", 700.0, "oreb@690")))
    new = rebuild(_rows((2020, 680.0, "dreb", 700.0, "oreb@690")))
    assert old.loc[0, "sc"] == 14.0          # 24 - 10
    assert new.loc[0, "sc"] == 4.0           # 14 - 10
    keep24 = rebuild(_rows((2020, 680.0, "dreb", 700.0, "oreb@690")), ClockRules(oreb_14_from=2100))
    assert keep24.loc[0, "sc"] == 14.0


def test_a_defensive_foul_tops_the_clock_up_to_14_only_when_lower():
    low = rebuild(_rows((2015, 670.0, "dreb", 700.0, "foul:def:Personal@680")))     # 4 left at the foul
    high = rebuild(_rows((2015, 690.0, "dreb", 700.0, "foul:def:Personal@695")))    # 19 left: no reset
    assert low.loc[0, "sc"] == 4.0 and low.loc[0, "reset_kind"] == "dfoul"         # 14 - 10
    assert high.loc[0, "sc"] == 14.0 and high.loc[0, "reset_kind"] == "dreb"
    shooting = rebuild(_rows((2015, 670.0, "dreb", 700.0, "foul:def:Shooting@680")))
    assert shooting.loc[0, "reset_kind"] == "dreb"


def test_the_clock_is_off_when_the_period_has_less_left():
    r = rebuild(_rows((2015, 5.0, "made_fg", 15.0, "")))
    assert r.loc[0, "clock_off"] and r.loc[0, "sc_eff"] == 5.0


def test_lags_are_added_by_reset_kind_and_fitted_as_medians():
    rows = _rows(*[(2015, 690.0, "made_fg", 700.0, "")] * 300, *[(2015, 690.0, "dreb", 700.0, "")] * 300)
    base = rebuild(rows)
    truth = np.r_[np.full(300, 11.0), np.full(300, 14.5)]
    lags = fit_lags(base, truth)
    assert abs(lags["made_fg"] + 3.0) < 1e-9 and abs(lags["dreb"] - 0.5) < 1e-9
    again = rebuild(rows, ClockRules(lag=lags))
    acc = accuracy(again["sc_eff"].to_numpy(), truth)
    assert acc["median_abs"] == 0.0 and acc["within_1"] == 1.0
