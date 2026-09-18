"""The single-year prior's inputs: the feature list, and the order the aggregation happens in.

Two things here have bitten already.  The board's hand-picked list left out every ratio and every
shot-quality column -- the features an axis-aligned tree cannot build for itself -- and nothing failed,
because `add_derived` skips a family whose inputs are absent instead of raising.  And a possession-weighted
mean of a RATIO is not the ratio of the means, so the aggregation order is a real choice and not a detail.
"""
import numpy as np
import pandas as pd
import pytest

from eracoef import singleyear as sy
from eracoef.bio import PLAYER_INPUTS
from eracoef.gbdt_prior import DERIVED, RATIOS, SHOTQ


def _panel(n=40, seed=0):
    """A frame with every `INPUT_COLUMNS` name, two seasons per player, plausible magnitudes."""
    rng = np.random.default_rng(seed)
    rows = []
    for pid in range(n):
        for season in (2014, 2015):
            r = {c: float(rng.normal(0, 2)) for c in sy.INPUT_COLUMNS}
            r.update(player_id=pid, season=season, side="O",
                     poss=float(rng.uniform(300, 4000)))
            for c in sy.INPUT_COLUMNS:
                if c.startswith("raw_"):
                    r[c] = float(rng.uniform(1, 12))
            r.update(shot_fg2a=float(rng.uniform(50, 900)), shot_fg3a=float(rng.uniform(10, 600)),
                     shot_lg2=0.485, shot_lg3=0.35, shot_lgpps=0.99)
            r.update(shot_fg2m=r["shot_fg2a"] * 0.5, shot_xl2=r["shot_fg2a"] * 0.49,
                     shot_fg3m=r["shot_fg3a"] * 0.36, shot_xl3=r["shot_fg3a"] * 0.35)
            rows.append(r)
    return pd.DataFrame(rows)


def test_the_bio_names_have_not_drifted():
    """`singleyear` repeats `bio.PLAYER_INPUTS` because it may not import a data-layer module."""
    assert sy.BIO == list(PLAYER_INPUTS)


def test_the_feature_list_carries_what_a_tree_cannot_build():
    assert "season" not in sy.PRIOR_FEATURES, "a row pooled over twelve seasons has no season"
    for family, name in ((DERIVED, "DERIVED"), (RATIOS, "RATIOS"), (SHOTQ, "SHOTQ")):
        assert set(family) <= set(sy.PRIOR_FEATURES), f"{name} missing from the single-year prior"
    assert len(sy.PRIOR_FEATURES) == len(set(sy.PRIOR_FEATURES)) == 54


def test_aggregate_builds_every_requested_feature():
    out = sy.aggregate(_panel())
    assert list(out.index.names) == ["player_id"]
    assert not [f for f in sy.PRIOR_FEATURES if f not in out.columns]
    assert out[sy.PRIOR_FEATURES].notna().all().all()


def test_a_missing_input_raises_instead_of_shortening_the_list():
    """The failure mode the guard exists for: drop the raw rates and the ratios quietly vanish."""
    panel = _panel().drop(columns=[c for c in sy.INPUT_COLUMNS if c.startswith("raw_")])
    with pytest.raises(KeyError, match="add_derived did not build"):
        sy.aggregate(panel)


def test_the_ratios_are_built_on_the_average_not_averaged():
    """`ts` of the mean, not the mean of `ts` -- the two differ, and the first is the one we want."""
    panel = _panel()
    got = sy.aggregate(panel)["ts"]

    per_season = sy.season_frame(panel)
    w = per_season.poss
    averaged = (per_season.ts * w).groupby(per_season.player_id).sum() / w.groupby(per_season.player_id).sum()

    num, den, pad, target = RATIOS["ts"]
    mean_raw = {c: (panel[f"raw_{c}"] * panel.poss).groupby(panel.player_id).sum()
                   / panel.poss.groupby(panel.player_id).sum() for c in set(num) | set(den)}
    expected = ((sum(k * mean_raw[c] for c, k in num.items()) + pad * target)
                / (sum(k * mean_raw[c] for c, k in den.items()) + pad))
    assert np.allclose(got.to_numpy(), expected.to_numpy())
    assert not np.allclose(got.to_numpy(), averaged.to_numpy())


def test_prior_rows_joins_an_external_target_instead_of_pooling_one():
    """The target is already a leave-one-out quantity; pooling it again would count it twice."""
    panel = _panel()
    target = pd.DataFrame({"player_id": np.arange(0, 40, 2), "offense": np.linspace(-3, 3, 20),
                           "possessions": np.linspace(500, 9000, 20)})
    rows = sy.prior_rows(target, panel, "offense")
    assert len(rows) == 20
    assert np.allclose(rows.target.to_numpy(), rows.offense.to_numpy())
    assert np.allclose(rows.weight.to_numpy(), rows.possessions.to_numpy())


def _per_team(shares: dict, total=1000.0) -> pd.DataFrame:
    """One row per (player_id, team_id) with `poss_on`, from a dict of player to share list."""
    rows = [{"player_id": pid, "team_id": i, "poss_on": share * total}
            for pid, spread in shares.items() for i, share in enumerate(spread)]
    return pd.DataFrame(rows)


def test_team_movement_is_the_chance_two_possessions_came_from_different_teams():
    """Gini-Simpson, `1 - sum(share ** 2)`, and the three cases that pin the shape down.

    The measure this replaced was `1 - the share on his most-played team`, which read the six-team
    player below as 0.50, the same as the two-team one; that is the defect the owner found on
    2026-09-16.  The first two rows are where the two forms must agree, the third is where they must
    not, and the fourth says a cameo on a seventh team counts a little rather than almost nothing.
    """
    spreads = {0: [1.0],                                  # one team
               1: [0.5, 0.5],                             # an even two-team split
               2: [0.5, 0.1, 0.1, 0.1, 0.1, 0.1],         # five independent contrasts
               3: [0.98, 0.02]}                           # a cameo
    got = sy.team_movement(_per_team(spreads), min_poss=100.0)

    assert got.loc[0] == pytest.approx(0.0)               # never left: no team variation at all
    assert got.loc[1] == pytest.approx(0.5)               # unchanged from the form this replaced
    assert got.loc[2] == pytest.approx(0.7)               # 0.50 under max-share, and 0.70 is right
    assert got.loc[3] == pytest.approx(0.0392)            # small, not zero, and not crushed to nothing
    assert got.loc[2] > got.loc[1], "the shape of the tail has to count"
    assert 1.0 / (1.0 - got.loc[2]) == pytest.approx(10 / 3), "the reciprocal is the effective teams"


def test_team_movement_is_a_share_and_ignores_how_many_possessions_they_are():
    """Doubling every count changes nothing: this measures identification, not exposure.

    Exposure is already the row weight in `reweight_by_movement`, so a movement measure that grew with
    possessions would count it twice.
    """
    spreads = {0: [0.7, 0.3], 1: [0.4, 0.4, 0.2]}
    small = sy.team_movement(_per_team(spreads, total=200.0))
    large = sy.team_movement(_per_team(spreads, total=40000.0))
    assert np.allclose(small.to_numpy(), large.to_numpy())


def test_team_movement_drops_a_player_with_too_little_evidence():
    """`min_poss` is on the player's TOTAL, so a thin career has no movement rather than a noisy one."""
    frame = _per_team({0: [0.5, 0.5], 1: [0.5, 0.5]}, total=1000.0)
    frame.loc[frame.player_id == 1, "poss_on"] = 20.0
    got = sy.team_movement(frame, min_poss=100.0)
    assert list(got.index) == [0]


def test_reweight_by_movement_keeps_the_total_and_moves_only_its_distribution():
    """The booster's regularisation has to mean the same thing before and after the reweighting."""
    train = pd.DataFrame({"target": [1.0, 2.0, 3.0, 4.0], "weight": [100.0, 200.0, 300.0, 400.0]},
                         index=pd.Index([0, 0, 1, 2], name="player_id"))
    movement = pd.Series({0: 0.5, 1: 0.0, 2: 0.25})
    out = sy.reweight_by_movement(train, movement, floor=0.0)

    assert out.weight.sum() == pytest.approx(train.weight.sum())
    assert out.loc[1, "weight"] == pytest.approx(0.0), "floor 0 drops a one-team player outright"
    assert out.loc[2, "weight"] > 0
    # the two rows of player 0 keep their ratio to each other: only players are reweighted, not rows
    first, second = out.loc[0, "weight"].to_numpy()
    assert second / first == pytest.approx(2.0)


def test_reweight_by_movement_with_a_floor_keeps_every_player():
    """The one run of this rule lost by zeroing 29% of the players, stars included; the floor is the fix."""
    train = pd.DataFrame({"target": [1.0, 2.0], "weight": [100.0, 100.0]},
                         index=pd.Index([0, 1], name="player_id"))
    out = sy.reweight_by_movement(train, pd.Series({0: 0.0, 1: 0.5}), floor=0.25)
    assert (out.weight > 0).all()
    assert out.weight.sum() == pytest.approx(200.0)


def test_reweight_by_movement_refuses_to_leave_no_weight_at_all():
    """Silently returning an all-zero training set would fit a constant and look like a bad idea."""
    train = pd.DataFrame({"target": [1.0], "weight": [100.0]},
                         index=pd.Index([0], name="player_id"))
    with pytest.raises(ValueError, match="no training weight"):
        sy.reweight_by_movement(train, pd.Series({0: 0.0}), floor=0.0)


def test_team_movement_is_never_below_the_form_it_replaced():
    """A strict generalisation: equal shares agree, an uneven tail is higher, never lower.

    `sum(share ** 2) <= max(share)` always, with equality only when every team he played for got the
    same share of him.  This is the property that makes the swap a refinement rather than a new idea.
    """
    rng = np.random.default_rng(7)
    spreads = {}
    for pid in range(200):
        raw = rng.dirichlet(np.full(int(rng.integers(1, 8)), float(rng.uniform(0.2, 3.0))))
        spreads[pid] = list(raw)
    frame = _per_team(spreads, total=5000.0)
    new = sy.team_movement(frame)
    totals = frame.groupby("player_id").poss_on.sum()
    old = 1.0 - frame.groupby("player_id").poss_on.max() / totals

    assert (new >= old - 1e-12).all()
    equal_shares = {pid for pid, s in spreads.items() if np.allclose(s, s[0])}
    assert {pid for pid in new.index if abs(new[pid] - old[pid]) < 1e-12} == equal_shares
