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
    assert np.allclose(rows.row_weight.to_numpy(), rows.possessions.to_numpy())


def _career_panel(seed=3):
    """Players with four, three, two and one season, one with a gap, on the sides and scale of `_panel`."""
    base = _panel(n=5, seed=seed)
    template = base.iloc[0].to_dict()
    careers = {0: [2010, 2011, 2012, 2013], 1: [2010, 2011, 2013], 2: [2012, 2013], 3: [2011], 4: [2010, 2012]}
    rng = np.random.default_rng(seed)
    rows = []
    for pid, seasons in careers.items():
        for s in seasons:
            r = {c: float(rng.normal(0, 2)) for c in sy.INPUT_COLUMNS}
            for c in sy.INPUT_COLUMNS:
                if c.startswith("raw_"):
                    r[c] = float(rng.uniform(1, 12))
            for c in ("shot_fg2a", "shot_fg3a", "shot_lg2", "shot_lg3", "shot_lgpps", "shot_fg2m",
                      "shot_xl2", "shot_fg3m", "shot_xl3"):
                r[c] = template[c]
            r.update(player_id=pid, season=s, side="O", poss=float(rng.uniform(300, 4000)))
            rows.append(r)
    return pd.DataFrame(rows)


def _labels_for(panel, sets, unseen=(2020,)):
    """A distinct, recognisable label per excluded set: player id + the sum of the excluded seasons / 1e4.

    A player with no season outside the set gets no row, exactly as `LeaveSeasonOutRAPM` gives none to a
    player below its possession count."""
    out = {}
    for chunk in sets:
        key = tuple(sorted(set(unseen) | set(chunk)))
        kept = panel[~panel.season.isin(key)]
        ids = np.sort(kept.player_id.unique())
        out[key] = pd.DataFrame({"player_id": ids, "offense": ids + sum(key) / 1e4,
                                 "possessions": kept.groupby("player_id").poss.sum().reindex(ids).to_numpy()})
    return out


def test_outside_labels_give_each_chunk_the_rapm_of_the_seasons_outside_it():
    """Experiment 25: a chunk row's label is fit without the chunk's own seasons; the career row keeps its own."""
    panel = _career_panel()
    unseen = (2020,)
    career = _labels_for(panel, [()], unseen)[(2020,)]
    sets = sy.chunk_season_sets(panel)
    labels = _labels_for(panel, sets, unseen)
    rows = sy.chunk_rows(career, panel, "offense", labels=labels, unseen=unseen)

    for pid, row in rows.iterrows():
        key = tuple(int(s) for s in row.label_key.split(","))
        assert set(unseen) <= set(key)
        frame = labels[key] if key in labels else career
        assert row.target == frame.set_index("player_id").offense.loc[pid]
    whole = rows[rows.label_key == "2020"]
    assert sorted(whole.index) == sorted(career.player_id), "one career row per player, labelled from unseen"
    chunks = rows[rows.label_key != "2020"]
    # player 3 has one season: its only chunk IS his career, so there is nothing outside it and it goes
    assert 3 not in chunks.index
    # player 1 played 2010, 2011, 2013: the 2011 + 2013 chunk is consecutive in his seasons, not the calendar
    assert "2011,2013,2020" in set(chunks.loc[1].label_key)
    # weights: every player's chunk rows together still weigh what his career row weighs
    for pid in chunks.index.unique():
        total = chunks.loc[[pid]].row_weight.sum()
        assert total == pytest.approx(whole.loc[pid].row_weight)


def test_a_one_season_chunk_carries_exactly_the_per_season_label():
    """The owner's check: a 1-season chunk of season s has `season_rows`'s label for s, to the bit."""
    panel = _career_panel()
    unseen = (2020,)
    sets = sy.chunk_season_sets(panel)
    labels = _labels_for(panel, sets, unseen)
    career = _labels_for(panel, [()], unseen)[(2020,)]
    rows = sy.chunk_rows(career, panel, "offense", labels=labels, unseen=unseen)
    per_season = sy.season_rows({s: labels[tuple(sorted({2020, s}))] for s in sorted(panel.season.unique())},
                                panel, "offense")
    ones = rows[(rows.chunk_seasons == 1) & (rows.label_key != "2020")]
    assert len(ones) > 0
    looked_up = per_season.set_index("season", append=True).target
    for pid, row in ones.iterrows():
        season = [s for s in map(int, row.label_key.split(",")) if s not in unseen][0]
        assert row.target == looked_up.loc[(pid, season)]


def test_label_weights_give_each_chunk_the_possessions_its_own_label_rests_on():
    """After experiment 25: a chunk row weighs `share x its own label's possessions`; the career row is untouched."""
    panel = _career_panel()
    unseen = (2020,)
    career = _labels_for(panel, [()], unseen)[(2020,)]
    labels = _labels_for(panel, sy.chunk_season_sets(panel), unseen)
    by_career = sy.chunk_rows(career, panel, "offense", labels=labels, unseen=unseen)
    by_label = sy.chunk_rows(career, panel, "offense", labels=labels, unseen=unseen, weight_by="label")
    whole = by_label.label_key == "2020"
    assert np.array_equal(by_label.row_weight[whole].to_numpy(), by_career.row_weight[whole].to_numpy())
    chunks_c, chunks_l = by_career[~whole], by_label[~whole]
    assert np.allclose(chunks_l.row_weight.to_numpy(),
                       (chunks_c.row_weight * chunks_c.label_possessions / chunks_c.possessions).to_numpy())
    assert (chunks_l.label_possessions < chunks_l.possessions).all(), "an outside label rests on less than the career"
    assert (chunks_l.row_weight < chunks_c.row_weight).all()
    with pytest.raises(ValueError, match="weight_by"):
        sy.chunk_rows(career, panel, "offense", weight_by="possessions")


def test_career_labels_are_the_incumbent_and_ignore_the_new_arguments():
    """`labels=None` is the incumbent: every chunk row carries the career label, as before."""
    panel = _career_panel()
    career = _labels_for(panel, [()])[(2020,)]
    rows = sy.chunk_rows(career, panel, "offense")
    per_player = career.set_index("player_id").offense
    assert np.array_equal(rows.target.to_numpy(), per_player.reindex(rows.index).to_numpy())
    assert (rows.label_key == "").all()


def test_a_missing_outside_label_raises_rather_than_dropping_rows():
    panel = _career_panel()
    career = _labels_for(panel, [()])[(2020,)]
    labels = _labels_for(panel, sy.chunk_season_sets(panel)[:2])
    with pytest.raises(KeyError, match="no label for the excluded season sets"):
        sy.chunk_rows(career, panel, "offense", labels=labels, unseen=(2020,))


def test_the_piece_names_have_not_drifted():
    """`singleyear` repeats `pieces.COLUMNS` because `pieces` imports it."""
    from eracoef import pieces
    assert sy.PIECES == pieces.COLUMNS
    assert set(sy.PIECES + sy.PO_SHARE) <= set(sy.INPUT_COLUMNS)


def test_the_charlotte_move_is_one_franchise():
    """1610612766 is New Orleans's franchise through 2002 and the Bobcats' from 2005; nothing else moves."""
    got = sy.franchise([1610612766, 1610612766, 1610612766, 1610612740, 1610612747],
                       [2002, 2005, 1997, 2003, 2002])
    assert got.tolist() == [1610612740, 1610612766, 1610612740, 1610612740, 1610612747]


def test_harmonic_overlap_is_one_for_identical_spreads_and_zero_for_none_shared():
    left = pd.DataFrame({"key": [1, 1, 2, 3], "team_id": [10, 11, 10, 12], "poss_on": [500.0, 500.0, 300.0, 100.0]})
    right = pd.DataFrame({"key": [1, 1, 2, 3], "team_id": [10, 11, 10, 13], "poss_on": [50.0, 50.0, 900.0, 100.0]})
    got = sy.harmonic_overlap(left, right)
    assert got.loc[1] == pytest.approx(1.0)       # the same 50/50 split, whatever the possessions
    assert got.loc[2] == pytest.approx(1.0)       # one team both times
    assert got.loc[3] == 0.0                      # no team in common
    half = sy.harmonic_overlap(left[left.key == 1], pd.DataFrame({"key": [1], "team_id": [10], "poss_on": [9.0]}))
    assert half.loc[1] == pytest.approx(2 * 0.5 * 1.0 / 1.5)


def _team_shares(panel, teams):
    """One (player_id, season, team_id, poss_on) row per panel row, from a {(player, season): team} map."""
    rows = panel[["player_id", "season", "poss"]].rename(columns={"poss": "poss_on"})
    return rows.assign(team_id=[teams.get((p, s), 99) for p, s in zip(rows.player_id, rows.season)])


def test_same_team_is_one_on_career_rows_and_zero_on_a_team_he_never_played_for_otherwise():
    panel = _career_panel()
    unseen = (2020,)
    career = _labels_for(panel, [()], unseen)[(2020,)]
    labels = _labels_for(panel, sy.chunk_season_sets(panel), unseen)
    # player 0: team 1 in 2010, 2011 and 2013, team 2 in 2012; player 2 one team throughout
    teams = {(0, 2010): 1, (0, 2011): 1, (0, 2012): 2, (0, 2013): 1, (2, 2012): 5, (2, 2013): 5}
    rows = sy.chunk_rows(career, panel, "offense", labels=labels, unseen=unseen,
                         shares=_team_shares(panel, teams))
    whole = rows.label_key == "2020"
    assert (rows.loc[whole, sy.SAME_TEAM] == 1.0).all()
    zero = rows.loc[[0]]
    only_2012 = zero[zero.label_key == "2012,2020"]
    assert only_2012[sy.SAME_TEAM].tolist() == [0.0], "2012 was his only season on team 2"
    only_2010 = zero[zero.label_key == "2010,2020"]
    # the label seasons are 2011, 2012, 2013: team 1 holds their 2011 and 2013 share, the chunk is all team 1
    rest = panel[(panel.player_id == 0) & panel.season.isin([2011, 2012, 2013])]
    share_1 = rest[rest.season != 2012].poss.sum() / rest.poss.sum()
    assert only_2010[sy.SAME_TEAM].iloc[0] == pytest.approx(2 * share_1 / (1 + share_1))
    assert (rows.loc[[2], sy.SAME_TEAM] == 1.0).all(), "one team throughout: every row is the same team"
    assert rows[sy.SAME_TEAM].between(0.0, 1.0 + 1e-12).all()


def test_team_season_is_the_rows_biggest_team_season():
    """Experiment 27's group: a chunk's biggest (franchise, season) by possessions; the career row's over his span."""
    panel = _career_panel()
    unseen = (2020,)
    career = _labels_for(panel, [()], unseen)[(2020,)]
    labels = _labels_for(panel, sy.chunk_season_sets(panel), unseen)
    teams = {(0, 2010): 1, (0, 2011): 1, (0, 2012): 2, (0, 2013): 1}
    shares = _team_shares(panel, teams)
    rows = sy.chunk_rows(career, panel, "offense", labels=labels, unseen=unseen, shares=shares)
    zero = rows.loc[[0]]
    only_2012 = zero[zero.label_key == "2012,2020"][sy.TEAM_SEASON].iloc[0]
    assert only_2012 == 2012 * 10**10 + 2
    mine = panel[panel.player_id == 0].set_index("season").poss
    top = int(mine.idxmax())
    assert zero[zero.label_key == "2020"][sy.TEAM_SEASON].iloc[0] == top * 10**10 + teams[(0, top)]
    pair = zero[zero.label_key == "2010,2011,2020"][sy.TEAM_SEASON].iloc[0]
    assert pair == int(mine.loc[[2010, 2011]].idxmax()) * 10**10 + 1
    assert (rows[sy.TEAM_SEASON] > 0).all(), "every row here has a team"


def test_the_aging_curve_recovers_a_known_curve_from_season_to_season_changes():
    """The delta method on noiseless data: a peak at 27 falling off as 0.1 x (age - 27)^2 on either side.

    A change from a to a + 1 is then exactly linear in a + 0.5, so the fit must return the curve itself."""
    rows = []
    for pid in range(40):
        start = 19 + pid % 12
        for k in range(8):
            age = start + k
            level = pid * 0.1 - 0.1 * (age - 27) ** 2
            rows.append(dict(player_id=pid, season=2000 + k, age=float(age), rating=level, poss=1000.0 + pid))
    curve = sy.AgingCurve(pd.DataFrame(rows), "rating", degree=2)
    for age in (20.0, 23.0, 27.0, 31.0, 36.0):
        assert curve(age) == pytest.approx(-0.1 * (age - 27) ** 2, abs=1e-9)
    assert sy.AgingCurve(pd.DataFrame(rows), "rating", exclude=range(2000, 2008)).pairs == 0


def test_age_adjusted_labels_move_chunks_to_their_own_age_and_never_the_career_row():
    panel = _career_panel()
    panel["age"] = 20.0 + (panel.season - 2010) * 3.0             # three years a season, so ages differ a lot
    unseen = (2020,)
    career = _labels_for(panel, [()], unseen)[(2020,)]
    labels = _labels_for(panel, sy.chunk_season_sets(panel), unseen)

    def curve(age):
        return 0.5 * (np.asarray(age, dtype=float) - 27.0)       # half a point a year, everywhere
    plain = sy.chunk_rows(career, panel, "offense", labels=labels, unseen=unseen)
    moved = sy.chunk_rows(career, panel, "offense", labels=labels, unseen=unseen, age_curve=curve)
    whole = moved.label_key == "2020"
    assert np.array_equal(moved.target[whole].to_numpy(), plain.target[whole].to_numpy())
    assert (moved.age_shift[whole] == 0).all()
    # player 0's 2010 chunk: his own age 20 against the possession-weighted age of 2011-2013
    mine = panel[panel.player_id == 0].set_index("season")
    rest = mine.loc[[2011, 2012, 2013]]
    expected = curve(20.0) - float(np.average(curve(rest.age), weights=rest.poss))
    row = moved.loc[[0]][moved.loc[[0]].label_key == "2010,2020"]
    assert row.age_shift.iloc[0] == pytest.approx(expected)
    assert row.target.iloc[0] == pytest.approx(plain.loc[[0]][plain.loc[[0]].label_key == "2010,2020"].target.iloc[0]
                                               + expected)
    halved = sy.chunk_rows(career, panel, "offense", labels=labels, unseen=unseen, age_curve=curve,
                           label_scale=lambda n: np.full(len(n), 0.5))
    assert np.allclose(halved.age_shift.to_numpy(), 0.5 * moved.age_shift.to_numpy())


def test_adjacent_rows_pair_each_stretch_with_the_seasons_next_to_it():
    """Experiment 30 (the owner, 2026-09-30): windows of 2k consecutive seasons split in the middle, each half
    predicting the other; the career row stays; one-team players keep full weight."""
    panel = _career_panel()
    unseen = (2020,)
    career = _labels_for(panel, [()], unseen)[(2020,)]
    halves = sy.adjacent_season_sets(panel)
    assert (2011, 2013) in halves or (2013,) in halves
    labels = {}
    for key in halves:
        kept = panel[panel.season.isin(key)]
        ids = np.sort(kept.player_id.unique())
        labels[key] = pd.DataFrame({"player_id": ids, "offense": ids + sum(key) / 1e4,
                                    "possessions": kept.groupby("player_id").poss.sum().reindex(ids).to_numpy()})
    teams = {(0, 2010): 1, (0, 2011): 1, (0, 2012): 2, (0, 2013): 1}
    rows = sy.adjacent_rows(career, panel, "offense", labels=labels, unseen=unseen,
                            shares=_team_shares(panel, teams))
    whole = rows.row_kind == "career"
    assert sorted(rows[whole].index) == sorted(career.player_id)
    assert (rows[whole][sy.SAME_TEAM] == 1).all()
    zero = rows.loc[[0]]
    # player 0 has four seasons: three 1+1 windows (six rows) and one 2+2 window (two rows)
    assert (zero.row_kind == "window").sum() == 8
    one = zero[(zero.chunk_seasons == 1) & (zero.label_half == "2012")]
    # the 2011 half labelled by 2012 and the 2013 half labelled by 2012: both "traded" (team 1 -> team 2)
    assert len(one) == 2 and (one[sy.SAME_TEAM] == 0).all()
    assert (one.target == labels[(2012,)].set_index("player_id").offense.loc[0]).all()
    stay = zero[(zero.feature_half == "2010") & (zero.label_half == "2011")]
    assert len(stay) == 1 and stay[sy.SAME_TEAM].iloc[0] == 1.0, "2010 on team 1 labelled by 2011 on team 1"
    back = zero[(zero.feature_half == "2012") & (zero.label_half == "2011")]
    assert back[sy.SAME_TEAM].iloc[0] == 0.0, "2012 on team 2 labelled by 2011 on team 1"
    # every player's window rows together weigh his career row's weight
    for pid in rows[~whole].index.unique():
        assert rows[~whole].loc[[pid]].row_weight.sum() == pytest.approx(rows[whole].loc[pid].row_weight)


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
    train = pd.DataFrame({"target": [1.0, 2.0, 3.0, 4.0], "row_weight": [100.0, 200.0, 300.0, 400.0]},
                         index=pd.Index([0, 0, 1, 2], name="player_id"))
    movement = pd.Series({0: 0.5, 1: 0.0, 2: 0.25})
    out = sy.reweight_by_movement(train, movement, floor=0.0)

    assert out.row_weight.sum() == pytest.approx(train.row_weight.sum())
    assert out.loc[1, "row_weight"] == pytest.approx(0.0), "floor 0 drops a one-team player outright"
    assert out.loc[2, "row_weight"] > 0
    # the two rows of player 0 keep their ratio to each other: only players are reweighted, not rows
    first, second = out.loc[0, "row_weight"].to_numpy()
    assert second / first == pytest.approx(2.0)


def test_reweight_by_movement_with_a_floor_keeps_every_player():
    """The one run of this rule lost by zeroing 29% of the players, stars included; the floor is the fix."""
    train = pd.DataFrame({"target": [1.0, 2.0], "row_weight": [100.0, 100.0]},
                         index=pd.Index([0, 1], name="player_id"))
    out = sy.reweight_by_movement(train, pd.Series({0: 0.0, 1: 0.5}), floor=0.25)
    assert (out.row_weight > 0).all()
    assert out.row_weight.sum() == pytest.approx(200.0)


def test_career_bands_keep_each_bands_weight_and_move_it_only_toward_movers():
    """The owner's fix (2026-09-30): the movement weight may not shift weight from short careers to long ones."""
    train = pd.DataFrame({"target": [0.0] * 6, "row_weight": [100.0, 100.0, 50.0, 50.0, 900.0, 900.0],
                          "possessions": [1000.0, 1000.0, 1500.0, 1500.0, 30000.0, 30000.0]},
                         index=pd.Index([0, 1, 2, 2, 3, 4], name="player_id"))
    movement = pd.Series({0: 0.0, 1: 0.6, 2: 0.2, 3: 0.0, 4: 0.5})
    out = sy.reweight_by_movement(train, movement, floor=0.5, bands=True)
    short, long = train.possessions < 2000, train.possessions >= 15000
    assert out.row_weight[short].sum() == pytest.approx(train.row_weight[short].sum())
    assert out.row_weight[long].sum() == pytest.approx(train.row_weight[long].sum())
    # inside a band the weights follow movement + floor
    assert out.loc[4, "row_weight"] / out.loc[3, "row_weight"] == pytest.approx((0.5 + 0.5) / (0.0 + 0.5))
    assert out.loc[1, "row_weight"] / out.loc[0, "row_weight"] == pytest.approx((0.6 + 0.5) / 0.5)
    assert (out.row_weight > 0).all(), "the floor keeps every one-team player"


def test_reweight_by_movement_refuses_to_leave_no_weight_at_all():
    """Silently returning an all-zero training set would fit a constant and look like a bad idea."""
    train = pd.DataFrame({"target": [1.0], "row_weight": [100.0]},
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


def test_a_deal_target_column_keeps_another_targets_deal():
    import pandas as pd
    from eracoef import singleyear as sy
    rng = np.random.default_rng(5)
    ids = np.repeat(np.arange(300), 2)
    a = np.repeat(rng.normal(0, 1, 300), 2)
    b = a + rng.normal(0, 0.05, a.size)                    # a slightly different target: the snake re-deals
    train_a = pd.DataFrame({"target": a, "row_weight": 1.0}, index=pd.Index(ids, name="player_id"))
    train_b = train_a.assign(target=b)
    fa, fb = sy.stratified_player_folds(train_a), sy.stratified_player_folds(train_b)
    assert (fa != fb).mean() > 0.2                         # a small label change moves many players
    pinned = sy.stratified_player_folds(train_b.assign(deal_target=a))
    assert np.array_equal(pinned, fa)                      # dealt on target a's labels: target a's folds exactly


def test_training_rows_keep_body_weight_and_carry_the_sample_weight_apart():
    """The Robustness pass (2026-10-08): the sample weight was written over the body-weight input, so every prior
    trained on label possessions under the name `weight` and was then asked about pounds on the rated row."""
    panel = _career_panel()
    panel["weight"] = 150.0 + 10.0 * panel.player_id.to_numpy(float)       # pounds, fixed per player
    unseen = (2020,)
    career = _labels_for(panel, [()], unseen)[(2020,)]
    rows = sy.chunk_rows(career, panel, "offense", unseen=unseen)
    assert np.allclose(rows["weight"].to_numpy(), 150.0 + 10.0 * rows.index.to_numpy(float)), \
        "the input `weight` is body weight on every training row"
    assert sy.ROW_WEIGHT in rows.columns and (rows[sy.ROW_WEIGHT] > 0).all()
    # the sample weight is where it always was: a player's chunk rows together weigh what his career row weighs
    per_player = rows.groupby(level=0)
    assert np.allclose(per_player[sy.ROW_WEIGHT].sum().to_numpy(), 2.0 * per_player.possessions.first().to_numpy())


def test_a_bookkeeping_column_named_like_an_input_is_refused():
    """No column the training rows add for their own bookkeeping may share a name with an input."""
    frame = pd.DataFrame({"target": [1.0], sy.ROW_WEIGHT: [2.0]})
    assert sy._guard(frame, ["pts", "weight"]) is frame
    with pytest.raises(ValueError, match="bookkeeping"):
        sy._guard(frame, ["pts", "target"])


def _with_counts(panel, seed=11):
    """`panel` with consistent padding columns: each row's 13 rates rebuilt as the panel stores them,
    (100 * count + k * target) / (poss + k), centred on a level of its own season (139's columns, invented)."""
    rng = np.random.default_rng(seed)
    out = panel.copy()
    n = out.poss.to_numpy(float)[:, None]
    shape = (len(out), len(sy.PAD_COUNTS))
    counts = rng.uniform(0.0, 0.15, shape) * n
    k = np.repeat(rng.uniform(40.0, 450.0, (1, shape[1])), len(out), axis=0) * rng.uniform(0.9, 1.1, (len(out), 1))
    target = rng.uniform(1.0, 12.0, shape)
    level = rng.uniform(2.0, 8.0, shape)
    raw = (100.0 * counts + k * target) / (n + k)
    from eracoef.design import FEATURES
    out[[f"raw_{c}" for c in FEATURES]] = raw
    out[list(FEATURES)] = raw - level
    out[sy.PAD_COUNTS], out[sy.PAD_K], out[sy.PAD_TARGET] = counts, k, target
    return out


def test_pad_once_gives_back_a_one_season_row():
    """The rated row and every one-season chunk must not move: a group of one season is its panel row."""
    from eracoef.design import FEATURES
    panel = _with_counts(_panel())
    one = panel[panel.season == 2014]
    got = sy.aggregate(one, pad_once=True)
    plain = sy.aggregate(one)
    cols = [f"raw_{c}" for c in FEATURES] + list(FEATURES)
    assert np.allclose(got[cols].to_numpy(), one.set_index("player_id")[cols].reindex(got.index).to_numpy(),
                       rtol=0, atol=1e-12)
    assert np.allclose(got[sy.PRIOR_FEATURES].to_numpy(), plain[sy.PRIOR_FEATURES].to_numpy(), rtol=0, atol=1e-12)


def test_pad_once_pads_a_multi_season_row_on_its_summed_counts():
    """Two seasons: (100 * summed counts + k * target) / (summed possessions + k), k and target blended by
    possessions -- and wider than the mean of the two padded rates, which shrinks the pair like one season."""
    from eracoef.design import FEATURES
    panel = _with_counts(_panel())
    got = sy.aggregate(panel, pad_once=True)
    plain = sy.aggregate(panel)
    for pid in (0, 7, 23):
        r = panel[panel.player_id == pid]
        n = r.poss.to_numpy(float)
        for j, c in enumerate(FEATURES[:4]):
            k = np.average(r[sy.PAD_K[j]], weights=n)
            t = np.average(r[sy.PAD_TARGET[j]], weights=n)
            want = (100.0 * r[sy.PAD_COUNTS[j]].sum() + k * t) / (n.sum() + k)
            assert np.isclose(got.loc[pid, f"raw_{c}"], want, rtol=0, atol=1e-10)
            level = np.average(r[f"raw_{c}"] - r[c], weights=n)
            assert np.isclose(got.loc[pid, c], want - level, rtol=0, atol=1e-10)
    assert not np.allclose(got["raw_stl"].to_numpy(), plain["raw_stl"].to_numpy())
    # the summed evidence is padded less: two identical seasons are one season's evidence twice, so the pair sits
    # farther from the target than either season -- where the mean of the two padded rates leaves it exactly there
    one = panel[panel.season == 2014]
    twice = pd.concat([one, one.assign(season=2015)], ignore_index=True)
    got2, plain2 = sy.aggregate(twice, pad_once=True), sy.aggregate(twice)
    single = one.set_index("player_id").reindex(got2.index)
    for j, c in enumerate(FEATURES):
        n, k, t = single.poss.to_numpy(float), single[sy.PAD_K[j]].to_numpy(float), single[sy.PAD_TARGET[j]].to_numpy(float)
        assert np.allclose(plain2[f"raw_{c}"].to_numpy(), single[f"raw_{c}"].to_numpy(), rtol=0, atol=1e-12)
        dev_once = got2[f"raw_{c}"].to_numpy() - t
        dev_one = single[f"raw_{c}"].to_numpy() - t
        assert np.allclose(dev_once, dev_one * (2 * n / (2 * n + k)) * ((n + k) / n), rtol=1e-10, atol=1e-12)


def test_pad_once_needs_the_count_columns():
    panel = _panel()
    with pytest.raises(KeyError, match="139_count_panel"):
        sy.aggregate(panel, pad_once=True)


def test_pad_once_moves_the_career_row_and_no_one_season_chunk():
    """chunk_rows with pad_once: the one-season chunks are the plain ones; the career row is padded once."""
    panel = _with_counts(_career_panel())
    unseen = (2020,)
    career = _labels_for(panel, [()], unseen)[(2020,)]
    plain = sy.chunk_rows(career, panel, "offense", unseen=unseen)
    once = sy.chunk_rows(career, panel, "offense", unseen=unseen, pad_once=True)
    assert plain.index.equals(once.index) and np.allclose(plain.row_weight, once.row_weight)
    single = (plain.chunk_seasons == 1).to_numpy()
    assert np.allclose(plain[sy.PRIOR_FEATURES].to_numpy()[single], once[sy.PRIOR_FEATURES].to_numpy()[single],
                       rtol=0, atol=1e-12)
    many = (plain.chunk_seasons > 1).to_numpy()
    assert not np.allclose(plain["stl"].to_numpy()[many], once["stl"].to_numpy()[many])


def test_the_own_side_lists_drop_exactly_the_other_sides_box_score():
    """Experiment 49: the defensive prior without the scoring inputs, the offensive without steals -- and each
    dropped name was on the list it is dropped from, so the set is not quietly the shipped one."""
    base, own = sy.feature_set("boruta_noonc"), sy.feature_set("boruta_noonc_ownside")
    for side in ("O", "D"):
        assert set(sy.CROSS_SIDE[side]) <= set(base[side])
        assert own[side] == [f for f in base[side] if f not in sy.CROSS_SIDE[side]]
    assert "stocks" in own["D"] and "stl" in own["D"] and "pts" in own["O"]
