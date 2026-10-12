"""Step 2 (src/openrapm/box_prior.py): the training rows, the folds, the LightGBM fits and the out-of-player prior.

Two things here have bitten already.  A possession-weighted mean of a ratio is not the ratio of the means, so the
order of averaging and deriving is a real choice.  And the sample weight was once written over the body-weight
input, so a bookkeeping column may never share a name with an input.
"""
import numpy as np
import pandas as pd
import pytest

from openrapm import box_prior
from openrapm.features import PANEL_INPUTS, PRIOR_FEATURES, RATIOS, ROW_FEATURES, SHIPPED
from openrapm.settings import PRODUCT

PARAMS = {"n_estimators": 60, "num_leaves": 7, "learning_rate": 0.1, "min_child_samples": 5}
FEATS = ["a", "b"]


def _panel_row(rng, pid, season, side="O"):
    r = {c: float(rng.normal(0, 2)) for c in PANEL_INPUTS}
    for c in PANEL_INPUTS:
        if c.endswith("_uncentred"):
            r[c] = float(rng.uniform(1, 12))
    r.update(shot_twos_attempted=float(rng.uniform(50, 900)), shot_threes_attempted=float(rng.uniform(10, 600)),
             league_two_point_pct=0.485, league_three_point_pct=0.35, league_points_per_shot=0.99)
    r.update(shot_twos_made=r["shot_twos_attempted"] * 0.5, shot_twos_expected_made=r["shot_twos_attempted"] * 0.49,
             shot_threes_made=r["shot_threes_attempted"] * 0.36,
             shot_threes_expected_made=r["shot_threes_attempted"] * 0.35)
    r.update(player_id=pid, season=season, side=side, possessions=float(rng.uniform(300, 4000)))
    return r


def _panel(n=40, seasons=(2014, 2015), seed=0):
    """Every `PANEL_INPUTS` column, plausible magnitudes, the same seasons for every player."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame([_panel_row(rng, pid, s) for pid in range(n) for s in seasons])


def _career_panel(seed=3):
    """Players with four, three, two and one season, one with a gap."""
    rng = np.random.default_rng(seed)
    careers = {0: [2010, 2011, 2012, 2013], 1: [2010, 2011, 2013], 2: [2012, 2013], 3: [2011], 4: [2010, 2012]}
    return pd.DataFrame([_panel_row(rng, pid, s) for pid, seasons in careers.items() for s in seasons])


def _label(panel, offset=0.0):
    ids = np.sort(panel.player_id.unique())
    return pd.DataFrame({"player_id": ids, "offense": ids + offset,
                         "possessions": panel.groupby("player_id").possessions.sum().reindex(ids).to_numpy()})


# ------------------------------------------------------------------ averaging, then deriving
def test_averaging_builds_every_requested_feature():
    out = box_prior.average_over_seasons(_panel(), PRIOR_FEATURES)
    assert list(out.index.names) == ["player_id"]
    assert out[PRIOR_FEATURES].notna().all().all()


def test_a_missing_input_raises_instead_of_shortening_the_list():
    panel = _panel().drop(columns=[c for c in PANEL_INPUTS if c.endswith("_uncentred")])
    with pytest.raises(KeyError, match="cannot build"):
        box_prior.average_over_seasons(panel, PRIOR_FEATURES)


def test_the_ratios_are_built_on_the_average_not_averaged():
    """True shooting of the average season, not the average of each season's true shooting."""
    panel = _panel()
    got = box_prior.average_over_seasons(panel, ["true_shooting_pct"])["true_shooting_pct"]
    num, den, pad, target = RATIOS["true_shooting_pct"]
    mean_raw = {c: (panel[f"{c}_uncentred"] * panel.possessions).groupby(panel.player_id).sum()
                / panel.possessions.groupby(panel.player_id).sum() for c in set(num) | set(den)}
    expected = ((sum(k * mean_raw[c] for c, k in num.items()) + pad * target)
                / (sum(k * mean_raw[c] for c, k in den.items()) + pad))
    assert np.allclose(got.to_numpy(), expected.to_numpy())
    per_season = box_prior.rated_rows(panel, ["true_shooting_pct"])
    averaged = (per_season.true_shooting_pct * per_season.possessions).groupby(per_season.player_id).sum() / \
        per_season.possessions.groupby(per_season.player_id).sum()
    assert not np.allclose(got.to_numpy(), averaged.to_numpy())


def test_career_rows_join_the_label_and_weigh_its_possessions():
    panel = _panel()
    label = pd.DataFrame({"player_id": np.arange(0, 40, 2), "offense": np.linspace(-3, 3, 20),
                          "possessions": np.linspace(500, 9000, 20)})
    rows = box_prior.career_rows(label, panel, "offense", SHIPPED["O"])
    assert len(rows) == 20
    assert np.array_equal(rows.label.to_numpy(), rows.offense.to_numpy())
    assert np.array_equal(rows.row_weight.to_numpy(), rows.label_possessions.to_numpy())


# ------------------------------------------------------------------ the chunk rows
def test_every_row_carries_the_career_label_and_each_player_weighs_twice_his_label():
    panel = _career_panel()
    rows = box_prior.training_rows(_label(panel), panel, "offense", SHIPPED["O"])
    assert np.array_equal(rows.label.to_numpy(), rows.index.to_numpy(float))
    per_player = rows.groupby(level=0)
    assert np.allclose(per_player.row_weight.sum().to_numpy(),
                       2.0 * per_player.label_possessions.first().to_numpy())
    # player 0: one career row, four single seasons, three pairs, two triples
    assert len(rows.loc[[0]]) == 1 + 4 + 3 + 2
    assert list(rows.columns[-4:]) == ["label", "row_weight", "row_possessions", "row_seasons"]


def test_a_run_is_consecutive_in_his_seasons_not_on_the_calendar():
    panel = _career_panel()
    runs = box_prior.season_runs(panel[panel.player_id == 1])
    pairs = runs[runs.chunk.str.contains(":2:")].groupby("chunk").season.apply(tuple)
    assert sorted(pairs) == [(2010, 2011), (2011, 2013)]


def test_a_chunk_row_weight_is_its_share_of_the_chunk_possessions():
    panel = _career_panel()
    rows = box_prior.training_rows(_label(panel), panel, "offense", SHIPPED["O"])
    chunks = rows.loc[[2]].iloc[1:]                      # player 2: 2012, 2013 and the pair
    share = chunks.row_possessions / chunks.row_possessions.sum()
    assert np.allclose(chunks.row_weight.to_numpy(), (share * chunks.label_possessions).to_numpy())
    assert sorted(chunks.row_seasons) == [1.0, 1.0, 2.0]


def test_body_weight_stays_an_input_beside_the_sample_weight():
    panel = _career_panel()
    panel["weight_lb"] = 150.0 + 10.0 * panel.player_id.to_numpy(float)
    rows = box_prior.training_rows(_label(panel), panel, "offense", SHIPPED["O"])
    assert np.allclose(rows["weight_lb"].to_numpy(), 150.0 + 10.0 * rows.index.to_numpy(float))
    assert (rows.row_weight > 0).all()


def test_a_bookkeeping_name_among_the_inputs_is_refused():
    panel = _career_panel()
    with pytest.raises(ValueError, match="bookkeeping"):
        box_prior.training_rows(_label(panel), panel, "offense", ["points_per100", "row_weight"])


def test_the_rated_row_reads_his_possessions_and_one_season():
    panel = _panel()
    rated = box_prior.rated_rows(panel[panel.season == 2015], SHIPPED["D"])
    assert np.array_equal(rated.row_possessions.to_numpy(), rated.possessions.to_numpy())
    assert (rated.row_seasons == 1.0).all()
    assert not set(SHIPPED["D"]) - set(rated.columns)


# ------------------------------------------------------------------ the folds
def _fold_frame(n_players=400, rows_each=3, seed=3):
    rng = np.random.default_rng(seed)
    ids = np.repeat(np.arange(n_players), rows_each)
    return pd.DataFrame({"label": np.repeat(rng.normal(0, 1, n_players), rows_each),
                         "row_weight": rng.uniform(1, 5, ids.size)}, index=pd.Index(ids, name="player_id"))


def test_the_folds_go_there_and_back_in_label_order():
    train = pd.DataFrame({"label": [5.0, 1.0, 3.0, 2.0, 4.0, 0.0], "row_weight": 1.0},
                         index=pd.Index([10, 11, 12, 13, 14, 15], name="player_id"))
    fold = box_prior.balanced_player_folds(train, n_folds=3)
    # sorted by label: 15, 11, 13, 12, 14, 10 -> folds 0, 1, 2, 2, 1, 0
    assert dict(zip(train.index, fold)) == {15: 0, 11: 1, 13: 2, 12: 2, 14: 1, 10: 0}


def test_a_seeded_assignment_is_as_balanced_and_a_different_make_up():
    train = _fold_frame()
    groups, y, w = train.index.to_numpy(), train.label.to_numpy(), train.row_weight.to_numpy()
    there_and_back = box_prior.balanced_player_folds(train, 5)
    seeded = box_prior.balanced_player_folds(train, 5, fold_assignment=1)
    assert np.array_equal(seeded, box_prior.balanced_player_folds(train, 5, fold_assignment=1))
    assert (there_and_back != seeded).mean() > 0.5
    spread = {}
    for name, fold in (("there_and_back", there_and_back), ("seeded", seeded)):
        assert all(len(set(fold[groups == g])) == 1 for g in range(0, 400, 37))      # every player whole
        counts = np.bincount(fold[::3], minlength=5)
        assert counts.min() >= 79 and counts.max() <= 81
        means = [np.average(y[fold == f], weights=w[fold == f]) for f in range(5)]
        spread[name] = max(means) - min(means)
    assert spread["seeded"] < 1.5 * spread["there_and_back"]


# ------------------------------------------------------------------ the boosters
def _rows(n_players=120, rows_each=2, seed=0):
    rng = np.random.default_rng(seed)
    pid = np.repeat(np.arange(n_players), rows_each)
    a, b = rng.normal(size=pid.size), rng.normal(size=pid.size)
    label = 0.8 * a + np.sin(2 * b) + rng.normal(scale=0.3, size=pid.size)
    return pd.DataFrame({"a": a, "b": b, "label": label, "row_weight": rng.uniform(100, 3000, pid.size)},
                        index=pd.Index(pid, name="player_id"))


def test_the_fit_is_deterministic_and_blind_to_the_weight_scale():
    train = _rows()
    x, y, w = train[FEATS].to_numpy(float), train.label.to_numpy(float), train.row_weight.to_numpy(float)
    one = box_prior.fit_lightgbm(PARAMS, x, y, w).predict(x)
    assert np.array_equal(one, box_prior.fit_lightgbm(PARAMS, x, y, w).predict(x))
    assert np.array_equal(one, box_prior.fit_lightgbm(PARAMS, x, y, 1000.0 * w).predict(x))
    noisy = {**PARAMS, "subsample": 0.5, "subsample_freq": 1, "colsample_bytree": 0.5, "random_state": 7}
    assert np.array_equal(one, box_prior.fit_lightgbm(noisy, x, y, w).predict(x))


def test_each_player_is_predicted_by_the_fold_model_that_never_saw_him():
    train = _rows()
    prior = box_prior.fit_out_of_player(train, FEATS, PARAMS, n_folds=5)
    assert len(prior.fold_models) == 5
    assert set().union(*prior.held_out_players) == set(train.index)
    frame = train.reset_index().drop_duplicates("player_id").iloc[:10].copy()
    frame.loc[frame.index[-1], "player_id"] = -1                     # a player with no training rows
    got = box_prior.predict(prior, frame)
    x = frame[FEATS].to_numpy(float)
    for i, p in enumerate(frame.player_id):
        held = [k for k, ex in enumerate(prior.held_out_players) if p in ex]
        model = prior.full if p == -1 else prior.fold_models[held[0]]
        assert got[i] == model.predict(x[i:i + 1])[0]


def test_one_fold_means_the_full_fit_alone():
    prior = box_prior.fit_out_of_player(_rows(), FEATS, PARAMS, n_folds=1)
    assert prior.fold_models == [] and prior.held_out_players == []


def test_the_season_priors_rate_every_rated_player_and_train_on_no_unseen_season():
    rng = np.random.default_rng(5)
    panel = pd.DataFrame([_panel_row(rng, pid, s, side) for side in ("O", "D") for pid in range(60)
                          for s in (2010, 2011, 2012, 2013)])
    unseen = [2013]
    labels = {"O": _label(panel[panel.side == "O"]),
              "D": _label(panel[panel.side == "D"]).rename(columns={"offense": "defense"})}
    params = {"O": PARAMS, "D": PARAMS}
    priors = box_prior.season_priors(labels, panel, 2013, unseen, PRODUCT, params)
    for side in ("O", "D"):
        assert list(priors[side].values) == list(range(60))
        assert np.isfinite(list(priors[side].values.values())).all()
        assert priors[side].model.inputs == SHIPPED[side] + ROW_FEATURES
        assert len(priors[side].model.fold_models) == PRODUCT.prior.n_folds
    # the full booster is the one fitted on 2010-2012 alone (LightGBM is deterministic here), not on 2010-2013
    column, inputs = "offense", SHIPPED["O"] + ROW_FEATURES
    rated = box_prior.rated_rows(panel[(panel.side == "O") & (panel.season == 2013)], SHIPPED["O"])

    def full_fit(seasons):
        rows = panel[(panel.side == "O") & panel.season.isin(seasons)]
        train = box_prior.training_rows(labels["O"], rows, column, SHIPPED["O"])
        model = box_prior.fit_lightgbm(PARAMS, train[inputs].to_numpy(float), train.label.to_numpy(float),
                                       train.row_weight.to_numpy(float))
        return model.predict(rated[inputs].to_numpy(float))

    shipped = priors["O"].model.full.predict(rated[inputs].to_numpy(float))
    assert np.array_equal(shipped, full_fit([2010, 2011, 2012]))
    assert not np.array_equal(shipped, full_fit([2010, 2011, 2012, 2013]))
