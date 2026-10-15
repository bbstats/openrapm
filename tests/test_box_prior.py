"""Step 2 (src/openrapm/box_prior.py): the training rows, the folds, the LightGBM fits and the out-of-player prior.

Two things here have bitten already.  A possession-weighted mean of a ratio is not the ratio of the means, so the
order of averaging and deriving is a real choice.  And the sample weight was once written over the body-weight
input, so a bookkeeping column may never share a name with an input.
"""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from openrapm import box_prior
from openrapm.features import FEATURE_SETS, PANEL_INPUTS, PRIOR_FEATURES, RATIOS, ROW_FEATURES, SHIPPED
from openrapm.settings import PRODUCT

# the input lists the product setting's prior reads (experiment 57's since 2026-10-13: features.SHIPPED with the career
# counted from the debut in place)
PRODUCT_LISTS = FEATURE_SETS[PRODUCT.prior.feature_set]

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


def test_a_column_a_season_may_lack_is_averaged_over_the_seasons_that_have_it():
    """features.MISSING_OK: a missing season is not an average one, and a row with none of the seasons is missing."""
    panel = _career_panel()
    column = "deflections_vs_league_avg"
    panel.loc[panel.season == 2010, column] = np.nan
    out = box_prior.average_over_seasons(panel, ["assists_per100", column])
    has = panel[panel.season != 2010]
    expected = ((has[column] * has.possessions).groupby(has.player_id).sum()
                / has.possessions.groupby(has.player_id).sum())
    assert out.loc[[0, 1, 2, 4], column].to_numpy() == pytest.approx(expected.loc[[0, 1, 2, 4]].to_numpy())
    assert out.loc[3, column] == pytest.approx(expected.loc[3]) or np.isnan(out.loc[3, column])
    assert out["assists_per100"].notna().all()                      # an ordinary column is untouched


def test_a_player_all_of_whose_seasons_lack_a_dashboard_column_has_it_missing_not_zero():
    panel = _career_panel()
    panel.loc[panel.player_id == 3, "deflections_vs_league_avg"] = np.nan             # player 3 has one season
    out = box_prior.average_over_seasons(panel, ["deflections_vs_league_avg"])
    assert np.isnan(out.loc[3, "deflections_vs_league_avg"])
    assert out.loc[[0, 1, 2, 4], "deflections_vs_league_avg"].notna().all()


def test_training_rows_keep_a_row_that_lacks_a_dashboard_input():
    """Players 3 (one season, 2011) and 4 (2010, 2012) have no season with the column before 2012: their career and
    chunk rows stay, with the column missing, so a tree can use the rows for everything else."""
    panel = _career_panel()
    column = "deflections_vs_league_avg"
    panel.loc[panel.season <= 2011, column] = np.nan
    names = ["assists_per100", column]
    rows = box_prior.training_rows(_label(panel), panel, "offense", names)
    assert rows[column].isna().any() and rows[column].notna().any()
    assert {0, 1, 2, 3, 4} <= set(rows.index)                                             # every player's career row
    assert set(box_prior.career_rows(_label(panel), panel, "offense", names).index) == {0, 1, 2, 3, 4}
    assert np.isnan(box_prior.career_rows(_label(panel), panel, "offense", names).loc[3, column])
    assert rows["assists_per100"].notna().all()


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
    # row subsampling and the seed in a settings file are overridden; the column share no longer is (check 150, below)
    noisy = {**PARAMS, "subsample": 0.5, "subsample_freq": 1, "random_state": 7}
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
        assert priors[side].model.inputs == PRODUCT_LISTS[side] + ROW_FEATURES
        assert len(priors[side].model.fold_models) == PRODUCT.prior.n_folds
    # the full booster is the one fitted on 2010-2012 alone (LightGBM is deterministic here), not on 2010-2013
    column, inputs = "offense", PRODUCT_LISTS["O"] + ROW_FEATURES
    rated = box_prior.rated_rows(panel[(panel.side == "O") & (panel.season == 2013)], PRODUCT_LISTS["O"])

    def full_fit(seasons):
        rows = panel[(panel.side == "O") & panel.season.isin(seasons)]
        train = box_prior.training_rows(labels["O"], rows, column, PRODUCT_LISTS["O"])
        model = box_prior.fit_lightgbm(PARAMS, train[inputs].to_numpy(float), train.label.to_numpy(float),
                                       train.row_weight.to_numpy(float))
        return model.predict(rated[inputs].to_numpy(float))

    shipped = priors["O"].model.full.predict(rated[inputs].to_numpy(float))
    assert np.array_equal(shipped, full_fit([2010, 2011, 2012]))
    assert not np.array_equal(shipped, full_fit([2010, 2011, 2012, 2013]))


# ------------------------------------------------------------------ the era kernel (experiment 56)
KERNEL_COLUMNS = [*PANEL_INPUTS, "era_kernel"]


def test_the_era_kernel_is_one_up_to_the_nearest_usable_season_then_halves_every_half_life():
    got = box_prior.era_kernel([2013, 2014, 2015, 2017, 2005, 2025], 2015, 8, start=2)
    assert np.allclose(got, [1.0, 1.0, 1.0, 1.0, 0.5, 0.5])
    assert np.allclose(box_prior.era_kernel([2014, 2006], 2015, 4, start=1), [1.0, 0.25])
    with pytest.raises(ValueError, match="positive"):
        box_prior.era_kernel([2015], 2015, 0)


def test_a_rows_kernel_is_the_possession_weighted_kernel_of_its_seasons():
    panel = _career_panel()
    panel = panel.assign(era_kernel=box_prior.era_kernel(panel.season, 2013, 1.0, start=1))
    rows = box_prior.training_rows(_label(panel), panel, "offense", SHIPPED["O"], panel_columns=KERNEL_COLUMNS)
    mine = panel[panel.player_id == 0]
    career = rows.loc[[0]].iloc[0]
    assert career.era_kernel == pytest.approx(np.average(mine.era_kernel, weights=mine.possessions))
    pair = panel[(panel.player_id == 2)]                                   # 2012 and 2013, his one pair
    chunk = rows.loc[[2]].iloc[1:].query("row_seasons == 2.0").iloc[0]
    assert chunk.era_kernel == pytest.approx(np.average(pair.era_kernel, weights=pair.possessions))


def test_the_kernel_leaves_the_row_weights_and_the_folds_alone():
    panel = _career_panel()
    plain = box_prior.training_rows(_label(panel), panel, "offense", SHIPPED["O"])
    weighted = box_prior.training_rows(
        _label(panel), panel.assign(era_kernel=box_prior.era_kernel(panel.season, 2013, 1.0)), "offense",
        SHIPPED["O"], panel_columns=KERNEL_COLUMNS)
    assert np.array_equal(plain.row_weight.to_numpy(), weighted.row_weight.to_numpy())
    assert np.array_equal(box_prior.balanced_player_folds(plain), box_prior.balanced_player_folds(weighted))
    assert np.array_equal(plain[SHIPPED["O"]].to_numpy(), weighted[SHIPPED["O"]].to_numpy())


def test_no_kernel_and_a_kernel_of_ones_are_the_shipped_fit_and_a_smaller_one_is_less_data():
    """The kernel multiplies the weights after they are rescaled, so a kernel of one half everywhere is half the
    data under the same leaf penalty, not the same fit."""
    train = _rows()
    x, y, w = train[FEATS].to_numpy(float), train.label.to_numpy(float), train.row_weight.to_numpy(float)
    penalised = {**PARAMS, "reg_lambda": 50.0}
    shipped = box_prior.fit_lightgbm(penalised, x, y, w).predict(x)
    assert np.array_equal(shipped, box_prior.fit_lightgbm(penalised, x, y, w, np.ones(y.size)).predict(x))
    assert not np.array_equal(shipped, box_prior.fit_lightgbm(penalised, x, y, w, np.full(y.size, 0.5)).predict(x))


def test_a_short_half_life_follows_the_rated_eras_price():
    """Two eras with opposite prices: the pooled fit learns neither, a fit weighted toward one era learns its price."""
    rng = np.random.default_rng(11)
    season = np.repeat([2000, 2020], 400)
    a = rng.normal(size=season.size)
    label = np.where(season == 2000, 1.0, -1.0) * a + rng.normal(scale=0.2, size=season.size)
    x, w = np.column_stack([a, rng.normal(size=season.size)]), np.ones(season.size)
    grid = np.column_stack([np.linspace(-1.5, 1.5, 7), np.zeros(7)])

    def price(kernel):
        return np.polyfit(grid[:, 0], box_prior.fit_lightgbm(PARAMS, x, label, w, kernel).predict(grid), 1)[0]

    assert price(box_prior.era_kernel(season, 2000, 2.0)) > 0.5
    assert price(box_prior.era_kernel(season, 2020, 2.0)) < -0.5
    assert abs(price(None)) < 0.3


def test_the_season_priors_weigh_each_row_by_its_kernel_and_keep_the_folds():
    rng = np.random.default_rng(5)
    panel = pd.DataFrame([_panel_row(rng, pid, s, side) for side in ("O", "D") for pid in range(60)
                          for s in (2008, 2009, 2010, 2011, 2012, 2013)])
    labels = {"O": _label(panel[panel.side == "O"]),
              "D": _label(panel[panel.side == "D"]).rename(columns={"offense": "defense"})}
    params = {"O": PARAMS, "D": PARAMS}
    era = replace(PRODUCT, prior=replace(PRODUCT.prior, era_halflife=2.0))
    priors = box_prior.season_priors(labels, panel, 2013, [2013], era, params)
    shipped = box_prior.season_priors(labels, panel, 2013, [2013], PRODUCT, params)
    assert "era_kernel" not in panel.columns, "the kernel goes on a copy, never on the shared panel"
    inputs = PRODUCT_LISTS["O"] + ROW_FEATURES
    rows = panel[(panel.side == "O") & (panel.season < 2013)]
    rows = rows.assign(era_kernel=box_prior.era_kernel(rows.season, 2013, 2.0, start=1))
    train = box_prior.training_rows(labels["O"], rows, "offense", PRODUCT_LISTS["O"], panel_columns=KERNEL_COLUMNS)
    by_hand = box_prior.fit_lightgbm(PARAMS, train[inputs].to_numpy(float), train.label.to_numpy(float),
                                     train.row_weight.to_numpy(float), train.era_kernel.to_numpy(float))
    rated = box_prior.rated_rows(panel[(panel.side == "O") & (panel.season == 2013)], PRODUCT_LISTS["O"])
    x = rated[inputs].to_numpy(float)
    assert np.array_equal(priors["O"].model.full.predict(x), by_hand.predict(x))
    assert not np.array_equal(priors["O"].model.full.predict(x), shipped["O"].model.full.predict(x))
    assert priors["O"].model.held_out_players == shipped["O"].model.held_out_players
    assert priors["O"].model.inputs == inputs, "the kernel is never an input"


def test_the_era_kernel_is_refused_as_an_input():
    panel = _career_panel()
    with pytest.raises(ValueError, match="bookkeeping"):
        box_prior.training_rows(_label(panel), panel, "offense", ["points_per100", "era_kernel"])


def test_a_settings_file_may_name_the_share_of_inputs_each_tree_sees_and_the_shipped_one_does_not():
    """Check 150: `colsample_bytree` in a settings file reaches LightGBM (the same rows still give the same booster);
    without it every tree sees every input, as before."""
    import json
    from pathlib import Path

    from openrapm.box_prior import COLUMN_SHARES, LIGHTGBM_FIXED, fit_lightgbm
    root = Path(__file__).resolve().parents[1]
    shipped = json.loads((root / "params" / "booster_params_lgb1.json").read_text(encoding="utf-8"))
    assert not any(name in shipped[side]["params"] for side in ("O", "D") for name in COLUMN_SHARES)
    cols = json.loads((root / "params" / "booster_params_lgb1_cols06.json").read_text(encoding="utf-8"))
    for side in ("O", "D"):
        assert cols[side]["params"] == {**shipped[side]["params"], "colsample_bytree": 0.6}
    rng = np.random.default_rng(0)
    x = rng.normal(size=(600, 8))
    y = x[:, 0] + rng.normal(size=600)
    params = dict(n_estimators=20, num_leaves=7, min_child_samples=20)
    whole = fit_lightgbm(params, x, y, np.ones(600))
    part = fit_lightgbm({**params, "colsample_bytree": 0.5}, x, y, np.ones(600))
    again = fit_lightgbm({**params, "colsample_bytree": 0.5}, x, y, np.ones(600))
    assert whole.get_params()["colsample_bytree"] == LIGHTGBM_FIXED["colsample_bytree"] == 1.0
    assert part.get_params()["colsample_bytree"] == 0.5
    assert np.array_equal(part.predict(x), again.predict(x)) and not np.allclose(part.predict(x), whole.predict(x))
