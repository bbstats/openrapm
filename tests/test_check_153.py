"""scripts/checks/153_swap_audit.py (what the swap step does): its step 5 from given inputs is swap_adjust's step 5,
its arms do what they say, its statistics are the ones named, and the script starts."""
import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from openrapm import swap_adjust as sa
from openrapm import swap_test as sw

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "checks" / "153_swap_audit.py"


@pytest.fixture(scope="module")
def audit():
    spec = importlib.util.spec_from_file_location("swap_audit_check", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _league(lineup_league, seasons=(2020, 2021, 2022, 2023)):
    """{season: SeasonSwaps}, the type features and the player folds of four synthetic seasons rated at their truth,
    and a step-4-like table for them."""
    swaps, features, tables = {}, [], []
    for season in seasons:
        stints, truth, _ = lineup_league(season, games=160)
        lineups = sw.prepare(sw.season_rows(stints), season)
        rng = np.random.default_rng(season)
        features.append(pd.DataFrame({"player_id": truth.player_id, "season": season,
                                      **{name: rng.normal(0, 1, len(truth)) for name in ("x", "y")}}))
        swaps[season] = (lineups, truth)
        poss = sa.team_possessions(lineups)["offense"].groupby("player_id").poss.sum()
        tables.append(pd.DataFrame({"player_id": poss.index.astype(np.int64), "season": season,
                                    "rating_off": rng.normal(0, 2, len(poss)),
                                    "rating_def": rng.normal(0, 1, len(poss)),
                                    "poss_off": poss.to_numpy(), "poss_def": poss.to_numpy()}))
    features = pd.concat(features).set_index(["player_id", "season"])
    fold_of = sa.player_folds(features.index.get_level_values("player_id"), 5)
    kept = {season: sa.season_swaps(lineups, truth, sa.season_features(features, season), fold_of, 5)
            for season, (lineups, truth) in swaps.items()}
    table = pd.concat(tables, ignore_index=True)
    table["rating_total"] = table.rating_off + table.rating_def
    table["offense"], table["defense"] = table.rating_off, -table.rating_def
    main_team = (table.player_id // 100).astype(float)
    main_team.index = pd.MultiIndex.from_arrays([table.player_id, table.season])
    return kept, features, fold_of, table, main_team


def test_help_lists_every_stage_and_its_flags():
    result = subprocess.run([sys.executable, str(SCRIPT), "--help"], capture_output=True, text=True,
                            encoding="utf-8", cwd=ROOT, timeout=300)
    assert result.returncode == 0, result.stderr
    for command in ("reproduce", "whole", "folds", "decider", "types", "luck", "whose"):
        assert command in result.stdout
    folds = subprocess.run([sys.executable, str(SCRIPT), "folds", "--help"], capture_output=True, text=True,
                           encoding="utf-8", cwd=ROOT, timeout=300)
    assert all(flag in folds.stdout for flag in ("--system", "--first", "--last"))


def test_step_5_from_given_inputs_is_swap_adjusts_step_5(audit, lineup_league):
    """The fold code's step 5, handed a season's possessions and type predictions but none of its swaps, gives the
    shipped step 5 on that season to the last bit: the swaps reach the give-back only through their players' ids."""
    swaps, features, fold_of, table, main_team = _league(lineup_league)
    settings = audit.SWAP            # step 5 at the half strength the check audits; off as shipped since experiment 82
    predictions = sa.type_predictions(features, swaps, fold_of, set(swaps), settings)
    shipped, factors = sa.adjust(table, sw.ratings_in_raw_sign(table), swaps, predictions, main_team, settings)
    grams = {season: kept.grams for season, kept in swaps.items()}
    betas = audit.season_betas(grams, set(swaps))
    for season in swaps:
        rows = table[table.season == season].reset_index(drop=True)
        features_now = sa.season_features(features, season)
        prediction = {side: audit.type_prediction(features_now, betas[(season, side)]) for side in sa.SIDES}
        for side in sa.SIDES:
            assert np.array_equal(prediction[side].to_numpy(), predictions[(season, side)].to_numpy())
        mine, mine_factors = audit.step5_rows(rows, prediction, swaps[season].team_possessions, main_team)
        theirs = shipped[shipped.season == season].reset_index(drop=True)
        for column in ("rating_off", "rating_def", "c_off", "c_def"):
            assert np.array_equal(mine[column].to_numpy(), theirs[column].to_numpy()), (season, column)
        shipped_factors = factors[factors.season == season]
        assert mine_factors == dict(zip(shipped_factors.side, shipped_factors.factor))


def test_without_the_hold_the_team_totals_and_the_season_mean_still_stay(audit, lineup_league):
    swaps, features, fold_of, table, main_team = _league(lineup_league)
    betas = audit.season_betas({season: kept.grams for season, kept in swaps.items()}, set(swaps))
    season = 2022
    rows = table[table.season == season].reset_index(drop=True)
    prediction = {side: 4.0 * audit.type_prediction(sa.season_features(features, season), betas[(season, side)])
                  for side in sa.SIDES}
    out, factors = audit.step5_rows(rows, prediction, swaps[season].team_possessions, main_team, hold=False)
    assert factors == {"offense": 1.0, "defense": 1.0}
    assert out.c_off.abs().sum() > 0
    team = out.player_id // 100
    assert np.allclose((out.poss_off * out.c_off).groupby(team).sum(), 0.0, atol=1e-9)
    assert abs(np.average(out.c_def, weights=out.poss_def)) < 1e-12


def test_shrink_within_keeps_team_means_and_multiplies_every_distance(audit):
    values = np.array([1.0, 3.0, -2.0, 4.0, 7.0])
    weight = np.array([1.0, 3.0, 2.0, 2.0, 5.0])
    team = np.array([1.0, 1.0, 2.0, 2.0, np.nan])
    out = audit.shrink_within(values, weight, team, 0.5)
    assert np.allclose(out[:2], 2.5 + 0.5 * (values[:2] - 2.5))          # team 1's mean, (1 + 9) / 4 = 2.5
    assert np.allclose(out[2:4], 1.0 + 0.5 * (values[2:4] - 1.0))        # team 2's mean, (-4 + 8) / 4 = 1
    assert out[4] == 7.0                                                  # no main team: kept
    for mask in (team == 1.0, team == 2.0):
        assert np.isclose(np.average(out[mask], weights=weight[mask]), np.average(values[mask], weights=weight[mask]))


def test_shrink_only_leaves_the_order_inside_each_team(audit, lineup_league):
    _, _, _, table, main_team = _league(lineup_league)
    out = audit.shrink_only(table, {"offense": 0.8, "defense": 0.7}, main_team)
    for (season, team), rows in out.groupby([out.season, out.player_id // 100]):
        before = table.loc[rows.index]
        assert (np.argsort(rows.rating_off.to_numpy()) == np.argsort(before.rating_off.to_numpy())).all()
    assert np.allclose(out.c_off + table.rating_off, out.rating_off)


def test_a_shuffle_deals_every_row_to_another_player_and_repeats_with_its_seed(audit):
    features = pd.DataFrame({"x": np.arange(10.0), "y": -np.arange(10.0)}, index=np.arange(100, 110))
    first = audit.shuffled(features, 1, (2024, 0, 1))
    assert (first.index == features.index).all()
    assert sorted(first.x) == sorted(features.x) and (first.x == -first.y).all()       # whole rows move together
    assert first.equals(audit.shuffled(features, 1, (2024, 0, 1)))
    assert not first.equals(audit.shuffled(features, 2, (2024, 0, 1)))


def test_the_normal_equations_are_swap_adjusts(audit, lineup_league):
    swaps, features, fold_of, _, _ = _league(lineup_league)
    kept = swaps[2021]
    for side in sa.SIDES:
        mine = audit.equations(kept.pairs[side], sa.season_features(features, 2021))
        assert np.allclose(mine.G, kept.grams[side].G) and np.allclose(mine.b, kept.grams[side].b)
        assert mine.n == kept.grams[side].n


def test_the_jackknife_standard_error_is_the_leave_one_season_out_formula(audit):
    rng = np.random.default_rng(0)
    parts = {}
    for season in range(2000, 2006):
        dx = rng.normal(size=(200, 2))
        r = dx @ np.array([1.0, -0.5]) + rng.normal(size=200)
        parts[season] = audit.Equations(dx.T @ dx, dx.T @ r, 200)
    beta, se = audit.jackknife(parts)
    drops = np.array([np.linalg.solve(sum(p.G for s, p in parts.items() if s != out)
                                      + 1e-4 * np.mean(np.diag(sum(p.G for s, p in parts.items() if s != out)))
                                      * np.eye(2), sum(p.b for s, p in parts.items() if s != out))
                      for out in parts])
    assert np.allclose(se, np.sqrt(5 / 6 * ((drops - drops.mean(axis=0)) ** 2).sum(axis=0)))
    assert np.allclose(beta, [1.0, -0.5], atol=0.1)
