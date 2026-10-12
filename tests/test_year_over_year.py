"""The year-over-year runner (src/openrapm/commands/year_over_year.py) on simulated seasons with a known truth."""
import numpy as np
import pandas as pd
import pytest

from openrapm.commands.year_over_year import (
    RESULT_COLUMNS,
    NeighbourTable,
    by_direction,
    fill_rule,
    neighbour_tables,
    report,
    run,
    scored_seasons,
)

CFG = {"first_season": 2001, "last_season": 2005}        # the seasons `scored_seasons` reads


class SimulatedDesigns:
    """What the runner asks of `data.designs.DesignCache`, from simulated stints (regular season only): `design` is
    the `synthetic_design` fixture."""

    def __init__(self, sim, design, cfg):
        self.sim, self.design, self.cfg = sim, design, cfg

    def season(self, season, target):
        assert target == "pts"
        return self.design(self.sim, [season], self.cfg)


@pytest.fixture(scope="module")
def results(synthetic_league, synthetic_design, small_cfg):
    sim = synthetic_league(n_seasons=5, seed=11)
    truth = sim["truth"]["ps"].rename(columns={"impact_O": "o", "impact_D": "d"})
    true = truth.assign(poss=np.where(truth.player_id % 1000 > 7, 100.0, 1000.0))[["player_id", "season", "o", "d",
                                                                                    "poss"]]
    rng = np.random.default_rng(3)
    noisy = true.assign(o=true.o + rng.normal(0, 1.0, len(true)), d=true.d + rng.normal(0, 1.0, len(true)))
    zero = true.assign(o=0.0, d=0.0)
    tables = {name: {"prev": t, "next": t} for name, t in (("true", true), ("noisy", noisy), ("zero", zero))}
    systems = neighbour_tables(tables)
    designs = SimulatedDesigns(sim, synthetic_design, small_cfg(**CFG))
    res = run(systems, designs, scored_seasons(CFG), splits=("game_bench_share",), verbose=False)
    return systems, res


def test_scored_seasons_have_a_season_on_both_sides():
    assert scored_seasons(CFG) == [2002, 2003, 2004]
    every = scored_seasons({"first_season": 1997, "last_season": 2026})
    assert (every[0], every[-1], len(every)) == (1998, 2025, 28)
    assert scored_seasons(CFG, first=2000, last=2003) == [2001, 2002, 2003], "clipped to the configured seasons"


def test_two_neighbour_tables_per_table_in_the_order_named(results):
    systems, _ = results
    assert [s.name for s in systems] == ["true:prev", "true:next", "noisy:prev", "noisy:next", "zero:prev",
                                         "zero:next"]
    assert systems[0].rated_season(2003) == 2002 and systems[1].rated_season(2003) == 2004
    assert systems[0].rated_season(2001) is None, "the table has no 2000"


def test_the_results_schema(results):
    _, res = results
    assert list(res.columns) == list(RESULT_COLUMNS)
    whole = res[res.split == "all"]
    assert set(whole.held_out) == {2002, 2003, 2004}
    assert whole.groupby(["held_out", "system"]).size().max() == 1
    assert (whole.k == 1).all() and (whole.lam == 0.0).all() and whole.cut.isna().all()
    assert set(whole.train.astype(int) - whole.held_out) == {-1, 1}


def test_the_split_groups_add_up_to_the_whole(results):
    _, res = results
    whole = res[res.split == "all"].set_index(["held_out", "system"]).n
    groups = res[res.split == "game_bench_share"].groupby(["held_out", "system"]).n.sum()
    assert np.allclose(groups.reindex(whole.index), whole)


def test_the_neighbouring_truth_beats_noise_and_beats_nothing(results):
    _, res = results
    frames = by_direction(res)
    for frame in (frames["prev"], frames["next"]):
        whole = frame[frame.split == "all"]
        mse = whole.groupby("system").mse.mean()
        assert mse["true"] < mse["noisy"] < mse["zero"]
    text = report(res[res.split == "all"], ref="zero")
    assert "paired by scored season against 'zero'" in text and "true" in text
    assert "paired" not in report(res[res.split == "all"], ref="zero", compare=False)


def test_the_fill_rules():
    rows = pd.DataFrame({"player_id": [1, 2, 3], "o": [-2.0, -4.0, 8.0], "d": [1.0, 3.0, -9.0],
                         "poss": [100.0, 300.0, 5000.0]})
    assert fill_rule("")(rows) == (0.0, 0.0)
    assert fill_rule(None)(rows) == (0.0, 0.0)
    assert fill_rule("-1.6/-0.5")(rows) == (-1.6, 0.5), "positive-good in, raw sign out"
    assert fill_rule("500")(rows) == pytest.approx((-3.5, 2.5))
    assert fill_rule("500x0.25")(rows) == pytest.approx((-0.875, 0.625))


def test_a_neighbour_table_carries_its_fill():
    table = pd.DataFrame({"player_id": [1, 2], "season": [2003, 2003], "o": [1.0, -2.0], "d": [0.5, 0.0],
                          "poss": [2000.0, 100.0]})
    system = NeighbourTable("t:prev", table, -1, frozenset({2003}), fill_rule("500"))
    ratings = system.ratings(system.rated_season(2004))
    assert (ratings.fill_o, ratings.fill_d) == (-2.0, 0.0)
    assert list(ratings.df.columns) == ["player_id", "o", "d", "poss"]
