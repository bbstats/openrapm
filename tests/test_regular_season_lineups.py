"""The regular-season rating behind WOSA reads regular-season lineups only (commands/build.py).

WOSA is locked when the regular season ends: it reads each player's rating from that regular season alone
(commands/wins.py).  Until 2026-11-09 the designs and the panel of that rating were cut to the regular season, but Box
Adj's inputs, his teammates' and opponents' priors and his team's shooting with him on and off the court, were still
read from every stint of the season, so a playoff run moved the WOSA of a playoff team's players.  The whole-season
rating reads the whole season, as before."""
import inspect

import pandas as pd
import pytest

from openrapm import swap_test
from openrapm.commands import build
from openrapm.design import SEASON_PHASES
from openrapm.settings import PRODUCT
from openrapm.simulate import simulate

SEASON = 2026


def _stint_files(tmp_path, playoff_points: float = 0.0) -> tuple[dict, set]:
    """(a config, the playoff games): a simulated season's stints on disk as two files, its last ten games the
    playoffs, with `playoff_points` added to every playoff stint's home points."""
    stints = simulate(n_seasons=1, n_teams=6, players_per_team=9, games_per_season=20, stints_per_game=(6, 8),
                      seed=5)["stints"].copy()
    for column in swap_test.TEAM_COUNT_COLUMNS:
        stints[column] = 1.0
    playoffs = set(sorted(stints.game_id.unique())[-10:])
    folder = tmp_path / "data" / "stints"
    folder.mkdir(parents=True, exist_ok=True)
    for phase in SEASON_PHASES:
        part = stints[stints.game_id.isin(playoffs) == (phase == "PO")].assign(phase=phase)
        if phase == "PO":
            part = part.assign(pts_h=part.pts_h + playoff_points)
        part.to_parquet(folder / f"{SEASON}_{phase}.parquet", index=False)
        teams = part.drop_duplicates("game_id").assign(home_team_id=lambda games: games.h1 // 1000,
                                                       away_team_id=lambda games: games.a1 // 1000)
        teams[["game_id", "home_team_id", "away_team_id"]].to_parquet(folder / f"{SEASON}_{phase}_diag.parquet",
                                                                      index=False)
    return {"_root": str(tmp_path), "paths": {"stints": "data/stints"}}, playoffs


def test_the_regular_seasons_lineup_rows_hold_no_playoff_stint_and_no_playoff_result_can_move_them(tmp_path):
    cfg, playoffs = _stint_files(tmp_path / "as_played")
    regular = build.season_lineup_rows(cfg, SEASON, phases=("RS",))
    whole = build.season_lineup_rows(cfg, SEASON)
    assert len(regular) and not set(regular.game_id) & playoffs
    assert playoffs <= set(whole.game_id) and len(whole) > len(regular)       # the whole season still reads both
    other, _ = _stint_files(tmp_path / "another_playoff_run", playoff_points=50.0)
    pd.testing.assert_frame_equal(build.season_lineup_rows(other, SEASON, phases=("RS",)), regular)
    assert not build.season_lineup_rows(other, SEASON).equals(whole)


def test_every_default_is_the_whole_season():
    """A season is its regular season and its playoffs everywhere but in the rating WOSA is locked to
    (tests/test_playoffs_in_fit.py)."""
    for function in (build.season_lineup_rows, build.rated_second_prior, build.rate_one_season):
        assert tuple(inspect.signature(function).parameters["phases"].default) == SEASON_PHASES, function.__name__


def test_the_rating_from_the_regular_season_alone_asks_for_regular_season_lineups(tmp_path, monkeypatch):
    """`rate_regular_seasons` hands the phases WOSA reads, settings.WinsSettings.phases, to `rate_one_season`, which
    hands them to Box Adj's lineup rows."""
    asked = {}

    class Designs:
        def __init__(self, cfg):
            self.cfg = cfg

        def clear(self):
            pass

    def rate(season, sums, panel, designs, settings, params, lines, label_cache, part_seasons, progress, phases=None):
        asked[season] = phases
        return (pd.DataFrame({"season": [season]}),)

    monkeypatch.setattr(build, "DesignCache", Designs)
    monkeypatch.setattr(build, "accumulate_rapm", lambda designs, seasons, settings: {})
    monkeypatch.setattr(build, "trainable", lambda cfg: type("Seasons", (), {"seasons": (2025, 2026)})())
    monkeypatch.setattr(build, "trainable_panel", lambda cfg: pd.DataFrame({"possessions": [1.0]}))
    monkeypatch.setattr(build, "booster_params", lambda cfg, file: {})
    monkeypatch.setattr(build, "line_setup", lambda cfg, tag, inputs: {})
    monkeypatch.setattr(build, "load_grams", lambda cfg, tag: {})
    monkeypatch.setattr(build, "part_season_panel", lambda cfg, settings: None)
    monkeypatch.setattr(build, "panel_sources", lambda cfg: None)
    monkeypatch.setattr(build, "regular_season_cut",
                        lambda season, cfg, designs, targets, sources: ({}, pd.DataFrame({"possessions": [1.0]})))
    monkeypatch.setattr(build, "rate_one_season", rate)
    build.rate_regular_seasons({}, PRODUCT, [SEASON], tmp_path / "parts")
    assert asked == {SEASON: PRODUCT.wins.phases} and PRODUCT.wins.phases == ("RS",)
    assert (tmp_path / "parts" / f"{SEASON}.parquet").exists()


def test_box_adj_reads_the_lineup_rows_of_the_phases_it_is_given(monkeypatch):
    seen = []

    class Stop(Exception):
        pass

    def rows(cfg, season, phases):
        seen.append(phases)
        raise Stop

    monkeypatch.setattr(build, "season_lineup_rows", rows)
    monkeypatch.setattr(build.second_prior, "season_coefficients", lambda *args: {"O": None, "D": None})
    monkeypatch.setattr(build.second_prior, "booster_answers", lambda model, rows: None)
    monkeypatch.setattr(build.box_prior, "rated_rows", lambda *args: None)
    monkeypatch.setattr(build.box_prior, "least_share", lambda settings: 0.25)
    panel = pd.DataFrame({"side": ["O", "D"], "season": [SEASON, SEASON]})
    priors = {side: type("Prior", (), {"model": None, "feature_names": [], "values": {}})() for side in ("O", "D")}
    lines = build.SecondPriorLines({}, {SEASON}, None)
    with pytest.raises(Stop):
        build.rated_second_prior({}, SEASON, panel, priors, lines, PRODUCT, phases=("RS",))
    assert seen == [("RS",)]
