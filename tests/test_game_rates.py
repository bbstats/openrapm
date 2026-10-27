"""The multi-stat network's inputs (game_rates.py) on a made-up league: a game's own outcome never reaches its own
inputs, a season never reaches another, the opponent factor reads only the opponent's earlier games, the padding
starts at the league level, and the per-game rows a fit trains on skip a season in progress.

The league: two seasons, four teams of six players, every team playing every other twice a season, counts drawn at
random.  Nothing here needs PyTorch or the data on disk.
"""
import numpy as np
import pandas as pd
import pytest

from openrapm import game_rates as G
from openrapm.data import player_games as P

BOX = ["fg3m", "fg3_miss", "fg2m", "fg2_miss", "ftm", "ft_miss", "orb", "drb", "ast", "tov", "stl", "blk", "pf"]


def _league(seasons=(2018, 2019), teams=4, roster=6, seed=0) -> pd.DataFrame:
    """Per-game rows in data/player_games.py's layout: every pair of teams meets twice a season, each team plays its
    first five, the sixth man every other game."""
    rng = np.random.default_rng(seed)
    rows = []
    for season in seasons:
        meetings = [(a, b) for a in range(teams) for b in range(teams) if a != b]
        for game_idx, (home, away) in enumerate(meetings):
            date = pd.Timestamp(f"{season - 1}-11-01") + pd.Timedelta(days=game_idx)
            for team, opponent, side in ((home, away, 1.0), (away, home, -1.0)):
                for slot in range(roster):
                    if slot == 5 and game_idx % 2:
                        continue
                    poss = float(rng.integers(20, 80))
                    row = {"season": season, "phase": "RS", "game_id": f"{season}{game_idx:04d}",
                           "game_date": date, "game_idx": game_idx, "player_id": 100 * team + slot,
                           "team_id": team, "opponent_id": opponent, "home": side, "started": float(slot < 5),
                           "poss_off": poss, "poss_def": poss + rng.integers(-2, 3), "team_poss_off": 100.0,
                           "team_poss_def": 100.0}
                    row.update({count: float(rng.poisson(3.0)) for count in BOX})
                    row.update({name: float(rng.poisson(2.0)) if season >= 2017 else np.nan for name in P.HUSTLE})
                    rows.append(row)
    return pd.DataFrame(rows)


def _inputs(rows: pd.DataFrame, league: pd.DataFrame, pads=2000.0, ks=300.0):
    """(games, inputs, factors, padding) with the league levels held at `league`: a season constant, the way a
    rating reads them, so a check of one game's outcome sees only what the sums carry."""
    games = G.games_from_rows(rows)
    league_row = G.league_by_row(games, league)
    factors = G.opponent_factors(G.opponent_sums(games, league_row), np.full(len(G.STATS), pads))
    padding = G.padding_forecasts(games, league_row, factors, np.full(len(G.STATS), ks))
    return games, G.network_inputs(games, league_row, factors, padding), factors, padding


def _row(games, player_id, game_id):
    frame = games.frame
    return int(np.flatnonzero((frame.player_id == player_id).to_numpy() & (frame.game_id == game_id).to_numpy())[0])


def test_a_games_own_outcome_never_reaches_its_own_inputs():
    """The handoff's leakage check on the real code: edit one game's counts; that game's inputs do not move, for
    him or for anyone in the game, and his next game's do."""
    rows = _league()
    league = G.league_levels(G.games_from_rows(rows))
    games, before, _, _ = _inputs(rows, league)
    edited = rows.copy()
    target = (edited.player_id == 102) & (edited.game_id == "20190005")
    edited.loc[target, ["ast", "fg3m", "drb", "deflections"]] += 6.0
    games_after, after, _, _ = _inputs(edited, league)
    in_the_game = np.flatnonzero((games.frame.game_id == "20190005").to_numpy())
    np.testing.assert_array_equal(before[in_the_game], after[in_the_game])
    his = np.flatnonzero((games.frame.player_id == 102).to_numpy() & (games.frame.season == 2019).to_numpy())
    position = int(np.flatnonzero(his == _row(games, 102, "20190005"))[0])
    assert not np.array_equal(before[his[position + 1]], after[his[position + 1]])


def test_a_season_never_reaches_another():
    """Single season: every input of 2018 is the same whatever 2019's games say."""
    rows = _league()
    league = G.league_levels(G.games_from_rows(rows))
    games, before, _, _ = _inputs(rows, league)
    noisy = rows.copy()
    later = noisy.season == 2019
    noisy.loc[later, BOX] = noisy.loc[later, BOX] * 3 + 1
    _, after, _, _ = _inputs(noisy, league)
    earlier = (games.season == 2018)
    np.testing.assert_array_equal(before[earlier], after[earlier])


def test_the_opponent_factor_reads_the_opponents_earlier_games_only():
    """A team's first game is met at a factor of 1; a defence that gave up many assists in its earlier games makes
    its next opponents' assists count for less."""
    rows = _league(seasons=(2018,))
    league = G.league_levels(G.games_from_rows(rows))
    games, _, factors, _ = _inputs(rows, league, pads=50.0)
    assists = G.NAMES.index("assists")
    first = games.frame.groupby("opponent_id").game_idx.transform("min").to_numpy() == games.frame.game_idx.to_numpy()
    real = ~games.frame.end.to_numpy()
    np.testing.assert_allclose(factors[first & real, assists], 1.0)
    generous = rows.copy()
    generous.loc[(generous.opponent_id == 3) & (generous.game_idx < 6), "ast"] += 10.0
    games2, _, factors2, _ = _inputs(generous, league, pads=50.0)
    later = (games2.frame.opponent_id == 3).to_numpy() & (games2.frame.game_idx > 6).to_numpy() & real
    assert (factors2[later, assists] > factors[later, assists]).all()


def test_the_padding_starts_each_season_at_the_league_level_and_percentages_have_no_opponent():
    rows = _league()
    games = G.games_from_rows(rows)
    league = G.league_levels(games)
    _, _, factors, padding = _inputs(rows, league)
    first = games.position == 0
    np.testing.assert_allclose(padding[first], G.league_by_row(games, league)[first])
    assert (factors[:, G.IS_PCT] == 1.0).all()


def test_end_rows_carry_the_whole_season_and_no_loss():
    """Each player-season ends with a row after his last game: no outcome, and his state is his whole season."""
    rows = _league(seasons=(2018,))
    games = G.games_from_rows(rows)
    end = games.frame.end.to_numpy()
    assert end.sum() == rows.groupby(["season", "player_id"]).ngroups
    assert not games.available[end].any()
    counted = games.frame.loc[end].set_index("player_id")
    games_played = rows.groupby("player_id").size()
    assert (games.position[end] == games_played.reindex(counted.index).to_numpy()).all()


def test_forward_numpy_starts_as_the_padding():
    """The last layer at zero: before training, the network's links are the padding's."""
    rng = np.random.default_rng(1)
    weights = [(rng.normal(size=(5, 4)), rng.normal(size=4)), (np.zeros((4, 3)), np.zeros(3))]
    offsets = rng.normal(size=(7, 3))
    np.testing.assert_allclose(G.forward_numpy(weights, rng.normal(size=(7, 5)), offsets), offsets)


def test_the_rows_a_fit_trains_on_skip_a_season_in_progress(tmp_path, monkeypatch):
    """`trainable_player_games` drops a season whose Finals are not over and says so."""
    cfg = {"_root": str(tmp_path), "paths": {"cache": "cache"}, "box_counts": BOX}
    rows = _league(seasons=(2018, 2019))
    folder = tmp_path / "cache" / "player_games"
    folder.mkdir(parents=True)
    for season, part in rows.groupby("season"):
        part.to_parquet(folder / f"{season}.parquet", index=False)
    monkeypatch.setattr("openrapm.data.seasons.trainable_through", lambda cfg: 2018)
    kept = P.trainable_player_games(cfg, [2018, 2019])
    assert set(kept.season) == {2018}
    with pytest.raises(FileNotFoundError):
        P.trainable_player_games(cfg, [2017])
