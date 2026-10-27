"""The multi-stat networks on disk: one file per rated season under a tag, written by `openrapm game-rates train`
and read by every rating that uses them (`openrapm rate --game-prior TAG` and the within-season builds).

    paths.outputs/game_rates/<tag>/season_<S>.npz

Each file holds what the network of rated season S needs to forecast and what it was trained on:
    W0, b0, W1, b1, ...        the layers, first to last (game_rates.forward_numpy)
    mean, sd                   the inputs' standardisation, from its training rows
    ks, opponent_pads          the padding's pad per stat and the opponent factor's (infinity: no factor)
    seasons                    the seasons it trained on: never S, S - 1 or S + 1, never a season in progress
    validation_players         the players whose rows stopped its training and never trained it
    input_names, stats         the inputs and the stats, in order, checked when the file is read back

Example: `load_network(cfg, "net1", 2010)` reads outputs/game_rates/net1/season_2010.npz, a network trained on
1997-2008 and 2012-2026.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import resolve


@dataclass
class SavedNetwork:
    """One rated season's network as read back."""
    season: int
    weights: list              # [(W, b), ...]
    mean: np.ndarray
    sd: np.ndarray
    ks: np.ndarray
    opponent_pads: np.ndarray
    seasons: np.ndarray
    validation_players: np.ndarray
    input_names: list
    stats: list


def network_dir(cfg: dict, tag: str) -> Path:
    return resolve(cfg, "outputs") / "game_rates" / str(tag)


def network_path(cfg: dict, tag: str, season: int) -> Path:
    return network_dir(cfg, tag) / f"season_{int(season)}.npz"


def save_network(cfg: dict, tag: str, network: SavedNetwork) -> Path:
    """Write one rated season's network (the module docstring); returns the path."""
    path = network_path(cfg, tag, network.season)
    path.parent.mkdir(parents=True, exist_ok=True)
    layers = {f"{part}{i}": array for i, (weight, bias) in enumerate(network.weights)
              for part, array in (("W", weight), ("b", bias))}
    np.savez_compressed(path, season=np.int64(network.season), mean=network.mean, sd=network.sd, ks=network.ks,
                        opponent_pads=network.opponent_pads, seasons=np.asarray(network.seasons, dtype=np.int64),
                        validation_players=np.asarray(network.validation_players, dtype=np.int64),
                        input_names=np.asarray(network.input_names), stats=np.asarray(network.stats), **layers)
    return path


def load_network(cfg: dict, tag: str, season: int, input_names=None, stats=None) -> SavedNetwork:
    """One rated season's network; with `input_names` and `stats`, refused when they differ from the file's (the
    inputs are read by position)."""
    path = network_path(cfg, tag, season)
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist; train it with `openrapm game-rates train --tag {tag}`")
    with np.load(path) as file:
        n_layers = sum(1 for key in file.files if key.startswith("W"))
        network = SavedNetwork(season=int(file["season"]),
                               weights=[(file[f"W{i}"], file[f"b{i}"]) for i in range(n_layers)],
                               mean=file["mean"], sd=file["sd"], ks=file["ks"], opponent_pads=file["opponent_pads"],
                               seasons=file["seasons"], validation_players=file["validation_players"],
                               input_names=[str(name) for name in file["input_names"]],
                               stats=[str(name) for name in file["stats"]])
    for what, expected, held in (("inputs", input_names, network.input_names), ("stats", stats, network.stats)):
        if expected is not None and list(expected) != list(held):
            raise ValueError(f"{path}: its {what} differ from today's code's; train the tag again")
    return network
