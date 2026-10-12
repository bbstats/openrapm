"""Load config.yaml, and resolve the data paths it names.

`load_config()` reads the file named by the environment variable OPENRAPM_CONFIG when it is set, and
the repository's own config.yaml otherwise, so a verification run can point every data path at
another folder without touching the code.  A relative path in `paths` is relative to the repository
root; an absolute one is used as it is.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import yaml

from .legacy_names import player_unit_name

ROOT = Path(__file__).resolve().parents[3]
CONFIG_ENV = "OPENRAPM_CONFIG"

# Where each data path points when the config does not name it: the folders the project has always
# used, so a config written before a key existed resolves to the same place on disk.
DEFAULT_PATHS = {
    "raw": "data/raw",                # the scraped game logs, play-by-play and box scores
    "stints": "data/stints",          # stints, their diagnostics and the per-game shooter tables
    "cache": "data/cache",            # rebuildable caches: roles, bio, the per-season design pieces
    "shot_curves": "data/shotcurve",  # the league's make probability by distance, one file per season
    "outputs": "outputs",             # the ratings tables, the saved prior models, the run logs
    "panel": "outputs/season_panel.parquet",   # the season panel: one row per player, season and side
    "within": "outputs/within",       # the within-season folds, one folder per tag
    "event_counts": "data/event_counts",   # the play-by-play counts per game, player and phase
}


def config_path(path: str | Path | None = None) -> Path:
    """The config file to read: `path`, else $OPENRAPM_CONFIG, else the repository's config.yaml."""
    if path is not None:
        return Path(path)
    env = os.environ.get(CONFIG_ENV, "").strip()
    return Path(env) if env else ROOT / "config.yaml"


def load_config(path: str | Path | None = None) -> dict:
    """The config file (`config_path`) as a dict, with the repository root under "_root".  A player_unit written
    before 2026-10-11 ("window", "season") is read as its plain name (data/legacy_names.py)."""
    with open(config_path(path), encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["_root"] = str(ROOT)
    if "player_unit" in cfg:
        cfg["player_unit"] = player_unit_name(cfg["player_unit"])
    return cfg


def resolve_path(cfg: dict, path: str | Path) -> Path:
    """A configured path: relative to the repository root, or absolute and used as it is."""
    p = Path(path)
    return p if p.is_absolute() else Path(cfg["_root"]) / p


def resolve(cfg: dict, key: str) -> Path:
    """Resolve a path from cfg['paths'] (falling back to DEFAULT_PATHS) relative to the repo root."""
    paths = cfg.get("paths", {})
    if key not in paths and key not in DEFAULT_PATHS:
        raise KeyError(f"config.yaml paths has no {key!r}")
    return resolve_path(cfg, paths.get(key, DEFAULT_PATHS.get(key)))


def booster_params(cfg: dict, path: str | Path) -> dict:
    """{"O": LightGBM settings, "D": LightGBM settings} from a tracked params/ file.

    `path` is `settings.PriorSettings.booster_file`, relative to the repository root.  Only the tracked copy is
    read: an untracked file of the same name elsewhere can never shadow it.
    """
    spec = json.loads(resolve_path(cfg, path).read_text(encoding="utf-8"))
    for side in ("O", "D"):
        if spec.get(side, {}).get("learner") != "lightgbm" or "params" not in spec[side]:
            raise ValueError(f"{path}: side {side} must name the LightGBM learner and its params")
    return {side: dict(spec[side]["params"]) for side in ("O", "D")}
