"""Load config.yaml, and resolve the data paths it names.

`load_config()` reads the file named by the environment variable OPENRAPM_CONFIG when it is set, and
the repository's own config.yaml otherwise, so a verification run can point every data path at
another folder without touching the code.  A relative path in `paths` is relative to the repository
root; an absolute one is used as it is.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
CONFIG_ENV = "OPENRAPM_CONFIG"

# Where each data path points when the config does not name it: the folders the project has always
# used, so a config written before a key existed resolves to the same place on disk.
DEFAULT_PATHS = {
    "raw": "data/raw",                # the scraped game logs, play-by-play and box scores
    "stints": "data/stints",          # stints, their diagnostics and the per-game shooter tables
    "cache": "data/cache",            # rebuildable caches: roles, bio, the per-season design pieces
    "shot_curves": "data/shotcurve",  # the league's make probability by distance, one file per season
}

# Windows consoles default to cp1252; player names are UTF-8 (Schröder, Ginóbili).
for _stream in ("stdout", "stderr"):
    try:
        getattr(sys, _stream).reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass


def config_path(path: str | Path | None = None) -> Path:
    """The config file to read: `path`, else $OPENRAPM_CONFIG, else the repository's config.yaml."""
    if path is not None:
        return Path(path)
    env = os.environ.get(CONFIG_ENV, "").strip()
    return Path(env) if env else ROOT / "config.yaml"


def load_config(path: str | Path | None = None) -> dict:
    with open(config_path(path), encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["_root"] = str(ROOT)
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
