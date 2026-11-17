"""The old outputs, sized, each marked by whether the live code names it.  Deletes nothing.

    python scratch/2026-11-16_cleanup/inventory.py      writes outputs/cleanup_proof/inventory.csv

An item is a file or folder at the top of outputs/, outputs/within/ or outputs/second_prior/, or data/cache/player_games.
Its stem, the name less its prefix and suffix (prior_models_fade141.pkl is fade141), is "named" when it appears in
src/, scripts/ outside scripts/checks/, tests/, params/, config.yaml or notes/HANDOFF.md, or is one of the published
names; the checks are history and do not keep a file.
"""
import os
import re
from pathlib import Path

import pandas as pd

ROOT = Path("A:/code/spmm")
OUT = ROOT / "outputs"
PREFIXES = ("prior_models_", "season_ratings_", "game_priors_", "game_rates_")
SUFFIXES = (".pkl", ".parquet", ".csv", ".json", ".log", ".txt", "_step3", "_step4", "_regular")


def size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    total = 0
    for folder, _, files in os.walk(path):
        for name in files:
            try:
                total += os.stat(os.path.join(folder, name)).st_size
            except OSError:
                pass
    return total


def stem(name: str) -> str:
    for prefix in PREFIXES:
        if name.startswith(prefix):
            name = name[len(prefix):]
    changed = True
    while changed:
        changed = False
        for suffix in SUFFIXES:
            if name.endswith(suffix):
                name, changed = name[: -len(suffix)], True
    return name


def named_text() -> str:
    parts = []
    for folder in ("src", "tests", "params", "scripts"):
        for path in (ROOT / folder).rglob("*"):
            if path.is_file() and "checks" not in path.parts and path.suffix in {".py", ".sh", ".json", ".yaml"}:
                parts.append(path.read_text(encoding="utf-8", errors="replace"))
    for path in (ROOT / "config.yaml", ROOT / "notes/HANDOFF.md"):
        parts.append(path.read_text(encoding="utf-8", errors="replace"))
    return "\n".join(parts)


TEXT = named_text()
items = [*OUT.iterdir(), *(OUT / "within").iterdir(), *(OUT / "second_prior").iterdir(),
         ROOT / "data/cache/player_games"]
rows = []
for path in items:
    if not path.exists() or path in (OUT / "within", OUT / "second_prior"):
        continue
    name = stem(path.name)
    named = bool(name) and re.search(rf"(?<![A-Za-z0-9]){re.escape(name)}(?![A-Za-z0-9])", TEXT) is not None
    rows.append(dict(path=path.relative_to(ROOT).as_posix(), stem=name, gb=size(path) / 1e9, named=named))
frame = pd.DataFrame(rows).sort_values("gb", ascending=False)
frame.to_csv(OUT / "cleanup_proof/inventory.csv", index=False)
print(f"{len(frame)} items, {frame.gb.sum():.0f} GB; named {frame[frame.named].gb.sum():.0f} GB, "
      f"not named {frame[~frame.named].gb.sum():.0f} GB")
print(frame.head(40).to_string(index=False))
