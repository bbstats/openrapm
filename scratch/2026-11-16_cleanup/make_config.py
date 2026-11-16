"""The proof's config: A:/code/spmm/config.yaml with every data path absolute, into A:/code/spmm.

Both sides of the proof read it through OPENRAPM_CONFIG, so the "before" code at A:/code/spmm_before and the "after"
code at A:/code/spmm_cleanup read the same data and write into the same outputs folder, under scratch names.  The
root a command reads params/ and data/external/ from stays the code's own checkout (data/config.py, ROOT).

    A:/code/spmm/.venv/Scripts/python scratch/2026-11-16_cleanup/make_config.py
"""
from pathlib import Path

import yaml

MAIN = Path("A:/code/spmm")
OUT = MAIN / "outputs" / "cleanup_proof" / "config.yaml"
DEFAULTS = {"raw": "data/raw", "stints": "data/stints", "cache": "data/cache", "shot_curves": "data/shotcurve",
            "outputs": "outputs", "panel": "outputs/season_panel.parquet",
            "panel_so_far": "outputs/season_panel_so_far.parquet", "within": "outputs/within",
            "event_counts": "data/event_counts", "public_ratings": "data/public_ratings"}

cfg = yaml.safe_load((MAIN / "config.yaml").read_text(encoding="utf-8"))
paths = {**DEFAULTS, **cfg.get("paths", {})}
cfg["paths"] = {key: (MAIN / value).as_posix() for key, value in paths.items()}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
print("wrote", OUT)
for key, value in cfg["paths"].items():
    print(f"  {key}: {value}")
