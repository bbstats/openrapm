"""The rebuilt Detail file against the one the site shows (docs/data/decompose_openrapm.json as of 2026-11-16): each
of a rating's five pieces, offense and defense, compared player for player.  The box-score piece comes from the season
panel through the booster, so a panel that differed would show here first.

    python scratch/2026-11-16_cleanup/pieces_vs_site.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("A:/code/spmm")


def frame(path: Path) -> pd.DataFrame:
    data = json.loads(path.read_text(encoding="utf-8"))
    pieces = data["meta"]["pieces"]
    rows = []
    for season, players in data["rows"].items():
        for row in players:
            name, pid, _, off, dfn = row[0], row[1], row[2], row[3], row[4]
            rows.append({"season": int(season), "player_id": pid, "name": name,
                         **{f"o_{p}": v for p, v in zip(pieces, off)}, **{f"d_{p}": v for p, v in zip(pieces, dfn)}})
    return pd.DataFrame(rows)


site = frame(ROOT / "docs/data/decompose_openrapm.json")
new = frame(ROOT / "outputs/rebuild/site/data/decompose_openrapm.json")
both = site.merge(new, on=["season", "player_id"], suffixes=("_site", "_new"))
print(f"site {len(site)} player-seasons, rebuilt {len(new)}, matched {len(both)}")
finished = both[both.season <= 2026]
rows = []
for column in [c for c in site.columns if c[:2] in ("o_", "d_")]:
    a, b = finished[f"{column}_site"], finished[f"{column}_new"]
    gap = (a - b).abs()
    rows.append(dict(piece=column, corr=round(float(np.corrcoef(a, b)[0, 1]), 5), mean_gap=round(float(gap.mean()), 3),
                     p99_gap=round(float(gap.quantile(0.99)), 2), max_gap=round(float(gap.max()), 2),
                     share_within_0_02=round(float((gap <= 0.02).mean()), 3)))
print("finished seasons, 1997-2026, points per 100 (the site rounds to 0.01):")
print(pd.DataFrame(rows).to_string(index=False))

# The box-score piece is the booster's prior times one number per season and side, the prior scale and the shrink's
# multiplier, which the rebuild's own held-back games refit.  So if the panel and the booster are the same, the rebuilt
# piece is the site's times one constant per season and side, and what is left over is the site's rounding.
left = []
for (season, ), g in finished.groupby(["season"]):
    for side in ("o", "d"):
        a, b = g[f"{side}_box_score_site"].to_numpy(), g[f"{side}_box_score_new"].to_numpy()
        c = float(a @ b / (a @ a))
        rest = np.abs(b - c * a)
        left.append(dict(season=season, side=side, constant=round(c, 4), mean_left=round(float(rest.mean()), 4),
                         max_left=round(float(rest.max()), 3)))
left = pd.DataFrame(left)
print("\nbox-score piece after one constant per season and side:")
print(left.describe().loc[["mean", "max"]].round(4).to_string())
print(left.sort_values("max_left", ascending=False).head(5).to_string(index=False))

# Which players moved: those whose height or weight the NBA's bio feed changed since the copy of 2026-09-07 that the
# deletion missed (data/cache_before_parser94/bio.parquet), against the rest
old_bio = pd.read_parquet(ROOT / "data/cache_before_parser94/bio.parquet")
new_bio = pd.read_parquet(ROOT / "data/cache/bio.parquet")
bio = old_bio.merge(new_bio, on="player_id", suffixes=("_old", "_new"))
changed = set(bio[(bio.height_old != bio.height_new) | (bio.weight_old != bio.weight_new)].player_id.astype(int))
moved = []
for (season, ), g in finished.groupby(["season"]):
    for side in ("o", "d"):
        a, b = g[f"{side}_box_score_site"].to_numpy(), g[f"{side}_box_score_new"].to_numpy()
        c = float(a @ b / (a @ a))
        moved.append(pd.DataFrame({"player_id": g.player_id.to_numpy(), "left": np.abs(b - c * a)}))
moved = pd.concat(moved)
moved["bio_changed"] = moved.player_id.astype(int).isin(changed)
print("\nwhat is left of the box-score piece, by whether the player's height or weight changed in the bio feed:")
print(moved.groupby("bio_changed").left.describe(percentiles=[0.5, 0.9, 0.99]).round(3).to_string())
