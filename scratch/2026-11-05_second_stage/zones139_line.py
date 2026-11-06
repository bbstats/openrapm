"""Writes outputs/second_prior/zones139_line/: the screen's sums cut to experiment 139's build set,
second_prior.INPUT_SETS["context_types_team_zones_centred"], so its line needs no new pass over the folds.  The cut is
the same normal equations on fewer columns, so the line is what `openrapm second-prior` would fit on these inputs."""
import json
import shutil
from pathlib import Path

import numpy as np

from openrapm import rating_line, second_prior as sp
from openrapm.data.config import load_config
from openrapm.data.second_prior_files import grams_directory, load_grams, rating_sums_path, write_rating_sums, write_run

cfg = load_config()
F = rating_line.FITTED_PARTS
union = sp.INPUT_SETS["context_types_team_rates_zones_centred"]
chosen = sp.INPUT_SETS["context_types_team_zones_centred"]
K = len(union)
at = np.array([0, 1, *[F + union.index(n) for n in chosen], *[F + K + union.index(n) for n in chosen]])
source, target = grams_directory(cfg, "zones139"), grams_directory(cfg, "zones139_line")
target.mkdir(parents=True, exist_ok=True)
for season, sums in load_grams(cfg, "zones139").items():
    write_rating_sums(rating_sums_path(target, season), rating_line.RatingSums(Q=sums.Q[np.ix_(at, at)], c=sums.c[at], yy=sums.yy, sw=sums.sw))
record = json.loads((source / "run.json").read_text(encoding="utf-8"))
record.update(inputs=chosen, cut_from="zones139", argv=["scratch/2026-11-05_second_stage/zones139_line.py"])
write_run(target, record)
print("wrote", target, len(chosen), "inputs an end")
