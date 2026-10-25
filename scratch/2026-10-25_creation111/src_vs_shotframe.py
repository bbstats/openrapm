"""The pipeline's shot reader (data/creation_counts.py) against the research table (data/shotframe), 2016."""
import sys
import time

import pandas as pd

sys.path.insert(0, "scratch/2026-10-25_creation111")
from cells import load_shots  # noqa: E402

from openrapm.data.config import load_config  # noqa: E402
from openrapm.data.creation_counts import season_shots  # noqa: E402

season = int(sys.argv[1]) if len(sys.argv) > 1 else 2016
started = time.time()
shots, trips = season_shots(season, load_config())
print(f"{season}: {len(shots)} shots, {int(trips.trips.sum())} trips, {int(trips.and_one_trips.sum())} and-ones "
      f"({time.time() - started:.0f}s)")
old = load_shots(season)
both = shots.merge(old, on=["game_id", "action_number"], suffixes=("", "_old"))
print(f"matched {len(both)} of {len(shots)} new and {len(old)} research rows")
both["assisted_old"] = both.post_assisted
for column in ("made", "assisted", "spot", "context", "cell"):
    print(f"  {column} agrees {(both[column] == both[column + '_old']).mean():.4f}")
print(f"  and-one agrees on makes {(both.and_one[both.made == 1] == both.post_and1[both.made == 1]).mean():.4f}")
print(pd.crosstab(both.context_old, both.context).to_string())
