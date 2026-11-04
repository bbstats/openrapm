"""Read-only: what win probability the garbage-time rule amounts to, from the win model and from how games ended."""
import sys

import numpy as np
import pandas as pd

from openrapm import wins
from openrapm.commands.wins import season_stints
from openrapm.data.config import load_config

sys.stdout.reconfigure(encoding="utf-8")
cfg = load_config()
SEASONS = list(range(2017, 2027))
print("the model's chance the leader wins, teams of average strength on a neutral floor")
stints, seconds = season_stints(cfg, 2026, ("RS",))
model = wins.fit_win_model(stints, seconds, 2026)
for margin, minutes in ((20, 12), (20, 9), (20, 6), (15, 5.99), (15, 3), (15, 1), (12, 1), (25, 18)):
    p = float(wins.win_probability(float(margin), minutes * 60.0 / 2880.0, model, neutral=True))
    print(f"  up {margin} with {minutes:>5} minutes left: {p:.4f}")
rows = []
first_rows = []
for season in SEASONS:
    stints, seconds = season_stints(cfg, season, ("RS",))
    model = wins.fit_win_model(stints, seconds, season)
    left = wins.share_left(stints.period, stints.start_clock, seconds)[0]
    margin = stints.margin_h.to_numpy(dtype=float)
    wp = wins.win_probability(np.abs(margin), left, model, neutral=True)
    poss = (stints.poss_h + stints.poss_a).to_numpy(dtype=float)
    flagged = stints.is_gt.to_numpy(dtype=bool)
    rows.append(pd.DataFrame({"season": season, "wp": wp, "poss": poss, "flagged": flagged,
                              "period": stints.period.to_numpy()}))
    results = wins.game_results(stints).set_index("game_id").margin
    gt = stints[flagged].groupby("game_id", sort=False).head(1)
    final = results.reindex(gt.game_id).to_numpy()
    start = gt.margin_h.to_numpy(dtype=float)
    first_rows.append(pd.DataFrame({"season": season, "margin": np.abs(start), "held": np.sign(start) == np.sign(final),
                                    "final_ok": final != 0, "clock": gt.start_clock.to_numpy()}))
all_rows = pd.concat(rows, ignore_index=True)
first = pd.concat(first_rows, ignore_index=True)
first = first[first.final_ok]
total = all_rows.poss.sum()
print("\n2017-2026, regular seasons")
print(f"  possessions flagged as garbage time: {all_rows.poss[all_rows.flagged].sum() / total:.4f}")
print(f"  games that ever reach it: {len(first):,}; the leader then won {first.held.mean():.4f}")
print("  the leader's model chance on flagged stints, possession-weighted: "
      + ", ".join(f"{q:.0%} of them below {np.quantile(np.repeat(all_rows.wp[all_rows.flagged].to_numpy(), 1), q):.4f}" for q in (0.01, 0.05, 0.25, 0.5)))
for threshold in (0.95, 0.97, 0.98, 0.99, 0.995):
    above = all_rows.wp >= threshold
    both = all_rows.poss[above & all_rows.flagged].sum()
    print(f"  chance {threshold:.3f} or more: {all_rows.poss[above].sum() / total:.4f} of possessions; "
          f"flagged now and above {both / all_rows.poss[all_rows.flagged].sum():.3f} of flagged; "
          f"above but not flagged {all_rows.poss[above & ~all_rows.flagged].sum() / total:.4f} of all, "
          f"of which before the fourth {all_rows.poss[above & ~all_rows.flagged & (all_rows.period < 4)].sum() / total:.4f}")
