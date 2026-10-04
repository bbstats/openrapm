"""Is "top players are overrated" just regression to the mean?  Rank players by a season OUTSIDE the test window.

    python scripts/96_rtm_test.py [--alpha=outputs/tradeset_swapadj_within_alpha.parquet]
                                  [--ratings=outputs/season_ratings_swapadj_within.parquet]

The owner, 2026-10-03, on the finding that the top 30 are overrated by 0.5+ twice as often as underrated: "I think
this might just be regression to the mean ... Lets test it."

A player-season's miss (alpha) is measured on the team-games of seasons H-1, H and H+1 (scripts/70_tradeset.py).
Ranking players by their season-H rating and then finding the top overrated mixes two things: a real overrating of
good players, and regression to the mean -- a season ranked near the top is partly a peak year or a lucky rating,
and the neighbouring seasons come back down.  Ranking by a season that shares no games with the window (H-2 or H+2)
removes the second: selection on H-2's rating cannot pick H's luck or a peak in H.  So:

  - top 30 by H-2 (or H+2) still overrated in H  -> a real overrating of good players;
  - the imbalance disappears                     -> it was regression to the mean from ranking on season H.

Each comparison is made on the same rows: players rated in both H and the ranking season.  H-1 and H+1 rankings are
printed too, but they share games with the window (their own luck enters alpha), so they lean toward "underrated".
z: the mean of alpha against 0, each player's seasons one cluster.  Reads only.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _cli import check_flags, flag  # noqa: E402

TIERS = [0, 30, 90, 150, 300, 10 ** 6]
TIER_NAMES = ["top 30", "31-90", "91-150", "151-300", "301+"]
CUT = 0.5


def clustered_z(values: np.ndarray, players: np.ndarray) -> float:
    mean = float(values.mean())
    by_player = pd.Series(values - mean).groupby(players).sum().to_numpy()
    se = float(np.sqrt(np.sum(by_player ** 2))) / len(values)
    return mean / se if se > 0 else np.nan


def main() -> None:
    check_flags()
    alpha = pd.read_parquet(ROOT / flag("alpha", "outputs/tradeset_swapadj_within_alpha.parquet"))
    alpha = alpha[alpha.eligible & (alpha.without_poss >= 1.0)][["player_id", "season", "side", "alpha_good"]]
    ratings = pd.read_parquet(ROOT / flag("ratings", "outputs/season_ratings_swapadj_within.parquet"))
    ratings["rank"] = ratings.groupby("season").rating_total.rank(ascending=False, method="first")
    rank = ratings.set_index(["player_id", "season"])["rank"]
    rows = []
    for offset, label in ((0, "H (the tested season)"), (-2, "H-2"), (2, "H+2"), (-1, "H-1 (shares games)"),
                          (1, "H+1 (shares games)")):
        x = alpha.copy()
        x["rank_H"] = rank.reindex(list(zip(x.player_id, x.season))).to_numpy()
        x["rank_sel"] = rank.reindex(list(zip(x.player_id, x.season + offset))).to_numpy()
        x = x.dropna(subset=["rank_H", "rank_sel"])
        for side in ("offense", "defense"):
            s = x[x.side == side]
            for by, column in (("rank in " + label, "rank_sel"), ("rank in H, same rows", "rank_H")):
                if offset == 0 and column == "rank_H":
                    continue
                tier = pd.cut(s[column], TIERS, labels=TIER_NAMES)
                for name in TIER_NAMES[:2]:
                    t = s[tier == name]
                    if len(t) < 30:
                        continue
                    a = t.alpha_good.to_numpy()
                    rows.append(dict(side=side, ranked_by=by, rows_for=label, tier=name, n=len(t),
                                     overrated=float(np.mean(a <= -CUT)), underrated=float(np.mean(a >= CUT)),
                                     mean_alpha=float(a.mean()), z=clustered_z(a, t.player_id.to_numpy())))
    out = pd.DataFrame(rows)
    show = out.assign(overrated=(100 * out.overrated).round(0).astype(int).astype(str) + "%",
                      underrated=(100 * out.underrated).round(0).astype(int).astype(str) + "%",
                      mean_alpha=out.mean_alpha.round(3), z=out.z.round(1))
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        for side in ("offense", "defense"):
            print(f"\n=== {side}: share of player-seasons the three-season games say are overrated / underrated by "
                  f"{CUT}+ per 100, and the mean miss (negative = overrated)")
            print(show[show.side == side].drop(columns="side").to_string(index=False))
    path = ROOT / "outputs" / "csv" / "rtm_test.csv"
    out.to_csv(path, index=False)
    print(f"\nwrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
