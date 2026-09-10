"""Archetypes without a hand-made axis: where does the board disagree with the consensus, by player TYPE?

`tests/test_vs_consensus.py` guards one hand-built axis, `bigness` = per-36 (orb + blk + 0.3 drb - 0.5 ast
- 0.4 fg3m), because that is where the board's original defect ran.  A single formula with five weights in it
is a guess about what a player type is.  This fits the types instead: a Bayesian Gaussian mixture over the
per-36 box profile, which chooses how many clusters it needs (an unused component's weight goes to zero), and
then reports the board's disagreement with the consensus per cluster, split into offense, defense and total.

    python scripts/58_archetype.py [--board=outputs/season_ratings.parquet] [--k=8] [--seasons=2024,2025,2026]

Read the TOTAL column first.  A cluster whose offense and defense gaps are large and opposite is an
attribution disagreement -- the board and the consensus agree about how good that type of player is and
disagree about which side of the ball it comes from.  Only a large total gap is a rating that is wrong.
"""
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eracoef.boxtable import season_box  # noqa: E402
from eracoef.config import load_config  # noqa: E402

pd.set_option("display.width", 250, "display.max_columns", 40)
SUFFIX = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b")
RATES = ["pts", "fg3m", "fg3_miss", "ftm", "ast", "orb", "drb", "stl", "blk", "tov", "pf"]


def flag(name, default=None):
    hit = [a for a in sys.argv[1:] if a.startswith(f"--{name}=")]
    return hit[0].split("=", 1)[1] if hit else default


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    s = s.replace("&", " ").replace("-", " ").replace("'", "").replace(".", " ")
    return re.sub(r"\s+", " ", SUFFIX.sub(" ", s)).strip()


def main():
    cfg = load_config()
    root = Path(cfg["_root"])
    seasons = [int(s) for s in flag("seasons", "2024,2025,2026").split(",")]
    board = pd.read_parquet(root / flag("board", "outputs/season_ratings.parquet"))
    con = pd.read_csv(root / "data" / "external" / "consensus.csv")
    con = con.dropna(subset=["player_name", "adj_overall"]).assign(key=lambda d: d.player_name.map(norm))
    con = con.drop_duplicates("key")[["key", "adj_offense", "adj_defense", "adj_overall"]]

    # the board pooled over the seasons the consensus covers, each season weighted by its possessions
    d = board[board.season.isin(seasons) & board.player_name.notna()].copy()
    w = d.poss_off.to_numpy(dtype=float)
    g = pd.DataFrame({c: (d[c] * w).groupby(d.player_id).sum() / d.groupby("player_id").poss_off.sum()
                      for c in ("rating_off", "rating_def", "rating_total")})
    g["poss"] = d.groupby("player_id").poss_off.sum()
    g["player_name"] = d.sort_values("season").drop_duplicates("player_id", keep="last").set_index("player_id").player_name
    g = g.reset_index()
    g = g[g.poss >= 1000].assign(key=lambda x: x.player_name.map(norm))
    g = g.sort_values("poss").drop_duplicates("key", keep="last").merge(con, on="key", how="inner")

    # the per-36 box profile, standardised: what the mixture clusters on
    box = season_box(seasons, ["RS"], cfg)
    b = box[box.phase == "RS"].groupby("player_id", as_index=False)[["minutes", *RATES]].sum()
    for c in RATES:
        b[c] = b[c] / b.minutes.clip(lower=1) * 36
    m = g.merge(b[["player_id", *RATES]], on="player_id", how="inner")
    X = m[RATES].to_numpy(dtype=float)
    X = (X - X.mean(0)) / X.std(0)

    from sklearn.mixture import BayesianGaussianMixture
    k = int(flag("k", 8))
    bgm = BayesianGaussianMixture(n_components=k, covariance_type="full", weight_concentration_prior=1.0 / k,
                                  max_iter=1000, n_init=5, random_state=0).fit(X)
    m["cluster"] = bgm.predict(X)
    z = lambda v: (v - v.mean()) / v.std()                                          # noqa: E731
    for side, col in (("off", "adj_offense"), ("def", "adj_defense"), ("total", "adj_overall")):
        m[f"gap_{side}"] = z(m[f"rating_{'total' if side == 'total' else side}"]) - z(m[col])

    rows = []
    for c, gg in m.groupby("cluster"):
        top = gg.sort_values("poss", ascending=False).player_name.head(3).tolist()
        rows.append(dict(cluster=int(c), n=len(gg), weight=float(bgm.weights_[c]),
                         off=gg.gap_off.mean(), dfn=gg.gap_def.mean(), total=gg.gap_total.mean(),
                         **{r: gg[r].mean() for r in ("pts", "fg3m", "ast", "orb", "blk")},
                         who=", ".join(top)))
    R = pd.DataFrame(rows).sort_values("total")
    print(f"\n{flag('board', 'outputs/season_ratings.parquet')}: {len(m)} players, "
          f"{(bgm.weights_ > 0.01).sum()} of {k} components used\n")
    print("gap = our z minus the consensus z; per-36 means beside it.  Read `total` first.\n")
    print(R.round(2).to_string(index=False))
    print(f"\nspread of the per-cluster TOTAL gap: {R.total.std():.3f} "
          f"(the number an archetype penalty would be asked to shrink)")


if __name__ == "__main__":
    main()
