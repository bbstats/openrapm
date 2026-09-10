"""Archetypes without a hand-made axis: where does the board disagree with the consensus, by player TYPE?

`tests/test_vs_consensus.py` guarded one hand-built axis, `bigness` = per-36 (orb + blk + 0.3 drb - 0.5 ast
- 0.4 fg3m), because that is where the board's original defect ran.  A formula with five chosen weights in it
is a guess about what a player type is.  This fits the types instead (src/eracoef/archetype.py): a Bayesian
Gaussian mixture over the per-36 box profile, then the board-minus-consensus gap per cluster.

    python scripts/58_archetype.py [--board=outputs/season_ratings.parquet] [--k=8] [--seed=0]
                                   [--seasons=2024,2025,2026]
    python scripts/58_archetype.py stability [--boards=a.parquet,b.parquet] [--seeds=0-9] [--ks=6,8,10,12]
        refit over seeds, over k, and over each season on its own, and report how much the statistic moves

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
from eracoef.archetype import RATES, cluster_gaps, fit_clusters, gap_columns, per36, spread  # noqa: E402
from eracoef.boxtable import season_box  # noqa: E402
from eracoef.config import load_config  # noqa: E402

pd.set_option("display.width", 250, "display.max_columns", 40)
SUFFIX = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b")
MIN_POSS = 1000


def flag(name, default=None):
    hit = [a for a in sys.argv[1:] if a.startswith(f"--{name}=")]
    return hit[0].split("=", 1)[1] if hit else default


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    s = s.replace("&", " ").replace("-", " ").replace("'", "").replace(".", " ")
    return re.sub(r"\s+", " ", SUFFIX.sub(" ", s)).strip()


def joined(board_path: Path, cfg: dict, seasons: list) -> pd.DataFrame:
    """The board pooled over `seasons` (possession-weighted, the convention of test_vs_consensus), joined to
    the consensus by normalised name and to the per-36 box profile by player id."""
    root = Path(cfg["_root"])
    con = pd.read_csv(root / "data" / "external" / "consensus.csv")
    con = con.dropna(subset=["player_name", "adj_overall"]).assign(key=lambda d: d.player_name.map(norm))
    con = con.drop_duplicates("key")[["key", "adj_offense", "adj_defense", "adj_overall"]]

    board = pd.read_parquet(board_path)
    d = board[board.season.isin(seasons) & board.player_name.notna()].copy()
    w = d.poss_off.to_numpy(dtype=float)
    g = pd.DataFrame({c: (d[c] * w).groupby(d.player_id).sum() / d.groupby("player_id").poss_off.sum()
                      for c in ("rating_off", "rating_def", "rating_total")})
    g["poss"] = d.groupby("player_id").poss_off.sum()
    g["player_name"] = d.sort_values("season").drop_duplicates("player_id", keep="last").set_index("player_id").player_name
    g = g.reset_index()
    g = g[g.poss >= MIN_POSS].assign(key=lambda x: x.player_name.map(norm))
    g = g.sort_values("poss").drop_duplicates("key", keep="last").merge(con, on="key", how="inner")

    box = season_box(seasons, ["RS"], cfg)
    b = per36(box[box.phase == "RS"])
    return g.merge(b[["player_id", *RATES]], on="player_id", how="inner").reset_index(drop=True)


def table_for(m: pd.DataFrame, k: int, seed: int) -> pd.DataFrame:
    labels, weights = fit_clusters(m[RATES].to_numpy(dtype=float), k=k, seed=seed)
    gaps = gap_columns(m, m)
    t = cluster_gaps(labels, gaps, extra=m[["poss", *RATES]])
    t["weight"] = [weights[int(c)] for c in t.cluster]
    who = pd.DataFrame({"cluster": labels, "player_name": m.player_name, "poss": m.poss})
    t["who"] = [", ".join(who[who.cluster == c].sort_values("poss", ascending=False).player_name.head(3))
                for c in t.cluster]
    return t


def report(cfg, seasons):
    src = flag("board", "outputs/season_ratings.parquet")
    m = joined(Path(cfg["_root"]) / src, cfg, seasons)
    k, seed = int(flag("k", 8)), int(flag("seed", 0))
    t = table_for(m, k, seed)
    show = ["cluster", "n", "weight", "gap_off", "gap_def", "gap_total", "pts", "fg3m", "ast", "orb", "blk", "who"]
    print(f"\n{src}: {len(m)} players, {(t.weight > 0.01).sum()} of {k} components used, seed {seed}\n")
    print("gap = our z minus the consensus z; per-36 means beside it.  Read `gap_total` first.\n")
    print(t[show].round(2).to_string(index=False))
    print(f"\nspread of the per-cluster TOTAL gap: {spread(t):.3f} "
          f"(the number an archetype penalty would be asked to shrink)")


def stability(cfg, seasons):
    """How much does the statistic move when nothing about the board changes?

    Three sources of movement, because a floor has to survive all three: the mixture's own seed, the choice
    of `k` (an upper bound, but not an irrelevant one), and which seasons are pooled."""
    boards = (flag("boards") or "outputs/season_ratings.parquet,outputs/season_ratings_ks00.parquet").split(",")
    spec = flag("seeds", "0-9")
    lo, dash, hi = spec.partition("-")
    seeds = list(range(int(lo), int(hi) + 1)) if dash else [int(x) for x in spec.split(",")]
    ks = [int(x) for x in flag("ks", "6,8,10,12").split(",")]
    frames = {b: joined(Path(cfg["_root"]) / b, cfg, seasons) for b in boards}
    per_season = {(b, s): joined(Path(cfg["_root"]) / b, cfg, [s]) for b in boards for s in seasons}

    print(f"\nseeds {seeds[0]}-{seeds[-1]} at k=8, then k over {ks} at seed 0, then each season alone\n")
    rows = []
    for b, m in frames.items():
        by_seed = [spread(table_for(m, 8, s)) for s in seeds]
        by_k = [spread(table_for(m, k, 0)) for k in ks]
        by_season = [spread(table_for(per_season[(b, s)], 8, 0)) for s in seasons]
        rows.append(dict(board=Path(b).stem, seed_mean=np.mean(by_seed), seed_sd=np.std(by_seed, ddof=1),
                         seed_min=min(by_seed), seed_max=max(by_seed),
                         k_min=min(by_k), k_max=max(by_k), season_min=min(by_season), season_max=max(by_season)))
    R = pd.DataFrame(rows)
    print(R.round(3).to_string(index=False))
    if len(R) == 2:
        a, b = R.iloc[0], R.iloc[1]
        sep = a.seed_min > b.seed_max or b.seed_min > a.seed_max
        print(f"\nthe two boards are{'' if sep else ' NOT'} separated across every seed "
              f"({a.board} {a.seed_min:.3f}-{a.seed_max:.3f}, {b.board} {b.seed_min:.3f}-{b.seed_max:.3f})")


def main():
    cfg = load_config()
    seasons = [int(s) for s in flag("seasons", "2024,2025,2026").split(",")]
    if len(sys.argv) > 1 and sys.argv[1] == "stability":
        stability(cfg, seasons)
    else:
        report(cfg, seasons)


if __name__ == "__main__":
    main()
