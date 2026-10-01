"""Compare two or more rankings by the trade loss, paired by season, on the same players.

    python scripts/73_tradeloss.py --alphas=incumbent=outputs/tradeset_team_alpha.parquet,
                                           noonc_d=outputs/tradeset_noonc_d_alpha.parquet
                                   [--ref=incumbent] [--tier=all|each|<tier>] [--min_without=1]
                                   [--quality=outputs/season_ratings_unshrinkdef.parquet] [--movers=1]

Each `--alphas` entry is one `scripts/70_tradeset.py` run's `<out>_alpha.parquet`, which carries a
correction per player, season and side: what that player's comings and goings say his rating did not
already know.  The trade loss is the size of those corrections -- `MAE_SCALE * sqrt(mean(alpha^2))`,
points per 100, every player counting once.  Smaller means the rating already knew more.

**Why this exists.**  The year-over-year test is an error per team-game, so a 200-possession man is a
rounding error in it, and it called many of the candidates on disk ties.  This loss counts a bench
player the same as a starter.  It is an instrument and never a fitting target: alpha reads the seasons
either side of the rated one, so nothing scored here can become a published rating (ruling 1).

**The two things this script exists to get right.**

1. **The same players.**  Eligibility depends on the rankings covering the player, so two runs can
   differ in who has a row as well as in how well they are rated, and the two cannot be told apart in
   a pooled number.  Every comparison below is on the intersection of eligible player-seasons, and
   the count dropped from each arm is printed.
2. **Paired by season.**  30 seasons per side, `cand - ref`, reported with its own standard error.
   The level of any single arm is not meaningful on its own -- each run's alpha penalty is chosen by
   its own grid, and the corrections of neighbouring seasons share games -- so read the paired
   difference and the seasons won, exactly as `scripts/63_yoy.py` is read.

What it cannot tell you: alpha is a three-season average deviation from a one-season rating, so it
mixes "the rating missed something" with "he was different the next year".  That confound is the same
for both arms and cancels in the difference; it does not cancel in the level.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _cli import check_flags, flag  # noqa: E402
from eracoef.tradeset import MAE_SCALE, possession_tier  # noqa: E402


def missed(alpha: np.ndarray) -> float:
    """The trade loss on one group: the root mean square correction, on the mean-absolute scale."""
    return MAE_SCALE * float(np.sqrt(np.mean(np.square(alpha)))) if len(alpha) else np.nan


QUALITY_EDGES = [0, 30, 90, 150, 300, 10**9]
QUALITY_LABELS = ["top 30", "31-90", "91-150", "151-300", "301+"]


def quality_tiers(path: Path) -> pd.Series:
    """(player_id, season) -> quality tier: his rank by total rating in that season of the rankings at `path`.

    The owner's player-quality clusters (2026-09-30): top 30, 31-90, 91-150, 151-300, 301+.  One table fixes the
    tiers for every arm, so all of them are cut the same way; name the reference's rankings."""
    t = pd.read_parquet(path, columns=["player_id", "season", "rating_total"])
    rank = t.groupby("season").rating_total.rank(ascending=False, method="first").to_numpy()
    labels = np.asarray(QUALITY_LABELS, dtype=object)[np.digitize(rank, QUALITY_EDGES[1:-1], right=True)]
    return pd.Series(labels, index=pd.MultiIndex.from_arrays([t.player_id.to_numpy(), t.season.to_numpy()]))


MOVER_LABELS = ["changed teams", "same team", "no neighbour season"]


def mover_tiers(seasons) -> pd.Series:
    """(player_id, season) -> whether he changed teams around the rated season: "changed teams" if his main team
    (most minutes, as a franchise) in the season before or after differs from the rated season's, "same team" if
    every neighbouring season he played has the same main team, "no neighbour season" if he played neither.  The
    trade set's correction reads exactly those three seasons' games (the owner, 2026-10-01: split every result
    into traded and not traded)."""
    from eracoef import singleyear as sy
    from eracoef.config import load_config
    from eracoef.holdout import Context
    ctx = Context.load(load_config(ROOT / "config.yaml"))
    have = sorted(set(int(s) for s in seasons))
    span = sorted(set(have) | {s - 1 for s in have} | {s + 1 for s in have})
    teams = {}
    for s in span:
        try:
            m = ctx.main_team(s)
        except Exception:                       # a season outside the data
            m = {}
        ids = np.fromiter(m.keys(), dtype=np.int64, count=len(m))
        fr = sy.franchise(np.fromiter(m.values(), dtype=np.int64, count=len(m)), np.full(len(m), s))
        teams[s] = dict(zip(ids.tolist(), fr.tolist()))
    keys, labels = [], []
    for s in have:
        for p, t in teams[s].items():
            near = [teams[n][p] for n in (s - 1, s + 1) if p in teams.get(n, {})]
            keys.append((p, s))
            labels.append(MOVER_LABELS[2] if not near else MOVER_LABELS[0] if any(x != t for x in near)
                          else MOVER_LABELS[1])
    return pd.Series(labels, index=pd.MultiIndex.from_tuples(keys))


def load(path: Path, min_without: float, quality: pd.Series | None = None) -> pd.DataFrame:
    """One run's eligible corrections, tiered by the rated season's possessions -- or, given `quality`, by
    the player-quality clusters of `quality_tiers`."""
    table = pd.read_parquet(path)
    rows = table[(table.eligible) & (table.without_poss >= min_without)].copy()
    if quality is None:
        rows["tier"] = possession_tier(rows.poss_on.to_numpy())
    else:
        key = pd.MultiIndex.from_arrays([rows.player_id.to_numpy(), rows.season.to_numpy()])
        rows["tier"] = quality.reindex(key).fillna("no rating").to_numpy()
    return rows.set_index(["player_id", "season", "side"]).sort_index()


def paired(cand: pd.DataFrame, ref: pd.DataFrame, tier: str) -> pd.DataFrame:
    """Per side: both losses, the paired per-season difference, its z, and the seasons won."""
    out = []
    for side in ("offense", "defense"):
        c = cand.xs(side, level="side")
        r = ref.xs(side, level="side")
        if tier != "all":
            c, r = c[c.tier == tier], r[r.tier == tier]
        seasons = sorted(set(c.index.get_level_values("season")))
        diffs, n_players = [], 0
        for season in seasons:
            cs, rs = c.xs(season, level="season"), r.xs(season, level="season")
            both = cs.index.intersection(rs.index)
            if len(both) == 0:
                continue
            n_players += len(both)
            diffs.append((season,
                          missed(cs.loc[both].alpha_good.to_numpy()),
                          missed(rs.loc[both].alpha_good.to_numpy())))
        frame = pd.DataFrame(diffs, columns=["season", "cand", "ref"])
        d = (frame.cand - frame.ref).to_numpy()
        se = float(np.std(d, ddof=1) / np.sqrt(len(d))) if len(d) > 1 else np.nan
        out.append({"side": side, "seasons": len(frame), "players": n_players,
                    "cand": float(frame.cand.mean()), "ref": float(frame.ref.mean()),
                    "mean_diff": float(np.mean(d)), "se": se,
                    "z": float(np.mean(d) / se) if se and np.isfinite(se) and se > 0 else np.nan,
                    "wins": int((d < 0).sum())})
    return pd.DataFrame(out)


def main() -> None:
    check_flags()
    spec = flag("alphas", "")
    if not spec:
        raise SystemExit(__doc__)
    min_without = float(flag("min_without", "1"))
    tier = flag("tier", "all")          # a tier's name, "all", or "each" for every tier in turn
    quality_path = flag("quality")      # rankings whose per-season rank sets the quality tiers
    movers = flag("movers", "0") not in ("0", "no", "false")      # tiers = changed teams / same team instead
    assert not (movers and quality_path), "--movers=1 and --quality= each set the tiers; pick one"
    quality = quality_tiers(ROOT / quality_path) if quality_path else None
    if movers:
        quality = mover_tiers(range(1997, 2027))
    arms = {}
    for part in spec.replace("\n", "").split(","):
        if not part.strip():
            continue
        name, _, path = part.strip().partition("=")
        if not path:
            raise SystemExit(f"--alphas wants name=path pairs; got {part!r}")
        arms[name] = load(ROOT / path if not Path(path).is_absolute() else Path(path), min_without, quality)
    ref_name = flag("ref", next(iter(arms)))
    if ref_name not in arms:
        raise SystemExit(f"--ref={ref_name} is not one of {', '.join(arms)}")

    print(f"the trade loss, points per 100, every player counting once, tier {tier!r}, "
          f"corrections on at least {min_without:g} possession(s) of absence")
    for name, rows in arms.items():
        pen = sorted(rows.penalty.unique())
        print(f"  {name}: {len(rows):,} eligible player-season-sides, "
              f"{rows.index.get_level_values('season').nunique()} seasons, "
              f"alpha penalty {'/'.join(f'{p:,.0f}' for p in pen[:6])}"
              f"{' ...' if len(pen) > 6 else ''}")

    ref = arms[ref_name]
    for name, rows in arms.items():
        if name == ref_name:
            continue
        shared = rows.index.intersection(ref.index)
        print(f"\n=== {name} against {ref_name} ===")
        print(f"  {len(shared):,} player-season-sides in both; dropped {len(rows) - len(shared):,} "
              f"from {name} and {len(ref) - len(shared):,} from {ref_name}")
        order = (MOVER_LABELS + ["no rating"] if movers else QUALITY_LABELS + ["no rating"] if quality is not None
                 else sorted(set(rows.tier)))
        for t in ([x for x in order if x in set(rows.tier)] if tier == "each" else [tier]):
            table = paired(rows.loc[shared], ref.loc[shared], t)
            if tier == "each":
                print(f"  -- tier {t}")
            print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        print("  mean_diff below zero means the rating leaves a SMALLER correction, so it is better; "
              "z is that difference over its own standard error, wins is out of `seasons`")


if __name__ == "__main__":
    main()
