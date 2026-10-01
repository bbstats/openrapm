"""Where a year-over-year difference comes from: the error difference between two tables of player rankings, split
among the players on the floor and added up by player group.

    python scripts/88_yoy_by_player.py --ref=incumbent=outputs/season_ratings_unshrinkdef.parquet
                                       --cands=team_intercept=outputs/season_ratings_team_intercept.parquet[,name=path]
                                       [,name=prev_path|next_path]   (one file per direction, scripts/89_stitch_by_move.py)
                                       [--first=1998] [--last=2025] [--tag=<name>]

The year-over-year test (scripts/63_yoy.py) scores a team-game, one team's points in one game, predicted from the
ten players' ratings of a neighbouring season.  Its splits label a stint or a team-game, so they cannot say which
PLAYERS a difference came from.  This script can, exactly:

    a team-game's squared-error difference (candidate minus reference) is shared among the players on the floor in
    proportion to their possessions in it -- each of the ten on court holds a tenth of a row's possessions -- and a
    player's contribution is the sum of his shares.

The contributions add up, season by season, to the paired team-game difference 63_yoy.py reports, so a group's
total is its part of that number and the groups recombine exactly (checked every season).  The same is done at stint
level.  Each group is read the way the test is read: per scored season and direction, the mean over seasons, its
standard error, z, and the seasons in which the group made the candidate better.  Below zero = the candidate is
better in that group.

The groups, fixed by the REFERENCE table or by facts, so every candidate is cut the same way:
  quality   the reference's rank by total rating in the RATED season (the season whose ratings are used):
            top 30, 31-90, 91-150, 151-300, 301+, and "no rating" (not in that season's table, so rated 0 by both)
  movers    "changed teams" if his main team (most minutes) in the scored season differs from the rated season's,
            "same team" if not, "not in both" if he has no main team in one of them
  age       his age in the scored season (the season panel's `age`): 23 and under, 24-26, 27-29, 30-32, 33+

`--splits=quality,movers,age` picks them.  Writes outputs/yoy_by_player_<tag>.parquet, one row per scored season,
direction, candidate, split and group.  Its `tg_abs` column shares the candidate's own team-game squared error out
the same way, so a group's `tg_abs` over its `poss_share` is the mean squared miss of the team-games it played in
(times sqrt(2/pi) after the root, the test's own scale): each group's accuracy, not only its part of a difference.
"""
import os
import sys
import time
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.sparse as sp  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _cli import check_flags, flag  # noqa: E402
from eracoef.config import load_config  # noqa: E402
from eracoef.holdout import Context, Ratings, predict_season  # noqa: E402
from eracoef.seasons import drop_untrainable  # noqa: E402

QUALITY_EDGES = [0, 30, 90, 150, 300, 10**9]
QUALITY_LABELS = ["top 30", "31-90", "91-150", "151-300", "301+"]
AGE_EDGES = [0, 24, 27, 30, 33, 200]
AGE_LABELS = ["23 and under", "24-26", "27-29", "30-32", "33+"]
DIRECTIONS = {"prev": -1, "next": +1}          # the rated season relative to the scored one


def load_table(path: Path) -> pd.DataFrame:
    """player_id, season, o, d (raw sign: d is points allowed) and poss, plus the total for ranking."""
    t = pd.read_parquet(path)
    return pd.DataFrame({"player_id": t.player_id.astype(np.int64), "season": t.season.astype(int),
                         "o": t.rating_off.astype(float), "d": -t.rating_def.astype(float),
                         "poss": t.poss_off.astype(float), "total": t.rating_total.astype(float)})


def load_by_direction(path: str) -> dict:
    """`a.parquet` for both directions, or `a.parquet|b.parquet`: the first for "prev" (its rankings look forward at
    the scored season), the second for "next".  The trade-flag test's stitched list depends on which neighbour it is
    scored on (scripts/89_stitch_by_move.py)."""
    paths = path.split("|")
    assert len(paths) in (1, 2), f"--cands {path}: one path, or two joined by |"
    return {"prev": load_table(ROOT / paths[0]), "next": load_table(ROOT / paths[-1])}


def team_game_errors(p, gid: np.ndarray, n_games: int) -> np.ndarray:
    """Per team-game: (actual - predicted points) / possessions x 100, the unit 63_yoy.py scores."""
    act = np.bincount(gid, weights=p.y * p.poss / 100.0, minlength=n_games)
    prd = np.bincount(gid, weights=p.pred * p.poss / 100.0, minlength=n_games)
    poss = np.bincount(gid, weights=p.poss, minlength=n_games)
    return (act - prd) / np.where(poss > 0, poss, 1.0) * 100.0


def quality_groups(ref: pd.DataFrame, season: int, ids: np.ndarray) -> np.ndarray:
    t = ref[ref.season == season]
    rank = t.total.rank(ascending=False, method="first").astype(int)
    by_id = pd.Series(rank.to_numpy(), index=t.player_id.to_numpy())
    r = by_id.reindex(ids).to_numpy()
    out = np.full(len(ids), "no rating", dtype=object)
    ok = ~np.isnan(r)
    out[ok] = np.asarray(QUALITY_LABELS, dtype=object)[np.digitize(r[ok], QUALITY_EDGES[1:-1], right=True)]
    return out


def mover_groups(ctx: Context, scored: int, rated: int, ids: np.ndarray) -> np.ndarray:
    now, then = ctx.main_team(scored), ctx.main_team(rated)
    return np.array(["not in both" if (q not in now or q not in then) else
                     ("changed teams" if now[q] != then[q] else "same team") for q in map(int, ids)], dtype=object)


def age_groups(ages: pd.Series, ids: np.ndarray) -> np.ndarray:
    a = ages.reindex(ids).to_numpy(dtype=float)
    out = np.full(len(ids), "unknown", dtype=object)
    ok = ~np.isnan(a)
    out[ok] = np.asarray(AGE_LABELS, dtype=object)[np.digitize(a[ok], AGE_EDGES[1:-1])]
    return out


def summarise(frame: pd.DataFrame, value: str) -> pd.DataFrame:
    """Per candidate, split, group: mean over observations, se, z and wins, per direction and both together."""
    out = []
    for (cand, split, group), g in frame.groupby(["candidate", "split", "group"], sort=False):
        row = dict(candidate=cand, split=split, group=group, poss_share=float(g.poss_share.mean()))
        for label, part in (("next", g[g.direction == "next"]), ("prev", g[g.direction == "prev"]), ("both", g)):
            v = part[value].to_numpy(float)
            se = float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else np.nan
            row.update({f"{label}_mean": float(v.mean()), f"{label}_z": float(v.mean() / se) if se else np.nan,
                        f"{label}_wins": int((v < 0).sum()), f"{label}_n": len(v)})
        out.append(row)
    return pd.DataFrame(out)


def main() -> None:
    check_flags()
    cfg = load_config(ROOT / "config.yaml")
    ref_name, ref_path = flag("ref", "incumbent=outputs/season_ratings_unshrinkdef.parquet").split("=", 1)
    cands = [part.split("=", 1) for part in flag("cands", "").split(",") if part]
    if not cands:
        raise SystemExit("--cands=name=path[,name=path...] is required")
    splits = [s for s in flag("splits", "quality,movers,age").split(",") if s]
    first, last = int(flag("first", cfg["holdout"]["first"])), int(flag("last", cfg["holdout"]["last"]))
    tag = flag("tag", "_".join(n for n, _ in cands))
    ref = load_table(ROOT / ref_path)
    tables = {name: load_by_direction(path) for name, path in cands}
    # ages only, but read through the season gate like every other reader of the panel
    panel, _ = drop_untrainable(pd.read_parquet(ROOT / "outputs/role_panel_season.parquet",
                                                columns=["window", "player_id", "season", "side", "age"]),
                                cfg, what="the season panel")
    panel = panel[panel.side == "O"]
    ctx = Context.load(cfg)
    rows, worst_gap, started = [], 0.0, time.time()
    for scored in range(first, last + 1):
        wd = ctx.design([scored], "pts")
        m = wd.spec.n_ps
        Z = sp.csr_matrix(wd.X[:, :2 * m])
        on_court = (Z[:, :m] + Z[:, m:]).T.tocsr()                 # player x row: on the floor, either end
        ids = wd.spec.ps_table["player_id"].to_numpy(np.int64)
        gkey = wd.rows.game_idx.to_numpy() * 2 + wd.rows.is_home_off.to_numpy().astype(int)
        gid, gvals = pd.factorize(gkey)
        n_games = len(gvals)
        ages = panel[panel.season == scored].set_index("player_id").age
        for direction, step in DIRECTIONS.items():
            rated = scored + step
            if not (ref.season == rated).any():
                continue
            p_ref = predict_season(Ratings(ref[ref.season == rated][["player_id", "o", "d", "poss"]]
                                           .reset_index(drop=True)), wd)
            poss_row = p_ref.poss
            poss_g = np.bincount(gid, weights=poss_row, minlength=n_games)
            share = np.asarray(on_court @ (poss_row / 10.0)).ravel() / poss_row.sum()
            e_ref = team_game_errors(p_ref, gid, n_games)
            labels = {"quality": quality_groups(ref, rated, ids),
                      "movers": mover_groups(ctx, scored, rated, ids),
                      "age": age_groups(ages, ids)}
            for name, by_direction in tables.items():
                table = by_direction[direction]
                if not (table.season == rated).any():
                    continue
                p = predict_season(Ratings(table[table.season == rated][["player_id", "o", "d", "poss"]]
                                           .reset_index(drop=True)), wd)
                e = team_game_errors(p, gid, n_games)
                delta_g = e ** 2 - e_ref ** 2                        # per team-game
                total_tg = float(np.sum(poss_g * delta_g) / poss_g.sum())
                per_row = poss_row * delta_g[gid] / (10.0 * poss_g.sum())
                contrib_tg = np.asarray(on_court @ per_row).ravel()
                w = p.w
                per_row_st = w * ((p.y - p.pred) ** 2 - (p.y - p_ref.pred) ** 2) / (10.0 * w.sum())
                contrib_st = np.asarray(on_court @ per_row_st).ravel()
                total_st = float(np.sum(w * ((p.y - p.pred) ** 2 - (p.y - p_ref.pred) ** 2)) / w.sum())
                worst_gap = max(worst_gap, abs(contrib_tg.sum() - total_tg), abs(contrib_st.sum() - total_st))
                # the candidate's own team-game squared error, shared out the same way: a group's `tg_abs` over its
                # `poss_share` is the mean squared miss of the team-games it played in, so groups compare directly
                total_abs = float(np.sum(poss_g * e ** 2) / poss_g.sum())
                contrib_abs = np.asarray(on_court @ (poss_row * (e ** 2)[gid] / (10.0 * poss_g.sum()))).ravel()
                rows.append(dict(season=scored, direction=direction, candidate=name, split="all", group="all",
                                 tg=total_tg, stint=total_st, poss_share=1.0, tg_abs=total_abs))
                for split in splits:
                    lab = labels[split]
                    for g in pd.unique(lab):
                        k = lab == g
                        rows.append(dict(season=scored, direction=direction, candidate=name, split=split,
                                         group=str(g), tg=float(contrib_tg[k].sum()), stint=float(contrib_st[k].sum()),
                                         poss_share=float(share[k].sum()), tg_abs=float(contrib_abs[k].sum())))
        print(f"  {scored} ({time.time() - started:.0f}s)", flush=True)
    frame = pd.DataFrame(rows)
    out = ROOT / "outputs" / f"yoy_by_player_{tag}.parquet"
    frame.to_parquet(out, index=False)
    print(f"\nthe groups add up to the whole in every season and direction: largest gap {worst_gap:.1e}")
    order = {"quality": QUALITY_LABELS + ["no rating"], "movers": ["same team", "changed teams", "not in both"],
             "age": AGE_LABELS + ["unknown"], "all": ["all"]}
    pd.set_option("display.width", 220, "display.max_columns", 30)
    for value, what in (("tg", "team-game level (what the test decides on)"), ("stint", "stint level")):
        s = summarise(frame, value)
        print(f"\n=== {what}: each group's part of the candidate-minus-{ref_name} difference in squared error; "
              f"below zero = the candidate is better there")
        for cand in s.candidate.unique():
            for split in ["all"] + splits:
                part = s[(s.candidate == cand) & (s.split == split)].copy()
                part["o"] = part.group.map({g: i for i, g in enumerate(order[split])})
                part = part.sort_values("o")
                show = part[["group", "poss_share", "next_mean", "next_z", "next_wins", "prev_mean", "prev_z",
                             "prev_wins", "both_mean", "both_z", "both_wins"]]
                print(f"\n--- {cand}, split by {split} (next/prev: {int(part.next_n.iloc[0])} scored seasons each; "
                      f"both: {int(part.both_n.iloc[0])})")
                print(show.round(4).to_string(index=False))
    print(f"\nwrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
