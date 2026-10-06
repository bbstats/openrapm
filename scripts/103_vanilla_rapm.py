"""B2 and B3, experiment 35's ridge baselines (src/eracoef/vanilla.py): vanilla RAPM, and RAPM on B1's prior.

    python scripts/103_vanilla_rapm.py [--tags=q1of4,q1of3,within2,q2of3,within,within10] [--stage=all|grid|select]
                                       [--prior=b1|openrapm]   (openrapm: B4, RAPM on OpenRAPM's own box prior, exp 36)

Stage `grid` (resumable, one file per fold size): every within-season fold and every season, each penalty pair of
the grid (offense 12 log-spaced values 500-200,000, defense/offense ratio 0.5/0.75/1/1.5/2), fit on actual points
and scored on the held-out games -- six stored numbers per pair give the error at any prior multiplier.

  within-season  the fold's rating games only (rows cut to them); scored on its held-out team-games with the
                 within-season stand-in (500 x 0.25); B3's prior is the fold's own B1 (asserted)
  year-over-year the whole season; scored on the seasons before and after, unrated players at 0 (63's rule)

Stage `select`: for each rated season H, the penalty pair (B2) and the pair plus multipliers (B3) that minimize
the pooled held-out error over blocks that touch no season in {H-1, H, H+1} -- separately per fold size and for the
year-over-year test -- then the chosen ratings, refit, written in the formats the scorecard and 63 read.  Also a
NON-nested B2 (the best pair over every season, H included), to size how much tuning on the test flatters.

Outputs: outputs/within/<tag>/baseline_{b2,b2g,b3}.parquet, outputs/season_ratings_base_{b2,b2g,b3}.parquet,
outputs/csv/baseline_penalties.csv, the grids in outputs/within/<tag>/ridge_grid.parquet and
outputs/csv/ridge_grid_yoy.parquet.
"""
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _cli import check_flags, flag  # noqa: E402
from eracoef import vanilla as va  # noqa: E402
from eracoef.config import load_config  # noqa: E402
from eracoef.holdout import Context  # noqa: E402

LAM_O = np.round(np.geomspace(500, 200000, 12), -1)
RATIOS = (0.5, 0.75, 1.0, 1.5, 2.0)
GRID = [(float(lo), float(r)) for lo in LAM_O for r in RATIOS]
FIRST, LAST = 1997, 2026


def _borrow(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


S104 = _borrow("_scorecard104", "104_scorecard.py")      # the fold and year-over-year loaders
S102 = _borrow("_boxspm102", "102_box_spm.py")           # `fold_fit_games`


def ridge_for(wd, keep_rows=None) -> va.Ridge:
    """The ridge of a design (optionally restricted to rows), with each player's possessions in those rows."""
    m = wd.spec.n_ps
    X, y, w, poss_rows = wd.X, wd.y, wd.w, wd.rows["poss"].to_numpy(float)
    if keep_rows is not None:
        X, y, w, poss_rows = X[keep_rows], y[keep_rows], w[keep_rows], poss_rows[keep_rows]
    X = X.tocsr()
    poss = np.asarray(X[:, :m].T @ poss_rows).ravel()
    # only the named fixed effects: the cached design's last column is the game index (experiment 37, stage 0)
    ridge = va.Ridge(X, y, w, m, wd.spec.ps_table["player_id"].to_numpy(np.int64), poss,
                     n_fixed=len(wd.spec.f_names))
    assert ridge.c == len(wd.spec.f_names), "the ridge's unpenalized block must be exactly the design's fixed effects"
    return ridge


def aligned_prior(ids: np.ndarray, table: pd.DataFrame) -> tuple:
    """A prior's raw-sign (o, d) on the design's player columns; 0 where the prior has no row."""
    t = table.drop_duplicates("player_id").set_index("player_id")
    s = pd.Series(ids)
    return s.map(t.o).fillna(0.0).to_numpy(float), s.map(t.d).fillna(0.0).to_numpy(float)


# The prior B3-style RAPM is centered on.  `b1` (experiment 35): the linear box-score SPM.  `openrapm` (experiment 36,
# the owner 2026-10-05): OpenRAPM's own boosted box prior -- the booster's raw prediction before its free scale
# (`prior_raw_*`), from the same saved models and the same rating-game inputs as the fold's OpenRAPM rating, so the two
# priors get identical ridge tuning and the comparison is the prior alone.
PRIORS = {"b1": dict(name="b3", grid="ridge_grid.parquet", grid_yoy="ridge_grid_yoy.parquet",
                     penalties="baseline_penalties.csv", with_b2=True),
          "openrapm": dict(name="b4", grid="ridge_grid_openrapm.parquet", grid_yoy="ridge_grid_yoy_openrapm.parquet",
                           penalties="baseline_penalties_b4.csv", with_b2=False)}


def openrapm_prior(players: pd.DataFrame) -> pd.DataFrame:
    """OpenRAPM's raw box prior for the rated players (raw sign), centered on their possessions; a player the prior
    has no value for is left out (prior 0)."""
    t = players.drop_duplicates("player_id")
    t = t[(t.poss_off > 0) & t.prior_raw_off.notna() & t.prior_raw_def.notna()]
    w = t.poss_off.to_numpy(float)
    o, d = t.prior_raw_off.to_numpy(float), -t.prior_raw_def.to_numpy(float)
    return pd.DataFrame({"player_id": t.player_id.astype(np.int64).to_numpy(),
                         "o": o - np.average(o, weights=w), "d": d - np.average(d, weights=w)})


def fold_prior(kind: str, directory: Path, stem: str, b1) -> pd.DataFrame:
    if kind == "b1":
        prior = b1[b1.key == stem]
        assert len(prior), f"{directory.name} {stem}: no fold-own B1 prior"
        return prior
    return openrapm_prior(pd.read_parquet(directory / f"players_{stem}.parquet"))


def season_prior(kind: str, S: int, b1) -> pd.DataFrame:
    if kind == "b1":
        return b1[b1.season == S]
    # the whole-season run of the same code at the test settings (97's reproduction, models never saw S-1..S+1)
    return openrapm_prior(pd.read_parquet(ROOT / "outputs" / "within" / "within" / f"players_{S}_all.parquet"))


def grid_within(ctx, tag: str, kind: str = "b1") -> pd.DataFrame:
    directory = ROOT / "outputs" / "within" / tag
    out_path = directory / PRIORS[kind]["grid"]
    if out_path.exists():
        return pd.read_parquet(out_path)
    b1 = pd.read_parquet(directory / "baseline_b1.parquet") if kind == "b1" else None
    rows, t0 = [], time.time()
    for H in range(FIRST, LAST + 1):
        if not (directory / f"folds_{H}.parquet").exists():
            continue
        wd = ctx.design([H], "pts")
        game_of = wd.games.drop_duplicates("game_idx").set_index("game_id").game_idx
        game_of.index = game_of.index.astype(str)
        assignment = pd.read_parquet(directory / f"folds_{H}.parquet")
        assignment.index = assignment.index.astype(str)
        row_game = wd.rows["game_idx"].to_numpy()
        for stem in sorted(p.stem[len("fold_"):] for p in directory.glob(f"fold_{H}_*.json")):
            fit_ids, mode, season = S102.fold_fit_games(directory, stem, assignment)
            fit_idx = game_of.reindex(sorted(fit_ids)).dropna().to_numpy()
            keep = np.isin(row_game, fit_idx)
            fold = S104.load_fold(directory, stem)
            test_ids = set(map(str, np.load(directory / f"test_{stem}.npz", allow_pickle=True)["game_id"]))
            assert not (test_ids & fit_ids), f"{tag} {stem}: a held-out game among the rating games"
            ridge = ridge_for(wd, keep)
            p_o, p_d = aligned_prior(ridge.player_ids, fold_prior(kind, directory, stem, b1))
            for lam_o, ratio in GRID:
                comp = ridge.components(lam_o, lam_o * ratio, p_o, p_d)
                rows.append(dict(tag=tag, key=stem, season=H, deal=fold["block"].deal, lam_o=lam_o, ratio=ratio,
                                 **va.quadratic(fold["block"], comp, fill=True)))
        print(f"  {tag} {H}: grid done ({time.time() - t0:.0f}s)", flush=True)
    grid = pd.DataFrame(rows)
    grid.to_parquet(out_path, index=False)
    return grid


def grid_yoy(ctx, kind: str = "b1") -> pd.DataFrame:
    out_path = ROOT / "outputs" / "csv" / PRIORS[kind]["grid_yoy"]
    if out_path.exists():
        return pd.read_parquet(out_path)
    b1 = pd.read_parquet(ROOT / "outputs" / "season_ratings_base_b1.parquet")
    b1 = b1.assign(o=b1.rating_off, d=-b1.rating_def)
    blocks = {T: S104.yoy_block(ctx, T) for T in range(FIRST + 1, LAST)}
    rows, t0 = [], time.time()
    for S in range(FIRST, LAST + 1):
        ridge = ridge_for(ctx.design([S], "pts"))
        p_o, p_d = aligned_prior(ridge.player_ids, season_prior(kind, S, b1))
        for lam_o, ratio in GRID:
            comp = ridge.components(lam_o, lam_o * ratio, p_o, p_d)
            for T, direction in ((S + 1, "prev"), (S - 1, "next")):
                if T in blocks:
                    rows.append(dict(rated=S, season=T, direction=direction, lam_o=lam_o, ratio=ratio,
                                     **va.quadratic(blocks[T], comp, fill=False)))
        print(f"  year-over-year: season {S} rated ({time.time() - t0:.0f}s)", flush=True)
    grid = pd.DataFrame(rows)
    grid.to_parquet(out_path, index=False)
    return grid


def choose(train: pd.DataFrame) -> dict:
    """B2: the pair with the lowest pooled error at m = 0; B3: the pair and multipliers with the lowest error."""
    best2, best3 = None, None
    for (lam_o, ratio), g in train.groupby(["lam_o", "ratio"]):
        e2 = g.g00.sum() / g.sw.sum()
        if best2 is None or e2 < best2[0]:
            best2 = (e2, lam_o, ratio)
        m_o, m_d = va.best_multipliers(g)
        e3 = (g.g00.sum() - 2 * (m_o * g.g0o.sum() + m_d * g.g0d.sum()) + m_o ** 2 * g.goo.sum()
              + 2 * m_o * m_d * g.god.sum() + m_d ** 2 * g.gdd.sum()) / g.sw.sum()
        if best3 is None or e3 < best3[0]:
            best3 = (e3, lam_o, ratio, m_o, m_d)
    return dict(b2_lam_o=best2[1], b2_ratio=best2[2], b3_lam_o=best3[1], b3_ratio=best3[2],
                b3_m_o=best3[3], b3_m_d=best3[4])


def select(grid: pd.DataFrame, rated_col: str, touch) -> pd.DataFrame:
    """Per rated season H: nested choices from blocks that touch no season in {H-1, H, H+1}; plus the non-nested B2."""
    rows = []
    every = choose(grid)
    for H in sorted(grid[rated_col].unique()):
        near = {H - 1, H, H + 1}
        train = grid[~touch(grid, near)]
        rows.append(dict(season=int(H), **choose(train), b2g_lam_o=every["b2_lam_o"], b2g_ratio=every["b2_ratio"],
                         blocks=int(train.groupby(["lam_o", "ratio"]).size().iloc[0])))
    return pd.DataFrame(rows)


def materialize_within(ctx, tag: str, chosen: pd.DataFrame, kind: str = "b1") -> None:
    directory = ROOT / "outputs" / "within" / tag
    b1 = pd.read_parquet(directory / "baseline_b1.parquet") if kind == "b1" else None
    pname = PRIORS[kind]["name"]
    out = {"b2": [], "b2g": [], pname: []} if PRIORS[kind]["with_b2"] else {pname: []}
    c = chosen.set_index("season")
    for H in sorted(c.index):
        if not (directory / f"folds_{H}.parquet").exists():
            continue
        wd = ctx.design([H], "pts")
        game_of = wd.games.drop_duplicates("game_idx").set_index("game_id").game_idx
        game_of.index = game_of.index.astype(str)
        assignment = pd.read_parquet(directory / f"folds_{H}.parquet")
        assignment.index = assignment.index.astype(str)
        row_game = wd.rows["game_idx"].to_numpy()
        for stem in sorted(p.stem[len("fold_"):] for p in directory.glob(f"fold_{H}_*.json")):
            fit_ids, _, _ = S102.fold_fit_games(directory, stem, assignment)
            keep = np.isin(row_game, game_of.reindex(sorted(fit_ids)).dropna().to_numpy())
            ridge = ridge_for(wd, keep)
            p_o, p_d = aligned_prior(ridge.player_ids, fold_prior(kind, directory, stem, b1))
            r = c.loc[H]
            arms = [("b2", r.b2_lam_o, r.b2_ratio, 0.0, 0.0), ("b2g", r.b2g_lam_o, r.b2g_ratio, 0.0, 0.0),
                    (pname, r.b3_lam_o, r.b3_ratio, r.b3_m_o, r.b3_m_d)]
            for name, lam_o, ratio, m_o, m_d in [a for a in arms if a[0] in out]:
                comp = ridge.components(lam_o, lam_o * ratio, p_o, p_d)
                t = comp.table(m_o, m_d).assign(key=stem, season=H)
                if name == pname:
                    # the prior part: m x B1 on the rated players, centered like the rating (the games part is the rest)
                    pid = pd.Series(comp.player_id)
                    po = pid.map(pd.Series(p_o, index=ridge.player_ids)).to_numpy(float) * m_o
                    pdd = pid.map(pd.Series(p_d, index=ridge.player_ids)).to_numpy(float) * m_d
                    t["prior_o"] = po - np.average(po, weights=comp.poss)
                    t["prior_d"] = pdd - np.average(pdd, weights=comp.poss)
                out[name].append(t)
    for name, parts in out.items():
        pd.concat(parts, ignore_index=True).to_parquet(directory / f"baseline_{name}.parquet", index=False)


def materialize_yoy(ctx, chosen: pd.DataFrame, kind: str = "b1") -> None:
    b1 = pd.read_parquet(ROOT / "outputs" / "season_ratings_base_b1.parquet")
    names = b1.drop_duplicates("player_id").set_index("player_id").player_name
    b1 = b1.assign(o=b1.rating_off, d=-b1.rating_def)
    pname = PRIORS[kind]["name"]
    out = {"b2": [], "b2g": [], pname: []} if PRIORS[kind]["with_b2"] else {pname: []}
    c = chosen.set_index("season")
    for S in range(FIRST, LAST + 1):
        ridge = ridge_for(ctx.design([S], "pts"))
        p_o, p_d = aligned_prior(ridge.player_ids, season_prior(kind, S, b1))
        r = c.loc[S] if S in c.index else c.iloc[(c.index.to_series() - S).abs().argmin()]
        arms = [("b2", r.b2_lam_o, r.b2_ratio, 0.0, 0.0), ("b2g", r.b2g_lam_o, r.b2g_ratio, 0.0, 0.0),
                (pname, r.b3_lam_o, r.b3_ratio, r.b3_m_o, r.b3_m_d)]
        for name, lam_o, ratio, m_o, m_d in [a for a in arms if a[0] in out]:
            t = ridge.components(lam_o, lam_o * ratio, p_o, p_d).table(m_o, m_d)
            out[name].append(pd.DataFrame({"player_id": t.player_id, "season": S, "rating_off": t.o,
                                           "rating_def": -t.d, "poss_off": t.poss, "poss_def": t.poss}))
    for name, parts in out.items():
        table = pd.concat(parts, ignore_index=True)
        table["player_name"] = table.player_id.map(names)
        table["rating_total"] = table.rating_off + table.rating_def
        table.to_parquet(ROOT / "outputs" / f"season_ratings_base_{name}.parquet", index=False)


def main():
    check_flags()
    tags = [t for t in flag("tags", "q1of4,q1of3,within2,q2of3,within,within10").split(",") if t]
    stage = flag("stage", "all")
    kind = flag("prior", "b1")
    assert kind in PRIORS, f"--prior={kind}: one of {sorted(PRIORS)}"
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(8, "blas")
    except ImportError:
        pass
    ctx = Context.load(load_config(ROOT / "config.yaml"))
    grids = {tag: grid_within(ctx, tag, kind) for tag in tags}
    gy = grid_yoy(ctx, kind)
    if stage == "grid":
        return
    penalties = []
    for tag, grid in grids.items():
        chosen = select(grid, "season", lambda g, near: g.season.isin(near))
        penalties.append(chosen.assign(test=tag))
        materialize_within(ctx, tag, chosen, kind)
        print(f"  {tag}: chosen and written", flush=True)
    chosen = select(gy, "rated", lambda g, near: g.rated.isin(near) | g.season.isin(near))
    penalties.append(chosen.assign(test="yoy"))
    materialize_yoy(ctx, chosen, kind)
    out = ROOT / "outputs" / "csv" / PRIORS[kind]["penalties"]
    new = pd.concat(penalties, ignore_index=True)
    if out.exists():                     # a run on some fold sizes keeps the other sizes' rows
        old = pd.read_csv(out)
        new = pd.concat([old[~old.test.isin(new.test.unique())], new], ignore_index=True)
    new.to_csv(out, index=False)
    print(f"wrote the {PRIORS[kind]['name'].upper()} tables and {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
