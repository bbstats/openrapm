"""Experiment 37, stage 1: build the held-out quadratics of a box-score prior fit directly on held-out games.

    python scripts/107_heldout_prior_build.py [--tags=q1of4,q1of3,within2,q2of3,within,within10] [--yoy=1]

For every within-season fold and every season (src/eracoef/heldoutprior.py): the ridge of the fold's rating games
(actual points, the design's named fixed effects only), each of ~65 inputs per side as a prior direction, and the
held-out team-game error as an exact quadratic in the input weights -- at 16 penalty pairs, with the level refit as
the scoring code does ("home") and with every team's level removed ("team", the within-team-split guardrail).
Pooled per season.  Year over year: each season's full ratings scored on the seasons before and after, unrated
players at 0 (63's rule).

Inputs (all from the rating games only): the fold's rebuilt panel (offense inputs from the offensive row, defense from
the defensive row), OpenRAPM's raw boosted prior, B1's prediction (0 where B1 has no row, 103's rule, so the anchor
reproduces B3 exactly) and B1's own 14 inputs (102's `player_inputs` on the rating games' box score, padding constants
from training seasons).  Every input is divided by a fixed scale (its standard deviation over the development
seasons' whole-season tables, lockbox seasons excluded) -- conditioning only, the fits are invariant to it.

Checks before anything is stored for a season: the test games are not among the rating games; B1's inputs read only
rating-game box rows; and on the season's first fold, weights 0 reproduce the B2 grid's stored error and the anchor
alone reproduces the B3 grid's six numbers (103's `ridge_grid.parquet`, same penalty pair), to 1e-8.

Writes outputs/heldout/quad_<tag>.npz (G: seasons x pairs x levels x (1+K) x (1+K), sw), quad_yoy.npz, inputs.json
(names and scales).  Resumable per tag.
"""
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _cli import check_flags, flag  # noqa: E402
from eracoef import boxspm as bx  # noqa: E402
from eracoef import heldoutprior as hp  # noqa: E402
from eracoef import scorecard as sc  # noqa: E402
from eracoef import tradeset as ts  # noqa: E402
from eracoef.boxtable import season_box  # noqa: E402
from eracoef.config import load_config  # noqa: E402
from eracoef.exposure import K_MAX  # noqa: E402
from eracoef.holdout import Context  # noqa: E402

LAMS = (2560.0, 4420.0, 7620.0, 13130.0, 22640.0, 39030.0, 67290.0)      # widened twice: the first build chose its low edge, the second its high edge (all inputs minus plus-minus groups)
RATIOS = (0.5, 0.75, 1.0, 1.5)
GRID = [(lam, r) for lam in LAMS for r in RATIOS]
LEVELS = ("home", "team")
FIRST, LAST = 1997, 2026
OUT = ROOT / "outputs" / "heldout"
PANEL_GROUPS = ["rates", "efficiency", "shot_location", "role", "body_career", "score_context", "on_court",
                "off_court", "same_games_rapm"]


def _borrow(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


S103 = _borrow("_vanilla103_for_107", "103_vanilla_rapm.py")
S104 = S103.S104
S102 = S103.S102
B70 = _borrow("_tradeset70_for_107", "70_tradeset.py")
COLUMNS = hp.columns_of(list(hp.GROUPS))           # the same list on both sides


def side_table(players: pd.DataFrame, side: str, b1: pd.DataFrame, bxin: pd.DataFrame) -> pd.DataFrame:
    """player_id -> every input for one side (NaN where missing; b1 left NaN here and set to 0 by the caller)."""
    rows = players[players.side == side].drop_duplicates("player_id").set_index("player_id")
    t = rows[hp.columns_of(PANEL_GROUPS)].astype(float).copy()
    t["booster"] = rows.prior_raw_off.astype(float) if side == "O" else -rows.prior_raw_def.astype(float)
    b = b1.drop_duplicates("player_id").set_index("player_id")
    t = t.reindex(t.index.union(b.index))
    t["b1"] = b.o if side == "O" else b.d
    x = bxin.set_index("player_id")
    for c in bx.INPUTS:
        t["bx_" + c[2:] if c.startswith("r_") else "bx_" + c] = x[c]
    return t[COLUMNS]


def design_inputs(ridge, tables: dict, scales: dict) -> tuple:
    """(F_o, F_d): the inputs on the ridge's player columns, imputed and scaled; B1's prediction 0 where absent."""
    out = []
    for side in ("O", "D"):
        F = tables[side].reindex(ridge.player_ids)
        b1 = F["b1"].fillna(0.0).to_numpy()
        F = hp.impute(F.to_numpy(float), ridge.poss, ridge.rated)
        F[:, COLUMNS.index("b1")] = b1
        out.append(F / np.array([scales[side][c] for c in COLUMNS]))
    return out[0], out[1]


def box_inputs(box: pd.DataFrame, k: np.ndarray, games=None) -> pd.DataFrame:
    sub = box if games is None else box[box.game_id.isin(games)]
    if games is not None:
        assert set(sub.game_id.unique()) <= set(games), "box inputs read a game outside the rating games"
    return bx.player_inputs(sub, k)


def accumulate(store, sidx, G_by, sw, sw_level=None):
    for (g, level), G in G_by.items():
        store["G"][sidx, g, LEVELS.index(level)] += G
    store["sw"][sidx] += sw
    for lv, v in (sw_level or {}).items():
        store["sw_level"][sidx, LEVELS.index(lv)] += v


def score_weights_for(block, moved) -> dict:
    """level -> row weights: "moved" / "stayed" re-weight the rows by the share of players on a new team (exp 39)."""
    out = {}
    for level in LEVELS:
        if level == "moved":
            out[level] = block.w * moved
        elif level == "stayed":
            out[level] = block.w * (1.0 - moved)
        else:
            out[level] = block.w
    return out


def grams_for(ridge, F_o, F_d, block, teams, fill: bool, moved=None, traded=None) -> dict:
    """(pair, level) -> gram; with `traded` (the block of traded players' entries, scorecard.split_traded) instead
    (pair, "split") -> the split gram, the inputs' contributions from players who stayed and who were traded apart."""
    out = {}
    weights = score_weights_for(block, moved) if moved is not None else {lv: block.w for lv in LEVELS}
    for g, (lam, ratio) in enumerate(GRID):
        ids, w, V_o, V_d = hp.responses(ridge, lam, lam * ratio, F_o, F_d)
        C = hp.contributions(block, ids, w, V_o, V_d, fill=fill)
        if traded is not None:
            out[(g, "split")] = hp.split_gram(block, C, hp.contributions(traded, ids, w, V_o, V_d, fill=fill))
            continue
        for level in LEVELS:
            if level in ("moved", "stayed"):
                out[(g, level)] = hp.gram(block, C, "home", teams, score_weights=weights[level])
            else:
                out[(g, level)] = hp.gram(block, C, level, teams)
    return out


def check_against_grid(G_by, sw, grid_rows, scales, tol=1e-8):
    """weights 0 = B2's stored error; the anchor alone = B3's six numbers (scaled back)."""
    jo, jd = 1 + COLUMNS.index("b1"), 1 + len(COLUMNS) + COLUMNS.index("b1")
    so, sd = scales["O"]["b1"], scales["D"]["b1"]
    for g, (lam, ratio) in enumerate(GRID):
        r = grid_rows[(grid_rows.lam_o == lam) & (np.isclose(grid_rows.ratio, ratio))]
        if not len(r):
            continue
        r = r.iloc[0]
        G = G_by[(g, "home")]
        got = np.array([G[0, 0], G[0, jo] * so, G[0, jd] * sd, G[jo, jo] * so * so, G[jo, jd] * so * sd,
                        G[jd, jd] * sd * sd])
        want = r[["g00", "g0o", "g0d", "goo", "god", "gdd"]].to_numpy(float)
        assert np.allclose(got, want, rtol=1e-8, atol=tol * max(1.0, abs(want[0]))), \
            f"the anchor does not reproduce B3's grid at {lam}/{ratio}: {got} vs {want}"
        assert abs(sw - r.sw) < 1e-6


def main():
    check_flags()
    global LEVELS
    tags = [t for t in flag("tags", "q1of4,q1of3,within2,q2of3,within,within10").split(",") if t]
    do_yoy = flag("yoy", "1") == "1"
    # experiment 39: --traded=1 writes quad_<tag>_traded.npz, the split grams (players who stayed / traded players)
    split = flag("traded", "0") == "1"
    LEVELS = tuple(x for x in flag("levels", ",".join(LEVELS)).split(",") if x)
    assert set(LEVELS) <= {"home", "team", "moved", "stayed"} and LEVELS[0] == "home", f"--levels={LEVELS}"
    cfg = load_config(ROOT / "config.yaml")
    ctx = Context.load(cfg)
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(8, "blas")
    except ImportError:
        pass
    t0 = time.time()
    seasons = list(range(FIRST, LAST + 1))
    box = {s: bx.with_possessions(season_box([s], ("RS", "PO"), cfg)) for s in seasons}
    k_season = {s: bx.padding_k(box[s]) for s in seasons}

    def k_for(H):
        k = np.nanmean(np.array([k_season[s] for s in seasons if abs(s - H) > 1]), axis=0)
        return np.where(np.isfinite(k), k, K_MAX)

    b1_full = pd.read_parquet(ROOT / "outputs" / "season_ratings_base_b1.parquet")
    b1_full = b1_full.assign(o=b1_full.rating_off, d=-b1_full.rating_def)

    def whole_tables(S):
        players = pd.read_parquet(ROOT / "outputs" / "within" / "within" / f"players_{S}_all.parquet")
        bxin = box_inputs(box[S], k_for(S))
        b1 = b1_full[b1_full.season == S][["player_id", "o", "d"]]
        return {side: side_table(players, side, b1, bxin) for side in ("O", "D")}

    # fixed scales: each input's standard deviation over the development seasons' whole-season tables
    scales_path = OUT / "inputs.json"
    if scales_path.exists():
        scales = json.loads(scales_path.read_text())["scales"]
    else:
        dev = [s for s in seasons if s not in hp.LOCKBOX]
        frames = {side: [] for side in ("O", "D")}
        for S in dev:
            for side, t in whole_tables(S).items():
                frames[side].append(t)
        scales = {}
        for side in ("O", "D"):
            sd = pd.concat(frames[side]).std()
            scales[side] = {c: float(sd[c]) if np.isfinite(sd[c]) and sd[c] > 0 else 1.0 for c in COLUMNS}
        scales_path.write_text(json.dumps({"columns": COLUMNS, "grid": GRID, "levels": LEVELS, "scales": scales,
                                           "lockbox": list(hp.LOCKBOX)}, indent=1))
    print(f"inputs: {len(COLUMNS)} per side; scales from the development seasons ({time.time() - t0:.0f}s)", flush=True)
    K = 2 * len(COLUMNS)

    for tag in tags:
        path = OUT / (f"quad_{tag}_traded.npz" if split else f"quad_{tag}.npz")
        if path.exists():
            print(f"  {path.name}: already built", flush=True)
            continue
        directory = ROOT / "outputs" / "within" / tag
        grid_rows = pd.read_parquet(directory / "ridge_grid.parquet")
        b1_folds = pd.read_parquet(directory / "baseline_b1.parquet")
        if split:
            store = {"G": np.zeros((len(seasons), len(GRID), 2 * K + 1, 2 * K + 1)), "sw": np.zeros(len(seasons)),
                     "traded_poss": np.zeros(len(seasons)), "all_poss": np.zeros(len(seasons)),
                     "traded_players": np.zeros(len(seasons))}
        else:
            store = {"G": np.zeros((len(seasons), len(GRID), len(LEVELS), K + 1, K + 1)), "sw": np.zeros(len(seasons)),
                     "sw_level": np.zeros((len(seasons), len(LEVELS)))}
        for H in seasons:
            if not (directory / f"folds_{H}.parquet").exists():
                continue
            sidx = seasons.index(H)
            wd = ctx.design([H], "pts")
            game_of = wd.games.drop_duplicates("game_idx").set_index("game_id").game_idx
            game_of.index = game_of.index.astype(str)
            assignment = pd.read_parquet(directory / f"folds_{H}.parquet")
            assignment.index = assignment.index.astype(str)
            row_game = wd.rows["game_idx"].to_numpy()
            kH = k_for(H)
            for i, stem in enumerate(sorted(p.stem[len("fold_"):] for p in directory.glob(f"fold_{H}_*.json"))):
                fit_ids, _, _ = S102.fold_fit_games(directory, stem, assignment)
                t = np.load(directory / f"test_{stem}.npz", allow_pickle=True)
                assert not (set(map(str, t["game_id"])) & fit_ids), f"{tag} {stem}: a held-out game among the rating games"
                keep = np.isin(row_game, game_of.reindex(sorted(fit_ids)).dropna().to_numpy())
                ridge = S103.ridge_for(wd, keep)
                fold = S104.load_fold(directory, stem)
                tables = {side: side_table(fold["players"], side, b1_folds[b1_folds.key == stem][["player_id", "o", "d"]],
                                           box_inputs(box[H], kH, fit_ids)) for side in ("O", "D")}
                F_o, F_d = design_inputs(ridge, tables, scales)
                if split:
                    b = fold["block"]
                    team_of = fold["players"].drop_duplicates("player_id").set_index("player_id").team_id
                    Zt = sc.traded_entries(b, team_of[team_of >= 0], t["team_off"], t["team_def"])
                    _, traded = sc.split_traded(b, Zt)
                    G_by = grams_for(ridge, F_o, F_d, b, None, fill=True, traded=traded)
                    if i == 0:
                        check_against_grid({(g, "home"): hp.collapse_split(G, K) for (g, _), G in G_by.items()},
                                           float(b.w.sum()), grid_rows[grid_rows.key == stem], scales)
                    for (g, _), G in G_by.items():
                        store["G"][sidx, g] += G
                    store["sw"][sidx] += float(b.w.sum())
                    store["traded_poss"][sidx] += float((Zt.T @ b.w).sum())
                    store["all_poss"][sidx] += float((b.Z.T @ b.w).sum())
                    store["traded_players"][sidx] += len(np.unique(Zt.tocoo().col % b.n_players))
                    continue
                moved = None
                if {"moved", "stayed"} & set(LEVELS):
                    team_of = fold["players"].drop_duplicates("player_id").set_index("player_id").team_id
                    moved = sc.moved_share(fold["block"], team_of[team_of >= 0], t["team_off"], t["team_def"])
                G_by = grams_for(ridge, F_o, F_d, fold["block"], (t["team_off"], t["team_def"]), fill=True, moved=moved)
                if i == 0:
                    check_against_grid(G_by, float(fold["block"].w.sum()), grid_rows[grid_rows.key == stem], scales)
                weights = score_weights_for(fold["block"], moved) if moved is not None else {}
                accumulate(store, sidx, G_by, float(fold["block"].w.sum()),
                           {lv: float(wv.sum()) for lv, wv in weights.items()})
            print(f"  {tag} {H}: done ({time.time() - t0:.0f}s)", flush=True)
        if split:
            np.savez_compressed(path, seasons=np.array(seasons), **store)
        else:
            np.savez_compressed(path, G=store["G"], sw=store["sw"], sw_level=store["sw_level"],
                                seasons=np.array(seasons), levels=np.array(LEVELS))
        print(f"  wrote {path.relative_to(ROOT)}", flush=True)

    if do_yoy and not (OUT / "quad_yoy.npz").exists():
        blocks = {}
        for T in range(FIRST + 1, LAST):
            pooled = ts.team_game_design(ctx.design([T], "pts"))
            names = list(pooled.control_names)
            F = np.asarray(pooled.F, dtype=float)
            teams = B70.team_of_rows(pooled.keys, cfg)
            blocks[T] = (sc.Block(key=str(T), season=T, deal=0, Z=sp.csr_matrix(pooled.Z), y=pooled.y.astype(float),
                                  w=pooled.w.astype(float), F=F, home=F[:, names.index("home")],
                                  player_ids=pooled.player_ids.astype(np.int64)),
                         (teams.team_off.to_numpy(), teams.team_def.to_numpy()))
        pairs = [(S, T) for S in seasons for T in (S - 1, S + 1) if T in blocks]
        G = np.zeros((len(pairs), len(GRID), len(LEVELS), K + 1, K + 1))
        sw = np.zeros(len(pairs))
        for S in seasons:
            ridge = S103.ridge_for(ctx.design([S], "pts"))
            F_o, F_d = design_inputs(ridge, whole_tables(S), scales)
            for T in (S - 1, S + 1):
                if T not in blocks:
                    continue
                block, teams = blocks[T]
                p = pairs.index((S, T))
                for (g, level), Gb in grams_for(ridge, F_o, F_d, block, teams, fill=False).items():
                    G[p, g, LEVELS.index(level)] = Gb
                sw[p] = block.w.sum()
            print(f"  year-over-year: {S} rated ({time.time() - t0:.0f}s)", flush=True)
        np.savez_compressed(OUT / "quad_yoy.npz", G=G, sw=sw, pairs=np.array(pairs))
        print("  wrote outputs/heldout/quad_yoy.npz", flush=True)
    print(f"done ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
