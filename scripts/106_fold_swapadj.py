"""Experiment 35, stage 4: OpenRAPM on every within-season fold exactly as shipped -- steps 1-3 (saved by 97), then the
prior shrink, then the swap adjustment -- built from the fold's rating games only.

    python scripts/106_fold_swapadj.py [--tags=q1of4,q1of3,within2,q2of3,within,within10] [--kappas=0.25,0.5,0.75,1]
                                       [--check_seasons=2017,2024] [--mult_tag=within] [--whole_tag=within]
                                       [--base=season_ratings_priorshrink_raw] [--shipped=season_ratings_priorshrink]
                                       [--out=openrapm_shipped]

The defaults reproduce the chimeraboost run of experiment 35.  For another incumbent name its folders and tables
(2026-10-08, the Robustness pass): `--mult_tag` the folds 99's multipliers are fitted on, `--whole_tag` the folder
holding the whole-season rows of the check, `--base` the full-season test table before the swap step (the type
model's residuals), `--shipped` the finished test table the check compares with, `--out` the file written into each
fold folder.  For the LightGBM incumbent: --mult_tag=lgb_noonc_within --whole_tag=lgb_noonc_within
--base=season_ratings_lgb_noonc_rs_shrunk_raw --shipped=season_ratings_lgb_noonc_rs.

  prior shrink   the fold's prior part times its season's pinned multipliers (99's `--rule=test`: fitted on the
                 2017-2026 folds outside {H-1, H, H+1}), the rating re-centred on the rating games' possessions
  swap adjustment, the shipped arm ("type x0.5, team version"): kappa x the type prediction, projected so every
                 team's possession-weighted total is unchanged, re-centred, the spread inside teams held to the shrunk
                 rating's (swapadjust.hold_spread "within").  What it reads:
                   * the type model: every OTHER season's swap residuals against the full-season test table
                     (`season_ratings_priorshrink_raw`), seasons within one of H left out, five player folds -- 91's
                     model, with ONE change: its features are standardised on the training seasons, not on every
                     season (91's `features` includes the rated season; differences of standardised features enter
                     the model, so only the scale moves, through the relative ridge -- measured below)
                   * the player's features: the fold's own rebuilt panel (rating games) and its shrunk prior
                   * the team constraint: each player's possessions for each team in the rating games' stints
                 kappa 0.5 is what ships; 0.25 / 0.75 / 1 are diagnostic rows only.
  check          the whole-season version of the same code against the shipped test table
                 (`season_ratings_priorshrink`): how much the standardising fix and the main-team source move it.

Writes outputs/within/<tag>/<out>.parquet (default openrapm_shipped): key, season, player_id, poss, variant (`shrunk`, `swap0.25`,
...), o, d (raw sign: d adds points allowed).
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
from eracoef import swapadjust as sa  # noqa: E402
from eracoef import swaptest as sw  # noqa: E402
from eracoef.config import load_config  # noqa: E402
from eracoef.seasons import drop_untrainable  # noqa: E402

FOLDS = 5
RIDGE = 1e-4


def _borrow(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


T90 = _borrow("_swap90_for_106", "90_swap_test.py")
S91 = _borrow("_swapadj91_for_106", "91_swap_adjust.py")      # PANEL, OWN, FEATURES
S99 = _borrow("_priorshrink99_for_106", "99_prior_shrink.py")
S102 = _borrow("_boxspm102_for_106", "102_box_spm.py")        # fold_fit_games
FEATURES, PANEL, OWN = S91.FEATURES, S91.PANEL, S91.OWN


def team_poss_from_rows(rows: pd.DataFrame) -> dict:
    """side -> player_id, team, poss: what `swapadjust.team_possessions` gives, straight from season_rows."""
    out = {}
    for side, cols, team in (("offense", sw.OFF, "team_off"), ("defense", sw.DEF, "team_def")):
        long = pd.DataFrame({"player_id": rows[cols].to_numpy().ravel(),
                             "team": np.repeat(rows[team].to_numpy(), 5), "poss": np.repeat(rows.poss.to_numpy(), 5)})
        out[side] = long.groupby(["player_id", "team"], as_index=False).poss.sum()
    return out


def project(type_pred: pd.Series, tposs: pd.DataFrame) -> pd.Series:
    """The type prediction moved to the nearest point where every team's possession-weighted total is zero
    (`swapadjust.solve` at tau = inf)."""
    empty = pd.DataFrame({"fifth_a": pd.Series([], dtype=np.int64), "fifth_b": pd.Series([], dtype=np.int64),
                          "h": pd.Series([], dtype=float), "r": pd.Series([], dtype=float)})
    return sa.solve(sa.swap_system(empty, tposs), type_pred, np.inf)


def fold_features(players: pd.DataFrame, t: pd.DataFrame, standardised) -> pd.DataFrame:
    """player_id -> the type model's features: the rebuilt panel's offensive-side rows (as 91 reads the season panel)
    and the SHRUNK prior (as 91 reads its base table), standardised on the training seasons."""
    o_rows = players[players.side == "O"].drop_duplicates("player_id").set_index("player_id")
    return standardised(o_rows[PANEL].join(t.set_index("player_id")[OWN], how="inner"))


def shrink_fold(players: pd.DataFrame, m_off: float, m_def: float) -> pd.DataFrame:
    """One fold's rated players, prior part times the multipliers, re-centred on possessions (99's `shrink`)."""
    t = players.drop_duplicates("player_id")
    t = t[t.poss_off > 0].copy()
    t["prior_off"], t["prior_def"] = m_off * t.prior_off, m_def * t.prior_def
    w = t.poss_off.to_numpy(float)
    for side in ("off", "def"):
        level = np.average(t[f"prior_{side}"] + t[f"u_{side}"], weights=w)
        t[f"prior_{side}"] -= level
        t[f"rating_{side}"] = t[f"prior_{side}"] + t[f"u_{side}"]
    return t


def adjust(t: pd.DataFrame, feats: pd.DataFrame, betas: dict, tposs: dict, kappas) -> dict:
    """variant -> (o, d) raw sign for the rated players of `t` (one fold or one season)."""
    o0, d0 = t.rating_off.to_numpy(float), -t.rating_def.to_numpy(float)
    w_o, w_d = t.poss_off.to_numpy(float), t.poss_def.to_numpy(float)
    team = t.team_id.to_numpy(dtype=float)
    team[team < 0] = np.nan
    ids = t.player_id.to_numpy(np.int64)
    fs = feats.reindex(ids)
    pred = {}
    for side in sa.SIDES:
        p = pd.Series(np.nan, index=ids, dtype=float)
        fold = ids % FOLDS
        ok = ~fs.isna().any(axis=1).to_numpy()
        for k in range(FOLDS):
            mine = (fold == k) & ok
            p[mine] = fs.to_numpy()[mine] @ betas[side][k]
        pred[side] = p
    out = {"shrunk": (o0, d0)}
    for kappa in kappas:
        res = []
        for side, base, w in (("offense", o0, w_o), ("defense", d0, w_d)):
            c = project(kappa * pred[side], tposs[side]).reindex(ids).fillna(0.0).to_numpy()
            c = c - np.average(c, weights=w)
            held, _ = sa.hold_spread(base + c, base, w, team, "within")
            res.append(held)
        out[f"swap{kappa:g}"] = tuple(res)
    return out


def main():
    check_flags()
    tags = [t for t in flag("tags", "q1of4,q1of3,within2,q2of3,within,within10").split(",") if t]
    kappas = [float(x) for x in flag("kappas", "0.25,0.5,0.75,1").split(",") if x]
    check_seasons = [int(x) for x in flag("check_seasons", "2017,2024").split(",") if x]
    mult_tag, whole_tag = flag("mult_tag", "within"), flag("whole_tag", "within")
    base_name = flag("base", "season_ratings_priorshrink_raw")
    shipped_name = flag("shipped", "season_ratings_priorshrink")
    out_name = flag("out", "openrapm_shipped")
    cfg = load_config(ROOT / "config.yaml")
    s0, s1 = int(cfg["first_season"]), int(cfg["last_season"])
    margin_clip = float(cfg.get("margin_clip", 25))
    t0 = time.time()

    # the multipliers the shipped test table used, per rated season (pinned 2017-2026 folds, rule test)
    seasons = list(range(s0, s1 + 1))
    mult = S99.multipliers_for(seasons, S99.load_folds(ROOT / "outputs" / "within" / mult_tag, (2017, 2026)), "test")
    mult = mult.set_index("season")

    # every season's swap residuals against the full-season test table, and the raw (unstandardised) features
    base_path = ROOT / "outputs" / f"{base_name}.parquet"
    base = pd.read_parquet(base_path)
    raw = T90.load_table(base_path)
    # the trust boundary (src/eracoef/seasons.py): a season still being played never trains the type model (91's rule)
    panel, _ = drop_untrainable(pd.read_parquet(ROOT / "outputs" / "role_panel_season.parquet"), cfg,
                                what="the season panel (type model training)")
    trainable = set(panel.season.astype(int))
    panel = panel[panel.side == "O"][["player_id", "season", *PANEL]]
    full_feats = panel.merge(base[["player_id", "season", *OWN]], on=["player_id", "season"], how="inner")
    pairs, rows_of = {}, {}
    for s in seasons:
        rows = sw.season_rows(T90.load_stints(s, cfg), margin_clip=margin_clip)
        rows_of[s] = rows
        pairs[s] = sa.residual_pairs(sw.prepare(rows, s), raw[raw.season == s], context="nofatigue")
    print(f"swap residuals for {len(seasons)} seasons ({time.time() - t0:.0f}s)", flush=True)

    out_rows = {tag: [] for tag in tags}
    checks = []
    for H in seasons:
        train = [s for s in seasons if abs(s - H) > 1 and s in trainable]
        f_train = full_feats[full_feats.season.isin(train)]
        mu, sd = f_train[FEATURES].mean(), f_train[FEATURES].std(ddof=0)

        def standardised(frame):
            return ((frame[FEATURES].astype(float) - mu) / sd)

        grams = {side: [] for side in sa.SIDES}
        for s in train:
            fs = standardised(full_feats[full_feats.season == s]).set_index(full_feats[full_feats.season == s].player_id)
            fold_of = pd.Series(fs.index.to_numpy() % FOLDS, index=fs.index)
            for side in sa.SIDES:
                grams[side].append(sa.pair_grams(pairs[s][side], fs, fold_of, FOLDS))
        betas = {side: [sa.type_coefficients(grams[side], k, RIDGE) for k in range(FOLDS)] for side in sa.SIDES}
        m_off, m_def = float(mult.at[H, "prior_off"]), float(mult.at[H, "prior_def"])

        # the whole-season check: the same code on all of H against the shipped test table
        if H in check_seasons:
            whole = pd.read_parquet(ROOT / "outputs" / "within" / whole_tag / f"players_{H}_all.parquet")
            t = shrink_fold(whole, m_off, m_def)
            got = adjust(t, fold_features(whole, t, standardised), betas, team_poss_from_rows(rows_of[H]), [0.5])
            shipped = pd.read_parquet(ROOT / "outputs" / f"{shipped_name}.parquet")
            shipped = shipped[shipped.season == H].set_index("player_id")
            o, d = got["swap0.5"]
            ids = t.player_id.to_numpy()
            gap = np.r_[np.abs(o - shipped.rating_off.reindex(ids).to_numpy()),
                        np.abs(d + shipped.rating_def.reindex(ids).to_numpy())]
            gap_shrunk = np.r_[np.abs(t.rating_off.to_numpy() - pd.read_parquet(base_path).query("season == @H")
                                      .set_index("player_id").rating_off.reindex(ids).to_numpy())]
            checks.append(dict(season=H, shrunk_max=float(np.nanmax(gap_shrunk)), shipped_max=float(np.nanmax(gap)),
                               shipped_median=float(np.nanmedian(gap))))
            print(f"  check {H}: shrunk vs shipped raw table max {np.nanmax(gap_shrunk):.2e}; swap 0.5 vs shipped "
                  f"max {np.nanmax(gap):.3f}, median {np.nanmedian(gap):.4f} pts/100", flush=True)

        for tag in tags:
            directory = ROOT / "outputs" / "within" / tag
            if not (directory / f"folds_{H}.parquet").exists():
                continue
            assignment = pd.read_parquet(directory / f"folds_{H}.parquet")
            assignment.index = assignment.index.astype(str)
            for stem in sorted(p.stem[len("fold_"):] for p in directory.glob(f"fold_{H}_*.json")):
                fit_ids, _, _ = S102.fold_fit_games(directory, stem, assignment)
                fit_rows = rows_of[H][rows_of[H].game_id.astype(str).isin(fit_ids)]
                assert set(fit_rows.game_id.astype(str).unique()) <= fit_ids
                players = pd.read_parquet(directory / f"players_{stem}.parquet")
                t = shrink_fold(players, m_off, m_def)
                got = adjust(t, fold_features(players, t, standardised), betas, team_poss_from_rows(fit_rows), kappas)
                for variant, (o, d) in got.items():
                    out_rows[tag].append(pd.DataFrame({"key": stem, "season": H, "player_id": t.player_id.to_numpy(),
                                                       "poss": t.poss_off.to_numpy(), "variant": variant, "o": o, "d": d}))
        print(f"  {H}: folds adjusted ({time.time() - t0:.0f}s)", flush=True)

    for tag, parts in out_rows.items():
        if parts:
            pd.concat(parts, ignore_index=True).to_parquet(ROOT / "outputs" / "within" / tag / f"{out_name}.parquet",
                                                           index=False)
    if checks:
        pd.DataFrame(checks).to_csv(ROOT / "outputs" / "csv" / "fold_swapadj_check.csv", index=False)
    print(f"done ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
