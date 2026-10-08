"""Within-season misses: rate a season from part of its games, exactly as the board rates a whole one, and keep
the games it never saw to measure the rating on.

    python scripts/97_within_season.py --models=<name> --base=<table> [--seasons=2017-2026] [--repeats=3]
                                       [--folds=4] [--rate_from=rest|one] [--tag=within] [--check_only=0]

Why (the owner, 2026-10-03).  A calibrator for the ratings' misses needs a target.  The trade set's misses are
measured on the NEIGHBOURING seasons' games, so a calibrator trained on them would learn that a peak season comes
back down the year after, and shrink every peak season.  The owner: "Let's smartly think through how to not shrink
peak seasons.  One idea I have is bootstrapping the heck out of the single seasons to get lots of holdout data?"
This is that: the target comes from the season's OWN games, held out.  Then: "Just keep really special good tabs on
it" -- every input, what it is built from and the check that proves it is in WITHIN_SEASON_LEDGER.md.

How.  For a season H and a repeat r, H's games (regular season and playoffs) are dealt at random into `--folds`
folds, seeded by (H, r).  For each fold k:

  fit games   every game NOT in fold k.  The rating is built from these alone, by the board's own functions
              (scripts/62_single_year_board.py, steps 1-3): the SAME fitted prior models (`--models`, written by
              62's `--save_models` on the incumbent's settings), asked about H's box score, on-court record,
              playing time, starts and tenure REBUILT FROM THE FIT GAMES; the same ridge (penalty 13,037, the free
              prior scale priced on cross-fitted columns rebuilt inside the fit games); the same centring.
  test games  fold k.  Nothing about them reaches the rating.  Stored: the test team-games (points per 100 as
              actually scored, each player's share of the possessions), and the rating's error on them.

The swap adjustment (scripts/91) is NOT applied: the ratings here are steps 1-3, the table 91 starts from.

The reproduction check.  Before any fold, each season is rebuilt with the fit games = ALL of its games, and that
rating must equal the board's own row (`--base`, written by the same 62 run that saved the models): the
part-season ratings then come from the shipped code path and nothing else.  The rebuilt inputs are compared with
the season panel the board reads, column by column.

Writes, under outputs/within/<tag>/:
  folds_<H>.parquet                  game_id -> fold, one column per repeat
  players_<H>_all.parquet            one row per player and side: inputs, rating and its pieces, whole season
  players_<H>_r<r>_f<k>.parquet      the same from the fit games, plus each player's possessions in the test games
  test_<H>_r<r>_f<k>.npz             the test team-games: shares, points per 100, possessions, controls, keys, teams
  fold_<H>_r<r>_f<k>.json            the fold's error on its test games and every leak check (a failed check stops)
  reproduce_<H>.json                 the reproduction check
  summary.parquet, checks.parquet    all folds, gathered at the end
A finished fold is skipped on a rerun, so a stopped run resumes where it stopped.
"""
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

# Thread pinning before numpy and chimeraboost, exactly as 62 does it (never two builds at once).
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
os.environ.setdefault("NUMBA_NUM_THREADS", os.environ.get("OPENRAPM_NUMBA_THREADS", "4"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eracoef import singleyear as sy  # noqa: E402
from eracoef import tradeset as ts_mod  # noqa: E402
from eracoef.bio import PLAYER_INPUTS, player_inputs  # noqa: E402
from eracoef.config import load_config, resolve  # noqa: E402
from eracoef.design import AWAY_SLOTS, FEATURES, HOME_SLOTS, WindowData  # noqa: E402
from eracoef.holdout import Context, Ratings, predict_season, score  # noqa: E402
from eracoef.ingest import GAME_PREFIX, game_table, load_gamelog, raw_dir  # noqa: E402
from eracoef.investigate import offcourt_rates, oncourt_rates  # noqa: E402
from eracoef.priorridge import armse  # noqa: E402
from eracoef.seasons import drop_untrainable  # noqa: E402
from eracoef.roles import (CAREER_INPUTS, RAW_INPUTS, ROLE_PHASES, career_inputs,  # noqa: E402
                           player_season_inputs, roles_from_boxes, season_ages, shares_from_stints,
                           window_inputs)
from eracoef.spm import apm_fit, season_of_units  # noqa: E402
from eracoef.windows import window_label  # noqa: E402
from eracoef.xshoot import DEFENSE_TARGETS, SHOT_LEAGUE_COLS, SHOT_TOTAL_COLS, player_shot_frame  # noqa: E402
from _cli import check_flags, flag  # noqa: E402


def _borrow(name: str, file: str):
    """Another script's functions, loaded the way 70 loads 63's: one definition, never a copy."""
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


B62 = _borrow("_board62", "62_single_year_board.py")      # the prior, the fold prior, the ridge
B70 = _borrow("_tradeset70", "70_tradeset.py")             # team_of_rows

# what the board's ridge runs at: 62's defaults (`--lambda_player=13037`, `--free_scale=1`, `--crossfit=scale`)
PENALTY = 13037.0
TARGETS = {"O": sy.OFFENSE_TARGET, "D": sy.DEFENSE_TARGET}       # xpts_ft, x3def_w0.25: what each ridge explains
ONC_TARGETS = ("xpts_ft", "x3def")                             # what the panel's on-court columns are built from
SCORE_TARGET = "pts"                                           # the test games are scored on points as scored
FILL = (500.0, 0.25)                                           # the year-over-year test's fill, 500x0.25
SHOT_COLS_ALL = [*SHOT_TOTAL_COLS, *SHOT_LEAGUE_COLS]
CLOSE_SLOTS = ([(f"h{i}", "poss_h", "poss_a") for i in range(1, 6)]
               + [(f"a{i}", "poss_a", "poss_h") for i in range(1, 6)])


# ------------------------------------------------------------------------------------------------ the folds
def deadline_folds(games: pd.DataFrame, cut: float) -> pd.Series:
    """game_id -> fold for the trade-deadline split: 1 = regular-season games before the date by which `cut` of the
    regular season had been played, 0 = every later game and the playoffs."""
    g = games.drop_duplicates("game_id")
    dates = pd.to_datetime(g.game_date)
    rs = np.sort(dates[(g.phase == "RS").to_numpy()].to_numpy())
    cut_date = rs[int(round(cut * (len(rs) - 1)))]
    late = (g.phase != "RS").to_numpy() | (dates.to_numpy() >= cut_date)
    return pd.Series(np.where(late, 0, 1), index=g.game_id.astype(str).to_numpy(), name="r0")


def deal_folds(game_ids, n_folds: int, season: int, repeat: int) -> pd.Series:
    """game_id -> fold: the season's games dealt at random, every fold the same size to within one game.  Seeded by
    (season, repeat), so a rerun deals the same folds and a new repeat deals new ones."""
    ids = np.sort(np.asarray(pd.unique(pd.Series(game_ids).astype(str))))
    rng = np.random.default_rng([int(season), int(repeat), 9701])
    return pd.Series(rng.permutation(len(ids)) % int(n_folds), index=ids, name=f"r{repeat}")


def cut_design(wd: WindowData, keep_idx: np.ndarray) -> WindowData:
    """The design restricted to the games `keep_idx` (game_idx values): its rows AND its per-game box-score and
    possession tables.

    `WindowData.subset` keeps the WHOLE season's `game_box` and `game_poss`, and those are what the padded box
    rates, their padding constants and the possession counts are built from -- a fit on a subset would still read
    every game's box score.  This is the leak FINDINGS 31 found in the in-season cut, closed the same way: the
    tables are cut with the rows.  `games` (ids, dates, phases: no outcomes) is kept whole so game_idx still
    indexes it."""
    keep = np.isin(wd.rows["game_idx"].to_numpy(), keep_idx)
    sub = wd.subset(keep)
    return WindowData(sub.X_src, sub.y, sub.w, sub.groups, sub.spec,
                      wd.game_box[wd.game_box["game_idx"].isin(keep_idx)].reset_index(drop=True),
                      wd.game_poss[wd.game_poss["game_idx"].isin(keep_idx)].reset_index(drop=True),
                      sub.rows, wd.games, sub.counters)


# ------------------------------------------------------------------------------- the season's raw tables
class SeasonTables:
    """One season's per-game tables, read once: the box scores (games, starts, minutes), the stints (possessions on
    the floor, score state, phase) and each game's two teams.  `roles.season_roles` reads the same files."""

    def __init__(self, season: int, cfg):
        self.season = int(season)
        box_dir = raw_dir(cfg) / "box" / str(season)
        self.boxes = {}
        stints, games = [], []
        for phase in ROLE_PHASES:
            for path in sorted(box_dir.glob(f"{GAME_PREFIX[phase]}*.parquet")):
                self.boxes[path.stem] = pd.read_parquet(path, columns=["teamId", "personId", "minutes"])
            path = Path(resolve(cfg, "stints")) / f"{season}_{phase}.parquet"
            if path.exists():
                part = pd.read_parquet(path, columns=["game_id", "period", *HOME_SLOTS, *AWAY_SLOTS, "poss_h",
                                                      "poss_a", "margin_h", "is_gt"])
                stints.append(part.assign(phase=phase))
                games.append(game_table(load_gamelog(season, phase, cfg)))
        self.stints = pd.concat(stints, ignore_index=True)
        self.stints["game_id"] = self.stints["game_id"].astype(str)
        self.games = pd.concat(games, ignore_index=True)
        self.games["game_id"] = self.games["game_id"].astype(str)
        self.ages = season_ages(season, cfg)


def roles_from_games(tables: SeasonTables, keep_ids: set) -> tuple:
    """`roles.season_roles` + `build_roles`' age join, on the games `keep_ids` only: the same arithmetic on fewer
    games.  `poss_pct` and `gs_pct` built from it are then shares OF THE FIT GAMES (the team's denominator is cut
    with the player's numerator).  Returns (roles rows, the box games read, the stint games read)."""
    boxes = {g: b for g, b in tables.boxes.items() if g in keep_ids}
    roles = roles_from_boxes(boxes)
    stints = tables.stints[tables.stints["game_id"].isin(keep_ids)]
    shares = shares_from_stints(stints, tables.games)
    out = roles.merge(shares, on=["player_id", "team_id"], how="outer")
    out[["games", "starts", "minutes", "poss_on"]] = out[["games", "starts", "minutes", "poss_on"]].fillna(0.0)
    team_poss = shares.drop_duplicates("team_id").set_index("team_id")["team_poss"]
    out["team_poss"] = out["team_poss"].fillna(out["team_id"].map(team_poss))
    out.insert(1, "season", tables.season)
    out = out.merge(tables.ages, on=["player_id", "season"], how="left")
    return out, set(boxes), set(stints["game_id"].unique())


def closeness_rows(stints: pd.DataFrame, season: int) -> pd.DataFrame:
    """scripts/69_closeness_panel.py's `exposure`, for the stints given: per (player, side) the share of his
    possessions in garbage time, the mean of 1 / max(|margin|, 1), the mean |margin|."""
    st = stints.copy()
    st["abs_margin_"] = st["margin_h"].abs()
    st["closeness_"] = 1.0 / np.maximum(st["abs_margin_"], 1.0)
    st["is_gt_"] = st["is_gt"].astype(float)
    rows = []
    for slot, own, opponent in CLOSE_SLOTS:
        for side, poss_col in (("O", own), ("D", opponent)):
            part = st[[slot, poss_col, "abs_margin_", "closeness_", "is_gt_"]].copy()
            part.columns = ["player_id", "poss", "abs_margin_", "closeness_", "is_gt_"]
            part["side"] = side
            rows.append(part)
    d = pd.concat(rows, ignore_index=True)
    d = d[(d["player_id"] > 0) & (d["poss"] > 0)]
    for column in ("abs_margin_", "closeness_", "is_gt_"):
        d[f"w_{column}"] = d["poss"] * d[column]
    g = d.groupby(["player_id", "side"], as_index=False).agg(
        poss=("poss", "sum"), w_gt=("w_is_gt_", "sum"), w_margin=("w_abs_margin_", "sum"),
        w_close=("w_closeness_", "sum"))
    g["season"] = season
    g["gt_share"] = g["w_gt"] / g["poss"]
    g["abs_margin"] = g["w_margin"] / g["poss"]
    g["closeness"] = g["w_close"] / g["poss"]
    return g[["player_id", "season", "side", *sy.CLOSENESS]]


def playoff_share(tables: SeasonTables, keep_ids: set) -> pd.Series:
    """scripts/86_context_panel.py's `po_share` on the games `keep_ids`: 1 - regular-season possessions on the floor
    / all possessions on the floor, summed over his teams."""
    stints = tables.stints[tables.stints["game_id"].isin(keep_ids)]
    both = shares_from_stints(stints, tables.games)
    regular = shares_from_stints(stints[stints["phase"] == "RS"], tables.games)
    both = both[both.poss_on > 0].groupby("player_id").poss_on.sum()
    regular = regular[regular.poss_on > 0].groupby("player_id").poss_on.sum()
    return (1.0 - regular.reindex(both.index).fillna(0.0) / both).clip(0.0, 1.0).rename(sy.PO_SHARE[0])


# ------------------------------------------------------------------------------------- one season's world
class World:
    """Season H rated from the games `fit_ids`, with every input the rating reads rebuilt from those games.

    There is ONE path.  The whole season is this same class handed every game, and the reproduction check holds it
    to the board's own table -- so a passing check is a check on the part-season code, not on a second copy."""

    def __init__(self, season: int, cfg, full: dict, tables: SeasonTables, inputs: pd.DataFrame,
                 roles: pd.DataFrame, fit_ids):
        self.season, self.cfg = int(season), cfg
        games = full["pts"].games
        all_ids = set(games["game_id"].astype(str))
        self.fit_ids = set(str(g) for g in fit_ids)
        assert self.fit_ids <= all_ids, "fit games that are not this season's"
        self.whole = self.fit_ids == all_ids
        self.fit_idx = np.sort(games.loc[games["game_id"].astype(str).isin(self.fit_ids), "game_idx"].to_numpy())
        self.checks: dict = {}
        cap = float(cfg.get("roles", {}).get("share_cap", 0.9))

        # -- the designs.  The two x3def targets reprice every opponent three at the shooter's 3P%, which must come
        #    from the fit games too (`keep`, FINDINGS 31's x3def row), and are rebuilt on the cut points design.
        keep = {self.season: sorted(self.fit_ids)}
        pts = cut_design(full["pts"], self.fit_idx)
        self.d = {"pts": pts, "xpts_ft": cut_design(full["xpts_ft"], self.fit_idx),
                  "x3def": DEFENSE_TARGETS["x3def"]([season], cfg, pts, keep=keep)[0],
                  "x3def_w0.25": DEFENSE_TARGETS["x3def_w0.25"]([season], cfg, pts, keep=keep)[0]}
        reference = self.d["pts"].rows
        for name, wd in self.d.items():
            games_used = set(wd.rows["game_idx"].unique())
            self.checks[f"design {name}: rows only from fit games"] = games_used <= set(self.fit_idx)
            self.checks[f"design {name}: box and possession tables only from fit games"] = (
                set(wd.game_box["game_idx"].unique()) <= set(self.fit_idx)
                and set(wd.game_poss["game_idx"].unique()) <= set(self.fit_idx))
            self.checks[f"design {name}: same rows as the points design"] = (
                np.array_equal(wd.rows["game_idx"].to_numpy(), reference["game_idx"].to_numpy())
                and np.array_equal(wd.rows["is_home_off"].to_numpy(), reference["is_home_off"].to_numpy()))
        for name in ("x3def", "x3def_w0.25"):
            # against the board's whole-season design on the same rows: 0 for the whole season (the rebuild is the
            # board's design), above 0 for a part season (evidence the shooters were repriced from the fit games)
            whole_y = full[name].y[np.isin(full[name].rows["game_idx"].to_numpy(), self.fit_idx)]
            self.checks[f"design {name}: response against the whole-season design (max change)"] = float(
                np.max(np.abs(self.d[name].y - whole_y)))

        # -- playing time, starts, tenure: the season's role rows rebuilt from the fit games
        roles_h, box_games, stint_games = roles_from_games(tables, self.fit_ids)
        self.checks["roles: box scores read only from fit games"] = box_games <= self.fit_ids
        self.checks["roles: stints read only from fit games"] = stint_games <= self.fit_ids
        if self.whole:
            # the rebuild of the role table itself, against the cached one the panel was built from
            stored = roles[roles.season == self.season].sort_values(["player_id", "team_id"]).reset_index(drop=True)
            mine = roles_h[list(roles.columns)].sort_values(["player_id", "team_id"]).reset_index(drop=True)
            same_rows = len(stored) == len(mine) and np.array_equal(stored[["player_id", "team_id"]].to_numpy(),
                                                                     mine[["player_id", "team_id"]].to_numpy())
            self.checks["roles: whole-season rebuild has the cached rows"] = bool(same_rows)
            if same_rows:
                for c in ("games", "starts", "minutes", "poss_on", "team_poss", "age"):
                    self.checks[f"roles: whole-season rebuild, max difference in {c}"] = float(
                        np.nanmax(np.abs(mine[c].to_numpy(float) - stored[c].to_numpy(float))))
        roles_f = pd.concat([roles[roles.season != self.season], roles_h[list(roles.columns)]], ignore_index=True)
        inputs_f = pd.concat([inputs[inputs.season != self.season],
                              player_season_inputs(roles_h[list(roles.columns)], cap=cap)], ignore_index=True)
        self.roles_h = roles_h

        # -- the panel rows: scripts/49_role_panel.py pass 1 and pass 4, on one season, on these designs
        wd_o, wd_d = self.d["xpts_ft"], self.d["x3def"]
        ids = wd_o.spec.ps_table["player_id"].to_numpy()
        inp = window_inputs(wd_o, inputs_f, cap=cap)
        season_col = season_of_units(wd_o)
        shots = player_shot_frame([season], cfg, ids, keep=keep)
        onc = oncourt_rates(wd_o, wd_d)
        offc = offcourt_rates(wd_o, wd_d)
        parts = []
        for side, wd in (("O", wd_o), ("D", wd_d)):
            a = apm_fit(wd, cfg)
            d = pd.DataFrame(a["ro"] if side == "O" else a["rd"], columns=FEATURES)
            raw = (a["pipe"]["exposure"].season_rates_ if side == "O"
                   else a["pipe"]["exposure"].season_rates_d_)
            for j, c in enumerate(FEATURES):
                d[f"raw_{c}"] = np.asarray(raw, dtype=float)[:, j]
            d.insert(0, "window", window_label([season]))
            d.insert(1, "side", side)
            d.insert(2, "player_id", ids)
            d.insert(3, "ps_idx", np.arange(wd.spec.n_ps))
            d.insert(4, "season", season_col)
            for c in SHOT_COLS_ALL:
                d[c] = shots[c].to_numpy(dtype=float)
            for c in sy.ONC:
                d[c] = onc[c].to_numpy(dtype=float)
            for c in sy.OFFC:
                d[c] = offc[c].to_numpy(dtype=float)
            d["poss"] = a["poss_o"] if side == "O" else a["poss_d"]
            for c in RAW_INPUTS:
                d[c] = inp[c].to_numpy()
            d["apm"] = a["u_o"] if side == "O" else a["u_d"]
            parts.append(d)
        panel = pd.concat(parts, ignore_index=True)
        panel["net_o"], panel["net_d"] = panel.onc_o - panel.offc_o, panel.onc_d - panel.offc_d
        ci = career_inputs(inputs_f, self.season, panel["player_id"].to_numpy(), age=panel["age"].to_numpy())
        panel[CAREER_INPUTS] = ci[CAREER_INPUTS].to_numpy(dtype=float)
        panel[PLAYER_INPUTS] = player_inputs(cfg, roles_f, [self.season],
                                             panel["player_id"].to_numpy())[PLAYER_INPUTS].to_numpy(dtype=float)
        # context columns the calibrator may read (not the prior): score state, playoff share
        stints = tables.stints[tables.stints["game_id"].isin(self.fit_ids)]
        self.checks["context: stints read only from fit games"] = set(stints["game_id"].unique()) <= self.fit_ids
        close = closeness_rows(stints, self.season)
        panel = panel.merge(close, on=["player_id", "season", "side"], how="left")
        panel[sy.PO_SHARE[0]] = panel["player_id"].map(playoff_share(tables, self.fit_ids)).fillna(0.0)
        self.panel = panel

    # --------------------------------------------------------------------------------- steps 1-3 of the board
    def rate(self, saved: dict) -> pd.DataFrame:
        """62's season loop on this world's inputs: the saved boosters asked about the rebuilt panel rows, the fold
        prior rebuilt inside these games, the two ridges, the merge and the centring."""
        prior, held_frames, model_feats_of, models, raw_prior = {}, {}, {}, {}, {}
        for side in ("O", "D"):
            m = saved[side]
            model = B62.OutOfPlayerSPM(m["params"], len(m["folds"]), intercept=m["intercept"],
                                       by_player=m["by_player"])
            model.full_, model.fold_models_, model.excluded_ = m["full"], m["folds"], m["excluded"]
            rows = self.panel[(self.panel.side == side) & (self.panel.poss > 0)]
            held = sy.season_frame(rows, m["feats"])
            held = held.assign(chunk_poss=held.poss.to_numpy(float), chunk_seasons=1.0)
            prior[side] = dict(zip(held.player_id.to_numpy(), model.predict(held, m["model_feats"])))
            raw_prior[side] = prior[side]
            models[side], held_frames[side], model_feats_of[side] = model, held, m["model_feats"]
        reads_pieces = any(c in model_feats_of[s] for s in model_feats_of for c in sy.PIECES)
        assert not reads_pieces, "the saved priors read the RAPM pieces; this script does not rebuild them"
        self.prior_raw = raw_prior
        _, def_poss = B62.side_possessions(self.d[TARGETS["O"]])
        fold_prior = B62._fold_prior_builder(self.d[ONC_TARGETS[0]], self.d[ONC_TARGETS[1]], models, held_frames,
                                             model_feats_of, teams=None)
        fits = {name: B62._ridge(self.d[name], prior, fold_prior=fold_prior, crossfit_penalty=False)
                for name in dict.fromkeys(TARGETS.values())}
        table = {side: fits[TARGETS[side]].ratings_ for side in ("O", "D")}
        merged = (table["O"][["player_id", "offense", "prior_offense", "possessions"]]
                  .merge(table["D"][["player_id", "defense", "prior_defense"]], on="player_id"))
        merged["poss_def"] = merged.player_id.map(def_poss).fillna(0.0).to_numpy(float)
        merged["prior_scale_off"] = fits[TARGETS["O"]].prior_scale_[0]
        merged["prior_scale_def"] = fits[TARGETS["D"]].prior_scale_[1]
        # 62's `finish`: per side, the possession-weighted mean of the rating is zero
        for side in ("offense", "defense"):
            level = float(np.average(merged[side], weights=merged.possessions))
            merged[side] -= level
            merged[f"prior_{side}"] -= level
        merged["season"] = self.season
        self.checks["ridge: possessions are the fit games' possessions"] = bool(np.isclose(
            merged.possessions.sum(), float(self.d[TARGETS["O"]].rows["poss"].sum()) * 5.0, rtol=1e-9))
        self.ratings = merged
        return merged

    def player_table(self) -> pd.DataFrame:
        """One row per player and side with poss > 0: every rebuilt input (the derived columns built), the rating in
        the board's positive-good schema, and the prior before its scale."""
        frame = sy.season_frame(self.panel[self.panel.poss > 0].copy(),
                                list(dict.fromkeys(sy.PRIOR_FEATURES + sy.OFFC + sy.NET + sy.CLOSENESS)))
        r = self.ratings
        board = pd.DataFrame({
            "player_id": r.player_id, "rating_off": r.offense, "rating_def": -r.defense,
            "prior_off": r.prior_offense, "prior_def": -r.prior_defense,
            "poss_off": r.possessions, "poss_def": r.poss_def,
            "prior_scale_off": r.prior_scale_off, "prior_scale_def": r.prior_scale_def})
        board["rating_total"] = board.rating_off + board.rating_def
        board["u_off"], board["u_def"] = board.rating_off - board.prior_off, board.rating_def - board.prior_def
        board["prior_raw_off"] = board.player_id.map(self.prior_raw["O"])
        board["prior_raw_def"] = -board.player_id.map(self.prior_raw["D"])
        # his main team in these games (most possessions on the court), for team-level inputs a calibrator builds
        played = self.roles_h[self.roles_h.poss_on > 0]
        main = (played.sort_values(["player_id", "poss_on"], ascending=[True, False])
                .drop_duplicates("player_id").set_index("player_id").team_id)
        board["team_id"] = board.player_id.map(main).fillna(-1).astype(np.int64)
        return frame.merge(board, on="player_id", how="left")


# ------------------------------------------------------------------------------------------ the test games
def test_games(world: World, full_pts: WindowData, test_idx: np.ndarray, cfg) -> tuple:
    """The fold's test games on POINTS AS SCORED: the rating's team-game error on them (the year-over-year test's
    own scoring, `predict_season` + `score`, the level and home edge refit on the scored games), and the team-games
    pooled the trade set's way (`tradeset.team_game_design`) for a calibrator to fit on."""
    keep = np.isin(full_pts.rows["game_idx"].to_numpy(), test_idx)
    wd = full_pts.subset(keep)
    r = world.ratings
    rated = pd.DataFrame({"player_id": r.player_id, "o": r.offense, "d": r.defense, "poss": r.possessions})
    rated = rated[rated.poss > 0].reset_index(drop=True)
    fill_o, fill_d = ts_mod.replacement_fill(rated, max_poss=FILL[0], shrink=FILL[1])
    result = score(predict_season(Ratings(rated, fill_o=fill_o, fill_d=fill_d), wd, level="home"))
    pooled = ts_mod.team_game_design(wd)
    teams = B70.team_of_rows(pooled.keys, cfg)
    checks = {"test: rows only from test games": set(wd.rows["game_idx"].unique()) <= set(test_idx),
              "test: no test game among the fit games": not (set(test_idx) & set(world.fit_idx)),
              "test: shares sum to five": bool(np.allclose(
                  np.asarray(pooled.Z[:, :pooled.n_players].sum(axis=1)).ravel(), 5.0, atol=1e-6))}
    with_poss = np.asarray(pooled.Z.T @ pooled.w).ravel()
    n = pooled.n_players
    test_poss = pd.DataFrame({"player_id": pooled.player_ids, "test_poss_off": with_poss[:n],
                              "test_poss_def": with_poss[n:]})
    summary = dict(tg=float(result["tg"]), tg_base=float(result["tg_base"]), tg_n=float(result["tg_n"]),
                   game_armse=float(armse(result["tg"])), scale_off=float(result["scale_off"]),
                   scale_def=float(result["scale_def"]), fill_o=fill_o, fill_d=fill_d,
                   test_team_games=int(pooled.Z.shape[0]))
    return pooled, teams, test_poss, summary, checks


def save_test(path: Path, pooled, teams) -> None:
    Z = pooled.Z.tocsr()
    np.savez_compressed(path, Z_data=Z.data, Z_indices=Z.indices, Z_indptr=Z.indptr, Z_shape=np.array(Z.shape),
                        y=pooled.y, w=pooled.w, F=pooled.F, control_names=np.array(pooled.control_names),
                        player_ids=pooled.player_ids, game_idx=pooled.keys.game_idx.to_numpy(),
                        is_home_off=pooled.keys.is_home_off.to_numpy(), game_id=pooled.keys.game_id.astype(str).to_numpy(),
                        phase=pooled.keys.phase.astype(str).to_numpy(),
                        team_off=teams.team_off.to_numpy(), team_def=teams.team_def.to_numpy())


# -------------------------------------------------------------------------------- the reproduction check
REPRO_TOL = 1e-6


def reproduce(world: World, base: pd.DataFrame, panel: pd.DataFrame, saved: dict) -> dict:
    """The whole-season world against the board's own row for the season, and its rebuilt inputs against the
    season panel the board read."""
    mine = world.ratings
    theirs = base[base.season == world.season]
    joined = mine.merge(theirs[["player_id", "offense", "defense", "prior_offense", "prior_defense"]],
                        on="player_id", suffixes=("", "_board"))
    out = dict(season=world.season, players=int(len(mine)), board_players=int(len(theirs)), joined=int(len(joined)))
    for c in ("offense", "defense", "prior_offense", "prior_defense"):
        out[f"max_diff_{c}"] = float(np.max(np.abs(joined[c] - joined[f"{c}_board"])))
    # the prior the board handed its ridge (saved with the models) against the rebuilt one
    for side in ("O", "D"):
        board_prior = pd.Series(saved[side]["prior"])
        rebuilt = pd.Series(world.prior_raw[side]).reindex(board_prior.index)
        out[f"max_diff_raw_prior_{side}"] = float(np.nanmax(np.abs(rebuilt - board_prior)))
        out[f"raw_prior_missing_{side}"] = int(rebuilt.isna().sum())
    stored = panel[panel.season == world.season]
    columns = [c for c in world.panel.columns if c in stored.columns and c not in ("window", "side", "player_id",
                                                                                   "ps_idx", "season")]
    both = world.panel.merge(stored, on=["player_id", "side"], suffixes=("", "_panel"))
    diffs = {c: float(np.nanmax(np.abs(both[c].to_numpy(float) - both[f"{c}_panel"].to_numpy(float))))
             for c in columns if pd.api.types.is_numeric_dtype(both[c])}
    out["panel_rows"], out["panel_rows_stored"] = int(len(world.panel)), int(len(stored))
    out["panel_max_diff"] = diffs
    out["world_checks"] = {k: (bool(v) if isinstance(v, (bool, np.bool_)) else v) for k, v in world.checks.items()}
    out["passed"] = bool(len(joined) == len(theirs) == len(mine)
                         and max(out[f"max_diff_{c}"] for c in ("offense", "defense")) < REPRO_TOL
                         and not _failed(world.checks))
    return out


def _failed(checks: dict) -> list:
    """The pass/fail checks that failed (the numeric ones are evidence, read by eye and in the ledger)."""
    return [name for name, ok in checks.items() if isinstance(ok, (bool, np.bool_)) and not ok]


# ------------------------------------------------------------------------------------------------- main
def _seasons(text: str) -> list:
    out = []
    for part in text.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def main() -> None:
    check_flags()
    cfg = load_config(ROOT / "config.yaml")
    models_name = flag("models")
    base_name = flag("base")
    if not models_name or not base_name:
        raise SystemExit("--models=<name> (62's --save_models) and --base=<table> (62's --out, the same run)")
    seasons = _seasons(flag("seasons", "2017-2026"))
    repeats, n_folds = int(flag("repeats", 3)), int(flag("folds", 4))
    # rest: rate from every fold but one and test on that one (3/4 of a season at four folds); one: rate from
    # ONE fold and test on the rest (1/4 at four folds) -- the owner, 2026-10-04: the split test "at different
    # sizes".  Ratings from different folds of one deal share no game only under `one` (or at two folds).
    rate_from = flag("rate_from", "rest")
    assert rate_from in ("rest", "one"), "--rate_from=rest|one"
    # `deadline` (experiment 39, the owner 2026-10-05: an in-season check that weighs traded players): fold 1 = the
    # regular-season games before the date by which `--cut` of them had been played, fold 0 = the rest of the regular
    # season and the playoffs.  With two folds and `rest`, fold 0 rates from the early games and scores the late ones
    # (a player traded in between is rated on his old team and scored on his new one); fold 1 is the reverse.
    split = flag("split", "random")
    cut_share = float(flag("cut", "0.6"))
    assert split in ("random", "deadline"), "--split=random|deadline"
    assert split == "random" or (n_folds == 2 and repeats == 1 and rate_from == "rest"), \
        "--split=deadline needs --folds=2 --repeats=1 --rate_from=rest"
    tag = flag("tag", "within")
    check_only = flag("check_only", "0") not in ("0", "no", "false")
    out_dir = ROOT / "outputs" / "within" / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    saved_all = pd.read_pickle(ROOT / "outputs" / f"prior_models_{models_name}.pkl")
    meta = saved_all.get("meta", {})
    # the settings the incumbent's steps 1-3 were built at (HANDOFF: the year-over-year build)
    # --booster_params=<name> (the LightGBM rematch, 2026-10-08): models saved from a build that named this settings
    # file, so the prior shrink can be refitted on the candidate's own priors; absent = the incumbent's boosters
    # --features=<set> (experiment 45, adopted 2026-10-08: the prior without the on-court inputs, boruta_noonc)
    expect = dict(exclude_neighbours=1, features=flag("features", "boruta"), rows="chunks", chunk_label="career",
                  unshrink=["defense"], player_folds=5, booster_params=flag("booster_params"), crossfit="scale")
    wrong = {k: (meta.get(k), v) for k, v in expect.items() if meta.get(k) != v}
    if wrong:
        raise SystemExit(f"the saved models were not built at the incumbent's settings: {wrong}")
    base = pd.read_parquet(ROOT / "outputs" / f"{base_name}.parquet")
    # gated like every reader that fits from the panel: a season still being played is never rated here
    panel, _ = drop_untrainable(pd.read_parquet(ROOT / "outputs" / "role_panel_season.parquet"), cfg,
                                what="the season panel")
    roles = pd.read_parquet(ROOT / "data" / "cache" / "roles_RSPO.parquet")
    inputs = player_season_inputs(roles, cap=float(cfg.get("roles", {}).get("share_cap", 0.9)))
    # the ridge at the board's shipped settings (62's main sets these from its defaults)
    B62.FREE_PRIOR_SCALE, B62.LAM_BUCKETS = True, {}
    B62.BOARD_PLAYER_LAMBDAS = B62.BOARD_OFFENSE_LAMBDAS = B62.BOARD_DEFENSE_LAMBDAS = np.array([PENALTY])
    B62.BOARD_CONTEXT_LAMBDAS = np.array([0.0])
    ctx = Context.load(cfg)
    ctx.cache_size = 4
    print(f"models {models_name} ({meta.get('argv')}); seasons {seasons[0]}-{seasons[-1]}; {repeats} repeats of "
          f"{n_folds} folds; writing {out_dir.relative_to(ROOT)}", flush=True)

    t0 = time.time()
    for season in seasons:
        if season not in saved_all:
            raise SystemExit(f"no saved models for {season}")
        saved = saved_all[season]
        unseen = set(saved["O"]["unseen"]) & set(saved["D"]["unseen"])
        # the prior models were never fit on this season or either neighbour (62 --exclude_neighbours=1)
        assert {season - 1, season, season + 1} <= unseen, unseen
        full = {"pts": ctx.design([season], SCORE_TARGET), "xpts_ft": ctx.design([season], "xpts_ft")}
        for name in ("x3def", "x3def_w0.25"):
            full[name] = ctx.design([season], DEFENSE_TARGETS[name])
        tables = SeasonTables(season, cfg)
        games = full["pts"].games
        id_of_idx = pd.Series(games["game_id"].astype(str).to_numpy(), index=games["game_idx"].to_numpy())

        # -- the reproduction check: the whole season through this script must be the board's row
        repro_path = out_dir / f"reproduce_{season}.json"
        if not repro_path.exists():
            world = World(season, cfg, full, tables, inputs, roles, fit_ids=set(id_of_idx))
            world.rate(saved)
            result = reproduce(world, base, panel, saved)
            repro_path.write_text(json.dumps(result, indent=1))
            world.player_table().assign(season=season, repeat=-1, fold=-1).to_parquet(
                out_dir / f"players_{season}_all.parquet", index=False)
            worst = sorted(result["panel_max_diff"].items(), key=lambda kv: -kv[1])[:4]
            print(f"  {season} reproduction: rating max diff {result['max_diff_offense']:.1e} / "
                  f"{result['max_diff_defense']:.1e}, raw prior {result['max_diff_raw_prior_O']:.1e} / "
                  f"{result['max_diff_raw_prior_D']:.1e}; {result['joined']} of {result['board_players']} players; "
                  f"largest input differences {[(k, f'{v:.1e}') for k, v in worst]}  "
                  f"({time.time() - t0:.0f}s)", flush=True)
            if not result["passed"]:
                raise SystemExit(f"{season}: the whole-season rebuild does not reproduce the board; see "
                                 f"{repro_path.relative_to(ROOT)}")
            del world
        if check_only:
            continue

        # -- the folds
        folds_path = out_dir / f"folds_{season}.parquet"
        if split == "deadline":
            dealt = deadline_folds(games, cut_share).to_frame("r0")
        else:
            dealt = pd.concat([deal_folds(id_of_idx.to_numpy(), n_folds, season, r) for r in range(repeats)], axis=1)
        if folds_path.exists():
            old = pd.read_parquet(folds_path)
            common = [c for c in old.columns if c in dealt.columns]
            assert old[common].equals(dealt.loc[old.index, common]), "the stored folds differ from the dealt ones"
        dealt.to_parquet(folds_path)
        for r in range(repeats):
            for k in range(n_folds):
                stem = f"{season}_r{r}_f{k}"
                if (out_dir / f"fold_{stem}.json").exists():
                    done = json.loads((out_dir / f"fold_{stem}.json").read_text())["summary"]
                    if done.get("rate_from", "rest") != rate_from:
                        raise SystemExit(f"{stem} in {out_dir.name} was built with --rate_from="
                                         f"{done.get('rate_from', 'rest')}; one folder holds one mode")
                    continue
                t1 = time.time()
                fold_ids = set(dealt.index[dealt[f"r{r}"] == k])
                test_ids = fold_ids if rate_from == "rest" else set(dealt.index) - fold_ids
                fit_ids = set(dealt.index) - test_ids
                test_idx = np.sort(games.loc[games["game_id"].astype(str).isin(test_ids), "game_idx"].to_numpy())
                world = World(season, cfg, full, tables, inputs, roles, fit_ids=fit_ids)
                world.rate(saved)
                pooled, teams, test_poss, summary, checks = test_games(world, full["pts"], test_idx, cfg)
                checks.update(world.checks)
                checks["folds: fit and test games disjoint"] = not (fit_ids & test_ids)
                checks["folds: fit and test games cover the season"] = (fit_ids | test_ids) == set(id_of_idx)
                checks["prior models: never trained on this season or its neighbours"] = (
                    {season - 1, season, season + 1} <= unseen)
                failed = _failed(checks)
                table = (world.player_table().merge(test_poss, on="player_id", how="left")
                         .fillna({"test_poss_off": 0.0, "test_poss_def": 0.0})
                         .assign(season=season, repeat=r, fold=k))
                table.to_parquet(out_dir / f"players_{stem}.parquet", index=False)
                save_test(out_dir / f"test_{stem}.npz", pooled, teams)
                summary.update(season=season, repeat=r, fold=k, rate_from=rate_from, fit_games=len(fit_ids),
                               split=split, cut=cut_share if split == "deadline" else None,
                               test_games=len(test_ids),
                               players=int((world.ratings.possessions > 0).sum()),
                               prior_scale_off=float(world.ratings.prior_scale_off.iloc[0]),
                               prior_scale_def=float(world.ratings.prior_scale_def.iloc[0]),
                               seconds=round(time.time() - t1, 1))
                (out_dir / f"fold_{stem}.json").write_text(json.dumps(
                    dict(summary=summary, checks={k_: (bool(v) if isinstance(v, (bool, np.bool_)) else v)
                                                  for k_, v in checks.items()}), indent=1))
                print(f"  {stem}: {summary['fit_games']} fit / {summary['test_games']} test games, "
                      f"{summary['players']} players; test error {summary['game_armse']:.3f} (base "
                      f"{armse(summary['tg_base']):.3f}); test scales {summary['scale_off']:.2f} / "
                      f"{summary['scale_def']:.2f}; prior scale {summary['prior_scale_off']:.2f} / "
                      f"{summary['prior_scale_def']:.2f}; checks {len(checks) - len(failed)} of {len(checks)} "
                      f"passed ({summary['seconds']:.0f}s, {time.time() - t0:.0f}s total)", flush=True)
                if failed:
                    raise SystemExit(f"{stem}: leak checks failed: {failed}")
                del world, pooled
        ctx._cache.clear()

    # -- gather
    folds = sorted(out_dir.glob("fold_*.json"))
    if folds:
        loaded = [json.loads(p.read_text()) for p in folds]
        pd.DataFrame([x["summary"] for x in loaded]).to_parquet(out_dir / "summary.parquet", index=False)
        # a check is either pass/fail or a measured size; one column each, or the file cannot hold them
        checks = pd.DataFrame([dict(season=x["summary"]["season"], repeat=x["summary"]["repeat"],
                                    fold=x["summary"]["fold"], check=k,
                                    passed=v if isinstance(v, bool) else None,
                                    measure=None if isinstance(v, bool) else float(v))
                               for x in loaded for k, v in x["checks"].items()])
        checks.to_parquet(out_dir / "checks.parquet", index=False)
        failed = checks[checks.passed.eq(False)]
        print(f"checks: {int(checks.passed.notna().sum())} pass/fail checks over {len(loaded)} folds, "
              f"{len(failed)} failed")
        print(f"\nwrote {out_dir.relative_to(ROOT)}/summary.parquet and checks.parquet: {len(loaded)} folds")


if __name__ == "__main__":
    main()
