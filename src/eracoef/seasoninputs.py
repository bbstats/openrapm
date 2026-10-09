"""One season rebuilt from any subset of its games: every input the rating reads, from those games alone.

Moved here from scripts/97_within_season.py (the Robustness pass, 2026-10-09) so the board itself can use it:
`honest_fold_prior` gives `priorridge.PriorRidgeCV` a prior for each of its cross-fitting folds whose EVERY input --
the padded box rates and their padding constants, playing time and starts, shot totals, on- and off-court numbers --
comes from the fold's training games.  62's own fold builder rebuilds only the on-court, off-court and RAPM-piece
columns, and since experiment 45 the prior reads none of them, so its "cross-fitted" scale was priced on box-score
columns that had seen the games it was regressed on (4-12% too large; DECISIONS.md, "The prior's scale, priced in
sample: measured").  97 keeps using the same class: its whole-season version reproduces the shipped ratings exactly.

Data layer: it reads the season's box scores and stints (`SeasonTables`).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import singleyear as sy
from .bio import PLAYER_INPUTS, player_inputs
from .config import resolve
from .design import AWAY_SLOTS, FEATURES, HOME_SLOTS, WindowData
from .ingest import GAME_PREFIX, game_table, load_gamelog, raw_dir
from .investigate import offcourt_rates, oncourt_rates
from .roles import (CAREER_INPUTS, RAW_INPUTS, ROLE_PHASES, career_inputs, player_season_inputs,  # noqa: F401
                    roles_from_boxes, season_ages, shares_from_stints, window_inputs)
from .spm import apm_fit, season_of_units
from .windows import window_label
from .xshoot import DEFENSE_TARGETS, SHOT_LEAGUE_COLS, SHOT_TOTAL_COLS, player_shot_frame

__all__ = ["SHOT_COLS_ALL", "CLOSE_SLOTS", "cut_design", "SeasonTables", "roles_from_games", "closeness_rows",
           "playoff_share", "SeasonWorld", "honest_fold_prior"]

SHOT_COLS_ALL = [*SHOT_TOTAL_COLS, *SHOT_LEAGUE_COLS]
CLOSE_SLOTS = ([(f"h{i}", "poss_h", "poss_a") for i in range(1, 6)]
               + [(f"a{i}", "poss_a", "poss_h") for i in range(1, 6)])



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


class SeasonWorld:
    """Season H rated from the games `fit_ids`, with every input the rating reads rebuilt from those games.

    There is ONE path.  The whole season is this same class handed every game, and the reproduction check holds it
    to the board's own table -- so a passing check is a check on the part-season code, not on a second copy."""

    def __init__(self, season: int, cfg, full: dict, tables: SeasonTables, inputs: pd.DataFrame,
                 roles: pd.DataFrame, fit_ids):
        self.season, self.cfg = int(season), cfg
        # kept so a world can rebuild a smaller world inside its own games (the honest cross-fitted scale)
        self.full, self.tables, self.inputs, self.roles = full, tables, inputs, roles
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


# ------------------------------------------------------------------------------------- the honest fold prior
def honest_fold_prior(world_args: tuple, rows_game_idx: np.ndarray, models: dict, model_feats: dict,
                      prior_full: dict):
    """`fold_prior(train_mask) -> (offence priors, defence priors)` for `PriorRidgeCV.fit`: the season rebuilt from the
    games of the rows where `train_mask` is True (a `SeasonWorld` on them), the same fitted boosters asked about every
    rebuilt row, each player answered by the fold model that never saw him.  A player with no game in those games keeps
    his full-season prior.  `world_args` = (season, cfg, full designs, SeasonTables, role inputs, roles);
    `rows_game_idx` is the game_idx of each row of the design the ridge is fitted on.  Folds are cached, so the two
    ridges of a season (one per target, the same rows and folds) build each fold once."""
    season, cfg, full, tables, inputs, roles = world_args
    id_of_idx = pd.Series(full["pts"].games["game_id"].astype(str).to_numpy(),
                          index=full["pts"].games["game_idx"].to_numpy())
    for side in ("O", "D"):
        reads = [c for c in model_feats[side] if c in sy.PIECES or c == sy.SAME_TEAM]
        assert not reads, f"the honest fold prior does not rebuild {reads}"
    built: dict = {}

    def fold_prior(train_mask):
        mask = np.asarray(train_mask, dtype=bool)
        key = np.packbits(mask).tobytes()
        if key not in built:
            fit_ids = set(id_of_idx.loc[np.unique(rows_game_idx[mask])])
            world = SeasonWorld(season, cfg, full, tables, inputs, roles, fit_ids=fit_ids)
            out = []
            for side in ("O", "D"):
                feats = [f for f in model_feats[side] if f not in sy.CHUNK_FEATURES]
                rows = world.panel[(world.panel.side == side) & (world.panel.poss > 0)]
                held = sy.season_frame(rows, feats)
                held = held.assign(chunk_poss=held.poss.to_numpy(float), chunk_seasons=1.0)
                got = dict(zip(held.player_id.to_numpy(), models[side].predict(held, model_feats[side])))
                out.append({p: got.get(p, v) for p, v in prior_full[side].items()})
            built[key] = tuple(out)
        return built[key]

    return fold_prior
