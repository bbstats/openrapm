"""The season panel: what the box-score prior reads, one row per player, season and side.

Words used below
    season panel  one row per player, season and side (offence "O", defence "D"): his possessions on that side, his
                  padded box-score rates, role, score state, shot totals, career, body and team, and his team's record
                  with him on and off the court.  `features.SEASON_PANEL_COLUMNS` lists the columns in order and
                  features.py says what each one means
    design        one season's stints as the rows of a regression (design.py): the ten players on the court, the
                  points per 100 possessions they explain, the possessions as the weight; one design per target
    target        what a design's rows explain: points with expected free throws (`OFFENCE`), points with every
                  opponent three repriced at the shooter's padded three-point percentage (`DEFENCE_ALL`), plain points
    fit games     the games a row may be built from: every game of the season, or, for `openrapm folds`, three
                  quarters of them

`build_panel` writes paths.panel (outputs/season_panel.parquet): `season_panel` for every season, from all of its
games.  `season_panel(..., fit_game_ids)` builds the same rows from part of a season, every input from those games
alone (commands/within_season.py); handed every game it gives the rows `build_panel` writes.

Where each column comes from:
    possessions, the 13 rates and their `_uncentred` twins     box_rates.box_rates on the offence design
    team_possession_share, games_started_share, age            the roles table (data/roles.py), per player-season
    garbage_time_share, closeness, mean_abs_margin             `score_state`, from the stints
    the shot totals and the league levels beside them          data/shots.py, player_shot_frame
    seasons_before, career_possessions_before_k, age_at_first_season    data/roles.py, career_inputs
    height_in, weight_lb, draft_pick, seasons_with_team, teams_this_season    data/bio.py, player_inputs
    on court, off court, on minus off                          on_court.py, on the offence design (scored) and the
                                                               DEFENCE_ALL design (allowed)
    playoff_share                                              `playoff_share`, from the stints

Example: Stephen Curry's 2016 offence row reads possessions 6,994 (his team's offensive possessions with him on the
court), assists_per100 4.25 above the average player (8.84 uncentred), team_possession_share 0.66 and playoff_share
0.18.  2026 has 582 players, so 1,164 rows.

Every fit reads the panel through `trainable_panel`, which first drops every row of a season whose Finals are not over
(`seasons.drop_untrainable`), so no prior trains on a season still being played; step 5 reads it through
`panel_and_trainable_seasons`.  `build_panel` itself reads and writes every season, the one in progress included.

`migrate_panel` wrote the panel once, on 2026-10-11, from the one the retired scripts/49_role_panel.py (with 65, 69
and 86) wrote under the names used before then, outputs/role_panel_season.parquet: every column renamed
(data/legacy_names.py), the columns no reader uses dropped, the rest in `SEASON_PANEL_COLUMNS` order, the rows in
theirs (`openrapm data panel --migrate-from outputs/role_panel_season.parquet`).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd

from ..box_rates import box_rates, rate_table
from ..design import AWAY_SLOTS, HOME_SLOTS, WindowData
from ..features import (
    BIO,
    BOX_RATES,
    CAREER,
    CLOSENESS,
    OFF_COURT,
    ON_COURT,
    ON_MINUS_OFF,
    PLAYOFF_SHARE,
    ROLE,
    SEASON_PANEL_COLUMNS,
    SHOT_LEAGUE,
    SHOT_TOTALS,
)
from ..on_court import offcourt_rates, oncourt_rates
from .bio import PLAYER_INPUTS, player_inputs
from .config import resolve
from .designs import DesignCache
from .ingest import GAME_PREFIX, game_table, load_gamelog, raw_dir
from .legacy_names import is_deleted, translate_frame
from .roles import (
    CAREER_INPUTS,
    RAW_INPUTS,
    ROLE_PHASES,
    build_roles,
    career_inputs,
    player_season_inputs,
    roles_from_boxes,
    season_ages,
    shares_from_stints,
    window_inputs,
)
from .seasons import drop_untrainable
from .shots import DEFENSE_TARGETS, SHOT_LEAGUE_COLS, SHOT_TOTAL_COLS, player_shot_frame

__all__ = ["OFFENCE", "DEFENCE_ALL", "PANEL_TARGETS", "SeasonTables", "roles_from_games", "score_state",
           "playoff_share", "cut_design", "fit_designs", "SeasonSources", "SeasonPanel", "season_panel",
           "fill_missing_score_state", "build_panel", "write_panel", "trainable_panel", "panel_and_trainable_seasons",
           "panel_from_old_names", "migrate_panel"]

OFFENCE = "points_with_expected_free_throws"
DEFENCE_ALL = "points_with_expected_opponent_threes_all"
PANEL_TARGETS = (OFFENCE, DEFENCE_ALL)            # the designs one season's rows are built from
SIDES = ("O", "D")
OLD_PANEL = "role_panel_season.parquet"          # the old-named panel's file name, beside the new one
STINT_COLUMNS = ["game_id", "period", *HOME_SLOTS, *AWAY_SLOTS, "poss_h", "poss_a", "margin_h", "is_gt"]
# The data layer's names for the role, career, body and shot columns, against the panel's (the pairs are pinned in
# tests/test_features.py and tests/test_panel.py).
ROLE_NAMES = dict(zip(RAW_INPUTS, ROLE))
CAREER_NAMES = dict(zip(CAREER_INPUTS, CAREER))
BIO_NAMES = dict(zip(PLAYER_INPUTS, BIO))
SHOT_NAMES = dict(zip([*SHOT_TOTAL_COLS, *SHOT_LEAGUE_COLS], [*SHOT_TOTALS, *SHOT_LEAGUE]))
# lineup slot -> (his team's possessions, the opponent's possessions)
SCORE_SLOTS = ([(f"h{i}", "poss_h", "poss_a") for i in range(1, 6)]
               + [(f"a{i}", "poss_a", "poss_h") for i in range(1, 6)])


def share_cap(cfg: dict) -> float:
    """config.yaml's roles.share_cap: team_possession_share stops at one full team-season of possessions (0.9)."""
    return float(cfg.get("roles", {}).get("share_cap", 0.9))


# ------------------------------------------------------------------------------------------ one season's games
class SeasonTables:
    """One season's per-game tables (regular season and playoffs): the stints and each game's two teams, read at once;
    the box scores and the ages, read when first asked for (only rebuilding the role rows from part of the season,
    `roles_from_games`, needs them)."""

    def __init__(self, season: int, cfg: dict):
        self.season, self.cfg = int(season), cfg
        stints, games = [], []
        for phase in ROLE_PHASES:
            path = resolve(cfg, "stints") / f"{season}_{phase}.parquet"
            if path.exists():
                stints.append(pd.read_parquet(path, columns=STINT_COLUMNS).assign(phase=phase))
                games.append(game_table(load_gamelog(season, phase, cfg)))
        self.stints = pd.concat(stints, ignore_index=True)
        self.stints["game_id"] = self.stints["game_id"].astype(str)
        self.games = pd.concat(games, ignore_index=True)
        self.games["game_id"] = self.games["game_id"].astype(str)

    @cached_property
    def boxes(self) -> dict:
        """{game_id: the game's box score (teamId, personId, minutes)}, regular season then playoffs."""
        box_dir = raw_dir(self.cfg) / "box" / str(self.season)
        return {path.stem: pd.read_parquet(path, columns=["teamId", "personId", "minutes"])
                for phase in ROLE_PHASES for path in sorted(box_dir.glob(f"{GAME_PREFIX[phase]}*.parquet"))}

    @cached_property
    def ages(self) -> pd.DataFrame:
        """player_id, season, age from the bio feed (data/roles.py, season_ages)."""
        return season_ages(self.season, self.cfg)


def roles_from_games(tables: SeasonTables, keep_ids: set) -> tuple:
    """The season's role rows (`roles.season_roles` and `build_roles`' age join) from the games `keep_ids` only: the
    same arithmetic on fewer games, so the team's possessions behind team_possession_share are its possessions in
    those games.  Returns (role rows, the box-score games read, the stint games read)."""
    boxes = {game: box for game, box in tables.boxes.items() if game in keep_ids}
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


def score_state(stints: pd.DataFrame, season: int) -> pd.DataFrame:
    """Per player and side, the score state his possessions came in (DECISIONS.md, experiment 14):

        garbage_time_share  share of his possessions in garbage time (config.yaml garbage_time)
        closeness           possession-weighted mean of 1 / max(|margin|, 1)
        mean_abs_margin     possession-weighted mean |margin|

    weighted on offence by his team's possessions, on defence by the opponent's.

    Example: 600 possessions with 120 in garbage time give garbage_time_share 0.2.
    """
    st = stints.copy()
    st["abs_margin_"] = st["margin_h"].abs()
    st["closeness_"] = 1.0 / np.maximum(st["abs_margin_"], 1.0)
    st["is_gt_"] = st["is_gt"].astype(float)
    rows = []
    for slot, own, opponent in SCORE_SLOTS:
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
    g["garbage_time_share"] = g["w_gt"] / g["poss"]
    g["mean_abs_margin"] = g["w_margin"] / g["poss"]
    g["closeness"] = g["w_close"] / g["poss"]
    return g[["player_id", "season", "side", *CLOSENESS]]


def playoff_share(stints: pd.DataFrame, games: pd.DataFrame) -> pd.Series:
    """player_id -> the share of his possessions on the court that came in the playoffs: 1 - regular-season
    possessions / all possessions, summed over his teams.

    Example: 2,400 possessions with 2,100 in the regular season give 1 - 2100 / 2400 = 0.125.
    """
    both = shares_from_stints(stints, games)
    regular = shares_from_stints(stints[stints["phase"] == "RS"], games)
    both = both[both.poss_on > 0].groupby("player_id").poss_on.sum()
    regular = regular[regular.poss_on > 0].groupby("player_id").poss_on.sum()
    return (1.0 - regular.reindex(both.index).fillna(0.0) / both).clip(0.0, 1.0).rename(PLAYOFF_SHARE[0])


# ---------------------------------------------------------------------------------- the designs of the fit games
def cut_design(design: WindowData, keep_idx: np.ndarray) -> WindowData:
    """The design restricted to the games `keep_idx` (game_idx values): its rows AND its per-game box-score and
    possession tables.

    `WindowData.subset` keeps the whole season's box and possession tables, and those are what the padded rates and
    the possessions are built from, so a fit on part of the season would still read every game's box score (the leak
    FINDINGS 31 found, closed the same way: the tables are cut with the rows).  `games` (ids, dates, phases: no
    outcomes) is kept whole so game_idx still indexes it."""
    keep = np.isin(design.rows["game_idx"].to_numpy(), keep_idx)
    sub = design.subset(keep)
    return WindowData(sub.X_src, sub.y, sub.w, sub.groups, sub.spec,
                      design.game_box[design.game_box["game_idx"].isin(keep_idx)].reset_index(drop=True),
                      design.game_poss[design.game_poss["game_idx"].isin(keep_idx)].reset_index(drop=True),
                      sub.rows, design.games, sub.counters)


def fit_designs(season: int, cfg: dict, whole: dict, fit_game_ids) -> dict:
    """{target: the design of the fit games}, one per target of `whole` (the season's designs, "pts" among them).

    "pts" and the other point targets are cut to the fit games (`cut_design`); a defensive target is rebuilt on the cut
    points design, the shooters' three-point percentages that reprice the opponents' threes counted from the fit games
    too (FINDINGS 31's x3def row)."""
    games = whole["pts"].games
    fit_ids = {str(game) for game in fit_game_ids}
    fit_idx = np.sort(games.loc[games["game_id"].astype(str).isin(fit_ids), "game_idx"].to_numpy())
    points = cut_design(whole["pts"], fit_idx)
    keep = {int(season): sorted(fit_ids)}
    out = {}
    for target, design in whole.items():
        if target in DEFENSE_TARGETS:
            out[target] = DEFENSE_TARGETS[target]([int(season)], cfg, points, keep=keep)[0]
        else:
            out[target] = points if target == "pts" else cut_design(design, fit_idx)
    return out


# ----------------------------------------------------------------------------------------- one season's rows
@dataclass
class SeasonSources:
    """What a season's rows are read from besides its designs."""
    tables: SeasonTables          # the season's stints, games and (on first use) box scores
    roles: pd.DataFrame           # the roles table, every season (data/roles.py, build_roles)
    inputs: pd.DataFrame          # player_season_inputs of the roles table


@dataclass
class SeasonPanel:
    """One season's panel rows and what they were built from."""
    season: int
    rows: pd.DataFrame            # SEASON_PANEL_COLUMNS
    roles: pd.DataFrame           # the season's role rows, one per player and team (from the fit games)
    games_read: dict = field(default_factory=dict)     # {what: the game ids it read}, for the folds' leak checks


@dataclass
class _RoleHistory:
    roles: pd.DataFrame           # the roles table with the season's rows from the fit games
    inputs: pd.DataFrame          # player_season_inputs of it
    season_roles: pd.DataFrame
    games_read: dict


def _role_history(season: int, sources: SeasonSources, fit: set | None, cap: float) -> _RoleHistory:
    """The roles table and its per-season inputs as the season's rows see them: as stored for every game, or with the
    season's rows rebuilt from the fit games (every other season as stored)."""
    if fit is None:
        return _RoleHistory(sources.roles, sources.inputs, sources.roles[sources.roles.season == season], {})
    rebuilt, box_games, stint_games = roles_from_games(sources.tables, fit)
    own = rebuilt[list(sources.roles.columns)]
    roles = pd.concat([sources.roles[sources.roles.season != season], own], ignore_index=True)
    inputs = pd.concat([sources.inputs[sources.inputs.season != season], player_season_inputs(own, cap=cap)],
                       ignore_index=True)
    return _RoleHistory(roles, inputs, rebuilt, {"role box scores": box_games, "role stints": stint_games})


def _player_columns(season: int, cfg: dict, designs: dict, history: _RoleHistory, fit: set | None) -> pd.DataFrame:
    """The columns that are the same on both of a player's rows, one row per design column: role, shot totals, on and
    off court, career, body and team."""
    offence, defence = designs[OFFENCE], designs[DEFENCE_ALL]
    players = offence.spec.ps_table["player_id"].to_numpy()
    role = window_inputs(offence, history.inputs, cap=share_cap(cfg)).rename(columns=ROLE_NAMES)
    shots = player_shot_frame([season], cfg, players, keep=None if fit is None else {season: sorted(fit)})
    frame = pd.DataFrame({"player_id": players})
    for name in ROLE:
        frame[name] = role[name].to_numpy()
    for old, name in SHOT_NAMES.items():
        frame[name] = shots[old].to_numpy(dtype=float)
    frame = pd.concat([frame, oncourt_rates(offence, defence), offcourt_rates(offence, defence)], axis=1)
    frame[ON_MINUS_OFF[0]] = frame[ON_COURT[0]] - frame[OFF_COURT[0]]
    frame[ON_MINUS_OFF[1]] = frame[ON_COURT[1]] - frame[OFF_COURT[1]]
    career = career_inputs(history.inputs, season, players, age=frame["age"].to_numpy())
    body = player_inputs(cfg, history.roles, [season], players)
    for old, name in {**CAREER_NAMES, **BIO_NAMES}.items():
        frame[name] = (career if old in CAREER_NAMES else body)[old].to_numpy(dtype=float)
    return frame


def _side_rows(side: str, season: int, rates, players: pd.DataFrame) -> pd.DataFrame:
    """A side's rows: the keys, his possessions and padded rates on that side, then the columns both sides share."""
    possessions = (rates.offence if side == "O" else rates.defence).possessions
    keys = pd.DataFrame({"side": side, "player_id": players["player_id"].to_numpy(), "season": season,
                         "possessions": possessions})
    return pd.concat([keys, rate_table(rates, side, BOX_RATES), players.drop(columns="player_id")], axis=1)


def season_panel(season: int, cfg: dict, designs: dict, sources: SeasonSources, fit_game_ids=None) -> SeasonPanel:
    """One season's panel rows, every input from the fit games: all of the season's games when `fit_game_ids` is None.

    `designs` holds the season's `OFFENCE` and `DEFENCE_ALL` designs of the same games (`fit_designs` cuts them to
    part of a season).  The rows: offence then defence, each in the design's player order; `SEASON_PANEL_COLUMNS`.
    """
    season = int(season)
    fit = None if fit_game_ids is None else {str(game) for game in fit_game_ids}
    history = _role_history(season, sources, fit, share_cap(cfg))
    rates = box_rates(designs[OFFENCE], pad_target=cfg["pad_target"])
    players = _player_columns(season, cfg, designs, history, fit)
    rows = pd.concat([_side_rows(side, season, rates, players) for side in SIDES], ignore_index=True)
    stints = sources.tables.stints
    if fit is not None:
        stints = stints[stints["game_id"].isin(fit)]
    rows = rows.merge(score_state(stints, season), on=["player_id", "season", "side"], how="left")
    rows[PLAYOFF_SHARE[0]] = rows["player_id"].map(playoff_share(stints, sources.tables.games)).fillna(0.0)
    read = {**history.games_read, "score-state stints": set(stints["game_id"].unique())}
    return SeasonPanel(season, rows[SEASON_PANEL_COLUMNS], history.season_roles, read)


# -------------------------------------------------------------------------------------------- every season's rows
def fill_missing_score_state(panel: pd.DataFrame) -> pd.DataFrame:
    """A row with no possessions on its side has no score state; it takes the whole panel's mean, as the published
    panel did (scripts/69_closeness_panel.py, retired).  Such a row weighs nothing in any fit.

    Example: in the published panel one row, a 2010 offence row with no offensive possessions, reads the means
    0.140, 0.309 and 9.54."""
    for column in CLOSENESS:
        panel[column] = panel[column].fillna(panel[column].mean())
    return panel


def build_panel(cfg: dict, seasons=None) -> pd.DataFrame:
    """Every season's rows from all of its games (`season_panel`), in season order, the score state filled
    (`fill_missing_score_state`).  The roles table is read from paths.cache, or built there first (data/roles.py)."""
    seasons = (range(int(cfg["first_season"]), int(cfg["last_season"]) + 1) if seasons is None else seasons)
    roles = build_roles(cfg)
    inputs = player_season_inputs(roles, cap=share_cap(cfg))
    designs = DesignCache(cfg)
    parts = []
    for season in seasons:
        started = time.time()
        whole = {target: designs.season(int(season), target) for target in PANEL_TARGETS}
        parts.append(season_panel(int(season), cfg, whole, SeasonSources(SeasonTables(season, cfg), roles,
                                                                           inputs)).rows)
        designs.clear()
        print(f"  {season}: {len(parts[-1])} rows ({time.time() - started:.1f}s)", flush=True)
    return fill_missing_score_state(pd.concat(parts, ignore_index=True))


def write_panel(cfg: dict, force: bool = False, seasons=None) -> tuple[Path, pd.DataFrame]:
    """Build the panel (`build_panel`) and write it to paths.panel: (where, the panel).  An existing panel is kept
    unless `force`, and the check comes before the build, which takes minutes."""
    target = resolve(cfg, "panel")
    if target.exists() and not force:
        raise FileExistsError(f"{target} exists; pass --force to rebuild it, or point paths.panel at a new file")
    panel = build_panel(cfg, seasons)
    target.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(target, index=False)
    return target, panel


# ---------------------------------------------------------------------------------------------- reading the panel
def _read(cfg: dict) -> pd.DataFrame:
    """The season panel as written, refused with the command to run when only the old-named one exists."""
    path = resolve(cfg, "panel")
    old = path.with_name(OLD_PANEL)
    if not path.exists() and old.exists():
        raise FileNotFoundError(f"{path} does not exist; the panel on disk is {old}, under the names used before "
                                f"2026-10-11.  Run `openrapm data panel --migrate-from {old}` once.")
    panel = pd.read_parquet(path)
    if "poss" in panel.columns and "possessions" not in panel.columns:
        raise ValueError(f"{path} carries the names used before 2026-10-11; point paths.panel at a new file and "
                         f"run `openrapm data panel --migrate-from {path}`")
    return panel


def trainable_panel(cfg: dict) -> pd.DataFrame:
    """The season panel without the seasons nothing may be fitted on (none while no season is in progress)."""
    panel, _ = drop_untrainable(_read(cfg), cfg, what="the season panel")
    return panel


def panel_and_trainable_seasons(cfg: dict) -> tuple[pd.DataFrame, set]:
    """(the whole season panel, the seasons a fit may train on): every season is read, only finished ones train."""
    panel = _read(cfg)
    trainable, _ = drop_untrainable(panel, cfg, what="the season panel (the swap type model's training)")
    return panel, set(trainable.season.astype(int))


# ------------------------------------------------------------------------------------ the one-off migration
def panel_from_old_names(old: pd.DataFrame) -> pd.DataFrame:
    """A panel built under the old names, under the plain ones: renamed, the unread columns dropped, the columns in
    `SEASON_PANEL_COLUMNS` order, the rows and every value as they were.

    Raises if a column of `SEASON_PANEL_COLUMNS` is missing or a column is left that the table does not know, so
    a migrated panel never silently lacks an input or carries one under an old name.
    """
    renamed = translate_frame(old)
    missing = [column for column in SEASON_PANEL_COLUMNS if column not in renamed.columns]
    unknown = [column for column in renamed.columns if column not in SEASON_PANEL_COLUMNS]
    if missing or unknown:
        raise ValueError(f"the old panel does not map onto the season panel: missing {missing}, unknown {unknown}")
    return renamed[SEASON_PANEL_COLUMNS]


def migrate_panel(source: Path, cfg: dict) -> tuple[Path, pd.DataFrame, list]:
    """Write `paths.panel` from the old-named panel at `source`: (where, the panel, the old columns dropped).

    Written once: an existing season panel is never overwritten (delete it to migrate again).
    """
    target = resolve(cfg, "panel")
    if target.exists():
        raise FileExistsError(f"{target} exists already; the migration writes it once (delete it to migrate again)")
    old = pd.read_parquet(source)
    panel = panel_from_old_names(old)
    target.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(target, index=False)
    return target, panel, [column for column in old.columns if is_deleted(column)]
