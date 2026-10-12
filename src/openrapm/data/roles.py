"""Role inputs of the season panel: the share of team possessions played, starts, and age.

Three inputs per player-season, all from data already on disk or one request away (the season panel calls them
team_possession_share, games_started_share and age; data/panel.py):

  poss_pct the player's on-floor possessions (both ends) summed over every team he played for,
           divided by ONE full team-season of possessions (the mean of his teams' season totals).
           The denominator counts the games he did not play, so injury lowers it; a player
           traded mid-season keeps his full-season figure because the numerator is summed, not
           averaged.  Capped at `share_cap` (0.9).
  gs_pct   games started / games played.  Starters are each team's first five rows in the V3 box
           file (the rule stint_parser.GameParser already relies on); played = minutes > 0.
  age      from nba_api LeagueDashPlayerBioStats, one request per season, cached under
           data/raw/bio/.  A missing age gets the season median and `age_imputed` = 1.

`build_roles` writes data/cache/roles_RSPO.parquet (one row per player-season-team, regular
season and playoffs pooled -- the name carries the phases so an older RS-only table cannot be read
back as this one); `window_inputs` blends the per-season inputs down to the design's player columns
the way box_rates.py blends its padding constants (possession-weighted over the player's seasons in
the design); `career_inputs` counts what a player had behind him before a season.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..design import AWAY_SLOTS, HOME_SLOTS

# The phases a player's role is measured over: `design.SEASON_PHASES`, imported by name so there is one
# place the answer lives.  Owner's ruling, 2026-09-10: one rating per player per season from that season's
# games, "regular season and playoffs together, nothing else".
from ..design import SEASON_PHASES as ROLE_PHASES
from .config import DEFAULT_PATHS, resolve
from .ingest import GAME_PREFIX, TIMEOUT, _retry, game_table, load_gamelog, raw_dir, refuse_if_offline, season_str

# The role and career inputs as this module names them.  The season panel calls them team_possession_share,
# games_started_share, age, and seasons_before, career_possessions_before_k, age_at_first_season
# (features.ROLE and features.CAREER; data/legacy_names.py maps one onto the other).
RAW_INPUTS = ["poss_pct", "gs_pct", "age"]
CAREER_INPUTS = ["exp_yrs", "exp_poss", "entry_age"]

# One franchise, two team ids: the Charlotte Hornets moved to New Orleans after 2001-02 and took a new id
# with them (1610612740 from 2003); from 2005 1610612766 is the Bobcats, now the Hornets, and is itself.
# Unmapped, a player who stayed through the move looks like he changed teams: 12 of Charlotte's 15 in
# 2002.  Every other team id in 1997-2026 covers all thirty seasons.  (old id, first, last season) -> id.
FRANCHISE_MOVES = {(1610612766, 1997, 2002): 1610612740}


def franchise(team_id, season) -> np.ndarray:
    """Team ids as franchises: `FRANCHISE_MOVES` applied, everything else unchanged."""
    team = np.array(team_id, dtype=np.int64)
    season = np.asarray(season)
    for (old, first, last), new in FRANCHISE_MOVES.items():
        team[(team == old) & (season >= first) & (season <= last)] = new
    return team


def parse_minutes(s) -> float:
    """'24:06' -> 24.1; '' / None / NaN -> 0.0."""
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return 0.0
    s = str(s).strip()
    if not s:
        return 0.0
    if ":" in s:
        m, sec = s.split(":", 1)
        try:
            return float(m) + float(sec) / 60.0
        except ValueError:
            return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0


def roles_from_boxes(boxes: dict) -> pd.DataFrame:
    """Per (player_id, team_id): games played, games started, minutes, from V3 box frames keyed by game_id.

    Each team's first five rows in a V3 box are its starters (position is not a starter flag).
    """
    parts = []
    for gid, b in boxes.items():
        if len(b) == 0:
            continue
        b = b[["teamId", "personId", "minutes"]].copy()
        b["minutes"] = b["minutes"].map(parse_minutes)
        b["rank"] = b.groupby("teamId").cumcount()
        b["start"] = (b["rank"] < 5).astype(float)
        b["played"] = (b["minutes"] > 0).astype(float)
        parts.append(b[["teamId", "personId", "played", "start", "minutes"]])
    if not parts:
        return pd.DataFrame(columns=["player_id", "team_id", "games", "starts", "minutes"])
    d = pd.concat(parts, ignore_index=True)
    g = d.groupby(["personId", "teamId"], as_index=False).agg(games=("played", "sum"), starts=("start", "sum"),
                                                              minutes=("minutes", "sum"))
    return g.rename(columns={"personId": "player_id", "teamId": "team_id"}).astype(
        {"player_id": np.int64, "team_id": np.int64})


def shares_from_stints(stints: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Per (player_id, team_id): on-floor possessions (both ends), and the team's season total.

    `stints`: game_id, h1..h5, a1..a5, poss_h, poss_a.  `games`: game_id, home_team_id, away_team_id.
    """
    st = stints.merge(games[["game_id", "home_team_id", "away_team_id"]], on="game_id", how="inner")
    both = (st["poss_h"].to_numpy(dtype=float) + st["poss_a"].to_numpy(dtype=float))
    parts = []
    for slots, team_col in ((HOME_SLOTS, "home_team_id"), (AWAY_SLOTS, "away_team_id")):
        team = st[team_col].to_numpy()
        for c in slots:
            parts.append(pd.DataFrame({"player_id": st[c].to_numpy(), "team_id": team, "poss_on": both}))
    p = pd.concat(parts, ignore_index=True).groupby(["player_id", "team_id"], as_index=False)["poss_on"].sum()
    team_poss = pd.concat([pd.DataFrame({"team_id": st["home_team_id"].to_numpy(), "poss": both}),
                           pd.DataFrame({"team_id": st["away_team_id"].to_numpy(), "poss": both})],
                          ignore_index=True).groupby("team_id", as_index=False)["poss"].sum().rename(
        columns={"poss": "team_poss"})
    return p.merge(team_poss, on="team_id", how="left").astype({"player_id": np.int64, "team_id": np.int64})


def _box_dir(season: int, cfg) -> Path:
    return raw_dir(cfg) / "box" / str(season)


def season_roles(season: int, cfg, phases: tuple = ROLE_PHASES) -> pd.DataFrame:
    """One season: player_id, season, team_id, games, starts, minutes, poss_on, team_poss.

    Pooled over `phases`.  A playoff game is a game: it is a start, minutes, and possessions on the floor
    for both the player and his team, so `poss_pct` and `gs_pct` stay shares and a deep run does not
    inflate them.  This was `("RS",)` while the playoffs were a separate delta block.
    """
    bdir = _box_dir(season, cfg)
    boxes, st_parts, gm_parts = {}, [], []
    for ph in phases:
        for f in sorted(bdir.glob(f"{GAME_PREFIX[ph]}*.parquet")):
            boxes[f.stem] = pd.read_parquet(f, columns=["teamId", "personId", "minutes"])
        sp = Path(resolve(cfg, "stints")) / f"{season}_{ph}.parquet"
        if sp.exists():
            st_parts.append(pd.read_parquet(sp, columns=["game_id", *HOME_SLOTS, *AWAY_SLOTS, "poss_h", "poss_a"]))
            gm_parts.append(game_table(load_gamelog(season, ph, cfg)))
    roles = roles_from_boxes(boxes)
    stints = pd.concat(st_parts, ignore_index=True)
    games = pd.concat(gm_parts, ignore_index=True)
    games["game_id"] = games["game_id"].astype(str)
    stints["game_id"] = stints["game_id"].astype(str)
    shares = shares_from_stints(stints, games)
    out = roles.merge(shares, on=["player_id", "team_id"], how="outer")
    out[["games", "starts", "minutes", "poss_on"]] = out[["games", "starts", "minutes", "poss_on"]].fillna(0.0)
    team_poss = shares.drop_duplicates("team_id").set_index("team_id")["team_poss"]
    out["team_poss"] = out["team_poss"].fillna(out["team_id"].map(team_poss))
    out.insert(1, "season", int(season))
    return out


def season_ages(season: int, cfg, force: bool = False) -> pd.DataFrame:
    """player_id, season, age from LeagueDashPlayerBioStats (cached data/raw/bio/{season}_RS.parquet)."""
    path = raw_dir(cfg) / "bio" / f"{season}_RS.parquet"
    if path.exists() and not force:
        df = pd.read_parquet(path)
    else:
        refuse_if_offline(path, f"the bio stats {season}")
        from nba_api.stats.endpoints import leaguedashplayerbiostats

        def pull():
            d = leaguedashplayerbiostats.LeagueDashPlayerBioStats(
                season=season_str(season), season_type_all_star="Regular Season", timeout=TIMEOUT).get_data_frames()[0]
            if len(d) == 0:
                raise ValueError("empty bio stats")
            return d
        df = _retry(pull, what=f"bio {season}")
        df = df.astype({c: str for c in df.columns if df[c].dtype == object})
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
    out = pd.DataFrame({"player_id": df["PLAYER_ID"].astype(np.int64), "season": int(season),
                        "age": pd.to_numeric(df["AGE"], errors="coerce")})
    return out.drop_duplicates("player_id")


def build_roles(cfg, seasons=None, force: bool = False, verbose: bool = True,
                phases: tuple = ROLE_PHASES) -> pd.DataFrame:
    """Every season's roles + ages -> data/cache/roles_RSPO.parquet (one row per player-season-team)."""
    path = roles_path(cfg, phases)
    if path.exists() and not force:
        return pd.read_parquet(path)
    seasons = list(range(int(cfg["first_season"]), int(cfg["last_season"]) + 1)) if seasons is None else list(seasons)
    parts = []
    for s in seasons:
        r = season_roles(s, cfg, phases)
        a = season_ages(s, cfg)
        r = r.merge(a, on=["player_id", "season"], how="left")
        miss = float(r.age.isna().mean()) if len(r) else 0.0
        parts.append(r)
        if verbose:
            print(f"  roles {s}: {r.player_id.nunique()} players, {len(r)} player-team rows, "
                  f"age missing {miss:.1%}", flush=True)
    out = pd.concat(parts, ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    return out


def roles_path(cfg, phases: tuple = ROLE_PHASES) -> Path:
    """The roles table: `roles.cache` (a file name) inside `paths.cache`, or `roles.cache` itself when it is an
    absolute path, so a config that moves the cache folder moves this table with it.

    The cache carries its phases in its name, so an RS-only table built before the playoffs joined the
    fit cannot be read back as an RS+PO one -- it would be a silent, plausible-looking undercount."""
    p = Path(cfg.get("roles", {}).get("cache", "roles.parquet"))
    p = p.with_name(f"{p.stem}_{''.join(phases)}{p.suffix}")
    if p.is_absolute():
        return p
    return resolve(cfg, "cache") / p if "_root" in cfg else Path(DEFAULT_PATHS["cache"]) / p


def main_teams(cfg) -> pd.DataFrame:
    """player_id, season, team_id: the team each player played the most possessions for that season.

    Read from the roles table (`roles_path`), one row per player, season and team: his possessions on the court
    for each team are summed and the largest kept.
    """
    roles = pd.read_parquet(roles_path(cfg), columns=["player_id", "season", "team_id", "poss_on"])
    return (roles[roles.poss_on > 0].groupby(["player_id", "season", "team_id"], as_index=False).poss_on.sum()
            .sort_values("poss_on").drop_duplicates(["player_id", "season"], keep="last")
            [["player_id", "season", "team_id"]].astype(np.int64))


def player_season_inputs(roles: pd.DataFrame, cap: float = 0.9) -> pd.DataFrame:
    """Per player-season: poss_pct (summed over teams / one team-season, capped), gs_pct, age (+ age_imputed)."""
    r = roles.copy()
    g = r.groupby(["player_id", "season"], as_index=False).agg(
        games=("games", "sum"), starts=("starts", "sum"), minutes=("minutes", "sum"), poss_on=("poss_on", "sum"),
        team_poss=("team_poss", "mean"), age=("age", "first"))
    g["poss_pct"] = np.where(g.team_poss > 0, g.poss_on / np.where(g.team_poss > 0, g.team_poss, 1.0),
                             0.0).clip(0.0, cap)
    g["gs_pct"] = np.where(g.games > 0, g.starts / np.where(g.games > 0, g.games, 1.0), 0.0).clip(0.0, 1.0)
    med = g.groupby("season")["age"].transform("median")
    g["age_imputed"] = g.age.isna().astype(int)
    g["age"] = g.age.fillna(med).fillna(float(g.age.median()) if g.age.notna().any() else 27.0)
    return g[["player_id", "season", "games", "starts", "minutes", "poss_on", "team_poss", "poss_pct", "gs_pct", "age",
              "age_imputed"]]


def career_inputs(inputs: pd.DataFrame, before_season: int, player_ids=None, age=None) -> pd.DataFrame:
    """Per player, what he had behind him BEFORE `before_season`: seasons played, possessions (thousands) and
    the age he entered at.

    `before_season` is the BLOCK's first season, not the row's, so the quantity is a property of the training
    block alone -- for a held-out season H at K = 3 the block is (H-2, H-1, H+1) and this counts only seasons
    up to H-3, which no covariate measured on H can reach.  A season with no games played does not count.

    The roles table starts in 1997, so everyone in the first window is left-censored to zero years: `entry_age`
    (the age in his first season SEEN, which for a censored veteran is his 1997 age, not 19) is what lets the
    tree tell a real rookie from a censored one, and `season` is a feature beside it.

    `player_ids` returns one row per id in that order; a player the past never saw has zero years, zero
    possessions and `entry_age` = his age now (`age`, aligned to the same ids), because he is entering now.
    """
    d = inputs[(inputs.season < int(before_season)) & (inputs.games > 0)]
    if len(d):
        g = d.groupby("player_id").agg(exp_yrs=("season", "nunique"), exp_poss=("poss_on", "sum"))
        g["entry_age"] = d.sort_values("season").groupby("player_id")["age"].first()
        g["exp_poss"] = g.exp_poss / 1000.0
        g = g[CAREER_INPUTS]
    else:
        g = pd.DataFrame(columns=CAREER_INPUTS, dtype=float)
    if player_ids is None:
        return g.rename_axis("player_id").reset_index()
    out = g.reindex(np.asarray(player_ids)).reset_index(drop=True)
    out["exp_yrs"] = out.exp_yrs.fillna(0.0).astype(float)
    out["exp_poss"] = out.exp_poss.fillna(0.0).astype(float)
    fill = (float(inputs.age.median()) if age is None else
            pd.Series(np.asarray(age, dtype=float), index=out.index))
    out["entry_age"] = out.entry_age.fillna(fill).astype(float)
    return out


def window_inputs(wd, inputs: pd.DataFrame, cap: float = 0.9) -> pd.DataFrame:
    """The per-season inputs blended to the design's Z unit, aligned to wd.spec.ps_table.

    Possession-weighted over the player's seasons in the window (RS offensive possessions from
    wd.game_poss, the weights the design itself uses); a player-season with no roles row takes the
    season's possession-weighted league mean and is counted in .attrs["n_missing"].
    """
    spec = wd.spec
    psx = spec.psx_table[["psx_idx", "player_id", "season", "ps_idx"]].copy()
    gp = wd.game_poss
    rs = wd.games.loc[wd.games["phase"] == "RS", "game_idx"].to_numpy()
    gp = gp[np.isin(gp["game_idx"], rs)]
    w_psx = np.zeros(len(psx))
    np.add.at(w_psx, gp["psx_idx"].to_numpy(), gp["poss_off"].to_numpy(dtype=float))
    psx["w"] = w_psx
    m = psx.merge(inputs[["player_id", "season", *RAW_INPUTS]], on=["player_id", "season"], how="left")
    missing = m["poss_pct"].isna()
    for c in RAW_INPUTS:
        lm = m.groupby("season").apply(
            lambda d: np.average(d[c].fillna(0.0), weights=np.where(d[c].notna(), d.w, 0.0))
            if (d[c].notna() & (d.w > 0)).any() else d[c].mean(), include_groups=False)
        m[c] = m[c].fillna(m["season"].map(lm)).fillna(float(m[c].mean()) if m[c].notna().any() else 0.0)
    m["poss_pct"] = m["poss_pct"].clip(0.0, cap)
    n_ps = spec.n_ps
    out = np.zeros((n_ps, len(RAW_INPUTS)))
    den = np.zeros(n_ps)
    cnt = np.zeros((n_ps, len(RAW_INPUTS)))
    n = np.zeros(n_ps)
    vals = m[RAW_INPUTS].to_numpy(dtype=float)
    idx = m["ps_idx"].to_numpy()
    ww = m["w"].to_numpy(dtype=float)
    np.add.at(out, idx, vals * ww[:, None])
    np.add.at(den, idx, ww)
    np.add.at(cnt, idx, vals)
    np.add.at(n, idx, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        blend = np.where(den[:, None] > 0, out / np.where(den > 0, den, 1.0)[:, None],
                         cnt / np.maximum(n, 1.0)[:, None])
    res = pd.DataFrame(blend, columns=RAW_INPUTS)
    res.insert(0, "ps_idx", np.arange(n_ps))
    res.insert(1, "player_id", spec.ps_table["player_id"].to_numpy())
    res.attrs["n_missing"] = int(missing.sum())
    res.attrs["n_psx"] = int(len(m))
    return res
