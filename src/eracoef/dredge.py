"""The Dredge counters: per (player, season) event counts read straight out of the play-by-play.

The 13 box rates are a nightly summary.  Every event behind them is on disk in
`data/raw/pbp/{season}/{game_id}.parquet`, and Justin Willard's Dredge (Nylon Calculus, 2016 -- an
elastic net onto 15-year RAPM, trained 2001-2015 and tested on 1997-2000 and 2016) says the summary
throws away most of what the events know:

  * blocks "in their pure form are highly overrated" (his coefficient 0.236) while a **Russell** --
    a block the defence recovers -- is worth 0.445, and a **rim block** 0.523, and blocking a three
    "did not test well".  Our own SHAP table gives `blk` 15.7% on defense, the single largest
    defensive feature, which is exactly the term he says is the wrong shape.
  * **unassisted** field goals are worth more than assisted ones (0.841 on UnAstShot%): "players get
    credit for assists, they should get more credit for unassisted shots".
  * **technicals and flagrants** carry a POSITIVE coefficient (+1.25), "a proxy for feisty defenders
    and guys who fight hard in the paint", and loose-ball fouls about +0.33.
  * a **stolen** turnover is roughly twice as costly as any other kind.

This module counts those events; `gbdt_prior.add_dredge` turns the counts into padded features and
`scripts/56_dredge.py` builds and caches the per-season tables.

WHAT OUR FEED CAN ATTRIBUTE.  `data/raw/pbp` is the v3 play-by-play: exactly ONE `personId` per
event.  So a counter is buildable here when the player we want is the one the row names.  Audited on
1997, 1999, 2001, 2003, 2006, 2010, 2015, 2019, 2023 and 2026 (`scratch/dredge_audit*.py`):

  BLOCK and STEAL are their own rows with the blocker / stealer named, in every season, and a BLOCK
  row's own `shotValue` says whether the blocked attempt was a two or a three (100% of them).  The
  blocked attempt is ALWAYS the row directly above (112 of 112 in 1997, 114 of 114 in 2026), so its
  distance is readable; the rebound that follows is the next `Rebound` row, and whether it belongs to
  the blocker's team is what makes the block a Russell.  A STEAL always sits directly beside its
  `Turnover` row, whose `personId` is the player who lost it.  `Foul` rows name the committer, which
  is what the loose-ball, technical, flagrant and offensive-foul terms want.  The file's row order is
  chronological in every era (zero clock inversions in the audit), so "the row above" is meaningful.

WHAT IT CANNOT.  **Offensive fouls DRAWN** -- Dredge's best find, coefficient 1.22 -- name only the
fouler (`"Asik OFF.Foul (P3)"`) in 1997 and 2026 alike.  That needs the v2 feed or pbpstats, i.e. an
ingest job, and is deliberately not here.  Offensive fouls COMMITTED are, and are the stepping stone.

TWO ERA TRAPS, both handled rather than assumed away:

  * `Foul / Offensive Charge` does not exist as a subType before ~2006; charges sit inside
    `Foul / Offensive`.  A column that is structurally zero for a third of the panel is learned as
    "old era", not as "none happened", and `season` is a feature.  So `foul_off` is the AGGREGATE of
    every `Offensive*` subtype and the modern split is not built -- which is also what Dredge found
    to be the better feature ("non-charges tested MORE valuable than charges").
  * `Violation / Defensive Goaltending` is counted, but Justin's footnote says 1997 has suspiciously
    FEW goaltends while our feed says it has MORE (1.27 a game against 0.43 now).  One of the two is
    wrong and a feature with a spurious era trend is the FINDINGS 22.2 failure mode wearing a hat, so
    `goaltend` is counted, reported per season by `scripts/56_dredge.py`, and left OUT of the
    default feature list until somebody reconciles it against a published source.

Everything is per (player, season) over the REGULAR SEASON games the stints were built from -- the
same games, so `poss_off` / `poss_def` (summed from the stints' own lineup slots, the way
`design.build_game_poss` does it) are the honest denominators for a per-100 rate.  A block's totals
are the sum over its seasons: `player_dredge_frame` is the `xshoot.player_shot_frame` of this block
and is the single source used by the panel build (`scripts/49_role_panel.py`) and by prediction
(`spm.chain_offset`) alike.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import re
import unicodedata

from .config import resolve
from .design import AWAY_SLOTS, HOME_SLOTS
from .shotcurve import shot_bin
from .stints import RIM_FT

# NBA team ids are 1610612737-1610612766.  Player ids are far below that and always will be (the
# newest are ~1.64e6), so this is how a team row -- a team rebound, a team technical -- is told from a
# player row.  Team rebounds carry the team in `personId` with `teamId` 0, which is why both are read.
TEAM_LO, TEAM_HI = 1_610_612_700, 1_610_612_800

# NBA shot-chart coordinates, tenths of a foot, (0, 0) at the basket.  The corner-three line is the straight
# segment at |x| = 22 ft running out to y = 9.25 ft, where the arc begins.
CORNER_X, CORNER_Y = 220.0, 92.5
# a two at or inside RIM_FT is the rim, out to SHORT_FT is the short mid-range, beyond it the long one
SHORT_FT = 13.0
ZONES = ["rim", "smr", "lmr", "c3", "ab3"]
AST_RX = re.compile(r"\(([^()]*?) (\d+) AST\)")
_SUFFIX = re.compile(r"\b(jr|sr|ii|iii|iv|v)\.?$")


def _norm(name: str) -> str:
    """A surname in a form both the description and the roster agree on: no accents, no suffix, no case.

    The description writes "Doncic" where `playerName` writes "Dončić", and drops "III" and "Jr.".  Both
    sides go through this, so neither convention has to be the right one.
    """
    t = unicodedata.normalize("NFKD", str(name))
    t = "".join(c for c in t if not unicodedata.combining(c)).lower().strip()
    t = _SUFFIX.sub("", t).strip().rstrip(".").strip()
    return re.sub(r"\s+", " ", t)

COUNTERS = [
    "fgm_unast",    # made field goals with no assist in the description
    "fgm_ast",      # ... and with one (the pair sums to his made field goals: a check against the box)
    "tov_all",      # his turnovers
    "tov_stolen",   # ... of which a STEAL row sits beside
    "foul_off",     # Foul / Offensive*  (charges included, never split: see the era trap above)
    "blk",          # BLOCK rows
    "blk_rus",      # ... a Russell: the next rebound belongs to the blocker's team
    "blk_rim",      # ... the blocked attempt was a two at or inside RIM_FT (shotcurve's own binning)
    "blk_3",        # ... the blocked attempt was a three ("did not test well", so it is a share to net out)
    "stl",          # STEAL rows
    "foul_loose",   # Foul / Loose Ball
    "foul_tech",    # any Technical or Flagrant subtype
    "goaltend",     # Violation / Defensive Goaltending  (counted, NOT in the default features)
    # assists credited to the PASSER, split by where the shot he created came from.  `ast_res` is how many
    # of his assists resolved at all, so the coverage is a column and not a footnote.
    "ast_res", "ast_rim", "ast_smr", "ast_lmr", "ast_c3", "ast_ab3",
    # and blocks by the same zones (blk_rim and blk_3 are already above; these complete the split)
    "blk_smr", "blk_lmr",
]
POSS = ["poss_off", "poss_def"]
DREDGE_COLS = [*COUNTERS, *POSS]


def shot_zone(shot_value: float, dist: float, x: float, y: float, desc: str) -> str | None:
    """The zone of one field-goal attempt, or None when a two has no usable location.

    Twos go through `shotcurve.shot_bin`, which is the project's own binning and is what reads a
    description for the rim when 1997 records a layup at distance 0.  Threes split on the corner line,
    which needs the coordinates rather than the distance -- a corner three is 22 ft and an above-break one
    23.75, and the feed drops `shotDistance` on some of them (FINDINGS 23.3).
    """
    if float(shot_value) == 3.0:
        return "c3" if (abs(float(x)) >= CORNER_X and float(y) <= CORNER_Y) else "ab3"
    b = shot_bin(float(dist), False, str(desc))
    if b < 0:
        return None
    return "rim" if b <= RIM_FT else ("smr" if b <= SHORT_FT else "lmr")


# ------------------------------------------------------------------------------------ one game
def game_counts(pbp: pd.DataFrame) -> pd.DataFrame:
    """`player_id` x `COUNTERS` for one game's play-by-play, in the file's own (chronological) row order."""
    df = pbp.reset_index(drop=True)
    n = len(df)
    at = df["actionType"].astype(str).str.strip()
    sub = df["subType"].astype(str).str.strip()
    desc = df["description"].astype(str)
    pid = pd.to_numeric(df["personId"], errors="coerce").fillna(0).astype(np.int64).to_numpy()
    tid = pd.to_numeric(df["teamId"], errors="coerce").fillna(0).astype(np.int64).to_numpy()
    sv = pd.to_numeric(df["shotValue"], errors="coerce").fillna(0).astype(float).to_numpy()
    sd = (pd.to_numeric(df["shotDistance"], errors="coerce").fillna(-1.0).astype(float).to_numpy()
          if "shotDistance" in df.columns else np.full(n, -1.0))
    is_team = (pid >= TEAM_LO) & (pid < TEAM_HI)
    player = (pid > 0) & ~is_team
    # the team a row belongs to: `teamId` when it holds one, else `personId` (team rebounds swap them)
    row_team = np.where((tid >= TEAM_LO) & (tid < TEAM_HI), tid, np.where(is_team, pid, 0))
    atv, subv, descv = at.to_numpy(), sub.to_numpy(), desc.to_numpy()

    hits: list[tuple[np.ndarray, str]] = []

    made = (atv == "Made Shot") & player
    assisted = desc.str.contains(r"\d+ AST", regex=True).to_numpy()
    hits.append((pid[made & assisted], "fgm_ast"))
    hits.append((pid[made & ~assisted], "fgm_unast"))

    # the assist goes to the PASSER, named only by surname in the shooter's description, and is filed under
    # the zone of the shot he created
    def _num(col):
        """A numeric column, or zeros when the frame does not carry it (1997 has every field, but the
        unit tests build the minimum shape and a missing column must not be an exception)."""
        if col not in df.columns:
            return np.zeros(n)
        return pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy(dtype=float)

    xl, yl = _num("xLegacy"), _num("yLegacy")
    namev = df["playerName"].astype(str).str.strip().to_numpy()
    # the roster, keyed on BOTH name forms the feed carries: `playerName` is the bare surname and
    # `playerNameI` the initial-plus-surname one, which is what tells two Williamses apart
    initv = (df["playerNameI"].astype(str).str.strip().to_numpy() if "playerNameI" in df.columns
             else np.full(n, "", dtype=object))
    roster: dict = {}
    for p_, t_, nm, ini in zip(pid, row_team, namev, initv):
        if not (0 < p_ < TEAM_LO):
            continue
        for form in (nm, ini):
            if form and str(form) != "nan":
                roster.setdefault((int(t_), _norm(form)), set()).add(int(p_))
    by_zone: dict = {z: [] for z in ZONES}
    resolved = []
    for i in np.flatnonzero(made & assisted):
        m = AST_RX.search(descv[i])
        if m is None:
            continue
        who, team = _norm(m.group(1)), int(row_team[i])
        hit = roster.get((team, who)) or set()
        if len(hit) != 1:                     # the bare surname behind an initial the roster does not carry
            tail = who.split(". ")[-1].split(" ")[-1]
            cand = {p_ for (t_, nm), ps in roster.items() if t_ == team and nm.split(" ")[-1] == tail
                    for p_ in ps}
            hit = cand if len(cand) == 1 else hit
        if len(hit) != 1:                     # still two of the same name: leave it out rather than guess
            continue
        who_id = next(iter(hit))
        resolved.append(who_id)
        z = shot_zone(sv[i], sd[i], xl[i], yl[i], descv[i])
        if z is not None:
            by_zone[z].append(who_id)
    hits.append((np.asarray(resolved, dtype=np.int64), "ast_res"))
    for z, v in by_zone.items():
        hits.append((np.asarray(v, dtype=np.int64), f"ast_{z}"))

    is_tov = (atv == "Turnover") & player
    hits.append((pid[is_tov], "tov_all"))

    is_foul = atv == "Foul"
    hits.append((pid[is_foul & player & pd.Series(subv).str.startswith("Offensive").to_numpy()], "foul_off"))
    hits.append((pid[is_foul & player & (subv == "Loose Ball")], "foul_loose"))
    tech = pd.Series(subv).str.contains("Technical|Flagrant", regex=True).to_numpy()
    hits.append((pid[is_foul & player & tech], "foul_tech"))

    hits.append((pid[(atv == "Violation") & player & (subv == "Defensive Goaltending")], "goaltend"))

    # blocks: the blocked attempt is the row above, the rebound that decides a Russell the next one below
    is_blk = desc.str.contains("BLOCK", regex=False).to_numpy() & player
    is_stl = desc.str.contains("STEAL", regex=False).to_numpy() & player
    hits.append((pid[is_stl], "stl"))
    if is_blk.any():
        is_reb = atv == "Rebound"
        nxt_reb = np.full(n + 1, -1, dtype=np.int64)        # nxt_reb[i] = the first Rebound row at or after i
        for i in range(n - 1, -1, -1):
            nxt_reb[i] = i if is_reb[i] else nxt_reb[i + 1]
        b_all, b_rus, b_rim, b_3 = [], [], [], []
        b_smr, b_lmr = [], []
        for i in np.flatnonzero(is_blk):
            b_all.append(pid[i])
            three = float(sv[i]) == 3.0
            if three:
                b_3.append(pid[i])
            elif i > 0 and atv[i - 1] == "Missed Shot":
                z = shot_zone(2.0, sd[i - 1], 0.0, 0.0, descv[i - 1])
                if z == "rim":
                    b_rim.append(pid[i])
                elif z == "smr":
                    b_smr.append(pid[i])
                elif z == "lmr":
                    b_lmr.append(pid[i])
            j = nxt_reb[i + 1] if i + 1 <= n else -1
            if j >= 0 and row_team[j] == row_team[i] and row_team[i] != 0:
                b_rus.append(pid[i])
        for v, name in ((b_all, "blk"), (b_rus, "blk_rus"), (b_rim, "blk_rim"), (b_3, "blk_3"),
                        (b_smr, "blk_smr"), (b_lmr, "blk_lmr")):
            hits.append((np.asarray(v, dtype=np.int64), name))
    if is_stl.any():
        st = []
        for i in np.flatnonzero(is_stl):
            for j in (i - 1, i + 1):                          # the audit says it is always one of the two
                if 0 <= j < n and is_tov[j]:
                    st.append(pid[j])
                    break
        hits.append((np.asarray(st, dtype=np.int64), "tov_stolen"))

    ids = np.unique(np.concatenate([h for h, _ in hits if len(h)] or [np.zeros(0, dtype=np.int64)]))
    out = pd.DataFrame({"player_id": ids})
    for c in COUNTERS:
        out[c] = 0.0
    idx = {int(p): k for k, p in enumerate(ids)}
    for arr, name in hits:
        if not len(arr):
            continue
        col = out.columns.get_loc(name)
        v = out.iloc[:, col].to_numpy()
        np.add.at(v, np.array([idx[int(p)] for p in arr], dtype=np.int64), 1.0)
        out.iloc[:, col] = v
    return out


# ------------------------------------------------------------------------------------ one season
def stint_possessions(season: int, cfg) -> tuple[pd.DataFrame, np.ndarray]:
    """Per player offensive and defensive possessions of one regular season, and the season's game ids.

    Exactly `design.build_game_poss`'s arithmetic (a home slot's player is on offense for `poss_h`),
    summed over the season instead of per game, and read from the same stints file the design is built
    from -- so the possessions and the events below cover the same set of games.
    """
    st = pd.read_parquet(Path(resolve(cfg, "stints")) / f"{season}_RS.parquet",
                         columns=["game_id", *HOME_SLOTS, *AWAY_SLOTS, "poss_h", "poss_a"])
    po, pa = st["poss_h"].to_numpy(dtype=float), st["poss_a"].to_numpy(dtype=float)
    ids, off, dfn = [], [], []
    for slots, own, opp in ((HOME_SLOTS, po, pa), (AWAY_SLOTS, pa, po)):
        for c in slots:
            ids.append(st[c].to_numpy(dtype=np.int64))
            off.append(own)
            dfn.append(opp)
    p = pd.DataFrame({"player_id": np.concatenate(ids), "poss_off": np.concatenate(off),
                      "poss_def": np.concatenate(dfn)})
    p = p[p.player_id > 0].groupby("player_id", as_index=False)[POSS].sum()
    return p, st["game_id"].astype(str).unique()


def build_season(season: int, cfg, verbose: bool = False) -> pd.DataFrame:
    """Count the season's play-by-play into one (player_id x COUNTERS + POSS) frame."""
    poss, game_ids = stint_possessions(season, cfg)
    pbp_dir = Path(resolve(cfg, "raw")) / "pbp" / str(int(season))
    parts, missing = [], 0
    for g in game_ids:
        p = pbp_dir / f"{g}.parquet"
        if not p.exists():
            missing += 1
            continue
        parts.append(game_counts(pd.read_parquet(p)))
    if missing:
        raise FileNotFoundError(f"{missing} of {len(game_ids)} {season} RS play-by-play files are missing "
                                f"under {pbp_dir}; run scripts/01_ingest.py")
    c = pd.concat(parts, ignore_index=True).groupby("player_id", as_index=False)[COUNTERS].sum()
    out = poss.merge(c, on="player_id", how="outer").fillna(0.0)
    out.insert(1, "season", int(season))
    if verbose:
        print(f"  {season}: {len(game_ids)} games, {len(out)} players, "
              f"{out.blk.sum():.0f} blocks ({out.blk_rus.sum() / max(out.blk.sum(), 1):.3f} Russell, "
              f"{out.blk_rim.sum() / max(out.blk.sum(), 1):.3f} rim), "
              f"{out.fgm_unast.sum() / max(out.fgm_unast.sum() + out.fgm_ast.sum(), 1):.3f} unassisted", flush=True)
    return out[["player_id", "season", *DREDGE_COLS]]


def dredge_path(season: int, cfg) -> Path:
    return Path(cfg["_root"]) / "data" / "dredge" / f"{int(season)}_RS.parquet"


_SEASON_CACHE: dict = {}


def season_dredge(season: int, cfg, force: bool = False, verbose: bool = False) -> pd.DataFrame:
    """The cached per-season counter table (built on first use, read once per process)."""
    key = int(season)
    if key in _SEASON_CACHE and not force:
        return _SEASON_CACHE[key]
    path = dredge_path(key, cfg)
    if path.exists() and not force:
        df = pd.read_parquet(path)
    else:
        df = build_season(key, cfg, verbose=verbose)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
    _SEASON_CACHE[key] = df
    return df


# ------------------------------------------------------------------------------------ a block
DREDGE_TOTAL_COLS = [f"dr_{c}" for c in DREDGE_COLS]              # dr_fgm_unast ... dr_poss_def
DREDGE_LEAGUE_COLS = [f"dr_lg_{c}" for c in DREDGE_COLS]          # the same, summed over every player


def player_dredge_frame(seasons, cfg, player_ids=None) -> pd.DataFrame:
    """Per-player regular-season Dredge counts over a block, with the block's own league totals beside them.

    `dr_*` are the player's counts and possessions summed over the block's seasons; `dr_lg_*` are the
    same summed over every player the block saw, constant down the frame, and they are what
    `gbdt_prior.add_dredge` pads toward -- a 300-possession player lands on HIS era's rate, not on the
    thirty-season average, which matters because these events have real era trends (blocks per 100
    possessions have fallen by a third since 1997).

    `player_ids` returns one row per id in that order, zeros for a player the block never saw; without
    it the frame carries a `player_id` column and only the players it did.
    """
    parts = [season_dredge(int(s), cfg) for s in seasons]
    t = pd.concat(parts, ignore_index=True).groupby("player_id")[DREDGE_COLS].sum()
    t.index = t.index.astype(np.int64)
    lg = t.sum(axis=0)
    tot = t.astype(float)
    tot.columns = DREDGE_TOTAL_COLS
    if player_ids is None:
        out = tot.rename_axis("player_id").reset_index()
    else:
        out = tot.reindex(np.asarray(player_ids, dtype=np.int64)).fillna(0.0).reset_index(drop=True)
    for c, v in zip(DREDGE_LEAGUE_COLS, lg[DREDGE_COLS].to_numpy(dtype=float)):
        out[c] = float(v)
    return out
