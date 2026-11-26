"""`openrapm site`: the data behind the published site, docs/.

    openrapm site                    docs/data/ratings.json, the ratings as one file (the ratings page reads
                                     ratings_list.json and decompose_openrapm.json, below)
    openrapm site --bias             also docs/data/bias.json, the Bias page: where OpenRAPM rates players higher or
                                     lower than a consensus of three outside systems, per standard deviation of a
                                     stat, and than the season's team-game results do
    openrapm site --openrapm-decomposition
                                     also docs/data/decompose_openrapm.json, the ratings page's Detail view: every
                                     published rating split into pieces that add up to it; and, cut from it,
                                     docs/data/ratings_list.json, the ratings alone with each player's teams, which
                                     is all the page needs to draw its list
    openrapm site --similarity       also docs/data/similarity/, the similarity page: each player-season's nearest
                                     player-seasons of other players
    openrapm site --power            also docs/data/power.json, the power ratings page: each team's current ratings
                                     times its players' shares of its possessions over its last 10 games

The pages' wording and look are not this command's to change: it writes data.

Words used below
    ratings table   outputs/season_ratings_product.parquet, one row per player per season (`openrapm build`)
    built           the date in a file's meta.built.  ratings.json shows the ratings table's modification date,
                    so re-exporting the same table does not tell readers the ratings changed; the Similarity and
                    Bias files show the day they were computed.  --built sets it for every file written
    consensus       on the Bias page, the weighted mean of three outside systems' ratings (bias.py), whose tables
                    `openrapm data public-ratings` pulls; not data/external/consensus.csv, the adoption check's

Example: `openrapm site --out-dir C:/tmp/site --built 2026-10-09` writes C:/tmp/site/data/ratings.json, whose
meta.built reads 2026-10-09 whatever the ratings table's date.
"""
from __future__ import annotations

import json
import time
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from ..bias import (
    CONSENSUS_WEIGHTS,
    OFF_COURT_PLUS_MINUS,
    ON_COURT_PLUS_MINUS,
    OURS,
    SIDES,
    STATS,
    bias_table,
    compared_players,
    consensus,
    player_stats,
)
from ..data import public_ratings
from ..data.config import resolve, resolve_path
from ..data.designs import DesignCache
from ..data.ingest import raw_dir
from ..data.panel import in_progress_panel, panel_of_season, rookies, trainable_panel
from ..data.power_files import load_knots, team_game_counts
from ..data.roles import main_teams
from ..decomposition import lineups, refit, row_keys, split_games_part
from ..features import FEATURE_SETS, meaning
from ..power_forecast import WEIGHTS_FILE, flat_shares, last_ten_stats, stats_parts, weights_at
from ..prior_shrink import team_split
from ..settings import PRODUCT
from ..similarity import closest, distances, own_season_ranks, player_seasons, similarity_inputs, standardise

RATINGS_TABLE = "season_ratings_product.parquet"
SITE_FOLDER = "docs"


def table_date(path) -> str:
    """A file's modification date in UTC, written 2026-10-09."""
    return pd.Timestamp(Path(path).stat().st_mtime, unit="s", tz="UTC").strftime("%Y-%m-%d")


def today() -> str:
    """Today's date in UTC, written 2026-10-09."""
    return pd.Timestamp.now("UTC").strftime("%Y-%m-%d")


def run_site(cfg: dict, ratings=None, out_dir=None, built: str | None = None, bias: bool = False,
             similarity: bool = False, openrapm_decomposition: bool = False,
             power: bool = False, power_forecast: bool = False) -> int:
    """`openrapm site`: ratings.json always; the similarity files, the OpenRAPM decomposition file, the power ratings
    file and the Bias page's file when asked.

    `ratings` defaults to the ratings table under config paths.outputs and `out_dir` to the repository's docs/.  The
    Bias page's file is written last: it alone needs the outside systems' tables, and a missing one then stops the
    command with every other file already written.
    """
    ratings_path = Path(ratings) if ratings else resolve(cfg, "outputs") / RATINGS_TABLE
    if not ratings_path.exists():
        raise SystemExit(f"{ratings_path} does not exist; `openrapm build` writes it")
    out_dir = Path(out_dir) if out_dir else resolve_path(cfg, SITE_FOLDER)
    write_ratings_json(ratings_path, out_dir / "data" / "ratings.json", built or table_date(ratings_path))
    if similarity:
        write_similarity(cfg, ratings_path, out_dir / "data" / SIMILARITY_FOLDER, built or today())
    if openrapm_decomposition:
        write_openrapm_decomposition(cfg, ratings_path, out_dir / "data", built or table_date(ratings_path))
    if power:
        write_power(cfg, ratings_path, out_dir / "data", built or table_date(ratings_path), forecast=power_forecast)
    if bias:
        write_bias(cfg, ratings_path, out_dir / "data", built or today())
    return 0


# ===================================================================================== ratings.json
# The ratings table's columns and their names in ratings.json: one letter each, because the page downloads the
# whole table.  The NBA id is in the file and the CSV download, not on the page.
RATINGS_FIELDS = {"season": "s", "player_name": "n", "rating_off": "o", "rating_def": "d", "rating_total": "t",
                  "poss_season": "p", "player_id": "id", "wosa": "w"}
# The owner, 2026-10-24: "i want to have a 60 possession minimum, site-wide".  A player-season with fewer offensive
# possessions is rated as before and left off every page and download; the similarity page asks 1,000.
MIN_POSSESSIONS_SHOWN = 60


def ratings_rows(table: pd.DataFrame) -> list[dict]:
    """One row per player-season with 60 or more offensive possessions (`MIN_POSSESSIONS_SHOWN`), ratings at two
    decimals, by season and then best total first.

    Example: {"s": 2026, "n": "Victor Wembanyama", "o": 2.16, "d": 5.3, "t": 7.45, "p": 5398, "id": 1641705,
    "w": 12.63}.
    A missing name is written "".
    """
    missing = [column for column in RATINGS_FIELDS if column not in table.columns]
    if missing:
        raise SystemExit(f"the ratings table lacks {missing}; `openrapm build` writes them all")
    short = table[list(RATINGS_FIELDS)].rename(columns=RATINGS_FIELDS)
    short["n"] = short["n"].fillna("").astype(str)
    short = short[short.p >= MIN_POSSESSIONS_SHOWN].sort_values(["s", "t"], ascending=[True, False])
    return [dict(s=int(r.s), n=r.n, o=round(float(r.o), 2), d=round(float(r.d), 2), t=round(float(r.t), 2),
                 p=int(r.p), id=int(r.id), w=round(float(r.w), 2))
            for r in short.itertuples(index=False)]


# The owner, 2026-10-23: "2027 should not be the DEFAULT on the site until the median player has 250 possessions
# played", and 2026-10-24, on offensive possessions.  The median is of the players the page shows, 60 possessions or
# more (the owner, 2026-11-02: "median player has 250 and site didnt default to 2027", at 266 shown and 217 of all 442).
DEFAULT_SEASON_POSSESSIONS = 250


def default_season(table: pd.DataFrame) -> int:
    """The season the pages open on: the latest whose median player, of those the pages show (`MIN_POSSESSIONS_SHOWN`),
    has 250 or more offensive possessions (poss_season, the pages' p).  Example: on 2026-11-02 the median of the 337
    players 2027 shows has 266, so the pages open on 2027."""
    shown = table[table.poss_season >= MIN_POSSESSIONS_SHOWN]
    medians = shown.groupby("season").poss_season.median()
    ready = medians[medians >= DEFAULT_SEASON_POSSESSIONS]
    return int(ready.index.max() if len(ready) else medians.index.max())


def ratings_document(table: pd.DataFrame, built: str) -> dict:
    """{"meta": the seasons, the season the pages open on, the built date and the number of players shown, "rows":
    `ratings_rows`}."""
    rows = ratings_rows(table)
    meta = dict(seasons=sorted({row["s"] for row in rows}), default_season=default_season(table), built=built,
                n_players=len({row["id"] for row in rows}))
    return dict(meta=meta, rows=rows)


def write_ratings_json(ratings_path: Path, path: Path, built: str) -> None:
    """ratings.json: compact JSON (no spaces), non-ASCII letters escaped, no newline at the end."""
    document = ratings_document(pd.read_parquet(ratings_path), built)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, separators=(",", ":")), encoding="utf-8")
    seasons = document["meta"]["seasons"]
    print(f"wrote {path}: {len(document['rows'])} rows over {len(seasons)} seasons ({seasons[0]}-{seasons[-1]}), "
          f"{path.stat().st_size / 1e6:.1f} MB, from {ratings_path}", flush=True)


# rows whose team the box scores cannot name: under this share of a season's rows (the Detail view's check)
UNPLACED_TOLERANCE = 0.001


class BoxScoreFiles:
    """The per-game box score files under config paths.raw/box/<season>/, each season's teams read once per run."""

    def __init__(self, cfg: dict):
        self.folder = raw_dir(cfg) / "box"
        self._teams: dict = {}

    def files(self, season: int) -> list:
        return sorted((self.folder / str(season)).glob("*.parquet"))

    def teams(self, season: int) -> pd.Series:
        """Every player's team in every game of a season: (game_id, player_id) -> team id."""
        if season not in self._teams:
            frames = []
            for path in self.files(season):
                box = pd.read_parquet(path, columns=["personId", "teamId"])
                frames.append(pd.DataFrame({"game_id": path.stem, "player_id": box.personId.to_numpy(np.int64),
                                            "team_id": box.teamId.to_numpy(np.int64)}))
            table = pd.concat(frames).drop_duplicates(["game_id", "player_id"]) if frames else \
                pd.DataFrame(columns=["game_id", "player_id", "team_id"])
            self._teams[season] = table.set_index(["game_id", "player_id"]).team_id
        return self._teams[season]

    def season_teams(self, season: int) -> dict:
        """Every player's teams in a season, as that season's box scores abbreviate them, in the order he joined
        them: {player_id: "SAC/IND"}.  Only games he played minutes in count, unless he has none."""
        frames = [pd.read_parquet(path, columns=["personId", "teamTricode", "minutes"]).assign(game=n)
                  for n, path in enumerate(self.files(season))]
        if not frames:
            return {}
        box = pd.concat(frames)
        box["played"] = box.minutes.fillna("").astype(str).str.strip().str.strip("0:.").str.len() > 0
        teams = {}
        for player_id, games in box.sort_values("game").groupby("personId", sort=False):
            games = games[games.played] if games.played.any() else games
            teams[int(player_id)] = "/".join(dict.fromkeys(games.teamTricode.astype(str)))
        return teams

    def names(self, ids, seasons) -> pd.Series:
        """The last resort, for a player the ratings table and the bio files both lack (Luca Vildoza played only
        in the 2022 playoffs): his name as the span's box scores print it."""
        wanted, found = {int(i) for i in ids}, {}
        for season in seasons:
            for path in self.files(season):
                box = pd.read_parquet(path, columns=["personId", "firstName", "familyName"])
                for row in box[box.personId.isin(wanted - set(found))].itertuples(index=False):
                    found[int(row.personId)] = f"{row.firstName} {row.familyName}".strip()
                if wanted <= set(found):
                    return pd.Series(found, dtype=object)
        return pd.Series(found, dtype=object)


def name_table(ratings_path: Path, cfg: dict) -> pd.DataFrame:
    """player_id, season, player_name: the ratings table first, then the bio files for the players it lacks."""
    names = pd.read_parquet(ratings_path, columns=["player_id", "season", "player_name"]) \
        .dropna(subset=["player_name"])
    bio = [pd.read_parquet(path, columns=["PLAYER_ID", "PLAYER_NAME"]).assign(season=int(path.stem[:4]))
           for path in sorted((raw_dir(cfg) / "bio").glob("*_RS.parquet"))]
    if bio:
        bio = pd.concat(bio).rename(columns={"PLAYER_ID": "player_id", "PLAYER_NAME": "player_name"}).dropna()
        names = pd.concat([names, bio[~bio.player_id.isin(names.player_id)]], ignore_index=True)
    return names.sort_values("season")


def span_names(names: pd.DataFrame, ids: np.ndarray, seasons, box_files: BoxScoreFiles) -> np.ndarray:
    """Each player's name as of the span's last season he has one in; any season if the span has none; the box
    scores if no table has him."""
    inside = names[names.season.isin(seasons)].drop_duplicates("player_id", keep="last").set_index("player_id")
    anywhere = names.drop_duplicates("player_id", keep="last").set_index("player_id")
    got = inside.player_name.reindex(ids).fillna(anywhere.player_name.reindex(ids))
    if got.isna().any():
        got = got.fillna(box_files.names(ids[got.isna().to_numpy()], seasons).reindex(ids))
    return got.to_numpy(object)


def check(ok: bool, what: str) -> None:
    """Print a check; stop the run on a failure, so nothing is written past one."""
    print(("  PASS  " if ok else "  FAIL  ") + what, flush=True)
    if not ok:
        raise SystemExit(f"check failed: {what}")


# ===================================================================================== the OpenRAPM decomposition page
# Every published rating split, on each side, into pieces that add up to it exactly:
#
#     rating = box score + Box Adj + on court + off court + team SOS
#
#   box score        what the box score says: the prior without its Box Adj, after step 4's multipliers
#   Box Adj          the part of the prior learned from lineup swaps, which only moves credit between teammates (the
#                    table's box_adj_off, box_adj_def), and step 5's adjustment, 0 since experiment 82 (c_off, c_def)
#   on court         the part of the season's games' part (u, how far step 3 moved him from his prior, times step 4's
#                    factor on it since experiment 65) that comes from the possessions he was on the court for
#   off court        the part from his team's possessions without him: in games he played and in games he missed
#   team SOS         the part from every other possession
#
# The total is the sum of the two sides' pieces.  Box score and Box Adj come from the ratings table's columns; u is
# split by solving step 3 again for each season and each side's target on the table's own priors (decomposition.py
# says why that splits u exactly), and the run stops before writing unless the refit reproduces every u and every
# prior scale in the table (to 1e-9; to 1e-6 in 2020, whose solve needs a jitter that follows the scale of the
# prior columns).  The one file, docs/data/decompose_openrapm.json:
#     {"meta": {built, penalty, seasons, fields, pieces}, "rows": {season: [[name, NBA id, possessions, offense,
#      defense, total, teams], ...]}}     each side's list holds OPENRAPM_PIECES in order, at two decimals; teams
#      are the box scores' abbreviations in the order he joined them, "SAC/IND"; then his wins over starter average,
#      his regular-season minutes, and per side, offense, defense and total, [his regular-season rating over the
#      starter average, his leverage, his wins over starter average] (WOSA_PARTS, wins.py says how they multiply)
# Rows are the ratings.json rows (possessions above 0), best total first.
#
# Example (2026): Victor Wembanyama's offense, 3.01 = 2.11 + 0.26 + 0.86 - 0.32 + 0.11; his total, 9.37 = 3.89 + 2.70
# + 2.98 - 0.43 + 0.23: box score, Box Adj, on court, off court, team SOS.  The page's headers for on court and off
# court are On adj and Off adj.
OPENRAPM_FILE = "decompose_openrapm.json"
OPENRAPM_FIELDS = ["n", "id", "p", "o", "d", "t", "tm", "w", "m", "wo", "wd", "wt"]
# The ratings' standard errors (rating_error.py), one more field at the end of each row when `openrapm errors` has
# written outputs/rating_errors_<name>.parquet beside the ratings table: [offense, defense, total] at two decimals,
# null for a player without one (every game of his in one fold).  Without that file the two site files are as they
# were, so a fresh clone and a build that never ran the errors write the same documents as before 2026-11-26.
ERRORS_FIELD = "se"
WOSA_PARTS = ["above_starter_average", "leverage", "wosa"]
OPENRAPM_PIECES = ["box_score", "on_court", "off_court", "team_sos", "box_adj", "rating"]
OPENRAPM_SIDES = ("off", "def", "total")
OPENRAPM_TOLERANCE = 1e-9         # each identity holds to this, per player and side
OPENRAPM_JITTER_TOLERANCE = 1e-6  # the same, in a season whose solve needed rapm.solved_ridge's jitter (2020)
OPENRAPM_TABLE_COLUMNS = ["player_id", "season", "player_name", "poss_season", "prior_off", "prior_def", "u_off",
                          "u_def", "box_adj_off", "box_adj_def", "c_off", "c_def", "rating_off", "rating_def",
                          "rating_total", "offense_lambda",
                          "wosa", "wosa_off", "wosa_def", "minutes_rs", "above_off", "above_def", "leverage",
                          "leverage_off", "leverage_def"]
OPENRAPM_STEP3_COLUMNS = ["player_id", "season", "prior_offense", "prior_defense", "prior_scale_off",
                          "prior_scale_def", "u_off", "u_def"]


def step3_path(ratings_path: Path) -> Path:
    """The step-3 table `openrapm build` writes beside the ratings table."""
    return ratings_path.with_name(f"{ratings_path.stem}_step3.parquet")


GAMES_FACTOR_TOLERANCE = 1e-12   # the table's games' part against its step-3 table's times one factor


class GamesLayers:
    """Step 4's two layers on one season and side's games' part (prior_shrink, "Since experiment 106"): the part
    split into each player's team mean, times `team`, and his deviation from it, times `within`; the team is his main
    team (data.roles.main_teams), the mean weighted by his possessions on that side.  The map is linear, so a piece of
    the games' part goes through it the same way and the pieces still add up to the part."""

    def __init__(self, team: float, within: float, team_of: dict, weight_of: dict):
        self.team, self.within, self.team_of, self.weight_of = float(team), float(within), team_of, weight_of

    def apply(self, player_ids, values) -> np.ndarray:
        ids = pd.Series(np.asarray(player_ids))
        team = np.array(ids.map(self.team_of).to_numpy(float), dtype=float)
        alone = np.isnan(team)
        team[alone] = -1.0 - np.arange(int(alone.sum()))
        mean, deviation = team_split(team, values, ids.map(self.weight_of).fillna(0.0).to_numpy(float))
        return self.team * mean + self.within * deviation


def games_image(factor, player_ids, values) -> np.ndarray:
    """A games' part (or one of its pieces) after step 4: `factor` times it with one layer, `GamesLayers.apply` with
    two.  `player_ids` and `values` are a whole season's table rows, as step 4 splits them."""
    values = np.asarray(values, dtype=float)
    if isinstance(factor, GamesLayers):
        return factor.apply(player_ids, values)
    return (1.0 if factor is None else float(factor)) * values


def layered_games_factors(both: pd.DataFrame, table: pd.DataFrame, teams: pd.DataFrame) -> tuple[dict, float]:
    """({season: {"off", "def"}: GamesLayers}, the largest miss): each season and side's two multipliers on the
    games' part read off the table by least squares on the step-3 part's team means and deviations."""
    team_of = teams.set_index(["player_id", "season"]).team_id
    poss = table.set_index(["player_id", "season"])[["poss_off", "poss_def"]]
    factors, miss = {}, 0.0
    for season, rows in both.groupby("season"):
        factors[int(season)] = {}
        ids = rows.player_id.to_numpy()
        key = pd.MultiIndex.from_arrays([ids, np.full(len(ids), season)])
        teams_now = dict(zip(ids, team_of.reindex(key).to_numpy(float)))
        teams_now = {player: team for player, team in teams_now.items() if not np.isnan(team)}
        for side in ("off", "def"):
            weight_of = dict(zip(ids, poss.reindex(key)[f"poss_{side}"].to_numpy(float)))
            unit = GamesLayers(1.0, 0.0, teams_now, weight_of)
            theirs = rows[f"u_{side}_step3"].to_numpy(float)
            mean = unit.apply(ids, theirs)
            design = np.column_stack([mean, theirs - mean])
            mine = rows[f"u_{side}"].to_numpy(float)
            team, within = np.linalg.lstsq(design, mine, rcond=None)[0]
            factors[int(season)][side] = GamesLayers(team, within, teams_now, weight_of)
            miss = max(miss, float(np.abs(mine - design @ np.array([team, within])).max()))
    return factors, miss


def published_parts(ratings_path: Path, cfg: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """(the ratings table, its step-3 table, {season: {"off", "def"}: step 4's factor on the games' part}), after
    checking they are one build's: the same player-seasons, the same games' part up to one factor per season and side
    (step 4 multiplies it since experiment 65, 1 before; step 5 leaves it alone), and the penalty the shipped settings
    name.  When one factor does not reproduce the table and `cfg` is given, the two layers of experiment 106: two
    multipliers per season and side, on each player's team mean and on his deviation from it (`GamesLayers`)."""
    step3_file = step3_path(ratings_path)
    if not step3_file.exists():
        raise SystemExit(f"{step3_file} does not exist; `openrapm build` writes it beside the ratings table")
    table, step3 = pd.read_parquet(ratings_path), pd.read_parquet(step3_file)
    for name, frame, columns in (("the ratings table", table, OPENRAPM_TABLE_COLUMNS),
                                 ("its step-3 table", step3, OPENRAPM_STEP3_COLUMNS)):
        missing = [column for column in columns if column not in frame.columns]
        if missing:
            raise SystemExit(f"{name} lacks {missing}; `openrapm build` writes them all")
    keys = ["player_id", "season"]
    both = table[keys + ["u_off", "u_def"]].merge(step3[keys + ["u_off", "u_def"]], on=keys, suffixes=("", "_step3"))
    if not len(both) == len(table) == len(step3):
        raise SystemExit("the ratings table and its step-3 table hold different player-seasons: they are not one "
                         "build's")
    factors, miss = {}, 0.0
    for season, rows in both.groupby("season"):
        factors[int(season)] = {}
        for side in ("off", "def"):
            mine, theirs = rows[f"u_{side}"].to_numpy(float), rows[f"u_{side}_step3"].to_numpy(float)
            factor = float(mine @ theirs / (theirs @ theirs)) if theirs @ theirs > 0 else 1.0
            factors[int(season)][side] = factor
            miss = max(miss, float(np.abs(mine - factor * theirs).max()))
    if miss >= GAMES_FACTOR_TOLERANCE and cfg:
        factors, miss = layered_games_factors(both, table, main_teams(cfg))
    check(miss < GAMES_FACTOR_TOLERANCE, f"the ratings table and its step-3 table have the same games' part, up to "
                                         f"step 4's factor per season and side (max miss {miss:.1e})")
    check(set(table.offense_lambda.unique()) == {float(PRODUCT.ridge.penalty)},
          f"the ratings table was built at the shipped penalty, {PRODUCT.ridge.penalty:,.0f}")
    return table, step3, factors


def season_games_parts(designs: DesignCache, box_files: BoxScoreFiles, season: int, step3: pd.DataFrame,
                       penalty: float, held: bool = False) -> tuple[dict, dict]:
    """One season's games' part split on both sides: ({"off": frame, "def": frame}, {"off": misses, "def": misses}).

    The offense half comes from the fit on the offense target, the defense half from the fit on the defense target,
    as `prior_rapm.rate_season` makes the table.  The priors are the step-3 table's own divided by its prior scale: a
    fit's deviations do not change with the prior's scale or with a constant added to it, because the scale is a free
    coefficient and the season intercept absorbs the constant.  `misses` adds the largest gap between the refit's
    prior scale and the table's, and the share of rows the box scores cannot place.  `held`: the build held the
    season's prior scales (a young season in progress, `held_scale_seasons`), so the refit holds them at the table's
    and its scale misses nothing by construction.
    """
    rows = step3[step3.season == season]
    scale = (float(rows.prior_scale_off.iloc[0]), float(rows.prior_scale_def.iloc[0])) if held else None
    prior_offense = dict(zip(rows.player_id, rows.prior_offense / rows.prior_scale_off))
    prior_defense = dict(zip(rows.player_id, rows.prior_defense / rows.prior_scale_def))
    parts, misses = {}, {}
    for side, target, scale_column, scale_index in (("off", PRODUCT.targets.offence, "prior_scale_off", -2),
                                                    ("def", PRODUCT.targets.defence, "prior_scale_def", -1)):
        design = designs.season(season, target)
        fit = refit(design, prior_offense, prior_defense, penalty, scale)
        absent = set(fit.player_ids) - set(rows.player_id)
        if absent:
            raise SystemExit(f"{season}: {len(absent)} players of the design are not in the step-3 table")
        lineup_o, lineup_d = lineups(fit.players, fit.n_players)
        keys = row_keys(design, lineup_o, lineup_d, box_files.teams(season))
        parts[side], misses[side] = split_games_part(fit, keys, side)
        misses[side].update(unplaced=keys["unplaced"], scale=0.0 if held else
                            abs(float(fit.theta[scale_index]) - float(rows[scale_column].iloc[0])))
    return parts, misses


def held_scale_seasons(cfg: dict) -> set:
    """The seasons whose prior scales the build held (commands/build.SeasonsInProgress): those in progress short of
    settings.RidgeSettings.held_scale_through of their regular season.  A configuration that names no seasons or no
    data folder, a simulated league's, has none in progress (as data.seasons.last_finished reads it)."""
    from ..data.seasons import in_progress, share_played
    if "_root" not in cfg or "first_season" not in cfg:
        return set()
    return {int(season) for season in in_progress(cfg)
            if share_played(int(season), cfg) < PRODUCT.ridge.held_scale_through}


def season_pieces(rows: pd.DataFrame, parts: dict, factors: dict | None = None) -> pd.DataFrame:
    """One season's pieces: a row per ratings-table row, and per side (off, def, total) a column for each of
    OPENRAPM_PIECES, named `<side>_<piece>`.  Positive is good on both sides; a total is the two sides added.  The
    games' part's pieces are the refit's times step 4's factor on it (`factors`, {"off", "def"}; 1 when None)."""
    pieces = pd.DataFrame({"player_id": rows.player_id.to_numpy()})
    for side in ("off", "def"):
        games = parts[side].set_index("player_id").reindex(rows.player_id)
        if games.u.isna().any():
            raise SystemExit(f"{int(rows.season.iloc[0])}: the refit has no row for some players of the ratings table")
        pieces[f"{side}_box_score"] = rows[f"prior_{side}"].to_numpy() - rows[f"box_adj_{side}"].to_numpy()
        factor = None if factors is None else factors[side]
        for piece in ("on_court", "off_court", "team_sos"):
            pieces[f"{side}_{piece}"] = games_image(factor, rows.player_id.to_numpy(), games[piece].to_numpy())
        pieces[f"{side}_box_adj"] = rows[f"box_adj_{side}"].to_numpy() + rows[f"c_{side}"].to_numpy()
        pieces[f"{side}_rating"] = rows[f"rating_{side}"].to_numpy()
    for piece in OPENRAPM_PIECES[:-1]:
        pieces[f"total_{piece}"] = pieces[f"off_{piece}"] + pieces[f"def_{piece}"]
    pieces["total_rating"] = rows.rating_total.to_numpy()
    return pieces


OPENRAPM_MISSES = ("identity", "games_part", "scale", "sum_hy", "leftover")


def new_openrapm_worst() -> dict:
    """The largest misses of every check, over the seasons whose solve was plain and over the jittered ones."""
    return dict(plain=dict.fromkeys(OPENRAPM_MISSES, 0.0), jittered=dict.fromkeys(OPENRAPM_MISSES, 0.0),
                unplaced=0.0, unnamed=0)


def update_openrapm_worst(worst: dict, rows: pd.DataFrame, pieces: pd.DataFrame, parts: dict, misses: dict,
                          factors: dict | None = None) -> None:
    """Fold one season's misses into the worst over every season; the refit's games' part is held to the table's
    after step 4's factor on it (`factors`, {"off", "def"}; 1 when None).

    A season whose solve needed rapm.solved_ridge's jitter (2020) is kept apart: the jitter is 1e-9 x the mean
    diagonal, and the prior columns here are the table's (scaled and centered) rather than the build's own, so the
    replayed matrix differs from the build's in the last digits and the games' part by about 1e-8.
    """
    season = worst["jittered" if any(misses[side]["retried"] for side in ("off", "def")) else "plain"]
    for side in OPENRAPM_SIDES:
        added = sum(pieces[f"{side}_{piece}"] for piece in OPENRAPM_PIECES[:-1])
        season["identity"] = max(season["identity"], float((added - pieces[f"{side}_rating"]).abs().max()))
    for side in ("off", "def"):
        refit_u = games_image(None if factors is None else factors[side], rows.player_id.to_numpy(),
                              parts[side].set_index("player_id").u.reindex(rows.player_id).to_numpy())
        season["games_part"] = max(season["games_part"], float(np.abs(refit_u - rows[f"u_{side}"].to_numpy()).max()))
        season["scale"] = max(season["scale"], misses[side]["scale"])
        season["sum_hy"] = max(season["sum_hy"], misses[side]["sum_hy"])
        season["leftover"] = max(season["leftover"], misses[side]["leftover"])
        worst["unplaced"] = max(worst["unplaced"], misses[side]["unplaced"])


def error_cell(errors: dict, player_id: int) -> list | None:
    """A row's `ERRORS_FIELD`: [offense, defense, total] at two decimals from {player_id: (se_off, se_def, se_total)},
    None for a player it lacks or whose standard errors are missing."""
    mine = errors.get(int(player_id))
    if mine is None or any(pd.isna(value) for value in mine):
        return None
    return [round(float(value), 2) for value in mine]


def openrapm_rows(rows: pd.DataFrame, pieces: pd.DataFrame, names: np.ndarray, teams: dict | None = None,
                  errors: dict | None = None) -> list:
    """One season's rows in OPENRAPM_FIELDS order, best total first, players under 60 possessions left out (as
    ratings.json leaves them): name, NBA id, possessions (whole, as ratings.json writes them), each side's pieces at
    two decimals, then his teams ({player_id: "SAC/IND"}; "" for a player it lacks), then his wins over starter average
    at two decimals, his regular-season minutes, whole, and each side's WOSA_PARTS at two decimals.  With `errors`,
    {player_id: (se_off, se_def, se_total)}, each row ends with its `ERRORS_FIELD` (`error_cell`)."""
    teams = teams or {}
    wosa = {"wo": ("above_off", "leverage_off", "wosa_off"), "wd": ("above_def", "leverage_def", "wosa_def")}
    frame = pieces.assign(n=names, p=rows.poss_season.to_numpy(), w=rows.wosa.to_numpy(),
                          m=rows.minutes_rs.to_numpy(),
                          above_total=(rows.above_off + rows.above_def).to_numpy(),
                          **{column: rows[column].to_numpy() for parts in wosa.values() for column in parts},
                          leverage=rows.leverage.to_numpy())
    wosa["wt"] = ("above_total", "leverage", "w")
    frame = frame[frame.p >= MIN_POSSESSIONS_SHOWN].sort_values("total_rating", ascending=False)
    return [[r["n"], int(r["player_id"]), int(r["p"]),
             *([round(float(r[f"{side}_{piece}"]), 2) for piece in OPENRAPM_PIECES] for side in OPENRAPM_SIDES),
             teams.get(int(r["player_id"]), ""), round(float(r["w"]), 2), int(round(float(r["m"]))),
             *([round(float(r[column]), 2) for column in wosa[field]] for field in ("wo", "wd", "wt")),
             *([] if errors is None else [error_cell(errors, r["player_id"])])]
            for r in frame.to_dict("records")]


def run_openrapm_checks(worst: dict, rows: dict, table: pd.DataFrame) -> None:
    """The checks on every season, before the file is written."""
    plain, jittered = worst["plain"], worst["jittered"]

    def holds(key: str) -> bool:
        return plain[key] < OPENRAPM_TOLERANCE and jittered[key] < OPENRAPM_JITTER_TOLERANCE

    def misses(key: str) -> str:
        return f"max miss {plain[key]:.1e}; in a jittered season {jittered[key]:.1e}"

    check(holds("identity"), f"box score + Box Adj + on court + off court + team SOS add up to every rating, "
                             f"each side and the total ({misses('identity')})")
    check(holds("games_part"), f"solving step 3 again gives every player's games' part in the ratings table "
                               f"({misses('games_part')})")
    check(holds("scale"), f"and every season's prior scale, both sides ({misses('scale')})")
    check(holds("sum_hy"), f"every player's coefficient is the sum over rows of weight x outcome ({misses('sum_hy')})")
    check(holds("leftover"), f"the three groups of rows add up to the games' part, up to the jitter of a singular "
                             f"season ({misses('leftover')})")
    check(worst["unplaced"] < UNPLACED_TOLERANCE,
          f"the box scores name both teams of nearly every row (largest share without, in one season: "
          f"{worst['unplaced']:.2%})")
    check(worst["unnamed"] == 0, f"every player has a name ({worst['unnamed']} without)")
    team_field = OPENRAPM_FIELDS.index("tm")
    teamless = sum(1 for season_rows in rows.values() for row in season_rows if not row[team_field])
    check(teamless == 0, f"every player has a team ({teamless} without)")
    published = {(row["s"], row["id"]): (row["o"], row["d"], row["t"], row["p"], row["w"])
                 for row in ratings_rows(table)}
    wins_field = OPENRAPM_FIELDS.index("w")
    mine = {(season, row[1]): (row[3][-1], row[4][-1], row[5][-1], row[2], row[wins_field])
            for season, season_rows in rows.items() for row in season_rows}
    check(published == mine, f"the file holds the {len(published):,} player-seasons of ratings.json with its "
                             "ratings, possessions and wins over starter average")
    total_field = OPENRAPM_FIELDS.index("wt")
    unequal = sum(1 for season_rows in rows.values() for row in season_rows if row[total_field][-1] != row[wins_field])
    check(unequal == 0, f"each row's total wins over starter average is its WOSA ({unequal} differ)")


PERCENTILE_POSSESSIONS = 500   # the ratings page's percentile pool: a finished season's players with this many


def percentile_minimums(cfg: dict) -> dict:
    """{season: possessions} for each season in progress: the fewest possessions a player needs to enter the ratings
    page's percentile pool, PERCENTILE_POSSESSIONS times the share of its regular season played, never under
    MIN_POSSESSIONS_SHOWN (the owner, 2026-11-01: with 500 alone, 2027 at 7% of its season pooled 1 player of the 334
    shown).  Finished seasons are not listed; the page reads 500 for them.  A configuration that names no seasons or no
    data folder has none in progress (`held_scale_seasons`).

    Example: 2027 at 7.2% of its season, 500 x 0.072 = 36, reads 60; at half a season, 250."""
    from ..data.seasons import in_progress, share_played
    if "_root" not in cfg or "first_season" not in cfg:
        return {}
    return {int(season): int(max(MIN_POSSESSIONS_SHOWN, round(PERCENTILE_POSSESSIONS * share_played(int(season), cfg))))
            for season in in_progress(cfg)}


def write_openrapm_file(path: Path, rows: dict, built: str, penalty: float, opens_on: int | None = None,
                        percentile_min: dict | None = None, rookie_ids: dict | None = None,
                        with_errors: bool = False) -> dict:
    """decompose_openrapm.json: meta and the rows of every season, latest first; compact, UTF-8 letters as they are.
    Returns the document written.
    `opens_on` is the season the ratings page opens on (`default_season`), the latest when not given;
    `percentile_min`, the seasons in progress' percentile pool minimums (`percentile_minimums`), none when not given;
    `rookie_ids`, each season's rookies (data/panel.py `rookies`), kept to the players the season's rows show, none
    when not given.  `with_errors`: the rows end with their standard errors, and the fields with `ERRORS_FIELD`."""
    seasons = sorted(rows, reverse=True)
    minimums = {str(season): int(value) for season, value in (percentile_min or {}).items()}
    shown = {season: {row[OPENRAPM_FIELDS.index("id")] for row in rows[season]} for season in seasons}
    rookie_lists = {str(season): [player for player in (rookie_ids or {}).get(season, []) if player in shown[season]]
                    for season in seasons}
    document = dict(meta=dict(built=built, penalty=penalty, seasons=seasons,
                              default_season=int(seasons[0] if opens_on is None else opens_on),
                              percentile_min=minimums, rookies=rookie_lists,
                              fields=OPENRAPM_FIELDS + ([ERRORS_FIELD] if with_errors else []),
                              pieces=OPENRAPM_PIECES, wosa_parts=WOSA_PARTS),
                    rows={str(season): rows[season] for season in seasons})
    path.write_text(json.dumps(document, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return document


# ratings_list.json: what the ratings page draws as it opens, cut from the Detail file's document, so the two can never
# disagree.  The page read the Detail file for everything until 2026-11-09, 2.9 MB, 1.0 MB as the server compresses it,
# before its first row was drawn.  This holds each player-season's name, NBA id, possessions, teams, three ratings and
# wins over starter average: 0.8 MB, 0.3 MB compressed.  The page asks for the Detail file when Detail is opened.
#
#   {"meta": {"built", "seasons", "default_season", "percentile_min", "rookies",     as the Detail file has them
#             "through": {season: the date of its last game},                        a season in progress only
#             "fields": RATINGS_LIST_FIELDS},
#    "rows": {season: [[name, id, possessions, teams, offense, defense, total, WOSA], ...]}}     the Detail file's order
# When the Detail file carries the standard errors (`ERRORS_FIELD`), each row ends with the total's, null without one,
# and the fields with "se".
RATINGS_LIST_FILE = "ratings_list.json"
RATINGS_LIST_FIELDS = ["n", "id", "p", "tm", "o", "d", "t", "w"]
RATINGS_LIST_META = ("built", "seasons", "default_season", "percentile_min", "rookies")


def ratings_list_document(detail: dict, through: dict | None = None) -> dict:
    """ratings_list.json's document from decompose_openrapm.json's (`write_openrapm_file`): every row's name, id,
    possessions and teams, each side's rating, the last of its pieces, and his wins over starter average; `through` is
    {season: the date of its last game} for the seasons in progress (`last_game_days`).

    Example: the Detail row ["Victor Wembanyama", 1641705, 620, [3.9, 0.3, 0.2, 0.0, 0.66, 5.06], [...], [...], "SAS",
    1.9, ...] gives ["Victor Wembanyama", 1641705, 620, "SAS", 5.06, 4.27, 9.33, 1.9]; ending [0.67, 0.7, 0.93], his
    standard errors, it gives the same with 0.93 after it."""
    fields = detail["meta"]["fields"]
    at = {name: fields.index(name) for name in RATINGS_LIST_FIELDS}
    rating = detail["meta"]["pieces"].index("rating")
    errors = fields.index(ERRORS_FIELD) if ERRORS_FIELD in fields else None

    def total_error(row) -> list:
        return [] if errors is None else [None if row[errors] is None else row[errors][-1]]

    rows = {season: [[row[at["n"]], row[at["id"]], row[at["p"]], row[at["tm"]], row[at["o"]][rating],
                      row[at["d"]][rating], row[at["t"]][rating], row[at["w"]], *total_error(row)]
                     for row in season_rows]
            for season, season_rows in detail["rows"].items()}
    meta = {key: detail["meta"][key] for key in RATINGS_LIST_META}
    meta.update(through={str(season): str(day) for season, day in sorted((through or {}).items())},
                fields=RATINGS_LIST_FIELDS + ([] if errors is None else [ERRORS_FIELD]))
    return dict(meta=meta, rows=rows)


def last_game_days(cfg: dict) -> dict:
    """{season: the date of its last game with stints, written 2026-11-08} for each season in progress: the day the
    ratings page says that season's numbers run through.  A configuration that names no seasons or no data folder has
    none in progress (`percentile_minimums`)."""
    from ..data.seasons import in_progress
    if "_root" not in cfg or "first_season" not in cfg:
        return {}
    return {int(season): pd.Timestamp(season_stints(int(season), cfg).game_date.max()).strftime("%Y-%m-%d")
            for season in in_progress(cfg)}


def errors_path(ratings_path: Path) -> Path:
    """The standard errors `openrapm errors` writes beside a ratings table: season_ratings_product.parquet's are
    rating_errors_product.parquet."""
    return ratings_path.with_name(f"rating_errors_{ratings_path.stem.removeprefix('season_ratings_')}.parquet")


def rating_errors(ratings_path: Path) -> dict | None:
    """{season: {player_id: (se_off, se_def, se_total)}} from `errors_path`, None when the file is not there."""
    path = errors_path(ratings_path)
    if not path.exists():
        return None
    table = pd.read_parquet(path)
    return {int(season): {int(row.player_id): (row.se_off, row.se_def, row.se_total)
                          for row in rows.itertuples(index=False)}
            for season, rows in table.groupby("season")}


def write_openrapm_decomposition(cfg: dict, ratings_path: Path, data_dir: Path, built: str) -> None:
    """decompose_openrapm.json in `data_dir`, after its checks pass.  Solves step 3 again for every season and both
    sides, so it must run after `openrapm build`, whenever the ratings table changes.  The rows carry the ratings'
    standard errors when `errors_path` is there."""
    table, step3, factors = published_parts(ratings_path, cfg)
    errors = rating_errors(ratings_path)
    names, box_files, designs = name_table(ratings_path, cfg), BoxScoreFiles(cfg), DesignCache(cfg)
    penalty = float(PRODUCT.ridge.penalty)
    started, rows, worst, held = time.time(), {}, new_openrapm_worst(), held_scale_seasons(cfg)
    for season in sorted(int(s) for s in table.season.unique()):
        season_rows = table[table.season == season].reset_index(drop=True)
        parts, misses = season_games_parts(designs, box_files, season, step3, penalty, season in held)
        designs.clear()
        pieces = season_pieces(season_rows, parts, factors[season])
        update_openrapm_worst(worst, season_rows, pieces, parts, misses, factors[season])
        named = span_names(names, season_rows.player_id.to_numpy(), [season], box_files)
        worst["unnamed"] += int(sum(1 for name in named if pd.isna(name) or not str(name).strip()))
        rows[season] = openrapm_rows(season_rows, pieces, named, box_files.season_teams(season),
                                     None if errors is None else errors.get(season, {}))
        top = rows[season][0]
        print(f"  {season}: {len(rows[season])} players, top {top[0]} {top[5][-1]:+.2f} "
              f"({time.time() - started:.0f}s)", flush=True)
    run_openrapm_checks(worst, rows, table)
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / OPENRAPM_FILE
    document = write_openrapm_file(path, rows, built, penalty, default_season(table), percentile_minimums(cfg),
                                   rookies(cfg), with_errors=errors is not None)
    if errors is not None:
        without = {season: sum(1 for row in season_rows if row[-1] is None) for season, season_rows in rows.items()}
        print(f"  standard errors from {errors_path(ratings_path).name}: {sum(without.values())} of "
              f"{sum(len(v) for v in rows.values()):,} rows without one"
              + "".join(f"; {season}: {count}" for season, count in without.items() if count > 20), flush=True)
    print(f"wrote {path}: {sum(len(v) for v in rows.values()):,} rows over {len(rows)} seasons, "
          f"{path.stat().st_size / 1e6:.2f} MB ({time.time() - started:.0f}s)", flush=True)
    list_path = data_dir / RATINGS_LIST_FILE
    _write_compact(list_path, ratings_list_document(document, last_game_days(cfg)))
    print(f"wrote {list_path}: the same rows' ratings for the ratings page's list, "
          f"{list_path.stat().st_size / 1e6:.2f} MB", flush=True)


# ===================================================================================== the similarity page
# Which other player-seasons a player-season is most like: the nearest player-seasons of other players, by the
# distance between their standardized inputs (similarity.py says how).  The owner's design, 2026-10-12.
#
# The files, in docs/data/similarity/; the page loads the first once, and a season's file when one of its
# player-seasons is picked:
#     player_seasons.json   {"meta": ..., "rows": [[id, name, season, possessions, offense, defense, total], ...]}:
#                           the pool in its order (season, then id), so a row's position is how the lists name it
#     <season>.json         {"meta": {"season", "first", "fields"}, "rows": one entry per row of that season, in pool
#                           order from row `first`: [its seasons list, their distances, its players list, their
#                           distances]}; distances at two decimals
SIMILARITY_MIN_POSSESSIONS = 1000         # the ratings page's default minimum (docs/index.html)
SIMILARITY_MATCHES = 50                   # each list's length (the owner, 2026-10-12)
SIMILARITY_DEFAULT = (1627780, 2022)      # Gary Payton II, 2021-22: the page's opening view (the owner, 2026-10-12)
# the similarity page's inputs: the box-score prior's lists before experiment 76 added the hustle counts, which are
# missing before 2017 and so cannot place a 1998 season beside a 2020 one; the page is unchanged by that adoption
SIMILARITY_FEATURE_SET = "no_career_counts"
SIMILARITY_FOLDER = "similarity"
SIMILARITY_FIELDS = ["id", "n", "s", "p", "o", "d", "t"]
SIMILARITY_LIST_FIELDS = ["k", "d", "pk", "pd"]
SIMILARITY_SHOWN = 20                     # the console report's list length
# The owner, 2026-11-18: "let's just turn similarity on when the median poss played = 500".  A season in progress joins
# the pool once the median of the players the pages show (`MIN_POSSESSIONS_SHOWN` or more) has 500 offensive
# possessions, the ratings page's rule for the season it opens on at 250 (`default_season`).  Its players need
# SIMILARITY_MIN_POSSESSIONS times the share of its regular season played, never under MIN_POSSESSIONS_SHOWN, as the
# percentile pool scales (`percentile_minimums`): with 1,000 alone, 2027 on 2026-11-18 at a median of 516 held 9
# players.  scripts/daily_files.py reads the same rule from ratings.json, so the morning job rebuilds the pool the day a
# season crosses it.
SIMILARITY_SEASON_POSSESSIONS = 500


def similarity_ready(table: pd.DataFrame) -> set[int]:
    """The seasons whose median player shown has SIMILARITY_SEASON_POSSESSIONS or more offensive possessions
    (poss_season, the pages' p).  Example: on 2026-11-18 2027's median of its shown players is 516, so 2027 is ready;
    every finished season's is over 1,700."""
    shown = table[table.poss_season >= MIN_POSSESSIONS_SHOWN]
    medians = shown.groupby("season").poss_season.median()
    return {int(season) for season in medians.index[medians >= SIMILARITY_SEASON_POSSESSIONS]}


def similarity_in_progress(cfg: dict, table: pd.DataFrame) -> dict:
    """{season: the fewest offensive possessions a player of it needs} for each season in progress that is ready
    (`similarity_ready`).  A configuration that names no seasons or no data folder has none in progress.

    Example: 2027 at 21% of its regular season, 1000 x 0.21 = 210."""
    from ..data.seasons import in_progress, share_played
    if "_root" not in cfg or "first_season" not in cfg:
        return {}
    ready = similarity_ready(table)
    return {int(season): int(max(MIN_POSSESSIONS_SHOWN,
                                 round(SIMILARITY_MIN_POSSESSIONS * share_played(int(season), cfg))))
            for season in in_progress(cfg) if int(season) in ready}


def _named(rows: pd.DataFrame, ratings_path: Path, cfg: dict) -> pd.DataFrame:
    """The pool with every name filled: the ratings table's, else the bio files' (`name_table`)."""
    missing = rows.player_name.isna() | (rows.player_name.astype(str).str.strip() == "")
    if missing.any():
        latest = name_table(ratings_path, cfg).drop_duplicates("player_id", keep="last").set_index("player_id")
        rows = rows.copy()
        rows.loc[missing, "player_name"] = rows.loc[missing, "player_id"].map(latest.player_name)
    if rows.player_name.isna().any():
        raise SystemExit(f"no name for players {sorted(rows.player_id[rows.player_name.isna()].unique())}")
    return rows


def similarity_frames(cfg: dict, ratings_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(the pool, its inputs).

    The pool holds every finished season (`trainable_panel`) and each season in progress whose median player shown has
    500 offensive possessions (`similarity_in_progress`); nothing is fitted on it, the inputs are only standardized
    on the pool.  Stops when the panel's possessions differ from the ratings table's (one of them rebuilt since the
    other) or when the opening player-season is not in the pool.
    """
    table = pd.read_parquet(ratings_path)
    current = similarity_in_progress(cfg, table)
    panel = trainable_panel(cfg)
    if current:
        later = in_progress_panel(cfg)
        panel = pd.concat([panel, later[later.season.astype(int).isin(current)]], ignore_index=True)
    seasons = sorted(set(panel.season.astype(int)))
    waiting = sorted(set(table.season.astype(int)) - set(seasons))
    if waiting:
        print(f"  left out until their median player shown has {SIMILARITY_SEASON_POSSESSIONS} offensive "
              f"possessions: {waiting}", flush=True)
    for season, floor in current.items():
        print(f"  {season}, in progress: players with {floor} offensive possessions or more", flush=True)
    floors = {season: current.get(season, SIMILARITY_MIN_POSSESSIONS) for season in seasons}
    rows = _named(player_seasons(table, floors, seasons), ratings_path, cfg)
    inputs = similarity_inputs(rows, panel, FEATURE_SETS[SIMILARITY_FEATURE_SET])
    for side in ("off", "def"):
        if not np.array_equal(inputs[f"poss_{side}"].to_numpy(), rows[f"poss_{side}"].to_numpy(float)):
            raise SystemExit(f"the season panel's possessions differ from {ratings_path.name}'s (poss_{side}): "
                             f"rebuild the older of the two")
    if not ((rows.player_id == SIMILARITY_DEFAULT[0]) & (rows.season == SIMILARITY_DEFAULT[1])).any():
        raise SystemExit(f"the opening player-season {SIMILARITY_DEFAULT} is not in the pool")
    return rows, inputs


def similarity_documents(rows: pd.DataFrame, lists: dict, built: str) -> tuple[dict, dict]:
    """player_seasons.json's document, and {season: its file's document} (the shapes are above)."""
    meta = dict(built=built, seasons=sorted(int(s) for s in rows.season.unique()),
                min_possessions=SIMILARITY_MIN_POSSESSIONS, matches=SIMILARITY_MATCHES,
                default=list(SIMILARITY_DEFAULT), fields=SIMILARITY_FIELDS)
    pool = [[int(r.player_id), str(r.player_name), int(r.season), int(round(r.poss_off)),
             round(float(r.rating_off), 2), round(float(r.rating_def), 2), round(float(r.rating_total), 2)]
            for r in rows.itertuples(index=False)]
    seasons = {}
    for season, positions in rows.groupby("season").indices.items():
        if not np.array_equal(positions, np.arange(positions[0], positions[0] + len(positions))):
            raise ValueError(f"the pool's {season} rows are not together; the pool must be in season order")
        entries = [[lists["seasons"][i].tolist(), np.round(lists["season_distances"][i], 2).tolist(),
                    lists["players"][i].tolist(), np.round(lists["player_distances"][i], 2).tolist()]
                   for i in positions]
        seasons[int(season)] = dict(meta=dict(season=int(season), first=int(positions[0]),
                                              fields=SIMILARITY_LIST_FIELDS), rows=entries)
    return dict(meta=meta, rows=pool), seasons


def _write_compact(path: Path, document: dict) -> None:
    """Compact JSON (no spaces), letters as they are, no newline at the end."""
    path.write_text(json.dumps(document, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


def write_similarity_files(data_dir: Path, documents: tuple) -> None:
    """player_seasons.json and one file per season in `data_dir`; a season file left from an earlier pool is
    removed."""
    pool, seasons = documents
    data_dir.mkdir(parents=True, exist_ok=True)
    for stale in data_dir.glob("[0-9][0-9][0-9][0-9].json"):
        if int(stale.stem) not in seasons:
            stale.unlink()
    _write_compact(data_dir / "player_seasons.json", pool)
    for season, document in seasons.items():
        _write_compact(data_dir / f"{season}.json", document)


def _shown_lists(rows: pd.DataFrame, lists: dict) -> list[str]:
    """The opening player-season's two lists, the first SIMILARITY_SHOWN of each."""
    at = int(np.flatnonzero((rows.player_id == SIMILARITY_DEFAULT[0]) & (rows.season == SIMILARITY_DEFAULT[1]))[0])
    lines = []
    for key, label in (("seasons", "player-seasons"), ("players", "players")):
        lines.append(f"\n=== {rows.player_name[at]} {rows.season[at]}: the {SIMILARITY_SHOWN} nearest {label}")
        pairs = zip(lists[key][at][:SIMILARITY_SHOWN], lists[f"{key[:-1]}_distances"][at][:SIMILARITY_SHOWN])
        lines.extend(f"  {rank:>2}. {rows.player_name[row]:<26} {rows.season[row]}  distance {distance:.2f}  "
                     f"offense {rows.rating_off[row]:+.2f}  defense {rows.rating_def[row]:+.2f}"
                     for rank, (row, distance) in enumerate(pairs, start=1))
    return lines


def similarity_summary(rows: pd.DataFrame, inputs: pd.DataFrame, lists: dict, ranks: np.ndarray) -> list[str]:
    """The console report: the inputs, how high each player's own other seasons rank, and the opening
    player-season's two lists."""
    ranked = ranks[~np.isnan(ranks)]
    lines = [f"  inputs: {', '.join(inputs.columns)}",
             f"  his nearest other season among every row: mean 1/rank {np.mean(1.0 / ranked):.3f}, median rank "
             f"{np.median(ranked):.0f}, first in {np.mean(ranked == 1):.0%} of {len(ranked):,} player-seasons"]
    return lines + _shown_lists(rows, lists)


def write_similarity(cfg: dict, ratings_path: Path, data_dir: Path, built: str) -> None:
    """The similarity page's files in `data_dir`."""
    started = time.time()
    rows, inputs = similarity_frames(cfg, ratings_path)
    print(f"similarity: {len(rows):,} player-seasons of {rows.player_id.nunique():,} players, "
          f"{len(inputs.columns)} inputs", flush=True)
    distance, players = partial(distances, standardise(inputs)), rows.player_id.to_numpy()
    lists = closest(distance, players, SIMILARITY_MATCHES)
    ranks = own_season_ranks(distance, players)
    write_similarity_files(data_dir, similarity_documents(rows, lists, built))
    print("\n".join(similarity_summary(rows, inputs, lists, ranks)))
    size = sum(path.stat().st_size for path in data_dir.glob("*.json"))
    print(f"wrote {data_dir}: player_seasons.json and {rows.season.nunique()} season files, {size / 1e6:.1f} MB "
          f"({time.time() - started:.0f}s)", flush=True)


# ===================================================================================== the power ratings page
# Each team as it has played lately: every player's current-season rating times his share of the team's possessions
# over its last 10 games, summed over the team (the owner's design, 2026-10-28).  The ratings are centered on every
# possession and the shares count blowouts half, so the plain sums put the average team at +0.60 and 17 of 30 teams
# above it (2026-10-30): every player's rating is shifted down by a fifth of the 30 plain sums' mean (meta
# "average_team"), and shares add up to 500%, five on the floor, so the sums are points per 100 possessions above an
# average team and each player's row, his shifted rating and its share of the sum, adds up to his team's total (the
# owner, 2026-11-01: "adjust the team ratings to avg at 0").
#
# A possession counts for both teams of a game, on offense and on defense alike.  A blowout possession, the season's
# garbage time (config garbage_time: the 4th quarter with a margin of 20 or more, or 15 or more in its last 6
# minutes), counts half: "blowouts weighted somewhat less".  Each of a team's 5 latest games counts double the 5 before
# them (the owner, 2026-10-28: "weigh the most recent 5 more than the previous 5").
#
# The Power tab is hidden until every team has played `POWER_READY_GAMES` games (meta.ready; site.js shows the tab), the
# owner, 2026-10-28, then 2026-11-09: "'minimum=8' is a fine rule to swap in".
#
# The file, docs/data/power.json: {"meta": {"season", "through", "games", "recent_games", "recent_weight",
#     "blowout_weight", "ready", "built"},
#     "teams": [{"tm": "OKC", "name": "Oklahoma City Thunder", "g": 10, "o", "d", "t",
#                "players": [[name, id, share, o, d, t], ...]}, ...]}
# teams best total first, each team's players by share; share a percent at one decimal, ratings at two.  Players
# under the site's 60 offensive possessions (`MIN_POSSESSIONS_SHOWN`) count in their team's sums and are shown as one
# last row, ["Other players", null, their summed share, their share-weighted ratings].
POWER_FILE = "power.json"
POWER_GAMES = 10
POWER_READY_GAMES = 8
RECENT_GAMES = 5
RECENT_WEIGHT = 2.0
BLOWOUT_WEIGHT = 0.5
SLOTS = [f"{side}{i}" for side in "ha" for i in range(1, 6)]


def season_stints(season: int, cfg: dict) -> pd.DataFrame:
    """The season's stints, regular season and playoffs, whichever exist."""
    paths = [resolve(cfg, "stints") / f"{season}_{phase}.parquet" for phase in ("RS", "PO")]
    parts = [pd.read_parquet(path, columns=["game_id", "game_date", "poss_h", "poss_a", "is_gt", *SLOTS])
             for path in paths if path.exists()]
    if not parts:
        raise SystemExit(f"no stints for {season}; `openrapm data stints` builds them")
    return pd.concat(parts, ignore_index=True)


def game_teams(season: int, cfg: dict) -> pd.DataFrame:
    """Each player's team in each game of the season, from its game logs: game_id, player_id, tm, name."""
    paths = [raw_dir(cfg) / "gamelog" / f"{season}_{phase}.parquet" for phase in ("RS", "PO")]
    logs = pd.concat([pd.read_parquet(path, columns=["GAME_ID", "PLAYER_ID", "TEAM_ABBREVIATION", "TEAM_NAME"])
                      for path in paths if path.exists()], ignore_index=True)
    return logs.set_axis(["game_id", "player_id", "tm", "name"], axis=1).drop_duplicates(["game_id", "player_id"])


def player_game_possessions(stints: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    """One row per player per game: game_id, game_date, player_id, tm, name and poss, his possessions at both ends
    with each blowout possession counted `BLOWOUT_WEIGHT`."""
    weight = np.where(stints.is_gt, BLOWOUT_WEIGHT, 1.0) * (stints.poss_h + stints.poss_a)
    long = pd.concat([pd.DataFrame({"game_id": stints.game_id, "game_date": stints.game_date,
                                    "player_id": stints[slot].astype("int64"), "poss": weight})
                      for slot in SLOTS], ignore_index=True)
    games = long.groupby(["game_id", "game_date", "player_id"], as_index=False).poss.sum()
    games = games.merge(teams, on=["game_id", "player_id"], how="left")
    if games.tm.isna().any():
        missing = games[games.tm.isna()]
        raise SystemExit(f"{len(missing)} player-games of the stints are not in the game logs, e.g. "
                         f"{missing.iloc[0].to_dict()}")
    return games


def team_shares(games: pd.DataFrame, n_games: int, recent_games: int = RECENT_GAMES,
                recent_weight: float = RECENT_WEIGHT) -> pd.DataFrame:
    """Each player's share of his team's possessions over the team's last `n_games` games, its latest
    `recent_games` counted `recent_weight` times: tm, name, player_id, share (5.0 summed over a team) and g, the games
    counted.  A game's team possessions are its players' summed over five."""
    team_games = games.groupby(["tm", "game_id"], as_index=False).agg(game_date=("game_date", "first"),
                                                                       poss=("poss", "sum"))
    team_games["poss"] /= 5
    last = team_games.sort_values(["tm", "game_date", "game_id"]).groupby("tm").tail(n_games).copy()
    games_back = last.groupby("tm").cumcount(ascending=False)          # 0 for the team's latest game
    last["weight"] = np.where(games_back < recent_games, recent_weight, 1.0)
    last["poss"] *= last.weight
    recent = games.merge(last[["tm", "game_id", "weight"]], on=["tm", "game_id"])
    recent["poss"] *= recent.weight
    totals = last.groupby("tm").agg(team_poss=("poss", "sum"), g=("game_id", "size"))
    players = recent.groupby(["tm", "player_id"], as_index=False).agg(name=("name", "last"), poss=("poss", "sum"))
    players = players.join(totals, on="tm")
    players["share"] = players.poss / players.team_poss
    return players[["tm", "name", "player_id", "share", "g"]]


STATS_ROW = "Last 10 Performance Over Expected"  # the owner's name for a team's stats part, 2026-11-09
SHARES_TOTAL = 5.0     # five players on the floor: a team's shares of its possessions add up to 500%


def power_numbers(share, offense, defense, total) -> list:
    """A player row's numbers: his share as a percent at one decimal, his three ratings at two."""
    return [round(100 * float(share), 1), *(round(float(value), 2) for value in (offense, defense, total))]


def power_document(table: pd.DataFrame, games: pd.DataFrame, season: int, built: str,
                   shares: pd.DataFrame | None = None, forecast: dict | None = None) -> dict:
    """power.json's document for `season`: the teams, best total first, each with its players by share.  `shares`, when
    given, in place of the page's own.  `forecast`, the one rating (power_forecast.py): {"offense", "defense", the
    multipliers; "parts", the stats parts by team; "share", the share of the regular season played; "flat", whether the
    shares count each game once}.  Each player's row then reads his rating at the multipliers, and a team's stats part
    is the first row, [`STATS_ROW`, null, null, offense, defense, total], so the rows still add up to the team's
    total."""
    ratings = table[table.season == season].set_index("player_id")
    shares = team_shares(games, POWER_GAMES) if shares is None else shares
    unrated = sorted(set(shares.player_id) - set(ratings.index))
    if unrated:
        raise SystemExit(f"{len(unrated)} players of {season}'s games have no {season} rating, e.g. {unrated[:5]}")
    shares = shares.join(ratings[["player_name", "rating_off", "rating_def", "rating_total", "poss_season"]],
                         on="player_id")
    columns = ["rating_off", "rating_def", "rating_total"]
    # the 30 plain sums' mean per side, taken off every player's rating a fifth at a time: shares add to 500%, so the
    # teams then average 0 exactly and each player's row adds up to his team's total (the module comment)
    plain = shares.assign(**{column: shares.share * shares[column] for column in columns}).groupby("tm")[columns].sum()
    average = {side: float(plain[column].mean()) for side, column in zip("odt", columns)}
    shares = shares.assign(**{column: shares[column] - average[side] / SHARES_TOTAL
                              for side, column in zip("odt", columns)})
    if forecast is not None:
        offense, defense = shares.rating_off * forecast["offense"], shares.rating_def * forecast["defense"]
        shares = shares.assign(rating_off=offense, rating_def=defense, rating_total=offense + defense)
    teams = []
    for tm, rows in shares.groupby("tm"):
        rows = rows.sort_values("share", ascending=False)
        sums = [float((rows.share * rows[column]).sum()) for column in columns]
        if forecast is not None:
            part_o, part_d = float(forecast["parts"].o[tm]), float(forecast["parts"].d[tm])
            sums = [sums[0] + part_o, sums[1] + part_d, sums[2] + part_o + part_d]
        shown, others = rows[rows.poss_season >= MIN_POSSESSIONS_SHOWN], rows[rows.poss_season < MIN_POSSESSIONS_SHOWN]
        players = [[r.player_name if isinstance(r.player_name, str) else "", int(r.player_id), *power_numbers(
                    r.share, r.rating_off, r.rating_def, r.rating_total)] for r in shown.itertuples(index=False)]
        if len(others):
            share = others.share.sum()
            players.append(["Other players", None, *power_numbers(
                share, *((others.share * others[column]).sum() / share for column in columns))])
        if forecast is not None and (part_o or part_d):
            players.insert(0, [STATS_ROW, None, None, round(part_o, 2), round(part_d, 2), round(part_o + part_d, 2)])
        teams.append(dict(tm=tm, name=rows.name.iloc[0], g=int(rows.g.iloc[0]),
                          **{side: round(value, 2) for side, value in zip("odt", sums)}, players=players))
    teams.sort(key=lambda team: -team["t"])
    meta = dict(season=season, through=pd.Timestamp(games.game_date.max()).strftime("%Y-%m-%d"), games=POWER_GAMES,
                recent_games=RECENT_GAMES, recent_weight=RECENT_WEIGHT, blowout_weight=BLOWOUT_WEIGHT,
                ready=min(team["g"] for team in teams) >= POWER_READY_GAMES, built=built,
                average_team={side: round(value, 2) for side, value in average.items()})
    if forecast is not None:
        meta.update(forecast=True, share_played=round(forecast["share"], 4), flat_shares=forecast["flat"],
                    recent_weight=1.0 if forecast["flat"] else RECENT_WEIGHT)
    return dict(meta=meta, teams=teams)


def write_power(cfg: dict, ratings_path: Path, data_dir: Path, built: str, forecast: bool = False) -> None:
    """power.json in `data_dir`, for the ratings table's latest season: with `forecast`, the one rating of
    power_forecast.py, which the morning job does not write until the owner's word."""
    table = pd.read_parquet(ratings_path)
    season = int(table.season.max())
    games = player_game_possessions(season_stints(season, cfg), game_teams(season, cfg))
    if forecast:
        from ..data.seasons import share_played
        share = share_played(season, cfg)
        weights = weights_at(share, load_knots(resolve_path(cfg, WEIGHTS_FILE)))
        flat = flat_shares(share)
        shares = team_shares(games, POWER_GAMES, recent_weight=1.0 if flat else RECENT_WEIGHT)
        parts = stats_parts(last_ten_stats(team_game_counts(season, cfg)), weights)
        document = power_document(table, games, season, built, shares=shares,
                                  forecast=dict(offense=weights["offense_multiplier"],
                                                defense=weights["defense_multiplier"], parts=parts, share=share,
                                                flat=flat))
    else:
        document = power_document(table, games, season, built)
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / POWER_FILE
    path.write_text(json.dumps(document, separators=(",", ":")), encoding="utf-8")
    top = document["teams"][0]
    print(f"wrote {path}: {len(document['teams'])} teams of {season} through {document['meta']['through']}, top "
          f"{top['tm']} {top['t']:+.2f}", flush=True)


# ===================================================================================== the bias page
# docs/data/bias.json, which docs/bias.html draws (the owner's design, 2026-11-05): for the season the pages open on,
# how much higher or lower OpenRAPM rates players one standard deviation higher in each stat than a consensus of three
# outside systems does ("vs consensus"); and than the results of games the ratings never saw do, over the last ten
# finished seasons (the owner, 2026-11-24, on the in-sample game-results columns this replaced, which graded each
# season's ratings on the games they were fit on: "Held-back, drop ’27").  That column is fixed until the next build
# is adopted: `BIAS_HELD_BACK` holds it, written by scratch/2026-11-24_bias_heldback/write_params.py from the
# incumbent's within-season test (notes/DECISIONS.md, "The Bias page's leans on held-back games").  bias.py holds the
# consensus column's equations; the file:
#
#   {"meta": {"season", "through", "built", "min_possessions", "players_shown", "players",
#             "multipliers": {"o", "d"}, "games_seasons": {"first", "last", "build"}, "fields": BIAS_FIELDS},
#    "systems": [{"key", "name", "by", "link", "weight", "as_of", "players"}, ...],    the three, heaviest first
#    "rows": [[stat, label, o, d, t, o_se, d_se, t_se, sd, gs, gs_se], ...]}          one per stat, bias.STATS order
#
# o and d are written at two decimals and t is their sum, so every row adds up on the page; gs, OpenRAPM minus the
# held-back games, and its standard error, at two.  players_shown is the season's players with 60 or more offensive
# possessions (`MIN_POSSESSIONS_SHOWN`), and players those of them all three systems list; their mean and spread of
# each stat are the standard deviation the consensus column is per (bias.stat_z).  A system's own `players` is how many
# of the shown it lists; its `as_of` is its own date, or the day it was pulled when its page prints none.  Only these
# slopes are published: the systems' ratings stay in paths.public_ratings, out of git.
BIAS_FILE = "bias.json"
BIAS_FIELDS = ["stat", "label", "o", "d", "t", "o_se", "d_se", "t_se", "sd", "gs", "gs_se"]
BIAS_HELD_BACK = "params/bias_held_back.json"
BIAS_LABELS = {ON_COURT_PLUS_MINUS: "On-court plus-minus", OFF_COURT_PLUS_MINUS: "Off-court plus-minus"}


def bias_label(stat: str) -> str:
    """A row's name on the page: the catalogue's words for a box rate (`features.meaning`, "Made threes"), and the
    two plus-minus rows by name."""
    return BIAS_LABELS.get(stat) or meaning(stat).capitalize()


def _rounded(value, digits: int = 2):
    """A number as the file holds it; None where there is none (a stat with no spread has no slope)."""
    return round(float(value), digits) if np.isfinite(value) else None


def bias_row(row) -> list:
    """One stat's row in `BIAS_FIELDS` order: offense and defense at two decimals, total their sum, then the
    held-back games' lean and its standard error."""
    offense, defense = _rounded(row.offense), _rounded(row.defense)
    total = None if offense is None or defense is None else round(offense + defense, 2)
    return [row.stat, bias_label(row.stat), offense, defense, total, _rounded(row.offense_se),
            _rounded(row.defense_se), _rounded(row.total_se), _rounded(row.sd, 3), _rounded(row.held_back),
            _rounded(row.held_back_se)]


def against_openrapm(players: pd.DataFrame, pulls: dict) -> list:
    """The systems whose ratings run against OpenRAPM's on a side among the compared players, as "DARKO defense":
    every system agrees with OpenRAPM far above 0 (0.5 to 0.8 on 2026-11-05), so one at 0 or below changed a sign."""
    out = []
    for key in CONSENSUS_WEIGHTS:
        theirs = players[["player_id"]].merge(pulls[key].table, on="player_id")
        out += [f"{public_ratings.SYSTEMS[key].name} {side}" for side in SIDES
                if not np.corrcoef(players[OURS[side]], theirs[side])[0, 1] > 0]
    return out


def held_back_leans(path: Path) -> dict:
    """`BIAS_HELD_BACK` as read: {"first", "last", "build", "rows": {stat: [lean, standard error]}}, every stat of
    bias.STATS present."""
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    missing = [stat for stat in STATS if stat not in document["rows"]]
    if missing:
        raise SystemExit(f"{path} has no held-back lean for {', '.join(missing)}")
    return document


def bias_document(table: pd.DataFrame, pulls: dict, season: int, built: str, through: str, panel: pd.DataFrame,
                  held_back: dict) -> dict:
    """bias.json's document for `season`, from the ratings table, each system's stored pull ({key:
    `public_ratings.PublicRatings`}), the season's panel rows and the held-back games' leans (`held_back_leans`)."""
    shown = table[(table.season == season) & (table.poss_season >= MIN_POSSESSIONS_SHOWN)]
    blend = consensus({key: pulls[key].table for key in CONSENSUS_WEIGHTS})
    stats = player_stats(panel)
    players = compared_players(shown, blend, stats)
    flipped = against_openrapm(players, pulls)
    if flipped:
        raise SystemExit(f"{', '.join(flipped)} runs against OpenRAPM's among {len(players)} players: its page "
                         f"changed a sign (data/public_ratings.py); the bias file is left as it was")
    slopes, multipliers = bias_table(players)
    leans = pd.DataFrame([(stat, *values) for stat, values in held_back["rows"].items()],
                         columns=["stat", "held_back", "held_back_se"])
    slopes = slopes.merge(leans, on="stat", how="left")
    systems = [dict(key=key, name=public_ratings.SYSTEMS[key].name, by=public_ratings.SYSTEMS[key].by,
                    link=public_ratings.SYSTEMS[key].link, weight=weight,
                    as_of=pulls[key].as_of or pulls[key].pulled,
                    players=int(shown.player_id.isin(pulls[key].table.player_id).sum()))
               for key, weight in CONSENSUS_WEIGHTS.items()]
    meta = dict(season=season, through=through, built=built, min_possessions=MIN_POSSESSIONS_SHOWN,
                players_shown=len(shown), players=len(players),
                multipliers=dict(o=_rounded(multipliers["offense"], 3), d=_rounded(multipliers["defense"], 3)),
                games_seasons=dict(first=held_back["first"], last=held_back["last"], build=held_back["build"]),
                fields=BIAS_FIELDS)
    return dict(meta=meta, systems=systems, rows=[bias_row(row) for row in slopes.itertuples(index=False)])


def bias_summary(document: dict) -> list[str]:
    """The console report: the players compared, the multipliers, each system's date, and the table: offense,
    defense, total (vs consensus), vs the held-back games."""
    meta = document["meta"]
    lines = [f"  {meta['players']} of the {meta['players_shown']} players shown for {meta['season']} are in all "
             f"{len(document['systems'])} systems; OpenRAPM multiplied by {meta['multipliers']['o']:.3f} on offense "
             f"and {meta['multipliers']['d']:.3f} on defense",
             "  " + ", ".join(f"{system['name']} {system['weight']:.0%} as of {system['as_of']}"
                              for system in document["systems"]),
             f"  held-back games of {meta['games_seasons']['first']} to {meta['games_seasons']['last']}, "
             f"build {meta['games_seasons']['build']}"]
    field = {name: position for position, name in enumerate(meta["fields"])}
    columns = ["o", "d", "t", "gs"]
    lines += [f"  {row[1]:<22}" + "".join("     n/a" if row[field[name]] is None else f"{row[field[name]]:+8.2f}"
                                          for name in columns)
              + f"   standard errors {row[field['t_se']]}, {row[field['gs_se']]}"
              for row in document["rows"]]
    return lines


def write_bias(cfg: dict, ratings_path: Path, data_dir: Path, built: str) -> None:
    """bias.json in `data_dir`, for the season the pages open on (`default_season`), from each system's newest stored
    pull, the season's panel and the held-back games' leans (`BIAS_HELD_BACK`).  A system never pulled for that
    season, or dated before the season's first game, stops the command with one line and the file as it was; so does
    a system that runs against OpenRAPM (`bias_document`)."""
    table = pd.read_parquet(ratings_path)
    season = default_season(table)
    pulls = public_ratings.latest(cfg, season)
    missing = [public_ratings.SYSTEMS[key].name for key in CONSENSUS_WEIGHTS if key not in pulls]
    if missing:
        raise SystemExit(f"no stored table of {', '.join(missing)} for {season}; `openrapm data public-ratings` "
                         f"pulls them")
    dates = pd.to_datetime(season_stints(season, cfg).game_date)
    first, through = dates.min().strftime("%Y-%m-%d"), dates.max().strftime("%Y-%m-%d")
    early = [f"{public_ratings.SYSTEMS[key].name} {pulls[key].as_of}" for key in CONSENSUS_WEIGHTS
             if pulls[key].as_of and pulls[key].as_of < first]
    if early:
        raise SystemExit(f"{', '.join(early)}: dated before {season}'s first game, {first}; the bias file is left "
                         f"as it was")
    held_back = held_back_leans(resolve_path(cfg, BIAS_HELD_BACK))
    document = bias_document(table, pulls, season, built, through, panel_of_season(cfg, season), held_back)
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / BIAS_FILE
    part = path.with_name(BIAS_FILE + ".part")      # written whole, then moved: the page never reads half a file
    part.write_text(json.dumps(document, separators=(",", ":")), encoding="utf-8")
    part.replace(path)
    print(f"wrote {path}: {season} through {through}", flush=True)
    print("\n".join(bias_summary(document)), flush=True)
