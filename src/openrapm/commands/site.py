"""`openrapm site`: the data behind the published site, docs/.

    openrapm site                    docs/data/ratings.json, the ratings as one file (the pages read
                                     decompose_openrapm.json, below, which carries them with their pieces)
    openrapm site --bias             also docs/data/bias.json, the Bias page: where OpenRAPM rates players higher or
                                     lower than a consensus of five outside systems, per standard deviation of a stat
    openrapm site --decomposition    also docs/data/decompose_1y.json and decompose_3y.json, the vanilla RAPM
                                     decomposition page
    openrapm site --openrapm-decomposition
                                     also docs/data/decompose_openrapm.json, the OpenRAPM decomposition page: every
                                     published rating split into pieces that add up to it
    openrapm site --similarity       also docs/data/similarity/, the similarity page: each player-season's nearest
                                     player-seasons of other players
    openrapm site --power            also docs/data/power.json, the power ratings page: each team's current ratings
                                     times its players' shares of its possessions over its last 10 games

The pages' wording and look are not this command's to change: it writes data.

Words used below
    ratings table   outputs/season_ratings_product.parquet, one row per player per season (`openrapm build`)
    built           the date in a file's meta.built.  ratings.json shows the ratings table's modification date,
                    so re-exporting the same table does not tell readers the ratings changed; the decomposition
                    files show the day they were computed.  --built sets it for every file written
    consensus       on the Bias page, the weighted mean of five outside systems' ratings (bias.py), whose tables
                    `openrapm data public-ratings` pulls; not data/external/consensus.csv, the adoption check's
    vanilla RAPM    plain ridge regression with no box-score prior: raw points, regular season and playoffs

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
import scipy.linalg as sla
import scipy.sparse as sp

from ..bias import (
    CONSENSUS_WEIGHTS,
    OFF_COURT_PLUS_MINUS,
    ON_COURT_PLUS_MINUS,
    OURS,
    SIDES,
    bias_table,
    compared_players,
    consensus,
    player_stats,
)
from ..data import public_ratings
from ..data.config import resolve, resolve_path
from ..data.designs import DesignCache, block_design
from ..data.ingest import raw_dir
from ..data.panel import panel_of_season, rookies, trainable_panel
from ..data.roles import main_teams
from ..decomposition import lineups, refit, row_groups, row_keys, side_team, split_games_part, team_codes
from ..features import FEATURE_SETS, meaning
from ..prior_shrink import team_split
from ..rapm import SeasonSums
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
             decomposition: bool = False, similarity: bool = False, openrapm_decomposition: bool = False,
             power: bool = False) -> int:
    """`openrapm site`: ratings.json always; the decomposition files, the similarity files, the OpenRAPM decomposition
    file, the power ratings file and the Bias page's file when asked.

    `ratings` defaults to the ratings table under config paths.outputs and `out_dir` to the repository's docs/.  The
    Bias page's file is written last: it alone needs the outside systems' tables, and a missing one then stops the
    command with every other file already written.
    """
    ratings_path = Path(ratings) if ratings else resolve(cfg, "outputs") / RATINGS_TABLE
    if not ratings_path.exists():
        raise SystemExit(f"{ratings_path} does not exist; `openrapm build` writes it")
    out_dir = Path(out_dir) if out_dir else resolve_path(cfg, SITE_FOLDER)
    write_ratings_json(ratings_path, out_dir / "data" / "ratings.json", built or table_date(ratings_path))
    if decomposition:
        write_decomposition(cfg, ratings_path, out_dir / "data", built or today())
    if similarity:
        write_similarity(cfg, ratings_path, out_dir / "data" / SIMILARITY_FOLDER, built or today())
    if openrapm_decomposition:
        write_openrapm_decomposition(cfg, ratings_path, out_dir / "data", built or table_date(ratings_path))
    if power:
        write_power(cfg, ratings_path, out_dir / "data", built or table_date(ratings_path))
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


# ===================================================================================== the decomposition page
# Every player's vanilla RAPM over one season and over each three-season window, split two ways that each add up
# to it exactly: by player and by possession.  One fit per span (each season alone, and each of config.yaml's ten
# three-season windows), penalty 3,000 per side, the owner's choice when the page was built (2026-09-27).
#
# By player, who was on the court:
#
#     RAPM = on-court rating + teammates + opponents + context + ridge penalty
#
#   on-court rating  his team's points per 100 possessions minus the opponents', while he is on the court
#   teammates        minus the RAPM of the four teammates on the court with him, added up, weighted by time together
#   opponents        the RAPM of the five opponents on the court with him, added up, weighted by time
#   context          minus what home court, playoffs, garbage time, the score margin and the season predict for his
#                    minutes
#   ridge penalty    the ridge's pull toward zero: minus penalty x his coefficient / his possessions, on each side
#
# It is exact because it is the ridge's own first-order condition for his offensive and his defensive column: the
# residuals of the possessions he played add up to penalty x his coefficient.  On the published zero point
# (possession-weighted zero per side) each side picks up a league-wide constant, 5 x the two sides' shifts, and the
# two cancel in the net.  Each piece averages zero over the league, weighted by possessions.
#
# By possession, which possessions the rating came from:
#
#     RAPM = his minutes + his team without him + games he missed + everything else
#
# RAPM is linear in the outcomes: his published rating is c'beta = sum over rows of h_r y_r, h = W X A^-1 c, with
# A = X'WX + the penalty, W the possessions and c picking his offence minus his defence less the zero point.  The
# context columns are unpenalised, so h is orthogonal to each of them, and measuring every outcome against its
# fitted context prediction leaves the total unchanged.  Each piece adds up h_r x (outcome - context prediction)
# over one group of rows: the possessions he played; his team's possessions without him in games he played; his
# teams' possessions in games he did not play, in seasons he played for them; and every other possession in the
# span.  A row's team comes from the box scores (the team of the players on that side).  A season he missed
# entirely lands in everything else; for a traded player, his other team's games count as games he missed.  The
# page also shows his actual off-court rating beside the pieces (not a piece).
#
# Example (2026, one season): Victor Wembanyama, 7.16 = 15.34 - 8.28 + 2.45 + 1.95 - 4.30 by player,
# and 7.20 - 0.19 - 0.45 + 0.60 by possession.  The command stops on a failed check, before writing the span's file.
DECOMPOSITION_PENALTY = 3000.0
# a row's fields: name, possessions, the five pieces by player, RAPM, the four by possession, the off-court rating,
# the NBA id (in the files and the CSV download, not on the page)
DECOMPOSITION_FIELDS = ["n", "p", "m", "t", "o", "x", "r", "a", "h", "w", "g", "e", "f", "id"]
PIECES = ["margin", "teammates", "opponents", "context", "ridge"]
SOURCES = ["his_minutes", "team_without_him", "games_missed", "everything_else"]
IDENTITY_TOLERANCE = 1e-9      # each identity holds to this, per player, per span
UNPLACED_TOLERANCE = 0.001     # rows whose team the box scores cannot name: under this share of the span's rows
CENTRE_TOLERANCE = 0.05        # a piece's possession-weighted league average must sit this close to zero


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


def span_label(seasons) -> str:
    """"2026" for one season, "2024-26" for a window."""
    return str(seasons[0]) if len(seasons) == 1 else f"{seasons[0]}-{str(seasons[-1])[-2:]}"


def span_sums(seasons, cfg: dict) -> tuple[SeasonSums, dict]:
    """Each season's points design, and the normal equations of all of them in one rapm.SeasonSums."""
    sums, designs = SeasonSums(min_possessions=0.0), {}
    for season in seasons:
        designs[season] = block_design([season], cfg, target="pts")
        sums.add(season, designs[season])
    return sums, designs


def season_rows(sums: SeasonSums, design, season: int, first_context: int, n_columns: int) -> dict:
    """One season's rows in the span's columns: the sparse rows (ten players, then the context), both lineups as
    span columns, the weights, the targets and each row's game id."""
    n_players = len(sums.player_ids)
    slot = sums.player_columns(season)
    context = np.asarray(design.parts["F"], dtype=float)
    rows, n_context = context.shape
    if n_context != sums.context_size(season):
        raise SystemExit(f"{season}: {n_context} context columns in the design, {sums.context_size(season)} in "
                         "the fit")
    lineup_o, lineup_d = slot[np.asarray(design.parts["lineup_o"])], slot[np.asarray(design.parts["lineup_d"])]
    indices = np.hstack([lineup_o, lineup_d + n_players,
                         np.broadcast_to(first_context + np.arange(n_context), (rows, n_context))])
    data = np.hstack([np.ones((rows, 10)), context])
    row_starts = np.arange(0, rows * (10 + n_context) + 1, 10 + n_context, dtype=np.int64)
    game_ids = design.games.set_index("game_idx").game_id.reindex(design.rows.game_idx.to_numpy()).to_numpy() \
        .astype(str)
    return dict(X=sp.csr_matrix((data.ravel(), indices.ravel(), row_starts), shape=(rows, n_columns)),
                lineup_o=lineup_o, lineup_d=lineup_d, w=np.asarray(design.w, dtype=float),
                y=np.asarray(design.y, dtype=float), game_ids=game_ids, n_context=n_context)


def span_rows(sums: SeasonSums, designs: dict, box_files: BoxScoreFiles, n_columns: int) -> dict:
    """Every row of the span, seasons in `assemble`'s order so its context columns are ours, with each row's
    attacking and defending team and season."""
    column, blocks = 2 * len(sums.player_ids), []
    for season in sums.seasons:
        block = season_rows(sums, designs[season], season, column, n_columns)
        teams = box_files.teams(season)
        block["team_att"] = side_team(teams, block["game_ids"], sums.player_ids[block["lineup_o"]])
        block["team_def"] = side_team(teams, block["game_ids"], sums.player_ids[block["lineup_d"]])
        block["season"] = np.full(len(block["w"]), season, dtype=np.int64)
        blocks.append(block)
        column += block["n_context"]
    joined = {key: np.concatenate([block[key] for block in blocks])
              for key in ("w", "y", "game_ids", "team_att", "team_def", "season")}
    joined.update(X=sp.vstack([block["X"] for block in blocks]).tocsr(),
                  lineup_o=np.vstack([block["lineup_o"] for block in blocks]),
                  lineup_d=np.vstack([block["lineup_d"] for block in blocks]))
    return joined


def ridge_solve(gram: np.ndarray, rhs: np.ndarray, n_players: int, n_context: int, penalty: float):
    """(coefficients, the diagonal of G, the Cholesky factor of G + penalty): the plain ridge, context unpenalised.

    An empty context column (the 2020 bubble's playoff home court) gets a token penalty of 1: its coefficient is
    then exactly 0 and nothing else moves, where unpenalised it makes the system singular.
    """
    diagonal = np.diagonal(gram)
    penalties = np.concatenate([np.full(2 * n_players, float(penalty)), np.zeros(n_context)])
    penalties[(diagonal <= 0) & (penalties == 0)] = 1.0
    factor = sla.cho_factor(gram + np.diag(penalties))
    return sla.cho_solve(factor, rhs), diagonal, factor


def published_zero(beta: np.ndarray, diagonal: np.ndarray, possessions: np.ndarray, n_players: int) -> dict:
    """Both sides centred as the ratings pages centre them, positive good: each side's possession-weighted mean
    over players with at least the published minimum possessions is zero.  Also the two weight vectors."""
    keep = possessions >= PRODUCT.labels.min_possessions
    weight_off = np.where(keep, possessions, 0.0)
    weight_def = np.where(keep, diagonal[n_players:2 * n_players], 0.0)
    offence, defence = beta[:n_players], beta[n_players:2 * n_players]
    return dict(off=offence - weight_off @ offence / weight_off.sum(),
                dfn=-(defence - weight_def @ defence / weight_def.sum()),
                w_off=weight_off / weight_off.sum(), w_def=weight_def / weight_def.sum())


def span_fit(seasons, cfg: dict, penalty: float, box_files: BoxScoreFiles) -> dict:
    """One span's vanilla RAPM: its rows in the span's columns, every row's ten players, the fit, the ratings."""
    sums, designs = span_sums(seasons, cfg)
    gram, rhs, possessions, n_context = sums.assemble(())
    n_players = possessions.size
    rows = span_rows(sums, designs, box_files, 2 * n_players + n_context)
    beta, diagonal, factor = ridge_solve(gram, rhs, n_players, n_context, penalty)
    return dict(n=n_players, X=rows["X"], lineup_o=rows["lineup_o"], lineup_d=rows["lineup_d"], w=rows["w"],
                y=rows["y"], beta=beta, diag=diagonal, poss=possessions, player_ids=sums.player_ids.copy(),
                lam=float(penalty), factor=factor, keys=team_codes(rows),
                **published_zero(beta, diagonal, possessions, n_players))


def weighted_mean(values: np.ndarray, rows, weights: np.ndarray) -> float:
    """The mean of `values` over `rows`, weighted by possessions."""
    return float(np.average(values[rows], weights=weights[rows]))


def row_terms(fit: dict) -> dict:
    """What every player's split reads per row: the residual, the context prediction, the outcome against it, the
    two lineups' rating sums, which players are on each row, and A^-1 with the zero point's part of A^-1 c."""
    n, X, w, beta = fit["n"], fit["X"], fit["w"], fit["beta"]
    rows = np.arange(len(w))
    context_row = X[:, 2 * n:] @ beta[2 * n:]
    zero_point = np.zeros(X.shape[1])
    zero_point[:n], zero_point[n:2 * n] = -fit["w_off"], fit["w_def"]
    inverse = sla.cho_solve(fit["factor"], np.eye(X.shape[1]))
    return dict(resid=fit["y"] - X @ beta, context_row=context_row, z=fit["y"] - context_row,
                sum_o=fit["off"][fit["lineup_o"]].sum(axis=1), sum_d=fit["dfn"][fit["lineup_d"]].sum(axis=1),
                at_o=sp.csc_matrix((np.ones(fit["lineup_o"].size), (np.repeat(rows, 5), fit["lineup_o"].ravel())),
                                   shape=(len(w), n)),
                at_d=sp.csc_matrix((np.ones(fit["lineup_d"].size), (np.repeat(rows, 5), fit["lineup_d"].ravel())),
                                   shape=(len(w), n)),
                inverse=inverse, u0=inverse @ zero_point)


def player_pieces(i: int, rows_off, rows_def, fit: dict, terms: dict) -> dict:
    """The five pieces by player and his RAPM (offence plus defence, positive good)."""
    y, w, off, dfn = fit["y"], fit["w"], fit["off"], fit["dfn"]
    return dict(margin=weighted_mean(y, rows_off, w) - weighted_mean(y, rows_def, w),
                teammates=-((weighted_mean(terms["sum_o"], rows_off, w) - off[i])
                            + (weighted_mean(terms["sum_d"], rows_def, w) - dfn[i])),
                opponents=weighted_mean(terms["sum_d"], rows_off, w) + weighted_mean(terms["sum_o"], rows_def, w),
                context=-(weighted_mean(terms["context_row"], rows_off, w)
                          - weighted_mean(terms["context_row"], rows_def, w)),
                ridge=-(weighted_mean(terms["resid"], rows_off, w) - weighted_mean(terms["resid"], rows_def, w)),
                rapm=off[i] + dfn[i])


def possession_groups(i: int, rows_off, rows_def, fit: dict, terms: dict) -> dict:
    """The four pieces by possession, and his actual off-court rating (his team's net points per 100 without
    him, in games he played; missing when one side has no such rows)."""
    n, w, y = fit["n"], fit["w"], fit["y"]
    hz = w * (fit["X"] @ (terms["u0"] + terms["inverse"][:, i] - terms["inverse"][:, n + i])) * terms["z"]
    rows = row_groups(rows_off, rows_def, fit["keys"], len(w))
    on, without, missed = rows["on"], rows["without"], rows["missed"]
    groups = dict(his_minutes=hz[on].sum(), team_without_him=hz[without].sum(), games_missed=hz[missed].sum(),
                  everything_else=hz[~(on | without | missed)].sum())
    off_att, off_def = rows["off_att"], rows["off_def"]
    both_sides = off_att.any() and off_def.any()
    groups["off_rtg"] = weighted_mean(y, off_att, w) - weighted_mean(y, off_def, w) if both_sides else np.nan
    return groups


def record_misses(miss: dict, record: dict, i: int, rows_off, rows_def, fit: dict, terms: dict) -> None:
    """Keep the largest miss of each identity the split rests on."""
    n, w, beta, diagonal = fit["n"], fit["w"], fit["beta"], fit["diag"]
    miss["identity"] = max(miss["identity"], abs(sum(record[k] for k in PIECES) - record["rapm"]))
    miss["sources"] = max(miss["sources"], abs(sum(record[k] for k in SOURCES) - record["rapm"]))
    miss["ridge"] = max(miss["ridge"],
                        abs(weighted_mean(terms["resid"], rows_off, w) - fit["lam"] * beta[i] / diagonal[i]),
                        abs(weighted_mean(terms["resid"], rows_def, w) - fit["lam"] * beta[n + i] / diagonal[n + i]))
    miss["weights"] = max(miss["weights"], abs(w[rows_off].sum() - fit["poss"][i]) / fit["poss"][i])


def decompose(fit: dict) -> tuple[pd.DataFrame, dict]:
    """Every player's five pieces by player and four by possession, and the largest misses of the identities they
    rest on.  Players who played on both sides only."""
    terms = row_terms(fit)
    at_o, at_d = terms["at_o"], terms["at_d"]
    miss = dict(identity=0.0, ridge=0.0, weights=0.0, sources=0.0)
    records = []
    for i in np.flatnonzero((np.diff(at_o.indptr) > 0) & (np.diff(at_d.indptr) > 0)):
        rows_off = at_o.indices[at_o.indptr[i]:at_o.indptr[i + 1]]
        rows_def = at_d.indices[at_d.indptr[i]:at_d.indptr[i + 1]]
        record = dict(i=i, poss=fit["poss"][i], **player_pieces(i, rows_off, rows_def, fit, terms))
        record.update(possession_groups(i, rows_off, rows_def, fit, terms))
        record_misses(miss, record, i, rows_off, rows_def, fit, terms)
        records.append(record)
    return pd.DataFrame(records), miss


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


def decomposition_row(r) -> list:
    """One player's row in DECOMPOSITION_FIELDS order: numbers at two decimals, possessions whole."""
    return [r.name, int(round(r.poss)), *(round(float(getattr(r, k)), 2) for k in PIECES), round(float(r.rapm), 2),
            *(round(float(getattr(r, k)), 2) for k in SOURCES),
            None if np.isnan(r.off_rtg) else round(float(r.off_rtg), 2), int(r.player_id)]


def new_worst() -> dict:
    return dict(identity=0.0, sources=0.0, ridge=0.0, weights=0.0, centre=0.0, unnamed=0, unplaced=0.0)


def update_worst(worst: dict, table: pd.DataFrame, miss: dict, unplaced: float) -> None:
    """Fold one span's misses into the worst over every span of its length."""
    worst["unplaced"] = max(worst["unplaced"], unplaced)
    for key in miss:
        worst[key] = max(worst[key], miss[key])
    for piece in PIECES:
        worst["centre"] = max(worst["centre"], abs(float(np.average(table[piece], weights=table.poss))))
    worst["unnamed"] += int(table["name"].isna().sum())


def check(ok: bool, what: str) -> None:
    """Print a check; stop the run on a failure, so nothing is written past one."""
    print(("  PASS  " if ok else "  FAIL  ") + what, flush=True)
    if not ok:
        raise SystemExit(f"check failed: {what}")


def run_checks(span_words: str, worst: dict) -> None:
    """The seven checks on every span of one length, before its file is written."""
    check(worst["identity"] < IDENTITY_TOLERANCE, f"{span_words}: the five pieces add up to RAPM for every player "
                                                  f"of every span (max miss {worst['identity']:.1e})")
    check(worst["sources"] < IDENTITY_TOLERANCE, f"{span_words}: the four possession groups add up to RAPM for "
                                                 f"every player of every span (max miss {worst['sources']:.1e})")
    check(worst["unplaced"] < UNPLACED_TOLERANCE, f"{span_words}: the box scores name both teams of nearly every "
                                                  f"row (largest share without, in one span: "
                                                  f"{worst['unplaced']:.2%})")
    check(worst["ridge"] < IDENTITY_TOLERANCE, f"{span_words}: 'ridge penalty' equals penalty x coefficient / "
                                               f"possessions on each side, the fit's own condition (max miss "
                                               f"{worst['ridge']:.1e})")
    check(worst["weights"] < IDENTITY_TOLERANCE, f"{span_words}: the fit's weights are his possessions, so 'per 100 "
                                                 f"possessions' is literal (max relative miss "
                                                 f"{worst['weights']:.1e})")
    check(worst["centre"] < CENTRE_TOLERANCE, f"{span_words}: every piece averages zero over the league, weighted "
                                              f"by possessions (largest {worst['centre']:.3f})")
    check(worst["unnamed"] == 0, f"{span_words}: every player has a name ({worst['unnamed']} without)")


def span_table(seasons, cfg: dict, names: pd.DataFrame, box_files: BoxScoreFiles, worst: dict) -> pd.DataFrame:
    """One span's players with names and NBA ids, best RAPM first; its misses folded into `worst`."""
    fit = span_fit(seasons, cfg, DECOMPOSITION_PENALTY, box_files)
    table, miss = decompose(fit)
    ids = fit["player_ids"][table.i.to_numpy()]
    table["name"] = span_names(names, ids, seasons, box_files)
    table["player_id"] = ids
    update_worst(worst, table, miss, fit["keys"]["unplaced"])
    return table.sort_values("rapm", ascending=False)


def decomposition_spans(cfg: dict) -> dict:
    """{file key: (the words for the span, the spans)}: every season alone, and config.yaml's three-season windows."""
    return {"1y": ("1 season", [[s] for s in range(int(cfg["first_season"]), int(cfg["last_season"]) + 1)]),
            "3y": ("3 seasons", [list(range(int(a), int(b) + 1)) for a, b in cfg["windows"]])}


def write_span_file(path: Path, span_words: str, rows: dict, built: str) -> None:
    """decompose_<key>.json: meta and the rows of every span, latest first; compact, UTF-8 letters as they are."""
    periods = sorted(rows, reverse=True)
    document = dict(meta=dict(span=span_words, penalty=DECOMPOSITION_PENALTY, built=built,
                              fields=DECOMPOSITION_FIELDS, periods=periods),
                    rows={period: rows[period] for period in periods})
    path.write_text(json.dumps(document, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


def write_decomposition(cfg: dict, ratings_path: Path, data_dir: Path, built: str) -> None:
    """decompose_1y.json and decompose_3y.json in `data_dir`, each after its checks pass."""
    names, box_files = name_table(ratings_path, cfg), BoxScoreFiles(cfg)
    data_dir.mkdir(parents=True, exist_ok=True)
    for key, (span_words, spans) in decomposition_spans(cfg).items():
        started, rows, worst = time.time(), {}, new_worst()
        for seasons in spans:
            table = span_table(seasons, cfg, names, box_files, worst)
            rows[span_label(seasons)] = [decomposition_row(r) for r in table.itertuples(index=False)]
            top = table.iloc[0]
            print(f"  {span_label(seasons)}: {len(table)} players, top {top['name']} {top.rapm:+.2f} "
                  f"({time.time() - started:.0f}s)", flush=True)
        run_checks(span_words, worst)
        path = data_dir / f"decompose_{key}.json"
        write_span_file(path, span_words, rows, built)
        print(f"wrote {path}: {sum(len(v) for v in rows.values()):,} rows over {len(rows)} periods, "
              f"{path.stat().st_size / 1e6:.2f} MB ({time.time() - started:.0f}s)", flush=True)


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
#     {"meta": {built, penalty, seasons, fields, pieces}, "rows": {season: [[name, NBA id, possessions, offence,
#      defence, total, teams], ...]}}     each side's list holds OPENRAPM_PIECES in order, at two decimals; teams
#      are the box scores' abbreviations in the order he joined them, "SAC/IND"; then his wins over starter average,
#      his regular-season minutes, and per side, offence, defence and total, [his regular-season rating over the
#      starter average, his leverage, his wins over starter average] (WOSA_PARTS, wins.py says how they multiply)
# Rows are the ratings.json rows (possessions above 0), best total first.
#
# Example (2026): Victor Wembanyama's offence, 3.01 = 2.11 + 0.26 + 0.86 - 0.32 + 0.11; his total, 9.37 = 3.89 + 2.70
# + 2.98 - 0.43 + 0.23: box score, Box Adj, on court, off court, team SOS.  The page's headers for on court and off
# court are On adj and Off adj.
OPENRAPM_FILE = "decompose_openrapm.json"
OPENRAPM_FIELDS = ["n", "id", "p", "o", "d", "t", "tm", "w", "m", "wo", "wd", "wt"]
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

    The offence half comes from the fit on the offence target, the defence half from the fit on the defence target,
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
    diagonal, and the prior columns here are the table's (scaled and centred) rather than the build's own, so the
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


def openrapm_rows(rows: pd.DataFrame, pieces: pd.DataFrame, names: np.ndarray, teams: dict | None = None) -> list:
    """One season's rows in OPENRAPM_FIELDS order, best total first, players under 60 possessions left out (as
    ratings.json leaves them): name, NBA id, possessions (whole, as ratings.json writes them), each side's pieces at
    two decimals, then his teams ({player_id: "SAC/IND"}; "" for a player it lacks), then his wins over starter average
    at two decimals, his regular-season minutes, whole, and each side's WOSA_PARTS at two decimals."""
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
             *([round(float(r[column]), 2) for column in wosa[field]] for field in ("wo", "wd", "wt"))]
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
                        percentile_min: dict | None = None, rookie_ids: dict | None = None) -> None:
    """decompose_openrapm.json: meta and the rows of every season, latest first; compact, UTF-8 letters as they are.
    `opens_on` is the season the ratings page opens on (`default_season`), the latest when not given;
    `percentile_min`, the seasons in progress' percentile pool minimums (`percentile_minimums`), none when not given;
    `rookie_ids`, each season's rookies (data/panel.py `rookies`), kept to the players the season's rows show, none
    when not given."""
    seasons = sorted(rows, reverse=True)
    minimums = {str(season): int(value) for season, value in (percentile_min or {}).items()}
    shown = {season: {row[OPENRAPM_FIELDS.index("id")] for row in rows[season]} for season in seasons}
    rookie_lists = {str(season): [player for player in (rookie_ids or {}).get(season, []) if player in shown[season]]
                    for season in seasons}
    document = dict(meta=dict(built=built, penalty=penalty, seasons=seasons,
                              default_season=int(seasons[0] if opens_on is None else opens_on),
                              percentile_min=minimums, rookies=rookie_lists, fields=OPENRAPM_FIELDS,
                              pieces=OPENRAPM_PIECES, wosa_parts=WOSA_PARTS),
                    rows={str(season): rows[season] for season in seasons})
    path.write_text(json.dumps(document, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


def write_openrapm_decomposition(cfg: dict, ratings_path: Path, data_dir: Path, built: str) -> None:
    """decompose_openrapm.json in `data_dir`, after its checks pass.  Solves step 3 again for every season and both
    sides, so it must run after `openrapm build`, whenever the ratings table changes."""
    table, step3, factors = published_parts(ratings_path, cfg)
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
        rows[season] = openrapm_rows(season_rows, pieces, named, box_files.season_teams(season))
        top = rows[season][0]
        print(f"  {season}: {len(rows[season])} players, top {top[0]} {top[5][-1]:+.2f} "
              f"({time.time() - started:.0f}s)", flush=True)
    run_openrapm_checks(worst, rows, table)
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / OPENRAPM_FILE
    write_openrapm_file(path, rows, built, penalty, default_season(table), percentile_minimums(cfg), rookies(cfg))
    print(f"wrote {path}: {sum(len(v) for v in rows.values()):,} rows over {len(rows)} seasons, "
          f"{path.stat().st_size / 1e6:.2f} MB ({time.time() - started:.0f}s)", flush=True)


# ===================================================================================== the similarity page
# Which other player-seasons a player-season is most like: the nearest player-seasons of other players, by the
# distance between their standardised inputs (similarity.py says how).  The owner's design, 2026-10-12.
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

    The inputs are standardised on the pool, so the pool holds only seasons a fit may train on (`trainable_panel`): a
    season in progress waits for its Finals.  Stops when the panel's possessions differ from the ratings table's (one
    of them rebuilt since the other) or when the opening player-season is not in the pool.
    """
    table = pd.read_parquet(ratings_path)
    panel = trainable_panel(cfg)
    seasons = sorted(set(panel.season.astype(int)))
    waiting = sorted(set(table.season.astype(int)) - set(seasons))
    if waiting:
        print(f"  left out until their Finals are over: {waiting}", flush=True)
    rows = _named(player_seasons(table, SIMILARITY_MIN_POSSESSIONS, seasons), ratings_path, cfg)
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
# over its last 10 games, summed over the team (the owner's design, 2026-10-28).  The ratings are centred on every
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
# The Power tab is hidden until every team has played 10 games (meta.ready; site.js shows the tab), the owner,
# 2026-10-28: "hide it until there are 10 games in the system".
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


SHARES_TOTAL = 5.0     # five players on the floor: a team's shares of its possessions add up to 500%


def power_numbers(share, offense, defense, total) -> list:
    """A player row's numbers: his share as a percent at one decimal, his three ratings at two."""
    return [round(100 * float(share), 1), *(round(float(value), 2) for value in (offense, defense, total))]


def power_document(table: pd.DataFrame, games: pd.DataFrame, season: int, built: str) -> dict:
    """power.json's document for `season`: the teams, best total first, each with its players by share."""
    ratings = table[table.season == season].set_index("player_id")
    shares = team_shares(games, POWER_GAMES)
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
    teams = []
    for tm, rows in shares.groupby("tm"):
        rows = rows.sort_values("share", ascending=False)
        sums = [float((rows.share * rows[column]).sum()) for column in columns]
        shown, others = rows[rows.poss_season >= MIN_POSSESSIONS_SHOWN], rows[rows.poss_season < MIN_POSSESSIONS_SHOWN]
        players = [[r.player_name if isinstance(r.player_name, str) else "", int(r.player_id), *power_numbers(
                    r.share, r.rating_off, r.rating_def, r.rating_total)] for r in shown.itertuples(index=False)]
        if len(others):
            share = others.share.sum()
            players.append(["Other players", None, *power_numbers(
                share, *((others.share * others[column]).sum() / share for column in columns))])
        teams.append(dict(tm=tm, name=rows.name.iloc[0], g=int(rows.g.iloc[0]),
                          **{side: round(value, 2) for side, value in zip("odt", sums)}, players=players))
    teams.sort(key=lambda team: -team["t"])
    meta = dict(season=season, through=pd.Timestamp(games.game_date.max()).strftime("%Y-%m-%d"), games=POWER_GAMES,
                recent_games=RECENT_GAMES, recent_weight=RECENT_WEIGHT, blowout_weight=BLOWOUT_WEIGHT,
                ready=min(team["g"] for team in teams) >= POWER_GAMES, built=built,
                average_team={side: round(value, 2) for side, value in average.items()})
    return dict(meta=meta, teams=teams)


def write_power(cfg: dict, ratings_path: Path, data_dir: Path, built: str) -> None:
    """power.json in `data_dir`, for the ratings table's latest season."""
    table = pd.read_parquet(ratings_path)
    season = int(table.season.max())
    games = player_game_possessions(season_stints(season, cfg), game_teams(season, cfg))
    document = power_document(table, games, season, built)
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / POWER_FILE
    path.write_text(json.dumps(document, separators=(",", ":")), encoding="utf-8")
    top = document["teams"][0]
    print(f"wrote {path}: {len(document['teams'])} teams of {season} through {document['meta']['through']}, top "
          f"{top['tm']} {top['t']:+.2f}", flush=True)


# ===================================================================================== the bias page
# docs/data/bias.json, which docs/bias.html draws (the owner's design, 2026-11-05): for the season the pages open on,
# how much higher or lower OpenRAPM rates players one standard deviation higher in each stat than a consensus of five
# outside systems does.  bias.py holds the equations; the file:
#
#   {"meta": {"season", "through", "built", "min_possessions", "players_shown", "players",
#             "multipliers": {"o", "d"}, "fields": BIAS_FIELDS},
#    "systems": [{"key", "name", "by", "link", "weight", "as_of", "players"}, ...],    the five, heaviest first
#    "rows": [[stat, label, o, d, t, o_se, d_se, t_se, sd], ...]}                      one per stat, bias.STATS order
#
# o and d are written at two decimals and t is their sum, so every row adds up on the page.  players_shown is the
# season's players with 60 or more offensive possessions (`MIN_POSSESSIONS_SHOWN`), and players those of them all
# five systems list.  A system's own `players` is how many of the shown it lists; its `as_of` is its own date, or the
# day it was pulled when its page prints none.  Only these slopes are published: the systems' ratings stay in
# paths.public_ratings, out of git.
BIAS_FILE = "bias.json"
BIAS_FIELDS = ["stat", "label", "o", "d", "t", "o_se", "d_se", "t_se", "sd"]
BIAS_LABELS = {ON_COURT_PLUS_MINUS: "On-court plus-minus", OFF_COURT_PLUS_MINUS: "Off-court plus-minus"}


def bias_label(stat: str) -> str:
    """A row's name on the page: the catalogue's words for a box rate (`features.meaning`, "Made threes"), and the
    two plus-minus rows by name."""
    return BIAS_LABELS.get(stat) or meaning(stat).capitalize()


def _rounded(value, digits: int = 2):
    """A number as the file holds it; None where there is none (a stat with no spread has no slope)."""
    return round(float(value), digits) if np.isfinite(value) else None


def bias_row(row) -> list:
    """One stat's row in `BIAS_FIELDS` order: offense and defense at two decimals, total their sum."""
    offense, defense = _rounded(row.offense), _rounded(row.defense)
    total = None if offense is None or defense is None else round(offense + defense, 2)
    return [row.stat, bias_label(row.stat), offense, defense, total, _rounded(row.offense_se),
            _rounded(row.defense_se), _rounded(row.total_se), _rounded(row.sd, 3)]


def against_openrapm(players: pd.DataFrame, pulls: dict) -> list:
    """The systems whose ratings run against OpenRAPM's on a side among the compared players, as "LOKI defense":
    every system agrees with OpenRAPM far above 0 (0.5 to 0.8 on 2026-11-05), so one at 0 or below changed a sign."""
    out = []
    for key in CONSENSUS_WEIGHTS:
        theirs = players[["player_id"]].merge(pulls[key].table, on="player_id")
        out += [f"{public_ratings.SYSTEMS[key].name} {side}" for side in SIDES
                if not np.corrcoef(players[OURS[side]], theirs[side])[0, 1] > 0]
    return out


def bias_document(table: pd.DataFrame, panel: pd.DataFrame, pulls: dict, season: int, built: str,
                  through: str) -> dict:
    """bias.json's document for `season`, from the ratings table, the season's panel rows and each system's stored
    pull ({key: `public_ratings.PublicRatings`})."""
    shown = table[(table.season == season) & (table.poss_season >= MIN_POSSESSIONS_SHOWN)]
    blend = consensus({key: pulls[key].table for key in CONSENSUS_WEIGHTS})
    players = compared_players(shown, blend, player_stats(panel))
    flipped = against_openrapm(players, pulls)
    if flipped:
        raise SystemExit(f"{', '.join(flipped)} runs against OpenRAPM's among {len(players)} players: its page "
                         f"changed a sign (data/public_ratings.py); the bias file is left as it was")
    slopes, multipliers = bias_table(players)
    systems = [dict(key=key, name=public_ratings.SYSTEMS[key].name, by=public_ratings.SYSTEMS[key].by,
                    link=public_ratings.SYSTEMS[key].link, weight=weight,
                    as_of=pulls[key].as_of or pulls[key].pulled,
                    players=int(shown.player_id.isin(pulls[key].table.player_id).sum()))
               for key, weight in CONSENSUS_WEIGHTS.items()]
    meta = dict(season=season, through=through, built=built, min_possessions=MIN_POSSESSIONS_SHOWN,
                players_shown=len(shown), players=len(players),
                multipliers=dict(o=_rounded(multipliers["offense"], 3), d=_rounded(multipliers["defense"], 3)),
                fields=BIAS_FIELDS)
    return dict(meta=meta, systems=systems, rows=[bias_row(row) for row in slopes.itertuples(index=False)])


def bias_summary(document: dict) -> list[str]:
    """The console report: the players compared, the multipliers, each system's date, and the table."""
    meta = document["meta"]
    lines = [f"  {meta['players']} of the {meta['players_shown']} players shown for {meta['season']} are in all "
             f"{len(document['systems'])} systems; OpenRAPM multiplied by {meta['multipliers']['o']:.3f} on offense "
             f"and {meta['multipliers']['d']:.3f} on defense",
             "  " + ", ".join(f"{system['name']} {system['weight']:.0%} as of {system['as_of']}"
                              for system in document["systems"])]
    lines += [f"  {row[1]:<22}" + "".join("     n/a" if value is None else f"{value:+8.2f}" for value in row[2:5])
              + f"   standard error of the total {row[7]}" for row in document["rows"]]
    return lines


def write_bias(cfg: dict, ratings_path: Path, data_dir: Path, built: str) -> None:
    """bias.json in `data_dir`, for the season the pages open on (`default_season`), from each system's newest stored
    pull.  A system never pulled for that season, or dated before the season's first game, stops the command with
    one line and the file as it was; so does a system that runs against OpenRAPM (`bias_document`)."""
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
    document = bias_document(table, panel_of_season(cfg, season), pulls, season, built, through)
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / BIAS_FILE
    part = path.with_name(BIAS_FILE + ".part")      # written whole, then moved: the page never reads half a file
    part.write_text(json.dumps(document, separators=(",", ":")), encoding="utf-8")
    part.replace(path)
    print(f"wrote {path}: {season} through {through}", flush=True)
    print("\n".join(bias_summary(document)), flush=True)
