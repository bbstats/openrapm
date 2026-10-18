"""`openrapm site`: the data behind the published site, docs/.

    openrapm site                    docs/data/ratings.json, the ratings page's table
    openrapm site --bias             also docs/bias.html, the bias by player type, with its two tables
                                     outputs/bias_groups.json and outputs/bgmm_proba.parquet
    openrapm site --decomposition    also docs/data/decompose_1y.json and decompose_3y.json, the vanilla RAPM
                                     decomposition page
    openrapm site --openrapm-decomposition
                                     also docs/data/decompose_openrapm.json, the OpenRAPM decomposition page: every
                                     published rating split into pieces that add up to it
    openrapm site --similarity       also docs/data/similarity/, the similarity page: each player-season's nearest
                                     player-seasons of other players

The pages' wording and look are not this command's to change: it writes data, and the bias page's text is
copied unchanged.

Words used below
    ratings table   outputs/season_ratings_product.parquet, one row per player per season (`openrapm build`)
    built           the date in a file's meta.built.  ratings.json shows the ratings table's modification date,
                    so re-exporting the same table does not tell readers the ratings changed; the decomposition
                    files show the day they were computed.  --built sets it for every file written
    consensus       data/external/consensus.csv: a GLS-weighted mean of DRIP, LEBRON, EPM, DARKO DPM, Time-Decay
                    RAPM and Time-Decay-Luck-Adjusted RAPM (the 2026 snapshot)
    vanilla RAPM    plain ridge regression with no box-score prior: raw points, regular season and playoffs

Example: `openrapm site --out-dir C:/tmp/site --built 2026-10-09` writes C:/tmp/site/data/ratings.json, whose
meta.built reads 2026-10-09 whatever the ratings table's date.
"""
from __future__ import annotations

import json
import re
import time
import unicodedata
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.linalg as sla
import scipy.sparse as sp

from ..data.box_scores import season_box
from ..data.config import resolve, resolve_path
from ..data.designs import DesignCache, block_design
from ..data.ingest import raw_dir
from ..data.panel import trainable_panel
from ..decomposition import lineups, refit, row_groups, row_keys, side_team, split_games_part, team_codes
from ..features import FEATURE_SETS
from ..player_types import RATES, bias_by_type, cluster_proba, describe_types, fit_mixture, per36
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
             decomposition: bool = False, bias_tables=None, similarity: bool = False,
             openrapm_decomposition: bool = False) -> int:
    """`openrapm site`: ratings.json always; the bias page, the decomposition files, the similarity files and the
    OpenRAPM decomposition file when asked.

    `ratings` defaults to the ratings table under config paths.outputs, `out_dir` to the repository's docs/, and
    `bias_tables` (where --bias writes bias_groups.json and bgmm_proba.parquet) to config paths.outputs.
    """
    ratings_path = Path(ratings) if ratings else resolve(cfg, "outputs") / RATINGS_TABLE
    if not ratings_path.exists():
        raise SystemExit(f"{ratings_path} does not exist; `openrapm build` writes it")
    out_dir = Path(out_dir) if out_dir else resolve_path(cfg, SITE_FOLDER)
    write_ratings_json(ratings_path, out_dir / "data" / "ratings.json", built or table_date(ratings_path))
    if bias:
        tables_dir = Path(bias_tables) if bias_tables else resolve(cfg, "outputs")
        write_bias_page(cfg, ratings_path, out_dir / "bias.html", tables_dir)
    if decomposition:
        write_decomposition(cfg, ratings_path, out_dir / "data", built or today())
    if similarity:
        write_similarity(cfg, ratings_path, out_dir / "data" / SIMILARITY_FOLDER, built or today())
    if openrapm_decomposition:
        write_openrapm_decomposition(cfg, ratings_path, out_dir / "data", built or table_date(ratings_path))
    return 0


# ===================================================================================== ratings.json
# The ratings table's columns and their names in ratings.json: one letter each, because the page downloads the
# whole table.  The NBA id is in the file and the CSV download, not on the page.
RATINGS_FIELDS = {"season": "s", "player_name": "n", "rating_off": "o", "rating_def": "d", "rating_total": "t",
                  "poss_season": "p", "player_id": "id"}


def ratings_rows(table: pd.DataFrame) -> list[dict]:
    """One row per player-season with possessions, ratings at two decimals, by season and then best total first.

    Example: {"s": 2026, "n": "Victor Wembanyama", "o": 2.16, "d": 5.3, "t": 7.45, "p": 5398, "id": 1641705}.
    A missing name is written "".
    """
    missing = [column for column in RATINGS_FIELDS if column not in table.columns]
    if missing:
        raise SystemExit(f"the ratings table lacks {missing}; `openrapm build` writes them all")
    short = table[list(RATINGS_FIELDS)].rename(columns=RATINGS_FIELDS)
    short["n"] = short["n"].fillna("").astype(str)
    short = short[short.p > 0].sort_values(["s", "t"], ascending=[True, False])
    return [dict(s=int(r.s), n=r.n, o=round(float(r.o), 2), d=round(float(r.d), 2), t=round(float(r.t), 2),
                 p=int(r.p), id=int(r.id))
            for r in short.itertuples(index=False)]


def ratings_document(table: pd.DataFrame, built: str) -> dict:
    """{"meta": the seasons, the built date and the number of players, "rows": `ratings_rows`}."""
    rows = ratings_rows(table)
    meta = dict(seasons=sorted({row["s"] for row in rows}), built=built,
                n_players=int(table.player_id.nunique()))
    return dict(meta=meta, rows=rows)


def write_ratings_json(ratings_path: Path, path: Path, built: str) -> None:
    """ratings.json: compact JSON (no spaces), non-ASCII letters escaped, no newline at the end."""
    document = ratings_document(pd.read_parquet(ratings_path), built)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, separators=(",", ":")), encoding="utf-8")
    seasons = document["meta"]["seasons"]
    print(f"wrote {path}: {len(document['rows'])} rows over {len(seasons)} seasons ({seasons[0]}-{seasons[-1]}), "
          f"{path.stat().st_size / 1e6:.1f} MB, from {ratings_path}", flush=True)


# ===================================================================================== the bias page
# Where the published ratings sit above or below two other sources, by player type, on the OVERALL rating.
#
# The types come from `player_types`: a Bayesian Gaussian mixture fitted on the 2023-2025 regular seasons' per-36
# profiles (one row per player-season with at least 1,000 possessions), then used to place each 2026 player.
# Offence and defence are not reported: a type whose two sides are large and opposite is a disagreement about which
# end a player's value comes from, and only the total says a rating is wrong.  Possession-weighted, because a
# 1,000-possession player and a 5,000-possession player are not equally part of how wrong the ratings are.
#
# Two columns, both "our rating minus the source", points per 100, positive = the ratings rate the type higher:
#   vs consensus      our overall rating, after one global rescale to the consensus's standard deviation (never a
#                     rescale per type, which would define the answer away), minus the consensus
#   vs 2026 observed  minus the per-player coefficient of a ridge fitted to what the games left over after our
#                     rating predicted every 2026 team-game, with one free scale per side and one level per team
#                     (outputs/tradeset_actual_b0_alpha.parquet; kept as it is until the owner decides on it)
BIAS_SEASONS = (2026,)
FIT_SEASONS = (2023, 2024, 2025)
BIAS_MIN_POSSESSIONS = 1000
MIXTURE_COMPONENTS = 8
MIXTURE_SEED = 0
MIN_COMPONENT_WEIGHT = 0.01       # a component with less weight than this counts as unused
CONSENSUS_FILE = "data/external/consensus.csv"
RESIDUAL_TABLE = "tradeset_actual_b0_alpha.parquet"
SUFFIX = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b")

# The hand-chosen names, keyed by the signature the fit produces for a group: the four rates the group is
# furthest from the league on, strongest first.  Four and not two because the two biggest scoring groups --
# Jokic's and Brunson's -- are both `+ftm +pts` and separate only on the fourth rate, rebounds against
# assists.  A signature is stable in a way a cluster number is not, and a group whose signature is not
# listed prints as UNNAMED rather than borrowing a neighbour's name.
NAMES = {
    "+ftm +pts +tov +ast": "High-usage scorers who draw fouls",
    "+ftm +blk +pts +drb": "Two-way big men who score",
    "+orb -fg3_miss +blk -fg3m": "Rim-protecting centers",
    "+fg3m +fg3_miss -drb -blk": "Wings who shoot threes",
    "+ast -orb +tov -drb": "Guards who handle the ball",
    "+stl +drb +pf +orb": "Busy all-round forwards",
    "-ftm -tov -pts -ast": "Low-usage wings who stay out of the way",
    "+stl -pts -ftm -fg3m": "Ball-hawk defenders who rarely shoot",
}
# The plain words for each rate a signature names.
RATE_WORDS = {"pts": "points", "fg3m": "threes made", "fg3_miss": "threes missed", "ftm": "free throws",
              "ast": "assists", "orb": "offensive rebounds", "drb": "defensive rebounds", "stl": "steals",
              "blk": "blocks", "tov": "turnovers", "pf": "fouls"}

# The only prose on the page: what the two columns are measured against.  Both are OpenRAPM minus that
# source, so positive means OpenRAPM rates the type higher.
BLURB = """<div class="blurb">
<p><b>vs 2026 observed.</b> We fit a RAPM model against the <em>residuals</em> of OpenRAPM's
predictions. These are the average results for each player type (possession-weighted).</p>
<p><b>vs consensus.</b> The consensus is a GLS-weighted mean of DRIP, LEBRON, EPM, DARKO DPM,
Time-Decay RAPM, and Time-Decay-Luck-Adjusted RAPM.</p>
</div>"""

# The download under the table: the table itself as CSV, read off the page so it can never disagree with it.
# UTF-8 with a byte-order mark, so a spreadsheet opens it as written.
DOWNLOAD = '<p class="dl"><a href="#" id="csv">Download CSV</a></p>'
CSV_SCRIPT = """<script>
document.getElementById("csv").addEventListener("click", e => {
  e.preventDefault();
  const q = v => /[",\\r\\n]/.test(v) ? '"' + v.replace(/"/g, '""') + '"' : v;
  const slug = s => s.trim().toLowerCase().replace(/[^a-z0-9]+/g, "_");
  const head = [...document.querySelectorAll("thead th")].map(th => slug(th.textContent));
  const rows = [[head[0], "signature", ...head.slice(1)]];
  document.querySelectorAll("tbody tr").forEach(tr => {
    const td = [...tr.children], sig = td[0].querySelector(".sig");
    rows.push([td[0].firstChild.textContent.trim(), sig ? sig.textContent.trim() : "",
               ...td.slice(1).map(c => c.textContent.trim().replace("\\u2013", ""))]);
  });
  const text = "\\uFEFF" + rows.map(r => r.map(q).join(",")).join("\\r\\n") + "\\r\\n";
  const url = URL.createObjectURL(new Blob([text], { type: "text/csv;charset=utf-8" }));
  const a = Object.assign(document.createElement("a"), { href: url, download: "openrapm_bias_by_player_type.csv" });
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});
</script>"""

PAGE_HEAD = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>OpenRAPM: bias by player type</title>
<link rel="stylesheet" href="site.css">
<script src="site.js"></script>
<style>
  :root { --line: #d9d8d3; --muted: #6b6a66; --ink: #1a1a19; --accent: #2a78d6;
          --over: 227 73 72; --under: 42 120 214; --zero: #f0efec; }
  :root[data-theme="dark"] { --line: #3a3a38; --muted: #a09f98; --ink: #f2f1ee; --accent: #3987e5;
          --over: 230 103 103; --under: 57 135 229; --zero: #383835; }
  body { color: var(--ink); font: 14px/1.4 -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; }
  h1 { font-size: 21px; margin: 0 0 10px; }
  table { border-collapse: separate; border-spacing: 2px 0; width: 100%;
          font-variant-numeric: tabular-nums; }
  th, td { padding: 4px 7px; border-bottom: 1px solid var(--line); text-align: right; white-space: nowrap; }
  th:nth-child(1), td:nth-child(1) { text-align: left; white-space: normal; }
  th { color: var(--muted); font-weight: 600; }
  th:nth-child(2), th:nth-child(3) { color: var(--ink); }
  td.over, td.under { background: var(--zero); border-radius: 4px; }
  td.over { background: rgb(var(--over) / calc(var(--depth) * 0.55)); }
  td.under { background: rgb(var(--under) / calc(var(--depth) * 0.55)); }
  .blurb { margin-top: 22px; color: var(--muted); max-width: 66ch; font-size: 13.5px; }
  .blurb p { margin: 0 0 10px; } .blurb b { color: var(--ink); }
  .sig { color: var(--muted); font-size: 12px; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
  .dl { margin: 10px 0 0; color: var(--muted); font-size: 13px; } .dl a { color: inherit; }
</style>
</head>
<body>
<main>
<nav class="site">
  <a href="./">Ratings</a>
  <a href="bias.html" aria-current="page">Bias</a>
  <a href="similarity.html">Similarity</a>
  """
# the dark-mode button, one line on the page
THEME_BUTTON = ('<button class="theme" type="button" aria-label="Dark mode" title="Dark mode">'
                '<svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true">'
                '<circle cx="8" cy="8" r="6.25" fill="none" stroke="currentColor" stroke-width="1.5"/>'
                '<path d="M8 1.75a6.25 6.25 0 0 1 0 12.5z" fill="currentColor"/></svg></button>')
PAGE_TITLE = """
</nav>
<h1>OpenRAPM: bias by player type</h1>
"""


def name_key(name) -> str:
    """A name as the consensus join matches it: accents dropped, lower case, letters and spaces, no suffix.

    Example: "Jabari Smith Jr." gives "jabari smith".
    """
    text = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    return SUFFIX.sub("", re.sub(r"[^a-z ]", "", text)).strip()


def consensus_overall(consensus: pd.DataFrame) -> pd.DataFrame:
    """key, adj_overall, var: one row per consensus name (the first of a repeated name)."""
    named = consensus.dropna(subset=["player_name", "adj_overall"]).assign(key=lambda d: d.player_name.map(name_key))
    return named.drop_duplicates("key")[["key", "adj_overall", "var"]]


def pooled_ratings(table: pd.DataFrame, seasons) -> pd.DataFrame:
    """One row per named player over `seasons`: each rating averaged with offensive-possession weights, his
    possessions (`poss`) and the name of his latest season."""
    rows = table[table.season.isin(seasons) & table.player_name.notna()].copy()
    weights = rows.poss_off.to_numpy(float)
    pooled = pd.DataFrame({column: (rows[column] * weights).groupby(rows.player_id).sum()
                           / rows.groupby("player_id").poss_off.sum()
                           for column in ("rating_off", "rating_def", "rating_total")})
    pooled["poss"] = rows.groupby("player_id").poss_off.sum()
    pooled["player_name"] = rows.sort_values("season").drop_duplicates("player_id", keep="last") \
                                .set_index("player_id").player_name
    return pooled.reset_index()


def bias_players(table: pd.DataFrame, consensus: pd.DataFrame, profile: pd.DataFrame, seasons) -> pd.DataFrame:
    """The players the page covers: at least BIAS_MIN_POSSESSIONS over `seasons`, matched to the consensus by name
    (the player with more possessions keeps a shared name), with a per-36 profile."""
    pooled = pooled_ratings(table, seasons)
    pooled = pooled[pooled.poss >= BIAS_MIN_POSSESSIONS].assign(key=lambda x: x.player_name.map(name_key))
    pooled = pooled.sort_values("poss").drop_duplicates("key", keep="last") \
                   .merge(consensus_overall(consensus), on="key", how="inner")
    return pooled.merge(profile[["player_id", "minutes", *RATES]], on="player_id", how="inner") \
                 .reset_index(drop=True)


def training_profiles(table: pd.DataFrame, cfg: dict, seasons) -> pd.DataFrame:
    """One row per player-season over `seasons`: his per-36 rates in that regular season alone, for players with
    at least BIAS_MIN_POSSESSIONS that season.

    One row per player-season and not one per player, because the page places one season's rates, and a training
    row should be the same kind of row as the row the mixture is asked about.
    """
    possessions = table[["player_id", "season", "poss_off"]]
    rows = []
    for season in seasons:
        box = season_box([season], ["RS"], cfg)
        profile = per36(box[box.phase == "RS"]).assign(season=season)
        played = possessions[(possessions.season == season) & (possessions.poss_off >= BIAS_MIN_POSSESSIONS)]
        rows.append(profile.merge(played[["player_id"]], on="player_id", how="inner"))
    return pd.concat(rows, ignore_index=True)


def bias_inputs(cfg: dict, ratings_path: Path) -> dict:
    """Everything the bias page reads: the ratings table, the consensus, the 2026 regular-season profiles, the
    training profiles and the residual table."""
    table = pd.read_parquet(ratings_path)
    box = season_box(list(BIAS_SEASONS), ["RS"], cfg)
    return dict(table=table, consensus=pd.read_csv(resolve_path(cfg, CONSENSUS_FILE)),
                profile=per36(box[box.phase == "RS"]), training=training_profiles(table, cfg, FIT_SEASONS),
                residuals=pd.read_parquet(resolve(cfg, "outputs") / RESIDUAL_TABLE))


def games_residual(residuals: pd.DataFrame, seasons) -> pd.DataFrame:
    """player_id, ours_scaled, residual: our rating at the scale the games ask for, and what the games say it
    missed, both sides added and pooled over `seasons` with offensive-possession weights.

    `residuals` is outputs/tradeset_actual_b0_alpha.parquet: per player, season and side, our rating (positive
    good), its fitted scale and the ridge coefficient on what the games left over (`alpha_good`).
    """
    eligible = residuals[residuals.season.isin(seasons) & residuals.eligible].copy()
    eligible["scaled"] = eligible.rating_scale * eligible.rating
    wide = eligible.pivot_table(index=["player_id", "season"], columns="side",
                                values=["scaled", "alpha_good", "poss_off"]).reset_index()
    wide.columns = [column[0] if not column[1] else f"{column[0]}_{column[1][:3]}" for column in wide.columns]
    wide = wide.dropna(subset=["scaled_off", "scaled_def", "alpha_good_off", "alpha_good_def"])
    wide["ours_scaled"] = wide.scaled_off + wide.scaled_def
    wide["residual"] = wide.alpha_good_off + wide.alpha_good_def
    weights = wide.poss_off_off.to_numpy(float)
    pooled = pd.DataFrame({column: (wide[column] * weights).groupby(wide.player_id).sum()
                           / wide.groupby("player_id").poss_off_off.sum()
                           for column in ("ours_scaled", "residual")})
    return pooled.reset_index()


def with_types(players: pd.DataFrame, model, mean: np.ndarray, sd: np.ndarray) -> pd.DataFrame:
    """Each player's probability of every type (p0, p1, ...), his type (`group`) and its probability."""
    probability = cluster_proba(model, mean, sd, players[RATES].to_numpy(float))
    columns = {f"p{i}": probability[:, i] for i in range(probability.shape[1])}
    return players.assign(**columns, group=probability.argmax(axis=1), group_proba=probability.max(axis=1))


def stretch_to_consensus(players: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Our overall rating stretched once, league-wide, to the consensus's standard deviation, and the bias.

        ours_vs_consensus = mean(consensus) + (ours - mean(ours)) x sd(consensus) / sd(ours)
        bias_vs_consensus = ours_vs_consensus - consensus
    """
    stretch = float(players.adj_overall.std(ddof=1) / players.rating_total.std(ddof=1))
    ours = players.adj_overall.mean() + (players.rating_total - players.rating_total.mean()) * stretch
    return players.assign(ours_vs_consensus=ours, bias_vs_consensus=ours - players.adj_overall), stretch


def with_games(players: pd.DataFrame, residual: pd.DataFrame) -> pd.DataFrame:
    """`games` = our scaled rating + what the games say it missed; `bias_vs_games` = minus that miss."""
    players = players.merge(residual, on="player_id", how="left")
    players["games"] = players.ours_scaled + players.residual
    players["bias_vs_games"] = -players.residual
    return players


def type_probabilities(players: pd.DataFrame, n_types: int, season: int) -> pd.DataFrame:
    """outputs/bgmm_proba.parquet: each player's type, its probability and the probability of every type."""
    table = players[["player_id", "player_name", "poss", "group", "group_proba",
                     *[f"p{i}" for i in range(n_types)]]].copy()
    table.insert(1, "season", season)
    return table


def bias_result(inputs: dict, ratings_name: str) -> tuple[dict, pd.DataFrame]:
    """outputs/bias_groups.json's content, and the type probabilities, from `bias_inputs`."""
    seasons = list(BIAS_SEASONS)
    players = bias_players(inputs["table"], inputs["consensus"], inputs["profile"], seasons)
    training = inputs["training"]
    model, mean, sd = fit_mixture(training[RATES].to_numpy(float), k=MIXTURE_COMPONENTS, seed=MIXTURE_SEED)
    players = with_types(players, model, mean, sd)
    # the types are described against the training seasons' league, the standardisation the fit used
    z = pd.DataFrame({rate: (players[rate] - mu) / s for rate, mu, s in zip(RATES, mean, sd)})
    players, stretch = stretch_to_consensus(players)
    described = describe_types(players, z, NAMES)
    players = with_games(players, games_residual(inputs["residuals"], seasons))
    both = players.dropna(subset=["adj_overall", "games"])
    rows_consensus = bias_by_type(players, described, "ours_vs_consensus", "adj_overall", "bias_vs_consensus")
    rows_games = bias_by_type(players, described, "ours_scaled", "games", "bias_vs_games")
    paired = {row["group"]: row["bias"] for row in rows_games}
    shared = [(row["bias"], paired[row["group"]]) for row in rows_consensus if row["group"] in paired]
    result = {"board": ratings_name, "seasons": seasons, "fit_seasons": list(FIT_SEASONS),
              "k": MIXTURE_COMPONENTS, "seed": MIXTURE_SEED,
              "r2_consensus_observed": float(np.corrcoef(both.adj_overall, both.games)[0, 1] ** 2),
              "r_types": float(np.corrcoef([a for a, _ in shared], [b for _, b in shared])[0, 1]),
              "train_rows": int(len(training)), "mean_top_proba": float(players.group_proba.mean()),
              "players": len(players), "components_used": int((model.weights_ > MIN_COMPONENT_WEIGHT).sum()),
              "global_stretch": stretch, "residual_players": int(players.residual.notna().sum()),
              "groups": rows_consensus, "groups_vs_games": rows_games}
    return result, type_probabilities(players, len(model.weights_), seasons[-1])


def tint_cell(value) -> str:
    """A diverging tint: red = OpenRAPM rates the type ABOVE that source, blue = below, grey at zero.

    The depth is `min(|bias| / 0.9, 1)`, so the largest bias on the page is the full tint and the
    owner's 0.1 threshold for "nothing" is a tenth of it -- barely visible, which is the point.
    """
    if value is None:
        return '<td>&ndash;</td>'
    depth = min(abs(value) / 0.9, 1.0)
    pole = "over" if value > 0 else "under"
    return (f'<td class="{pole}" style="--depth: {depth:.3f}"><b>{value:+.3f}</b></td>')


def bias_page(result: dict) -> str:
    """docs/bias.html: one row per type, in the consensus table's order, with both biases."""
    by_group = {row["group"]: row for row in result["groups_vs_games"]}
    out = ['<table><thead><tr><th>Player type</th><th>vs consensus</th>'
           '<th>vs 2026 observed</th></tr></thead><tbody>']
    for row in result["groups"]:
        other = by_group.get(row["group"])
        out.append(f"<tr><td>{row['name']}<br><span class=\"sig\">{row['signature']}</span></td>"
                   f"{tint_cell(row['bias'])}{tint_cell(other['bias'] if other else None)}</tr>")
    return (PAGE_HEAD + THEME_BUTTON + PAGE_TITLE + "".join(out) + "</tbody></table>" + DOWNLOAD + BLURB + "</main>"
            + CSV_SCRIPT + "</body></html>")


def bias_summary(result: dict) -> list[str]:
    """The console report: coverage, any unnamed type, and both tables."""
    lines = [f"  {len(result['groups'])} groups over {result['players']} players, {result['components_used']} of "
             f"{result['k']} components used, OpenRAPM's overall rating stretched "
             f"x{result['global_stretch']:.3f} to the consensus's spread; the residual RAPM covers "
             f"{result['residual_players']} of them"]
    unnamed = [row["signature"] for row in result["groups"] if row["name"] == "UNNAMED"]
    if unnamed:
        lines.append(f"  WARNING: no name for {unnamed} -- the fit moved and `NAMES` needs the new signature")
    for key, label in (("groups", "against the consensus"), ("groups_vs_games", "against the games")):
        lines.append(f"\n=== {label}, ordered by |bias| x possessions")
        lines.extend(f"  {row['name']:<30} bias {row['bias']:+.3f}  OpenRAPM {row['ours']:+.2f}  "
                     f"theirs {row['theirs']:+.2f}  {row['players']:>3} players, "
                     f"{row['possessions'] / 1000:>5.0f}k possessions  {row['signature']}" for row in result[key])
    return lines


def write_bias_page(cfg: dict, ratings_path: Path, html_path: Path, tables_dir: Path) -> None:
    """docs/bias.html, and bias_groups.json and bgmm_proba.parquet in `tables_dir`."""
    result, probabilities = bias_result(bias_inputs(cfg, ratings_path), ratings_path.name)
    tables_dir.mkdir(parents=True, exist_ok=True)
    probabilities.to_parquet(tables_dir / "bgmm_proba.parquet", index=False)
    (tables_dir / "bias_groups.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(bias_page(result), encoding="utf-8")
    print(f"wrote {html_path}, and bias_groups.json and bgmm_proba.parquet in {tables_dir}")
    print("\n".join(bias_summary(result)), flush=True)


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
#     rating = box score + on court + off court + team SOS + swap adjustment
#
#   box score        what the box score says: step 2's prior after step 4's multiplier (prior_off, prior_def)
#   on court         the part of the season's games' part (u, how far step 3 moved him from his prior, times step 4's
#                    factor on it since experiment 65) that comes from the possessions he was on the court for
#   off court        the part from his team's possessions without him: in games he played and in games he missed
#   team SOS         the part from every other possession
#   swap adjustment  step 5's adjustment inside his team (the table's c_off, c_def)
#
# The total is the sum of the two sides' pieces.  Box score and swap adjustment are columns of the ratings table; u is
# split by solving step 3 again for each season and each side's target on the table's own priors (decomposition.py
# says why that splits u exactly), and the run stops before writing unless the refit reproduces every u and every
# prior scale in the table (to 1e-9; to 1e-6 in 2020, whose solve needs a jitter that follows the scale of the
# prior columns).  The one file, docs/data/decompose_openrapm.json:
#     {"meta": {built, penalty, seasons, fields, pieces}, "rows": {season: [[name, NBA id, possessions, offence,
#      defence, total], ...]}}     each side's list holds OPENRAPM_PIECES in order, at two decimals
# Rows are the ratings.json rows (possessions above 0), best total first.
#
# Example (2026): Victor Wembanyama's offence, 2.07 = 0.69 + 1.41 - 0.27 + 0.06 + 0.18; his total, 7.16 = 3.98 + 2.65
# - 0.21 + 0.14 + 0.61.
OPENRAPM_FILE = "decompose_openrapm.json"
OPENRAPM_FIELDS = ["n", "id", "p", "o", "d", "t"]
OPENRAPM_PIECES = ["box_score", "on_court", "off_court", "team_sos", "swap", "rating"]
OPENRAPM_SIDES = ("off", "def", "total")
OPENRAPM_TOLERANCE = 1e-9         # each identity holds to this, per player and side
OPENRAPM_JITTER_TOLERANCE = 1e-6  # the same, in a season whose solve needed rapm.solved_ridge's jitter (2020)
OPENRAPM_TABLE_COLUMNS = ["player_id", "season", "player_name", "poss_season", "prior_off", "prior_def", "u_off",
                          "u_def", "c_off", "c_def", "rating_off", "rating_def", "rating_total", "offense_lambda"]
OPENRAPM_STEP3_COLUMNS = ["player_id", "season", "prior_offense", "prior_defense", "prior_scale_off",
                          "prior_scale_def", "u_off", "u_def"]


def step3_path(ratings_path: Path) -> Path:
    """The step-3 table `openrapm build` writes beside the ratings table."""
    return ratings_path.with_name(f"{ratings_path.stem}_step3.parquet")


GAMES_FACTOR_TOLERANCE = 1e-12   # the table's games' part against its step-3 table's times one factor


def published_parts(ratings_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """(the ratings table, its step-3 table, {season: {"off", "def"}: step 4's factor on the games' part}), after
    checking they are one build's: the same player-seasons, the same games' part up to one factor per season and side
    (step 4 multiplies it since experiment 65, 1 before; step 5 leaves it alone), and the penalty the shipped settings
    name."""
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
    check(miss < GAMES_FACTOR_TOLERANCE, f"the ratings table and its step-3 table have the same games' part, up to "
                                         f"step 4's factor per season and side (max miss {miss:.1e})")
    check(set(table.offense_lambda.unique()) == {float(PRODUCT.ridge.penalty)},
          f"the ratings table was built at the shipped penalty, {PRODUCT.ridge.penalty:,.0f}")
    return table, step3, factors


def season_games_parts(designs: DesignCache, box_files: BoxScoreFiles, season: int, step3: pd.DataFrame,
                       penalty: float) -> tuple[dict, dict]:
    """One season's games' part split on both sides: ({"off": frame, "def": frame}, {"off": misses, "def": misses}).

    The offence half comes from the fit on the offence target, the defence half from the fit on the defence target,
    as `prior_rapm.rate_season` makes the table.  The priors are the step-3 table's own divided by its prior scale: a
    fit's deviations do not change with the prior's scale or with a constant added to it, because the scale is a free
    coefficient and the season intercept absorbs the constant.  `misses` adds the largest gap between the refit's
    prior scale and the table's, and the share of rows the box scores cannot place.
    """
    rows = step3[step3.season == season]
    prior_offense = dict(zip(rows.player_id, rows.prior_offense / rows.prior_scale_off))
    prior_defense = dict(zip(rows.player_id, rows.prior_defense / rows.prior_scale_def))
    parts, misses = {}, {}
    for side, target, scale_column, scale_index in (("off", PRODUCT.targets.offence, "prior_scale_off", -2),
                                                    ("def", PRODUCT.targets.defence, "prior_scale_def", -1)):
        design = designs.season(season, target)
        fit = refit(design, prior_offense, prior_defense, penalty)
        absent = set(fit.player_ids) - set(rows.player_id)
        if absent:
            raise SystemExit(f"{season}: {len(absent)} players of the design are not in the step-3 table")
        lineup_o, lineup_d = lineups(fit.players, fit.n_players)
        keys = row_keys(design, lineup_o, lineup_d, box_files.teams(season))
        parts[side], misses[side] = split_games_part(fit, keys, side)
        misses[side].update(unplaced=keys["unplaced"],
                            scale=abs(float(fit.theta[scale_index]) - float(rows[scale_column].iloc[0])))
    return parts, misses


def season_pieces(rows: pd.DataFrame, parts: dict, factors: dict | None = None) -> pd.DataFrame:
    """One season's pieces: a row per ratings-table row, and per side (off, def, total) a column for each of
    OPENRAPM_PIECES, named `<side>_<piece>`.  Positive is good on both sides; a total is the two sides added.  The
    games' part's pieces are the refit's times step 4's factor on it (`factors`, {"off", "def"}; 1 when None)."""
    pieces = pd.DataFrame({"player_id": rows.player_id.to_numpy()})
    for side in ("off", "def"):
        games = parts[side].set_index("player_id").reindex(rows.player_id)
        if games.u.isna().any():
            raise SystemExit(f"{int(rows.season.iloc[0])}: the refit has no row for some players of the ratings table")
        pieces[f"{side}_box_score"] = rows[f"prior_{side}"].to_numpy()
        factor = 1.0 if factors is None else factors[side]
        for piece in ("on_court", "off_court", "team_sos"):
            pieces[f"{side}_{piece}"] = factor * games[piece].to_numpy()
        pieces[f"{side}_swap"] = rows[f"c_{side}"].to_numpy()
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
        refit_u = (1.0 if factors is None else factors[side]) * parts[side].set_index("player_id").u.reindex(
            rows.player_id).to_numpy()
        season["games_part"] = max(season["games_part"], float(np.abs(refit_u - rows[f"u_{side}"].to_numpy()).max()))
        season["scale"] = max(season["scale"], misses[side]["scale"])
        season["sum_hy"] = max(season["sum_hy"], misses[side]["sum_hy"])
        season["leftover"] = max(season["leftover"], misses[side]["leftover"])
        worst["unplaced"] = max(worst["unplaced"], misses[side]["unplaced"])


def openrapm_rows(rows: pd.DataFrame, pieces: pd.DataFrame, names: np.ndarray) -> list:
    """One season's rows in OPENRAPM_FIELDS order, best total first, players with no possessions left out (as
    ratings.json leaves them): name, NBA id, possessions (whole, as ratings.json writes them), then each side's
    pieces at two decimals."""
    frame = pieces.assign(n=names, p=rows.poss_season.to_numpy())
    frame = frame[frame.p > 0].sort_values("total_rating", ascending=False)
    return [[r["n"], int(r["player_id"]), int(r["p"]),
             *([round(float(r[f"{side}_{piece}"]), 2) for piece in OPENRAPM_PIECES] for side in OPENRAPM_SIDES)]
            for r in frame.to_dict("records")]


def run_openrapm_checks(worst: dict, rows: dict, table: pd.DataFrame) -> None:
    """The checks on every season, before the file is written."""
    plain, jittered = worst["plain"], worst["jittered"]

    def holds(key: str) -> bool:
        return plain[key] < OPENRAPM_TOLERANCE and jittered[key] < OPENRAPM_JITTER_TOLERANCE

    def misses(key: str) -> str:
        return f"max miss {plain[key]:.1e}; in a jittered season {jittered[key]:.1e}"

    check(holds("identity"), f"box score + on court + off court + team SOS + swap adjustment add up to every rating, "
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
    published = {(row["s"], row["id"]): (row["o"], row["d"], row["t"], row["p"]) for row in ratings_rows(table)}
    mine = {(season, row[1]): (row[3][-1], row[4][-1], row[5][-1], row[2])
            for season, season_rows in rows.items() for row in season_rows}
    check(published == mine, f"the file holds the {len(published):,} player-seasons of ratings.json with its "
                             "ratings and possessions")


def write_openrapm_file(path: Path, rows: dict, built: str, penalty: float) -> None:
    """decompose_openrapm.json: meta and the rows of every season, latest first; compact, UTF-8 letters as they are."""
    seasons = sorted(rows, reverse=True)
    document = dict(meta=dict(built=built, penalty=penalty, seasons=seasons, fields=OPENRAPM_FIELDS,
                              pieces=OPENRAPM_PIECES),
                    rows={str(season): rows[season] for season in seasons})
    path.write_text(json.dumps(document, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


def write_openrapm_decomposition(cfg: dict, ratings_path: Path, data_dir: Path, built: str) -> None:
    """decompose_openrapm.json in `data_dir`, after its checks pass.  Solves step 3 again for every season and both
    sides, so it must run after `openrapm build`, whenever the ratings table changes."""
    table, step3, factors = published_parts(ratings_path)
    names, box_files, designs = name_table(ratings_path, cfg), BoxScoreFiles(cfg), DesignCache(cfg)
    penalty = float(PRODUCT.ridge.penalty)
    started, rows, worst = time.time(), {}, new_openrapm_worst()
    for season in sorted(int(s) for s in table.season.unique()):
        season_rows = table[table.season == season].reset_index(drop=True)
        parts, misses = season_games_parts(designs, box_files, season, step3, penalty)
        designs.clear()
        pieces = season_pieces(season_rows, parts, factors[season])
        update_openrapm_worst(worst, season_rows, pieces, parts, misses, factors[season])
        named = span_names(names, season_rows.player_id.to_numpy(), [season], box_files)
        worst["unnamed"] += int(sum(1 for name in named if pd.isna(name) or not str(name).strip()))
        rows[season] = openrapm_rows(season_rows, pieces, named)
        top = rows[season][0]
        print(f"  {season}: {len(rows[season])} players, top {top[0]} {top[5][-1]:+.2f} "
              f"({time.time() - started:.0f}s)", flush=True)
    run_openrapm_checks(worst, rows, table)
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / OPENRAPM_FILE
    write_openrapm_file(path, rows, built, penalty)
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
