"""`openrapm site`: the data behind the published site, docs/.

    openrapm site                    docs/data/ratings.json, the ratings page's table
    openrapm site --bias             also docs/bias.html, the bias by player type, with its two tables
                                     outputs/bias_groups.json and outputs/bgmm_proba.parquet
    openrapm site --decomposition    also docs/data/decompose_1y.json and decompose_3y.json, the decomposition page

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
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.linalg as sla
import scipy.sparse as sp

from ..data.box_scores import season_box
from ..data.config import resolve, resolve_path
from ..data.designs import block_design
from ..data.ingest import raw_dir
from ..player_types import RATES, bias_by_type, cluster_proba, describe_types, fit_mixture, per36
from ..rapm import SeasonSums
from ..settings import PRODUCT

RATINGS_TABLE = "season_ratings_product.parquet"
SITE_FOLDER = "docs"


def table_date(path) -> str:
    """A file's modification date in UTC, written 2026-10-09."""
    return pd.Timestamp(Path(path).stat().st_mtime, unit="s", tz="UTC").strftime("%Y-%m-%d")


def today() -> str:
    """Today's date in UTC, written 2026-10-09."""
    return pd.Timestamp.now("UTC").strftime("%Y-%m-%d")


def run_site(cfg: dict, ratings=None, out_dir=None, built: str | None = None, bias: bool = False,
             decomposition: bool = False, bias_tables=None) -> int:
    """`openrapm site`: ratings.json always; the bias page and the decomposition files when asked.

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
  body { color: var(--ink); font: 15px/1.45 -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; }
  h1 { font-size: 26px; margin: 0 0 14px; }
  table { border-collapse: separate; border-spacing: 2px 0; width: 100%;
          font-variant-numeric: tabular-nums; }
  th, td { padding: 6px 9px; border-bottom: 1px solid var(--line); text-align: right; white-space: nowrap; }
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
  <a href="decompose.html">Decomposition</a>
  <a href="bias.html" aria-current="page">Bias</a>
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
TEAM_CODE = 10 ** 10           # season (or game) x TEAM_CODE + team id: one number per team-season or team-game


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


def side_team(teams: pd.Series, game_ids: np.ndarray, players: np.ndarray) -> np.ndarray:
    """Each row's team on one side: the team of that side's first player the box score lists (-1 if none is)."""
    got = np.full(len(game_ids), np.nan)
    for k in range(players.shape[1]):
        todo = np.isnan(got)
        if not todo.any():
            break
        index = pd.MultiIndex.from_arrays([game_ids[todo], players[todo, k]])
        got[todo] = teams.reindex(index).to_numpy(dtype=float)
    return np.where(np.isnan(got), -1, got).astype(np.int64)


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


def team_codes(rows: dict) -> dict:
    """Each row's attacking and defending team as team-season and team-game codes, and the share of rows the box
    scores cannot place.  A side the box scores cannot name gets a code of its own, so it is nobody's team."""
    team_att, team_def, season = rows["team_att"], rows["team_def"], rows["season"]
    game = pd.factorize(rows["game_ids"])[0].astype(np.int64)
    n_rows = len(team_att)
    unknown_att, unknown_def = -1 - np.arange(n_rows), -1 - n_rows - np.arange(n_rows)
    team_season = np.concatenate([np.where(team_att >= 0, season * TEAM_CODE + team_att, unknown_att),
                                  np.where(team_def >= 0, season * TEAM_CODE + team_def, unknown_def)])
    team_game = np.concatenate([np.where(team_att >= 0, game * TEAM_CODE + team_att, unknown_att),
                                np.where(team_def >= 0, game * TEAM_CODE + team_def, unknown_def)])
    season_code, game_code = pd.factorize(team_season)[0], pd.factorize(team_game)[0]
    return dict(ts_att=season_code[:n_rows], ts_def=season_code[n_rows:], gt_att=game_code[:n_rows],
                gt_def=game_code[n_rows:], n_ts=int(season_code.max()) + 1, n_gt=int(game_code.max()) + 1,
                unplaced=float(((team_att < 0) | (team_def < 0)).mean()))


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


def his_teams(rows_off, rows_def, keys: dict) -> tuple:
    """Per row: is the attacking (defending) side one of his team-seasons, and one of his team-games."""
    his_season, his_game = np.zeros(keys["n_ts"], dtype=bool), np.zeros(keys["n_gt"], dtype=bool)
    his_season[keys["ts_att"][rows_off]], his_season[keys["ts_def"][rows_def]] = True, True
    his_game[keys["gt_att"][rows_off]], his_game[keys["gt_def"][rows_def]] = True, True
    return (his_season[keys["ts_att"]], his_season[keys["ts_def"]],
            his_game[keys["gt_att"]], his_game[keys["gt_def"]])


def possession_groups(i: int, rows_off, rows_def, fit: dict, terms: dict) -> dict:
    """The four pieces by possession, and his actual off-court rating (his team's net points per 100 without
    him, in games he played; missing when one side has no such rows)."""
    n, w, y = fit["n"], fit["w"], fit["y"]
    hz = w * (fit["X"] @ (terms["u0"] + terms["inverse"][:, i] - terms["inverse"][:, n + i])) * terms["z"]
    on = np.zeros(len(w), dtype=bool)
    on[rows_off], on[rows_def] = True, True
    team_att, team_def, played_att, played_def = his_teams(rows_off, rows_def, fit["keys"])
    without = ((team_att & played_att) | (team_def & played_def)) & ~on
    missed = ((team_att & ~played_att) | (team_def & ~played_def)) & ~on & ~without
    groups = dict(his_minutes=hz[on].sum(), team_without_him=hz[without].sum(), games_missed=hz[missed].sum(),
                  everything_else=hz[~(on | without | missed)].sum())
    off_att, off_def = team_att & played_att & ~on, team_def & played_def & ~on
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
