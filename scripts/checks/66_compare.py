"""Which players does a change move?  Two or more ratings tables side by side for one season, as an interactive,
self-contained HTML page.

Every experiment ends with the 2026 top 20 (HANDOFF.md), and this is the page it is read on.  The first table named
is the reference: name the incumbent first.

Reads
    each NAME=PATH table: player_id, player_name, season, rating_off, rating_def, rating_total, poss_off
    `--yoy`: a scripts/checks/63_yoy.py parquet, for the summary block (optional)
Writes
    outputs/compare_<names>_<season>.html (or `--out`): one page, no network, opened in the browser unless
    `--open=0`
    the same name with `_phone` added: the same rows as a static table with inline styles, for the body of an
    email (mail clients strip scripts)

How to read the page
    summary      per table, when `--yoy` is given: the year-over-year error per team-game (points per 100), what
                 the next season wants each side multiplied by, and the paired comparison with `--ref`: the mean
                 difference in team-game mean squared error (below zero = better), its paired t-statistic z, and
                 the seasons won out of 56
    main table   every player in any table's top N for the season (a tick box shows every player): possessions,
                 then rank, offence, defence and total per table, and the change in rank against the reference
                 (+3 = three places higher than in the reference).  Highlighted rows are in one table's top N but
                 not another's.  Click a header to sort, type to filter by name, hover a rank change for the ranks
Example: a player ranked 12th by the incumbent and 9th by the candidate shows a change of +3, in green.

    python scripts/checks/66_compare.py incumbent=outputs/season_ratings_weightfix.parquet \\
        new=outputs/season_ratings_new.parquet --season=2026 --top=20 --yoy=outputs/yoy_new.parquet --ref=incumbent
"""
import argparse
import html
import json
import subprocess

import numpy as np
import pandas as pd

from openrapm.commands.console import use_utf8
from openrapm.data.config import load_config, resolve, resolve_path
from openrapm.scoring import paired, pooled

COLUMNS = ("player_id", "player_name", "rating_off", "rating_def", "rating_total", "poss_off")
PAGE_COLUMNS = ("player_id", "player_name", "poss", "in_top", "marked")
DO_NOT_OPEN = ("0", "no", "false")
CELL = "padding:4px 6px;border-bottom:1px solid #e5e7eb;text-align:right;white-space:nowrap;font-size:13px"
HEAD_CELL = "padding:4px 6px;background:#f3f4f6;border-bottom:2px solid #d1d5db;font-size:12px;text-align:right"
UP, DOWN, FLAT = "#0f766e", "#b91c1c", "#6b7280"


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="An HTML comparison of two or more tables of player rankings.",
                                     allow_abbrev=False)
    parser.add_argument("tables", nargs="*", metavar="NAME=PATH", help="the tables, the reference first")
    parser.add_argument("--season", type=int, default=2026, help="default %(default)s")
    parser.add_argument("--top", type=int, default=20, help="default %(default)s")
    parser.add_argument("--yoy", help="a scripts/checks/63_yoy.py parquet for the summary block")
    parser.add_argument("--ref", help="the year-over-year reference (default the first table)")
    parser.add_argument("--alias", default="", help="page_name:yoy_name,... when the two names differ")
    parser.add_argument("--title", help="the page title")
    parser.add_argument("--out", help="the page (default outputs/compare_<names>_<season>.html)")
    parser.add_argument("--open", default="1", help="0: do not open the page (default %(default)s)")
    return parser.parse_args(argv)


# ------------------------------------------------------------------------------------------------ the tables
def load(path, season: int) -> pd.DataFrame:
    """One season of a ratings table, best total first, with its `rank` (1 = best)."""
    table = pd.read_parquet(path)
    table = table[table.season == season].copy()
    missing = [c for c in COLUMNS if c not in table.columns]
    if missing:
        raise SystemExit(f"{path}: missing {missing}")
    table = table.sort_values("rating_total", ascending=False).reset_index(drop=True)
    table["rank"] = np.arange(1, len(table) + 1)
    return table[[*COLUMNS, "rank"]]


def read_tables(specs: list, season: int, cfg: dict) -> dict:
    """{name: one season of the table} from the NAME=PATH arguments, in the order given."""
    tables = {}
    for spec in specs:
        name, path = spec.split("=", 1)
        tables[name] = load(resolve_path(cfg, path), season)
    return tables


def merge_tables(tables: dict, top: int) -> pd.DataFrame:
    """Every player of any table, one row each: rank_, off_, def_, tot_ and poss_ per table name, the most
    possessions any table gives him (`poss`), whether he is in any top `top` (`in_top`) and in some but not all
    (`marked`), and per later table his rank change against the first (`drank_`, positive = moved up).

    A player missing from the first table has no name: the name is read from the first table alone.
    """
    names = list(tables)
    renamed = [table.rename(columns={"rank": f"rank_{name}", "rating_off": f"off_{name}",
                                     "rating_def": f"def_{name}", "rating_total": f"tot_{name}",
                                     "poss_off": f"poss_{name}"}) for name, table in tables.items()]
    merged = renamed[0]
    for table in renamed[1:]:
        merged = merged.merge(table.drop(columns=["player_name"]), on="player_id", how="outer")
    merged["player_name"] = merged["player_name"].fillna("")
    merged["poss"] = merged[[f"poss_{n}" for n in names]].max(axis=1)
    in_tops = np.column_stack([(merged[f"rank_{n}"] <= top).fillna(False).to_numpy() for n in names])
    merged["in_top"] = in_tops.any(axis=1)
    merged["marked"] = in_tops.any(axis=1) & ~in_tops.all(axis=1)
    for n in names[1:]:
        merged[f"drank_{n}"] = merged[f"rank_{names[0]}"] - merged[f"rank_{n}"]
    return merged


def as_integers(record: dict) -> dict:
    """A page record with its ranks, rank changes and possessions as whole numbers (None stays None)."""
    for key, value in record.items():
        if key.startswith(("rank_", "drank_")) or key == "poss":
            record[key] = None if value is None else int(value)
    return record


def page_records(merged: pd.DataFrame, names: list) -> list[dict]:
    """The rows the page's script draws: one dictionary per player, missing values as None."""
    keep = [*PAGE_COLUMNS, *(c for n in names for c in (f"rank_{n}", f"off_{n}", f"def_{n}", f"tot_{n}")),
            *(f"drank_{n}" for n in names[1:])]
    return [as_integers(record) for record in json.loads(merged[keep].to_json(orient="records"))]


# -------------------------------------------------------------------------------- the year-over-year summary
def with_directions(res: pd.DataFrame) -> pd.DataFrame:
    """A 63_yoy.py result's whole-season rows, both directions as separate observations.

    The direction is read BEFORE the name is stripped, or both directions collapse onto one scored-season key and
    the paired test runs over 28 averaged observations instead of 56.  A system named without a direction counts
    as "prev".
    """
    res = res[res.split == "all"].copy()
    has_direction = res.system.str.contains(":")
    res["direction"] = np.where(has_direction, res.system.str.rsplit(":", n=1).str[-1], "prev")
    res["system"] = res.system.str.rsplit(":", n=1).str[0]
    res["held_out"] = res.held_out + res.direction.map({"prev": 0.0, "next": 0.5}).fillna(0.0)
    return res


def summary(yoy_path, names: list, ref: str) -> list[dict]:
    """Per table: pooled year-over-year numbers and the paired test against `ref`, from a 63_yoy.py parquet."""
    both = with_directions(pd.read_parquet(yoy_path))
    pooled_table = pooled(both).set_index("system")
    paired_table = paired(both, ref, "tg").set_index("system") if ref in set(both.system) else None
    rows = []
    for n in (n for n in names if n in pooled_table.index):
        p = pooled_table.loc[n]
        row = dict(name=n, game_armse=float(p["game_armse"]), scale_off=float(p["scale_off"]),
                   scale_def=float(p["scale_def"]), seasons=int(p["seasons"]))
        if paired_table is not None and n in paired_table.index:
            t = paired_table.loc[n]
            row.update(mean_diff=float(t["mean_diff"]), z=float(t["z"]), wins=int(t["wins"]),
                       n_seasons=int(t["n_seasons"]))
        rows.append(row)
    return rows


def aliased_summary(yoy_path, names: list, ref: str, alias_spec: str) -> list[dict]:
    """`summary` when the page's short names differ from the year-over-year run's: `alias_spec` is
    "page_name:yoy_name,...", and the rows come back under the page's names."""
    alias = dict(part.split(":") for part in alias_spec.split(",") if part)
    rows = summary(yoy_path, [alias.get(n, n) for n in names], alias.get(ref, ref))
    back = {yoy_name: page_name for page_name, yoy_name in alias.items()}
    for row in rows:
        row["name"] = back.get(row["name"], row["name"])
    return rows


# ------------------------------------------------------------------------------------------------- the page
PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>__TITLE__</title>
<style>
  :root { --ink:#1f2933; --muted:#6b7280; --line:#e5e7eb; --up:#0f766e; --down:#b91c1c; --mark:#fef3c7; \
--head:#f3f4f6; }
  body { font: 14px/1.4 system-ui, -apple-system, Segoe UI, Roboto, sans-serif; color: var(--ink); margin: 24px; }
  h1 { font-size: 20px; margin: 0 0 4px; } .sub { color: var(--muted); margin-bottom: 16px; }
  .controls { display:flex; gap:16px; align-items:center; margin: 12px 0; flex-wrap: wrap; }
  input[type=search] { padding: 6px 10px; border: 1px solid var(--line); border-radius: 6px; \
width: 260px; font-size: 14px; }
  table { border-collapse: collapse; width: 100%; } th, td { padding: 6px 8px; border-bottom: 1px solid var(--line); \
text-align: right; white-space: nowrap; }
  th { background: var(--head); cursor: pointer; position: sticky; top: 0; user-select: none; }
  th.l, td.l { text-align: left; } th .arrow { color: var(--muted); font-size: 11px; margin-left: 4px; }
  tr.marked td { background: var(--mark); }
  .up { color: var(--up); font-weight: 600; } .down { color: var(--down); font-weight: 600; } \
.flat { color: var(--muted); }
  .group { border-left: 2px solid var(--line); } .summary { margin-bottom: 18px; } .summary table { width: auto; }
  .legend { color: var(--muted); font-size: 12px; margin-top: 8px; }
</style></head><body>
<h1>__TITLE__</h1>
<div class="sub">__SUBTITLE__</div>
<div class="summary" id="summary"></div>
<div class="controls">
  <input type="search" id="q" placeholder="filter by player name">
  <label><input type="checkbox" id="all"> show every player, not just the top __TOP__</label>
  <span class="legend" id="count"></span>
</div>
<table id="t"><thead></thead><tbody></tbody></table>
<div class="legend">Rank change is against the first table (<b>__REF__</b>): green moved up, red moved down.
Highlighted rows are in one table's top __TOP__ but not another's.  Click a header to sort; click again to reverse.
Ratings in points per 100 possessions, positive good on both ends.</div>
<script>
const DATA = __DATA__;
const NAMES = __NAMES__;
const TOP = __TOP__;
const SUMMARY = __SUMMARY__;
const fmt = (x, d=2) => (x === null || x === undefined || Number.isNaN(x)) ? "" : Number(x).toFixed(d);
function renderSummary() {
  if (!SUMMARY.length) return;
  let h = '<table><thead><tr><th class="l">rankings</th><th>year-over-year error per team-game</th>\
<th>next season wants offence &times;</th><th>defence &times;</th><th>vs reference, team-game MSE</th>\
<th>z</th><th>seasons better</th></tr></thead><tbody>';
  for (const r of SUMMARY) {
    h += `<tr><td class="l">${r.name}</td><td>${fmt(r.game_armse,3)}</td><td>${fmt(r.scale_off)}</td>\
<td>${fmt(r.scale_def)}</td>` +
         `<td>${r.mean_diff === undefined ? "reference" : fmt(r.mean_diff)}</td>\
<td>${r.z === undefined ? "" : fmt(r.z,1)}</td>` +
         `<td>${r.wins === undefined ? "" : r.wins + " of " + r.n_seasons}</td></tr>`;
  }
  document.getElementById("summary").innerHTML = h + "</tbody></table>";
}
let sortKey = "rank_" + NAMES[0], sortDir = 1;
function columns() {
  const cols = [{k:"player_name", label:"player", cls:"l"}, {k:"poss", label:"poss"}];
  NAMES.forEach((n, i) => {
    cols.push({k:"rank_"+n, label:n + " rank", cls:"group"});
    if (i > 0) cols.push({k:"drank_"+n, label:"change", delta:true});
    cols.push({k:"off_"+n, label:"off"}, {k:"def_"+n, label:"def"}, {k:"tot_"+n, label:"total"});
  });
  return cols;
}
function render() {
  const q = document.getElementById("q").value.trim().toLowerCase();
  const all = document.getElementById("all").checked;
  const cols = columns();
  let rows = DATA.filter(r => (all || r.in_top) && (!q || r.player_name.toLowerCase().includes(q)));
  rows.sort((a, b) => {
    const x = a[sortKey], y = b[sortKey];
    if (x === null || x === undefined) return 1; if (y === null || y === undefined) return -1;
    return (typeof x === "string" ? x.localeCompare(y) : x - y) * sortDir;
  });
  document.querySelector("#t thead").innerHTML = "<tr>" + cols.map(c =>
    `<th class="${c.cls||""}" data-k="${c.k}">${c.label}\
${c.k===sortKey ? '<span class="arrow">'+(sortDir>0?'&#9650;':'&#9660;')+'</span>' : ''}</th>`).join("") + "</tr>";
  document.querySelector("#t tbody").innerHTML = rows.map(r => "<tr" + (r.marked ? ' class="marked"' : "") + \
">" + cols.map(c => {
    const v = r[c.k];
    if (c.delta) {
      if (v === null || v === undefined) return '<td class="flat"></td>';
      const cls = v > 0 ? "up" : v < 0 ? "down" : "flat";
      const n = c.k.slice(6);
      return `<td class="${cls}" title="${r['rank_'+NAMES[0]]} in ${NAMES[0]} &rarr; ${r['rank_'+n]} in ${n}">\
${v > 0 ? "+" + v : v}</td>`;
    }
    if (c.k === "player_name") return `<td class="l">${v}</td>`;
    if (c.k === "poss" || c.k.startsWith("rank_")) return \
`<td class="${c.cls||""}">${v === null || v === undefined ? "" : v}</td>`;
    return `<td>${fmt(v)}</td>`;
  }).join("") + "</tr>").join("");
  document.getElementById("count").textContent = rows.length + " players shown";
  document.querySelectorAll("#t th").forEach(th => th.onclick = () => {
    const k = th.dataset.k; if (sortKey === k) sortDir = -sortDir; \
else { sortKey = k; sortDir = k === "player_name" ? 1 : (k.startsWith("rank_") ? 1 : -1); }
    render();
  });
}
document.getElementById("q").oninput = render; document.getElementById("all").onchange = render;
renderSummary(); render();
</script></body></html>
"""


def interactive_page(records: list, names: list, top: int, title: str, subtitle: str, summ: list) -> str:
    """The interactive page: PAGE with its placeholders filled."""
    return (PAGE.replace("__TITLE__", html.escape(title)).replace("__SUBTITLE__", html.escape(subtitle))
            .replace("__DATA__", json.dumps(records)).replace("__NAMES__", json.dumps(names))
            .replace("__TOP__", str(top)).replace("__REF__", html.escape(names[0]))
            .replace("__SUMMARY__", json.dumps(summ)))


# ------------------------------------------------------------------------------------ the phone version
def phone_header(names: list) -> str:
    """The phone table's header cells: rank and player, then per table its ratings (and, after the first, its rank
    and rank change)."""
    head = f'<th style="{HEAD_CELL}">rank</th><th style="{HEAD_CELL};text-align:left">player</th>'
    for n in names:
        head += f'<th style="{HEAD_CELL}">{html.escape(n)} off / def / total</th>'
        if n != names[0]:
            head += f'<th style="{HEAD_CELL}">rank</th><th style="{HEAD_CELL}">change</th>'
    return head


def ratings_cell(row: dict, name: str) -> str:
    """One table's "off / def / total" cell, empty when the table has no row for the player."""
    offence, defence, total = row[f"off_{name}"], row[f"def_{name}"], row[f"tot_{name}"]
    if pd.isna(total):
        return f'<td style="{CELL}"></td>'
    return f'<td style="{CELL}">{offence:.1f} / {defence:.1f} / <b>{total:.1f}</b></td>'


def rank_change_cells(row: dict, name: str) -> str:
    """A later table's rank and its change against the first: green up, red down, grey level or absent."""
    rank, change = row[f"rank_{name}"], row[f"drank_{name}"]
    colour = UP if pd.notna(change) and change > 0 else DOWN if pd.notna(change) and change < 0 else FLAT
    text = "" if pd.isna(change) else (f"+{int(change)}" if change > 0 else str(int(change)))
    return (f'<td style="{CELL}">{int(rank) if pd.notna(rank) else ""}</td>'
            f'<td style="{CELL};color:{colour};font-weight:600">{text}</td>')


def phone_row(row: dict, names: list) -> str:
    """One player's row of the phone table, highlighted when he is in some tops and not others."""
    ref_rank = row[f"rank_{names[0]}"]
    mark = ' style="background:#fef3c7"' if row["marked"] else ""
    cells = (f'<td style="{CELL}">{int(ref_rank) if pd.notna(ref_rank) else ""}</td>'
             f'<td style="{CELL};text-align:left">{html.escape(str(row["player_name"]))}</td>')
    for n in names:
        cells += ratings_cell(row, n)
        if n != names[0]:
            cells += rank_change_cells(row, n)
    return f"<tr{mark}>{cells}</tr>"


def summary_line(row: dict, ref: str) -> str:
    """One table's year-over-year summary as a sentence."""
    extra = ("reference" if "mean_diff" not in row
             else f"{row['mean_diff']:+.2f} team-game MSE vs {html.escape(ref)}, z {row['z']:.1f}, "
                  f"{row['wins']} of {row['n_seasons']} seasons better")
    return (f"{html.escape(row['name'])}: year-over-year error {row['game_armse']:.3f} per 100; "
            f"next season wants offence x{row['scale_off']:.2f}, defence x{row['scale_def']:.2f}; {extra}")


def phone_body(merged: pd.DataFrame, names: list, season: int, top: int, title: str, summ: list) -> str:
    """The union of the top N as one static HTML table with inline styles: the body of an email."""
    ref = names[0]
    shown = merged[merged.in_top].sort_values(f"rank_{ref}")
    body = "".join(phone_row(row, names) for row in shown.to_dict(orient="records"))
    summary_html = "".join(f'<p style="margin:0 0 6px;color:#6b7280;font-size:13px">{summary_line(row, ref)}</p>'
                           for row in summ)
    return (f'<div style="font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;color:#1f2933">'
            f'<h2 style="font-size:18px;margin:0 0 6px">{html.escape(title)}</h2>{summary_html}'
            f'<table style="border-collapse:collapse;width:100%"><thead><tr>{phone_header(names)}</tr></thead>'
            f'<tbody>{body}</tbody></table>'
            f'<p style="color:#6b7280;font-size:12px">Every player in any top {top} for {season}, ordered by '
            f'{html.escape(ref)}.  Change is rank places moved against {html.escape(ref)}: green up, red down.  '
            f'Highlighted rows are in one top {top} but not another.  Ratings in points per 100 possessions, '
            f'positive good on both ends.</p></div>')


# ------------------------------------------------------------------------------------------------- the run
def main() -> None:
    use_utf8()
    args = parse_args()
    cfg = load_config()
    specs = [a for a in args.tables if "=" in a]
    if not specs:
        raise SystemExit(__doc__)
    tables = read_tables(specs, args.season, cfg)
    names = list(tables)
    merged = merge_tables(tables, args.top)
    summ = []
    if args.yoy:
        summ = aliased_summary(resolve_path(cfg, args.yoy), names, names[0] if args.ref is None else args.ref,
                               args.alias)
    title = (f"Experiment comparison, {args.season}: " + " vs ".join(names)) if args.title is None else args.title
    subtitle = (f"{len(merged)} players in {args.season}; the union of the top {args.top} shown by default.  "
                + (f"Year-over-year from {args.yoy}." if args.yoy else ""))
    out = (resolve(cfg, "outputs") / f"compare_{'_'.join(names)}_{args.season}.html" if args.out is None
           else resolve_path(cfg, args.out))
    out.write_text(interactive_page(page_records(merged, names), names, args.top, title, subtitle, summ),
                   encoding="utf-8")
    print(f"wrote {out}")
    phone = out.with_name(out.stem + "_phone.html")
    phone.write_text(phone_body(merged, names, args.season, args.top, title, summ), encoding="utf-8")
    print(f"wrote {phone} (email body)")
    if args.open not in DO_NOT_OPEN:
        subprocess.run(["cmd", "/c", "start", "", str(out)], check=False)


if __name__ == "__main__":
    main()
