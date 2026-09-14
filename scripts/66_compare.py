"""An interactive, self-contained HTML comparison of two or more tables of player rankings.

    python scripts/66_compare.py name=outputs/a.parquet name2=outputs/b.parquet [--season=2026] [--top=20]
                                 [--yoy=outputs/yoy_x.parquet --ref=name] [--alias=name:yoy_name,...]
                                 [--out=outputs/compare_<tag>.html] [--title=...] [--open=1]

One page, no network, opens in the browser.  What is on it:

  * a summary block, when `--yoy=` is given: per table the year-over-year error per team-game
    (`game_armse`, points per 100), what the scored season wants each side multiplied by, and the paired
    comparison against `--ref` (mean difference in team-game MSE, its z, seasons won of 56);
  * the main table: every player in any table's top N for the season (toggle to every player), with
    possessions, then rank / offence / defence / total per table, and the rank change against the FIRST
    table named.  Click a header to sort, type to filter by name, hover a rank change for the numbers.
    Players in one table's top N but not another's are marked.

The first table named is the reference; name the incumbent first.
"""
import html
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

ROOT = Path(__file__).resolve().parents[1]


def _flag(name, default=None):
    hit = [a for a in sys.argv[1:] if a.startswith(f"--{name}=")]
    return hit[0].split("=", 1)[1] if hit else default


def load(path: Path, season: int) -> pd.DataFrame:
    t = pd.read_parquet(path)
    t = t[t.season == season].copy()
    need = ["player_id", "player_name", "rating_off", "rating_def", "rating_total", "poss_off"]
    missing = [c for c in need if c not in t.columns]
    if missing:
        raise SystemExit(f"{path}: missing {missing}")
    t = t.sort_values("rating_total", ascending=False).reset_index(drop=True)
    t["rank"] = np.arange(1, len(t) + 1)
    return t[need + ["rank"]]


def summary(yoy_path: Path, names: list, ref: str) -> list:
    """Per table: pooled year-over-year numbers and the paired test against `ref`, from a 63_yoy.py parquet."""
    from eracoef.holdout import paired, pooled
    res = pd.read_parquet(yoy_path)
    res = res[res.split == "all"].copy()
    res["system"] = res.system.str.rsplit(":", n=1).str[0]
    res["direction"] = res.system.str.rsplit(":", n=1).str[1] if res.system.str.contains(":").any() else "prev"
    both = res.copy()
    if "direction" in both:
        both["held_out"] = both.held_out + both.direction.map({"prev": 0.0, "next": 0.5}).fillna(0.0)
    P = pooled(both).set_index("system")
    T = paired(both, ref, "tg").set_index("system") if ref in set(both.system) else None
    rows = []
    for n in names:
        if n not in P.index:
            continue
        r = dict(name=n, game_armse=float(P.loc[n, "game_armse"]), scale_off=float(P.loc[n, "scale_off"]),
                 scale_def=float(P.loc[n, "scale_def"]), seasons=int(P.loc[n, "seasons"]))
        if T is not None and n in T.index:
            r.update(mean_diff=float(T.loc[n, "mean_diff"]), z=float(T.loc[n, "z"]),
                     wins=int(T.loc[n, "wins"]), n_seasons=int(T.loc[n, "n_seasons"]))
        rows.append(r)
    return rows


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>__TITLE__</title>
<style>
  :root { --ink:#1f2933; --muted:#6b7280; --line:#e5e7eb; --up:#0f766e; --down:#b91c1c; --mark:#fef3c7; --head:#f3f4f6; }
  body { font: 14px/1.4 system-ui, -apple-system, Segoe UI, Roboto, sans-serif; color: var(--ink); margin: 24px; }
  h1 { font-size: 20px; margin: 0 0 4px; } .sub { color: var(--muted); margin-bottom: 16px; }
  .controls { display:flex; gap:16px; align-items:center; margin: 12px 0; flex-wrap: wrap; }
  input[type=search] { padding: 6px 10px; border: 1px solid var(--line); border-radius: 6px; width: 260px; font-size: 14px; }
  table { border-collapse: collapse; width: 100%; } th, td { padding: 6px 8px; border-bottom: 1px solid var(--line); text-align: right; white-space: nowrap; }
  th { background: var(--head); cursor: pointer; position: sticky; top: 0; user-select: none; }
  th.l, td.l { text-align: left; } th .arrow { color: var(--muted); font-size: 11px; margin-left: 4px; }
  tr.marked td { background: var(--mark); }
  .up { color: var(--up); font-weight: 600; } .down { color: var(--down); font-weight: 600; } .flat { color: var(--muted); }
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
  let h = '<table><thead><tr><th class="l">rankings</th><th>year-over-year error per team-game</th><th>next season wants offence &times;</th><th>defence &times;</th><th>vs reference, team-game MSE</th><th>z</th><th>seasons better</th></tr></thead><tbody>';
  for (const r of SUMMARY) {
    h += `<tr><td class="l">${r.name}</td><td>${fmt(r.game_armse,3)}</td><td>${fmt(r.scale_off)}</td><td>${fmt(r.scale_def)}</td>` +
         `<td>${r.mean_diff === undefined ? "reference" : fmt(r.mean_diff)}</td><td>${r.z === undefined ? "" : fmt(r.z,1)}</td>` +
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
    `<th class="${c.cls||""}" data-k="${c.k}">${c.label}${c.k===sortKey ? '<span class="arrow">'+(sortDir>0?'&#9650;':'&#9660;')+'</span>' : ''}</th>`).join("") + "</tr>";
  document.querySelector("#t tbody").innerHTML = rows.map(r => "<tr" + (r.marked ? ' class="marked"' : "") + ">" + cols.map(c => {
    const v = r[c.k];
    if (c.delta) {
      if (v === null || v === undefined) return '<td class="flat"></td>';
      const cls = v > 0 ? "up" : v < 0 ? "down" : "flat";
      const n = c.k.slice(6);
      return `<td class="${cls}" title="${r['rank_'+NAMES[0]]} in ${NAMES[0]} &rarr; ${r['rank_'+n]} in ${n}">${v > 0 ? "+" + v : v}</td>`;
    }
    if (c.k === "player_name") return `<td class="l">${v}</td>`;
    if (c.k === "poss" || c.k.startsWith("rank_")) return `<td class="${c.cls||""}">${v === null || v === undefined ? "" : v}</td>`;
    return `<td>${fmt(v)}</td>`;
  }).join("") + "</tr>").join("");
  document.getElementById("count").textContent = rows.length + " players shown";
  document.querySelectorAll("#t th").forEach(th => th.onclick = () => {
    const k = th.dataset.k; if (sortKey === k) sortDir = -sortDir; else { sortKey = k; sortDir = k === "player_name" ? 1 : (k.startsWith("rank_") ? 1 : -1); }
    render();
  });
}
document.getElementById("q").oninput = render; document.getElementById("all").onchange = render;
renderSummary(); render();
</script></body></html>
"""


def phone_body(merged: pd.DataFrame, names: list, season: int, top: int, title: str, summ: list) -> str:
    """The union of the top N as one static HTML table with inline styles: the body of an email."""
    ref = names[0]
    td = "padding:4px 6px;border-bottom:1px solid #e5e7eb;text-align:right;white-space:nowrap;font-size:13px"
    th = "padding:4px 6px;background:#f3f4f6;border-bottom:2px solid #d1d5db;font-size:12px;text-align:right"
    head = f'<th style="{th}">rank</th><th style="{th};text-align:left">player</th>'
    for n in names:
        head += f'<th style="{th}">{html.escape(n)} off / def / total</th>'
        if n != ref:
            head += f'<th style="{th}">rank</th><th style="{th}">change</th>'
    rows = merged[merged.in_top].sort_values(f"rank_{ref}")
    body = ""
    for r in rows.itertuples():
        d = r._asdict()
        mark = ' style="background:#fef3c7"' if d["marked"] else ""
        cells = f'<td style="{td}">{int(d[f"rank_{ref}"]) if pd.notna(d[f"rank_{ref}"]) else ""}</td>' \
                f'<td style="{td};text-align:left">{html.escape(str(d["player_name"]))}</td>'
        for n in names:
            o, de, t = d[f"off_{n}"], d[f"def_{n}"], d[f"tot_{n}"]
            cells += (f'<td style="{td}">{o:.1f} / {de:.1f} / <b>{t:.1f}</b></td>' if pd.notna(t)
                      else f'<td style="{td}"></td>')
            if n != ref:
                rk, ch = d[f"rank_{n}"], d[f"drank_{n}"]
                colour = "#0f766e" if pd.notna(ch) and ch > 0 else "#b91c1c" if pd.notna(ch) and ch < 0 else "#6b7280"
                chs = "" if pd.isna(ch) else (f"+{int(ch)}" if ch > 0 else str(int(ch)))
                cells += (f'<td style="{td}">{int(rk) if pd.notna(rk) else ""}</td>'
                          f'<td style="{td};color:{colour};font-weight:600">{chs}</td>')
        body += f"<tr{mark}>{cells}</tr>"
    lines = []
    for s in summ:
        extra = ("reference" if "mean_diff" not in s
                 else f"{s['mean_diff']:+.2f} team-game MSE vs {html.escape(ref)}, z {s['z']:.1f}, "
                      f"{s['wins']} of {s['n_seasons']} seasons better")
        lines.append(f"{html.escape(s['name'])}: year-over-year error {s['game_armse']:.3f} per 100; "
                     f"next season wants offence x{s['scale_off']:.2f}, defence x{s['scale_def']:.2f}; {extra}")
    summary_html = "".join(f'<p style="margin:0 0 6px;color:#6b7280;font-size:13px">{ln}</p>' for ln in lines)
    return (f'<div style="font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;color:#1f2933">'
            f'<h2 style="font-size:18px;margin:0 0 6px">{html.escape(title)}</h2>{summary_html}'
            f'<table style="border-collapse:collapse;width:100%"><thead><tr>{head}</tr></thead>'
            f'<tbody>{body}</tbody></table>'
            f'<p style="color:#6b7280;font-size:12px">Every player in any top {top} for {season}, ordered by '
            f'{html.escape(ref)}.  Change is rank places moved against {html.escape(ref)}: green up, red down.  '
            f'Highlighted rows are in one top {top} but not another.  Ratings in points per 100 possessions, '
            f'positive good on both ends.</p></div>')


def main():
    specs = [a for a in sys.argv[1:] if not a.startswith("--") and "=" in a]
    if len(specs) < 1:
        raise SystemExit(__doc__)
    season = int(_flag("season", 2026))
    top = int(_flag("top", 20))
    tables = {}
    for spec in specs:
        name, path = spec.split("=", 1)
        tables[name] = load(ROOT / path, season)
    names = list(tables)
    ref = names[0]

    merged = None
    for name, t in tables.items():
        t = t.rename(columns={"rank": f"rank_{name}", "rating_off": f"off_{name}", "rating_def": f"def_{name}",
                              "rating_total": f"tot_{name}", "poss_off": f"poss_{name}"})
        merged = t if merged is None else merged.merge(
            t.drop(columns=["player_name"]), on="player_id", how="outer")
        if merged is not None and "player_name" not in merged:
            merged["player_name"] = t["player_name"]
    merged["player_name"] = merged["player_name"].fillna("")
    merged["poss"] = merged[[f"poss_{n}" for n in names]].max(axis=1)
    in_top = np.zeros(len(merged), dtype=bool)
    in_all_tops = np.ones(len(merged), dtype=bool)
    for n in names:
        r = merged[f"rank_{n}"]
        in_top |= (r <= top).fillna(False).to_numpy()
        in_all_tops &= (r <= top).fillna(False).to_numpy()
    merged["in_top"] = in_top
    merged["marked"] = in_top & ~in_all_tops
    for n in names[1:]:
        merged[f"drank_{n}"] = merged[f"rank_{ref}"] - merged[f"rank_{n}"]      # positive = moved up
    keep = ["player_id", "player_name", "poss", "in_top", "marked"] + [
        c for n in names for c in (f"rank_{n}", f"off_{n}", f"def_{n}", f"tot_{n}")] + [f"drank_{n}" for n in names[1:]]
    records = json.loads(merged[keep].to_json(orient="records"))
    for r in records:
        for k, v in list(r.items()):
            if k.startswith("rank_") or k.startswith("drank_") or k == "poss":
                r[k] = None if v is None else int(v)

    # --alias=page_name:yoy_name,... when the page's short names differ from the year-over-year run's
    alias = {k: v for k, v in (p.split(":") for p in _flag("alias", "").split(",") if p)}
    summ = []
    if _flag("yoy"):
        yoy_names = [alias.get(n, n) for n in names]
        summ = summary(ROOT / _flag("yoy"), yoy_names, alias.get(_flag("ref", ref), _flag("ref", ref)))
        back = {v: k for k, v in alias.items()}
        for r in summ:
            r["name"] = back.get(r["name"], r["name"])
    title = _flag("title", f"Experiment comparison, {season}: " + " vs ".join(names))
    sub = (f"{len(merged)} players in {season}; the union of the top {top} shown by default.  "
           + (f"Year-over-year from {_flag('yoy')}." if _flag("yoy") else ""))
    page = (PAGE.replace("__TITLE__", html.escape(title)).replace("__SUBTITLE__", html.escape(sub))
            .replace("__DATA__", json.dumps(records)).replace("__NAMES__", json.dumps(names))
            .replace("__TOP__", str(top)).replace("__REF__", html.escape(ref))
            .replace("__SUMMARY__", json.dumps(summ)))
    out = ROOT / _flag("out", f"outputs/compare_{'_'.join(names)}_{season}.html")
    out.write_text(page, encoding="utf-8")
    print(f"wrote {out}")
    # a phone version: the same rows as a static table with inline styles, for the body of an email
    # (mail clients strip scripts).  Two tables only; the first is the reference.
    phone = out.with_name(out.stem + "_phone.html")
    phone.write_text(phone_body(merged, names, season, top, title, summ), encoding="utf-8")
    print(f"wrote {phone} (email body)")
    if _flag("open", "1") not in ("0", "no", "false"):
        subprocess.run(["cmd", "/c", "start", "", str(out)], check=False)


if __name__ == "__main__":
    main()
