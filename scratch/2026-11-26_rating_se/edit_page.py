"""One-off: the standard error column on the Ratings page, docs/index.html (run once)."""
from edit_folds import edit

PAGE = [
    # the columns: the total's standard error beside Total; in Detail the shown side's beside its rating
    ('''  const COLUMNS = { ratings: [["o", "Offense"], ["d", "Defense"], ["t", "Total"], ["w", "WOSA"], ["p", "Poss."]],
                    detail: [...PIECES, ["a", "OpenRAPM"], ["p", "Poss."], ...WOSA_COLS] };''',
     '''  // se is the total's standard error and sa the shown side's (meta.fields has "se" once openrapm errors has run);
  // without them in the file the two columns are left out
  const COLUMNS = { ratings: [["o", "Offense"], ["d", "Defense"], ["t", "Total"], ["se", "\\u00b1"], ["w", "WOSA"], ["p", "Poss."]],
                    detail: [...PIECES, ["a", "OpenRAPM"], ["sa", "\\u00b1"], ["p", "Poss."], ...WOSA_COLS] };
  const ERROR_KEYS = ["se", "sa"];
  const SIDE_ERROR = { off: 0, def: 1, total: 2 };
  const columns = () => COLUMNS[state.view].filter(([key]) => data && data.errors || !ERROR_KEYS.includes(key));'''),
    ('''                 t: "Offense plus defense, points per 100 possessions",''',
     '''                 t: "Offense plus defense, points per 100 possessions",
                 se: "Standard error of Total",
                 sa: "Standard error of OpenRAPM",'''),
    ('''    const doc = await (await fetch("data/ratings_list.json")).json();
    return { meta: doc.meta, rows: rowsOf(doc) };''',
     '''    const doc = await (await fetch("data/ratings_list.json")).json();
    return { meta: doc.meta, rows: rowsOf(doc), errors: doc.meta.fields.includes("se") };'''),
    ('''        r.x = { o: r.o, d: r.d, t: r.t, wo: r.wo, wd: r.wd, wt: r.wt };
        r.o = r.x.o[5]; r.d = r.x.d[5]; r.t = r.x.t[5];
      }
      data = { meta: Object.assign({}, doc.meta, data.meta), rows, detail: true };''',
     '''        r.x = { o: r.o, d: r.d, t: r.t, wo: r.wo, wd: r.wd, wt: r.wt, se: r.se || null };
        r.o = r.x.o[5]; r.d = r.x.d[5]; r.t = r.x.t[5];
        r.se = r.x.se ? r.x.se[2] : null;
      }
      data = { meta: Object.assign({}, doc.meta, data.meta), rows, detail: true, errors: doc.meta.fields.includes("se") };'''),
    ('''    const row = { n: r.n, id: r.id, p: r.p, tm: r.tm, w: r.w, o: r.o, d: r.d, t: r.t, rt: r.t };
    if (!r.x) return row;                             // a row of the list file: the ratings alone
    const v = r.x[SIDE_FIELD[side]], x = r.x[WOSA_SIDE[side]];
    return Object.assign(row, { box: v[0], adj: v[4], on: v[1], off: v[2], sos: v[3], a: v[5], wl: x[1], ws: x[2] });''',
     '''    const row = { n: r.n, id: r.id, p: r.p, tm: r.tm, w: r.w, o: r.o, d: r.d, t: r.t, rt: r.t,
                  se: r.se === undefined ? null : r.se, sa: null };
    if (!r.x) return row;                             // a row of the list file: the ratings alone
    const v = r.x[SIDE_FIELD[side]], x = r.x[WOSA_SIDE[side]];
    if (r.x.se) row.sa = r.x.se[SIDE_ERROR[side]];
    return Object.assign(row, { box: v[0], adj: v[4], on: v[1], off: v[2], sos: v[3], a: v[5], wl: x[1], ws: x[2] });'''),
    ('''      `<th class="team">Team</th>` + COLUMNS[state.view].map(([key, name]) => th(key, name)).join("") + `</tr>`;''',
     '''      `<th class="team">Team</th>` + columns().map(([key, name]) => th(key, name)).join("") + `</tr>`;'''),
    ('''    const cols = COLUMNS[state.view], bold = detail ? ["a", "ws"] : [SIDE_FIELD[state.side]];''',
     '''    const cols = columns(), bold = detail ? ["a", "ws"] : [SIDE_FIELD[state.side]];'''),
    ('''    const detailCell = (key, v) => key === "wl" ? v.toFixed(2) : fmt(v);''',
     '''    const detailCell = (key, v) => key === "wl" ? v.toFixed(2) : fmt(v);
    // a standard error as written: no sign, a dash without one; the same with the percentiles on, which it has none of
    const errorCell = v => v === null ? "\\u2014" : v.toFixed(detail ? 2 : 1);'''),
    ('''        `<td${bold.includes(key) ? ' class="total"' : ""}${p.c ? shade(p.c[key]) : ""}>${key === "p" ? whole.format(p.p)
          : state.pct''',
     '''        `<td${bold.includes(key) ? ' class="total"' : ""}${p.c ? shade(p.c[key]) : ""}>${key === "p" ? whole.format(p.p)
          : ERROR_KEYS.includes(key) ? errorCell(p[key])
          : state.pct'''),
    # the sort follows the column between the two views
    ('''      if (view === "ratings" ? s.k === "ws" : s.k === "w") s.k = view === "ratings" ? "w" : "ws";
      else if''',
     '''      if (view === "ratings" ? s.k === "ws" : s.k === "w") s.k = view === "ratings" ? "w" : "ws";
      else if (view === "ratings" ? s.k === "sa" : s.k === "se") s.k = view === "ratings" ? "se" : "sa";
      else if'''),
    # the downloads
    ('''      ? [["season", "player_id", "player", "team", "offense", "defense", "total", "wosa", "possessions"],
         ...pairs.map(([per, x]) => [per, x.id, x.n, x.tm, x.o, x.d, x.t, x.w, x.p])]''',
     '''      ? [["season", "player_id", "player", "team", "offense", "defense", "total", "wosa", "possessions",
          ...(data.errors ? ["total_standard_error"] : [])],
         ...pairs.map(([per, x]) => [per, x.id, x.n, x.tm, x.o, x.d, x.t, x.w, x.p, ...(data.errors ? [x.se] : [])])]'''),
    ('''          ...["offense", "defense", "total"].flatMap(s => CSV_WOSA.map(k => s + "_" + k))],
         ...pairs.map(([per, { id, n, tm, p, x }]) => [per, id, n, tm, p, ...csvPieces(x.o), ...csvPieces(x.d),
                                                      ...csvPieces(x.t), ...x.wo.slice(1), ...x.wd.slice(1),
                                                      ...x.wt.slice(1)])];''',
     '''          ...["offense", "defense", "total"].flatMap(s => CSV_WOSA.map(k => s + "_" + k)),
          ...(data.errors ? ["offense", "defense", "total"].map(s => s + "_standard_error") : [])],
         ...pairs.map(([per, { id, n, tm, p, x }]) => [per, id, n, tm, p, ...csvPieces(x.o), ...csvPieces(x.d),
                                                      ...csvPieces(x.t), ...x.wo.slice(1), ...x.wd.slice(1),
                                                      ...x.wt.slice(1),
                                                      ...(data.errors ? x.se || [null, null, null] : [])])];'''),
]

if __name__ == "__main__":
    edit("docs/index.html", PAGE)
