
(async function () {
  const data = await (await fetch("data/ratings.json")).json();
  const rows = data.rows, windows = data.meta.windows;
  const seasonRows = data.seasons || [], seasons = data.meta.seasons || [];
  // the playoff DELTA (scripts/63_playoff_delta.py): how far a postseason moved a player off his
  // regular-season number, which is what the estimator produces -- not a standalone playoff rating
  const poRows = data.playoffs || [], poWindows = data.meta.playoff_windows || [];
  const $ = id => document.getElementById(id);
  const view = $("view"), block = $("block"), minp = $("minp"), find = $("find"), tbody = $("table").querySelector("tbody");
  let sortKey = "t", sortDir = -1;      // -1 sorts high to low, +1 low to high; a second click on the
                                       // same column flips it, a click on a new one starts high to low
  // the season board is the default view when it has been built (scripts/60_season_board.py); the
  // three-year blocks stay available in the same picker
  view.value = seasons.length ? "season" : "block";
  if (!seasons.length) view.parentElement.style.display = "none";
  if (!poRows.length) view.querySelector('option[value="playoff"]').remove();
  function fillPicker() {
    const v = view.value;
    const keys = v === "season" ? seasons : (v === "playoff" ? poWindows : windows);
    block.replaceChildren();
    keys.forEach(w => { const o = document.createElement("option"); o.value = w; o.textContent = w; block.appendChild(o); });
    block.value = keys[keys.length - 1];
    $("picklabel").textContent = v === "season" ? "Season" : "Block";
    // the playoff view has one extra column (his regular-season total) and its own units, and a playoff
    // possession floor of two thousand would empty the table -- a whole postseason is a few hundred
    const po = v === "playoff";
    $("th_r").hidden = !po;
    $("th_o").textContent = po ? "Δ Offense" : "Offense";
    $("th_d").textContent = po ? "Δ Defense" : "Defense";
    $("th_t").textContent = po ? "Δ Total" : "Total";
    $("th_p").textContent = po ? "Playoff poss." : "Poss.";
    minp.value = po ? 400 : 2000;
    $("note").hidden = !po;
  }
  fillPicker();
  view.addEventListener("change", () => { fillPicker(); render(); });
  $("built").textContent = "Built " + data.meta.built + ".";
  document.querySelectorAll("th[data-k]").forEach(th => th.addEventListener("click", () => {
    sortDir = sortKey === th.dataset.k ? -sortDir : -1;
    sortKey = th.dataset.k;
    const dir = sortDir < 0 ? "descending" : "ascending";
    document.querySelectorAll("th[data-k]").forEach(x => x.setAttribute("aria-sort", x === th ? dir : "none"));
    render();
  }));
  [block, minp].forEach(el => el.addEventListener("change", render));
  find.addEventListener("input", render);
  const fmt = v => (v > 0 ? "+" : "") + v.toFixed(1);
  const fmt2 = v => (v > 0 ? "+" : "") + v.toFixed(2);   // a delta lives in hundredths, not tenths
  function render() {
    const w = block.value, mp = +minp.value || 0, q = find.value.trim().toLowerCase();
    const v = view.value, po = v === "playoff";
    let r = v === "season" ? seasonRows.filter(x => String(x.s) === String(w) && x.p >= mp)
          : po ? poRows.filter(x => x.w === w && x.p >= mp)
               : rows.filter(x => x.w === w && x.p >= mp);
    if (q) r = r.filter(x => x.n.toLowerCase().includes(q));
    r.sort((a, b) => (a[sortKey] - b[sortKey]) * sortDir);
    tbody.replaceChildren();
    r.forEach((p, i) => {
      const tr = document.createElement("tr");
      const cells = po ? [i + 1, p.n, fmt(p.r), fmt2(p.o), fmt2(p.d), fmt2(p.t), p.p.toLocaleString()]
                       : [i + 1, p.n, fmt(p.o), fmt(p.d), fmt(p.t), p.p.toLocaleString()];
      cells.forEach(v => {
        const td = document.createElement("td"); td.textContent = v; tr.appendChild(td);
      });
      tbody.appendChild(tr);
    });
    $("count").textContent = r.length + " players, " + w + ", at least " + mp.toLocaleString()
      + (po ? " playoff possessions" : " possessions");
  }
  render();
})();
