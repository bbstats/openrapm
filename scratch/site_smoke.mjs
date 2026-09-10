// Run docs/index.html's script headlessly against the real docs/data/ratings.json.
//
// The page is static and has no test, so a change to it is otherwise unverified until somebody opens a
// browser.  This stubs just enough DOM for the script to run, drives the view picker through every mode,
// and prints the first rows of each -- which is what a reader would see.  It is a smoke test, not a
// rendering test: it proves the data reaches the table and the columns line up, nothing about layout.
//
//   node scratch/site_smoke.mjs
import { readFileSync } from "node:fs";

const html = readFileSync("docs/index.html", "utf8");
const src = html.match(/<script>([\s\S]*?)<\/script>/)[1];
const data = JSON.parse(readFileSync("docs/data/ratings.json", "utf8"));

const listeners = new Map();
function el(id, tag = "div") {
  return {
    id, tagName: tag, value: "", textContent: "", hidden: false, dataset: {}, children: [],
    parentElement: { style: {} },
    appendChild(c) { this.children.push(c); }, replaceChildren(...c) { this.children = c; },
    querySelector() { return tbody; }, querySelectorAll() { return []; },
    remove() { this.removed = true; },
    setAttribute() {}, getAttribute() { return null; },
    addEventListener(ev, fn) { listeners.set(`${id}:${ev}`, fn); },
  };
}
const tbody = el("tbody", "tbody");
const ids = ["view", "block", "minp", "find", "table", "count", "built", "picklabel",
             "th_r", "th_o", "th_d", "th_t", "th_p"];
const nodes = Object.fromEntries(ids.map(i => [i, el(i)]));
nodes.table.querySelector = () => tbody;
nodes.view.querySelector = () => ({ remove() { this.removed = true; } });

global.document = {
  getElementById: id => nodes[id] ?? el(id),
  createElement: tag => el("_" + tag, tag),
  querySelectorAll: () => [],
};
global.fetch = async () => ({ json: async () => data });

await eval(`(${src.trim().replace(/^\(/, "(").replace(/\)\(\);?$/, ")")})()`);

function show(view, label) {
  nodes.view.value = view;
  listeners.get("view:change")?.();
  const rows = tbody.children.slice(0, 3).map(tr => tr.children.map(td => td.textContent).join(" | "));
  console.log(`\n--- ${label}: ${nodes.block.value}, ${nodes.count.textContent}`);
  console.log(`    season column hidden: ${nodes.th_r.hidden}`);
  console.log(`    columns: ${["#", "Player", nodes.th_r.hidden ? null : (nodes.th_r.textContent || "Season"),
    nodes.th_o.textContent, nodes.th_d.textContent, nodes.th_t.textContent,
    nodes.th_p.textContent].filter(Boolean).join(" | ")}`);
  rows.forEach(r => console.log("    " + r));
  const want = nodes.th_r.hidden ? 6 : 7;
  const got = tbody.children[0]?.children.length ?? 0;
  if (got !== want) { console.error(`COLUMN MISMATCH: header wants ${want}, row has ${got}`); process.exitCode = 1; }
  if (!rows.length) { console.error("EMPTY TABLE"); process.exitCode = 1; }
}
show("season", "Season");
show("block", "3-year block");
show("playoff", "Playoff delta");
