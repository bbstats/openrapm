"""Where the published rankings sit above or below the consensus, by player type, on the OVERALL rating.

    python scripts/76_bias_groups.py [--board=outputs/season_ratings_product.parquet] [--k=8] [--seed=0]
                                     [--seasons=2024,2025,2026] [--out=docs/bias.html] [--json=1]

The types are not hand-made: a Bayesian Gaussian mixture over each player's per-36 box profile
(`eracoef.archetype`, the same fit `scripts/58_archetype.py` uses), which prunes the components the data
does not support instead of splitting a real group in two.  What is new here is the quantity reported.

**One number per group: the possession-weighted bias on the OVERALL rating.**  Offence and defence are
deliberately not reported.  A group whose two sides are large and opposite is a disagreement about WHICH
END a player's value comes from, and both sources can be equally right about how good he is; only the
total says a rating is wrong.  Possession-weighted because a 1,000-possession man and a 5,000-possession
man are not equally part of how wrong the rankings are.

**Two bias columns, because the two sources are not on one scale.**  Our published overall spread is
narrower than the consensus's, so a group of stars would show a bias that is really the amplitude
difference and nothing about that type of player.  `bias` is measured after our overall is stretched ONCE,
globally, to the consensus's spread -- never per group, which would define the answer away.  `bias_raw` is
the same without that stretch.  Both are points per 100, positive = we rate the group ABOVE the consensus.

The names are chosen by hand and each is checked against the group's own profile at run time: every group
carries a `signature` built from the rates it is most extreme on, and the page prints it beside the name,
so a name that has drifted onto the wrong group is visible rather than silent.
"""
import json
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _cli import check_flags, flag  # noqa: E402
from eracoef.archetype import RATES, fit_clusters, per36  # noqa: E402
from eracoef.boxtable import season_box  # noqa: E402
from eracoef.config import load_config  # noqa: E402

MIN_POSS = 1000
SUFFIX = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b")

# The hand-chosen names, keyed by the signature the fit produces for a group: the four rates the group is
# furthest from the league on, strongest first.  Four and not two because the two biggest scoring groups --
# Jokic's and Brunson's -- are both `+ftm +pts` and separate only on the fourth rate, rebounds against
# assists.  A signature is stable in a way a cluster number is not, and a group whose signature is not
# listed prints as UNNAMED rather than borrowing a neighbour's name.
NAMES = {
    "+ftm +pts +tov +drb": "Big men who run the offence",
    "+ftm +pts +tov +ast": "High-usage guards and wings",
    "+fg3m +fg3_miss -drb -orb": "Three-point specialists",
    "+blk +drb -ast -stl": "Shooting big men",
    "+orb -fg3_miss -fg3m +blk": "Rim-protecting centres",
    "+stl -pts -tov -ftm": "Ball-hawk defenders",
    "+ast -orb +tov -blk": "Guards who pass and shoot",
    "-tov -ast -pts -blk": "Low-usage wings and forwards",
}
RATE_WORDS = {"pts": "points", "fg3m": "threes made", "fg3_miss": "threes missed", "ftm": "free throws",
              "ast": "assists", "orb": "offensive rebounds", "drb": "defensive rebounds", "stl": "steals",
              "blk": "blocks", "tov": "turnovers", "pf": "fouls"}


def norm(name) -> str:
    text = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    return SUFFIX.sub("", re.sub(r"[^a-z ]", "", text)).strip()


def joined(board_path: Path, cfg: dict, seasons: list) -> pd.DataFrame:
    """The rankings pooled over `seasons`, possession-weighted, joined to the consensus and the profile."""
    con = pd.read_csv(ROOT / "data" / "external" / "consensus.csv")
    con = con.dropna(subset=["player_name", "adj_overall"]).assign(key=lambda d: d.player_name.map(norm))
    con = con.drop_duplicates("key")[["key", "adj_overall", "var"]]

    board = pd.read_parquet(board_path)
    d = board[board.season.isin(seasons) & board.player_name.notna()].copy()
    w = d.poss_off.to_numpy(float)
    g = pd.DataFrame({c: (d[c] * w).groupby(d.player_id).sum() / d.groupby("player_id").poss_off.sum()
                      for c in ("rating_off", "rating_def", "rating_total")})
    g["poss"] = d.groupby("player_id").poss_off.sum()
    g["player_name"] = d.sort_values("season").drop_duplicates("player_id", keep="last") \
                        .set_index("player_id").player_name
    g = g.reset_index()
    g = g[g.poss >= MIN_POSS].assign(key=lambda x: x.player_name.map(norm))
    g = g.sort_values("poss").drop_duplicates("key", keep="last").merge(con, on="key", how="inner")

    box = season_box(seasons, ["RS"], cfg)
    profile = per36(box[box.phase == "RS"])
    return g.merge(profile[["player_id", "minutes", *RATES]], on="player_id", how="inner") \
            .reset_index(drop=True)


def signature(z_profile: pd.Series, n: int = 4) -> str:
    """The rates a group is furthest from the league average on, strongest first, as `+pts -ast`."""
    order = z_profile.abs().sort_values(ascending=False).index[:n]
    return " ".join(f"{'+' if z_profile[c] > 0 else '-'}{c}" for c in order)


def samples(group: pd.DataFrame, z: pd.DataFrame, centre: pd.Series) -> dict:
    """Who to show so a name can be judged: the biggest names, and the most and least typical members."""
    distance = np.sqrt(((z.loc[group.index, RATES] - centre) ** 2).sum(axis=1))
    typical = group.assign(d=distance).sort_values("d")
    return {"most_played": group.sort_values("poss", ascending=False).player_name.head(6).tolist(),
            "most_typical": typical.player_name.head(6).tolist(),
            "least_typical": typical.player_name.tail(3).tolist()[::-1]}


def build(board_path: Path, cfg: dict, seasons: list, k: int, seed: int) -> dict:
    m = joined(board_path, cfg, seasons)
    labels, weights = fit_clusters(m[RATES].to_numpy(float), k=k, seed=seed)
    m["group"] = labels
    z = pd.DataFrame({c: (m[c] - m[c].mean()) / m[c].std() for c in RATES})

    # one global stretch, so a group's bias is about that group and not about the two spreads
    scale = float(m.adj_overall.std(ddof=1) / m.rating_total.std(ddof=1))
    ours = m.adj_overall.mean() + (m.rating_total - m.rating_total.mean()) * scale
    m["gap"] = ours - m.adj_overall
    m["gap_raw"] = m.rating_total - m.adj_overall

    rows = []
    for g, group in m.groupby("group"):
        if len(group) < 10:                     # a five-player component's mean is noise, not a type
            continue
        w = group.poss.to_numpy(float)
        centre = z.loc[group.index, RATES].mean()
        rows.append({"group": int(g), "players": len(group), "possessions": float(w.sum()),
                     "bias": float(np.average(group.gap, weights=w)),
                     "bias_raw": float(np.average(group.gap_raw, weights=w)),
                     "spread_of_gap": float(group.gap.std(ddof=1)),
                     "our_overall": float(np.average(group.rating_total, weights=w)),
                     "consensus_overall": float(np.average(group.adj_overall, weights=w)),
                     "signature": signature(centre), "profile": centre.round(2).to_dict(),
                     "minutes_per_game_median": float(group.minutes.median()),
                     **samples(group, z, centre)})
    rows.sort(key=lambda r: -abs(r["bias"]))
    share = sum(r["possessions"] for r in rows)
    for r in rows:
        r["possession_share"] = r["possessions"] / share
        r["name"] = NAMES.get(r["signature"], "UNNAMED")
    return {"board": board_path.name, "seasons": seasons, "k": k, "seed": seed,
            "players": len(m), "components_used": int((weights > 0.01).sum()),
            "global_stretch": scale, "groups": rows}


def page(result: dict) -> str:
    """One self-contained page: the bias per group, each group's profile, and who is in it."""
    head = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>OpenRAPM: bias by player type</title>
<style>
  :root { color-scheme: light dark; --line: #d9d8d3; --muted: #6b6a66; --accent: #2a78d6;
          --over: #b3452b; --under: #2a6fb3; }
  @media (prefers-color-scheme: dark) { :root { --line: #3a3a38; --muted: #a09f98; --accent: #3987e5;
          --over: #e07a5f; --under: #7fb2e5; } }
  body { margin: 0; font: 15px/1.45 -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; }
  main { max-width: 900px; margin: 0 auto; padding: 28px 20px 48px; }
  h1 { font-size: 26px; margin: 0 0 6px; }
  h2 { font-size: 17px; margin: 26px 0 4px; }
  p.note { color: var(--muted); max-width: 70ch; margin: 0 0 14px; }
  table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; margin-bottom: 8px; }
  th, td { padding: 5px 8px; border-bottom: 1px solid var(--line); text-align: right; white-space: nowrap; }
  th:nth-child(1), td:nth-child(1) { text-align: left; white-space: normal; }
  th { color: var(--muted); font-weight: 600; }
  td.over { color: var(--over); } td.under { color: var(--under); }
  .sig { color: var(--muted); font-size: 12px; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
  .who { max-width: 70ch; }
  .who div { margin: 2px 0; } .who span { color: var(--muted); }
  footer { margin-top: 30px; color: var(--muted); font-size: 13px; }
  footer a, h1 a { color: inherit; }
</style>
</head>
<body>
<main>
<h1>Bias by player type</h1>
"""
    g = result["groups"]
    intro = (f"<p class=\"note\">How far the published overall rating sits above or below the consensus, "
             f"for each player type, over {'-'.join(str(s) for s in (result['seasons'][0], result['seasons'][-1]))}. "
             f"Possession-weighted, points per 100 possessions, positive means <b>we rate the type above the "
             f"consensus</b>. {result['players']} players with 1,000+ possessions, "
             f"{len(g)} types from a Bayesian Gaussian mixture over per-36 box rates "
             f"({result['components_used']} of {result['k']} components used, seed {result['seed']}).</p>"
             f"<p class=\"note\">Only the overall rating is shown. A type whose offence and defence are "
             f"wrong in opposite directions is a disagreement about which end the value comes from, and both "
             f"sources can be right about how good the player is; only the total says a rating is wrong. Our "
             f"overall is stretched once, globally, by x{result['global_stretch']:.3f} to match the "
             f"consensus's spread, so that a type of stars does not read as biased merely because the two "
             f"scales differ; the unstretched figure is in the last column.</p>")

    rows = []
    for r in g:
        cls = "over" if r["bias"] > 0 else "under"
        rows.append(
            f"<tr><td>{r['name']}<br><span class=\"sig\">{r['signature']}</span></td>"
            f"<td class=\"{cls}\"><b>{r['bias']:+.3f}</b></td>"
            f"<td>{r['our_overall']:+.2f}</td><td>{r['consensus_overall']:+.2f}</td>"
            f"<td>{r['players']}</td><td>{100 * r['possession_share']:.1f}%</td>"
            f"<td>{r['bias_raw']:+.3f}</td></tr>")
    table = ("<table><thead><tr><th>Player type</th><th>Bias</th><th>Ours</th><th>Consensus</th>"
             "<th>Players</th><th>Possessions</th><th>Unstretched</th></tr></thead><tbody>"
             + "".join(rows) + "</tbody></table>")

    blocks = []
    for r in g:
        profile = ", ".join(f"{RATE_WORDS[c]} {v:+.2f}" for c, v in
                            sorted(r["profile"].items(), key=lambda kv: -abs(kv[1]))[:6])
        blocks.append(
            f"<h2>{r['name']} <span class=\"sig\">{r['bias']:+.3f}</span></h2>"
            f"<div class=\"who\">"
            f"<div><span>per-36 rates against the league, strongest first:</span> {profile}</div>"
            f"<div><span>most possessions:</span> {', '.join(r['most_played'])}</div>"
            f"<div><span>most typical of the type:</span> {', '.join(r['most_typical'])}</div>"
            f"<div><span>least typical, still in it:</span> {', '.join(r['least_typical'])}</div>"
            f"</div>")

    foot = (f"<footer>Built by <code>scripts/76_bias_groups.py</code> from "
            f"<code>outputs/{result['board']}</code> and <code>data/external/consensus.csv</code>. "
            f"The consensus is a sanity check and never a fitting target. "
            f"<a href=\"index.html\">Back to the rankings</a>.</footer></main></body></html>")
    return head + intro + table + "".join(blocks) + foot


def main() -> None:
    check_flags()
    cfg = load_config()
    seasons = [int(s) for s in flag("seasons", "2024,2025,2026").split(",")]
    result = build(ROOT / flag("board", "outputs/season_ratings_product.parquet"), cfg, seasons,
                   int(flag("k", 8)), int(flag("seed", 0)))
    out = ROOT / flag("json_out", "outputs/bias_groups.json")
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    html = ROOT / flag("out", "docs/bias.html")
    html.write_text(page(result), encoding="utf-8")
    print(f"wrote {html.relative_to(ROOT).as_posix()}")
    unnamed = [r["signature"] for r in result["groups"] if r["name"] == "UNNAMED"]
    if unnamed:
        print(f"  WARNING: no name for {unnamed} -- the fit moved and `NAMES` needs the new signature")
    print(f"wrote {out.relative_to(ROOT).as_posix()}: {len(result['groups'])} groups over "
          f"{result['players']} players, {result['components_used']} of {result['k']} components used, "
          f"our overall stretched x{result['global_stretch']:.3f} to the consensus's spread")
    for r in result["groups"]:
        print(f"\n  group {r['group']}  {r['signature']:<24} {r['name']}")
        print(f"    {r['players']} players, {100 * r['possession_share']:.1f}% of possessions, "
              f"bias {r['bias']:+.3f} (raw {r['bias_raw']:+.3f}), "
              f"ours {r['our_overall']:+.2f} theirs {r['consensus_overall']:+.2f}")
        print(f"    profile: " + ", ".join(f"{c} {v:+.2f}" for c, v in r["profile"].items()))
        print(f"    most played:   {', '.join(r['most_played'])}")
        print(f"    most typical:  {', '.join(r['most_typical'])}")
        print(f"    least typical: {', '.join(r['least_typical'])}")


if __name__ == "__main__":
    main()
