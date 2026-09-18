"""Where the published rankings sit above or below the consensus, by player type, on the OVERALL rating.

    python scripts/76_bias_groups.py [--board=outputs/season_ratings_product.parquet] [--k=8] [--seed=0]
                                     [--seasons=2026] [--out=docs/bias.html]

The types are not hand-made: a Bayesian Gaussian mixture over each player's per-36 box profile
(`eracoef.archetype`, the same fit `scripts/58_archetype.py` uses), which prunes the components the data
does not support instead of splitting a real group in two.  What is new here is the quantity reported.

**One number per group: the possession-weighted bias on the OVERALL rating.**  Offence and defence are
deliberately not reported.  A group whose two sides are large and opposite is a disagreement about WHICH
END a player's value comes from, and both sources can be equally right about how good he is; only the
total says a rating is wrong.  Possession-weighted because a 1,000-possession man and a 5,000-possession
man are not equally part of how wrong the rankings are.

**Two tables, and one bias column in each.**  The first compares the published overall rating with the
consensus; the second compares it with what the season's own games say, which is
`scripts/70_tradeset.py --block=0`: the rating is the prediction for every team-game, and a ridge is
fitted to what is left, so the coefficient is the part of the residual belonging to the player.  Both are
read the same way -- a single global rescale is taken out first, never a per-group one, which would define
the answer away -- and in both, positive means WE RATE THE TYPE ABOVE the other source, points per 100.

Ordered by `|bias| x possessions`, so a small bias over a quarter of the league outranks a large one over
a twentieth.

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
    "+ftm +pts +tov +ast": "High-usage scorers who draw fouls",
    "+blk +drb +pts +ftm": "Big men who score and block shots",
    "+fg3_miss +fg3m +pts -orb": "High-volume three-point shooters",
    "+drb +blk -stl -ast": "Forwards who rebound and block",
    "+orb -fg3_miss -fg3m +blk": "Rim-protecting centers",
    "+stl -pts -ftm -drb": "Ball-hawk defenders",
    "+ast -blk -orb +tov": "Guards who handle the ball",
    "-tov -ast -ftm -pts": "Low-usage wings",
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


def residual_rapm(alpha_path: Path, seasons: list) -> pd.DataFrame:
    """What the games say our rating missed, per player, from a RAPM fitted to our own residual.

    `scripts/70_tradeset.py --block=0` takes the published rating as the prediction for every team-game
    of the rated season, subtracts it, and fits a ridge on the player columns of what is left, with one
    free unpenalised column per side for the amplitude and one free level per team.  So the coefficient
    is the per-player part of the residual after the best single rescale, which is the same treatment the
    consensus comparison gets, and the two tables are read the same way.

    `rating` in that table is positive-good on both sides.  Returned per player, pooled over `seasons`
    possession-weighted: `ours_scaled` is our rating at the amplitude the games ask for, `games` is that
    plus the per-player residual.
    """
    a = pd.read_parquet(alpha_path)
    a = a[a.season.isin(seasons) & a.eligible].copy()
    a["scaled"] = a.rating_scale * a.rating
    wide = a.pivot_table(index=["player_id", "season"], columns="side",
                         values=["scaled", "alpha_good", "poss_off"]).reset_index()
    wide.columns = [c[0] if not c[1] else f"{c[0]}_{c[1][:3]}" for c in wide.columns]
    wide = wide.dropna(subset=["scaled_off", "scaled_def", "alpha_good_off", "alpha_good_def"])
    wide["ours_scaled"] = wide.scaled_off + wide.scaled_def
    wide["residual"] = wide.alpha_good_off + wide.alpha_good_def
    w = wide.poss_off_off.to_numpy(float)
    out = pd.DataFrame({c: (wide[c] * w).groupby(wide.player_id).sum()
                        / wide.groupby("player_id").poss_off_off.sum()
                        for c in ("ours_scaled", "residual")})
    return out.reset_index()


def group_meta(m: pd.DataFrame, z: pd.DataFrame) -> dict:
    """Per group, the things that describe the TYPE: its profile, its signature, its name, its members.

    Computed once from the whole group and shared by every table, because the second table covers fewer
    players -- a residual needs the player to have been on the floor enough for the fit to identify him --
    and a group's name must not change between two tables about the same group.
    """
    meta = {}
    for g, group in m.groupby("group"):
        centre = z.loc[group.index, RATES].mean()
        sig = signature(centre)
        meta[int(g)] = {"signature": sig, "profile": centre.round(2).to_dict(),
                        "name": NAMES.get(sig, "UNNAMED"), "centre": centre,
                        **samples(group, z, centre)}
    return meta


def group_rows(m: pd.DataFrame, meta: dict, ours: str, theirs: str, bias: str) -> list:
    """One row per group: the two levels and the bias, possession-weighted, ordered by what it costs.

    The order is `|bias| x possessions` -- how much of the league a type's bias applies to, not how
    large the bias is on one player, because a 0.6 miss over a quarter of the possessions is a bigger
    problem than a 1.1 miss over a twentieth.
    """
    rows = []
    for g, group in m.groupby("group"):
        group = group.dropna(subset=[ours, theirs])
        if len(group) < 10:                     # a five-player component's mean is noise, not a type
            continue
        w = group.poss.to_numpy(float)
        described = {k: v for k, v in meta[int(g)].items() if k != "centre"}
        rows.append({"group": int(g), "players": len(group), "possessions": float(w.sum()),
                     "bias": float(np.average(group[bias], weights=w)),
                     "ours": float(np.average(group[ours], weights=w)),
                     "theirs": float(np.average(group[theirs], weights=w)), **described})
    for r in rows:
        r["cost"] = abs(r["bias"]) * r["possessions"]
    rows.sort(key=lambda r: -r["cost"])
    return rows


def build(board_path: Path, cfg: dict, seasons: list, k: int, seed: int, alpha_path: Path) -> dict:
    m = joined(board_path, cfg, seasons)
    labels, weights = fit_clusters(m[RATES].to_numpy(float), k=k, seed=seed)
    m["group"] = labels
    z = pd.DataFrame({c: (m[c] - m[c].mean()) / m[c].std() for c in RATES})

    # one global stretch, so a group's bias is about that group and not about the two spreads
    scale = float(m.adj_overall.std(ddof=1) / m.rating_total.std(ddof=1))
    m["ours_vs_consensus"] = m.adj_overall.mean() + (m.rating_total - m.rating_total.mean()) * scale
    m["bias_vs_consensus"] = m.ours_vs_consensus - m.adj_overall

    meta = group_meta(m, z)
    games = residual_rapm(alpha_path, seasons)
    m = m.merge(games, on="player_id", how="left")
    m["games"] = m.ours_scaled + m.residual
    m["bias_vs_games"] = -m.residual                     # positive = we rate him above what the games say

    return {"board": board_path.name, "seasons": seasons, "k": k, "seed": seed,
            "players": len(m), "components_used": int((weights > 0.01).sum()),
            "global_stretch": scale, "residual_players": int(m.residual.notna().sum()),
            "groups": group_rows(m, meta, "ours_vs_consensus", "adj_overall", "bias_vs_consensus"),
            "groups_vs_games": group_rows(m, meta, "ours_scaled", "games", "bias_vs_games")}


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
    by_group = {r["group"]: r for r in result["groups_vs_games"]}
    out = ['<table><thead><tr><th>Player type</th><th>Consensus</th><th>Ours</th>'
           '<th>Bias vs consensus</th><th>Bias vs Season Performance+</th></tr></thead><tbody>']
    for r in result["groups"]:
        other = by_group.get(r["group"])
        cls = "over" if r["bias"] > 0 else "under"
        cell = "&ndash;"
        if other is not None:
            cell = (f"<b>{other['bias']:+.3f}</b>", )[0]
            cell = f"<span class=\"{'over' if other['bias'] > 0 else 'under'}\"><b>{cell}</b></span>"
        out.append(f"<tr><td>{r['name']}<br><span class=\"sig\">{r['signature']}</span></td>"
                   f"<td>{r['theirs']:+.2f}</td><td>{r['ours']:+.2f}</td>"
                   f"<td class=\"{cls}\"><b>{r['bias']:+.3f}</b></td>"
                   f"<td>{cell}</td></tr>")
    return head + "".join(out) + "</tbody></table></main></body></html>"


def main() -> None:
    check_flags()
    cfg = load_config()
    seasons = [int(s) for s in flag("seasons", "2026").split(",")]
    result = build(ROOT / flag("board", "outputs/season_ratings_product.parquet"), cfg, seasons,
                   int(flag("k", 8)), int(flag("seed", 0)),
                   ROOT / flag("alpha", "outputs/tradeset_actual_b0_alpha.parquet"))
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
    print(f"  the residual RAPM covers {result['residual_players']} of {result['players']} players")
    for key, label in (("groups", "against the consensus"), ("groups_vs_games", "against the games")):
        print(f"\n=== {label}, ordered by |bias| x possessions")
        for r in result[key]:
            print(f"  {r['name']:<30} bias {r['bias']:+.3f}  ours {r['ours']:+.2f}  "
                  f"theirs {r['theirs']:+.2f}  {r['players']:>3} players, "
                  f"{r['possessions'] / 1000:>5.0f}k possessions  {r['signature']}")


if __name__ == "__main__":
    main()
