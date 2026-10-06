"""Experiment 40: a portable rating next to the team rating.

    python scripts/112_portable.py

The owner (2026-10-05), after experiment 39: "Yes portable rating. It would be good to deal w/ hey these are often
role players as well and ideally openrapm can surface a traded rating and non (provided we adjust for the fact that
traded players are worse often)."

  team rating      OpenRAPM as shipped (prior shrink + swap adjustment x0.5): best for his own team's games
  portable rating  the same rating with its box-prior part and its part beyond the box score (games part + swap
                   adjustment) each times a ratio: how much of that part holds up on a NEW team, judged against
                   players of the same playing-time tier who stayed (src/eracoef/portable.py)

Two places a player lands on a new team, each fit and scored:
  deadline   traded during the season (experiment 39's folds: rated before the cut, scored after, and the reverse)
  yoy        on a different team the next (or previous) season than his team in the rated season -- a different and
             larger sample, with more good players
Variants (portable.groups_for): both parts per side (4 ratios), the part beyond the box only (2), one ratio for it
(1), and per playing-time tier (12).  Everything cross-fitted: season H's ratios come from development seasons outside
{H-1, H, H+1}; development seasons only.  Each portable rating is given to the players on a new team only (everyone
else keeps the team rating) and scored by the paired t over seasons of the team-game MSE change and by the blend
weight of that change (1 = all signal, 0 = none).

Writes outputs/portable/*.csv, outputs/portable/production.json (the published ratio: one ratio for the part beyond
the box score, both sides, pooled over the two samples by inverse variance) and the portable rating beside the test
table (outputs/season_ratings_portable.parquet) and beside the product table the site publishes
(outputs/season_ratings_product_portable.parquet).
"""
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _cli import check_flags, flag  # noqa: E402
from eracoef import heldoutprior as hp  # noqa: E402
from eracoef import portable as pt  # noqa: E402
from eracoef import scorecard as sc  # noqa: E402
from eracoef import tradeset as ts  # noqa: E402
from eracoef.config import load_config  # noqa: E402
from eracoef.holdout import Context  # noqa: E402

OUT = ROOT / "outputs" / "portable"
TAG = "deadline"
SEASONS = list(range(1997, 2027))
DEV = [s for s in SEASONS if s not in hp.LOCKBOX]
VARIANTS = ("box+beyond", "beyond", "beyond-one", "box+beyond-tier")
SOURCES = ("deadline", "yoy")


def _borrow(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def paired_t(d) -> tuple:
    d = pd.Series(d).dropna()
    se = d.std(ddof=1) / np.sqrt(len(d))
    return float(d.mean()), (float(d.mean() / se) if se > 0 else np.nan), int((d < 0).sum()), len(d)


def jack_se(values: np.ndarray) -> float:
    k = len(values)
    return float(np.sqrt((k - 1) / k * ((values - values.mean()) ** 2).sum()))


def centred(t: pd.DataFrame) -> pd.DataFrame:
    w = t.poss.to_numpy(float)
    return t.assign(o=t.o - np.average(t.o, weights=w), d=t.d - np.average(t.d, weights=w))


def as_rating(parts: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"player_id": parts.player_id.to_numpy(), "o": (parts.o_prior + parts.o_beyond).to_numpy(),
                         "d": (parts.d_prior + parts.d_beyond).to_numpy(), "poss": parts.poss.to_numpy()})


# ------------------------------------------------------------------------------------------ the scored sets
def load_folds(S104) -> list:
    """The deadline folds with OpenRAPM-as-shipped split into its parts (raw sign)."""
    directory = ROOT / "outputs" / "within" / TAG
    shipped = pd.read_parquet(directory / "openrapm_shipped.parquet")
    O, W = shipped[shipped.variant == "shrunk"], shipped[shipped.variant == "swap0.5"]
    out = []
    for stem in S104.fold_stems(directory):
        f = S104.load_fold(directory, stem)
        b = f["block"]
        if b.season not in DEV:
            continue
        s = f["summary"]
        share = float(s["fit_games"]) / (float(s["fit_games"]) + float(s["test_games"]))
        players = f["players"].drop_duplicates("player_id").set_index("player_id")
        o = O[O.key == stem].set_index("player_id")
        w = W[W.key == stem].set_index("player_id")
        assert o.index.equals(w.index), stem
        u_o, u_d = players.u_off.reindex(w.index), -players.u_def.reindex(w.index)
        parts = pd.DataFrame({"player_id": w.index.to_numpy(), "o_prior": (o.o - u_o).to_numpy(),
                              "d_prior": (o.d - u_d).to_numpy(), "poss": w.poss.to_numpy()})
        parts["o_beyond"] = w.o.to_numpy() - parts.o_prior
        parts["d_beyond"] = w.d.to_numpy() - parts.d_prior
        # the shrunk prior part is the fold's prior times one multiplier per side plus a constant (106's shrink)
        for side, col, sign in (("o", "prior_off", 1.0), ("d", "prior_def", -1.0)):
            x = sign * players[col].reindex(w.index).to_numpy(float)
            A = np.column_stack([np.ones(len(x)), x])
            y = parts[f"{side}_prior"].to_numpy()
            assert np.abs(y - A @ np.linalg.lstsq(A, y, rcond=None)[0]).max() < 1e-8, f"{stem}: prior part ({side})"
        parts["tier"] = pt.tier_of(parts.poss.to_numpy(), 1.0 / share)
        team = as_rating(parts)
        assert np.allclose(team.o, w.o.to_numpy()) and np.allclose(team.d, w.d.to_numpy())
        team_of = players.team_id
        Zt = sc.traded_entries(b, team_of[team_of >= 0], *f["teams"])
        fill = ts.replacement_fill(team, max_poss=500.0, shrink=0.25)
        p = sc.parts(b, team, fill)
        eq = sc.normal(b, pt.columns(b, Zt, parts), offset=p.s_o + p.s_d, level="full")
        out.append(dict(source="deadline", key=stem, season=b.season, touches={b.season}, share=share, block=b,
                        Zt=Zt, parts=parts, team=team, fill=fill, eq=eq))
    return out


def load_yoy(B70) -> list:
    """Season H rated (the shipped table), its development neighbours scored; players on a team other than their
    season-H team are the movers.  Stand-ins at 0, as the year-over-year test scores them."""
    cfg = load_config(ROOT / "config.yaml")
    ctx = Context.load(cfg)
    inc = pd.read_parquet(ROOT / "outputs" / "season_ratings_priorshrink.parquet")
    inc = inc[inc.poss_off > 0]
    roles = pd.read_parquet(ROOT / "data/cache/roles_RSPO.parquet", columns=["player_id", "season", "team_id", "poss_on"])
    main_team = (roles[roles.poss_on > 0].groupby(["player_id", "season", "team_id"], as_index=False).poss_on.sum()
                 .sort_values("poss_on").drop_duplicates(["player_id", "season"], keep="last"))
    blocks, out = {}, []
    for H in DEV:
        r = inc[inc.season == H]
        parts = pd.DataFrame({"player_id": r.player_id.astype(np.int64).to_numpy(), "o_prior": r.prior_off.to_numpy(float),
                              "o_beyond": (r.u_off + r.c_off).to_numpy(float), "d_prior": -r.prior_def.to_numpy(float),
                              "d_beyond": -(r.u_def + r.c_def).to_numpy(float), "poss": r.poss_off.to_numpy(float)})
        parts["tier"] = pt.tier_of(parts.poss.to_numpy(), 1.0)
        team = as_rating(parts)
        assert np.allclose(team.o, r.rating_off.to_numpy()) and np.allclose(team.d, -r.rating_def.to_numpy())
        tm = main_team[main_team.season == H].set_index("player_id").team_id
        for T in (H - 1, H + 1):
            if T not in DEV:
                continue
            if T not in blocks:
                pooled = ts.team_game_design(ctx.design([T], "pts"))
                names = list(pooled.control_names)
                F = np.asarray(pooled.F, dtype=float)
                blk = sc.Block(key=str(T), season=T, deal=0, Z=sp.csr_matrix(pooled.Z), y=pooled.y.astype(float),
                               w=pooled.w.astype(float), F=F, home=F[:, names.index("home")],
                               player_ids=pooled.player_ids.astype(np.int64))
                blocks[T] = (blk, B70.team_of_rows(pooled.keys, cfg))
            blk, teams = blocks[T]
            Zt = sc.traded_entries(blk, tm, teams.team_off.to_numpy(), teams.team_def.to_numpy())
            eq = sc.normal(blk, pt.columns(blk, Zt, parts), level="full")
            out.append(dict(source="yoy", key=f"{H}->{T}", season=H, scored=T, touches={H, T}, block=blk, Zt=Zt,
                            parts=parts, team=team, fill=(0.0, 0.0), eq=eq))
    return out


def gram(sets, keep) -> tuple:
    eqs = [x["eq"] for x in sets if keep(x)]
    return sum(e.xx for e in eqs), sum(e.xy for e in eqs)


def describe(folds) -> pd.DataFrame:
    """Players on a new team vs everyone else on the scored games (the owner: traded players are often role players)."""
    rows = []
    for f in folds:
        b, Zt, parts = f["block"], f["Zt"], f["parts"].set_index("player_id")
        n = b.n_players
        poss_all = np.asarray(b.Z.T @ b.w).ravel()
        poss_tr = np.asarray(Zt.T @ b.w).ravel()
        on = (poss_all[:n] + poss_all[n:]) > 0
        moved = (poss_tr[:n] + poss_tr[n:]) > 0
        for mask, group in ((moved & on, "new team"), (~moved & on, "same team")):
            p = parts.reindex(b.player_ids[mask]).dropna(subset=["o_prior"])
            rows.append(dict(source=f["source"], key=f["key"], group=group, players=len(p),
                             rating=float((p.o_prior + p.o_beyond - p.d_prior - p.d_beyond).mean()),
                             box_part=float((p.o_prior - p.d_prior).mean()),
                             beyond_part=float((p.o_beyond - p.d_beyond).mean()),
                             season_poss=float((p.poss / f.get("share", 1.0)).median()),
                             tier0=float((p.tier == 0).mean()), tier1=float((p.tier == 1).mean()),
                             tier2=float((p.tier == 2).mean())))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------ scoring
def score(x, port, fill_p) -> dict:
    """The portable rating given to players on a new team only, to the others only, to everyone; and the blend
    weights of its change, split the same way."""
    b = x["block"]
    stay, moved = sc.split_traded(b, x["Zt"])
    ts_, tm_ = sc.parts(stay, x["team"], x["fill"]).total, sc.parts(moved, x["team"], x["fill"]).total
    ps_, pm_ = sc.parts(stay, port, fill_p).total, sc.parts(moved, port, fill_p).total
    eq = sc.normal(b, {"stayed": ps_ - ts_, "moved": pm_ - tm_}, offset=ts_ + tm_, level="home")
    return dict(sw=float(b.w.sum()), ref=sc.error(b, ts_ + tm_), moved=sc.error(b, ts_ + pm_),
                stayed=sc.error(b, ps_ + tm_), both=sc.error(b, ps_ + pm_), eq=eq)


def evaluate(sets, ratio_of) -> tuple:
    """Per development season: the four errors (possession-weighted over the season's sets) and the blend equations."""
    rows, eqs = [], {}
    for x in sets:
        H = x["season"]
        port = centred(pt.portable(x["parts"], ratio_of[H]))
        fill_p = (0.0, 0.0) if x["source"] == "yoy" else ts.replacement_fill(port, max_poss=500.0, shrink=0.25)
        r = score(x, port, fill_p)
        eqs.setdefault(H, []).append(r.pop("eq"))
        rows.append(dict(season=H, **r))
    d = pd.DataFrame(rows)
    per = d.assign(**{c: d[c] * d.sw for c in ("ref", "moved", "stayed", "both")}).groupby("season")[
        ["sw", "ref", "moved", "stayed", "both"]].sum()
    for c in ("ref", "moved", "stayed", "both"):
        per[c] = per[c] / per.sw
    return per, eqs


def blend(eqs: dict) -> tuple:
    """Pooled blend weights (stayed, moved) and their season-jackknife standard errors."""
    seasons = sorted(eqs)
    M = {s: sum(np.column_stack([e.xx, e.xy[:, None]]) for e in eqs[s]) for s in seasons}
    total = sum(M.values())
    coef = np.linalg.lstsq(total[:, :-1], total[:, -1], rcond=None)[0]
    jack = np.array([np.linalg.lstsq((total - M[s])[:, :-1], (total - M[s])[:, -1], rcond=None)[0] for s in seasons])
    k = len(seasons)
    se = np.sqrt((k - 1) / k * ((jack - jack.mean(axis=0)) ** 2).sum(axis=0))
    return coef, se


def main():
    check_flags()
    pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_rows", 200)
    OUT.mkdir(parents=True, exist_ok=True)
    S104 = _borrow("_scorecard104_for_112", "104_scorecard.py")
    B70 = _borrow("_tradeset70_for_112", "70_tradeset.py")
    t0 = time.time()
    sets = {"deadline": load_folds(S104)}
    sets["yoy"] = load_yoy(B70)
    print(f"{len(sets['deadline'])} deadline folds, {len(sets['yoy'])} year-over-year pairs ({time.time() - t0:.0f}s)",
          flush=True)

    # 1. who lands on a new team
    desc = pd.concat([describe(sets["deadline"]), describe(sets["yoy"])])
    desc.to_csv(OUT / "describe.csv", index=False)
    cols = ["players", "rating", "box_part", "beyond_part", "season_poss", "tier0", "tier1", "tier2"]
    print("\n=== players on a new team vs the same team (mean per fold or pair; rating = offense + defense, pts/100)")
    print(desc.groupby(["source", "group"])[cols].mean().round(3).to_string())
    for src in SOURCES:
        x = desc[desc.source == src].pivot_table(index="key", columns="group", values="rating")
        m, t, _, n = paired_t(x["new team"] - x["same team"])
        print(f"  {src}: new team minus same team, mean rating {m:+.3f} (paired t {t:.1f}, n {n})")

    # 2. production ratios per source and variant (every development season), season jackknife
    rows = []
    prod = {}
    for src in SOURCES:
        xx, xy = gram(sets[src], lambda x: True)
        for v in VARIANTS:
            f = pt.fit(xx, xy, v)
            prod[(src, v)] = f
            jk = [pt.fit(*gram(sets[src], lambda x, s=s: s not in x["touches"]), v)["keys"] for s in DEV]
            for key, val in f["keys"].items():
                rows.append(dict(source=src, variant=v, ratio=" ".join(map(str, key)), value=val,
                                 se=jack_se(np.array([j[key] for j in jk]))))
            for key in ("O traded level", "D traded level"):
                rows.append(dict(source=src, variant=v, ratio=key, value=f["levels"][key], se=np.nan))
    ratios = pd.DataFrame(rows)
    ratios.to_csv(OUT / "ratios.csv", index=False)
    print("\n=== ratios (fit on every development season; season jackknife se)")
    print(ratios.round(3).to_string(index=False))

    # role check: each tier's own slopes, players on a new team vs the same team
    trows = []
    for src in SOURCES:
        xx, xy = gram(sets[src], lambda x: True)
        free = pt.free_fit(xx, xy)
        jf = pd.DataFrame([pt.free_fit(*gram(sets[src], lambda x, s=s: s not in x["touches"])) for s in DEV])
        for s in pt.SIDES:
            for c in pt.PARTS:
                for k in pt.TIERS:
                    a, b = f"{s} {c} t{k} stayed", f"{s} {c} t{k} traded"
                    trows.append(dict(source=src, side=s, part=c, tier=["<1,000", "1,000-2,500", "2,500+"][k],
                                      same_team=free[a], same_team_se=jack_se(jf[a].to_numpy()), new_team=free[b],
                                      new_team_se=jack_se(jf[b].to_numpy()), ratio=free[b] / free[a],
                                      ratio_se=jack_se((jf[b] / jf[a]).to_numpy())))
    tiers = pd.DataFrame(trows)
    tiers.to_csv(OUT / "tiers.csv", index=False)
    print("\n=== role check: each playing-time tier's own slopes (season-equivalent possessions)")
    print(tiers.round(3).to_string(index=False))

    # 3. the ablation: every source x variant, cross-fitted, scored on both sets
    arows = []
    for src in SOURCES:
        for v in VARIANTS:
            ratio_of = {H: pt.fit(*gram(sets[src], lambda x, H=H: not (x["touches"] & {H - 1, H, H + 1})), v)["ratio"]
                        for H in DEV}
            row = dict(fit_on=src, variant=v)
            for test in SOURCES:
                per, eqs = evaluate(sets[test], ratio_of)
                for grp in ("moved", "stayed"):
                    m, t, better, n = paired_t(per[grp] - per.ref)
                    row[f"{test}_{grp}_diff"], row[f"{test}_{grp}_t"] = m, t
                    row[f"{test}_{grp}_better"] = f"{better} of {n}"
                coef, se = blend(eqs)
                row[f"{test}_blend_moved"], row[f"{test}_blend_moved_se"] = coef[1], se[1]
                row[f"{test}_blend_stayed"], row[f"{test}_blend_stayed_se"] = coef[0], se[0]
            arows.append(row)
            print(f"  {src} / {v} ({time.time() - t0:.0f}s)", flush=True)
    abl = pd.DataFrame(arows)
    abl.to_csv(OUT / "ablation.csv", index=False)
    show = ["fit_on", "variant", "deadline_moved_t", "deadline_moved_better", "deadline_blend_moved",
            "deadline_blend_moved_se", "yoy_moved_t", "yoy_moved_better", "yoy_blend_moved", "yoy_blend_moved_se",
            "deadline_stayed_t", "yoy_stayed_t"]
    print("\n=== the portable rating given to players on a new team only (cross-fitted; t: paired over 20 seasons, "
          "negative = better; blend weight: 1 = the change is all signal)")
    print(abl[show].round(3).to_string(index=False))

    # 4. production table and the 2026 top 20 for the variant named by --pick (default: one ratio, both samples)
    # "both": each ratio pooled over the two samples by inverse variance (season-jackknife variances)
    pick = tuple(flag("pick", "both,beyond-one").split(","))
    if pick[0] == "both":
        R = {}
        for nm in pt.slope_names():
            key = pt.groups_for(pick[1])[nm]
            if key is None:
                R[nm] = 1.0
                continue
            label = " ".join(map(str, key))
            r = ratios[(ratios.variant == pick[1]) & (ratios.ratio == label)].set_index("source")
            wts = 1.0 / r.se ** 2
            R[nm] = float((r.value * wts).sum() / wts.sum())
        pooled_se = {lab: float(1.0 / np.sqrt((1.0 / g.se ** 2).sum()))
                     for lab, g in ratios[(ratios.variant == pick[1]) & ratios.se.notna()].groupby("ratio")}
        shown = {k: round(v, 3) for k, v in R.items() if "beyond" in k and k.endswith("t0")}
        print("")
        print(f"=== production ratios pooled over both samples ({pick[1]}): {shown}, se {pooled_se}")
    else:
        R = prod[pick]["ratio"]
    # the site (scripts/52_site.py) applies the same single ratio to the PRODUCT table with portable.portable_table
    assert pick[1] == "beyond-one", "the published portable rating is the one-ratio version"
    r_b = R["O beyond t0"]
    se_b = (pooled_se["both beyond"] if pick[0] == "both" else
            float(ratios[(ratios.source == pick[0]) & (ratios.variant == pick[1])].se.iloc[0]))
    (OUT / "production.json").write_text(json.dumps(dict(
        variant=pick[1], fit_on=pick[0], ratio_beyond=r_b, se=se_b, seasons=DEV,
        note="portable = box-prior part + ratio_beyond x (games part + swap adjustment), re-centered per season"),
        indent=1), encoding="utf-8")
    tables = {"season_ratings_portable": "season_ratings_priorshrink",          # the test table (scored above)
              "season_ratings_product_portable": "season_ratings_product"}      # what the site publishes
    for out_name, src in tables.items():
        base = pd.read_parquet(ROOT / "outputs" / f"{src}.parquet")
        tab = base[["player_id", "player_name", "season", "poss_season", "rating_off", "rating_def",
                    "rating_total"]].copy()
        port = pt.portable_table(base, r_b)
        for c in ("portable_off", "portable_def", "portable_total"):
            tab[c] = port[c].to_numpy()
        tab.to_parquet(ROOT / "outputs" / f"{out_name}.parquet", index=False)
        top = tab[tab.season == 2026].copy()
        top["team_rank"] = top.rating_total.rank(ascending=False, method="first").astype(int)
        top["portable_rank"] = top.portable_total.rank(ascending=False, method="first").astype(int)
        top = top.sort_values("portable_total", ascending=False)
        top.to_csv(OUT / f"top2026_{src}.csv", index=False)
        print("")
        print(f"=== 2026 top 20 by the portable rating, {src} (ratio {r_b:.3f})")
        print(top.head(20)[["player_name", "poss_season", "rating_total", "portable_total", "team_rank",
                            "portable_rank"]].round(2).to_string(index=False))
    print("")
    print(f"done ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
