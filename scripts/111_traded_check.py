"""Experiment 39: which pieces of a rating travel with a traded player, and which belong to his team or to the season.

    python scripts/111_traded_check.py [--parts=systems,stats]

The owner (2026-10-05): "an in-season check that boosts the weight of traded players in the scoring ... which pieces
are regression to the mean, and which pieces are 'real things about a player'."

The trade-deadline folds (97 --split=deadline): each season rated from its games before the date by which 60% of the
regular season was played and scored on the rest (playoffs included), and the reverse.  On the scored games every
player-entry is either TRADED (the player is on a team other than his team in the rating games) or STAYED.  Any linear
prediction splits exactly into the two (scorecard.split_traded), so a change to the rating can be handed to one group
at a time with everyone else held at the reference:

    stayed   the change given only to players who stayed: same team, same season, other games
    traded   the change given only to traded players: the rating from his old team, scored on his new one

Weighting whole team-game rows by "has a traded player" instead would mostly score the nine other players on the
court; swapping only the traded players' ratings isolates them.  Per change, two statistics per group:

    error change    the held-out team-game error with the change given to that group only, minus the reference's;
                    paired by season over the development seasons (t on 19-20 seasons)
    blend weight    the coefficient of the reference's miss on that group's part of the change (forecast
                    encompassing): 1 = the change is all signal for that group, 0 = none; season-clustered jackknife

A piece that helps players who stayed but not traded players is about the team (lineups, scheme, teammates); one
that helps both is about the player.  Experiment 38's year-over-year effects say whether it lasts into the next
season.

  systems   per rating system, the calibration slope on traded vs stayed players (per side), the same for the
            prior part and the games part, and the model decisions as hybrids (fold data, outputs/within/deadline)
  stats     every input (and input group) added alone to RAPM on the linear box prior, its shape fit on the random
            within-season folds exactly as in experiment 38, scored on the split held-out quadratics
            (107 --tags=deadline --traded=1)

Development seasons only (heldoutprior.LOCKBOX untouched).  Writes outputs/heldout/traded_check_*.csv.
"""
import importlib.util
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _cli import check_flags, flag  # noqa: E402
from eracoef import heldoutprior as hp  # noqa: E402
from eracoef import scorecard as sc  # noqa: E402

OUT = ROOT / "outputs" / "heldout"
TAG = "deadline"
DEV = [s for s in range(1997, 2027) if s not in hp.LOCKBOX]
SYSTEMS = ("B1", "P", "B2", "B3", "B4", "O13", "O", "O+swap0.5", "B2:team-mean", "O:team-mean")
NAMES = {"B1": "linear box score alone", "P": "OpenRAPM's boosted prior alone", "B2": "plain RAPM (tuned)",
         "B3": "RAPM on the linear box prior", "B4": "RAPM on the boosted prior", "O13": "OpenRAPM before the prior shrink",
         "O": "OpenRAPM, prior shrink", "O+swap0.5": "OpenRAPM as shipped (with the swap adjustment)",
         "B2:team-mean": "control: plain RAPM's team average for everyone",
         "O:team-mean": "control: OpenRAPM's team average for everyone"}
DECISIONS = [("add a linear box-score prior to vanilla RAPM", "B2", "B3"),
             ("boosted (OpenRAPM) prior instead of the linear one", "B3", "B4"),
             ("OpenRAPM's prior shrink", "O13", "O"),
             ("the swap adjustment (OpenRAPM, x0.5)", "O", "O+swap0.5"),
             ("OpenRAPM as shipped instead of RAPM on the linear box prior", "B3", "O+swap0.5")]


def _borrow(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def paired_t(d) -> tuple:
    d = pd.Series(d).dropna()
    se = d.std(ddof=1) / np.sqrt(len(d))
    return float(d.mean()), (float(d.mean() / se) if se > 0 else np.nan), int((d < 0).sum()), len(d)


def jackknife(blocks: dict, fn) -> tuple:
    """fn(sum of the blocks) and its leave-one-season-out jackknife standard error; `blocks`: season -> array(s)."""
    seasons = sorted(blocks)
    total = sum(blocks[s] for s in seasons)
    value = np.asarray(fn(total), dtype=float)
    jack = np.array([np.asarray(fn(total - blocks[s]), dtype=float) for s in seasons])
    k = len(seasons)
    return value, np.sqrt((k - 1) / k * ((jack - jack.mean(axis=0)) ** 2).sum(axis=0))


def blend(Q: np.ndarray) -> np.ndarray:
    """(stayed, traded) blend weights of an increment gram."""
    return np.linalg.solve(Q[1:, 1:], Q[1:, 0])


def armse(mse):
    return np.sqrt(mse) * np.sqrt(2 / np.pi)


# ------------------------------------------------------------------------------------------ part 1: systems
def systems_part(direction: str = "both", seasons=None) -> dict:
    """`direction` "f0": rated on the games before the cut, scored after; "f1": rated after the cut, scored before --
    the trade was decided on the early games, so in "f1" the rating games carry no selection on the trade."""
    S104 = _borrow("_scorecard104_for_111", "104_scorecard.py")
    seasons = DEV if seasons is None else [s for s in seasons if s in DEV]
    directory = ROOT / "outputs" / "within" / TAG
    tables = {}
    for name, file in (("B1", "baseline_b1"), ("B2", "baseline_b2"), ("B3", "baseline_b3"), ("B4", "baseline_b4")):
        tables[name] = pd.read_parquet(directory / f"{file}.parquet")
    for variant, g in pd.read_parquet(directory / "openrapm_shipped.parquet").groupby("variant"):
        tables["O" if variant == "shrunk" else f"O+{variant}"] = g
    rng = np.random.default_rng(3901)
    exposure, slopes, parts, dec_rows, dec_eqs = [], {}, {}, [], {}
    t0 = time.time()
    for stem in S104.fold_stems(directory):
        if direction != "both" and not stem.endswith(direction):
            continue
        f = S104.load_fold(directory, stem)
        b = f["block"]
        if b.season not in seasons:
            continue
        systems, fills, parts_of = S104.within_systems(stem, f, tables, rng)
        team_of = f["players"].drop_duplicates("player_id").set_index("player_id").team_id
        Zt = sc.traded_entries(b, team_of[team_of >= 0], *f["teams"])
        stay, traded = sc.split_traded(b, Zt)
        exposure.append(dict(season=b.season, key=stem, traded_share=float((Zt.T @ b.w).sum() / (b.Z.T @ b.w).sum()),
                             traded_players=len(np.unique(Zt.tocoo().col % b.n_players)),
                             players=len(np.unique(b.Z.tocoo().col % b.n_players))))
        split = {}
        for name in [n for n in SYSTEMS if n in systems]:
            ps, pt = sc.parts(stay, systems[name], fills[name]), sc.parts(traded, systems[name], fills[name])
            split[name] = (ps, pt)
            eq = sc.normal(b, {"O stayed": ps.c_o, "D stayed": ps.c_d, "O traded": pt.c_o, "D traded": pt.c_d},
                           offset=ps.s_o + ps.s_d + pt.s_o + pt.s_d, level="full")
            slopes.setdefault(name, {}).setdefault(b.season, []).append(eq)
        for name, (prior, games) in parts_of.items():
            whole = sc.parts(b, systems[name], fills[name])
            cols = {}
            for part, table in (("prior", prior), ("games", games)):
                for side, col in (("O", "o"), ("D", "d")):
                    cols[f"{side} {part} stayed"] = sc.side_column(stay, table, col, side)
                    cols[f"{side} {part} traded"] = sc.side_column(traded, table, col, side)
            parts.setdefault(name, {}).setdefault(b.season, []).append(
                sc.normal(b, cols, offset=whole.s_o + whole.s_d, level="full"))
        for label, a, c in DECISIONS:
            if a not in split or c not in split:
                continue
            (as_, at), (cs, ct) = split[a], split[c]
            pa_s, pa_t, pc_s, pc_t = as_.total, at.total, cs.total, ct.total
            dec_rows.append(dict(decision=label, season=b.season, key=stem, sw=float(b.w.sum()),
                                 ref=sc.error(b, pa_s + pa_t), stayed=sc.error(b, pc_s + pa_t),
                                 traded=sc.error(b, pa_s + pc_t), both=sc.error(b, pc_s + pc_t)))
            eq = sc.normal(b, {"stayed": pc_s - pa_s, "traded": pc_t - pa_t}, offset=pa_s + pa_t, level="home")
            dec_eqs.setdefault(label, {}).setdefault(b.season, []).append(eq)
    print(f"  systems: {len(exposure)} development folds ({time.time() - t0:.0f}s)", flush=True)

    def summed(per_season):
        """season -> [xx | xy] summed over the season's folds."""
        return {s: sum(np.column_stack([e.xx, e.xy[:, None]]) for e in eqs) for s, eqs in per_season.items()}

    def coef(M):
        return np.linalg.solve(M[:, :-1], M[:, -1])

    rows = []
    for name, per in slopes.items():
        names = per[next(iter(per))][0].names
        val, se = jackknife(summed(per), coef)
        r = dict(system=name, description=NAMES[name])
        for side in ("O", "D"):
            js, jt = names.index(f"{side} stayed"), names.index(f"{side} traded")
            r[f"{side}_stayed"], r[f"{side}_stayed_se"] = val[js], se[js]
            r[f"{side}_traded"], r[f"{side}_traded_se"] = val[jt], se[jt]
            d, dse = jackknife(summed(per), lambda M, js=js, jt=jt: coef(M)[jt] - coef(M)[js])
            r[f"{side}_traded_minus_stayed"], r[f"{side}_diff_se"] = float(d), float(dse)
        rows.append(r)
    slope_table = pd.DataFrame(rows)

    rows = []
    for name, per in parts.items():
        names = per[next(iter(per))][0].names
        val, se = jackknife(summed(per), coef)
        for part in ("prior", "games"):
            r = dict(system=name, part=part)
            for side in ("O", "D"):
                js, jt = names.index(f"{side} {part} stayed"), names.index(f"{side} {part} traded")
                r[f"{side}_stayed"], r[f"{side}_stayed_se"] = val[js], se[js]
                r[f"{side}_traded"], r[f"{side}_traded_se"] = val[jt], se[jt]
                d, dse = jackknife(summed(per), lambda M, js=js, jt=jt: coef(M)[jt] - coef(M)[js])
                r[f"{side}_traded_minus_stayed"], r[f"{side}_diff_se"] = float(d), float(dse)
            rows.append(r)
    part_table = pd.DataFrame(rows)

    d = pd.DataFrame(dec_rows)
    per_season = (d.assign(**{c: d[c] * d.sw for c in ("ref", "stayed", "traded", "both")})
                  .groupby(["decision", "season"])[["sw", "ref", "stayed", "traded", "both"]].sum())
    for c in ("ref", "stayed", "traded", "both"):
        per_season[c] = per_season[c] / per_season.sw
    rows = []
    for label, a, c in DECISIONS:
        ps = per_season.loc[label]
        r = dict(decision=label, reference=a, candidate=c)
        for grp in ("stayed", "traded", "both"):
            r[f"{grp}_diff"], r[f"{grp}_t"], r[f"{grp}_better"], r[f"{grp}_n"] = paired_t(ps[grp] - ps.ref)
        val, se = jackknife(summed(dec_eqs[label]), coef)
        r["blend_stayed"], r["blend_stayed_se"], r["blend_traded"], r["blend_traded_se"] = val[0], se[0], val[1], se[1]
        dd, dse = jackknife(summed(dec_eqs[label]), lambda M: coef(M)[1] - coef(M)[0])
        r["blend_traded_minus_stayed"], r["blend_diff_se"] = float(dd), float(dse)
        rows.append(r)
    dec_table = pd.DataFrame(rows)
    # experiment 38's year-over-year effect of the same decisions
    x38 = pd.read_csv(OUT / "in_vs_out_decisions.csv").set_index("decision")

    def parse(s):
        m = re.match(r"([+-]?[\d.]+) \(t ([+-]?[\d.]+)", str(s))
        return (float(m.group(1)), float(m.group(2))) if m else (np.nan, np.nan)

    dec_table["yoy_diff"] = [parse(x38.out_of_season.get(l, ""))[0] for l in dec_table.decision]
    dec_table["yoy_t"] = [parse(x38.out_of_season.get(l, ""))[1] for l in dec_table.decision]
    return dict(exposure=pd.DataFrame(exposure), slopes=slope_table, parts=part_table, decisions=dec_table)


# ------------------------------------------------------------------------------------------ part 2: stats
def stats_part() -> pd.DataFrame:
    S108 = _borrow("_select108_for_111", "108_heldout_prior_select.py")
    D = S108.Data()
    z = np.load(OUT / f"quad_{TAG}_traded.npz")
    Gs_all, sw_all = z["G"], z["sw"]
    K = 2 * D.k1
    assert Gs_all.shape[-1] == 1 + 2 * K, f"split gram {Gs_all.shape} for K = {K}"
    seasons = [int(s) for s in z["seasons"]]
    # the split build agrees with the plain one (107 run twice, two code paths)
    plain = np.load(OUT / f"quad_{TAG}.npz")
    for si in (seasons.index(1997), seasons.index(2016)):
        for g in (0, len(D.grid) // 2):
            P = plain["G"][si, g, list(plain["levels"]).index("home")]
            assert np.allclose(hp.collapse_split(Gs_all[si, g], K), P, rtol=1e-9, atol=1e-6 * abs(P[0, 0])), (si, g)
    ref_idx, ref_mask = D.inputs(S108.Config("B3 (anchor only)", []))

    def fit_ref_at(g, train):
        total = sum(sum(D.within_norm(tag, [s], g, "home", ref_idx) for tag in S108.FIT_SIZES) for s in train)
        return hp.fit(total, 1.0, np.zeros(len(ref_idx)))

    def run(cfg) -> dict:
        idx, mask = D.inputs(cfg)
        assert list(idx[:2]) == list(ref_idx), "the anchor comes first"
        Qs = {}
        for H in D.dev:
            train = [s for s in D.dev if abs(s - H) > 1]
            g, rho, beta = S108.fit_shape(D, cfg, train, idx, mask)
            beta_ref = np.r_[fit_ref_at(g, train), np.zeros(len(idx) - len(ref_idx))]
            si = seasons.index(H)
            Qs[H] = (hp.increment_gram(hp.split_subset(Gs_all[si, g], idx, K), beta_ref, beta), sw_all[si])
        return Qs

    def err(Q, sw, x):
        x = np.asarray(x, dtype=float)
        return (Q[0, 0] - 2 * x @ Q[1:, 0] + x @ Q[1:, 1:] @ x) / sw

    configs = [(c, S108.Config(f"+{c}", columns=[c])) for c in D.columns if c != "b1"]
    configs += [(f"group: {g}", S108.Config(f"+group {g}", [g])) for g in S108.ALL_GROUPS]
    configs += [("all inputs", S108.Config("all inputs", S108.ALL_GROUPS)),
                ("all inputs except plus-minus", S108.Config("all inputs except plus-minus",
                                                             [g for g in S108.ALL_GROUPS if g not in S108.PLUS_MINUS]))]
    out = []
    t0 = time.time()
    for label, cfg in configs:
        Qs = run(cfg)
        row = {"stat": label}
        per = pd.DataFrame({H: {"ref": err(Q, sw, [0, 0]), "stayed": err(Q, sw, [1, 0]), "traded": err(Q, sw, [0, 1]),
                                "both": err(Q, sw, [1, 1])} for H, (Q, sw) in Qs.items()}).T
        for grp in ("stayed", "traded", "both"):
            row[f"{grp}_diff"], row[f"{grp}_t"], row[f"{grp}_better"], row[f"{grp}_n"] = paired_t(per[grp] - per.ref)
            row[f"{grp}_pts"] = float((armse(per[grp]) - armse(per.ref)).mean())
        blocks = {H: Q for H, (Q, sw) in Qs.items()}
        val, se = jackknife(blocks, blend)
        row["blend_stayed"], row["blend_stayed_se"], row["blend_traded"], row["blend_traded_se"] = val[0], se[0], val[1], se[1]
        d, dse = jackknife(blocks, lambda Q: blend(Q)[1] - blend(Q)[0])
        row["blend_traded_minus_stayed"], row["blend_diff_se"] = float(d), float(dse)
        out.append(row)
    print(f"  stats: {len(out)} inputs and groups ({time.time() - t0:.0f}s)", flush=True)
    st = pd.DataFrame(out)
    # experiment 38's within-season (random folds) and year-over-year effects of the same single inputs
    x38 = pd.read_csv(OUT / "in_vs_out_stats.csv")[["stat", "name", "group", "within_diff", "within_t", "yoy_diff",
                                                     "yoy_t"]]
    st = st.merge(x38, on="stat", how="left")
    # groups: experiment 37's ledger, the same procedure
    led = pd.read_parquet(OUT / "ledger.parquet")
    led = led[led.season.isin(DEV)].assign(yoy=lambda x: x[["yoy_prev", "yoy_next"]].mean(axis=1),
                                           within=lambda x: x[["q2of3_home", "within_home", "within10_home"]].mean(axis=1))
    ref = led[led.config == "B3 (anchor only)"].set_index("season")
    for label, name in [(f"group: {g}", f"B3 (anchor only) + {g}") for g in S108.ALL_GROUPS] + [("all inputs", "all inputs")]:
        r = led[led.config == name].set_index("season")
        if not len(r):
            continue
        i = st.index[st.stat == label][0]
        for metric in ("within", "yoy"):
            dd, tt, _, _ = paired_t(r[metric] - ref[metric])
            st.loc[i, f"{metric}_diff"], st.loc[i, f"{metric}_t"] = dd, tt
        st.loc[i, "group"] = label.replace("group: ", "")
    return st


def main():
    check_flags()
    pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_rows", 200)
    do = [p for p in flag("parts", "systems,stats").split(",") if p]
    direction = flag("direction", "both")
    era = flag("seasons", "")
    era_seasons = None
    if era:
        a, b = (int(x) for x in era.split("-"))
        era_seasons = list(range(a, b + 1))
    suffix = ("" if direction == "both" else f"_{direction}") + (f"_{era}" if era else "")
    if "systems" in do:
        r = systems_part(direction, era_seasons)
        for k, v in r.items():
            v.to_csv(OUT / f"traded_check_{k}{suffix}.csv", index=False)
        e = r["exposure"]
        print(f"\ntraded players' share of the scored player-possessions: {e.traded_share.mean():.3f} "
              f"(range {e.traded_share.min():.3f}-{e.traded_share.max():.3f}); traded players per fold "
              f"{e.traded_players.mean():.1f} of {e.players.mean():.0f}")
        print("\n=== calibration slope, traded vs stayed players (pooled over development seasons; jackknife se)")
        print(r["slopes"].round(3).to_string(index=False))
        print("\n=== the same for the prior part and the games part")
        print(r["parts"].round(3).to_string(index=False))
        print("\n=== model decisions given to one group at a time (error change in MSE, paired t; blend weights)")
        print(r["decisions"].round(3).to_string(index=False))
    if "stats" in do:
        st = stats_part()
        st.to_csv(OUT / "traded_check_stats.csv", index=False)
        cols = ["stat", "stayed_pts", "stayed_t", "traded_pts", "traded_t", "blend_stayed", "blend_traded",
                "blend_traded_minus_stayed", "blend_diff_se", "yoy_t", "within_t"]
        print("\n=== each input added alone to RAPM on the linear box prior (negative = better)")
        print(st.sort_values("stayed_t")[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
