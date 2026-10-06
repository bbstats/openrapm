"""Experiment 35's scorecard: error and slope of every rating system on held-out team-games (src/eracoef/scorecard.py).

    python scripts/104_scorecard.py --mode=reproduce [--tags=q1of4,q1of3,within2,q2of3,within] [--yoy=shotmix_pin]

`--mode=reproduce` proves the loaders and the scorecard's error are the ones the project already scores with, before
any new number is read:

  * within-season: every fold's stored error (`tg`) and no-ratings error (`tg_base`) in outputs/within/<tag>/fold_*.json
    (scripts/97_within_season.py, `test_games`), recomputed from the saved held-out team-games and the fold's own
    OpenRAPM ratings and stand-in -- to 1e-8;
  * year-over-year: the incumbent's per-season error in outputs/yoy_<yoy>.parquet (scripts/63_yoy.py), recomputed from
    whole-season team-game designs and the neighbouring season's ratings, unrated players at 0 as 63 scores them.

Loaders live here (data layer); the arithmetic lives in the model layer.
"""
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
from eracoef import scorecard as sc  # noqa: E402
from eracoef import tradeset as ts  # noqa: E402

SIZES = ["q1of4", "q1of3", "within2", "q2of3", "within", "within10"]
MOVED_TAGS = {"deadline"}     # experiment 39: also score rows with players on a new team, and the rest, separately
SHARE = {"q1of4": 0.25, "q1of3": 1 / 3, "within2": 0.5, "q2of3": 2 / 3, "within": 0.75, "within10": 0.9}


# ------------------------------------------------------------------------------------------ within-season folds
def fold_stems(directory: Path) -> list:
    return sorted(p.stem[len("fold_"):] for p in directory.glob("fold_*.json"))


def load_fold(directory: Path, stem: str) -> dict:
    """One fold: its held-out team-games as a Block, its OpenRAPM players (raw-sign o/d for rated players), its
    stand-in and stored scores."""
    info = json.loads((directory / f"fold_{stem}.json").read_text())
    summary = info["summary"]
    t = np.load(directory / f"test_{stem}.npz", allow_pickle=False)
    Z = sp.csr_matrix((t["Z_data"], t["Z_indices"], t["Z_indptr"]), shape=tuple(t["Z_shape"]))
    names = [str(c) for c in t["control_names"]]
    F = np.asarray(t["F"], dtype=float)
    block = sc.Block(key=stem, season=int(summary["season"]), deal=int(summary["repeat"]), Z=Z,
                     y=t["y"].astype(float), w=t["w"].astype(float), F=F, home=F[:, names.index("home")],
                     player_ids=t["player_ids"].astype(np.int64))
    players = pd.read_parquet(directory / f"players_{stem}.parquet")
    rated = players.drop_duplicates("player_id")
    rated = rated[rated.poss_off > 0]
    ratings = pd.DataFrame({"player_id": rated.player_id.astype(np.int64), "o": rated.rating_off.astype(float),
                            "d": -rated.rating_def.astype(float), "poss": rated.poss_off.astype(float)})
    return dict(block=block, ratings=ratings, players=players, summary=summary,
                fill=(float(summary["fill_o"]), float(summary["fill_d"])),
                teams=(t["team_off"].astype(np.int64), t["team_def"].astype(np.int64)))


def reproduce_folds(tags) -> pd.DataFrame:
    rows = []
    for tag in tags:
        directory = ROOT / "outputs" / "within" / tag
        t0 = time.time()
        for stem in fold_stems(directory):
            f = load_fold(directory, stem)
            b = f["block"]
            refill = ts.replacement_fill(f["ratings"], max_poss=500.0, shrink=0.25)
            p = sc.parts(b, f["ratings"], f["fill"])
            rows.append(dict(tag=tag, key=stem, season=b.season,
                             tg=sc.error(b, p.total), tg_stored=float(f["summary"]["tg"]),
                             tg_base=sc.error(b, None), tg_base_stored=float(f["summary"]["tg_base"]),
                             fill_gap=max(abs(refill[0] - f["fill"][0]), abs(refill[1] - f["fill"][1]))))
        print(f"  {tag}: {sum(r['tag'] == tag for r in rows)} folds ({time.time() - t0:.0f}s)", flush=True)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------ year-over-year
def yoy_block(ctx, season: int) -> sc.Block:
    """The whole season's team-games (regular season and playoffs), pooled the trade set's way."""
    pooled = ts.team_game_design(ctx.design([season], "pts"))
    names = list(pooled.control_names)
    F = np.asarray(pooled.F, dtype=float)
    return sc.Block(key=str(season), season=season, deal=0, Z=sp.csr_matrix(pooled.Z), y=pooled.y.astype(float),
                    w=pooled.w.astype(float), F=F, home=F[:, names.index("home")],
                    player_ids=pooled.player_ids.astype(np.int64))


def rating_table(path: Path) -> pd.DataFrame:
    """A rankings table in raw sign (63's `load_table`)."""
    t = pd.read_parquet(path)
    return pd.DataFrame({"player_id": t.player_id.astype(np.int64), "season": t.season.astype(int),
                         "o": t.rating_off.astype(float), "d": -t.rating_def.astype(float),
                         "poss": t.poss_off.astype(float)})


def reproduce_yoy(yoy_tag: str, table_path: Path) -> pd.DataFrame:
    from eracoef.config import load_config
    from eracoef.holdout import Context
    stored = pd.read_parquet(ROOT / "outputs" / f"yoy_{yoy_tag}.parquet")
    stored = stored[(stored.split == "all") & (stored.group == "all") & stored.system.str.startswith("incumbent:")]
    ctx = Context.load(load_config(ROOT / "config.yaml"))
    table = rating_table(table_path)
    rows = []
    for season in sorted(stored.held_out.unique()):
        b = yoy_block(ctx, int(season))
        for direction, offset in (("prev", -1), ("next", 1)):
            r = table[table.season == season + offset]
            got = sc.error(b, sc.parts(b, r, (0.0, 0.0)).total)
            want = stored[(stored.held_out == season) & (stored.system == f"incumbent:{direction}")].tg
            rows.append(dict(season=int(season), direction=direction, tg=got,
                             tg_stored=float(want.iloc[0]) if len(want) else np.nan))
    return pd.DataFrame(rows)


def armse(mse) -> float:
    return float(np.sqrt(mse) * np.sqrt(2 / np.pi))


# ------------------------------------------------------------------------------------------ scoring
ENCOMPASS = [("B2", "O"), ("B3", "O"), ("B2", "B3"), ("B1", "P"), ("O13", "O"), ("O", "O+swap0.5"),
             ("B2", "incumbent"), ("B3", "incumbent"),
             ("B3", "B4"), ("B4", "O"), ("B2", "B4"), ("B4", "incumbent")]       # experiment 36: B4, RAPM on OpenRAPM's prior
TIER_EDGES = (1000.0, 2500.0)                     # full-season-equivalent possessions: <1,000 / 1,000-2,500 / 2,500+


def raw_sign(frame: pd.DataFrame, o: str, d: str, poss: str = "poss") -> pd.DataFrame:
    return pd.DataFrame({"player_id": frame.player_id.astype(np.int64).to_numpy(), "o": frame[o].astype(float).to_numpy(),
                         "d": frame[d].astype(float).to_numpy(), "poss": frame[poss].astype(float).to_numpy()})


def centred(t: pd.DataFrame) -> pd.DataFrame:
    w = t.poss.to_numpy(float)
    return t.assign(o=t.o - np.average(t.o, weights=w), d=t.d - np.average(t.d, weights=w))


def tier_columns(block, ratings: pd.DataFrame, scale: float) -> dict:
    """Each side's rated contribution split by the player's possession tier (full-season equivalent)."""
    tier = np.digitize(ratings.poss.to_numpy(float) * scale, TIER_EDGES)
    cols = {}
    for k, name in enumerate(("lt1000", "1000-2500", "2500+")):
        sub = ratings.assign(o=np.where(tier == k, ratings.o, 0.0), d=np.where(tier == k, ratings.d, 0.0))
        cols[f"O {name}"] = sc.side_column(block, sub, "o", "O")
        cols[f"D {name}"] = sc.side_column(block, sub, "d", "D")
    return cols


def score_block(test, block, systems: dict, fills: dict, tier_scale: float, parts_of: dict,
                weightings: dict | None = None) -> tuple:
    """Every system on one block: its quadratic (error, slopes, rescale), its full-level side equations, tier and
    part equations for some, and the blend-weight equations for the ENCOMPASS pairs."""
    quads, normals, cache = [], [], {}
    meta = dict(test=test, key=block.key, season=block.season, deal=block.deal)
    sw = float(block.w.sum())
    quads.append(dict(**meta, system="B0", sw=sw, g00=sc.error(block, None) * sw, g0o=0.0, g0d=0.0, goo=1.0,
                      god=0.0, gdd=1.0))
    for name, ratings in systems.items():
        p = sc.parts(block, ratings, fills[name])
        cache[name] = p
        quads.append(dict(**meta, system=name, **sc.quadratic(block, p)))
        for wname, wv in (weightings or {}).items():
            # the same residuals scored with re-weighted rows (experiment 39: rows with traded players / the rest)
            quads.append(dict(**meta, system=name, **sc.quadratic(block, p, score_weights=wv)) | {"test": f"{test}_{wname}"})
        normals.append(dict(**meta, system=name, kind="full", eq=sc.side_normal(block, p, level="full")))
        if name in ("O13", "O", "B2", "B3", "B4", "B1", "incumbent"):
            normals.append(dict(**meta, system=name, kind="tier",
                                eq=sc.normal(block, tier_columns(block, ratings, tier_scale), offset=p.s_o + p.s_d,
                                             level="full")))
    for name, (prior, games) in parts_of.items():
        p = cache[name]
        cols = {"O prior": sc.side_column(block, prior, "o", "O"), "O games": sc.side_column(block, games, "o", "O"),
                "D prior": sc.side_column(block, prior, "d", "D"), "D games": sc.side_column(block, games, "d", "D")}
        normals.append(dict(**meta, system=name, kind="part",
                            eq=sc.normal(block, cols, offset=p.s_o + p.s_d, level="full")))
    for a, b in ENCOMPASS:
        if a in cache and b in cache:
            normals.append(dict(**meta, system=f"{b} vs {a}", kind="enc",
                                eq=sc.encompass_normal(block, cache[a], cache[b], level="home", by_side=True)))
            normals.append(dict(**meta, system=f"{b} vs {a}", kind="enc_total",
                                eq=sc.encompass_normal(block, cache[a], cache[b], level="home", by_side=False)))
    return quads, normals


def controls(systems: dict, bases, team: pd.Series, weight: pd.Series, rng) -> None:
    """Add the team-mean, within-team shuffle and league-wide shuffle controls of each base system."""
    for base in bases:
        if base in systems:
            systems[f"{base}:team-mean"] = sc.team_mean_control(systems[base], team, weight)
            systems[f"{base}:team-shuffle"] = sc.shuffle_within_team(systems[base], team, weight, rng)
            systems[f"{base}:league-shuffle"] = sc.shuffle_within_bins(systems[base], weight, rng, n_bins=10)


def within_systems(stem: str, f: dict, tables: dict, rng) -> tuple:
    """name -> raw-sign ratings for one fold, their stand-ins, and the prior/games parts for O13 and B3."""
    players = f["players"].drop_duplicates("player_id")
    players = players[players.poss_off > 0]
    systems = {"O13": f["ratings"]}
    fills = {"O13": f["fill"]}
    # the box prior alone; a player the prior has no value for (a missing input) is unrated, scored at the stand-in
    has_prior = players[players.prior_raw_off.notna() & players.prior_raw_def.notna()]
    systems["P"] = centred(raw_sign(has_prior.assign(pd_=-has_prior.prior_raw_def), "prior_raw_off", "pd_", "poss_off"))
    for name, table in tables.items():
        sub = table[table.key == stem]
        if len(sub):
            systems[name] = sub[["player_id", "o", "d", "poss"]].reset_index(drop=True)
    team = players.set_index("player_id").team_id
    controls(systems, ("O", "B2"), team[team >= 0], players.set_index("player_id").poss_off, rng)
    for name, r in systems.items():
        if name not in fills:
            fills[name] = ts.replacement_fill(r, max_poss=500.0, shrink=0.25)
    parts_of = {"O13": (raw_sign(players.assign(pd_=-players.prior_def), "prior_off", "pd_", "poss_off"),
                        raw_sign(players.assign(ud_=-players.u_def), "u_off", "ud_", "poss_off"))}
    for name in ("B3", "B4"):
        if name in tables:
            b = tables[name][tables[name].key == stem]
            if len(b):
                parts_of[name] = (raw_sign(b, "prior_o", "prior_d"),
                                  raw_sign(b.assign(go=b.o - b.prior_o, gd=b.d - b.prior_d), "go", "gd"))
    return systems, fills, parts_of


def score_within(tags) -> tuple:
    quads, normals = [], []
    rng = np.random.default_rng(3501)
    for tag in tags:
        directory = ROOT / "outputs" / "within" / tag
        tables = {}
        for name, file in (("B1", "baseline_b1"), ("B2", "baseline_b2"), ("B2g", "baseline_b2g"), ("B3", "baseline_b3"),
                           ("B4", "baseline_b4")):
            if (directory / f"{file}.parquet").exists():
                tables[name] = pd.read_parquet(directory / f"{file}.parquet")
        shipped = directory / "openrapm_shipped.parquet"
        if shipped.exists():
            for variant, g in pd.read_parquet(shipped).groupby("variant"):
                tables["O" if variant == "shrunk" else f"O+{variant}"] = g
        t0 = time.time()
        for stem in fold_stems(directory):
            f = load_fold(directory, stem)
            systems, fills, parts_of = within_systems(stem, f, tables, rng)
            s = f["summary"]
            share = float(s["fit_games"]) / (float(s["fit_games"]) + float(s["test_games"]))
            weightings = None
            if tag in MOVED_TAGS:
                b = f["block"]
                team_of = f["players"].drop_duplicates("player_id").set_index("player_id").team_id
                moved = sc.moved_share(b, team_of[team_of >= 0], *f["teams"])
                weightings = {"moved": b.w * moved, "stayed": b.w * (1.0 - moved)}
                for wname, wv in weightings.items():
                    z = np.zeros(len(b.y))
                    q0 = sc.quadratic(b, sc.Parts(z, z, z, z), score_weights=wv)
                    quads.append(dict(test=f"{tag}_{wname}", key=b.key, season=b.season, deal=b.deal, system="B0",
                                      sw=q0["sw"], g00=q0["g00"], g0o=0.0, g0d=0.0, goo=1.0, god=0.0, gdd=1.0))
            q, n = score_block(tag, f["block"], systems, fills, 1.0 / share, parts_of, weightings)
            quads += q
            normals += n
        print(f"  scored {tag} ({time.time() - t0:.0f}s)", flush=True)
    return quads, normals


def score_yoy() -> tuple:
    from eracoef.config import load_config
    from eracoef.holdout import Context
    ctx = Context.load(load_config(ROOT / "config.yaml"))
    paths = {"incumbent": "season_ratings_priorshrink", "O13": "season_ratings_unshrinkdef",
             "O": "season_ratings_priorshrink_raw", "B1": "season_ratings_base_b1", "B2": "season_ratings_base_b2",
             "B2g": "season_ratings_base_b2g", "B3": "season_ratings_base_b3", "B4": "season_ratings_base_b4"}
    tables = {k: rating_table(ROOT / "outputs" / f"{v}.parquet") for k, v in paths.items()
              if (ROOT / "outputs" / f"{v}.parquet").exists()}
    o13 = pd.read_parquet(ROOT / "outputs" / "season_ratings_unshrinkdef.parquet")
    roles = pd.read_parquet(ROOT / "data/cache/roles_RSPO.parquet", columns=["player_id", "season", "team_id", "poss_on"])
    teams = (roles[roles.poss_on > 0].groupby(["player_id", "season", "team_id"], as_index=False).poss_on.sum()
             .sort_values("poss_on").drop_duplicates(["player_id", "season"], keep="last"))
    rng = np.random.default_rng(3502)
    quads, normals = [], []
    for T in range(1998, 2026):
        block0 = yoy_block(ctx, T)
        for direction, offset, deal in (("prev", -1, 0), ("next", 1, 1)):
            S = T + offset
            block = sc.Block(key=f"{T}_{direction}", season=T, deal=deal, Z=block0.Z, y=block0.y, w=block0.w,
                             F=block0.F, home=block0.home, player_ids=block0.player_ids)
            systems = {k: v[v.season == S].reset_index(drop=True) for k, v in tables.items()}
            weight = systems["incumbent"].set_index("player_id").poss
            controls(systems, ("incumbent", "B2"), teams[teams.season == S].set_index("player_id").team_id, weight, rng)
            fills = {k: (0.0, 0.0) for k in systems}
            o = o13[o13.season == S]
            parts_of = {"O13": (raw_sign(o.assign(pd_=-o.prior_def), "prior_off", "pd_", "poss_off"),
                                raw_sign(o.assign(ud_=-o.u_def), "u_off", "ud_", "poss_off"))}
            q, n = score_block("yoy", block, systems, fills, 1.0, parts_of)
            quads += q
            normals += n
        print(f"  scored year-over-year {T}", flush=True)
    return quads, normals


REFERENCE = {"yoy": "incumbent"}               # the paired comparisons' reference; within-season: O (shipped minus swaps)
SHOW = ["B0", "B1", "P", "B2", "B2g", "B3", "B4", "O13", "O", "O+swap0.25", "O+swap0.5", "O+swap0.75", "O+swap1",
        "incumbent", "O:team-mean", "O:team-shuffle", "O:league-shuffle", "B2:team-mean", "B2:team-shuffle",
        "B2:league-shuffle", "incumbent:team-mean", "incumbent:team-shuffle", "incumbent:league-shuffle"]


def _pooled_mse(g: pd.DataFrame, values) -> float:
    return float((values * g.sw).sum() / g.sw.sum())


def summarise(out: Path) -> None:
    q = pd.read_parquet(out / "quadratics.parquet")
    q["mse"] = sc.error_at(q, 1.0, 1.0)
    normals = pd.DataFrame(pd.read_pickle(out / "normals.pkl"))
    bad = q[~np.isfinite(q[["sw", "g00", "g0o", "g0d", "goo", "god", "gdd"]]).all(axis=1)]
    assert bad.empty, f"non-finite scores (pandas would skip them silently):\n{bad[['test', 'key', 'system']].head()}"
    finite = normals["eq"].map(lambda e: bool(np.isfinite(e.xx).all() and np.isfinite(e.xy).all()))
    assert finite.all(), f"non-finite equations:\n{normals.loc[~finite, ['test', 'key', 'system', 'kind']].head()}"
    rows, blend, tiers, parts, variance = [], [], [], [], []
    for test, qt in q.groupby("test", sort=False):
        ref = REFERENCE.get(test, "O")
        win = 2 if test == "yoy" else 1
        base_mse = _pooled_mse(qt[qt.system == "B0"], qt[qt.system == "B0"].mse)
        nt = normals[normals.test == test]
        frame = qt[["season", "key", "system", "mse"]]
        for system, g in qt.groupby("system"):
            mse = _pooled_mse(g, g.mse)
            row = dict(test=test, system=system, error=armse(mse), removed=1 - mse / base_mse)
            if system != "B0":
                resc = 0.0
                for H, gh in g.groupby("season"):
                    train = g[(g.season < H - win) | (g.season > H + win)]
                    m = sc.multipliers(train) if len(train) else (np.nan, np.nan)
                    resc += float((sc.error_at(gh, m[0], m[1]) * gh.sw).sum())
                row["rescaled_error"] = armse(resc / g.sw.sum())
                jk = sc.multipliers_jackknife(g)
                row.update(slope_o=jk.coef["O"], slope_o_se=jk.se["O"], slope_d=jk.coef["D"], slope_d_se=jk.se["D"])
                full = nt[(nt.system == system) & (nt.kind == "full")]
                if len(full):
                    pf = sc.pooled(list(full["eq"]))
                    row.update(full_slope_o=pf.coef["O"], full_slope_d=pf.coef["D"])
                if system != ref and ref in set(qt.system):
                    pr = sc.paired_by_season(frame[frame.system.isin([ref, system])], ref, system, "mse")
                    row.update(vs_ref=ref, diff_mse=pr["mean_diff"], z=pr["z"], wins=pr["wins"], seasons=pr["seasons"])
            rows.append(row)
        for kind, sink in (("enc", blend), ("enc_total", blend), ("tier", tiers), ("part", parts)):
            for system, g in nt[nt.kind == kind].groupby("system"):
                pf = sc.pooled(list(g["eq"]))
                for name, r in pf.iterrows():
                    sink.append(dict(test=test, system=system, kind=kind, term=name, coef=r.coef, se=r.se))
        if test != "yoy" and {"O", "B2"} <= set(qt.system):
            p = qt[qt.system.isin(["O", "B2"])].pivot_table(index=["season", "deal", "key"], columns="system", values="mse")
            d = (p["O"] - p["B2"]).rename("d").reset_index()
            variance.append(dict(test=test, **sc.variance_components(d)))
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "summary.csv", index=False)
    pd.DataFrame(blend).to_csv(out / "blend_weights.csv", index=False)
    pd.DataFrame(tiers).to_csv(out / "tier_slopes.csv", index=False)
    pd.DataFrame(parts).to_csv(out / "part_slopes.csv", index=False)
    pd.DataFrame(variance).to_csv(out / "variance.csv", index=False)
    pd.set_option("display.width", 250, "display.max_columns", 30)
    cols = ["system", "error", "removed", "rescaled_error", "slope_o", "slope_o_se", "slope_d", "slope_d_se",
            "full_slope_o", "full_slope_d", "diff_mse", "z", "wins"]
    for test, g in summary.groupby("test", sort=False):
        g = g.set_index("system").reindex([s for s in SHOW if s in set(g.system)]).reset_index()
        print(f"\n=== {test}: error per team-game (pts/100), share of the no-ratings error removed, error after a "
              f"cross-fitted per-side multiplier, slopes (home-level, jackknife SE; full-level), paired vs "
              f"{REFERENCE.get(test, 'O')}")
        print(g[[c for c in cols if c in g.columns]].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    b = pd.DataFrame(blend)
    if len(b):
        print("\n=== blend weights (1 = the candidate's change is all signal, 0 = all noise, > 0.5 = lower error)")
        print(b.pivot_table(index=["system", "kind", "term"], columns="test", values="coef")
              .to_string(float_format=lambda v: f"{v:.3f}"))
    if variance:
        print("\n=== O minus B2: standard error now and with unlimited deals")
        print(pd.DataFrame(variance).to_string(index=False, float_format=lambda v: f"{v:.4f}"))


def main():
    check_flags()
    mode = flag("mode", "reproduce")
    tags = [t for t in flag("tags", ",".join(SIZES)).split(",") if t]
    out = ROOT / "outputs" / "scorecard"
    if mode == "score":
        out.mkdir(exist_ok=True)
        q1, n1 = score_within(tags)
        q2, n2 = score_yoy()
        pd.DataFrame(q1 + q2).to_parquet(out / "quadratics.parquet", index=False)
        pd.to_pickle(n1 + n2, out / "normals.pkl")
        print(f"wrote {out.relative_to(ROOT)}: {len(q1) + len(q2)} system-blocks")
        return
    if mode == "summary":
        summarise(out)
        return
    if mode != "reproduce":
        raise SystemExit("--mode=reproduce|score|summary")
    folds = reproduce_folds(tags)
    gap = (folds.tg - folds.tg_stored).abs().max()
    gap_base = (folds.tg_base - folds.tg_base_stored).abs().max()
    print(f"within-season: {len(folds)} folds; largest error difference {gap:.2e}, no-ratings {gap_base:.2e}, "
          f"stand-in recomputed within {folds.fill_gap.max():.1e}")
    for tag, g in folds.groupby("tag", sort=False):
        print(f"  {tag:8s} error {armse(g.tg.mean()):.4f} pts/100 per team-game (no ratings {armse(g.tg_base.mean()):.4f})")
    assert gap < 1e-8 and gap_base < 1e-8, "the scorecard does not reproduce the stored fold errors"
    yoy_tag = flag("yoy", "shotmix_pin")
    yoy = reproduce_yoy(yoy_tag, ROOT / "outputs" / "season_ratings_priorshrink.parquet")
    gap_y = (yoy.tg - yoy.tg_stored).abs().max()
    print(f"year-over-year: {len(yoy)} season-directions; largest difference {gap_y:.2e}; pooled "
          f"{armse(yoy.tg.mean()):.4f} (stored {armse(yoy.tg_stored.mean()):.4f})")
    assert gap_y < 1e-8, "the scorecard does not reproduce the stored year-over-year errors"


if __name__ == "__main__":
    main()
