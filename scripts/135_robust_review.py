"""A robust review of an experiment: every test the chain ran, cut into slices and read for a systemic pattern.

The owner (2026-10-08): "want to review our results in a not-brittle fashion. single player small moves shouldnt
concern us but - systemic improvements should be seen positive".  One pooled z against a cutoff is brittle: a change
that helps a little nearly everywhere can sit at z -1.7 while one that helps a lot in three seasons passes.  So each
instrument the standard chain already ran is cut into slices -- direction, era, side, player-quality tier, movers,
age -- and the report counts how many slices move each way, beside the same counts for a control that changed nothing
but noise (C1: one regrouping of the box-score prior's training players), so what noise alone does is on the page.
With a replicate (`--cands=a,b`) every slice is read twice.

    python scripts/135_robust_review.py --cands=prior_avg5,prior_avg5b [--controls=redeal_incumbent]
                                        [--ref=priorshrink] [--out=robust_prior_avg5]

Reads what the chain saved -- outputs/yoy_<name>.parquet (63), yoy_by_player_<name>.parquet (88),
swaptest_<name>.parquet (90), tradeset_<name>_alpha.parquet (70) against tradeset_<ref>_alpha.parquet -- and the
ratings tables, for the consensus (64's join) and for the side split, which runs 63 on two mixed tables (the
candidate's offence with the reference's defence, and the reverse; ~10 s, cached as yoy_<name>_sides.parquet).

Every slice is oriented so that a NEGATIVE difference is better.  The slices overlap (tiers add up to the whole, the
side split roughly does too); the eras crossed with the two directions are the six cells that share no scored games,
and the summary counts those apart.  Writes outputs/<out>.csv, one row per name and slice.
"""
import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _cli import check_flags, flag  # noqa: E402

ERAS = {"1998-2006": (1998, 2006), "2007-2016": (2007, 2016), "2017-2025": (2017, 2025)}
DIRECTION = {"prev": "rating looks forward", "next": "rating looks back"}
SWAP_DIRECTION = {"forward": "rating looks forward", "back": "rating looks back"}


def _module(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def era_of(season) -> np.ndarray:
    season = np.asarray(season, dtype=int)
    out = np.full(season.shape, "", dtype=object)
    for label, (a, b) in ERAS.items():
        out[(season >= a) & (season <= b)] = label
    return out


def summarise(diff, cluster=None) -> dict:
    """One slice: the mean difference (negative = better), its z, how many observations are better, the median, and
    the Wilcoxon signed-rank p (ranks, so one large season cannot carry it).  `cluster`: a z whose standard error
    lets observations that share a rated season move together."""
    d = np.asarray(diff, dtype=float)
    d = d[np.isfinite(d)]
    n = d.size
    se = float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    out = dict(n=n, mean_diff=float(d.mean()) if n else np.nan,
               z=float(d.mean() / se) if n > 1 and se > 0 else np.nan,
               better=int((d < 0).sum()), median=float(np.median(d)) if n else np.nan,
               wilcoxon_p=float(stats.wilcoxon(d).pvalue) if n >= 6 and np.any(d != 0) else np.nan)
    if cluster is not None and n > 2:
        c = np.asarray(cluster)[np.isfinite(np.asarray(diff, dtype=float))]
        dev = pd.Series(d - d.mean()).groupby(c).sum().to_numpy()
        g = dev.size
        var = (g / (g - 1)) * float((dev ** 2).sum()) / n ** 2 if g > 1 else np.nan
        out["z_by_rated_season"] = float(d.mean() / np.sqrt(var)) if var and var > 0 else np.nan
    return out


def paired_yoy(path: Path, system: str, ref_system: str = "incumbent") -> pd.DataFrame:
    """One 63 output, the candidate minus the reference per scored season and direction: team-game error `tg`, stint
    error `stint`, and the season the ratings were made from (`rated`)."""
    t = pd.read_parquet(path)
    t = t[(t.split == "all") & (t.group == "all")]
    t = t.assign(base_name=t.system.str.rsplit(":", n=1).str[0], direction=t.system.str.rsplit(":", n=1).str[1])
    cand = t[t.base_name == system].set_index(["held_out", "direction"])
    ref = t[t.base_name == ref_system].set_index(["held_out", "direction"])
    both = cand.index.intersection(ref.index)
    frame = pd.DataFrame({"tg": cand.loc[both, "tg"] - ref.loc[both, "tg"],
                          "stint": cand.loc[both, "mse"] - ref.loc[both, "mse"],
                          "rated": cand.loc[both, "train"].astype(int)}).reset_index()
    frame["era"] = era_of(frame.held_out)
    return frame


def yoy_slices(name: str) -> list:
    """63's team-game and stint errors per scored season and direction, paired against the reference."""
    frame = paired_yoy(ROOT / "outputs" / f"yoy_{name}.parquet", name)
    rows = [dict(family="year over year, team-game", slice="all", **summarise(frame.tg, frame.rated))]
    for key, label in DIRECTION.items():
        part = frame[frame.direction == key]
        rows.append(dict(family="year over year, team-game", slice=label, **summarise(part.tg)))
    for era in ERAS:
        for key, label in DIRECTION.items():
            part = frame[(frame.era == era) & (frame.direction == key)]
            rows.append(dict(family="year over year, team-game", slice=f"{era}, {label}", cell=True,
                             **summarise(part.tg)))
    rows.append(dict(family="year over year, stint", slice="all", **summarise(frame.stint, frame.rated)))
    for era in ERAS:
        rows.append(dict(family="year over year, stint", slice=era, **summarise(frame[frame.era == era].stint)))
    return rows


def side_slices(name: str, ref_table: Path, refresh: bool) -> list:
    """Which side carries it: 63 on the candidate's offence with the reference's defence, and the reverse."""
    out_path = ROOT / "outputs" / f"yoy_{name}_sides.parquet"
    if refresh or not out_path.exists():
        cand = pd.read_parquet(ROOT / "outputs" / f"season_ratings_{name}.parquet")
        ref = pd.read_parquet(ref_table)
        keys = ["player_id", "season"]
        mixed = ref.merge(cand[keys + ["rating_off", "rating_def"]], on=keys, how="left", suffixes=("", "_cand"),
                          validate="one_to_one")
        if mixed[["rating_off_cand", "rating_def_cand"]].isna().any().any():
            raise SystemExit(f"{name}: the candidate's table does not cover every row of the reference")
        paths = {}
        for side, column in (("offside", "rating_off"), ("defside", "rating_def")):
            table = mixed.copy()
            table[column] = table[f"{column}_cand"]
            table["rating_total"] = table.rating_off + table.rating_def
            path = ROOT / "outputs" / f"season_ratings_{name}_{side}.parquet"
            table.drop(columns=["rating_off_cand", "rating_def_cand"]).to_parquet(path, index=False)
            paths[side] = path
        spec = ",".join([f"{name}_{s}=outputs/{p.name}" for s, p in paths.items()]
                        + [f"incumbent={ref_table.relative_to(ROOT).as_posix()}"])
        subprocess.run([sys.executable, "-u", str(ROOT / "scripts" / "63_yoy.py"), f"--rankings={spec}",
                        "--ref=incumbent", f"--tag={name}_sides", "--splits="], check=True, cwd=ROOT,
                       stdout=subprocess.DEVNULL)
    rows = []
    for side, label in (("offside", "offence only (defence as now)"), ("defside", "defence only (offence as now)")):
        part = paired_yoy(out_path, f"{name}_{side}")
        rows.append(dict(family="year over year, team-game", slice=label, **summarise(part.tg, part.rated)))
    return rows


def by_player_slices(name: str) -> list:
    """88: the team-game difference shared among the players on the floor, summed by group (adds up to 63's)."""
    t = pd.read_parquet(ROOT / "outputs" / f"yoy_by_player_{name}.parquet")
    t = t[(t.candidate == name) & (t.split != "all")]
    rows = []
    for (split, group), part in t.groupby(["split", "group"], sort=False):
        rows.append(dict(family=f"year over year by player {split}", slice=str(group), **summarise(part.tg)))
    return rows


def trade_slices(name: str, ref: str, tl, quality: pd.Series) -> list:
    """70 + 73: the with/without correction each rating leaves, every player once, paired by season."""
    cand = tl.load(ROOT / "outputs" / f"tradeset_{name}_alpha.parquet", 1.0, quality)
    base = tl.load(ROOT / "outputs" / f"tradeset_{ref}_alpha.parquet", 1.0, quality)
    rows = []
    for side in ("offense", "defense"):
        c, r = cand.xs(side, level="side"), base.xs(side, level="side")
        per = []
        for season in sorted(set(c.index.get_level_values("season"))):
            cs, rs = c.xs(season, level="season"), r.xs(season, level="season")
            both = cs.index.intersection(rs.index)
            if len(both) == 0:
                continue
            item = {"season": season,
                    "all": tl.missed(cs.loc[both].alpha_good.to_numpy()) - tl.missed(rs.loc[both].alpha_good.to_numpy())}
            for tier in tl.QUALITY_LABELS:
                keep = both[rs.loc[both].tier.to_numpy() == tier]
                item[tier] = (tl.missed(cs.loc[keep].alpha_good.to_numpy()) - tl.missed(rs.loc[keep].alpha_good.to_numpy())
                              if len(keep) else np.nan)
            per.append(item)
        per = pd.DataFrame(per)
        # rated seasons 1997-2026: the eras' outer seasons join the first and last decade
        decade = {"1998-2006": "1997-2006", "2007-2016": "2007-2016", "2017-2025": "2017-2026"}
        per["era"] = [decade[e] for e in era_of(per.season.clip(1998, 2025))]
        family = f"trade loss, {side}"
        rows.append(dict(family=family, slice="all", **summarise(per["all"])))
        for era in decade.values():
            rows.append(dict(family=family, slice=era, cell=True, **summarise(per.loc[per.era == era, "all"])))
        for tier in tl.QUALITY_LABELS:
            rows.append(dict(family=family, slice=tier, **summarise(per[tier])))
    return rows


def swap_slices(name: str) -> list:
    """90: does the ranking order teammates the way the neighbouring seasons' lineup swaps do (order higher is better,
    so it is negated; gaps lower is better)."""
    t = pd.read_parquet(ROOT / "outputs" / f"swaptest_{name}.parquet")
    t = t[(t.group == "all")]
    key = ["side", "scored", "direction"]
    cand = t[t.ranking == name].set_index(key)
    ref = t[t.ranking == "incumbent"].set_index(key)
    both = cand.index.intersection(ref.index)
    frame = pd.DataFrame({"order": -(cand.loc[both, "order"] - ref.loc[both, "order"]),
                          "gaps": cand.loc[both, "gaps"] - ref.loc[both, "gaps"]}).reset_index()
    frame["era"] = era_of(frame.scored)
    rows = []
    for side in ("net", "offense", "defense"):
        part = frame[frame.side == side]
        for measure in ("order", "gaps"):
            rows.append(dict(family="lineup-swap test", slice=f"{side} {measure}", **summarise(part[measure])))
    net = frame[frame.side == "net"]
    for era in ERAS:
        for key, label in SWAP_DIRECTION.items():
            part = net[(net.era == era) & (net.direction == key)]
            rows.append(dict(family="lineup-swap test", slice=f"net order, {era}, {label}", cell=True,
                             **summarise(part.order)))
    return rows


def consensus_slices(name: str, ref_table: Path, tvc, draws: int = 2000) -> list:
    """64's join; the rank agreement's change with a paired bootstrap over players (higher agreement is better, so
    the change is negated)."""
    con = pd.read_csv(tvc.CONSENSUS)[["player_name", "adj_offense", "adj_defense", "adj_overall"]]
    con = con.dropna(subset=["player_name", "adj_overall"])
    con["key"] = con.player_name.map(tvc._norm)
    con = con.drop_duplicates("key")

    def joined(path):
        ours = tvc._pooled(pd.read_parquet(path))
        ours["key"] = ours.player_name.map(tvc._norm)
        ours = ours.sort_values("poss_off").drop_duplicates("key", keep="last")
        b = ours.merge(con, on="key", how="inner")
        return b[b.poss_off >= tvc.MIN_POSS].set_index("key")

    a, b = joined(ROOT / "outputs" / f"season_ratings_{name}.parquet"), joined(ref_table)
    keys = a.index.intersection(b.index)
    a, b = a.loc[keys], b.loc[keys]
    rng = np.random.default_rng(0)
    rows = []
    for side, mine, theirs in (("offence", "rating_off", "adj_offense"), ("defence", "rating_def", "adj_defense"),
                               ("total", "rating_total", "adj_overall")):
        x_a, x_b, y = a[mine].to_numpy(float), b[mine].to_numpy(float), a[theirs].to_numpy(float)

        def rho(x, idx):
            return stats.spearmanr(x[idx], y[idx]).statistic
        whole = np.arange(len(y))
        diff = -(rho(x_a, whole) - rho(x_b, whole))
        boots = np.array([-(rho(x_a, i) - rho(x_b, i)) for i in
                          (rng.integers(0, len(y), len(y)) for _ in range(draws))])
        se = float(boots.std(ddof=1))
        rows.append(dict(family="consensus agreement", slice=side, n=len(y), mean_diff=float(diff),
                         z=float(diff / se) if se > 0 else np.nan, better=np.nan, median=float(diff)))
    return rows


def main() -> None:
    check_flags()
    cands = [c for c in flag("cands", "").split(",") if c]
    controls = [c for c in flag("controls", "redeal_incumbent").split(",") if c]
    ref = flag("ref", "lgb_noonc_rs")             # the incumbent since 2026-10-08 (experiment 45)
    refresh = flag("refresh", "0") not in ("0", "no", "false")
    out = ROOT / "outputs" / f"{flag('out', 'robust_' + (cands[0] if cands else 'review'))}.csv"
    if not cands:
        raise SystemExit("--cands=<name>[,<replicate>]: the chain's run names")
    ref_table = ROOT / "outputs" / f"season_ratings_{ref}.parquet"
    tl = _module("tradeloss_73", "scripts/73_tradeloss.py")
    tvc = _module("tvc_135", "tests/test_vs_consensus.py")
    quality = tl.quality_tiers(ref_table)
    rows = []
    for role, names in (("candidate", cands), ("control", controls)):
        for name in names:
            got = (yoy_slices(name) + side_slices(name, ref_table, refresh) + by_player_slices(name)
                   + trade_slices(name, ref, tl, quality) + swap_slices(name) + consensus_slices(name, ref_table, tvc))
            rows += [dict(name=name, role=role, **r) for r in got]
            print(f"  {name}: {len(got)} slices", flush=True)
    table = pd.DataFrame(rows)
    table["cell"] = table.get("cell", pd.Series(False, index=table.index)).fillna(False).astype(bool)
    table.to_parquet(out.with_suffix(".parquet"), index=False)
    table.to_csv(out, index=False)

    pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_rows", 400)
    print("\n=== every slice; NEGATIVE mean_diff = better than the reference; better = observations better")
    show = table.pivot_table(index=["family", "slice"], columns="name", values=["mean_diff", "z"], sort=False)
    print(show.round(4).to_string())

    print("\n=== the pattern: slices better / worse by sign, and those past z 2 either way")
    summary = []
    for (name, family), part in table.groupby(["name", "family"], sort=False):
        summary.append(dict(name=name, family=family, slices=len(part), better=int((part.mean_diff < 0).sum()),
                            worse=int((part.mean_diff > 0).sum()), clearly_better=int((part.z <= -2).sum()),
                            clearly_worse=int((part.z >= 2).sum())))
    cells = table[table.cell]
    for (name, family), part in cells.groupby(["name", "family"], sort=False):
        summary.append(dict(name=name, family=f"{family}: cells sharing no scored games", slices=len(part),
                            better=int((part.mean_diff < 0).sum()), worse=int((part.mean_diff > 0).sum()),
                            clearly_better=int((part.z <= -2).sum()), clearly_worse=int((part.z >= 2).sum())))
    summary = pd.DataFrame(summary)
    summary.to_csv(out.with_name(out.stem + "_summary.csv"), index=False)
    print(summary.pivot_table(index="family", columns="name", values=["better", "worse", "clearly_better",
                                                                      "clearly_worse"], sort=False)
          .fillna(0).astype(int).to_string())
    print(f"\nwrote {out.relative_to(ROOT)} and .parquet ({len(table)} rows)")


if __name__ == "__main__":
    main()
