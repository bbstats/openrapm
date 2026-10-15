"""Check 135, the robust review: does a change help nearly everywhere, by more than noise alone moves things?

The owner (2026-10-08): "want to review our results in a not-brittle fashion. single player small moves shouldnt
concern us but - systemic improvements should be seen positive".  One pooled z against a cutoff is brittle: a change
that helps a little nearly everywhere can sit at z -1.7 while one that helps a lot in three seasons passes.  So every
test the experiment chain already ran is cut into slices, and the report counts how many slices move each way, beside
the same counts for a noise control.

Words used below
    reference   the incumbent's run (--ref): every slice is the candidate minus the reference
    candidate   an experiment's run name (--cands); a second name is its replicate, the same change built again with
                other random groupings of the box prior's training players, so every slice is read twice
    control     a run that changed nothing but that random grouping (--controls): what noise alone does
    slice       one test restricted to one group: a direction, an era, a side, a quality tier, the movers, an age
    cell        an era crossed with a direction: the six cells share no scored games, so the summary counts them apart
    mean_diff   the slice's mean difference, oriented so NEGATIVE is better (an agreement that should rise is negated)
    z           mean_diff over its standard error.  z_by_rated_season lets observations that share a rated season
                move together (a cluster-robust standard error)
    better      how many observations in the slice are better; median, the median difference
    wilcoxon_p  the Wilcoxon signed-rank p-value: ranks, so one large season cannot carry it

The tests, each from a file the chain saved for that run name (outputs/):
    year over year (check 63)       yoy_<name>.parquet: team-game and stint error per scored season and direction
    side split                      check 63 run here on two mixed tables, the candidate's offence with the
                                    reference's defence and the reverse (cached as yoy_<name>_sides.parquet)
    year over year by player (88)   yoy_by_player_<name>.parquet: the team-game difference shared among the players
    trade loss (checks 70, 73)      tradeset_<name>_alpha.parquet against tradeset_<ref>_alpha.parquet
    lineup-swap test (check 90)     swaptest_<name>.parquet
    consensus agreement (check 64)  season_ratings_<name>.parquet joined to the consensus; the change in rank
                                    agreement, its standard error from 2,000 bootstrap draws of players

The slices overlap (the tiers add up to the whole, the side split roughly does too).  Writes outputs/<out>.csv and
.parquet (one row per run name and slice) and outputs/<out>_summary.csv (the pattern counts).

How to read it: the summary counts, per test, the slices better and worse and those past z 2 either way.  A change
worth keeping is better in most slices of most tests, by more than the control is; a z between two noise draws is
about standard normal whatever their size, so compare sizes, not only z.  Example (experiment 43, averaging the prior
over five groupings): better in 42 of 47 prediction slices, its replicate in 34, one random regrouping in 8 -- a
small change, but a systemic one.

    python scripts/checks/135_robust_review.py --cands=prior_avg5,prior_avg5b --controls=redeal_incumbent \\
                                               --ref=weightfix --out=robust_prior_avg5
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from openrapm.commands import year_over_year as yoy
from openrapm.commands.console import use_utf8
from openrapm.consensus import matched
from openrapm.data.config import load_config, resolve
from openrapm.data.designs import DesignCache
from openrapm.data.tables import load_consensus, load_ratings
from openrapm.scoring import QUALITY_LABELS, quality_tiers
from openrapm.trade_set import SIDES, eligible_corrections, missed

ERAS = {"1998-2006": (1998, 2006), "2007-2016": (2007, 2016), "2017-2025": (2017, 2025)}
# The trade loss is read by rated season, 1997-2026: the outer seasons join the first and last era.
TRADE_ERAS = {"1998-2006": "1997-2006", "2007-2016": "2007-2016", "2017-2025": "2017-2026"}
DIRECTION = {"prev": "rating looks forward", "next": "rating looks back"}
SWAP_DIRECTION = {"forward": "rating looks forward", "back": "rating looks back"}
SIDE_SPLIT = (("offside", "rating_off", "offence only (defence as now)"),
              ("defside", "rating_def", "defence only (offence as now)"))
CONSENSUS_SIDES = (("offence", "rating_off", "adj_offense"), ("defence", "rating_def", "adj_defense"),
                   ("total", "rating_total", "adj_overall"))
PATTERN = ("better", "worse", "clearly_better", "clearly_worse")
OFF = ("0", "no", "false")


def on_off(text: str) -> bool:
    """A switch as typed: 0, no or false is off, anything else on."""
    return str(text) not in OFF


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A robust review of an experiment, every test cut into slices.",
                                     allow_abbrev=False)
    parser.add_argument("--cands", default="", help="the chain's run names: <name>[,<replicate>]")
    # the name README.md's noise-control chain gives it; on the project's machine the incumbent's noise control is
    # noise_regroup2 (settings.PriorSettings.fold_assignment)
    parser.add_argument("--controls", default="noise_control", help="noise controls (default %(default)s)")
    # the incumbent since 2026-10-09 (experiment 46; 45 was lgb_noonc_rs)
    parser.add_argument("--ref", default="cols06", help="the reference run (default %(default)s)")
    parser.add_argument("--refresh", default="0", type=on_off, help="1: rerun the side split (default %(default)s)")
    parser.add_argument("--out", help="outputs/<OUT>.csv (default robust_<first candidate>)")
    return parser.parse_args(argv)


# --------------------------------------------------------------------------------------------- one slice
def era_of(season) -> np.ndarray:
    """Each season's era label (ERAS); "" for a season outside them."""
    season = np.asarray(season, dtype=int)
    out = np.full(season.shape, "", dtype=object)
    for label, (low, high) in ERAS.items():
        out[(season >= low) & (season <= high)] = label
    return out


def clustered_z(diff: np.ndarray, cluster: np.ndarray) -> float:
    """The mean difference over a standard error that lets observations of one cluster move together:

        variance = G / (G - 1) x sum over clusters of (sum of the cluster's deviations from the mean)^2 / n^2

    G clusters, n observations.  Example: two rated seasons each scored twice, deviations (+1, +1) and (-1, -1),
    give 2 x (4 + 4) / 16 = 1.0, where treating the four as independent would give 4 / 16 x 4 / 3 = 0.33.
    """
    n = diff.size
    deviations = pd.Series(diff - diff.mean()).groupby(cluster).sum().to_numpy()
    groups = deviations.size
    variance = (groups / (groups - 1)) * float((deviations ** 2).sum()) / n ** 2 if groups > 1 else np.nan
    return float(diff.mean() / np.sqrt(variance)) if variance and variance > 0 else np.nan


def summarise(diff, cluster=None) -> dict:
    """One slice: n, mean_diff (negative = better), z, better, median and wilcoxon_p (module docstring); with
    `cluster` (each observation's rated season) also z_by_rated_season."""
    values = np.asarray(diff, dtype=float)
    finite = np.isfinite(values)
    d = values[finite]
    n = d.size
    se = float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    out = dict(n=n, mean_diff=float(d.mean()) if n else np.nan,
               z=float(d.mean() / se) if n > 1 and se > 0 else np.nan,
               better=int((d < 0).sum()), median=float(np.median(d)) if n else np.nan,
               wilcoxon_p=float(stats.wilcoxon(d).pvalue) if n >= 6 and np.any(d != 0) else np.nan)
    if cluster is not None and n > 2:
        out["z_by_rated_season"] = clustered_z(d, np.asarray(cluster)[finite])
    return out


# ------------------------------------------------------------------------------------- year over year
def paired_yoy(path: Path, system: str, ref_system: str = "incumbent") -> pd.DataFrame:
    """One check 63 output, the candidate minus the reference per scored season and direction: team-game error `tg`,
    stint error `stint`, and the season the ratings were made from (`rated`)."""
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


def yoy_slices(name: str, outputs: Path) -> list:
    """Check 63's team-game and stint errors per scored season and direction, paired against the reference."""
    frame = paired_yoy(outputs / f"yoy_{name}.parquet", name)
    family = "year over year, team-game"
    rows = [dict(family=family, slice="all", **summarise(frame.tg, frame.rated))]
    for key, label in DIRECTION.items():
        rows.append(dict(family=family, slice=label, **summarise(frame[frame.direction == key].tg)))
    for era in ERAS:
        for key, label in DIRECTION.items():
            part = frame[(frame.era == era) & (frame.direction == key)]
            rows.append(dict(family=family, slice=f"{era}, {label}", cell=True, **summarise(part.tg)))
    rows.append(dict(family="year over year, stint", slice="all", **summarise(frame.stint, frame.rated)))
    for era in ERAS:
        rows.append(dict(family="year over year, stint", slice=era, **summarise(frame[frame.era == era].stint)))
    return rows


def write_mixed_tables(name: str, ref_table: Path, outputs: Path) -> dict:
    """{side: path} of the two mixed tables: the reference with the candidate's offence, and with its defence."""
    cand = pd.read_parquet(outputs / f"season_ratings_{name}.parquet")
    ref = pd.read_parquet(ref_table)
    keys = ["player_id", "season"]
    mixed = ref.merge(cand[keys + ["rating_off", "rating_def"]], on=keys, how="left", suffixes=("", "_cand"),
                      validate="one_to_one")
    if mixed[["rating_off_cand", "rating_def_cand"]].isna().any().any():
        raise SystemExit(f"{name}: the candidate's table does not cover every row of the reference")
    paths = {}
    for side, column, _ in SIDE_SPLIT:
        table = mixed.copy()
        table[column] = table[f"{column}_cand"]
        table["rating_total"] = table.rating_off + table.rating_def
        path = outputs / f"season_ratings_{name}_{side}.parquet"
        table.drop(columns=["rating_off_cand", "rating_def_cand"]).to_parquet(path, index=False)
        paths[side] = path
    return paths


def score_mixed_tables(name: str, paths: dict, ref_table: Path, out_path: Path, cfg: dict) -> None:
    """Check 63 on the two mixed tables and the reference (named "incumbent"), the whole season only; the rows are
    the ones `scripts/checks/63_yoy.py --splits=` writes."""
    files = {f"{name}_{side}": path for side, path in paths.items()} | {"incumbent": ref_table}
    tables = {}
    for label, path in files.items():
        table = load_ratings(path, "rating", possessions=True)
        tables[label] = {"prev": table, "next": table}
    yoy.run(yoy.neighbour_tables(tables), DesignCache(cfg), yoy.scored_seasons(cfg), out=out_path, verbose=False)


def side_slices(name: str, ref_table: Path, refresh: bool, outputs: Path, cfg: dict) -> list:
    """Which side carries it: check 63 on the candidate's offence with the reference's defence, and the reverse."""
    out_path = outputs / f"yoy_{name}_sides.parquet"
    if refresh or not out_path.exists():
        score_mixed_tables(name, write_mixed_tables(name, ref_table, outputs), ref_table, out_path, cfg)
    rows = []
    for side, _, label in SIDE_SPLIT:
        part = paired_yoy(out_path, f"{name}_{side}")
        rows.append(dict(family="year over year, team-game", slice=label, **summarise(part.tg, part.rated)))
    return rows


def by_player_slices(name: str, outputs: Path) -> list:
    """Check 88: the team-game difference shared among the players on the court, summed by group (adds up to 63's)."""
    t = pd.read_parquet(outputs / f"yoy_by_player_{name}.parquet")
    t = t[(t.candidate == name) & (t.split != "all")]
    return [dict(family=f"year over year by player {split}", slice=str(group), **summarise(part.tg))
            for (split, group), part in t.groupby(["split", "group"], sort=False)]


# ----------------------------------------------------------------------------------------- the trade loss
def loss_change(cand: pd.DataFrame, ref: pd.DataFrame, players) -> float:
    """The candidate's trade loss minus the reference's on `players`."""
    return missed(cand.loc[players].alpha_good.to_numpy()) - missed(ref.loc[players].alpha_good.to_numpy())


def season_loss_changes(cand: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    """One side, per rated season: the change in trade loss on the players both have, overall ("all") and in each
    quality tier (NaN for a tier with nobody), with the season's era."""
    per = []
    for season in sorted(set(cand.index.get_level_values("season"))):
        cand_season, ref_season = cand.xs(season, level="season"), ref.xs(season, level="season")
        both = cand_season.index.intersection(ref_season.index)
        if len(both) == 0:
            continue
        item = {"season": season, "all": loss_change(cand_season, ref_season, both)}
        for tier in QUALITY_LABELS:
            keep = both[ref_season.loc[both].tier.to_numpy() == tier]
            item[tier] = loss_change(cand_season, ref_season, keep) if len(keep) else np.nan
        per.append(item)
    per = pd.DataFrame(per)
    per["era"] = [TRADE_ERAS[e] for e in era_of(per.season.clip(1998, 2025))]
    return per


def trade_slices(name: str, ref: str, quality: pd.Series, outputs: Path) -> list:
    """Checks 70 + 73: the with/without correction each rating leaves, every player once, paired by season."""
    cand = eligible_corrections(pd.read_parquet(outputs / f"tradeset_{name}_alpha.parquet"), 1.0, quality)
    base = eligible_corrections(pd.read_parquet(outputs / f"tradeset_{ref}_alpha.parquet"), 1.0, quality)
    rows = []
    for side in SIDES:
        per = season_loss_changes(cand.xs(side, level="side"), base.xs(side, level="side"))
        family = f"trade loss, {side}"
        rows.append(dict(family=family, slice="all", **summarise(per["all"])))
        for era in TRADE_ERAS.values():
            rows.append(dict(family=family, slice=era, cell=True, **summarise(per.loc[per.era == era, "all"])))
        for tier in QUALITY_LABELS:
            rows.append(dict(family=family, slice=tier, **summarise(per[tier])))
    return rows


# --------------------------------------------------------------------------------- the lineup-swap test
def swap_slices(name: str, outputs: Path) -> list:
    """Check 90: does the ranking order teammates the way the neighbouring seasons' lineup swaps do (order: higher is
    better, so it is negated; gaps: lower is better)."""
    t = pd.read_parquet(outputs / f"swaptest_{name}.parquet")
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


# ---------------------------------------------------------------------------------------- the consensus
def rank_agreement(ratings: np.ndarray, consensus: np.ndarray, rows: np.ndarray) -> float:
    """Spearman's rank correlation of a ratings table with the consensus, on the players at `rows`."""
    return stats.spearmanr(ratings[rows], consensus[rows]).statistic


def agreement_change(cand: np.ndarray, ref: np.ndarray, consensus: np.ndarray, rows: np.ndarray) -> float:
    """The candidate's rank agreement minus the reference's, negated so that negative is better."""
    return -(rank_agreement(cand, consensus, rows) - rank_agreement(ref, consensus, rows))


def consensus_slices(name: str, ref_table: Path, consensus: pd.DataFrame, outputs: Path, draws: int = 2000) -> list:
    """Check 64's join (`openrapm.consensus.matched`); the change in rank agreement per side, with its standard
    error from `draws` bootstrap samples of the players both tables match."""
    cand = matched(pd.read_parquet(outputs / f"season_ratings_{name}.parquet"), consensus).set_index("key")
    ref = matched(pd.read_parquet(ref_table), consensus).set_index("key")
    keys = cand.index.intersection(ref.index)
    cand, ref = cand.loc[keys], ref.loc[keys]
    rng = np.random.default_rng(0)
    rows = []
    for side, rating_column, consensus_column in CONSENSUS_SIDES:
        cand_ratings, ref_ratings = cand[rating_column].to_numpy(float), ref[rating_column].to_numpy(float)
        consensus_ratings = cand[consensus_column].to_numpy(float)
        count = len(consensus_ratings)
        diff = agreement_change(cand_ratings, ref_ratings, consensus_ratings, np.arange(count))
        boots = np.array([agreement_change(cand_ratings, ref_ratings, consensus_ratings,
                                           rng.integers(0, count, count)) for _ in range(draws)])
        se = float(boots.std(ddof=1))
        rows.append(dict(family="consensus agreement", slice=side, n=count, mean_diff=float(diff),
                         z=float(diff / se) if se > 0 else np.nan, better=np.nan, median=float(diff)))
    return rows


# --------------------------------------------------------------------------------------------- the review
def slices_of(name: str, ref: str, ref_table: Path, refresh: bool, quality: pd.Series, consensus: pd.DataFrame,
              cfg: dict) -> list:
    """Every slice of every test for one run name, in the report's order."""
    outputs = resolve(cfg, "outputs")
    return (yoy_slices(name, outputs) + side_slices(name, ref_table, refresh, outputs, cfg)
            + by_player_slices(name, outputs) + trade_slices(name, ref, quality, outputs)
            + swap_slices(name, outputs) + consensus_slices(name, ref_table, consensus, outputs))


def review(runs: tuple, ref: str, refresh: bool, cfg: dict) -> pd.DataFrame:
    """One row per run name and slice, for (role, names) pairs such as ("candidate", [...]), ("control", [...])."""
    ref_table = resolve(cfg, "outputs") / f"season_ratings_{ref}.parquet"
    consensus = load_consensus(cfg)
    quality = quality_tiers(pd.read_parquet(ref_table, columns=["player_id", "season", "rating_total"]))
    rows = []
    for role, names in runs:
        for name in names:
            got = slices_of(name, ref, ref_table, refresh, quality, consensus, cfg)
            rows += [dict(name=name, role=role, **r) for r in got]
            print(f"  {name}: {len(got)} slices", flush=True)
    table = pd.DataFrame(rows)
    table["cell"] = table.get("cell", pd.Series(False, index=table.index)).fillna(False).astype(bool)
    return table


def pattern_row(name: str, family: str, part: pd.DataFrame) -> dict:
    """One row of the pattern: how many slices are better and worse by sign, and how many past z 2 either way."""
    return dict(name=name, family=family, slices=len(part), better=int((part.mean_diff < 0).sum()),
                worse=int((part.mean_diff > 0).sum()), clearly_better=int((part.z <= -2).sum()),
                clearly_worse=int((part.z >= 2).sum()))


def pattern(table: pd.DataFrame) -> pd.DataFrame:
    """Per run name and test: the slices better and worse by sign, and those past z 2 either way; then the same for
    the cells that share no scored games."""
    rows = [pattern_row(name, family, part) for (name, family), part in table.groupby(["name", "family"], sort=False)]
    rows += [pattern_row(name, f"{family}: cells sharing no scored games", part)
             for (name, family), part in table[table.cell].groupby(["name", "family"], sort=False)]
    return pd.DataFrame(rows)


def report(table: pd.DataFrame, out: Path) -> None:
    """Print every slice and the pattern, and write the pattern beside the table."""
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.max_rows", 400):
        print("\n=== every slice; NEGATIVE mean_diff = better than the reference; better = observations better")
        show = table.pivot_table(index=["family", "slice"], columns="name", values=["mean_diff", "z"], sort=False)
        print(show.round(4).to_string())
        print("\n=== the pattern: slices better / worse by sign, and those past z 2 either way")
        summary = pattern(table)
        summary.to_csv(out.with_name(out.stem + "_summary.csv"), index=False)
        print(summary.pivot_table(index="family", columns="name", values=list(PATTERN), sort=False)
              .fillna(0).astype(int).to_string())


def main(argv=None) -> None:
    use_utf8()
    args = parse_args(argv)
    cands = [c for c in args.cands.split(",") if c]
    controls = [c for c in args.controls.split(",") if c]
    if not cands:
        raise SystemExit("--cands=<name>[,<replicate>]: the chain's run names")
    cfg = load_config()
    out = resolve(cfg, "outputs") / f"{args.out or 'robust_' + cands[0]}.csv"
    table = review((("candidate", cands), ("control", controls)), args.ref, args.refresh, cfg)
    table.to_parquet(out.with_suffix(".parquet"), index=False)
    table.to_csv(out, index=False)
    report(table, out)
    print(f"\nwrote outputs/{out.name} and .parquet ({len(table)} rows)")


if __name__ == "__main__":
    main()
