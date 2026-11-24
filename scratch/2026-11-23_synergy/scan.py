"""Lineup synergies, the scan: which lineup statistics make a five score more, or allow less, than the sum of its
players' ratings says, on games the ratings never saw, and which part of that no player rating can carry.

The owner, 2026-11-23: "what are the 'statistical synergies' that improve lineups on offense AND on defense? i think we
have shown that self-creation helps! ... what else? look at everything".  Read-only; no rating moves.

The stints and the players are extract.py's.  For one statistic, each player's value from his fold's fit games is
centered on the fold's possession-weighted average; a stand-in for a player the fold did not rate carries the
average of the fold's players under 500 possessions, as check 154 fills one.  For a five's values a1..a5:

    lean      a1 + ... + a5                the five's sum: a straight line in it is the same as every player rated a
                                           little off in proportion to his own value, which a player rating can carry
    own bend  a1^2 + ... + a5^2            still one player at a time: a rating can carry it too
    synergy   the sum of ai x aj over the ten pairs of teammates: the one part no player rating can carry.  Above 0:
              two players high in the statistic help each other, more than the sum of their ratings.  Below 0: they
              get in each other's way, the second is worth less beside the first (check 154's concentration, at the
              same amount, is this term with the opposite sign)

Every stint row has an offense five and a defense five, so each statistic is read on both ends at once, the
offense's five against the points it scored and the defense's five against the points it allowed.  Beside them,
check 154's free terms with the lineup shape held, and the defense's playing time and rating spread: team levels per
fold (fitted out of every column first), home, playoffs, garbage time, margin, margin times time left, both rating
sums, their squares and product, the spread of each five's ratings, each five's summed share of team possessions, and
the stand-ins.

Read per season, weighted least squares on possessions, then across seasons as check 154 reads it.  Reported in
points per 100 for the lineup, positive better on both ends (on defense, fewer points allowed):

    synergy per lineup sd    the synergy term's coefficient times its possession-weighted standard deviation over
                             the season's stints: how far it moves a typical lineup
    synergy per pair         times the squared player standard deviation: two teammates each one standard deviation
                             above average, beyond what they each add alone
    lean per player          the lean's coefficient times the player standard deviation

Also: pairs of different statistics (a player high in one beside a teammate high in the other, the term
sum over i != j of ai x bj, with each statistic's own three terms and the player's own product held); a joint fit of
every statistic's three terms on both ends at once; and each statistic's count table, the mean miss after the free
terms by how many of the five sit in its fold's top fifth by possessions (0, 1, 2, 3 or more).

    .venv/Scripts/python scratch/2026-11-23_synergy/scan.py --data=outputs/synergy/rd143

Writes DATA/scan_<season>.parquet per season (a season on disk is skipped), then DATA/scan_summary.csv and
DATA/scan_counts.csv.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from openrapm.commands.console import one_thread, use_utf8

HERE = Path(__file__).resolve().parent
CHECKS = HERE.parents[1] / "scripts" / "checks"
STAND_IN_POSS = 500
ERAS = (("all", 1997, 2026), ("1997-2006", 1997, 2006), ("2007-2016", 2007, 2016), ("2017-2026", 2017, 2026))
CONTEXT = ("is_gt", "margin", "margin_frem")
FREE_RAW = (*CONTEXT, "rating_sum_off", "rating_sum_def", "playing_time", "playing_time_def", "stand_ins",
            "talent_spread_off", "talent_spread_def")
SHAPE = ("rating_curve_off", "rating_curve_def", "rating_cross")
FREE = (*FREE_RAW, *SHAPE)
ENDS = (("off", "O", "o"), ("def", "D", "d"))
PAIRS = (
    ("self_creation_per100", "three_point_attempts_per100"),       # a creator beside shooters
    ("self_creation_per100", "off_rebounds_per100_uncentred"),     # a creator beside an offensive rebounder
    ("self_creation_per100", "height_in"),                         # a creator beside size
    ("assists_per100_uncentred", "threes_made_per100_uncentred"),  # a passer beside shooters
    ("assists_to_rim_per100", "height_in"),                        # a passer to the rim beside size
    ("height_in", "three_point_attempts_per100"),                  # size beside shooters
    ("blocks_per100_uncentred", "steals_per100_uncentred"),        # a shot blocker beside a ball hawk
    ("def_rebounds_per100_uncentred", "blocks_per100_uncentred"),  # a rebounder beside a shot blocker
)
COUNT_BINS = ("0", "1", "2", "3+")


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def weighted_sd(values: np.ndarray, w: np.ndarray) -> float:
    mean = np.average(values, weights=w)
    return float(np.sqrt(np.average((values - mean) ** 2, weights=w)))


def weighted_quantile(values: np.ndarray, w: np.ndarray, q: float) -> float:
    order = np.argsort(values, kind="stable")
    middle = (np.cumsum(w[order]) - 0.5 * w[order]) / w.sum()
    return float(np.interp(q, middle, values[order]))


def fit(target: np.ndarray, columns: np.ndarray, w: np.ndarray) -> np.ndarray:
    root = np.sqrt(w)
    coef, *_ = np.linalg.lstsq(columns * root[:, None], target * root, rcond=None)
    return coef


class FoldValues:
    """One fold side's players: each statistic centered on the fold's possession-weighted average, the stand-in, the
    player standard deviation and the top-fifth line."""

    def __init__(self, players: pd.DataFrame, side: str, traits: tuple):
        rows = players[players.side == side].reset_index(drop=True)
        poss = rows["poss_off" if side == "O" else "poss_def"].to_numpy(float)
        keep = poss > 0
        rows, poss = rows[keep].reset_index(drop=True), poss[keep]
        self.index = pd.Index(rows.player_id.to_numpy())
        self.values, self.fill, self.sd, self.top = {}, {}, {}, {}
        low = poss < STAND_IN_POSS
        for trait in traits:
            raw = rows[trait].to_numpy(float)
            centred = raw - np.average(raw, weights=poss)
            self.values[trait] = centred
            self.fill[trait] = float(np.average(centred[low], weights=poss[low])) if low.any() else 0.0
            self.sd[trait] = weighted_sd(centred, poss)
            self.top[trait] = weighted_quantile(centred, poss, 0.8)

    def lineup(self, ids: np.ndarray, trait: str) -> np.ndarray:
        """n x 5: each player's centered value, the stand-in where the fold has none."""
        place = self.index.get_indexer(ids.ravel()).reshape(ids.shape)
        values = self.values[trait]
        return np.where(place >= 0, values[np.maximum(place, 0)], self.fill[trait])


def fold_columns(m154, stints: pd.DataFrame, players: pd.DataFrame, traits: tuple) -> tuple:
    """(projected columns {name: array}, raw columns for the scales and counts, player sds {(end, trait): sd})."""
    w = stints.w.to_numpy(float)
    raw = {name: stints[name].to_numpy(float) for name in FREE_RAW}
    offense = raw["rating_sum_off"] - np.average(raw["rating_sum_off"], weights=w)
    defense = raw["rating_sum_def"] - np.average(raw["rating_sum_def"], weights=w)
    raw["rating_curve_off"], raw["rating_curve_def"], raw["rating_cross"] = offense ** 2, defense ** 2, offense * defense
    raw["miss"] = stints.y.to_numpy(float) - (stints.rating_sum_off.to_numpy(float)
                                              + stints.rating_sum_def.to_numpy(float))
    sds, counts = {}, {}
    for end, side, prefix in ENDS:
        values = FoldValues(players, side, traits)
        ids = stints[[f"{prefix}{k}" for k in range(1, 6)]].to_numpy(np.int64)
        lineups = {}
        for trait in traits:
            a = values.lineup(ids, trait)
            lineups[trait] = a
            total, own = a.sum(axis=1), (a ** 2).sum(axis=1)
            raw[f"S|{end}|{trait}"], raw[f"Q|{end}|{trait}"] = total, own
            raw[f"C|{end}|{trait}"] = (total ** 2 - own) / 2.0
            sds[(end, trait)] = values.sd[trait]
            counts[(end, trait)] = (a >= values.top[trait]).sum(axis=1)
        for first, second in (pair for pair in PAIRS if pair[0] in lineups and pair[1] in lineups):
            a, b = lineups[first], lineups[second]
            own = (a * b).sum(axis=1)
            raw[f"P|{end}|{first}|{second}"] = own
            raw[f"X|{end}|{first}|{second}"] = a.sum(axis=1) * b.sum(axis=1) - own
    names = list(raw)
    levels = m154.team_levels(stints.rename(columns={}))
    projected = m154.project(levels, w, np.column_stack([raw[name] for name in names]))
    return dict(zip(names, projected.T)), raw, sds, counts


def season_scan(m154, season: int, data: Path, traits: tuple) -> pd.DataFrame:
    """Every reading of the season, one row per (analysis, end, trait, term)."""
    stints = pd.read_parquet(data / f"stints_{season}.parquet")
    players = pd.read_parquet(data / f"players_{season}.parquet")
    parts, raws, counts, sds = [], [], [], {}
    for (repeat, fold), group in stints.groupby(["repeat", "fold"], sort=True):
        group = group.reset_index(drop=True)
        mine = players[(players.repeat == repeat) & (players.fold == fold)]
        projected, raw, fold_sds, fold_counts = fold_columns(m154, group, mine, traits)
        parts.append(pd.DataFrame(projected))
        raws.append(pd.DataFrame({k: v for k, v in raw.items() if k.startswith(("S|", "C|", "X|"))}))
        counts.append(pd.DataFrame({f"N|{e}|{t}": v for (e, t), v in fold_counts.items()}))
        for key, value in fold_sds.items():
            sds.setdefault(key, []).append(value)
        parts[-1]["w"] = group.w.to_numpy(float)
    p = pd.concat(parts, ignore_index=True)
    raw = pd.concat(raws, ignore_index=True)
    count = pd.concat(counts, ignore_index=True)
    sd = {key: float(np.mean(values)) for key, values in sds.items()}
    w = p.w.to_numpy(float)
    target = p.miss.to_numpy(float)
    free = p[list(FREE)].to_numpy(float)
    rows = [dict(analysis="baseline", end="", trait="", term="error with team levels only",
                 value=float(np.sum(w * target ** 2) / w.sum()))]
    sign = {"off": 1.0, "def": -1.0}

    def lineup_sd(name):
        return weighted_sd(raw[name].to_numpy(float), w)

    # one statistic at a time, both ends
    for trait in traits:
        names = [f"{t}|{end}|{trait}" for end, _, _ in ENDS for t in ("S", "Q", "C")]
        coef = fit(target, np.hstack([p[names].to_numpy(float), free]), w)
        for k, (end, _, _) in enumerate(ENDS):
            lean, synergy = coef[3 * k], coef[3 * k + 2]
            s = sd[(end, trait)]
            rows += [dict(analysis="single", end=end, trait=trait, term="synergy per lineup sd",
                          value=sign[end] * synergy * lineup_sd(f"C|{end}|{trait}")),
                     dict(analysis="single", end=end, trait=trait, term="synergy per pair",
                          value=sign[end] * synergy * s * s),
                     dict(analysis="single", end=end, trait=trait, term="lean per player",
                          value=sign[end] * lean * s),
                     dict(analysis="single", end=end, trait=trait, term="player sd", value=s)]
    # pairs of different statistics, both ends
    for first, second in PAIRS:
        names = []
        for end, _, _ in ENDS:
            names += [f"{t}|{end}|{x}" for x in (first, second) for t in ("S", "Q", "C")]
            names += [f"P|{end}|{first}|{second}", f"X|{end}|{first}|{second}"]
        coef = fit(target, np.hstack([p[names].to_numpy(float), free]), w)
        per_end = len(names) // 2
        for k, (end, _, _) in enumerate(ENDS):
            cross = coef[per_end * k + per_end - 1]
            rows += [dict(analysis="pair", end=end, trait=f"{first} x {second}", term="synergy per lineup sd",
                          value=sign[end] * cross * lineup_sd(f"X|{end}|{first}|{second}")),
                     dict(analysis="pair", end=end, trait=f"{first} x {second}", term="synergy per pair",
                          value=sign[end] * cross * sd[(end, first)] * sd[(end, second)])]
    # every statistic at once
    names = [f"{t}|{end}|{trait}" for end, _, _ in ENDS for trait in traits for t in ("S", "Q", "C")]
    coef = fit(target, np.hstack([p[names].to_numpy(float), free]), w)
    for k, name in enumerate(names):
        term, end, trait = name.split("|")
        if term == "C":
            rows.append(dict(analysis="joint", end=end, trait=trait, term="synergy per lineup sd",
                             value=sign[end] * coef[k] * lineup_sd(name)))
        elif term == "S":
            rows.append(dict(analysis="joint", end=end, trait=trait, term="lean per player",
                             value=sign[end] * coef[k] * sd[(end, trait)]))
    # counts of top-fifth players, the miss after the free terms alone
    left = target - free @ fit(target, free, w)
    for end, _, _ in ENDS:
        for trait in traits:
            n = np.minimum(count[f"N|{end}|{trait}"].to_numpy(), 3)
            for k, label in enumerate(COUNT_BINS):
                inside = n == k
                rows += [dict(analysis="count", end=end, trait=trait, term=f"{label} share", value=float(w[inside].sum() / w.sum())),
                         dict(analysis="count", end=end, trait=trait, term=f"{label} mean",
                              value=float(sign[end] * np.sum(w[inside] * left[inside]) / w[inside].sum())
                              if inside.any() else np.nan)]
    out = pd.DataFrame(rows)
    out.insert(0, "season", season)
    return out


def across(values: pd.Series, newey_west) -> dict:
    n = len(values)
    mean = float(values.mean())
    se = float(values.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    se_nw = newey_west(values) if n > 2 else np.nan
    return dict(mean=mean, standard_error=se, z=mean / se if se > 0 else np.nan,
                z_newey_west=mean / se_nw if se_nw > 0 else np.nan,
                seasons_with_its_sign=int((np.sign(values) == np.sign(mean)).sum()), seasons=n)


def summarize(per_season: pd.DataFrame, newey_west) -> pd.DataFrame:
    rows = []
    keep = per_season[per_season.analysis.isin(["single", "pair", "joint"])]
    for (analysis, end, trait, term), group in keep.groupby(["analysis", "end", "trait", "term"], sort=False):
        values = group.set_index("season").value.sort_index()
        for label, first, last in ERAS:
            part = values[(values.index >= first) & (values.index <= last)]
            if len(part):
                rows.append(dict(analysis=analysis, end=end, trait=trait, term=term, era=label,
                                 **across(part, newey_west)))
    return pd.DataFrame(rows)


def count_table(per_season: pd.DataFrame) -> pd.DataFrame:
    """Each statistic's count table: the mean over seasons of each season's mean, by era."""
    part = per_season[per_season.analysis == "count"]
    rows = []
    for (end, trait, term), group in part.groupby(["end", "trait", "term"], sort=False):
        values = group.set_index("season").value
        for label, first, last in ERAS:
            inside = values[(values.index >= first) & (values.index <= last)].dropna()
            rows.append(dict(end=end, trait=trait, term=term, era=label, value=float(inside.mean()),
                             standard_error=float(inside.std(ddof=1) / np.sqrt(len(inside))) if len(inside) > 1
                             else np.nan, seasons=len(inside)))
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    use_utf8()
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--data", default="outputs/synergy/rd143")
    parser.add_argument("--first", type=int, default=1997)
    parser.add_argument("--last", type=int, default=2026)
    parser.add_argument("--check", default="outputs/csv/wowy_rd143_seasons.csv:rd143")
    args = parser.parse_args(argv)
    m154 = _load("selfcreation_check", CHECKS / "154_self_creation.py")
    wowy = m154._load("wowy_check", "151_wowy.py")
    traits = _load("synergy_extract", HERE / "extract.py").TRAITS
    data = Path(args.data)
    started = time.time()
    with one_thread():
        for season in range(args.first, args.last + 1):
            out = data / f"scan_{season}.parquet"
            if out.exists() or not (data / f"stints_{season}.parquet").exists():
                continue
            season_scan(m154, season, data, traits).to_parquet(out, index=False)
            print(f"  {season}: scanned ({time.time() - started:.0f}s)", flush=True)
    found = [data / f"scan_{s}.parquet" for s in range(args.first, args.last + 1)]
    per_season = pd.concat([pd.read_parquet(f) for f in found if f.exists()], ignore_index=True)
    if args.check:
        path, system = args.check.rsplit(":", 1)
        theirs = pd.read_csv(path)
        theirs = theirs[(theirs.system == system) & (theirs.stage == "step4") & (theirs.slice == "all")]
        mine = per_season[per_season.analysis == "baseline"].set_index("season").value
        diff = (mine - theirs.set_index("season").error).dropna()
        print(f"error with team levels only against check 151 ({system}): largest difference "
              f"{diff.abs().max():.2e} over {len(diff)} seasons")
    summarize(per_season, wowy.newey_west_se).to_csv(data / "scan_summary.csv", index=False)
    count_table(per_season).to_csv(data / "scan_counts.csv", index=False)
    print(f"done in {time.time() - started:.0f}s: {data / 'scan_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
