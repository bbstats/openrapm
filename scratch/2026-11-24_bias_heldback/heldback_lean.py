"""The Bias page's leans on held-back games: does any of the page's 13 box stats mark players OpenRAPM over- or
under-credits, on games the ratings never saw?

The owner, 2026-11-24: "What lessons can we take from the biases we see?  Credit error to players?  Or is this
team/coach level credit we can't confidently give (or can't confidently deduct)".  Read-only; no rating moves.  The
rule was registered in notes/DECISIONS.md, "The Bias page's leans on held-back games", before any result.

The games: every held-back stint of the incumbent's within-season test, saved by scratch/2026-11-23_synergy/extract.py
in outputs/synergy/rd143/stints_<season>.parquet, both fives, the miss being points per 100 minus the ten players'
ratings after the shipped shrink.

The stat: from each fold's own fit games only (outputs/within/rd143_early or rd143_within,
players_<season>_r<repeat>_f<fold>.parquet; never the season's _all file, which holds the held-back games).  A
player's rate is the mean of his offense and defense rows, as bias.player_stats takes it, in standard deviations of
the fold's players with 60 or more offensive possessions, weighted by them, as the page standardizes.  A player the
fold has no row for counts at 0, the mean.  A five's value is the sum of its five players'.

The fit, per season and stat: weighted least squares on possessions of the miss on the offense five's sum and the
defense five's sum, team levels per fold fitted out of every column first, with one of three sets of free terms
(`RUNS`).  Sign as the page's, OpenRAPM minus the games: offense is minus the offense five's coefficient (a positive
coefficient means the games score those lineups higher than their ratings), defense is the defense five's coefficient
(a positive one means they allow more than their ratings say), total is their sum.

The shuffled copy: check 154's career null.  Every player is handed another player's standing in the stat, season by
season along that player's career (the donor's last season when his career is shorter); in a fold the donated
standing is standardized again on the fold's shown players.  The same draws for every season.

    .venv/Scripts/python scratch/2026-11-24_bias_heldback/heldback_lean.py --only=1997:2004
    ... (one process per range) ...
    .venv/Scripts/python scratch/2026-11-24_bias_heldback/heldback_lean.py --summary

Writes OUT/real_<season>.parquet and OUT/null_<season>.parquet (a season on disk is skipped), then with --summary
OUT/summary.csv and OUT/seasons.csv.
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
from openrapm.features import BOX_RATES

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CHECKS = ROOT / "scripts" / "checks"
SYNERGY = ROOT / "scratch" / "2026-11-23_synergy"
SEED = 20261124
MIN_SHOWN = 60
# The page's 13 box rates, then its two plus-minus rows, his team's points scored minus allowed per 100 with him on the
# court and off it (bias.player_stats); a fold rebuilds both from its fit games.  The rule was registered on the 13.
PLUS_MINUS = ("on_court_plus_minus_per100", "off_court_plus_minus_per100")
COURT = ("on_court_scored_per100", "on_court_allowed_per100", "off_court_scored_per100", "off_court_allowed_per100")
STATS = (*BOX_RATES, *PLUS_MINUS)
ERAS = (("all", 1997, 2026), ("1997-2006", 1997, 2006), ("2007-2016", 2007, 2016), ("2017-2026", 2017, 2026))
CONTEXT = ("is_gt", "margin", "margin_frem")
TIME = ("playing_time", "playing_time_def")
FROM_RATINGS = ("rating_sum_off", "rating_sum_def", "talent_spread_off", "talent_spread_def",
                "rating_curve_off", "rating_curve_def", "rating_cross")
RUNS = {
    "width free": (*CONTEXT, *TIME, "stand_ins", *FROM_RATINGS),
    "own width": (*CONTEXT, *TIME, "stand_ins"),
    "width free, no playing time": (*CONTEXT, "stand_ins", *FROM_RATINGS),
}
ALL_FREE = tuple(dict.fromkeys(c for columns in RUNS.values() for c in columns))
ENDS = (("offense", "o"), ("defense", "d"))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def weighted_mean(values: np.ndarray, w: np.ndarray) -> np.ndarray:
    return (values * w[:, None]).sum(axis=0) / w.sum()


def standardize(values: np.ndarray, poss: np.ndarray) -> np.ndarray:
    """players x stats: each column in standard deviations of the players with `MIN_SHOWN` or more offensive
    possessions, weighted by them."""
    shown = poss >= MIN_SHOWN
    w = poss[shown]
    centre = weighted_mean(values[shown], w)
    spread = np.sqrt(weighted_mean((values[shown] - centre) ** 2, w))
    return (values - centre) / np.where(spread > 0, spread, 1.0)


def fold_file(within: Path, season: int, repeat: int, fold: int) -> Path:
    name = f"players_{season}_r{repeat}_f{fold}.parquet"
    for folder in ("rd143_early", "rd143_within"):
        if (within / folder / name).exists():
            return within / folder / name
    raise FileNotFoundError(name)


def fold_rates(within: Path, season: int, repeat: int, fold: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(player ids, players x stats raw rates as the mean of his two rows, offensive possessions), from the fold's fit
    games."""
    rows = pd.read_parquet(fold_file(within, season, repeat, fold),
                           columns=["player_id", "side", "poss_off", *BOX_RATES, *COURT])
    rates = rows.groupby("player_id")[[*BOX_RATES, *COURT]].mean()
    rates["on_court_plus_minus_per100"] = rates.on_court_scored_per100 - rates.on_court_allowed_per100
    rates["off_court_plus_minus_per100"] = rates.off_court_scored_per100 - rates.off_court_allowed_per100
    rates = rates[list(STATS)]
    poss = rows.groupby("player_id").poss_off.max().reindex(rates.index)
    return rates.index.to_numpy(np.int64), rates.to_numpy(float), poss.to_numpy(float)


class Careers:
    """Every player's standing in each stat, season by season: his offensive-possession-weighted mean over the
    season's folds, in standard deviations of the season's players weighted by their summed possessions."""

    def __init__(self, within: Path, first: int, last: int):
        frames = []
        for season in range(first, last + 1):
            parts = []
            for repeat in range(3):
                for fold in range(4):
                    try:
                        ids, rates, poss = fold_rates(within, season, repeat, fold)
                    except FileNotFoundError:
                        continue
                    parts.append(pd.DataFrame(rates * poss[:, None], columns=STATS).assign(player_id=ids, poss=poss))
            table = pd.concat(parts).groupby("player_id").sum()
            table = table[table.poss > 0]
            mean = table[list(STATS)].div(table.poss, axis=0)
            w = table.poss.to_numpy(float)
            centre = weighted_mean(mean.to_numpy(), w)
            spread = np.sqrt(weighted_mean((mean.to_numpy() - centre) ** 2, w))
            standing = (mean - centre) / spread
            frames.append(standing.assign(season=season).reset_index())
        table = pd.concat(frames, ignore_index=True).sort_values(["player_id", "season"], kind="stable")
        table["step"] = table.groupby("player_id").cumcount()
        self.players = np.sort(table.player_id.unique())
        self.place = pd.Index(self.players)
        self.step = {(p, s): k for p, s, k in table[["player_id", "season", "step"]].itertuples(index=False)}
        self.length = (table.groupby("player_id").step.max() + 1).reindex(self.players).to_numpy(int)
        grid = np.full((len(self.players), int(self.length.max()), len(STATS)), np.nan)
        grid[self.place.get_indexer(table.player_id), table.step.to_numpy(int)] = table[list(STATS)].to_numpy(float)
        self.values = grid

    def donated(self, permutation: np.ndarray, players: np.ndarray, season: int) -> np.ndarray:
        """players x stats: each player's donor's standing at the same step of the donor's career."""
        place = self.place.get_indexer(players)
        step = np.array([self.step.get((p, season), 0) for p in players])
        donor = permutation[place]
        at = np.minimum(step, self.length[donor] - 1)
        return np.nan_to_num(self.values[donor, at], nan=0.0)


def lineup_sum(z: np.ndarray, where: np.ndarray) -> np.ndarray:
    """n x stats: the five's summed z, 0 for a player the fold has no row for."""
    picked = z[np.maximum(where, 0)]
    picked[where < 0] = 0.0
    return picked.sum(axis=1)


def season_folds(m154, data: Path, within: Path, season: int) -> list:
    """One dict per fold: its projector, the projected miss and free terms, the lineup places, the raw rates and
    possessions, and its player ids."""
    stints = pd.read_parquet(data / f"stints_{season}.parquet")
    folds = []
    for (repeat, fold), group in stints.groupby(["repeat", "fold"], sort=True):
        group = group.reset_index(drop=True)
        w = group.w.to_numpy(float)
        raw = {name: group[name].to_numpy(float) for name in ALL_FREE if name in group.columns}
        offense = raw["rating_sum_off"] - np.average(raw["rating_sum_off"], weights=w)
        defense = raw["rating_sum_def"] - np.average(raw["rating_sum_def"], weights=w)
        raw["rating_curve_off"], raw["rating_curve_def"], raw["rating_cross"] = offense ** 2, defense ** 2, offense * defense
        miss = group.y.to_numpy(float) - (group.rating_sum_off.to_numpy(float) + group.rating_sum_def.to_numpy(float))
        project = m154.projector(m154.team_levels(group), w)
        ids, rates, poss = fold_rates(within, season, int(repeat), int(fold))
        index = pd.Index(ids)
        where = {end: index.get_indexer(group[[f"{p}{k}" for k in range(1, 6)]].to_numpy(np.int64).ravel())
                 .reshape(-1, 5) for end, p in ENDS}
        folds.append(dict(
            w=w, project=project, miss=project(miss[:, None])[:, 0], raw_miss=miss,
            free=pd.DataFrame(project(np.column_stack([raw[name] for name in ALL_FREE])), columns=ALL_FREE),
            where=where, rates=rates, poss=poss, ids=ids))
    return folds


def leans(folds: list, z_of_fold) -> dict:
    """{(run, end, stat): the season's lean in the page's sign}: `z_of_fold(k, fold)` gives fold k's players x stats z."""
    columns = []
    for k, fold in enumerate(folds):
        z = z_of_fold(k, fold)
        block = np.hstack([lineup_sum(z, fold["where"][end]) for end, _ in ENDS])
        columns.append(fold["project"](block))
    x_all = np.vstack(columns)
    w = np.concatenate([f["w"] for f in folds])
    target = np.concatenate([f["miss"] for f in folds])
    free = pd.concat([f["free"] for f in folds], ignore_index=True)
    out = {}
    n = len(STATS)
    xw_all = x_all * w[:, None]
    for run, names in RUNS.items():
        f = free[list(names)].to_numpy(float)
        fw = f * w[:, None]
        ff, fy = fw.T @ f, fw.T @ target
        xf = xw_all.T @ f
        xy = xw_all.T @ target
        for s, stat in enumerate(STATS):
            picks = [s, n + s]
            x = x_all[:, picks]
            gram = np.block([[xw_all[:, picks].T @ x, xf[picks]], [xf[picks].T, ff]])
            coef = np.linalg.lstsq(gram, np.concatenate([xy[picks], fy]), rcond=None)[0]
            offense, defense = -coef[0], coef[1]
            out[(run, "offense", stat)] = float(offense)
            out[(run, "defense", stat)] = float(defense)
            out[(run, "total", stat)] = float(offense + defense)
    return out


def frame(values: dict, **extra) -> pd.DataFrame:
    return pd.DataFrame([dict(**extra, run=r, end=e, stat=s, value=v) for (r, e, s), v in values.items()])


def season_read(m154, careers, data: Path, within: Path, season: int, permutations: np.ndarray,
                out: Path) -> None:
    folds = season_folds(m154, data, within, season)
    w = np.concatenate([f["w"] for f in folds])
    miss = np.concatenate([f["miss"] for f in folds])
    baseline = float(np.sum(w * miss ** 2) / w.sum())
    real = leans(folds, lambda k, fold: standardize(fold["rates"], fold["poss"]))
    frame(real, season=season).assign(baseline=baseline).to_parquet(out / f"real_{season}.parquet", index=False)
    players = np.unique(np.concatenate([f["ids"] for f in folds]))
    rows = []
    for draw, permutation in enumerate(permutations):
        donated = pd.DataFrame(careers.donated(permutation, players, season), index=players)
        values = leans(folds, lambda k, fold: standardize(donated.loc[fold["ids"]].to_numpy(), fold["poss"]))
        rows.append(frame(values, season=season, draw=draw))
    pd.concat(rows, ignore_index=True).to_parquet(out / f"null_{season}.parquet", index=False)


def summary(out: Path, first: int, last: int, check: str) -> None:
    real = pd.concat([pd.read_parquet(out / f"real_{s}.parquet") for s in range(first, last + 1)], ignore_index=True)
    null = pd.concat([pd.read_parquet(out / f"null_{s}.parquet") for s in range(first, last + 1)], ignore_index=True)
    if check:
        path, system = check.rsplit(":", 1)
        theirs = pd.read_csv(path)
        theirs = theirs[(theirs.system == system) & (theirs.stage == "step4") & (theirs.slice == "all")]
        mine = real.drop_duplicates("season").set_index("season").baseline
        diff = (mine - theirs.set_index("season").error).dropna()
        print(f"error with team levels only against check 151 ({system}): largest difference "
              f"{diff.abs().max():.2e} over {len(diff)} seasons")
    real.to_csv(out / "seasons.csv", index=False)
    rows = []
    for (run, end, stat), group in real.groupby(["run", "end", "stat"], sort=False):
        values = group.set_index("season").value.sort_index()
        draws = null[(null.run == run) & (null.end == end) & (null.stat == stat)] \
            .pivot(index="draw", columns="season", values="value")
        row = dict(run=run, end=end, stat=stat)
        for label, lo, hi in ERAS:
            part = values[(values.index >= lo) & (values.index <= hi)]
            means = draws[[s for s in draws.columns if lo <= s <= hi]].mean(axis=1).to_numpy()
            row[f"{label} mean"] = float(part.mean())
            row[f"{label} standard error"] = float(part.std(ddof=1) / np.sqrt(len(part)))
            row[f"{label} null mean"] = float(means.mean())
            row[f"{label} null spread"] = float(means.std(ddof=1))
        row["seasons with its sign"] = int((np.sign(values) == np.sign(values.mean())).sum())
        row["seasons"] = len(values)
        rows.append(row)
    table = pd.DataFrame(rows)
    table["z"] = table["all mean"] / table["all standard error"]
    table["z against null"] = (table["all mean"] - table["all null mean"]) / table["all null spread"]
    table["same sign in every era"] = (np.sign(table[["1997-2006 mean", "2007-2016 mean", "2017-2026 mean"]])
                                       .nunique(axis=1) == 1)
    table["null centred"] = table["all null mean"].abs() < 0.5 * table["all null spread"]
    table["beyond 2"] = (table.z.abs() > 2) & (table["z against null"].abs() > 2)
    table.to_csv(out / "summary.csv", index=False)
    print(f"summary: {out / 'summary.csv'}")


def main(argv=None) -> int:
    use_utf8()
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--data", default="outputs/synergy/rd143")
    parser.add_argument("--within", default="outputs/within")
    parser.add_argument("--out", default="outputs/bias_heldback")
    parser.add_argument("--draws", type=int, default=200)
    parser.add_argument("--first", type=int, default=1997)
    parser.add_argument("--last", type=int, default=2026)
    parser.add_argument("--only", default="", help="first:last, the seasons this process reads")
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--check", default="outputs/csv/wowy_rd143_seasons.csv:rd143")
    args = parser.parse_args(argv)
    data, within, out = Path(args.data), Path(args.within), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.summary:
        summary(out, args.first, args.last, args.check)
        return 0
    m154 = _load("selfcreation_check", CHECKS / "154_self_creation.py")
    lo, hi = (int(x) for x in args.only.split(":")) if args.only else (args.first, args.last)
    started = time.time()
    with one_thread():
        careers = Careers(within, args.first, args.last)
        rng = np.random.default_rng(SEED)
        permutations = np.array([rng.permutation(len(careers.players)) for _ in range(args.draws)])
        print(f"careers of {len(careers.players):,} players ({time.time() - started:.0f}s)", flush=True)
        for season in range(lo, hi + 1):
            if (out / f"real_{season}.parquet").exists() and (out / f"null_{season}.parquet").exists():
                continue
            season_read(m154, careers, data, within, season, permutations, out)
            print(f"  {season}: done ({time.time() - started:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
