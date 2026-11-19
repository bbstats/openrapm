"""Read-only: experiment 143, read 4, trees on what Box Adj's line leaves, the step repeated (notes/DECISIONS.md,
"Experiment 143: the reads on Box Adj without age").

scratch/2026-11-10_boost_vs_line/boost_vs_line.py made one tree column on the targets at the line (check 167: 0.100
better, 21 of 30).  Here, from player_sums.py's per-player sums of today's folds and line (noage142_players):

    step k   at the weights of [the line | the tree columns of steps 1 .. k-1], solved jointly, each player's target
             g / h on each end (boost_vs_line.py's docstring), the slope now carrying the earlier tree columns too;
             ChimeraBoost at its defaults on those targets, weighted h; its output one more column an end
    scored   each rated season's error with the line and steps 1 .. k, every weight solved on the seasons more than one
             from it, each counted second_prior.season_weight(other, rated, 8), against the line alone

Seasons go in blocks of three; a block's trees are fitted on the seasons two or more from every season of the block,
each counted by its distance from the block's middle season, as are the weights its targets are taken at.  From step 2
on, a training season's earlier tree columns were made by models fitted on other blocks, which saw the block's own
seasons: the shuffled control, run the same way, says whether that matters.

    BOOST_SEED=100     the boosters' random draws moved
    BOOST_SHUFFLE=1    every target handed to another player of its season and end, at every step

Writes outputs/csv/reads143_trees<tag>_seasons.csv and reads143_trees<tag>.csv.
    .venv/Scripts/python scratch/2026-11-19_box_adj_reads/trees_steps.py
"""
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("NUMBA_NUM_THREADS", "4")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from chimeraboost import ChimeraBoostRegressor  # noqa: E402

from openrapm import rating_line, second_prior  # noqa: E402
from openrapm.data.config import load_config, resolve  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
SEED = int(os.environ.get("BOOST_SEED", "0"))
SHUFFLE = os.environ.get("BOOST_SHUFFLE", "0") == "1"
STEPS = int(os.environ.get("BOOST_STEPS", "5"))
TAG = ("_shuffled" if SHUFFLE else "") + (f"_seed{SEED}" if SEED else "")
F = rating_line.FITTED_PARTS
PENALTY = 1e-4
HALF_LIFE = 8.0
SEASONS = list(range(1997, 2027))
BLOCKS = [SEASONS[k:k + 3] for k in range(0, 30, 3)]
ERAS = (("1997-2006", 1997, 2006), ("2007-2016", 2007, 2016), ("2017-2026", 2017, 2026), ("all 30", 1997, 2026))
NAMES = second_prior.INPUT_SETS["context_types_no_age_team_zones_centred"]
K = len(NAMES)
LINE_COLUMNS = F + 2 * K
DIRECTORY = "noage142_players"


def raw_to_prior(ids: np.ndarray, side_ids: np.ndarray, team_poss: np.ndarray) -> np.ndarray:
    """(n x n_side): a number per player with inputs on his end of the prior, moved so each team's possession-weighted
    total is 0, as `rating_line.prior_columns` moves every input but the box-score prior."""
    frame = pd.DataFrame({"player_id": team_poss[:, 0].astype(np.int64), "team": team_poss[:, 1].astype(np.int64),
                          "poss": team_poss[:, 2]})
    moved_ids, moved = rating_line.team_free(np.eye(len(side_ids)), side_ids, frame)
    rows = pd.Series(np.arange(len(ids)), index=ids).reindex(moved_ids).to_numpy()
    found = ~np.isnan(rows)
    out = np.zeros((len(ids), len(side_ids)))
    out[rows[found].astype(int)] = moved[found]
    return out


class Fold:
    def __init__(self, path: Path, season: int):
        self.path, self.season = path, season
        with np.load(path) as data:
            Q, c, P, ids = data["Q"], data["c"], data["P"], data["ids"]
            self.yy, self.sw = float(data["yy"]), float(data["sw"])
            self.side_ids = {"O": data["ids_O"], "D": data["ids_D"]}
            self.X = {"O": data["X_O"], "D": data["X_D"]}
            self.maps = {side: raw_to_prior(ids, data[f"ids_{side}"], data[f"team_poss_{side}"]) for side in "OD"}
        n = self.n = len(ids)
        across = np.zeros((F + 2 * n, LINE_COLUMNS))
        across[:F, :F] = np.eye(F)
        across[F:, F:] = P
        self.line = rating_line.RatingSums(Q=across.T @ Q @ across, c=across.T @ c, yy=self.yy, sw=self.sw)
        q_wp, q_pp, c_p = Q[:F, F:], Q[F:, F:], c[F:]
        q_pp_p = q_pp @ P
        self.a0, self.m_w, self.m_b, self.h, self.R = {}, {}, {}, {}, {"O": [], "D": []}
        for half, side in enumerate("OD"):
            rows = slice(half * n, (half + 1) * n)
            a = self.maps[side]
            self.a0[side] = a.T @ c_p[rows]
            self.m_w[side] = a.T @ q_wp[:, rows].T
            self.m_b[side] = a.T @ q_pp_p[rows]
            self.h[side] = np.einsum("ij,ij->j", a, q_pp[rows, rows] @ a)
        self.added = np.zeros((2 * n, 0))

    def slope(self, side: str, weights: np.ndarray) -> np.ndarray:
        """g for every player with inputs on `side`, at `weights` over [the line | the tree columns so far]."""
        g = self.a0[side] - self.m_w[side] @ weights[:F] - self.m_b[side] @ weights[F:LINE_COLUMNS]
        for k, r in enumerate(self.R[side]):
            g = g - r @ weights[LINE_COLUMNS + 2 * k:LINE_COLUMNS + 2 * k + 2]
        return g

    def add(self, values: dict) -> rating_line.RatingSums:
        """One more tree column an end, `values` {side: a number per player with inputs}: the fold's sums over its
        two new columns against every column so far, laid out as the block to append."""
        with np.load(self.path) as data:
            Q, c, P = data["Q"], data["c"], data["P"]
        n = self.n
        new = np.zeros((2 * n, 2))
        new[:n, 0] = self.maps["O"] @ values["O"]
        new[n:, 1] = self.maps["D"] @ values["D"]
        q_new = Q[F:, F:] @ new
        for half, side in enumerate("OD"):
            self.R[side].append(self.maps[side].T @ q_new[half * n:(half + 1) * n])
        cross = np.vstack([Q[:F, F:] @ new, P.T @ q_new, self.added.T @ q_new])
        block = dict(cross=cross, self=new.T @ q_new, c=new.T @ c[F:])
        self.added = np.column_stack([self.added, new])
        return block


def grown(total: rating_line.RatingSums, blocks: list) -> rating_line.RatingSums:
    """`total` with two more columns, from the folds' blocks of `Fold.add`, summed."""
    size = total.Q.shape[0]
    cross = sum(b["cross"] for b in blocks)
    Q = np.zeros((size + 2, size + 2))
    Q[:size, :size], Q[:size, size:], Q[size:, :size] = total.Q, cross, cross.T
    Q[size:, size:] = sum(b["self"] for b in blocks)
    return rating_line.RatingSums(Q=Q, c=np.concatenate([total.c, sum(b["c"] for b in blocks)]), yy=total.yy,
                                  sw=total.sw)


def solved(total: rating_line.RatingSums, at: list) -> np.ndarray:
    weights = np.zeros(total.Q.shape[0])
    weights[at] = rating_line._solve(total, np.asarray(at), PENALTY)
    return weights


def faded(sums: dict, seasons: list, centre: int) -> rating_line.RatingSums:
    parts = [sums[s].scaled(second_prior.season_weight(s, centre, HALF_LIFE)) for s in seasons]
    return sum(parts[1:], parts[0])


def block_training(block: list) -> list:
    return [season for season in SEASONS if min(abs(season - inside) for inside in block) >= 2]


def targets(folds: dict, seasons: list, side: str, weights: np.ndarray, centre: int, step: int) -> pd.DataFrame:
    frames = []
    for season in seasons:
        rows = []
        for fold in folds[season]:
            frame = pd.DataFrame(fold.X[side], columns=NAMES)
            frame["player_id"], frame["g"], frame["h"] = fold.side_ids[side], fold.slope(side, weights), fold.h[side]
            rows.append(frame)
        every = pd.concat(rows, ignore_index=True)
        one = every.groupby("player_id").agg({**{name: "mean" for name in NAMES}, "g": "sum", "h": "sum"}).reset_index()
        frames.append(one.assign(season=season, fade=second_prior.season_weight(season, centre, HALF_LIFE)))
    table = pd.concat(frames, ignore_index=True)
    table = table[table.h > 0].reset_index(drop=True)
    table["target"] = table.g / table.h
    if SHUFFLE:
        rng = np.random.default_rng(1000 + SEED + 10 * step + (0 if side == "O" else 1) + centre)
        table["target"] = table.groupby("season").target.transform(lambda v: rng.permutation(v.to_numpy()))
    return table


def by_era(change: np.ndarray, label: str) -> list:
    seasons = np.array(SEASONS)
    rows = []
    for era, first, last in ERAS:
        part = change[(seasons >= first) & (seasons <= last)]
        error = part.std(ddof=1) / np.sqrt(len(part))
        rows.append(dict(reading=label, era=era, change=round(float(part.mean()), 4),
                         standard_error=round(float(error), 4), paired_z=round(float(part.mean() / error), 2),
                         seasons_better=int((part < 0).sum()), seasons=len(part)))
    return rows


def main() -> None:
    started = time.time()
    directory = resolve(load_config(), "outputs") / "second_prior" / DIRECTORY
    folds = {}
    for season in SEASONS:
        paths = sorted(directory.glob(f"*{season}_r*_f*.npz"))
        assert len(paths) == 12, (season, len(paths))
        folds[season] = [Fold(path, season) for path in paths]
    print(f"read 360 folds ({time.time() - started:.0f}s); tag '{TAG}'", flush=True)
    sums = {season: sum((f.line for f in folds[season][1:]), folds[season][0].line) for season in SEASONS}
    line_at = list(range(LINE_COLUMNS))

    def season_errors(at: list) -> np.ndarray:
        return np.array([sums[r].error(solved(faded(sums, [s for s in SEASONS if abs(s - r) > 1], r), at))
                         for r in SEASONS])

    line_err = season_errors(line_at)
    table = pd.DataFrame({"season": SEASONS, "line": line_err})
    summary, earned = [], []
    print(f"the line, 56 inputs an end: mean error {line_err.mean():.4f}", flush=True)
    previous = line_err
    for step in range(1, STEPS + 1):
        at = list(range(sums[SEASONS[0]].Q.shape[0]))
        values = {season: [dict() for _ in folds[season]] for season in SEASONS}
        for number, block in enumerate(BLOCKS):
            train, centre = block_training(block), block[1]
            weights = solved(faded(sums, train, centre), at)
            for side in "OD":
                rows = targets(folds, train, side, weights, centre, step)
                model = ChimeraBoostRegressor(random_state=number + SEED + 1000 * step, thread_count=4)
                model.fit(rows[NAMES].to_numpy(float), rows.target.to_numpy(float),
                          sample_weight=(rows.h * rows.fade).to_numpy(float), groups=rows.player_id.to_numpy())
                for season in block:
                    for k, fold in enumerate(folds[season]):
                        values[season][k][side] = np.asarray(model.predict(fold.X[side]), dtype=float)
        for season in SEASONS:
            sums[season] = grown(sums[season], [fold.add(values[season][k]) for k, fold in enumerate(folds[season])])
        err = season_errors(list(range(sums[SEASONS[0]].Q.shape[0])))
        table[f"line + {step} steps"] = err
        summary += by_era(err - line_err, f"line + {step} steps, against the line")
        summary += by_era(err - previous, f"step {step}, against step {step - 1}")
        weights = solved(faded(sums, [s for s in SEASONS if abs(s - 2026) > 1], 2026), list(range(sums[2026].Q.shape[0])))
        earned.append(dict(step=step, offense=weights[-2], defense=weights[-1]))
        print(f"step {step}: {(err - line_err).mean():+.4f} against the line, {(err - previous).mean():+.4f} against "
              f"step {step - 1}; the new column's weights for 2026, {weights[-2]:.3f} and {weights[-1]:.3f} "
              f"({time.time() - started:.0f}s)", flush=True)
        previous = err
    out = resolve(load_config(), "outputs") / "csv"
    table.to_csv(out / f"reads143_trees{TAG}_seasons.csv", index=False)
    frame = pd.DataFrame(summary)
    frame.to_csv(out / f"reads143_trees{TAG}.csv", index=False)
    pd.set_option("display.width", 200)
    print(frame.to_string(index=False))
    print(pd.DataFrame(earned).round(3).to_string(index=False))
    print(f"done in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
