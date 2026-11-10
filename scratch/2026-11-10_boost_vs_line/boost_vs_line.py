"""Read-only: for Box Adj, boosted trees at ChimeraBoost's default settings against the straight line.

Box Adj's weights are a straight line in 57 inputs an end, solved on the within-season test itself
(src/openrapm/rating_line.py).  A tree model cannot be solved that way, so it is given the same job in the one form a
booster takes: a number per player to forecast.  From the per-player sums of player_sums.py, at a base prior,

    g_i = how much the test's error falls per unit added to player i's prior on an end     (half the slope)
    h_i = the curvature of the error in that one number
    target r_i = g_i / h_i, weight h_i

which is the step each player's prior should take if he were alone: what a second stage is fitted on.  Summed over a
season's folds, one row per player, season and end.

Arms, each scored on the real test through the per-player sums, exactly as the line is:

    box only                 the fitted parts and each end's box-score prior: the base
    line                     the shipped line, 57 inputs an end, solved jointly: today's Box Adj
    box + line, one step     a weighted least-squares line on the targets at the base, its output one more input
    box + trees              ChimeraBoost at its defaults on the same rows, targets and weights, its output one more
                             input
    line + trees             ChimeraBoost on the targets at the LINE's own solution, added to the line

"One step" and "trees" are fitted on identical rows, so that pair is learner against learner; "line" is what the joint
solve gets, which a one-step fit may fall short of; "line + trees" asks whether trees find anything the line left.

No season's column is made by a model that saw it: seasons go in blocks of three, and a block's models train on the
seasons two or more away from every season of the block.  A column's weight is then solved like any input's, on the
seasons more than one from the rated one.

Two controls, by environment variable, each writing its own files (the tag joins the names):
    BOOST_SEED=100       the boosters' random draws moved by that much: the same check again
    BOOST_SHUFFLE=1      every target handed to another player of its season and end, weights and inputs left: what
                         the arms read when there is nothing to learn

Writes outputs/csv/boost_vs_line<tag>_seasons.csv and outputs/csv/boost_vs_line<tag>.csv.
    .venv/Scripts/python scratch/2026-11-10_boost_vs_line/boost_vs_line.py
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
TAG = ("_shuffled" if SHUFFLE else "") + (f"_seed{SEED}" if SEED else "")
F = rating_line.FITTED_PARTS
PENALTY = 1e-4
SEASONS = list(range(1997, 2027))
BLOCKS = [SEASONS[k:k + 3] for k in range(0, 30, 3)]
ERAS = (("1997-2006", 1997, 2006), ("2007-2016", 2007, 2016), ("2017-2026", 2017, 2026), ("all 30", 1997, 2026))
NAMES = second_prior.INPUT_SETS["context_types_team_zones_centred"]
K = len(NAMES)
BOX = NAMES.index("box_prior")
LINE_COLUMNS = F + 2 * K
# the candidate columns after the line's: each a pair, offence's then defence's
CANDIDATES = ("one step", "trees", "trees on the line")
ARMS = {
    "box only": [0, 1, F + BOX, F + K + BOX],
    "line": list(range(LINE_COLUMNS)),
    "box + line, one step": [0, 1, F + BOX, F + K + BOX, LINE_COLUMNS, LINE_COLUMNS + 1],
    "box + trees": [0, 1, F + BOX, F + K + BOX, LINE_COLUMNS + 2, LINE_COLUMNS + 3],
    "line + trees": [*range(LINE_COLUMNS), LINE_COLUMNS + 4, LINE_COLUMNS + 5],
}


def raw_to_prior(ids: np.ndarray, side_ids: np.ndarray, team_poss: np.ndarray) -> np.ndarray:
    """(n x n_side): what a number per player with inputs becomes on his end of the prior, moved so each team's
    possession-weighted total is 0, as `rating_line.prior_columns` moves every input but the box-score prior."""
    frame = pd.DataFrame({"player_id": team_poss[:, 0].astype(np.int64), "team": team_poss[:, 1].astype(np.int64),
                          "poss": team_poss[:, 2]})
    moved_ids, moved = rating_line.team_free(np.eye(len(side_ids)), side_ids, frame)
    rows = pd.Series(np.arange(len(ids)), index=ids).reindex(moved_ids).to_numpy()
    found = ~np.isnan(rows)
    out = np.zeros((len(ids), len(side_ids)))
    out[rows[found].astype(int)] = moved[found]
    return out


class Fold:
    """One fold's small arrays, read from its file once (`first_pass`)."""

    def __init__(self, path: Path, season: int):
        self.path, self.season = path, season
        with np.load(path) as data:
            Q, c, P, ids = data["Q"], data["c"], data["P"], data["ids"]
            self.yy, self.sw = float(data["yy"]), float(data["sw"])
            self.side_ids = {"O": data["ids_O"], "D": data["ids_D"]}
            self.X = {"O": data["X_O"], "D": data["X_D"]}
            maps = self.maps(data)
        n = len(ids)
        self.n = n
        q_wp, q_pp, c_p = Q[:F, F:], Q[F:, F:], c[F:]
        # the check of `raw_to_prior`: the line's own inputs other than the box-score prior are this map of the raw ones
        for half, side in enumerate(("O", "D")):
            other = [k for k in range(K) if k != BOX]
            rebuilt = maps[side] @ self.X[side][:, other]
            stored = P[half * n:(half + 1) * n, [half * K + k for k in other]]
            gap = float(np.abs(rebuilt - stored).max())
            assert gap < 1e-9, (path.name, side, gap)
        across = np.zeros((F + 2 * n, LINE_COLUMNS))
        across[:F, :F] = np.eye(F)
        across[F:, F:] = P
        self.line = rating_line.RatingSums(Q=across.T @ Q @ across, c=across.T @ c, yy=self.yy, sw=self.sw)
        q_pp_p = q_pp @ P
        self.a0, self.m_w, self.m_b, self.h = {}, {}, {}, {}
        for half, side in enumerate(("O", "D")):
            rows = slice(half * n, (half + 1) * n)
            a = maps[side]
            self.a0[side] = a.T @ c_p[rows]
            self.m_w[side] = a.T @ q_wp[:, rows].T
            self.m_b[side] = a.T @ q_pp_p[rows]
            self.h[side] = np.einsum("ij,ij->j", a, q_pp[rows, rows] @ a)

    def maps(self, data) -> dict:
        """{"O", "D"}: `raw_to_prior` for each end, from the fold's file; too large to keep for every fold."""
        return {side: raw_to_prior(data["ids"], data[f"ids_{side}"], data[f"team_poss_{side}"]) for side in ("O", "D")}

    def slope(self, side: str, weights: np.ndarray) -> np.ndarray:
        """g for every player with inputs on `side`, at the line weights `weights` (zeros where an arm has none)."""
        return self.a0[side] - self.m_w[side] @ weights[:F] - self.m_b[side] @ weights[F:LINE_COLUMNS]

    def extended(self, columns: dict) -> rating_line.RatingSums:
        """The fold's sums over [the line's columns | each candidate's two], `columns` {candidate: {side: values}}."""
        with np.load(self.path) as data:
            Q, c, P = data["Q"], data["c"], data["P"]
            maps = self.maps(data)
        n = self.n
        added = np.zeros((2 * n, 2 * len(CANDIDATES)))
        for at, name in enumerate(CANDIDATES):
            added[:n, 2 * at] = maps["O"] @ columns[name]["O"]
            added[n:, 2 * at + 1] = maps["D"] @ columns[name]["D"]
        q_added = Q[F:, F:] @ added
        size = LINE_COLUMNS + added.shape[1]
        out_q, out_c = np.zeros((size, size)), np.zeros(size)
        out_q[:LINE_COLUMNS, :LINE_COLUMNS], out_c[:LINE_COLUMNS] = self.line.Q, self.line.c
        cross = np.vstack([Q[:F, F:] @ added, P.T @ q_added])
        out_q[:LINE_COLUMNS, LINE_COLUMNS:], out_q[LINE_COLUMNS:, :LINE_COLUMNS] = cross, cross.T
        out_q[LINE_COLUMNS:, LINE_COLUMNS:] = added.T @ q_added
        out_c[LINE_COLUMNS:] = added.T @ c[F:]
        return rating_line.RatingSums(Q=out_q, c=out_c, yy=self.yy, sw=self.sw)


def solved(total: rating_line.RatingSums, at: list) -> np.ndarray:
    """The ridge of `rating_line._solve` on the columns `at`, zeros elsewhere."""
    weights = np.zeros(total.Q.shape[0])
    weights[at] = rating_line._solve(total, np.asarray(at), PENALTY)
    return weights


def added_up(parts: list) -> rating_line.RatingSums:
    return sum(parts[1:], parts[0])


def block_training(block: list) -> list:
    """The seasons two or more from every season of `block`."""
    return [season for season in SEASONS if min(abs(season - inside) for inside in block) >= 2]


def targets(folds: dict, seasons: list, side: str, weights: np.ndarray) -> pd.DataFrame:
    """One row per player and season on `side`: the target, its weight and his inputs averaged over the folds."""
    frames = []
    for season in seasons:
        rows = []
        for fold in folds[season]:
            frame = pd.DataFrame(fold.X[side], columns=NAMES)
            frame["player_id"], frame["g"], frame["h"] = fold.side_ids[side], fold.slope(side, weights), fold.h[side]
            rows.append(frame)
        every = pd.concat(rows, ignore_index=True)
        one = every.groupby("player_id").agg({**{name: "mean" for name in NAMES}, "g": "sum", "h": "sum"}).reset_index()
        frames.append(one.assign(season=season))
    table = pd.concat(frames, ignore_index=True)
    table = table[table.h > 0].reset_index(drop=True)
    table["target"] = table.g / table.h
    if SHUFFLE:
        rng = np.random.default_rng(1000 + SEED + (0 if side == "O" else 1))
        table["target"] = table.groupby("season").target.transform(lambda values: rng.permutation(values.to_numpy()))
    return table


def one_step_line(table: pd.DataFrame):
    """Weighted least squares of the target on the inputs, each standardised, a level free: x -> forecast."""
    x, weight = table[NAMES].to_numpy(float), table.h.to_numpy(float)
    mean = np.average(x, axis=0, weights=weight)
    spread = np.sqrt(np.average((x - mean) ** 2, axis=0, weights=weight))
    spread[spread == 0] = 1.0
    design = np.column_stack([np.ones(len(x)), (x - mean) / spread])
    gram = design.T @ (design * weight[:, None])
    gram[np.diag_indices_from(gram)] += PENALTY * float(np.mean(np.diag(gram)))
    beta = np.linalg.solve(gram, design.T @ (weight * table.target.to_numpy(float)))
    return lambda rows: np.column_stack([np.ones(len(rows)), (rows - mean) / spread]) @ beta


def trees(table: pd.DataFrame, seed: int):
    model = ChimeraBoostRegressor(random_state=seed, thread_count=4)
    model.fit(table[NAMES].to_numpy(float), table.target.to_numpy(float), sample_weight=table.h.to_numpy(float),
              groups=table.player_id.to_numpy())
    return model.predict


def read_folds(directory: Path, started: float) -> dict:
    """{season: its 12 folds}."""
    folds = {}
    for season in SEASONS:
        paths = sorted(directory.glob(f"*{season}_r*_f*.npz"))
        assert len(paths) == 12, (season, len(paths))
        folds[season] = [Fold(path, season) for path in paths]
        if season % 5 == 0:
            print(f"  read {season} ({time.time() - started:.0f}s)", flush=True)
    return folds


def block_columns(folds: dict, line_sums: dict, started: float) -> dict:
    """{season: [one {candidate: {side: a number per player}} per fold]}: each block's models, fitted on the seasons
    two or more away, forecasting the block's own folds."""
    columns = {season: [dict() for _ in folds[season]] for season in SEASONS}
    notes = []
    for number, block in enumerate(BLOCKS):
        train = block_training(block)
        total = added_up([line_sums[season] for season in train])
        base = {"box only": solved(total, ARMS["box only"]), "line": solved(total, ARMS["line"])}
        for side in ("O", "D"):
            at_base = targets(folds, train, side, base["box only"])
            at_line = targets(folds, train, side, base["line"])
            makers = {"one step": one_step_line(at_base), "trees": trees(at_base, number + SEED),
                      "trees on the line": trees(at_line, number + SEED)}
            notes.append(dict(block=f"{block[0]}-{block[-1]}", side=side, rows=len(at_base),
                              target_sd_base=float(np.sqrt(np.average(at_base.target ** 2, weights=at_base.h))),
                              target_sd_line=float(np.sqrt(np.average(at_line.target ** 2, weights=at_line.h)))))
            for season in block:
                for at, fold in enumerate(folds[season]):
                    for name, maker in makers.items():
                        columns[season][at].setdefault(name, {})[side] = np.asarray(maker(fold.X[side]), dtype=float)
        print(f"  block {block[0]}-{block[-1]}: models fitted on {len(train)} seasons ({time.time() - started:.0f}s)",
              flush=True)
    print(pd.DataFrame(notes).round(3).to_string(index=False), flush=True)
    return columns


def scores(extended: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(each rated season's error under every arm, the weight each added column earned): every arm's weights solved
    on the seasons more than one from the rated one."""
    rows, shares = [], []
    for season in SEASONS:
        total = added_up([extended[other] for other in SEASONS if abs(other - season) > 1])
        record = {"season": season}
        for arm, at in ARMS.items():
            weights = solved(total, at)
            record[arm] = extended[season].error(weights)
            if at[-1] >= LINE_COLUMNS:                     # an arm with an added column on each end
                shares.append(dict(season=season, arm=arm, offence=weights[at[-2]], defence=weights[at[-1]]))
        rows.append(record)
    return pd.DataFrame(rows), pd.DataFrame(shares)


def by_era(table: pd.DataFrame) -> pd.DataFrame:
    """Each arm minus the line, paired by season, by era."""
    summary = []
    for arm in (arm for arm in ARMS if arm != "line"):
        for era, first, last in ERAS:
            change = (table[arm] - table["line"])[(table.season >= first) & (table.season <= last)].to_numpy()
            error = change.std(ddof=1) / np.sqrt(len(change))
            summary.append(dict(arm=arm, era=era, change=round(float(change.mean()), 4),
                                standard_error=round(float(error), 4), paired_z=round(float(change.mean() / error), 2),
                                seasons_better=int((change < 0).sum()), seasons=len(change)))
    return pd.DataFrame(summary)


def main() -> None:
    started = time.time()
    directory = resolve(load_config(), "outputs") / "second_prior" / "zones139_players"
    folds = read_folds(directory, started)
    line_sums = {season: added_up([fold.line for fold in folds[season]]) for season in SEASONS}
    missing = int(sum(np.isnan(fold.X[side]).sum() for season in SEASONS for fold in folds[season] for side in "OD"))
    print(f"read {sum(len(v) for v in folds.values())} folds ({time.time() - started:.0f}s); inputs missing: {missing}",
          flush=True)
    columns = block_columns(folds, line_sums, started)
    extended = {season: added_up([fold.extended(columns[season][at]) for at, fold in enumerate(folds[season])])
                for season in SEASONS}
    print(f"the candidates' columns scored through the per-player sums ({time.time() - started:.0f}s)", flush=True)
    table, shares = scores(extended)
    table.to_csv(f"outputs/csv/boost_vs_line{TAG}_seasons.csv", index=False)
    summary = by_era(table)
    summary.to_csv(f"outputs/csv/boost_vs_line{TAG}.csv", index=False)
    pd.set_option("display.width", 200)
    print(f"\nthe line, today's Box Adj: mean error {table['line'].mean():.4f}; "
          f"box only {table['box only'].mean():.4f}")
    print("each arm minus the line, season by season; below 0 is better than the line")
    print(summary.to_string(index=False))
    print("\nthe weight each added column earned, mean over the 30 rated seasons")
    print(shares.groupby("arm")[["offence", "defence"]].mean().round(3).to_string())
    print(f"done in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
