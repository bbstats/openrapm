"""Check 160: which of the last-10 stats to keep in check 159's forecast of a team's next 10 games.

The owner, 2026-11-08: "smartly do feature selection".  notes/DECISIONS.md, "Check 160", holds the rule, written before
any result.

Words used below
    stat           one of check 159's 14 stats, on both ends: the team's and its opponents'.  It is kept or left whole
    copy           a stat dealt to random teams of the same season, its two ends together: same spread, no information
    importance     the square root of the sum of a stat's two squared coefficients, on columns of unit spread
    hit            a round in which a real stat's importance beats the largest copy's
    verdict        accepted, tentative or rejected, from a binomial test of the hits, as in Boruta (`thresholds`)
    held-out test  one season predicted from the other 29: everything chosen, the stats and the ridge penalty, is chosen
                   inside those 29

    python scripts/checks/160_power_next10_select.py [--out outputs/csv/check160]
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def load_159():
    """Check 159 as a module: its rows, its scoring and its Power."""
    spec = importlib.util.spec_from_file_location("check159", ROOT / "scripts" / "checks" / "159_power_next10.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["check159"] = module
    spec.loader.exec_module(module)
    return module


C159 = load_159()
from openrapm.data.config import load_config  # noqa: E402

ROUNDS = 100
SEED = 160
LEVEL = 0.05
BASE = 3  # the home edge, Power on offense, Power on defense
PENALTIES = C159.PENALTIES
ZONE_LABELS = {"rim": "Rim", "smid": "Short mid", "lmid": "Long mid", "thr": "Three", "lthr": "Long three"}
UNITS = {
    **{f"{ZONE_LABELS[z]} attempts": [f"attempts_{z}_o", f"attempts_{z}_d"] for z in C159.ZONES},
    **{f"{ZONE_LABELS[z]} makes over expected": [f"makes_over_{z}_o", f"makes_over_{z}_d"] for z in C159.ZONES},
    "Offensive rebounds over expected": ["orb_over_o", "orb_over_d"],
    "Turnovers": ["tov_o", "tov_d"],
    "Free throw attempts": ["ft_attempts_o", "ft_attempts_d"],
    "Free throws made over expected": ["ft_makes_over_o", "ft_makes_over_d"],
}
NAMES = list(UNITS)
COLUMNS = [column for columns in UNITS.values() for column in columns]
if set(COLUMNS) != set(C159.ADDED) or len(COLUMNS) != len(C159.ADDED):
    raise SystemExit("check 160's stats are not check 159's")
UNIT_COLUMNS = [np.array([BASE + 2 * k, BASE + 2 * k + 1]) for k in range(len(NAMES))]
ALL_IN = np.arange(len(NAMES))


def thresholds(rounds: int, stats: int, level: float = LEVEL) -> tuple[int, int]:
    """(accepted at this many hits or more, rejected at this many or fewer): a two-sided binomial test against one hit
    in two rounds at `level` / `stats`, so half of that in each tail.  100 rounds and 14 stats give 65 and 35."""
    alpha, tail, k = level / stats / 2, 0.0, rounds + 1
    while k > 0:
        extra = math.comb(rounds, k - 1) * 0.5**rounds
        if tail + extra > alpha:
            break
        tail += extra
        k -= 1
    return k, rounds - k


ACCEPT_AT, REJECT_AT = thresholds(ROUNDS, len(NAMES))


class Day:
    """One day's rows as arrays: `x` the standardized columns (base, then the stats' two ends), `y`, the season sums."""

    def __init__(self, team_rows: pd.DataFrame, game_rows: pd.DataFrame, prefix: str = ""):
        team_rows = team_rows.reset_index(drop=True)
        where = {key: i for i, key in enumerate(zip(team_rows.season, team_rows.tm))}
        self.me = np.array([where[key] for key in zip(game_rows.season, game_rows.tm)])
        self.opp = np.array([where[key] for key in zip(game_rows.season, game_rows.opp)])
        self.stats = team_rows[[prefix + column for column in COLUMNS]].to_numpy(float)
        power = team_rows[["power_o", "power_d"]].to_numpy(float)
        base = np.hstack([game_rows.home.to_numpy(float)[:, None], power[self.me] - power[self.opp]])
        raw = np.hstack([base, self.stats[self.me] - self.stats[self.opp]])
        self.spread = raw.std(axis=0)
        self.x = raw / self.spread
        self.y = game_rows.margin.to_numpy(float)
        self.season = game_rows.season.to_numpy()
        self.seasons = np.unique(self.season)
        self.game_rows = game_rows
        self.team_groups = list(team_rows.groupby("season").indices.values())
        self.xtx, self.xty, self.yy, self.rows = season_sums(self.x, self.y, self.season, self.seasons)
        self.masks = {s: self.season == s for s in self.seasons}


def season_sums(x, y, season, seasons) -> tuple[dict, dict, dict, dict]:
    """Each season's X'X, X'y, y'y and rows."""
    xtx, xty, yy, rows = {}, {}, {}, {}
    for s in seasons:
        mask = season == s
        xtx[s], xty[s], yy[s], rows[s] = (
            x[mask].T @ x[mask],
            x[mask].T @ y[mask],
            float(y[mask] @ y[mask]),
            int(mask.sum()),
        )
    return xtx, xty, yy, rows


def solve(a: np.ndarray, b: np.ndarray, rows: int, penalty: float, penalized: np.ndarray) -> np.ndarray:
    return np.linalg.solve(a + rows * penalty * np.diag(penalized.astype(float)) + 1e-9 * np.eye(a.shape[0]), b)


def columns_of(units) -> np.ndarray:
    """The base columns and the columns of `units`."""
    return np.concatenate([np.arange(BASE), *[UNIT_COLUMNS[k] for k in units]]) if len(units) else np.arange(BASE)


def left_out_error(day: Day, train: list, keep: np.ndarray, penalty: float) -> float:
    """The squared error on each season of `train` from a fit on the rest of `train`, summed."""
    a = sum(day.xtx[s][np.ix_(keep, keep)] for s in train)
    b = sum(day.xty[s][keep] for s in train)
    rows = sum(day.rows[s] for s in train)
    penalized = keep >= BASE
    total = 0.0
    for s in train:
        xtx, xty = day.xtx[s][np.ix_(keep, keep)], day.xty[s][keep]
        beta = solve(a - xtx, b - xty, rows - day.rows[s], penalty, penalized)
        total += day.yy[s] - 2 * beta @ xty + beta @ xtx @ beta
    return float(total)


def best_penalty(day: Day, train: list, keep: np.ndarray) -> float:
    """The penalty, of the grid, with the least left-out error on `train`; 0 when only the base columns are in."""
    if not (keep >= BASE).any():
        return 0.0
    errors = [left_out_error(day, train, keep, penalty) for penalty in PENALTIES]
    return float(PENALTIES[int(np.argmin(errors))])


def fitted(day: Day, train: list, keep: np.ndarray) -> np.ndarray:
    """The coefficients on `train`, the penalty picked inside it."""
    a = sum(day.xtx[s][np.ix_(keep, keep)] for s in train)
    b = sum(day.xty[s][keep] for s in train)
    rows = sum(day.rows[s] for s in train)
    return solve(a, b, rows, best_penalty(day, train, keep), keep >= BASE)


def screen(day: Day) -> dict:
    """The shuffled-copy screen for every held-out test and for all seasons (key None): each gets the stats' hits, mean
    importance and the mean largest copy."""
    rng = np.random.default_rng(SEED)
    tests = [*day.seasons, None]
    trains = {test: [s for s in day.seasons if s != test] for test in tests}
    real = BASE + len(COLUMNS)
    penalties = {test: best_penalty(day, trains[test], np.arange(real)) for test in tests}
    penalized = np.array([False] * BASE + [True] * (2 * len(COLUMNS)))
    out = {test: dict(hits=np.zeros(len(NAMES)), importance=np.zeros(len(NAMES)), largest=0.0) for test in tests}
    pairs = [np.array([2 * k, 2 * k + 1]) for k in range(len(NAMES))]
    for _ in range(ROUNDS):
        copies = day.stats.copy()
        for group in day.team_groups:
            for pair in pairs:
                copies[np.ix_(group, pair)] = day.stats[np.ix_(rng.permutation(group), pair)]
        x = np.hstack([day.x, (copies[day.me] - copies[day.opp]) / day.spread[BASE:]])
        xtx, xty, _, rows = season_sums(x, day.y, day.season, day.seasons)
        total_a, total_b, total_rows = sum(xtx.values()), sum(xty.values()), sum(rows.values())
        for test in tests:
            a, b, n = total_a, total_b, total_rows
            if test is not None:
                a, b, n = a - xtx[test], b - xty[test], n - rows[test]
            beta = solve(a, b, n, penalties[test], penalized)
            stats = beta[BASE:]
            importance = np.array([np.hypot(*stats[pair]) for pair in pairs])
            shadows = np.array([np.hypot(*stats[len(COLUMNS) + pair]) for pair in pairs])
            out[test]["hits"] += importance > shadows.max()
            out[test]["importance"] += importance / ROUNDS
            out[test]["largest"] += shadows.max() / ROUNDS
    return out


def verdict(hits: float) -> str:
    return "accepted" if hits >= ACCEPT_AT else "rejected" if hits <= REJECT_AT else "tentative"


def forward(day: Day, train: list) -> list:
    """Stats added one at a time, each the one that most lowers the left-out error on `train`, until none lowers it.
    The penalty stays the one all 14 stats take on `train`."""
    penalty = best_penalty(day, train, np.arange(BASE + len(COLUMNS)))
    chosen: list = []
    best = left_out_error(day, train, columns_of(chosen), penalty)
    while len(chosen) < len(NAMES):
        trials = {
            k: left_out_error(day, train, columns_of(sorted([*chosen, k])), penalty) for k in ALL_IN if k not in chosen
        }
        k = min(trials, key=trials.get)
        if trials[k] >= best:
            break
        best = trials[k]
        chosen.append(k)
    return sorted(chosen)


def backward(day: Day, train: list) -> list:
    """All 14 stats, then the stat whose removal most lowers the left-out error on `train` taken out, until no removal
    lowers it.  The penalty stays the one all 14 stats take on `train`."""
    penalty = best_penalty(day, train, np.arange(BASE + len(COLUMNS)))
    kept = list(ALL_IN)
    best = left_out_error(day, train, columns_of(kept), penalty)
    while kept:
        trials = {k: left_out_error(day, train, columns_of([j for j in kept if j != k]), penalty) for k in kept}
        k = min(trials, key=trials.get)
        if trials[k] >= best:
            break
        best = trials[k]
        kept.remove(k)
    return sorted(int(k) for k in kept)


def predictions(day: Day, choose) -> tuple[np.ndarray, dict]:
    """Every held-out season's predictions, the stats chosen by `choose(test)`, and what each test chose."""
    out, chosen = np.zeros(len(day.y)), {}
    for test in day.seasons:
        train = [s for s in day.seasons if s != test]
        units = chosen[test] = list(choose(test, train))
        keep = columns_of(units)
        beta = fitted(day, train, keep)
        out[day.masks[test]] = day.x[day.masks[test]][:, keep] @ beta
    return out, chosen


def score_day(day: Day, control: Day) -> tuple[pd.DataFrame, dict, dict]:
    """One row per method and season for one day, the screen's output for the real stats, and the stats each control
    method kept per test."""
    found, control_found = screen(day), screen(control)

    def kept(results, test, accepted_only=False):
        return [
            k
            for k, hits in enumerate(results[test]["hits"])
            if hits >= ACCEPT_AT or (not accepted_only and hits > REJECT_AT)
        ]

    methods = {
        "Power at its own scale": (day, lambda test, train: []),
        "Power + all 14 stats": (day, lambda test, train: ALL_IN),
        "Power + screen: accepted and tentative": (day, lambda test, train: kept(found, test)),
        "Power + screen: accepted only": (day, lambda test, train: kept(found, test, True)),
        "Power + forward selection": (day, lambda test, train: forward(day, train)),
        "Power + backward elimination": (day, lambda test, train: backward(day, train)),
        "control: screen, wrong teams": (control, lambda test, train: kept(control_found, test)),
        "control: forward, wrong teams": (control, lambda test, train: forward(control, train)),
        "control: backward, wrong teams": (control, lambda test, train: backward(control, train)),
    }
    frames, counts = [], {}
    for name, (data, choose) in methods.items():
        made, chosen = predictions(data, choose)
        frames.append(C159.score_frame(name, data.game_rows, data.y, made))
        counts[name] = chosen
    return pd.concat(frames, ignore_index=True), found, counts


def drop_each(day: Day) -> pd.DataFrame:
    """Each stat dropped from the 14, change in the held-out squared error per season: below 0 is better without it."""
    full, _ = predictions(day, lambda test, train: ALL_IN)
    base = C159.score_frame("all", day.game_rows, day.y, full).set_index("season").game
    rows = []
    for k, name in enumerate(NAMES):
        others = [j for j in ALL_IN if j != k]
        made, _ = predictions(day, lambda test, train: others)
        diff = C159.score_frame(name, day.game_rows, day.y, made).set_index("season").game - base
        se = float(diff.std(ddof=1) / np.sqrt(len(diff)))
        rows.append(
            dict(
                stat=name,
                drop_change=float(diff.mean()),
                drop_se=se,
                drop_z=float(diff.mean() / se) if se > 0 else 0.0,
                drop_better=int((diff < 0).sum()),
            )
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="outputs/csv/check160")
    args = parser.parse_args()
    started = time.time()
    cfg = load_config()
    within = ROOT / "outputs" / "within"
    wowy = C159.load_wowy()
    tables = {season: C159.season_game_table(season, cfg) for season in C159.SEASONS}
    stints = {season: C159.season_stints_ids(season, cfg) for season in C159.SEASONS}
    teams = {season: C159.regular_game_teams(season, cfg) for season in C159.SEASONS}
    print(f"accepted at {ACCEPT_AT}+ hits of {ROUNDS}, rejected at {REJECT_AT} or fewer", flush=True)
    results, against_all, today_day, today_found, today_counts = [], [], None, None, None
    for cut, folder in C159.CUTS.items():
        team_rows, game_rows = C159.day_rows(cut, C159.day_ratings(wowy, within, folder), tables, stints, teams)
        team_rows = C159.with_shuffled(team_rows)
        day, control = Day(team_rows, game_rows), Day(team_rows, game_rows, "shuffled_")
        scores, found, counts = score_day(day, control)
        table = C159.compared(scores).assign(cut=cut)
        table["mean_stats_kept"] = table.model.map(lambda m: float(np.mean([len(v) for v in counts[m].values()])))
        results.append(table)
        against_all.append(C159.compared(scores, "Power + all 14 stats").assign(cut=cut))
        shown = table[table.era == "all seasons"]
        print(f"\n{cut} ({time.time() - started:.0f}s)", flush=True)
        print(
            shown[
                [
                    "model",
                    "game_rmse",
                    "team_rmse",
                    "game_change",
                    "game_z",
                    "game_better",
                    "team_change",
                    "team_z",
                    "mean_stats_kept",
                ]
            ]
            .round(4)
            .to_string(index=False),
            flush=True,
        )
        if cut == C159.TODAY_CUT:
            today_day, today_found, today_counts = day, found, counts
    result = pd.concat(results, ignore_index=True)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(f"{out}_compared.csv", index=False)
    versus = pd.concat(against_all, ignore_index=True)
    versus.to_csv(f"{out}_against_all14.csv", index=False)
    print("\nagainst all 14 stats, all seasons", flush=True)
    print(
        versus[versus.era == "all seasons"][
            ["cut", "model", "game_change", "game_z", "game_better", "team_change", "team_z"]
        ]
        .round(4)
        .to_string(index=False),
        flush=True,
    )
    names = [
        "Power + all 14 stats",
        "Power + screen: accepted and tentative",
        "Power + screen: accepted only",
        "Power + forward selection",
        "Power + backward elimination",
        "control: screen, wrong teams",
        "control: forward, wrong teams",
        "control: backward, wrong teams",
    ]
    print("\nby era, the 10% day", flush=True)
    part = result[(result.cut == C159.TODAY_CUT) & result.model.isin(names)]
    print(
        part[["era", "model", "game_rmse", "game_change", "game_z", "game_better", "team_change", "team_z"]]
        .round(4)
        .to_string(index=False),
        flush=True,
    )

    day = today_day
    final = today_found[None]
    tests = [s for s in day.seasons]
    screen_keep = today_counts["Power + screen: accepted and tentative"]
    forward_keep = today_counts["Power + forward selection"]
    backward_keep = today_counts["Power + backward elimination"]
    backward_final = backward(day, list(day.seasons))
    kept_units = [k for k, hits in enumerate(final["hits"]) if hits > REJECT_AT]
    table = pd.DataFrame(
        {
            "stat": NAMES,
            "hits": final["hits"].astype(int),
            "verdict": [verdict(h) for h in final["hits"]],
            "importance": final["importance"],
            "largest_copy": final["largest"],
            "screen_kept_tests": [sum(k in screen_keep[s] for s in tests) for k in ALL_IN],
            "forward_kept_tests": [sum(k in forward_keep[s] for s in tests) for k in ALL_IN],
            "backward_kept_tests": [sum(k in backward_keep[s] for s in tests) for k in ALL_IN],
            "backward_final": [k in backward_final for k in ALL_IN],
        }
    )

    def weights_of(units) -> pd.Series:
        keep = columns_of(units)
        beta = fitted(day, list(day.seasons), keep) / day.spread[keep]
        return pd.Series(beta, index=[("home", "power_o", "power_d", *COLUMNS)[i] for i in keep])

    all_weights, backward_weights = weights_of(ALL_IN), weights_of(backward_final)
    for label, weights in (("all14", all_weights), ("backward", backward_weights)):
        table[f"{label}_offense_weight"] = [float(weights.get(UNITS[n][0], np.nan)) for n in NAMES]
        table[f"{label}_opponent_weight"] = [float(weights.get(UNITS[n][1], np.nan)) for n in NAMES]
    table = table.merge(drop_each(day), on="stat")
    table.to_csv(f"{out}_verdicts.csv", index=False)
    print("\nevery stat, the 10% day", flush=True)
    print(table.round(4).to_string(index=False), flush=True)
    print(f"\nscreen kept: {[NAMES[k] for k in kept_units]}", flush=True)
    print(f"backward kept on all 30 seasons: {[NAMES[k] for k in backward_final]}", flush=True)

    now = C159.today_teams(cfg)
    centered = now[C159.ADDED + ["power_o", "power_d"]] - now[C159.ADDED + ["power_o", "power_d"]].mean()
    for label, weights in (("backward", backward_weights), ("all14", all_weights)):
        parts = pd.DataFrame({"power_part": centered.power_o * weights.power_o + centered.power_d * weights.power_d})
        for group, columns in C159.GROUPS.items():
            present = [column for column in columns if column in weights.index]
            parts[group] = (centered[present] * weights[present]).sum(axis=1) if present else 0.0
        parts["new_total"] = parts.sum(axis=1)
        parts["power"] = now.power_o + now.power_d
        parts = parts.sort_values("new_total", ascending=False)
        parts.to_csv(f"{out}_today_{label}.csv")
        print(f"\n{label}, multipliers {weights.power_o:.3f} {weights.power_d:.3f}", flush=True)
        print(parts.round(2).to_string(), flush=True)
    print(f"done in {time.time() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
