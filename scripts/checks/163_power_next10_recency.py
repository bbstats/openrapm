"""Check 163: the best recency weighting over a team's last 10 games, for Power's shares and for the last-10 stats.

The owner, 2026-11-08: "find best recency weighting, and only use most recent 10. for both parts".  notes/DECISIONS.md,
"Check 163", holds the rule, written before any result.

Words used below
    decay       a game counts decay^k, k the games before the team's latest game.  1.00 is a flat last 10.  At 0.80 the
                latest game counts 1, the one before 0.80, the one before that 0.64, and the tenth 0.13
    step        today's Power page: the latest 5 games count 2, the 5 before them 1
    pair        a decay for the shares and a decay for the stats
    one-SE pair of the pairs whose error is within one standard error of the lowest error's, paired by season, the
                flattest: the largest sum of the two decays
    held-out test  one season predicted from the others: the pair and the ridge penalty are chosen inside the others

    POWER_NEXT10_FAMILY=zones139 python scripts/checks/163_power_next10_recency.py [--out outputs/csv/check163]
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def load_161():
    """Check 161 as a module; checks 160 and 159 come with it."""
    spec = importlib.util.spec_from_file_location("check161", ROOT / "scripts" / "checks" / "161_power_next10_audit.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["check161"] = module
    spec.loader.exec_module(module)
    return module


C161 = load_161()
C160, C159 = C161.C160, C161.C159
from openrapm.commands.site import RECENT_GAMES, RECENT_WEIGHT, team_shares  # noqa: E402
from openrapm.data.config import load_config, resolve  # noqa: E402

DECAYS = [1.0, 0.97, 0.94, 0.91, 0.88, 0.85, 0.80, 0.75, 0.70, 0.60, 0.50]
STEP = "step"
SEED = 163
WINDOW = C159.LAST_GAMES
GRID = [(stats, shares) for stats in DECAYS for shares in DECAYS]
TODAY_ARM, DOUBLED_ARM = (1.0, STEP), (STEP, STEP)
FLAT_ARM = (1.0, 1.0)
ALL_STATS = C160.columns_of(C160.ALL_IN)
POWER = "Power alone, today's shares"
ARMS = {
    "today's shares, plain stats": TODAY_ARM,
    "today's shares, doubled stats": DOUBLED_ARM,
    "flat shares, flat stats": FLAT_ARM,
}
CHOSEN = ["one-SE pair", "lowest-error pair", "one shared decay"]


def weights_for(decay) -> np.ndarray:
    """The weight of a game by how many games before the team's latest it is."""
    if decay == STEP:
        return np.where(np.arange(WINDOW) < RECENT_GAMES, RECENT_WEIGHT, 1.0)
    return float(decay) ** np.arange(WINDOW)


def team_orders(games: pd.DataFrame, rng) -> dict:
    """{tm: a random order of the team's window of games}: the noise control's deal of the last 10 games."""
    counts = games.groupby("tm").game_id.nunique().clip(upper=WINDOW)
    return {tm: rng.permutation(int(count)) for tm, count in counts.items()}


def game_weighter(decay, orders=None):
    """`last_games_stats`'s `game_weights`: the weights by games back, through a random order when `orders` is given."""
    table = weights_for(decay)

    def weigh(tm: np.ndarray, back: np.ndarray) -> np.ndarray:
        if orders is None:
            return table[back]
        return table[[orders[t][b] for t, b in zip(tm, back)]]

    return weigh


def weighted_shares(games: pd.DataFrame, weigh) -> pd.DataFrame:
    """site.team_shares with the weights of `weigh`: tm, player_id, share (5.0 summed over a team)."""
    team_games = games.groupby(["tm", "game_id"], as_index=False).agg(
        game_date=("game_date", "first"), poss=("poss", "sum")
    )
    team_games["poss"] /= 5
    last = team_games.sort_values(["tm", "game_date", "game_id"]).groupby("tm").tail(WINDOW).copy()
    back = last.groupby("tm").cumcount(ascending=False).to_numpy()
    last["weight"] = weigh(last.tm.to_numpy(), back)
    last["poss"] *= last.weight
    recent = games.merge(last[["tm", "game_id", "weight"]], on=["tm", "game_id"])
    recent["poss"] *= recent.weight
    totals = last.groupby("tm").poss.sum().rename("team_poss")
    players = recent.groupby(["tm", "player_id"], as_index=False).poss.sum().join(totals, on="tm")
    players["share"] = players.poss / players.team_poss
    return players[["tm", "player_id", "share"]]


def power_from(shares: pd.DataFrame, rated: pd.DataFrame) -> pd.DataFrame:
    """Each team's Power from `shares`, centered over the teams: power_o, power_d, indexed by tm."""
    joined = shares.join(rated.set_index("player_id")[["rating_off", "rating_def"]], on="player_id").fillna(0.0)
    power = pd.DataFrame(
        {"tm": joined.tm, "power_o": joined.share * joined.rating_off, "power_d": joined.share * joined.rating_def}
    )
    power = power.groupby("tm").sum()
    return power - power.mean()


def day_tables(cut: str, ratings: dict, tables: dict, stints: dict, teams: dict, shuffle: bool) -> tuple:
    """({decay: stats by (season, tm)}, {decay: Power by (season, tm)}, game rows) of one day."""
    rng = np.random.default_rng(SEED)
    decays = [*DECAYS, STEP]
    stats, power = {d: [] for d in decays}, {d: [] for d in decays}
    game_rows = []
    for season in C159.SEASONS:
        rated, played_ids = ratings[season]
        table = tables[season]
        played = table[table.game_id.isin(played_ids)]
        games = C159.player_game_possessions(stints[season][stints[season].game_id.isin(played_ids)], teams[season])
        orders = team_orders(games, rng) if shuffle else None
        for decay in decays:
            frame = C159.last_games_stats(played, game_weighter(decay, orders))
            stats[decay].append(frame.assign(season=season).set_index("season", append=True).swaplevel())
            shares = weighted_shares(games, game_weighter(decay, orders))
            frame = power_from(shares, rated)
            power[decay].append(frame.assign(season=season).set_index("season", append=True).swaplevel())
        day = played.game_date.max()
        later = table[table.game_date > day].sort_values(["tm", "game_date", "game_id"], kind="stable")
        next_games = later.groupby("tm").head(C159.NEXT_GAMES)
        game_rows.append(next_games[["tm", "opp", "home", "margin", "game_id"]].assign(season=season))
    return (
        {d: pd.concat(v) for d, v in stats.items()},
        {d: pd.concat(v) for d, v in power.items()},
        pd.concat(game_rows, ignore_index=True),
    )


def make_day(stats: pd.DataFrame, power: pd.DataFrame, game_rows: pd.DataFrame):
    """Check 160's Day for one pair."""
    rows = pd.concat([stats[C159.ADDED], power], axis=1).reset_index().rename(columns={"level_0": "season"})
    rows.columns = ["season", "tm", *rows.columns[2:]]
    return C160.Day(rows, game_rows)


def left_out_errors(day, train: list, penalty: float) -> np.ndarray:
    """Each season of `train`'s mean squared error from a fit on the rest of `train`."""
    keep = ALL_STATS
    a = sum(day.xtx[s][np.ix_(keep, keep)] for s in train)
    b = sum(day.xty[s][keep] for s in train)
    rows = sum(day.rows[s] for s in train)
    penalized = keep >= C160.BASE
    out = np.zeros(len(train))
    for k, s in enumerate(train):
        xtx, xty = day.xtx[s][np.ix_(keep, keep)], day.xty[s][keep]
        beta = C160.solve(a - xtx, b - xty, rows - day.rows[s], penalty, penalized)
        out[k] = (day.yy[s] - 2 * beta @ xty + beta @ xtx @ beta) / day.rows[s]
    return out


def grid_errors(days: dict, train: list) -> dict:
    """{pair: each `train` season's left-out mean squared error at the penalty chosen on `train`}."""
    return {pair: left_out_errors(day, train, C160.best_penalty(day, train, ALL_STATS)) for pair, day in days.items()}


def choose(errors: dict) -> dict:
    """The three chosen arms' pairs from `errors`."""
    means = {pair: float(values.mean()) for pair, values in errors.items()}
    best = min(means, key=means.get)
    near = []
    for pair, values in errors.items():
        gap = values - errors[best]
        if gap.mean() <= gap.std(ddof=1) / np.sqrt(len(gap)):
            near.append(pair)
    flattest = max(near, key=lambda pair: (pair[0] + pair[1], -means[pair]))
    shared = min([pair for pair in means if pair[0] == pair[1]], key=means.get)
    return {"one-SE pair": flattest, "lowest-error pair": best, "one shared decay": shared}


def held_out(days: dict, arms: dict, train_of) -> tuple[dict, dict]:
    """Predictions for every test season of `train_of`, for `arms` and the chosen arms, and the pairs chosen."""
    reference = days[TODAY_ARM]
    tests = [s for s in reference.seasons if train_of(s)]
    grid = {pair: days[pair] for pair in GRID}
    made = {name: np.full(len(reference.y), np.nan) for name in [POWER, *arms, *CHOSEN]}
    picked: dict = {name: {} for name in CHOSEN}
    for test in tests:
        train = train_of(test)
        mask = reference.masks[test]
        power_keep = C160.columns_of([])
        made[POWER][mask] = reference.x[mask][:, power_keep] @ C160.fitted(reference, train, power_keep)
        for name, pair in arms.items():
            made[name][mask] = days[pair].x[mask][:, ALL_STATS] @ C160.fitted(days[pair], train, ALL_STATS)
        chosen = choose(grid_errors(grid, train))
        for name, pair in chosen.items():
            picked[name][test] = pair
            made[name][mask] = days[pair].x[mask][:, ALL_STATS] @ C160.fitted(days[pair], train, ALL_STATS)
    return made, picked


def scores_of(reference, made: dict, keep_seasons=None) -> pd.DataFrame:
    """Check 159's squared errors by season for each arm."""
    frames = []
    for name, predicted in made.items():
        mask = ~np.isnan(predicted)
        rows = reference.game_rows[mask]
        frames.append(C159.score_frame(name, rows, reference.y[mask], predicted[mask]))
    return pd.concat(frames, ignore_index=True)


def walk_table(scores: pd.DataFrame) -> pd.DataFrame:
    """Each arm minus today's shares with plain stats, on seasons predicted from earlier seasons alone."""
    wide = scores.pivot(index="season", columns="model", values="game")
    rows = []
    for era, (first, last) in {
        "2007-2026": (2007, 2026),
        **{k: v for k, v in C159.ERAS.items() if v[0] >= 2007},
    }.items():
        part = wide.loc[first:last]
        for model in wide.columns:
            diff = part[model] - part["today's shares, plain stats"]
            se = float(diff.std(ddof=1) / np.sqrt(len(diff)))
            rows.append(
                dict(
                    era=era,
                    model=model,
                    change=float(diff.mean()),
                    se=se,
                    z=float(diff.mean() / se) if se else 0.0,
                    seasons_better=int((diff < 0).sum()),
                    seasons=len(diff),
                )
            )
    return pd.DataFrame(rows)


def run_day(cut: str, ratings: dict, tables: dict, stints: dict, teams: dict, shuffle: bool) -> dict:
    """Everything one day yields, for the real order of games or the control's."""
    stats, power, game_rows = day_tables(cut, ratings, tables, stints, teams, shuffle)
    pairs = {*GRID, TODAY_ARM, DOUBLED_ARM, FLAT_ARM}
    days = {pair: make_day(stats[pair[0]], power[pair[1]], game_rows) for pair in pairs}
    reference = days[TODAY_ARM]
    seasons = list(reference.seasons)
    made, picked = held_out(days, ARMS, lambda test: [s for s in seasons if s != test])
    scores = scores_of(reference, made)
    table = C159.compared(scores, POWER).assign(cut=cut)
    against = C159.compared(scores, "today's shares, plain stats").assign(cut=cut)
    walk_made, walk_picked = held_out(
        days, ARMS, lambda test: [s for s in seasons if s < test] if test >= C161.WALK_FIRST else []
    )
    walk = walk_table(scores_of(reference, walk_made)).assign(cut=cut)
    return dict(
        days=days,
        stats=stats,
        power=power,
        table=table,
        against=against,
        walk=walk,
        picked=picked,
        walk_picked=walk_picked,
        game_rows=game_rows,
        seasons=seasons,
    )


def verify_shares(ratings: dict, stints: dict, teams: dict) -> float:
    """The largest gap, over a season's players, between `weighted_shares` at the step and site.team_shares."""
    season = 2026
    _, played = ratings[season]
    games = C159.player_game_possessions(stints[season][stints[season].game_id.isin(played)], teams[season])
    mine = weighted_shares(games, game_weighter(STEP)).set_index(["tm", "player_id"]).share
    theirs = team_shares(games, WINDOW).set_index(["tm", "player_id"]).share
    return float((mine - theirs.reindex(mine.index)).abs().max())


def today_table(result: dict, final: tuple, weights: pd.Series, cfg: dict) -> pd.DataFrame:
    """2027 through today at the final pair: scaled Power, the four groups, the rating; and today's arm's rating."""
    table = C159.season_game_table(2027, cfg)
    published = pd.read_parquet(
        resolve(cfg, "outputs") / "season_ratings_product.parquet",
        columns=["season", "player_id", "rating_off", "rating_def"],
    )
    rated = published[published.season == 2027]
    stints = C159.season_stints_ids(2027, cfg)
    teams = C159.regular_game_teams(2027, cfg)
    games = C159.player_game_possessions(stints, teams)
    stats = C159.last_games_stats(table, game_weighter(final[0]))
    power = power_from(weighted_shares(games, game_weighter(final[1])), rated)
    power_today = power_from(weighted_shares(games, game_weighter(STEP)), rated)
    stats = stats[C159.ADDED]
    out = pd.DataFrame({"power_part": power.power_o * weights.power_o + power.power_d * weights.power_d})
    for group, columns in C159.GROUPS.items():
        out[group] = (stats[columns] * weights[columns]).sum(axis=1)
    out["new_total"] = out.sum(axis=1)
    out["power"] = power_today.sum(axis=1)
    return out.sort_values("new_total", ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="outputs/csv/check163")
    args = parser.parse_args()
    started = time.time()
    cfg = load_config()
    within = ROOT / "outputs" / "within"
    wowy = C159.load_wowy()
    tables = {season: C159.season_game_table(season, cfg) for season in C159.SEASONS}
    stints = {season: C159.season_stints_ids(season, cfg) for season in C159.SEASONS}
    teams = {season: C159.regular_game_teams(season, cfg) for season in C159.SEASONS}
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"ratings: {C159.FAMILY}", flush=True)
    tables_out, against_out, walk_out, control_out, today_result = [], [], [], [], None
    for cut, folder in C159.CUTS.items():
        ratings = C159.day_ratings(wowy, within, folder)
        if cut == C159.TODAY_CUT:
            print(
                f"the step against site.team_shares, largest gap: {verify_shares(ratings, stints, teams):.2e}",
                flush=True,
            )
        result = run_day(cut, ratings, tables, stints, teams, shuffle=False)
        print(f"{cut} real ({time.time() - started:.0f}s)", flush=True)
        control = run_day(cut, ratings, tables, stints, teams, shuffle=True)
        print(f"{cut} control ({time.time() - started:.0f}s)", flush=True)
        tables_out.append(result["table"])
        against_out.append(result["against"])
        walk_out.append(result["walk"])
        control_out.append(control["against"].assign(arm="control"))
        if cut == C159.TODAY_CUT:
            today_result = result
        del control
    table, against, walk, noise = (
        pd.concat(frames, ignore_index=True) for frames in (tables_out, against_out, walk_out, control_out)
    )
    table.to_csv(f"{out}_against_power.csv", index=False)
    against.to_csv(f"{out}_against_today.csv", index=False)
    walk.to_csv(f"{out}_walk.csv", index=False)
    noise.to_csv(f"{out}_control.csv", index=False)
    shown = [
        "cut",
        "era",
        "model",
        "game_rmse",
        "team_rmse",
        "game_change",
        "game_z",
        "game_better",
        "team_change",
        "team_z",
    ]
    print("\nagainst today's shares with plain stats", flush=True)
    print(against[against.model != POWER][shown].round(3).to_string(index=False), flush=True)
    print("\nagainst Power alone", flush=True)
    print(table[table.era == "all seasons"][shown].round(3).to_string(index=False), flush=True)
    print("\ncontrol: the last 10 games in a random order, against today's shares with plain stats", flush=True)
    print(
        noise[(noise.model.isin(CHOSEN)) & (noise.era == "all seasons")][shown].round(3).to_string(index=False),
        flush=True,
    )
    print("\neach season predicted only from the seasons before it", flush=True)
    print(walk.round(3).to_string(index=False), flush=True)

    result = today_result
    for name, picks in result["picked"].items():
        counts = pd.Series(list(picks.values())).value_counts().head(8)
        print(
            f"\n{name}, pairs chosen in the 30 held-out tests, (stats decay, shares decay):\n{counts.to_string()}",
            flush=True,
        )
    grid = {pair: result["days"][pair] for pair in GRID}
    errors = grid_errors(grid, result["seasons"])
    today_errors = left_out_errors(
        result["days"][TODAY_ARM],
        result["seasons"],
        C160.best_penalty(result["days"][TODAY_ARM], result["seasons"], ALL_STATS),
    )
    surface = pd.DataFrame(
        [
            dict(stats_decay=pair[0], shares_decay=pair[1], change=float((values - today_errors).mean()))
            for pair, values in errors.items()
        ]
    )
    surface.to_csv(f"{out}_grid.csv", index=False)
    print(
        "\nthe grid, change in left-out squared error against today's arm: rows stats decay, columns shares decay",
        flush=True,
    )
    print(
        surface.pivot(index="stats_decay", columns="shares_decay", values="change")
        .sort_index(ascending=False)
        .sort_index(axis=1, ascending=False)
        .round(2)
        .to_string(),
        flush=True,
    )
    final = choose(errors)
    print(f"\nthe final pairs on all 30 seasons, (stats decay, shares decay): {final}", flush=True)
    pair = final["one-SE pair"]
    day = result["days"][pair]
    names = ["home", "power_o", "power_d", *C160.COLUMNS]
    weights = pd.Series(C160.fitted(day, result["seasons"], ALL_STATS) / day.spread[ALL_STATS], index=names)
    weights.to_csv(f"{out}_weights.csv", header=["points_per_unit"])
    print(weights.round(4).to_string(), flush=True)
    now = today_table(result, pair, weights, cfg)
    now.to_csv(f"{out}_today.csv")
    print("\n2027 through the latest games at the final pair", flush=True)
    print(now.round(2).to_string(), flush=True)
    print(f"done in {time.time() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
