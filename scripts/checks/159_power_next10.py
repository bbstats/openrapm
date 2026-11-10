"""Check 159: Power as the starting point for a team's next 10 games, with its last 10 games' stats added.

The owner, 2026-11-08: "using current power ratings, then add in L10 shot zones, orb%, turnovers, and ft stuff, i want
to use these power ratings as a prior for the next 10 games!".  notes/DECISIONS.md, "Check 159", holds the rule, written
before any result.

Words used below
    day            the day a share of a season's regular-season games had been played (`CUTS`): every rating read here
                   saw only the games through that day (outputs/within/<folder>, `openrapm folds --cut`)
    Power          the Power page's sum (commands/site.py): each player's rating on that day times his share of the
                   team's possessions over its last 10 games, the latest 5 counted double and blowouts half, less the
                   mean of the season's teams, on offense and on defense
    last 10        a team's last 10 games through the day, unweighted, or as many as it has played
    next 10        a team's next 10 regular-season games after the day
    over expected  a count less what the league's rate expects from the team's chances, per 100 possessions.  A team
                   that takes 30 rim shots per 100 and makes 21 where the league makes 65% is +1.5 makes over expected
    margin         a game's points for less points against, per 100 possessions
    difference     a stat of the team less the same stat of its opponent, both as of the day

The fit, one for each day.  Next-10 margin = two free multipliers x the two teams' difference in Power on offense and on
defense + a home edge (+1 at home, -1 away, 0 on a neutral court) + a ridge regression on the differences of the added
stats, the penalty on those alone.  Each season is predicted from the other 29; the penalty is picked inside the 29 by
leaving each of them out in turn.  Season 2027 is never fitted on: it only receives the finished fit.

    python scripts/checks/159_power_next10.py [--out outputs/csv/check159]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from openrapm.commands.site import RECENT_GAMES, game_teams, player_game_possessions, team_shares  # noqa: E402
from openrapm.data.config import load_config, resolve  # noqa: E402
from openrapm.prior_shrink import WITHIN_COLUMNS, season_factors, shrink_fold  # noqa: E402

FAMILY = os.environ.get("POWER_NEXT10_FAMILY", "rating134")  # which build's season-so-far ratings are read
CUTS = {
    "7%": f"{FAMILY}_cut0715_held",
    "10%": f"{FAMILY}_cut10_held",
    "25%": f"{FAMILY}_cut25_held",
    "50%": f"{FAMILY}_cut50_held",
}
TODAY_CUT = "10%"
SEASONS = range(1997, 2027)
ERAS = {"1997-2006": (1997, 2006), "2007-2016": (2007, 2016), "2017-2026": (2017, 2026)}
LAST_GAMES = NEXT_GAMES = 10
PENALTIES = np.array([0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0])  # on columns of unit spread
ZONES = ["rim", "smid", "lmid", "thr", "lthr"]
ZONE_NAMES = dict(rim="rim", smid="short mid", lmid="long mid", thr="three", lthr="long three")
COUNTS = [
    "poss",
    "pts",
    "fta",
    "ftm",
    "fta_tech",
    "ftm_tech",
    "tov",
    "reb_chance",
    "reb_cont",
    *[f"{kind}_{zone}" for kind in ("fga", "fgm") for zone in ("rim", "smid", "lmid", "thr", "lthr")],
]
SHUFFLE_SEED = 159
# what each of a team's latest `RECENT_GAMES` games counts for in the last-10 stats; 1 is a plain last 10, as checks 159
# to 161 ran.  Check 162 sets it to the Power page's site.RECENT_WEIGHT
STATS_RECENT_WEIGHT = 1.0


# ----------------------------------------------------------------------------------------- the games of a season
def season_game_table(season: int, cfg: dict) -> pd.DataFrame:
    """One row per team per regular-season game: game_id, game_date, tm, opp, home, the team's counts `*_o`, its
    opponent's `*_d`, margin (per 100 possessions a team)."""
    columns = ["game_id", "game_date", "neutral", *[f"{c}_{end}" for c in COUNTS for end in "ha"]]
    stints = pd.read_parquet(resolve(cfg, "stints") / f"{season}_RS.parquet", columns=columns)
    stints["game_id"] = stints.game_id.astype(str).str.zfill(10)
    games = stints.groupby("game_id", as_index=False).agg(
        game_date=("game_date", "first"),
        neutral=("neutral", "first"),
        **{f"{c}_{end}": (f"{c}_{end}", "sum") for c in COUNTS for end in "ha"},
    )
    for end in "ha":  # the three zone counts as the file keeps them:
        games[f"fga_thr_{end}"] -= games[f"fga_lthr_{end}"]  # fga_thr holds the long threes, take them out
        games[f"fgm_thr_{end}"] -= games[f"fgm_lthr_{end}"]
    log = pd.read_parquet(
        resolve(cfg, "raw") / "gamelog" / f"{season}_RS.parquet",
        columns=["GAME_ID", "TEAM_ID", "TEAM_ABBREVIATION", "MATCHUP"],
    ).drop_duplicates(["GAME_ID", "TEAM_ID"])
    log["GAME_ID"] = log.GAME_ID.astype(str).str.zfill(10)
    # a neutral-site game lists both teams as "@": the stints' home side is then the lower team id (data/ingest.py)
    log["is_home"] = log.MATCHUP.str.contains(" vs. ", regex=False)
    log = log.sort_values(["GAME_ID", "is_home", "TEAM_ID"], ascending=[True, False, True], kind="stable")
    home = log.groupby("GAME_ID").TEAM_ABBREVIATION.first()
    away = log.groupby("GAME_ID").TEAM_ABBREVIATION.last()
    games["home_tm"], games["away_tm"] = games.game_id.map(home), games.game_id.map(away)
    missing = games.home_tm.isna() | games.away_tm.isna()
    if missing.any():
        raise SystemExit(f"{season}: {int(missing.sum())} games have no teams in the game log")
    rows = []
    for mine, theirs, sign, tm, opp in (("h", "a", 1, "home_tm", "away_tm"), ("a", "h", -1, "away_tm", "home_tm")):
        part = games[["game_id", "game_date"]].assign(
            tm=games[tm], opp=games[opp], home=np.where(games.neutral, 0, sign)
        )
        for c in COUNTS:
            part[f"{c}_o"], part[f"{c}_d"] = games[f"{c}_{mine}"], games[f"{c}_{theirs}"]
        rows.append(part)
    table = pd.concat(rows, ignore_index=True)
    poss = (table.poss_o + table.poss_d) / 2
    table["margin"] = 100 * (table.pts_o - table.pts_d) / poss
    return table.sort_values(["tm", "game_date", "game_id"], kind="stable").reset_index(drop=True)


def last_games_stats(played: pd.DataFrame, game_weights=None) -> pd.DataFrame:
    """The added stats of each team over its last `LAST_GAMES` games of `played`: one row per team, indexed by tm.
    Every column is centered over the teams, so a season's or a day's level cannot enter.  `game_weights(tm, back)`,
    given each game's team and how many games before the team's latest it is, returns the games' weights; by default
    the latest `RECENT_GAMES` count `STATS_RECENT_WEIGHT`."""
    last = played.sort_values(["tm", "game_date", "game_id"], kind="stable").groupby("tm").tail(LAST_GAMES).copy()
    columns = [f"{c}_{side}" for c in COUNTS for side in "od"]
    # each game's counts and possessions take the game's weight before any rate is formed, so a rate is
    # (2 x the latest 5 games' count + the 5 before them) / (2 x their possessions + the 5 before them)
    games_back = last.groupby("tm").cumcount(ascending=False)  # 0 for the team's latest game
    if game_weights is None:
        weight = np.where(games_back < RECENT_GAMES, STATS_RECENT_WEIGHT, 1.0)
    else:
        weight = game_weights(last.tm.to_numpy(), games_back.to_numpy())
    last[columns] = last[columns].astype(float).mul(weight, axis=0)
    sums = last.groupby("tm")[columns].sum()
    league = sums.sum()
    out = pd.DataFrame(index=sums.index)
    for side in "od":
        per100 = 100 / sums[f"poss_{side}"]
        for zone in ZONES:
            attempts, makes = sums[f"fga_{zone}_{side}"], sums[f"fgm_{zone}_{side}"]
            pct = (league[f"fgm_{zone}_o"] + league[f"fgm_{zone}_d"]) / (
                league[f"fga_{zone}_o"] + league[f"fga_{zone}_d"]
            )
            out[f"attempts_{zone}_{side}"] = attempts * per100
            out[f"makes_over_{zone}_{side}"] = (makes - pct * attempts) * per100
        rate = (league.reb_cont_o + league.reb_cont_d) / (league.reb_chance_o + league.reb_chance_d)
        out[f"orb_over_{side}"] = (sums[f"reb_cont_{side}"] - rate * sums[f"reb_chance_{side}"]) * per100
        out[f"tov_{side}"] = sums[f"tov_{side}"] * per100
        attempts = sums[f"fta_{side}"] - sums[f"fta_tech_{side}"]
        makes = sums[f"ftm_{side}"] - sums[f"ftm_tech_{side}"]
        pct = (league.ftm_o + league.ftm_d - league.ftm_tech_o - league.ftm_tech_d) / (
            league.fta_o + league.fta_d - league.fta_tech_o - league.fta_tech_d
        )
        out[f"ft_attempts_{side}"] = attempts * per100
        out[f"ft_makes_over_{side}"] = (makes - pct * attempts) * per100
    out["last10_margin"] = 200 * (sums.pts_o - sums.pts_d) / (sums.poss_o + sums.poss_d)
    return out - out.mean()


GROUPS = {
    "zones": [f"{kind}_{zone}_{side}" for kind in ("attempts", "makes_over") for zone in ZONES for side in "od"],
    "offensive rebounds": ["orb_over_o", "orb_over_d"],
    "turnovers": ["tov_o", "tov_d"],
    "free throws": [f"{name}_{side}" for name in ("ft_attempts", "ft_makes_over") for side in "od"],
}
ADDED = [column for group in GROUPS.values() for column in group]


# ------------------------------------------------------------------------------------- Power as of a day
def power_on_day(
    ratings: pd.DataFrame, played_ids: set, stints_season: pd.DataFrame, teams: pd.DataFrame
) -> pd.DataFrame:
    """Each team's Power from `ratings` (player_id, rating_off, rating_def), through the games `played_ids`: the same
    shares and centering as the Power page.  One row per team, indexed by tm: power_o, power_d."""
    stints = stints_season[stints_season.game_id.isin(played_ids)]
    shares = team_shares(player_game_possessions(stints, teams), LAST_GAMES)
    shares = shares.join(ratings.set_index("player_id")[["rating_off", "rating_def"]], on="player_id")
    unrated = shares.rating_off.isna()
    if unrated.any():  # a player with no row at all counts as average
        shares.loc[unrated, ["rating_off", "rating_def"]] = 0.0
    power = (
        pd.DataFrame(
            {"power_o": shares.share * shares.rating_off, "power_d": shares.share * shares.rating_def, "tm": shares.tm}
        )
        .groupby("tm")
        .sum()
    )
    return power - power.mean()


def season_stints_ids(season: int, cfg: dict) -> pd.DataFrame:
    """site.season_stints, regular season only, with game_id as ten characters."""
    slots = [f"{side}{i}" for side in "ha" for i in range(1, 6)]
    stints = pd.read_parquet(
        resolve(cfg, "stints") / f"{season}_RS.parquet",
        columns=["game_id", "game_date", "poss_h", "poss_a", "is_gt", *slots],
    )
    stints["game_id"] = stints.game_id.astype(str).str.zfill(10)
    return stints


def regular_game_teams(season: int, cfg: dict) -> pd.DataFrame:
    """site.game_teams with game_id as ten characters."""
    teams = game_teams(season, cfg)
    teams["game_id"] = teams.game_id.astype(str).str.zfill(10)
    return teams


def load_wowy():
    """The with-or-without-you check as a module: it knows how a day's folds take the shipped shrink."""
    spec = importlib.util.spec_from_file_location("wowy151", ROOT / "scripts" / "checks" / "151_wowy.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["wowy151"] = module
    spec.loader.exec_module(module)
    return module


PUBLISHED_MULTIPLIERS = ROOT / "outputs" / "csv" / "season_ratings_product_step4_multipliers.csv"


def day_ratings(wowy, within: Path, folder: str, step4: str = "folds") -> dict:
    """{season: player_id, rating_off, rating_def} of the day `folder`, through the shipped shrink, and each season's
    games through that day: {season: (ratings, set of game ids)}.

    `step4` says where the shrink's multipliers come from.  "folds": fitted on the day's own season-so-far folds, as
    checks 159 to 165 read them.  "published": each season's row of the published table's multipliers
    (`PUBLISHED_MULTIPLIERS`), fitted on whole seasons without that season, which is what the season in progress is
    published with and so what the Power page multiplies (check 166)."""
    folds, where = wowy.system_folds(within, folder)
    folds = [fold for fold in folds if fold.season in SEASONS]
    seasons = sorted({fold.season for fold in folds})
    if step4 == "published":
        published = pd.read_csv(PUBLISHED_MULTIPLIERS)
    else:
        multipliers, layer = wowy.step4_tables(
            within, folds, where, seasons, wowy.DEFENCE_FACTORS, folder, wowy.WITHIN_FACTORS
        )
    out = {}
    for fold in folds:
        if step4 == "published":
            factors = season_factors(published, fold.season)
        else:
            row = multipliers.loc[fold.season]
            factor = wowy.factor_for(wowy.DEFENCE_FACTORS, fold.season)
            factors = (
                float(row.multiplier_off),
                factor * float(row.multiplier_def),
                float(row.get("games_multiplier_off", 1.0)),
                factor * float(row.get("games_multiplier_def", 1.0)),
            )
            if layer is not None:
                factors += tuple(float(layer.at[fold.season, column]) for column in WITHIN_COLUMNS)
        rated = shrink_fold(fold.rows, fold.season, factors)
        file = pd.read_parquet(where[fold.stem] / f"folds_{fold.season}.parquet")
        ids = file.game_id if "game_id" in file.columns else file.index.to_series()
        kept = file.r0.to_numpy() == -1
        out[fold.season] = (rated[["player_id", "rating_off", "rating_def"]], set(ids.astype(str).str.zfill(10)[kept]))
    return out


# -------------------------------------------------------------------------------------------- rows of a day
def day_rows(cut: str, ratings: dict, tables: dict, stints: dict, teams: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(team rows: season, tm, power, added stats; game rows: season, tm, opp, home, margin) of one day."""
    team_rows, game_rows = [], []
    for season in SEASONS:
        rated, played_ids = ratings[season]
        table = tables[season]
        played = table[table.game_id.isin(played_ids)]
        day = played.game_date.max()
        features = last_games_stats(played).join(power_on_day(rated, played_ids, stints[season], teams[season]))
        team_rows.append(features.reset_index().assign(season=season, cut=cut))
        later = table[table.game_date > day].sort_values(["tm", "game_date", "game_id"], kind="stable")
        next_games = later.groupby("tm").head(NEXT_GAMES)
        next_games = next_games[next_games.opp.isin(features.index)]
        game_rows.append(next_games[["tm", "opp", "home", "margin", "game_id"]].assign(season=season))
    return pd.concat(team_rows, ignore_index=True), pd.concat(game_rows, ignore_index=True)


def design(team_rows: pd.DataFrame, game_rows: pd.DataFrame, columns: list) -> np.ndarray:
    """The difference between each game row's team and its opponent on `columns`, row by row."""
    table = team_rows.set_index(["season", "tm"])[columns]
    mine = table.reindex(pd.MultiIndex.from_frame(game_rows[["season", "tm"]]))
    theirs = table.reindex(pd.MultiIndex.from_arrays([game_rows.season, game_rows.opp]))
    return mine.to_numpy(float) - theirs.to_numpy(float)


def fit_season_out(x: np.ndarray, y: np.ndarray, season: np.ndarray, penalized: np.ndarray) -> tuple[np.ndarray, list]:
    """Each season's predictions from the other seasons' fit, the penalty picked inside them; and the penalty each
    season took.  `x` has columns of unit spread; `penalized` says which columns the penalty falls on."""
    seasons = np.unique(season)
    xtx, xty = {}, {}
    for s in seasons:
        mask = season == s
        xtx[s], xty[s] = x[mask].T @ x[mask], x[mask].T @ y[mask]
    total_xtx, total_xty, n = sum(xtx.values()), sum(xty.values()), len(y)

    def solve(a: np.ndarray, b: np.ndarray, rows: int, penalty: float) -> np.ndarray:
        return np.linalg.solve(a + rows * penalty * np.diag(penalized.astype(float)) + 1e-9 * np.eye(a.shape[0]), b)

    grid = PENALTIES if penalized.any() else np.array([0.0])
    predictions, picked = np.zeros(n), []
    for s in seasons:
        a, b, rows = total_xtx - xtx[s], total_xty - xty[s], n - int((season == s).sum())
        errors = np.zeros(len(grid))
        for inner in seasons:
            if inner == s:
                continue
            ai, bi, rowsi = a - xtx[inner], b - xty[inner], rows - int((season == inner).sum())
            mask = season == inner
            for k, penalty in enumerate(grid):
                beta = solve(ai, bi, rowsi, penalty)
                errors[k] += float(((y[mask] - x[mask] @ beta) ** 2).sum())
        best = grid[int(errors.argmin())]
        picked.append(float(best))
        predictions[season == s] = x[season == s] @ solve(a, b, rows, best)
    return predictions, picked


def final_fit(x: np.ndarray, y: np.ndarray, season: np.ndarray, penalized: np.ndarray) -> tuple[np.ndarray, float]:
    """The fit on every season, its penalty picked by leaving each season out."""
    seasons = np.unique(season)
    a, b, n = x.T @ x, x.T @ y, len(y)
    grid = PENALTIES if penalized.any() else np.array([0.0])
    errors = np.zeros(len(grid))
    for s in seasons:
        mask = season == s
        ai, bi, rows = a - x[mask].T @ x[mask], b - x[mask].T @ y[mask], n - int(mask.sum())
        for k, penalty in enumerate(grid):
            beta = np.linalg.solve(
                ai + rows * penalty * np.diag(penalized.astype(float)) + 1e-9 * np.eye(a.shape[0]), bi
            )
            errors[k] += float(((y[mask] - x[mask] @ beta) ** 2).sum())
    best = float(grid[int(errors.argmin())])
    return np.linalg.solve(a + n * best * np.diag(penalized.astype(float)) + 1e-9 * np.eye(a.shape[0]), b), best


# ----------------------------------------------------------------------------------------------- the models
def models() -> dict:
    """{name: (unpenalized columns, penalized columns)}; every model also has a free home edge, unpenalized."""
    power = ["power_o", "power_d"]
    out = {
        "home edge only": ([], []),
        "Power at its own scale": (power, []),
        "last-10 margin alone": (["last10_margin"], []),
        "Power + last-10 margin": (power, ["last10_margin"]),
    }
    for name, columns in GROUPS.items():
        out[f"Power + {name}"] = (power, columns)
    out["Power + all four groups"] = (power, ADDED)
    out["control: all four shuffled"] = (power, [f"shuffled_{column}" for column in ADDED])
    return out


def with_shuffled(team_rows: pd.DataFrame) -> pd.DataFrame:
    """`team_rows` plus each added stat dealt to a random team of the same season and day, a single draw, the same draw
    for every stat: every stat's spread is kept and no team's stats are its own."""
    rng = np.random.default_rng(SHUFFLE_SEED)
    order = np.arange(len(team_rows))
    for _, index in team_rows.groupby(["season", "cut"]).indices.items():
        order[index] = rng.permutation(index)
    shuffled = team_rows[ADDED].to_numpy(float)[order]
    return pd.concat(
        [team_rows, pd.DataFrame(shuffled, columns=[f"shuffled_{c}" for c in ADDED], index=team_rows.index)], axis=1
    )


def score_models(team_rows: pd.DataFrame, game_rows: pd.DataFrame) -> pd.DataFrame:
    """One row per model and season: the per-game squared error and the squared error of a team's 10-game average."""
    y, season = game_rows.margin.to_numpy(float), game_rows.season.to_numpy()
    home = game_rows.home.to_numpy(float)[:, None]
    records = []
    for name, (free, penalized) in models().items():
        columns = [*free, *penalized]
        x = np.hstack([home, design(team_rows, game_rows, columns)]) if columns else home
        x = x / x.std(axis=0)
        mask = np.array([False] * (1 + len(free)) + [True] * len(penalized))
        predictions, _ = fit_season_out(x, y, season, mask)
        records.append(score_frame(name, game_rows, y, predictions))
    fixed = design(team_rows, game_rows, ["power_o", "power_d"]).sum(axis=1)
    # Power as it reads: multiplier 1, only the home edge fitted, on the other seasons
    predictions = np.zeros(len(y))
    for s in np.unique(season):
        out = season == s
        edge = float(((y - fixed)[~out] * home[~out, 0]).sum() / (home[~out, 0] ** 2).sum())
        predictions[out] = fixed[out] + edge * home[out, 0]
    records.append(score_frame("Power as it reads", game_rows, y, predictions))
    return pd.concat(records, ignore_index=True)


def score_frame(name: str, game_rows: pd.DataFrame, y: np.ndarray, predictions: np.ndarray) -> pd.DataFrame:
    """One model's squared errors by season: every game, and each team's average over its next 10 games."""
    frame = pd.DataFrame(
        {"season": game_rows.season.to_numpy(), "tm": game_rows.tm.to_numpy(), "miss": y - predictions}
    )
    frame["game"] = frame.miss**2
    per_team = frame.groupby(["season", "tm"]).miss.mean().pow(2).groupby("season").mean().rename("team")
    return frame.groupby("season").game.mean().to_frame().join(per_team).reset_index().assign(model=name)


def compared(scores: pd.DataFrame, reference: str = "Power at its own scale") -> pd.DataFrame:
    """Each model minus the reference, season by season: mean, standard error, paired t-statistic, seasons better; for
    every game and for a team's 10-game average, overall and by era."""
    out = []
    wide = scores.pivot(index="season", columns="model", values=["game", "team"])
    eras = {"all seasons": (1997, 2026), **ERAS}
    for era, (first, last) in eras.items():
        part = wide.loc[first:last]
        for model in scores.model.unique():
            row = dict(
                era=era,
                model=model,
                game_rmse=float(np.sqrt(part["game"][model].mean())),
                team_rmse=float(np.sqrt(part["team"][model].mean())),
            )
            for unit in ("game", "team"):
                diff = part[unit][model] - part[unit][reference]
                row[f"{unit}_change"] = float(diff.mean())
                se = float(diff.std(ddof=1) / np.sqrt(len(diff)))
                row[f"{unit}_z"] = float(diff.mean() / se) if se > 0 else 0.0
                row[f"{unit}_better"] = int((diff < 0).sum())
            row["seasons"] = len(part)
            out.append(row)
    return pd.DataFrame(out)


# --------------------------------------------------------------------------------------------- today's teams
def today_teams(cfg: dict) -> pd.DataFrame:
    """2027 through today: the Power page's numbers from the published ratings, and the added stats."""
    table = season_game_table(2027, cfg)
    published = pd.read_parquet(
        resolve(cfg, "outputs") / "season_ratings_product.parquet",
        columns=["season", "player_id", "rating_off", "rating_def"],
    )
    rated = published[published.season == 2027]
    ids = set(table.game_id)
    features = last_games_stats(table).join(
        power_on_day(rated, ids, season_stints_ids(2027, cfg), regular_game_teams(2027, cfg))
    )
    games = table.groupby("tm").game_id.nunique().rename("games")
    return features.join(games)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="outputs/csv/check159")
    args = parser.parse_args()
    started = time.time()
    cfg = load_config()
    within = ROOT / "outputs" / "within"
    wowy = load_wowy()
    print("reading the seasons' games", flush=True)
    tables = {season: season_game_table(season, cfg) for season in SEASONS}
    stints = {season: season_stints_ids(season, cfg) for season in SEASONS}
    teams = {season: regular_game_teams(season, cfg) for season in SEASONS}
    print(f"read ({time.time() - started:.0f}s)", flush=True)

    all_compared, keep = [], {}
    for cut, folder in CUTS.items():
        ratings = day_ratings(wowy, within, folder)
        team_rows, game_rows = day_rows(cut, ratings, tables, stints, teams)
        team_rows = with_shuffled(team_rows)
        print(
            f"{cut}: {len(team_rows)} team rows, {len(game_rows)} game rows ({time.time() - started:.0f}s)", flush=True
        )
        if team_rows[["power_o", "power_d", *ADDED]].isna().any().any():
            raise SystemExit(f"{cut}: missing values in the team rows")
        table = compared(score_models(team_rows, game_rows)).assign(cut=cut)
        all_compared.append(table)
        keep[cut] = (team_rows, game_rows)
        shown = table[table.era == "all seasons"]
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
                    "team_better",
                ]
            ]
            .round(4)
            .to_string(index=False),
            flush=True,
        )
    result = pd.concat(all_compared, ignore_index=True)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(f"{out}_compared.csv", index=False)
    print("\nby era, the 10% day", flush=True)
    today = result[
        (result.cut == TODAY_CUT)
        & result.model.isin(
            ["Power + all four groups", "control: all four shuffled", "Power + last-10 margin", "Power as it reads"]
        )
    ]
    print(
        today[
            ["era", "model", "game_change", "game_z", "game_better", "team_change", "team_z", "team_better", "seasons"]
        ]
        .round(4)
        .to_string(index=False),
        flush=True,
    )

    # the finished fit on every season at today's day, applied to 2027
    team_rows, game_rows = keep[TODAY_CUT]
    columns = ["power_o", "power_d", *ADDED]
    home = game_rows.home.to_numpy(float)[:, None]
    x = np.hstack([home, design(team_rows, game_rows, columns)])
    spread = x.std(axis=0)
    y, season = game_rows.margin.to_numpy(float), game_rows.season.to_numpy()
    penalized = np.array([False, False, False] + [True] * len(ADDED))
    beta, penalty = final_fit(x / spread, y, season, penalized)
    beta = beta / spread  # back to points per unit of each stat
    weights = pd.Series(beta[1:], index=columns)
    base_x = np.hstack([home, design(team_rows, game_rows, ["power_o", "power_d"])])
    base_spread = base_x.std(axis=0)
    base, _ = final_fit(base_x / base_spread, y, season, np.array([False, False, False]))
    base = base / base_spread
    print(
        f"\nfit, 10% day: penalty {penalty}; home edge {beta[0]:.2f}; Power multipliers offense {beta[1]:.3f}, "
        f"defense {beta[2]:.3f}; Power alone multipliers offense {base[1]:.3f}, defense {base[2]:.3f}",
        flush=True,
    )
    now = today_teams(cfg)
    centered = now[columns] - now[columns].mean()
    parts = pd.DataFrame(
        {"power_part": centered.power_o * weights.power_o + centered.power_d * weights.power_d}, index=now.index
    )
    for name, group in GROUPS.items():
        parts[name] = (centered[group] * weights[group]).sum(axis=1)
    parts["new_total"] = parts.sum(axis=1)
    parts["power"] = now.power_o + now.power_d
    parts["power_own_scale"] = centered.power_o * base[1] + centered.power_d * base[2]
    parts["games"] = now.games
    parts = parts.sort_values("new_total", ascending=False)
    parts.to_csv(f"{out}_today.csv")
    weights.to_csv(f"{out}_weights.csv", header=["points_per_unit"])
    print(parts.round(2).to_string(), flush=True)
    print("\nweights, points per unit of each stat", flush=True)
    print(weights.round(4).to_string(), flush=True)
    published = json.load(open(ROOT.parent / "spmm_site" / "docs" / "data" / "power.json", encoding="utf-8"))
    gap = max(abs(team["t"] - float(now.power_o[team["tm"]] + now.power_d[team["tm"]])) for team in published["teams"])
    print(f"\nlargest gap from the published Power page: {gap:.3f}", flush=True)
    print(f"done in {time.time() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
