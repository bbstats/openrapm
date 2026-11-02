"""Check 157: is the wins over starter average built on a sound win model, and what penalty does the pace ridge take?

The owner, 2026-11-02: an MVP score in wins (src/openrapm/wins.py).  Everything here reads finished seasons.

Words used below
    calibration   the possession-weighted chance the win model gives the team ahead, against how often it won, by bins
                  of time left and margin; once at each game's real strength gap, once at the average strength the
                  leverage itself reads
    net points per win     from teams' records alone, 1 over the slope of wins on net points: the reading that
                  shares no construction with the leverage
    rating points per win  1 over the slope of wins on the expected margin from the published ratings, in points and
                  weighted by relative leverage: what settings.WinsSettings.points_per_win is read from
    model points per win   1 over the win model's mean leverage.  Not the scale: leads shrink, so it values a point at
                  what reaches the final margin, while the ratings already sit below teams' actual margins
    leverage sum  a team's leverage times its actual margin in each stint, summed over its season, against its wins
                  above half its games.  Above 1 by what the shrinking leads take back
    pace search   the pace ridge fitted on each season's half-A games and scored on its half-B games, and the reverse,
                  at a grid of penalties.  The chosen penalty must be interior, with both neighbours worse
    pace repeat   the correlation among regulars of pace effects from half-A games and half-B games: whether a pace
                  effect belongs to the player, as the owner asked ("only if the player is causing-ish it")
    long way      `wins.counterfactual_wins`, each player's wins one stint at a time, against `wins.season_wins`

Usage: .venv/Scripts/python scripts/checks/157_wins_model.py --part model --tag first
       .venv/Scripts/python scripts/checks/157_wins_model.py --part pace --tag first
writes outputs/csv/wins_model_<tag>_seasons.csv, _calibration.csv, _pace_search.csv and _pace_agreement.csv.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from openrapm import swap_test, wins
from openrapm.commands.console import one_thread, use_utf8
from openrapm.data.config import load_config, resolve
from openrapm.data.roles import season_boxes, starts_by_game
from openrapm.data.seasons import last_finished
from openrapm.data.stints import load_season_stints
from openrapm.rapm import normal_equations
from openrapm.settings import PRODUCT

TIME_BINS = [("first half", 1440.0, 1e9), ("to 5 minutes left", 300.0, 1440.0), ("to 2 minutes left", 120.0, 300.0),
             ("last 2 minutes", 0.0, 120.0)]
MARGIN_BINS = [("0-2", 0.0, 3.0), ("3-6", 3.0, 7.0), ("7-12", 7.0, 13.0), ("13+", 13.0, 1e9)]
PENALTIES = (1e3, 3e3, 1e4, 3e4, 1e5, 3e5, 1e6, 3e6, 1e7, 3e7)
SEARCH_SEASONS = (2001, 2006, 2011, 2016, 2021, 2026)
ERAS = ((1997, 2006), (2007, 2016), (2017, 2026))
MINIMUM_SECONDS = (10.0, 15.0, 30.0)


def season_stints(cfg: dict, season: int) -> tuple[pd.DataFrame, np.ndarray]:
    stints = swap_test.order_stints(load_season_stints(season, cfg, wins.STINT_COLUMNS, phases=("RS",)))
    return stints, swap_test.stint_durations(stints)


def _game_values(results: pd.DataFrame, stints: pd.DataFrame, values) -> np.ndarray:
    return pd.Series(np.asarray(values, dtype=float), index=results.game_id.to_numpy()).reindex(
        stints.game_id.to_numpy()).to_numpy()


def calibration(stints: pd.DataFrame, seconds: np.ndarray, model: wins.WinModel) -> pd.DataFrame:
    """Per bin of time left and margin: the possessions, the model's chance for the team ahead at the real strength gap
    and at average strength, and how often the team ahead won.  Games of final margin 0 are left out."""
    results = wins.game_results(stints)
    final = _game_values(results, stints, results.margin)
    gap = _game_values(results, stints, wins.strength_gap(results))
    left = wins.share_left(stints.period, stints.start_clock, seconds)[1]
    margin = stints.margin_h.to_numpy(dtype=float)
    neutral = stints.neutral.fillna(False).to_numpy(dtype=float)
    ahead_at_home = margin >= 0.0
    chance = {name: wins.win_probability(margin, left, model, neutral, strength) for name, strength in
              (("real_strength", gap), ("average_strength", 0.0))}
    frame = pd.DataFrame({"seconds_left": left * 2880.0, "abs_margin": np.abs(margin),
                          "possessions": (stints.poss_h + stints.poss_a).to_numpy(dtype=float),
                          "won": np.where(ahead_at_home, final > 0.0, final < 0.0).astype(float),
                          **{name: np.where(ahead_at_home, value, 1.0 - value) for name, value in chance.items()}})
    frame = frame[final != 0.0]
    rows = []
    for time_name, low, high in TIME_BINS:
        for margin_name, least, most in MARGIN_BINS:
            cell = frame[(frame.seconds_left >= low) & (frame.seconds_left < high) & (frame.abs_margin >= least) &
                         (frame.abs_margin < most) & (frame.possessions > 0.0)]
            if len(cell):
                weight = cell.possessions
                rows.append(dict(time_left=time_name, margin=margin_name, possessions=float(weight.sum()),
                                 won=np.average(cell.won, weights=weight),
                                 real_strength=np.average(cell.real_strength, weights=weight),
                                 average_strength=np.average(cell.average_strength, weights=weight)))
    return pd.DataFrame(rows)


def team_sums(stints: pd.DataFrame, seconds: np.ndarray, model: wins.WinModel, ratings: pd.DataFrame,
              points_per_win: float) -> pd.DataFrame:
    """Per team: its wins above half its games, its leverage sum, and its wins from the ratings at `points_per_win`.  A
    team's view of a home margin is the home team's margin, or its negative."""
    results = wins.game_results(stints)
    relative = wins.relative_leverage(stints, seconds, model)
    lever = relative * wins.mean_leverage(stints, seconds, model)
    actual = (stints.pts_h - stints.pts_a).to_numpy(dtype=float)
    margin = wins.expected_margins(stints, ratings)[0]
    possessions = (stints.poss_h + stints.poss_a).to_numpy(dtype=float) / 2.0
    home_view = pd.DataFrame({"game_id": stints.game_id.to_numpy(), "leverage_sum": lever * actual,
                              "rating_wins": relative * margin * possessions / 100.0 / points_per_win
                              }).groupby("game_id").sum()
    games = results.set_index("game_id").join(home_view)
    margin_final = games.margin.to_numpy(dtype=float)
    teams = pd.DataFrame({"team": np.concatenate([games.home_team_id, games.away_team_id]),
                          "wins_above": np.concatenate([(margin_final > 0.0) - 0.5, (margin_final < 0.0) - 0.5]),
                          "leverage_sum": np.concatenate([games.leverage_sum, -games.leverage_sum]),
                          "rating_wins": np.concatenate([games.rating_wins, -games.rating_wins])})
    return teams.groupby("team").sum()


def slope_and_correlation(x, y) -> tuple[float, float]:
    return float(np.polyfit(x, y, 1)[0]), float(np.corrcoef(x, y)[0, 1])


def long_way_gap(stints: pd.DataFrame, seconds: np.ndarray, ratings: pd.DataFrame, model: wins.WinModel) -> float:
    """The largest difference, over a season's players, between the wins as shipped and the wins the long way."""
    choices = PRODUCT.wins
    pace = wins.pace_effects(stints, seconds, choices.pace_penalty, choices.pace_min_seconds)
    bar = wins.starter_bar(ratings, wins.starts_and_minutes(stints, seconds), pace, choices.starters)
    relative = wins.relative_leverage(stints, seconds, model)
    shipped = wins.season_wins(stints, seconds, ratings, relative, pace, bar, choices.points_per_win)
    long_way = wins.counterfactual_wins(stints, seconds, ratings, relative, pace, bar, choices.points_per_win)
    return float((shipped.set_index("player_id").wosa - long_way.reindex(shipped.player_id)).abs().max())


def box_starts_gap(cfg: dict, season: int, starts: pd.DataFrame) -> tuple[int, int]:
    """(players whose starts differ from the box scores' first five, the largest difference): is a game's opening stint
    its starters?  Regular season alone, since the roles table pools the playoffs in."""
    boxes = starts_by_game(season_boxes(season, cfg, phases=("RS",)))
    from_box = boxes.groupby("player_id").started.sum()
    gap = (starts.set_index("player_id").starts_rs.reindex(from_box.index).fillna(0.0) - from_box).abs()
    return int((gap > 0.0).sum()), int(gap.max())


def season_row(cfg: dict, season: int, stints: pd.DataFrame, seconds: np.ndarray, ratings: pd.DataFrame,
               cells: pd.DataFrame, model: wins.WinModel) -> dict:
    """One season's line of the model part."""
    start, middle = wins.share_left(stints.period, stints.start_clock, seconds)
    margin = stints.margin_h.to_numpy(dtype=float)
    neutral = stints.neutral.fillna(False).to_numpy(dtype=float)
    possessions = (stints.poss_h + stints.poss_a).to_numpy(dtype=float)
    results = wins.game_results(stints)
    gap = _game_values(results, stints, wins.strength_gap(results))
    above = wins.win_probability(margin + 1.0, middle, model, neutral, gap)
    aware = (above - wins.win_probability(margin - 1.0, middle, model, neutral, gap)) / 2.0
    teams = team_sums(stints, seconds, model, ratings, PRODUCT.wins.points_per_win)
    leverage_slope, leverage_r = slope_and_correlation(teams.leverage_sum, teams.wins_above)
    rating_slope, rating_r = slope_and_correlation(teams.rating_wins, teams.wins_above)
    relative = wins.relative_leverage(stints, seconds, model)
    period, clock = stints.period.to_numpy(), stints.start_clock.to_numpy(dtype=float)
    close_late = (period >= swap_test.CLUTCH_PERIOD) & (clock <= swap_test.CLUTCH_SECONDS) & (
        np.abs(margin) <= swap_test.CLUTCH_MARGIN)
    garbage = stints.is_gt.to_numpy(dtype=bool)
    starts = wins.starts_and_minutes(stints, seconds)
    differ, largest = box_starts_gap(cfg, season, starts)
    weight = cells.possessions
    return dict(season=season, games=model.n_games, games_margin_zero=model.n_ties, home_edge=model.home_edge,
                strength_slope=model.strength_slope, lead_kept_at_end=float(wins.lead_kept(0.0, model)),
                lead_kept_at_half=float(wins.lead_kept(0.5, model)),
                lead_kept_at_tipoff=float(wins.lead_kept(1.0, model)),
                spread_tipoff=float(wins.spread_at(1.0, model)),
                spread_final_margins=float((results.margin - results.margin.mean()).std()),
                model_points_per_win=1.0 / float(np.average(wins.leverage(margin, middle, model, neutral),
                                                            weights=possessions)),
                model_points_per_win_at_stint_start=1.0 / float(np.average(wins.leverage(margin, start, model, neutral),
                                                                           weights=possessions)),
                model_points_per_win_real_strength=1.0 / float(np.average(aware, weights=possessions)),
                net_points_per_win=wins.net_points_per_win(results),
                rating_points_per_win=wins.rating_points_per_win(stints, relative,
                                                                 wins.expected_margins(stints, ratings)[0]),
                relative_leverage_close_late=float(np.average(relative[close_late], weights=possessions[close_late])),
                relative_leverage_garbage=float(np.average(relative[garbage], weights=possessions[garbage])),
                relative_leverage_first_quarter=float(np.average(relative[period == 1],
                                                                 weights=possessions[period == 1])),
                miss_real_strength_mean=np.average(np.abs(cells.real_strength - cells.won), weights=weight),
                miss_real_strength_largest=float(np.abs(cells.real_strength - cells.won).max()),
                miss_average_strength_mean=np.average(np.abs(cells.average_strength - cells.won), weights=weight),
                leverage_sum_slope=leverage_slope, leverage_sum_correlation=leverage_r,
                record_on_rating_wins_slope=rating_slope, record_on_rating_wins_correlation=rating_r,
                players_with_other_starts_than_box=differ, largest_starts_gap=largest,
                largest_gap_to_the_long_way=long_way_gap(stints, seconds, ratings, model))


def model_part(cfg: dict, seasons: list, table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    lines, calibrated = [], []
    for season in seasons:
        stints, seconds = season_stints(cfg, season)
        model = wins.fit_win_model(stints, seconds, season)
        cells = calibration(stints, seconds, model)
        line = season_row(cfg, season, stints, seconds, table[table.season == season], cells, model)
        lines.append(line)
        calibrated.append(cells.assign(season=season))
        print(f"  {season}: home edge {line['home_edge']:.2f}, spread {line['spread_tipoff']:.1f}, lead kept at the "
              f"half {line['lead_kept_at_half']:.2f}, rating points per win {line['rating_points_per_win']:.1f}, net "
              f"{line['net_points_per_win']:.1f}, calibration miss {line['miss_real_strength_mean']:.3f}, record on "
              f"rating wins {line['record_on_rating_wins_slope']:.2f} r "
              f"{line['record_on_rating_wins_correlation']:.2f}, starts differ for "
              f"{line['players_with_other_starts_than_box']} by up to {line['largest_starts_gap']}", flush=True)
    return pd.DataFrame(lines), pd.concat(calibrated, ignore_index=True)


def search_season(stints: pd.DataFrame, seconds: np.ndarray, penalties, min_seconds: float, season: int) -> list:
    """The held-back error of one season at every penalty, in both directions, plus the error with no player at all."""
    halves = {half: stints.half.to_numpy() == half for half in ("A", "B")}
    rows = {half: wins.pace_rows(stints[mask], seconds[mask], min_seconds) for half, mask in halves.items()}
    sums = {half: normal_equations(kept.players, kept.context, kept.target, kept.weight) for half, kept in rows.items()}
    lines = []
    for penalty in penalties:
        for fit_half, score_half in (("A", "B"), ("B", "A")):
            fit = wins.solve_pace(rows[fit_half], penalty, sums[fit_half])
            lines.append(dict(season=season, penalty=penalty, fit_half=fit_half,
                              error=wins.pace_error(fit, rows[score_half])))
    return lines


def chosen_penalty(search: pd.DataFrame) -> tuple[float, bool]:
    """(the penalty with the least mean error, whether it is interior with both neighbours worse)."""
    means = search.groupby("penalty").error.mean().sort_index()
    place = int(np.argmin(means.to_numpy()))
    interior = 0 < place < len(means) - 1
    return float(means.index[place]), interior


def pace_repeat(stints: pd.DataFrame, seconds: np.ndarray, penalty: float, min_seconds: float, season: int) -> dict:
    """Half A's pace effects against half B's, among players with 500 minutes in each half."""
    effects, minutes = {}, {}
    for half in ("A", "B"):
        mask = stints.half.to_numpy() == half
        effects[half] = wins.pace_effects(stints[mask], seconds[mask], penalty, min_seconds).set_index(
            "player_id").pace_effect
        minutes[half] = wins.starts_and_minutes(stints[mask], seconds[mask]).set_index("player_id").minutes_rs
    regulars = minutes["A"][minutes["A"] >= 500.0].index.intersection(minutes["B"][minutes["B"] >= 500.0].index)
    return dict(season=season, regulars=len(regulars),
                correlation=float(np.corrcoef(effects["A"].reindex(regulars), effects["B"].reindex(regulars))[0, 1]))


def agreement(stints: pd.DataFrame, seconds: np.ndarray, penalty: float, season: int) -> list:
    """How far the pace effects of regulars move with the shortest stint the fit keeps, against the shipped minimum."""
    effects = {least: wins.pace_effects(stints, seconds, penalty, least).set_index("player_id").pace_effect
               for least in MINIMUM_SECONDS}
    minutes = wins.starts_and_minutes(stints, seconds).set_index("player_id").minutes_rs
    regulars = minutes[minutes >= 1000.0].index
    base = effects[15.0].reindex(regulars)
    return [dict(season=season, minimum_seconds=least,
                 correlation=float(np.corrcoef(base, kept.reindex(regulars))[0, 1]),
                 largest_move=float((kept.reindex(regulars) - base).abs().max()),
                 sd_regulars=float(kept.reindex(regulars).std())) for least, kept in effects.items()]


def pace_part(cfg: dict, seasons=SEARCH_SEASONS) -> tuple[pd.DataFrame, pd.DataFrame]:
    search, agreeing = [], []
    for season in seasons:
        stints, seconds = season_stints(cfg, season)
        search += search_season(stints, seconds, PENALTIES, PRODUCT.wins.pace_min_seconds, season)
        print(f"  {season}: searched", flush=True)
    search = pd.DataFrame(search)
    best, interior = chosen_penalty(search)
    by_penalty = search.groupby(["season", "penalty"]).error.mean().unstack("season")
    print((by_penalty - by_penalty.min()).round(2).to_string(), flush=True)
    print(f"the least mean error: penalty {best:g}, interior with both neighbours worse: {interior}", flush=True)
    repeats = []
    for season in seasons:
        stints, seconds = season_stints(cfg, season)
        agreeing += agreement(stints, seconds, best, season)
        repeats.append(pace_repeat(stints, seconds, best, PRODUCT.wins.pace_min_seconds, season))
    print(pd.DataFrame(repeats).round(3).to_string(index=False), flush=True)
    return search, pd.DataFrame(agreeing), pd.DataFrame(repeats)


def era_lines(seasons: pd.DataFrame) -> pd.DataFrame:
    """The season table's means by era."""
    keep = ["home_edge", "strength_slope", "lead_kept_at_end", "lead_kept_at_half", "lead_kept_at_tipoff",
            "spread_tipoff", "model_points_per_win", "model_points_per_win_at_stint_start",
            "model_points_per_win_real_strength", "net_points_per_win", "rating_points_per_win",
            "relative_leverage_close_late", "relative_leverage_garbage", "relative_leverage_first_quarter",
            "miss_real_strength_mean", "miss_real_strength_largest",
            "miss_average_strength_mean", "leverage_sum_slope", "record_on_rating_wins_slope",
            "record_on_rating_wins_correlation"]
    rows = []
    for low, high in ERAS:
        part = seasons[(seasons.season >= low) & (seasons.season <= high)]
        if len(part):
            rows.append(dict(era=f"{low}-{high}", **part[keep].mean().to_dict()))
    return pd.DataFrame(rows)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="The win model and the pace penalty of the wins over starter average.")
    parser.add_argument("--part", choices=("model", "pace"), required=True)
    parser.add_argument("--tag", required=True, help="the outputs are outputs/csv/wins_model_<TAG>_*.csv")
    parser.add_argument("--table", default="season_ratings_product", help="the ratings table the rating sum reads")
    parser.add_argument("--first", type=int, default=1997)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    use_utf8()
    args = parse_args(argv)
    cfg = load_config()
    folder = resolve(cfg, "outputs") / "csv"
    folder.mkdir(parents=True, exist_ok=True)
    stem = folder / f"wins_model_{args.tag}"
    with one_thread():
        if args.part == "pace":
            search, agreeing, repeats = pace_part(cfg)
            search.to_csv(f"{stem}_pace_search.csv", index=False)
            agreeing.to_csv(f"{stem}_pace_agreement.csv", index=False)
            repeats.to_csv(f"{stem}_pace_repeat.csv", index=False)
            print(agreeing.round(3).to_string(index=False), flush=True)
            return 0
        table = pd.read_parquet(resolve(cfg, "outputs") / f"{args.table}.parquet",
                                columns=["season", "player_id", "rating_off", "rating_def", "rating_total"])
        seasons, cells = model_part(cfg, list(range(args.first, last_finished(cfg) + 1)), table)
        seasons.to_csv(f"{stem}_seasons.csv", index=False)
        cells.to_csv(f"{stem}_calibration.csv", index=False)
        print(era_lines(seasons).round(3).T.to_string(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
