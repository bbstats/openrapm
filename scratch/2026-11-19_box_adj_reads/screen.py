"""Read-only: the direct screen on today's Box Adj line, the line without age (notes/DECISIONS.md, "Experiment 143: the
reads on Box Adj"; the owner, 2026-11-18: "box adj is good ... the rest is fine").

Each fold behind the shipped line `noage142_line` (`rebuilt_base_early,rebuilt_base_within`) gets its share of the
within-season test's normal equations, built by the package's own code (`commands.second_prior.fold_rating_sums`'s
steps: `rating_line.prior_columns`, `ridge_parts`, `held_back_sums`), over today's 56 inputs an end followed by every
candidate and a copy of each candidate shuffled among the fold's players.  The first 2 + 2 x 56 columns must give back
`noage142_line`'s own sums_<season>.npz: `build` checks that for every season.

Candidates, each from the fold's fit games only:
    team rates       second_prior.ALL_TEAM_RATES not in the line (team made and missed threes and twos, attempts)
    teammates        his teammates' 13 standardized box rates with him on the court, and his team's when he sits
    panel            every numeric column of the fold's players file a fold rebuilds from its fit games, less
                     identifiers, outcomes, other seasons, age in any form, playoff_share, and any column equal across
                     two folds of one season (a season total: measurement trap 37)
    clutch           his share of his team's possessions in the last five minutes of the fourth quarter or overtime
                     with the margin within 5, minus his share of all its possessions, both over the games he played;
                     raw and padded by 50 clutch possessions
    per game         his possessions per game he played: minutes per game, counted in possessions
    box x share      the box-score prior times his share of team possessions, centered on the end's players

    .venv/Scripts/python scratch/2026-11-19_box_adj_reads/screen.py prepare
    .venv/Scripts/python scratch/2026-11-19_box_adj_reads/screen.py build --seasons 1997,1998
"""
import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from openrapm import rating_line, second_prior
from openrapm.commands.build import season_lineup_rows, step4_folds, within_layer_multipliers
from openrapm.commands.second_prior import (RatingSeason, check_fit_designs, fold_answers, fold_players,
                                            held_back_stints, stored_box)
from openrapm.commands.within_season import load_models, season_models
from openrapm.data.config import load_config, resolve
from openrapm.data.designs import DesignCache
from openrapm.data.panel import fit_designs, trainable_panel
from openrapm.data.within_folds import fold_games
from openrapm.features import BOX_RATES
from openrapm.prior_shrink import WITHIN_COLUMNS
from openrapm.rapm import design_blocks
from openrapm.second_prior import ALL_TEAM_RATES, _Entries, pad_toward_level
from openrapm.settings import TEST
from openrapm.swap_test import DEF, OFF

sys.stdout.reconfigure(encoding="utf-8")
FOLDS_TAG = "rebuilt_base_early,rebuilt_base_within"
MODELS = "rebuilt_base"
LINE = "noage142_line"
INPUT_SET = "context_types_no_age_team_zones_centred"
TAG = "reads143"
SEASONS = list(range(1997, 2027))
CLUTCH_PAD = 50.0
NOT_INPUTS = {"side", "player_id", "season", "possessions", "team_id", "repeat", "fold", "test_poss_off", "test_poss_def",
              "rating_off", "rating_def", "rating_total", "prior_off", "prior_def", "poss_off", "poss_def",
              "prior_scale_off", "prior_scale_def", "u_off", "u_def", "prior_raw_off", "prior_raw_def",
              "second_prior_raw_off", "second_prior_raw_def", "seasons_before", "career_possessions_before_k",
              "age", "age_at_first_season", "age_at_debut", "draft_pick", "seasons_with_team", "seasons_before_from_debut",
              "career_possessions_before_from_debut_k", "seasons_with_team_from_debut", "playoff_share",
              "season_from_2017", "league_two_point_pct", "league_three_point_pct", "league_points_per_shot",
              "shot_twos_attempted", "shot_twos_made", "shot_twos_expected_made", "shot_threes_attempted",
              "shot_threes_made", "shot_threes_expected_made", "on_court_off_possessions", "on_court_def_possessions",
              "off_court_off_possessions", "off_court_def_possessions"}
BUILT = ["clutch_share_over_share", "clutch_share_over_share_padded", "possessions_per_game", "box_prior_x_share"]


def out_dir(cfg) -> Path:
    return resolve(cfg, "outputs") / "direct" / TAG


def prepare(args) -> int:
    """names.json (every column, in order), the within layer every fold reads, and the line's type scale."""
    cfg = load_config()
    out = out_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    shrink = replace(TEST.shrink, folds_tag=FOLDS_TAG)
    folds, where = step4_folds(cfg, shrink)
    today = list(second_prior.INPUT_SETS[INPUT_SET])
    shipped = json.loads((resolve(cfg, "outputs") / "second_prior" / LINE / "run.json").read_text())
    assert shipped["inputs"] == today, "the line's run.json names other inputs"
    # the panel columns a fold rebuilds from its fit games: equal across two folds of a season means a season total
    first = [f for f in folds if f.season == 2019 and f.repeat == 0][:2]
    a, b = [fold_players(where[f.stem], f.stem).query("side == 'O'").set_index("player_id") for f in first]
    shared = a.index.intersection(b.index)
    panel, leaky = [], []
    for name in a.columns:
        if name in NOT_INPUTS or name in today or name.endswith("_uncentred") or name.endswith("_shuffled"):
            continue
        if not pd.api.types.is_numeric_dtype(a[name]):
            continue
        x, y = a[name].reindex(shared).to_numpy(float), b[name].reindex(shared).to_numpy(float)
        seen = np.isfinite(x) & np.isfinite(y)
        same = np.mean(np.isclose(x[seen], y[seen])) if seen.any() else 1.0
        (leaky if same > 0.9 else panel).append(name)
    team = [c for c in ALL_TEAM_RATES if c not in today]
    mates = [f"mates_{r.replace('_per100', '')}_{w}" for w in ("on", "bench") for r in BOX_RATES]
    candidates = [*team, *mates, *panel, *BUILT]
    per_side = [*today, *candidates, *[f"shuffled_{c}" for c in candidates]]
    names = ["O:fitted", "D:fitted", *[f"O:{c}" for c in per_side], *[f"D:{c}" for c in per_side]]
    within = within_layer_multipliers(cfg, sorted({f.season for f in folds}), folds, where, shrink)
    within.to_csv(out / "within.csv", index=False)
    full = trainable_panel(cfg)
    scale = second_prior.type_scale(full[full.side == "O"])
    (out / "names.json").write_text(json.dumps(dict(names=names, today=today, candidates=candidates, team=team,
                                                    mates=mates, panel=panel, built=BUILT, leaky=leaky, scale=scale),
                                               indent=0))
    print(f"{len(candidates)} candidates an end ({len(team)} team rates, {len(mates)} teammates, {len(panel)} panel, "
          f"{len(BUILT)} built); {len(names)} columns; dropped as season totals: {leaky}")
    return 0


def teammate_numbers(entries, values, keep):
    """Each of the 13 box rates over his teammates on the court and his team's five when he sits, padded."""
    raw, levels = entries.numbers(values)
    out = {}
    on, bench = entries.on_weight, entries.bench_weight
    for j, stat in enumerate(BOX_RATES):
        noise = entries.noise(values[:, j], keep)
        short = stat.replace("_per100", "")
        out[f"mates_{short}_on"] = pad_toward_level(raw["teammates_{}_on"][:, j], on, levels["teammates"][:, j],
                                                    noise["teammates_{}_on"])
        out[f"mates_{short}_bench"] = pad_toward_level(raw["teammates_{}_bench"][:, j], bench, levels["teammates"][:, j],
                                                       noise["teammates_{}_bench"])
    return out


def usage_numbers(stints: pd.DataFrame, side: str) -> pd.DataFrame:
    """player_id -> the clutch shares and possessions per game, on one end, over the games he played."""
    five, team = (OFF, "team_off") if side == "O" else (DEF, "team_def")
    poss = stints.poss.to_numpy(float)
    clutch = poss * stints.clutch.to_numpy(float)
    games = stints.game_id.astype(str).to_numpy()
    teams = stints[team].to_numpy(dtype=np.int64)
    team_game = (pd.DataFrame({"team": teams, "game": games, "poss": poss, "clutch": clutch})
                 .groupby(["team", "game"], as_index=False)[["poss", "clutch"]].sum())
    long = pd.DataFrame({"player_id": stints[five].to_numpy(dtype=np.int64).ravel(),
                         "team": np.repeat(teams, 5), "game": np.repeat(games, 5),
                         "poss": np.repeat(poss, 5), "clutch": np.repeat(clutch, 5)})
    his = long.groupby(["player_id", "team", "game"], as_index=False)[["poss", "clutch"]].sum()
    his = his.merge(team_game, on=["team", "game"], suffixes=("", "_team"))
    total = his.groupby("player_id")[["poss", "clutch", "poss_team", "clutch_team"]].sum()
    n_games = his.groupby("player_id").game.nunique()
    share = total.poss / total.poss_team.where(total.poss_team > 0, 1.0)
    clutch_share = (total.clutch / total.clutch_team.where(total.clutch_team > 0, 1.0)).where(total.clutch_team > 0,
                                                                                              share)
    over = clutch_share - share
    return pd.DataFrame({"clutch_share_over_share": over,
                         "clutch_share_over_share_padded": over * total.clutch_team / (total.clutch_team + CLUTCH_PAD),
                         "possessions_per_game": total.poss / n_games})


def with_candidates(inputs: dict, meta: dict, players: pd.DataFrame, fit_stints: pd.DataFrame, rng) -> dict:
    """{"O", "D"}: today's inputs followed by every candidate and its shuffled copy, one row per player of `inputs`."""
    team_rates = second_prior.team_rate_numbers(fit_stints)
    everyone = np.unique(np.concatenate([fit_stints[OFF].to_numpy(dtype=np.int64).ravel(),
                                         fit_stints[DEF].to_numpy(dtype=np.int64).ravel()]))
    entries = _Entries(fit_stints, everyone)
    own = inputs["O"][BOX_RATES].astype(float).reindex(everyone).fillna(0.0).to_numpy()
    mates = pd.DataFrame(teammate_numbers(entries, own, np.flatnonzero(entries.on_weight > 0)), index=everyone)
    out = {}
    for side in ("O", "D"):
        frame = inputs[side]
        ids = frame.index.to_numpy(dtype=np.int64)
        rows = players[players.side == side].drop_duplicates("player_id").set_index("player_id")
        usage = usage_numbers(fit_stints, side).reindex(ids)
        weight = second_prior.team_possessions(fit_stints, side).groupby("player_id").poss.sum().reindex(ids).fillna(0.0)
        share = frame.team_possession_share.to_numpy(float)
        centred = share - np.average(share, weights=weight.to_numpy(float))
        usage["box_prior_x_share"] = frame.box_prior.to_numpy(float) * centred
        values = np.column_stack([team_rates[side].reindex(ids)[meta["team"]].to_numpy(float),
                                  mates.reindex(ids)[meta["mates"]].to_numpy(float),
                                  rows.reindex(ids)[meta["panel"]].to_numpy(float),
                                  usage[BUILT].to_numpy(float)])
        means = np.nanmean(values, axis=0)
        values = np.where(np.isfinite(values), values, np.where(np.isfinite(means), means, 0.0))
        shuffled = np.column_stack([rng.permutation(values[:, j]) for j in range(values.shape[1])])
        extra = pd.DataFrame(np.column_stack([values, shuffled]), index=frame.index,
                             columns=[*meta["candidates"], *[f"shuffled_{c}" for c in meta["candidates"]]])
        out[side] = pd.concat([frame[meta["today"]].astype(float), extra], axis=1)
    return out


def build(args) -> int:
    cfg = load_config()
    out = out_dir(cfg)
    meta = json.loads((out / "names.json").read_text())
    within = pd.read_csv(out / "within.csv").set_index("season")
    shrink = replace(TEST.shrink, folds_tag=FOLDS_TAG)
    folds, where = step4_folds(cfg, shrink)
    saved_all = load_models(resolve(cfg, "outputs"), MODELS)
    designs = DesignCache(cfg, keep=1)
    k_today, k_all = len(meta["today"]), len(meta["names"]) // 2 - 1
    today_at = np.concatenate([[0, 1], 2 + np.arange(k_today), 2 + k_all + np.arange(k_today)])
    seasons = SEASONS if not args.seasons else [int(s) for s in args.seasons.split(",")]
    started = time.time()
    for season in seasons:
        path = out / f"season_{season}.npz"
        if path.exists():
            continue
        stints = season_lineup_rows(cfg, season)
        whole = {t: designs.season(season, t) for t in ("pts", TEST.targets.offence, TEST.targets.defence)}
        row = within.loc[season]
        shared = RatingSeason(cfg, whole, design_blocks(whole["pts"])[0],
                              tuple(float(row[c]) for c in WITHIN_COLUMNS), TEST.ridge.penalty)
        saved = season_models(saved_all, season)
        rng = np.random.default_rng(season)
        total = None
        for number, fold in enumerate(f for f in folds if f.season == season):
            directory = where[fold.stem]
            fit_ids, _ = fold_games(directory, fold)
            players = fold_players(directory, fold.stem)
            answers, gap = fold_answers(players, saved)
            assert gap < 1e-9, (fold.stem, gap)
            fit_stints = stints[stints.game_id.astype(str).isin(fit_ids).to_numpy()]
            context = second_prior.context_numbers(fit_stints, answers["O"], answers["D"])
            types = second_prior.standardised_types(players[players.side == "O"], meta["scale"])
            inputs = second_prior.side_inputs(stored_box(players), context, types, INPUT_SET,
                                              second_prior.team_rate_numbers(fit_stints))
            inputs = with_candidates(inputs, meta, players, fit_stints, rng)
            fit = fit_designs(season, cfg, whole, fit_ids)
            if number == 0:
                ridge_gap = check_fit_designs(fold, fit, shared.penalty)
                assert ridge_gap < 1e-3, (fold.stem, ridge_gap)
            design = fit[TEST.targets.offence]
            z, context_columns = design_blocks(design)
            ids = design.spec.players["player_id"].to_numpy(dtype=np.int64)
            team_poss = {side: second_prior.team_possessions(fit_stints, side) for side in ("O", "D")}
            priors = rating_line.prior_columns(inputs, ids, team_poss)
            offence, defence, explained = rating_line.ridge_parts(
                z, context_columns, np.asarray(design.w, dtype=float), np.asarray(design.y, dtype=float),
                np.asarray(fit[TEST.targets.defence].y, dtype=float), priors, shared.penalty)
            sums = rating_line.held_back_sums(held_back_stints(shared, directory, fold), (offence, defence), priors,
                                              explained, shared.within)
            total = sums if total is None else total + sums
        with np.load(resolve(cfg, "outputs") / "second_prior" / LINE / f"sums_{season}.npz") as line:
            gap_q = np.max(np.abs(total.Q[np.ix_(today_at, today_at)] - line["Q"])) / np.max(np.abs(line["Q"]))
            gap_c = np.max(np.abs(total.c[today_at] - line["c"])) / np.max(np.abs(line["c"]))
            gap_y = abs(total.yy - float(line["yy"])) / float(line["yy"])
        np.savez(path, Q=total.Q, c=total.c, yy=total.yy, sw=total.sw, gap=max(gap_q, gap_c, gap_y))
        print(f"{season}: today's columns against {LINE}'s sums, largest relative gap Q {gap_q:.1e}, c {gap_c:.1e}, "
              f"yy {gap_y:.1e} ({time.time() - started:.0f}s)", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "build"))
    parser.add_argument("--seasons", default="")
    args = parser.parse_args()
    return prepare(args) if args.command == "prepare" else build(args)


if __name__ == "__main__":
    sys.exit(main())
