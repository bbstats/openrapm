"""Check 152, the booster tune on the decider: the box-score prior's LightGBM settings chosen by the with-or-without-you
test on the held-back games (check 151), not by the prior's error against career labels.

Why (the owner, 2026-10-16): today's settings came from a 2026-10-08 tune on the prior's error against career labels, on
2015 and 2026 only; the decider never chose them.  "If we haven't done that, we are basically just wasting our time."

Words used below
    tune season     one of the ten seasons the tune scores, 1998, 2001, ..., 2025; the other twenty are kept untouched
                    for the confirming build
    fold            as in `openrapm folds`: a tune season rated from three quarters of its games, the other quarter
                    held back; three repeats of four folds, the same seeded draw as every fold folder
    cache           what does not depend on the booster settings, built once: each tune season's training rows (step 1's
                    labels and the panel, per feature set), and per fold the inputs rebuilt from the fit games, the
                    ridge's design on them, the held-back team-games and the held-back stints
    trial           one setting of both boosters and one feature set: the boosters retrained for every tune season (the
                    full fit and the five player folds, both sides), each fold rated through steps 2-3 as `openrapm
                    folds` rates it, step 4 fitted on the ten seasons' folds by the test rule (8 nearest), and the
                    decider's error, stint by stint, on every fold
    score           the mean over the ten tune seasons of the decider's error after step 4, the `step4`/`all` reading of
                    check 151; below is better
    reference       the incumbent's settings (lgb1_cols06, feature set no_career_counts) as a trial: each trial is read
                    against it season by season, the mean change, its standard error and the z, as check 151 reads

Step 4 here fits on the ten tune seasons' folds only, where the shipped build fits on all thirty, so a trial's numbers
are not check 151's; the reference is built the same way, so the comparison is fair.  The winner is confirmed through
the full chain (scripts/experiment_chain.sh), with its own noise control, on all thirty seasons.

The search (Optuna's TPE): straight-line leaves on or off, the same on both sides (the same-reason rule); per side the
learning rate, trees, leaves, depth, rows per leaf, both penalties, the column share, the bins, the smoothing and the
straight-line leaves' penalty; the era marker on or off with the hustle inputs on.  The first two trials are the
incumbent's settings and experiment 71's, both with the hustle inputs.

    python scripts/checks/152_decider_tune.py cache --tag tune1                    (about 15 minutes, once)
    python scripts/checks/152_decider_tune.py trial --tag tune1 --params lgb1_cols06_flat --features hustle \
        --compare flathustle                                                       (one trial; checks the cache)
    python scripts/checks/152_decider_tune.py tune --tag tune1 --trials 60         (resumes where it stopped)
    python scripts/checks/152_decider_tune.py cull --tag tune1                     (the input cull, below; resumes)
    python scripts/checks/152_decider_tune.py cull-table --tag tune1               (its full table)

Writes under outputs/tune/<tag>/: season_<season>.pkl (the cache), reference.json, study.db (Optuna's), trials.csv
(one row per trial and season) and best.json (the best trial's settings as a params/ file).
"""
from __future__ import annotations

import os

THREADS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
for _name in THREADS:
    os.environ.setdefault(_name, "1")

import argparse  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import pickle  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import zlib  # noqa: E402
from dataclasses import replace  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.sparse as sp  # noqa: E402

TUNE_SEASONS = tuple(range(1998, 2026, 3))          # 1998, 2001, ..., 2025
FEATURE_SETS = ("no_career_counts", "hustle", "hustle_era_marker")
REFERENCE = ("lgb1_cols06", "no_career_counts")
NEAREST = 8
STAGES = ("step3", "step4")
SLICES = ("all", "dnp")


def _load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(file))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _wowy():
    return _load("wowy_check", "151_wowy.py")


def tune_dir(tag: str) -> Path:
    from openrapm.data.config import load_config, resolve
    return resolve(load_config(), "outputs") / "tune" / tag


def settings_for(feature_set: str):
    from openrapm.settings import TEST
    return replace(TEST, prior=replace(TEST.prior, feature_set=feature_set))


# ------------------------------------------------------------------------------------------------- the cache
def duck_design(ridge: dict) -> SimpleNamespace:
    """What prior_rapm.fit reads of a design, from the cached pieces: the same matrix, so the same bits."""
    spec = SimpleNamespace(n_players=ridge["n_players"], context_names=ridge["context_names"],
                           players=pd.DataFrame({"player_id": ridge["player_ids"]}))
    return SimpleNamespace(spec=spec, X=ridge["X"], y=ridge["y"], w=ridge["w"],
                           rows=pd.DataFrame({"poss": ridge["poss"]}))


def ridge_pieces(design) -> dict:
    n_players, n_context = design.spec.n_players, len(design.spec.context_names)
    return dict(X=design.X[:, :2 * n_players + n_context].tocsr(), y=np.asarray(design.y), w=np.asarray(design.w),
                poss=design.rows["poss"].to_numpy(), player_ids=design.spec.players["player_id"].to_numpy(),
                n_players=n_players, context_names=list(design.spec.context_names))


def rated_inputs(panel: pd.DataFrame, feature_set: str) -> dict:
    """{side: (player ids, the boosters' input matrix)} of a fold's rebuilt rows, for one feature set."""
    from openrapm import box_prior
    from openrapm.features import FEATURE_SETS as SETS
    from openrapm.features import ROW_FEATURES
    out = {}
    for side in ("O", "D"):
        names = list(SETS[feature_set][side])
        rows = box_prior.rated_rows(panel[(panel.side == side) & (panel.possessions > 0)], names)
        out[side] = (rows.player_id.to_numpy(), rows[names + ROW_FEATURES].to_numpy(float))
    return out


def stint_pieces(wowy, fold, teams: dict, design) -> dict:
    """The held-back stints of one fold as check 151's `stint_fold_errors` reads them, before any rating."""
    absent = wowy.regulars_absent(fold, teams)
    key = pd.DataFrame({"game_id": teams["game_id"].astype(str), "is_home_off": teams["is_home_off"].astype(bool),
                        "team_off": teams["team_off"], "team_def": teams["team_def"], "dnp": absent})
    games = design.games.set_index("game_idx").game_id.astype(str)
    rows = design.rows.assign(game_id=design.rows.game_idx.map(games).to_numpy())
    rows = rows.reset_index().merge(key, on=["game_id", "is_home_off"], how="inner")
    index = rows["index"].to_numpy()
    y, w = design.y[index], design.w[index]
    columns = design.spec.players.set_index("player_column").player_id
    lineup_o = columns.reindex(design.parts["lineup_o"][index].ravel()).to_numpy().reshape(-1, 5)
    lineup_d = columns.reindex(design.parts["lineup_d"][index].ravel()).to_numpy().reshape(-1, 5)
    off_codes, _ = pd.factorize(rows.team_off)
    def_codes, _ = pd.factorize(rows.team_def)
    n = len(rows)
    levels = sp.hstack([
        sp.csr_matrix((np.ones(n), (np.arange(n), off_codes)), shape=(n, off_codes.max() + 1)),
        sp.csr_matrix((np.ones(n), (np.arange(n), def_codes)), shape=(n, def_codes.max() + 1))[:, 1:],
        sp.csr_matrix(rows.is_home_off.to_numpy(float)[:, None]),
        sp.csr_matrix((rows.phase == "PO").to_numpy(float)[:, None])]).tocsr()
    return dict(y=y, w=w, lineup_o=lineup_o, lineup_d=lineup_d, levels=levels, dnp=rows.dnp.to_numpy())


def build_cache(tag: str, seasons, check_models: str) -> int:
    """Each tune season's cache file: training rows per feature set, and per fold everything a trial reads."""
    from openrapm import box_prior, trade_set
    from openrapm.commands import within_season as ws
    from openrapm.commands.build import accumulate_rapm, unseen_seasons
    from openrapm.data.careers import careers_before
    from openrapm.data.config import load_config, resolve
    from openrapm.data.designs import DesignCache
    from openrapm.data.panel import SeasonSources, SeasonTables, share_cap, trainable_panel
    from openrapm.data.roles import player_season_inputs, roles_path
    from openrapm.data.seasons import trainable
    from openrapm.data.tables import team_of_rows
    from openrapm.data.within_folds import Fold, _base, _rated, _side_blocks
    from openrapm.features import FEATURE_SETS as SETS
    from openrapm.features import PANEL_INPUTS
    from openrapm.rapm import career_labels

    wowy = _wowy()
    cfg = load_config()
    out_dir = tune_dir(tag)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    settings = settings_for("hustle")
    designs = DesignCache(cfg)
    print("step 1's sums over every trainable season", flush=True)
    sums = accumulate_rapm(designs, trainable(cfg).seasons, settings)
    panel = trainable_panel(cfg)
    panel = panel[panel.possessions > 0].reset_index(drop=True)
    roles = pd.read_parquet(roles_path(cfg))
    inputs = player_season_inputs(roles, cap=share_cap(cfg))
    debut = careers_before(cfg, roles)
    saved_all = ws.load_models(resolve(cfg, "outputs"), check_models)
    print(f"  done ({time.time() - started:.0f}s)", flush=True)
    for season in seasons:
        path = out_dir / f"season_{season}.pkl"
        if path.exists():
            print(f"{season}: cached already", flush=True)
            continue
        unseen = unseen_seasons(season, settings)
        labels = career_labels(sums, unseen, settings.labels, settings.targets)
        train = {}
        for feature_set in FEATURE_SETS:
            train[feature_set] = {}
            for side, column in (("O", "offense"), ("D", "defense")):
                names = list(SETS[feature_set][side])
                training = panel[(panel.side == side) & ~panel.season.isin(unseen)]
                rows = box_prior.training_rows(labels[side], training, column, names, settings.prior.chunk_sizes,
                                               PANEL_INPUTS)
                keep = [*names, *box_prior.ROW_FEATURES, "label", "row_weight"]
                train[feature_set][side] = rows[keep].copy()
        saved = ws.season_models(saved_all, season)
        full = {target: designs.season(season, target) for target in ws.WORLD_TARGETS}
        run = ws.SeasonRun(season, cfg, full, SeasonSources(SeasonTables(season, cfg), roles, inputs, debut), saved,
                           out_dir)
        pts = full[ws.SCORE_TARGET]
        folds = []
        for repeat in range(ws.REPEATS):
            assignment = ws.assign_folds(run.game_ids.to_numpy(), ws.N_FOLDS, season, repeat)
            for k in range(ws.N_FOLDS):
                stem = f"{season}_r{repeat}_f{k}"
                test_ids = set(assignment.index[assignment == k])
                fit_ids = set(assignment.index) - test_ids
                games = pts.games
                test_idx = np.sort(games.loc[games["game_id"].astype(str).isin(test_ids), "game_idx"].to_numpy())
                world = run.world(fit_ids)
                failed = ws.failed_checks(world.checks)
                if failed:
                    raise SystemExit(f"{stem}: leak checks failed: {failed}")
                # the check models' rating of this fold: the cache's skeleton (rated players, the regulars), and the
                # numbers a trial on the same settings must give back
                rated = ws.rate_world(world, saved)
                players = ws.player_table(rated)
                keep = pts.rows["game_idx"].isin(test_idx).to_numpy()
                pooled = trade_set.team_game_design(pts.subset(keep))
                teams_frame = team_of_rows(pooled.keys, cfg)
                shares = pooled.Z.tocsr()
                test = dict(shares=shares, player_ids=np.asarray(pooled.player_ids), y=np.asarray(pooled.y, float),
                            w=np.asarray(pooled.w, float), F=np.asarray(pooled.F, float),
                            game_idx=pooled.keys.game_idx.to_numpy())
                teams = dict(team_off=teams_frame.team_off.to_numpy(), team_def=teams_frame.team_def.to_numpy(),
                             game_id=pooled.keys.game_id.astype(str).to_numpy(),
                             phase=pooled.keys.phase.astype(str).to_numpy(),
                             is_home_off=pooled.keys.is_home_off.to_numpy())
                table = rated.table
                ratings = pd.DataFrame({"o": table.offense, "d": table.defense, "poss": table.possessions,
                                        "player_id": table.player_id})
                fill = trade_set.replacement_fill(ratings[ratings.poss > 0].reset_index(drop=True),
                                                  max_poss=ws.FILL[0], shrink=ws.FILL[1])
                rated_players = _rated(players)
                rows, blocks = _side_blocks(players, rated_players, shares, test["player_ids"])
                fold = Fold(stem=stem, season=season, repeat=repeat, fold=k, fill_o=fill[0], fill_d=fill[1],
                            y=test["y"], w=test["w"], F=test["F"], game_idx=test["game_idx"],
                            base=_base(shares, test["player_ids"], rated_players, fill[0], fill[1]), rows=rows,
                            Z=blocks)
                world_panel = world.panel[world.panel.possessions > 0].reset_index(drop=True)
                played = world.roles[world.roles.poss_on > 0]
                main_team = (played.sort_values(["player_id", "poss_on"], ascending=[True, False])
                             .drop_duplicates("player_id").set_index("player_id").team_id)
                folds.append(dict(
                    stem=stem, repeat=repeat, fold=k,
                    keys=world_panel[["player_id", "side"]].copy(),
                    inputs={feature_set: rated_inputs(world_panel, feature_set) for feature_set in FEATURE_SETS},
                    ridge={target: ridge_pieces(world.designs[target]) for target in ws.SETTINGS.targets.distinct()},
                    poss_def=prior_rapm_poss_def(world.designs[ws.SETTINGS.targets.offence]),
                    main_team=main_team, test=test, teams=teams,
                    stints=stint_pieces(wowy, fold, teams, pts),
                    check=dict(models=check_models, rating_off=players.set_index(["player_id", "side"]).rating_off,
                               rating_def=players.set_index(["player_id", "side"]).rating_def)))
                print(f"  {stem}: {len(fit_ids)} fit / {len(test_ids)} test games ({time.time() - started:.0f}s)",
                      flush=True)
        with open(path, "wb") as handle:
            pickle.dump(dict(season=season, unseen=list(unseen), train=train, folds=folds), handle, protocol=5)
        designs.clear()
        print(f"{season}: cached, {path.stat().st_size / 1e6:.0f} MB ({time.time() - started:.0f}s)", flush=True)
    return 0


def prior_rapm_poss_def(design) -> dict:
    from openrapm.prior_rapm import defence_possessions
    return defence_possessions(design)


def guard_design(design) -> dict:
    """What scoring.predict_season reads of a scored season's points design, kept small: the player columns and the
    home column, the target, the weights and the rows' keys."""
    n_players = design.spec.n_players
    home = design.X[:, design.spec.context_column("home")].tocsr()
    return dict(X=sp.hstack([design.X[:, :2 * n_players], home]).tocsr(), y=np.asarray(design.y),
                w=np.asarray(design.w), player_ids=design.spec.players["player_id"].to_numpy(), n_players=n_players,
                rows=design.rows[["poss", "game_idx", "is_home_off"]].reset_index(drop=True))


def duck_guard(stored: dict) -> SimpleNamespace:
    n_players = stored["n_players"]
    spec = SimpleNamespace(n_players=n_players, players=pd.DataFrame({"player_id": stored["player_ids"]}),
                           context_column=lambda name: 2 * n_players)
    return SimpleNamespace(spec=spec, X=stored["X"], y=stored["y"], w=stored["w"], rows=stored["rows"])


def build_whole_cache(tag: str, seasons, check_models: str) -> int:
    """Per tune season, the whole season's rebuilt inputs and ridge pieces (whole_<season>.pkl, read by the season's
    worker), and the next-season check's pieces (guard_<season>.pkl, read by the parent): the points designs of the
    seasons either side, and the season's lineups for step 5."""
    from openrapm import swap_test
    from openrapm.commands import within_season as ws
    from openrapm.data.careers import careers_before
    from openrapm.data.config import load_config, resolve
    from openrapm.data.designs import DesignCache
    from openrapm.data.panel import SeasonSources, SeasonTables, share_cap
    from openrapm.data.roles import player_season_inputs, roles_path
    from openrapm.data.stints import load_season_stints
    cfg = load_config()
    out_dir = tune_dir(tag)
    started = time.time()
    designs = DesignCache(cfg)
    roles = pd.read_parquet(roles_path(cfg))
    inputs = player_season_inputs(roles, cap=share_cap(cfg))
    debut = careers_before(cfg, roles)
    saved_all = ws.load_models(resolve(cfg, "outputs"), check_models)
    base = pd.read_parquet(resolve(cfg, "outputs") / f"season_ratings_{check_models}_step3.parquet")
    margin_clip = float(cfg.get("margin_clip", 25))
    for season in seasons:
        whole_path, guard_path = out_dir / f"whole_{season}.pkl", out_dir / f"guard_{season}.pkl"
        if whole_path.exists() and guard_path.exists():
            print(f"{season}: cached already", flush=True)
            continue
        saved = ws.season_models(saved_all, season)
        full = {target: designs.season(season, target) for target in ws.WORLD_TARGETS}
        run = ws.SeasonRun(season, cfg, full, SeasonSources(SeasonTables(season, cfg), roles, inputs, debut), saved,
                           out_dir)
        world = run.world(set(run.game_ids))
        rated = ws.rate_world(world, saved)
        theirs = base[base.season == season].set_index("player_id")
        mine = rated.table.set_index("player_id")
        worst = float(np.nanmax(np.abs(mine.offense - theirs.offense.reindex(mine.index))))
        if worst > ws.REPRODUCE_TOLERANCE:
            raise SystemExit(f"{season}: the whole-season rebuild misses {check_models}'s step 3 by {worst:.1e}")
        world_panel = world.panel[world.panel.possessions > 0].reset_index(drop=True)
        whole = dict(
            inputs={feature_set: rated_inputs(world_panel, feature_set) for feature_set in FEATURE_SETS},
            ridge={target: ridge_pieces(world.designs[target]) for target in ws.SETTINGS.targets.distinct()},
            poss_def=prior_rapm_poss_def(world.designs[ws.SETTINGS.targets.offence]),
            check=mine[["offense", "defense"]].copy())
        with open(whole_path, "wb") as handle:
            pickle.dump(whole, handle, protocol=5)
        designs.clear()
        stints = load_season_stints(season, cfg, swap_test.STINT_COLUMNS)
        guard = dict(lineups=swap_test.prepare(swap_test.season_rows(stints, margin_clip=margin_clip), season),
                     scored={scored: guard_design(designs.season(scored, "pts"))
                             for scored in (season - 1, season + 1)})
        with open(guard_path, "wb") as handle:
            pickle.dump(guard, handle, protocol=5)
        designs.clear()
        print(f"{season}: whole season and guard cached; step 3 reproduced to {worst:.1e} "
              f"({time.time() - started:.0f}s)", flush=True)
    return 0


class Guard:
    """The next-season check on the tune seasons, in the parent: the whole seasons' step-3 tables through step 4 (the
    folds' multipliers) and step 5, then each season's ratings predicting the seasons either side of it, scored per
    team-game as scripts/checks/63_yoy.py scores them (fill 0, the level and home edge refitted on the scored games)."""

    def __init__(self, tag: str, seasons):
        from openrapm.data.config import load_config
        from openrapm.data.panel import panel_and_trainable_seasons
        from openrapm.data.roles import main_teams
        from openrapm.data.tables import player_names
        self.cfg = load_config()
        self.seasons = list(seasons)
        self.lineups, self.scored = {}, {}
        for season in self.seasons:
            with open(tune_dir(tag) / f"guard_{season}.pkl", "rb") as handle:
                stored = pickle.load(handle)
            self.lineups[season] = stored["lineups"]
            for scored, design in stored["scored"].items():
                self.scored[(season, scored)] = duck_guard(design)
        self.panel, self.trainable = panel_and_trainable_seasons(self.cfg)
        self.main_team = main_teams(self.cfg).set_index(["player_id", "season"]).team_id
        self.names = player_names(self.cfg)

    def swap(self, table: pd.DataFrame) -> pd.DataFrame:
        """Step 5 on a step-4 table, `build.swap_seasons` with the lineups cached."""
        from openrapm import swap_adjust, swap_test
        from openrapm.settings import TEST
        raw = swap_test.ratings_in_raw_sign(table)
        features = swap_adjust.type_features(self.panel, table)
        n_folds = TEST.swap.n_folds
        fold_of = swap_adjust.player_folds(raw.player_id, n_folds)
        swaps = {season: swap_adjust.season_swaps(self.lineups[season], raw[raw.season == season],
                                                  swap_adjust.season_features(features, season), fold_of, n_folds)
                 for season in sorted(int(s) for s in raw.season.unique())}
        predictions = swap_adjust.type_predictions(features, swaps, fold_of, self.trainable, TEST.swap)
        adjusted, _ = swap_adjust.adjust(table, raw, swaps, predictions, self.main_team, TEST.swap)
        return adjusted

    def rows(self, step3: pd.DataFrame, multipliers: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """(one row per rated and scored season: the team-game error, the step-5 table)."""
        from openrapm import prior_rapm, prior_shrink
        published = prior_rapm.published_table(step3, self.names)
        adjusted = self.swap(prior_shrink.shrink(published, multipliers))
        return self.score(adjusted), adjusted

    def score(self, table: pd.DataFrame) -> pd.DataFrame:
        from openrapm import scoring
        out = []
        for (rated, scored), design in self.scored.items():
            rows = table[table.season == rated]
            ratings = pd.DataFrame({"player_id": rows.player_id.to_numpy(), "o": rows.rating_off.to_numpy(float),
                                    "d": -rows.rating_def.to_numpy(float), "poss": rows.poss_off.to_numpy(float)})
            result = scoring.score(scoring.predict_season(scoring.Ratings(ratings, fill_o=0.0, fill_d=0.0), design))
            out.append(dict(rated=rated, scored=scored, tg=float(result["tg"]), mse=float(result["mse"])))
        return pd.DataFrame(out)


def guard_paired(mine: pd.DataFrame, theirs: pd.DataFrame, value: str = "tg") -> dict:
    key = ["rated", "scored"]
    return paired(mine.set_index(key)[value], theirs.set_index(key)[value])


# ------------------------------------------------------------------------------------------------- an input changed
# The input cull (the owner, 2026-10-16: "time to do boruta (on the new test)").  Boruta keeps an input when it beats a
# shuffled copy of itself, its shadow.  Here the decider is the yardstick in place of the booster's own importance: the
# shipped boosters' straight-line leaves have no SHAP values, and a booster's importance is read on its own training
# labels, which the decider replaced.  A change is (side, arm, input):
#     drop      the input taken off that side's booster
#     shadow    the input's values shuffled among the rows that have one, in training and in rating; missing stays
#               missing, so what is left of it is its missing pattern (for the hustle counts, a marker of the seasons
#               before 2017) and noise
# The booster's seed (LightGBM's random_state) draws the inputs each tree is shown; a trial at another seed, nothing
# else changed, is the noise control.
CULL_ARMS = ("drop", "shadow")


def side_changes(changes: tuple, side: str) -> list:
    """[(arm, input)] of the changes on one side."""
    return [(arm, name) for change_side, arm, name in changes if change_side == side]


def booster_seed(seed: int) -> None:
    """Every LightGBM fit in this process from here on drawn with `seed` (box_prior pins random_state at 0)."""
    from openrapm import box_prior
    box_prior.LIGHTGBM_FIXED = {**box_prior.LIGHTGBM_FIXED, "random_state": int(seed)}


def shadow(values: np.ndarray, seed: int, name: str, stream: str) -> np.ndarray:
    """`values` shuffled among the rows that have one; the draw fixed by the seed, the input and the stream."""
    out = np.asarray(values, dtype=float).copy()
    present = np.flatnonzero(~np.isnan(out))
    rng = np.random.default_rng([int(seed), zlib.crc32(f"{name}|{stream}".encode())])
    out[present] = out[rng.permutation(present)]
    return out


def changed_train(train: pd.DataFrame, inputs: list, changes: list, seed: int, stream: str
                  ) -> tuple[pd.DataFrame, list]:
    """(the training rows, the booster's input names) after one side's changes."""
    inputs = list(inputs)
    if not changes:
        return train, inputs
    train = train.copy()
    for arm, name in changes:
        if arm == "drop":
            inputs.remove(name)
        else:
            train[name] = shadow(train[name].to_numpy(float), seed, name, f"{stream}_train")
    return train, inputs


def changed_matrix(matrix: np.ndarray, inputs: list, changes: list, seed: int, stream: str) -> np.ndarray:
    """A rated input matrix, columns in the order of `inputs`, after one side's changes: dropped columns removed,
    shadowed ones shuffled."""
    if not changes:
        return matrix
    matrix = np.asarray(matrix, dtype=float).copy()
    for arm, name in changes:
        if arm == "shadow":
            column = inputs.index(name)
            matrix[:, column] = shadow(matrix[:, column], seed, name, stream)
    dropped = [inputs.index(name) for arm, name in changes if arm == "drop"]
    return np.delete(matrix, dropped, axis=1)


# Several boosters averaged (the owner, 2026-10-17: "Better to average or to do bagging?"), K of them per side, each
# answering for a player as one booster does, the K answers averaged:
#     one       one booster at the trial's seed, as shipped
#     seeds     boosters at seeds s, s+1, ..., s+K-1 on the same rows: only the inputs each tree is shown differ
#     bags      the same seeds, each booster on a bootstrap of the players (box_prior.player_bootstrap): with s 0,
#               the bags `openrapm rate --bags K` fits
ENSEMBLES = ("one", "seeds", "bags")


def parse_ensemble(text: str) -> tuple:
    kind, size = text.split(":")
    if kind not in ENSEMBLES or int(size) < 1:
        raise SystemExit(f"--ensemble {text}: one of {ENSEMBLES} and a size of at least 1")
    return kind, int(size)


def parse_changes(text: str, arm: str) -> tuple:
    """"O:weight_lb,D:height_in" -> (("O", arm, "weight_lb"), ("D", arm, "height_in"))."""
    out = []
    for item in filter(None, (part.strip() for part in text.split(","))):
        side, name = item.split(":", 1)
        out.append((side, arm, name))
    return tuple(out)


# ------------------------------------------------------------------------------------------------- one trial
class SeasonWorker:
    """One tune season held in memory: retrains its boosters for a trial, rates its folds, scores them."""

    def __init__(self, path: Path):
        with open(path, "rb") as handle:
            self.cache = pickle.load(handle)
        self.season = int(self.cache["season"])
        whole = path.with_name(f"whole_{self.season}.pkl")
        self.whole = pickle.loads(whole.read_bytes()) if whole.exists() else None
        self.wowy = _wowy()
        self.audit = self.wowy._audit()
        self.folds = None

    def rate(self, params: dict, feature_set: str, changes: tuple = (), seed: int = 0, ensemble: tuple = ("one", 1)
             ) -> tuple[list, pd.DataFrame | None]:
        """Every fold of the season rated through steps 2-3 on the trial's boosters, and the whole season when it is
        cached: (the `Fold`s, the whole season's centred step-3 table in raw sign, or None).  `changes` and `seed` as
        in `changed_matrix` and `booster_seed`; `ensemble` as in `ENSEMBLES`."""
        from openrapm import box_prior, trade_set
        from openrapm.commands import within_season as ws
        from openrapm.data.within_folds import Fold, _base, _rated, _side_blocks
        from openrapm.features import FEATURE_SETS as SETS
        from openrapm.features import ROW_FEATURES
        settings = settings_for(feature_set)
        kind, size = ensemble
        models, names = {}, {}
        for side in ("O", "D"):
            names[side] = list(SETS[feature_set][side]) + ROW_FEATURES
            train = self.cache["train"][feature_set][side]
            train, inputs = changed_train(train, names[side], side_changes(changes, side), seed, f"{self.season}")
            models[side] = []
            for member in range(int(size)):
                booster_seed(seed + member)
                rows = train if kind != "bags" else box_prior.player_bootstrap(train, seed + member)[0]
                models[side].append(box_prior.fit_out_of_player(rows, inputs, params[side], settings.prior.n_folds,
                                                                settings.prior.fold_assignment))
        folds = []
        for cached in self.cache["folds"]:
            table = self.step3(cached, models, feature_set, settings, names, changes, seed, cached["stem"])
            ratings = pd.DataFrame({
                "player_id": table.player_id, "rating_off": table.offense, "rating_def": -table.defense,
                "prior_off": table.prior_offense, "prior_def": -table.prior_defense,
                "poss_off": table.possessions, "poss_def": table.poss_def})
            ratings["u_off"] = ratings.rating_off - ratings.prior_off
            ratings["u_def"] = ratings.rating_def - ratings.prior_def
            ratings["team_id"] = ratings.player_id.map(cached["main_team"]).fillna(-1).astype(np.int64)
            players = cached["keys"].merge(ratings, on="player_id", how="left")
            plain = pd.DataFrame({"o": table.offense, "d": table.defense, "poss": table.possessions,
                                  "player_id": table.player_id})
            fill_o, fill_d = trade_set.replacement_fill(plain[plain.poss > 0].reset_index(drop=True),
                                                        max_poss=ws.FILL[0], shrink=ws.FILL[1])
            test = cached["test"]
            rated = _rated(players)
            rows, blocks = _side_blocks(players, rated, test["shares"], test["player_ids"])
            folds.append(Fold(stem=cached["stem"], season=self.season, repeat=cached["repeat"], fold=cached["fold"],
                              fill_o=fill_o, fill_d=fill_d, y=test["y"], w=test["w"], F=test["F"],
                              game_idx=test["game_idx"],
                              base=_base(test["shares"], test["player_ids"], rated, fill_o, fill_d),
                              rows=rows, Z=blocks))
        self.folds = folds
        whole = None if self.whole is None else self.step3(self.whole, models, feature_set, settings, names, changes,
                                                           seed, f"{self.season}_whole")
        return folds, whole

    def step3(self, cached: dict, models: dict, feature_set: str, settings, names: dict, changes: tuple, seed: int,
              stream: str) -> pd.DataFrame:
        """Steps 2-3 on one set of cached inputs and ridge pieces: the centred table in raw sign."""
        from openrapm import prior_rapm
        prior = {}
        for side in ("O", "D"):
            ids, matrix = cached["inputs"][feature_set][side]
            matrix = changed_matrix(matrix, names[side], side_changes(changes, side), seed, stream)
            prior[side] = dict(zip(ids, np.mean([_predict(model, ids, matrix) for model in models[side]], axis=0)))
        targets = settings.targets
        fits = {name: prior_rapm.fit(duck_design(cached["ridge"][name]), prior["O"], prior["D"],
                                     settings.ridge.penalty) for name in targets.distinct()}
        return prior_rapm.centre(prior_rapm.season_table(fits[targets.offence], fits[targets.defence],
                                                         cached["poss_def"], self.season))

    def compare(self) -> dict:
        """The largest rating difference against the check models' folds (a trial on the same settings: 0)."""
        worst = {"off": 0.0, "def": 0.0}
        for fold, cached in zip(self.folds, self.cache["folds"]):
            for side in ("O", "D"):
                rows = fold.rows[side].set_index(["player_id", "side"])
                for part in ("off", "def"):
                    theirs = cached["check"][f"rating_{part}"].reindex(rows.index)
                    worst[part] = max(worst[part], float(np.nanmax(np.abs(rows[f"rating_{part}"] - theirs))))
        return worst

    def score(self, step4: tuple) -> dict:
        """{(stage, slice): (sum of w x WOWY miss squared, sum of w)} over the season's folds, check 151's stint way."""
        sums: dict = {}
        for fold, cached in zip(self.folds, self.cache["folds"]):
            stints = cached["stints"]
            y, w, levels, dnp = stints["y"], stints["w"], stints["levels"], stints["dnp"]
            lineup_o, lineup_d = pd.DataFrame(stints["lineup_o"]), pd.DataFrame(stints["lineup_d"])
            keep = w > 0
            n = len(y)
            for stage, (rating_off, allowed) in self.wowy.stage_ratings(self.audit, fold, step4).items():
                pred = (lineup_o.apply(lambda c: c.map(rating_off)).fillna(fold.fill_o).to_numpy().sum(axis=1)
                        + lineup_d.apply(lambda c: c.map(allowed)).fillna(fold.fill_d).to_numpy().sum(axis=1))
                left = np.zeros(n)
                left[keep] = self.wowy.wowy_miss((y - pred)[keep], levels[keep], w[keep])
                for slice_, mask in (("all", keep), ("dnp", keep & dnp)):
                    acc = sums.setdefault((stage, slice_), [0.0, 0.0])
                    acc[0] += float(np.sum(w[mask] * left[mask] ** 2))
                    acc[1] += float(w[mask].sum())
        return sums


def _predict(model, ids: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """box_prior.predict on a cached input matrix: each player answered by the fold model that never saw him."""
    out = model.full.predict(matrix)
    for fold_model, held_out in zip(model.fold_models, model.held_out_players):
        mine = np.fromiter((pid in held_out for pid in ids), dtype=bool, count=len(ids))
        if mine.any():
            out[mine] = fold_model.predict(matrix[mine]) + 0.0
    return out


def _worker(path: str, connection) -> None:
    worker = SeasonWorker(Path(path))
    connection.send(("ready", worker.season))
    while True:
        message = connection.recv()
        if message[0] == "stop":
            return
        try:
            if message[0] == "rate":
                started = time.time()
                folds, whole = worker.rate(*message[1:])
                connection.send(("rated", folds, whole, time.time() - started))
            elif message[0] == "compare":
                connection.send(("compared", worker.compare()))
            elif message[0] == "score":
                connection.send(("scored", worker.score(message[1])))
        except Exception as error:          # noqa: BLE001 -- the parent reports it and stops the trial
            import traceback
            connection.send(("error", f"{worker.season}: {error}\n{traceback.format_exc()}"))


class Pool:
    """One worker process per tune season, each holding its cache."""

    def __init__(self, tag: str, seasons):
        context = mp.get_context("spawn")
        self.links = {}
        for season in seasons:
            parent, child = context.Pipe()
            process = context.Process(target=_worker, args=(str(tune_dir(tag) / f"season_{season}.pkl"), child),
                                      daemon=True)
            process.start()
            self.links[season] = (parent, process)
        for season, (link, _) in self.links.items():
            self._expect(link, "ready")

    @staticmethod
    def _expect(link, kind: str):
        message = link.recv()
        if message[0] == "error":
            raise RuntimeError(message[1])
        if message[0] != kind:
            raise RuntimeError(f"expected {kind}, got {message[0]}")
        return message

    def trial(self, params: dict, feature_set: str, compare: bool = False, def_factor: float = 1.0,
              guard: Guard | None = None, changes: tuple = (), seed: int = 0, ensemble: tuple = ("one", 1)
              ) -> tuple[pd.DataFrame, dict]:
        """(one row per season, stage and slice: the decider's error; timings, comparisons and, with `guard`, the
        next-season check's rows in extra["guard"]).  `def_factor` multiplies both of step 4's defensive factors;
        `changes` and `seed` as in `SeasonWorker.rate`."""
        from openrapm.prior_shrink import multipliers_by_season
        started = time.time()
        for link, _ in self.links.values():
            link.send(("rate", params, feature_set, tuple(changes), int(seed), tuple(ensemble)))
        folds, seconds, wholes = [], {}, []
        for season, (link, _) in self.links.items():
            _, season_folds, whole, took = self._expect(link, "rated")
            folds.extend(season_folds)
            wholes.append(whole)
            seconds[season] = took
        extra = {"rate_seconds_max": max(seconds.values()), "rate_seconds_sum": sum(seconds.values())}
        if compare:
            for link, _ in self.links.values():
                link.send(("compare",))
            extra["compare"] = {season: self._expect(link, "compared")[1] for season, (link, _) in self.links.items()}
        multipliers = multipliers_by_season(sorted(self.links), folds, "test", NEAREST, free_games=True)
        multipliers["multiplier_def"] *= def_factor
        multipliers["games_multiplier_def"] *= def_factor
        if guard is not None:
            guard_started = time.time()
            extra["guard"], _ = guard.rows(pd.concat(wholes, ignore_index=True), multipliers)
            extra["guard_seconds"] = time.time() - guard_started
        multipliers = multipliers.set_index("season")
        for season, (link, _) in self.links.items():
            row = multipliers.loc[season]
            link.send(("score", (float(row.multiplier_off), float(row.multiplier_def),
                                 float(row.games_multiplier_off), float(row.games_multiplier_def))))
        rows = []
        for season, (link, _) in self.links.items():
            for (stage, slice_), (top, bottom) in self._expect(link, "scored")[1].items():
                rows.append(dict(season=season, stage=stage, slice=slice_, error=top / bottom))
        extra["multipliers"] = multipliers.reset_index().to_dict("records")
        extra["seconds"] = time.time() - started
        return pd.DataFrame(rows), extra

    def close(self) -> None:
        for link, process in self.links.values():
            try:
                link.send(("stop",))
            except (BrokenPipeError, OSError):
                pass
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()


def score_of(table: pd.DataFrame, stage: str = "step4", slice_: str = "all") -> pd.Series:
    """The decider's error per season for one stage and slice."""
    return table[(table.stage == stage) & (table.slice == slice_)].set_index("season").error.sort_index()


def paired(mine: pd.Series, theirs: pd.Series) -> dict:
    diff = (mine - theirs.reindex(mine.index)).dropna()
    se = float(diff.std(ddof=1) / np.sqrt(len(diff))) if len(diff) > 1 else float("nan")
    return dict(mean_change=float(diff.mean()), se=se, z=float(diff.mean() / se) if se and se > 0 else float("nan"),
                seasons_better=int((diff < 0).sum()), seasons=int(len(diff)))


def params_file(name: str) -> dict:
    from openrapm.data.config import booster_params, load_config
    return booster_params(load_config(), f"params/booster_params_{name}.json")


def reference_guard(pool: Pool, tag: str, guard: Guard) -> pd.DataFrame:
    """The incumbent's next-season rows on the tune seasons, computed once and kept in reference_guard.json."""
    path = tune_dir(tag) / "reference_guard.json"
    if path.exists():
        return pd.DataFrame(json.loads(path.read_text(encoding="utf-8"))["rows"])
    _, extra = pool.trial(params_file(REFERENCE[0]), REFERENCE[1], guard=guard)
    path.write_text(json.dumps(dict(params=REFERENCE[0], features=REFERENCE[1], rows=extra["guard"].to_dict("records")),
                               indent=1, default=float), encoding="utf-8")
    return extra["guard"]


def describe_guard(rows: pd.DataFrame, reference: pd.DataFrame) -> str:
    team, stint = guard_paired(rows, reference, "tg"), guard_paired(rows, reference, "mse")
    return (f"next-season team-game {team['mean_change']:+.4f} z {team['z']:+.2f} {team['seasons_better']}/"
            f"{team['seasons']}; stint {stint['mean_change']:+.4f} z {stint['z']:+.2f}")


def reference_scores(pool: Pool, tag: str) -> pd.DataFrame:
    """The incumbent's trial, computed once and kept in reference.json."""
    path = tune_dir(tag) / "reference.json"
    if path.exists():
        return pd.DataFrame(json.loads(path.read_text(encoding="utf-8"))["rows"])
    table, extra = pool.trial(params_file(REFERENCE[0]), REFERENCE[1])
    path.write_text(json.dumps(dict(params=REFERENCE[0], features=REFERENCE[1], rows=table.to_dict("records"),
                                    extra=extra), indent=1, default=float), encoding="utf-8")
    print(f"reference ({REFERENCE[0]}, {REFERENCE[1]}): score {score_of(table).mean():.4f} "
          f"({extra['seconds']:.0f}s)", flush=True)
    return table


def describe(table: pd.DataFrame, reference: pd.DataFrame) -> str:
    parts = []
    for stage in STAGES:
        for slice_ in SLICES:
            result = paired(score_of(table, stage, slice_), score_of(reference, stage, slice_))
            parts.append(f"{stage}/{slice_} {result['mean_change']:+.4f} z {result['z']:+.2f} "
                         f"{result['seasons_better']}/{result['seasons']}")
    return "; ".join(parts)


def run_trial(args) -> int:
    pool = Pool(args.tag, TUNE_SEASONS)
    try:
        reference = reference_scores(pool, args.tag)
        guard = Guard(args.tag, TUNE_SEASONS) if args.guard else None
        guard_reference = reference_guard(pool, args.tag, guard) if guard else None
        changes = parse_changes(args.drop, "drop") + parse_changes(args.shadow, "shadow")
        table, extra = pool.trial(params_file(args.params), args.features, compare=bool(args.compare),
                                  def_factor=args.def_factor, guard=guard, changes=changes, seed=args.seed,
                                  ensemble=parse_ensemble(args.ensemble))
        if args.save:
            out_dir = tune_dir(args.tag)
            key = dict(trial=args.save)
            table.assign(**key).to_csv(out_dir / "trials_saved.csv", mode="a",
                                       header=not (out_dir / "trials_saved.csv").exists(), index=False)
            if guard:
                extra["guard"].assign(**key).to_csv(out_dir / "trials_saved_guard.csv", mode="a",
                                                    header=not (out_dir / "trials_saved_guard.csv").exists(),
                                                    index=False)
        if guard:
            print("next-season check against the reference: " + describe_guard(extra["guard"], guard_reference))
        print(f"{args.params}, {args.features}: score {score_of(table).mean():.4f}  ({extra['seconds']:.0f}s, "
              f"slowest season's boosters and folds {extra['rate_seconds_max']:.0f}s)")
        print("against the reference: " + describe(table, reference))
        if args.compare:
            worst = max(max(value.values()) for value in extra["compare"].values())
            print(f"largest rating difference against the cache's check folds: {worst:.2e}")
            print({season: {k: f"{v:.1e}" for k, v in value.items()} for season, value in extra["compare"].items()})
        if args.against:
            from openrapm.data.config import load_config, resolve
            seasons = pd.read_csv(resolve(load_config(), "outputs") / "csv" / f"wowy_{args.against}_seasons.csv")
            for system in seasons.system.unique():
                theirs = seasons[seasons.system == system]
                step3 = paired(score_of(table, "step3"), score_of(theirs, "step3"))
                print(f"step3/all against check 151's {system} ({args.against}): largest season difference "
                      f"{(score_of(table, 'step3') - score_of(theirs, 'step3').reindex(TUNE_SEASONS)).abs().max():.2e}"
                      f", mean {step3['mean_change']:+.2e}")
    finally:
        pool.close()
    return 0


# ------------------------------------------------------------------------------------------------- the input cull
def cull_stem(args) -> str:
    return f"cull_{args.params}_{args.features}"


def cull_plan(args) -> list:
    """[(arm, side, input, seed)]: the shipped settings at every base seed first, then every input's arms one seed at a
    time, so a run stopped part way has whole seeds."""
    from openrapm.features import FEATURE_SETS as SETS
    arms = [arm for arm in args.arms.split(",") if arm]
    plan = [("base", "-", "-", seed) for seed in range(args.base_seeds)]
    for seed in range(args.seeds):
        for side in ("O", "D"):
            for name in SETS[args.features][side]:
                plan.extend((arm, side, name, seed) for arm in arms)
    return plan


def run_cull(args) -> int:
    """Every input's arms at every seed, appended to cull_<params>_<features>.csv and its _guard.csv; resumes."""
    out_dir = tune_dir(args.tag)
    path, guard_path = out_dir / f"{cull_stem(args)}.csv", out_dir / f"{cull_stem(args)}_guard.csv"
    done = set()
    if path.exists():
        old = pd.read_csv(path, keep_default_na=False)
        done = set(zip(old.arm, old.side, old.input, old.seed.astype(int)))
    todo = [key for key in cull_plan(args) if key not in done]
    print(f"{len(todo)} trials to run, {len(done)} done", flush=True)
    if not todo:
        return 0
    params = params_file(args.params)
    base, base_guard = {}, {}
    if path.exists():
        old = pd.read_csv(path, keep_default_na=False)
        old_guard = pd.read_csv(guard_path, keep_default_na=False)
        for seed in old[old.arm == "base"].seed.unique():
            base[int(seed)] = score_of(old[(old.arm == "base") & (old.seed == seed)])
            base_guard[int(seed)] = old_guard[(old_guard.arm == "base") & (old_guard.seed == seed)]
    pool = Pool(args.tag, TUNE_SEASONS)
    try:
        guard = Guard(args.tag, TUNE_SEASONS)
        for number, (arm, side, name, seed) in enumerate(todo, start=1):
            changes = () if arm == "base" else ((side, arm, name),)
            table, extra = pool.trial(params, args.features, def_factor=args.def_factor, guard=guard,
                                      changes=changes, seed=seed)
            key = dict(arm=arm, side=side, input=name, seed=seed)
            table.assign(**key).to_csv(path, mode="a", header=not path.exists(), index=False)
            extra["guard"].assign(**key).to_csv(guard_path, mode="a", header=not guard_path.exists(), index=False)
            if arm == "base":
                base[seed], base_guard[seed] = score_of(table), extra["guard"]
            reading = f"decider {score_of(table).mean():.4f}"
            if arm != "base" or seed != 0:
                theirs = base[seed if arm != "base" else 0]
                result = paired(score_of(table), theirs)
                reading += (f", {result['mean_change']:+.4f} z {result['z']:+.2f} {result['seasons_better']}/"
                            f"{result['seasons']}; " + describe_guard(extra["guard"],
                                                                      base_guard[seed if arm != "base" else 0]))
            print(f"[{number}/{len(todo)}] {arm} {side} {name} seed {seed}: {reading}  ({extra['seconds']:.0f}s)",
                  flush=True)
    finally:
        pool.close()
    return 0


def run_cull_table(args) -> int:
    """The full table: per side and input, each arm against the shipped settings at the same seeds, the seed means
    paired by season (the decider) and by rated-and-scored season (the next-season check); the noise control rows
    on top."""
    from openrapm.data.config import load_config, resolve
    from openrapm.features import meaning
    out_dir = tune_dir(args.tag)
    rows = pd.read_csv(out_dir / f"{cull_stem(args)}.csv", keep_default_na=False)
    guard = pd.read_csv(out_dir / f"{cull_stem(args)}_guard.csv", keep_default_na=False)
    rows = rows[(rows.stage == "step4") & (rows.slice == "all")]
    base = rows[rows.arm == "base"].pivot_table(index="season", columns="seed", values="error")
    base_guard = guard[guard.arm == "base"].pivot_table(index=["rated", "scored"], columns="seed", values="tg")
    sides = {"O": "offence", "D": "defence"}
    records = []
    for seed in base.columns[1:]:
        decider, nxt = paired(base[seed], base[0]), paired(base_guard[seed], base_guard[0])
        records.append(dict(side="both", input=f"seed_{seed}", meaning=f"noise control: the shipped settings at seed "
                            f"{seed} against seed 0", drop_change=decider["mean_change"], drop_se=decider["se"],
                            drop_z=decider["z"], drop_seasons_better=decider["seasons_better"],
                            drop_guard_change=nxt["mean_change"], drop_guard_z=nxt["z"]))
    for (side, name), group in rows[rows.arm != "base"].groupby(["side", "input"], sort=False):
        record = dict(side=sides[side], input=name, meaning=meaning(name))
        for arm in CULL_ARMS:
            mine = group[group.arm == arm].pivot_table(index="season", columns="seed", values="error")
            seeds = [seed for seed in mine.columns if seed in base.columns]
            if not seeds:
                continue
            decider = paired(mine[seeds].mean(axis=1), base[seeds].mean(axis=1))
            theirs = guard[(guard.arm == arm) & (guard.side == side) & (guard.input == name)].pivot_table(
                index=["rated", "scored"], columns="seed", values="tg")
            nxt = paired(theirs[seeds].mean(axis=1), base_guard[seeds].mean(axis=1))
            record.update({f"{arm}_change": decider["mean_change"], f"{arm}_se": decider["se"],
                           f"{arm}_z": decider["z"], f"{arm}_seasons_better": decider["seasons_better"],
                           f"{arm}_seeds": len(seeds), f"{arm}_guard_change": nxt["mean_change"],
                           f"{arm}_guard_z": nxt["z"]})
            for seed in seeds:
                record[f"{arm}_change_seed_{seed}"] = float((mine[seed] - base[seed]).mean())
        if "shadow_z" in record:
            record["boruta_verdict"] = ("accepted" if record["shadow_z"] >= 2 else
                                        "rejected" if record["shadow_z"] <= -2 else "tentative")
        if "drop_z" in record:
            record["drop_verdict"] = ("better without" if record["drop_z"] <= -2 else
                                      "needed" if record["drop_z"] >= 2 else "tie")
        records.append(record)
    table = pd.DataFrame(records)
    out = resolve(load_config(), "outputs") / "csv" / f"{cull_stem(args)}_table.csv"
    table.to_csv(out, index=False)
    show = [c for c in ("side", "input", "drop_change", "drop_z", "drop_seasons_better", "drop_guard_z",
                        "shadow_change", "shadow_z", "shadow_guard_z", "boruta_verdict", "drop_verdict")
            if c in table.columns]
    with pd.option_context("display.width", 250, "display.max_rows", 200, "display.float_format", "{:+.3f}".format):
        print(table[show].to_string(index=False))
    print(f"wrote {out}")
    return 0


# ------------------------------------------------------------------------------------------------- the search
LINEAR_KEYS = ("linear_lambda",)


def suggest(trial) -> tuple[dict, str]:
    """One trial's settings, both sides, and its feature set."""
    linear = trial.suggest_categorical("linear_tree", [True, False])
    era = trial.suggest_categorical("era_marker", [False, True])
    params = {}
    for side in ("O", "D"):
        p = dict(
            learning_rate=trial.suggest_float(f"{side}_learning_rate", 0.005, 0.15, log=True),
            n_estimators=trial.suggest_int(f"{side}_n_estimators", 50, 2000, log=True),
            num_leaves=trial.suggest_int(f"{side}_num_leaves", 4, 96, log=True),
            max_depth=trial.suggest_int(f"{side}_max_depth", 2, 12),
            min_child_samples=trial.suggest_int(f"{side}_min_child_samples", 20, 2000, log=True),
            reg_lambda=trial.suggest_float(f"{side}_reg_lambda", 1e-3, 1000.0, log=True),
            reg_alpha=trial.suggest_float(f"{side}_reg_alpha", 1e-4, 100.0, log=True),
            max_bin=trial.suggest_categorical(f"{side}_max_bin", [15, 31, 63, 127, 255]),
            path_smooth=trial.suggest_float(f"{side}_path_smooth", 1e-3, 30.0, log=True),
            colsample_bytree=trial.suggest_float(f"{side}_colsample_bytree", 0.3, 1.0),
            linear_tree=linear)
        if linear:
            p["linear_lambda"] = trial.suggest_float(f"{side}_linear_lambda", 1e-2, 1000.0, log=True)
        params[side] = p
    return params, ("hustle_era_marker" if era else "hustle")


def enqueue_from(study, name: str, era: bool = False) -> None:
    """A params/ file as a trial to run first (the hustle inputs on)."""
    spec = params_file(name)
    values = {"linear_tree": bool(spec["O"].get("linear_tree", False)), "era_marker": era}
    for side in ("O", "D"):
        for key, value in spec[side].items():
            if key == "linear_tree" or (key in LINEAR_KEYS and not values["linear_tree"]):
                continue
            values[f"{side}_{key}"] = value
    study.enqueue_trial(values, skip_if_exists=True)


def run_tune(args) -> int:
    import optuna
    out_dir = tune_dir(args.tag)
    storage = f"sqlite:///{(out_dir / 'study.db').as_posix()}"
    sampler = optuna.samplers.TPESampler(seed=args.seed, multivariate=True, group=True,
                                         n_startup_trials=args.startup, constant_liar=False)
    study = optuna.create_study(study_name=args.tag, storage=storage, direction="minimize", sampler=sampler,
                                load_if_exists=True)
    if len(study.trials) == 0:
        enqueue_from(study, "lgb1_cols06")
        enqueue_from(study, "lgb1_cols06_flat")
    pool = Pool(args.tag, TUNE_SEASONS)
    trials_csv = out_dir / "trials.csv"
    try:
        reference = reference_scores(pool, args.tag)
        reference_score = float(score_of(reference).mean())

        def objective(trial):
            params, feature_set = suggest(trial)
            table, extra = pool.trial(params, feature_set)
            score = float(score_of(table).mean())
            result = paired(score_of(table), score_of(reference))
            trial.set_user_attr("params", params)
            trial.set_user_attr("features", feature_set)
            trial.set_user_attr("vs_reference", result)
            trial.set_user_attr("seconds", extra["seconds"])
            trial.set_user_attr("multipliers", extra["multipliers"])
            table.assign(trial=trial.number, features=feature_set).to_csv(
                trials_csv, mode="a", header=not trials_csv.exists(), index=False)
            print(f"trial {trial.number}: {score - reference_score:+.4f} against the reference, z {result['z']:+.2f}, "
                  f"{result['seasons_better']}/{result['seasons']} seasons; {describe(table, reference)}; "
                  f"{feature_set}, linear {params['O']['linear_tree']}  ({extra['seconds']:.0f}s)", flush=True)
            return score

        remaining = args.trials - len([t for t in study.trials if t.state.is_finished()])
        if remaining > 0:
            study.optimize(objective, n_trials=remaining, gc_after_trial=True)
    finally:
        pool.close()
    write_best(study, out_dir)
    return 0


# The second search (the owner, 2026-10-16: "the best most modern HPO on distributions"): Optuna's Gaussian-process
# sampler, every setting drawn from a distribution over one fixed space, the next-season check a hard limit.
GUARD_Z_LIMIT = 1.0      # the next-season check on the 20 neighbour seasons; about z +1.7 on the full test's 56
DEF_FACTOR = (0.7, 1.2)


def suggest_fixed(trial) -> tuple[dict, str, float]:
    """One trial's settings on a fixed space (the Gaussian process wants one): the straight-line penalty is drawn
    every trial and read only with straight-line leaves; the defence factor multiplies step 4's two defensive
    factors."""
    linear = trial.suggest_categorical("linear_tree", [True, False])
    era = trial.suggest_categorical("era_marker", [False, True])
    params = {}
    for side in ("O", "D"):
        p = dict(
            learning_rate=trial.suggest_float(f"{side}_learning_rate", 0.003, 0.15, log=True),
            n_estimators=trial.suggest_int(f"{side}_n_estimators", 30, 2500, log=True),
            num_leaves=trial.suggest_int(f"{side}_num_leaves", 3, 96, log=True),
            max_depth=trial.suggest_int(f"{side}_max_depth", 1, 12),
            min_child_samples=trial.suggest_int(f"{side}_min_child_samples", 10, 2500, log=True),
            reg_lambda=trial.suggest_float(f"{side}_reg_lambda", 1e-3, 1000.0, log=True),
            reg_alpha=trial.suggest_float(f"{side}_reg_alpha", 1e-4, 100.0, log=True),
            max_bin=trial.suggest_categorical(f"{side}_max_bin", [15, 31, 63, 127, 255]),
            path_smooth=trial.suggest_float(f"{side}_path_smooth", 1e-3, 30.0, log=True),
            colsample_bytree=trial.suggest_float(f"{side}_colsample_bytree", 0.3, 1.0),
            linear_tree=linear)
        line_penalty = trial.suggest_float(f"{side}_linear_lambda", 1e-3, 1000.0, log=True)
        if linear:
            p["linear_lambda"] = line_penalty
        params[side] = p
    def_factor = trial.suggest_float("def_factor", *DEF_FACTOR)
    return params, ("hustle_era_marker" if era else "hustle"), def_factor


def enqueue_fixed(study, name: str, def_factor: float = 1.0) -> None:
    spec = params_file(name)
    values = {"linear_tree": bool(spec["O"].get("linear_tree", False)), "era_marker": False, "def_factor": def_factor}
    for side in ("O", "D"):
        for key, value in spec[side].items():
            if key != "linear_tree":
                values[f"{side}_{key}"] = value
    study.enqueue_trial(values, skip_if_exists=True)


def run_tune_guarded(args) -> int:
    import optuna
    out_dir = tune_dir(args.tag)
    study_name = args.study
    storage = f"sqlite:///{(out_dir / f'{study_name}.db').as_posix()}"
    sampler = optuna.samplers.GPSampler(seed=args.seed, n_startup_trials=args.startup, deterministic_objective=True,
                                        constraints_func=lambda frozen: frozen.user_attrs.get("constraint", [1.0]))
    study = optuna.create_study(study_name=study_name, storage=storage, direction="minimize", sampler=sampler,
                                load_if_exists=True)
    if len(study.trials) == 0:
        enqueue_fixed(study, "lgb1_tune1")
        enqueue_fixed(study, "lgb1_tune1", 0.893)
        enqueue_fixed(study, "lgb1_tune1_offonly")
        enqueue_fixed(study, "lgb1_cols06")
        enqueue_fixed(study, "lgb1_cols06_flat")
    pool = Pool(args.tag, TUNE_SEASONS)
    trials_csv = out_dir / f"{study_name}_trials.csv"
    try:
        guard = Guard(args.tag, TUNE_SEASONS)
        reference = reference_scores(pool, args.tag)
        guard_reference = reference_guard(pool, args.tag, guard)
        reference_score = float(score_of(reference).mean())

        def objective(trial):
            params, feature_set, def_factor = suggest_fixed(trial)
            table, extra = pool.trial(params, feature_set, def_factor=def_factor, guard=guard)
            score = float(score_of(table).mean())
            result = paired(score_of(table), score_of(reference))
            team = guard_paired(extra["guard"], guard_reference, "tg")
            stint = guard_paired(extra["guard"], guard_reference, "mse")
            trial.set_user_attr("constraint", [float(team["z"]) - GUARD_Z_LIMIT])
            for key, value in (("params", params), ("features", feature_set), ("def_factor", def_factor),
                               ("vs_reference", result), ("guard_team_game", team), ("guard_stint", stint),
                               ("seconds", extra["seconds"]), ("multipliers", extra["multipliers"])):
                trial.set_user_attr(key, value)
            table.assign(trial=trial.number, features=feature_set, def_factor=def_factor).to_csv(
                trials_csv, mode="a", header=not trials_csv.exists(), index=False)
            print(f"trial {trial.number}: decider {score - reference_score:+.4f} z {result['z']:+.2f} "
                  f"{result['seasons_better']}/{result['seasons']}; {describe_guard(extra['guard'], guard_reference)}"
                  f"{'' if team['z'] <= GUARD_Z_LIMIT else ', FAILS the limit'}; {feature_set}, linear "
                  f"{params['O']['linear_tree']}, defence factor {def_factor:.3f}  ({extra['seconds']:.0f}s)",
                  flush=True)
            return score

        remaining = args.trials - len([t for t in study.trials if t.state.is_finished()])
        if remaining > 0:
            study.optimize(objective, n_trials=remaining, gc_after_trial=True)
    finally:
        pool.close()
    feasible = [t for t in study.trials if t.state.is_finished() and t.value is not None
                and t.user_attrs.get("constraint", [1.0])[0] <= 0]
    best = min(feasible, key=lambda t: t.value)
    params = best.user_attrs["params"]
    spec = {side: {"learner": "lightgbm", "params": params[side]} for side in ("O", "D")}
    spec["def_factor"] = best.user_attrs["def_factor"]
    spec["note"] = (f"check 152's guarded search {study_name}, trial {best.number} of {len(study.trials)}; feature set "
                    f"{best.user_attrs['features']}; decider {best.user_attrs['vs_reference']}; next-season "
                    f"{best.user_attrs['guard_team_game']}")
    (out_dir / f"{study_name}_best.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
    print(f"best within the limit: trial {best.number}, {best.user_attrs['vs_reference']}; next-season "
          f"{best.user_attrs['guard_team_game']}; defence factor {best.user_attrs['def_factor']:.3f}")
    return 0


def write_best(study, out_dir: Path) -> None:
    best = study.best_trial
    params = best.user_attrs["params"]
    spec = {side: {"learner": "lightgbm", "params": params[side]} for side in ("O", "D")}
    spec["note"] = (f"check 152's tune on the decider, trial {best.number} of {len(study.trials)}; feature set "
                    f"{best.user_attrs['features']}; against the reference {best.user_attrs['vs_reference']}")
    (out_dir / "best.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
    print(f"best: trial {best.number}, score {best.value:.4f}, {best.user_attrs['features']}; "
          f"{best.user_attrs['vs_reference']}; wrote {out_dir / 'best.json'}")


def main(argv=None) -> int:
    from openrapm.commands.console import use_utf8
    use_utf8()
    parser = argparse.ArgumentParser(description="The booster tune on the decider.")
    commands = parser.add_subparsers(dest="command", required=True)
    cache = commands.add_parser("cache", help="build each tune season's cache, once")
    cache.add_argument("--tag", required=True)
    cache.add_argument("--seasons", default="", help="comma-separated; default the ten tune seasons")
    cache.add_argument("--check-models", default="flathustle",
                       help="saved boosters whose fold ratings the cache keeps, to check a trial against")
    trial = commands.add_parser("trial", help="one trial from a params/ file")
    whole = commands.add_parser("cache-whole", help="each tune season's whole-season and next-season pieces, once")
    whole.add_argument("--tag", required=True)
    whole.add_argument("--seasons", default="")
    whole.add_argument("--check-models", default="flathustle")
    trial.add_argument("--tag", required=True)
    trial.add_argument("--params", required=True, help="params/booster_params_<name>.json")
    trial.add_argument("--features", required=True)
    trial.add_argument("--compare", action="store_true", help="the ratings against the cache's check folds")
    trial.add_argument("--guard", action="store_true", help="also the next-season check (needs cache-whole)")
    trial.add_argument("--def-factor", type=float, default=1.0, help="multiplies step 4's defensive factors")
    trial.add_argument("--against", default="", help="a check-151 tag whose step3 errors the trial must equal")
    trial.add_argument("--drop", default="", help="inputs taken off, as O:weight_lb,D:height_in")
    trial.add_argument("--shadow", default="", help="inputs shuffled among the rows that have one, as --drop")
    trial.add_argument("--seed", type=int, default=0, help="the boosters' random_state (default %(default)s)")
    trial.add_argument("--ensemble", default="one:1", help="seeds:K or bags:K, as ENSEMBLES says (default one:1)")
    trial.add_argument("--save", default="", help="append the trial's rows to trials_saved.csv under this name")
    for name, text in (("cull", "every input dropped and shadowed in turn, at several seeds; resumes"),
                       ("cull-table", "the cull's full table")):
        cull = commands.add_parser(name, help=text)
        cull.add_argument("--tag", required=True)
        cull.add_argument("--params", default="lgb1_tune2")
        cull.add_argument("--features", default="hustle")
        cull.add_argument("--def-factor", type=float, default=0.866)
        cull.add_argument("--seeds", type=int, default=3, help="seeds per input and arm (default %(default)s)")
        cull.add_argument("--base-seeds", type=int, default=5, help="seeds of the shipped settings (default "
                                                                    "%(default)s); those past --seeds are the noise "
                                                                    "control")
        cull.add_argument("--arms", default=",".join(CULL_ARMS))
    tune = commands.add_parser("tune", help="the Optuna search")
    tune.add_argument("--tag", required=True)
    tune.add_argument("--trials", type=int, default=60, help="finished trials to reach (default %(default)s)")
    tune.add_argument("--startup", type=int, default=12, help="random trials before TPE (default %(default)s)")
    tune.add_argument("--seed", type=int, default=0)
    tune.add_argument("--guard", action="store_true",
                      help="the second search: the Gaussian-process sampler with the next-season check as a limit")
    tune.add_argument("--study", default="guarded", help="the guarded search's name (default %(default)s)")
    args = parser.parse_args(argv)
    if args.command == "cache":
        seasons = [int(s) for s in args.seasons.split(",") if s] or list(TUNE_SEASONS)
        return build_cache(args.tag, seasons, args.check_models)
    if args.command == "cache-whole":
        seasons = [int(s) for s in args.seasons.split(",") if s] or list(TUNE_SEASONS)
        return build_whole_cache(args.tag, seasons, args.check_models)
    if args.command == "trial":
        return run_trial(args)
    if args.command == "cull":
        return run_cull(args)
    if args.command == "cull-table":
        return run_cull_table(args)
    return run_tune_guarded(args) if args.guard else run_tune(args)


if __name__ == "__main__":
    raise SystemExit(main())
