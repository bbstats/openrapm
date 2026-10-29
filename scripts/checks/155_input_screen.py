"""Check 155, the input screen on the within-season test (experiment 112; the owner, 2026-10-25: "Probably time to do
boruta instead of ditching ... we need to be robust in how we add in new features").

Boruta keeps an input when it beats a shuffled copy of itself, its shadow.  Here the yardstick is the decider, check
151's with-or-without-you error on the held-back games after step 4, as in check 152's input cull (experiment 77).
Check 152 rates the pipeline of 2026-10-16; this check rates today's through the code `openrapm rate` and `openrapm
folds` run, so a trial on the shipped lists gives back creation110's folds.

Words used below
    tune season   one of check 152's ten, 1998, 2001, ..., 2025; the other twenty are left for the build that confirms
    cache         built once per tune season: the boosters' training rows with every candidate (feature set
                  `features.FEATURE_SETS["screen112_pool"]`, step 1's labels as `openrapm rate` reads them, the
                  part-season rows), each of the twelve folds rebuilt from its fit games and the whole season rebuilt
                  the same way (commands/within_season.SeasonRun.world), and the points designs of the seasons either
                  side for the guard
    trial         one arm on one input at one seed: both boosters refitted for every tune season (10 bags), every fold
                  rated through steps 2-3 with the second prior (`rate_world`) and written as `openrapm folds` writes
                  it, step 4 with both layers fitted on the tune seasons' folds (test rule, 8 nearest), and check
                  151's error stint by stint
    arm           base: the shipped lists; add: the candidate appended to its side's list; add_shadow: the same with
                  its values shuffled among the rows that have one, in training and in rating; shadow: a shipped
                  input shuffled in place; pool: every candidate of the tag appended, the lists the tune fits
    screen seasons the seasons a tag's screen reads (`SCREENS`): the ten tune seasons, or for an input that has values
                  in a few seasons only, those seasons (experiment 120's matchup inputs, 2018-2025)
    seed          bag k of a trial at seed s is drawn and fitted at k + 100 s, so seed 0 is the shipped build; a
                  trial at another seed, nothing else changed, is the noise control
    variant       the box-score prior at other settings (`VARIANTS`): the base arm alone at every base seed, read
                  against the shipped prior's base arm by `variants` (2026-10-26: is a prior that averages over more
                  ways of splitting its players both steadier from draw to draw and better on average?)
    guard         each tune season's whole-season ratings after the trial's step 4 predicting the seasons either side,
                  team-game by team-game (check 152's guard; step 5 is off)
    tune          the booster settings searched again on today's pipeline with the tag's pool in the lists
                  (experiment 120; the owner, 2026-10-29: "hyperparameter tuning + boruta should probably always
                  happen in tandem (or - if it shouldn't we should prove it)"): experiment 76's guarded search, its
                  space less the era marker and the defence factor, on the ten tune seasons at seed 0
    tandem        the screen at two booster settings side by side: each candidate's reading against its shuffled
                  copy at both, and how far it moved, paired by season

The rule (DECISIONS.md, experiment 112, registered before the run): a candidate passes when, over the tune seasons,
seeds averaged within each season and paired by season, add beats add_shadow at z -2 or below, add beats the shipped
lists on the mean, and the guard is not worse at z +2.

    python scripts/checks/155_input_screen.py cache --tag screen112 [--seasons ...]  (once, resumes)
    python scripts/checks/155_input_screen.py dry --tag screen112 --seasons 2025  (against creation110)
    python scripts/checks/155_input_screen.py run --tag screen112                (resumes)
    python scripts/checks/155_input_screen.py table --tag screen112
    python scripts/checks/155_input_screen.py run --tag screen112 --variant out_of_bag60   (the base arm alone)
    python scripts/checks/155_input_screen.py variants --tag screen112
    python scripts/checks/155_input_screen.py tune --tag screen120 --study tune120         (resumes)
    python scripts/checks/155_input_screen.py run --tag screen120 --variant tune120 --all-arms
    python scripts/checks/155_input_screen.py tandem --tag screen120 --variant tune120

Each tag screens its own pool (`SCREENS`): screen112 experiment 112's candidates, screen119 experiment 119's (the owner,
2026-10-28: goaltends, fouls, blocked shots and counts over expected, "go for it and remember to boruta"), screen120
experiment 120's matchup inputs on 2018-2025.

Writes under outputs/screen/<tag>/: season_<season>.pkl (the cache), trials.csv (one row per trial, season, stage and
slice), guard.csv (one row per trial and rated-scored pair), and outputs/csv/screen_<tag>_table.csv (the full table);
a variant's trials_<variant>.csv and guard_<variant>.csv, and outputs/csv/screen_<tag>_variants.csv.
"""
from __future__ import annotations

import os

THREADS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
for _name in THREADS:
    os.environ.setdefault(_name, "1")

import argparse  # noqa: E402
import importlib.util  # noqa: E402
import multiprocessing as mp  # noqa: E402
import pickle  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import zlib  # noqa: E402
from dataclasses import dataclass, replace  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

TUNE_SEASONS = tuple(range(1998, 2026, 3))          # 1998, 2001, ..., 2025, check 152's
# the matchup inputs have values from 2018 on; 2026's next season is the one in progress, so it is left out
MATCHUP_SEASONS = tuple(range(2018, 2026))
# {tag: (the feature set holding the shipped lists and every candidate, the features.py name of the candidates, the
# seasons the screen reads)}
SCREENS = {"screen112": ("screen112_pool", "SCREEN_CANDIDATES", TUNE_SEASONS),
           "screen119": ("screen119_pool", "SCREEN_CANDIDATES_119", TUNE_SEASONS),
           "screen120": ("screen120_pool", "SCREEN_CANDIDATES_MATCHUPS", MATCHUP_SEASONS)}
REFERENCE = "creation110"                           # the shipped build whose saved boosters name the second prior
SEED_STEP = 100                                     # box_prior.BAG_DRAW_STEP
NEAREST = 8
# the base arm at another setting of the box-score prior (HANDOFF, "START HERE" of 2026-10-26: the prior made robust
# to how its players are split into the five training groups), each a dict of settings.PriorSettings fields
VARIANTS = {"shipped": {}, "out_of_bag60": dict(prediction="out_of_bag", bags=60), "bags30": dict(bags=30),
            "split_per_bag": dict(split_per_bag=True),
            # experiment 118 (2026-10-28): the pretrained Kumo network in place of LightGBM, no bags; a random draw
            # picks each member's training rows (in_context_prior.py)
            "kumo_small": dict(booster_file="params/booster_params_kumo_small.json", bags=0),
            "kumo_medium": dict(booster_file="params/booster_params_kumo_medium.json", bags=0),
            # its second look: every training row the same chance to be shown, whatever its weight
            "kumo_small_equal": dict(booster_file="params/booster_params_kumo_small_equal.json", bags=0),
            # experiment 120: the booster settings the tune found with the matchup inputs in the lists
            "tune120": dict(booster_file="params/booster_params_tune120.json")}
ERAS = {"1997-2006": (1998, 2001, 2004), "2007-2016": (2007, 2010, 2013, 2016), "2017-2026": (2019, 2022, 2025)}
ERA_SPANS = {"1997-2006": (1997, 2006), "2007-2016": (2007, 2016), "2017-2026": (2017, 2026)}


def _load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(file))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def pool(tag: str) -> str:
    """The feature set a tag screens: the shipped lists with every candidate appended."""
    if tag not in SCREENS:
        raise SystemExit(f"--tag is one of {sorted(SCREENS)}, not {tag!r}; a new screen adds its pool to SCREENS")
    return SCREENS[tag][0]


def candidates(tag: str) -> dict:
    """{side: the inputs a tag screens, each appended alone to its side's shipped list}."""
    from openrapm import features
    pool(tag)
    return getattr(features, SCREENS[tag][1])


def screen_seasons(tag: str) -> tuple:
    """The seasons a tag's screen reads."""
    pool(tag)
    return SCREENS[tag][2]


def screen_dir(tag: str) -> Path:
    from openrapm.data.config import load_config, resolve
    return resolve(load_config(), "outputs") / "screen" / tag


def within_dir() -> Path:
    from openrapm.data.config import load_config, resolve
    return resolve(load_config(), "within")


def shipped_lists() -> dict:
    from openrapm.features import FEATURE_SETS
    from openrapm.settings import TEST
    return {side: list(FEATURE_SETS[TEST.prior.feature_set][side]) for side in ("O", "D")}


# ------------------------------------------------------------------------------------------------- the cache
def build_cache(tag: str, seasons) -> int:
    """Each tune season's cache file, after checking the shipped boosters refitted on its training rows give back the
    saved ones' whole-season priors."""
    from openrapm import box_prior, rapm
    from openrapm.commands import within_season as ws
    from openrapm.commands.build import accumulate_rapm, part_season_panel, row_label_tables, unseen_seasons
    from openrapm.data.careers import careers_before
    from openrapm.data.config import load_config, resolve
    from openrapm.data.designs import DesignCache
    from openrapm.data.panel import SeasonSources, SeasonTables, share_cap, trainable_panel
    from openrapm.data.roles import player_season_inputs, roles_path
    from openrapm.data.seasons import trainable
    from openrapm.features import FEATURE_SETS, PANEL_INPUTS, ROW_FEATURES
    from openrapm.settings import TEST

    guard = _load("decider_tune", "152_decider_tune.py")
    cfg = load_config()
    out_dir = screen_dir(tag)
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [season for season in seasons if not (out_dir / f"season_{season}.pkl").exists()]
    if not todo:
        print("every season cached", flush=True)
        return 0
    started = time.time()
    settings = replace(TEST, prior=replace(TEST.prior, feature_set=pool(tag)))
    designs = DesignCache(cfg)
    sums = accumulate_rapm(designs, trainable(cfg).seasons, settings)
    panel = trainable_panel(cfg)
    panel = panel[panel.possessions > 0].reset_index(drop=True)
    part_seasons = part_season_panel(cfg, settings)
    roles = pd.read_parquet(roles_path(cfg))
    inputs = player_season_inputs(roles, cap=share_cap(cfg))
    debut = careers_before(cfg, roles)
    saved_all = ws.load_models(resolve(cfg, "outputs"), REFERENCE, todo)
    meta = saved_all["meta"]
    fold_settings = ws.fold_settings(meta)
    lines = ws.fold_lines(cfg, meta, False)
    params = box_prior_params(cfg)
    shipped = shipped_lists()
    print(f"step 1's sums and the panels read ({time.time() - started:.0f}s)", flush=True)
    for season in todo:
        unseen = unseen_seasons(season, settings)
        labels = rapm.career_labels(sums, unseen, settings.labels, settings.targets)
        outside = row_label_tables(sums, panel, unseen, settings, None)
        train = {}
        for side, column in (("O", "offense"), ("D", "defense")):
            names = list(FEATURE_SETS[pool(tag)][side])
            side_outside = None if outside is None else {key: tables[side] for key, tables in outside.items()}
            train[side] = box_prior.training_rows(
                labels[side], box_prior.training_seasons(panel, side, unseen, season, settings), column, names,
                settings.prior.chunk_sizes, PANEL_INPUTS, side_outside,
                box_prior.training_seasons(part_seasons, side, unseen, season, settings))
        saved = ws.season_models(saved_all, season)
        # the shipped boosters refitted here must give back the saved ones' whole-season priors
        worst = 0.0
        rows_of = panel[panel.season == season]
        for side in ("O", "D"):
            model = fit_bags(train[side], shipped[side] + ROW_FEATURES, params[side], settings, 0)
            rated = box_prior.rated_rows(rows_of[rows_of.side == side], shipped[side])
            mine = pd.Series(box_prior.predict(model, rated), index=rated.player_id.to_numpy())
            theirs = pd.Series(saved[side]["prior"]).reindex(mine.index)
            worst = max(worst, float(np.nanmax(np.abs(mine.to_numpy() - theirs.to_numpy()))))
        if worst > 1e-9:
            raise SystemExit(f"{season}: the shipped boosters refitted here miss {REFERENCE}'s saved priors by "
                             f"{worst:.1e}")
        full = {target: designs.season(season, target) for target in ws.world_targets(fold_settings)}
        run = ws.SeasonRun(season, cfg, full, SeasonSources(SeasonTables(season, cfg), roles, inputs, debut), saved,
                           out_dir, settings=fold_settings, part_season_rows=ws.trained_on_part_seasons(meta))
        ws.attach_line(run, lines, False)
        folds = ws.season_folds(run)
        worlds = {}
        for repeat in range(run.repeats):
            column = folds[f"r{repeat}"]
            for fold in range(run.n_folds):
                fit_ids = set(folds.index[(column != fold) & (column != ws.OUTSIDE)].astype(str))
                world = run.world(fit_ids)
                failed = ws.failed_checks(world.checks)
                if failed:
                    raise SystemExit(f"{season}_r{repeat}_f{fold}: leak checks failed: {failed}")
                worlds[frozenset(fit_ids)] = world
        worlds[frozenset(run.game_ids)] = run.world(set(run.game_ids))
        scored = {other: guard.guard_design(designs.season(other, "pts")) for other in (season - 1, season + 1)}
        # the training rows apart too, for the workers that only fit boosters
        with open(out_dir / f"train_{season}.pkl", "wb") as handle:
            pickle.dump(dict(season=season, unseen=list(unseen), train=train), handle, protocol=5)
        with open(out_dir / f"season_{season}.pkl", "wb") as handle:
            pickle.dump(dict(season=season, unseen=list(unseen), train=train, worlds=worlds, scored=scored,
                             line=run.line), handle, protocol=5)
        designs.clear()
        size = (out_dir / f"season_{season}.pkl").stat().st_size / 1e6
        print(f"{season}: cached, {size:.0f} MB; the shipped boosters refitted give back the saved priors to "
              f"{worst:.1e} ({time.time() - started:.0f}s)", flush=True)
    return 0


def box_prior_params(cfg: dict, booster_file: str | None = None) -> dict:
    """{"O", "D"}: the booster settings of `booster_file`, the shipped file by default."""
    from openrapm.commands.build import booster_params
    from openrapm.settings import TEST
    return booster_params(cfg, booster_file or TEST.prior.booster_file)


def fit_bags(train: pd.DataFrame, inputs: list, params: dict, settings, seed: int):
    """box_prior.fit_prior with bag k drawn and fitted at k + SEED_STEP x seed (its bag_draw): at seed 0 and the
    shipped settings, the shipped fit."""
    from openrapm import box_prior
    prior = settings.prior
    return box_prior.fit_prior(train, inputs, params, prior.n_folds, prior.fold_assignment, None, prior.bags,
                               prior.prediction, prior.split_per_bag, bag_draw=int(seed))


def variant_settings(variant: str, tag: str):
    """The test setting on the tag's pool with a variant's prior settings (`VARIANTS`)."""
    from openrapm.settings import TEST
    if variant not in VARIANTS:
        raise SystemExit(f"--variant is one of {sorted(VARIANTS)}, not {variant!r}")
    return replace(TEST, prior=replace(TEST.prior, feature_set=pool(tag), **VARIANTS[variant]))


# ------------------------------------------------------------------------------------------------- a trial
@dataclass(frozen=True)
class Trial:
    arm: str                     # base, add, add_shadow, shadow, pool
    side: str                    # "O", "D", or "-" for base and pool
    name: str                    # the input, or "-" for base and pool; a pool trial's tag
    seed: int
    variant: str = "shipped"     # the box-score prior's settings, a key of VARIANTS

    @property
    def key(self) -> str:
        variant = "" if self.variant == "shipped" else f"__{self.variant}"
        return f"{self.arm}__{self.side}__{self.name}__s{self.seed}{variant}"


def trial_lists(trial: Trial) -> dict:
    """{side: the boosters' inputs} of a trial."""
    lists = shipped_lists()
    if trial.arm in ("add", "add_shadow"):
        lists[trial.side] = lists[trial.side] + [trial.name]
    if trial.arm == "pool":
        lists = {side: lists[side] + list(candidates(trial.name)[side]) for side in ("O", "D")}
    return lists


def shuffled(values: np.ndarray, seed: int, name: str, stream: str) -> np.ndarray:
    """`values` with the non-missing ones permuted, the draw seeded by the seed, the input and the stream."""
    out = np.array(values, dtype=float)
    has = np.flatnonzero(~np.isnan(out))
    rng = np.random.default_rng([int(seed), zlib.crc32(f"{name}|{stream}".encode())])
    out[has] = out[has][rng.permutation(has.size)]
    return out


def saved_side(model, names: list, unseen: list) -> dict:
    """One side's boosters in the layout `openrapm folds` reads (within_season.saved_booster, prior_rows, and the
    fold check that they never trained on the season or a neighbour)."""
    from openrapm.commands.build import booster_layout
    from openrapm.features import ROW_FEATURES
    return dict(**booster_layout(model), feats=list(names), model_feats=list(names) + ROW_FEATURES,
                unseen=list(unseen))


class Season:
    """One tune season in a worker: its SeasonRun, rated again for every trial, and its cache, read from disk for each
    trial and let go after it (a season's cache holds about 2.5 GB; a worker holds one at a time)."""

    def __init__(self, tag: str, season: int):
        from openrapm.commands import within_season as ws
        from openrapm.data.careers import careers_before
        from openrapm.data.config import load_config, resolve
        from openrapm.data.designs import DesignCache
        from openrapm.data.panel import SeasonSources, SeasonTables, share_cap
        from openrapm.data.roles import player_season_inputs, roles_path
        from openrapm.settings import TEST
        self.tag, self.season, self.cfg = tag, season, load_config()
        self.train = None
        self.settings = replace(TEST, prior=replace(TEST.prior, feature_set=pool(tag)))
        self.params = box_prior_params(self.cfg)
        saved_all = ws.load_models(resolve(self.cfg, "outputs"), REFERENCE, [season])
        meta = saved_all["meta"]
        fold_settings = ws.fold_settings(meta)
        designs = DesignCache(self.cfg)
        full = {target: designs.season(season, target) for target in ws.world_targets(fold_settings)}
        roles = pd.read_parquet(roles_path(self.cfg))
        sources = SeasonSources(SeasonTables(season, self.cfg), roles, player_season_inputs(roles, cap=share_cap(
            self.cfg)), careers_before(self.cfg, roles))
        self.run = cached_run_class()(season, self.cfg, full, sources, ws.season_models(saved_all, season),
                                      screen_dir(tag), settings=fold_settings,
                                      part_season_rows=ws.trained_on_part_seasons(meta))
        ws.attach_line(self.run, ws.fold_lines(self.cfg, meta, False), False)

    def rate(self, trial: Trial, out_dir: Path, models: dict) -> pd.DataFrame:
        """Rate and write every fold into `out_dir` with the trial's boosters `models` ({side: BaggedPrior}, from
        `fit_task`); return the whole season's step-3 table (raw sign)."""
        from openrapm.commands import within_season as ws
        with open(screen_dir(self.tag) / f"season_{self.season}.pkl", "rb") as handle:
            cache = pickle.load(handle)
        self.run.worlds, unseen = cache["worlds"], cache["unseen"]
        del cache
        lists = trial_lists(trial)
        saved = {side: saved_side(models[side], lists[side], unseen) for side in ("O", "D")}
        original = ws.prior_rows

        def prior_rows(panel, side, side_saved, share=1.0):
            rows = original(panel, side, side_saved, share)
            if trial.arm in ("add_shadow", "shadow") and trial.side == side:
                stream = f"{self.season}|{len(rows)}|{int(rows.possessions.sum())}"
                rows[trial.name] = shuffled(rows[trial.name].to_numpy(float), trial.seed, trial.name, stream)
            return rows
        ws.prior_rows = prior_rows
        try:
            self.run.saved, self.run.out_dir = saved, out_dir
            for side in ("O", "D"):
                prime_in_context(saved[side], [ws.prior_rows(world.panel, side, saved[side])
                                               for world in self.run.worlds.values()])
            folds = ws.season_folds(self.run)
            started = time.time()
            for repeat in range(self.run.repeats):
                for fold in range(self.run.n_folds):
                    ws.run_fold(self.run, folds, repeat, fold, started)
            whole = self.run.rate(set(self.run.game_ids))
        finally:
            ws.prior_rows = original
            self.run.worlds = None
        return whole.table.assign(season=self.season)


def prime_in_context(side_saved: dict, rows: list) -> None:
    """Ask a side's in-context priors (in_context_prior.py) about every row of `rows` (one frame per world) in one
    pass each, so the season's folds read kept answers; LightGBM boosters are left alone."""
    from openrapm.in_context_prior import InContextPrior
    boosters = [side_saved["full"], *side_saved["folds"]]
    for bag in side_saved.get("bags", []):
        boosters += [bag["full"], *bag["folds"]]
    boosters = [booster for booster in boosters if isinstance(booster, InContextPrior)]
    if not boosters:
        return
    matrix = np.concatenate([frame[side_saved["model_feats"]].to_numpy(float) for frame in rows])
    for booster in boosters:
        booster.prime(matrix)


_FIT = {}            # in a fitting process: the booster parameters by file and the last seasons' training rows


def fit_task(tag: str, season: int, side: str, trial: Trial, params: dict | None = None):
    """(season, side, the trial's boosters on that side): the season's training rows, the trial's input list, the
    input shuffled when the arm asks, bag k at k + SEED_STEP x seed, the prior at the trial's variant; `params`, one
    side's LightGBM settings, in place of the variant's file (a tune's trial)."""
    from openrapm.data.config import load_config
    from openrapm.features import ROW_FEATURES
    from openrapm.in_context_prior import LEARNER
    settings = variant_settings(trial.variant, tag)
    if "params" not in _FIT:
        _FIT["params"], _FIT["rows"] = {}, {}
    if params is None and settings.prior.booster_file not in _FIT["params"]:
        _FIT["params"][settings.prior.booster_file] = box_prior_params(load_config(), settings.prior.booster_file)
    if season not in _FIT["rows"]:
        if len(_FIT["rows"]) >= 2:
            _FIT["rows"].pop(next(iter(_FIT["rows"])))
        with open(screen_dir(tag) / f"train_{season}.pkl", "rb") as handle:
            _FIT["rows"][season] = pickle.load(handle)["train"]
    train = _FIT["rows"][season][side]
    if trial.arm in ("add_shadow", "shadow") and trial.side == side:
        train = train.copy()
        train[trial.name] = shuffled(train[trial.name].to_numpy(float), trial.seed, trial.name, "train")
    names = trial_lists(trial)[side]
    if params is None:
        params = _FIT["params"][settings.prior.booster_file][side]
    if params.get("learner") == LEARNER:
        # no bags to draw: the seed picks the Kumo members' training rows instead
        return season, side, fit_bags(train, names + ROW_FEATURES, {**params, "draw": trial.seed}, settings, 0)
    return season, side, fit_bags(train, names + ROW_FEATURES, params, settings, trial.seed)


class Fitters:
    """A pool of small processes that only fit boosters; a trial's fits are sent off before the trial before it is
    rated, so the two overlap."""

    def __init__(self, tag: str, n_processes: int):
        self.tag = tag
        self.pool = mp.get_context("spawn").Pool(n_processes)

    def submit(self, trial: Trial, seasons, params: dict | None = None) -> list:
        """Every season's fits of `trial`, at {side: settings} `params` when given (a tune's trial)."""
        # offence first: its 1,091-tree boosters take most of the time
        return [self.pool.apply_async(fit_task, (self.tag, season, side, trial, None if params is None
                                                 else params[side]))
                for side in ("O", "D") for season in seasons]

    @staticmethod
    def collect(pending: list) -> dict:
        models: dict = {}
        for result in pending:
            season, side, model = result.get()
            models.setdefault(season, {})[side] = model
        return models

    def close(self) -> None:
        self.pool.close()
        self.pool.join()


def cached_run_class():
    """A SeasonRun whose worlds come from the cache instead of being rebuilt from the fit games each trial."""
    from openrapm.commands import within_season as ws

    class CachedRun(ws.SeasonRun):
        worlds: dict = None

        def world(self, fit_ids):
            return self.worlds[frozenset(str(game) for game in fit_ids)]
    return CachedRun


def _worker(tag: str, seasons: list, connection) -> None:
    """A worker process: holds its seasons, rates each trial it is sent, answers with the whole-season tables."""
    import warnings
    warnings.filterwarnings("ignore")
    held = {season: Season(tag, season) for season in seasons}
    connection.send(("ready", seasons))
    while True:
        message = connection.recv()
        if message is None:
            return
        trial, out_dir, models = message
        try:
            tables = [held[season].rate(trial, Path(out_dir), models[season]) for season in seasons]
            connection.send(("done", pd.concat(tables, ignore_index=True)))
        except Exception as error:          # the parent stops the run and prints it
            import traceback
            connection.send(("error", f"{error}\n{traceback.format_exc()}"))


class Workers:
    """Persistent rating workers, the tune seasons dealt out among them; each holds its seasons' SeasonRuns and reads
    one season's cache at a time."""

    def __init__(self, tag: str, n_workers: int, seasons):
        context = mp.get_context("spawn")
        chunks = [list(tuple(seasons)[k::n_workers]) for k in range(n_workers)]
        self.chunks = chunks
        self.links, self.processes = [], []
        for chunk in chunks:
            parent, child = context.Pipe()
            process = context.Process(target=_worker, args=(tag, chunk, child), daemon=True)
            process.start()
            self.links.append(parent)
            self.processes.append(process)
        for link in self.links:
            kind, value = link.recv()
            if kind != "ready":
                raise SystemExit(f"a worker failed to start: {value}")
        print(f"{n_workers} workers ready", flush=True)

    def rate(self, trial: Trial, out_dir: Path, models: dict) -> pd.DataFrame:
        for link, seasons in zip(self.links, self.chunks):
            link.send((trial, str(out_dir), {season: models[season] for season in seasons}))
        tables = []
        for link in self.links:
            kind, value = link.recv()
            if kind != "done":
                raise SystemExit(f"a worker failed on {trial.key}:\n{value}")
            tables.append(value)
        return pd.concat(tables, ignore_index=True)

    def close(self) -> None:
        for link in self.links:
            link.send(None)
        for process in self.processes:
            process.join(timeout=60)


# ------------------------------------------------------------------------------------------------- scoring
def score_folds(name: str, seasons) -> pd.DataFrame:
    """Check 151's error per season, stage and slice on the trial's folds (outputs/within/<name>), step 4 fitted on
    them alone."""
    wowy = _load("wowy_check", "151_wowy.py")
    return wowy.system_errors(wowy._audit(), within_dir(), name, (min(seasons), max(seasons)))


class Guard:
    """The whole seasons after the trial's step 4 predicting the seasons either side, team-game by team-game."""

    def __init__(self, tag: str, seasons):
        from openrapm.data.config import load_config
        from openrapm.data.roles import main_teams
        from openrapm.data.tables import player_names
        guard = _load("decider_tune", "152_decider_tune.py")
        self.cfg = load_config()
        self.seasons = sorted(seasons)
        self.scored = {}
        for season in self.seasons:
            with open(screen_dir(tag) / f"season_{season}.pkl", "rb") as handle:
                cache = pickle.load(handle)
            for other, design in cache["scored"].items():
                self.scored[(season, other)] = guard.duck_guard(design)
        self.teams = main_teams(self.cfg)
        self.names = player_names(self.cfg)

    def score(self, whole: pd.DataFrame, name: str) -> pd.DataFrame:
        from openrapm import prior_rapm, prior_shrink, scoring
        from openrapm.commands.build import step4_multipliers, within_layer_multipliers
        from openrapm.settings import PRODUCT
        wowy = _load("wowy_check", "151_wowy.py")
        folds, where = wowy.system_folds(within_dir(), name)
        shrink = replace(PRODUCT.shrink, rule="test", nearest=NEAREST, extrapolate_folds_tag="")
        seasons = self.seasons
        multipliers = step4_multipliers(seasons, folds, shrink, None)
        if shrink.within_team_layer:
            layer = within_layer_multipliers(self.cfg, seasons, folds, where, shrink)
            multipliers = multipliers.merge(layer, on="season")
        published = prior_rapm.published_table(whole, self.names)
        table = prior_shrink.shrink(published, multipliers, self.teams)
        out = []
        for (rated, scored), design in self.scored.items():
            rows = table[table.season == rated]
            ratings = pd.DataFrame({"player_id": rows.player_id.to_numpy(), "o": rows.rating_off.to_numpy(float),
                                    "d": -rows.rating_def.to_numpy(float), "poss": rows.poss_off.to_numpy(float)})
            result = scoring.score(scoring.predict_season(scoring.Ratings(ratings, fill_o=0.0, fill_d=0.0), design))
            out.append(dict(rated=rated, scored=scored, tg=float(result["tg"])))
        return pd.DataFrame(out)


def plan(tag: str, seeds: int, base_seeds: int, variant: str = "shipped", all_arms: bool = False) -> list:
    """Every trial, the shipped lists at every base seed first, then each seed's arms in turn, so a run stopped part
    way has whole seeds.  At another variant of the prior, the shipped lists at every base seed only, unless
    `all_arms` asks for the whole screen at it (the screen at the tune's settings, experiment 120)."""
    from openrapm.features import CREATION
    trials = [Trial("base", "-", "-", seed, variant) for seed in range(base_seeds)]
    if variant != "shipped" and not all_arms:
        return trials
    for seed in range(seeds):
        for side in ("O", "D"):
            for name in candidates(tag)[side]:
                trials += [Trial("add", side, name, seed, variant), Trial("add_shadow", side, name, seed, variant)]
        trials.append(Trial("shadow", "O", CREATION[0], seed, variant))
    return trials


def compare_reference(name: str) -> float:
    """The largest rating difference between the trial's folds and the reference build's folds on disk."""
    worst = 0.0
    trial_dir = within_dir() / name
    for path in sorted(trial_dir.glob("players_*.parquet")):
        season = int(path.stem.split("_")[1])
        folder = within_dir() / (f"{REFERENCE}_early" if season < 2017 else f"{REFERENCE}_within")
        mine = pd.read_parquet(path).set_index(["player_id", "side"])
        theirs = pd.read_parquet(folder / path.name).set_index(["player_id", "side"])
        for column in ("rating_off", "rating_def"):
            diff = np.abs(mine[column] - theirs[column].reindex(mine.index)).to_numpy(float)
            worst = max(worst, float(np.nanmax(diff)))
    return worst


def run_dry(args) -> int:
    """The shipped lists at seed 0 on the seasons asked, in this process: the folds against the reference build's on
    disk, and the time a season takes."""
    seasons = [int(s) for s in args.seasons.split(",") if s]
    name = f"screen/{args.tag}/dry"
    folder = within_dir() / name
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    trial = Trial("base", "-", "-", 0)
    for season in seasons:
        started = time.time()
        held = Season(args.tag, season)
        loaded = time.time()
        models = {side: fit_task(args.tag, season, side, trial)[2] for side in ("O", "D")}
        fitted = time.time()
        held.rate(trial, folder, models)
        print(f"{season}: loaded in {loaded - started:.0f}s, fitted in {fitted - loaded:.0f}s, rated in "
              f"{time.time() - fitted:.0f}s; largest rating difference against {REFERENCE}'s folds "
              f"{compare_reference(name):.1e}", flush=True)
    return 0


def wait_for_the_daily_job() -> None:
    """Between trials from 5:50 AM, hold until the morning's update (scripts/daily_update.sh, 6:00 AM) has written
    today's result to outputs/daily/last.txt, or until 7:30 AM: the machine is not to be thrashed."""
    from datetime import datetime

    from openrapm.data.config import load_config, resolve
    last = resolve(load_config(), "outputs") / "daily" / "last.txt"
    while True:
        now = datetime.now()
        if not (now.replace(hour=5, minute=50) <= now <= now.replace(hour=7, minute=30)):
            return
        if last.exists() and last.read_text().startswith(now.strftime("%Y-%m-%d")):
            return
        print(f"  {now:%I:%M %p}: waiting for the daily update", flush=True)
        time.sleep(120)


def result_paths(out_dir: Path, variant: str) -> tuple[Path, Path]:
    """(trials, guard) of a variant: trials.csv and guard.csv for the shipped prior, trials_<variant>.csv and
    guard_<variant>.csv for another."""
    suffix = "" if variant == "shipped" else f"_{variant}"
    return out_dir / f"trials{suffix}.csv", out_dir / f"guard{suffix}.csv"


def run_screen(args) -> int:
    out_dir = screen_dir(args.tag)
    seasons = screen_seasons(args.tag)
    variant_settings(args.variant, args.tag)
    trials_path, guard_path = result_paths(out_dir, args.variant)
    done = set()
    if trials_path.exists():
        done = set(pd.read_csv(trials_path).trial.unique())
    todo = [trial for trial in plan(args.tag, args.seeds, args.base_seeds, args.variant, args.all_arms)
            if trial.key not in done]
    print(f"{len(todo)} trials to run, {len(done)} done; seasons {', '.join(map(str, seasons))}", flush=True)
    if not todo:
        return 0
    fitters = Fitters(args.tag, args.fitters)
    workers = Workers(args.tag, args.workers, seasons)
    guard = Guard(args.tag, seasons)
    try:
        wait_for_the_daily_job()
        pending = fitters.submit(todo[0], seasons)
        for number, trial in enumerate(todo, start=1):
            started = time.time()
            models = fitters.collect(pending)
            fitted = time.time()
            if number < len(todo):
                wait_for_the_daily_job()
                pending = fitters.submit(todo[number], seasons)
            name = f"screen/{args.tag}/{trial.key}"
            folder = within_dir() / name
            if folder.exists():
                shutil.rmtree(folder)
            folder.mkdir(parents=True)
            whole = workers.rate(trial, folder, models)
            rated = time.time()
            errors = score_folds(name, seasons).assign(trial=trial.key, arm=trial.arm, side=trial.side,
                                                       input=trial.name, seed=trial.seed)
            scores = guard.score(whole, name).assign(trial=trial.key, arm=trial.arm, side=trial.side,
                                                     input=trial.name, seed=trial.seed)
            reading = errors[(errors.stage == "step4") & (errors.slice == "all")].error.mean()
            extra = ""
            if trial.arm == "base" and trial.seed == 0 and trial.variant == "shipped":
                worst = compare_reference(name)
                extra = f"; against {REFERENCE}'s folds on disk, largest rating difference {worst:.1e}"
                if worst > 1e-6:
                    raise SystemExit(f"the shipped lists at seed 0 miss {REFERENCE}'s folds by {worst:.1e}{extra}")
            errors.to_csv(trials_path, mode="a", header=not trials_path.exists(), index=False)
            scores.to_csv(guard_path, mode="a", header=not guard_path.exists(), index=False)
            if not (trial.arm == "base" and trial.seed == 0):
                shutil.rmtree(folder)
            print(f"[{number}/{len(todo)}] {trial.key}: decider {reading:.4f}{extra} (waited {fitted - started:.0f}s "
                  f"for the fits, rated in {rated - fitted:.0f}s, scored in {time.time() - rated:.0f}s)", flush=True)
    finally:
        workers.close()
        fitters.close()
    return 0


# ------------------------------------------------------------------------------------------------- the table
def paired(mine: pd.Series, theirs: pd.Series) -> dict:
    diff = (mine - theirs).dropna()
    se = float(diff.std(ddof=1) / np.sqrt(len(diff))) if len(diff) > 1 else float("nan")
    return dict(change=float(diff.mean()), se=se, z=float(diff.mean() / se) if se > 0 else float("nan"),
                better=int((diff < 0).sum()), n=len(diff))


def season_means(frame: pd.DataFrame, seeds) -> pd.Series:
    """Each season's error, `seeds` averaged."""
    return frame[frame.seed.isin(seeds)].groupby("season").error.mean()


def guard_means(frame: pd.DataFrame, seeds) -> pd.Series:
    return frame[frame.seed.isin(seeds)].groupby(["rated", "scored"]).tg.mean()


def screen_rows(tag: str, variant: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(every trial's step-4 error per season over all held-back stints, every trial's guard rows) of a variant."""
    trials_path, guard_path = result_paths(screen_dir(tag), variant)
    rows = pd.read_csv(trials_path, keep_default_na=False)
    return (rows[(rows.stage == "step4") & (rows.slice == "all")],
            pd.read_csv(guard_path, keep_default_na=False))


def add_minus_shadow(rows: pd.DataFrame, side: str, name: str) -> pd.Series:
    """Each season's add arm minus its add_shadow arm, the seeds both ran averaged within each."""
    add = rows[(rows.arm == "add") & (rows.side == side) & (rows.input == name)]
    shadow = rows[(rows.arm == "add_shadow") & (rows.side == side) & (rows.input == name)]
    both = sorted(set(add.seed.unique()) & set(shadow.seed.unique()))
    return season_means(add, both) - season_means(shadow, both)


def screen_table(tag: str, variant: str = "shipped") -> tuple[pd.DataFrame, list]:
    """(one row per arm and input: its change against the shipped lists and, for add, against its shuffled copy, the
    guard, by era against the copy, and the verdict by experiment 112's rule; the noise floor, the shipped lists at
    each other seed against seed 0)."""
    from openrapm.features import meaning
    rows, guard = screen_rows(tag, variant)
    base, base_guard = rows[rows.arm == "base"], guard[guard.arm == "base"]
    noise = [paired(season_means(base, [seed]), season_means(base, [0])) for seed in sorted(base.seed.unique())
             if seed != 0]
    table = []
    for (arm, side, name), group in rows[rows.arm != "base"].groupby(["arm", "side", "input"]):
        if arm == "add_shadow":
            continue
        seeds = sorted(group.seed.unique())
        mine = season_means(group, seeds)
        against_base = paired(mine, season_means(base, seeds))
        guarded = paired(guard_means(guard[(guard.arm == arm) & (guard.side == side) & (guard.input == name)], seeds),
                         guard_means(base_guard, seeds))
        row = dict(input=name, plain_name=meaning(name), side=side, arm=arm, seeds=len(seeds),
                   change_vs_shipped=against_base["change"], z_vs_shipped=against_base["z"],
                   seasons_better_vs_shipped=against_base["better"], guard_change=guarded["change"],
                   guard_z=guarded["z"])
        if arm == "add":
            difference = add_minus_shadow(rows, side, name)
            against_shadow = paired(difference, pd.Series(0.0, index=difference.index))
            row.update(change_vs_shadow=against_shadow["change"], z_vs_shadow=against_shadow["z"],
                       seasons_better_vs_shadow=against_shadow["better"])
            for era, (first, last) in ERA_SPANS.items():
                inside = difference[(difference.index >= first) & (difference.index <= last)]
                if len(inside):
                    row[f"vs_shadow_{era}"] = float(inside.mean())
            row["verdict"] = ("passes" if against_shadow["z"] <= -2 and against_base["change"] < 0
                              and not guarded["z"] >= 2 else "fails")
        table.append(row)
    return pd.DataFrame(table).sort_values(["arm", "side", "z_vs_shadow"], na_position="last"), noise


def table_path(tag: str, variant: str, kind: str = "table") -> Path:
    from openrapm.data.config import load_config, resolve
    suffix = "" if variant == "shipped" else f"_{variant}"
    return resolve(load_config(), "outputs") / "csv" / f"screen_{tag}{suffix}_{kind}.csv"


def run_table(args) -> int:
    out, noise = screen_table(args.tag, args.variant)
    path = table_path(args.tag, args.variant)
    out.to_csv(path, index=False)
    pd.set_option("display.width", 250)
    print("the noise floor: the shipped lists at each other seed against seed 0")
    for k, result in enumerate(noise, start=1):
        print(f"  seed {k}: {result['change']:+.4f}, z {result['z']:+.2f}, {result['better']} of {result['n']}")
    print(out.round(4).to_string(index=False))
    print(f"\nwrote {path}")
    return 0


def verdict_of(row: pd.Series) -> str:
    return row["verdict"] if isinstance(row.get("verdict"), str) else "-"


def run_tandem(args) -> int:
    """The screen at the shipped booster settings and at a variant's side by side (experiment 120): for each candidate,
    its change against its shuffled copy at both, its verdict at both, and the variant's reading minus the shipped
    one, season by season (each season's seeds averaged within each arm, the two settings paired by season); and the
    shipped lists at the variant's settings against the shipped settings, every base seed averaged."""
    from openrapm.features import meaning
    first, _ = screen_table(args.tag, "shipped")
    second, _ = screen_table(args.tag, args.variant)
    rows_first, _ = screen_rows(args.tag, "shipped")
    rows_second, _ = screen_rows(args.tag, args.variant)
    table = []
    for side in ("O", "D"):
        for name in candidates(args.tag)[side]:
            at = {label: frame[(frame.arm == "add") & (frame.side == side) & (frame.input == name)]
                  for label, frame in (("shipped", first), (args.variant, second))}
            if any(frame.empty for frame in at.values()):
                continue
            moved = paired(add_minus_shadow(rows_second, side, name), add_minus_shadow(rows_first, side, name))
            row = dict(input=name, plain_name=meaning(name), side=side)
            for label, frame in at.items():
                one = frame.iloc[0]
                row.update({f"vs_copy_{label}": one.change_vs_shadow, f"z_{label}": one.z_vs_shadow,
                            f"verdict_{label}": verdict_of(one)})
            row.update(moved=moved["change"], moved_se=moved["se"], moved_z=moved["z"], seasons=moved["n"])
            row["verdict_flips"] = row["verdict_shipped"] != row[f"verdict_{args.variant}"]
            row["matters"] = bool(row["verdict_flips"] and abs(moved["z"]) >= 2)
            table.append(row)
    out = pd.DataFrame(table)
    base_first = rows_first[rows_first.arm == "base"]
    base_second = rows_second[rows_second.arm == "base"]
    retune = paired(season_means(base_second, sorted(base_second.seed.unique())),
                    season_means(base_first, sorted(base_first.seed.unique())))
    path = table_path(args.tag, args.variant, "tandem")
    out.to_csv(path, index=False)
    pd.set_option("display.width", 250)
    print(out.round(4).to_string(index=False))
    print(f"\nthe shipped lists at {args.variant}'s settings against the shipped settings, every base seed averaged: "
          f"{retune['change']:+.4f}, z {retune['z']:+.2f}, {retune['better']} of {retune['n']} seasons better")
    print(f"verdicts that flip: {int(out.verdict_flips.sum())} of {len(out)}; flips whose reading moved beyond z 2: "
          f"{int(out.matters.sum())}")
    print(f"\nwrote {path}")
    return 0


def run_variants(args) -> int:
    """Each variant of the prior against the shipped one, the shipped lists at every random draw: the change on the
    decider and the guard, each season's draws averaged and paired by season, by era, and how far the draws spread
    (each season's standard deviation over its draws, averaged over the seasons)."""
    from openrapm.data.config import load_config, resolve
    out_dir = screen_dir(args.tag)
    errors, guards = {}, {}
    for variant in VARIANTS:
        trials_path, guard_path = result_paths(out_dir, variant)
        if not trials_path.exists():
            continue
        rows = pd.read_csv(trials_path, keep_default_na=False)
        rows = rows[(rows.arm == "base") & (rows.stage == "step4") & (rows.slice == "all")]
        guard = pd.read_csv(guard_path, keep_default_na=False)
        errors[variant] = rows.pivot_table(index="season", columns="seed", values="error")
        guards[variant] = guard[guard.arm == "base"].pivot_table(index=["rated", "scored"], columns="seed",
                                                                 values="tg")
    if "shipped" not in errors:
        raise SystemExit(f"no shipped base arm in {out_dir}")
    shipped_spread = float(errors["shipped"].std(axis=1, ddof=1).mean())
    table = []
    for variant, mine in errors.items():
        change = paired(mine.mean(axis=1), errors["shipped"].mean(axis=1))
        guarded = paired(guards[variant].mean(axis=1), guards["shipped"].mean(axis=1))
        spread = float(mine.std(axis=1, ddof=1).mean())
        row = dict(variant=variant, draws=mine.shape[1], change_vs_shipped=change["change"], se=change["se"],
                   z=change["z"], seasons_better=change["better"], seasons=change["n"])
        for era, seasons in ERAS.items():
            row[f"change_{era}"] = float((mine.mean(axis=1) - errors["shipped"].mean(axis=1)).reindex(seasons).mean())
        row.update(draw_spread=spread, draw_spread_vs_shipped=spread / shipped_spread,
                   guard_change=guarded["change"], guard_z=guarded["z"])
        table.append(row)
    out = pd.DataFrame(table)
    path = resolve(load_config(), "outputs") / "csv" / f"screen_{args.tag}_variants.csv"
    out.to_csv(path, index=False)
    pd.set_option("display.width", 250)
    print(out.round(4).to_string(index=False))
    print("\neach season, the draws averaged, minus the shipped prior's")
    seasons = pd.DataFrame({variant: mine.mean(axis=1) - errors["shipped"].mean(axis=1)
                            for variant, mine in errors.items() if variant != "shipped"})
    print(seasons.round(4).to_string())
    print(f"\nwrote {path}")
    return 0


# ------------------------------------------------------------------------------------------------- the tune
# Experiment 120 (the owner, 2026-10-29: "hyperparameter tuning + boruta should probably always happen in tandem (or -
# if it shouldn't we should prove it)"): experiment 76's guarded search again, on today's pipeline, with the tag's
# pool in the lists, so the settings it finds are the ones a screen run in tandem with a tune would read the candidates
# at.  The shipped settings were searched on the pipeline of 2026-10-16: one booster a fold, no part-season rows, no
# second prior, one layer of step 4.
GUARD_Z_LIMIT = 1.0      # check 152's: the next-season check's z on its 20 readings, about +1.7 on the full test
IN_FLIGHT = 2            # trials at once: a trial's boosters are fitted while the trial before it is rated


def suggest(trial) -> dict:
    """{side: one trial's LightGBM settings} on experiment 76's space (check 152's suggest_fixed), less its era marker
    and its defence factor: the inputs are the tag's pool, and step 4 stays as shipped."""
    linear = trial.suggest_categorical("linear_tree", [True, False])
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
    return params


def enqueue(study, booster_file: str) -> None:
    """Queue a params file's settings as one of the search's first trials."""
    from openrapm.data.config import load_config
    spec = box_prior_params(load_config(), booster_file)
    values = {"linear_tree": bool(spec["O"].get("linear_tree", False))}
    for side in ("O", "D"):
        for key, value in spec[side].items():
            if key != "linear_tree":
                values[f"{side}_{key}"] = value
    study.enqueue_trial(values, skip_if_exists=True)


def rate_and_score(tag: str, label: str, trial: Trial, models: dict, workers: Workers, guard: Guard,
                   seasons) -> tuple[pd.Series, pd.Series]:
    """(each season's step-4 error over every held-back stint, each guard reading's team-game error) of a trial's
    boosters; its folds are written under outputs/within/screen/<tag>/<label> and removed after."""
    name = f"screen/{tag}/{label}"
    folder = within_dir() / name
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    whole = workers.rate(trial, folder, models)
    errors = score_folds(name, seasons)
    errors = errors[(errors.stage == "step4") & (errors.slice == "all")].set_index("season").error
    readings = guard.score(whole, name).set_index(["rated", "scored"]).tg
    shutil.rmtree(folder)
    return errors, readings


def tune_reference(args, fitters: Fitters, workers: Workers, guard: Guard) -> tuple[pd.Series, pd.Series]:
    """(each season's error, each guard reading) of the pool lists at the shipped settings, seed 0, which the search's
    limit is read against; computed once and kept beside the search as <study>_reference.pkl."""
    path = screen_dir(args.tag) / f"{args.study}_reference.pkl"
    if not path.exists():
        started = time.time()
        trial = Trial("pool", "-", args.tag, 0)
        models = fitters.collect(fitters.submit(trial, TUNE_SEASONS))
        pd.to_pickle(rate_and_score(args.tag, f"{args.study}_reference", trial, models, workers, guard, TUNE_SEASONS),
                     path)
        print(f"the reference, the pool lists at the shipped settings, seed 0 ({time.time() - started:.0f}s)",
              flush=True)
    return pd.read_pickle(path)


def run_tune(args) -> int:
    """The guarded search: Optuna's Gaussian-process sampler on experiment 76's space, two trials at once (the sampler
    reads a trial still running as if it had come out as well as the best so far), the score the within-season test
    after step 4 on the ten tune seasons at seed 0, the next-season check a hard limit against the pool lists at the
    shipped settings; the best trial within the limit written as params/booster_params_<study>.json."""
    from collections import deque

    import optuna
    from optuna.trial import TrialState

    from openrapm.settings import TEST
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    out_dir = screen_dir(args.tag)
    seasons = TUNE_SEASONS
    storage = f"sqlite:///{(out_dir / f'{args.study}.db').as_posix()}"
    sampler = optuna.samplers.GPSampler(seed=args.seed, n_startup_trials=args.startup, deterministic_objective=True,
                                        constraints_func=lambda frozen: frozen.user_attrs.get("constraint", [1.0]))
    study = optuna.create_study(study_name=args.study, storage=storage, direction="minimize", sampler=sampler,
                                load_if_exists=True)
    for stale in study.get_trials(deepcopy=False, states=(TrialState.RUNNING,)):
        study.tell(stale.number, state=TrialState.FAIL)       # a run stopped part way
    if not study.trials:
        enqueue(study, TEST.prior.booster_file)
        enqueue(study, "params/booster_params_lgb1_tune1.json")
    pool_trial = Trial("pool", "-", args.tag, 0)
    trials_csv = out_dir / f"{args.study}_trials.csv"

    def finished() -> int:
        return len(study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,)))

    fitters = Fitters(args.tag, args.fitters)
    workers = Workers(args.tag, args.workers, seasons)
    guard = Guard(args.tag, seasons)
    flight: deque = deque()

    def launch() -> None:
        wait_for_the_daily_job()
        trial = study.ask()
        params = suggest(trial)
        flight.append((trial, params, fitters.submit(pool_trial, seasons, params), time.time()))

    try:
        reference_errors, reference_readings = tune_reference(args, fitters, workers, guard)
        while len(flight) < IN_FLIGHT and finished() + len(flight) < args.trials:
            launch()
        while flight:
            trial, params, pending, submitted = flight.popleft()
            models = fitters.collect(pending)
            if finished() + len(flight) + 1 < args.trials:
                launch()
            errors, readings = rate_and_score(args.tag, f"{args.study}_t{trial.number}", pool_trial, models, workers,
                                              guard, seasons)
            score = float(errors.mean())
            change = paired(errors, reference_errors)
            team = paired(readings, reference_readings)
            limit = team["z"] if np.isfinite(team["z"]) else 0.0
            trial.set_user_attr("constraint", [float(limit) - GUARD_Z_LIMIT])
            for key, value in (("params", params), ("vs_reference", change), ("guard_team_game", team),
                               ("seconds", time.time() - submitted)):
                trial.set_user_attr(key, value)
            errors.rename("error").reset_index().assign(trial=trial.number, guard_change=team["change"],
                                                        guard_z=team["z"]).to_csv(
                trials_csv, mode="a", header=not trials_csv.exists(), index=False)
            study.tell(trial, score)
            print(f"trial {trial.number}: decider {change['change']:+.4f} z {change['z']:+.2f} {change['better']}/"
                  f"{change['n']}; next-season {team['change']:+.4f} z {team['z']:+.2f}"
                  f"{'' if limit <= GUARD_Z_LIMIT else ', FAILS the limit'}; linear {params['O']['linear_tree']}  "
                  f"({time.time() - submitted:.0f}s)", flush=True)
    finally:
        workers.close()
        fitters.close()
    return write_tuned(study, args)


def write_tuned(study, args) -> int:
    """The five best trials within the limit, and the best written as params/booster_params_<study>.json."""
    import json

    from optuna.trial import TrialState

    from openrapm.data.config import load_config, resolve_path
    feasible = [trial for trial in study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
                if trial.user_attrs.get("constraint", [1.0])[0] <= 0]
    ranked = sorted(feasible, key=lambda trial: trial.value)
    for trial in ranked[:5]:
        change, team = trial.user_attrs["vs_reference"], trial.user_attrs["guard_team_game"]
        print(f"  trial {trial.number}: decider {change['change']:+.4f} z {change['z']:+.2f}; next-season "
              f"{team['change']:+.4f} z {team['z']:+.2f}")
    best = ranked[0]
    spec = {side: {"learner": "lightgbm", "params": best.user_attrs["params"][side]} for side in ("O", "D")}
    spec["note"] = (f"check 155's tune {args.study} on tag {args.tag} (experiment 120, {time.strftime('%Y-%m-%d')}): "
                    f"trial {best.number} of {len(study.trials)}, Optuna's GP sampler on the ten tune seasons, the "
                    f"pool lists, the next-season check a hard limit; against the pool lists at the shipped settings "
                    f"{best.user_attrs['vs_reference']}; next-season {best.user_attrs['guard_team_game']}")
    path = resolve_path(load_config(), f"params/booster_params_{args.study}.json")
    path.write_bytes(json.dumps(spec, indent=2).encode("utf-8"))
    print(f"best within the limit: trial {best.number}; wrote {path}", flush=True)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="The input screen on the within-season test (experiment 112).")
    commands = parser.add_subparsers(dest="command", required=True)
    cache = commands.add_parser("cache", help="each season's cache, once; resumes")
    cache.add_argument("--tag", required=True)
    cache.add_argument("--seasons", default="", help="comma-separated, default the tag's screen seasons and the ten "
                                                     "tune seasons")
    run = commands.add_parser("run", help="every trial; resumes")
    run.add_argument("--tag", required=True)
    run.add_argument("--workers", type=int, default=2, help="rating processes, about 6 GB each (default %(default)s)")
    run.add_argument("--fitters", type=int, default=6, help="fitting processes, under 1 GB each (default %(default)s)")
    run.add_argument("--seeds", type=int, default=3)
    run.add_argument("--base-seeds", type=int, default=5)
    run.add_argument("--variant", default="shipped", choices=sorted(VARIANTS),
                     help="the box-score prior's settings; another than shipped runs the base arm alone, at every "
                          "base seed, into trials_<variant>.csv (default %(default)s)")
    run.add_argument("--all-arms", action="store_true", help="at another variant, every arm of the screen")
    variants = commands.add_parser("variants", help="each variant of the prior against the shipped one")
    variants.add_argument("--tag", required=True)
    dry = commands.add_parser("dry", help="the shipped lists at seed 0 on a few seasons, against the reference")
    dry.add_argument("--tag", required=True)
    dry.add_argument("--seasons", required=True)
    table = commands.add_parser("table", help="the full table")
    table.add_argument("--tag", required=True)
    table.add_argument("--variant", default="shipped", choices=sorted(VARIANTS))
    tune = commands.add_parser("tune", help="the booster settings searched again with the tag's pool in; resumes")
    tune.add_argument("--tag", required=True)
    tune.add_argument("--study", default="tune120", help="the search's name and its params file's (default "
                                                         "%(default)s)")
    tune.add_argument("--trials", type=int, default=80, help="finished trials to reach (default %(default)s)")
    tune.add_argument("--startup", type=int, default=10, help="trials before the Gaussian process, the queued ones "
                                                              "among them (default %(default)s)")
    tune.add_argument("--seed", type=int, default=0)
    tune.add_argument("--workers", type=int, default=2)
    tune.add_argument("--fitters", type=int, default=6)
    tandem = commands.add_parser("tandem", help="the screen at the shipped settings and at a variant's, side by side")
    tandem.add_argument("--tag", required=True)
    tandem.add_argument("--variant", required=True, choices=sorted(VARIANTS))
    args = parser.parse_args(argv)
    if args.command == "cache":
        seasons = ([int(s) for s in args.seasons.split(",") if s]
                   or sorted(set(screen_seasons(args.tag)) | set(TUNE_SEASONS)))
        return build_cache(args.tag, seasons)
    if args.command == "run":
        return run_screen(args)
    if args.command == "dry":
        return run_dry(args)
    if args.command == "variants":
        return run_variants(args)
    if args.command == "tune":
        return run_tune(args)
    if args.command == "tandem":
        return run_tandem(args)
    return run_table(args)


if __name__ == "__main__":
    raise SystemExit(main())
