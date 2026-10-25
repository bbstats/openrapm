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
                  input shuffled in place
    seed          bag k of a trial at seed s is drawn and fitted at k + 100 s, so seed 0 is the shipped build; a
                  trial at another seed, nothing else changed, is the noise control
    guard         each tune season's whole-season ratings after the trial's step 4 predicting the seasons either side,
                  team-game by team-game (check 152's guard; step 5 is off)

The rule (DECISIONS.md, experiment 112, registered before the run): a candidate passes when, over the tune seasons,
seeds averaged within each season and paired by season, add beats add_shadow at z -2 or below, add beats the shipped
lists on the mean, and the guard is not worse at z +2.

    python scripts/checks/155_input_screen.py cache --tag screen112 --jobs 4     (once, resumes)
    python scripts/checks/155_input_screen.py run --tag screen112 --workers 5    (resumes)
    python scripts/checks/155_input_screen.py table --tag screen112

Writes under outputs/screen/<tag>/: season_<season>.pkl (the cache), trials.csv (one row per trial, season, stage and
slice), guard.csv (one row per trial and rated-scored pair), and outputs/csv/screen_<tag>_table.csv (the full table).
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
POOL = "screen112_pool"
REFERENCE = "creation110"                           # the shipped build whose saved boosters name the second prior
SEED_STEP = 100
NEAREST = 8


def _load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(file))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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
    settings = replace(TEST, prior=replace(TEST.prior, feature_set=POOL))
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
            names = list(FEATURE_SETS[POOL][side])
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
        with open(out_dir / f"season_{season}.pkl", "wb") as handle:
            pickle.dump(dict(season=season, unseen=list(unseen), train=train, worlds=worlds, scored=scored,
                             line=run.line), handle, protocol=5)
        designs.clear()
        size = (out_dir / f"season_{season}.pkl").stat().st_size / 1e6
        print(f"{season}: cached, {size:.0f} MB; the shipped boosters refitted give back the saved priors to "
              f"{worst:.1e} ({time.time() - started:.0f}s)", flush=True)
    return 0


def box_prior_params(cfg: dict) -> dict:
    from openrapm.commands.build import booster_params
    from openrapm.settings import TEST
    return booster_params(cfg, TEST.prior.booster_file)


def fit_bags(train: pd.DataFrame, inputs: list, params: dict, settings, seed: int):
    """box_prior.fit_prior with bag k drawn and fitted at k + SEED_STEP x seed: at seed 0, the shipped fit."""
    from openrapm import box_prior
    members = []
    for bag in range(int(settings.prior.bags)):
        k = bag + SEED_STEP * int(seed)
        rows, _ = box_prior.player_bootstrap(train, k)
        members.append(box_prior.fit_out_of_player(rows, inputs, params, settings.prior.n_folds,
                                                   settings.prior.fold_assignment, None, seed=k))
    return box_prior.BaggedPrior(members=members, inputs=list(inputs))


# ------------------------------------------------------------------------------------------------- a trial
@dataclass(frozen=True)
class Trial:
    arm: str                     # base, add, add_shadow, shadow
    side: str                    # "O", "D", or "-" for base
    name: str                    # the input, or "-" for base
    seed: int

    @property
    def key(self) -> str:
        return f"{self.arm}__{self.side}__{self.name}__s{self.seed}"


def trial_lists(trial: Trial) -> dict:
    """{side: the boosters' inputs} of a trial."""
    lists = shipped_lists()
    if trial.arm in ("add", "add_shadow"):
        lists[trial.side] = lists[trial.side] + [trial.name]
    return lists


def shuffled(values: np.ndarray, seed: int, name: str, stream: str) -> np.ndarray:
    """`values` with the non-missing ones permuted, the draw seeded by the seed, the input and the stream."""
    out = np.array(values, dtype=float)
    has = np.flatnonzero(~np.isnan(out))
    rng = np.random.default_rng([int(seed), zlib.crc32(f"{name}|{stream}".encode())])
    out[has] = out[has][rng.permutation(has.size)]
    return out


def saved_side(model, names: list) -> dict:
    """One side's boosters in the layout `openrapm folds` reads (within_season.saved_booster, prior_rows)."""
    from openrapm.features import ROW_FEATURES
    first = model.members[0]
    return dict(full=first.full, folds=first.fold_models, excluded=first.held_out_players, feats=list(names),
                model_feats=list(names) + ROW_FEATURES,
                bags=[dict(full=m.full, folds=m.fold_models, excluded=m.held_out_players) for m in model.members])


class Season:
    """One tune season in a worker: its cache and its SeasonRun, rated again for every trial."""

    def __init__(self, tag: str, season: int):
        from openrapm.commands import within_season as ws
        from openrapm.data.careers import careers_before
        from openrapm.data.config import load_config, resolve
        from openrapm.data.designs import DesignCache
        from openrapm.data.panel import SeasonSources, SeasonTables, share_cap
        from openrapm.data.roles import player_season_inputs, roles_path
        from openrapm.settings import TEST
        self.tag, self.season, self.cfg = tag, season, load_config()
        with open(screen_dir(tag) / f"season_{season}.pkl", "rb") as handle:
            cache = pickle.load(handle)
        self.train, self.worlds = cache["train"], cache["worlds"]
        self.settings = replace(TEST, prior=replace(TEST.prior, feature_set=POOL))
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
        self.run.worlds = self.worlds
        ws.attach_line(self.run, ws.fold_lines(self.cfg, meta, False), False)

    def rate(self, trial: Trial, out_dir: Path) -> pd.DataFrame:
        """Fit the trial's boosters, rate and write every fold into `out_dir`; return the whole season's step-3
        table (raw sign)."""
        from openrapm.commands import within_season as ws
        from openrapm.features import ROW_FEATURES
        lists = trial_lists(trial)
        saved = {}
        for side in ("O", "D"):
            train = self.train[side]
            if trial.arm in ("add_shadow", "shadow") and trial.side == side:
                train = train.copy()
                train[trial.name] = shuffled(train[trial.name].to_numpy(float), trial.seed, trial.name, "train")
            model = fit_bags(train, lists[side] + ROW_FEATURES, self.params[side], self.settings, trial.seed)
            saved[side] = saved_side(model, lists[side])
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
            folds = ws.season_folds(self.run)
            started = time.time()
            for repeat in range(self.run.repeats):
                for fold in range(self.run.n_folds):
                    ws.run_fold(self.run, folds, repeat, fold, started)
            whole = self.run.rate(set(self.run.game_ids))
        finally:
            ws.prior_rows = original
        return whole.table.assign(season=self.season)


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
        trial, out_dir = message
        try:
            tables = [held[season].rate(trial, Path(out_dir)) for season in seasons]
            connection.send(("done", pd.concat(tables, ignore_index=True)))
        except Exception as error:          # the parent stops the run and prints it
            import traceback
            connection.send(("error", f"{error}\n{traceback.format_exc()}"))


class Workers:
    """Persistent workers, the tune seasons dealt out among them."""

    def __init__(self, tag: str, n_workers: int):
        context = mp.get_context("spawn")
        chunks = [list(TUNE_SEASONS[k::n_workers]) for k in range(n_workers)]
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

    def rate(self, trial: Trial, out_dir: Path) -> pd.DataFrame:
        for link in self.links:
            link.send((trial, str(out_dir)))
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
def score_folds(name: str) -> pd.DataFrame:
    """Check 151's error per season, stage and slice on the trial's folds (outputs/within/<name>), step 4 fitted on
    them alone."""
    wowy = _load("wowy_check", "151_wowy.py")
    return wowy.system_errors(wowy._audit(), within_dir(), name, (TUNE_SEASONS[0], TUNE_SEASONS[-1]))


class Guard:
    """The whole seasons after the trial's step 4 predicting the seasons either side, team-game by team-game."""

    def __init__(self, tag: str):
        from openrapm.data.config import load_config
        from openrapm.data.roles import main_teams
        from openrapm.data.tables import player_names
        guard = _load("decider_tune", "152_decider_tune.py")
        self.cfg = load_config()
        self.scored = {}
        for season in TUNE_SEASONS:
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
        seasons = sorted(TUNE_SEASONS)
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


def plan(seeds: int, base_seeds: int) -> list:
    """Every trial, the shipped lists at every base seed first, then each seed's arms in turn, so a run stopped part
    way has whole seeds."""
    from openrapm.features import CREATION, SCREEN_CANDIDATES
    trials = [Trial("base", "-", "-", seed) for seed in range(base_seeds)]
    for seed in range(seeds):
        for side in ("O", "D"):
            for name in SCREEN_CANDIDATES[side]:
                trials += [Trial("add", side, name, seed), Trial("add_shadow", side, name, seed)]
        trials.append(Trial("shadow", "O", CREATION[0], seed))
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
    for season in seasons:
        started = time.time()
        held = Season(args.tag, season)
        loaded = time.time() - started
        held.rate(Trial("base", "-", "-", 0), folder)
        print(f"{season}: loaded in {loaded:.0f}s, rated in {time.time() - started - loaded:.0f}s; largest rating "
              f"difference against {REFERENCE}'s folds {compare_reference(name):.1e}", flush=True)
    return 0


def run_screen(args) -> int:
    out_dir = screen_dir(args.tag)
    trials_path, guard_path = out_dir / "trials.csv", out_dir / "guard.csv"
    done = set()
    if trials_path.exists():
        done = set(pd.read_csv(trials_path).trial.unique())
    todo = [trial for trial in plan(args.seeds, args.base_seeds) if trial.key not in done]
    print(f"{len(todo)} trials to run, {len(done)} done", flush=True)
    if not todo:
        return 0
    workers = Workers(args.tag, args.workers)
    guard = Guard(args.tag)
    try:
        for number, trial in enumerate(todo, start=1):
            started = time.time()
            name = f"screen/{args.tag}/{trial.key}"
            folder = within_dir() / name
            if folder.exists():
                shutil.rmtree(folder)
            folder.mkdir(parents=True)
            whole = workers.rate(trial, folder)
            errors = score_folds(name).assign(trial=trial.key, arm=trial.arm, side=trial.side, input=trial.name,
                                              seed=trial.seed)
            scores = guard.score(whole, name).assign(trial=trial.key, arm=trial.arm, side=trial.side,
                                                     input=trial.name, seed=trial.seed)
            reading = errors[(errors.stage == "step4") & (errors.slice == "all")].error.mean()
            extra = ""
            if trial.arm == "base" and trial.seed == 0:
                worst = compare_reference(name)
                extra = f"; against {REFERENCE}'s folds on disk, largest rating difference {worst:.1e}"
                if worst > 1e-6:
                    raise SystemExit(f"the shipped lists at seed 0 miss {REFERENCE}'s folds by {worst:.1e}{extra}")
            errors.to_csv(trials_path, mode="a", header=not trials_path.exists(), index=False)
            scores.to_csv(guard_path, mode="a", header=not guard_path.exists(), index=False)
            if not (trial.arm == "base" and trial.seed == 0):
                shutil.rmtree(folder)
            print(f"[{number}/{len(todo)}] {trial.key}: decider {reading:.4f}{extra} ({time.time() - started:.0f}s)",
                  flush=True)
    finally:
        workers.close()
    return 0


# ------------------------------------------------------------------------------------------------- the table
def paired(mine: pd.Series, theirs: pd.Series) -> dict:
    diff = (mine - theirs).dropna()
    se = float(diff.std(ddof=1) / np.sqrt(len(diff))) if len(diff) > 1 else float("nan")
    return dict(change=float(diff.mean()), se=se, z=float(diff.mean() / se) if se > 0 else float("nan"),
                better=int((diff < 0).sum()), n=len(diff))


def run_table(args) -> int:
    from openrapm.data.config import load_config, resolve
    from openrapm.features import meaning
    out_dir = screen_dir(args.tag)
    rows = pd.read_csv(out_dir / "trials.csv", keep_default_na=False)
    rows = rows[(rows.stage == "step4") & (rows.slice == "all")]
    guard = pd.read_csv(out_dir / "guard.csv", keep_default_na=False)

    def season_means(frame: pd.DataFrame, seeds) -> pd.Series:
        return frame[frame.seed.isin(seeds)].groupby("season").error.mean()

    def guard_means(frame: pd.DataFrame, seeds) -> pd.Series:
        return frame[frame.seed.isin(seeds)].groupby(["rated", "scored"]).tg.mean()
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
            shadow = rows[(rows.arm == "add_shadow") & (rows.side == side) & (rows.input == name)]
            both = sorted(set(seeds) & set(shadow.seed.unique()))
            against_shadow = paired(season_means(group, both), season_means(shadow, both))
            row.update(change_vs_shadow=against_shadow["change"], z_vs_shadow=against_shadow["z"],
                       seasons_better_vs_shadow=against_shadow["better"])
            row["verdict"] = ("passes" if against_shadow["z"] <= -2 and against_base["change"] < 0
                              and not guarded["z"] >= 2 else "fails")
        table.append(row)
    out = pd.DataFrame(table).sort_values(["arm", "side", "z_vs_shadow"], na_position="last")
    path = resolve(load_config(), "outputs") / "csv" / f"screen_{args.tag}_table.csv"
    out.to_csv(path, index=False)
    pd.set_option("display.width", 250)
    print("the noise floor: the shipped lists at each other seed against seed 0")
    for k, result in enumerate(noise, start=1):
        print(f"  seed {k}: {result['change']:+.4f}, z {result['z']:+.2f}, {result['better']} of {result['n']}")
    print(out.round(4).to_string(index=False))
    print(f"\nwrote {path}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="The input screen on the within-season test (experiment 112).")
    commands = parser.add_subparsers(dest="command", required=True)
    cache = commands.add_parser("cache", help="each tune season's cache, once; resumes")
    cache.add_argument("--tag", required=True)
    cache.add_argument("--seasons", default="", help="comma-separated, default every tune season")
    run = commands.add_parser("run", help="every trial; resumes")
    run.add_argument("--tag", required=True)
    run.add_argument("--workers", type=int, default=5)
    run.add_argument("--seeds", type=int, default=3)
    run.add_argument("--base-seeds", type=int, default=5)
    dry = commands.add_parser("dry", help="the shipped lists at seed 0 on a few seasons, against the reference")
    dry.add_argument("--tag", required=True)
    dry.add_argument("--seasons", required=True)
    table = commands.add_parser("table", help="the full table")
    table.add_argument("--tag", required=True)
    args = parser.parse_args(argv)
    if args.command == "cache":
        seasons = [int(s) for s in args.seasons.split(",") if s] or list(TUNE_SEASONS)
        return build_cache(args.tag, seasons)
    if args.command == "run":
        return run_screen(args)
    if args.command == "dry":
        return run_dry(args)
    return run_table(args)


if __name__ == "__main__":
    raise SystemExit(main())
