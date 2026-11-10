"""Check 156: the shrink's four fixed settings refitted for one system, on the within-season WOWY test with the
year-over-year check as a hard limit (the owner, 2026-10-27: "refit").

After step 4 fits its multipliers on a system's own held-back games, four fixed numbers multiply them
(settings.ShrinkSettings): the within layer's factors on offense and on defense, 0.75 and 1.0 (experiment 106: 0.75 is
the year-over-year test's own scale), and the defense factor by era, 0.89 before 2017 and 0.84 from 2017 (experiment
98: picked by eye from a width test).  All four were set on today's ratings over all thirty seasons, so a system built
on another prior is read at settings fitted to someone else's.  This check searches them for one system:

    score       check 151's within-season WOWY error after step 4, stint unit, every held-back stint, the mean over
                check 152's ten tune seasons, 1998, 2001, ..., 2025; the other twenty seasons are kept for the verdict
    hard limit  the next-season check: each tune season's whole-season ratings after step 4 predicting the seasons
                either side, team-game error per 100 as scripts/checks/63_yoy.py scores it, paired against today's
                published ratings; z at most +1.0 over the twenty readings (check 152's limit, about z +1.7 on 63's
                fifty-six)
    sampler     Optuna's Gaussian-process sampler, the four settings drawn on fixed ranges, the shipped four enqueued
                first (check 152's guarded search); the study in outputs/tune/refit156_<system>/study.db, so a stopped
                search resumes

Step 4's multipliers do not depend on the four settings (commands/build.step4_multipliers and
within_layer_multipliers apply them last), so each system's are fitted once with all four at 1 and a trial only
multiplies and scores: about 20 seconds.  Before the first trial the fast path is checked at the shipped settings
against check 151's own `system_errors` and against the system's step-4 table on disk, to the last digit.

`judge` reads each system's best trial that kept the limit and scores it on all thirty seasons with check 151's
`system_errors`, paired by season against the reference at its own best and as shipped: the twenty untouched
seasons, each era's untouched seasons, the ten tune seasons and all thirty.

    python scripts/checks/156_shrink_refit.py search --system=net1b --trials=40
    python scripts/checks/156_shrink_refit.py judge --systems=net1b,creation110 --ref=creation110
"""
from __future__ import annotations

import os

THREADS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
for _name in THREADS:
    os.environ.setdefault(_name, "2")

import argparse  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from dataclasses import replace  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

TUNE_SEASONS = tuple(range(1998, 2026, 3))          # 1998, 2001, ..., 2025, check 152's
ALL_SEASONS = (1997, 2026)
GUARD_SCORED = tuple(sorted({season + step for season in TUNE_SEASONS for step in (-1, 1)
                             if ALL_SEASONS[0] <= season + step <= ALL_SEASONS[1]}))
ERAS = (("1997-2006", 1997, 2006), ("2007-2016", 2007, 2016), ("2017-2026", 2017, 2026))
NAMES = ("within_off", "within_def", "def_early", "def_late")
WITHIN_RANGE = (0.4, 1.3)
DEFENCE_RANGE = (0.6, 1.2)
GUARD_Z_LIMIT = 1.0                                 # check 152's: about z +1.7 on 63's fifty-six readings
TODAY = "zones139"                                  # the published ratings the guard is paired against


def _load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(file))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _wowy():
    return _load("wowy_check", "151_wowy.py")


def outputs_dir() -> Path:
    from openrapm.data.config import load_config, resolve
    return resolve(load_config(), "outputs")


def within_dir() -> Path:
    from openrapm.data.config import load_config, resolve
    return resolve(load_config(), "within")


def study_path(system: str) -> Path:
    return outputs_dir() / "tune" / f"refit156_{system}" / "study.db"


def shipped_settings() -> tuple:
    """(within offense, within defense, defense before 2017, defense from 2017) as shipped."""
    from openrapm.settings import PRODUCT, factor_for
    factors = PRODUCT.shrink.defence_factors
    return (*PRODUCT.shrink.within_factors, factor_for(factors, 2016), factor_for(factors, 2017))


def defence_pairs(settings: tuple) -> tuple:
    """The defense factors as settings.ShrinkSettings holds them: ((0, before 2017), (2017, from 2017))."""
    return ((0, float(settings[2])), (2017, float(settings[3])))


def shrink_flags(settings: tuple) -> str:
    """The `openrapm shrink` flags that give these settings."""
    return (f"--within-factors {settings[0]:.4f},{settings[1]:.4f} "
            f"--defence-factor {settings[2]:.4f},2017:{settings[3]:.4f}")


def raw(table: pd.DataFrame) -> pd.DataFrame:
    from openrapm.swap_test import ratings_in_raw_sign
    return ratings_in_raw_sign(table).assign(poss=table.poss_off.astype(float).to_numpy())


class Refit:
    """One system's step 4 fitted once with the four settings at 1, then multiplied by a trial's settings: its
    within-season WOWY error on the tune seasons, and its whole tune seasons for the next-season check."""

    def __init__(self, system: str):
        from openrapm.commands.build import step4_multipliers
        from openrapm.data.config import load_config
        from openrapm.data.designs import DesignCache
        from openrapm.data.roles import main_teams
        from openrapm.settings import PRODUCT
        started = time.time()
        self.cfg, self.system = load_config(), system
        self.wowy = _wowy()
        self.audit = self.wowy._audit()
        folds, where = self.wowy.system_folds(within_dir(), system)
        self.folds = [fold for fold in folds if fold.season in TUNE_SEASONS]     # check 151's order, so its sums
        self.teams = {fold.stem: self.wowy.team_arrays(where[fold.stem], fold.stem) for fold in self.folds}
        unit = replace(PRODUCT.shrink, rule="test", nearest=self.wowy.NEAREST, extrapolate_folds_tag="",
                       folds_tag=f"{system}_early,{system}_within", defence_factors=((0, 1.0),),
                       within_factors=(1.0, 1.0))
        self.team_layer = step4_multipliers(list(TUNE_SEASONS), folds, unit, None)
        layer = self.wowy.within_layer(folds, where, list(TUNE_SEASONS), ((0, 1.0),), (1.0, 1.0))
        self.multipliers = self.team_layer.merge(layer.reset_index(), on="season").set_index("season")
        self.designs = DesignCache(self.cfg, keep=1)
        self.year_designs = DesignCache(self.cfg, keep=4)
        step3 = pd.read_parquet(outputs_dir() / f"season_ratings_{system}_step3.parquet")
        self.step3 = step3[step3.season.isin(TUNE_SEASONS)].reset_index(drop=True)
        self.main_teams = main_teams(self.cfg)
        today = pd.read_parquet(outputs_dir() / f"season_ratings_{TODAY}.parquet")
        self.today_rows = self.year_rows("today", today[today.season.isin(TUNE_SEASONS)])
        print(f"{system}: {len(self.folds)} tune folds, step 4 fitted once ({time.time() - started:.0f}s)", flush=True)

    def factors(self, settings: tuple) -> pd.DataFrame:
        """The tune seasons' step-4 multipliers times the settings, as build.step4_multipliers and
        within_layer_multipliers apply them."""
        from openrapm.prior_shrink import WITHIN_COLUMNS
        within_off, within_def = float(settings[0]), float(settings[1])
        out = self.multipliers.copy()
        factor = np.where(out.index.to_numpy() < 2017, float(settings[2]), float(settings[3]))
        out["multiplier_def"] = out["multiplier_def"] * factor
        out["games_multiplier_def"] = out["games_multiplier_def"] * factor
        for column in WITHIN_COLUMNS[:2]:
            out[column] = out[column] * within_off
        for column in WITHIN_COLUMNS[2:]:
            out[column] = out[column] * (within_def * factor)
        return out

    def wowy_errors(self, settings: tuple) -> pd.DataFrame:
        """One row per tune season and slice: the step-4 WOWY error over the season's folds (check 151's
        `system_errors`, step4 stage)."""
        from openrapm.prior_shrink import WITHIN_COLUMNS
        by_season = self.factors(settings)
        sums: dict = {}
        for fold in self.folds:
            row = by_season.loc[fold.season]
            step4 = (float(row.multiplier_off), float(row.multiplier_def), float(row.games_multiplier_off),
                     float(row.games_multiplier_def), *(float(row[column]) for column in WITHIN_COLUMNS))
            errors = self.wowy.stint_fold_errors(self.audit, fold, self.teams[fold.stem],
                                                 self.designs.season(fold.season, "pts"), step4)
            for (stage, slice_), (top, bottom) in errors.items():
                if stage == "step4":
                    acc = sums.setdefault((fold.season, slice_), [0.0, 0.0])
                    acc[0] += top
                    acc[1] += bottom
        return pd.DataFrame([dict(season=season, slice=slice_, error=top / bottom)
                             for (season, slice_), (top, bottom) in sums.items()])

    def table(self, settings: tuple) -> pd.DataFrame:
        """The tune seasons' whole-season ratings after step 4 at these settings, as `openrapm shrink` writes them."""
        from openrapm.prior_shrink import shrink
        return shrink(self.step3, self.factors(settings).reset_index(), self.main_teams)

    def year_rows(self, name: str, table: pd.DataFrame) -> pd.DataFrame:
        """63's rows for `table`'s seasons predicting the seasons either side, rated seasons the tune seasons."""
        from openrapm.commands import year_over_year as yoy
        systems = yoy.neighbour_tables({name: {"prev": raw(table), "next": raw(table)}})
        rows = yoy.run(systems, self.year_designs, list(GUARD_SCORED), verbose=False)
        return rows[rows.train.astype(int).isin(TUNE_SEASONS)]

    def guard(self, settings: tuple) -> dict:
        """The next-season check against today's ratings: mean change in team-game error, its standard error, z and
        readings better, over the twenty readings."""
        from openrapm import scoring
        from openrapm.commands import year_over_year as yoy
        rows = pd.concat([self.today_rows, self.year_rows("trial", self.table(settings))], ignore_index=True)
        paired = scoring.paired(yoy.by_direction(rows)["both"], "today", "tg")
        line = paired[paired.system == "trial"].iloc[0]
        z = float(line.z) if np.isfinite(line.z) else 0.0
        return dict(mean=float(line.mean_diff), se=float(line.se), z=z, better=int(line.wins),
                    readings=int(line.n_seasons))


def check_fast_path(refit: Refit) -> None:
    """At the shipped settings: the fast path's errors equal check 151's `system_errors`, and its whole-season table
    equals the system's step-4 table on disk, both to the last digit; otherwise stop."""
    shipped = shipped_settings()
    mine = refit.wowy_errors(shipped).set_index(["season", "slice"]).error
    theirs = refit.wowy.system_errors(refit.audit, within_dir(), refit.system, ALL_SEASONS, only=set(TUNE_SEASONS))
    theirs = theirs[theirs.stage == "step4"].set_index(["season", "slice"]).error
    gap = float((mine - theirs.reindex(mine.index)).abs().max())
    table = refit.table(shipped)
    disk = pd.read_parquet(outputs_dir() / f"season_ratings_{refit.system}_step4.parquet")
    joined = table.merge(disk, on=["player_id", "season"], suffixes=("", "_disk"))
    table_gap = max(float((joined[column] - joined[f"{column}_disk"]).abs().max())
                    for column in ("rating_off", "rating_def", "prior_off", "u_off", "prior_def", "u_def"))
    print(f"fast path at the shipped settings: WOWY {gap:.1e} from check 151, table {table_gap:.1e} from "
          f"season_ratings_{refit.system}_step4 over {len(joined):,} of {len(table):,} rows", flush=True)
    if gap > 1e-9 or table_gap > 1e-9 or len(joined) != len(table):
        raise SystemExit("the fast path does not reproduce check 151 and the step-4 table; stopped")


def run_search(args) -> int:
    import optuna
    refit = Refit(args.system)
    check_fast_path(refit)
    path = study_path(args.system)
    path.parent.mkdir(parents=True, exist_ok=True)
    sampler = optuna.samplers.GPSampler(seed=args.seed, n_startup_trials=args.startup, deterministic_objective=True,
                                        constraints_func=lambda frozen: frozen.user_attrs.get("constraint", [1.0]))
    study = optuna.create_study(study_name="refit", storage=f"sqlite:///{path.as_posix()}", direction="minimize",
                                sampler=sampler, load_if_exists=True)
    study.enqueue_trial(dict(zip(NAMES, shipped_settings())), skip_if_exists=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial) -> float:
        started = time.time()
        settings = (trial.suggest_float("within_off", *WITHIN_RANGE), trial.suggest_float("within_def", *WITHIN_RANGE),
                    trial.suggest_float("def_early", *DEFENCE_RANGE), trial.suggest_float("def_late", *DEFENCE_RANGE))
        errors = refit.wowy_errors(settings)
        every = errors[errors.slice == "all"].set_index("season").error
        guard = refit.guard(settings)
        trial.set_user_attr("constraint", [guard["z"] - GUARD_Z_LIMIT])
        trial.set_user_attr("errors", {str(season): float(value) for season, value in every.items()})
        trial.set_user_attr("guard", guard)
        print(f"  trial {trial.number:3d}  {' '.join(f'{v:.3f}' for v in settings)}  WOWY {every.mean():.4f}  "
              f"next season {guard['mean']:+.4f} z {guard['z']:+.2f}"
              f"{'' if guard['z'] <= GUARD_Z_LIMIT else ', FAILS the limit'}  ({time.time() - started:.0f}s)",
              flush=True)
        return float(every.mean())

    done = sum(1 for trial in study.trials if trial.state.is_finished())
    study.optimize(objective, n_trials=max(args.trials - done, 0))
    best = best_trial(args.system)
    print(f"best that keeps the limit: trial {best['number']}, WOWY {best['value']:.4f}, "
          f"{shrink_flags(best['settings'])}", flush=True)
    return 0


def best_trial(system: str) -> dict:
    """The study's best finished trial that kept the limit: number, value, settings, guard."""
    import optuna
    study = optuna.load_study(study_name="refit", storage=f"sqlite:///{study_path(system).as_posix()}")
    kept = [trial for trial in study.trials if trial.state.is_finished() and trial.value is not None
            and trial.user_attrs.get("constraint", [1.0])[0] <= 0]
    if not kept:
        raise SystemExit(f"{system}: no finished trial kept the limit")
    trial = min(kept, key=lambda t: t.value)
    return dict(number=trial.number, value=trial.value, settings=tuple(trial.params[name] for name in NAMES),
                guard=trial.user_attrs["guard"])


def paired_rows(errors: pd.DataFrame, system: str, ref: str, label: str) -> list:
    """`system` minus `ref` season by season on the step-4 error, every held-back stint: the twenty untouched seasons,
    each era's untouched ones, the ten tune seasons and all thirty."""
    table = errors[(errors.stage == "step4") & (errors.slice == "all")].pivot(index="season", columns="system",
                                                                              values="error")
    diff = (table[system] - table[ref]).dropna()
    untouched = ~diff.index.isin(TUNE_SEASONS)
    groups = [("untouched 20", untouched), *((f"untouched {era}", untouched & (diff.index >= first) &
                                              (diff.index <= last)) for era, first, last in ERAS),
              ("tune 10", ~untouched), ("all 30", np.ones(len(diff), dtype=bool))]
    rows = []
    for group, mask in groups:
        part = diff[mask]
        se = part.std(ddof=1) / np.sqrt(len(part)) if len(part) > 1 else np.nan
        rows.append(dict(comparison=label, seasons=group, change=part.mean(), se=se, z=part.mean() / se,
                         better=int((part < 0).sum()), n=len(part)))
    return rows


def run_judge(args) -> int:
    from openrapm.commands.console import use_utf8
    use_utf8()
    wowy = _wowy()
    audit = wowy._audit()
    systems = [name for name in args.systems.split(",") if name]
    frames, settings = [], {}
    for system in systems:
        best = best_trial(system)
        settings[system] = best["settings"]
        print(f"{system}: best trial {best['number']}, {shrink_flags(best['settings'])}; next season "
              f"{best['guard']['mean']:+.4f} z {best['guard']['z']:+.2f}", flush=True)
        within_factors = tuple(best["settings"][:2])
        frames.append(wowy.system_errors(audit, within_dir(), system, ALL_SEASONS, label=f"{system}@best",
                                         defence_factors=defence_pairs(best["settings"]),
                                         within_factors=within_factors))
        frames.append(wowy.system_errors(audit, within_dir(), system, ALL_SEASONS, label=f"{system}@shipped"))
    errors = pd.concat(frames, ignore_index=True)
    out = outputs_dir() / "csv" / f"refit156_{'_'.join(systems)}_seasons.csv"
    errors.to_csv(out, index=False)
    ref = args.ref
    rows = []
    for system in systems:
        if system == ref:
            continue
        rows += paired_rows(errors, f"{system}@best", f"{ref}@best", f"{system} at its best against {ref} at its best")
        rows += paired_rows(errors, f"{system}@best", f"{ref}@shipped", f"{system} at its best against {ref} shipped")
        rows += paired_rows(errors, f"{system}@shipped", f"{ref}@shipped", f"{system} shipped against {ref} shipped")
    rows += paired_rows(errors, f"{ref}@best", f"{ref}@shipped", f"{ref} at its best against {ref} shipped")
    summary = pd.DataFrame(rows)
    summary.to_csv(out.with_name(out.name.replace("_seasons", "")), index=False)
    pd.set_option("display.width", 200)
    print(summary.round(4).to_string(index=False))
    print(json.dumps({system: shrink_flags(value) for system, value in settings.items()}, indent=1))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    search = commands.add_parser("search")
    search.add_argument("--system", required=True)
    search.add_argument("--trials", type=int, default=40)
    search.add_argument("--startup", type=int, default=10)
    search.add_argument("--seed", type=int, default=0)
    search.set_defaults(run=run_search)
    judge = commands.add_parser("judge")
    judge.add_argument("--systems", required=True)
    judge.add_argument("--ref", required=True)
    judge.set_defaults(run=run_judge)
    args = parser.parse_args(argv)
    return args.run(args)


if __name__ == "__main__":
    raise SystemExit(main())
