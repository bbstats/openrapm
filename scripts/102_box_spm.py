"""B1, experiment 35's first baseline: a linear box-score-only SPM, rated on full seasons and on every within-season fold.

    python scripts/102_box_spm.py [--tags=q1of4,q1of3,within2,q2of3,within,within10] [--first=1997] [--last=2026]

For each rated season H (src/eracoef/boxspm.py):
  * the label is the same family the shipped prior learns from: a leave-season-out RAPM over every season except
    {H-1, H, H+1} (looseason.LeaveSeasonOutRAPM, the offensive and defensive targets and penalties of
    singleyear.py), the defensive label put back on one scale (62's `unshrink_label`, the shipped `--unshrink_label=def`);
  * the training rows are every player-season outside {H-1, H, H+1}: box-score inputs only, weighted by box
    possessions; padding constants are the mean over those training seasons;
  * the rating is the fit applied to H's box score -- all of H for the year-over-year table, and only the fold's
    rating games for each within-season fold -- centered on possessions.  League rates come from the same games.

Writes outputs/season_ratings_base_b1.parquet (63's format), outputs/within/<tag>/baseline_b1.parquet (one row per
fold and player, raw sign: `o` adds points scored, `d` adds points allowed), outputs/csv/baseline_b1_coefficients.csv
and outputs/within/baseline_b1_checks.parquet (a failed check stops the run).
"""
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _cli import check_flags, flag  # noqa: E402
from eracoef import boxspm as bx  # noqa: E402
from eracoef import singleyear as sy  # noqa: E402
from eracoef.boxtable import season_box  # noqa: E402
from eracoef.config import load_config  # noqa: E402
from eracoef.exposure import K_MAX  # noqa: E402
from eracoef.holdout import Context  # noqa: E402
from eracoef.looseason import LeaveSeasonOutRAPM  # noqa: E402

PHASES = ("RS", "PO")
UNSHRINK_FLOOR = 4444.0          # 62's --unshrink_floor default


def _borrow(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


B62 = _borrow("_board62_for_b1", "62_single_year_board.py")      # `_target`, `unshrink_label`


def labels(rapm: dict, unseen: list) -> pd.DataFrame:
    """player_id, offense, defense (raw sign), label possessions: the shipped prior's label family, unseen left out."""
    kw = dict(held_out_season=unseen, offense_lambda=sy.RAPM_OFFENSE_LAMBDA, defense_lambda=sy.RAPM_DEFENSE_LAMBDA,
              context_lambda=sy.RAPM_CONTEXT_LAMBDA)
    o = rapm["O"].ratings(**kw)[["player_id", "offense", "possessions"]]
    d = rapm["D"].ratings(**kw)
    d = B62.unshrink_label(d, sy.RAPM_OFFENSE_LAMBDA, sy.RAPM_DEFENSE_LAMBDA, UNSHRINK_FLOOR, sides=("defense",))
    return o.merge(d[["player_id", "defense"]], on="player_id", how="inner")


def fold_fit_games(directory: Path, stem: str, assignment: pd.DataFrame) -> tuple:
    """(fit game ids, rate_from) for one fold, from the fold's json and the season's deal table."""
    summary = json.loads((directory / f"fold_{stem}.json").read_text())["summary"]
    mode = summary.get("rate_from")
    if mode is None or (isinstance(mode, float) and np.isnan(mode)):
        mode = "rest"
    col = assignment[f"r{int(summary['repeat'])}"]
    k = int(summary["fold"])
    fit = col.index[(col != k) if mode == "rest" else (col == k)]
    return set(map(str, fit)), mode, int(summary["season"])


def rate(model_o, model_d, inputs: pd.DataFrame) -> pd.DataFrame:
    out = inputs[["player_id", "poss"]].copy()
    out["o"] = bx.centered(model_o.predict(inputs), inputs.poss.to_numpy(float))
    out["d"] = bx.centered(model_d.predict(inputs), inputs.poss.to_numpy(float))
    return out


def main():
    check_flags()
    tags = [t for t in flag("tags", "q1of4,q1of3,within2,q2of3,within,within10").split(",") if t]
    first, last = int(flag("first", "1997")), int(flag("last", "2026"))
    seasons = list(range(first, last + 1))
    cfg = load_config(ROOT / "config.yaml")
    ctx = Context.load(cfg)
    t0 = time.time()

    box = {s: bx.with_possessions(season_box([s], PHASES, cfg)) for s in seasons}
    names = box[seasons[-1]].drop_duplicates("player_id", keep="last").set_index("player_id").player_name
    k_season = {s: bx.padding_k(box[s]) for s in seasons}
    print(f"box scores and padding constants for {len(seasons)} seasons ({time.time() - t0:.0f}s)", flush=True)

    rapm = {}
    for side, target in (("O", sy.OFFENSE_TARGET), ("D", sy.DEFENSE_TARGET)):
        rapm[side] = LeaveSeasonOutRAPM(min_possessions=sy.MIN_POSSESSIONS)
        for s in seasons:
            rapm[side].add_season(s, ctx.design([s], B62._target(target)))
        print(f"  accumulated {target} ({time.time() - t0:.0f}s)", flush=True)

    tables, coefs, fold_rows, checks = [], [], {t: [] for t in tags}, []
    deals = {}
    for H in seasons:
        unseen = [H - 1, H, H + 1]
        train_seasons = [s for s in seasons if s not in unseen]
        k = np.nanmean(np.array([k_season[s] for s in train_seasons]), axis=0)
        k = np.where(np.isfinite(k), k, K_MAX)
        lab = labels(rapm, unseen)
        rows = []
        for s in train_seasons:
            x = bx.player_inputs(box[s], k).merge(lab, on="player_id", how="inner")
            rows.append(x)
        train = pd.concat(rows, ignore_index=True)
        w = train.poss.to_numpy(float)
        model_o = bx.LinearBoxSPM().fit(train, train.offense.to_numpy(float), w)
        model_d = bx.LinearBoxSPM().fit(train, train.defense.to_numpy(float), w)
        coefs.append(pd.concat({"O": model_o.coefficients(), "D": model_d.coefficients()}, axis=1)
                     .assign(season=H).reset_index(names="input"))

        whole = rate(model_o, model_d, bx.player_inputs(box[H], k))
        tables.append(whole.assign(season=H))
        checks.append(dict(scope="season", key=str(H), check="label excludes the rated season and its neighbours",
                           value=float(set(unseen).isdisjoint(train_seasons)), passed=set(unseen).isdisjoint(train_seasons)))

        for tag in tags:
            directory = ROOT / "outputs" / "within" / tag
            if not (directory / f"folds_{H}.parquet").exists():
                continue
            if (tag, H) not in deals:
                deals[(tag, H)] = pd.read_parquet(directory / f"folds_{H}.parquet")
            assignment = deals[(tag, H)]
            assignment.index = assignment.index.astype(str)
            for stem in sorted(p.stem[len("fold_"):] for p in directory.glob(f"fold_{H}_*.json")):
                fit_ids, mode, season = fold_fit_games(directory, stem, assignment)
                assert season == H
                sub = box[H][box[H].game_id.isin(fit_ids)]
                found = set(sub.game_id.unique())
                coverage = len(found) / max(len(fit_ids), 1)
                ok_rows = found <= fit_ids
                checks.append(dict(scope=tag, key=stem, check="box rows only from fit games", value=float(ok_rows),
                                   passed=bool(ok_rows)))
                checks.append(dict(scope=tag, key=stem, check="fit games found in the game log", value=coverage,
                                   passed=bool(coverage > 0.99)))
                r = rate(model_o, model_d, bx.player_inputs(sub, k))
                fold_rows[tag].append(r.assign(key=stem, season=H))
        print(f"  {H}: {len(train)} training player-seasons; rated {len(whole)} ({time.time() - t0:.0f}s)", flush=True)

    checks = pd.DataFrame(checks)
    (ROOT / "outputs" / "within").mkdir(exist_ok=True)
    checks.to_parquet(ROOT / "outputs" / "within" / "baseline_b1_checks.parquet", index=False)
    failed = checks[~checks.passed]
    assert failed.empty, f"{len(failed)} checks failed:\n{failed.head(10)}"

    t = pd.concat(tables, ignore_index=True)
    out = pd.DataFrame({"player_id": t.player_id.astype(np.int64), "season": t.season.astype(int),
                        "player_name": t.player_id.map(names), "rating_off": t.o, "rating_def": -t.d,
                        "poss_off": t.poss, "poss_def": t.poss})
    out["rating_total"] = out.rating_off + out.rating_def
    out.to_parquet(ROOT / "outputs" / "season_ratings_base_b1.parquet", index=False)
    for tag, parts in fold_rows.items():
        if parts:
            pd.concat(parts, ignore_index=True).to_parquet(ROOT / "outputs" / "within" / tag / "baseline_b1.parquet",
                                                           index=False)
    pd.concat(coefs, ignore_index=True).to_csv(ROOT / "outputs" / "csv" / "baseline_b1_coefficients.csv", index=False)
    print(f"wrote season_ratings_base_b1 ({len(out)} rows) and {sum(len(v) for v in fold_rows.values())} fold tables; "
          f"{len(checks)} checks, 0 failed ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
