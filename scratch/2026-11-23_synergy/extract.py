"""Lineup synergies, the extract: every held-back stint of the incumbent's within-season test, saved once per season
with both fives, so any lineup statistic can be scored afterwards without rebuilding the season.

The owner, 2026-11-23: "what are the 'statistical synergies' that improve lineups on offense AND on defense? i think we
have shown that self-creation helps! ... what else? look at everything".  Read-only; no rating moves.

The stints are check 154's: each held-back stint of every fold, its miss (points per 100 minus the ten players'
ratings after the shipped shrink, both layers and the defense factor, check 151's stand-in for an unrated player) and
its free terms.  Check 154's own frame carries the offense's five only; this one also keeps the defense's five, the
defense's playing time and the spread of the five defensive ratings.  Beside it, each fold's players with their padded
rates from the fold's fit games, the inputs the fold's box-score prior reads, so no held-back game is in them.

    .venv/Scripts/python scratch/2026-11-23_synergy/extract.py --folds=rd143 --out=outputs/synergy/rd143

Writes OUT/stints_<season>.parquet and OUT/players_<season>.parquet; a season already written is skipped, so a
stopped run resumes.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from openrapm.commands.console import one_thread, use_utf8

CHECKS = Path(__file__).resolve().parents[2] / "scripts" / "checks"

# Box-score and play-by-play rates a fold rebuilds from its own fit games, and height.  No tracking or hustle column:
# those are season totals a fold only rescales, and they leak the held-back games (DECISIONS.md, measurement traps).
TRAITS = (
    "self_creation_per100", "used_possessions_per100", "true_shooting_pct", "assists_per100_uncentred",
    "turnovers_per100_uncentred", "three_point_attempts_per100", "threes_made_per100_uncentred", "three_point_pct",
    "free_throw_attempts_per100", "off_rebounds_per100_uncentred", "def_rebounds_per100_uncentred",
    "blocks_per100_uncentred", "steals_per100_uncentred", "fouls_per100_uncentred", "height_in", "weight_lb",
    "unassisted_points_per100_uncentred", "assisted_points_per100", "transition_attempts_per100_uncentred",
    "putback_attempts_per100_uncentred", "assists_to_rim_per100", "rim_blocks_per100", "points_per100",
)
KEEP = ("player_id", "team_id", "poss_off", "poss_def", "team_possession_share")


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def fold_stints(m154, wowy, audit, fold, teams: dict, design, step4: tuple) -> pd.DataFrame:
    """One fold's held-back stints: check 154's frame plus the defense's five, its playing time and rating spread."""
    rows = m154.held_back_rows(teams, design)
    index = rows["index"].to_numpy()
    lineup_o, lineup_d = m154.lineup_ids(design, index)
    rating_off, allowed = wowy.stage_ratings(audit, fold, step4)["step4"]
    off, missing_o = m154.mapped(lineup_o, rating_off, fold.fill_o)
    dfn, missing_d = m154.mapped(lineup_d, allowed, fold.fill_d)
    share_o = dict(zip(fold.rows["O"].player_id, fold.rows["O"].team_possession_share))
    share_d = dict(zip(fold.rows["D"].player_id, fold.rows["D"].team_possession_share))
    frame = pd.DataFrame({
        "repeat": np.int8(fold.repeat), "fold": np.int8(fold.fold), "game_idx": rows.game_idx.to_numpy(np.int32),
        "team_off": rows.team_off.to_numpy(), "team_def": rows.team_def.to_numpy(),
        "is_home_off": rows.is_home_off.to_numpy(np.float32), "playoffs": (rows.phase == "PO").to_numpy(np.float32),
        "y": design.y[index], "w": design.w[index],
        "rating_sum_off": off.sum(axis=1), "rating_sum_def": dfn.sum(axis=1),
        "talent_spread_off": off.std(axis=1), "talent_spread_def": dfn.std(axis=1),
        "playing_time": m154.mapped(lineup_o, share_o, 0.0)[0].sum(axis=1),
        "playing_time_def": m154.mapped(lineup_d, share_d, 0.0)[0].sum(axis=1),
        "stand_ins": (missing_o.sum(axis=1) + missing_d.sum(axis=1)).astype(float),
        **m154.context_columns(design, index)})
    for k in range(5):
        frame[f"o{k + 1}"] = lineup_o[:, k].astype(np.int64)
        frame[f"d{k + 1}"] = lineup_d[:, k].astype(np.int64)
    frame = frame[frame.w > 0].reset_index(drop=True)
    for column in frame.columns:
        if frame[column].dtype == np.float64:
            frame[column] = frame[column].astype(np.float32)
    return frame


def fold_players(fold) -> pd.DataFrame:
    """The fold's players, both sides, with the traits from the fold's fit games."""
    parts = []
    for side in ("O", "D"):
        rows = fold.rows[side]
        part = rows[[c for c in (*KEEP, *TRAITS) if c in rows.columns]].copy()
        part.insert(0, "side", side)
        parts.append(part)
    out = pd.concat(parts, ignore_index=True)
    out.insert(0, "fold", np.int8(fold.fold))
    out.insert(0, "repeat", np.int8(fold.repeat))
    return out


def main(argv=None) -> int:
    use_utf8()
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--folds", default="rd143")
    parser.add_argument("--out", default="outputs/synergy/rd143")
    parser.add_argument("--first", type=int, default=1997)
    parser.add_argument("--last", type=int, default=2026)
    args = parser.parse_args(argv)
    m154 = _load("selfcreation_check", CHECKS / "154_self_creation.py")
    wowy = m154._load("wowy_check", "151_wowy.py")
    from openrapm.data.config import load_config, resolve
    from openrapm.data.designs import DesignCache
    cfg = load_config()
    outputs = resolve(cfg, "outputs")
    within = resolve(cfg, "within") if "within" in cfg.get("paths", {}) else outputs / "within"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    spec = argparse.Namespace(folds=args.folds, first=args.first, last=args.last, step4_from="", trade=0, null=0,
                              save_rows="")
    started = time.time()
    with one_thread():
        inputs = m154.run_inputs(spec, cfg, within, wowy)
        cache = DesignCache(cfg, keep=1)
        for season in sorted({fold.season for fold in inputs.folds}):
            if (out / f"stints_{season}.parquet").exists() and (out / f"players_{season}.parquet").exists():
                print(f"  {season}: on disk", flush=True)
                continue
            design = cache.season(season, "pts")
            stints, players = [], []
            for fold in (f for f in inputs.folds if f.season == season):
                teams = wowy.team_arrays(inputs.where[fold.stem], fold.stem)
                step4 = m154.step4_of(fold, inputs.multipliers, inputs.layer)
                stints.append(fold_stints(m154, wowy, inputs.audit, fold, teams, design, step4))
                players.append(fold_players(fold))
            pd.concat(players, ignore_index=True).to_parquet(out / f"players_{season}.parquet", index=False)
            pd.concat(stints, ignore_index=True).to_parquet(out / f"stints_{season}.parquet", index=False)
            print(f"  {season}: {sum(len(s) for s in stints):,} stints ({time.time() - started:.0f}s)", flush=True)
    print(f"done in {time.time() - started:.0f}s: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
