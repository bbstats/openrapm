"""The second prior's files on disk: the normal equations its line is fitted on, one file per season.

`openrapm second-prior --tag TAG` writes them under outputs/second_prior/<TAG>/ (paths.outputs); `openrapm rate
--second-prior TAG` and `openrapm folds` read them back (second_prior.py, the line).

    grams_<season>.npz        per side ("offense", "defense"): G, b, G_fold, b_fold and n of swap_adjust.PairGrams,
                              summed over the season's within-season folds
    inputs_<season>.parquet   every fold player's inputs: season, repeat, fold, side, player_id and
                              second_prior.INPUTS
    run.json                  the command that wrote them and the folds and boosters it read

Example: outputs/second_prior/ctx81/grams_2019.npz holds 2019's share of both lines, from its twelve folds.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..swap_adjust import PairGrams
from .config import resolve

__all__ = ["SIDES", "grams_directory", "grams_path", "write_grams", "read_grams", "load_grams", "write_run",
           "read_run", "line_setup"]

SIDES = ("offense", "defense")
FIELDS = ("G", "b", "G_fold", "b_fold", "n")


def grams_directory(cfg: dict, tag: str) -> Path:
    """outputs/second_prior/<tag>/."""
    return resolve(cfg, "outputs") / "second_prior" / tag


def grams_path(directory: Path, season: int) -> Path:
    return Path(directory) / f"grams_{int(season)}.npz"


def write_grams(path: Path, grams: dict) -> None:
    """One season's {side: PairGrams} to `path`."""
    np.savez(path, **{f"{side}_{field}": np.asarray(getattr(grams[side], field)) for side in SIDES
                      for field in FIELDS})


def read_grams(path: Path) -> dict:
    """{side: PairGrams} from one season's file."""
    stored = np.load(path)
    return {side: PairGrams(G=stored[f"{side}_G"], b=stored[f"{side}_b"], G_fold=stored[f"{side}_G_fold"],
                            b_fold=stored[f"{side}_b_fold"], n=int(stored[f"{side}_n"])) for side in SIDES}


def load_grams(cfg: dict, tag: str, padded: bool | None = True) -> dict:
    """{season: {side: PairGrams}} for every season under outputs/second_prior/<tag>/; stops, naming the command
    that writes them, when there are none, and when run.json does not say the context numbers were padded as
    `padded` asks (None: either), so a line is never applied to inputs of another kind than it was fitted on."""
    directory = grams_directory(cfg, tag)
    paths = sorted(directory.glob("grams_*.npz"))
    if not paths:
        raise SystemExit(f"no second-prior grams in {directory}: `openrapm second-prior --tag {tag}` writes them")
    if padded is not None and bool(read_run(directory).get("padded", False)) != padded:
        raise SystemExit(f"{directory}/run.json does not say padded={padded}: these grams were fitted on context "
                         "numbers of another kind; write them again with `openrapm second-prior`")
    return {int(path.stem.removeprefix("grams_")): read_grams(path) for path in paths}


def write_run(directory: Path, record: dict) -> None:
    (Path(directory) / "run.json").write_text(json.dumps(record, indent=1), encoding="utf-8")


def line_setup(cfg: dict, tag: str, inputs: list) -> dict:
    """run.json of outputs/second_prior/<tag>/, after checking its line read `inputs` in that order (a line applied
    to other inputs than it was fitted on would read each weight on the wrong number); it carries `type_scale` when
    the line reads step 5's numbers."""
    record = read_run(grams_directory(cfg, tag))
    if list(record.get("inputs", [])) != list(inputs):
        raise SystemExit(f"outputs/second_prior/{tag} was fitted on the inputs {record.get('inputs')}, not {inputs}")
    return record


def read_run(directory: Path) -> dict:
    path = Path(directory) / "run.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
