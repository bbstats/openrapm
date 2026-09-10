"""The layer boundary: model code may not load its own data.

The project has two layers and one rule between them.

    DATA   reads the disk and the network.  Scraping, stints, box tables, caches, loaders.
           A contributor may add a source here -- tracking data, a different feed, anything.
           This is where a maintainer looks.

    MODEL  gets handed data and returns numbers.  The ridge, the box prior, the role prior,
           the padding, the design assembly, the calibration.  This is where a pull request
           that improves the model belongs, and it may not open a file.

The rule exists because the season boundary (src/eracoef/seasons.py) is only as good as the
narrowest channel through which data can reach a fit.  If a model module may call `read_parquet`,
it can reach past every guard and read the season in progress directly, and no amount of care in
the loaders would stop it.  Keeping the model layer I/O-free makes the loaders the only way in.

`MODEL_LAYER` below is the list of modules that are I/O-free today and must stay that way.  It is
a ratchet, not an aspiration: a module joins it when it is clean and never leaves.  `NEEDS_SPLIT`
is the honest remainder -- model code that still loads its own tables and should be separated.
"""
import ast
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "eracoef"
sys.path.insert(0, str(SRC.parents[1]))

# Modules that are handed their data and must never fetch it.  Verified I/O-free.
MODEL_LAYER = [
    "boxtable.py",      # box-score frames from tables already in memory
    "cv.py",            # pipelines, cross-fitted beta, the plug-in fit
    "design.py",        # the sparse design: players, fixed effects, box exposures
    "estimator.py",     # the mixed model (Henderson), the eigen lambda path
    "exposure.py",      # cross-fitted, empirical-Bayes padded per-100 rates
    "fastfit.py",       # the shipped one-pass fit
    "gbdt_prior.py",    # the boosted box prior
    "inseason.py",      # the rolling kernel and the game cut
    "investigate.py",   # the attribution instrument
    "pad.py",           # the padding helper
    "simulate.py",      # the synthetic fixture the tests fit against
    "spm.py",           # APM, the role prior, the chain's offset
]

# Model code that still reads its own tables.  Each one is a place the boundary is not yet real.
# Shrinking this list is the work; nothing may be added to it.
NEEDS_SPLIT = {
    "calmap.py": "loads its own parameter table; fitting and applying should take a frame",
    "systems.py": "the `panel=` override reads a parquet instead of taking one",
}

# The verbs that reach the disk or the network.  `.get` and `.exists` are deliberately absent:
# they are overwhelmingly `dict.get` and would drown the rule in false positives.
IO_CALLS = {"read_parquet", "read_csv", "read_json", "to_parquet", "to_csv", "to_json",
            "open", "glob", "rglob", "iterdir", "read_text", "write_text", "read_bytes",
            "write_bytes", "urlopen", "urlretrieve", "request"}
IO_MODULES = {"requests", "urllib", "nba_api", "httpx", "socket"}


def _io_in(path: Path):
    """(line, what) for every disk or network reach in a module."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    out = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            f = n.func
            name = f.attr if isinstance(f, ast.Attribute) else (f.id if isinstance(f, ast.Name) else None)
            if name in IO_CALLS:
                out.append((n.lineno, f"{name}()"))
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            mod = n.module if isinstance(n, ast.ImportFrom) else None
            names = [mod] if mod else [a.name for a in n.names]
            for nm in names:
                if nm and nm.split(".")[0] in IO_MODULES:
                    out.append((n.lineno, f"import {nm}"))
    return out


@pytest.mark.parametrize("module", MODEL_LAYER)
def test_a_model_module_does_no_io(module):
    path = SRC / module
    assert path.exists(), f"{module} is in MODEL_LAYER but does not exist -- update the list"
    found = _io_in(path)
    assert not found, (
        f"src/eracoef/{module} is model-layer code and may not reach the disk or the network, but "
        f"it does at {found}.  Take the data as an argument instead; the loader belongs in the data "
        f"layer.  See the docstring of this file.")


def test_the_model_layer_and_the_split_list_do_not_overlap():
    assert not set(MODEL_LAYER) & set(NEEDS_SPLIT), "a module cannot be both clean and needing a split"


def test_every_module_is_classified():
    """A new module must be placed in one layer or the other, so nothing lands here unnoticed."""
    data_layer = {"__init__.py", "config.py", "seasons.py", "ingest.py", "stints.py", "roles.py",
                  "bio.py", "shotcurve.py", "designcache.py", "turnover.py", "xshoot.py",
                  "windows.py", "holdout.py", "context.py", "checks.py", "factors.py", "xpts.py",
                  "teamloo.py"}
    known = set(MODEL_LAYER) | set(NEEDS_SPLIT) | data_layer
    actual = {p.name for p in SRC.glob("*.py")}
    unclassified = actual - known
    assert not unclassified, (
        f"new module(s) {sorted(unclassified)}: add each to MODEL_LAYER (if it takes its data as an "
        f"argument) or to the data-layer set in this test (if it loads data).  See the docstring.")
