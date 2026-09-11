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

Two rules, because the first alone is not enough
------------------------------------------------
Checking for `read_parquet` in the module itself misses the interesting case.  `boxtable.py` calls
no I/O verb at all, and `boxtable.season_box` still reaches stats.nba.com -- it calls
`ingest.load_gamelog`, which SCRAPES when the cache is cold.  That is how the test suite was
quietly downloading three game logs on a "fresh clone with no data" run.  So a model module must
also not IMPORT a data-layer module.  That is rule two here; the network itself is blocked in
conftest.py so no test can silently fetch what it is missing.

`MODEL_LAYER` is the list of modules that pass both rules today and must keep passing.  It is a
ratchet: a module joins it when it is clean and never leaves.  `NEEDS_SPLIT` is the honest
remainder -- model code that still reaches its own data, with what it reaches through.  Shrinking
that list is the work.
"""
import ast
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "eracoef"
sys.path.insert(0, str(SRC.parents[1]))

# Clean on both rules: no I/O verb, no data-layer import.
MODEL_LAYER = [
    "archetype.py",     # the box-profile mixture and the per-cluster gap
    "cv.py",            # pipelines, cross-fitted beta, the plug-in fit
    "estimator.py",     # the mixed model (Henderson), the eigen lambda path
    "exposure.py",      # cross-fitted, empirical-Bayes padded per-100 rates
    "investigate.py",   # the attribution instrument
    "pad.py",           # the padding helper
    "pbo.py",           # the probability of backtest overfitting
    "rloocv.py",        # rebalanced leave-one-out: the splitter and the partner rule, numpy only
    "simulate.py",      # the synthetic fixture the tests fit against
]

# Model code that still reaches its own data, and what through.  Nothing may be added here.
NEEDS_SPLIT = {
    "boxtable.py": "season_box -> ingest.load_gamelog, which SCRAPES when the cache is cold",
    "design.py": "imports stints to load them; should take the frame",
    "fastfit.py": "imports holdout for Context, which owns every loader",
    "gbdt_prior.py": "imports context, roles, windows for prediction-time features",
    "inseason.py": "imports windows for the block list",
    "spm.py": "imports bio, context, roles, turnover, xshoot to rebuild features at predict time",
    "calmap.py": "loads its own parameter table; fitting and applying should take a frame",
    "systems.py": "the `panel=` override reads a parquet instead of taking one",
}

# Modules whose job IS to reach the disk or the network.
DATA_LAYER = {
    "__init__.py", "config.py", "seasons.py", "ingest.py", "stints.py", "roles.py", "bio.py",
    "shotcurve.py", "designcache.py", "turnover.py", "xshoot.py", "windows.py", "holdout.py",
    "context.py", "checks.py", "factors.py", "xpts.py", "teamloo.py",
}

# The verbs that reach the disk or the network.  `.get` and `.exists` are deliberately absent:
# they are overwhelmingly `dict.get` and would drown the rule in false positives.
IO_CALLS = {"read_parquet", "read_csv", "read_json", "to_parquet", "to_csv", "to_json",
            "open", "glob", "rglob", "iterdir", "read_text", "write_text", "read_bytes",
            "write_bytes", "urlopen", "urlretrieve", "request"}
IO_MODULES = {"requests", "urllib", "nba_api", "httpx", "socket"}


def _parse(path: Path):
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _io_in(path: Path):
    """(line, what) for every direct disk or network reach in a module."""
    out = []
    for n in ast.walk(_parse(path)):
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


def _sibling_imports(path: Path):
    """(line, module) for every import of another module in this package."""
    out = []
    for n in ast.walk(_parse(path)):
        if isinstance(n, ast.ImportFrom):
            if n.level and n.module:                       # from .roles import ...
                out.append((n.lineno, n.module.split(".")[0]))
            elif n.level and not n.module:                 # from . import roles
                out.extend((n.lineno, a.name) for a in n.names)
            elif n.module and n.module.startswith("eracoef"):
                out.append((n.lineno, n.module.split(".")[-1]))
        elif isinstance(n, ast.Import):
            out.extend((n.lineno, a.name.split(".")[-1]) for a in n.names
                       if a.name.startswith("eracoef."))
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


@pytest.mark.parametrize("module", MODEL_LAYER)
def test_a_model_module_does_not_import_the_data_layer(module):
    """The rule that catches the interesting case.  boxtable.py calls no I/O verb and still scrapes,
    because it calls ingest.load_gamelog.  Reaching data through a sibling is still reaching data."""
    bad = [(ln, m) for ln, m in _sibling_imports(SRC / module) if f"{m}.py" in DATA_LAYER]
    assert not bad, (
        f"src/eracoef/{module} imports data-layer module(s) {sorted({m for _, m in bad})} at lines "
        f"{[ln for ln, _ in bad]}.  Whatever it needs from them should be passed in.")


def test_the_lists_do_not_overlap():
    assert not set(MODEL_LAYER) & set(NEEDS_SPLIT), "a module cannot be both clean and needing a split"
    assert not set(MODEL_LAYER) & DATA_LAYER, "a module cannot be both model-layer and data-layer"


def test_every_module_is_classified():
    """A new module must be placed in one layer or the other, so nothing lands here unnoticed."""
    known = set(MODEL_LAYER) | set(NEEDS_SPLIT) | DATA_LAYER
    unclassified = {p.name for p in SRC.glob("*.py")} - known
    assert not unclassified, (
        f"new module(s) {sorted(unclassified)}: add each to MODEL_LAYER (if it takes its data as an "
        f"argument and imports no loader), to NEEDS_SPLIT (if it is model code that still loads), or "
        f"to DATA_LAYER (if loading is its job).  See the docstring.")


def test_the_split_list_is_accurate():
    """Every module claimed to need a split must actually reach data -- otherwise it belongs in
    MODEL_LAYER and the ratchet should have moved."""
    for module in NEEDS_SPLIT:
        path = SRC / module
        if not path.exists():
            continue
        reaches = _io_in(path) or [(ln, m) for ln, m in _sibling_imports(path) if f"{m}.py" in DATA_LAYER]
        assert reaches, (
            f"src/eracoef/{module} is in NEEDS_SPLIT but no longer reaches data -- move it to "
            f"MODEL_LAYER.")
