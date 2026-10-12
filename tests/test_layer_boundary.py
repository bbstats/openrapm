"""The layer boundary: model code may not load its own data.

The project has two layers, one rule between them, and a third folder that joins them.

    DATA   reads the disk and the network.  Scraping, stints, box tables, caches, loaders.
           A contributor may add a source here -- tracking data, a different feed, anything.
           This is where a maintainer looks.  It lives in src/openrapm/data/: every module in that
           folder is data layer by where it sits.

    MODEL  gets handed data and returns numbers.  The ridge, the box prior, the role prior,
           the padding, the design assembly, the calibration.  This is where a pull request
           that improves the model belongs, and it may not open a file.

    COMMANDS  src/openrapm/commands/: `openrapm rate`, `openrapm folds`, ...  Each reads its inputs
           through the data layer, runs the model steps and writes its outputs, so it may import
           anything.  Nothing else may import it: a model module or a loader that reached a command
           would reach everything the command reaches.

The rule exists because the season boundary (src/openrapm/data/seasons.py) is only as good as the
narrowest channel through which data can reach a fit.  If a model module may call `read_parquet`,
it can reach past every guard and read the season in progress directly, and no amount of care in
the loaders would stop it.  Keeping the model layer I/O-free makes the loaders the only way in.

Two rules, because the first alone is not enough
------------------------------------------------
Checking for `read_parquet` in the module itself misses the interesting case.
`data/box_scores.py` calls no I/O verb at all, and `box_scores.season_box` still reaches
stats.nba.com -- it calls `ingest.load_gamelog`, which SCRAPES when the cache is cold.  That is how
the test suite was quietly downloading three game logs on a "fresh clone with no data" run.  So a
model module must also not IMPORT a data-layer module.  That is rule two here; the network itself is
blocked in conftest.py so no test can silently fetch what it is missing.

`MODEL_LAYER` is the list of modules that pass both rules today and must keep passing.  It is a
ratchet: a module joins it when it is clean and never leaves.  `NEEDS_SPLIT` is the honest
remainder -- model code that still reaches its own data, with what it reaches through.  Shrinking
that list is the work.  `DATA_LAYER` names the top-level modules that still load data and have not
yet moved into data/.  Both of those lists may only shrink (`test_the_exception_lists_only_shrink`),
and a model module may not import a `NEEDS_SPLIT` module either, since that reaches the same data one
step removed.  `PACKAGE_FILES` (`__init__.py`, and `__main__.py` once it exists) belong to neither
layer: `__init__.py` imports nothing from the package, `__main__.py` only the command line.
"""
import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "openrapm"
DATA_DIR = "data"            # src/openrapm/data/: the data layer, by location
COMMANDS_DIR = "commands"    # src/openrapm/commands/: the commands, by location; they may import anything

# Clean on both rules: no I/O verb, no data-layer import.
MODEL_LAYER = [
    "player_types.py",  # the box-profile mixture, the per-type gap and bias, each type's signature
    "design.py",        # the design matrix from the frames it is handed; the stint column and counter names
    "box_rates.py",     # the season panel's padded per-100 box rates, and the padding helpers
    "on_court.py",      # his team's record with him on the court and off it
    "simulate.py",      # the simulated league, and its season panel, that the tests and the tutorial fit
    "swap_adjust.py",   # step 5: the swap type model, the give-back, team totals held, the spread held
    "swap_test.py",     # the swap test: same-four lineup pairs, scored against a neighbouring season's ranking
    "trade_set.py",     # the team-game with-and-without-you ridge and its losses
    "consensus.py",     # the consensus check: the name join, the pooling, rank agreement and spreads
    "settings.py",      # the shipped settings, each with the experiment that set it
    "features.py",      # the prior's input catalogue, the derived columns, the shipped lists
    "rapm.py",          # step 1: the season sums, the leave-season-out RAPM, the un-shrunk label
    "box_prior.py",     # step 2: training rows, balanced player folds, the out-of-player LightGBM prior
    "prior_rapm.py",    # step 3: the ridge around the prior, the season's table, centring, the schema
    "prior_shrink.py",  # step 4: the prior part's multipliers from the within-season folds, the shrink
    "scoring.py",       # ratings scored on unseen games, the splits, the pooled and paired summaries
]

# Model code that still reaches its own data, and what through.  Nothing may be added here.
NEEDS_SPLIT: dict = {}

# Top-level modules whose job IS to reach the disk or the network, not yet moved into data/.
DATA_LAYER: set = set()

# What the two exception lists may hold at most (2026-10 cleanup): nothing, since phase 9a.  holdout.py left in
# phase 7 (split into scoring.py and commands/year_over_year.py); in phase 9a spm.py was deleted and seasoninputs.py
# was merged into data/panel.py (the rows) and commands/within_season.py (the leak checks).
NEEDS_SPLIT_AT_MOST: set = set()
DATA_LAYER_AT_MOST: set = set()

# The package's own entry files: neither layer (see test_the_package_files_import_nothing_from_the_package).
PACKAGE_FILES = {"__init__.py", "__main__.py"}

# The verbs that reach the disk or the network.  `.get` and `.exists` are deliberately absent:
# they are overwhelmingly `dict.get` and would drown the rule in false positives.
IO_CALLS = {"read_parquet", "read_csv", "read_json", "to_parquet", "to_csv", "to_json",
            "read_pickle", "to_pickle", "read_feather", "to_feather", "read_excel", "to_excel",
            "read_hdf", "to_hdf", "read_table", "read_fwf",
            "load", "save", "savez", "savez_compressed", "loadtxt", "savetxt", "genfromtxt", "fromfile",
            "tofile", "memmap", "dump", "savefig",
            "open", "glob", "rglob", "iterdir", "read_text", "write_text", "read_bytes",
            "write_bytes", "urlopen", "urlretrieve", "request"}
IO_MODULES = {"requests", "urllib", "nba_api", "httpx", "socket"}


def _modules() -> set:
    """Every module of the package, as a path relative to src/openrapm ("design.py", "data/roles.py")."""
    return {p.relative_to(SRC).as_posix() for p in SRC.rglob("*.py") if "__pycache__" not in p.parts}


def _is_data(module: str) -> bool:
    return module.startswith(DATA_DIR + "/") or module in DATA_LAYER


def _is_command(module: str) -> bool:
    return module.startswith(COMMANDS_DIR + "/")


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


def _module_file(parts: list, name: str | None = None) -> str:
    """The package module that `parts` (+ an imported `name`) names, relative to src/openrapm."""
    if (SRC.joinpath(*parts)).is_dir():                     # a package: the imported name may be a module in it
        if name and (SRC.joinpath(*parts) / f"{name}.py").exists():
            return "/".join([*parts, f"{name}.py"])
        if name and (SRC.joinpath(*parts) / name).is_dir():  # or a subpackage: `from . import data`
            return "/".join([*parts, name, "__init__.py"])
        return "/".join([*parts, "__init__.py"])
    return "/".join(parts) + ".py"


def _sibling_imports(path: Path):
    """(line, module) for every import of another module in this package, the module named by its path
    relative to src/openrapm, so `from .data.roles import x`, `from .data import roles`, `from ..design
    import y` and `import openrapm.data.roles` all resolve to the file they load."""
    return _imports_of(_parse(path), list(path.relative_to(SRC).parent.parts))


def _imports_of(tree, package: list):
    """`_sibling_imports` on a parsed module that sits in `package` ([] at the top level, ["data"] in data/)."""
    out = []
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom):
            if n.level:
                up = n.level - 1
                if up > len(package):
                    continue
                parts = package[:len(package) - up] + (n.module.split(".") if n.module else [])
            elif n.module and (n.module == "openrapm" or n.module.startswith("openrapm.")):
                parts = n.module.split(".")[1:]
            else:
                continue
            for a in n.names:
                out.append((n.lineno, _module_file(parts, a.name)))
        elif isinstance(n, ast.Import):
            out.extend((n.lineno, _module_file(a.name.split(".")[1:])) for a in n.names
                       if a.name.startswith("openrapm."))
    return out


def test_the_import_resolver_reads_every_form():
    """The rule is only as good as the resolver: each import form must name the file it loads."""
    top = ast.parse("from .data.roles import a\nfrom .data import roles\nfrom . import box_rates\n"
                    "import openrapm.data.stints\nfrom openrapm.data.bio import b\nfrom .design import c\n")
    assert sorted(m for _, m in _imports_of(top, [])) == sorted(
        ["data/roles.py", "data/roles.py", "box_rates.py", "data/stints.py", "data/bio.py", "design.py"])
    inner = ast.parse("from .config import resolve\nfrom ..design import HOME_SLOTS\nfrom .. import box_rates\n")
    assert sorted(m for _, m in _imports_of(inner, [DATA_DIR])) == ["box_rates.py", "data/config.py", "design.py"]
    # inside commands/, `.data` is the `openrapm data` command (commands/data.py) and `..data` the data layer
    command = ast.parse("from .data import add_data\nfrom ..data.panel import write_panel\n")
    assert sorted(m for _, m in _imports_of(command, [COMMANDS_DIR])) == ["commands/data.py", "data/panel.py"]
    assert _is_command("commands/data.py") and not _is_data("commands/data.py")
    # the subpackage itself, imported by name: data/__init__.py, not the package's own __init__.py
    whole = ast.parse("from . import data\nfrom openrapm import data\nimport openrapm.data\n"
                      "from openrapm import __version__\n")
    assert sorted(m for _, m in _imports_of(whole, [])) == sorted(
        ["data/__init__.py", "data/__init__.py", "data/__init__.py", "__init__.py"])
    assert all(_is_data(m) for m in ("data/roles.py", "data/stints.py", "data/__init__.py", "data/panel.py"))
    assert not any(_is_data(m) for m in ("design.py", "box_rates.py", "__init__.py"))


@pytest.mark.parametrize("module", MODEL_LAYER)
def test_a_model_module_does_no_io(module):
    path = SRC / module
    assert path.exists(), f"{module} is in MODEL_LAYER but does not exist -- update the list"
    found = _io_in(path)
    assert not found, (
        f"src/openrapm/{module} is model-layer code and may not reach the disk or the network, but "
        f"it does at {found}.  Take the data as an argument instead; the loader belongs in the data "
        f"layer.  See the docstring of this file.")


@pytest.mark.parametrize("module", MODEL_LAYER)
def test_a_model_module_does_not_import_the_data_layer(module):
    """The rule that catches the interesting case.  box_scores.py calls no I/O verb and still scrapes,
    because it calls ingest.load_gamelog.  Reaching data through a sibling is still reaching data."""
    bad = [(ln, m) for ln, m in _sibling_imports(SRC / module) if _is_data(m) or m in NEEDS_SPLIT or _is_command(m)]
    assert not bad, (
        f"src/openrapm/{module} imports data-layer or command module(s) {sorted({m for _, m in bad})} at lines "
        f"{[ln for ln, _ in bad]} (a NEEDS_SPLIT module counts: it reaches data one step removed).  "
        f"Whatever it needs from them should be passed in.")


def test_only_the_commands_and_the_entry_file_import_the_commands():
    """A loader (data/) or a model module that imported a command would reach
    everything the command reaches.  commands/ itself may import anything."""
    bad = {}
    for module in sorted(_modules()):
        if _is_command(module) or module == "__main__.py":
            continue
        reached = sorted({m for _, m in _sibling_imports(SRC / module) if _is_command(m)})
        if reached:
            bad[module] = reached
    assert not bad, f"modules outside commands/ import a command: {bad}"


def test_the_entry_file_calls_the_command_line():
    assert [m for _, m in _sibling_imports(SRC / "__main__.py")] == [COMMANDS_DIR + "/cli.py"]


def test_the_exception_lists_only_shrink():
    """Adding a module to NEEDS_SPLIT or DATA_LAYER would excuse it from both rules; neither list may grow."""
    grew = sorted(set(NEEDS_SPLIT) - NEEDS_SPLIT_AT_MOST)
    assert not grew, f"NEEDS_SPLIT grew: {grew}"
    assert DATA_LAYER <= DATA_LAYER_AT_MOST, f"DATA_LAYER grew: {sorted(DATA_LAYER - DATA_LAYER_AT_MOST)}"


@pytest.mark.parametrize("module", sorted(m for m in PACKAGE_FILES if (SRC / m).exists()))
def test_the_package_files_import_nothing_from_the_package(module):
    """`import openrapm` must stay cheap and must not load data: __init__.py holds the version and nothing else.
    __main__.py may reach the command line (commands/) and nothing else."""
    path = SRC / module
    allowed = "commands/" if module == "__main__.py" else None
    imports = [m for _, m in _sibling_imports(path) if not (allowed and m.startswith(allowed))]
    assert not _io_in(path), f"src/openrapm/{module} reaches the disk or the network"
    assert not imports, f"src/openrapm/{module} imports {sorted(set(imports))} from the package"


def test_the_lists_do_not_overlap():
    assert not set(MODEL_LAYER) & set(NEEDS_SPLIT), "a module cannot be both clean and needing a split"
    assert not set(MODEL_LAYER) & DATA_LAYER, "a module cannot be both model-layer and data-layer"
    assert not PACKAGE_FILES & (set(MODEL_LAYER) | set(NEEDS_SPLIT) | DATA_LAYER), \
        "a package entry file belongs to neither layer"
    assert not any(m.startswith(DATA_DIR + "/") for m in [*MODEL_LAYER, *NEEDS_SPLIT]), \
        "a module in data/ is data layer by location; it cannot be listed as model code"
    assert not any(_is_command(m) for m in [*MODEL_LAYER, *NEEDS_SPLIT, *DATA_LAYER]), \
        "a module in commands/ is placed by location; it cannot be listed in a layer"


def test_every_module_is_classified():
    """A new module must be placed in one layer or the other, so nothing lands here unnoticed.  A module
    in data/ or commands/ is placed by being there."""
    known = set(MODEL_LAYER) | set(NEEDS_SPLIT) | DATA_LAYER | PACKAGE_FILES
    unclassified = {m for m in _modules() if m not in known and not m.startswith(DATA_DIR + "/")
                    and not _is_command(m)}
    assert not unclassified, (
        f"new module(s) {sorted(unclassified)}: add each to MODEL_LAYER (if it takes its data as an "
        f"argument and imports no loader), to NEEDS_SPLIT (if it is model code that still loads), or "
        f"move it into src/openrapm/data/ (if loading is its job).  See the docstring.")


def test_the_data_layer_is_the_data_folder():
    """The modules that moved into data/ in the 2026-10 cleanup are there and nowhere else."""
    moved = {"config.py", "ingest.py", "stints.py", "stint_parser.py", "shot_curves.py", "box_scores.py",
             "roles.py", "bio.py", "seasons.py", "shots.py", "designs.py"}
    have = {m.split("/", 1)[1] for m in _modules() if m.startswith(DATA_DIR + "/")}
    assert moved <= have, f"data/ is missing {sorted(moved - have)}"
    stale = {"config.py", "ingest.py", "stints.py", "shotcurve.py", "boxtable.py", "roles.py", "bio.py",
             "seasons.py", "xshoot.py", "designcache.py"} & {m for m in _modules() if "/" not in m}
    assert not stale, f"{sorted(stale)} still sit at the top level of src/openrapm"


def test_the_split_list_is_accurate():
    """Every module claimed to need a split must actually reach data -- otherwise it belongs in
    MODEL_LAYER and the ratchet should have moved."""
    for module in NEEDS_SPLIT:
        path = SRC / module
        if not path.exists():
            continue
        reaches = _io_in(path) or [(ln, m) for ln, m in _sibling_imports(path) if _is_data(m)]
        assert reaches, (
            f"src/openrapm/{module} is in NEEDS_SPLIT but no longer reaches data -- move it to "
            f"MODEL_LAYER.")
