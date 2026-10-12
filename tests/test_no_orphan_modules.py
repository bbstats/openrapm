"""Every module of the package is reached from something a person runs.

A module nothing imports still has to be read, kept passing its tests and kept inside the layer rules, and nobody can
tell from the code that it is dead.  The things a person runs are the command line (src/openrapm/commands/cli.py, and
src/openrapm/__main__.py, which is `python -m openrapm`), the experiment checks (scripts/checks/*.py) and the tutorial
(examples/tutorial.py).  Every import in them is followed through the package, at the top of a file or inside a
function (the command line imports each command only when it runs), and every module under src/openrapm must be
reached.  Importing a module also runs its packages' __init__.py, so those count as reached with it.
"""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "openrapm"
PACKAGE = "openrapm"


def _entry_points() -> list:
    return [SRC / "commands" / "cli.py", SRC / "__main__.py", *sorted((ROOT / "scripts" / "checks").glob("*.py")),
            ROOT / "examples" / "tutorial.py"]


def _file(module: str) -> Path | None:
    """The file of a dotted module name of the package, or None when it names no module (a function, a class)."""
    parts = module.split(".")[1:]
    candidate = SRC.joinpath(*parts)
    if candidate.with_suffix(".py").is_file():
        return candidate.with_suffix(".py")
    if (candidate / "__init__.py").is_file():
        return candidate / "__init__.py"
    return None


def _name(path: Path) -> str:
    """The dotted module name of a file under src/openrapm: data/config.py -> openrapm.data.config."""
    parts = list(path.relative_to(SRC).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join([PACKAGE, *parts])


def _base(node: ast.ImportFrom, path: Path) -> str | None:
    """The module a `from ... import` reads from, its relative dots resolved against the importing file."""
    if not node.level:
        return node.module
    if SRC not in path.parents:
        return None
    package = _name(path).split(".") if path.name == "__init__.py" else _name(path).split(".")[:-1]
    stem = package[:len(package) - (node.level - 1)]
    return ".".join(stem + ([node.module] if node.module else []))


def _imported(path: Path) -> set:
    """Every package module a file imports, anywhere in the file."""
    found = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), filename=str(path))):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = _base(node, path)
            if base:
                found.add(base)
                found.update(f"{base}.{alias.name}" for alias in node.names)
    return {module for module in found if module.split(".")[0] == PACKAGE and _file(module)}


def _with_packages(module: str) -> set:
    """A module and every package above it: openrapm.data.config -> itself, openrapm.data, openrapm."""
    parts = module.split(".")
    return {".".join(parts[:end]) for end in range(1, len(parts) + 1)}


def _reached() -> set:
    reached, waiting = set(), list(_entry_points())
    while waiting:
        path = waiting.pop()
        for module in _imported(path):
            for name in _with_packages(module) - reached:
                reached.add(name)
                waiting.append(_file(name))
    return reached


def test_every_module_is_reached_from_a_command_a_check_or_the_tutorial():
    every = {_name(path) for path in SRC.rglob("*.py") if "__pycache__" not in path.parts}
    entry = {_name(path) for path in _entry_points() if SRC in path.parents}
    orphans = sorted(every - _reached() - entry)
    assert not orphans, f"modules nothing a person runs imports: {orphans}"


def test_the_walk_follows_relative_and_lazy_imports():
    """The command line imports `build` inside a function, relative to itself; `build` imports the step modules."""
    reached = _reached()
    assert {"openrapm.commands.build", "openrapm.rapm", "openrapm.box_prior", "openrapm.prior_rapm",
            "openrapm.data.config", "openrapm.simulate"} <= reached
