"""CONTRIBUTING.md's code rules, and the exceptions to them by name, so an exception cannot be added unseen.

The rules: no `# noqa`, no `global`, no `sys.path` changes, no loading one script from another.  The package and the
tests keep all four.  The checks under scripts/checks/ break them in three known ways, each listed below with why;
a new file that breaks one fails here until it is added to the list, in the same commit, with its reason.

    noqa        `# noqa: E402` on the imports of the checks that set the thread variables before numpy loads, which
                the import order then breaks on purpose; and check 152's `# noqa: BLE001` on the one handler that
                passes any worker's error back to the parent.  No other code, no other rule.
    global      the screens that hand their workers' shared state to a multiprocessing initializer
    sys.path    the Power checks 159-166, which run from a checkout without the package installed
    by path     the checks that reuse another check's code, mostly check 151's scoring, by loading its file
"""
import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECKS = "scripts/checks/"
NOQA_E402 = {f"{CHECKS}{name}" for name in (
    "152_decider_tune.py", "155_input_screen.py", "156_shrink_refit.py", "159_power_next10.py",
    "160_power_next10_select.py", "161_power_next10_audit.py", "162_power_next10_recent.py",
    "163_power_next10_recency.py", "164_power_next10_schedule.py", "165_power_forecast_weights.py",
    "166_power_forecast_refit.py")}
NOQA_OTHER = {(f"{CHECKS}152_decider_tune.py", "BLE001")}
GLOBAL = {f"{CHECKS}{name}" for name in (
    "141_stepwise_inputs.py", "142_era_screen.py", "143_warm_start_screen.py", "146_dashboard_screen.py",
    "147_era_flag_screen.py", "148_noise_column_screen.py", "149_matchup_screen.py",
    "150_column_subsampling_screen.py", "153_swap_audit.py")}
SYS_PATH = {f"{CHECKS}{number}_{name}.py" for number, name in (
    (159, "power_next10"), (160, "power_next10_select"), (161, "power_next10_audit"), (162, "power_next10_recent"),
    (163, "power_next10_recency"), (164, "power_next10_schedule"), (165, "power_forecast_weights"),
    (166, "power_forecast_refit"))}
BY_PATH = {f"{CHECKS}{name}" for name in (
    "143_warm_start_screen.py", "146_dashboard_screen.py", "147_era_flag_screen.py", "148_noise_column_screen.py",
    "149_matchup_screen.py", "150_column_subsampling_screen.py", "151_wowy.py", "152_decider_tune.py",
    "153_swap_audit.py", "154_self_creation.py", "155_input_screen.py", "156_shrink_refit.py",
    "159_power_next10.py", "160_power_next10_select.py", "161_power_next10_audit.py", "162_power_next10_recent.py",
    "163_power_next10_recency.py", "164_power_next10_schedule.py", "165_power_forecast_weights.py",
    "166_power_forecast_refit.py")}


def _code_files(*folders) -> dict:
    """{repository path: text} of every .py file under the folders, this one left out: it names the rules."""
    return {path.relative_to(ROOT).as_posix(): path.read_text(encoding="utf-8")
            for folder in folders for path in sorted((ROOT / folder).rglob("*.py"))
            if "__pycache__" not in path.parts and path.resolve() != Path(__file__).resolve()}


def test_noqa_only_where_listed():
    found_e402, found_other = set(), set()
    for name, text in _code_files("src", "scripts", "tests", "examples").items():
        for codes in re.findall(r"#\s*noqa(?::\s*([A-Z0-9, ]+))?", text):
            for code in [c.strip() for c in codes.split(",")] if codes else ["(bare)"]:
                if code == "E402":
                    found_e402.add(name)
                else:
                    found_other.add((name, code))
    assert found_e402 == NOQA_E402
    assert found_other == NOQA_OTHER


def test_global_only_where_listed():
    found = {name for name, text in _code_files("src", "scripts", "tests", "examples").items()
             if any(isinstance(node, ast.Global) for node in ast.walk(ast.parse(text)))}
    assert found == GLOBAL


def test_sys_path_changed_only_where_listed():
    found = {name for name, text in _code_files("src", "scripts", "tests", "examples").items()
             if re.search(r"sys\.path\.(insert|append|extend)\(|sys\.path\s*=", text)}
    assert found == SYS_PATH


def test_a_script_loads_another_by_path_only_where_listed():
    found = {name for name, text in _code_files("src", "scripts", "examples").items()
             if "spec_from_file_location" in text}
    assert found == BY_PATH


def _module_exists(module: str) -> bool:
    parts = module.split(".")
    path = ROOT / "src" / Path(*parts)
    return path.with_suffix(".py").is_file() or (path / "__init__.py").is_file()


def test_every_import_of_the_package_names_a_module_that_exists():
    """Imports inside functions included: check 155 imports the package only inside functions, so a module that was
    removed fails only when that function runs (the three priors removed on 2026-11-16)."""
    bad = []
    for name, text in _code_files("src", "scripts", "tests", "examples").items():
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                modules = [node.module]
                package = ROOT / "src" / Path(*node.module.split(".")) / "__init__.py"
                if package.is_file():          # from a package: each name is one of its modules, or defined in it
                    defined = package.read_text(encoding="utf-8")
                    modules += [f"{node.module}.{alias.name}" for alias in node.names
                                if not re.search(rf"\b{re.escape(alias.name)}\b", defined)]
            else:
                continue
            bad += [f"{name}: {module}" for module in modules
                    if module.split(".")[0] == "openrapm" and not _module_exists(module)]
    assert not bad, f"imports of modules that do not exist: {bad}"
