"""Three things that rot silently: a config key nobody reads, a citation of a deleted script, and a documented flag
no script understands.

Neither breaks anything, which is exactly why both survive.  A key with a comment telling you to
change it, that nothing reads, costs a day; so does a command line in a doc naming a file that was
deleted a release ago.  Both are cheap to check and nobody checks them by hand.

`UNREAD` is a ratchet in the same spirit as `tests/test_layer_boundary.py`: a key joins it when it
goes dead, and the test fails both when a NEW key goes unread and when one on the list comes back to
life, so the list cannot quietly stop being true.
"""
import ast
import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
CODE = "\n".join(p.read_text(encoding="utf-8", errors="replace")
                 for d in ("src", "scripts") for p in sorted((ROOT / d).rglob("*.py")))

# Read by nothing in src/ or scripts/.  Empty since the 2026-10 cleanup deleted every dead key; a key
# that goes dead is deleted from config.yaml, or listed here with a comment beside it there saying so.
# What is NOT allowed is one of these carrying an instruction to change it, because changing it does nothing.
UNREAD: set = set()

# Docs and config that a stranger follows literally.  The files in notes/ (HANDOFF.md, DECISIONS.md,
# WITHIN_SEASON_LEDGER.md) are deliberately excluded from the script check: they are a record of what was done,
# they cite retired scripts on purpose, and each opens with a header that maps those scripts to today's commands.
# Their command lines are still checked for flags below.
FOLLOWED = ["config.yaml", "README.md", "PIPELINE.md", "CONTRIBUTING.md", "GLOSSARY.md"]
DOCS = sorted(ROOT.glob("*.md")) + sorted((ROOT / "notes").glob("*.md"))
SCRIPT = re.compile(r"scripts/(?:checks/)?\d+_[A-Za-z0-9_]+\.py")


def _named(key) -> bool:
    return bool(re.search(r"""["']""" + re.escape(str(key)) + r"""["']""", CODE))


def _leaves(node, path=()):
    if isinstance(node, dict):
        for k, v in node.items():
            yield path + (str(k),), v
            yield from _leaves(v, path + (str(k),))


def _unread_keys() -> set:
    """Config paths whose own name appears in no source file, with dead subtrees reported once."""
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    out, covered = set(), set()
    for path, _ in _leaves(cfg):
        dotted = ".".join(path)
        if _named(path[-1]):
            continue
        if any(dotted.startswith(c + ".") for c in covered):
            continue
        out.add(dotted)
        covered.add(dotted)
    return out


def test_no_config_key_is_unread_without_being_listed():
    """A new key that nothing reads is almost always a rename that missed a caller."""
    extra = sorted(_unread_keys() - UNREAD)
    assert not extra, (
        f"config.yaml key(s) {extra} are read by nothing in src/ or scripts/.  Either wire them up, or "
        f"add them to UNREAD here with a comment in config.yaml saying so -- and make sure no comment "
        f"beside them tells a reader to change a value that has no effect.")


# ------------------------------------------------------------- the data paths and the box counts
BOX_COUNTS = ["fg3m", "fg3_miss", "fg2m", "fg2_miss", "ftm", "ft_miss", "orb", "drb", "ast", "tov", "stl", "blk", "pf"]


def test_box_counts_replaced_features_with_the_same_value_in_the_same_order():
    """`features` was renamed `box_counts` in the 2026-10 cleanup.  The value and its order are load-bearing:
    the design's box columns follow it and every cached design piece records it."""
    from openrapm.design import FEATURES
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    assert "features" not in cfg, "the old key name is back; the code reads cfg['box_counts']"
    assert cfg["box_counts"] == BOX_COUNTS == list(FEATURES)
    assert _named("box_counts")
    assert not re.search(r"""cfg\[["']features["']\]""", CODE), "a reader of the old `features` key survives"


def test_every_data_path_is_configured_and_read():
    """Every folder the data layer reads or writes is named in config.yaml paths, read by name, and by default
    points where it always has, so nothing moves on disk (the season panel moved once, on 2026-10-11, when its
    columns got their plain names: outputs/role_panel_season.parquet became outputs/season_panel.parquet)."""
    from openrapm.data.config import DEFAULT_PATHS
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    expected = {"raw": "data/raw", "stints": "data/stints", "cache": "data/cache", "shot_curves": "data/shotcurve",
                "outputs": "outputs", "panel": "outputs/season_panel.parquet", "within": "outputs/within"}
    for key, value in expected.items():
        assert cfg["paths"][key] == value == DEFAULT_PATHS[key], key
        assert _named(key), f"paths.{key} is configured but nothing resolves it"
    assert cfg["roles"]["cache"] == "roles.parquet", "the roles table is a file name inside paths.cache"


def test_the_roles_table_sits_where_it_always_has_and_moves_with_the_cache(tmp_path):
    """roles.cache is a file name inside paths.cache: under the repository's config the table is still
    data/cache/roles_RSPO.parquet, and a config that moves paths.cache moves the table with it."""
    from openrapm.data.config import load_config
    from openrapm.data.roles import roles_path
    assert roles_path(load_config(ROOT / "config.yaml")) == ROOT / "data" / "cache" / "roles_RSPO.parquet"
    assert roles_path(load_config(ROOT / "config.yaml"), ("RS",)) == ROOT / "data" / "cache" / "roles_RS.parquet"
    moved = {"_root": str(tmp_path / "repo"), "paths": {"cache": str(tmp_path / "verify" / "cache")},
             "roles": {"cache": "roles.parquet"}}
    assert roles_path(moved) == tmp_path / "verify" / "cache" / "roles_RSPO.parquet"
    assert roles_path({"_root": str(tmp_path / "repo")}) == tmp_path / "repo" / "data" / "cache" / "roles_RSPO.parquet"


def test_resolve_takes_relative_paths_from_the_root_and_absolute_ones_as_they_are(tmp_path):
    from openrapm.data.config import resolve
    from openrapm.data.roles import roles_path
    cfg = {"_root": str(tmp_path / "repo"), "paths": {"raw": "data/raw", "cache": str(tmp_path / "elsewhere")},
           "roles": {"cache": str(tmp_path / "elsewhere" / "roles.parquet")}}
    assert resolve(cfg, "raw") == tmp_path / "repo" / "data" / "raw"
    assert resolve(cfg, "cache") == tmp_path / "elsewhere"
    assert resolve(cfg, "shot_curves") == tmp_path / "repo" / "data" / "shotcurve"      # not configured: the default
    assert roles_path(cfg) == tmp_path / "elsewhere" / "roles_RSPO.parquet"
    with pytest.raises(KeyError):
        resolve(cfg, "no_such_path")


def test_load_config_honours_OPENRAPM_CONFIG(tmp_path, monkeypatch):
    """A verification run points every data path at another folder by naming another config.yaml."""
    from openrapm.data.config import CONFIG_ENV, load_config, resolve
    from openrapm.data.config import ROOT as CODE_ROOT
    assert CONFIG_ENV == "OPENRAPM_CONFIG"
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    other = tmp_path / "verify"
    cfg["paths"] = {key: str(other / key) for key in ("raw", "stints", "cache", "shot_curves")}
    alt = tmp_path / "alt_config.yaml"
    with open(alt, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh)

    monkeypatch.delenv(CONFIG_ENV, raising=False)
    assert resolve(load_config(), "cache") == ROOT / "data" / "cache"          # unset: the repository's own

    monkeypatch.setenv(CONFIG_ENV, str(alt))
    got = load_config()
    assert got["_root"] == str(CODE_ROOT) == str(ROOT)
    for key in ("raw", "stints", "cache", "shot_curves"):
        assert resolve(got, key) == other / key
    from openrapm.data.roles import roles_path
    assert roles_path(got) == other / "cache" / "roles_RSPO.parquet", "the roles table must follow paths.cache"
    assert resolve(load_config(ROOT / "config.yaml"), "cache") == ROOT / "data" / "cache"   # an explicit path wins


@pytest.mark.parametrize("written, read", [("window", "player"), ("season", "player_season"),
                                           ("player", "player"), ("player_season", "player_season")])
def test_a_player_unit_written_before_the_rename_is_read_as_its_plain_name(tmp_path, written, read):
    """The verification configs written before 2026-10-11 say player_unit "window"; they must build the same
    designs, so the old value is read as "player" (and "season" as "player_season")."""
    from openrapm.data.config import load_config
    from openrapm.design import PLAYER_UNITS
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    cfg["player_unit"] = written
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    assert load_config(path)["player_unit"] == read and read in PLAYER_UNITS


def test_no_listed_key_has_quietly_come_back_to_life():
    revived = sorted(UNREAD - _unread_keys())
    assert not revived, (f"{revived} are read by code again; take them off UNREAD.")


@pytest.mark.parametrize("doc", FOLLOWED)
def test_every_script_a_followed_doc_names_exists_or_says_it_is_retired(doc):
    text = (ROOT / doc).read_text(encoding="utf-8", errors="replace")
    bad = []
    for m in SCRIPT.finditer(text):
        if (ROOT / m.group(0)).exists():
            continue
        around = text[max(0, m.start() - 150):m.end() + 150]
        if "retired" not in around and "deleted" not in around:
            bad.append(m.group(0))
    assert not bad, (
        f"{doc} names {sorted(set(bad))}, which do not exist.  Point at the script that does the job "
        f"now, or say beside it that it was retired and where the numbers went -- a reader should never "
        f"have to run a command to find out it cannot be run.")


# --------------------------------------------------------------- the documented command lines
# Every script reads its command line with argparse, which stops a run on a flag it was not told about (and,
# with allow_abbrev=False, on an abbreviation of one) instead of ignoring it.  That only helps if the flags the
# docs tell you to type are the ones the scripts answer to, so check the docs against the source.  Every .md
# at the top and in notes/, including notes/HANDOFF.md, because its commands are meant to be run.
def declared_options(path) -> set:
    """Every option a script's argparse parser declares, without the dashes: the literal "--name" arguments of
    each `add_argument` call in its source."""
    tree = ast.parse(Path(path).read_text(encoding="utf-8"), filename=str(path))
    return {arg.value[2:] for node in ast.walk(tree)
            if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "add_argument"
            for arg in node.args if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
            and arg.value.startswith("--")}


def script_file(named: str) -> Path | None:
    """The script a doc names: the path as written, else the file of that name in scripts/checks/ or scripts/ (a
    check that moved keeps its number and name), else None."""
    path = ROOT / named
    if path.exists():
        return path
    name = Path(named).name
    return next((p for p in (ROOT / "scripts" / "checks" / name, ROOT / "scripts" / name) if p.exists()), None)


COMMAND = re.compile(r"(?:python|\S*python(?:\.exe)?)\s+(scripts/(?:checks/)?[0-9]\w*\.py)((?:\s+-[^\s`]+)*)")


def test_the_option_scan_reads_every_form(tmp_path):
    source = ('import argparse\np = argparse.ArgumentParser()\np.add_argument("--tag", default="x")\n'
              'p.add_argument("--team_effects")\np.add_argument("--dry", action="store_true")\n'
              'p.add_argument("tables", nargs="*")\nsub = p.add_subparsers().add_parser("x")\n'
              'sub.add_argument("-q", "--quiet")\n')
    path = tmp_path / "sample.py"
    path.write_text(source, encoding="utf-8")
    assert declared_options(path) == {"tag", "team_effects", "dry", "quiet"}


def test_every_flag_a_doc_tells_you_to_type_is_one_the_script_understands():
    checked, bad = 0, []
    for md in DOCS:
        for line in md.read_text(encoding="utf-8", errors="replace").splitlines():
            for m in COMMAND.finditer(line):
                script = script_file(m.group(1))
                if script is None:
                    continue                      # the retired-script rule above covers this
                known = declared_options(script)
                for tok in re.findall(r"--([A-Za-z_][\w-]*)", m.group(2)):
                    checked += 1
                    if tok not in known:
                        bad.append(f"{md.name}: {m.group(1)} --{tok}")
    assert checked > 20, f"only {checked} documented flags found; the command regex has stopped matching"
    assert not bad, (
        f"documented flag(s) no script understands: {sorted(set(bad))}.  argparse stops the run on these "
        f"instead of ignoring them, so a doc that names one is a command nobody can run.")


# The experiment chain builds for hours before its first reading, and a reading that fails does not stop it, so a
# moved check or a renamed flag would show up only as "No such file" or "unrecognized arguments" in the log at the
# end.  No allowance for retired scripts here: the chain is run, not read.
CHAIN = ROOT / "scripts" / "experiment_chain.sh"
CHAIN_CALL = re.compile(r"(scripts/checks/\d+_\w+\.py)((?:[ \t]+[^\s>]+)*)")


def _chain_call_problems(script: str, arguments: str) -> list:
    """What is wrong with one call of a check in the chain: a missing script, or flags its parser does not declare."""
    if not (ROOT / script).exists():
        return [f"{script} does not exist"]
    known = declared_options(ROOT / script)
    return [f"{script} --{flag}" for flag in re.findall(r"(?<![\w-])--([A-Za-z_][\w-]*)", arguments)
            if flag not in known]


def test_every_check_the_experiment_chain_calls_exists_and_declares_its_flags():
    calls, bad = 0, []
    for line in CHAIN.read_text(encoding="utf-8").splitlines():
        for match in CHAIN_CALL.finditer(line):
            calls += 1
            bad += _chain_call_problems(match.group(1), match.group(2))
    assert calls >= 10, f"only {calls} check calls found in the chain; the pattern has stopped matching"
    assert not bad, f"the experiment chain would fail at these readings: {sorted(set(bad))}"


def test_the_chain_looks_for_every_check_it_calls_before_it_builds():
    """READINGS is the list the chain checks for before the build starts; a call missing from it is a check whose
    absence would be found only after the build."""
    text = CHAIN.read_text(encoding="utf-8")
    listed = re.search(r'^READINGS="([^"]*)"', text, re.M)
    assert listed, "the chain no longer lists its readings in READINGS"
    called = set(re.findall(r"scripts/checks/(\d+_\w+)\.py", text))
    assert called == set(listed.group(1).split())


def test_no_script_reads_its_own_command_line():
    """argparse is the one reader: a script that also scans sys.argv for "--x=" accepts what the parser does not
    know about, which is how a misspelled flag used to be ignored silently."""
    scanners = sorted(p.relative_to(ROOT).as_posix() for p in (ROOT / "scripts").rglob("*.py")
                      if re.search(r"sys\.argv\[1:\]|\b_cli\b", p.read_text(encoding="utf-8")))
    assert not scanners, f"scripts reading sys.argv or scripts/_cli.py themselves: {scanners}"
    assert not (ROOT / "scripts" / "_cli.py").exists(), "scripts/_cli.py was retired for argparse"


def test_the_booster_settings_are_never_read_from_config_yaml():
    """They live in params/booster_params_<name>.json alone; the chimeraboost block config.yaml once carried
    was a fallback only the retired scripts/62_single_year_board.py read (at the tag archive/research-2026-10)."""
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    assert "gbdt" not in cfg
    assert not re.search(r"""cfg\[["']gbdt["']\]""", CODE)


# --------------------------------------------------------------- the openrapm command lines
OPENRAPM = re.compile(r"(?:-m openrapm|\bopenrapm)\s+(data\s+(?:ingest|stints|panel)|build|rate|shrink|swap|folds|"
                      r"site|compare|glossary)\b([^\n`]*)")
# A command as a doc types it: in a code block (`python -m openrapm data panel`) or a code span (`openrapm build`).
NAMED_COMMAND = re.compile(r"(?:-m openrapm|`openrapm)\s+([a-z]+)(?:\s+([a-z]+))?")


def _subcommands(parser) -> dict:
    """{name: its parser} for a parser's subcommands, or {} when it has none."""
    import argparse
    actions = [action for action in parser._actions if isinstance(action, argparse._SubParsersAction)]
    return dict(actions[0].choices) if actions else {}


def _openrapm_parsers() -> dict:
    """{command: its parser}, with each `data` subcommand under its two words ("data ingest", ...)."""
    from openrapm.commands.cli import build_parser
    parsers = {}
    for name, sub in _subcommands(build_parser()).items():
        parsers[name] = sub
        parsers.update({f"{name} {inner}": parser for inner, parser in _subcommands(sub).items()})
    return parsers


def _openrapm_flags() -> dict:
    """{command: every option string its parser accepts}."""
    return {name: {option for action in sub._actions for option in action.option_strings}
            for name, sub in _openrapm_parsers().items()}


def test_every_openrapm_flag_the_docs_and_the_chain_use_exists():
    """argparse stops a run on an unknown flag, so a documented one that does not exist is a command nobody can run."""
    known = _openrapm_flags()
    files = DOCS + [ROOT / "scripts" / "experiment_chain.sh"]
    checked, bad = 0, []
    for path in files:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            for match in OPENRAPM.finditer(line):
                command = " ".join(match.group(1).split())
                for flag in re.findall(r"(?<![\w-])(--[a-z][\w-]*)", match.group(2)):
                    checked += 1
                    if flag not in known[command]:
                        bad.append(f"{path.name}: openrapm {command} {flag}")
    assert checked >= 20, f"only {checked} openrapm flags found; the pattern has stopped matching"
    assert not bad, f"flags `openrapm` does not understand: {sorted(set(bad))}"


def test_every_openrapm_command_the_docs_name_exists():
    """`python -m openrapm NAME` in a code block or `openrapm NAME` in a code span must name a real command, and
    `openrapm data NAME` a real data step: a renamed command would otherwise live on in the docs."""
    parsers = _openrapm_parsers()
    checked, bad = 0, []
    for path in DOCS:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            for match in NAMED_COMMAND.finditer(line):
                checked += 1
                name, inner = match.group(1), match.group(2)
                if name not in parsers:
                    bad.append(f"{path.name}: openrapm {name}")
                elif _subcommands(parsers[name]) and inner and f"{name} {inner}" not in parsers:
                    bad.append(f"{path.name}: openrapm {name} {inner}")
    assert checked >= 30, f"only {checked} openrapm commands found; the pattern has stopped matching"
    assert not bad, f"commands `openrapm` does not have: {sorted(set(bad))}"
