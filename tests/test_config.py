"""Two things that rot silently: a config key nobody reads, and a citation of a deleted script.

Neither breaks anything, which is exactly why both survive.  A key with a comment telling you to
change it, that nothing reads, costs a day; so does a command line in a doc naming a file that was
deleted a release ago.  Both are cheap to check and nobody checks them by hand.

`UNREAD` is a ratchet in the same spirit as `tests/test_layer_boundary.py`: a key joins it when it
goes dead, and the test fails both when a NEW key goes unread and when one on the list comes back to
life, so the list cannot quietly stop being true.
"""
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

# Docs and config that a stranger follows literally.  HANDOFF.md and DECISIONS.md are deliberately
# excluded: they are a record of what was done, and they cite retired scripts on purpose.
FOLLOWED = ["config.yaml", "README.md", "PIPELINE.md", "CONTRIBUTING.md"]
SCRIPT = re.compile(r"scripts/\d+_[A-Za-z0-9_]+\.py")


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
    points where it always has, so nothing moves on disk."""
    from openrapm.data.config import DEFAULT_PATHS
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    expected = {"raw": "data/raw", "stints": "data/stints", "cache": "data/cache", "shot_curves": "data/shotcurve",
                "outputs": "outputs", "panel": "outputs/role_panel_season.parquet", "within": "outputs/within"}
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
# scripts/_cli.py made a misspelled flag an error instead of a silent default.  That only helps if
# the flags the docs tell you to type are the ones the scripts answer to, so check the docs against
# the source.  Every .md here, including HANDOFF.md, because its commands are meant to be run.
def _declared_in(path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("_cli_for_tests", ROOT / "scripts" / "_cli.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.declared_in(path)


COMMAND = re.compile(r"(?:python|\S*python(?:\.exe)?)\s+(scripts/[0-9]\w*\.py)((?:\s+-[^\s`]+)*)")


def test_every_flag_a_doc_tells_you_to_type_is_one_the_script_understands():
    checked, bad = 0, []
    for md in sorted(ROOT.glob("*.md")):
        for line in md.read_text(encoding="utf-8", errors="replace").splitlines():
            for m in COMMAND.finditer(line):
                script = ROOT / m.group(1)
                if not script.exists():
                    continue                      # the retired-script rule above covers this
                known = _declared_in(script)
                for tok in re.findall(r"--([A-Za-z_][\w-]*)", m.group(2)):
                    checked += 1
                    if tok not in known:
                        bad.append(f"{md.name}: {m.group(1)} --{tok}")
    assert checked > 20, f"only {checked} documented flags found; the command regex has stopped matching"
    assert not bad, (
        f"documented flag(s) no script understands: {sorted(set(bad))}.  Since scripts/_cli.py these "
        f"stop the run instead of being ignored, so a doc that names one is a command nobody can run.")


def test_the_booster_settings_are_never_read_from_config_yaml():
    """They live in params/booster_params_<name>.json alone; the chimeraboost block config.yaml once carried
    was a fallback only the retired scripts/62_single_year_board.py read."""
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    assert "gbdt" not in cfg
    assert not re.search(r"""cfg\[["']gbdt["']\]""", CODE)


# --------------------------------------------------------------- the openrapm command lines
OPENRAPM = re.compile(r"(?:-m openrapm|\bopenrapm)\s+(rate|folds|site|compare)\b([^\n`]*)")


def _openrapm_flags() -> dict:
    """{command: every option string its parser accepts}."""
    import argparse

    from openrapm.commands.cli import build_parser
    commands = next(action for action in build_parser()._actions if isinstance(action, argparse._SubParsersAction))
    return {name: {option for action in sub._actions for option in action.option_strings}
            for name, sub in commands.choices.items()}


def test_every_openrapm_flag_the_docs_and_the_chain_use_exists():
    """argparse stops a run on an unknown flag, so a documented one that does not exist is a command nobody can run."""
    known = _openrapm_flags()
    files = sorted(ROOT.glob("*.md")) + [ROOT / "scripts" / "experiment_chain.sh"]
    checked, bad = 0, []
    for path in files:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            for match in OPENRAPM.finditer(line):
                for flag in re.findall(r"(?<![\w-])(--[a-z][\w-]*)", match.group(2)):
                    checked += 1
                    if flag not in known[match.group(1)]:
                        bad.append(f"{path.name}: openrapm {match.group(1)} {flag}")
    assert checked >= 8, f"only {checked} openrapm flags found; the pattern has stopped matching"
    assert not bad, f"flags `openrapm` does not understand: {sorted(set(bad))}"
