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

# Read by nothing in src/ or scripts/ as of 2026-09-17.  Each is left named rather than deleted
# because it records a decision (see the comments beside it in config.yaml); what is NOT allowed is
# one of these carrying an instruction to change it, because changing it does nothing.
UNREAD = {
    "lam_buckets", "lam_delta", "lam_ratio_grid", "lam_delta_grid",
    "boost_min_poss", "boost_quality", "boost_nuisance",
    "pad_scale_grid", "min_half_poss", "gt_weight_robustness", "run_tags",
    "ratings_prior.role_prior", "ratings_prior.boost", "ratings_prior.defense_target",
    "ratings_prior.rank_map", "ratings_prior.gbdt_panel", "ratings_prior.lam_scale",
    "ratings_prior.gbdt_turn",
    "cv.n_folds", "cv.lambda_windows", "cv.validation_window", "cv.poss_buckets",
    "gbdt.features_O", "gbdt.features_D", "gbdt.features_full_O", "gbdt.features_full_D",
    "paths.figs", "paths.lambda_file",
}

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


# Blocks handed to something else whole -- the booster, mostly -- so their leaves are consumed by
# name somewhere other than this repo and must not be reported as unread.
FORWARDED = ("gbdt.params", "gbdt.params_def", "ratings_prior.season_board", "ratings_prior.cal_map")


def _unread_keys() -> set:
    """Config paths whose own name appears in no source file, with dead subtrees reported once."""
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    out, covered = set(), set()
    for path, _ in _leaves(cfg):
        dotted = ".".join(path)
        if dotted.startswith(FORWARDED) or _named(path[-1]):
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
