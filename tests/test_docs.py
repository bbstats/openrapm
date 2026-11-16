"""The docs say what ships.  Each test here fails when the code moves on and a doc does not: a check or a module the
docs leave out, a shipped setting PIPELINE.md does not name, an incumbent the glossary and the handoff no longer
name.  The cleanup of 2026-11-16 found every one of these stale at once."""
import re
from pathlib import Path

from openrapm.settings import PRODUCT

ROOT = Path(__file__).resolve().parents[1]


def _read(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


def test_every_check_is_in_pipelines_table():
    pipeline = _read("PIPELINE.md")
    missing = [path.name for path in sorted((ROOT / "scripts" / "checks").glob("*.py"))
               if f"| `{path.name}` |" not in pipeline]
    assert not missing, f"PIPELINE.md's table of checks leaves out {missing}"


def test_every_module_is_named_in_the_readme_or_pipeline():
    docs = _read("README.md") + _read("PIPELINE.md")
    src = ROOT / "src" / "openrapm"
    missing = []
    for path in sorted(src.rglob("*.py")):
        relative = path.relative_to(src).as_posix()
        if path.name in ("__init__.py", "__main__.py") or "__pycache__" in path.parts:
            continue
        if relative not in docs:
            missing.append(relative)
    assert not missing, f"README.md and PIPELINE.md never name {missing}"


def test_pipeline_names_every_shipped_setting_it_describes():
    """The values a reader of PIPELINE.md acts on, read from settings.py: a setting changed by an adoption must be
    changed in the guide in the same commit."""
    pipeline = " ".join(_read("PIPELINE.md").split())         # a value may wrap across a line
    prior, shrink = PRODUCT.prior, PRODUCT.shrink
    wanted = [prior.feature_set, Path(prior.booster_file).name, prior.second_prior, prior.second_prior_inputs,
              PRODUCT.targets.offence, PRODUCT.targets.defence, shrink.extrapolate_folds_tag,
              *shrink.folds_tag.split(","), f"{PRODUCT.ridge.penalty:,.0f}", f"{PRODUCT.labels.offense_penalty:,.0f}",
              *(f"{factor:g}" for _, factor in shrink.defence_factors), *(f"{f:g}" for f in shrink.within_factors),
              f"over {prior.second_prior_halflife:g} seasons"]
    missing = [value for value in wanted if value not in pipeline]
    assert not missing, f"PIPELINE.md does not name the shipped {missing}"


def _chain_default(name: str) -> str:
    text = _read("scripts/experiment_chain.sh")
    return re.search(rf"^{name}=\$\{{{name}:-([^}}]+)\}}", text, re.M).group(1)


def test_the_glossary_and_the_handoff_name_the_chains_incumbent_and_noise_control():
    incumbent, control = _chain_default("INC_NAME"), _chain_default("CONTROL_NAME")
    for doc in ("GLOSSARY.md", "notes/HANDOFF.md"):
        text = _read(doc)
        assert f"`{incumbent}`" in text and f"`{control}`" in text, (doc, incumbent, control)


def test_no_doc_a_reader_follows_names_a_retired_baseline_as_todays():
    """The incumbents before today's, named as today's: CONTRIBUTING.md kept zones139 for a week after it was
    replaced."""
    incumbent = _chain_default("INC_NAME")
    for doc in ("README.md", "CONTRIBUTING.md", "PIPELINE.md", "GLOSSARY.md"):
        for name in ("zones139", "noise_zones139", "mid136", "creation110", "weightfix", "bags10"):
            if name == incumbent:
                continue
            assert not re.search(rf"\b(incumbent|noise control)\b[^.\n]{{0,40}}`{name}`", _read(doc)), (doc, name)
