"""`openrapm glossary`: GLOSSARY.md's table of the prior's columns, written from features.py.

GLOSSARY.md is hand-written prose around one generated block, the table of every column the prior can read
(`features.glossary_markdown`), which sits between these two lines:

    <!-- generated: features -->
    <!-- end generated -->

`openrapm glossary` rewrites what lies between them and leaves every other line alone; `openrapm glossary --check`
only compares, and exits 1 when the block and the catalogue differ (tests/test_features.py makes the same
comparison, so a renamed or added feature cannot leave the glossary behind).
"""
from __future__ import annotations

from pathlib import Path

from ..data.config import ROOT
from ..features import glossary_markdown

__all__ = ["GLOSSARY", "START", "END", "generated_block", "with_block", "run_glossary"]

GLOSSARY = ROOT / "GLOSSARY.md"
START = "<!-- generated: features -->"
END = "<!-- end generated -->"


def _span(text: str) -> tuple[int, int]:
    """(where the block starts, where it ends): just after the START line and at the END marker."""
    start, end = text.find(START), text.find(END)
    if start < 0 or end < start:
        raise ValueError(f"GLOSSARY.md needs the lines {START} and {END}, in that order")
    return start + len(START) + 1, end


def generated_block(text: str) -> str:
    """The text between the two marker lines, line endings read as LF."""
    text = text.replace("\r\n", "\n")
    start, end = _span(text)
    return text[start:end]


def with_block(text: str, block: str) -> str:
    """`text` with the lines between the markers replaced by `block` (which ends with a line break)."""
    text = text.replace("\r\n", "\n")
    start, end = _span(text)
    return text[:start] + block + text[end:]


def run_glossary(check: bool = False, path: Path = GLOSSARY) -> int:
    """Write the catalogue's table into the glossary, or with `check` report whether it is already there."""
    text = path.read_text(encoding="utf-8")
    table = glossary_markdown()
    if generated_block(text) == table:
        print(f"{path.name}: the feature table matches features.py", flush=True)
        return 0
    if check:
        print(f"{path.name}: the feature table differs from features.py; run `openrapm glossary`", flush=True)
        return 1
    path.write_text(with_block(text, table), encoding="utf-8", newline="\n")
    print(f"{path.name}: the feature table rewritten from features.py", flush=True)
    return 0
