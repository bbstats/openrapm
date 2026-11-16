"""Each module of src/openrapm and the first sentence of its docstring, one line each, for PIPELINE.md."""
import ast
from pathlib import Path

SRC = Path("src/openrapm")
for path in sorted(SRC.rglob("*.py")):
    if "__pycache__" in path.parts:
        continue
    doc = ast.get_docstring(ast.parse(path.read_text(encoding="utf-8"))) or ""
    first = " ".join(doc.split("\n\n")[0].split())
    print(f"{path.relative_to(SRC).as_posix()} | {first[:200]}")
