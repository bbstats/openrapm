"""Each check script's name and the first paragraph of its docstring, one line each, for PIPELINE.md's table."""
import ast
from pathlib import Path

for path in sorted(Path("scripts/checks").glob("*.py"), key=lambda p: int(p.name.split("_")[0])):
    doc = ast.get_docstring(ast.parse(path.read_text(encoding="utf-8"))) or ""
    print(f"{path.name} | {' '.join(doc.split(chr(10) * 2)[0].split())[:260]}")
