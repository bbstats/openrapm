"""Did an edit change anything but comments, docstrings and help text?

    python scratch/2026-11-16_cleanup/same_code.py BASE [PATH...]

For every .py file that differs between the commit BASE and the working tree (or each PATH given): the syntax tree
before and after, with every docstring and the literal text of every `help=`, `description=` and `epilog=` argument
blanked, must be equal; comments are not in the tree at all.  For every .sh file: the lines, with full-line comments
and blank lines dropped and line 1 kept, must be equal.  Prints one line per file and exits 1 when any differs.
"""
import ast
import subprocess
import sys
from pathlib import Path

TEXT_ARGS = {"help", "description", "epilog"}


class Blank(ast.NodeTransformer):
    def _strip_doc(self, node):
        body = getattr(node, "body", None)
        if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                and isinstance(body[0].value.value, str):
            body[0].value = ast.Constant("")
        return node

    def visit_Module(self, node):
        self.generic_visit(node)
        return self._strip_doc(node)

    visit_FunctionDef = visit_AsyncFunctionDef = visit_ClassDef = visit_Module

    def visit_keyword(self, node):
        self.generic_visit(node)
        if node.arg in TEXT_ARGS:
            node.value = blank_text(node.value)
        return node


def blank_text(value):
    """A string's literal parts emptied; an f-string keeps its expressions; anything else is left alone."""
    if isinstance(value, ast.Constant) and isinstance(value.value, str):
        return ast.Constant("")
    if isinstance(value, ast.JoinedStr):
        return ast.JoinedStr([part for part in value.values if not isinstance(part, ast.Constant)])
    if isinstance(value, ast.BinOp):
        return ast.BinOp(blank_text(value.left), value.op, blank_text(value.right))
    return value


def python_shape(text: str) -> str:
    return ast.dump(Blank().visit(ast.parse(text)))


def shell_shape(text: str) -> list:
    lines = text.replace("\r\n", "\n").split("\n")
    return lines[:1] + [line.rstrip() for line in lines[1:] if line.strip() and not line.lstrip().startswith("#")]


def old_text(base: str, path: str) -> str | None:
    done = subprocess.run(["git", "show", f"{base}:{path}"], capture_output=True)
    return done.stdout.decode("utf-8") if done.returncode == 0 else None


def main() -> int:
    base, paths = sys.argv[1], sys.argv[2:]
    if not paths:
        listed = subprocess.run(["git", "diff", "--name-only", base, "--", "*.py", "*.sh"], capture_output=True,
                                text=True, check=True).stdout.split()
        paths = [p for p in listed if not p.startswith("scratch/")]
    bad = 0
    for path in paths:
        before, now = old_text(base, path), Path(path).read_text(encoding="utf-8") if Path(path).exists() else None
        if before is None or now is None:
            print(f"ADDED OR REMOVED  {path}")
            bad += 1
            continue
        shape = python_shape if path.endswith(".py") else shell_shape
        same = shape(before) == shape(now)
        print(f"{'same code' if same else 'CODE CHANGED':16s}  {path}")
        bad += not same
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
