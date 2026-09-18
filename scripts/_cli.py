"""The scripts' command line: read a flag, and refuse a flag nobody asked for.

Every script here used to carry its own copy of

    def _flag(name, default=None):
        hit = [a for a in sys.argv[1:] if a.startswith(f"--{name}=")]
        return hit[0].split("=", 1)[1] if hit else default

nineteen copies of it, and not one of them looked at the flags it was NOT given.  So

    python scripts/62_single_year_board.py --exclude_neighbors=1 ...

ran happily with the American spelling, the prior saw the two seasons it was about to be scored on,
and the number that came out looked exactly like a number that had not.  That is the shape of defect
this project's own DECISIONS.md keeps a list of: not a crash, a confident wrong answer.

`check_flags()` compares what was typed against what the script understands, and stops on anything
left over.  Put it at the top of `main()`, before the expensive work:

    from _cli import flag, switch, check_flags

    def main():
        check_flags()
        tag = flag("tag", "run")

What the script understands is read from the script's own source, not from the flags it has got
round to reading yet -- almost every script here reads its flags scattered through `main()`, so a
check that only knew about the reads that had already happened would reject the ones below it.  The
scan finds every literal name handed to `flag`/`switch`, and to any local wrapper that forwards its
first argument to one of them (`_list`, `_int`, `_blend_flag` and friends are all found this way).
Names computed at run time are covered by the second half: `flag` and `switch` also record every
name they are actually asked for, and the two sets are unioned.

It is deliberately a no-op unless the calling module is `__main__`: scripts 70-72 exec script 62's
module body to borrow its loaders, and 62's flags are not theirs to validate.
"""
import ast
import sys
from pathlib import Path

_REQUESTED: set = set()


def flag(name, default=None):
    """The value of `--name=value`, or `default`.  Records `name` as one this script understands."""
    _REQUESTED.add(name)
    if f"--{name}" in sys.argv[1:]:
        raise SystemExit(f"--{name} needs a value: write --{name}=<value>.")
    hit = [a for a in sys.argv[1:] if a.startswith(f"--{name}=")]
    return hit[0].split("=", 1)[1] if hit else default


def switch(name) -> bool:
    """Whether a bare `--name` was given.  Records it the same way `flag` does.

    A value is refused rather than ignored.  `scripts/49_role_panel.py --season=2026` used to read as
    "no --season at all": it rebuilt the block panel over the tracked one and said nothing.
    """
    _REQUESTED.add(name)
    if any(a.startswith(f"--{name}=") for a in sys.argv[1:]):
        raise SystemExit(f"--{name} is a switch and takes no value: write --{name} on its own, or "
                         f"leave it off.")
    return f"--{name}" in sys.argv[1:]


def requested() -> frozenset:
    """Every flag name asked for so far.  For tests, and for a script that prints its own usage."""
    return frozenset(_REQUESTED)


def _callee(node) -> str | None:
    f = node.func
    return f.id if isinstance(f, ast.Name) else (f.attr if isinstance(f, ast.Attribute) else None)


def declared_in(path) -> set:
    """Every flag name written literally in a script: what it understands, read off its own source.

    A wrapper counts as a flag reader if it passes its own first parameter through to one -- which is
    how `_list("splits", ...)` and `_int("k", 4)` are found without naming them here.
    """
    tree = ast.parse(Path(path).read_text(encoding="utf-8"), filename=str(path))
    functions = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    readers = {"flag", "_flag", "switch", "_switch"}
    for _ in range(len(functions) + 1):                 # transitive, and it always terminates
        grew = False
        for fn in functions:
            if fn.name in readers or not fn.args.args:
                continue
            first = fn.args.args[0].arg
            for n in ast.walk(fn):
                if (isinstance(n, ast.Call) and _callee(n) in readers and n.args
                        and isinstance(n.args[0], ast.Name) and n.args[0].id == first):
                    readers.add(fn.name)
                    grew = True
                    break
        if not grew:
            break
    return {n.args[0].value for n in ast.walk(tree)
            if isinstance(n, ast.Call) and _callee(n) in readers and n.args
            and isinstance(n.args[0], ast.Constant) and isinstance(n.args[0].value, str)}


def check_flags(*also: str) -> None:
    """Stop on any `--flag` in argv that no `flag()`/`switch()` call asked for.

    What the script understands is its source's literal flag names (`declared_in`) plus whatever has
    already been read, so this may sit at the top of `main()` above every read.  `also` is for a name
    that is neither -- built at run time on a branch not yet taken.
    """
    frame = sys._getframe(1).f_globals
    if frame.get("__name__") != "__main__":   # a module exec'd by another script: not its argv to police
        return
    known = _REQUESTED | set(also) | declared_in(frame["__file__"])
    given = {a[2:].split("=", 1)[0] for a in sys.argv[1:] if a.startswith("--") and a != "--"}
    unknown = sorted(given - known)
    if unknown:
        raise SystemExit(
            f"unknown flag(s) {', '.join('--' + u for u in unknown)}.\n"
            f"{sys.argv[0]} understands: {', '.join('--' + k for k in sorted(known))}\n"
            "A misspelled flag used to be ignored silently, which is how a run can look right and be "
            "wrong; it is an error now.")
