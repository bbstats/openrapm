"""British spellings made American in comments, docstrings and argparse help text only (CLAUDE.md, "American English
only"; the cleanup of 2026-11-16).

    python scratch/2026-11-16_cleanup/american.py PATH...     rewrites each file in place, prints the count per file

Only these spans change: a Python file's comments (tokenize), its docstrings and the string literals given as `help=`,
`description=` or `epilog=` (ast); a shell file's full-line comments, line 1 kept.  Every other string is data, a
column name, a check name or a printed line another script parses, and is left alone.  A word is left alone too
where it touches `.`, `_`, `-`, a quote, a backtick or `/`, `(` or `=` after it: there it is a name,
`Targets.offence`, `defence_factors`, `--defence-factor`.  same_code.py proves the code did not change.
"""
import ast
import io
import re
import sys
import tokenize
from pathlib import Path

WORDS = {
    "offence": "offense", "offences": "offenses", "defence": "defense", "defences": "defenses",
    "centre": "center", "centres": "centers", "centred": "centered", "centring": "centering",
    "uncentred": "uncentered", "recentred": "recentered", "recentring": "recentering",
    "neighbour": "neighbor", "neighbours": "neighbors", "neighbouring": "neighboring",
    "neighbourhood": "neighborhood", "neighbourhoods": "neighborhoods",
    "behaviour": "behavior", "behaviours": "behaviors", "colour": "color", "colours": "colors",
    "coloured": "colored", "colouring": "coloring", "favour": "favor", "favours": "favors", "favoured": "favored",
    "favourite": "favorite", "favourable": "favorable", "honour": "honor", "honours": "honors",
    "catalogue": "catalog", "catalogues": "catalogs",
    "labelled": "labeled", "labelling": "labeling", "modelled": "modeled", "modelling": "modeling",
    "travelled": "traveled", "travelling": "traveling", "cancelled": "canceled", "levelled": "leveled",
    "totalled": "totaled", "signalled": "signaled",
    "grey": "gray", "artefact": "artifact", "artefacts": "artifacts", "judgement": "judgment",
    "judgements": "judgments", "analyse": "analyze", "analysed": "analyzed", "analysing": "analyzing",
    "whilst": "while", "learnt": "learned",
}
for stem in ("standard", "penal", "minim", "maxim", "memor", "recogn", "summar", "optim", "real", "regular",
             "normal", "initial", "stabil", "random", "categor", "character", "util", "organ", "priorit", "final",
             "general", "visual", "apolog", "emphas", "special", "local", "central", "neutral", "parametr"):
    for british, american in (("ise", "ize"), ("ised", "ized"), ("ises", "izes"), ("ising", "izing"),
                              ("isation", "ization"), ("isations", "izations")):
        WORDS[stem + british] = stem + american
WORDS.pop("emphasises", None)            # emphasis/emphases are nouns; only the verb forms change
PATTERN = re.compile(r"\b(" + "|".join(sorted(WORDS, key=len, reverse=True)) + r")\b", re.IGNORECASE)
BEFORE = set("._-/\"'`$:\\")
AFTER = set("_-\"'`(=/\\")


def americanize(text: str) -> tuple[str, int]:
    count = 0

    def swap(match):
        nonlocal count
        word, start, end = match.group(0), match.start(), match.end()
        if (start and text[start - 1] in BEFORE) or (end < len(text) and text[end] in AFTER) or word.isupper():
            return word
        new = WORDS[word.lower()]
        count += 1
        return new.capitalize() if word[0].isupper() else new
    return PATTERN.sub(swap, text), count


def python_spans(source: str) -> list:
    """(start, end) character offsets into `source` of every comment, docstring and help-text literal."""
    lines = source.splitlines(keepends=True)
    starts = [0]
    for line in lines:
        starts.append(starts[-1] + len(line))

    def at(lineno, col_bytes):
        line = lines[lineno - 1]
        return starts[lineno - 1] + len(line.encode("utf-8")[:col_bytes].decode("utf-8", "ignore"))

    spans = []
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.COMMENT:
            (r1, c1), (r2, c2) = token.start, token.end
            spans.append((starts[r1 - 1] + c1, starts[r2 - 1] + c2))
    for node in ast.walk(ast.parse(source)):
        targets = []
        body = getattr(node, "body", None)
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and body \
                and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                and isinstance(body[0].value.value, str):
            targets.append(body[0].value)
        if isinstance(node, ast.keyword) and node.arg in ("help", "description", "epilog") \
                and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            targets.append(node.value)
        for target in targets:
            spans.append((at(target.lineno, target.col_offset), at(target.end_lineno, target.end_col_offset)))
    return spans


def rewrite_python(source: str) -> tuple[str, int]:
    out, last, total = [], 0, 0
    for start, end in sorted(set(python_spans(source))):
        if start < last:
            continue
        new, count = americanize(source[start:end])
        out += [source[last:start], new]
        last, total = end, total + count
    out.append(source[last:])
    return "".join(out), total


def rewrite_shell(source: str) -> tuple[str, int]:
    lines, total = source.split("\n"), 0
    for k, line in enumerate(lines):
        if k and line.lstrip().startswith("#"):
            lines[k], count = americanize(line)
            total += count
    return "\n".join(lines), total


def main() -> int:
    grand = 0
    for name in sys.argv[1:]:
        path = Path(name)
        raw = path.read_bytes()
        crlf = b"\r\n" in raw
        source = raw.decode("utf-8").replace("\r\n", "\n")
        new, count = (rewrite_python if path.suffix == ".py" else rewrite_shell)(source)
        if count:
            path.write_bytes((new.replace("\n", "\r\n") if crlf else new).encode("utf-8"))
            print(f"{count:5d}  {name}")
            grand += count
    print(f"{grand} words in all")
    return 0


if __name__ == "__main__":
    sys.exit(main())
