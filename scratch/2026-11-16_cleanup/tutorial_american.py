"""examples/tutorial.py's printed sentences in American spelling: "offence"/"defence" as words of prose, after a space
or an opening quote and before a space or punctuation.  Keys such as ['offence'] and names such as "offence", are
not touched.  Prints each changed line."""
import re
from pathlib import Path

PATH = Path("examples/tutorial.py")
WORD = re.compile(r'(?<=[ "])(offence|defence)(?=[ ;.,:])')
raw = PATH.read_bytes().decode("utf-8")
out = []
for number, line in enumerate(raw.splitlines(keepends=True), 1):
    new = line if line.lstrip().startswith("#") else WORD.sub(lambda m: m.group(1).replace("ce", "se"), line)
    if new != line:
        print(number, new.rstrip())
    out.append(new)
PATH.write_bytes("".join(out).encode("utf-8"))
