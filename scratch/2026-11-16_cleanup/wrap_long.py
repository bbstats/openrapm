"""Lines of comment or docstring the spelling sweep pushed past 120 characters, wrapped at their last space before 120.

    python wrap_long.py PATH:LINE...

A comment line is wrapped into a second comment line with the same indentation; a docstring line into a second line
with the same indentation.  Only lines whose text at the break is comment or docstring prose are touched; same_code.py
checks the result."""
import sys
from pathlib import Path

for item in sys.argv[1:]:
    name, number = item.rsplit(":", 1)
    path = Path(name)
    raw = path.read_bytes()
    crlf = b"\r\n" in raw
    lines = raw.decode("utf-8").replace("\r\n", "\n").split("\n")
    k = int(number) - 1
    line = lines[k]
    indent = line[:len(line) - len(line.lstrip())]
    prefix = indent + ("# " if line.lstrip().startswith("#") else "")
    cut = line.rfind(" ", 0, 120)
    assert cut > len(prefix), item
    lines[k:k + 1] = [line[:cut], prefix + line[cut + 1:]]
    text = "\n".join(lines)
    path.write_bytes((text.replace("\n", "\r\n") if crlf else text).encode("utf-8"))
    print("wrapped", item)
