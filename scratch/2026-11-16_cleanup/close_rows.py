"""experiments/runs.csv: the seven rows of the three box-score priors removed on 2026-11-16 read not adopted.  Only the
verdict field of those rows changes; every other byte of the file stays as it was."""
import csv
import io
from pathlib import Path

PATH = Path("experiments/runs.csv")
ROWS = {"114", "115", "116", "117", "118", "121", "122"}
VERDICT = "Not adopted 2026-11-16: lost the within-season test; code removed, kept at the tag archive/research-2026-11"

raw = PATH.read_bytes()
crlf = b"\r\n" in raw
text = raw.decode("utf-8")
lines = text.splitlines(keepends=True)
header = next(csv.reader([lines[0]]))
at = header.index("verdict")
changed = 0
for k, line in enumerate(lines):
    fields = next(csv.reader([line]))
    if fields and fields[0] in ROWS:
        assert fields[at] == "Pending", (fields[0], fields[at])
        fields[at] = VERDICT
        out = io.StringIO()
        csv.writer(out, lineterminator="\r\n" if crlf else "\n").writerow(fields)
        assert len(next(csv.reader([out.getvalue()]))) == len(header)
        lines[k] = out.getvalue()
        changed += 1
assert changed == len(ROWS), changed
PATH.write_bytes("".join(lines).encode("utf-8"))
print(f"{changed} rows now read: {VERDICT}")
