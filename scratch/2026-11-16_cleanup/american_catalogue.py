"""The catalogue's words in American English: features.py from `def _linear` to `CATALOGUE = _catalogue()`, the text
GLOSSARY.md's generated table and the reports print.  The scale value "centred" becomes "centered"; every other word
by american.py's rule, so `_uncentred` column names and other names keep their spelling."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from american import americanize  # noqa: E402

p = Path("src/openrapm/features.py")
raw = p.read_bytes()
crlf = b"\r\n" in raw
t = raw.decode("utf-8").replace("\r\n", "\n")
start, end = t.index("def _linear("), t.index("CATALOGUE = _catalogue()")
region, count = americanize(t[start:end].replace('"centred"', '"centered"'))
t = t[:start] + region + t[end:]
t = t.replace('    scale: str          # "centred" or "level"', '    scale: str          # "centered" or "level"')
p.write_bytes((t.replace("\n", "\r\n") if crlf else t).encode("utf-8"))
print(f"{count} words, and the scale value, in {p}")
