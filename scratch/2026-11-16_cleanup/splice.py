"""Put a new head on a doc, keeping the old text from a marker on: python splice.py DOC HEAD MARKER [DEMOTE]

With DEMOTE, every heading kept from the old text goes one level down, under the new head's last section."""
import sys
from pathlib import Path

doc, head, marker = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
old = doc.read_text(encoding="utf-8").replace("\r\n", "\n")
rest = old[old.index(marker):]
if len(sys.argv) > 4:
    rest = "\n".join("#" + line if line.startswith("#") else line for line in rest.split("\n"))
doc.write_text(head.read_text(encoding="utf-8").replace("\r\n", "\n") + rest, encoding="utf-8", newline="\n")
print(f"{doc}: {len(old.splitlines())} -> {len(doc.read_text(encoding='utf-8').splitlines())} lines")
