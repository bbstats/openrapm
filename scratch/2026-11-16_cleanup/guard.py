"""Before each side of the proof: which code this process imports, and a checksum of the inputs both sides read.

    python guard.py before      stops unless openrapm comes from A:/code/spmm_before and openrapm.game_prior exists
    python guard.py after       stops unless openrapm comes from A:/code/spmm_cleanup and openrapm.game_prior is gone

The input checksums are written to outputs/cleanup_proof/inputs_<side>.txt; the two files must be equal, or the data
moved between the sides (the morning job rewrites 2027's stints, the panel and the published tables).
"""
import hashlib
import importlib.util
import sys
from pathlib import Path

MAIN = Path("A:/code/spmm")
TREES = {"before": Path("A:/code/spmm_before"), "after": Path("A:/code/spmm_cleanup")}
INPUTS = ["outputs/season_panel.parquet", "outputs/season_panel_so_far.parquet", "data/cache/roles_RSPO.parquet",
          "data/stints/2027_RS.parquet", "data/stints/2027_PO.parquet", "outputs/season_ratings_product_step3.parquet",
          "outputs/season_ratings_product_step4.parquet", "outputs/season_ratings_product.parquet",
          "outputs/season_ratings_product_regular.parquet"]

side = sys.argv[1]
import openrapm  # noqa: E402

here = Path(openrapm.__file__).resolve()
assert TREES[side].resolve() in here.parents, f"openrapm comes from {here}, not {TREES[side]}"
has_network = importlib.util.find_spec("openrapm.game_prior") is not None
assert has_network == (side == "before"), f"openrapm.game_prior present: {has_network} on the {side} side"
lines = []
for name in INPUTS:
    path = MAIN / name
    digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else "missing"
    lines.append(f"{digest}  {name}")
out = MAIN / "outputs" / "cleanup_proof" / f"inputs_{side}.txt"
out.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"{side}: openrapm from {here.parent}; game_prior present: {has_network}")
print("\n".join(lines))
