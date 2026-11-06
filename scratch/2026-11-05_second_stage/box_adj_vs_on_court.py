"""Read-only: does Box Adj's three-point piece give back what the same shooting took from On adj?  Regulars of 2026."""
import contextlib
import io
import json
import runpy
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
class _Quiet(io.StringIO):
    def reconfigure(self, **_):
        pass


with contextlib.redirect_stdout(_Quiet()):
    g = runpy.run_path("scratch/2026-11-05_second_stage/box_adj_sources.py")
parts, reg, w, wcorr, threes = g["parts"], g["reg"], g["w"], g["wcorr"], g["threes"]
site = json.load(open("A:/code/spmm_site/docs/data/decompose_openrapm.json", encoding="utf-8"))["rows"]["2026"]
on = pd.DataFrame({"on_off": {r[1]: r[3][1] for r in site}, "on_def": {r[1]: r[4][1] for r in site}})
for side, label in (("O", "on_off"), ("D", "on_def")):
    piece = parts[side][threes].sum(axis=1).reindex(reg)
    other = on[label].reindex(reg)
    ok = other.notna().to_numpy()
    slope = np.polyfit(piece[ok], other[ok], 1, w=np.sqrt(w[ok]))[0]
    print(f"{'offense' if side == 'O' else 'defense'}: three-point piece against On adj, regulars on the page: correlation {wcorr(piece.where(ok), other):+.2f}, On adj per point of the piece {slope:+.2f}")
