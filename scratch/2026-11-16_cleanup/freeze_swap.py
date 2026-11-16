"""Freeze step 5 again: crossfit80's golden step-4 tables through today's step 5 at share 0.5, one thread, into
A:/code/spmm_verify/golden/swap141/.  tests/test_swap_adjust.py reads them.  Refuses to overwrite.

    PYTHONPATH=src OPENRAPM_CONFIG=... python scratch/2026-11-16_cleanup/freeze_swap.py
"""
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd
import threadpoolctl

from openrapm.commands.build import swap_seasons
from openrapm.data.config import load_config
from openrapm.settings import PRODUCT, TEST

GOLDEN = Path("A:/code/spmm_verify/golden")
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else GOLDEN / "swap141"
OUT.mkdir(exist_ok=True)
cfg = load_config()
for name, settings in (("product", PRODUCT), ("test", TEST)):
    path = OUT / f"{name}_step5.parquet"
    assert not path.exists(), path
    with threadpoolctl.threadpool_limits(1):
        got, _, _ = swap_seasons(cfg, pd.read_parquet(GOLDEN / "crossfit80" / f"{name}_step4.parquet"),
                                 replace(settings, swap=replace(settings.swap, applied_share=0.5)))
    got.to_parquet(path, index=False)
    print(f"{path}: {len(got)} rows")
