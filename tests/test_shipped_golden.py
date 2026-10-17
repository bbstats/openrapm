"""2026 rated again from the data on disk equals the published step-3 table's 2026 rows, value for value.

`openrapm rate --seasons 2026` runs steps 1-3 for one season: the labels from every other season, the box-score prior
trained on them, the ridge around it.  Each season is rated on its own, so its rows must equal the 2026 rows of the
thirty-season run that built the published table (outputs/season_ratings_product_bags10_step3.parquet, a frozen
copy of the step-3 table `openrapm build` wrote on 2026-10-17 with the booster settings of check 152's guarded search,
the hustle inputs less three-point % on offence and 10 bags, experiment 78; before then
season_ratings_product_nothreepct_step3.parquet, season_ratings_product_tune2_step3.parquet,
season_ratings_product_cols06_step3.parquet,
season_ratings_product_nocareer_step3.parquet, season_ratings_product_fromdebut_step3.parquet
and season_ratings_product_weightfix_pre_swap.parquet).  The run
writes into a temporary folder: a copy of config.yaml whose paths.outputs points there, every other path the real one.

Slow (a few minutes) and local: it needs the stints, the design pieces, the season panel and the published table, so
`pytest -m slow` runs it and a fresh clone skips it.
"""
import os
import subprocess
import sys

import pytest
import yaml

from openrapm.commands.cli import THREAD_VARIABLES
from openrapm.commands.compare import compare
from openrapm.data.config import DEFAULT_PATHS, ROOT, load_config, resolve

PUBLISHED = "season_ratings_product_bags10_step3.parquet"


def _config_writing_into(folder) -> str:
    """A config.yaml like the one in use, every path absolute, paths.outputs the given folder."""
    cfg = load_config()
    paths = {key: str(resolve(cfg, key)) for key in {*DEFAULT_PATHS, *cfg.get("paths", {})}}
    written = {key: value for key, value in cfg.items() if not key.startswith("_")}
    written["paths"] = {**paths, "outputs": str(folder)}
    path = folder / "config.yaml"
    path.write_text(yaml.safe_dump(written, sort_keys=False), encoding="utf-8")
    return str(path)


@pytest.mark.slow
@pytest.mark.needs_data(f"outputs/{PUBLISHED}", "panel", "stints/2026_RS.parquet")
def test_2026_rated_again_is_the_published_step_3(tmp_path):
    config = _config_writing_into(tmp_path)
    one_thread = {**os.environ, **{variable: "1" for variable in THREAD_VARIABLES}}   # as the published build ran
    result = subprocess.run([sys.executable, "-m", "openrapm", "--config", config, "rate", "--seasons", "2026",
                             "--name", "golden"], capture_output=True, text=True, encoding="utf-8", cwd=ROOT,
                            env=one_thread, timeout=3600)
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
    published = resolve(load_config(), "outputs") / PUBLISHED
    verdict, notes, rows, _ = compare(tmp_path / "season_ratings_golden_step3.parquet", published, season=2026)
    assert verdict.startswith("IDENTICAL"), notes
    assert rows > 0, "no 2026 rows were compared"
