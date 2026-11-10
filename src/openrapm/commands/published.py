"""The published tables' name, kept from a command typed by hand.

`openrapm build`, `rate` and `wins` write under the name "product" unless told another, and `shrink` and `swap` under
the name of the table they read.  The morning job (scripts/daily_update.sh) rates the season in progress into those
tables and publishes from them with nobody looking.  So a command typed during the day that writes the same name
changes what the site shows the next morning, and nothing says so: `openrapm rate --seasons 2026` leaves a one-season
season_ratings_product_step3.parquet, the table the job writes the season in progress into and reads its held prior
scales from.

Where the morning job publishes from, the name is therefore refused unless the caller says the published tables are
what it means to write: the environment variable OPENRAPM_WRITE_PUBLISHED=1, which the morning job sets, and which a
full build after an adopted change is typed with.  "Where the morning job publishes from" is an outputs folder that
holds daily/last.txt, the job's own record of how its last morning ended.  A fresh clone has none, so README's
`openrapm build` runs there as written, and so do the tests in their temporary folders.

Example: in A:/code/spmm, `openrapm build --seasons 2026 --features hustle` stops before it reads anything.  The same
command with `--name trial` runs, and so does `OPENRAPM_WRITE_PUBLISHED=1 openrapm build`.

A table copied over the published one by hand is not seen here; an adoption that copies takes the morning job's lock
(scripts/daily_lock.sh) and says so in its script.
"""
from __future__ import annotations

import os

from ..data.config import resolve

PUBLISHED_NAME = "product"
WRITE_PUBLISHED_ENV = "OPENRAPM_WRITE_PUBLISHED"
JOB_RECORD = ("daily", "last.txt")       # under paths.outputs: how the morning job's last run ended


def refuse_published_name(cfg: dict, name: str) -> None:
    """Stop a command that is about to write the published tables' name where the morning job publishes from, unless
    OPENRAPM_WRITE_PUBLISHED=1.  Called first by every command that writes a ratings table, before anything is read."""
    if name != PUBLISHED_NAME or os.environ.get(WRITE_PUBLISHED_ENV, "").strip() == "1":
        return
    record = resolve(cfg, "outputs").joinpath(*JOB_RECORD)
    if record.exists():
        raise SystemExit(
            f"the name {PUBLISHED_NAME!r} is the published tables': the morning job publishes from "
            f"{record.parents[1]} (it wrote {record.name} there), and this command would write them.  Give another "
            f"name with --name, or set {WRITE_PUBLISHED_ENV}=1 for a build meant to replace the published tables "
            f"(src/openrapm/commands/published.py).")
