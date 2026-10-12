"""`openrapm data`: the data the five steps read, built in order.

    openrapm data ingest FIRST [LAST] [PHASES]    game logs, play-by-play and box scores from stats.nba.com, cached
    openrapm data stints FIRST [LAST] [PHASES]    the stints from the cached play-by-play, and each season's checks
    openrapm data panel                           the season panel (paths.panel) from the stints, box scores, roles
                                                  and bio feed
    openrapm data panel --migrate-from FILE       the season panel from one written under the names used before
                                                  2026-10-11 (done once, on 2026-10-11)

Example, from nothing: `openrapm data ingest 1997 2026 RS,PO` (days, resumable: every file is cached), then
`openrapm data stints 1997 2026 RS,PO`, then `openrapm data panel` (about half a minute).

Seasons are named by the year they end (2026 = 2025-26); phases are RS (regular season) and PO (playoffs).  numpy
and the data layer are imported inside each command, after the command line has pinned the threads (cli.py).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

DEFAULT_PHASES = "RS,PO"


def season_range(args: argparse.Namespace, cfg: dict) -> range:
    """FIRST to LAST as typed: FIRST defaults to config.yaml's first_season, LAST to FIRST."""
    first = args.first_season if args.first_season is not None else int(cfg["first_season"])
    last = args.last_season if args.last_season is not None else first
    return range(first, last + 1)


def phases(args: argparse.Namespace) -> tuple:
    """"RS,PO" -> ("RS", "PO")."""
    return tuple(args.phases.split(","))


def run_ingest(args: argparse.Namespace, cfg: dict) -> int:
    """Every season and phase asked for: game log, play-by-play and box scores, each file fetched once and cached."""
    from ..data.ingest import ingest_season
    for season in season_range(args, cfg):
        ingest_season(season, phases(args), cfg, workers=args.workers, delay=args.delay)
    return 0


def _stint_season(season: int, phase: str, cfg: dict, force: bool) -> dict:
    """One season and phase's stints built (or read, when built already and not `force`), its diagnostics printed."""
    import pandas as pd

    from ..data.stints import build_season, season_diagnostics
    started = time.time()
    build_season(season, phase, cfg, force=force)
    diagnostics = season_diagnostics(season, phase, cfg)
    diagnostics["wall_seconds"] = time.time() - started
    print(pd.Series(diagnostics).to_string())
    print()
    return diagnostics


def run_stints(args: argparse.Namespace, cfg: dict) -> int:
    """Every season and phase asked for: the stints, then the diagnostics of every season built so far in
    diagnostics_summary.parquet beside them (a season built again replaces its earlier row)."""
    import pandas as pd

    from ..data.config import resolve
    pd.set_option("display.width", 250, "display.max_columns", 40, "display.precision", 3)
    out = pd.DataFrame([_stint_season(season, phase, cfg, args.force)
                        for season in season_range(args, cfg) for phase in phases(args)])
    path = resolve(cfg, "stints") / "diagnostics_summary.parquet"
    if path.exists():
        old = pd.read_parquet(path)
        old = old[~(old.season.astype(str) + old.phase).isin(out.season.astype(str) + out.phase)]
        out = pd.concat([old, out], ignore_index=True)
    out.to_parquet(path, index=False)
    print(out.to_string(index=False))
    return 0


def _migrate(args: argparse.Namespace, cfg: dict) -> None:
    from ..data.panel import migrate_panel
    target, panel, dropped = migrate_panel(Path(args.migrate_from), cfg)
    print(f"wrote {target}: {len(panel)} rows, {panel.shape[1]} columns under the plain names, in the rows' order "
          f"of {args.migrate_from}; dropped {len(dropped)} columns no reader uses: {', '.join(dropped)}", flush=True)


def _build(args: argparse.Namespace, cfg: dict) -> None:
    from ..data.panel import write_panel
    started = time.time()
    target, panel = write_panel(cfg, force=args.force)
    print(f"wrote {target}: {len(panel)} rows, {panel.shape[1]} columns, {panel.season.nunique()} seasons "
          f"({time.time() - started:.0f}s)", flush=True)


def run_panel(args: argparse.Namespace, cfg: dict) -> int:
    """The season panel: built from the stints and box scores, or migrated from the old-named one.  An existing
    panel stops the command with one line saying so."""
    try:
        (_migrate if args.migrate_from else _build)(args, cfg)
    except FileExistsError as error:
        raise SystemExit(str(error)) from None
    return 0


def _add_seasons(parser) -> None:
    parser.add_argument("first_season", nargs="?", type=int, metavar="FIRST",
                        help="the first season (default config.yaml's first_season)")
    parser.add_argument("last_season", nargs="?", type=int, metavar="LAST", help="the last season (default FIRST)")
    parser.add_argument("phases", nargs="?", default=DEFAULT_PHASES, metavar="PHASES",
                        help="RS, PO or both, comma-separated (default %(default)s)")


def _add_ingest(steps) -> None:
    ingest = steps.add_parser(
        "ingest", help="game logs, play-by-play and box scores from stats.nba.com, cached",
        description="Fetches each season's game log and every game's play-by-play and box score into paths.raw "
                    "(data/raw).  A file already there is not fetched again, so a stopped run resumes.  With "
                    "OPENRAPM_OFFLINE=1 a missing file stops the run instead of being fetched.")
    _add_seasons(ingest)
    ingest.add_argument("--workers", type=int, default=2, help="requests in flight at once (default %(default)s)")
    ingest.add_argument("--delay", type=float, default=0.6, help="seconds between requests (default %(default)s)")
    ingest.set_defaults(run=run_ingest)


def _add_stints(steps) -> None:
    stints = steps.add_parser(
        "stints", help="the stints from the cached play-by-play, with each season's diagnostics",
        description="Builds paths.stints (data/stints): per season and phase, one row per span of play with the same "
                    "ten players on the court, its possessions, points and counters, and the shots table; prints each "
                    "season's diagnostics and gathers them in diagnostics_summary.parquet.")
    _add_seasons(stints)
    stints.add_argument("--force", action="store_true", help="rebuild stints that are already on disk")
    stints.set_defaults(run=run_stints)


def _add_panel(steps) -> None:
    panel = steps.add_parser(
        "panel", help="the season panel (paths.panel, outputs/season_panel.parquet)",
        description="Builds the season panel, one row per player, season and side, from the stints, box scores, "
                    "roles table and bio feed (src/openrapm/data/panel.py), every season from all of its games, and "
                    "writes it to paths.panel.  An existing panel is kept unless --force.  With --migrate-from it "
                    "writes the panel instead from one built under the names used before 2026-10-11: every column "
                    "renamed, the columns no reader uses dropped, the rows in their order, every value unchanged.")
    panel.add_argument("--force", action="store_true", help="replace an existing season panel")
    panel.add_argument("--migrate-from", metavar="FILE",
                       help="the old-named panel, e.g. outputs/role_panel_season.parquet")
    panel.set_defaults(run=run_panel)


def add_data(commands) -> None:
    """`openrapm data` and its three steps."""
    data = commands.add_parser("data", help="the data the ratings are built from: raw files, stints, season panel",
                               description="Builds the data the five steps read: `ingest`, then `stints`, then "
                                           "`panel`.")
    steps = data.add_subparsers(dest="data_command", required=True, metavar="DATA")
    _add_ingest(steps)
    _add_stints(steps)
    _add_panel(steps)
