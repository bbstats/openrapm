"""`openrapm data`: the data the five steps read, built in order.

    openrapm data ingest FIRST [LAST] [PHASES]    game logs, play-by-play and box scores from stats.nba.com, cached
    openrapm data ingest 2027 2027 RS --refresh   the season in progress: its game log fetched again, through
                                                  yesterday in US Eastern time, and every new game's files
    openrapm data stints FIRST [LAST] [PHASES]    the stints from the cached play-by-play, and each season's checks
    openrapm data events [FIRST] [LAST]           the play-by-play parts of four box counts, per game, player and
                                                  phase (paths.event_counts), each season held against its box score
    openrapm data careers                         the seasons before the first one of every veteran the roles table
                                                  sees (data/careers.py), from stats.nba.com, cached
    openrapm data dashboards --probe              a few seasons of the hustle, tracking, play-type and matchup
                                                  dashboards (data/dashboards.py), and how many rated players each
                                                  covers; nothing reads them yet
    openrapm data hustle-games [FIRST] [LAST]     the hustle numbers game by game (BoxScoreHustleV2, 2017 on), so a
                                                  within-season fold can rebuild them from its fit games
    openrapm data tracking-games [FIRST] [LAST]   the same for the per-game tracking box score (BoxScorePlayerTrackV3,
                                                  2014 on)
    openrapm data matchup-games [FIRST] [LAST]    the same for the per-game matchups, who guarded whom
                                                  (BoxScoreMatchupsV3, 2018 on)
    openrapm data player-games [FIRST] [LAST]     each player's games, one row per player and game, from the stored
                                                  design pieces, game logs, box scores and hustle box scores
                                                  (paths.cache/player_games): what the multi-stat network reads
    openrapm data panel                           the season panel (paths.panel) from the stints, box scores, roles,
                                                  bio feed, event counts and careers
    openrapm data panel --migrate-from FILE       the season panel from one written under the names used before
                                                  2026-10-11 (done once, on 2026-10-11)
    openrapm data panel --so-far 0.25,0.5,0.75    each season so far at those shares of its regular season
                                                  (paths.panel_so_far), the box-score prior's part-season rows

Example, from nothing: `openrapm data ingest 1997 2026 RS,PO` (days, resumable: every file is cached), then
`openrapm data stints 1997 2026 RS,PO`, then `openrapm data events 1997 2026` (minutes), then `openrapm data careers`
(about 13 minutes, resumable), then `openrapm data panel` (about a minute).

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
        ingest_season(season, phases(args), cfg, workers=args.workers, delay=args.delay, refresh=args.refresh)
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
    from ..data.ingest import played_phases
    pd.set_option("display.width", 250, "display.max_columns", 40, "display.precision", 3)
    out = pd.DataFrame([_stint_season(season, phase, cfg, args.force)
                        for season in season_range(args, cfg) for phase in played_phases(season, cfg, phases(args))])
    path = resolve(cfg, "stints") / "diagnostics_summary.parquet"
    if path.exists():
        old = pd.read_parquet(path)
        old = old[~(old.season.astype(str) + old.phase).isin(out.season.astype(str) + out.phase)]
        out = pd.concat([old, out], ignore_index=True)
    out.to_parquet(path, index=False)
    print(out.to_string(index=False))
    return 0


def event_check(totals, box):
    """Each season's parts against its box score (regular season and playoffs), one row per season: what the parts
    add up to minus the box score's count, as a share of the box score's count.

        defensive rebounds   the four rebound parts against the box score's defensive rebounds (a rebound whose row
                             before is not a miss is in no part, so a little below 0 is expected)
        steals               bad-pass plus lost-ball steals against steals
        points               assisted plus unassisted field-goal points against 2 x made twos + 3 x made threes
        threes attempted     corner plus above-the-break against threes attempted
        threes made          assisted plus unassisted against made threes
        blocks               rim, mid-range and three-point blocks against blocks
        assists              the four assist zones against assists (an assist whose passer matches no single name
                             is in no zone, so a little below 0 is expected)

    and three shares, which have no box-score total to meet: Russell blocks of blocks, stolen turnovers of turnovers,
    offensive fouls of personal fouls.
    """
    import pandas as pd

    from ..data.event_counts import EVENT_COUNTS
    box = box.groupby("season")[["drb", "stl", "fg2m", "fg3m", "fg3_miss", "blk", "ast", "tov", "pf"]].sum()
    parts = {"defensive rebounds": (EVENT_COUNTS[0:4], box.drb), "steals": (EVENT_COUNTS[4:6], box.stl),
             "points": (EVENT_COUNTS[6:8], 2 * box.fg2m + 3 * box.fg3m),
             "threes attempted": (EVENT_COUNTS[8:10], box.fg3m + box.fg3_miss),
             "threes made": (EVENT_COUNTS[10:12], box.fg3m), "blocks": (EVENT_COUNTS[13:16], box.blk),
             "assists": (EVENT_COUNTS[18:22], box.ast)}
    out = pd.DataFrame({name: totals[columns].sum(axis=1) / count.reindex(totals.index) - 1.0
                        for name, (columns, count) in parts.items()})
    for name, column, count in (("Russell share of blocks", EVENT_COUNTS[12], box.blk),
                                ("stolen share of turnovers", EVENT_COUNTS[17], box.tov),
                                ("offensive share of fouls", EVENT_COUNTS[16], box.pf)):
        out[name] = totals[column] / count.reindex(totals.index)
    return out


def run_events(args: argparse.Namespace, cfg: dict) -> int:
    """Every season asked for: the play-by-play counts, self-creation's counts (data/creation_counts.py) and the
    fouls, goaltends, shots blocked and three-shot trips (data/foul_counts.py) cached (a cached season is read, unless
    --force), then each season's parts against its box score (`event_check`)."""
    import pandas as pd

    from ..data.box_scores import season_box
    from ..data.creation_counts import build_creation_counts
    from ..data.event_counts import build_event_counts
    from ..data.foul_counts import build_foul_counts
    if args.first_season is None:                 # no season typed: every season config.yaml names
        seasons = range(int(cfg["first_season"]), int(cfg["last_season"]) + 1)
    else:
        seasons = season_range(args, cfg)
    totals = build_event_counts(cfg, seasons, force=args.force)
    build_creation_counts(cfg, seasons, force=args.force)
    build_foul_counts(cfg, seasons, force=args.force)
    check = event_check(totals, season_box(list(seasons), ["RS", "PO"], cfg))
    pd.set_option("display.width", 200)
    print("\nparts minus the box score, as a share of the box score (0.001 = 0.1%):")
    print(check.round(4).to_string())
    return 0


def run_careers(args: argparse.Namespace, cfg: dict) -> int:
    """The careers before the first season: every career data/careers.py names fetched (a cached one is read), then
    the rows before the first season built again and cached, and what they add."""
    from ..data.careers import careers_before, draft_years, players_to_pull, pull_careers
    from ..data.ingest import raw_dir
    from ..data.roles import build_roles
    first = int(cfg["first_season"])
    roles = build_roles(cfg)
    players = players_to_pull(roles, draft_years(raw_dir(cfg) / "bio"), first)
    fetched = pull_careers(cfg, players, delay=args.delay, force=args.force)
    rows = careers_before(cfg, roles, force=True)
    seasons = rows.groupby("player_id").season.nunique()
    print(f"{len(players)} players may have played before {first} ({fetched} careers fetched now); {len(seasons)} did: "
          f"{int(seasons.sum())} seasons, {rows.minutes.sum() / 1e6:.2f} million minutes, the most "
          f"{int(seasons.max())} seasons", flush=True)
    return 0


def run_dashboards(args: argparse.Namespace, cfg: dict) -> int:
    """The hustle, tracking, play-type and matchup dashboards: a probe (a few seasons of every pull) or the full
    fetch, then the coverage of the season panel's rated players, printed and written to outputs/csv."""
    import pandas as pd

    from ..data import dashboards
    from ..data.config import resolve
    wanted_sources = tuple(args.sources.split(","))
    unknown = set(wanted_sources) - set(dashboards.SOURCES)
    if unknown:
        raise SystemExit(f"unknown sources {sorted(unknown)}; choose from {list(dashboards.SOURCES)}")
    last_season = args.last_season if args.last_season is not None else int(cfg["last_season"])
    wanted = []
    for pull in dashboards.pulls(wanted_sources):
        if args.probe:
            seasons = dashboards.probe_seasons(pull, last_season)
        else:
            first = args.first_season if args.first_season is not None else pull.first_guess
            seasons = list(range(first, last_season + 1))
        wanted.append((pull, seasons))
    records = dashboards.pull_all(cfg, wanted, phases(args), delay=args.delay, force=args.force)
    table = dashboards.coverage(records, dashboards.panel_rows(cfg))
    csv = resolve(cfg, "outputs") / "csv"
    csv.mkdir(parents=True, exist_ok=True)
    stem = "dashboard_probe" if args.probe else "dashboard_coverage"
    table.to_csv(csv / f"{stem}.csv", index=False)
    dashboards.fields(records).to_csv(csv / f"{stem}_fields.csv", index=False)
    summary = (table.assign(has_rows=table.rows > 0, failed=table.status == "failed")
               .groupby(["source", "season", "phase"], as_index=False)
               .agg(pulls=("kind", "size"), with_rows=("has_rows", "sum"), failed=("failed", "sum"),
                    share_players=("share_players", "median"), share_possessions=("share_possessions", "median")))
    pd.set_option("display.width", 200)
    failed = int((table.status == "failed").sum())
    print(f"\n{len(table)} files ({failed} failed); wrote {csv / (stem + '.csv')}", flush=True)
    print(summary.round(3).to_string(index=False), flush=True)
    # The hustle tables feed the box-score prior's shipped inputs, and a failed pull leaves yesterday's file in place
    # (fetch_dashboard writes on success only): the season in progress would divide yesterday's totals by today's
    # possessions and the morning would publish them (2026-11-01).  So a failed hustle pull fails the run; the other
    # sources feed no shipped input and are only reported.
    hustle_failed = table[(table.source == "hustle") & (table.status == "failed")]
    if len(hustle_failed):
        print("  hustle pulls failed, the run stops: "
              + ", ".join(f"{r.kind} {r.season} {r.phase}" for r in hustle_failed.itertuples()), flush=True)
        return 1
    return 0


def run_hustle_games(args: argparse.Namespace, cfg: dict) -> int:
    """The per-game box score of `args.source` (hustle, tracking or matchups) for every game of the seasons, one
    request a game, cached; failed games are listed and fail the run.  A game of the season in progress the league
    has not processed yet is named, not cached, and does not fail the run: the next run asks again."""
    from ..data import dashboards
    first = args.first_season if args.first_season is not None else dashboards.GAME_BOXES[args.source][1]
    last = args.last_season if args.last_season is not None else int(cfg["last_season"])
    started = time.time()
    out = dashboards.fetch_game_boxes(cfg, args.source, range(first, last + 1), phases(args), delay=args.delay,
                                      limit=getattr(args, "limit", None), force=getattr(args, "force", False))
    print(f"{out['games']} games {first}-{last}: {out['fetched']} fetched now ({out['empty']} with no rows), "
          f"{len(out['failed'])} failed, {len(out['pending'])} not processed by the league yet "
          f"({time.time() - started:.0f}s)"
          f"{'; stopped at the limit, run it again for the rest' if out['stopped'] else ''}", flush=True)
    if out["pending"]:
        print(f"  not processed yet, asked again next time: {' '.join(out['pending'][:20])}", flush=True)
    for game_id, error in out["failed"][:20]:
        print(f"  failed {game_id}: {error}", flush=True)
    return 1 if out["failed"] else 0


def run_player_games(args: argparse.Namespace, cfg: dict) -> int:
    """Each season's game rows (data/player_games.py), written once; --force writes them again."""
    from ..data.player_games import write_player_games
    started = time.time()
    first = args.first_season if args.first_season is not None else int(cfg["first_season"])
    last = args.last_season if args.last_season is not None else int(cfg["last_season"])
    written = write_player_games(cfg, range(first, last + 1), force=args.force)
    print(f"wrote {len(written)} seasons of game rows ({time.time() - started:.0f}s)", flush=True)
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


def _build_so_far(args: argparse.Namespace, cfg: dict) -> None:
    from ..data.panel import write_panel_so_far
    from .cli import parse_shares
    started = time.time()
    target, panel = write_panel_so_far(cfg, parse_shares(args.so_far), force=args.force)
    print(f"wrote {target}: {len(panel)} rows, {panel.season.nunique()} seasons, shares "
          f"{', '.join(f'{share:g}' for share in sorted(panel.share.unique()))} ({time.time() - started:.0f}s)",
          flush=True)


def run_panel(args: argparse.Namespace, cfg: dict) -> int:
    """The season panel: built from the stints and box scores, or migrated from the old-named one; with --so-far,
    the panel of each season so far.  An existing panel stops the command with one line saying so."""
    build = _migrate if args.migrate_from else (_build_so_far if args.so_far else _build)
    try:
        build(args, cfg)
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
    ingest.add_argument("--refresh", action="store_true",
                        help="the season in progress: fetch its game log again instead of reading the cache, keeping "
                             "the games dated before today in US Eastern time, then every new game's files")
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


def _add_events(steps) -> None:
    events = steps.add_parser(
        "events", help="the play-by-play parts of box counts (paths.event_counts)",
        description="Counts, per game, player and phase, the parts the season panel pads beside four box counts "
                    "(src/openrapm/data/event_counts.py): defensive rebounds by the miss they ended, steals by the "
                    "turnover they took, field-goal points and made threes by whether an assist is written beside "
                    "them, threes attempted from the corner or above the break, blocks by the shot and by who got "
                    "the ball back, stolen turnovers, offensive fouls, and assists by the zone of the make they set "
                    "up; and self-creation's counts (src/openrapm/data/creation_counts.py): every shot's spot and "
                    "possession situation, the free-throw trips, and the passes expected on the misses and trips; "
                    "and the fouls, goaltends, shots blocked and three-shot trips "
                    "(src/openrapm/data/foul_counts.py).  Caches them per season in paths.event_counts "
                    "(data/event_counts) and prints each season's parts against its box score.")
    events.add_argument("first_season", nargs="?", type=int, metavar="FIRST",
                        help="the first season (default config.yaml's first_season)")
    events.add_argument("last_season", nargs="?", type=int, metavar="LAST",
                        help="the last season (default FIRST when FIRST is given, else config.yaml's last_season)")
    events.add_argument("--force", action="store_true", help="count again seasons that are already cached")
    events.set_defaults(run=run_events)


def _add_careers(steps) -> None:
    careers = steps.add_parser(
        "careers", help="the seasons before the first one of every veteran, from stats.nba.com, cached",
        description="Fetches the season totals of every player the roles table first sees by 2002 who was drafted by "
                    "1995 or not drafted (src/openrapm/data/careers.py) into paths.raw/careers, one file per player "
                    "(a file already there is not fetched again), then writes the seasons before the first one as "
                    "roles-table rows to paths.cache/careers_before.parquet, which `openrapm data panel` reads for "
                    "the career columns counted from the debut.  With OPENRAPM_OFFLINE=1 a missing file stops the run.")
    careers.add_argument("--delay", type=float, default=0.6, help="seconds between requests (default %(default)s)")
    careers.add_argument("--force", action="store_true", help="fetch every career again")
    careers.set_defaults(run=run_careers)


def _add_dashboards(steps) -> None:
    dashboards = steps.add_parser(
        "dashboards", help="hustle, tracking, play-type and matchup dashboards from stats.nba.com, cached",
        description="Fetches the league dashboards the ratings do not read yet (src/openrapm/data/dashboards.py) into "
                    "paths.raw/dashboards, one file per pull, season and phase (a file already there is not fetched "
                    "again), and prints how many of the season panel's rated players each covers.  --probe asks only "
                    "a few seasons of every pull, to learn which fields exist and from which season.  With "
                    "OPENRAPM_OFFLINE=1 a missing file stops the run.")
    dashboards.add_argument("first_season", nargs="?", type=int, metavar="FIRST",
                            help="the first season of a full fetch (default: each pull's first season with data)")
    dashboards.add_argument("last_season", nargs="?", type=int, metavar="LAST",
                            help="the last season (default config.yaml's last_season)")
    dashboards.add_argument("--probe", action="store_true",
                            help="only the season before each pull should start, its first, 2018 and the latest")
    dashboards.add_argument("--sources", default=",".join(("hustle", "tracking", "synergy", "matchups")),
                            help="hustle, tracking, synergy, matchups, comma-separated (default all four)")
    dashboards.add_argument("--phases", default=DEFAULT_PHASES, help="RS, PO or both (default %(default)s)")
    dashboards.add_argument("--delay", type=float, default=0.6, help="seconds between requests (default %(default)s)")
    dashboards.add_argument("--force", action="store_true", help="fetch files that are already on disk again")
    dashboards.set_defaults(run=run_dashboards)


def _add_hustle_games(steps) -> None:
    hustle = steps.add_parser(
        "hustle-games", help="the hustle numbers game by game from stats.nba.com, cached",
        description="Fetches every game's hustle box score (BoxScoreHustleV2: deflections, contested shots, charges "
                    "drawn, loose balls, screen assists, box-outs) into paths.raw/dashboards/hustle/games/<season>/, "
                    "one file a game (a file already there is not fetched again), so a within-season fold can "
                    "sum them over its fit games.  With OPENRAPM_OFFLINE=1 a missing file stops the run.")
    hustle.add_argument("first_season", nargs="?", type=int, metavar="FIRST",
                        help="the first season (default 2017, the first with every player)")
    hustle.add_argument("last_season", nargs="?", type=int, metavar="LAST",
                        help="the last season (default config.yaml's last_season)")
    hustle.add_argument("--phases", default=DEFAULT_PHASES, help="RS, PO or both (default %(default)s)")
    hustle.add_argument("--delay", type=float, default=0.6, help="seconds between requests (default %(default)s)")
    hustle.add_argument("--force", action="store_true",
                        help="fetch every game again, cached or not: the league fills a game's box score in over "
                             "the days after it, so a season fetched while in progress is fetched again once its "
                             "Finals are over")
    hustle.set_defaults(run=run_hustle_games, source="hustle")
    tracking = steps.add_parser(
        "tracking-games", help="the per-game tracking box scores from stats.nba.com, cached",
        description="Fetches every game's tracking box score (BoxScorePlayerTrackV3: distance, rebound chances, "
                    "touches, passes, contested shots, shots defended at the rim) into "
                    "paths.raw/dashboards/tracking/games/<season>/, one file a game, as hustle-games does.")
    tracking.add_argument("first_season", nargs="?", type=int, metavar="FIRST", help="the first season (default 2014)")
    tracking.add_argument("last_season", nargs="?", type=int, metavar="LAST",
                          help="the last season (default config.yaml's last_season)")
    tracking.add_argument("--phases", default=DEFAULT_PHASES, help="RS, PO or both (default %(default)s)")
    tracking.add_argument("--delay", type=float, default=0.6, help="seconds between requests (default %(default)s)")
    tracking.add_argument("--force", action="store_true",
                        help="fetch every game again, cached or not: the league fills a game's box score in over "
                             "the days after it, so a season fetched while in progress is fetched again once its "
                             "Finals are over")
    tracking.set_defaults(run=run_hustle_games, source="tracking")
    matchups = steps.add_parser(
        "matchup-games", help="the per-game matchups from stats.nba.com, who guarded whom, cached",
        description="Fetches every game's matchup box score (BoxScoreMatchupsV3: each attacker and each defender who "
                    "guarded him, with the matchup's partial possessions, time, share of the time both were on the "
                    "floor and box score) into paths.raw/dashboards/matchups/games/<season>/, one file a game, as "
                    "hustle-games does.  The matchup inputs (data/matchups.py) sum them over a fold's fit games.")
    matchups.add_argument("first_season", nargs="?", type=int, metavar="FIRST",
                          help="the first season (default 2018, the first the endpoint answers)")
    matchups.add_argument("last_season", nargs="?", type=int, metavar="LAST",
                          help="the last season (default config.yaml's last_season)")
    matchups.add_argument("--phases", default=DEFAULT_PHASES, help="RS, PO or both (default %(default)s)")
    matchups.add_argument("--delay", type=float, default=0.6, help="seconds between requests (default %(default)s)")
    matchups.add_argument("--force", action="store_true",
                        help="fetch every game again, cached or not: the league fills a game's box score in over "
                             "the days after it, so a season fetched while in progress is fetched again once its "
                             "Finals are over")
    matchups.add_argument("--limit", type=int, default=None, metavar="N",
                          help="stop after N games fetched; a rerun goes on from there.  One process stalled after "
                               "about 600 requests on 2026-10-28, so a long fetch runs best as several processes")
    matchups.set_defaults(run=run_hustle_games, source="matchups")


def _add_player_games(steps) -> None:
    games = steps.add_parser(
        "player-games", help="each player's games, one row per player and game (paths.cache/player_games)",
        description="Writes one file a season to paths.cache/player_games/<season>.parquet: one row per player and "
                    "game he played possessions in, with his team and the opponent, home or away, whether he "
                    "started, his and his team's possessions at each end, his 13 box counts and, from 2017, his "
                    "nine hustle counts (src/openrapm/data/player_games.py).  The multi-stat network reads it.")
    games.add_argument("first_season", nargs="?", type=int, metavar="FIRST",
                       help="the first season (default config.yaml's first_season)")
    games.add_argument("last_season", nargs="?", type=int, metavar="LAST",
                       help="the last season (default config.yaml's last_season)")
    games.add_argument("--force", action="store_true", help="write seasons already on disk again")
    games.set_defaults(run=run_player_games)


def _add_panel(steps) -> None:
    panel = steps.add_parser(
        "panel", help="the season panel (paths.panel, outputs/season_panel.parquet)",
        description="Builds the season panel, one row per player, season and side, from the stints, box scores, "
                    "roles table, bio feed and careers (src/openrapm/data/panel.py), every season from all of its "
                    "games, and "
                    "writes it to paths.panel.  An existing panel is kept unless --force.  With --migrate-from it "
                    "writes the panel instead from one built under the names used before 2026-10-11: every column "
                    "renamed, the columns no reader uses dropped, the rows in their order, every value unchanged.")
    panel.add_argument("--force", action="store_true", help="replace an existing season panel")
    panel.add_argument("--migrate-from", metavar="FILE",
                       help="the old-named panel, e.g. outputs/role_panel_season.parquet")
    panel.add_argument("--so-far", metavar="SHARES", default="",
                       help="instead, each season so far at these shares of its regular season, comma-separated "
                            "(0.25,0.5,0.75), every input from those games alone, written to paths.panel_so_far for "
                            "the box-score prior's part-season rows (`openrapm rate --part-season-rows`)")
    panel.set_defaults(run=run_panel)


def add_data(commands) -> None:
    """`openrapm data` and its five steps."""
    data = commands.add_parser("data", help="the data the ratings are built from: raw files, stints, season panel",
                               description="Builds the data the five steps read: `ingest`, then `stints`, then "
                                           "`events`, then `careers`, then `panel`.")
    steps = data.add_subparsers(dest="data_command", required=True, metavar="DATA")
    _add_ingest(steps)
    _add_stints(steps)
    _add_events(steps)
    _add_careers(steps)
    _add_dashboards(steps)
    _add_hustle_games(steps)
    _add_player_games(steps)
    _add_panel(steps)
