"""One-off edit for the bug-fix pass of 2026-10-30: `--force` on the per-game fetch commands (kept for the record)."""
from pathlib import Path


def edit(path: str, pairs: list) -> None:
    p = Path(path)
    s = p.read_text(encoding="utf-8")
    for old, new, count in pairs:
        assert s.count(old) == count, (path, old[:70], s.count(old))
        s = s.replace(old, new)
    p.write_text(s, encoding="utf-8", newline="\n")


FORCE_HELP = ('    {name}.add_argument("--force", action="store_true",\n'
              '                        help="fetch every game again, cached or not: the league fills a game\'s box score in over the days "\n'
              '                             "after it, so a season fetched while in progress is fetched again once its Finals are over")\n')

edit("src/openrapm/commands/data.py", [
    ('    hustle.add_argument("--delay", type=float, default=0.6, help="seconds between requests (default %(default)s)")\n',
     '    hustle.add_argument("--delay", type=float, default=0.6, help="seconds between requests (default %(default)s)")\n'
     + FORCE_HELP.format(name="hustle"), 1),
    ('    tracking.add_argument("--delay", type=float, default=0.6, help="seconds between requests (default %(default)s)")\n',
     '    tracking.add_argument("--delay", type=float, default=0.6, help="seconds between requests (default %(default)s)")\n'
     + FORCE_HELP.format(name="tracking"), 1),
    ('    matchups.add_argument("--delay", type=float, default=0.6, help="seconds between requests (default %(default)s)")\n',
     '    matchups.add_argument("--delay", type=float, default=0.6, help="seconds between requests (default %(default)s)")\n'
     + FORCE_HELP.format(name="matchups"), 1),
    ('''    out = dashboards.fetch_game_boxes(cfg, args.source, range(first, last + 1), phases(args), delay=args.delay,
                                      limit=getattr(args, "limit", None))''',
     '''    out = dashboards.fetch_game_boxes(cfg, args.source, range(first, last + 1), phases(args), delay=args.delay,
                                      limit=getattr(args, "limit", None), force=getattr(args, "force", False))''', 1),
])

edit("src/openrapm/data/dashboards.py", [
    ('''def fetch_game_boxes(cfg: dict, source: str, seasons, phases=("RS", "PO"), delay: float = 0.6,
                     log_every: int = 250, limit: int | None = None) -> dict:''',
     '''def fetch_game_boxes(cfg: dict, source: str, seasons, phases=("RS", "PO"), delay: float = 0.6,
                     log_every: int = 250, limit: int | None = None, force: bool = False) -> dict:''', 1),
    ('''    One bad game does not stop the run, and a rerun fetches only what is missing.  OfflineFetch is not forgiven.''',
     '''    One bad game does not stop the run, and a rerun fetches only what is missing, or every game with `force`: the
    league fills a game's box score in over the days after it (2026-10-30: last night's games answered no box-outs, a
    game of the week before 8 deflections against 42 later), so a season fetched while in progress is fetched again
    once its Finals are over.  OfflineFetch is not forgiven.''', 1),
    ('''            frame, cached = fetch_game_box(source, game_id, cfg)''',
     '''            frame, cached = fetch_game_box(source, game_id, cfg, force=force)''', 1),
])
print("ok")
