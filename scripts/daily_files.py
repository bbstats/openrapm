"""The morning job's two small readings (scripts/daily_update.sh).

    python scripts/daily_files.py changed SITE FILE...      exit 0 when a file differs from SITE's HEAD beyond the
                                                            date it was built ("built" in its meta), 3 when none does
    python scripts/daily_files.py last-game GAMELOG SEASON  the date of the season's last game in the game logs
                                                            GAMELOG/<season>_RS.parquet and _PO.parquet

The site's files carry the date they were built, so they differ from the published ones every morning: a plain
`git diff --quiet` never came back clean, and a morning with no new games published anyway (2026-11-01).  The last
game's date named the regular season's log alone, so a playoff commit would have named the regular season's last day.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path


def without_built(document: dict) -> dict:
    """The document with the built date out of its meta."""
    meta = {key: value for key, value in document.get("meta", {}).items() if key != "built"}
    return {**document, "meta": meta}


def published(site: Path, file: str) -> dict | None:
    """The file's JSON as SITE's HEAD holds it; None when HEAD has no such file."""
    done = subprocess.run(["git", "-C", str(site), "show", f"HEAD:{file}"], capture_output=True, text=True,
                          encoding="utf-8")
    return None if done.returncode else json.loads(done.stdout)


def changed(site: Path, files: list) -> list:
    """The files whose content differs from HEAD's beyond the built date."""
    out = []
    for file in files:
        now = json.loads((site / file).read_text(encoding="utf-8"))
        before = published(site, file)
        if before is None or without_built(before) != without_built(now):
            out.append(file)
    return out


def last_game(gamelog: Path, season: int) -> str:
    """The latest game date in the season's regular-season and playoff game logs, as YYYY-MM-DD.  A playoff log
    fetched before the playoffs holds no dated row and is passed over."""
    import pandas as pd
    dates = []
    for phase in ("RS", "PO"):
        path = gamelog / f"{season}_{phase}.parquet"
        if path.exists():
            played = pd.read_parquet(path, columns=["GAME_DATE"]).GAME_DATE.dropna()
            if len(played):
                dates.append(str(played.max())[:10])
    if not dates:
        raise SystemExit(f"no game log of {season} in {gamelog}")
    return max(dates)


def parser() -> argparse.ArgumentParser:
    top = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = top.add_subparsers(dest="command", required=True)
    files = commands.add_parser("changed", help="exit 0 when a file differs from SITE's HEAD beyond its built date, "
                                                "3 when none does")
    files.add_argument("site", type=Path)
    files.add_argument("files", nargs="+")
    game = commands.add_parser("last-game", help="the date of the season's last game in its game logs")
    game.add_argument("gamelog", type=Path)
    game.add_argument("season", type=int)
    return top


def main(argv) -> int:
    args = parser().parse_args(argv)
    if args.command == "changed":
        files = changed(args.site, args.files)
        print(f"changed beyond the built date: {', '.join(files) if files else 'nothing'}")
        return 0 if files else 3
    print(last_game(args.gamelog, args.season))
    return 0


if __name__ == "__main__":
    sys.exit(main(None))
