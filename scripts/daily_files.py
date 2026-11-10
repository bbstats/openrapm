"""The morning job's three small readings (scripts/daily_update.sh).

    python scripts/daily_files.py changed SITE FILE...      exit 0 when a file differs from SITE's HEAD beyond the
                                                            date it was built ("built" in its meta), 3 when none does
    python scripts/daily_files.py last-game GAMELOG SEASON  the date of the season's last game in the game logs
                                                            GAMELOG/<season>_RS.parquet and _PO.parquet
    python scripts/daily_files.py similarity-stale RATINGS POOL
                                                            exit 0 when a player-season of the Similarity page's pool
                                                            (POOL, docs/data/similarity/player_seasons.json) carries
                                                            other ratings than RATINGS (docs/data/ratings.json), 3
                                                            when every one carries the published ones

The site's files carry the date they were built, so they differ from the published ones every morning: a plain
`git diff --quiet` never came back clean, and a morning with no new games published anyway (2026-11-01).  The last
game's date named the regular season's log alone, so a playoff commit would have named the regular season's last day.

The Similarity page's pool holds finished seasons, so its ratings change only when a new build of the ratings is
published.  The five-zone shooting change reached the site through the morning job on 2026-11-07, the job did not write
the pool, and for three days 8,850 of its 10,160 player-seasons showed a total the Ratings page no longer had; the job
now asks every morning and rebuilds the pool when the answer is yes.
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


def stale_pool(ratings: dict, pool: dict) -> list:
    """The pool's player-seasons, [id, name, season], whose ratings are not the published ones: the player-season is
    not in `ratings` (ratings.json's document), or its offense, defense or total differs, or its possessions differ by
    more than 1.  The rule of tests/test_similarity_site.py, on the two documents.

    Example: a pool row [203999, "Nikola Jokić", 2026, 5269, 7.0, 2.3, 9.2] against a published defense of 1.8 and
    total of 8.7 is stale."""
    fields = pool["meta"]["fields"]
    at = {name: fields.index(name) for name in ("id", "n", "s", "p", "o", "d", "t")}
    published = {(row["id"], row["s"]): row for row in ratings["rows"]}
    stale = []
    for row in pool["rows"]:
        now = published.get((row[at["id"]], row[at["s"]]))
        if (now is None or any(now[key] != row[at[key]] for key in "odt")
                or abs(now["p"] - row[at["p"]]) > 1):
            stale.append([row[at["id"]], row[at["n"]], row[at["s"]]])
    return stale


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
    similar = commands.add_parser("similarity-stale", help="exit 0 when the Similarity page's pool carries other "
                                                           "ratings than the published ones, 3 when it is in step")
    similar.add_argument("ratings", type=Path)
    similar.add_argument("pool", type=Path)
    return top


def main(argv) -> int:
    args = parser().parse_args(argv)
    if args.command == "changed":
        files = changed(args.site, args.files)
        print(f"changed beyond the built date: {', '.join(files) if files else 'nothing'}")
        return 0 if files else 3
    if args.command == "similarity-stale":
        stale = stale_pool(json.loads(args.ratings.read_text(encoding="utf-8")),
                           json.loads(args.pool.read_text(encoding="utf-8")))
        # the id, not the name: the job's log is written in the console's code page, which has no ć
        print(f"the Similarity page's pool: {len(stale)} player-seasons carry other ratings than the published ones"
              + (f", the first player {stale[0][0]} in {stale[0][2]}" if stale else ""))
        return 0 if stale else 3
    print(last_game(args.gamelog, args.season))
    return 0


if __name__ == "__main__":
    sys.exit(main(None))
