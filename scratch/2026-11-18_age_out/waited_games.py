"""The games check learns a list of games it waits for (the owner, 2026-11-19: "skip the game for now but we should
keep checking for it").  Every replacement must find its text once, or nothing is written.

    .venv/Scripts/python -I scratch/2026-11-18_age_out/waited_games.py [--dry]
"""
import sys
from pathlib import Path

CHANGES = {
    "src/openrapm/data/stints.py": [
        ("""#   a game passes when     it is in the game log, has stints, and its play-by-play scores what its box score says
""", """#   a game passes when     it is in the game log, has stints, and its play-by-play scores what its box score says
#   a game is waited for   when config.yaml's `no_stints_accepted` names it and it has no stints: the morning goes on
#                          without it, its cached files are still removed, and every morning's fetch asks for it again
#                          (the owner, 2026-11-19, of a game the league posted with no substitution text: "skip the
#                          game for now but we should keep checking for it")
"""),
        ("""    gone: tuple              # games the last morning that passed held and the game log no longer does

    @property""", """    gone: tuple              # games the last morning that passed held and the game log no longer does
    waited_for: dict = field(default_factory=dict)    # game id -> why, for the games without stints the config accepts
    arrived: tuple = ()      # the games the config accepts that have stints now: they can come off its list

    @property"""),
        ("""def check_games(season: int, cfg, accepted=()) -> list:
    \"\"\"One GamesCheck for each phase of `season` that has games: the cached game log against the stints and their
    diagnostics on disk, and against the record of the last morning that passed.  Nothing is fetched, built or
    removed.  `accepted` names games whose points are known to differ for good (config `points_mismatch_accepted`).
""", """def _without_stints(logged: list, built: set, errors: dict, waited: set) -> tuple[dict, dict]:
    \"\"\"({game: why} for the logged games without stints that stop the morning, the same for the ones it waits for).\"\"\"
    missing = {game: errors.get(game, "no stints") for game in logged if game not in built}
    return ({game: why for game, why in missing.items() if game not in waited},
            {game: why for game, why in missing.items() if game in waited})


def check_games(season: int, cfg, accepted=(), waited_for=()) -> list:
    \"\"\"One GamesCheck for each phase of `season` that has games: the cached game log against the stints and their
    diagnostics on disk, and against the record of the last morning that passed.  Nothing is fetched, built or
    removed.  `accepted` names games whose points are known to differ for good (config `points_mismatch_accepted`);
    `waited_for` names games that may be without stints for now (config `no_stints_accepted`).
"""),
        ("""    folder, accepted, record = resolve(cfg, "stints"), set(_ten(accepted)), checked_games(season, cfg)
""", """    folder, accepted, record = resolve(cfg, "stints"), set(_ten(accepted)), checked_games(season, cfg)
    waited = set(_ten(waited_for))
"""),
        ("""        no_stints = {game: errors.get(game, "no stints") for game in logged if game not in built}
        gone = tuple(game for game in _ten(record.get(phase, [])) if game not in set(logged))
        checks.append(GamesCheck(phase, tuple(logged), no_stints, _points_differ(diagnostics, built, accepted), gone))
""", """        no_stints, waiting = _without_stints(logged, built, errors, waited)
        gone = tuple(game for game in _ten(record.get(phase, [])) if game not in set(logged))
        checks.append(GamesCheck(phase, tuple(logged), no_stints, _points_differ(diagnostics, built, accepted), gone,
                                 waiting, tuple(sorted(waited & built & set(logged)))))
"""),
    ],
    "src/openrapm/commands/data.py": [
        ("""    again.  A finished season is refused: its files are history, and some of its games never did add up.\"\"\"
""", """    again.  A game of config.yaml's `no_stints_accepted` that has no stints does not stop the morning: it is named,
    its cached files are removed all the same, so every morning asks the league for it again, and the day it has
    stints the check says it can come off the list.  A finished season is refused: its files are history, and some of
    its games never did add up.\"\"\"
"""),
        ("""    checks = check_games(season, cfg, accepted=cfg.get("points_mismatch_accepted") or ())
""", """    checks = check_games(season, cfg, accepted=cfg.get("points_mismatch_accepted") or (),
                         waited_for=cfg.get("no_stints_accepted") or ())
"""),
        ("""        for game_id in check.gone:
""", """        for game_id, why in sorted(check.waited_for.items()):
            removed = forget_game(game_id, cfg)
            print(f"  {game_id}: {why}, and config.yaml's no_stints_accepted waits for it: the season is rated without "
                  f"it; {len(removed)} cached files removed, asked for again by the next ingest", flush=True)
        for game_id in check.arrived:
            print(f"  {game_id}: has stints now; take it off config.yaml's no_stints_accepted", flush=True)
        for game_id in check.gone:
"""),
        ("""                    "them again.  config.yaml's points_mismatch_accepted lists games whose points differ for good.  "
""", """                    "them again.  config.yaml's points_mismatch_accepted lists games whose points differ for good, "
                    "and its no_stints_accepted games the season is rated without for now, asked for again every "
                    "morning.  "
"""),
    ],
    "config.yaml": [
        ("""# The role inputs of the season panel (src/openrapm/data/roles.py).
""", """# The morning job's games check (data/stints.check_games) stops when a game of the season in progress has no stints.
# A game named here does not stop it: the season is rated without that game, and every morning asks the league for it
# again and says here when it can come off.  0022600265, Hawks at Thunder of 2026-11-18: the league posted its
# play-by-play with every description blank, so no substitution says who came in (the owner, 2026-11-19: "skip the game
# for now but we should keep checking for it").
no_stints_accepted: ["0022600265"]

# The role inputs of the season panel (src/openrapm/data/roles.py).
"""),
    ],
}


def main() -> None:
    texts = {}
    for path, pairs in CHANGES.items():
        text = Path(path).read_text(encoding="utf-8")
        for old, new in pairs:
            if text.count(old) != 1:
                raise SystemExit(f"{path}: found {text.count(old)} times, nothing written:\n{old}")
            text = text.replace(old, new)
        texts[path] = text
    if "--dry" in sys.argv:
        print(f"every replacement found once in {len(texts)} files; nothing written")
        return
    for path, text in texts.items():
        Path(path).write_bytes(text.encode("utf-8"))
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
