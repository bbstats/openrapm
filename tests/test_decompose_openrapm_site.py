"""The OpenRAPM decomposition page's committed data (docs/data/decompose_openrapm.json, written by `openrapm site
--openrapm-decomposition`): on each side and in total the five pieces add up to the rating, and the rows are
ratings.json's, with its ratings and possessions, so a ratings rebuild without a decomposition rebuild fails here
instead of showing stale numbers.

The command checks the identities to 1e-9 before it writes; this checks what was actually committed, at the two
decimals the page shows."""
import json
from pathlib import Path

import pytest

from openrapm.commands.site import OPENRAPM_FIELDS, OPENRAPM_PIECES

DATA = Path(__file__).resolve().parents[1] / "docs" / "data"
ROUNDING = 0.031          # six numbers at two decimals, the five pieces and the rating, each off by up to 0.005


@pytest.fixture(scope="module")
def decomposition():
    path = DATA / "decompose_openrapm.json"
    if not path.exists():
        pytest.skip("decompose_openrapm.json is not built")
    return json.loads(path.read_text(encoding="utf-8"))


def test_the_file_holds_every_season_latest_first_with_the_published_field_names(decomposition):
    meta = decomposition["meta"]
    assert meta["fields"] == OPENRAPM_FIELDS and meta["pieces"] == OPENRAPM_PIECES
    assert meta["seasons"] == sorted(meta["seasons"], reverse=True)
    assert list(decomposition["rows"]) == [str(season) for season in meta["seasons"]]


def test_on_each_side_and_in_total_the_five_pieces_add_up_to_the_rating(decomposition):
    worst = 0.0
    for season, rows in decomposition["rows"].items():
        for name, player_id, possessions, *sides, team in rows:
            assert len(sides) == 3 and all(len(pieces) == len(OPENRAPM_PIECES) for pieces in sides)
            for pieces in sides:
                worst = max(worst, abs(sum(pieces[:-1]) - pieces[-1]))
    assert worst <= ROUNDING, f"a row's five pieces miss its rating by {worst:.3f}"


def test_a_total_is_the_two_sides_added(decomposition):
    worst = 0.0
    for rows in decomposition["rows"].values():
        for _, _, _, off, deff, total, _ in rows:
            worst = max(worst, *(abs(t - (a + b)) for t, a, b in zip(total, off, deff)))
    assert worst <= 0.0101, f"a total misses its two sides by {worst:.3f}"       # three roundings of half a cent


def test_every_row_has_a_name_an_id_possessions_and_a_team_and_no_player_appears_twice_in_a_season(decomposition):
    for season, rows in decomposition["rows"].items():
        assert rows, f"{season} has no players"
        ids = [row[1] for row in rows]
        assert len(set(ids)) == len(ids), f"{season}: a player appears twice"
        for name, player_id, possessions, *_, team in rows:
            assert name and isinstance(player_id, int) and player_id > 0 and possessions > 0, (season, name)
            assert team and all(len(code) == 3 for code in team.split("/")), (season, name, team)


def test_the_rows_are_ratings_jsons_with_its_ratings_and_possessions(decomposition):
    path = DATA / "ratings.json"
    if not path.exists():
        pytest.skip("ratings.json is not built")
    published = {(row["s"], row["id"]): (row["o"], row["d"], row["t"], row["p"])
                 for row in json.loads(path.read_text(encoding="utf-8"))["rows"]}
    mine = {(int(season), row[1]): (row[3][-1], row[4][-1], row[5][-1], row[2])
            for season, rows in decomposition["rows"].items() for row in rows}
    stale = sorted(set(published) ^ set(mine))[:3] or [key for key in published if published[key] != mine[key]][:3]
    assert published == mine, f"the file differs from ratings.json, e.g. {stale}: run `openrapm site " \
                              f"--openrapm-decomposition` after a ratings build"
