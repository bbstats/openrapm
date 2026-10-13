"""The similarity page's committed data (docs/data/similarity/, written by `openrapm site --similarity`): the pool and
its lists hang together, no list holds the picked player's own seasons, and the pool's ratings are the published
ones, so a ratings rebuild without a similarity rebuild fails here instead of showing stale numbers."""
import json
from pathlib import Path

import pytest

from openrapm.commands.site import SIMILARITY_FIELDS, SIMILARITY_LIST_FIELDS

DATA = Path(__file__).resolve().parents[1] / "docs" / "data"
FOLDER = DATA / "similarity"


@pytest.fixture(scope="module")
def pool():
    path = FOLDER / "player_seasons.json"
    if not path.exists():
        pytest.skip("the similarity files are not built")
    return json.loads(path.read_text(encoding="utf-8"))


def _season_files(pool):
    return {season: json.loads((FOLDER / f"{season}.json").read_text(encoding="utf-8"))
            for season in pool["meta"]["seasons"]}


def test_the_pool_is_in_season_then_id_order_and_holds_the_opening_view(pool):
    assert pool["meta"]["fields"] == SIMILARITY_FIELDS
    keys = [(row[2], row[0]) for row in pool["rows"]]
    assert keys == sorted(keys) and len(set(keys)) == len(keys)
    assert tuple(pool["meta"]["default"][::-1]) in set(keys)
    assert sorted({row[2] for row in pool["rows"]}) == pool["meta"]["seasons"]


def test_the_pool_carries_the_published_ratings(pool):
    path = DATA / "ratings.json"
    if not path.exists():
        pytest.skip("ratings.json is not built")
    published = {(r["id"], r["s"]): r for r in json.loads(path.read_text(encoding="utf-8"))["rows"]}
    stale = [row[:3] for row in pool["rows"]
             if (row[0], row[2]) not in published
             or [published[(row[0], row[2])][key] for key in "odt"] != row[4:7]
             or abs(published[(row[0], row[2])]["p"] - row[3]) > 1]
    assert not stale, f"{len(stale)} pool rows differ from ratings.json, e.g. {stale[:3]}: run `openrapm site " \
                      f"--similarity` after a ratings build"


def test_every_list_is_whole_sorted_and_free_of_the_players_own_seasons(pool):
    players = [row[0] for row in pool["rows"]]
    matches, n = pool["meta"]["matches"], len(players)
    for season, document in _season_files(pool).items():
        first = document["meta"]["first"]
        assert document["meta"]["fields"] == SIMILARITY_LIST_FIELDS and pool["rows"][first][2] == season
        assert len(document["rows"]) == sum(row[2] == season for row in pool["rows"])
        for offset, (seasons, near, others, other_near) in enumerate(document["rows"]):
            own = players[first + offset]
            assert len(seasons) == len(near) == len(others) == len(other_near) == matches
            assert all(0 <= i < n and players[i] != own for i in seasons + others), f"{season} row {offset}"
            assert near == sorted(near) and other_near == sorted(other_near)
            assert len({players[i] for i in others}) == matches


def test_a_pair_reads_the_same_from_either_end(pool):
    """The distance is symmetric, so a pair that is on both players' lists carries one distance."""
    files = _season_files(pool)
    lists = {}
    for document in files.values():
        for offset, (seasons, near, _, _) in enumerate(document["rows"]):
            lists[document["meta"]["first"] + offset] = dict(zip(seasons, near))
    both = [(r, t) for r, near in lists.items() for t in near if r in lists.get(t, {})]
    assert both, "no pair is on both players' lists"
    assert all(lists[r][t] == lists[t][r] for r, t in both)
