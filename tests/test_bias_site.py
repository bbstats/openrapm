"""The Bias page's committed file (docs/data/bias.json, written by `openrapm site --bias`) and the page that draws
it (docs/bias.html): the file has the published fields, its fifteen rows add up, each has a lean against the game
results, its three systems carry the owner's weights and their makers' names, and nothing in it is a system's own
rating.

The command builds the file from the outside systems' stored tables, which are not in git; this checks what was
actually committed.  The page reads the file by name, so a page committed without its file fails here.
"""
import json
from pathlib import Path

import pytest

from openrapm.bias import CONSENSUS_WEIGHTS, STATS
from openrapm.commands.site import BIAS_FIELDS, BIAS_FILE, MIN_POSSESSIONS_SHOWN
from openrapm.data.public_ratings import SYSTEMS

DOCS = Path(__file__).resolve().parents[1] / "docs"


@pytest.fixture(scope="module")
def document():
    return json.loads((DOCS / "data" / BIAS_FILE).read_text(encoding="utf-8"))


def test_the_page_reads_the_file_that_is_committed_beside_it():
    page = (DOCS / "bias.html").read_text(encoding="utf-8")
    assert f'fetch("data/{BIAS_FILE}")' in page
    assert (DOCS / "data" / BIAS_FILE).exists(), "docs/bias.html is committed without docs/data/bias.json"


def test_the_file_holds_the_fifteen_stats_in_order_under_the_published_fields(document):
    meta = document["meta"]
    assert meta["fields"] == BIAS_FIELDS and meta["min_possessions"] == MIN_POSSESSIONS_SHOWN
    assert [row[0] for row in document["rows"]] == STATS
    assert all(len(row) == len(BIAS_FIELDS) for row in document["rows"])
    assert 0 < meta["players"] <= meta["players_shown"]
    assert meta["multipliers"]["o"] > 0 and meta["multipliers"]["d"] > 0
    assert meta["through"] <= meta["built"]


def test_every_row_adds_up_at_the_two_decimals_the_page_shows(document):
    for row in document["rows"]:
        stat = dict(zip(BIAS_FIELDS, row))
        assert stat["t"] == round(stat["o"] + stat["d"], 2), stat["label"]
        assert all(stat[key] > 0 for key in ("o_se", "d_se", "t_se", "sd")), stat["label"]


def test_every_row_has_its_leans_against_the_game_results_and_the_file_says_what_they_were_read_on(document):
    meta = document["meta"]
    games, seasons = meta["games"], meta["games_seasons"]
    assert games["team_games"] > 0 and games["multipliers"]["o"] > 0 and games["multipliers"]["d"] > 0
    assert seasons["last"] == meta["season"] and seasons["last"] - seasons["first"] == 9
    assert seasons["team_games"] > 8 * games["team_games"]
    for row in document["rows"]:
        stat = dict(zip(BIAS_FIELDS, row))
        for key in ("g", "gs"):
            assert stat[key] is not None and stat[key + "_se"] > 0, (key, stat["label"])
            assert round(stat[key], 2) == stat[key], (key, stat["label"])


def test_the_page_names_its_columns_for_the_season_and_the_last_10_seasons():
    """The owner, 2026-11-24: "the first one should say 'vs consensus' and the 2nd one should now say 'vs game
    results'", then "Clarify the other 2 columns as just being '27" and "last 10 seasons (inclusive of current)"."""
    page = (DOCS / "bias.html").read_text(encoding="utf-8")
    assert '["t", m => `vs consensus ${short(m.season)}`' in page
    assert '["g", m => `vs game results ${short(m.season)}`' in page
    assert '["gs", () => "vs game results, last 10 seasons"' in page


def test_the_three_systems_carry_the_owners_weights_and_their_makers_names(document):
    systems = document["systems"]
    assert {system["key"]: system["weight"] for system in systems} == CONSENSUS_WEIGHTS
    assert [system["weight"] for system in systems] == sorted((s["weight"] for s in systems), reverse=True)
    for system in systems:
        known = SYSTEMS[system["key"]]
        assert (system["name"], system["by"], system["link"]) == (known.name, known.by, known.link)
        assert system["as_of"] and 0 < system["players"] <= document["meta"]["players_shown"]
    assert min(system["players"] for system in systems) >= document["meta"]["players"]


def test_the_file_publishes_slopes_and_no_players_rating(document):
    """The systems' ratings are their makers': the page shows how OpenRAPM leans against their mean, stat by stat,
    and never a player's number from any of them."""
    assert set(document) == {"meta", "systems", "rows"}
    text = json.dumps(document)
    assert "player_id" not in text and "nba_id" not in text
    assert len(text) < 5000, "fifteen rows and three systems; anything much longer is carrying something else"
