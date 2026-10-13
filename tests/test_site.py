"""`openrapm site` (src/openrapm/commands/site.py) on small made-up tables: the form of ratings.json, the bias
page's cells, and the decomposition's row bookkeeping.

The published files themselves are pinned byte for byte against the old scripts' output by the cleanup's gate 6
(A:/code/spmm_verify); tests/test_decompose_site.py checks the committed decomposition files add up.
"""
import json
import os

import numpy as np
import pandas as pd
import pytest

from openrapm.commands import site


def _table():
    """Three player-seasons with possessions and one without; one empty name; an accented name."""
    return pd.DataFrame({"season": [2025, 2026, 2026, 2026],
                         "player_name": ["Nikola Jokić", "Nikola Jokić", None, "Bench Player"],
                         "rating_off": [4.894, 4.886, 1.0, -1.0], "rating_def": [0.9, 0.904, 0.5, -0.2],
                         "rating_total": [5.794, 5.79, 1.5, -1.2], "poss_season": [5282.0, 5100.7, 300.0, 0.0],
                         "player_id": [203999, 203999, 1, 2]})


def test_ratings_rows_keep_players_with_possessions_best_first_at_two_decimals():
    rows = site.ratings_rows(_table())
    assert [(row["s"], row["id"]) for row in rows] == [(2025, 203999), (2026, 203999), (2026, 1)]
    assert rows[1] == {"s": 2026, "n": "Nikola Jokić", "o": 4.89, "d": 0.9, "t": 5.79, "p": 5100, "id": 203999}
    assert rows[2]["n"] == ""                       # a missing name is written empty


def test_ratings_document_counts_every_player_of_the_table():
    meta = site.ratings_document(_table(), "2026-10-09")["meta"]
    assert meta == {"seasons": [2025, 2026], "built": "2026-10-09", "n_players": 3}


def test_a_table_without_a_published_column_is_refused():
    with pytest.raises(SystemExit):
        site.ratings_rows(_table().drop(columns="poss_season"))


def test_ratings_json_is_compact_escaped_and_dated_by_the_table(tmp_path):
    ratings = tmp_path / "outputs" / "season_ratings_product.parquet"
    ratings.parent.mkdir()
    _table().to_parquet(ratings)
    stamp = pd.Timestamp("2026-10-09 18:37", tz="UTC").timestamp()
    os.utime(ratings, (stamp, stamp))
    cfg = {"_root": str(tmp_path), "paths": {"outputs": str(tmp_path / "outputs")}}
    assert site.run_site(cfg, out_dir=tmp_path / "site") == 0
    text = (tmp_path / "site" / "data" / "ratings.json").read_text(encoding="utf-8")
    assert text.startswith('{"meta":{"seasons":[2025,2026],"built":"2026-10-09","n_players":3},"rows":[{"s":2025,')
    assert "Joki\\u0107" in text and not text.endswith("\n")
    site.run_site(cfg, out_dir=tmp_path / "site", built="2027-01-01")
    assert json.loads((tmp_path / "site" / "data" / "ratings.json").read_text())["meta"]["built"] == "2027-01-01"


def test_a_missing_ratings_table_stops_the_command(tmp_path):
    with pytest.raises(SystemExit):
        site.run_site({"_root": str(tmp_path)}, ratings=tmp_path / "none.parquet", out_dir=tmp_path)


@pytest.mark.parametrize("value, cell", [
    (None, "<td>&ndash;</td>"),
    (0.45, '<td class="over" style="--depth: 0.500"><b>+0.450</b></td>'),
    (-1.8, '<td class="under" style="--depth: 1.000"><b>-1.800</b></td>'),
])
def test_a_bias_cell_is_tinted_by_its_size_and_sign(value, cell):
    assert site.tint_cell(value) == cell


def test_the_bias_page_lists_the_consensus_order_and_a_dash_where_the_games_have_no_row():
    result = {"groups": [{"group": 3, "name": "Rim-protecting centers", "signature": "+orb", "bias": 0.95},
                         {"group": 1, "name": "Wings who shoot threes", "signature": "+fg3m", "bias": -0.46}],
              "groups_vs_games": [{"group": 1, "bias": -0.06}]}
    page = site.bias_page(result)
    assert page.index("Rim-protecting centers") < page.index("Wings who shoot threes")
    assert '<span class="sig">+orb</span></td><td class="over" style="--depth: 1.000"><b>+0.950</b></td>' \
           '<td>&ndash;</td></tr>' in page
    assert page.startswith("<!doctype html>") and page.endswith("</script></body></html>")
    assert "\n</nav>\n<h1>OpenRAPM: bias by player type</h1>\n<table>" in page


def test_a_consensus_name_drops_accents_punctuation_and_suffixes():
    assert site.name_key("Jabari Smith Jr.") == "jabari smith"
    assert site.name_key("Nikola Jokić") == "nikola jokic"


def test_span_labels():
    assert site.span_label([2026]) == "2026"
    assert site.span_label([2024, 2025, 2026]) == "2024-26"


def test_a_side_takes_the_team_of_its_first_player_the_box_score_lists():
    teams = pd.Series([10, 20, 20], index=pd.MultiIndex.from_tuples([("g1", 1), ("g1", 2), ("g2", 3)]))
    game_ids = np.array(["g1", "g1", "g2"])
    players = np.array([[9, 1], [2, 1], [7, 8]])         # row 1: first player unknown; row 3: nobody known
    assert site.side_team(teams, game_ids, players).tolist() == [10, 20, -1]


def test_team_codes_give_an_unplaced_side_a_code_of_its_own():
    rows = {"team_att": np.array([10, 10, -1]), "team_def": np.array([20, 20, 20]),
            "season": np.array([2026, 2026, 2026]), "game_ids": np.array(["g1", "g2", "g2"])}
    codes = site.team_codes(rows)
    assert codes["ts_att"][0] == codes["ts_att"][1] != codes["ts_att"][2]       # one team-season, one unknown
    assert codes["gt_att"][0] != codes["gt_att"][1]                             # two team-games
    assert codes["unplaced"] == pytest.approx(1 / 3)


# ------------------------------------------------------------------------------ the similarity page
def _similarity_world(tmp_path, seed=3):
    """A made-up season panel and ratings table: four players over three seasons, every possession count above the
    minimum; the role, career and body columns equal on a player's two panel rows, as in the real panel."""
    from openrapm.features import BIO, CAREER, PANEL_INPUTS, ROLE
    rng = np.random.default_rng(seed)
    shared, panel, table = set(ROLE) | set(CAREER) | set(BIO), [], []
    for player in (101, 102, 103, 104):
        for season in (2024, 2025, 2026):
            same = {name: rng.uniform(1.0, 3.0) for name in shared}
            possessions = {side: float(rng.integers(1000, 3000)) for side in "OD"}
            for side in "OD":
                panel.append({"side": side, "player_id": player, "season": season, "possessions": possessions[side],
                              **{name: rng.uniform(1.0, 3.0) for name in PANEL_INPUTS if name not in shared}, **same})
            off, deff = rng.normal(), rng.normal()
            table.append({"player_id": player, "season": season, "player_name": f"Player {player}",
                          "poss_off": possessions["O"], "poss_def": possessions["D"], "poss_season": possessions["O"],
                          "u_off": rng.normal(), "u_def": rng.normal(), "rating_off": off, "rating_def": deff,
                          "rating_total": off + deff})
    ratings = tmp_path / "outputs" / "season_ratings_product.parquet"
    ratings.parent.mkdir()
    pd.DataFrame(table).to_parquet(ratings)
    cfg = {"_root": str(tmp_path), "paths": {"outputs": str(tmp_path / "outputs")}}
    return cfg, pd.DataFrame(panel)


@pytest.fixture
def small_similarity(tmp_path, monkeypatch):
    cfg, panel = _similarity_world(tmp_path)
    monkeypatch.setattr(site, "trainable_panel", lambda cfg: panel)
    monkeypatch.setattr(site, "SIMILARITY_MATCHES", 3)
    monkeypatch.setattr(site, "SIMILARITY_SHOWN", 3)
    monkeypatch.setattr(site, "SIMILARITY_DEFAULT", (102, 2025))
    return cfg, tmp_path / "site" / "data" / "similarity"


def _files(folder):
    return {path.name: path.read_bytes() for path in sorted(folder.glob("*.json"))}


def test_the_similarity_files_hold_the_pool_and_lists_without_the_players_own_seasons(small_similarity):
    cfg, folder = small_similarity
    assert site.run_site(cfg, out_dir=folder.parents[1], built="2026-10-12", similarity=True) == 0
    assert sorted(_files(folder)) == ["2024.json", "2025.json", "2026.json", "player_seasons.json"]
    pool = json.loads((folder / "player_seasons.json").read_text(encoding="utf-8"))
    assert pool["meta"]["default"] == [102, 2025] and pool["meta"]["fields"] == site.SIMILARITY_FIELDS
    assert [row[2] for row in pool["rows"]] == [2024] * 4 + [2025] * 4 + [2026] * 4          # season, then id
    players = [row[0] for row in pool["rows"]]
    for season in (2024, 2025, 2026):
        document = json.loads((folder / f"{season}.json").read_text(encoding="utf-8"))
        first = document["meta"]["first"]
        assert pool["rows"][first][2] == season and len(document["rows"]) == 4
        for offset, (seasons, near, others, other_near) in enumerate(document["rows"]):
            own = players[first + offset]
            assert len(seasons) == len(others) == 3 and own not in [players[i] for i in seasons + others]
            assert near == sorted(near) and len({players[i] for i in others}) == 3
    assert pool["meta"]["built"] == "2026-10-12"


def test_a_rebuild_writes_the_same_bytes_and_drops_a_season_no_longer_in_the_pool(small_similarity):
    cfg, folder = small_similarity
    site.run_site(cfg, out_dir=folder.parents[1], built="2026-10-12", similarity=True)
    fresh = _files(folder)
    (folder / "1999.json").write_text("{}", encoding="utf-8")
    site.run_site(cfg, out_dir=folder.parents[1], built="2026-10-12", similarity=True)
    assert _files(folder) == fresh


def test_every_page_carries_the_same_four_tabs_with_its_own_marked():
    import re
    from pathlib import Path
    docs = Path(__file__).resolve().parents[1] / "docs"
    tabs = ["./", "decompose.html", "bias.html", "similarity.html"]
    navs = {page: (docs / page).read_text(encoding="utf-8") for page in ("index.html", "decompose.html",
                                                                         "bias.html", "similarity.html")}
    navs["PAGE_HEAD"] = site.PAGE_HEAD + "</nav>"
    for page, text in navs.items():
        nav = re.search(r'<nav class="site">(.*?)</nav>', text, re.S).group(1)
        links = re.findall(r'<a href="([^"]+)"( aria-current="page")?>', nav)
        assert [href for href, _ in links] == tabs, page
        own = "./" if page == "index.html" else "bias.html" if page == "PAGE_HEAD" else page
        assert [href for href, current in links if current] == [own], page

