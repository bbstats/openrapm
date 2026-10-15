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
    tabs = ["./", "decompose_openrapm.html", "bias.html", "similarity.html"]
    pages = ("index.html", "decompose_openrapm.html", "decompose.html", "bias.html", "similarity.html")
    navs = {page: (docs / page).read_text(encoding="utf-8") for page in pages}
    navs["PAGE_HEAD"] = site.PAGE_HEAD + "</nav>"
    for page, text in navs.items():
        nav = re.search(r'<nav class="site">(.*?)</nav>', text, re.S).group(1)
        links = re.findall(r'<a href="([^"]+)"( aria-current="page")?>', nav)
        assert [href for href, _ in links] == tabs, page
        # both decomposition pages are the one Decomposition tab
        own = ("./" if page == "index.html" else "bias.html" if page == "PAGE_HEAD" else
               "decompose_openrapm.html" if page.startswith("decompose") else page)
        assert [href for href, current in links if current] == [own], page


def test_both_decomposition_pages_carry_the_same_toggle_with_their_own_marked():
    import re
    from pathlib import Path
    docs = Path(__file__).resolve().parents[1] / "docs"
    seen = {}
    for page in ("decompose_openrapm.html", "decompose.html"):
        text = (docs / page).read_text(encoding="utf-8")
        toggle = re.search(r'<nav class="toggle"[^>]*>(.*?)</nav>', text, re.S).group(1)
        seen[page] = re.findall(r'<a href="([^"]+)"( aria-current="page")?>([^<]+)</a>', toggle)
    links = {page: [(href, label) for href, _, label in found] for page, found in seen.items()}
    assert links["decompose_openrapm.html"] == [("decompose_openrapm.html", "OpenRAPM"),
                                                ("decompose.html", "Vanilla RAPM")]
    assert links["decompose.html"] == links["decompose_openrapm.html"]
    assert [href for href, current, _ in seen["decompose_openrapm.html"] if current] == ["decompose_openrapm.html"]
    assert [href for href, current, _ in seen["decompose.html"] if current] == ["decompose.html"]



# ------------------------------------------------------------------------------ the OpenRAPM decomposition page
@pytest.fixture(scope="module")
def rated_season(synthetic_league, synthetic_design):
    """2002 of a simulated league rated through step 3 as the build does (both sides on the one design): (the
    design, the step-3 table in the published schema, (game id, player id) -> team id)."""
    from openrapm import prior_rapm, rapm
    sim = synthetic_league(n_seasons=3, seed=4, po_series_per_season=2)
    design = synthetic_design(sim, [2002])
    base = rapm.SeasonSums(0.0).add(2002, design).ratings((), (3000.0, 3000.0, 0.0))
    fit = prior_rapm.fit(design, dict(zip(base.player_id, base.offense)), dict(zip(base.player_id, base.defense)),
                         penalty=13037.0)
    table = prior_rapm.centre(prior_rapm.season_table(fit, fit, prior_rapm.defence_possessions(design), 2002))
    ids = design.spec.players["player_id"]
    names = pd.DataFrame({"player_id": ids, "player_name": [f"Player {i}" for i in ids]})
    ps, box = sim["truth"]["ps"], sim["box"]
    box = box[box.season == 2002].merge(ps[ps.season == 2002][["player_id", "team"]], on="player_id")
    return design, prior_rapm.published_table(table, names), box.set_index(["game_id", "player_id"]).team


@pytest.fixture
def openrapm_site(rated_season, tmp_path, monkeypatch):
    """The ratings table (the step-3 table plus a swap adjustment) and the step-3 table on disk, and the three
    readers of real files (designs, box scores, names) replaced by the simulated season."""
    design, step3, teams = rated_season
    rng = np.random.default_rng(1)
    final = step3.copy()
    final["c_off"], final["c_def"] = rng.normal(0, 0.3, len(final)), rng.normal(0, 0.3, len(final))
    final["rating_off"], final["rating_def"] = step3.rating_off + final.c_off, step3.rating_def + final.c_def
    final["rating_total"] = final.rating_off + final.rating_def
    folder = tmp_path / "outputs"
    folder.mkdir()
    ratings = folder / "season_ratings_product.parquet"
    final.to_parquet(ratings)
    step3.to_parquet(site.step3_path(ratings))

    class Designs:
        def __init__(self, cfg):
            pass

        def season(self, season, target):
            return design

        def clear(self):
            pass

    class Boxes:
        def __init__(self, cfg):
            pass

        def teams(self, season):
            return teams

        def names(self, ids, seasons):
            return pd.Series(dtype=object)

    monkeypatch.setattr(site, "DesignCache", Designs)
    monkeypatch.setattr(site, "BoxScoreFiles", Boxes)
    monkeypatch.setattr(site, "name_table",
                        lambda path, cfg: pd.read_parquet(path, columns=["player_id", "season", "player_name"]))
    return ratings, final, step3, tmp_path / "site" / "data"


def test_the_openrapm_file_adds_up_and_carries_ratings_jsons_ratings(openrapm_site):
    ratings, final, _, data = openrapm_site
    assert site.run_site({}, ratings=ratings, out_dir=data.parent, built="2026-10-13", openrapm_decomposition=True) == 0
    document = json.loads((data / site.OPENRAPM_FILE).read_text(encoding="utf-8"))
    meta, rows = document["meta"], document["rows"]["2002"]
    assert meta["fields"] == site.OPENRAPM_FIELDS and meta["pieces"] == site.OPENRAPM_PIECES
    assert meta["built"] == "2026-10-13" and meta["penalty"] == 13037.0
    assert meta["seasons"] == [2002] and list(document["rows"]) == ["2002"]
    published = {row["id"]: row for row in json.loads((data / "ratings.json").read_text(encoding="utf-8"))["rows"]}
    assert sorted(row[1] for row in rows) == sorted(published)
    assert [row[5][-1] for row in rows] == sorted((row[5][-1] for row in rows), reverse=True)
    for name, player_id, possessions, off, deff, total in rows:
        assert name == f"Player {player_id}" and possessions == published[player_id]["p"]
        for pieces, rating in ((off, "o"), (deff, "d"), (total, "t")):
            assert len(pieces) == 6 and pieces[-1] == published[player_id][rating]
            assert abs(sum(pieces[:-1]) - pieces[-1]) < 0.031           # five pieces at two decimals each
        assert all(abs(t - (a + b)) < 0.0101 for t, a, b in zip(total, off, deff))      # a total is the two sides


def test_the_pieces_are_the_tables_columns_and_a_split_of_the_games_part(openrapm_site):
    """Box score and swap adjustment are prior_* and c_* as the table has them; the other three add up to u."""
    ratings, final, _, data = openrapm_site
    site.run_site({}, ratings=ratings, out_dir=data.parent, built="2026-10-13", openrapm_decomposition=True)
    rows = json.loads((data / site.OPENRAPM_FILE).read_text(encoding="utf-8"))["rows"]["2002"]
    table = final.set_index("player_id")
    for _, player_id, _, off, deff, _ in rows[:25]:
        for pieces, side in ((off, "off"), (deff, "def")):
            assert pieces[0] == round(table.loc[player_id, f"prior_{side}"], 2)
            assert pieces[4] == round(table.loc[player_id, f"c_{side}"], 2)
            assert abs(sum(pieces[1:4]) - table.loc[player_id, f"u_{side}"]) < 0.0151     # three pieces at two decimals


def test_the_run_stops_when_solving_step_3_again_does_not_give_the_tables_games_part(openrapm_site):
    ratings, final, step3, data = openrapm_site
    for table, path in ((final, ratings), (step3, site.step3_path(ratings))):
        table.assign(u_off=table.u_off + 0.01 * (table.index == 0)).to_parquet(path)
    with pytest.raises(SystemExit, match="check failed"):
        site.write_openrapm_decomposition({}, ratings, data, "2026-10-13")
    assert not (data / site.OPENRAPM_FILE).exists()


def test_a_ratings_table_and_step_3_table_of_different_builds_are_refused(openrapm_site):
    ratings, _, step3, _ = openrapm_site
    step3.assign(u_off=step3.u_off + 0.01).to_parquet(site.step3_path(ratings))
    with pytest.raises(SystemExit, match="same games' part"):
        site.published_parts(ratings)
    site.step3_path(ratings).unlink()
    with pytest.raises(SystemExit, match="does not exist"):
        site.published_parts(ratings)


def test_openrapm_rows_drop_players_without_possessions_and_sort_by_total():
    rows = pd.DataFrame({"poss_season": [1000.7, 0.0, 500.2]})
    columns = {f"{side}_{piece}": [0.111, 0.222, 0.333] for side in site.OPENRAPM_SIDES
               for piece in site.OPENRAPM_PIECES}
    columns["total_rating"] = [1.234, 9.0, 3.456]
    pieces = pd.DataFrame({"player_id": [1, 2, 3], **columns})
    got = site.openrapm_rows(rows, pieces, np.array(["A", "B", "C"], dtype=object))
    assert [(row[0], row[1], row[2]) for row in got] == [("C", 3, 500), ("A", 1, 1000)]       # whole possessions
    assert got[0][3] == [0.33] * 6 and got[0][5] == [0.33] * 5 + [3.46]
    assert got[1][4] == [0.11] * 6 and got[1][5] == [0.11] * 5 + [1.23]


def test_a_games_part_multiplied_by_step_4_is_split_with_its_factor(openrapm_site):
    """Experiment 65: step 4 multiplies u by one factor per season and side; the three pieces carry that factor."""
    ratings, final, step3, data = openrapm_site
    scaled = final.assign(u_off=1.3 * final.u_off, u_def=1.5 * final.u_def)
    scaled["rating_off"] = scaled.prior_off + scaled.u_off + scaled.c_off
    scaled["rating_def"] = scaled.prior_def + scaled.u_def + scaled.c_def
    scaled["rating_total"] = scaled.rating_off + scaled.rating_def
    scaled.to_parquet(ratings)
    _, _, factors = site.published_parts(ratings)
    assert factors[2002]["off"] == pytest.approx(1.3) and factors[2002]["def"] == pytest.approx(1.5)
    assert site.run_site({}, ratings=ratings, out_dir=data.parent, built="2026-10-15", openrapm_decomposition=True) == 0
    rows = json.loads((data / site.OPENRAPM_FILE).read_text(encoding="utf-8"))["rows"]["2002"]
    table = scaled.set_index("player_id")
    for _, player_id, _, off, deff, _ in rows[:25]:
        for pieces, side in ((off, "off"), (deff, "def")):
            assert abs(sum(pieces[1:4]) - table.loc[player_id, f"u_{side}"]) < 0.0151
