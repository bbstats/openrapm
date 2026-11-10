"""`openrapm site` (src/openrapm/commands/site.py) on small made-up tables: the form of ratings.json, the Bias
page's file, and the decomposition's row bookkeeping.

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
                         "player_id": [203999, 203999, 1, 2], "wosa": [14.126, 15.9, 0.4, -0.1]})


def test_ratings_rows_keep_players_with_possessions_best_first_at_two_decimals():
    rows = site.ratings_rows(_table())
    assert [(row["s"], row["id"]) for row in rows] == [(2025, 203999), (2026, 203999), (2026, 1)]
    assert rows[1] == {"s": 2026, "n": "Nikola Jokić", "o": 4.89, "d": 0.9, "t": 5.79, "p": 5100, "id": 203999,
                       "w": 15.9}
    assert rows[0]["w"] == 14.13
    assert rows[2]["n"] == ""                       # a missing name is written empty


def test_ratings_document_counts_every_player_of_the_table():
    meta = site.ratings_document(_table(), "2026-10-09")["meta"]
    assert meta == {"seasons": [2025, 2026], "default_season": 2026, "built": "2026-10-09", "n_players": 2}


def test_the_pages_show_players_with_60_or_more_possessions():
    """The owner, 2026-10-24: "i want to have a 60 possession minimum, site-wide"."""
    table = _table().assign(poss_season=[5282.0, 59.9, 60.0, 2.0])
    rows = site.ratings_rows(table)
    assert [(row["s"], row["id"]) for row in rows] == [(2025, 203999), (2026, 1)]
    assert site.ratings_document(table, "2026-10-24")["meta"]["n_players"] == 2


def test_a_season_in_progress_is_the_default_once_its_median_player_has_250_possessions():
    """The owner, 2026-10-23: "2027 should not be the DEFAULT on the site until the median player has 250
    possessions played", on offensive possessions; 2026-11-02, the median of the players the page shows."""
    finished = pd.DataFrame({"season": 2026, "poss_season": [5000.0, 3000.0, 400.0]})
    early = pd.DataFrame({"season": 2027, "poss_season": [400.0, 249.0, 100.0, 30.0, 0.0]})  # of the 3 shown: 249
    assert site.default_season(pd.concat([finished, early])) == 2026
    later = early.assign(poss_season=[400.0, 250.0, 100.0, 30.0, 0.0])                       # 30 and 0 not shown
    assert site.default_season(pd.concat([finished, later])) == 2027


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
    assert text.startswith('{"meta":{"seasons":[2025,2026],"default_season":2026,"built":"2026-10-09","n_players":2},'
                           '"rows":[{"s":2025,')
    assert "Joki\\u0107" in text and not text.endswith("\n")
    site.run_site(cfg, out_dir=tmp_path / "site", built="2027-01-01")
    assert json.loads((tmp_path / "site" / "data" / "ratings.json").read_text())["meta"]["built"] == "2027-01-01"


def test_a_missing_ratings_table_stops_the_command(tmp_path):
    with pytest.raises(SystemExit):
        site.run_site({"_root": str(tmp_path)}, ratings=tmp_path / "none.parquet", out_dir=tmp_path)


# ------------------------------------------------------------------------------ the bias page
def _bias_world(seed=2):
    """A made-up 2027: players 1-40 with 300 or more possessions, whom all five systems list; player 41 with 59
    possessions, under the pages' minimum; player 42, shown and missing from xRAPM.  Each system is OpenRAPM plus
    noise, the 3-year RAPM without a date of its own.  Returns (ratings table, panel rows, {key: PublicRatings})."""
    from openrapm.bias import CONSENSUS_WEIGHTS
    from openrapm.data.public_ratings import PublicRatings
    from openrapm.features import BOX_RATES
    rng = np.random.default_rng(seed)
    ids = np.arange(1, 43)
    offense, defense = rng.normal(0, 2, 42), rng.normal(0, 1, 42)
    table = pd.DataFrame({"season": 2027, "player_id": ids, "player_name": [f"Player {i}" for i in ids],
                          "rating_off": offense, "rating_def": defense, "rating_total": offense + defense,
                          "poss_season": np.where(ids == 41, 59.0, rng.integers(300, 900, 42).astype(float)),
                          "wosa": 0.0})
    court = ["on_court_scored_per100", "on_court_allowed_per100", "off_court_scored_per100", "off_court_allowed_per100"]
    rates = {name: rng.normal(0, 1, 42) for name in [*BOX_RATES, *court]}
    panel = pd.concat([pd.DataFrame({"side": side, "player_id": ids, "season": 2027, **rates}) for side in "OD"],
                      ignore_index=True)
    pulls = {}
    for key in CONSENSUS_WEIGHTS:
        listed = ids[ids != 42] if key == "xrapm" else ids
        ratings = pd.DataFrame({"player_id": ids, "offense": 0.7 * offense + rng.normal(0, 1, 42),
                                "defense": 0.7 * defense + rng.normal(0, 0.5, 42)})
        pulls[key] = PublicRatings(ratings[ratings.player_id.isin(listed)].reset_index(drop=True),
                                   None if key == "rapm_3y" else "2026-11-04", "2026-11-05")
    return table, panel, pulls


def test_the_bias_document_counts_its_players_and_names_its_five_systems():
    table, panel, pulls = _bias_world()
    document = site.bias_document(table, panel, pulls, 2027, "2026-11-05", "2026-11-04")
    meta = document["meta"]
    assert {key: meta[key] for key in ("season", "through", "built", "min_possessions", "players_shown", "players")} \
        == dict(season=2027, through="2026-11-04", built="2026-11-05", min_possessions=60, players_shown=41,
                players=40)
    assert meta["fields"] == site.BIAS_FIELDS and meta["multipliers"]["o"] > 0 and meta["multipliers"]["d"] > 0
    assert [(system["key"], system["weight"]) for system in document["systems"]] == [
        ("xrapm", 0.3), ("rapm_3y", 0.2), ("loki", 0.2), ("darko", 0.15), ("pulse", 0.15)]
    assert [system["players"] for system in document["systems"]] == [40, 41, 41, 41, 41]
    # a system's own date, or the day it was pulled when its page prints none
    assert [system["as_of"] for system in document["systems"]] == ["2026-11-04", "2026-11-05", "2026-11-04",
                                                                  "2026-11-04", "2026-11-04"]
    assert document["systems"][2]["by"] == "LUMA Basketball Research and CourtShare"
    assert set(document["systems"][0]) == {"key", "name", "by", "link", "weight", "as_of", "players"}


def test_the_bias_rows_are_the_fifteen_stats_and_every_row_adds_up_at_two_decimals():
    document = site.bias_document(*_bias_world(), 2027, "2026-11-05", "2026-11-04")
    rows = [dict(zip(site.BIAS_FIELDS, row)) for row in document["rows"]]
    assert [row["label"] for row in rows] == [
        "Made threes", "Missed threes", "Made twos", "Missed twos", "Made free throws", "Missed free throws",
        "Offensive rebounds", "Defensive rebounds", "Assists", "Turnovers", "Steals", "Blocks", "Personal fouls",
        "On-court plus-minus", "Off-court plus-minus"]
    assert rows[0]["stat"] == "threes_made_per100" and rows[13]["stat"] == "on_court_plus_minus_per100"
    for row in rows:
        assert row["t"] == round(row["o"] + row["d"], 2), row["label"]
        assert round(row["o"], 2) == row["o"] and round(row["d"], 2) == row["d"]
        assert row["o_se"] > 0 and row["d_se"] > 0 and row["t_se"] > 0 and row["sd"] > 0


def test_a_system_that_runs_against_openrapm_stops_the_document():
    """A defense column printed with the other sign would pass every other check and turn the page upside down."""
    from openrapm.data.public_ratings import PublicRatings
    table, panel, pulls = _bias_world()
    flipped = pulls["loki"].table.assign(defense=-pulls["loki"].table.defense)
    pulls["loki"] = PublicRatings(flipped, "2026-11-01", "2026-11-05")
    with pytest.raises(SystemExit, match="LOKI defense runs against OpenRAPM's"):
        site.bias_document(table, panel, pulls, 2027, "2026-11-05", "2026-11-04")


@pytest.fixture
def bias_site(tmp_path, monkeypatch):
    """`run_site(bias=True)` on the made-up 2027: the stored pulls, the panel and the game dates handed in."""
    table, panel, pulls = _bias_world()
    ratings = tmp_path / "outputs" / "season_ratings_product.parquet"
    ratings.parent.mkdir()
    table.to_parquet(ratings)
    stints = pd.DataFrame({"game_date": ["2026-10-20", "2026-11-04"]})
    monkeypatch.setattr(site.public_ratings, "latest", lambda cfg, season: dict(pulls) if season == 2027 else {})
    monkeypatch.setattr(site, "panel_of_season", lambda cfg, season: panel[panel.season == season])
    monkeypatch.setattr(site, "season_stints", lambda season, cfg: stints)
    cfg = {"_root": str(tmp_path), "paths": {"outputs": str(tmp_path / "outputs")}}
    return cfg, tmp_path / "site", pulls


def test_the_bias_file_is_written_whole_and_the_same_bytes_twice(bias_site, capsys):
    cfg, out_dir, _ = bias_site
    assert site.run_site(cfg, out_dir=out_dir, built="2026-11-05", bias=True) == 0
    path = out_dir / "data" / "bias.json"
    first = path.read_bytes()
    document = json.loads(first)
    assert document["meta"]["season"] == 2027 and document["meta"]["through"] == "2026-11-04"
    assert len(document["rows"]) == 15 and first.startswith(b'{"meta":{"season":2027,"through":"2026-11-04",')
    assert "40 of the 41 players shown for 2027 are in all 5 systems" in capsys.readouterr().out
    site.run_site(cfg, out_dir=out_dir, built="2026-11-05", bias=True)
    assert path.read_bytes() == first and sorted(p.name for p in path.parent.iterdir()) == ["bias.json",
                                                                                             "ratings.json"]


def test_a_system_never_pulled_or_dated_before_the_season_leaves_the_bias_file_as_it_was(bias_site):
    from openrapm.data.public_ratings import PublicRatings
    cfg, out_dir, pulls = bias_site
    site.run_site(cfg, out_dir=out_dir, built="2026-11-05", bias=True)
    path = out_dir / "data" / "bias.json"
    before = path.read_bytes()
    kept = pulls.pop("xrapm")
    with pytest.raises(SystemExit, match="no stored table of xRAPM for 2027; `openrapm data public-ratings`"):
        site.run_site(cfg, out_dir=out_dir, built="2026-11-06", bias=True)
    pulls["xrapm"] = PublicRatings(kept.table, "2026-04-12", "2026-11-05")       # last season's table still up
    with pytest.raises(SystemExit, match="xRAPM 2026-04-12: dated before 2027's first game, 2026-10-20"):
        site.run_site(cfg, out_dir=out_dir, built="2026-11-06", bias=True)
    assert path.read_bytes() == before
    # the ratings file is written first either way, so a missing system never holds the ratings back
    assert json.loads((out_dir / "data" / "ratings.json").read_text())["meta"]["built"] == "2026-11-06"


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
                          "rating_total": off + deff, "wosa": rng.normal()})
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


def test_every_page_carries_the_same_tabs_with_its_own_marked():
    import re
    from pathlib import Path
    docs = Path(__file__).resolve().parents[1] / "docs"
    tabs = ["./", "power.html", "similarity.html", "bias.html", "about.html", "changelog.html"]
    pages = ("about.html", "index.html", "power.html", "similarity.html", "bias.html", "changelog.html")
    navs = {page: (docs / page).read_text(encoding="utf-8") for page in pages}
    for page, text in navs.items():
        nav = re.search(r'<nav class="site">(.*?)</nav>', text, re.S).group(1)
        links = re.findall(r'<a href="([^"]+)"( aria-current="page")?(?: hidden)?>', nav)
        assert [href for href, _ in links] == tabs, page
        own = "./" if page == "index.html" else page
        assert [href for href, current in links if current] == [own], page


def test_the_old_decomposition_address_sends_its_links_to_the_ratings_page():
    from pathlib import Path
    text = (Path(__file__).resolve().parents[1] / "docs" / "decompose_openrapm.html").read_text(encoding="utf-8")
    assert 'location.replace("./#"' in text and 'h.set("view", "detail")' in text


def test_the_vanilla_pages_old_address_opens_the_openrapm_decomposition_and_nothing_links_it():
    from pathlib import Path
    docs = Path(__file__).resolve().parents[1] / "docs"
    text = (docs / "decompose.html").read_text(encoding="utf-8")
    assert 'location.replace("./#"' in text and 'to.set("view", "detail")' in text
    assert '<nav class="site">' not in text
    for page in ("about.html", "index.html", "similarity.html", "bias.html", "site.css"):
        assert "decompose.html" not in (docs / page).read_text(encoding="utf-8"), page
    assert not (docs / "data" / "decompose_1y.json").exists() and not (docs / "data" / "decompose_3y.json").exists()



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
    final["box_adj_off"], final["box_adj_def"] = rng.normal(0, 0.5, len(final)), rng.normal(0, 0.5, len(final))
    final["rating_off"], final["rating_def"] = step3.rating_off + final.c_off, step3.rating_def + final.c_def
    final["rating_total"] = final.rating_off + final.rating_def
    final["wosa_off"], final["wosa_def"] = rng.normal(0, 2.0, len(final)), rng.normal(0, 2.0, len(final))
    final["wosa"] = final.wosa_off + final.wosa_def + rng.normal(0, 0.05, len(final))
    final["minutes_rs"] = rng.uniform(0, 2500, len(final))
    final["above_off"], final["above_def"] = final.rating_off - 0.7, final.rating_def - 0.3
    final["leverage_off"], final["leverage_def"] = rng.uniform(0.9, 1.1, len(final)), rng.uniform(0.9, 1.1, len(final))
    final["leverage"] = (final.leverage_off + final.leverage_def) / 2.0
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

        def season_teams(self, season):
            return {int(player_id): "BOS" for player_id in final.player_id}

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
    assert meta["percentile_min"] == {}                   # a configuration naming no seasons has none in progress
    published = {row["id"]: row for row in json.loads((data / "ratings.json").read_text(encoding="utf-8"))["rows"]}
    assert sorted(row[1] for row in rows) == sorted(published)
    assert [row[5][-1] for row in rows] == sorted((row[5][-1] for row in rows), reverse=True)
    assert meta["wosa_parts"] == site.WOSA_PARTS
    by_id = final.set_index("player_id")
    for name, player_id, possessions, off, deff, total, team, wosa, minutes, *wosa_sides in rows:
        assert name == f"Player {player_id}" and possessions == published[player_id]["p"] and team == "BOS"
        assert wosa == published[player_id]["w"] and minutes == round(by_id.minutes_rs[player_id])
        mine = by_id.loc[player_id]
        assert wosa_sides == [[round(mine.above_off, 2), round(mine.leverage_off, 2), round(mine.wosa_off, 2)],
                              [round(mine.above_def, 2), round(mine.leverage_def, 2), round(mine.wosa_def, 2)],
                              [round(mine.above_off + mine.above_def, 2), round(mine.leverage, 2), wosa]]
        for pieces, rating in ((off, "o"), (deff, "d"), (total, "t")):
            assert len(pieces) == 6 and pieces[-1] == published[player_id][rating]
            assert abs(sum(pieces[:-1]) - pieces[-1]) < 0.031           # five pieces at two decimals each
        assert all(abs(t - (a + b)) < 0.0101 for t, a, b in zip(total, off, deff))      # a total is the two sides


def test_the_list_file_is_the_detail_files_ratings_and_nothing_else(openrapm_site):
    """ratings_list.json, what the ratings page draws as it opens: written beside the Detail file, from its document."""
    ratings, _, _, data = openrapm_site
    site.run_site({}, ratings=ratings, out_dir=data.parent, built="2026-10-13", openrapm_decomposition=True)
    detail = json.loads((data / site.OPENRAPM_FILE).read_text(encoding="utf-8"))
    listed = json.loads((data / site.RATINGS_LIST_FILE).read_text(encoding="utf-8"))
    assert listed["meta"] == {"built": "2026-10-13", "seasons": [2002], "default_season": 2002, "percentile_min": {},
                              "rookies": detail["meta"]["rookies"], "through": {}, "fields": site.RATINGS_LIST_FIELDS}
    assert list(listed["rows"]) == ["2002"] and len(listed["rows"]["2002"]) == len(detail["rows"]["2002"])
    for short, row in zip(listed["rows"]["2002"], detail["rows"]["2002"]):
        name, player_id, possessions, off, deff, total, team, wosa = row[:8]
        assert short == [name, player_id, possessions, team, off[-1], deff[-1], total[-1], wosa]
    assert (data / site.RATINGS_LIST_FILE).stat().st_size < 0.5 * (data / site.OPENRAPM_FILE).stat().st_size


def test_the_list_file_names_the_day_a_season_in_progress_runs_through():
    detail = {"meta": {"built": "2026-11-09", "seasons": [2027, 2026], "default_season": 2027,
                       "percentile_min": {"2027": 60}, "rookies": {"2027": [7], "2026": []},
                       "fields": site.OPENRAPM_FIELDS, "pieces": site.OPENRAPM_PIECES},
              "rows": {"2027": [["Victor Wembanyama", 1641705, 620, [3.9, 0.3, 0.2, 0.0, 0.66, 5.06],
                                 [3.9, -0.3, 0.6, 0.0, 0.07, 4.27], [7.8, 0.0, 0.8, 0.0, 0.73, 9.33], "SAS", 1.9, 500,
                                 [4.0, 1.0, 1.0], [4.0, 1.0, 0.9], [8.0, 1.0, 1.9]]],
                       "2026": []}}
    listed = site.ratings_list_document(detail, {2027: "2026-11-08"})
    assert listed["rows"] == {"2027": [["Victor Wembanyama", 1641705, 620, "SAS", 5.06, 4.27, 9.33, 1.9]], "2026": []}
    assert listed["meta"]["through"] == {"2027": "2026-11-08"} and listed["meta"]["percentile_min"] == {"2027": 60}
    assert site.ratings_list_document(detail)["meta"]["through"] == {}        # every season finished
    assert site.last_game_days({}) == {}                                       # no data folder, none in progress


def test_the_pieces_are_the_tables_columns_and_a_split_of_the_games_part(openrapm_site):
    """Box score and swap adjustment are prior_* and c_* as the table has them; the other three add up to u."""
    ratings, final, _, data = openrapm_site
    site.run_site({}, ratings=ratings, out_dir=data.parent, built="2026-10-13", openrapm_decomposition=True)
    rows = json.loads((data / site.OPENRAPM_FILE).read_text(encoding="utf-8"))["rows"]["2002"]
    table = final.set_index("player_id")
    for _, player_id, _, off, deff, *_ in rows[:25]:
        for pieces, side in ((off, "off"), (deff, "def")):
            adj = table.loc[player_id, f"box_adj_{side}"]
            assert pieces[0] == round(table.loc[player_id, f"prior_{side}"] - adj, 2)
            assert pieces[4] == round(adj + table.loc[player_id, f"c_{side}"], 2)
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
    rows = pd.DataFrame({"poss_season": [1000.7, 0.0, 500.2], "wosa": [4.567, 0.0, -1.234],
                         "wosa_off": [4.0, 0.0, -1.0], "wosa_def": [0.5, 0.0, -0.2], "minutes_rs": [2000.4, 0.0, 999.6],
                         "above_off": [5.555, 0.0, -1.0], "above_def": [1.0, 0.0, -0.5], "leverage": [1.054, 0.0, 0.9],
                         "leverage_off": [1.06, 0.0, 0.91], "leverage_def": [1.05, 0.0, 0.89]})
    columns = {f"{side}_{piece}": [0.111, 0.222, 0.333] for side in site.OPENRAPM_SIDES
               for piece in site.OPENRAPM_PIECES}
    columns["total_rating"] = [1.234, 9.0, 3.456]
    pieces = pd.DataFrame({"player_id": [1, 2, 3], **columns})
    got = site.openrapm_rows(rows, pieces, np.array(["A", "B", "C"], dtype=object), {3: "SAC/IND"})
    assert [(row[0], row[1], row[2]) for row in got] == [("C", 3, 500), ("A", 1, 1000)]       # whole possessions
    assert [row[6] for row in got] == ["SAC/IND", ""]
    assert [row[7] for row in got] == [-1.23, 4.57]
    assert [row[8] for row in got] == [1000, 2000]                                             # whole minutes
    assert got[1][9:] == [[5.55, 1.06, 4.0], [1.0, 1.05, 0.5], [6.55, 1.05, 4.57]]
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
    for _, player_id, _, off, deff, *_ in rows[:25]:
        for pieces, side in ((off, "off"), (deff, "def")):
            assert abs(sum(pieces[1:4]) - table.loc[player_id, f"u_{side}"]) < 0.0151


# ------------------------------------------------------------------------------ the power ratings page
def test_a_rookie_is_a_player_with_no_season_before_counted_from_the_debut(tmp_path, monkeypatch):
    assert site.rookies({}) == {}
    panel = pd.DataFrame({"season": [2004, 2004, 2004, 2004, 2005, 2005],
                          "side": ["off", "def", "off", "def", "off", "def"],
                          "player_id": [23, 23, 7, 7, 23, 23],
                          "seasons_before_from_debut": [0.0, 0.0, 11.0, 11.0, 1.0, 1.0]})
    panel.to_parquet(tmp_path / "panel.parquet")
    monkeypatch.setattr("openrapm.data.panel.resolve", lambda cfg, name: tmp_path / "panel.parquet")
    # one player on two sides counted once; the veteran and the sophomore left out
    assert site.rookies({"_root": ".", "paths": {}}) == {2004: [23]}


def test_the_file_lists_only_the_rookies_the_season_shows(tmp_path):
    def row(player):
        return [f"Player {player}", player, 500, [0.0] * 6, [0.0] * 6, [0.0] * 6, "BOS", 0.0]

    path = tmp_path / "file.json"
    site.write_openrapm_file(path, {2004: [row(23)], 2005: [row(7)]}, "2026-11-02", 13037.0,
                             rookie_ids={2004: [23, 99]})         # 99 played too little to be shown
    assert json.loads(path.read_text(encoding="utf-8"))["meta"]["rookies"] == {"2004": [23], "2005": []}


def test_the_percentile_pool_of_a_season_in_progress_grows_with_the_share_played(monkeypatch):
    assert site.percentile_minimums({}) == {}
    shares = {2027: 0.072}
    monkeypatch.setattr("openrapm.data.seasons.in_progress", lambda cfg: list(shares))
    monkeypatch.setattr("openrapm.data.seasons.share_played", lambda season, cfg: shares[season])
    cfg = {"_root": ".", "first_season": 1997}
    assert site.percentile_minimums(cfg) == {2027: 60}                # 500 x 0.072 = 36, floored at the 60 shown
    shares[2027] = 0.5
    assert site.percentile_minimums(cfg) == {2027: 250}
    shares[2027] = 0.9
    assert site.percentile_minimums(cfg) == {2027: 450}


def test_power_ratings_weigh_each_rating_by_its_share_of_the_last_games_with_blowouts_at_half():
    slots = {f"{side}{i}": [10 * (side == "a") + i] * 3 for side in "ha" for i in range(1, 6)}
    slots["h5"] = [5, 6, 6]                                # player 6 replaces player 5 in the later stints
    dates = pd.to_datetime(["2026-10-20", "2026-10-20", "2026-10-22"])
    stints = pd.DataFrame({"game_id": ["g1", "g1", "g2"], "game_date": dates,
                           "poss_h": [40, 10, 50], "poss_a": [40, 10, 50], "is_gt": [False, True, False], **slots})
    ids = list(range(1, 7)) + list(range(11, 16))
    teams = pd.DataFrame([(game, pid, "HOM" if pid < 10 else "AWY", "Home" if pid < 10 else "Away")
                          for game in ("g1", "g2") for pid in ids], columns=["game_id", "player_id", "tm", "name"])
    games = site.player_game_possessions(stints, teams)
    # game 1 counts 80 + 20 x 0.5 = 90 possessions; player 5 played the 80, player 6 the 10 of blowout and game 2's 100
    both = site.team_shares(games, n_games=10)
    shares = both.set_index("player_id").share
    assert shares[5] == pytest.approx(80 / 190) and shares[6] == pytest.approx(110 / 190)
    assert shares[1] == pytest.approx(1.0) and both.groupby("tm").share.sum().tolist() == pytest.approx([5, 5])
    # the latest game counted double: player 6 played all of it
    latest = site.team_shares(games, n_games=10, recent_games=1).set_index("player_id").share
    assert latest[5] == pytest.approx(80 / 290) and latest[6] == pytest.approx(210 / 290)
    # the last game only: player 5 drops out
    assert 5 not in set(site.team_shares(games, n_games=1).player_id)
    table = pd.DataFrame({"season": 2027, "player_id": ids, "player_name": [f"P{i}" for i in ids],
                          "rating_off": [1.0] * 5 + [3.0] + [0.0] * 5, "rating_def": 0.0,
                          "rating_total": [1.0] * 5 + [3.0] + [0.0] * 5, "poss_season": [500] * 5 + [40] + [500] * 5})
    document = site.power_document(table, games, 2027, "2026-10-28")
    home = next(team for team in document["teams"] if team["tm"] == "HOM")
    plain = 4 + 80 / 190 + 3 * 110 / 190                             # the away team's ratings are all 0
    # each team against the average team, here half the home team's plain sum (2026-10-30)
    assert home["o"] == pytest.approx(round(plain / 2, 2))
    assert document["meta"]["average_team"]["o"] == pytest.approx(round(plain / 2, 2))
    assert document["teams"][0]["tm"] == "HOM" and document["meta"]["through"] == "2026-10-22"
    assert document["meta"]["ready"] is False                         # two games, not 10
    # player 6 is under 60 possessions: counted in the sums, shown as the team's last row
    assert home["players"][-1][:3] == ["Other players", None, round(100 * 110 / 190, 1)]
    # every player's rating is shifted down by a fifth of the average team (2026-11-01), so his row, share times
    # rating, adds up to the team's total with the others, and the away team's players read minus that fifth
    assert home["players"][0][3] == pytest.approx(round(1.0 - plain / 2 / 5, 2))
    # the rows add up to the total but for their rounding, two decimals a rating and one a share, never the average
    # team (1.54 here)
    assert sum(row[2] / 100 * row[3] for row in home["players"]) == pytest.approx(home["o"], abs=0.03)
    away = next(team for team in document["teams"] if team["tm"] == "AWY")
    assert away["o"] == pytest.approx(round(-plain / 2, 2)) and away["players"][0][3] == round(-plain / 2 / 5, 2)
