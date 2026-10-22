"""data/bio.py: who he is (height, weight, draft slot) and where he has been (tenure, teams), for the prior."""
import pandas as pd
import pytest

from openrapm.data.bio import (
    BIO_INPUTS,
    PLAYER_INPUTS,
    TENURE_INPUTS,
    UNDRAFTED,
    bio_inputs,
    season_tenure,
    tenure_inputs,
)


def _roles():
    # player 1: three seasons with team A, then two with team B; player 2: A, B, A (no chain across the
    # change); player 3: traded mid-season 2002 (two rows), more possessions with team B
    rows = [(1, 2000, 10, 1000), (1, 2001, 10, 1000), (1, 2002, 10, 1000), (1, 2003, 20, 1000), (1, 2004, 20, 1000),
            (2, 2001, 10, 500), (2, 2002, 20, 500), (2, 2003, 10, 500),
            (3, 2001, 10, 800), (3, 2002, 10, 300), (3, 2002, 20, 700), (3, 2003, 20, 900),
            (4, 2002, 10, 0)]                                     # on the roster, never played
    return pd.DataFrame(rows, columns=["player_id", "season", "team_id", "poss_on"])


def test_season_tenure_counts_consecutive_seasons_with_the_main_team():
    t = season_tenure(_roles()).set_index(["player_id", "season"]).tenure
    assert t[(1, 2000)] == 1 and t[(1, 2002)] == 3 and t[(1, 2003)] == 1 and t[(1, 2004)] == 2
    assert t[(2, 2001)] == 1 and t[(2, 2002)] == 1 and t[(2, 2003)] == 1
    assert t[(3, 2002)] == 1 and t[(3, 2003)] == 2        # 2002's main team is B (700 > 300), so 2003 is his second
    assert (4, 2002) not in t.index


def test_tenure_chain_steps_over_an_excluded_season():
    t = season_tenure(_roles(), exclude_seasons=[2001]).set_index(["player_id", "season"]).tenure
    assert t[(1, 2002)] == 2 and (1, 2001) not in t.index   # 2000 and 2002 chain across the gap


def test_tenure_inputs_are_possession_weighted_over_the_block_and_aligned():
    out = tenure_inputs(_roles(), [2002, 2003, 2004], [1, 2, 3, 4, 99])
    assert list(out.columns) == TENURE_INPUTS and len(out) == 5
    assert abs(out.tenure[0] - (3 + 1 + 2) / 3) < 1e-12 and out.n_teams[0] == 2
    assert out.tenure[1] == 1.0 and out.n_teams[1] == 2
    assert abs(out.tenure[2] - (1 * 1000 + 2 * 900) / 1900) < 1e-12 and out.n_teams[2] == 2
    assert out.tenure[3] == 1.0 and out.n_teams[3] == 1                # never played: the defaults
    assert out.tenure[4] == 1.0 and out.n_teams[4] == 1                # never seen at all


def test_bio_inputs_fill_the_unknown_with_medians_and_undrafted():
    bio = pd.DataFrame({"player_id": [1, 2, 3], "height": [72.0, 80.0, 84.0], "weight": [180.0, 220.0, 260.0],
                        "draft_pick": [1.0, UNDRAFTED, 30.0]})
    out = bio_inputs(bio, [3, 1, 42])
    assert list(out.columns) == BIO_INPUTS
    assert out.height.tolist() == [84.0, 72.0, 80.0] and out.weight.tolist() == [260.0, 180.0, 220.0]
    assert out.draft_pick.tolist() == [30.0, 1.0, UNDRAFTED]


def test_the_player_inputs_carry_the_body_columns():
    assert set(PLAYER_INPUTS) >= set(BIO_INPUTS)


def _feed(tmp_path):
    """A bio feed of two seasons, 2014 before the barefoot heights and 2021 after them."""
    folder = tmp_path / "raw" / "bio"
    folder.mkdir(parents=True)
    pd.DataFrame({"PLAYER_ID": [1, 2], "PLAYER_HEIGHT_INCHES": [81, 84], "PLAYER_WEIGHT": [205, 250],
                  "DRAFT_NUMBER": [15, 2]}).to_parquet(folder / "2014_RS.parquet", index=False)
    pd.DataFrame({"PLAYER_ID": [1], "PLAYER_HEIGHT_INCHES": [82], "PLAYER_WEIGHT": [242],
                  "DRAFT_NUMBER": [15]}).to_parquet(folder / "2021_RS.parquet", index=False)
    return {"_root": str(tmp_path), "paths": {"raw": "raw", "cache": "cache"}}


def test_each_season_reads_its_own_listing_on_the_barefoot_ruler(tmp_path):
    """Player 1 is listed at 81 in and 205 lb in 2014 and 82 in and 242 lb in 2021, as Giannis Antetokounmpo was; his
    career median would read the same 81.5 in and 223.5 lb in both.  Heights before 2020 lose the 0.575 in of the
    shoes; a player a season does not list reads that season's medians."""
    from openrapm.data.bio import SHOE_INCHES, season_body
    cfg = _feed(tmp_path)
    early = season_body(cfg, 2014, [1, 2, 3])
    assert early.height.tolist() == pytest.approx([81 - SHOE_INCHES, 84 - SHOE_INCHES, 82.5 - SHOE_INCHES])
    assert early.weight.tolist() == [205.0, 250.0, 227.5]
    late = season_body(cfg, 2021, [1, 2])
    assert late.height.tolist() == [82.0, 82.0] and late.weight.tolist() == [242.0, 242.0]


def test_the_player_inputs_read_the_seasons_listing_when_the_config_asks(tmp_path):
    from openrapm.data.bio import player_inputs
    cfg = _feed(tmp_path)
    shipped = player_inputs(cfg, _roles(), [2014], [1])
    assert shipped.weight.tolist() == [223.5]                          # the career median, both seasons
    season = player_inputs({**cfg, "body_inputs": "season_listing"}, _roles(), [2014], [1])
    assert season.weight.tolist() == [205.0] and season.draft_pick.tolist() == shipped.draft_pick.tolist()
    with pytest.raises(ValueError):
        player_inputs({**cfg, "body_inputs": "listed"}, _roles(), [2014], [1])
