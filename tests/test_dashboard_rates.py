"""The dashboard inputs of the season panel (src/openrapm/data/dashboard_rates.py): the pad constant by the method of
moments, a rate against the league's, the totals read from the cached dashboards, and what a missing season or player
gives.  The dashboards are made up: tests cannot reach the network."""
import numpy as np
import pandas as pd
import pytest

from openrapm import box_rates, features
from openrapm.data import dashboard_rates as dr
from openrapm.data.dashboards import pulls, read_dashboard


# ------------------------------------------------------------------------------------------------- the pad constant
def test_the_pad_constant_is_recovered_from_a_simulated_league():
    """True rates with variance 0.36 around 1.6 per 100: noise per possession 160, so k = 160 / 0.36 = 444."""
    rng = np.random.default_rng(0)
    n = rng.uniform(300, 3000, 4000)
    true = rng.gamma(shape=1.6 ** 2 / 0.36, scale=0.36 / 1.6, size=len(n))          # mean 1.6, variance 0.36
    counts = rng.poisson(true * n / 100.0)
    assert 340 < dr.moment_k(counts, n) < 600


def test_a_rate_nobody_differs_on_is_padded_all_the_way_and_a_tiny_league_cannot_be_told_apart():
    n = np.full(60, 1000.0)
    assert dr.moment_k(np.full(60, 16.0), n) == box_rates.K_MAX            # no spread between players
    assert dr.moment_k(np.arange(10.0), np.full(10, 1000.0)) == box_rates.K_MAX      # under 20 players
    assert dr.moment_k(np.zeros(60), n) == box_rates.K_MAX                 # a league rate of 0


def test_a_league_relative_rate_is_the_padded_rate_over_the_league_rate_minus_one():
    """A measured quantity takes the fixed pad.  50 players of 1,000 possessions at 7.0 per 100 and one at 8.4: the bin
    target is the bin's rate, and the answer is (padded - league) / league."""
    n = np.full(51, 1000.0)
    counts = np.r_[np.full(50, 70.0), 84.0]
    out = dr.padded_relative(counts, n, measured=True)
    league = 100.0 * counts.sum() / n.sum()
    padded = (1000.0 * 8.4 + dr.FIXED_PAD * league) / (1000.0 + dr.FIXED_PAD)       # one bin, which holds everyone
    assert out[-1] == pytest.approx((padded - league) / league, rel=1e-9)
    ordinary = (1000.0 * 7.0 + dr.FIXED_PAD * league) / (1000.0 + dr.FIXED_PAD)
    assert out[0] == pytest.approx((ordinary - league) / league, rel=1e-9)
    assert out[-1] > 0 > out[0]


def test_a_player_without_a_count_or_possessions_is_missing_not_average():
    n = np.array([1000.0, 1000.0, 0.0, 1000.0] + [1000.0] * 40)
    counts = np.array([70.0, np.nan, 50.0, 80.0] + [75.0] * 40)
    out = dr.padded_relative(counts, n, measured=True)
    assert np.isnan(out[1]) and np.isnan(out[2]) and np.isfinite(out[[0, 3]]).all()
    assert np.isnan(dr.padded_relative(np.full(40, np.nan), np.full(40, 1000.0))).all()
    assert np.isnan(dr.padded_relative(np.zeros(40), np.full(40, 1000.0))).all()      # a league rate of 0


# ----------------------------------------------------------------------------------------- the cached dashboards
IDS = np.arange(1001, 1061)
NEEDED = {}
for _spec in dr.COUNTS:
    NEEDED.setdefault((_spec.source, _spec.kind), {"id": _spec.id_column, "columns": set()})["columns"].add(
        _spec.column)
for _spec in dr.OVER_EXPECTED:
    NEEDED.setdefault(("tracking", _spec.kind), {"id": "CLOSE_DEF_PERSON_ID", "columns": set()})["columns"].update(
        {_spec.made, _spec.attempts, _spec.expected_pct})


@pytest.fixture
def cfg(tmp_path):
    return {"_root": str(tmp_path), "paths": {"raw": str(tmp_path / "raw")}}


def write_matchups(cfg, season: int, pairs: pd.DataFrame):
    """The matchups dashboard: `pairs` (DEF_PLAYER_ID, OFF_PLAYER_ID, PARTIAL_POSS, PLAYER_PTS) split in two halves, one
    per phase, so that the phases must be added."""
    for phase in ("RS", "PO"):
        half = pairs.assign(PARTIAL_POSS=pairs.PARTIAL_POSS / 2.0, PLAYER_PTS=pairs.PLAYER_PTS / 2.0)
        path = cfg_path(cfg, "matchups", "season", season, phase)
        path.parent.mkdir(parents=True, exist_ok=True)
        half.to_parquet(path, index=False)


def random_pairs(listed, seed=0):
    """Five attackers for each listed defender, made up."""
    rng = np.random.default_rng(seed)
    rows = [(int(d), int(a), rng.uniform(50, 300)) for d in listed for a in rng.choice(listed, 5, replace=False)
            if a != d]
    frame = pd.DataFrame(rows, columns=["DEF_PLAYER_ID", "OFF_PLAYER_ID", "PARTIAL_POSS"])
    return frame.assign(PLAYER_PTS=frame.PARTIAL_POSS * rng.uniform(0.8, 1.3, len(frame)))


def write_dashboards(cfg, season: int, listed=IDS, sources=("tracking", "hustle", "matchups"), seed=0):
    """Every dashboard the module reads, for `listed` players, both phases, counts made up."""
    rng = np.random.default_rng(seed)
    if "matchups" in sources:
        write_matchups(cfg, season, random_pairs(listed, seed))
    for (source, kind), need in NEEDED.items():
        if source not in sources:
            continue
        for phase in ("RS", "PO"):
            frame = pd.DataFrame({need["id"]: listed})
            for column in sorted(need["columns"]):
                pct = column.startswith("NS_")
                frame[column] = (rng.uniform(0.3, 0.6, len(listed)) if pct
                                 else rng.integers(5, 400, len(listed)).astype(float))
            path = cfg_path(cfg, source, kind, season, phase)
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(path, index=False)


def cfg_path(cfg, source, kind, season, phase):
    from openrapm.data.dashboards import dashboard_path
    return dashboard_path(cfg, next(p for p in pulls((source,)) if p.kind == kind), season, phase)


def test_the_dashboards_before_they_start_read_as_no_table_and_a_missing_file_after_it_names_the_command(cfg):
    assert read_dashboard(cfg, "tracking", "pt_drives", 2013) is None                # tracking starts in 2014
    assert read_dashboard(cfg, "hustle", "players", 2015) is None                    # hustle in 2016
    with pytest.raises(FileNotFoundError, match="openrapm data dashboards"):
        read_dashboard(cfg, "tracking", "pt_drives", 2014)


def test_the_totals_add_the_phases_and_make_an_over_expected_count_from_the_dashboards_own_expected_pct(cfg):
    write_dashboards(cfg, 2026)
    totals = dr.season_counts(cfg, 2026)
    both = sum(pd.read_parquet(cfg_path(cfg, "hustle", "players", 2026, phase)).set_index("PLAYER_ID").DEFLECTIONS
               for phase in ("RS", "PO"))
    assert totals["deflections_vs_league_avg"].reindex(IDS).to_numpy() == pytest.approx(both.reindex(IDS).to_numpy())
    rim = [pd.read_parquet(cfg_path(cfg, "tracking", "defend_less_than_6ft", 2026, phase)).set_index(
        "CLOSE_DEF_PERSON_ID") for phase in ("RS", "PO")]
    expected = sum((f.FGM_LT_06 - f.NS_LT_06_PCT * f.FGA_LT_06) for f in rim)
    assert totals["rim_makes_over_expected_per_shot"].reindex(IDS).to_numpy() == pytest.approx(
        expected.reindex(IDS).to_numpy())
    assert totals["rim_makes_over_expected_per_shot__shots"].reindex(IDS).to_numpy() == pytest.approx(
        sum(f.FGA_LT_06 for f in rim).reindex(IDS).to_numpy())


def test_a_listed_player_gets_a_value_an_unlisted_one_is_missing_and_a_season_before_the_data_is_all_missing(cfg):
    write_dashboards(cfg, 2026, listed=IDS[:-1])
    n = {"O": np.full(len(IDS), 2000.0), "D": np.full(len(IDS), 2100.0)}
    out = dr.dashboard_rates(cfg, 2026, IDS, n)
    assert list(out["O"].columns) == features.DASHBOARD_RATES == list(out["D"].columns)
    assert out["O"].iloc[:-1].notna().all().all() and out["O"].iloc[-1].isna().all()
    assert out["D"].iloc[:-1].notna().all().all()
    early = dr.dashboard_rates(cfg, 2013, IDS, n)
    assert early["O"].isna().all().all() and early["D"].isna().all().all()


def test_a_season_with_tracking_but_no_hustle_has_only_the_hustle_columns_missing(cfg):
    write_dashboards(cfg, 2015, sources=("tracking",))
    n = {"O": np.full(len(IDS), 2000.0), "D": np.full(len(IDS), 2000.0)}
    out = dr.dashboard_rates(cfg, 2015, IDS, n)["O"]
    assert out[features.TRACKING_RATES].notna().all().all() and out[features.HUSTLE_RATES].isna().all().all()
    assert out[features.MATCHUP_RATES].isna().all().all(), "the matchups start in 2017"


# ---------------------------------------------------------------------------------------------------- assignments
# Defender 1 guarded attacker 1 only (100 points on 100 possessions); defender 2 guarded attacker 1 (150 on 100) and
# attacker 2 (50 on 100).  The league scores 300 points on 300 possessions, 1.0 per possession.
PAIRS = pd.DataFrame({"DEF_PLAYER_ID": [1, 2, 2], "OFF_PLAYER_ID": [1, 1, 2],
                      "PARTIAL_POSS": [100.0, 100.0, 100.0], "PLAYER_PTS": [100.0, 150.0, 50.0]})


def test_an_attackers_rate_leaves_out_the_rows_against_the_defender_and_pads_toward_the_league():
    """Defender 1: attacker 1 without the rows against him scored 150 on 100, padded (150 + 50) / (100 + 50) = 4/3, and
    the 100 possessions are the defender's only ones, so (100 x 4/3 + 50 x 1.0) / (100 + 50) = 1.2222.  Defender 2:
    attacker 1 without his rows scored 100 on 100 (1.0) and attacker 2 has no other rows (the pad alone, 1.0)."""
    guarded, league = dr.assignment_rates(PAIRS, "DEF_PLAYER_ID", "OFF_PLAYER_ID")
    assert league == pytest.approx(1.0)
    assert guarded[1] == pytest.approx((100 * 4 / 3 + 50) / 150) and guarded[2] == pytest.approx(1.0)


def test_the_matchup_columns_read_against_the_league_and_the_mirror_is_sign_flipped(cfg):
    """Attackers' rates against the league: defender 1 guarded scorers 22.2% above it.  Attacker 1 faced defender 1 (no
    other rows: the pad, 1.0) and defender 2 (without attacker 1's rows: 50 on 100, padded 100 / 150 = 2/3):
    (100 x 1.0 + 100 x 2/3 + 50) / (200 + 50) = 0.8667, so the defenders were 13.3% stingier than the league's."""
    write_matchups(cfg, 2026, PAIRS)
    out = dr.matchup_rates(cfg, 2026, [1, 2, 3])
    first, second = features.MATCHUP_RATES
    assert out[first].iloc[:2].to_numpy() == pytest.approx([(100 * 4 / 3 + 50) / 150 - 1.0, 0.0])
    assert out[second].iloc[0] == pytest.approx(1.0 - (100 * 1.0 + 100 * 2 / 3 + 50) / 250)
    assert out[second].iloc[1] == pytest.approx(1.0 - (100 * (200 / 150) + 50) / 150)
    assert out.iloc[2].isna().all(), "a player no pair names is missing, not average"


def test_a_fold_scales_the_matchups_so_the_pad_weighs_more(cfg):
    """Half the season: defender 1's attacker, without the rows against him, scored 75 on 50, padded
    (75 + 50) / (50 + 50) = 1.25, and the defender's 50 possessions give (50 x 1.25 + 50) / (50 + 50) = 1.125."""
    write_matchups(cfg, 2026, PAIRS)
    out = dr.matchup_rates(cfg, 2026, [1, 2], fit_share=0.5)
    assert out[features.MATCHUP_RATES[0]].iloc[0] == pytest.approx(0.125)


def test_the_matchups_before_2017_are_all_missing(cfg):
    assert dr.matchup_rates(cfg, 2016, [1, 2]).isna().all().all()


def test_a_fold_scales_the_over_expected_columns_by_its_share_of_the_season(cfg):
    write_dashboards(cfg, 2026)
    n = {"O": np.full(len(IDS), 2000.0), "D": np.full(len(IDS), 2000.0)}
    whole = dr.dashboard_rates(cfg, 2026, IDS, n)["D"]
    fold = dr.dashboard_rates(cfg, 2026, IDS, n, fit_share=0.75)["D"]
    totals = dr.season_counts(cfg, 2026)
    spec = dr.OVER_EXPECTED[1]
    made, shots = (totals[spec.name].reindex(IDS).to_numpy(), totals[f"{spec.name}__shots"].reindex(IDS).to_numpy())
    assert whole[spec.name].to_numpy() == pytest.approx(made / (shots + spec.pad))
    assert fold[spec.name].to_numpy() == pytest.approx(0.75 * made / (0.75 * shots + spec.pad))
