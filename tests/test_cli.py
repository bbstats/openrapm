"""The command line (src/openrapm/commands/cli.py): the parser's wiring, the seasons it reads, the settings it picks.

Nothing here rates anything: each command's run function is replaced by one that records what it was handed.
"""
import argparse
import os
import subprocess
import sys

import pytest

from openrapm.commands import cli
from openrapm.settings import PRODUCT, TEST


@pytest.mark.parametrize("text, seasons", [
    ("2026", [2026]),
    ("2017-2019", [2017, 2018, 2019]),
    ("2026,2017-2018", [2017, 2018, 2026]),
    ("2018, 2017,2018", [2017, 2018]),
])
def test_seasons_are_read_sorted_and_without_repeats(text, seasons):
    assert cli.parse_seasons(text) == seasons


@pytest.mark.parametrize("text", ["", "2026-2017", "twenty", "2017-", ","])
def test_a_season_list_that_cannot_be_read_is_refused(text):
    with pytest.raises(argparse.ArgumentTypeError):
        cli.parse_seasons(text)


def _parse(*argv):
    return cli.build_parser().parse_args(list(argv))


def test_rate_defaults_are_the_product_setting_itself():
    args = _parse("rate")
    assert (args.setting, args.seasons, args.name, args.save_models) == ("product", None, None, None)
    assert cli.chosen_settings(args) == PRODUCT


def test_the_test_setting_is_the_shipped_test_setting():
    assert cli.chosen_settings(_parse("rate", "--setting", "test")) == TEST


def test_each_rate_option_reaches_its_setting(capsys):
    args = _parse("rate", "--setting", "test", "--defensive-target", "points_with_expected_opponent_threes_all",
                  "--features", "box_score_and_bio", "--booster-params", "other", "--fold-assignment", "1",
                  "--seasons", "2025-2026", "--name", "trial", "--save-models", "trial")
    chosen = cli.chosen_settings(args)
    assert chosen.exclude_neighbours == 1 and chosen.targets.defence == "points_with_expected_opponent_threes_all"
    assert chosen.targets.offence == TEST.targets.offence
    assert chosen.prior.booster_file == "params/booster_params_other.json"
    assert chosen.prior.fold_assignment == 1 and chosen.prior.feature_set == "box_score_and_bio"
    assert chosen.prior.era_halflife == 0.0                   # every season alike unless asked
    assert (args.seasons, args.name, args.save_models) == ([2025, 2026], "trial", "trial")
    assert capsys.readouterr().out == ""                      # plain names: nothing to say


def test_the_era_half_life_reaches_the_prior_and_a_negative_one_is_refused():
    """Experiment 56: on rate and build alike."""
    for command in ("rate", "build"):
        assert cli.chosen_settings(_parse(command, "--era-halflife", "8")).prior.era_halflife == 8.0
    with pytest.raises(SystemExit):
        cli.chosen_settings(_parse("rate", "--era-halflife", "-1"))


def test_the_team_season_penalties_reach_the_labels_and_a_half_pair_is_refused():
    """Experiment 60: on rate and build alike; 0,0 (the default) is the shipped setting."""
    for command in ("rate", "build"):
        labels = cli.chosen_settings(_parse(command, "--team-season-penalties", "9000,6000")).labels
        assert (labels.team_offense_penalty, labels.team_defense_penalty) == (9000.0, 6000.0)
        assert labels.team_flags
    assert cli.chosen_settings(_parse("rate", "--team-season-penalties", "0,0")) == PRODUCT
    for text in ("9000", "9000,0", "a,b", "1,2,3", "-1,5"):
        with pytest.raises(argparse.ArgumentTypeError):
            cli.parse_penalty_pair(text)


def test_the_names_used_before_the_rename_still_work_and_say_so(capsys):
    """x3def and boruta_noonc, typed as before 2026-10-11, run their plain-named settings, one line each."""
    chosen = cli.chosen_settings(_parse("rate", "--defensive-target", "x3def", "--features", "boruta_noonc"))
    assert chosen.targets.defence == "points_with_expected_opponent_threes_all"
    assert chosen.prior.feature_set == "box_score_and_bio"
    said = capsys.readouterr().out.splitlines()
    assert len(said) == 2 and "x3def" in said[0] and "boruta_noonc" in said[1]
    assert cli.chosen_settings(_parse("rate", "--defensive-target", "x3def_w0.25")) == PRODUCT


def test_the_last_occurrence_of_a_flag_wins():
    """scripts/experiment_chain.sh puts the incumbent's flags first so an experiment's own can override them."""
    args = _parse("rate", "--features", "box_score_and_bio", "--booster-params", "lgb1", "--booster-params", "other")
    assert args.booster_params == "other"


def test_data_panel_builds_by_default_and_migrates_when_asked():
    from openrapm.commands import data
    built = _parse("data", "panel")
    assert (built.migrate_from, built.force) == (None, False) and built.run is data.run_panel
    assert _parse("data", "panel", "--force").force
    migrated = _parse("data", "panel", "--migrate-from", "outputs/role_panel_season.parquet")
    assert migrated.migrate_from == "outputs/role_panel_season.parquet" and migrated.run is data.run_panel
    with pytest.raises(SystemExit):
        _parse("data")


def test_data_events_reads_seasons_and_holds_the_parts_to_the_box_score():
    """`openrapm data events 1997 2026`; the check is the parts' sum over the box score's count, minus 1."""
    import numpy as np
    import pandas as pd

    from openrapm.commands import data
    from openrapm.data.event_counts import EVENT_COUNTS
    args = _parse("data", "events", "1997", "2026", "--force")
    assert (args.first_season, args.last_season, args.force, args.run) == (1997, 2026, True, data.run_events)
    assert (_parse("data", "events").first_season, _parse("data", "events").force) == (None, False)
    totals = pd.DataFrame([dict.fromkeys(EVENT_COUNTS, 1.0)], index=[2026])
    totals[EVENT_COUNTS[0]] = 9.0                                  # 9 + 1 + 1 + 1 = 12 defensive rebounds
    box = pd.DataFrame({"season": [2026, 2026], "drb": [6.0, 6.0], "stl": [1.0, 1.0], "fg2m": [0.0, 0.5],
                        "fg3m": [0.0, 0.0], "fg3_miss": [1.0, 1.0], "blk": [1.5, 1.5], "ast": [2.0, 2.0],
                        "tov": [2.0, 2.0], "pf": [5.0, 5.0]})
    check = data.event_check(totals, box).loc[2026]
    assert check["defensive rebounds"] == 0.0 and check["steals"] == 0.0 and check["points"] == 1.0
    assert check["threes attempted"] == 0.0 and np.isinf(check["threes made"])
    assert check["blocks"] == 0.0 and check["assists"] == 0.0              # 3 of 3 blocks, 4 of 4 assists
    assert (check["Russell share of blocks"], check["stolen share of turnovers"]) == (1 / 3, 1 / 4)
    assert check["offensive share of fouls"] == 0.1


def test_data_careers_reads_its_pacing_and_force():
    """`openrapm data careers` (experiment 57): one request at a time, --delay seconds apart."""
    from openrapm.commands import data
    args = _parse("data", "careers")
    assert (args.delay, args.force, args.run) == (0.6, False, data.run_careers)
    assert _parse("data", "careers", "--delay", "1.5", "--force").delay == 1.5


@pytest.mark.parametrize("command", ["ingest", "stints"])
def test_data_ingest_and_stints_read_seasons_and_phases_as_the_old_scripts_did(command):
    """`openrapm data ingest 1997 2026 RS,PO` is what scripts/01_ingest.py 1997 2026 RS,PO was (02 for stints)."""
    from openrapm.commands import data
    args = _parse("data", command, "1997", "2026", "RS,PO")
    cfg = {"first_season": 1997}
    assert list(data.season_range(args, cfg)) == list(range(1997, 2027)) and data.phases(args) == ("RS", "PO")
    alone = _parse("data", command, "2026")
    assert list(data.season_range(alone, cfg)) == [2026] and data.phases(alone) == ("RS", "PO")
    assert list(data.season_range(_parse("data", command), cfg)) == [1997]
    assert args.run is (data.run_ingest if command == "ingest" else data.run_stints)


def test_data_stints_force_and_ingest_pacing():
    assert _parse("data", "stints", "2026", "--force").force
    args = _parse("data", "ingest", "2026", "2026", "PO", "--workers", "4", "--delay", "1.5")
    assert (args.workers, args.delay, args.phases) == (4, 1.5, "PO")


def test_data_panel_checks_for_an_existing_panel_before_building(tmp_path, monkeypatch):
    """The build takes minutes; an existing panel is refused first, without --force, and nothing is built."""
    from openrapm.data import panel as panel_module
    stored = tmp_path / "season_panel.parquet"
    stored.write_bytes(b"kept")
    monkeypatch.setattr(panel_module, "build_panel", lambda *a, **k: pytest.fail("built before checking"))
    with pytest.raises(FileExistsError, match="--force"):
        panel_module.write_panel({"_root": str(tmp_path), "paths": {"panel": str(stored)}})
    assert stored.read_bytes() == b"kept"


def test_a_negative_fold_assignment_is_refused():
    with pytest.raises(SystemExit):
        cli.chosen_settings(_parse("rate", "--fold-assignment", "-1"))


def test_folds_needs_its_models_and_base_and_defaults_to_2017_2026():
    args = _parse("folds", "--models", "m", "--base", "b")
    assert (args.models, args.base, args.tag, args.reproduce_only) == ("m", "b", "within", False)
    assert args.seasons == list(range(2017, 2027))
    with pytest.raises(SystemExit):
        _parse("folds", "--models", "m")


def test_folds_split_the_games_part_into_pieces_only_when_asked():
    assert _parse("folds", "--models", "m", "--base", "b").pieces is False
    assert _parse("folds", "--models", "m", "--base", "b", "--pieces").pieces is True


def test_folds_read_whole_season_inputs_only_when_asked():
    assert _parse("folds", "--models", "m", "--base", "b").whole_season_inputs is False
    assert _parse("folds", "--models", "m", "--base", "b", "--whole-season-inputs").whole_season_inputs is True


def test_main_reads_the_config_given_before_the_command(monkeypatch, tmp_path):
    seen = {}

    def record(args, cfg):
        seen.update(command=args.command, root=cfg["_root"], first=cfg["first_season"], argv=args.argv)
        return 0

    config = tmp_path / "config.yaml"
    config.write_text("first_season: 2001\nlast_season: 2002\n", encoding="utf-8")
    monkeypatch.setattr(cli, "_run_compare", record)
    assert cli.main(["--config", str(config), "compare", "a.parquet", "b.parquet"]) == 0
    assert seen["command"] == "compare" and seen["first"] == 2001
    assert seen["argv"] == ["openrapm", "--config", str(config), "compare", "a.parquet", "b.parquet"]


def test_the_threads_are_pinned_before_numpy_loads():
    """In a fresh interpreter `python -m openrapm --help` leaves numpy unloaded and every thread variable at 1."""
    code = ("import os, sys; sys.argv = ['openrapm', '--help']\n"
            "import openrapm.commands.cli as c\n"
            "assert 'numpy' not in sys.modules, 'numpy loaded before the threads were pinned'\n"
            "print(','.join(os.environ[v] for v in c.THREAD_VARIABLES))")
    env = {k: v for k, v in os.environ.items() if k not in cli.THREAD_VARIABLES}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True)
    assert out.stdout.strip() == "1,1,1"


def test_python_dash_m_openrapm_answers_help():
    out = subprocess.run([sys.executable, "-m", "openrapm", "--help"], capture_output=True, text=True, check=True)
    assert all(command in out.stdout for command in ("data", "build", "rate", "shrink", "swap", "folds", "site",
                                                     "compare", "glossary"))


@pytest.mark.parametrize("base, step, name", [("season_ratings_trial_step3", "_step3", "trial"),
                                               ("season_ratings_trial_step3.parquet", "_step3", "trial"),
                                               ("season_ratings_trial_step4", "_step4", "trial"),
                                               ("other_table", "_step3", "other_table")])
def test_a_step_names_its_output_after_the_table_it_reads(base, step, name):
    assert cli.output_name(base, step) == name


def _recorded(monkeypatch, function: str, *argv):
    """Run `openrapm *argv` with commands.build.<function> replaced by a recorder; what it was handed."""
    from openrapm.commands import build
    seen = {}

    def record(cfg, settings, *args, **kwargs):
        seen.update(settings=settings, args=args, kwargs=kwargs)
        return 0

    monkeypatch.setattr(build, function, record)
    assert cli.main(list(argv)) == 0
    return seen


def test_build_defaults_to_the_product_setting_and_the_published_name(monkeypatch):
    seen = _recorded(monkeypatch, "run_build", "build")
    assert seen["settings"] == PRODUCT
    assert seen["args"][:2] == (None, "product") and seen["kwargs"]["with_types"] is False


def test_build_takes_the_test_setting_and_another_fold_folder(monkeypatch):
    seen = _recorded(monkeypatch, "run_build", "build", "--setting", "test", "--folds", "other_within",
                     "--seasons", "2026", "--types")
    assert seen["settings"].exclude_neighbours == 1 and seen["settings"].shrink.rule == "test"
    assert seen["settings"].shrink.folds_tag == "other_within" and seen["settings"].swap.exclude_near == 1
    assert seen["args"][:2] == ([2026], "test") and seen["kwargs"]["with_types"] is True


def test_shrink_reads_its_base_and_setting(monkeypatch):
    seen = _recorded(monkeypatch, "run_shrink", "shrink", "--base", "season_ratings_trial_step3", "--setting",
                     "test")
    assert seen["settings"].shrink == TEST.shrink
    assert seen["args"] == ("season_ratings_trial_step3", "trial")
    seen = _recorded(monkeypatch, "run_shrink", "shrink", "--base", "x", "--folds", "f", "--name", "y")
    assert seen["settings"].shrink.rule == "product" and seen["settings"].shrink.folds_tag == "f"
    assert seen["args"] == ("x", "y")
    with pytest.raises(SystemExit):
        _parse("shrink")


def test_swap_reads_its_base_and_setting(monkeypatch):
    seen = _recorded(monkeypatch, "run_swap", "swap", "--base", "season_ratings_trial_step4.parquet", "--setting",
                     "test")
    assert seen["settings"].swap == TEST.swap
    assert seen["args"] == ("season_ratings_trial_step4", "trial") and seen["kwargs"]["with_types"] is False


def test_site_writes_ratings_json_alone_by_default(monkeypatch):
    from openrapm.commands import site
    seen = {}
    monkeypatch.setattr(site, "run_site", lambda cfg, **kwargs: seen.update(kwargs) or 0)
    assert cli.main(["site"]) == 0
    assert seen == dict(ratings=None, out_dir=None, built=None, bias=False, decomposition=False, bias_tables=None,
                        similarity=False, openrapm_decomposition=False)
    assert cli.main(["site", "--ratings", "r.parquet", "--out-dir", "out", "--built", "2026-10-09", "--bias",
                     "--bias-tables", "tables", "--decomposition", "--similarity", "--openrapm-decomposition"]) == 0
    assert seen == dict(ratings="r.parquet", out_dir="out", built="2026-10-09", bias=True, decomposition=True,
                        bias_tables="tables", similarity=True, openrapm_decomposition=True)


@pytest.mark.parametrize("text", ["2026-13-01", "09/10/2026", "yesterday"])
def test_site_refuses_a_date_it_cannot_read(text):
    with pytest.raises(SystemExit):
        _parse("site", "--built", text)
