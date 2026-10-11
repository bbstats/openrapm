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


def test_each_rate_option_reaches_its_setting():
    args = _parse("rate", "--setting", "test", "--defensive-target", "x3def", "--features", "boruta_noonc",
                  "--booster-params", "other", "--fold-assignment", "1", "--seasons", "2025-2026",
                  "--name", "trial", "--save-models", "trial")
    chosen = cli.chosen_settings(args)
    assert chosen.exclude_neighbours == 1 and chosen.targets.defence == "x3def"
    assert chosen.targets.offence == TEST.targets.offence
    assert chosen.prior.booster_file == "params/booster_params_other.json"
    assert chosen.prior.fold_assignment == 1 and chosen.prior.feature_set == "boruta_noonc"
    assert (args.seasons, args.name, args.save_models) == ([2025, 2026], "trial", "trial")


def test_the_last_occurrence_of_a_flag_wins():
    """scripts/experiment_chain.sh puts the incumbent's flags first so an experiment's own can override them."""
    args = _parse("rate", "--features", "boruta_noonc", "--booster-params", "lgb1", "--booster-params", "other")
    assert args.booster_params == "other"


def test_a_negative_fold_assignment_is_refused():
    with pytest.raises(SystemExit):
        cli.chosen_settings(_parse("rate", "--fold-assignment", "-1"))


def test_folds_needs_its_models_and_base_and_defaults_to_2017_2026():
    args = _parse("folds", "--models", "m", "--base", "b")
    assert (args.models, args.base, args.tag, args.reproduce_only) == ("m", "b", "within", False)
    assert args.seasons == list(range(2017, 2027))
    with pytest.raises(SystemExit):
        _parse("folds", "--models", "m")


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
    assert "rate" in out.stdout and "folds" in out.stdout and "compare" in out.stdout
