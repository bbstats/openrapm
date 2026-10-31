"""scripts/checks/101_log_run.py reads three other checks' printed tables off the chain log: 64's consensus row,
90's net swap-order block and 73's with/without block.  A change to any of those formats would leave the experiment
table's columns blank without an error, so this runs 101 on a small log written in the formats the checks print
today and checks every column it should fill."""
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "checks" / "101_log_run.py"

# The lines as 64, 90 and 73 print them (outputs/noise_regroup2_chain.log, 2026-10-09), names changed.
CHAIN_LOG = """\
=== against data/external/consensus.csv, 2024-26, 1,000+ possessions
           rankings   n  rho_off  rho_def  rho_total  spread_off  spread_def  top5  def_share  team_r2_def  team_r2_off
season_ratings_cand 475    0.850    0.781      0.804       0.800       1.126     4      0.562        0.086        0.095
 season_ratings_inc 475    0.851    0.780      0.805       0.778       1.131     4      0.578        0.085        0.096
--- net, all pairs: 69,930 swap pairs and 968,658 possessions of information per observation, 56 observations
  ranking  order      gaps  slope             order_vs_ref              gaps_vs_ref
incumbent 1.8632 3806.1164 0.8361
     cand 1.8386 3806.3001 0.8301 -0.0246 z -1.2, 25 of 56 +0.1837 z +0.6, 30 of 56
=== cand against incumbent ===
  24,206 player-season-sides in both; dropped 0 from cand and 0 from incumbent
   side  seasons  players   cand    ref  mean_diff     se       z  wins
offense       30    12103 0.2942 0.2952    -0.0010 0.0005 -2.2090    21
defense       30    12103 0.2488 0.2487     0.0001 0.0005  0.1578    13
"""


def logged_row(tmp_path: Path) -> dict:
    """101's printed row for run "cand" against incumbent "inc", with every output read from `tmp_path`."""
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    cfg["paths"]["outputs"] = str(tmp_path)
    config = tmp_path / "config.yaml"
    with open(config, "w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle)
    with open(tmp_path / "cand_chain.log", "w", encoding="utf-8") as handle:
        handle.write(CHAIN_LOG)
    printed = subprocess.run([sys.executable, str(SCRIPT), "--name=cand", "--exp=p7test", "--incumbent=inc", "--dry"],
                             capture_output=True, text=True, encoding="utf-8", check=True, cwd=ROOT,
                             env={**os.environ, "OPENRAPM_CONFIG": str(config)}).stdout
    return {line.split()[0]: " ".join(line.split()[1:]) for line in printed.splitlines() if line.strip()}


def test_every_column_read_off_the_chain_log_is_filled(tmp_path):
    row = logged_row(tmp_path)
    assert row["consensus"] == "0.804" and row["offense_spread"] == "0.8"
    assert row["consensus_baseline"] == "0.805" and row["offense_spread_baseline"] == "0.778"
    assert row["swap_order_change"] == "-0.0246" and row["z_swap_order"] == "-1.2"
    assert row["z_with_without_offense"] == "-2.209" and row["z_with_without_defense"] == "0.1578"
    assert row["baseline"] == "outputs/season_ratings_inc.parquet"


def test_a_missing_year_over_year_file_leaves_its_columns_blank(tmp_path):
    row = logged_row(tmp_path)
    assert row["error"] == "NaN" and row["z"] == "NaN" and row["z_top30"] == "NaN"


def run_101(tmp_path: Path, log: str = CHAIN_LOG, seasons=range(1997, 2027), extra=()) -> subprocess.CompletedProcess:
    """101 on run "cand" against "inc", with a within-season file for `seasons`: cand's stint error 0.1 below inc's in
    even seasons, 0.05 above in odd ones."""
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    cfg["paths"]["outputs"] = str(tmp_path)
    config = tmp_path / "config.yaml"
    with open(config, "w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle)
    (tmp_path / "cand_chain.log").write_text(log, encoding="utf-8")
    rows = [f"{system},{season},step4,all,{100.0 + (0.0 if system == 'inc' else -0.1 if season % 2 == 0 else 0.05)}"
            for season in seasons for system in ("inc", "cand")]
    (tmp_path / "csv").mkdir(exist_ok=True)
    (tmp_path / "csv" / "wowy_cand_seasons.csv").write_text("system,season,stage,slice,error\n" + "\n".join(rows),
                                                             encoding="utf-8")
    return subprocess.run([sys.executable, str(SCRIPT), "--name=cand", "--exp=p7test", "--incumbent=inc", "--dry",
                           *extra], capture_output=True, text=True, encoding="utf-8", cwd=ROOT,
                          env={**os.environ, "OPENRAPM_CONFIG": str(config)})


def test_the_within_season_test_is_read_by_era_and_a_partial_or_stale_reading_stops(tmp_path):
    """2026-10-30: the table reads the test that decides, 151, and refuses a reading that cannot be this build's."""
    done = run_101(tmp_path)
    row = {line.split()[0]: " ".join(line.split()[1:]) for line in done.stdout.splitlines() if line.strip()}
    assert done.returncode == 0 and row["wowy_seasons_better"] == "15 of 30"
    assert float(row["wowy_change"]) == pytest.approx(-0.025)
    assert all(row[f"wowy_z_{era}"] != "NaN" for era in ("1997_2006", "2007_2016", "2017_2026"))
    partial = run_101(tmp_path, seasons=range(2017, 2027))
    assert partial.returncode != 0 and "10 of 30 seasons" in partial.stderr
    assert run_101(tmp_path, seasons=range(2017, 2027), extra=["--allow-partial"]).returncode == 0
    table = tmp_path / "season_ratings_cand.parquet"
    table.write_bytes(b"")
    os.utime(table, (time.time() + 60, time.time() + 60))
    stale = run_101(tmp_path)
    assert stale.returncode != 0 and "scores an earlier build" in stale.stderr


def test_the_log_is_read_from_its_last_run(tmp_path):
    """A resumed run (REUSE=1) appends to its log: the readings of the last "chain start" win."""
    later = CHAIN_LOG.replace("0.850    0.781      0.804", "0.860    0.781      0.814")
    log = "=== chain start: cand\n" + CHAIN_LOG + "=== chain start: cand\n" + later
    row = {line.split()[0]: " ".join(line.split()[1:]) for line in run_101(tmp_path, log=log).stdout.splitlines()
           if line.strip()}
    assert row["consensus"] == "0.814" and row["z_with_without_offense"] == "-2.209"
