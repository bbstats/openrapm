"""Add (or replace) one experiment's row in the experiment table, experiments/runs.csv.

One row per experiment: who asked, what changed, the headline test numbers, the verdict.  Every number is read
from what the standard chain (notes/HANDOFF.md, "The test"; scripts/experiment_chain.sh) already wrote, and nothing is
typed in by hand.  A number whose source is missing is left blank rather than guessed.  Rows are keyed by `exp`;
logging the same `exp` again replaces its row, so a corrected run does not leave a stale one behind.  The table is
mirrored to the "OpenRAPM experiments" Google Sheet, whose header row is HEADERS below; `--sheet` prints the whole
table in the form the Sheets connector writes.

Reads
    outputs/csv/wowy_<tag>_seasons.csv   the within-season WOWY test (151), the test that decides: each season's
                                         stint error at the shipped stage, the candidate against `--wowy-ref`
    outputs/yoy_<tag>.parquet            the year-over-year test (63): the error per team-game, the paired z and
                                         seasons better, each direction's z, and the order-only z (stint level,
                                         each side rescaled to the scored season)
    outputs/yoy_by_player_<tag>.parquet  the same split by quality tier (88): the top-30 z
    the chain log                        consensus agreement (64), the swap test's net order (90), the trade loss
                                         (73), read off the printed tables of the log's last run ("chain start")
Refuses (2026-10-30) a reading that cannot be this build's: a year-over-year file short of its 56 readings or a
within-season file short of its 30 seasons (a run that stopped part way; `--allow-partial` for one meant to be
partial), and either file older than the candidate's ratings table (an earlier build's, left by a reading that
failed).
Writes
    experiments/runs.csv (not with `--dry`, which prints the row and writes nothing)

How to read a row
    wowy_change, wowy_z     the within-season WOWY test: the mean over seasons of the candidate's stint mean squared
                            error minus the reference's (below zero = better), its paired t-statistic over seasons,
                            seasons better, and the paired t-statistic in each era
    error, baseline_error   the year-over-year error per team-game, points per 100, candidate and incumbent
    change, z               the paired mean difference in team-game mean squared error (below zero = better) and
                            its paired t-statistic; seasons_better counts the scored-season observations won
    z_next_season           63's "prev" rows: the PREVIOUS season's ratings predicting the next one
    z_previous_season       63's "next" rows: the next season's ratings predicting the previous one
Example: the noise control (the incumbent rebuilt with the box prior's players assigned to other folds) logged change
-0.0108, z -0.24, seasons_better "29 of 56": no difference, as it should be.  A real gain reads like the quarter
replacement level of experiment 11: -0.144, z -5.24, 42 of 56.

    python scripts/checks/101_log_run.py --name=shotmix --exp=34 --idea=owner --verdict=pending
            --changed="one sentence, plain words" [--notes="..."] [--date=2026-10-04]
            [--incumbent=mid136] [--wowy-ref=<system>] [--log=outputs/<name>_chain.log] [--command="..."]
            [--dry] [--allow-partial]
            [--tag=<name>] [--system=<name>] [--compared=outputs/season_ratings_<incumbent>.parquet]
    python scripts/checks/101_log_run.py --sheet      (print the table as rows for the Google Sheet)

`--tag` names the output files (yoy_<tag>.parquet), `--system` the candidate inside them, when an older run used
different names; `--compared` records what the "incumbent" in those files was at the time.  The date defaults to
the day the year-over-year file was written.
"""
import argparse
import json
import re
import subprocess
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from openrapm.commands.console import use_utf8
from openrapm.data.config import load_config, resolve, resolve_path
from openrapm.scoring import paired, pooled

# The table's columns, in order; plain names because the owner reads this table directly.  American spelling and
# standard terms: this table is read outside the project (the owner, 2026-10-04).
COLUMNS = ("exp", "date", "name", "proposed_by", "change_made", "verdict",
           "wowy_change", "wowy_z", "wowy_seasons_better", "wowy_z_1997_2006", "wowy_z_2007_2016", "wowy_z_2017_2026",
           "error", "baseline_error", "change", "z", "seasons_better",
           "z_next_season", "z_previous_season", "z_order_only", "z_top30",
           "swap_order_change", "z_swap_order",
           "consensus", "consensus_baseline", "offense_spread", "offense_spread_baseline",
           "z_with_without_offense", "z_with_without_defense",
           "baseline", "notes", "command", "commit")
# The Google Sheet's header row ("OpenRAPM experiments", tab Experiments), one per column above.
HEADERS = ("Exp", "Date", "Name", "Proposed by", "Change", "Result",
           "Within-season WOWY: change in stint MSE (neg = better)", "Within-season WOWY: z (paired by season)",
           "Within-season WOWY: seasons improved", "Within-season WOWY: z, 1997-2006",
           "Within-season WOWY: z, 2007-2016", "Within-season WOWY: z, 2017-2026",
           "Holdout error (pts/100 per team-game)", "Baseline holdout error",
           "Change in team-game MSE (neg = better)",
           "z (paired by season)", "Seasons improved", "z, predicting next season", "z, predicting prior season",
           "z, rank order only (scale-adjusted)", "z, top-30 players",
           "Lineup-swap test: rank-order change (pos = better)",
           "Lineup-swap test: z", "Spearman vs. consensus", "Baseline Spearman", "Offense SD ratio vs. consensus",
           "Baseline offense SD ratio", "With/without residual, offense: z (neg = better)",
           "With/without residual, defense: z (neg = better)", "Baseline ratings file", "Notes", "Command", "Commit")
TEXT_COLUMNS = ("exp", "commit")          # given a leading apostrophe in the sheet so Sheets keeps them as text
NUM = r"([+-]?\d+\.\d+)"                  # a printed decimal number
# 63's "prev" rows are the PREVIOUS season's rating looking forward, i.e. predicting the next season.
DIRECTION_COLUMNS = (("prev", "z_next_season"), ("next", "z_previous_season"))
TRADE_LOSS_COLUMNS = (("offense", "z_with_without_offense"), ("defense", "z_with_without_defense"))
ERAS = (("1997_2006", 1997, 2006), ("2007_2016", 2007, 2016), ("2017_2026", 2017, 2026))
YOY_READINGS = 56                         # 28 scored seasons, each predicted from the season before and the one after
WOWY_SEASONS = 30


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add or replace one experiment's row in experiments/runs.csv.",
                                     allow_abbrev=False)
    parser.add_argument("--sheet", action="store_true", help="print the table as rows for the Google Sheet")
    parser.add_argument("--name", help="the run's name (required unless --sheet)")
    parser.add_argument("--exp", help="the experiment number, the row's key (required unless --sheet)")
    # the incumbent since 2026-10-25 (experiment 110); weightfix (46) from 2026-10-09 to 2026-10-12
    parser.add_argument("--incumbent", default="mid136", help="default %(default)s")
    parser.add_argument("--wowy-ref", help="the within-season test's reference system (default --incumbent)")
    parser.add_argument("--allow-partial", action="store_true",
                        help="log a run with fewer readings than the full tests have, on purpose")
    parser.add_argument("--compared", help="the incumbent's table (default outputs/season_ratings_<incumbent>.parquet)")
    parser.add_argument("--tag", help="the output files' name (default --name)")
    parser.add_argument("--system", help="the candidate's name inside them (default --name)")
    parser.add_argument("--log", help="the chain log (default outputs/<tag>_chain.log)")
    parser.add_argument("--date", help="default the day the year-over-year file was written")
    parser.add_argument("--idea", default="", help="who proposed it")
    parser.add_argument("--changed", default="", help="one sentence, plain words")
    parser.add_argument("--verdict", default="pending", help="default %(default)s")
    parser.add_argument("--notes", default="")
    parser.add_argument("--command", default="")
    parser.add_argument("--dry", action="store_true", help="print the row and write nothing")
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------------- the year-over-year files
def game_armse(frame: pd.DataFrame, system: str) -> float:
    """`system`'s pooled year-over-year error per team-game, NaN when the file has no such system."""
    pooled_table = pooled(frame)
    row = pooled_table[(pooled_table.system == system) & (pooled_table.split == "all")
                       & (pooled_table.group == "all")]
    return float(row.game_armse.iloc[0]) if len(row) else np.nan


def paired_row(frame: pd.DataFrame, ref: str, name: str, value: str = "tg"):
    """`name`'s row of the paired comparison with `ref` on `value`, or None when either is missing."""
    table = paired(frame, ref, value)
    table = table[table.system == name]
    return table.iloc[0] if len(table) else None


def paired_z(frame: pd.DataFrame, ref: str, name: str, value: str = "tg") -> float:
    """The paired t-statistic of `name` against `ref`, NaN when either is missing."""
    row = paired_row(frame, ref, name, value)
    return row.z if row is not None else np.nan


def yoy(path: Path, name: str, ref: str, allow_partial: bool = False) -> dict:
    """The year-over-year columns of `name` against `ref` from a 63_yoy.py parquet; {} when it does not exist.  A
    file short of `YOY_READINGS` paired readings stops the run unless `allow_partial`: 63 rewrites it after every
    season, so a run that stopped part way leaves a partial one."""
    if not path.exists():
        return {}
    res = pd.read_parquet(path)
    res["direction"] = res.system.str.rsplit(":", n=1).str[1]
    res["system"] = res.system.str.rsplit(":", n=1).str[0]
    res = res[(res.split == "all") & (res.group == "all")]
    both = res.copy()
    both["held_out"] = both.held_out + both.direction.map({"prev": 0.0, "next": 0.5})
    out = {}
    row = paired_row(both, ref, name)
    if row is not None and int(row.n_seasons) < YOY_READINGS and not allow_partial:
        raise SystemExit(f"{path.name} pairs {name} with {ref} on {int(row.n_seasons)} of {YOY_READINGS} readings: "
                         f"a run that stopped part way.  Run 63 again, or --allow-partial")
    if row is not None:
        out.update(change=row.mean_diff, z=row.z, seasons_better=f"{row.wins} of {row.n_seasons}")
    out["error"] = game_armse(both, name)
    out["baseline_error"] = game_armse(both, ref)
    for direction, column in DIRECTION_COLUMNS:
        out[column] = paired_z(res[res.direction == direction], ref, name)
    out["z_order_only"] = paired_z(both, ref, name, "calib_side")
    return out


def paired_t(diffs: np.ndarray) -> float:
    """The mean of paired differences over its standard error; NaN with fewer than two."""
    diffs = np.asarray(diffs, dtype=float)
    return float(diffs.mean() / (diffs.std(ddof=1) / np.sqrt(len(diffs)))) if len(diffs) > 1 else np.nan


def wowy(path: Path, name: str, ref: str, allow_partial: bool = False) -> dict:
    """The within-season WOWY columns of `name` against `ref` from 151's `wowy_<tag>_seasons.csv`: each season's
    stint error at the shipped stage (step4, every stint), candidate minus reference, its mean, paired t-statistic,
    seasons better, and the t-statistic in each era.  {} when the file or either system is missing; fewer than
    `WOWY_SEASONS` paired seasons stops the run unless `allow_partial`."""
    if not path.exists():
        return {}
    d = pd.read_csv(path)
    d = d[(d.stage == "step4") & (d.slice == "all")].pivot_table(index="season", columns="system", values="error")
    if name not in d.columns or ref not in d.columns:
        return {}
    diff = (d[name] - d[ref]).dropna()
    if len(diff) < WOWY_SEASONS and not allow_partial:
        raise SystemExit(f"{path.name} pairs {name} with {ref} in {len(diff)} of {WOWY_SEASONS} seasons: a run that "
                         f"stopped part way.  Run 151 again, or --allow-partial")
    out = dict(wowy_change=float(diff.mean()), wowy_z=paired_t(diff.to_numpy()),
               wowy_seasons_better=f"{int((diff < 0).sum())} of {len(diff)}")
    for label, first, last in ERAS:
        out[f"wowy_z_{label}"] = paired_t(diff[(diff.index >= first) & (diff.index <= last)].to_numpy())
    return out


def refuse_older(reading: Path, table: Path) -> None:
    """A reading written before the candidate's ratings table scores an earlier build: a reading that failed leaves
    the last run's file in place.  Stop rather than log it."""
    if reading.exists() and table.exists() and reading.stat().st_mtime < table.stat().st_mtime:
        raise SystemExit(f"{reading} was written before {table}: it scores an earlier build.  Run its check again")


def top30(path: Path, name: str) -> dict:
    """The top-30 tier's z from an 88_yoy_by_player.py parquet: the mean of its team-game part over observations,
    over its standard error; {} when the file or the tier is missing."""
    if not path.exists():
        return {}
    d = pd.read_parquet(path)
    d = d[(d.candidate == name) & (d.split == "quality") & (d.group == "top 30")].tg
    return {"z_top30": float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d))))} if len(d) > 1 else {}


# ------------------------------------------------------------------------------------------- the chain log
def consensus_from_log(text: str, name: str, ref_file: str) -> dict:
    """64's rank agreement on total and offensive spread for the candidate and the baseline.

    64's rows read "rankings n rho_off rho_def rho_total spread_off spread_def top5 ...".
    """
    out = {}
    for who, suffix in ((f"season_ratings_{name}", ""), (Path(ref_file).stem, "_baseline")):
        m = re.search(rf"^\s*{re.escape(who)}\s+\d+\s+{NUM}\s+{NUM}\s+{NUM}\s+{NUM}", text, re.M)
        if m:
            out[f"consensus{suffix}"] = float(m.group(3))
            out[f"offense_spread{suffix}"] = float(m.group(4))
    return out


def swap_from_log(text: str, name: str) -> dict:
    """90's net order change and its z: the candidate's "order_vs_ref" cell in the first "--- net, all pairs"
    block."""
    block = text.split("--- net, all pairs", 1)
    if len(block) < 2:
        return {}
    m = re.search(rf"^\s*{re.escape(name)}\s+{NUM}\s+{NUM}\s+{NUM}\s+([+-]\d+\.\d+) z ([+-]\d+\.\d+)",
                  block[1], re.M)
    return dict(swap_order_change=float(m.group(4)), z_swap_order=float(m.group(5))) if m else {}


def trade_loss_from_log(text: str, name: str) -> dict:
    """73's with/without residual z per side, from the first "=== <name> against" block (the tier 'all'), whose rows
    read "side seasons players cand ref diff se z wins"."""
    block = text.split(f"=== {name} against", 1)
    if len(block) < 2:
        return {}
    out = {}
    for side, column in TRADE_LOSS_COLUMNS:
        m = re.search(rf"^\s*{side}\s+\d+\s+\d+\s+{NUM}\s+{NUM}\s+{NUM}\s+{NUM}\s+{NUM}", block[1], re.M)
        if m:
            out[column] = float(m.group(5))
    return out


RUN_START = "chain start:"               # the line every run of scripts/experiment_chain.sh opens its log with


def from_log(log: Path, name: str, ref_file: str) -> dict:
    """Every column read off the chain log, from its last run on (`RUN_START`; the whole log when it has none, as
    before 2026-10-30); {} when the log does not exist."""
    if not log.exists():
        return {}
    text = log.read_text(encoding="utf-8", errors="replace")
    if RUN_START in text:
        text = text[text.rindex(RUN_START):]
    return {**consensus_from_log(text, name, ref_file), **swap_from_log(text, name), **trade_loss_from_log(text, name)}


# ------------------------------------------------------------------------------------------------ the table
def sheet_cell(column: str, value):
    """One cell as the sheet takes it: blank for missing, numbers to three places, ids and commits as text."""
    if isinstance(value, float) and np.isnan(value):
        return ""
    if isinstance(value, float):
        return round(value, 3)
    if column in TEXT_COLUMNS:
        return "'" + str(value)
    return str(value)


def sheet_rows(table_path: Path) -> str:
    """The whole table as the JSON 2-D array the Sheets connector writes from A1: HEADERS, then one row per
    experiment."""
    table = pd.read_csv(table_path, dtype={"exp": str, "commit": str}).reindex(columns=list(COLUMNS))
    rows = [list(HEADERS)] + [[sheet_cell(c, r[c]) for c in COLUMNS] for _, r in table.iterrows()]
    return json.dumps(rows)


def current_commit(root: Path) -> str:
    """The checkout's short commit, recorded with the row; empty when git cannot be run."""
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root, capture_output=True,
                              text=True).stdout.strip()
    except OSError:
        return ""


def given(value, default):
    """A flag's value, or `default` when the flag was not given (an empty `--flag=` counts as given)."""
    return default if value is None else value


def new_row(args: argparse.Namespace, cfg: dict) -> dict:
    """The experiment's row: the flags as given, then every number read from the chain's outputs."""
    tag = given(args.tag, args.name)
    system = given(args.system, args.name)
    ref_file = given(args.compared, f"outputs/season_ratings_{args.incumbent}.parquet")
    outputs = resolve(cfg, "outputs")
    yoy_path = outputs / f"yoy_{tag}.parquet"
    when = (date.fromtimestamp(yoy_path.stat().st_mtime).isoformat() if yoy_path.exists()
            else date.today().isoformat())
    log = outputs / f"{tag}_chain.log" if args.log is None else resolve_path(cfg, args.log)
    row = dict(exp=str(args.exp), date=given(args.date, when), name=args.name, proposed_by=args.idea,
               change_made=args.changed, verdict=args.verdict.capitalize(), baseline=ref_file, notes=args.notes,
               command=args.command, commit=current_commit(Path(cfg["_root"])))
    wowy_path = outputs / "csv" / f"wowy_{tag}_seasons.csv"
    table = outputs / f"season_ratings_{system}.parquet"
    for reading in (yoy_path, wowy_path):
        refuse_older(reading, table)
    row.update(wowy(wowy_path, system, given(args.wowy_ref, args.incumbent), args.allow_partial))
    row.update(yoy(yoy_path, system, "incumbent", args.allow_partial))
    row.update(top30(outputs / f"yoy_by_player_{tag}.parquet", system))
    row.update(from_log(log, system, ref_file))
    return row


def with_row(table_path: Path, row: dict) -> pd.DataFrame:
    """The table with `row` in place of any earlier row of the same experiment, last; numbers to four places."""
    table = (pd.read_csv(table_path, dtype={"exp": str}) if table_path.exists()
             else pd.DataFrame(columns=list(COLUMNS)))
    table = table[table.exp != row["exp"]]
    table = pd.concat([table, pd.DataFrame([row])], ignore_index=True).reindex(columns=list(COLUMNS))
    for c in table.columns:
        if table[c].dtype == float:
            table[c] = table[c].round(4)
    return table


def main() -> None:
    use_utf8()
    args = parse_args()
    cfg = load_config()
    table_path = Path(cfg["_root"]) / "experiments" / "runs.csv"
    if args.sheet:
        print(sheet_rows(table_path))
        return
    if not args.name or not args.exp:
        raise SystemExit("--name and --exp are required")
    table = with_row(table_path, new_row(args, cfg))
    print(pd.Series(table.iloc[-1]).to_string())
    if args.dry:
        return
    table_path.parent.mkdir(exist_ok=True)
    table.to_csv(table_path, index=False, lineterminator="\n")
    print(f"\nwrote {table_path} ({len(table)} rows)")


if __name__ == "__main__":
    main()
