"""Add (or replace) one experiment's row in the experiment table, experiments/runs.csv.

    python scripts/101_log_run.py --name=shotmix --exp=34 --idea=owner --verdict=pending
            --changed="one sentence, plain words" [--notes="..."] [--date=2026-10-04]
            [--incumbent=priorshrink] [--log=outputs/<name>_chain.log] [--command="..."] [--dry]
            [--tag=<name>] [--system=<name>] [--compared=outputs/season_ratings_<incumbent>.parquet]
    python scripts/101_log_run.py --sheet      (print the table as rows for the Google Sheet)

`--tag` names the output files (yoy_<tag>.parquet), `--system` the candidate inside them, when an older run used
different names; `--compared` records what the "incumbent" in those files was at the time.  The date defaults to
the day the year-over-year file was written.

One row per experiment, MLflow-style: who asked, what changed, the headline test numbers, the verdict.  Every number
is read from what the standard chain (HANDOFF.md, "The test") already wrote -- nothing is typed in by hand:

  outputs/yoy_<name>.parquet            year-over-year (63): the error per team-game, z, seasons better, each
                                        direction, the order-only row (stint level, each side rescaled)
  outputs/yoy_by_player_<name>.parquet  the same split by quality tier (88): the top-30 z
  the chain log                         consensus agreement (64), the swap test's net order (90), the trade loss (73)

A number whose source is missing is left blank rather than guessed.  Rows are keyed by `exp`; logging the same
`exp` again replaces its row, so a corrected run does not leave a stale one behind.
"""
import re
import subprocess
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eracoef.holdout import paired, pooled  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / "experiments" / "runs.csv"

from _cli import check_flags, flag as _flag, switch   # noqa: E402

# the table's columns, in order; plain names because the owner reads this table directly
# American spelling and standard terms: this table is read outside the project (the owner, 2026-10-04)
COLUMNS = ["exp", "date", "name", "proposed_by", "change_made", "verdict",
           "error", "baseline_error", "change", "z", "seasons_better",
           "z_next_season", "z_previous_season", "z_order_only", "z_top30",
           "swap_order_change", "z_swap_order",
           "consensus", "consensus_baseline", "offense_spread", "offense_spread_baseline",
           "z_with_without_offense", "z_with_without_defense",
           "baseline", "notes", "command", "commit"]
# the Google Sheet's header row ("OpenRAPM experiments", tab Experiments), one per column above
HEADERS = ["Exp", "Date", "Name", "Proposed by", "Change", "Result",
           "Holdout error (pts/100 per team-game)", "Baseline holdout error", "Change in team-game MSE (neg = better)",
           "z (paired by season)", "Seasons improved", "z, predicting next season", "z, predicting prior season",
           "z, rank order only (scale-adjusted)", "z, top-30 players", "Lineup-swap test: rank-order change (pos = better)",
           "Lineup-swap test: z", "Spearman vs. consensus", "Baseline Spearman", "Offense SD ratio vs. consensus",
           "Baseline offense SD ratio", "With/without residual, offense: z (neg = better)",
           "With/without residual, defense: z (neg = better)", "Baseline ratings file", "Notes", "Command", "Commit"]


def game_armse(frame: pd.DataFrame, system: str) -> float:
    P = pooled(frame)
    row = P[(P.system == system) & (P.split == "all") & (P.group == "all")]
    return float(row.game_armse.iloc[0]) if len(row) else np.nan


def yoy(tag: str, name: str, ref: str) -> dict:
    path = ROOT / "outputs" / f"yoy_{tag}.parquet"
    if not path.exists():
        return {}
    res = pd.read_parquet(path)
    res["direction"] = res.system.str.rsplit(":", n=1).str[1]
    res["system"] = res.system.str.rsplit(":", n=1).str[0]
    res = res[(res.split == "all") & (res.group == "all")]
    both = res.copy()
    both["held_out"] = both.held_out + both.direction.map({"prev": 0.0, "next": 0.5})

    def z(frame, value="tg"):
        t = paired(frame, ref, value)
        t = t[t.system == name]
        return t.iloc[0] if len(t) else None

    out = {}
    t = z(both)
    if t is not None:
        out.update(change=t.mean_diff, z=t.z, seasons_better=f"{t.wins} of {t.n_seasons}")
    out["error"] = game_armse(both, name)
    out["baseline_error"] = game_armse(both, ref)
    # 63's "prev" rows are the PREVIOUS season's rating looking forward, i.e. predicting the next season
    for d, col in (("prev", "z_next_season"), ("next", "z_previous_season")):
        t = z(res[res.direction == d])
        out[col] = t.z if t is not None else np.nan
    t = z(both, "calib_side")
    out["z_order_only"] = t.z if t is not None else np.nan
    return out


def top30(tag: str, name: str) -> dict:
    path = ROOT / "outputs" / f"yoy_by_player_{tag}.parquet"
    if not path.exists():
        return {}
    d = pd.read_parquet(path)
    d = d[(d.candidate == name) & (d.split == "quality") & (d.group == "top 30")].tg
    return {"z_top30": float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d))))} if len(d) > 1 else {}


NUM = r"([+-]?\d+\.\d+)"


def from_log(log: Path, name: str, ref_file: str) -> dict:
    if not log.exists():
        return {}
    text = log.read_text(encoding="utf-8", errors="replace")
    out = {}
    # 64: "rankings n rho_off rho_def rho_total spread_off spread_def top5 ..."
    for who, suffix in ((f"season_ratings_{name}", ""), (Path(ref_file).stem, "_baseline")):
        m = re.search(rf"^\s*{re.escape(who)}\s+\d+\s+{NUM}\s+{NUM}\s+{NUM}\s+{NUM}", text, re.M)
        if m:
            out[f"consensus{suffix}"] = float(m.group(3))
            out[f"offense_spread{suffix}"] = float(m.group(4))
    # 90: the first "--- net, all pairs" block, the candidate's "order_vs_ref" cell
    blk = text.split("--- net, all pairs", 1)
    if len(blk) == 2:
        m = re.search(rf"^\s*{re.escape(name)}\s+{NUM}\s+{NUM}\s+{NUM}\s+([+-]\d+\.\d+) z ([+-]\d+\.\d+)",
                      blk[1], re.M)
        if m:
            out["swap_order_change"] = float(m.group(4))
            out["z_swap_order"] = float(m.group(5))
    # 73: the first "<name> against incumbent" block is tier 'all'; rows "side seasons players cand ref diff se z wins"
    blk = text.split(f"=== {name} against", 1)
    if len(blk) == 2:
        for side, col in (("offense", "z_with_without_offense"), ("defense", "z_with_without_defense")):
            m = re.search(rf"^\s*{side}\s+\d+\s+\d+\s+{NUM}\s+{NUM}\s+{NUM}\s+{NUM}\s+{NUM}", blk[1], re.M)
            if m:
                out[col] = float(m.group(5))
    return out


def sheet_rows() -> str:
    """The whole table as the JSON 2-D array the Sheets connector writes from A1: HEADERS, then one row per experiment.
    Experiment ids and commits get a leading apostrophe so Sheets keeps them as text."""
    import json
    t = pd.read_csv(TABLE, dtype={"exp": str, "commit": str})[COLUMNS]
    rows = [HEADERS]
    for _, r in t.iterrows():
        row = []
        for c in COLUMNS:
            v = r[c]
            if isinstance(v, float) and np.isnan(v):
                row.append("")
            elif isinstance(v, float):
                row.append(round(v, 3))
            elif c in ("exp", "commit"):
                row.append("'" + str(v))
            else:
                row.append(str(v))
        rows.append(row)
    return json.dumps(rows)


def main():
    check_flags()
    if switch("sheet"):
        print(sheet_rows())
        return
    name = _flag("name", None)
    exp = _flag("exp", None)
    if not name or not exp:
        raise SystemExit("--name and --exp are required")
    inc = _flag("incumbent", "priorshrink")
    ref_file = _flag("compared", f"outputs/season_ratings_{inc}.parquet")
    tag = _flag("tag", name)
    system = _flag("system", name)
    log = ROOT / _flag("log", f"outputs/{tag}_chain.log")
    yoy_path = ROOT / "outputs" / f"yoy_{tag}.parquet"
    when = (date.fromtimestamp(yoy_path.stat().st_mtime).isoformat() if yoy_path.exists()
            else date.today().isoformat())
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                                text=True).stdout.strip()
    except OSError:
        commit = ""
    row = dict(exp=str(exp), date=_flag("date", when), name=name,
               proposed_by=_flag("idea", ""), change_made=_flag("changed", ""),
               verdict=_flag("verdict", "pending").capitalize(), baseline=ref_file, notes=_flag("notes", ""), command=_flag("command", ""), commit=commit)
    row.update(yoy(tag, system, "incumbent"))
    row.update(top30(tag, system))
    row.update(from_log(log, system, ref_file))

    table = pd.read_csv(TABLE, dtype={"exp": str}) if TABLE.exists() else pd.DataFrame(columns=COLUMNS)
    table = table[table.exp != str(exp)]
    table = pd.concat([table, pd.DataFrame([row])], ignore_index=True)[COLUMNS]
    for c in table.columns:
        if table[c].dtype == float:
            table[c] = table[c].round(4)
    print(pd.Series(table.iloc[-1]).to_string())
    if switch("dry"):
        return
    TABLE.parent.mkdir(exist_ok=True)
    table.to_csv(TABLE, index=False, lineterminator="\n")
    print(f"\nwrote {TABLE} ({len(table)} rows)")


if __name__ == "__main__":
    main()
