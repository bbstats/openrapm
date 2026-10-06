"""Public SportVU tracking data, read and joined to the shot frame (the shot-quality build, stage 3).

Sources, each downloaded by hand into its own folder under data/tracking/ (the owner approved each, 2026-10-06):

    shotlog_openml/dataset_42806.pq   the 2014-15 shot log (OpenML 42806, an exact mirror of Kaggle's
                                      dansbecker/nba-shot-logs): 128,069 attempts, 2014-10-28 to 2015-03-04,
                                      about 280 selected shooters, with the closest defender and his distance,
                                      the shot clock, dribbles and touch time.  The STATS era, so its defender
                                      distance is never mixed with Second Spectrum's (the owner: "split eras").

The log has no play-by-play event number and its GAME_CLOCK runs about 2 s ahead of the play-by-play's, so a
clock join matches 2% of rows.  It joins by ORDER instead: within a game, a shooter's n-th attempt in the log is
his n-th field-goal attempt in the play-by-play (period, then clock, then action number), kept only where the
period, the shot value and the result all agree -- 98% of rows in the research's check.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

LOG_COLUMNS = {"SHOT_CLOCK": "shot_clock", "DRIBBLES": "dribbles", "TOUCH_TIME": "touch_time",
               "SHOT_DIST": "log_dist", "CLOSE_DEF_DIST": "def_dist", "CLOSEST_DEFENDER_PLAYER_ID": "def_id"}


def tracking_dir(cfg) -> Path:
    return Path(cfg["_root"]) / "data" / "tracking"


def _game_clock(s) -> float:
    m, _, sec = str(s).partition(":")
    try:
        return int(m) * 60 + float(sec)
    except ValueError:
        return np.nan


def load_shotlog(cfg) -> pd.DataFrame:
    """The 2014-15 shot log in the frame's vocabulary: game_id, shooter, period, order, value, made, the
    log's own game clock, and the tracking fields."""
    p = tracking_dir(cfg) / "shotlog_openml" / "dataset_42806.pq"
    if not p.exists():
        raise FileNotFoundError(f"{p} is missing: download https://data.openml.org/datasets/0004/42806/dataset_42806.pq")
    d = pd.read_parquet(p)
    out = pd.DataFrame({
        "game_id": "00" + d["GAME_ID"].astype("int64").astype(str).str.zfill(8),
        "shooter": d["player_id"].astype("int64"),
        "period": d["PERIOD"].astype(int),
        "order": d["SHOT_NUMBER"].astype(int),
        "value": d["PTS_TYPE"].astype(int),
        "made": d["FGM"].astype(int),
        "log_clock": d["GAME_CLOCK"].astype(str).map(_game_clock),
    })
    for src, dst in LOG_COLUMNS.items():
        out[dst] = pd.to_numeric(d[src], errors="coerce")
    out["def_id"] = out["def_id"].astype("Int64")
    return out


# ------------------------------------------------------------------------------------------ the dashboards
# The bins as the dashboards name them, with the edges our continuous values are cut at (left-closed).
DASH_BINS = {
    "def": ([0.0, 2.0, 4.0, 6.0, np.inf],
            ["0-2 Feet - Very Tight", "2-4 Feet - Tight", "4-6 Feet - Open", "6+ Feet - Wide Open"]),
    "dribble": ([0, 1, 2, 3, 7, np.inf], ["0 Dribbles", "1 Dribble", "2 Dribbles", "3-6 Dribbles", "7+ Dribbles"]),
    "touch": ([0.0, 2.0, 6.0, np.inf], ["Touch < 2 Seconds", "Touch 2-6 Seconds", "Touch 6+ Seconds"]),
    # counting down: (22, 24] is "24-22"; a clock that is off is its own bin
    "clock": ([-np.inf, 4.0, 7.0, 15.0, 18.0, 22.0, np.inf],
              ["4-0 Very Late", "7-4 Late", "15-7 Average", "18-15 Early", "22-18 Very Early", "24-22"]),
}


def dashboard_counts(cfg, season: int, family: str) -> pd.DataFrame:
    """Per player and date: FGA in each bin of one family, from data/tracking/dashboards (scripts/123)."""
    root = tracking_dir(cfg) / "dashboards" / str(int(season))
    paths = sorted(root.glob(f"*/{family}__*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no {family} dashboards for {season} under {root}; run scripts/123_dashboards.py")
    d = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    if not len(d):
        return pd.DataFrame()
    return d.pivot_table(index=["date", "PLAYER_ID"], columns="bin", values="FGA", aggfunc="sum", fill_value=0)


def to_bins(values, family: str) -> np.ndarray:
    """Our continuous values cut into the dashboard's bin names (the clock's NaN is 'ShotClock Off')."""
    edges, names = DASH_BINS[family]
    v = np.asarray(values, dtype=float)
    idx = np.searchsorted(np.asarray(edges[1:-1]), v, side="right" if family != "clock" else "left")
    out = np.asarray(names, dtype=object)[np.clip(idx, 0, len(names) - 1)]
    if family == "clock":
        out = np.where(np.isnan(v), "ShotClock Off", out)
    return out


def bin_agreement(ours: pd.DataFrame, dash: pd.DataFrame, totals: pd.Series | None = None) -> dict:
    """Agreement of two per (date, player) x bin count tables: the share of attempts that can be paired within
    the same bin.  With `totals` (the dashboards' attempts per player-date) it is read only on the player-dates
    where we found every attempt, so attempts we missed do not count against the bins."""
    a, d = ours.align(dash, join="inner", fill_value=0)
    cols = sorted(set(a.columns) | set(d.columns))
    a, d = a.reindex(columns=cols, fill_value=0), d.reindex(columns=cols, fill_value=0)
    out = dict(player_dates=len(a), attempts=int(d.values.sum()),
               agree_all=float(np.minimum(a.values, d.values).sum() / max(d.values.sum(), 1)))
    if totals is not None:
        full = (a.sum(axis=1) == totals.reindex(a.index)).to_numpy()
        out["complete_player_dates"] = int(full.sum())
        out["agree_complete"] = float(np.minimum(a.values[full], d.values[full]).sum() / max(d.values[full].sum(), 1))
    out["share_ours"] = (a.sum() / max(a.values.sum(), 1)).round(3).to_dict()
    out["share_dash"] = (d.sum() / max(d.values.sum(), 1)).round(3).to_dict()
    return out


def frame_order(frame: pd.DataFrame) -> pd.Series:
    """Each attempt's place among its shooter's attempts in that game: period, then clock (counting down),
    then action number."""
    f = frame.sort_values(["game_id", "shooter", "period", "clock", "action_number"],
                          ascending=[True, True, True, False, True])
    return f.groupby(["game_id", "shooter"]).cumcount().add(1).reindex(frame.index)


def join_shotlog(log: pd.DataFrame, frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """The log's rows attached to the frame's attempts by order; (joined, report).  `joined` is one row per
    agreeing attempt with the frame's game_id and action_number, so any frame column can be brought over."""
    f = frame.assign(order=frame_order(frame))
    keys = ["game_id", "shooter", "order"]
    m = log.merge(f[keys + ["action_number", "period", "value", "made", "clock"]], on=keys, how="left",
                  suffixes=("", "_pbp"), indicator=True)
    found = m["_merge"] == "both"
    agree = found & (m["period"] == m["period_pbp"]) & (m["value"] == m["value_pbp"]) & (m["made"] == m["made_pbp"])
    # the shooter-games whose attempt counts differ: the log is missing (or has an extra) attempt there
    nl = log.groupby(["game_id", "shooter"]).size()
    nf = f[f["game_id"].isin(set(log["game_id"]))].groupby(["game_id", "shooter"]).size()
    nl, nf = nl.align(nf, join="left")
    report = dict(log_rows=len(log), found=int(found.sum()), agree=int(agree.sum()),
                  share_agree=float(agree.mean()), games=int(log["game_id"].nunique()),
                  shooter_games=int(len(nl)), shooter_games_same_count=int((nl == nf).sum()))
    j = m[agree].drop(columns=["_merge", "period_pbp", "value_pbp", "made_pbp"])
    j["clock_lag"] = j["clock"] - j["log_clock"]         # play-by-play minus tracking: about -2 s
    return j.reset_index(drop=True), report
