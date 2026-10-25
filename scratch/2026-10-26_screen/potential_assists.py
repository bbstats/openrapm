"""Experiment 112: estimated potential assists from where the assists went, against the NBA's, 2014-2026.

    the regression   the NBA's potential assists on a player's assists by zone (rim, other twos, corner three,
                     above-the-break three; the play-by-play's passer, data/event_counts.py), no intercept, fitted on
                     the 2014-2017 regular seasons of one parity and read on the other, then on 2018-2026
    the math         each zone's assists over that season's make rate of passed shots there: assisted makes over
                     assisted makes plus the misses expected off a pass (data/creation_counts.py's cell shares and
                     miss factors), zones as data/event_counts.py draws them (rim: 6 ft or closer, or no location)
    flat             assists times one number, the baseline

Readings by season, mean over the seasons of each part, players with 100+ assists: the correlation with the NBA's
count, the correlation of potential assists per assist (what the estimate adds to the assists), and the error of
potential assists per assist after each season's level is matched (the tracking provider changed in 2018).
"""
import numpy as np
import pandas as pd

from openrapm.data.creation_counts import MISS_FACTOR, cell_shares, expit, logit

ZONES = ["assists_to_rim", "assists_to_mid_range", "assists_to_corner_threes", "assists_to_above_break_threes"]
SEASONS = range(2014, 2027)


def zone_of(shots: pd.DataFrame) -> np.ndarray:
    """data/event_counts.py's zones for each shot: a two within 6 ft or without a location, any other two, a corner
    three, any other three."""
    x, y = shots.x.to_numpy(float), shots.y.to_numpy(float)
    three = shots.value.to_numpy() == 3
    corner = three & (np.abs(x) >= 220) & (y <= 92.5)
    rim = ~three & ((np.hypot(x, y) <= 60) | ((x == 0) & (y == 0)))
    return np.select([rim, ~three, corner], ZONES[:3], ZONES[3])


def make_rates(season: int) -> pd.Series:
    """The season's make rate of passed shots by zone, regular season."""
    shots = pd.read_parquet(f"data/event_counts/{season}_creation_shots.parquet")
    shots = shots[shots.phase == "RS"].copy()
    share = cell_shares(shots)
    shots["p"] = np.where(shots.made == 1, shots.assisted,
                          expit(logit(shots.cell.map(share)) + shots.cell.map(MISS_FACTOR).fillna(0.0)))
    shots["zone"] = zone_of(shots)
    made = shots[shots.made == 1].groupby("zone").assisted.sum()
    passed = shots.groupby("zone").p.sum()
    return (made / passed).reindex(ZONES)


def table() -> pd.DataFrame:
    rows = []
    for season in SEASONS:
        e = pd.read_parquet(f"data/event_counts/{season}.parquet")
        e = e[e.phase == "RS"].groupby("player_id")[ZONES].sum()
        p = pd.read_parquet(f"data/raw/dashboards/tracking/pt_passing/{season}_RS.parquet")
        p = p.groupby("PLAYER_ID")[["AST", "POTENTIAL_AST"]].sum()
        m = e.join(p, how="inner")
        rate = make_rates(season)
        m["math"] = sum(m[z] / rate[z] for z in ZONES)
        m["season"] = season
        rows.append(m)
        print(f"  {season}: passed-shot make rates " + ", ".join(f"{z.removeprefix('assists_to_')} {rate[z]:.3f}"
                                                            for z in ZONES), flush=True)
    out = pd.concat(rows).reset_index(names="player_id")
    out["assists"] = out[ZONES].sum(axis=1)
    return out[out.assists >= 100]


def readings(d: pd.DataFrame, name: str, part: str) -> dict:
    per = []
    for _, g in d.groupby("season"):
        est, nba = g[name].to_numpy(float), g.POTENTIAL_AST.to_numpy(float)
        ratio_est, ratio_nba = est / g.assists, nba / g.assists
        level = ratio_nba.mean() / ratio_est.mean()
        per.append({"count_corr": np.corrcoef(est, nba)[0, 1], "ratio_corr": np.corrcoef(ratio_est, ratio_nba)[0, 1],
                    "ratio_error": np.sqrt(np.mean((ratio_est * level - ratio_nba) ** 2)),
                    "ratio_sd": ratio_nba.std()})
    per = pd.DataFrame(per)
    return {"estimate": name, "seasons": part, "players": len(d), **per.mean().round(4).to_dict()}


def main() -> None:
    d = table()
    early = d[d.season <= 2017]
    for parity in (0, 1):
        fit = early[early.season % 2 == parity]
        coef, *_ = np.linalg.lstsq(fit[ZONES].to_numpy(float), fit.POTENTIAL_AST.to_numpy(float), rcond=None)
        test = d.season.between(2014, 2017) & (d.season % 2 != parity)
        d.loc[test, "regression"] = d.loc[test, ZONES].to_numpy(float) @ coef
        print(f"fit on {'even' if parity == 0 else 'odd'} 2014-2017 seasons: " +
              ", ".join(f"{z.removeprefix('assists_to_')} {c:.2f}" for z, c in zip(ZONES, coef)))
    coef, *_ = np.linalg.lstsq(early[ZONES].to_numpy(float), early.POTENTIAL_AST.to_numpy(float), rcond=None)
    late = d.season >= 2018
    d.loc[late, "regression"] = d.loc[late, ZONES].to_numpy(float) @ coef
    print("fit on every 2014-2017 season: " + ", ".join(f"{z.removeprefix('assists_to_')} {c:.3f}"
                                                        for z, c in zip(ZONES, coef)))
    d["flat"] = d.assists
    rows = []
    for part, mask in (("2014-2017, held out by parity", d.season <= 2017), ("2018-2026", d.season >= 2018)):
        for name in ("flat", "regression", "math"):
            rows.append(readings(d[mask], name, part))
    print(pd.DataFrame(rows).to_string(index=False))
    pd.DataFrame(rows).to_csv("outputs/csv/potential_assists_check.csv", index=False)
    pd.Series(dict(zip(ZONES, coef))).to_json("outputs/csv/potential_assists_weights.json")


if __name__ == "__main__":
    main()
