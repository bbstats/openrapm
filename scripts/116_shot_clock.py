"""The rebuilt shot clock against the 2014-15 truth, and the shot-clock violations of every season.

    python scripts/116_shot_clock.py [--first=1997] [--last=2026] [--oreb_14_from=2019] [--oreb_14_mode=set]

1. Joins the 2014-15 shot log to the shot frame by shot order (tracking.join_shotlog) and reports the join.
2. Rebuilds the clock (shotclock.rebuild), fits the logging lags on the half-A games and scores the half-B
   games, and the reverse, so no score is in-sample; then fits the lags on all of 2014-15 and writes them to
   data/shotq/clock_lags.json for the models.
3. Accuracy overall, by reset kind and by arena; the 24.0 resets the log records on tips and putbacks; the
   clock-off flag.
4. Every season: each shot-clock violation should rebuild to zero.  For 2019 on, the 14-second offensive-
   rebound reset against the old 24 ("--oreb_14_from=2100" turns it off).

The stage-3 gate of the shot-quality plan: >= 125,000 rows joined, median error <= 1.5 s, >= 75% within 3 s,
>= 80% of the 24.0 resets recovered, and the violations within 1 s of zero in >= 90% of rows in every season.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _cli import check_flags, flag  # noqa: E402
from eracoef.config import load_config  # noqa: E402
from eracoef.shotclock import ClockRules, accuracy, fit_lags, rebuild  # noqa: E402
from eracoef.shotframe import load_frame  # noqa: E402
from eracoef.tracking import join_shotlog, load_shotlog  # noqa: E402

pd.set_option("display.width", 220, "display.max_columns", 30, "display.precision", 3)


def main():
    check_flags()
    cfg = load_config()
    rules_kw = dict(oreb_14_from=int(flag("oreb_14_from", "2019")), oreb_14_mode=flag("oreb_14_mode", "set"))
    out_dir = Path(cfg["_root"]) / "outputs" / "shotclock"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. the join
    f = load_frame([2015], cfg, phases=("RS",))
    log = load_shotlog(cfg)
    j, rep = join_shotlog(log, f)
    print("join:", rep)
    print(f"  play-by-play clock minus tracking clock: median {j.clock_lag.median():+.2f} s, "
          f"middle half {j.clock_lag.quantile(.25):+.1f} to {j.clock_lag.quantile(.75):+.1f}")
    J = j.merge(f[["game_id", "action_number", "season", "half", "arena", "poss_start", "poss_start_clock", "events",
                   "dist"]], on=["game_id", "action_number"], how="left")
    truth = J["shot_clock"].to_numpy(float)

    # 2. lags fitted on one half, scored on the other
    base = rebuild(J, ClockRules(**rules_kw))
    rows = []
    for fit_half, score_half in (("A", "B"), ("B", "A")):
        m_fit = (J["half"] == fit_half).to_numpy()
        lags = fit_lags(base[m_fit], truth[m_fit])
        r = rebuild(J[~m_fit], ClockRules(**rules_kw, lag=lags))
        acc = accuracy(r["sc_eff"].to_numpy(), truth[~m_fit], r["clock_off"].to_numpy())
        rows.append(dict(fit=fit_half, scored=score_half, **acc))
    print("\nheld-out accuracy (lags fitted on the other half of the games):")
    print(pd.DataFrame(rows).to_string(index=False))
    lags = fit_lags(base, truth)
    print("\nlags, all of 2014-15:", {k: round(v, 2) for k, v in lags.items()})
    full = rebuild(J, ClockRules(**rules_kw, lag=lags))
    J = pd.concat([J, full], axis=1)
    J["err"] = J["sc_eff"] - J["shot_clock"]

    # 3. by reset kind, by arena, the 24.0 resets, the off flag
    on = J["shot_clock"].notna()
    by_kind = J[on].groupby("reset_kind")["err"].agg(n="size", bias="median",
                                                     mae=lambda e: float(np.abs(e).mean()),
                                                     within3=lambda e: float((np.abs(e) <= 3).mean()))
    print("\nby the kind of the last reset (in-sample lags):")
    print(by_kind.to_string())
    by_arena = J[on].groupby("arena")["err"].agg(bias="median", mae=lambda e: float(np.abs(e).mean()), n="size")
    print(f"\nby arena: bias sd {by_arena.bias.std():.2f} s (range {by_arena.bias.min():+.2f} to {by_arena.bias.max():+.2f}),"
          f" mean absolute error {by_arena.mae.min():.2f} to {by_arena.mae.max():.2f}")
    r24 = J["shot_clock"] == 24.0
    rec24 = float(((J["reset_kind"] == "oreb") & (J["since_reset"] <= 2.0))[r24].mean())
    print(f"24.0 resets: {int(r24.sum())} rows; rebuilt as an offensive-rebound reset within 2 s: {rec24:.3f}")
    off_truth = J["shot_clock"].isna()
    print(f"clock off in the log: {int(off_truth.sum())} rows; rebuilt off {J.loc[off_truth, 'clock_off'].mean():.3f};"
          f" rebuilt off where the log has a clock: {J.loc[~off_truth, 'clock_off'].mean():.4f}")
    overall = accuracy(J["sc_eff"].to_numpy(), truth, J["clock_off"].to_numpy())
    print("all of 2014-15, in-sample lags:", {k: round(v, 3) for k, v in overall.items()})

    p = Path(cfg["_root"]) / "data" / "shotq"
    p.mkdir(parents=True, exist_ok=True)
    (p / "clock_lags.json").write_text(json.dumps(dict(lags=lags, rules=rules_kw, fitted_on="2015 RS shot log",
                                                       rows=int(on.sum())), indent=1), encoding="utf-8")
    J.to_parquet(out_dir / "joined_2015.parquet", index=False)

    # 4. the violations, every season.  An offensive rebound logged within 1 s of the whistle is the feed's team
    # rebound after a shot that never reached the rim, not a reset, so it is dropped.  The rest are read as the
    # seconds from the last reset to the whistle against the value it was reset to: the median says whether the
    # era's rules and the lags are right, the spread around it how tight the rebuild is (the feed's clock is in
    # whole seconds and each kind of event is logged with its own delay).
    first, last = int(flag("first", "1997")), int(flag("last", "2026"))
    vrows = []
    for s in range(first, last + 1):
        try:
            a = load_frame([s], cfg, phases=("RS",), aux=True)
        except FileNotFoundError:
            continue
        v = a[(a["kind"] == "tov") & (a["sub"] == "Shot Clock Turnover")].copy()
        if not len(v):
            continue
        v["events"] = [";".join(tok for tok in str(e).split(";") if tok and not (
            tok.startswith("oreb") and abs(float(tok.rpartition("@")[2]) - c) <= 1.0)) for e, c in zip(v["events"].fillna(""), v["clock"])]
        variants = {"rules": ClockRules(**rules_kw, lag=lags)}
        if s >= 2019:
            variants["oreb_24"] = ClockRules(oreb_14_from=2100, lag=lags)
        for name, rr in variants.items():
            r = rebuild(v, rr)
            on = ~r["clock_off"].to_numpy()
            x = r["sc_raw"].to_numpy()[on]
            med = float(np.median(x))
            vrows.append(dict(season=s, variant=name, n=len(x), median=med,
                              within_1=float(np.mean(np.abs(x) <= 1.0)), within_2=float(np.mean(np.abs(x) <= 2.0)),
                              within_2_of_median=float(np.mean(np.abs(x - med) <= 2.0)),
                              after_oreb=float((r["reset_kind"] == "oreb").mean())))
    V = pd.DataFrame(vrows)
    print("\nshot-clock violations: the rebuilt clock at the whistle (should be 0)")
    print(V.to_string(index=False))
    V.to_csv(out_dir / "violations.csv", index=False)
    by_kind.to_csv(out_dir / "by_kind_2015.csv")
    by_arena.to_csv(out_dir / "by_arena_2015.csv")


if __name__ == "__main__":
    main()
