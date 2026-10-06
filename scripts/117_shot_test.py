"""The direct shot test (eracoef.shottest) for any set of shot-quality arms.

    python scripts/117_shot_test.py [--first=1997] [--last=2026] [--arms=lp,flat] [--pred=name=dir,...]
                                    [--ref=lp] [--tests=makes,arena,half,dash] [--tag=shottest]

Arms: `lp` (the shipped distance curve, logged in the shot frame), `flat` (the season's league rate by shot
value), and any model's predictions under data/shotq/<dir>/<season>.parquet (game_id, action_number, q).
Regular season only (the halves are the regular season's A/B games).  Heaves (36 ft or more with 2 s or less
left) are left out of every test: they get a fixed rate in every arm.

    makes   log loss and Brier per season and shot value, each arm against --ref paired by season; the
            calibration slope per era block
    arena   the arena / offence / defence signal sd of each arm's expected points per 100 attempts, and the
            arena residual's correlation with it (the scorer-artefact check)
    half    other-half shooting for shooters (100+ attempts a half), team offence and team defence, twos and
            threes, the padding constant chosen on the other seasons
    dash    (2014-2017 only) how much contest the arm leaves: the slope of each player-game's shooting beyond the
            arm's expectation on its share of tightly guarded attempts, from the official dashboards

An arm named <x>_after (the owner's update, which has seen each shot's own result) is never scored on makes or
in the arena check, and in the other-half test it prices the predicting half only, with its twin <x> pricing
the half being predicted.

Writes outputs/shottest/<tag>_<test>.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _cli import check_flags, flag  # noqa: E402
from eracoef import shottest as st  # noqa: E402
from eracoef.config import load_config  # noqa: E402
from eracoef.shotframe import derive, load_frame  # noqa: E402

pd.set_option("display.width", 240, "display.max_columns", 40, "display.precision", 4)


def load(seasons, cfg, arms, preds) -> pd.DataFrame:
    parts = []
    for s in seasons:
        f = derive(load_frame([s], cfg, phases=("RS",)))
        f = f[~f["heave"]].copy()
        if "flat" in arms:
            f["flat"] = f.groupby("value")["made"].transform("mean")
        for name, d in preds.items():
            d, _, col = d.partition(":")
            p = pd.read_parquet(Path(cfg["_root"]) / "data" / "shotq" / d / f"{s}.parquet",
                                columns=["game_id", "action_number", col or "q"])
            f = f.merge(p.rename(columns={col or "q": name}), on=["game_id", "action_number"], how="left")
            miss = int(f[name].isna().sum())
            if miss:
                raise ValueError(f"{d} {s}: {miss} attempts without a prediction")
        keep = ["season", "game_id", "action_number", "half", "arena", "neutral", "team", "opp", "home", "shooter",
                "value", "made", "dist", "dist_xy", *arms]
        parts.append(f[keep])
    return pd.concat(parts, ignore_index=True)


def load_meta(seasons, cfg) -> pd.DataFrame:
    """game_date per attempt (the dashboards are per player and date)."""
    return pd.concat([load_frame([s], cfg, phases=("RS",))[["game_id", "action_number", "game_date"]] for s in seasons],
                     ignore_index=True)


def main():
    check_flags()
    cfg = load_config()
    first, last = int(flag("first", "1997")), int(flag("last", "2026"))
    preds = dict(p.split("=", 1) for p in flag("pred", "").split(",") if p)
    arms = [a for a in flag("arms", "lp,flat").split(",") if a] + list(preds)
    ref = flag("ref", arms[0])
    tests = flag("tests", "makes,arena,half").split(",")
    tag = flag("tag", "shottest")
    out = Path(cfg["_root"]) / "outputs" / "shottest"
    out.mkdir(parents=True, exist_ok=True)
    F = load(range(first, last + 1), cfg, arms, preds)
    print(f"{len(F):,} attempts, {F.season.nunique()} seasons, arms {arms}")

    before = [a for a in arms if not a.endswith("_after")]
    if "makes" in tests:
        S = st.season_scores(F, before)
        S.to_csv(out / f"{tag}_makes.csv", index=False)
        print("\n1. held-out makes: log loss, each arm minus", ref, "(paired by season; negative is better)")
        for v in (2, 3):
            for a in before:
                if a == ref:
                    continue
                p = st.paired(S[S.value == v], a, ref)
                print(f"   {v}s  {a:<16} {p['diff']:+.5f}  z {p['z']:+6.2f}  better in {p['won']}/{p['n']} seasons")
        C = st.calibration(F, before)
        C.to_csv(out / f"{tag}_calibration.csv", index=False)
        print("\n   calibration slope by era (1 is calibrated)")
        print(C.pivot_table(index=["era", "value"], columns="arm", values="slope").round(3).to_string())

    if "arena" in tests:
        A = pd.concat([st.arena_signal(F, a) for a in before], ignore_index=True)
        A.to_csv(out / f"{tag}_arena.csv", index=False)
        print("\n2. arena checks: signal sd of expected points per 100 attempts (arena / offence / defence), and the")
        print("   correlation of the arena's expected-points effect with its residual (near -0.9 is an artefact)")
        cols = ["xp_arena_sd", "xp_off_sd", "xp_def_sd", "res_arena_sd", "arena_xp_res_corr"]
        print(A.pivot_table(index="season", columns="arm", values=cols).round(2).to_string())

    if "teach" in tests:
        lp_ = Path(cfg["_root"]) / "data" / "shotq" / "teacher" / "labels.parquet"
        if lp_.exists():
            T = st.teacher_gap(F, before, pd.read_parquet(lp_))
            T.to_csv(out / f"{tag}_teach.csv", index=False)
            print()
            print("2(b). gap to the tracking teacher's out-of-fold quality (squared points), each arm minus", ref)
            for s in sorted(T.season.unique()):
                line = f"   {s}  {ref} {T[(T.arm == ref) & (T.season == s)].gap.mean():.2f}"
                for a in before:
                    if a == ref:
                        continue
                    p_ = st.paired_games(T[T.season == s], a, ref)
                    line += f" | {a} {p_['diff']:+.2f} z {p_['z']:+.1f}"
                print(line)

    if "dash" in tests:
        from eracoef.tracking import dashboard_counts
        parts = []
        for s in sorted(set(F.season) & {2014, 2015, 2016, 2017}):
            try:
                d = dashboard_counts(cfg, s, "def")
            except FileNotFoundError:
                continue
            d = d.reset_index()
            tight = d.get("0-2 Feet - Very Tight", 0) + d.get("2-4 Feet - Tight", 0)
            fga = d[[c for c in d.columns if "Feet" in str(c)]].sum(axis=1)
            parts.append(pd.DataFrame(dict(date=d["date"], PLAYER_ID=d["PLAYER_ID"], tight=tight, fga=fga)))
        if parts:
            D = pd.concat(parts, ignore_index=True)
            meta = load_meta(sorted(set(F.season) & {2014, 2015, 2016, 2017}), cfg)
            G = F.merge(meta, on=["game_id", "action_number"], how="left")
            C = st.contest_left(G, before, D)
            C.to_csv(out / f"{tag}_dash.csv", index=False)
            print()
            print("2(c). contest the arm leaves: slope of (makes - expected) per player-game on the share of tightly")
            print("     guarded attempts (official dashboards, STATS era); nearer zero is better")
            print(C.pivot_table(index="season", columns="arm", values="slope").round(4).to_string())

    if "half" in tests:
        rows = []
        for unit in ("shooter", "offence", "defence"):
            for v in (2, 3):
                for a in arms:
                    dest = a[: -len("_after")] if a.endswith("_after") else None
                    rows.append(st.other_half(F, a, unit=unit, value=v, dest_arm=dest,
                                              min_fga=100.0 if unit == "shooter" else 0.0))
        H = pd.concat(rows, ignore_index=True)
        H.to_csv(out / f"{tag}_half.csv", index=False)
        print("\n3. other-half shooting: held-out error (squared points of make rate), each arm minus", ref)
        for unit in ("shooter", "offence", "defence"):
            for v in (2, 3):
                h = H[(H.unit == unit) & (H.value == v)]
                base = h[h.arm == ref]
                line = f"   {unit:<8} {v}s  {ref} {base.err.mean():.3f}"
                for a in arms:
                    if a == ref:
                        continue
                    p = st.paired_seasons(h[h.arm == a], base)
                    line += f" | {a} {p['diff']:+.3f} z {p['z']:+.2f} ({p['won']}/{p['n']})"
                print(line)


if __name__ == "__main__":
    main()
