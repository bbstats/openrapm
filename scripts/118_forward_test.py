"""The forward test: a team's first half of games predicts its second half's points per 100, no ratings in between.

    python scripts/118_forward_test.py [--first=1997] [--last=2026] [--arms=lp,flat] [--pred=name=dir,...]

Restored from the retired scratch/forward.py (git show 73f0ab9^:scratch/forward.py; FINDINGS 35, memory
"Luck adjustment: what works"), with the scoring moved to eracoef.shottest.loso_mse.  For every team-season
and side: build the profile from the FIRST half of its own games, predict each SECOND-half game's points per
100.  Each arm is scored by a leave-one-season-out regression, so a global rescale cannot win; only the
structure of an arm can.

    raw         the realised first-half rating
    adjusted    every component shrunk toward the league by its own constant (constants from other seasons)
    selective   only the OUTCOME rates shrunk (turnovers, offensive rebounds, free throws, the three make
                rates); the shot mix and the other style rates as they happened.  The shipped finding: -6.6% on
                offence, -10.4% on defence.  The stage-2 gate: raw 8.907 / 8.293, selective 8.316 / 7.433.
    sel_<arm>   selective, except the three make rates (rim, other twos, threes) are padded toward the team's
                OWN expected make rate from a shot model instead of the league's -- so a team that took (or
                allowed) easy shots keeps that, and only the making beyond it is shrunk.  The constant is read
                by the method of moments on the residual (made minus expected) of the other seasons' teams.

--arms: prediction columns already in the shot frame (`lp`, the shipped curve) or `flat` (the season's league
rate by shot value).  --pred=name=dir: a model's predictions, data/shotq/<dir>/<season>.parquet with game_id,
action_number and `q`.  Needs data/shotframe (scripts/114_shot_frame.py) for any sel_ arm.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _cli import check_flags, flag  # noqa: E402
from eracoef import teamloo as tl  # noqa: E402
from eracoef.config import load_config  # noqa: E402
from eracoef.shottest import loso_mse, zone_of  # noqa: E402

SIDES = {"off": "team_id", "def": "opp_id"}
ZONES = {"rim": "rim", "mid": "mid", "fg3": "thr"}          # teamloo's rate name -> the frame's zone


def halves(tg: pd.DataFrame, key: str) -> tuple[np.ndarray, np.ndarray]:
    r = tg.groupby(key)["rank"].rank(method="first").to_numpy()
    g = tg.groupby(key)["rank"].transform("count").to_numpy()
    first = r <= np.ceil(g / 2.0)
    return first, ~first


def pool_constants(kts: list, side: str) -> dict:
    K = pd.concat(kts, ignore_index=True)
    K = K[K.side == side]
    out = {}
    for c, g in K.groupby("component"):
        w = g["att"].to_numpy(float)
        tau2 = float(np.average(g["tau2_half"].to_numpy(float), weights=w))
        within = float(np.average(g["within"].to_numpy(float), weights=w))
        out[c] = within / tau2 if tau2 > 1e-9 else tl.K_CAP
    return out


def league_of(tg: pd.DataFrame) -> dict:
    return {c: float(tg[f"m_{c}"].sum() / max(tg[f"n_{c}"].sum(), 1.0)) for c in tl.MODEL_RATES}


def expected_by_game(frames: dict, arms: list) -> dict:
    """Per season: (game_id, team_id) -> attempts and expected makes per zone and arm, from the shot frame."""
    out = {}
    for s, f in frames.items():
        z = zone_of(f)
        d = f.assign(zone=z)
        parts = {}
        for arm in arms:
            g = d.groupby(["game_id", "team", "zone"]).agg(n=("made", "size"), x=(arm, "sum")).unstack("zone").fillna(0.0)
            g.columns = [f"q{a}_{b}" for a, b in g.columns]     # qn_rim, qx_rim, ...: never the team-game names
            parts[arm] = g
        out[s] = parts
    return out


def residual_k(n, m, x) -> tuple[float, float]:
    """(within, tau2) of a zone's make rate around the team's expected rate, by the method of moments."""
    n, m, x = (np.asarray(v, dtype=float) for v in (n, m, x))
    keep = n >= 50
    if keep.sum() < 10:
        return np.nan, np.nan
    n, m, x = n[keep], m[keep], x[keep]
    p, e = m / n, x / n
    r = p - e
    mu = np.average(r, weights=n)
    between = np.average((r - mu) ** 2, weights=n)
    within_var = np.average(p * (1.0 - p) / n, weights=n)
    return float(np.average(p * (1.0 - p), weights=n)), float(between - within_var)


def build(seasons, cfg, arms: list, frames: dict) -> pd.DataFrame:
    tgs = {s: tl.team_games(s, cfg) for s in seasons}
    kts = {s: tl.k_table(tgs[s], tl.loo_rates(tgs[s], rates=tl.MODEL_RATES), rates=tl.MODEL_RATES) for s in seasons}
    exp = expected_by_game(frames, arms) if arms else {}
    # the residual constants per arm, side and zone, from each season's full team totals
    rk = {}
    for arm in arms:
        for s in seasons:
            tg = tgs[s]
            e = exp[s][arm].reset_index().rename(columns={"team": "team_id"})
            j = tg.merge(e, on=["game_id", "team_id"], how="left")
            for side, key in SIDES.items():
                tot = j.groupby(key)[[f"m_{c}" for c in ZONES] + [f"n_{c}" for c in ZONES]
                                     + [f"qx_{z}" for z in ZONES.values()]].sum()
                for c, z in ZONES.items():
                    rk[(arm, s, side, c)] = residual_k(tot[f"n_{c}"], tot[f"m_{c}"], tot[f"qx_{z}"])
    rows = []
    for s in seasons:
        tg = tgs[s]
        lg = league_of(tg)
        for side, key in SIDES.items():
            k = pool_constants([kts[t] for t in seasons if t != s], side)
            first, second = halves(tg, key)
            cols = [f"{a}_{c}" for c in tl.MODEL_RATES for a in ("m", "n")]
            A = tg.loc[first, [key, *cols, "pts", "poss"]].groupby(key).sum()
            B = tg.loc[second, [key, *cols, "pts", "poss"]].groupby(key).sum()
            tot = {c: A[c].to_numpy() for c in A.columns}
            raw = tl.possession_points(tl.rates_from_totals(tot))
            adj = tl.possession_points(tl.rates_from_totals(tot, shrink_k=k, league=lg))
            ksel = {x: (k[x] if x in tl.OUTCOME_RATES else 0.0) for x in tl.MODEL_RATES}
            sel_rates = tl.rates_from_totals(tot, shrink_k=ksel, league=lg)
            sel = tl.possession_points(sel_rates)
            j = B.reindex(A.index)
            row = dict(season=s, side=side, team=A.index.to_numpy(), raw=raw["pts100"], adjusted=adj["pts100"],
                       selective=sel["pts100"],
                       truth=100.0 * j["pts"].to_numpy() / np.maximum(j["poss"].to_numpy(), 1.0),
                       w=j["poss"].to_numpy())
            for arm in arms:
                e = exp[s][arm].reset_index().rename(columns={"team": "team_id"})
                tj = tg.loc[first].merge(e, on=["game_id", "team_id"], how="left").fillna(0.0)
                X = tj.groupby(key)[[f"qx_{z}" for z in ZONES.values()] + [f"qn_{z}" for z in ZONES.values()]].sum().reindex(A.index)
                r = dict(sel_rates)
                for c, z in ZONES.items():
                    pairs = [rk[(arm, t, side, c)] for t in seasons if t != s]
                    within = np.nanmean([p[0] for p in pairs])
                    tau2 = np.nanmean([p[1] for p in pairs])
                    kq = within / tau2 if tau2 > 1e-9 else tl.K_CAP
                    n = A[f"n_{c}"].to_numpy(float)
                    m = A[f"m_{c}"].to_numpy(float)
                    nx = X[f"qn_{z}"].to_numpy(float)
                    target = np.where(nx > 0, X[f"qx_{z}"].to_numpy(float) / np.where(nx > 0, nx, 1.0), lg[c])
                    p = np.where(n > 0, m / np.where(n > 0, n, 1.0), 0.0)
                    r[c] = (n * p + kq * target) / (n + kq)
                row[f"sel_{arm}"] = tl.possession_points(r)["pts100"]
            rows.append(pd.DataFrame(row))
    return pd.concat(rows, ignore_index=True).dropna(subset=["truth"])


def load_frames(seasons, cfg, arms, preds: dict) -> dict:
    from eracoef.shotframe import load_frame
    out = {}
    for s in seasons:
        f = load_frame([s], cfg, phases=("RS",))
        if "flat" in arms:
            f["flat"] = f.groupby("value")["made"].transform("mean")
        for name, d in preds.items():
            d, _, col = d.partition(":")
            p = pd.read_parquet(Path(cfg["_root"]) / "data" / "shotq" / d / f"{s}.parquet", columns=["game_id", "action_number", col or "q"])
            f = f.merge(p.rename(columns={col or "q": name}), on=["game_id", "action_number"], how="left")
            if f[name].isna().any():
                raise ValueError(f"{d} {s}: {int(f[name].isna().sum())} shots without a prediction")
        out[s] = f
    return out


def main():
    check_flags()
    cfg = load_config()
    first = int(flag("first", cfg["first_season"]))
    last = int(flag("last", cfg["last_season"]))
    arms = [a for a in flag("arms", "").split(",") if a]
    preds = dict(p.split("=", 1) for p in flag("pred", "").split(",") if p)
    arms += list(preds)
    seasons = list(range(first, last + 1))
    pd.set_option("display.width", 220)
    frames = load_frames(seasons, cfg, arms, preds) if arms else {}
    D = build(seasons, cfg, arms, frames)
    print(f"\n=== forward test: first half predicts second half, {len(D)} team-seasons")
    print("    scored by leave-one-season-out regression, so a global rescale cannot win it\n")
    print(f"  {'side':<5} {'arm':<16} {'held-out MSE':>13} {'vs raw':>9} {'z':>7} {'seasons won':>12} {'vs selective':>13} {'z':>7}")
    out = []
    for side in ("off", "def"):
        S = D[D.side == side]
        mse_r, per_r = loso_mse(S, ["raw"])
        mse_s, per_s = loso_mse(S, ["selective"])
        for nm in ["raw", "adjusted", "selective"] + [f"sel_{a}" for a in arms]:
            ms, ps = loso_mse(S, [nm])
            d = ps - per_r
            z = float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))) if nm != "raw" else np.nan
            ds = ps - per_s
            zs = float(ds.mean() / (ds.std(ddof=1) / np.sqrt(len(ds)))) if nm not in ("raw", "selective") else np.nan
            print(f"  {side:<5} {nm:<16} {ms:13.4f} {ms - mse_r:+9.4f} {z:7.2f} {int((d < 0).sum()):>5}/{len(d):<6}"
                  f" {ms - mse_s:+13.4f} {zs:7.2f}")
            out.append(dict(side=side, arm=nm, mse=ms, vs_raw=ms - mse_r, z_raw=z, vs_sel=ms - mse_s, z_sel=zs,
                            won_raw=int((d < 0).sum()), won_sel=int((ds < 0).sum()), n=len(d)))
        # each arm with raw beside it in the same regression: does it carry anything once raw is in?  The
        # recorded headline (FINDINGS 35.3: 8.316 / 7.433) is the selective arm read this way.
        mse_sb, per_sb = loso_mse(S, ["raw", "selective"])
        for nm in ["selective"] + [f"sel_{a}" for a in arms]:
            ms, ps = loso_mse(S, ["raw", nm])
            d = ps - per_r
            z = float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d))))
            ds = ps - per_sb
            zs = float(ds.mean() / (ds.std(ddof=1) / np.sqrt(len(ds)))) if nm != "selective" else np.nan
            print(f"  {side:<5} {'raw + ' + nm:<16} {ms:13.4f} {ms - mse_r:+9.4f} {z:7.2f} {int((d < 0).sum()):>5}/{len(d):<6}"
                  f" {ms - mse_sb:+13.4f} {zs:7.2f}")
            out.append(dict(side=side, arm=f"raw+{nm}", mse=ms, vs_raw=ms - mse_r, z_raw=z, vs_sel=ms - mse_sb,
                            z_sel=zs, won_raw=int((d < 0).sum()), won_sel=int((ds < 0).sum()), n=len(d)))
        print()
    p = Path(cfg["_root"]) / "outputs" / "shottest"
    p.mkdir(parents=True, exist_ok=True)
    tag = flag("tag", "forward")
    pd.DataFrame(out).to_csv(p / f"{tag}.csv", index=False)
    D.to_parquet(p / f"{tag}_rows.parquet", index=False)
    print(f"wrote {p / f'{tag}.csv'}")


if __name__ == "__main__":
    main()
