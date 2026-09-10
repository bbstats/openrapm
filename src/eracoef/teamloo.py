"""Team-game leave-one-out rates, and the luck-adjusted target built from them.

The owner's proposal, 2026-09-09: for every team-game, build the team's shooting rates from its OTHER
games, ask which of them predict this game's points per 100, and use the answer to strip luck out of
the ridge's target.  It is deliberately self-contained: one season of play-by-play is enough, which is
what "Open" needs.  Nothing here reads a second season.

Three things live in this module and they are separable:

1. **The rates.**  For team T and game j, the possession-weighted rate of each shooting component over
   T's other games.  `loo_rates`.
2. **The constant.**  How much of a game's OWN rate survives the blend, estimated by a method of
   moments on those rates.  `estimate_k`.  The blend itself is `pad.shrink`, like every other rate in
   this project (`pad.py`'s rule).
3. **The target.**  A stint's attempts valued at its game's adjusted team rate.  `team_design`, which
   registers into `fastfit.MspiFast` through `TARGETS` exactly as `xshoot.DEFENSE_TARGETS` does.

## Distributional bias, and what is actually done about it

Austin, Pe'er and Korem (*Distributional bias compromises leave-one-out cross-validation*, Science
Advances 2025) point out that leaving row j out shifts the remaining mean away from y_j, so a
leave-one-out feature carries a mechanical negative correlation with its own held-out label.  Their fix
("rebalanced LOO") drops one further row per fold, chosen so the remaining mean lands back on the full
mean.  This project has met the artifact before: `x3def` uses the other HALF of the block, never
leave-one-game-out, for exactly this reason.

Here the mechanical part is exact and worth writing down, because it decides where rebalancing is
needed.  With weights n_j and the team's own weighted mean pbar,

    p_loo(j) - pbar  ==  -n_j (p_j - pbar) / (S_n - n_j)                             (A)

identically.  So within a team the leave-one-out rate is an exactly negative multiple of the game's own
deviation.  Two consequences, in opposite directions:

* **For the matchup regression it is a real problem.**  Regressing a game's points on the team's LOO
  composite attenuates the coefficient, because (A) puts a negative within-team component into a
  regressor that is meant to carry only between-team signal.  This is the paper's case, and it is why
  `matchup_regression` uses the rebalanced rates.
* **For the moment estimate of tau2 it very nearly cancels.**  The between-team variance of a team's
  own mean is inflated by that mean's own sampling noise, and (A) subtracts an amount of the same
  order, so the plain-LOO covariance is close to unbiased for it.  So `estimate_k` computes tau2 FOUR
  ways -- rebalanced LOO, plain LOO, split-half (independent halves, no LOO mechanics) and the
  ordinary method of moments on team totals -- and reports all four rather than assuming.  **Measured
  over 1997-2026: rebalanced runs 9-23% above split-half on offense while plain and the method of
  moments agree with it to 4-9%.**  Rebalancing over-corrects here, so the constant is taken from the
  method of moments (`K_SOURCE`) and the rebalanced rates are used where the bias actually bites.

Rebalancing is defined on the LABEL (points scored for offense, points allowed for defense), which is
what the paper's rule specifies and what the regression needs.  `rebalance="rate"` rebalances on the
component's own rate instead and is slightly BETTER on the simulation, which bounds how much of (A) a
label-chosen partner removes.

**Two attempts to do better than the paper's heuristic, both measured and both rejected** (the composite
is points per possession, so an unattenuated coefficient is 100; the numbers are the mean absolute miss
across the three offensive components on `tests/test_teamloo.simulate_season`):

| | fg3 | fg2 | ft | mean miss |
|---|---|---|---|---|
| plain leave-one-out | 101.9 | 81.7 | 44.5 | 25.2 |
| the paper's rule, partner on the label | 107.9 | 96.7 | 67.4 | 14.6 |
| partner on the component's own rate | 107.4 | 97.6 | 69.2 | **13.5** |
| the exact within-team column (rejected: leaks) | 61.6 | 60.0 | 56.4 | 40.7 |
| split-half instrument (rejected: weak) | 79.8 | 124.5 | -280.1 | 141.6 |

The within-team column fails because identity (A) makes it a deterministic function of the game's own
rate, which drives the game's own points: the fit reaches R2 0.99 by regressing the outcome on itself.
The instrument (`split_loo_rates`, `matchup_regression(iv=True)`) is valid in principle -- two disjoint
halves of a team's other games have independent noise -- and fails in practice because 40 games is a weak
first stage, which is the classic weak-instrument blow-up.  So the paper's heuristic stands as the best
available correction, and `split_loo_rates` is kept because the split itself is reusable.

**None of this touches the target.**  The regression is a diagnostic; the target uses the rates and the
constant, so the coefficient's attenuation has no downstream effect on any rating.

## What the target actually does, stated plainly

The shrinkage constant comes out in the hundreds of attempts against 25-90 attempts in a game, so
`p_adj` sits close to the team's other-games rate: this is nearly full replacement of a game's shooting
by the team's, not a mild shrink.  `estimate_k` reports `shrink_mean`, the share of a game's own rate
that survives, so nobody has to infer it.  That matters because make-rate replacement has LOST here
before -- at shooter level (FINDINGS 18, +1.5 to +2.4 per 100) and at lineup level (FINDINGS 17) --
since shot-making is real signal, not only luck.  Free throws won (FINDINGS 15) because their constant
is about 24 attempts.  So the targets below are a pre-registered LADDER, gated rung by rung, and
`a` (the partial-adjustment scalar) exists so "how much of it" is a measured quantity rather than 1.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import pad
from .config import resolve

CACHE_VERSION = 1

# the three shooting components, their point value, and the counters each is built from
COMPONENTS = ("fg3", "fg2", "ft")
VALUE = {"fg3": 3.0, "fg2": 2.0, "ft": 1.0}

# EVERY rate a possession's points depend on, as (made, attempts) so ONE estimator covers all of them:
# a make rate, a turnover rate and a share of attempts are all proportions, and the same method of
# moments reads each.  The owner, 2026-09-09: *"we should be empirical rather than picking our %s"* --
# so nothing here is a chosen constant, every entry gets its own k on each side from the same code.
#   name        numerator        denominator      what it is
RATE_SPECS = {
    "fg3":       ("fg3m", "fg3a"),        # three-point percentage
    "rim":       ("fgm_rim", "fga_rim"),  # two-point percentage at the rim
    "mid":       ("fgm_mid", "fga_mid"),  # the LONG TWOS
    "fg2":       ("m_fg2", "n_fg2"),      # both twos together, kept for the existing target
    "ft":        ("ftm", "fta"),          # free-throw percentage
    "tov":       ("tov", "poss"),         # turnover rate per possession
    "oreb":      ("reb_cont", "reb_chance"),   # offensive rebound rate, per chance
    "rim_share": ("fga_rim", "fga"),      # the shot MIX: where the attempts went
    "mid_share": ("fga_mid", "fga"),
    "thr_share": ("fga_thr", "fga"),
    "ftr":       ("fta", "att"),          # free throws drawn per attempt
    "fga_rate":  ("fga", "att"),          # the share of attempts that are field goals (the rest are FT trips)
    "chance":    ("reb_chance", "att"),   # how often an attempt leaves a rebound to be had
}
# what a component is worth in points, where that is defined (a mix or a rate has no single value)
POINT_VALUE = {"fg3": 3.0, "rim": 2.0, "mid": 2.0, "fg2": 2.0, "ft": 1.0}

# the per-possession counters summed into a team-game row (stints.POSS_COUNTERS names)
TG_COUNTERS = ("pts", "pts_tech", "poss", "fga", "fgm", "fg3a", "fg3m", "fta", "ftm",
               "fta_tech", "ftm_tech", "tov", "att", "reb_cont", "reb_chance", "xftm", "xftm_tech",
               "fga_rim", "fgm_rim", "fga_mid", "fgm_mid", "fga_thr", "fgm_thr")

P_CLIP = (0.02, 0.98)          # a blended rate outside this is a bug, not a team
K_CAP = 1.0e6                  # tau2 <= 0: no between-team spread to preserve, so replace outright
CALIB_BAND = (0.995, 1.005)    # the season-total gate, as xshoot uses it

# Which estimate of the between-team variance sets the shrinkage constant.  Measured over 1997-2026
# (scripts/62_teamloo.py, attempt-weighted pool): the rebalanced-LOO covariance runs 9-23% ABOVE the
# split-half reference on offense, while plain LOO and the method of moments agree with it to 4-9%.
# The two bounds are understood -- rebalancing over-corrects the moment (its partner is chosen on the
# label, which is correlated with the team's own shooting), and the split-half reference is attenuated
# by real within-season change in a team -- and `mom` sits between them, needs no leave-one-out
# mechanics at all, and is the estimator `pad.py` already declares as this project's one convention.
# So the CONSTANT comes from the method of moments and the RATES stay rebalanced, which is the split
# the derivation asks for: rebalance where the paper's bias bites (the regression), not where it
# cancels (the moment).  Every candidate is reported side by side; `k_loo` etc. carry the others.
K_SOURCE = "mom"

def rates_in(tg) -> tuple:
    """The `RATE_SPECS` names this frame carries both sides of, in registry order.  A frame built
    by hand or by a simulation serves only the ones it defines, so nothing has to be stubbed."""
    return tuple(r for r in RATE_SPECS if f"n_{r}" in tg.columns and f"m_{r}" in tg.columns)


_TG_CACHE: dict = {}


# --------------------------------------------------------------------------- the team-game table
def _cache_path(season: int, cfg) -> Path:
    return Path(cfg["_root"]) / "data" / "cache" / "teamloo" / f"{season}_RS_v{CACHE_VERSION}.parquet"


def _stamp(season: int, cfg) -> tuple:
    """What a cached season must have been built from: the stints file and the game log."""
    out = []
    for path in (resolve(cfg, "stints") / f"{season}_RS.parquet",
                 resolve(cfg, "raw") / "gamelog" / f"{season}_RS.parquet"):
        info = path.stat() if path.exists() else None
        out.append((int(info.st_mtime_ns), int(info.st_size)) if info else None)
    return tuple(out)


def team_games(season: int, cfg, keep=None) -> pd.DataFrame:
    """One row per (regular-season game, team on offense), with that team's counters for the game.

    The stint counters are suffixed `_h` / `_a` by the team ON OFFENSE (stints.py:770), so a game
    contributes two rows: the home team's offensive possessions and the away team's.  `team_id` is the
    team those counters belong to and `opp_id` the team defending them.

    `keep`: the game_ids of this season the caller may see (`inseason.keep_games`).  A cut season is
    built fresh and never cached, the rule `xshoot.season_totals` follows -- the table then depends on
    the cut and not only on the season.
    """
    season = int(season)
    if keep is None:
        stamp = _stamp(season, cfg)
        hit = _TG_CACHE.get(season)
        if hit is not None and hit[0] == stamp:
            return hit[1]
        path = _cache_path(season, cfg)
        if path.exists():
            tg = pd.read_parquet(path)
            if tuple(tg.attrs.get("stamp", ())) == stamp or _stamp_of(tg) == stamp:
                _TG_CACHE[season] = (stamp, tg)
                return tg
    tg = _build_team_games(season, cfg, keep)
    if keep is None:
        stamp = _stamp(season, cfg)
        _TG_CACHE[season] = (stamp, tg)
        path = _cache_path(season, cfg)
        path.parent.mkdir(parents=True, exist_ok=True)
        tg.to_parquet(path, index=False)
    return tg


def _stamp_of(tg: pd.DataFrame) -> tuple:
    s = tg.attrs.get("stamp")
    return tuple(tuple(x) if x is not None else None for x in s) if s else ()


def _build_team_games(season: int, cfg, keep=None) -> pd.DataFrame:
    from .ingest import game_table, load_gamelog
    from .windows import load_stints

    st = load_stints(season, "RS", cfg)
    if keep is not None:
        st = st[st["game_id"].isin(set(map(str, keep)))]
    have = [c for c in TG_COUNTERS if f"{c}_h" in st.columns and f"{c}_a" in st.columns]
    missing = [c for c in ("pts", "poss", "fga", "fgm", "fg3a", "fg3m", "fta", "ftm") if c not in have]
    if missing:
        raise KeyError(f"stints {season} RS lack counters {missing}; rebuild the stints (stints.py)")
    frames = []
    gm = game_table(load_gamelog(season, "RS", cfg))[["game_id", "game_date", "home_team_id", "away_team_id", "neutral"]]
    for side, is_home in (("h", True), ("a", False)):
        g = st.groupby("game_id", sort=False)[[f"{c}_{side}" for c in have]].sum()
        g.columns = list(have)
        g = g.reset_index()
        g["is_home_off"] = is_home
        frames.append(g)
    tg = pd.concat(frames, ignore_index=True).merge(gm, on="game_id", how="left")
    if tg["home_team_id"].isna().any():
        n = int(tg["home_team_id"].isna().sum())
        raise KeyError(f"{n} team-game rows of {season} have no team id: the game log and the stints disagree")
    tg["season"] = season
    tg["team_id"] = np.where(tg["is_home_off"], tg["home_team_id"], tg["away_team_id"]).astype(np.int64)
    tg["opp_id"] = np.where(tg["is_home_off"], tg["away_team_id"], tg["home_team_id"]).astype(np.int64)
    tg["home"] = np.where(tg["neutral"].to_numpy(dtype=bool), 0.0,
                          np.where(tg["is_home_off"], 1.0, -1.0))
    tg = tg.drop(columns=["home_team_id", "away_team_id"])
    # the three components: attempts and makes.  `fta` / `ftm` are the non-technical free throws
    # (`fta_tech` / `ftm_tech` are held separately and are not a team's shooting in any useful sense).
    tg["n_fg2"], tg["m_fg2"] = tg["fga"] - tg["fg3a"], tg["fgm"] - tg["fg3m"]
    for name, (num, den) in RATE_SPECS.items():
        if num in tg.columns and den in tg.columns:
            tg[f"m_{name}"], tg[f"n_{name}"] = tg[num].to_numpy(float), tg[den].to_numpy(float)
    for c in rates_in(tg):
        n = tg[f"n_{c}"].to_numpy(dtype=float)
        tg[f"p_{c}"] = np.where(n > 0, tg[f"m_{c}"].to_numpy(dtype=float) / np.where(n > 0, n, 1.0), np.nan)
    poss = tg["poss"].to_numpy(dtype=float)
    tg["pts100"] = 100.0 * tg["pts"].to_numpy(dtype=float) / np.where(poss > 0, poss, 1.0)
    tg["tov100"] = 100.0 * tg["tov"].to_numpy(dtype=float) / np.where(poss > 0, poss, 1.0) if "tov" in tg else np.nan
    tg = tg.sort_values(["game_date", "game_id", "is_home_off"]).reset_index(drop=True)
    tg["rank"] = np.arange(len(tg))
    tg.attrs["stamp"] = _stamp(season, cfg) if keep is None else None
    return tg


# --------------------------------------------------------------------------- rebalanced leave-one-out
def rebalance_partners(label: np.ndarray, w: np.ndarray) -> np.ndarray:
    """For each row j, the other row whose removal alongside j puts the remaining weighted mean of
    `label` closest to the full weighted mean WITHOUT crossing it; -1 when no partner improves on
    leaving j out alone.  This is the rule of Austin, Pe'er and Korem (2025) for a continuous label.

    G is at most 82 here, so the G x G table is built outright.
    """
    label = np.asarray(label, dtype=float)
    w = np.asarray(w, dtype=float)
    G = label.size
    if G < 3:
        return np.full(G, -1, dtype=np.int64)
    S, W = float((w * label).sum()), float(w.sum())
    M = S / W
    s_j, w_j = w * label, w
    d1 = (S - s_j) / np.maximum(W - w_j, 1e-9) - M                       # what leaving j out alone does
    num = S - s_j[:, None] - s_j[None, :]
    den = W - w_j[:, None] - w_j[None, :]
    d2 = np.where(den > 1e-9, num / np.where(den > 1e-9, den, 1.0), np.nan) - M
    np.fill_diagonal(d2, np.nan)
    # keep only partners that move the mean back toward the full mean without overshooting it
    ok = np.isfinite(d2) & (np.sign(d2) * np.sign(d1)[:, None] >= 0) & (np.abs(d2) < np.abs(d1)[:, None])
    cost = np.where(ok, np.abs(d2), np.inf)
    partner = np.argmin(cost, axis=1).astype(np.int64)
    return np.where(np.isfinite(cost[np.arange(G), partner]), partner, -1)


def loo_rates(tg: pd.DataFrame, rebalance: str = "label", rates=None) -> pd.DataFrame:
    """Per row, each component's rate over the team's OTHER games, both as the team on offense
    (grouped by `team_id`) and as the team defending (grouped by `opp_id`).

    Columns per side in ("off", "def") and component c:
        `{side}_p_{c}`   the rebalanced-LOO rate      `{side}_pl_{c}`  the plain-LOO rate
        `{side}_n_{c}`   the attempts behind it       `{side}_partner` the dropped partner's row, or -1
    A team's own game contributes to neither.  With fewer than three games the LOO rate falls back to
    the team's pooled rate, and with no attempts at all to the season's league rate.

    `rebalance`: "label" (the paper's rule, on points scored / allowed), "rate" (on the component's own
    rate -- a diagnostic, one partner per component), or "none" (plain LOO in both columns).
    """
    if rebalance not in ("label", "rate", "none"):
        raise ValueError(f"rebalance must be label|rate|none, got {rebalance!r}")
    n_rows = len(tg)
    rates = tuple(rates) if rates is not None else rates_in(tg)
    out = {}
    league = {c: float(tg[f"m_{c}"].sum() / max(tg[f"n_{c}"].sum(), 1.0)) for c in rates}
    for side, key in (("off", "team_id"), ("def", "opp_id")):
        partner = np.full(n_rows, -1, dtype=np.int64)
        cols = {f"{side}_p_{c}": np.full(n_rows, np.nan) for c in rates}
        cols.update({f"{side}_pl_{c}": np.full(n_rows, np.nan) for c in rates})
        cols.update({f"{side}_n_{c}": np.zeros(n_rows) for c in rates})
        label = tg["pts100"].to_numpy(dtype=float)       # points scored (off) / allowed (def), same column
        poss = tg["poss"].to_numpy(dtype=float)
        for _, idx in tg.groupby(key, sort=False).indices.items():
            idx = np.asarray(idx, dtype=np.int64)
            pj = rebalance_partners(label[idx], poss[idx]) if rebalance == "label" \
                else np.full(idx.size, -1, dtype=np.int64)
            partner[idx] = np.where(pj >= 0, idx[np.maximum(pj, 0)], -1)
            for c in rates:
                n = tg[f"n_{c}"].to_numpy(dtype=float)[idx]
                m = tg[f"m_{c}"].to_numpy(dtype=float)[idx]
                Sn, Sm = n.sum(), m.sum()
                p_plain = _safe_div(Sm - m, Sn - n, Sm, Sn, league[c])
                if rebalance == "rate":
                    pj = rebalance_partners(np.where(n > 0, m / np.maximum(n, 1.0), league[c]), n)
                if rebalance == "none" or (pj < 0).all():
                    p_reb = p_plain
                else:
                    take = pj >= 0
                    mk = np.where(take, m[np.maximum(pj, 0)], 0.0)
                    nk = np.where(take, n[np.maximum(pj, 0)], 0.0)
                    p_reb = _safe_div(Sm - m - mk, Sn - n - nk, Sm, Sn, league[c])
                cols[f"{side}_p_{c}"][idx] = p_reb
                cols[f"{side}_pl_{c}"][idx] = p_plain
                cols[f"{side}_n_{c}"][idx] = n
        out[f"{side}_partner"] = partner
        out.update(cols)
    return pd.DataFrame(out, index=tg.index)


def _safe_div(num, den, tot_m, tot_n, league):
    """num/den, falling back to the team's pooled rate and then to the league rate."""
    pooled = float(tot_m / tot_n) if tot_n > 0 else league
    return np.where(den > 0, num / np.where(den > 0, den, 1.0), pooled)


# --------------------------------------------------------------------------- the shrinkage constant
def estimate_k(tg: pd.DataFrame, loo: pd.DataFrame, c: str, side: str) -> dict:
    """The between-team variance of a component's true rate, four ways, and the k that follows.

    `k = E_w[p(1-p)] / tau2` in ATTEMPT units, the convention of `pad.mom_k` (pad.py:56), so it feeds
    `pad.shrink` unchanged.  The four estimates of tau2:

      `tau2`        attempt-weighted covariance of a game's own rate with the REBALANCED-LOO rate
      `tau2_plain`  the same against the plain-LOO rate
      `tau2_half`   covariance of the team's two half-season rates, weighted by their harmonic mean
                    attempts -- the reference: the halves are independent, so this needs no correction
      `tau2_mom`    the ordinary method of moments on team-season totals (`pad.mom_k`) -- the one the
                    constant is taken from, see `K_SOURCE`

    `bias_exact` is the mechanical within-team term of identity (A), which is what a plain-LOO
    covariance carries and a rebalanced one carries less of; it is computed, not predicted from a
    formula, so the two columns can be read against each other honestly.
    `k_loo` / `k_plain` / `k_half` / `k_mom` are the constant each estimate implies; `k` is the one
    `K_SOURCE` selects.  `shrink_mean` is E_w[n_j/(n_j+k)], the share of a game's own rate that survives.
    """
    key = "team_id" if side == "off" else "opp_id"
    n = loo[f"{side}_n_{c}"].to_numpy(dtype=float)
    m = tg[f"m_{c}"].to_numpy(dtype=float)
    keep = n > 0
    n, m = n[keep], m[keep]
    p = m / n
    p_reb = loo[f"{side}_p_{c}"].to_numpy(dtype=float)[keep]
    p_pln = loo[f"{side}_pl_{c}"].to_numpy(dtype=float)[keep]
    W = n.sum()
    pbar = float((n * p).sum() / W)
    tau2 = float((n * (p - pbar) * (p_reb - (n * p_reb).sum() / W)).sum() / W)
    tau2_plain = float((n * (p - pbar) * (p_pln - (n * p_pln).sum() / W)).sum() / W)
    within = float((n * p * (1.0 - p)).sum() / W)
    # identity (A): the exact within-team component a plain-LOO covariance carries
    team = tg[key].to_numpy()[keep]
    df = pd.DataFrame(dict(team=team, n=n, m=m))
    g = df.groupby("team")[["n", "m"]].transform("sum")
    dev = p - g["m"].to_numpy() / g["n"].to_numpy()
    bias_exact = float(-(n * n * dev * dev / np.maximum(g["n"].to_numpy() - n, 1e-9)).sum() / W)
    # the reference: the two halves of the season are independent samples of the same team
    tau2_half, n_half = _half_cov(tg, c, key)
    # `pad.mom_k` returns its conservative K_FALLBACK when fewer than 20 units qualify, and 40 attempts
    # is a player-scale constant that would be nonsense for a team rate.  With fewer than 20 teams the
    # method of moments simply has no estimate, so say so and let `k` fall through to the LOO one.
    tot = tg.groupby(key)[[f"m_{c}", f"n_{c}"]].sum()
    qualified = int((tot[f"n_{c}"] >= 50.0).sum())
    if qualified >= 20:
        _, k_mom = pad.mom_k(tot[f"m_{c}"].to_numpy(dtype=float), tot[f"n_{c}"].to_numpy(dtype=float), min_att=50.0)
        tau2_mom = within / k_mom if k_mom > 0 else np.nan
    else:
        tau2_mom = np.nan
    out = dict(component=c, side=side, n_tg=int(keep.sum()), att=float(W), p=pbar, within=within,
               tau2=tau2, tau2_plain=tau2_plain, tau2_half=tau2_half, tau2_mom=float(tau2_mom),
               bias_exact=bias_exact, n_half=n_half)
    for src, t in (("loo", tau2), ("plain", tau2_plain), ("half", tau2_half), ("mom", tau2_mom)):
        out[f"k_{src}"] = (float(within / t) if t > 1e-9 else K_CAP) if np.isfinite(t) else np.nan
    out["k_source"] = K_SOURCE if np.isfinite(out[f"k_{K_SOURCE}"]) else "loo"
    out["k"] = out[f"k_{out['k_source']}"]
    out["at_floor"] = bool(out["k"] >= K_CAP)
    out["shrink_mean"] = float((n * (n / (n + out["k"]))).sum() / W)
    return out


def _half_cov(tg: pd.DataFrame, c: str, key: str) -> tuple[float, int]:
    """Covariance of a team's first-half and second-half rate, weighted by the harmonic mean of the
    two attempt counts.  Independent samples, so no leave-one-out correction is involved."""
    if "rank" not in tg.columns:
        return float("nan"), 0
    r = tg.groupby(key)["rank"].rank(method="first")
    g = tg.groupby(key)["rank"].transform("count")
    first = (r <= g / 2.0).to_numpy()
    d = pd.DataFrame({"team": tg[key].to_numpy(), "n": tg[f"n_{c}"].to_numpy(dtype=float),
                      "m": tg[f"m_{c}"].to_numpy(dtype=float), "first": first})
    a = d[d["first"]].groupby("team")[["n", "m"]].sum()
    b = d[~d["first"]].groupby("team")[["n", "m"]].sum()
    j = a.join(b, lsuffix="_a", rsuffix="_b", how="inner")
    j = j[(j["n_a"] > 0) & (j["n_b"] > 0)]
    if len(j) < 5:
        return float("nan"), int(len(j))
    pa, pb = j["m_a"] / j["n_a"], j["m_b"] / j["n_b"]
    w = 2.0 / (1.0 / j["n_a"] + 1.0 / j["n_b"])
    ma = float((w * pa).sum() / w.sum())
    mb = float((w * pb).sum() / w.sum())
    return float((w * (pa - ma) * (pb - mb)).sum() / w.sum()), int(len(j))


def k_table(tg: pd.DataFrame, loo: pd.DataFrame, rates=None) -> pd.DataFrame:
    """`estimate_k` for every rate the frame serves, on both sides."""
    rates = tuple(rates) if rates is not None else rates_in(tg)
    return pd.DataFrame([estimate_k(tg, loo, c, side) for side in ("off", "def") for c in rates])


def skill_table(kt: pd.DataFrame, per_game: float | None = None, games: float = 82.0) -> pd.DataFrame:
    """Turn the constants into the numbers the question is actually about: how much of what you SEE is
    real, on each side, at the sample size you are looking at.

    For a proportion measured over n attempts the sampling variance is p(1-p)/n, so with tau2 the true
    between-team variance the real share of the observed spread is

        real(n) = tau2 / (tau2 + p(1-p)/n)

    which is the same quantity `pad.shrink` applies row by row -- it is `n / (n + k)`.  Nothing is chosen:
    the answer moves with the sample, which is why "3P defense is 90% luck" is only true at one sample
    size.  `sd_pp` is the true spread in PERCENTAGE POINTS, the most legible form of tau2.

    `per_game` overrides the attempts per team-game implied by the table (`att / n_tg`).
    """
    rows = []
    for (side, c), g in kt.groupby(["side", "component"], sort=False):
        w = g["att"].to_numpy(float)
        tau2 = float(np.average(g["tau2_half"].to_numpy(float), weights=w))
        within = float(np.average(g["within"].to_numpy(float), weights=w))
        n_g = float(per_game if per_game is not None else np.average(g["att"] / g["n_tg"], weights=w))
        real = lambda n: tau2 / (tau2 + within / n) if tau2 > 0 and n > 0 else 0.0
        rows.append(dict(side=side, component=c, p=float(np.average(g["p"], weights=w)),
                         sd_pp=100.0 * np.sqrt(max(tau2, 0.0)), att_per_game=n_g,
                         k=within / tau2 if tau2 > 1e-9 else K_CAP,
                         real_game=real(n_g), real_10=real(10 * n_g), real_season=real(games * n_g)))
    D = pd.DataFrame(rows)
    off = D[D.side == "off"].set_index("component")["sd_pp"]
    D["def_vs_off"] = [np.nan if r["side"] == "off" or off.get(r["component"], 0) <= 0
                       else r["sd_pp"] / off[r["component"]] for _, r in D.iterrows()]
    return D


# ------------------------------------------------------------------- the possession model
# What a set of rates is worth in points per 100 possessions.  One function, used for BOTH the raw and
# the shrunk rates, so any error in the model itself is common to the two and cancels out of the paired
# comparison -- which is the only reason a structural model is safe to put in a test at all.
#
#   a possession is a turnover with probability `tov`, otherwise it produces at least one attempt
#   an attempt is a shot from one of the three zones in the proportions `*_share`, plus `ftr` free
#     throws drawn per attempt at `ft`
#   a missed attempt is rebounded by the offense with probability `oreb`, giving another attempt, so
#     the attempts per live possession are the geometric series 1 / (1 - miss * oreb)
MODEL_RATES = ("tov", "oreb", "chance", "fga_rate", "ft", "ftr",
               "fg3", "rim", "mid", "rim_share", "mid_share", "thr_share")

# Which of them are OUTCOMES, where a short sample is mostly luck, and which are STYLE, where it is
# mostly choice.  A team decides how many of its shots are threes and it does not decide whether they
# go in; a miss mechanically creates a rebound chance.  Shrinking a style rate toward the league throws
# away a real, stable team property, and `scratch/forward.py` measures exactly that -- shrinking
# `chance` or `fga_rate` costs 0.09 to 0.17 in forward MSE at z 3 to 4, on both sides.
OUTCOME_RATES = ("tov", "oreb", "ft", "ftr", "fg3", "rim", "mid")
STYLE_RATES = ("chance", "fga_rate", "rim_share", "mid_share", "thr_share")


def possession_points(r: dict) -> dict:
    """Points per 100 possessions implied by a dict of rates, with the pieces it went through.

    `r` needs `MODEL_RATES`.  The three shares are renormalised to sum to one, so a shrunk set that no
    longer adds up exactly is still a valid shot mix.
    """
    sh = np.array([r["rim_share"], r["mid_share"], r["thr_share"]], dtype=float)
    tot = sh.sum(axis=0) if sh.ndim > 1 else sh.sum()
    sh = sh / np.where(tot > 0, tot, 1.0)
    q = np.array([r["rim"], r["mid"], r["fg3"]], dtype=float)
    val = np.array([2.0, 2.0, 3.0])[:, None] if sh.ndim > 1 else np.array([2.0, 2.0, 3.0])
    # an ATTEMPT is a field goal or a free-throw trip (stints.py's definition), so the zone shares --
    # which are shares of FIELD GOALS -- have to be scaled by `fga_rate` before they can be mixed with
    # the free throws, and the rebound chance is per attempt rather than per miss
    fga_rate = np.asarray(r["fga_rate"], float)
    pts_att = fga_rate * (sh * q * val).sum(axis=0) + np.asarray(r["ftr"], float) * np.asarray(r["ft"], float)
    live = 1.0 / np.maximum(1.0 - np.asarray(r["chance"], float) * np.asarray(r["oreb"], float), 1e-6)
    att = (1.0 - np.asarray(r["tov"], float)) * live
    return dict(pts100=100.0 * att * pts_att, att_per_poss=att, pts_per_att=pts_att, live=live)


def rates_from_totals(tot: dict, shrink_k: dict | None = None, league: dict | None = None) -> dict:
    """The `MODEL_RATES` from summed counters, optionally shrunk toward `league` with `shrink_k`.

    `tot` holds `m_<rate>` and `n_<rate>` sums.  With `shrink_k` None the realised rates come back
    unchanged, which is the raw arm of any comparison.
    """
    out = {}
    for c in MODEL_RATES:
        n = np.asarray(tot[f"n_{c}"], dtype=float)
        m = np.asarray(tot[f"m_{c}"], dtype=float)
        p = np.where(n > 0, m / np.where(n > 0, n, 1.0), 0.0)
        if shrink_k is None:
            out[c] = p
        else:
            out[c] = pad.shrink(p, n, float(shrink_k[c]), float(league[c]))
    return out


# --------------------------------------------------------------------------- the adjusted rates
def adjusted_rates(tg: pd.DataFrame, loo: pd.DataFrame, ks: dict, prior: str = "off",
                   comps=COMPONENTS, k_fixed: float | None = None) -> pd.DataFrame:
    """Each row's blended make rate per component: `pad.shrink(p_game, n_game, k, p_prior)`.

    `prior`: "off" the offensive team's other-games rate; "def" the defending team's other-games rate
    allowed; "both" the additive matchup `off + def - league`, whose k uses the sum of the two
    between-team variances.  `k_fixed` overrides every k (0 reproduces the game's own rate exactly,
    which is the closure test).
    """
    if prior not in ("off", "def", "both"):
        raise ValueError(f"prior must be off|def|both, got {prior!r}")
    out = {}
    for c in comps:
        n = tg[f"n_{c}"].to_numpy(dtype=float)
        p = np.where(n > 0, tg[f"m_{c}"].to_numpy(dtype=float) / np.where(n > 0, n, 1.0), 0.0)
        lg = float(tg[f"m_{c}"].sum() / max(tg[f"n_{c}"].sum(), 1.0))
        if prior == "both":
            m_prior = loo[f"off_p_{c}"].to_numpy(dtype=float) + loo[f"def_p_{c}"].to_numpy(dtype=float) - lg
            tau = max(ks[("off", c)]["tau2"], 0.0) + max(ks[("def", c)]["tau2"], 0.0)
            k = ks[("off", c)]["within"] / tau if tau > 1e-9 else K_CAP
        else:
            m_prior = loo[f"{prior}_p_{c}"].to_numpy(dtype=float)
            k = ks[(prior, c)]["k"]
        m_prior = np.clip(m_prior, *P_CLIP)
        out[f"p_adj_{c}"] = np.clip(pad.shrink(p, n, float(k) if k_fixed is None else float(k_fixed), m_prior),
                                    *P_CLIP)
    return pd.DataFrame(out, index=tg.index)


# --------------------------------------------------------------------------- the matchup regression
def _composites(tg: pd.DataFrame, loo: pd.DataFrame, side: str) -> pd.DataFrame:
    """The owner's composites for one side: each component's LOO make rate x the team's LOO share of
    attempts x (1 - LOO turnover rate), i.e. its contribution to points per possession."""
    key = "team_id" if side == "off" else "opp_id"
    g = tg.groupby(key)
    tot = {c: g[f"n_{c}"].transform("sum").to_numpy(dtype=float) for c in COMPONENTS}
    own = {c: tg[f"n_{c}"].to_numpy(dtype=float) for c in COMPONENTS}
    poss_t = g["poss"].transform("sum").to_numpy(dtype=float)
    tov_t = g["tov"].transform("sum").to_numpy(dtype=float)
    poss_j = tg["poss"].to_numpy(dtype=float)
    tov_j = tg["tov"].to_numpy(dtype=float)
    keep = 1.0 - (tov_t - tov_j) / np.maximum(poss_t - poss_j, 1e-9)
    out = {}
    for c in COMPONENTS:
        share = (tot[c] - own[c]) / np.maximum(poss_t - poss_j, 1e-9)
        out[f"{side}_{c}"] = VALUE[c] * loo[f"{side}_p_{c}"].to_numpy(dtype=float) * share * keep
    out[f"{side}_tov"] = 100.0 * (tov_t - tov_j) / np.maximum(poss_t - poss_j, 1e-9)
    return pd.DataFrame(out, index=tg.index)


def split_loo_rates(tg: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Two leave-one-out rate sets per row, built from DISJOINT halves of the team's other games.

    For team T and game j, partition T's games by alternating chronological rank.  Set A is the rate over
    T's parity-0 games with j removed if j is one of them; set B the same for parity 1.  Both exclude j,
    and given the team's true rate their sampling noise is INDEPENDENT of each other -- which is what
    makes one a valid instrument for the other (`matchup_regression(iv=True)`).

    This is the same idea as `x3def`'s other-half-of-the-block rate, applied per team and per game.
    """
    out = []
    for parity in (0, 1):
        cols = {}
        league = {c: float(tg[f"m_{c}"].sum() / max(tg[f"n_{c}"].sum(), 1.0)) for c in COMPONENTS}
        for side, key in (("off", "team_id"), ("def", "opp_id")):
            for c in list(COMPONENTS) + ["tov", "poss"]:
                cols[f"{side}_{c}"] = np.full(len(tg), np.nan)
            for _, idx in tg.groupby(key, sort=False).indices.items():
                idx = np.asarray(idx, dtype=np.int64)
                order = np.argsort(tg["rank"].to_numpy()[idx], kind="stable")
                par = np.empty(idx.size, dtype=np.int64)
                par[order] = np.arange(idx.size) % 2
                mine = par == parity
                for c in COMPONENTS:
                    n = tg[f"n_{c}"].to_numpy(float)[idx]
                    m = tg[f"m_{c}"].to_numpy(float)[idx]
                    Sn, Sm = n[mine].sum(), m[mine].sum()
                    # remove j from the half only when j belongs to it
                    dn = np.where(mine, n, 0.0)
                    dm = np.where(mine, m, 0.0)
                    cols[f"{side}_{c}"][idx] = _safe_div(Sm - dm, Sn - dn, Sm, Sn, league[c])
                pos = tg["poss"].to_numpy(float)[idx]
                tov = tg["tov"].to_numpy(float)[idx]
                Sp, St = pos[mine].sum(), tov[mine].sum()
                dp = np.where(mine, pos, 0.0)
                dt = np.where(mine, tov, 0.0)
                cols[f"{side}_poss"][idx] = np.maximum(Sp - dp, 1e-9)
                cols[f"{side}_tov"][idx] = _safe_div(St - dt, Sp - dp, St, Sp, 0.14)
                for c in COMPONENTS:
                    n = tg[f"n_{c}"].to_numpy(float)[idx]
                    dn = np.where(mine, n, 0.0)
                    cols[f"{side}_share_{c}"] = cols.get(f"{side}_share_{c}", np.full(len(tg), np.nan))
                    cols[f"{side}_share_{c}"][idx] = (n[mine].sum() - dn) / np.maximum(Sp - dp, 1e-9)
        out.append(pd.DataFrame(cols, index=tg.index))
    return out[0], out[1]


def _split_composites(half: pd.DataFrame, side: str) -> pd.DataFrame:
    """The owner's composites built from ONE half of a team's other games (`split_loo_rates`)."""
    keep = 1.0 - half[f"{side}_tov"].to_numpy(float)
    d = {f"{side}_{c}": VALUE[c] * half[f"{side}_{c}"].to_numpy(float)
         * half[f"{side}_share_{c}"].to_numpy(float) * keep for c in COMPONENTS}
    d[f"{side}_tov"] = 100.0 * half[f"{side}_tov"].to_numpy(float)
    return pd.DataFrame(d, index=half.index)


def matchup_regression(tg: pd.DataFrame, loo: pd.DataFrame, design: str = "composite",
                       iv: bool = False) -> dict:
    """Points per 100 of a team-game on the two teams' leave-one-out rates and home.

    `design`: "composite" (the owner's three products per side plus turnovers), "raw" (the six rates
    plus turnovers per side) or "realised" (the row's OWN rates and mix -- an identity, R2 = 1, which
    is what the test checks).  Possession weighted; R2 is the weighted R2 of the fit.  `pred` is the
    fitted points per 100 of each team-game, the predictive rating the owner asked for.
    """
    y = tg["pts100"].to_numpy(dtype=float)
    w = tg["poss"].to_numpy(dtype=float)
    if design == "realised":
        n = {c: tg[f"n_{c}"].to_numpy(dtype=float) for c in COMPONENTS}
        poss = tg["poss"].to_numpy(dtype=float)
        X = pd.DataFrame({f"real_{c}": 100.0 * VALUE[c] * tg[f"m_{c}"].to_numpy(dtype=float) / poss
                          for c in COMPONENTS})
        X["real_tech"] = 100.0 * tg.get("ftm_tech", pd.Series(0.0, index=tg.index)).to_numpy(dtype=float) / poss
        del n
    elif design == "composite":
        X = pd.concat([_composites(tg, loo, "off"), _composites(tg, loo, "def")], axis=1)
        X["home"] = tg["home"].to_numpy(dtype=float)
    elif design == "raw":
        cols = {}
        for side in ("off", "def"):
            for c in COMPONENTS:
                cols[f"{side}_p_{c}"] = loo[f"{side}_p_{c}"].to_numpy(dtype=float)
        X = pd.DataFrame(cols, index=tg.index)
        X["home"] = tg["home"].to_numpy(dtype=float)
    else:
        raise ValueError(f"design must be composite|raw|realised, got {design!r}")
    Z = None
    if iv and design == "composite":
        # Errors in variables, not the paper's mean shift, is what attenuates this fit: a team's rate over
        # its other games is a NOISY measure of its true rate, and the noise pulls the coefficient toward
        # zero however the mean is balanced.  The fix uses the structure already here -- split the team's
        # other games in two, regress on one half and INSTRUMENT with the other.  Both exclude game j and
        # their noise is independent, so the instrument is valid and the attenuation cancels.
        A, B = split_loo_rates(tg)
        X = pd.concat([_split_composites(A, "off"), _split_composites(A, "def")], axis=1)
        X["home"] = tg["home"].to_numpy(dtype=float)
        Z = pd.concat([_split_composites(B, "off"), _split_composites(B, "def")], axis=1)
        Z["home"] = tg["home"].to_numpy(dtype=float)
    names = list(X.columns)
    A = np.column_stack([np.ones(len(tg)), X.to_numpy(dtype=float)])
    sw = np.sqrt(w)
    if Z is not None:
        # two-stage least squares, possession weighted: beta = (Z'X)^-1 Z'y with an intercept in both
        Zm = np.column_stack([np.ones(len(tg)), Z.to_numpy(dtype=float)]) * sw[:, None]
        Xm = A * sw[:, None]
        beta = np.linalg.lstsq(Zm.T @ Xm, Zm.T @ (y * sw), rcond=None)[0]
    else:
        beta, *_ = np.linalg.lstsq(A * sw[:, None], y * sw, rcond=None)
    pred = A @ beta
    ybar = float((w * y).sum() / w.sum())
    sse = float((w * (y - pred) ** 2).sum())
    sst = float((w * (y - ybar) ** 2).sum())
    return dict(design=design, iv=bool(iv), coef=dict(zip(["const"] + names, beta.tolist())),
                r2=1.0 - sse / sst if sst > 0 else float("nan"), rmse=float(np.sqrt(sse / w.sum())),
                pred=pred, n=len(tg))


# --------------------------------------------------------------------------- the season table
def season_table(season: int, cfg, keep=None, rebalance: str = "label"):
    """(tg, loo, k-frame) for one season; the unit everything above is assembled from."""
    tg = team_games(season, cfg, keep=keep)
    loo = loo_rates(tg, rebalance=rebalance)
    return tg, loo, k_table(tg, loo)


def adjusted_table(seasons, cfg, keep=None, prior: str = "off", comps=COMPONENTS,
                   k_fixed: float | None = None, rebalance: str = "label"):
    """The per-(season, game, side) adjusted rates for a block, with the per-season k frames.

    Each season is built ALONE -- its own rates, its own constants -- so a one-season run and a
    three-season run give the same numbers for the season they share.
    """
    parts, ktabs = [], []
    for s in seasons:
        s = int(s)
        tg, loo, kt = season_table(s, cfg, keep=None if keep is None else keep.get(s), rebalance=rebalance)
        ks = {(r["side"], r["component"]): r for r in kt.to_dict("records")}
        adj = adjusted_rates(tg, loo, ks, prior=prior, comps=comps, k_fixed=k_fixed)
        adj = adj.assign(season=s, game_id=tg["game_id"].to_numpy(), is_home_off=tg["is_home_off"].to_numpy(),
                         covered=True)
        parts.append(adj)
        ktabs.append(kt.assign(season=s))
    return pd.concat(parts, ignore_index=True), pd.concat(ktabs, ignore_index=True)


# --------------------------------------------------------------------------- the stint-level target
def team_design(seasons, cfg, wd_pts, comps=COMPONENTS, prior: str = "off", a: float = 1.0,
                ft: str = "team", calibrate: bool = True, keep=None, k_fixed: float | None = None,
                rebalance: str = "label"):
    """Actual points with each shooting component moved `a` of the way to its game's adjusted team rate.

    Per design row (a stint, one side on offense), with p_adj the team-game rate of THAT game:

        adj_c = a * (p_adj_c * attempts_c - makes_c)          for c in `comps`
        y     = 100 * (3*(fg3m + adj_fg3) + 2*(fg2m + adj_fg2) + ft_term + tech) / poss

    Attempts, attempt shares and turnovers stay exactly as they happened; only the make RATE moves.
    And-1s need no special case: the basket is in `fgm`, its free throw in `fta`/`ftm`.

    `ft`: "team" the free throws at the team's adjusted rate (and technicals at `xftm_tech`);
    "shooter" the shipped shooter-level `xftm` (what `xpts_ft` uses); "none" the free throws as they
    fell.  The free-throw term is NOT scaled by `a` -- it is the rung of the ladder that is already
    known to win, and `a` exists to measure the shot terms.

    Rows the team-game table does not cover (playoffs, since it is built from the regular season) keep
    their realised points; the report counts them.

    **The closure is a TEAM-GAME identity, not a row identity, and that is the whole point of reading
    this ladder carefully.**  At `k_fixed=0` the adjusted rate is the game's own team rate, so the
    team-game total returns actual points exactly (to 1e-13 on real data) while every STINT of that
    game changes: the game's makes have been spread evenly over its attempts.  So even before any
    shrinkage toward the other games, this target erases which LINEUP did the shooting within a game --
    the mechanism that lost at lineup level in FINDINGS 17.  The between-game shrink (`k` in the
    hundreds of attempts) is a second, larger effect on top of it.  Neither is visible in the season
    gates, which is memory trap 6 exactly: a target's gates cannot tell you whether it should be the
    target.  Only the held-out criterion can.
    """
    if ft not in ("team", "shooter", "none"):
        raise ValueError(f"ft must be team|shooter|none, got {ft!r}")
    comps = tuple(comps)
    c = wd_pts.counters
    if c is None:
        raise KeyError("teamloo targets need the stint counters; the design was built without them")
    need = {"pts", "poss", "fgm", "fg3m", "fga", "fg3a", "ftm", "ftm_tech"}
    need |= {"xftm", "xftm_tech"} if ft == "shooter" else set()
    need |= {"fta", "xftm_tech"} if ft == "team" else set()
    if miss := sorted(need - set(c.columns)):
        raise KeyError(f"teamloo target needs counters {miss}; declare them in TARGET_COLUMNS")

    adj, ktab = adjusted_table(seasons, cfg, keep=keep, prior=prior,
                               comps=tuple(set(comps) | ({"ft"} if ft == "team" else set())),
                               k_fixed=k_fixed, rebalance=rebalance)
    gid = wd_pts.games.set_index("game_idx")["game_id"]
    rows = pd.DataFrame({"season": wd_pts.rows["season"].to_numpy(),
                         "game_id": gid.reindex(wd_pts.rows["game_idx"].to_numpy()).to_numpy(),
                         "is_home_off": wd_pts.rows["is_home_off"].to_numpy()})
    j = rows.merge(adj, on=["season", "game_id", "is_home_off"], how="left")
    if len(j) != len(rows):
        raise ValueError("the team-game join duplicated design rows: game_id is not unique per season")
    # rows the regular-season team-game table does not reach (playoffs) keep their realised points
    uncovered = j["covered"].isna().to_numpy()

    poss = c["poss"].to_numpy(dtype=float)
    fg3m = c["fg3m"].to_numpy(dtype=float)
    fg3a = c["fg3a"].to_numpy(dtype=float)
    fg2m = (c["fgm"] - c["fg3m"]).to_numpy(dtype=float)
    fg2a = (c["fga"] - c["fg3a"]).to_numpy(dtype=float)
    made = {"fg3": fg3m, "fg2": fg2m}
    att = {"fg3": fg3a, "fg2": fg2a}
    pts_adj = 3.0 * fg3m + 2.0 * fg2m
    for comp in comps:
        if comp == "ft":
            continue
        p = np.nan_to_num(j[f"p_adj_{comp}"].to_numpy(dtype=float), nan=0.0)
        delta = np.where(uncovered, 0.0, a * (p * att[comp] - made[comp]))
        pts_adj = pts_adj + VALUE[comp] * delta
    if ft == "none":
        pts_adj = pts_adj + c["ftm"].to_numpy(dtype=float) + c["ftm_tech"].to_numpy(dtype=float)
        ft_ratio_num = c["ftm"].to_numpy(dtype=float)
    elif ft == "shooter":
        pts_adj = pts_adj + c["xftm"].to_numpy(dtype=float) + c["xftm_tech"].to_numpy(dtype=float)
        ft_ratio_num = c["xftm"].to_numpy(dtype=float)
    else:
        p = np.nan_to_num(j["p_adj_ft"].to_numpy(dtype=float), nan=0.0)
        xft = np.where(uncovered, c["ftm"].to_numpy(dtype=float), p * c["fta"].to_numpy(dtype=float))
        pts_adj = pts_adj + xft + c["xftm_tech"].to_numpy(dtype=float)
        ft_ratio_num = xft
    y = 100.0 * pts_adj / np.where(poss > 0, poss, 1.0)

    season = wd_pts.rows["season"].to_numpy()
    g = _gates(c, y, poss, season, ft_ratio_num, comps, j, uncovered)
    cal = pd.DataFrame()
    if calibrate:
        y, cal = align(y, wd_pts.y, wd_pts.w, season, poss)
    name = f"tl_{prior}_{''.join(sorted(comps))}_{ft}_a{a:g}"
    return wd_pts.with_target(y), dict(target=name, gates=g, calibration=cal, k=ktab,
                                       uncovered=float(uncovered.mean()), comps=comps, prior=prior, a=a, ft=ft)


def align(y, y_pts, w, season, poss, band=CALIB_BAND):
    """One scalar per season so the target's season total equals actual points: the LEVEL, nothing
    else.  `xshoot.align` is the same function and the same reasoning (a row-level affine map is a
    partial shrink of the target in disguise); it is re-exported here so this module has no import
    cycle with xshoot."""
    from .xshoot import align as _align
    return _align(y, y_pts, w, season, poss, band=band)


def _gates(c, y, poss, season, ft_num, comps, j, uncovered) -> pd.DataFrame:
    """Per season: the adjusted total against the realised one, by component and overall."""
    rows = []
    real = {"fg3": c["fg3m"].to_numpy(dtype=float), "fg2": (c["fgm"] - c["fg3m"]).to_numpy(dtype=float),
            "ft": c["ftm"].to_numpy(dtype=float)}
    exp = {"fg3": np.nan_to_num(j.get("p_adj_fg3", pd.Series(np.nan, index=j.index)).to_numpy(dtype=float)) * c["fg3a"].to_numpy(dtype=float),
           "fg2": np.nan_to_num(j.get("p_adj_fg2", pd.Series(np.nan, index=j.index)).to_numpy(dtype=float)) * (c["fga"] - c["fg3a"]).to_numpy(dtype=float),
           "ft": ft_num}
    for s in np.unique(season):
        m = season == s
        row = dict(season=int(s), uncovered=float(uncovered[m].mean()))
        for k in ("fg3", "fg2", "ft"):
            row[f"r_{k}"] = float(exp[k][m].sum() / max(real[k][m].sum(), 1.0)) if (k in comps or k == "ft") else 1.0
        row["pts"] = float((y[m] * poss[m] / 100.0).sum() / max(c["pts"].to_numpy(dtype=float)[m].sum(), 1.0))
        rows.append(row)
    g = pd.DataFrame(rows)
    for col in ("r_fg3", "r_fg2", "r_ft"):
        g[f"{col}_ok"] = g[col].between(0.98, 1.02)
    g["pts_ok"] = g["pts"].between(*CALIB_BAND)
    return g


# --------------------------------------------------------------------------- the registries
def _named(fn, name, **kw):
    def target(seasons, cfg, wd_pts, **more):
        return fn(seasons, cfg, wd_pts, **kw, **more)
    target.__name__ = name
    return target


# the counters every teamloo target reads (fastfit.MspiFast.counter_columns -> designcache's whitelist)
_COLS = {"pts", "pts_tech", "poss", "fga", "fgm", "fg3a", "fg3m", "fta", "ftm", "fta_tech", "ftm_tech",
         "xftm", "xftm_tech"}


def _ladder() -> dict:
    """The pre-registered ladder.  Each rung adds one component; the `a` sweep hangs off the top one.

    `tlft*` free throws at the team rate; `tlxft*` the shipped shooter-level free throws underneath a
    team-level shot term, which is the fallback if rung R1 says the team level loses shooter identity.
    """
    out = {"tlfto": dict(comps=(), prior="off", ft="team")}
    for tag, comps in (("3", ("fg3",)), ("32", ("fg3", "fg2"))):
        for pk, prior in (("o", "off"), ("d", "def"), ("b", "both")):
            out[f"tlft{tag}{pk}"] = dict(comps=comps, prior=prior, ft="team")
            out[f"tlxft{tag}{pk}"] = dict(comps=comps, prior=prior, ft="shooter")
    # the partial scalar, on the rungs the criterion actually reached.  `tlxft3o` is there because it
    # WON at a = 1 on the search half (-0.17 per 100, z -2.7) and the sweep asks whether all of the
    # adjustment is wanted or only part; the `32` rungs are there because they lost badly and the
    # sweep bounds how much of that is the two-point term.
    for base in ("tlxft3o", "tlft32o", "tlft32b", "tlxft32o", "tlxft32b"):
        for a in (0.25, 0.5, 0.75):
            out[f"{base}_a{int(a * 100)}"] = dict(out[base], a=a)
    return out


TARGETS = {name: _named(team_design, name, **kw) for name, kw in _ladder().items()}
TARGET_COLUMNS = {name: set(_COLS) for name in TARGETS}
