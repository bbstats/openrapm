"""The direct shot test: is one shot-quality model better than another, before any ratings run.

Stage 2 of the shot-quality build (the owner, 2026-10-06: "shot test first, then the ratings chain decides").
Every function takes a frame of shots (shotframe.derive) carrying one prediction column per ARM -- the
probability a league-average shooter makes that shot -- and returns scores.  Nothing here reads a file.

The tests, each scored on seasons or games the arm never saw, paired by season like 63_yoy:

1. Held-out makes (`season_scores`, `paired`, `calibration`): log loss and Brier per season and shot value,
   and the calibration slope of logit(y) on logit(p) per era block.
2. Arena checks (`arena_signal`): how much of an arm's expected points per 100 attempts is the ARENA rather
   than the teams.  A scorer artefact shows up as an arena effect mirrored by the residual (correlation
   near -0.9, the research's followup_1); real shot quality belongs to the teams.
3. Other-half shooting (`other_half`): a shooter's (team's, defence's) made shots in one half of the season's
   games predicted from the other half, the half being predicted never priced with its own results:
       predicted = (expected makes this half) x (padded made / expected over the other half)
   The padding constant is chosen on the OTHER seasons for every arm separately, so an arm cannot win by a
   lucky constant; with a flat league rate as the arm this is the ordinary padded FG%.
The forward test (first half of a team's games predicts its second half) lives in scripts/118_forward_test.py
because it is built on teamloo's team-game tables.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-6
ERAS = ((1997, 2001), (2002, 2010), (2011, 2016), (2017, 2022), (2023, 2026))
K_GRID = (25.0, 50.0, 100.0, 200.0, 400.0, 800.0, 1600.0, 3200.0)


def _clip(p):
    return np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)


def logloss(y, p) -> np.ndarray:
    p = _clip(p)
    y = np.asarray(y, dtype=float)
    return -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))


def era_of(season) -> np.ndarray:
    s = np.asarray(season)
    out = np.full(s.shape, "", dtype=object)
    for a, b in ERAS:
        out[(s >= a) & (s <= b)] = f"{a}-{b}"
    return out


# ------------------------------------------------------------------------------------------ 1. held-out makes
def season_scores(frame: pd.DataFrame, arms, by=("season", "value")) -> pd.DataFrame:
    """Mean log loss and Brier per group and arm (long format), with the group's mean prediction and make rate."""
    rows = []
    y = frame["made"].to_numpy(float)
    keys = frame[list(by)]
    for arm in arms:
        p = _clip(frame[arm].to_numpy(float))
        d = keys.assign(ll=logloss(y, p), br=(y - p) ** 2, p=p, y=y)
        g = d.groupby(list(by)).agg(n=("ll", "size"), logloss=("ll", "mean"), brier=("br", "mean"),
                                    mean_p=("p", "mean"), mean_y=("y", "mean")).reset_index()
        rows.append(g.assign(arm=arm))
    return pd.concat(rows, ignore_index=True)


def paired(scores: pd.DataFrame, a: str, b: str, metric: str = "logloss", by=("season",)) -> dict:
    """Arm a minus arm b, attempt-weighted within each season, paired across seasons: mean, z, seasons won."""
    def per(arm):
        s = scores[scores.arm == arm]
        return s.groupby(list(by)).apply(lambda g: np.average(g[metric], weights=g["n"]), include_groups=False)
    d = (per(a) - per(b)).dropna()
    if len(d) < 2:
        return dict(diff=float(d.mean()) if len(d) else np.nan, z=np.nan, won=int((d < 0).sum()), n=len(d))
    z = float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d))))
    return dict(diff=float(d.mean()), z=z, won=int((d < 0).sum()), n=len(d))


def _irls(X, y, iters=30, ridge=1e-8):
    beta = np.zeros(X.shape[1])
    for _ in range(iters):
        mu = 1.0 / (1.0 + np.exp(-(X @ beta)))
        s = np.maximum(mu * (1.0 - mu), 1e-9)
        z = X @ beta + (y - mu) / s
        new = np.linalg.solve((X * s[:, None]).T @ X + ridge * np.eye(X.shape[1]), (X * s[:, None]).T @ z)
        if np.max(np.abs(new - beta)) < 1e-9:
            return new
        beta = new
    return beta


def calibration(frame: pd.DataFrame, arms, by=("era", "value")) -> pd.DataFrame:
    """Calibration slope and intercept of logit(y) on logit(p) per group: slope 1 is calibrated, under 1 too
    confident (the spread of p is too wide), over 1 too timid."""
    f = frame.assign(era=era_of(frame["season"].to_numpy()))
    rows = []
    for arm in arms:
        for key, g in f.groupby(list(by)):
            p = _clip(g[arm].to_numpy(float))
            X = np.column_stack([np.ones(len(p)), np.log(p / (1.0 - p))])
            b = _irls(X, g["made"].to_numpy(float))
            rows.append(dict(zip(by, key if isinstance(key, tuple) else (key,)), arm=arm, n=len(g),
                             intercept=float(b[0]), slope=float(b[1]),
                             gap=float(g["made"].mean() - p.mean())))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------ 2. arena checks
def _effects(rows: pd.DataFrame, ycol: str) -> dict:
    """Weighted least squares of a game-side rate on arena + offence + defence + home; centred effects."""
    a = pd.Categorical(rows["arena"])
    o = pd.Categorical(rows["team"])
    d = pd.Categorical(rows["opp"])
    n = len(rows)
    blocks = [np.ones((n, 1)), rows[["home"]].to_numpy(float)]
    names = []
    for cat, tag in ((a, "arena"), (o, "off"), (d, "def")):
        M = np.zeros((n, len(cat.categories)))
        M[np.arange(n), cat.codes] = 1.0
        blocks.append(M[:, 1:])                     # the first level is the reference
        names.append((tag, cat.categories))
    X = np.hstack(blocks)
    w = rows["w"].to_numpy(float)
    sw = np.sqrt(w)
    beta = np.linalg.lstsq(X * sw[:, None], rows[ycol].to_numpy(float) * sw, rcond=None)[0]
    out, j = {}, 2
    for tag, cats in names:
        e = np.r_[0.0, beta[j:j + len(cats) - 1]]
        j += len(cats) - 1
        out[tag] = pd.Series(e - e.mean(), index=cats)
    return out


def arena_signal(frame: pd.DataFrame, arm: str, seed: int = 0) -> pd.DataFrame:
    """Per season: the signal sd of arena, offence and defence effects on the arm's expected points per 100
    attempts, and on the residual make rate (made - p, in percentage points), with the correlation between the
    arena's expected-points effect and its residual effect.

    Signal sd is sqrt(cov) of the effects estimated on two random halves of the season's games, so binomial
    noise drops out.  Neutral-site games are left out (their arena is not the home team's)."""
    f = frame[~frame["neutral"].astype(bool)] if "neutral" in frame.columns else frame
    rows = []
    for season, g in f.groupby("season"):
        p = g[arm].to_numpy(float)
        gg = g.assign(xp=100.0 * g["value"].to_numpy(float) * p,
                      res=100.0 * (g["made"].to_numpy(float) - p))
        side = gg.groupby(["game_id", "team", "opp", "arena", "home"]).agg(
            w=("xp", "size"), xp=("xp", "mean"), res=("res", "mean")).reset_index()
        side["home"] = np.where(side["home"] == 1, 1.0, -1.0)
        games = side["game_id"].unique()
        rng = np.random.default_rng(seed + int(season))
        h = set(rng.permutation(games)[: len(games) // 2])
        part = side["game_id"].isin(h).to_numpy()
        res = {}
        for ycol in ("xp", "res"):
            e1, e2 = _effects(side[part], ycol), _effects(side[~part], ycol)
            for tag in ("arena", "off", "def"):
                a, b = e1[tag].align(e2[tag], join="inner")
                c = float(np.cov(a.to_numpy(), b.to_numpy())[0, 1]) if len(a) > 2 else np.nan
                res[f"{ycol}_{tag}_sd"] = float(np.sqrt(max(c, 0.0)))
            res[f"_{ycol}_arena"] = (e1["arena"] + e2["arena"].reindex(e1["arena"].index)) / 2.0
        ax, ar = res.pop("_xp_arena"), res.pop("_res_arena")
        ax, ar = ax.align(ar, join="inner")
        res["arena_xp_res_corr"] = float(np.corrcoef(ax.to_numpy(), ar.to_numpy())[0, 1]) if len(ax) > 2 else np.nan
        rows.append(dict(season=int(season), arm=arm, **res))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------ 3. other-half shooting
UNITS = {"shooter": "shooter", "offence": "team", "defence": "opp"}


def _halves(frame: pd.DataFrame, arm: str, unit: str, value, dest_arm: str | None = None) -> pd.DataFrame:
    """Per season and unit: attempts, makes and expected makes in each half of the season's games.  `x` is the
    arm's expected makes (the predicting half's pricing), `xd` the `dest_arm`'s (the predicted half's)."""
    f = frame if value is None else frame[frame["value"] == value]
    f = f[f["half"].isin(["A", "B"])]
    key = UNITS[unit]
    g = f.assign(q=f[arm].to_numpy(float), qd=f[dest_arm or arm].to_numpy(float)).groupby(["season", key, "half"]).agg(
        n=("made", "size"), m=("made", "sum"), x=("q", "sum"), xd=("qd", "sum")).unstack("half")
    g.columns = [f"{a}_{b}" for a, b in g.columns]
    return g.fillna(0.0).reset_index()


def _priced_error(h: pd.DataFrame, k: float, min_fga: float) -> pd.DataFrame:
    """Both directions (A predicts B, B predicts A): attempt-weighted squared error of the predicted make rate."""
    out = []
    for src, dst in (("A", "B"), ("B", "A")):
        keep = (h[f"n_{src}"] >= min_fga) & (h[f"n_{dst}"] >= min_fga)
        s = h[keep]
        n_s, m_s, x_s = s[f"n_{src}"].to_numpy(float), s[f"m_{src}"].to_numpy(float), s[f"x_{src}"].to_numpy(float)
        p_mix = x_s / n_s
        p_pad = (m_s + k * p_mix) / (n_s + k)
        ratio = p_pad / np.maximum(p_mix, EPS)
        n_d, m_d, x_d = s[f"n_{dst}"].to_numpy(float), s[f"m_{dst}"].to_numpy(float), s[f"xd_{dst}"].to_numpy(float)
        pred = np.clip(ratio * x_d / n_d, EPS, 1.0 - EPS)
        out.append(pd.DataFrame(dict(season=s["season"].to_numpy(), n=n_d, se=n_d * (m_d / n_d - pred) ** 2)))
    return pd.concat(out, ignore_index=True)


def other_half(frame: pd.DataFrame, arm: str, unit: str = "shooter", value=None, min_fga: float = 100.0,
               ks=K_GRID, dest_arm: str | None = None) -> pd.DataFrame:
    """Per season: the held-out error of predicting one half's make rate from the other's, with the padding
    constant chosen to minimise the pooled error of the OTHER seasons.  Error in squared percentage points
    of make rate, attempt-weighted; `k` the constant the other seasons chose.

    `dest_arm`: the column that prices the PREDICTED half.  An arm that has seen each shot's own result (the
    owner's "after" quality) may price the predicting half only, so its before-twin prices the other."""
    h = _halves(frame, arm, unit, value, dest_arm)
    grid = {k: _priced_error(h, k, min_fga) for k in ks}
    per = {k: g.groupby("season").agg(se=("se", "sum"), n=("n", "sum")) for k, g in grid.items()}
    seasons = sorted(h["season"].unique())
    rows = []
    for s in seasons:
        best, best_e = None, np.inf
        for k, t in per.items():
            o = t.drop(index=s, errors="ignore")
            e = float(o["se"].sum() / max(o["n"].sum(), 1.0))
            if e < best_e:
                best, best_e = k, e
        t = per[best]
        if s in t.index and t.loc[s, "n"] > 0:
            rows.append(dict(season=int(s), arm=arm, unit=unit, value=value if value is not None else 0,
                             k=best, n=float(t.loc[s, "n"]), err=1e4 * float(t.loc[s, "se"] / t.loc[s, "n"])))
    return pd.DataFrame(rows)


def paired_seasons(a: pd.DataFrame, b: pd.DataFrame, col: str = "err") -> dict:
    """Two per-season tables (same seasons), a minus b: mean, z, seasons where a is lower."""
    m = a.set_index("season")[col].sub(b.set_index("season")[col]).dropna()
    if len(m) < 2:
        return dict(diff=float(m.mean()) if len(m) else np.nan, z=np.nan, won=int((m < 0).sum()), n=len(m))
    return dict(diff=float(m.mean()), z=float(m.mean() / (m.std(ddof=1) / np.sqrt(len(m)))),
                won=int((m < 0).sum()), n=len(m))


# ------------------------------------------------------------------------------------------ the forward test's pieces
def loso_mse(D: pd.DataFrame, cols: list) -> tuple[float, pd.Series]:
    """Held-out weighted MSE of a leave-one-season-out regression of `truth` on `cols` (+ intercept), and the
    per-season error.  A regression absorbs any global rescaling, so a uniformly shrunk arm scores exactly as
    its unshrunk twin and only the STRUCTURE of an arm can win (the old scratch/forward.py, verbatim)."""
    per, tot_e, tot_w = {}, 0.0, 0.0
    for s in sorted(D.season.unique()):
        tr, te = D[D.season != s], D[D.season == s]
        X = np.column_stack([np.ones(len(tr))] + [tr[c].to_numpy(float) for c in cols])
        sw = np.sqrt(tr.w.to_numpy(float))
        beta = np.linalg.lstsq(X * sw[:, None], tr.truth.to_numpy(float) * sw, rcond=None)[0]
        Xt = np.column_stack([np.ones(len(te))] + [te[c].to_numpy(float) for c in cols])
        e = (te.truth.to_numpy(float) - Xt @ beta) ** 2
        w = te.w.to_numpy(float)
        per[s] = float((e * w).sum() / w.sum())
        tot_e += float((e * w).sum())
        tot_w += float(w.sum())
    return tot_e / tot_w, pd.Series(per)


def zone_of(frame: pd.DataFrame) -> np.ndarray:
    """The stints' zone of each attempt (stints.py: `rim` at 0-3 feet by the feed's distance, `mid` every other
    two, `thr` a three), so per-game zone counts reconcile with the team-game counters exactly."""
    three = frame["value"].to_numpy() == 3
    d = frame["dist"].to_numpy(float)
    return np.where(three, "thr", np.where((d >= 0) & (d <= 3.0), "rim", "mid"))


# ------------------------------------------------------------------------------------------ 2(c). the dashboards
def contest_left(frame: pd.DataFrame, arms, dash: pd.DataFrame, min_fga: int = 5) -> pd.DataFrame:
    """Per season and arm: how much of a player-game's shooting beyond the arm's expectation the tracking
    dashboards' contest still explains.

    `dash` has date, PLAYER_ID, `tight` (attempts with the closest defender within 4 ft) and `fga`.  For each
    player-game with `min_fga`+ attempts: residual = (makes - sum of the arm's q) / attempts, and the slope of the
    residual on the share of tightly guarded attempts (attempt-weighted, one level per season).  An arm that sees
    how contested the shots were leaves a slope near zero; one that does not leaves it negative (tightly guarded
    shots go in less than it expects).  Before-quality arms only: an arm that saw each shot's result has no
    business here."""
    f = frame.assign(date=pd.to_datetime(frame["game_date"]).dt.strftime("%Y-%m-%d"))
    rows = []
    for arm in arms:
        g = f.groupby(["season", "date", "shooter"]).agg(n=("made", "size"), m=("made", "sum"), x=(arm, "sum")).reset_index()
        j = g.merge(dash, left_on=["date", "shooter"], right_on=["date", "PLAYER_ID"], how="inner")
        j = j[(j["n"] >= min_fga) & (j["fga"] == j["n"])]          # the dashboard saw every attempt we did
        for season, h in j.groupby("season"):
            r = (h["m"] - h["x"]) / h["n"]
            s = h["tight"] / h["fga"]
            w = h["n"].to_numpy(float)
            X = np.column_stack([np.ones(len(h)), s.to_numpy(float)])
            sw = np.sqrt(w)
            b, *_ = np.linalg.lstsq(X * sw[:, None], r.to_numpy(float) * sw, rcond=None)
            e = r.to_numpy(float) - X @ b
            se = float(np.sqrt((w * e ** 2).sum() / (len(h) - 2) / (w * (s - np.average(s, weights=w)) ** 2).sum()))
            rows.append(dict(season=int(season), arm=arm, player_games=len(h), slope=float(b[1]), se=se))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------ 2(b). the teacher
def teacher_gap(frame: pd.DataFrame, arms, labels: pd.DataFrame) -> pd.DataFrame:
    """Per tracked game and arm: the mean squared gap between the arm's quality and the tracking teacher's
    out-of-fold quality (data/shotq/teacher/labels.parquet), in squared points of make probability.  Before-quality
    arms only."""
    j = frame.merge(labels[["game_id", "action_number", "q_teacher"]], on=["game_id", "action_number"], how="inner")
    rows = []
    for arm in arms:
        g = (1e4 * (j[arm] - j["q_teacher"]) ** 2).groupby([j["season"], j["game_id"]]).mean()
        rows.append(g.rename("gap").reset_index().assign(arm=arm))
    return pd.concat(rows, ignore_index=True)


def paired_games(gaps: pd.DataFrame, a: str, b: str) -> dict:
    d = gaps[gaps.arm == a].set_index(["season", "game_id"])["gap"] - gaps[gaps.arm == b].set_index(["season", "game_id"])["gap"]
    d = d.dropna()
    return dict(diff=float(d.mean()), z=float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))), won=int((d < 0).sum()), n=len(d))
