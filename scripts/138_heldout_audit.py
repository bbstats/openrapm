"""The held-out audit: how far each season's own held-out games say the ratings lean on every input, and on every
group of correlated inputs (the Robustness pass, step 1; DECISIONS.md, "The Robustness pass").

    python scripts/138_heldout_audit.py [--random=lgb_noonc_within] [--mult_tag=lgb_noonc_within] [--deadline=]
                                        [--tradeset=outputs/tradeset_lgb_noonc_rs_pts_alpha.parquet]
                                        [--out=audit_lgb_noonc] [--draws=400] [--threshold_from=]
                                        [--reference=<tag>[:<mult tag>]] [--control=<tag>[:<mult tag>]]
                                        [--freeze=0] [--shares=2017,2021,2026]

A candidate (the Robustness pass's builds) is read with its own folds and multipliers (`--random=<its tag>
--mult_tag=<its tag>`), on the frozen null (`--draws=0 --threshold_from=<the baseline's --out>`), and PAIRED with the
incumbent's folds (`--reference=lgb_noonc_within`) and the noise control's (`--control=<tag>`) on identical test games:
every lean's change with a jackknife of the paired difference, how far the ratings themselves moved along each axis,
and the paired held-out error per fold (outputs/csv/<out>_paired.csv, outputs/<out>_paired_error.json).

What is measured, in one sentence.  For every held-out team-game of the within-season folds (scripts/97: a season
rated from three quarters of its games, scored on the fourth), the points per 100 as scored minus what the ratings
predict -- each rated player's share of the possessions times his rating as it ships (the box-score prior part
times the multiplier scripts/99 fits without that fold's season, plus the games' part, re-centred), the stand-in
ratings of unrated players held at their own values, the level and home edge free per fold -- is regressed on a
correction per player that depends on one input (or one group's combined axis), entering the team-game the way a
rating does; the correction's slope says how far the games say players high on that input are underrated (+) or
overrated (-), in points per 100.

Why this and not 137.  137 reads the with/without correction, which comes from the NEIGHBOURING seasons (it teaches
"peak seasons regress"; the owner's 2026-10-03 rule forbids fitting on it), scores luck-adjusted targets, and its
null shuffles inputs within a season -- which inflates the z of anything a player carries from season to season
3-4x (body weight 16.9 -> about 4).  Here: actual points on the season's own games; each fold's REBUILT inputs on
the axis (the full-season panel holds the held-out games' own steals: that leak turns the steal lean's z from -3.1
into -0.8); a null that moves whole careers; and one threshold for all the tests together.

The axis.  An input becomes a within-fold possession-weighted percentile (ties share their midpoint) and its normal
score.  A group is a set of inputs that move together: average linkage on 1 - |rank correlation| over the incumbent's
whole-season rated rows (500+ possessions, the seasons of the first --random folder), cut at 0.4, 0.5 and 0.6; its
axis is the frozen first principal component of its members' normal scores, ranked again.  Groups and loadings are
frozen in params/audit_groups.json (--freeze=1 writes it, once, before the first build; trap 22).

The statistics, each per side and on its own:
  line    the correction is a straight line in the axis' normal score: points per 100 per standard deviation
  bend    a straight line plus a square: the square's coefficient (a hump is negative, a U positive)
  tenths  one level per possession-weighted tenth: the top tenth minus the bottom tenth, points per 100
The z and the THRESHOLD come from a null that moves whole careers: every player is handed another player's inputs
(all of them, so the inputs keep their correlations), season by season along the donor's own career (so a player's
inputs persist across his seasons as much as real ones do).  Each test's line and bend estimates are recomputed for
every reassignment; a test's z is its estimate over its spread under that null, and the threshold is the 95th
percentile of the largest |z| over all the tests that can count (team context, playing time and score state are
measured, never corrected, so they are left out of it).  A season jackknife is printed beside it but does not decide:
a player's misfit carries over between seasons, so it is too small for anything a player carries with him.  Every
reading prints the smallest lean it could have detected: the threshold times its spread under the null.

The stages, the scale pinned to what ships:
  prior    the prior part alone (times its multiplier), the season's games' part left out
  raw      the rating before the prior shrink (prior part + games' part, as the fold built it)
  shrunk   the rating as it ships before the swap step -- the decision stage
Versions at the decision stage: `pinned` (the primary), `prior_free` (the two prior multipliers refitted together with
the lean), `rating_free` (one free scale per side on the whole rating), `nuisance` (pinned, plus log possessions and
the team-game shares of rated and of under-500-possession players).  A lean COUNTS only if, at the shrunk stage, its
line or bend z passes the threshold, its tenths reading is at least 0.1 per 100 from bottom to top, its line keeps
its sign (and its size within a factor of two) under all three scale versions, and -- once the folds span more than
one era or a deadline folder is given -- it keeps its sign in every era and the deadline folds do not contradict it
at |z| 2 or more.  Team context, playing time, score state and the nuisance never count.

`--shares=2017,2021,2026` also writes each group's share of the prior: Shapley values with every group of correlated
inputs as one player, against a background of the season's rated rows, on the incumbent's saved prior models
(`group_shapley_shares`; LightGBM cannot give SHAP values for linear leaves).  About a minute a season.

Writes outputs/csv/<out>.csv (every reading), outputs/csv/<out>_tenths.csv (each tenth's level),
outputs/csv/<out>_summary.csv (one line per test and side at the decision stage), outputs/<out>_thresholds.json, and
with --shares outputs/csv/<out>_shares.csv.
"""
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.cluster.hierarchy import fcluster, linkage  # noqa: E402
from scipy.spatial.distance import squareform  # noqa: E402
from scipy.stats import norm  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eracoef.config import load_config  # noqa: E402
from eracoef.glossary import plain  # noqa: E402
from eracoef.seasons import drop_untrainable  # noqa: E402
from _cli import check_flags, flag  # noqa: E402


def _borrow(name: str, file: str):
    """Another script's functions, loaded the way 99 loads 98's: one definition, never a copy."""
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


C98 = _borrow("_calib98", "98_calibrator.py")      # Fold
C99 = _borrow("_shrink99", "99_prior_shrink.py")   # load_folds, multipliers_for, fit_prior_multipliers

# ------------------------------------------------------------------------------------------- what is audited
# Every input a fold's player rows carry that a prior could read, by meaning (GLOSSARY.md has the plain names).
CLASSES = {
    "box score": ["fg3m", "fg3_miss", "fg2m", "fg2_miss", "ftm", "ft_miss", "orb", "drb", "ast", "tov", "stl",
                  "blk", "pf", "pts", "fga", "fta", "fg3a", "usage", "reb", "stocks", "creation", "shotmix"],
    "shooting and shares": ["efg", "ts", "fg3p", "fg2p", "ftp", "p3r", "ftr", "astr", "tovr", "orbsh"],
    "shot quality": ["q2", "q3", "m2", "m3", "xps", "mpts"],
    "body": ["height", "weight"],
    "age and career": ["age", "exp_yrs", "exp_poss", "entry_age", "tenure", "n_teams", "draft_pick"],
    "playing time": ["poss_pct", "gs_pct"],
    "score state": ["gt_share", "closeness", "abs_margin", "po_share"],
    "team context": ["onc_o", "onc_d", "offc_o", "offc_d", "net_o", "net_d", "team_net"],
}
MEASURED_ONLY = {"team context", "playing time", "score state"}   # measured, never corrected, outside the threshold
GROUPED = [c for k, v in CLASSES.items() if k != "team context" for c in v]
CLASS_OF = {c: k for k, v in CLASSES.items() for c in v}
SIDES = ("O", "D")
SIGN = {"O": 1.0, "D": -1.0}           # a correction raises a player's rating; on defence that lowers points allowed
CUTS = (0.4, 0.5, 0.6)
GROUP_MIN_POSS = 500.0
TENTHS = 10
REFERENCE_TENTH = 4                     # the dropped level; the top-minus-bottom reading does not depend on it
NCOL = 2 + TENTHS - 1                   # each test's columns: z, z squared, the nine tenths kept
STAGES = ("prior", "raw", "shrunk")
VERSIONS = ("prior_free", "rating_free", "nuisance")
GROUPS_FILE = ROOT / "params" / "audit_groups.json"
ERAS = ((1997, 2006), (2007, 2016), (2017, 2026))


def era_of(season: int) -> str:
    return next(f"{a}-{b}" for a, b in ERAS if a <= season <= b)


def poss_of(rows: pd.DataFrame, side: str) -> np.ndarray:
    return rows[("poss_off" if side == "O" else "poss_def")].to_numpy(float)


def percentile(x: np.ndarray, w: np.ndarray, broken: bool = False) -> np.ndarray:
    """Possession-weighted percentile in (0, 1); tied values share the midpoint of their block.  `broken` breaks ties
    by possessions instead, so an input with a large tied block (one team, no draft slot) still fills its tenths."""
    x = np.asarray(x, dtype=float)
    if np.isnan(x).any():
        x = np.where(np.isnan(x), np.nanmedian(x) if np.isfinite(x).any() else 0.0, x)
    if broken:
        order = np.lexsort((w, x))
        cw = np.cumsum(w[order])
        p = np.empty_like(x)
        p[order] = (cw - w[order] / 2.0) / cw[-1]
        return p
    order = np.argsort(x, kind="mergesort")
    xs, ws = x[order], w[order]
    _, start = np.unique(xs, return_index=True)
    block_w = np.add.reduceat(ws, start)
    before = np.concatenate([[0.0], np.cumsum(block_w)[:-1]])
    mid = (before + block_w / 2.0) / block_w.sum()
    p = np.empty_like(x)
    p[order] = np.repeat(mid, np.diff(np.append(start, len(xs))))
    return p


def normal_score(p: np.ndarray) -> np.ndarray:
    return norm.ppf(np.clip(p, 1e-4, 1.0 - 1e-4))


def centre_columns(values: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Each column's possession-weighted mean taken out (a correction is centred within its fold and side)."""
    values = np.asarray(values, dtype=float)
    return values - (w @ values) / w.sum()


# --------------------------------------------------------------------------------------------- the groups
def freeze_groups(directory: Path) -> dict:
    """Average linkage on 1 - |rank correlation| per side, over the incumbent's whole-season rated rows with 500+
    possessions; the clusters at each cut with two or more members, and their axis loadings."""
    files = sorted(directory.glob("players_*_all.parquet"))
    assert files, f"no whole-season rows in {directory}"
    frame = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
    out = {"source": str(directory.relative_to(ROOT)).replace("\\", "/"),
           "seasons": sorted(int(s) for s in frame.season.unique()), "min_possessions": GROUP_MIN_POSS,
           "cuts": list(CUTS), "codes": GROUPED, "sides": {}}
    for side in SIDES:
        rows = frame[(frame.side == side) & (poss_of(frame, side) >= GROUP_MIN_POSS)].reset_index(drop=True)
        z = np.column_stack([norm.ppf((rows.groupby("season")[c].rank(method="average") - 0.5)
                                      / rows.groupby("season")[c].transform("count")) for c in GROUPED])
        dist = 1.0 - np.abs(np.corrcoef(z.T))
        np.fill_diagonal(dist, 0.0)
        tree = linkage(squareform(np.clip(dist, 0.0, None), checks=False), method="average")
        labels = {c: fcluster(tree, t=c, criterion="distance") for c in CUTS}
        groups, seen = [], {}
        for cut in sorted(CUTS, reverse=True):                           # coarsest first
            for lab in np.unique(labels[cut]):
                idx = np.where(labels[cut] == lab)[0]
                key = tuple(sorted(GROUPED[j] for j in idx))
                if len(idx) < 2:
                    continue
                if key in seen:
                    seen[key]["cuts"].append(cut)
                    continue
                vals, vecs = np.linalg.eigh(np.corrcoef(z[:, idx].T))
                load = vecs[:, -1] * np.sign(vecs[np.argmax(np.abs(vecs[:, -1])), -1])
                order = np.argsort(-np.abs(load))
                g = dict(members=[GROUPED[idx[i]] for i in order], loadings=[float(load[i]) for i in order],
                         cuts=[cut], explained=float(vals[-1] / vals.sum()))
                seen[key] = g
                groups.append(g)
        for g in groups:
            g["cuts"] = sorted(g["cuts"])
        out["sides"][side] = dict(linkage=tree.tolist(), groups=groups, rows=int(len(rows)))
    return out


def group_name(members: list) -> str:
    head = ", ".join(plain(m) for m in members[:3])
    return head + (f" (+{len(members) - 3} more)" if len(members) > 3 else "")


def build_tests(groups: dict) -> list:
    tests = [dict(id=c, kind="input", code=c, cls=CLASS_OF[c], name=plain(c)) for k, v in CLASSES.items() for c in v]
    for side in SIDES:
        for g in groups["sides"][side]["groups"]:
            gid = "group:" + "+".join(sorted(g["members"]))
            if any(t["id"] == gid for t in tests):
                continue
            only = all(CLASS_OF[m] in MEASURED_ONLY for m in g["members"])
            tests.append(dict(id=gid, kind="group", members=g["members"], loadings=g["loadings"],
                              cls="playing time" if only else "group", name=group_name(g["members"]),
                              cuts=g["cuts"]))
    return tests


def decision_test(t: dict) -> bool:
    return t["cls"] not in MEASURED_ONLY


# ---------------------------------------------------------------------------------------------- the folds
class AuditFold:
    """One fold, reduced to what the audit needs.  With the level and home edge projected out (weighted), a
    correction basis B (one row per rated player of a side, centred) has X'X = B'HB, X'r = sign B'g and X'N = sign
    B'M for any pooled free columns N, so every regression is read off these small matrices."""

    def __init__(self, fold, m_o: float, m_d: float, setting: str, swap: pd.DataFrame | None = None):
        self.season, self.setting, self.stem = fold.season, setting, fold.stem
        self.repeat, self.fold = fold.repeat, fold.fold
        self.rows = fold.rows
        sw = np.sqrt(fold.w)
        u, s, _ = np.linalg.svd(fold.F * sw[:, None], full_matrices=False)
        Q = u[:, s > 1e-9 * s.max()]

        def proj(v):
            v = np.asarray(v, dtype=float)
            sv = (sw * v.T).T if v.ndim == 2 else sw * v
            return sv - Q @ (Q.T @ sv)

        o, d = fold.rows["O"], fold.rows["D"]
        po, uo = o.prior_off.to_numpy(float), o.u_off.to_numpy(float)
        pdf, ud = d.prior_def.to_numpy(float), d.u_def.to_numpy(float)
        Zo, Zd = fold.Z["O"], fold.Z["D"]
        prior_o, games_o = Zo @ po, Zo @ uo
        prior_d, games_d = Zd @ (-pdf), Zd @ (-ud)
        stand_in = fold.base - prior_o - games_o - prior_d - games_d
        # 99's shrink on this fold: the prior part times its season's multiplier, re-centred per side by possessions
        lo = np.average(m_o * po + uo, weights=o.poss_off.to_numpy(float))
        ld = np.average(m_d * pdf + ud, weights=d.poss_off.to_numpy(float))
        sp_o, sp_d = m_o * po - lo, m_d * pdf - ld
        self.shrunk_rating = {"O": sp_o + uo, "D": sp_d + ud}            # positive = good on both sides
        self.game_idx = np.asarray(fold.game_idx)
        self.total_w = float(fold.w.sum())
        miss = {"prior": fold.y - stand_in - Zo @ sp_o - Zd @ (-sp_d),
                "raw": fold.y - fold.base,
                "shrunk": fold.y - stand_in - Zo @ (sp_o + uo) - Zd @ (-(sp_d + ud))}
        if swap is not None:
            # 106's ratings on this fold: its own shrink (99's test rule) and the swap step on top of it, read as two
            # more stages so the swap step's own lean is the difference between them, multipliers held equal
            for stage, variant in (("shrunk106", "shrunk"), ("swap", "swap0.5")):
                v = swap[swap.variant == variant].set_index("player_id")
                o_r = v.o.reindex(o.player_id.to_numpy()).to_numpy(float)
                d_r = v.d.reindex(d.player_id.to_numpy()).to_numpy(float)
                assert np.isfinite(o_r).all() and np.isfinite(d_r).all(), f"{fold.stem}: 106 lacks rated players"
                miss[stage] = fold.y - stand_in - Zo @ o_r - Zd @ d_r
        free = {"prior_free": (fold.y - stand_in - games_o - games_d, np.column_stack([prior_o, prior_d])),
                "rating_free": (fold.y - stand_in, np.column_stack([prior_o + games_o, prior_d + games_d]))}
        self.weights = {s_: np.maximum(poss_of(fold.rows[s_], s_), 1e-9) for s_ in SIDES}
        logp = {s_: centre_columns(np.log1p(poss_of(fold.rows[s_], s_)), self.weights[s_]) for s_ in SIDES}
        rated = np.asarray(Zo.sum(axis=1)).ravel() + np.asarray(Zd.sum(axis=1)).ravel()
        bench = sum(np.asarray(fold.Z[s_][:, np.where(poss_of(fold.rows[s_], s_) < 500)[0]].sum(axis=1)).ravel()
                    for s_ in SIDES)
        nuis = np.column_stack([SIGN[s_] * (fold.Z[s_] @ logp[s_]) for s_ in SIDES] + [rated, bench])
        free["nuisance"] = (miss["shrunk"], nuis)

        r = {k: proj(v) for k, v in miss.items()}
        rv = {k: proj(v[0]) for k, v in free.items()}
        Nv = {k: proj(v[1]) for k, v in free.items()}
        # pooled terms of the free columns (the same whatever the test)
        self.NN = {k: Nv[k].T @ Nv[k] for k in free}
        self.Nr = {k: Nv[k].T @ rv[k] for k in free}
        self.H, self.g, self.gv, self.M, self.Zt_shape = {}, {}, {}, {}, {}
        self.rshrunk = r["shrunk"]
        self.proj = proj
        self.Z = fold.Z
        for side in SIDES:
            Zt = proj(fold.Z[side].toarray())
            self.H[side] = (Zt.T @ Zt).astype(np.float32)
            self.g[side] = {k: Zt.T @ v for k, v in r.items()}
            self.gv[side] = {k: Zt.T @ v for k, v in rv.items()}
            self.M[side] = {k: Zt.T @ v for k, v in Nv.items()}


def load_setting(tags: list, setting: str, multipliers: pd.DataFrame) -> list:
    """Every fold of the named folders (any size or split), with its season's shipped multipliers."""
    m = multipliers.set_index("season")
    folds = []
    for tag in tags:
        directory = ROOT / "outputs" / "within" / tag
        swap_file = directory / "openrapm_shipped.parquet"           # 106's ratings, when it has run on the folder
        swap = pd.read_parquet(swap_file) if swap_file.exists() else None
        if swap is not None:
            swap = swap[swap.variant.isin(["shrunk", "swap0.5"])]
            swap = dict(tuple(swap.groupby("key")))
        for path in sorted(directory.glob("fold_*.json")):
            stem = path.stem[len("fold_"):]
            info = json.loads(path.read_text())
            failed = [k for k, v in info["checks"].items() if isinstance(v, bool) and not v]
            assert not failed, f"{tag} fold {stem}: leak checks failed {failed}"
            fold = C98.Fold(directory, stem)
            folds.append(AuditFold(fold, float(m.at[fold.season, "prior_off"]),
                                   float(m.at[fold.season, "prior_def"]), setting,
                                   swap.get(stem) if swap is not None else None))
    return folds


def shipped_multipliers(seasons, mult_tag: str = "lgb_noonc_within"):
    """The product rule's multipliers for every season (99 as it ships: the folds of `mult_tag` pinned to 2017-2026,
    the rated season's own folds left out; other seasons borrow the pooled value).  A candidate is audited on its own
    multipliers, fitted on its own folds -- what would ship with it."""
    folds = C99.load_folds(ROOT / "outputs" / "within" / mult_tag, (2017, 2026))
    return C99.multipliers_for(seasons, folds, "product"), folds


# ------------------------------------------------------------------------------------------------- the axes
def axis_scores(rows: pd.DataFrame, w: np.ndarray, tests: list, broken: bool = False) -> np.ndarray:
    """Each test's percentile for the rated rows (columns in the order of `tests`); `broken` = ties broken by
    possessions (for the tenths), otherwise tied values share a percentile (for the line and the bend)."""
    member = {}
    out = np.empty((len(rows), len(tests)))
    for j, t in enumerate(tests):
        if t["kind"] == "input":
            out[:, j] = percentile(rows[t["code"]].to_numpy(float), w, broken)
        else:
            for m in t["members"]:
                if m not in member:
                    member[m] = normal_score(percentile(rows[m].to_numpy(float), w))
            out[:, j] = percentile(sum(a * member[m] for m, a in zip(t["members"], t["loadings"])), w, broken)
    return out


def basis_block(p: np.ndarray, p_tenths: np.ndarray) -> np.ndarray:
    """[z, z squared, the nine kept tenths] for one test: z from the shared-ties percentile, the tenths from the
    percentile with ties broken."""
    z = normal_score(p)
    tenth = np.minimum((p_tenths * TENTHS).astype(int), TENTHS - 1)
    return np.column_stack([z, z ** 2, np.delete(np.eye(TENTHS)[tenth], REFERENCE_TENTH, axis=1)])


# ------------------------------------------------------------------------------------------- the regressions
LINE, BEND, TEN = [0], [0, 1], list(range(2, NCOL))


def solve_block(G: np.ndarray, h: np.ndarray, cols: list, extra_G=None, extra_c=None, extra_h=None) -> np.ndarray:
    """The coefficients of `cols` (and any pooled free columns) by least squares from the accumulated sums."""
    Gs, hs = G[np.ix_(cols, cols)], h[cols]
    if extra_G is not None:
        c = extra_c[cols, :]
        Gs = np.block([[Gs, c], [c.T, extra_G]])
        hs = np.concatenate([hs, extra_h])
    return np.linalg.lstsq(Gs, hs, rcond=None)[0][:len(cols)]


def statistic(kind: str, beta: np.ndarray) -> float:
    if kind == "line":
        return float(beta[0])
    if kind == "bend":
        return float(beta[1])
    full = np.insert(beta, REFERENCE_TENTH, 0.0)
    return float(full[-1] - full[0])


def run_tests(folds: list, tests: list, by_era: bool) -> tuple:
    """Every test x side x stage (pinned) and every version at the shrunk stage: estimate, jackknife se, z; plus the
    tenths' levels.  Sums are kept per season (and era) so a season can be dropped."""
    ntest = len(tests)
    stages = [st for st in STAGES + ("shrunk106", "swap") if all(st in f.g["O"] for f in folds)]
    acc = {}                         # (side, era) -> season -> dict of arrays
    for fold in folds:
        for side in SIDES:
            P = axis_scores(fold.rows[side], fold.weights[side], tests)
            Pt = axis_scores(fold.rows[side], fold.weights[side], tests, broken=True)
            B = np.concatenate([basis_block(P[:, j], Pt[:, j])[:, :, None] for j in range(ntest)], axis=2)
            # B: players x NCOL x tests; centre within the fold and side
            w = fold.weights[side]
            B = B - np.einsum("i,ijk->jk", w, B)[None] / w.sum()
            flat = B.reshape(len(B), -1)
            HB = (fold.H[side] @ flat.astype(np.float32)).astype(float).reshape(B.shape)
            G = np.einsum("ijk,ilk->kjl", B, HB)                                   # tests x NCOL x NCOL
            h = {st: SIGN[side] * np.einsum("ijk,i->kj", B, fold.g[side][st]) for st in stages}
            hv = {v: SIGN[side] * np.einsum("ijk,i->kj", B, fold.gv[side][v]) for v in VERSIONS}
            cv = {v: SIGN[side] * np.einsum("ijk,il->kjl", B, fold.M[side][v]) for v in VERSIONS}
            for era in ("all",) + ((era_of(fold.season),) if by_era else ()):
                slot = acc.setdefault((side, era), {}).setdefault(fold.season, {
                    "G": 0.0, **{f"h_{st}": 0.0 for st in stages}, **{f"hv_{v}": 0.0 for v in VERSIONS},
                    **{f"cv_{v}": 0.0 for v in VERSIONS}, **{f"NN_{v}": 0.0 for v in VERSIONS},
                    **{f"Nr_{v}": 0.0 for v in VERSIONS}})
                slot["G"] = slot["G"] + G
                for st in stages:
                    slot[f"h_{st}"] = slot[f"h_{st}"] + h[st]
                for v in VERSIONS:
                    slot[f"hv_{v}"] = slot[f"hv_{v}"] + hv[v]
                    slot[f"cv_{v}"] = slot[f"cv_{v}"] + cv[v]
                    if side == "O":                                  # pooled terms once per fold, shared by sides
                        slot[f"NN_{v}"] = slot[f"NN_{v}"] + fold.NN[v]
                        slot[f"Nr_{v}"] = slot[f"Nr_{v}"] + fold.Nr[v]
    if any(k[0] == "D" for k in acc):                                 # copy the pooled terms to the defence slots
        for (side, era), seasons in acc.items():
            if side == "D":
                for s, slot in seasons.items():
                    for v in VERSIONS:
                        slot[f"NN_{v}"] = acc[("O", era)][s][f"NN_{v}"]
                        slot[f"Nr_{v}"] = acc[("O", era)][s][f"Nr_{v}"]
    rows, tenth_rows, dropped = [], [], {}
    kinds = {"line": LINE, "bend": BEND, "tenths": TEN}
    for (side, era), seasons in acc.items():
        keys = sorted(seasons)
        tot = {k: sum(seasons[s][k] for s in keys) for k in seasons[keys[0]]}
        left_out = {s: {k: tot[k] - seasons[s][k] for k in tot} for s in keys} if len(keys) > 2 else {}
        variants = [("pinned", st) for st in stages] + ([(v, "shrunk") for v in VERSIONS] if era == "all" else [])
        for j, t in enumerate(tests):
            for version, st in variants:
                for kind, cols in kinds.items():
                    def est(sums):
                        if version == "pinned":
                            return statistic(kind, solve_block(sums["G"][j], sums[f"h_{st}"][j], cols))
                        return statistic(kind, solve_block(sums["G"][j], sums[f"hv_{version}"][j], cols,
                                                           sums[f"NN_{version}"], sums[f"cv_{version}"][j],
                                                           sums[f"Nr_{version}"]))
                    e = est(tot)
                    drops = np.array([est(left_out[s]) for s in keys]) if left_out else None
                    se = (float(np.sqrt((len(keys) - 1) / len(keys) * ((drops - drops.mean()) ** 2).sum()))
                          if drops is not None else float("nan"))
                    rows.append(dict(test=t["id"], side=side, stage=st, version=version, statistic=kind, era=era,
                                     estimate=e, se=se, z=e / se if se > 0 else float("nan"), seasons=len(keys)))
                    dropped[(t["id"], side, st, version, kind, era)] = (keys, drops)
                if version == "pinned" and era == "all":
                    lv = np.insert(solve_block(tot["G"][j], tot[f"h_{st}"][j], TEN), REFERENCE_TENTH, 0.0)
                    for k, x in enumerate(lv - lv.mean()):
                        tenth_rows.append(dict(test=t["id"], side=side, stage=st, tenth=k + 1, level=float(x)))
    return pd.DataFrame(rows), pd.DataFrame(tenth_rows), dropped


def run_joint(folds: list, tests: list) -> pd.DataFrame:
    """All groups' lines together, both sides, at the decision stage: each group's lean holding the others fixed."""
    groups = [t for t in tests if t["kind"] == "group"]
    G, h = {}, {}
    for fold in folds:
        cols = []
        for side in SIDES:
            z = normal_score(axis_scores(fold.rows[side], fold.weights[side], groups))
            z = centre_columns(z, fold.weights[side])
            cols.append(SIGN[side] * fold.proj(np.asarray(fold.Z[side] @ z)))
        X = np.column_stack(cols)
        G[fold.season] = G.get(fold.season, 0.0) + X.T @ X
        h[fold.season] = h.get(fold.season, 0.0) + X.T @ fold.rshrunk
    keys = sorted(G)
    Gt, ht = sum(G.values()), sum(h.values())
    beta = np.linalg.lstsq(Gt, ht, rcond=None)[0]
    drops = np.array([np.linalg.lstsq(Gt - G[s], ht - h[s], rcond=None)[0] for s in keys])
    se = np.sqrt((len(keys) - 1) / len(keys) * ((drops - drops.mean(axis=0)) ** 2).sum(axis=0))
    rows, j = [], 0
    for side in SIDES:
        for t in groups:
            rows.append(dict(test=t["id"], side=side, stage="shrunk", version="joint", statistic="line", era="all",
                             estimate=float(beta[j]), se=float(se[j]), z=float(beta[j] / se[j]), seasons=len(keys)))
            j += 1
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------- the null: whole careers
def career_null(folds: list, tests: list, draws: int, seed: int = 20261008) -> dict:
    """Every decision test's line and bend estimates (both sides) for `draws` reassignments of whole careers: each
    player takes a donor's inputs, season by season along the donor's career.  Returns {"line": draws x sides x
    tests, "bend": the same, "jack_max": the largest season-jackknife |z| of each draw, "ids": the tests}.

    The season jackknife treats seasons as independent, but a player's misfit carries over from season to season, so
    an input fixed per player reaches |z| near 4 by chance in one test of twenty: each test is therefore scaled by
    its own spread under this null (the studentized max-T of Westfall and Young), not by its jackknife."""
    tests = [t for t in tests if decision_test(t)]
    nt = len(tests)
    # one reference fold per season: every test's normal score for every rated player
    ref, ref_rows = {}, {}
    for fold in folds:
        key = (fold.setting, fold.season)
        if key in ref:
            continue
        for side in SIDES:
            rows = fold.rows[side]
            ref[(key, side)] = normal_score(axis_scores(rows, fold.weights[side], tests))
            ref_rows[(key, side)] = rows.player_id.to_numpy()
        ref[key] = True
    players = np.unique(np.concatenate([v for v in ref_rows.values()]))
    pidx = {p: i for i, p in enumerate(players)}
    seasons_of = {}
    for (key, side), ids in ref_rows.items():
        if side == "O":
            for p in ids:
                seasons_of.setdefault(pidx[p], []).append(key)
    for i in seasons_of:
        seasons_of[i] = sorted(seasons_of[i], key=lambda k: k[1])
    # where each (season key, side) reference row of a player sits
    row_of = {(key, side): {pidx[p]: r for r, p in enumerate(ids)} for (key, side), ids in ref_rows.items()}
    rng = np.random.default_rng(seed)
    maxima = np.empty(draws)
    est_line, est_bend = np.empty((draws, len(SIDES), nt)), np.empty((draws, len(SIDES), nt))
    have = np.array(sorted(seasons_of))
    t0 = time.time()
    for d in range(draws):
        donor = dict(zip(have, rng.permutation(have)))
        sums = {}
        cache = {}
        for fold in folds:
            key = (fold.setting, fold.season)
            for side in SIDES:
                if (key, side) not in cache:
                    ids = ref_rows[(key, side)]
                    block = np.zeros((len(ids), nt))
                    for r_, p in enumerate(ids):
                        i = pidx[p]
                        k = seasons_of[i].index(key) if key in seasons_of[i] else 0
                        dn = donor.get(i, i)
                        dk = seasons_of[dn][min(k, len(seasons_of[dn]) - 1)]
                        rr = row_of[(dk, side)].get(dn)
                        if rr is not None:
                            block[r_] = ref[(dk, side)][rr]
                    cache[(key, side)] = (dict(zip(ids, range(len(ids)))), block)
                where, block = cache[(key, side)]
                pos = np.array([where.get(p, -1) for p in fold.rows[side].player_id.to_numpy()])
                C = np.where(pos[:, None] >= 0, block[np.maximum(pos, 0)], 0.0)
                w = fold.weights[side]
                C1 = centre_columns(C, w)
                C2 = centre_columns(C ** 2, w)
                HC = (fold.H[side] @ np.hstack([C1, C2]).astype(np.float32)).astype(float)
                g = fold.g[side]["shrunk"]
                parts = np.stack([np.einsum("ij,ij->j", C1, HC[:, :nt]), np.einsum("ij,ij->j", C1, HC[:, nt:]),
                                  np.einsum("ij,ij->j", C2, HC[:, nt:]), SIGN[side] * (C1.T @ g),
                                  SIGN[side] * (C2.T @ g)])
                sk = (fold.season, side)
                sums[sk] = sums.get(sk, 0.0) + parts
        zmax = 0.0
        for si, side in enumerate(SIDES):
            keys = sorted(s for (s, sd) in sums if sd == side)
            S = np.array([sums[(s, side)] for s in keys])
            tot = S.sum(axis=0)
            m = len(keys)
            line = tot[3] / tot[0]
            line_d = (tot[3] - S[:, 3]) / (tot[0] - S[:, 0])
            det = lambda P: P[0] * P[2] - P[1] ** 2                                # noqa: E731
            bend = (tot[0] * tot[4] - tot[1] * tot[3]) / det(tot)
            Pd = tot[None] - S
            bend_d = (Pd[:, 0] * Pd[:, 4] - Pd[:, 1] * Pd[:, 3]) / (Pd[:, 0] * Pd[:, 2] - Pd[:, 1] ** 2)
            est_line[d, si], est_bend[d, si] = line, bend
            for est, dr in ((line, line_d), (bend, bend_d)):
                se = np.sqrt((m - 1) / m * ((dr - dr.mean(axis=0)) ** 2).sum(axis=0))
                zmax = max(zmax, float(np.nanmax(np.abs(est / se))))
        maxima[d] = zmax
        if (d + 1) % 100 == 0:
            print(f"    career null: {d + 1} of {draws} draws ({time.time() - t0:.0f}s)", flush=True)
    return {"line": est_line, "bend": est_bend, "jack_max": maxima, "ids": [t["id"] for t in tests]}


def studentized_threshold(null: dict, level: float = 0.95) -> tuple:
    """Each test's null centre and spread, and the `level` quantile of the largest standardized |estimate| over all
    tests, sides and both statistics in each draw."""
    centre, spread, zs = {}, {}, []
    for kind in ("line", "bend"):
        est = null[kind]                                             # draws x sides x tests
        mu, sd = est.mean(axis=0), est.std(axis=0, ddof=1)
        centre[kind], spread[kind] = mu, sd
        zs.append(np.abs((est - mu[None]) / sd[None]).reshape(len(est), -1))
    maxima = np.concatenate(zs, axis=1).max(axis=1)
    return float(np.quantile(maxima, level)), centre, spread, maxima


# ------------------------------------------------------------------------------------- each group's share
def models_for(season: int) -> str:
    """The incumbent's saved prior models for a season (62 --save_models, one file per ten-season chunk)."""
    return ("lgb_noonc_early_1997" if season <= 2006 else "lgb_noonc_early_2007" if season <= 2016
            else "lgb_noonc_within")


def group_shapley_shares(seasons: list, groups: dict, out: Path, orderings: int = 32, background: int = 30,
                         seed: int = 0, threads: int = 4) -> pd.DataFrame:
    """Each group's share of the prior: Shapley values with every group of correlated inputs as ONE player (the
    audit's 0.5-cut groups; an input in no group stands alone), interventional against a background of the season's
    own rated rows, by sampled orderings of the groups, each coalition's prediction made in one batch.  A group's share
    is its mean |value| over the rated rows over the sum of all groups'.  LightGBM cannot give SHAP values for linear
    leaves (4.7.0: "not implemented for linear trees"), and shap's own model-agnostic Partition explainer took 35
    minutes a season and side; this is the same group-level quantity in a few minutes (linear leaves predict slowly,
    so `threads` cores; run it only when no build is running)."""
    from eracoef import singleyear as sy
    panel, _ = drop_untrainable(pd.read_parquet(ROOT / "outputs" / "role_panel_season.parquet"),
                                load_config(ROOT / "config.yaml"), what="the season panel")
    loaded, rows_out = {}, []
    rng = np.random.default_rng(seed)
    for season in seasons:
        name = models_for(season)
        if name not in loaded:
            loaded[name] = pd.read_pickle(ROOT / "outputs" / f"prior_models_{name}.pkl")
        for side in SIDES:
            s = loaded[name][season][side]
            feats, model_feats = s["feats"], s["model_feats"]
            held = sy.season_frame(panel[(panel.side == side) & (panel.season == season)], feats)
            held = held.assign(chunk_poss=held.poss.to_numpy(float), chunk_seasons=1.0)
            X = held[model_feats].to_numpy(float)
            n, p = X.shape
            member_of = {}
            for g in groups["sides"][side]["groups"]:
                if 0.5 in g["cuts"]:
                    for m in g["members"]:
                        member_of.setdefault(m, group_name(g["members"]))
            labels = [member_of.get(f, plain(f)) for f in model_feats]
            names = list(dict.fromkeys(labels))
            cols = [np.array([j for j, lab in enumerate(labels) if lab == nm]) for nm in names]
            B = X[rng.choice(n, size=min(background, n), replace=False)]
            phi = np.zeros((n, len(names)))
            for _ in range(orderings):
                current = np.repeat(B[None], n, axis=0)
                prev = s["full"].predict(current.reshape(-1, p), num_threads=threads).reshape(n, len(B)).mean(axis=1)
                for g in rng.permutation(len(names)):
                    current[:, :, cols[g]] = X[:, None, cols[g]]
                    val = s["full"].predict(current.reshape(-1, p), num_threads=threads).reshape(n, len(B)).mean(axis=1)
                    phi[:, g] += val - prev
                    prev = val
            phi /= orderings
            size = np.abs(phi).mean(axis=0)
            for nm, v, sz in zip(names, size / size.sum(), size):
                rows_out.append(dict(season=season, side=side, group=nm, share=float(v), mean_abs=float(sz)))
            top = np.argsort(-size)[:3]
            print(f"  group shares {season} {side}: " + "; ".join(f"{names[i]} {size[i] / size.sum():.2f}"
                                                                   for i in top), flush=True)
    df = pd.DataFrame(rows_out)
    df.to_csv(out, index=False)
    return df


# ---------------------------------------------------------------------------------------- the trade set (a veto)
def tradeset_leans(tests: list, alpha_path: Path) -> pd.DataFrame:
    """Each decision test's straight-line lean of the neighbouring-season miss (the trade set's alpha on actual points,
    eligible rows) on the input's within-season normal score, standard errors clustered by player.  It never fits
    anything: a lean the season's own games find and this contradicts at |z| 2 does not count."""
    from eracoef import singleyear as sy
    alpha = pd.read_parquet(alpha_path)
    alpha = alpha[alpha.eligible]
    # gated like every reader of the panel: a season still being played never enters a reading
    panel, _ = drop_untrainable(pd.read_parquet(ROOT / "outputs" / "role_panel_season.parquet"),
                                load_config(ROOT / "config.yaml"), what="the season panel")
    codes = sorted({c for t in tests if decision_test(t)
                    for c in ([t["code"]] if t["kind"] == "input" else t["members"])})
    rows = []
    for side, name in (("O", "offense"), ("D", "defense")):
        frame = sy.season_frame(panel[panel.side == side], codes)
        frame = frame.merge(alpha[alpha.side == name][["player_id", "season", "alpha_good"]],
                            on=["player_id", "season"], how="inner").reset_index(drop=True)
        w = np.maximum(frame["poss"].to_numpy(float), 1e-9)          # the panel's possessions on this side
        per_season = list(frame.groupby("season").indices.values())
        member = {}
        for t in tests:
            if not decision_test(t):
                continue
            p = np.empty(len(frame))
            if t["kind"] == "input":
                x = frame[t["code"]].to_numpy(float)
                for idx in per_season:
                    p[idx] = percentile(x[idx], w[idx])
            else:
                for m in t["members"]:
                    if m not in member:
                        x, member[m] = frame[m].to_numpy(float), np.empty(len(frame))
                        for idx in per_season:
                            member[m][idx] = normal_score(percentile(x[idx], w[idx]))
                combo = sum(a * member[m] for m, a in zip(t["members"], t["loadings"]))
                for idx in per_season:
                    p[idx] = percentile(combo[idx], w[idx])
            z = normal_score(p)
            y = frame.alpha_good.to_numpy(float)
            zc, yc = z - z.mean(), y - y.mean()
            slope = float(zc @ yc / (zc @ zc))
            score = pd.Series(zc * (yc - slope * zc)).groupby(frame.player_id.to_numpy()).sum().to_numpy()
            se = float(np.sqrt((score ** 2).sum()) / (zc @ zc))
            rows.append(dict(test=t["id"], side=side, tradeset_line=slope, tradeset_z=slope / se))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------ the paired mode
def audit_setting(tags: list, mult_tag: str, setting: str, tests: list, seasons=None) -> dict:
    """Load a fold folder set with its own multipliers and read every test on it."""
    if seasons is None:
        seasons = sorted({int(p.stem.split("_")[1]) for tag in tags
                          for p in (ROOT / "outputs" / "within" / tag).glob("fold_*.json")})
    multipliers, mult_folds = shipped_multipliers(seasons, mult_tag)
    folds = load_setting(tags, setting, multipliers)
    by_era = len({era_of(f.season) for f in folds}) > 1
    readings, tenths, dropped = run_tests(folds, tests, by_era)
    est = {(r.test, r.side, r.stage, r.version, r.statistic, r.era): r.estimate for r in readings.itertuples()}
    return dict(folds=folds, readings=readings, tenths=tenths, drops=dropped, est=est, by_era=by_era,
                pooled=C99.fit_prior_multipliers(mult_folds), mult_tag=mult_tag, tags=tags)


def paired(cand: dict, ref: dict, tests: list) -> tuple:
    """A candidate against a reference on identical test games: the change in every lean (a jackknife over seasons of
    the paired difference), how far the ratings themselves moved along each axis, and the paired held-out error."""
    by_stem = {f.stem: f for f in ref["folds"]}
    common = [f for f in cand["folds"] if f.stem in by_stem]
    assert common, "the two folders share no fold"
    for f in common:
        assert np.array_equal(f.game_idx, by_stem[f.stem].game_idx), f"{f.stem}: the two folders' test games differ"
    rows = []
    for key, (keys_c, drops_c) in cand["drops"].items():
        tid, side, st, version, kind, era = key
        if version != "pinned" or era != "all" or key not in ref["drops"]:
            continue
        keys_r, drops_r = ref["drops"][key]
        diff = cand["est"][key] - ref["est"][key]
        se = float("nan")
        if drops_c is not None and drops_r is not None and list(keys_c) == list(keys_r):
            dd = drops_c - drops_r
            se = float(np.sqrt((len(dd) - 1) / len(dd) * ((dd - dd.mean()) ** 2).sum()))
        rows.append(dict(test=tid, side=side, stage=st, statistic=kind, reference=ref["est"][key],
                         candidate=cand["est"][key], change=diff, change_se=se,
                         change_z=diff / se if se > 0 else float("nan")))
    leans = pd.DataFrame(rows)
    # how far the ratings moved along each axis: the candidate's shrunk rating minus the reference's, same players
    acc = {}
    for f in common:
        g = by_stem[f.stem]
        for side in SIDES:
            ids = g.rows[side].player_id.to_numpy()
            rc = pd.Series(f.shrunk_rating[side], index=f.rows[side].player_id.to_numpy()).reindex(ids).to_numpy()
            delta = rc - g.shrunk_rating[side]
            ok = np.isfinite(delta)
            w = g.weights[side][ok]
            P = axis_scores(g.rows[side], g.weights[side], tests)[ok]
            Pt = axis_scores(g.rows[side], g.weights[side], tests, broken=True)[ok]
            d = delta[ok]
            for j, t in enumerate(tests):
                z = normal_score(P[:, j])
                zc, dc = z - np.average(z, weights=w), d - np.average(d, weights=w)
                top, bot = Pt[:, j] >= 0.9, Pt[:, j] < 0.1
                acc.setdefault((t["id"], side), []).append(
                    (float(np.sum(w * zc * dc) / np.sum(w * zc * zc)),
                     float(np.average(d[top], weights=w[top]) - np.average(d[bot], weights=w[bot]))))
    moves = pd.DataFrame([dict(test=k[0], side=k[1], move_line=float(np.mean([v[0] for v in vals])),
                               move_tenths=float(np.mean([v[1] for v in vals]))) for k, vals in acc.items()])
    # the paired held-out error per fold at the decision stage (team-game error, level and home edge refitted)
    err = pd.DataFrame([dict(stem=f.stem, season=f.season,
                             candidate=float(np.sum(f.rshrunk ** 2) / f.total_w),
                             reference=float(np.sum(by_stem[f.stem].rshrunk ** 2) / by_stem[f.stem].total_w))
                        for f in common])
    err["change"] = err.candidate - err.reference
    per = err.groupby("season").change.mean()
    drops = np.array([per.drop(s_).mean() for s_ in per.index])
    se = float(np.sqrt((len(per) - 1) / len(per) * ((drops - drops.mean()) ** 2).sum()))
    error = dict(change=float(per.mean()), se=se, z=float(per.mean() / se) if se > 0 else float("nan"),
                 better_seasons=int((per < 0).sum()), seasons=int(len(per)), folds=int(len(err)))
    return leans, moves, error


# --------------------------------------------------------------------------------------------------- the run
def parse_folder(spec: str) -> tuple:
    """`tag[:mult_tag]`: a fold folder and the folder its multipliers are fitted on (default: itself)."""
    tag, _, mult = spec.partition(":")
    return tag, (mult or tag)


def null_from_file(name: str) -> tuple:
    """An earlier audit's threshold and each test's null centre and spread (a candidate is judged on the frozen
    yardstick, not on a null drawn again)."""
    saved = json.loads((ROOT / "outputs" / f"{name}_thresholds.json").read_text())
    table = pd.read_csv(ROOT / "outputs" / "csv" / f"{name}_null.csv")
    ids = list(dict.fromkeys(table.test))
    centre = {k: np.full((len(SIDES), len(ids)), np.nan) for k in ("line", "bend")}
    spread = {k: np.full((len(SIDES), len(ids)), np.nan) for k in ("line", "bend")}
    for r in table.itertuples():
        j, si = ids.index(r.test), SIDES.index(r.side)
        centre[r.statistic][si, j], spread[r.statistic][si, j] = r.centre, r.spread
    return float(saved["threshold"]), centre, spread, ids


def main() -> None:
    check_flags()
    random_tags = [t for t in flag("random", "lgb_noonc_within").split(",") if t]
    mult_tag = flag("mult_tag", "lgb_noonc_within")
    deadline_tags = [t for t in flag("deadline", "").split(",") if t]
    reference = flag("reference", "")
    control = flag("control", "")
    out_name = flag("out", "audit_lgb_noonc")
    draws = int(flag("draws", 400))
    threshold_from = flag("threshold_from", "")
    pts_alpha = ROOT / "outputs" / "tradeset_lgb_noonc_rs_pts_alpha.parquet"
    tradeset_path = flag("tradeset", "outputs/tradeset_lgb_noonc_rs_pts_alpha.parquet" if pts_alpha.exists() else "")
    share_seasons = [int(s) for s in flag("shares", "").split(",") if s]
    t0 = time.time()

    if flag("freeze", "0") not in ("0", "no", "false"):
        assert not GROUPS_FILE.exists(), f"{GROUPS_FILE.relative_to(ROOT)} is frozen; delete it to refreeze"
        groups = freeze_groups(ROOT / "outputs" / "within" / random_tags[0])
        GROUPS_FILE.write_text(json.dumps(groups, indent=1))
        print(f"froze {GROUPS_FILE.relative_to(ROOT)}: " + "; ".join(
            f"{s} {len(g['groups'])} groups over {g['rows']} rows" for s, g in groups["sides"].items()), flush=True)
    groups = json.loads(GROUPS_FILE.read_text())
    tests = build_tests(groups)

    main_set = audit_setting(random_tags, mult_tag, "random", tests)
    folds, by_era = main_set["folds"], main_set["by_era"]
    print(f"99's multipliers on {mult_tag} 2017-2026, all folds pooled: offence {main_set['pooled'][0]:.4f}, defence "
          f"{main_set['pooled'][1]:.4f}", flush=True)
    print(f"{len(folds)} random folds from {random_tags}, seasons {min(f.season for f in folds)}-"
          f"{max(f.season for f in folds)}; {len(tests)} tests ({sum(t['kind'] == 'group' for t in tests)} groups) "
          f"({time.time() - t0:.0f}s)", flush=True)
    parts = [main_set["readings"].assign(setting="random"), run_joint(folds, tests).assign(setting="random")]
    if deadline_tags:
        dl = audit_setting(deadline_tags, mult_tag, "deadline", tests)
        r = dl["readings"]
        parts.append(r[(r.stage == "shrunk") & (r.version == "pinned")].assign(setting="deadline"))
        print(f"{len(dl['folds'])} deadline folds read ({time.time() - t0:.0f}s)", flush=True)
        del dl
    allr = pd.concat(parts, ignore_index=True)

    if draws > 0:
        print(f"the career null ({draws} draws)", flush=True)
        null = career_null(folds, tests, draws)
        thr, null_centre, null_spread, maxima = studentized_threshold(null)
        null_ids = null["ids"]
    else:
        assert threshold_from, "--draws=0 needs --threshold_from=<an earlier audit's --out>"
        thr, null_centre, null_spread, null_ids = null_from_file(threshold_from)
        null, maxima = None, np.array([thr])
        print(f"threshold {thr:.2f} and each test's null spread read from {threshold_from} (no new null)", flush=True)
    null_pos = {tid: j for j, tid in enumerate(null_ids)}
    tradeset = tradeset_leans(tests, ROOT / tradeset_path).set_index(["test", "side"]) if tradeset_path else None

    pick = lambda st, v, kind, era="all", setting="random": allr[  # noqa: E731
        (allr.stage == st) & (allr.version == v) & (allr.statistic == kind) & (allr.era == era)
        & (allr.setting == setting)].set_index(["test", "side"])
    line, bend = pick("shrunk", "pinned", "line"), pick("shrunk", "pinned", "bend")
    ten = pick("shrunk", "pinned", "tenths")
    pf, rf = pick("shrunk", "prior_free", "line"), pick("shrunk", "rating_free", "line")
    nu = pick("shrunk", "nuisance", "line")
    jt = allr[allr.version == "joint"].set_index(["test", "side"])
    prior_ten, raw_ten = pick("prior", "pinned", "tenths"), pick("raw", "pinned", "tenths")
    has_swap = bool((allr.stage == "swap").any())
    s106_ten = pick("shrunk106", "pinned", "tenths") if has_swap else None
    swap_ten = pick("swap", "pinned", "tenths") if has_swap else None
    era_line = {f"{a}-{b}": pick("shrunk", "pinned", "line", era=f"{a}-{b}") for a, b in ERAS} if by_era else {}
    era_line = {k: v for k, v in era_line.items() if len(v)}
    dl_line = pick("shrunk", "pinned", "line", setting="deadline") if deadline_tags else None
    rows = []
    for t in tests:
        for side in SIDES:
            k = (t["id"], side)
            l, b, n = line.loc[k], bend.loc[k], ten.loc[k]
            same = bool(l.estimate != 0 and all(np.sign(x) == np.sign(l.estimate) and 0.5 <= x / l.estimate <= 2.0
                                                for x in (pf.loc[k].estimate, rf.loc[k].estimate)))
            if t["id"] in null_pos:
                j, si = null_pos[t["id"]], SIDES.index(side)
                nz_line = (l.estimate - null_centre["line"][si, j]) / null_spread["line"][si, j]
                nz_bend = (b.estimate - null_centre["bend"][si, j]) / null_spread["bend"][si, j]
                spread_line = float(null_spread["line"][si, j])
            else:
                nz_line = nz_bend = spread_line = float("nan")
            passes = bool(np.isfinite(nz_line) and max(abs(nz_line), abs(nz_bend)) >= thr)
            era_ok = (all(np.sign(v.loc[k].estimate) == np.sign(l.estimate) for v in era_line.values())
                      if era_line else None)
            # a veto that cannot be read (no spread: the trade set's eligible players are one-team players) vetoes nothing
            dl_ok = (bool(dl_line.loc[k].z * np.sign(l.estimate) > -2)
                     if dl_line is not None and np.isfinite(dl_line.loc[k].z) else None)
            in_ts = tradeset is not None and k in tradeset.index and np.isfinite(tradeset.loc[k].tradeset_z)
            ts_ok = bool(tradeset.loc[k].tradeset_z * np.sign(l.estimate) > -2) if in_ts else None
            counts = bool(decision_test(t) and passes and abs(n.estimate) >= 0.1 and same
                          and era_ok in (None, True) and dl_ok in (None, True) and ts_ok in (None, True))
            rows.append(dict(
                test=t["id"], kind=t["kind"], cls=t["cls"], name=t["name"], side=side,
                tenths_bottom_to_top=n.estimate, tenths_se=n.se, line_per_sd=l.estimate, line_null_sd=spread_line,
                line_z=nz_line, bend=b.estimate, bend_z=nz_bend, line_jackknife_z=l.z, bend_jackknife_z=b.z,
                smallest_detectable_line=thr * spread_line,
                joint_line=jt.loc[k].estimate if k in jt.index else np.nan,
                joint_z=jt.loc[k].z if k in jt.index else np.nan,
                prior_free_line=pf.loc[k].estimate, rating_free_line=rf.loc[k].estimate,
                nuisance_line=nu.loc[k].estimate, nuisance_z=nu.loc[k].z,
                prior_stage_tenths=prior_ten.loc[k].estimate, raw_stage_tenths=raw_ten.loc[k].estimate,
                # the swap step's own lean: 106's swapped ratings against 106's shrunk ones (the same multipliers)
                swap_stage_tenths=swap_ten.loc[k].estimate if has_swap else np.nan,
                swap_step_tenths=(swap_ten.loc[k].estimate - s106_ten.loc[k].estimate) if has_swap else np.nan,
                **{f"line_{e}": v.loc[k].estimate for e, v in era_line.items()},
                deadline_line_z=dl_line.loc[k].z if dl_line is not None else np.nan,
                tradeset_line=tradeset.loc[k].tradeset_line if in_ts else np.nan,
                tradeset_z=tradeset.loc[k].tradeset_z if in_ts else np.nan,
                passes_threshold=passes, at_least_0_1=bool(abs(n.estimate) >= 0.1), same_sign_all_scales=same,
                same_sign_every_era=era_ok, deadline_not_against=dl_ok, tradeset_not_against=ts_ok, counts=counts,
                # information, not part of the rule: the line loses its sign or half its size once playing time
                # (log possessions, the bench's share of the team-game) is held -- the closed playing-time topic
                playing_time_explains=bool(np.sign(nu.loc[k].estimate) != np.sign(l.estimate)
                                           or abs(nu.loc[k].estimate) < 0.5 * abs(l.estimate))))
    # offence plus defence: does the lean survive on the total, or is it credit landing on the wrong side?
    for t in tests:
        k_o, k_d = (t["id"], "O"), (t["id"], "D")
        tot_line = line.loc[k_o].estimate + line.loc[k_d].estimate
        z_tot = float("nan")
        if t["id"] in null_pos and null is not None:
            draws_tot = null["line"][:, 0, null_pos[t["id"]]] + null["line"][:, 1, null_pos[t["id"]]]
            z_tot = (tot_line - draws_tot.mean()) / draws_tot.std(ddof=1)
        rows.append(dict(test=t["id"], kind=t["kind"], cls=t["cls"], name=t["name"], side="total",
                         tenths_bottom_to_top=ten.loc[k_o].estimate + ten.loc[k_d].estimate,
                         line_per_sd=tot_line, line_z=z_tot, counts=False))
    summary = pd.DataFrame(rows)

    (ROOT / "outputs" / "csv").mkdir(exist_ok=True)
    allr.to_csv(ROOT / "outputs" / "csv" / f"{out_name}.csv", index=False)
    main_set["tenths"].to_csv(ROOT / "outputs" / "csv" / f"{out_name}_tenths.csv", index=False)
    summary.to_csv(ROOT / "outputs" / "csv" / f"{out_name}_summary.csv", index=False)
    if null is not None:
        pd.DataFrame([dict(test=tid, side=side, statistic=kind, centre=float(null_centre[kind][si, j]),
                           spread=float(null_spread[kind][si, j]))
                      for j, tid in enumerate(null_ids) for si, side in enumerate(SIDES) for kind in ("line", "bend")]
                     ).to_csv(ROOT / "outputs" / "csv" / f"{out_name}_null.csv", index=False)
        (ROOT / "outputs" / f"{out_name}_thresholds.json").write_text(json.dumps(dict(
            threshold=thr, draws=draws,
            null_maxima_quantiles={q: float(np.quantile(maxima, q)) for q in (0.5, 0.9, 0.95, 0.99)},
            jackknife_z_maxima_quantiles={q: float(np.quantile(null["jack_max"], q)) for q in (0.5, 0.95)},
            decision_tests=sum(decision_test(t) for t in tests), random=random_tags, mult_tag=mult_tag,
            deadline=deadline_tags, tradeset=tradeset_path, pooled_multipliers=list(main_set["pooled"]),
            folds=len(folds), seasons=sorted({f.season for f in folds})), indent=1))

    print(f"\nthreshold |z| {thr:.2f}: the 95th percentile of the largest standardized estimate over "
          f"{2 * 2 * sum(decision_test(t) for t in tests)} line and bend tests, whole careers reassigned"
          + (f" ({draws} draws; the season jackknife's own z would have needed "
             f"{np.quantile(null['jack_max'], 0.95):.2f})" if null is not None else f" (from {threshold_from})")
          + f"; {int(summary.counts.sum())} leans count", flush=True)
    show = summary.reindex(summary.tenths_bottom_to_top.abs().sort_values(ascending=False).index)
    for side in SIDES:
        print(f"\n{'offence' if side == 'O' else 'defence'}: bottom tenth to top, points per 100 (+ = underrated); "
              f"line z; bend z; joint z")
        for r in show[show.side == side].head(30).itertuples():
            print(f"  {r.name[:56]:56s} {r.tenths_bottom_to_top:+.2f}  line z {r.line_z:+5.1f}  bend z "
                  f"{r.bend_z:+5.1f}  joint z {r.joint_z:+5.1f}  {r.cls[:12]:12s} {'COUNTS' if r.counts else ''}")
    print("\noffence plus defence: bottom tenth to top, points per 100; line z")
    for r in show[show.side == "total"].head(15).itertuples():
        print(f"  {r.name[:56]:56s} {r.tenths_bottom_to_top:+.2f}  line z {r.line_z:+5.1f}")

    # the paired mode: the candidate against the incumbent's folds, beside the noise control's own movement
    if reference:
        seasons = sorted({f.season for f in folds})
        ref_tag, ref_mult = parse_folder(reference)
        ref = audit_setting([ref_tag], ref_mult, "reference", tests, seasons=seasons)
        leans, moves, error = paired(main_set, ref, tests)
        out = leans.merge(moves, on=["test", "side"], how="left")
        if control:
            ctl_tag, ctl_mult = parse_folder(control)
            ctl = audit_setting([ctl_tag], ctl_mult, "control", tests, seasons=seasons)
            c_leans, c_moves, c_error = paired(ctl, ref, tests)
            out = out.merge(c_leans[["test", "side", "stage", "statistic", "change"]].rename(
                columns={"change": "control_change"}), on=["test", "side", "stage", "statistic"], how="left")
            out = out.merge(c_moves.rename(columns={"move_line": "control_move_line",
                                                    "move_tenths": "control_move_tenths"}),
                            on=["test", "side"], how="left")
            error["control_change"], error["control_z"] = c_error["change"], c_error["z"]
        names = {t["id"]: t["name"] for t in tests}
        out.insert(2, "name", out.test.map(names))
        out.to_csv(ROOT / "outputs" / "csv" / f"{out_name}_paired.csv", index=False)
        (ROOT / "outputs" / f"{out_name}_paired_error.json").write_text(json.dumps(error, indent=1))
        print(f"\npaired against {ref_tag}: held-out team-game error {error['change']:+.4f} (z {error['z']:+.1f}, "
              f"{error['better_seasons']} of {error['seasons']} seasons better)"
              + (f"; the noise control {error['control_change']:+.4f}" if control else ""), flush=True)
        focus = out[(out.stage == "shrunk") & (out.statistic == "tenths")]
        focus = focus.reindex(focus.change.abs().sort_values(ascending=False).index).head(20)
        for r in focus.itertuples():
            ctl_txt = f"  control {r.control_change:+.2f}, moved {r.control_move_tenths:+.2f}" if control else ""
            print(f"  {r.side} {r.name[:50]:50s} lean {r.reference:+.2f} -> {r.candidate:+.2f} ({r.change:+.2f}, "
                  f"z {r.change_z:+.1f}); ratings moved {r.move_tenths:+.2f}{ctl_txt}")
    if share_seasons:
        print("\neach group's share of the prior (group-level Shapley values)", flush=True)
        group_shapley_shares(share_seasons, groups, ROOT / "outputs" / "csv" / f"{out_name}_shares.csv")
    print(f"\ndone in {time.time() - t0:.0f}s: outputs/csv/{out_name}.csv, _summary.csv, _tenths.csv", flush=True)


if __name__ == "__main__":
    main()
