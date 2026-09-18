"""Fit the prior's replacement level against APM, as a model on covariates rather than one constant.

    python scripts/67_blend_apm.py [--priors=sy_base] [--first=1997] [--last=2026] [--boards=2015,2024]
                                   [--out=blend_apm] [--k_grid=14] [--fix_a=1] [--a_grid=8]
                                   [--covs=level|scalar|<comma list>] [--drop=height]

The owner's design (2026-09-15), the stage between the box prior and the final ridge:

    new_prior = old_prior * w(n) + x * (1 - w(n)),      w(n) = n^a / (n^a + k^a)

`n` is the player's OFFENSIVE possessions on offence and DEFENSIVE on defence.  Fitted with `x` as a
single number it broke: 2026 came back with twenty players below -10 points per 100 and one at -22.41 on
four possessions, where the unblended board has none below -6.  **A constant target sits entirely in the
one direction this loss cannot see.**  Profiling the context out takes the season intercept with it, and
adding a constant to all five offensive ratings on the floor IS the intercept column, so `x`'s level
floats and only its variation with possessions is pinned; the fitted `x` was a curve parameter, not a
bench level.  The real bench level is measurable and the box prior already sits on it, -1.55 offence and
-0.45 defence under a hundred possessions.

So `x` becomes `g(z_i) = z_i . beta` over `singleyear.LEVEL_COVARIATES` -- age, experience, career
possessions, entry age, height, weight, draft pick, possession share, start share, tenure, teams -- every
one of them measured EXACTLY however few minutes a man played, which is the point: a player with few possessions is handed
to the covariates that still work rather than to a floating constant.  This is not merely more flexible.
It removes the pathology, because `beta` is pinned by players who DO have minutes and a 155-possession man
borrows strength from players with heavy minutes who look like him.  `--covs=scalar` restores the one-constant
version exactly (`z` = the intercept alone), so the owner's original is a point in this family.

**Fitted against APM, never RAPM.**  A ridge penalty pulls every player toward zero, so the RAPM of a player with few possessions is
near zero by construction, and a blend fitted on it would return a level of zero -- "a man we know
nothing about is league average" -- which is the error being corrected.  The season's normal equations are
solved with NO penalty on either player block.

The loss, per rated season and per side's target:

    L(theta) = (b - apm)' G (b - apm)
    b_off    = s_off * (w_off * prior_off) + (1 - w_off) * (Z_off @ beta_off)     (and the mirror)

`G` is the player block's gram after the context block (season intercept, home, playoff, garbage time,
margin, margin-by-time) is profiled out, and `apm = G+ r` is the unpenalised least-squares solution.  `L`
is identical up to a constant to least squares on the season's own stints with the context refit freely,
which is why it needs no inversion of a forty-possession player's information.

**`b` is linear in `theta`**, so given `(k, a)` the whole thing is one solve: the basis carries one column
`w * prior` per side plus one column `(1 - w) * Z[:, j]` per covariate, and the minimum is
`apm'G apm - theta' V'G apm`.  The grid over `(k, a)` is swept exhaustively on both sides at once.

**Every fit carries its own free prior scale `s`.**  Without it the fit runs away: as `k` falls and `a`
flattens, `1 - w` becomes constant, `x (1 - w)` becomes free and `w * prior` becomes a plain RESCALE of
the prior, so `k` pinned at the bottom of the grid in every season with `|x|` at 40 to 64.  The ridge
downstream already has a free scale and would absorb exactly that, so the reference this has to beat is
`s * prior` alone -- the scale by itself, nothing else.

**One season cannot price this; thirty can.**  Per season the parameters wander (the scalar version read
`x_off` sd 11.95 on a mean of -16.30).  Pooled over 29 they are stable, so the deliverable is the
leave-one-season-out fit: fitted on every other season, scored on the one it never saw.

**The loss is blind at the bottom.**  Each player's weight is his own information, `G_ii`, so a
four-possession man contributes almost nothing and the runaway cost the loss nothing.  `beta` is
identified by players with heavy minutes and APPLIED to those with few possessions; that is legitimate borrowing of strength
and it is also why the bottom of the list has to be read separately, by hand, every time.

Reads the priors `scripts/62_single_year_board.py --save_priors=<name>` left in outputs/priors_<name>.pkl,
so no booster is refitted.  Writes outputs/<out>.parquet, outputs/<out>_heldout.parquet and
outputs/<out>_coefs.parquet (the pooled coefficients, which is what the board consumes).
"""
import os
import sys
import time
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
os.environ.setdefault("NUMBA_NUM_THREADS", os.environ.get("OPENRAPM_NUMBA_THREADS", "4"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eracoef import singleyear as sy  # noqa: E402
from eracoef.config import load_config
from eracoef.seasons import drop_untrainable  # noqa: E402
from eracoef.holdout import Context  # noqa: E402
from eracoef.priorridge import PriorRidgeCV  # noqa: E402
from eracoef.xshoot import DEFENSE_TARGETS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

# printed at the end so the fitted level is legible as players rather than as coefficients
PROFILES = {
    "19-year-old undrafted rookie, deep bench": dict(age=19, exp_yrs=0, exp_poss=0.0, entry_age=19,
                                                     height=78, weight=210, draft_pick=61.0,
                                                     poss_pct=0.02, gs_pct=0.0, tenure=1, n_teams=1),
    "21-year-old lottery pick, deep bench": dict(age=21, exp_yrs=0, exp_poss=0.0, entry_age=21,
                                                 height=79, weight=215, draft_pick=5.0,
                                                 poss_pct=0.02, gs_pct=0.0, tenure=1, n_teams=1),
    "27-year-old journeyman, deep bench": dict(age=27, exp_yrs=5, exp_poss=12.0, entry_age=22,
                                               height=78, weight=215, draft_pick=45.0,
                                               poss_pct=0.03, gs_pct=0.0, tenure=1, n_teams=2),
    "36-year-old veteran, hurt, had been a starter": dict(age=36, exp_yrs=15, exp_poss=180.0,
                                                          entry_age=21, height=77, weight=205,
                                                          draft_pick=7.0, poss_pct=0.20, gs_pct=0.60,
                                                          tenure=3, n_teams=1),
    "26-year-old starter, full season": dict(age=26, exp_yrs=4, exp_poss=60.0, entry_age=22,
                                             height=79, weight=220, draft_pick=15.0,
                                             poss_pct=0.45, gs_pct=0.95, tenure=3, n_teams=1),
}


# The shared command line (scripts/_cli.py): `flag` records every name it is asked for so
# `check_flags` below can refuse one that was never asked for.  A misspelled flag used to be
# ignored silently, which is how a run looks right and is wrong.
from _cli import check_flags, flag as _flag, switch   # noqa: E402


def _target(name):
    return DEFENSE_TARGETS[name] if name in DEFENSE_TARGETS else name


def blend_weight(n, k, a):
    """w = n^a / (n^a + k^a), the owner's curve with a steepness.  a = 1 is n / (n + k) exactly.

    In logs, so a 6,000-possession player at a = 4 does not overflow; n = 0 gives w = 0.
    """
    n = np.asarray(n, dtype=float)
    with np.errstate(divide="ignore"):
        ratio = a * (np.log(np.maximum(n, 0.0)) - np.log(k))
    return 1.0 / (1.0 + np.exp(-ratio))


def season_pieces(design, prior_o, prior_d):
    """The season's APM, the profiled player gram, the two priors, and the per-side possessions.

    `G` is the player block's gram with the context block profiled out -- a Schur complement, so still
    positive semi-definite -- and `apm` minimises `(b - apm)' G (b - apm)`.
    """
    ridge = PriorRidgeCV(free_prior_scale=False)
    player_ids, players, context, zero = ridge._parts(design, None, None)
    n_players = design.spec.n_ps
    gram, rhs = ridge._blocks(players, context, zero, design.y, design.w, np.arange(design.y.size))

    p, c = slice(0, 2 * n_players), slice(2 * n_players, gram.shape[0])
    g_pp, g_pc, g_cc = gram[p, p], gram[p, c], gram[c, c]
    g_cc_inv = np.linalg.pinv(g_cc)
    G = g_pp - g_pc @ g_cc_inv @ g_pc.T
    r = rhs[p] - g_pc @ g_cc_inv @ rhs[c]

    # `G` is singular BY CONSTRUCTION: profiling the context out takes the season intercept with it, and
    # "add a constant to every offensive rating" puts five times that constant on every row, which IS the
    # intercept column.  Both all-ones directions are exactly null -- the objective cannot see the level
    # of either side, only its shape against possessions.  `np.linalg.solve` does not RAISE on that, it
    # returns a huge vector, so `priorridge.solve_diag`'s lstsq fallback never fires: it gave |apm| = 4e20
    # in 2018, which has two further null directions of its own, and that season swamped every pooled fit.
    # The minimum-norm solution is the right representative: every direction it drops is one L is flat on.
    eigenvalue, vector = np.linalg.eigh(G)
    keep = eigenvalue > 1e-9 * eigenvalue.max()
    apm = vector[:, keep] @ ((vector[:, keep].T @ r) / eigenvalue[keep])
    rank_note = float(np.linalg.norm(G @ apm - r) / max(np.linalg.norm(r), 1e-9))

    # the residual variance of the season's own stints at the OLS optimum, for the coefficient errors
    ss_y = float((design.w * design.y ** 2).sum())
    ss_resid = ss_y - float(rhs[c] @ g_cc_inv @ rhs[c]) - float(apm @ r)
    dof = max(design.y.size - int(keep.sum()) - g_cc.shape[0], 1)

    prior = np.concatenate([PriorRidgeCV._align(prior_o, player_ids, n_players),
                            PriorRidgeCV._align(prior_d, player_ids, n_players)])
    poss = (design.game_poss.groupby("psx_idx")[["poss_off", "poss_def"]].sum()
            .reindex(range(design.spec.n_psx)).fillna(0.0))
    ps_of_psx = (design.spec.ps_of_psx if design.spec.ps_of_psx is not None
                 else np.arange(design.spec.n_psx))
    n_off = np.bincount(ps_of_psx, weights=poss.poss_off.to_numpy(), minlength=n_players)
    n_def = np.bincount(ps_of_psx, weights=poss.poss_def.to_numpy(), minlength=n_players)
    return dict(G=G, apm=apm, prior=prior, n_off=n_off, n_def=n_def, player_ids=player_ids,
                n_players=n_players, rank_note=rank_note, null_extra=int((~keep).sum()),
                ss_resid=ss_resid, dof=dof)


def grid_tables(G, apm, prior, n_off, n_def, n_players, Z_off, Z_def, ks, alphas):
    """Per pair of settings, the normal equations of the blend as a linear model in `1 + p` per side.

    The free numbers are `[s_off, beta_off (p), s_def, beta_def (p)]`.  `s` is there because without it
    the fit runs away into the corner where `1 - w` is constant, which is a plain rescale of the prior --
    something `PriorRidgeCV.free_prior_scale` already does and would absorb downstream.
    """
    n_set = len(ks) * len(alphas)
    settings = [(float(k), float(a)) for k in ks for a in alphas]
    twice, p = 2 * n_players, Z_off.shape[1]
    off, dfe = slice(0, n_players), slice(n_players, twice)
    width = 1 + p

    Vo, Vd = np.zeros((n_set, width, twice)), np.zeros((n_set, width, twice))
    for i, (k, a) in enumerate(settings):
        w_o, w_d = blend_weight(n_off, k, a), blend_weight(n_def, k, a)
        Vo[i, 0, off] = w_o * prior[off]
        Vo[i, 1:, off] = ((1.0 - w_o)[:, None] * Z_off).T
        Vd[i, 0, dfe] = w_d * prior[dfe]
        Vd[i, 1:, dfe] = ((1.0 - w_d)[:, None] * Z_def).T

    GVo, GVd = Vo @ G, Vd @ G          # G is symmetric, so this is G applied to each basis vector
    # Put every basis column on unit G-norm before the system is formed.  Unnormalised, the scaled-prior
    # column and the level's carriers differ by orders of magnitude and the corner where w -> 0 (the blend
    # collapsing to a bare constant, which G cannot see at all) came back as the argmin with |theta| = 1e5
    # of pure rounding error.  The norms are kept so the answer can be reported in the owner's units.
    norm_o = np.sqrt(np.maximum(np.einsum("iax,iax->ia", Vo, GVo), 0.0))
    norm_d = np.sqrt(np.maximum(np.einsum("jax,jax->ja", Vd, GVd), 0.0))
    floor_o = 1e-8 * max(norm_o.max(), 1e-30)
    floor_d = 1e-8 * max(norm_d.max(), 1e-30)
    ok_o, ok_d = (norm_o > floor_o).all(axis=1), (norm_d > floor_d).all(axis=1)
    Vo = Vo / np.where(norm_o > floor_o, norm_o, 1.0)[:, :, None]
    Vd = Vd / np.where(norm_d > floor_d, norm_d, 1.0)[:, :, None]
    GVo, GVd = Vo @ G, Vd @ G
    g_apm = G @ apm

    M = np.zeros((n_set, n_set, 2 * width, 2 * width))
    M[:, :, :width, :width] = np.einsum("iax,ibx->iab", Vo, GVo)[:, None, :, :]
    M[:, :, width:, width:] = np.einsum("jax,jbx->jab", Vd, GVd)[None, :, :, :]
    cross = np.einsum("iax,jbx->ijab", Vo, GVd)
    M[:, :, :width, width:] = cross
    M[:, :, width:, :width] = np.swapaxes(cross, 2, 3)
    r = np.zeros((n_set, n_set, 2 * width))
    r[:, :, :width] = (Vo @ g_apm)[:, None, :]
    r[:, :, width:] = (Vd @ g_apm)[None, :, :]
    return dict(settings=settings, M=M, r=r, quad=float(apm @ g_apm), width=width,
                norm_o=norm_o, norm_d=norm_d, ok=ok_o[:, None] & ok_d[None, :])


def solve_blend(M, r, quad=None, ok=None):
    """argmin over theta of `theta' M theta - 2 theta' r`, and `r' M+ r`.

    `pinv`, not `solve`: where a setting leaves a direction with no information the objective is flat
    along it and the minimum-norm answer is the honest one.  With `quad` given, grid points whose
    minimised objective lands outside [0, quad] are returned NaN -- a least-squares minimum cannot do
    that, so any point that does has broken numerically and must be dropped rather than picked, which is
    exactly what an unguarded argmin did on the degenerate corner.
    """
    theta = np.einsum("...ab,...b->...a", np.linalg.pinv(M), r)
    drop = np.einsum("...a,...a->...", theta, r)
    if quad is not None:
        bad = ~np.isfinite(drop) | (drop < -1e-6 * abs(quad)) | (drop > quad * (1.0 + 1e-6))
        if ok is not None:
            bad = bad | ~ok
        drop = np.where(bad, np.nan, drop)
        theta = np.where(bad[..., None], np.nan, theta)
    return theta, drop


def scale_only(G, apm, prior, n_players):
    """The reference the blend must beat: the prior with a free scale per side and nothing else.

    This is what `PriorRidgeCV.free_prior_scale` already does, so a blend that only matches it has found
    nothing the pipeline did not already have.
    """
    V = np.zeros((2, 2 * n_players))
    V[0, :n_players] = prior[:n_players]
    V[1, n_players:] = prior[n_players:]
    GV = V @ G
    M, r = V @ GV.T, V @ (G @ apm)
    theta = np.linalg.pinv(M) @ r
    return float(apm @ G @ apm - theta @ r), theta


def unnormalise(values, norm_o, norm_d, i, j):
    """Basis columns were put on unit G-norm for the solve; put the answer back in the owner's units."""
    scale = np.concatenate([norm_o[i], norm_d[j]])
    out = np.array(values, dtype=float).copy()
    good = scale > 0
    out[good] = out[good] / scale[good]
    return out


def main():
    check_flags()      # refuse a flag this script does not understand (scripts/_cli.py)
    cfg = load_config(ROOT / "config.yaml")
    ctx = Context.load(cfg)
    priors_name = _flag("priors", "sy_base")
    saved = pd.read_pickle(ROOT / "outputs" / f"priors_{priors_name}.pkl")
    first, last = int(_flag("first", 1997)), int(_flag("last", 2026))
    seasons = [s for s in range(first, last + 1) if s in saved]
    boards = [int(x) for x in _flag("boards", "").split(",") if x] or seasons
    out = ROOT / "outputs" / f"{_flag('out', 'blend_apm')}.parquet"

    # a = 1 is the default and not an accident: freeing the exponent read 0.9883 pooled against 0.9895 for
    # a = 1 and lost a season on the count (experiment 12).  It also keeps the grid small, and the system
    # is (2 + 2p) square at EVERY grid point, so a wide `a` grid costs real memory once p is eleven.
    ks = np.round(np.logspace(np.log10(1.0), np.log10(50000.0), int(_flag("k_grid", 14))), 2)
    fix_a = _flag("fix_a", "1")
    alphas = (np.array([float(fix_a)]) if fix_a else
              np.array([0.2, 0.3, 0.5, 0.7, 1.0, 1.4, 2.0, 3.0])[:int(_flag("a_grid", 8))])

    covs_flag = _flag("covs", "level")
    dropped = [c for c in (_flag("drop", "") or "").split(",") if c]
    if covs_flag == "scalar":
        covs = []                               # the intercept alone: the owner's original, exactly
    elif covs_flag == "level":
        covs = [c for c in sy.LEVEL_COVARIATES if c not in dropped]
    else:
        covs = [c for c in covs_flag.split(",") if c and c not in dropped]

    # The trust boundary (src/eracoef/seasons.py): rows whose unit reaches into a season still
    # being played may not reach a fit.  Gating at the read is what keeps every fit below honest;
    # it drops nothing while no season is in progress, and says so when it does.
    panel, _ = drop_untrainable(pd.read_parquet(ROOT / "outputs/role_panel_season.parquet"),
                                cfg, what="the season panel")
    missing = [c for c in covs if c not in panel.columns]
    assert not missing, f"covariates not in the season panel: {missing}"
    # standardise on the WHOLE panel, once, so a coefficient means the same thing in every season and the
    # per-season fits can be compared with each other at all
    centre = panel[covs].mean() if covs else pd.Series(dtype=float)
    spread = panel[covs].std().replace(0.0, 1.0) if covs else pd.Series(dtype=float)
    names = ["intercept"] + list(covs)

    targets = {"O": sy.OFFENSE_TARGET, "D": sy.DEFENSE_TARGET}
    print(f"priors {priors_name}; seasons {boards[0]}-{boards[-1]} ({len(boards)}); "
          f"k grid {ks[0]:g}..{ks[-1]:g} ({len(ks)}), a grid {alphas.tolist()}; "
          f"targets O {targets['O']}, D {targets['D']}")
    print(f"level model: {len(names)} columns -- {', '.join(names)}"
          + (f"   (dropped {', '.join(dropped)})" if dropped else ""))
    print("every fit carries a free prior scale per side, so the level is measured on top of the rescale "
          "the ridge would have found anyway; the reference to beat is that scale alone.\n", flush=True)

    def covariates(season, side, player_ids):
        """The standardised covariate matrix in the design's player order, intercept first."""
        rows = panel[(panel.side == side) & (panel.season == season)]
        frame = (pd.DataFrame({"player_id": player_ids})
                 .merge(rows[["player_id"] + covs], on="player_id", how="left"))
        Z = np.ones((len(player_ids), 1 + len(covs)))
        if covs:
            # a player in the design but absent from the panel sits at the league mean, i.e. zero once
            # standardised -- the only neutral choice, and it is rare (the panel keeps every poss > 0 row)
            values = frame[covs].astype(float)
            Z[:, 1:] = ((values - centre) / spread).fillna(0.0).to_numpy()
        return Z

    per_season, store, t0 = [], {}, time.time()
    for season in boards:
        for side, target in targets.items():
            design = ctx.design([season], _target(target))
            P = season_pieces(design, saved[season]["O"], saved[season]["D"])
            if P["rank_note"] > 1e-6:
                print(f"  {season} {side}: SKIPPED -- the unpenalised solve misses its own equations by "
                      f"{P['rank_note']:.2e}; this season's APM does not exist", flush=True)
                continue
            Z_off = covariates(season, "O", P["player_ids"])
            Z_def = covariates(season, "D", P["player_ids"])
            T = grid_tables(P["G"], P["apm"], P["prior"], P["n_off"], P["n_def"], P["n_players"],
                            Z_off, Z_def, ks, alphas)
            theta, drop_v = solve_blend(T["M"], T["r"], T["quad"], T["ok"])
            obj = T["quad"] - drop_v
            d0 = P["prior"] - P["apm"]
            obj_raw = float(d0 @ P["G"] @ d0)
            obj_scale, _ = scale_only(P["G"], P["apm"], P["prior"], P["n_players"])
            best = np.unravel_index(int(np.nanargmin(obj)), obj.shape)
            width = T["width"]
            th = unnormalise(theta[best], T["norm_o"], T["norm_d"], best[0], best[1])
            store[(season, side)] = dict(settings=T["settings"], M=T["M"], r=T["r"], quad=T["quad"],
                                         ok=T["ok"], norm_o=T["norm_o"], norm_d=T["norm_d"], width=width,
                                         obj_raw=obj_raw, obj_scale=obj_scale,
                                         ss_resid=P["ss_resid"], dof=P["dof"])
            per_season.append(dict(
                season=season, side=side, target=str(target), n_players=P["n_players"],
                k_off=T["settings"][best[0]][0], a_off=T["settings"][best[0]][1],
                k_def=T["settings"][best[1]][0], a_def=T["settings"][best[1]][1],
                s_off=float(th[0]), s_def=float(th[width]),
                level_off=float(th[1]), level_def=float(th[width + 1]),
                obj=float(obj[best]), obj_raw=obj_raw, obj_scale=obj_scale,
                gain_vs_scale=float(1.0 - obj[best] / obj_scale), rank_note=P["rank_note"],
                null_extra=P["null_extra"]))
            print(f"  {season} {side}: k {T['settings'][best[0]][0]:,.0f} / "
                  f"{T['settings'][best[1]][0]:,.0f}  scale {th[0]:.2f} / {th[width]:.2f}  "
                  f"level at the mean player {th[1]:+.2f} / {th[width + 1]:+.2f}   "
                  f"APM distance {obj[best]:.4g}; free scale alone {obj_scale:.4g} "
                  f"({100 * (1 - obj[best] / obj_scale):+.2f}%)  ({time.time() - t0:.0f}s)", flush=True)

    D = pd.DataFrame(per_season)
    D.to_parquet(out, index=False)
    pd.set_option("display.width", 240, "display.max_columns", 30, "display.precision", 3)
    print(f"\n=== fitted on each season's own APM ({len(D)} season-sides)")
    print(D.round(3).to_string(index=False))

    # ---------------------------------------------------------- the pooled fit, and the held-out score
    print("\n=== fitted on every OTHER season, scored on the season it never saw")
    print("    `ratio_scale` below 1.00 = the blend beat a free prior scale on a season it never saw.")
    rows, coef_rows = [], []
    for side in ("O", "D"):
        have = [s for s in boards if (s, side) in store]
        for s in have:
            others = [t for t in have if t != s]
            if not others:
                continue
            M = sum(store[(t, side)]["M"] for t in others)
            r = sum(store[(t, side)]["r"] for t in others)
            quad = sum(store[(t, side)]["quad"] for t in others)
            ok = np.logical_and.reduce([store[(t, side)]["ok"] for t in others])
            theta, drop_v = solve_blend(M, r, quad, ok)
            b = np.unravel_index(int(np.nanargmin(quad - drop_v)), drop_v.shape)
            T = store[(s, side)]
            th_n, width = theta[b], T["width"]
            held = float(T["quad"] - 2 * th_n @ T["r"][b] + th_n @ T["M"][b] @ th_n)
            th = unnormalise(th_n, T["norm_o"], T["norm_d"], b[0], b[1])
            rows.append(dict(season=s, side=side, k_off=T["settings"][b[0]][0],
                             a_off=T["settings"][b[0]][1], k_def=T["settings"][b[1]][0],
                             s_off=float(th[0]), s_def=float(th[width]),
                             held_out=held, scale_only=T["obj_scale"], raw_prior=T["obj_raw"],
                             ratio_scale=held / T["obj_scale"], ratio_raw=held / T["obj_raw"]))
            if s == have[-1]:           # one coefficient table, from the fit that leaves the last out
                sigma2 = (sum(store[(t, side)]["ss_resid"] for t in others)
                          / sum(store[(t, side)]["dof"] for t in others))
                se_n = np.sqrt(np.maximum(np.diag(np.linalg.pinv(M[b])) * sigma2, 0.0))
                se = unnormalise(se_n, T["norm_o"], T["norm_d"], b[0], b[1])
                for w, half in ((0, "O"), (width, "D")):
                    for m, nm in enumerate(["scale"] + names):
                        coef_rows.append(dict(side=side, block=half, name=nm,
                                              coef=float(th[w + m]), se=float(se[w + m]),
                                              z=float(th[w + m] / se[w + m]) if se[w + m] else np.nan,
                                              k=T["settings"][b[0] if half == "O" else b[1]][0],
                                              a=T["settings"][b[0] if half == "O" else b[1]][1]))
    H = pd.DataFrame(rows)
    H.to_parquet(out.with_name(out.stem + "_heldout.parquet"), index=False)
    C = pd.DataFrame(coef_rows)
    C.to_parquet(out.with_name(out.stem + "_coefs.parquet"), index=False)

    for side in ("O", "D"):
        h = H[H.side == side]
        if not len(h):
            continue
        print(f"\n  --- {'offence' if side == 'O' else 'defence'} target, fitted without the scored "
              f"season")
        print(h[["season", "k_off", "k_def", "s_off", "s_def", "held_out", "scale_only",
                 "ratio_scale", "ratio_raw"]].round(4).to_string(index=False))
        print(f"      pooled ratio vs the free scale {h.ratio_scale.mean():.4f}; better than it in "
              f"{int((h.ratio_scale < 1).sum())} of {len(h)} seasons")

    print("\n=== the pooled level model, standardised coefficients (|z| above 2 is signal)")
    print("    read the offence block of the offensive target and the defence block of the defensive one;")
    print("    `scale` is the free prior scale, not part of the level.")
    for side in ("O", "D"):
        block = C[(C.side == side) & (C.block == side)]
        if len(block):
            print(f"\n  --- {'offence' if side == 'O' else 'defence'}  (k {block.k.iloc[0]:,.0f}, "
                  f"a {block.a.iloc[0]:g})")
            print(block[["name", "coef", "se", "z"]].round(3).to_string(index=False))

    if covs:
        print("\n=== what the fitted level says about particular players, points per 100")
        print("    (raw sign: on defence, positive means he allows MORE, i.e. worse)")
        print("  " + " " * 46 + "offence   defence")
        for label, profile in PROFILES.items():
            line = []
            for side in ("O", "D"):
                block = C[(C.side == side) & (C.block == side)]
                if not len(block):
                    line.append(float("nan"))
                    continue
                beta = block.set_index("name").coef
                level = beta.get("intercept", 0.0)
                for c in covs:
                    level += beta.get(c, 0.0) * (profile[c] - centre[c]) / spread[c]
                line.append(level)
            print(f"  {label:46s}{line[0]:+7.2f}   {line[1]:+7.2f}")

    print(f"\nwrote {out}, {out.with_name(out.stem + '_heldout.parquet')} and "
          f"{out.with_name(out.stem + '_coefs.parquet')} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
