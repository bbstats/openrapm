"""The named systems the holdout CLI (scripts/45_holdout.py) and the parallel runner can build.

A registry function rather than a module-level dict so that a spawned worker process can rebuild
systems by name (closures such as the role-prior offset do not pickle), and so that the same names
mean the same thing in every script.

    rapm            no box prior
    pi              the team-priced prior (prior-informed RAPM)
    hybrid          player-priced offense, no defensive prior
    hybrid_xft      hybrid on the free-throw-adjusted target
    hybrid_c<c>     hybrid offense with the team-priced defensive prior scaled by c
    hybrid_<t>      hybrid on a shooter-level target t (xshoot.TARGET_REGISTRY); split_<t> = offense only
    def3_p<n>       offense from hybrid_xft, defense from the opponent-3PM-replaced target (the board)
    spm             the role prior alone on both sides (APM -> Simple SPM -> ridge), no box prior
    mspi_resid        role prior + the boosted box prior on both sides
    mspi_resid_o      role prior on both sides + the boosted box prior on offense only
    rankmap_<s>     any of the above with the leave-one-season-out rank map (needs a rank table)
    <s>_<family>    any of the above with the smooth calibration map (calmap.py; needs the parameter table
                    from scripts/53_calmap.py fit, passed as `calmap=`); the names are the table's own
"""
from __future__ import annotations

import pandas as pd

from .holdout import (PluginSystem, RankMappedSystem, ReplacementSystem, SplitSystem, beta_hybrid, beta_mixed, beta_none,
                      beta_team)


def registry(cfg, rankmap=None, calmap=None) -> dict:
    S = {
        "rapm": PluginSystem("rapm", beta=beta_none),
        "pi": PluginSystem("pi", beta=beta_team),
        "hybrid": PluginSystem("hybrid", beta=beta_hybrid),
        "hybrid_xft": PluginSystem("hybrid_xft", target="xpts_ft", beta=beta_hybrid),
        "pi_xft": PluginSystem("pi_xft", target="xpts_ft", beta=beta_team),
    }
    for c in (0.25, 0.5, 0.75, 1.0):
        S[f"hybrid_c{c:g}"] = PluginSystem(f"hybrid_c{c:g}", beta=beta_mixed(c))
        S[f"hybrid_xft_c{c:g}"] = PluginSystem(f"hybrid_xft_c{c:g}", target="xpts_ft", beta=beta_mixed(c))
    try:                                                   # the shooter-level targets, once built
        from . import xshoot
        for name, target in xshoot.TARGET_REGISTRY.items():
            S[f"hybrid_{name}"] = PluginSystem(f"hybrid_{name}", target=target, beta=beta_hybrid)
            S[f"split_{name}"] = SplitSystem(f"split_{name}", offense=S[f"hybrid_{name}"], defense=S["hybrid"])
        # defensive targets: offense from the free-throw system, defense from a fit on the target
        for name, target in xshoot.DEFENSE_TARGETS.items():
            S[f"def3_{name[6:] or 'p0'}"] = SplitSystem(f"def3_{name[6:] or 'p0'}", offense=S["hybrid_xft"],
                                                        defense=PluginSystem(f"hybrid_{name}", target=target, beta=beta_hybrid))
        # the role-prior chain (spm.py, gbdt_prior.py): no linear box prior, the offset carries everything;
        # offense on the free-throw target, defense on the opponent-3PM-replaced one, like the board
        from .spm import chain_offset

        def chain(name, sides, mode="residual", scale=1.0, target="rapm1"):
            o = PluginSystem(f"{name}_o", target="xpts_ft", beta=beta_none,
                             offset=chain_offset(sides, mode, scale=scale, target=target))
            d = PluginSystem(f"{name}_d", target=xshoot.DEFENSE_TARGETS["x3def"], beta=beta_none,
                             offset=chain_offset(sides, mode, scale=scale, target=target))
            return SplitSystem(name, offense=o, defense=d)

        S["spm"] = chain("spm", ())
        S["mspi_resid"] = chain("mspi_resid", ("O", "D"))                    # SPM + GBDT on the residual beyond it
        S["mspi_resid_o"] = chain("mspi_resid_o", ("O",))
        S["mspi"] = chain("mspi", ("O", "D"), mode="full")     # the GBDT (rates + season + role) alone
        S["mspi_o"] = chain("mspi_o", ("O",), mode="full")     # ... on offense; SPM on defense
        for sc in (1.25, 1.5, 2.0):                            # the GBDT prior scaled before the ridge
            S[f"mspi_s{sc:g}".replace(".", "")] = chain(f"mspi_s{sc:g}".replace(".", ""), ("O", "D"), mode="full", scale=sc)
        S["mspi_apm"] = chain("mspi_apm", ("O", "D"), mode="full", target="apm")   # trained on unshrunk APM
        for sc in (1.25, 1.5):
            S[f"mspi_apm_s{sc:g}".replace(".", "")] = chain(f"mspi_apm_s{sc:g}".replace(".", ""), ("O", "D"), mode="full",
                                                          scale=sc, target="apm")

        # the same board in one pass per block (fastfit.py): identical numbers, about half the time
        from .fastfit import MspiFast
        S["mspi1"] = MspiFast("mspi1")
        # the GBDT prior without its audition fits (linear leaves / cross features are validation-selected by
        # default, each about 2x the fit time)
        S["mspi1_ll0"] = MspiFast("mspi1_ll0", gbdt_params={"linear_leaves": False})
        S["mspi1_cf0"] = MspiFast("mspi1_cf0", gbdt_params={"cross_features": False})
        S["mspi1_ll0cf0"] = MspiFast("mspi1_ll0cf0", gbdt_params={"linear_leaves": False, "cross_features": False})
        S["mspi1_ll1cf0"] = MspiFast("mspi1_ll1cf0", gbdt_params={"linear_leaves": True, "cross_features": False})
        # the ridge moved from the stint-CV choice (lam_plugin, lam_ratio_plugin)
        for f in (0.5, 0.7, 1.4, 2.0):
            S[f"mspi1_lam{f:g}".replace(".", "")] = MspiFast(f"mspi1_lam{f:g}".replace(".", ""), lam=float(cfg["lam_plugin"]) * f)
        for r in (0.2, 0.4, 0.6):
            S[f"mspi1_ratio{r:g}".replace(".", "")] = MspiFast(f"mspi1_ratio{r:g}".replace(".", ""), lam_ratio=r)
        for f in (0.15, 0.25, 0.35):
            S[f"mspi1_lam{f:g}".replace(".", "")] = MspiFast(f"mspi1_lam{f:g}".replace(".", ""), lam=float(cfg["lam_plugin"]) * f)
        # the defensive ratio at a weaker offensive ridge (lam_ratio = lambda_D / lambda_O)
        for f in (0.5, 0.35):
            for r in (0.15, 0.2, 0.4, 0.6, 0.8):
                n = f"mspi1_lam{f:g}_r{r:g}".replace(".", "")
                S[n] = MspiFast(n, lam=float(cfg["lam_plugin"]) * f, lam_ratio=r)
        L05 = float(cfg["lam_plugin"]) * 0.5
        # the low-possession players' ridge on its own (cfg low_poss_threshold), at the x0.5 ridge
        for b in (0.5, 2.0, 4.0):
            n = f"mspi1_lam05_low{b:g}".replace(".", "")
            S[n] = MspiFast(n, lam=L05, lam_buckets={"low_poss": b})
        # the GBDT prior's shape, at the x0.5 ridge (chimeraboost overrides)
        for tag, prm in (("d4", {"depth": 4}), ("d8", {"depth": 8}), ("l2x5", {"l2_leaf_reg": 5.0}),
                         ("l2x20", {"l2_leaf_reg": 20.0}), ("bag3", {"n_ensembles": 3}),
                         ("lr05", {"learning_rate": 0.05}), ("mcw20", {"min_child_weight": 20.0})):
            S[f"mspi1_lam05_{tag}"] = MspiFast(f"mspi1_lam05_{tag}", lam=L05, gbdt_params=prm)
        S["mspi1_lam05_po"] = MspiFast("mspi1_lam05_po", lam=L05, phases=("RS", "PO"))
        S["mspi1_lam05"] = MspiFast("mspi1_lam05", lam=L05)
        # the current best on the true loss: the x0.5 ridge and the GBDT prior without its audition fits
        FAST = {"linear_leaves": False, "cross_features": False}
        S["mspi1_lam05_fast"] = MspiFast("mspi1_lam05_fast", lam=L05, gbdt_params=FAST)
        # the farther training season (H-2 at K=3) down-weighted in the ridge
        for d in (0.7, 0.5, 0.3):
            n = f"mspi1_lam05_fast_dec{d:g}".replace(".", "")
            S[n] = MspiFast(n, lam=L05, gbdt_params=FAST, decay=d)
        for f in (0.35, 0.7):
            n = f"mspi1_lam{f:g}_fast_dec05".replace(".", "")
            S[n] = MspiFast(n, lam=float(cfg["lam_plugin"]) * f, gbdt_params=FAST, decay=0.5)
        S["mspi1_lam05_fast_dec05x"] = MspiFast("mspi1_lam05_fast_dec05x", lam=L05, gbdt_params=FAST, decay=0.5, decay_exposure=True)
        # the best on the criterion: the APM-trained prior at the shipped ridge (x0.5 / x0.7 / x1 within 0.03), the
        # GBDT without audition fits, H-2 at half weight in the rows and the exposure
        S["best"] = MspiFast("best", gbdt_params=FAST, decay=0.5, decay_exposure=True, target="apm")
        # the same fit with no held-out season: what 08_ratings.py ships (decay inert, no age term in its map)
        S["ship"] = MspiFast("ship", gbdt_params=FAST, target="apm")
        # the shipping candidates that must pass the consensus floors (tests/test_vs_consensus.py): the APM prior
        # on offense with the RAPM_1 prior on defense, and the RAPM_1 prior on both sides
        S["ship_mix"] = MspiFast("ship_mix", gbdt_params=FAST, target="apm", target_d="rapm1")
        S["ship_rapm1"] = MspiFast("ship_rapm1", gbdt_params=FAST)
        S["ship_p05"] = MspiFast("ship_p05", gbdt_params=FAST, panel="outputs/role_panel_lam0.5.parquet")
        # a blended offensive target (w APM + (1 - w) RAPM_1), RAPM_1 on defense: between the two floors
        for w in (0.5, 0.7, 0.85):
            S[f"ship_blend{w:g}".replace(".", "")] = MspiFast(f"ship_blend{w:g}".replace(".", ""), gbdt_params=FAST,
                                                               target=f"blend{w}", target_d="rapm1")
        S["ship_b07d03"] = MspiFast("ship_b07d03", gbdt_params=FAST, target="blend0.7", target_d="blend0.3")
        from .design import FEATURES as _F
        ALLF = [*_F, "season", "poss_pct", "gs_pct", "age"]
        S["best_allfeat"] = MspiFast("best_allfeat", gbdt_params=FAST, decay=0.5, decay_exposure=True, target="apm",
                                     gbdt_features={"O": ALLF, "D": ALLF})
        S["best_apm300"] = MspiFast("best_apm300", gbdt_params=FAST, decay=0.5, decay_exposure=True, target="apm",
                                    panel="outputs/role_panel_apm300.parquet")
        # the GBDT prior is a fifth of a tracked run's fit seconds; cheaper trees at the same loss
        for tag, prm in (("g4", {"depth": 4}), ("g4b64", {"depth": 4, "max_bins": 64}),
                         ("glr2", {"learning_rate": 0.2}), ("gb64", {"max_bins": 64})):
            S[f"best_{tag}"] = MspiFast(f"best_{tag}", gbdt_params={**FAST, **prm}, decay=0.5, decay_exposure=True,
                                        target="apm")
        # chimeraboost's named operating points: quality 4 = 5 bagged members, 5 = 8 (ensemble_n_jobs=1 so the
        # bag does not fork inside the holdout's own workers)
        S["best_q4"] = MspiFast("best_q4", gbdt_params={**FAST, "quality": 4, "ensemble_n_jobs": 1},
                                decay=0.5, decay_exposure=True, target="apm")
        S["best_q5"] = MspiFast("best_q5", gbdt_params={**FAST, "quality": 5, "ensemble_n_jobs": 1},
                                decay=0.5, decay_exposure=True, target="apm")
        S["best_q4full"] = MspiFast("best_q4full", gbdt_params={"quality": 4, "ensemble_n_jobs": 1},
                                    decay=0.5, decay_exposure=True, target="apm")
        # the prior's features: the 13 rates + role, plus linear aggregations of the rates (gbdt_prior.DERIVED)
        from .gbdt_prior import DERIVED, FULL_FEATURES as _FF, RATIOS
        from .gbdt_prior import DERIVED_FEATURES as _DF, RATIO_FEATURES as _RF
        S["best_agg"] = MspiFast("best_agg", gbdt_params=FAST, decay=0.5, decay_exposure=True, target="apm",
                                 gbdt_features={"O": list(_DF), "D": list(_DF)})
        # the efficiency ratios (built from the panel's uncentred rates) on top, and on their own
        S["best_ratio"] = MspiFast("best_ratio", gbdt_params=FAST, decay=0.5, decay_exposure=True, target="apm",
                                   gbdt_features={"O": list(_RF), "D": list(_RF)})
        _RO = [*_FF, *RATIOS]
        S["best_ratio_only"] = MspiFast("best_ratio_only", gbdt_params=FAST, decay=0.5, decay_exposure=True,
                                        target="apm", gbdt_features={"O": _RO, "D": _RO})
        # the prior's TARGET pooled over the player's nearby windows instead of all of them
        for wd_ in (0.3, 0.5, 0.7):
            n = f"best_wd{wd_:g}".replace(".", "")
            S[n] = MspiFast(n, gbdt_params=FAST, decay=0.5, decay_exposure=True, target="apm", win_decay=wd_)
        # accuracy first (the owner, 2026-09-06: the clock is not binding at ~25 s): the ratio feature set
        # with the gains that were only ever rejected on time
        _RK = dict(gbdt_features={"O": list(_RF), "D": list(_RF)}, decay=0.5, decay_exposure=True, target="apm")
        S["best_ratio_wd03"] = MspiFast("best_ratio_wd03", gbdt_params=FAST, win_decay=0.3, **_RK)
        S["best_ratio_q4"] = MspiFast("best_ratio_q4", gbdt_params={**FAST, "quality": 4, "ensemble_n_jobs": 1},
                                      **_RK)
        S["best_ratio_wd03_q4"] = MspiFast("best_ratio_wd03_q4", win_decay=0.3,
                                           gbdt_params={**FAST, "quality": 4, "ensemble_n_jobs": 1}, **_RK)
        S["best_ratio_full"] = MspiFast("best_ratio_full", gbdt_params={"quality": 4, "ensemble_n_jobs": 1},
                                        win_decay=0.3, **_RK)
        S["best_ratio_full1"] = MspiFast("best_ratio_full1", gbdt_params={"quality": 4, "ensemble_n_jobs": 1},
                                         **_RK)                      # the full recipe, pooling every window
        S["best_ratio_q5full"] = MspiFast("best_ratio_q5full", gbdt_params={"quality": 5, "ensemble_n_jobs": 1},
                                          win_decay=0.3, **_RK)
        # the SHIPPING shape of the accuracy-first line (21.25): no held-out season, so no decay; the
        # offensive target blended back toward RAPM_1 for the consensus floors, RAPM_1 on defense
        _FULLQ4 = {"quality": 4, "ensemble_n_jobs": 1}
        _SK = dict(gbdt_features={"O": list(_RF), "D": list(_RF)}, gbdt_params=_FULLQ4, win_decay=0.3,
                   target_d="rapm1")
        for w in ("apm", "blend0.85", "blend0.7", "blend0.6", "blend0.5"):
            n = "ship_ratio_" + w.replace("blend", "b").replace(".", "")
            S[n] = MspiFast(n, target=w, **_SK)
        # the ratio feature set on OFFENSE only: the defensive agreement with the consensus is the binding
        # floor and it is that side's prior that moves it (FINDINGS 21.26)
        for w in ("blend0.7", "blend0.6"):
            n = "ship_ratio_o" + w.replace("blend0.", "")
            S[n] = MspiFast(n, target=w, target_d="rapm1", gbdt_params=_FULLQ4, win_decay=0.3,
                            gbdt_features={"O": list(_RF), "D": list(_FF)})
        # what actually ships (21.26): the accuracy-first prior on OFFENSE, today's shipped prior on DEFENSE
        for w in ("blend0.85", "blend0.7", "blend0.6"):
            n = "ship_side" + w.replace("blend0.", "")
            S[n] = MspiFast(n, target=w, target_d="rapm1", gbdt_params=_FULLQ4, win_decay=0.3,
                            gbdt_params_d=dict(FAST), win_decay_d=1.0,
                            gbdt_features={"O": list(_RF), "D": list(_FF)})
        # SHOT QUALITY (FINDINGS 22.1): the six features built from the shooter's own locations
        # (gbdt_prior.SHOTQ) on top of the ratio set.  The first thing added to the prior that is new
        # INFORMATION rather than a re-expression of the 13 rates.
        from .gbdt_prior import SHOT_FEATURES as _SF, SHOTQ as _SQ
        _FSQ = [*_FF, *_SQ]                  # the plain list plus shot quality: the defensive candidate
        S["best_shot"] = MspiFast("best_shot", gbdt_params=_FULLQ4, win_decay=0.3, decay=0.5,
                                  decay_exposure=True, target="apm",
                                  gbdt_features={"O": list(_SF), "D": list(_SF)})
        # the shipping shape: the accuracy-first prior on OFFENSE, the cheap one on DEFENSE, with and
        # without shot quality on the defensive side (the consensus floors decide that one, not the criterion)
        for w in ("blend0.7", "blend0.6"):
            n_ = "ship_shot" + w.replace("blend0.", "")
            S[n_] = MspiFast(n_, target=w, target_d="rapm1", gbdt_params=_FULLQ4, win_decay=0.3,
                             gbdt_params_d=dict(FAST), win_decay_d=1.0,
                             gbdt_features={"O": list(_SF), "D": list(_FF)})
            S[n_ + "d"] = MspiFast(n_ + "d", target=w, target_d="rapm1", gbdt_params=_FULLQ4, win_decay=0.3,
                                   gbdt_params_d=dict(FAST), win_decay_d=1.0,
                                   gbdt_features={"O": list(_SF), "D": list(_FSQ)})
        # EXPERIENCE (FINDINGS 22.2) on top of shot quality: `best_both` is the two information blocks
        # together, which is what the prior's own leave-window-out fit likes best
        from .gbdt_prior import CAREER as _CA, PRIOR_FEATURES as _PF
        S["best_career"] = MspiFast("best_career", gbdt_params=_FULLQ4, win_decay=0.3, decay=0.5,
                                    decay_exposure=True, target="apm",
                                    gbdt_features={"O": [*_RF, *_CA], "D": [*_RF, *_CA]})
        S["best_both"] = MspiFast("best_both", gbdt_params=_FULLQ4, win_decay=0.3, decay=0.5,
                                  decay_exposure=True, target="apm",
                                  gbdt_features={"O": list(_PF), "D": list(_PF)})
        # DEFENSIVE LINEAR LEAVES (the owner, 2026-09-06: "it would not kill us to put season in as a linear
        # term").  A season INTERCEPT has nothing to capture -- the panel's target is possession-centred
        # inside every window, so it drifts 0.078 over 29 years against a target sd of 2.2 -- but season is
        # worth +0.066 as a CONDITIONER, and defense is the one side that ships with `linear_leaves: False`.
        # A ridge per leaf over the split features is that conditioning made local and linear: -0.009 weighted
        # MSE on the prior's own fit, the largest defensive gain measured this pass.
        _LLD = {"linear_leaves": True, "cross_features": False}
        # and the pieces of quality=4 decomposed on defense: linear leaves are the good part (-0.009), cross
        # features are HARMFUL alone (+0.004) and the best thing available beside them (-0.013 together), and
        # the BAG is what widens the prior (slope 1.08 -> 1.15) -- which is what the defensive spread floor
        # polices, and a mechanism for why 21.26 had to reject the whole package on this side.
        S["ship_shot7d_llcf"] = MspiFast("ship_shot7d_llcf", target="blend0.7", target_d="rapm1",
                                         gbdt_params=_FULLQ4, win_decay=0.3, win_decay_d=1.0,
                                         gbdt_params_d={"linear_leaves": True, "cross_features": True},
                                         gbdt_features={"O": list(_SF), "D": [*_FF, *_SQ]})
        S["ship_shot7d_ll"] = MspiFast("ship_shot7d_ll", target="blend0.7", target_d="rapm1",
                                       gbdt_params=_FULLQ4, win_decay=0.3, gbdt_params_d=_LLD,
                                       win_decay_d=1.0,
                                       gbdt_features={"O": list(_SF), "D": [*_FF, *_SQ]})
        # ---------------------------------------------------------------- the search's frontier (FINDINGS 22.7)
        # scripts/55_tune.py, 625 trials over the whole estimator, scored on 14 alternating held-out seasons
        # with the other 14 kept back.  These four are the confirm-half frontier: 234 the most accurate, 501
        # the parsimony pick, 596 and 609 most of the gain at a fraction of the cost.  Written out literally
        # so the board does not depend on outputs/tune_all.db.
        #
        # All four agree on three things the hand tuning had wrong or half-right: linear leaves on BOTH sides,
        # cross features on offense only, no defensive bag (22.6 found the same by hand), and a defensive
        # nearby-window discount of ~0.25-0.31 where the shipped board pools every window alike.
        from .gbdt_prior import FULL_FEATURES as _FF2, SHOT_FEATURES as _SF2, SHOTQ as _SQ2
        TUNED: dict = {}
        TUNED["tune234"] = dict(lam_mult=0.99103, lam_ratio=0.543385, blend=0.858776, win_decay=0.269307, win_decay_d=0.243613, o_depth=7, o_lr=0.11344, o_l2=0.416167, o_bins=128, o_subsample=0.756377, o_colsample=0.730909, o_mcw=1.4135, o_ll=True, o_cf=True, o_bag=5, d_depth=5, d_lr=0.0603821, d_l2=14.9619, d_bins=254, d_subsample=0.708535, d_colsample=0.678646, d_mcw=31.0206, d_ll=True, d_cf=False, d_bag=1)
        TUNED["tune501"] = dict(lam_mult=0.624047, lam_ratio=0.624519, blend=0.99967, win_decay=0.514318, win_decay_d=0.280024, o_depth=5, o_lr=0.0882469, o_l2=2.35039, o_bins=64, o_subsample=0.681911, o_colsample=0.604113, o_mcw=2.10825, o_ll=True, o_cf=True, o_bag=5, d_depth=4, d_lr=0.0872486, d_l2=18.8061, d_bins=128, d_subsample=0.654803, d_colsample=0.83231, d_mcw=3.26392, d_ll=True, d_cf=False, d_bag=1)
        TUNED["tune596"] = dict(lam_mult=0.427274, lam_ratio=0.91966, blend=0.971711, win_decay=0.403848, win_decay_d=0.308089, o_depth=4, o_lr=0.16041, o_l2=3.37425, o_bins=64, o_subsample=0.66728, o_colsample=0.754107, o_mcw=1.06444, o_ll=True, o_cf=True, o_bag=1, d_depth=4, d_lr=0.106781, d_l2=29.8289, d_bins=128, d_subsample=0.526268, d_colsample=0.708751, d_mcw=1.62074, d_ll=True, d_cf=False, d_bag=1)
        TUNED["tune609"] = dict(lam_mult=0.458471, lam_ratio=0.839104, blend=0.981735, win_decay=0.796479, win_decay_d=0.251493, o_depth=7, o_lr=0.0468551, o_l2=1.92763, o_bins=64, o_subsample=0.684056, o_colsample=0.740265, o_mcw=1.02222, o_ll=True, o_cf=True, o_bag=1, d_depth=3, d_lr=0.0419275, d_l2=6.0647, d_bins=64, d_subsample=0.78782, d_colsample=0.716312, d_mcw=29.9372, d_ll=True, d_cf=False, d_bag=1)

        def _tuned(name, p, blend=None):
            """A tuner parameter dict -> the MspiFast it describes (scripts/55_tune.py `build`)."""
            def booster(side):
                b = dict(depth=int(p[f"{side}_depth"]), learning_rate=float(p[f"{side}_lr"]),
                         l2_leaf_reg=float(p[f"{side}_l2"]), max_bins=int(p[f"{side}_bins"]),
                         subsample=float(p[f"{side}_subsample"]), colsample=float(p[f"{side}_colsample"]),
                         min_child_weight=float(p[f"{side}_mcw"]), linear_leaves=bool(p[f"{side}_ll"]),
                         cross_features=bool(p[f"{side}_cf"]))
                if int(p[f"{side}_bag"]) > 1:
                    b.update(n_ensembles=int(p[f"{side}_bag"]), ensemble_n_jobs=1)
                return b
            w = float(p["blend"] if blend is None else blend)
            return MspiFast(name, target=("apm" if w >= 0.999 else f"blend{round(w, 3)}"), target_d="rapm1",
                            lam=float(cfg["lam_plugin"]) * float(p["lam_mult"]),
                            lam_ratio=float(p["lam_ratio"]), win_decay=float(p["win_decay"]),
                            win_decay_d=float(p["win_decay_d"]),
                            gbdt_params=booster("o"), gbdt_params_d=booster("d"),
                            gbdt_features={"O": list(_SF2), "D": [*_FF2, *_SQ2]})

        for _n, _p in TUNED.items():
            S[_n] = _tuned(_n, _p)
        # the offensive target every candidate wants is nearly pure APM, which is what broke the bigness
        # floor in 21.26.  The same boosters with the target blended back toward RAPM_1, for the floors.
        for _b in (0.7, 0.6):
            for _n in ("tune501", "tune234"):
                _nm = f"{_n}_b{str(_b).replace('0.', '')}"
                S[_nm] = _tuned(_nm, TUNED[_n], blend=_b)
        # Every tuned candidate fails the consensus.  tune501_b7 fails ONE floor by 0.0008 (defensive
        # agreement 0.7592 against 0.76) while its defensive SPREAD improves to 1.28, so the prior is not too
        # wide -- the agreement alone moved.  The suspect is the defensive nearby-window discount, which the
        # search set to 0.28 and the shipped board leaves at 1.0: 21.26's pattern exactly, where what the
        # criterion wants on defense is what the floors refuse.  Back that one knob off and keep the rest.
        from dataclasses import replace as _replace
        for _wd in (1.0, 0.6):
            _t = f"{_wd:g}".replace(".", "")
            S[f"tune501_b7_wd{_t}"] = _replace(S["tune501_b7"], name=f"tune501_b7_wd{_t}", win_decay_d=_wd)
            S[f"tune501_b6_wd{_t}"] = _replace(S["tune501_b6"], name=f"tune501_b6_wd{_t}", win_decay_d=_wd)
        # and the same with the SHIPPED defensive booster, if the booster rather than the pooling is at fault
        S["tune501_b7_dship"] = _replace(S["tune501_b7"], name="tune501_b7_dship", win_decay_d=1.0,
                                         gbdt_params_d={"linear_leaves": True, "cross_features": False})
        # ---------------------------------------------------------------- the Dredge block (HANDOFF 3.1)
        # The play-by-play counters (dredge.py): Russells, rim blocks, blocked threes, unassisted makes,
        # stolen turnovers, loose-ball fouls, technicals and flagrants, offensive fouls committed.  Only
        # the DEFENSIVE side is tried, because that is the only side the pre-filter found anything on --
        # and even there what it found has the FINDINGS 22.2 signature, so the criterion is the gate.
        #
        # On the shipped defensive list and the shipped defensive booster, `scratch/prior_bench.py D`:
        #   + unast, unastsh                             -0.0129 pooled,  +0.0354 on the low-exposure rows
        #   + the whole block                            -0.0118 pooled,  +0.1157 low
        #   + loose, techflg, offoul                     -0.0011 pooled,  +0.0321 low
        #   + stolen, stolensh                           +0.0055 pooled,  +0.0012 low
        #   + russ, russsh, blkrim, blkrimsh, blk3sh     +0.0270 pooled,  +0.0825 low
        # Every pooled gain costs low-exposure accuracy, which is the half of the fit the held-out season
        # can feel, and the block split -- the thing Dredge most promised, against `blk` at 15.7% of the
        # defensive SHAP -- is the WORST of the five.  Two candidates go to the criterion anyway: the
        # bench "has been wrong by 0.13" and shot quality was flat on the line before it shipped.
        # ---------------------------------------------------------------- trade calibration (FINDINGS 24)
        # The same board with the TURNOVER-AWARE prior: trained on window pairs with the teammate turnover of
        # the target window as a feature, the ridge shrinking toward its settled-context value (turn 0.35) and
        # the rating carrying the delta to each player's turnover in the held-out season.  On the prior's own
        # pair rows the feature is -0.043 (z -2.7) on offense and -0.031 (z -4.1) on defense; the criterion
        # decides whether a team-game feels it.
        S["tune501_b7_turn"] = _replace(S["tune501_b7"], name="tune501_b7_turn", turn=True)
        # the control: the same pair-row prior evaluated at the settled value for everyone, no delta -- whether
        # un-pooling the target with `turn` in the booster changes the prior the ridge shrinks toward
        S["tune501_b7_turnref"] = _replace(S["tune501_b7"], name="tune501_b7_turnref", turn="ref")
        # two more controls: the pair rows with NO turnover feature (is it the un-pooling?), and the turnover
        # prior evaluated at the training pairs' own mean turnover 0.7 (is it the settled-context evaluation?)
        S["tune501_b7_pairs"] = _replace(S["tune501_b7"], name="tune501_b7_pairs", turn="pairs")
        S["tune501_b7_turn07"] = _replace(S["tune501_b7"], name="tune501_b7_turn07", turn="ref", turn_ref=0.7)
        # and one side at a time: where the gain lives, and whether the defensive floors need to be asked
        S["tune501_b7_turnref_o"] = _replace(S["tune501_b7"], name="tune501_b7_turnref_o", turn="ref", turn_sides=("O",))
        S["tune501_b7_turnref_d"] = _replace(S["tune501_b7"], name="tune501_b7_turnref_d", turn="ref", turn_sides=("D",))
        # ---------------------------------------------------------------- the four-factor defence (HANDOFF 3.2)
        # The defensive residual from four factor fits on the same layout -- opponents' eFG%, turnovers forced,
        # offensive rebounds allowed, free-throw rate allowed -- each with its own ridge and ratio (FINDINGS 15's,
        # fixed a priori), the points prior shared out across them, and the four recombined into points allowed
        # (fastfit.factor_defense).  On top of the shipped board; `ff5` blends half and half.
        S["tune501_b7_turnref_o_ff"] = _replace(S["tune501_b7_turnref_o"], name="tune501_b7_turnref_o_ff", def_factors=1.0)
        S["tune501_b7_turnref_o_ff5"] = _replace(S["tune501_b7_turnref_o"], name="tune501_b7_turnref_o_ff5", def_factors=0.5)
        for _s in (0.5, 2.0):
            _t = f"ff_ls{_s:g}".replace(".", "")
            S[f"tune501_b7_turnref_o_{_t}"] = _replace(S["tune501_b7_turnref_o"], name=f"tune501_b7_turnref_o_{_t}",
                                                       def_factors=1.0, factor_lam_scale=_s)
        # the factor ridges re-selected by REML inside each fit, on the residual around the prior share (FINDINGS 15's
        # were chosen with no prior and leave the residual 1.6x too wide); `ffr62` then applies the same 0.62
        # discount the estimator search put on the points ridge (tune501's lam_mult), fixed a priori
        S["tune501_b7_turnref_o_ffr"] = _replace(S["tune501_b7_turnref_o"], name="tune501_b7_turnref_o_ffr",
                                                 def_factors=1.0, factor_reml=True)
        S["tune501_b7_turnref_o_ffr62"] = _replace(S["tune501_b7_turnref_o"], name="tune501_b7_turnref_o_ffr62",
                                                   def_factors=1.0, factor_reml=True, factor_lam_scale=0.624047)
        S["tune501_b7_turnref_o_ffr5"] = _replace(S["tune501_b7_turnref_o"], name="tune501_b7_turnref_o_ffr5",
                                                  def_factors=0.5, factor_reml=True)
        # what the split-half read said (FINDINGS 25): the eFG% defensive half is noise at ratio 1.5, and the ratio
        # was fixed by a joint fit the offensive half dominates.  REML over the ratio too ("2d"), and the eFG%
        # numerator repriced at the shooters' expected threes (x3def's repair on the factor that needs it)
        S["tune501_b7_turnref_o_ff2d"] = _replace(S["tune501_b7_turnref_o"], name="tune501_b7_turnref_o_ff2d",
                                                  def_factors=1.0, factor_reml="2d")
        S["tune501_b7_turnref_o_ffx"] = _replace(S["tune501_b7_turnref_o"], name="tune501_b7_turnref_o_ffx",
                                                 def_factors=1.0, factor_reml="2d", factor_x3=True)
        S["tune501_b7_turnref_o_ffx5"] = _replace(S["tune501_b7_turnref_o"], name="tune501_b7_turnref_o_ffx5",
                                                  def_factors=0.5, factor_reml="2d", factor_x3=True)
        # the factor ridges tightened well past REML: the split-half read on one block (2021-2024 minus 2023) put the
        # factor sum's cross-half prediction of the points residual level with the points fit's own at 16-32x,
        # and below it everywhere looser.  Chosen on that read, not on the criterion.
        for _s in (16, 32):
            S[f"tune501_b7_turnref_o_ffx{_s}"] = _replace(S["tune501_b7_turnref_o"], name=f"tune501_b7_turnref_o_ffx{_s}",
                                                          def_factors=1.0, factor_reml=True, factor_x3=True,
                                                          factor_lam_scale=float(_s))
        # ---------------------------------------------------------------- who he is (bio.py, FINDINGS 26)
        # height and weight from the bio feed, a constant per player.  Offline (prior_bench) -0.145 on the defensive
        # line and -0.27 on the offensive one -- but binned to 2 in / 15 lb the defensive gain keeps 0.12 of it and the
        # offensive one 0.03: on offense the pair is naming the player (the 22.2 trap), on defense it is physiology.
        # So: defense fine and binned, both sides binned, both sides fine as the identification control.
        S["tune501_b7_turnref_o_hw"] = _replace(S["tune501_b7_turnref_o"], name="tune501_b7_turnref_o_hw",
                                                gbdt_features={"O": list(_SF2), "D": [*_FF2, *_SQ2, "height", "weight"]})
        S["tune501_b7_turnref_o_h"] = _replace(S["tune501_b7_turnref_o"], name="tune501_b7_turnref_o_h",
                                               gbdt_features={"O": list(_SF2), "D": [*_FF2, *_SQ2, "height"]})
        S["tune501_b7_turnref_o_hwc"] = _replace(S["tune501_b7_turnref_o"], name="tune501_b7_turnref_o_hwc",
                                                 gbdt_features={"O": list(_SF2), "D": [*_FF2, *_SQ2, "height2", "weight15"]})
        S["tune501_b7_turnref_o_hwb"] = _replace(S["tune501_b7_turnref_o"], name="tune501_b7_turnref_o_hwb",
                                                 gbdt_features={"O": [*_SF2, "height2", "weight15"], "D": [*_FF2, *_SQ2, "height2", "weight15"]})
        S["tune501_b7_turnref_o_hwbf"] = _replace(S["tune501_b7_turnref_o"], name="tune501_b7_turnref_o_hwbf",
                                                  gbdt_features={"O": [*_SF2, "height", "weight"], "D": [*_FF2, *_SQ2, "height", "weight"]})
        # and the draft slot (1-60, undrafted 61): -0.05 offline on top of the pair on either side
        S["tune501_b7_turnref_o_hwbd"] = _replace(S["tune501_b7_turnref_o"], name="tune501_b7_turnref_o_hwbd",
                                                  gbdt_features={"O": [*_SF2, "height2", "weight15", "draft_pick"],
                                                                 "D": [*_FF2, *_SQ2, "height2", "weight15", "draft_pick"]})
        S["tune501_b7_turnref_o_dp"] = _replace(S["tune501_b7_turnref_o"], name="tune501_b7_turnref_o_dp",
                                                gbdt_features={"O": [*_SF2, "draft_pick"], "D": [*_FF2, *_SQ2, "draft_pick"]})
        # ---------------------------------------------------------------- plus-minus as an input (FINDINGS 28)
        # his own on-court record before the block (gbdt_prior.PAST: discounted APM, the possessions behind it,
        # and the shrunk RAPM_1), on pair rows so nothing of the target is in the feature.  The investigator (27)
        # says the prior is compressed at the top on both sides; this is the information that would lift it.
        from .gbdt_prior import PAST as _PAST
        _HW = ["height2", "weight15"]
        S["tune501_b7_turnref_o_hwb_past"] = _replace(S["tune501_b7_turnref_o"], name="tune501_b7_turnref_o_hwb_past",
                                                      gbdt_features={"O": [*_SF2, *_HW, *_PAST], "D": [*_FF2, *_SQ2, *_HW, *_PAST]})
        S["tune501_b7_turnref_o_hwb_pasto"] = _replace(S["tune501_b7_turnref_o"], name="tune501_b7_turnref_o_hwb_pasto",
                                                       gbdt_features={"O": [*_SF2, *_HW, *_PAST], "D": [*_FF2, *_SQ2, *_HW]})
        S["tune501_b7_turnref_o_hwb_pastd"] = _replace(S["tune501_b7_turnref_o"], name="tune501_b7_turnref_o_hwb_pastd",
                                                       gbdt_features={"O": [*_SF2, *_HW], "D": [*_FF2, *_SQ2, *_HW, *_PAST]})
        # APM alone (no shrunk twin), and the pair-row control with no PAST at all on defense (offense already pairs)
        S["tune501_b7_turnref_o_hwb_pasta"] = _replace(S["tune501_b7_turnref_o"], name="tune501_b7_turnref_o_hwb_pasta",
                                                       gbdt_features={"O": [*_SF2, *_HW, "past_apm", "past_poss"], "D": [*_FF2, *_SQ2, *_HW, "past_apm", "past_poss"]})
        # the owner, 2026-09-07: "we need to split up o/d plus minus per 100" -- they are (the panel's APM is per
        # side); this gives each prior the OTHER side's record too, named: the offensive prior sees his past
        # defensive APM beside his past offensive one, and the defensive prior (pooled as shipped) the reverse
        _PX = ["past_apm_o", "past_poss_o", "past_apm_d", "past_poss_d"]
        S["tune501_b7_turnref_o_hwb_pastx"] = _replace(S["tune501_b7_turnref_o"], name="tune501_b7_turnref_o_hwb_pastx",
                                                       gbdt_features={"O": [*_SF2, *_HW, "past_rapm", *_PX], "D": [*_FF2, *_SQ2, *_HW]})
        S["tune501_b7_turnref_o_hwb_pastxd"] = _replace(S["tune501_b7_turnref_o"], name="tune501_b7_turnref_o_hwb_pastxd",
                                                        gbdt_features={"O": [*_SF2, *_HW, "past_rapm", *_PX], "D": [*_FF2, *_SQ2, *_HW, *_PX]})
        # the other bound: the defensive prior ALONE (the factor ridges at 1e6 leave no residual at all), so the
        # value of a defensive residual at team-game level is a number
        S["tune501_b7_turnref_o_dprior"] = _replace(S["tune501_b7_turnref_o"], name="tune501_b7_turnref_o_dprior",
                                                    def_factors=1.0, factor_lam_scale=1e6)
        # the diagnostic pair: no defensive prior at all, points residual against factor residual
        S["tune501_b7_turnref_o_nodp"] = _replace(S["tune501_b7_turnref_o"], name="tune501_b7_turnref_o_nodp", no_def_prior=True)
        S["tune501_b7_turnref_o_nodp_ffx"] = _replace(S["tune501_b7_turnref_o"], name="tune501_b7_turnref_o_nodp_ffx",
                                                      no_def_prior=True, def_factors=1.0, factor_reml=True, factor_x3=True)
        from .gbdt_prior import DREDGE as _DR
        S["tune501_b7_drd"] = _replace(S["tune501_b7"], name="tune501_b7_drd",
                                       gbdt_features={"O": list(_SF2), "D": [*_FF2, *_SQ2, *_DR]})
        S["tune501_b7_dru"] = _replace(S["tune501_b7"], name="tune501_b7_dru",
                                       gbdt_features={"O": list(_SF2), "D": [*_FF2, *_SQ2, "unast", "unastsh"]})
        # The ERA-RELATIVE form of the same block (add_dredge's `_r` columns: each feature divided by its
        # own block's league level, so a change in how the feed RECORDS an event divides out).  On the
        # prior's own fit the whole relative block is -0.029 against -0.020 for the absolute one, and the
        # two features the era artifact actually contaminates -- the rim share of blocked twos and the
        # disputed goaltend count -- are -0.021 on their own at a fifth of the low-exposure cost.
        from .gbdt_prior import DREDGE_R as _DRR
        S["tune501_b7_drr"] = _replace(S["tune501_b7"], name="tune501_b7_drr",
                                       gbdt_features={"O": list(_SF2), "D": [*_FF2, *_SQ2, *_DRR]})
        # POTENTIAL ASSISTS, reconstructed from assists by zone (gbdt_prior.POTENTIAL_AST, the owner's 2019
        # fit at r-squared ~1).  Potential assists are a TRACKING statistic and begin in 2013-14; assist
        # location is in the play-by-play from 1997, so this carries the measure back over the whole panel.
        # It is the only thing in the Dredge block the prior's own fit likes on OFFENSE (-0.018, where every
        # other group is worse) and it is the most reliable feature in the block year over year (0.922,
        # against 0.918 for blocks per 100 and 0.126 for the Russell share).
        S["tune501_b7_past"] = _replace(S["tune501_b7"], name="tune501_b7_past",
                                        gbdt_features={"O": [*_SF2, "pot_ast"], "D": [*_FF2, *_SQ2]})
        S["tune501_b7_past2"] = _replace(S["tune501_b7"], name="tune501_b7_past2",
                                         gbdt_features={"O": [*_SF2, "pot_ast"],
                                                        "D": [*_FF2, *_SQ2, "pot_ast"]})
        S["tune501_b7_dcal"] = _replace(S["tune501_b7"], name="tune501_b7_dcal",
                                        gbdt_features={"O": list(_SF2),
                                                       "D": [*_FF2, *_SQ2, "blkrimsh_r", "goalt_r"]})
        # the same without the nearby-window discount (the consensus floors, not the criterion, may want it)
        S["ship_ratio_b07_wd1"] = MspiFast("ship_ratio_b07_wd1", target="blend0.7",
                                           **{**_SK, "win_decay": 1.0})
        S["ship_ratio_b05_wd1"] = MspiFast("ship_ratio_b05_wd1", target="blend0.5",
                                           **{**_SK, "win_decay": 1.0})
        # the aggregations INSTEAD of the raw rates they are made of: a smaller, better-conditioned set
        _SM = ["season", "poss_pct", "gs_pct", "age", *DERIVED, *RATIOS]
        S["best_small"] = MspiFast("best_small", gbdt_params=FAST, decay=0.5, decay_exposure=True, target="apm",
                                   gbdt_features={"O": _SM, "D": _SM})
        # the single-possession stints dropped: a quarter of the rows, 7% of the weight
        for md in (2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 12.0):
            n = f"best_md{md:g}"
            S[n] = MspiFast(n, gbdt_params=FAST, decay=0.5, decay_exposure=True, target="apm", min_den=md)
        S["best_b07"] = MspiFast("best_b07", gbdt_params=FAST, target="blend0.7", target_d="rapm1", decay=0.5, decay_exposure=True)
        S["ship_p035"] = MspiFast("ship_p035", gbdt_params=FAST, panel="outputs/role_panel_lam0.35.parquet")
        S["best_mix"] = MspiFast("best_mix", gbdt_params=FAST, target="apm", target_d="rapm1", decay=0.5, decay_exposure=True)
        BEST = dict(lam=L05, gbdt_params=FAST, decay=0.5, decay_exposure=True)
        # the padding of the box rates behind the prior: its constants halved / doubled, the league target
        S["mspi1_best_pad05"] = MspiFast("mspi1_best_pad05", pad_scale=0.5, **BEST)
        S["mspi1_best_pad2"] = MspiFast("mspi1_best_pad2", pad_scale=2.0, **BEST)
        S["mspi1_best_padleague"] = MspiFast("mspi1_best_padleague", pad_target="league", **BEST)
        # the GBDT prior trained on a role panel whose RAPM_1 used the halved ridge (scratch/panel_lam.py 0.5)
        S["mspi1_best_panel05"] = MspiFast("mspi1_best_panel05", panel="outputs/role_panel_lam0.5.parquet", **BEST)
        S["mspi1_best_panel07"] = MspiFast("mspi1_best_panel07", panel="outputs/role_panel_lam0.7.parquet", **BEST)
        S["mspi1_best_panel035"] = MspiFast("mspi1_best_panel035", panel="outputs/role_panel_lam0.35.parquet", **BEST)
        S["mspi1_best_panel025"] = MspiFast("mspi1_best_panel025", panel="outputs/role_panel_lam0.25.parquet", **BEST)
        S["mspi1_best_panel015"] = MspiFast("mspi1_best_panel015", panel="outputs/role_panel_lam0.15.parquet", **BEST)
        S["mspi1_best_apm"] = MspiFast("mspi1_best_apm", target="apm", **BEST)      # the prior trained on unshrunk APM
        APM = {**BEST, "target": "apm"}
        for f in (0.35, 0.7, 1.0):
            n = f"mspi1_apm_lam{f:g}".replace(".", "")
            S[n] = MspiFast(n, **{**APM, "lam": float(cfg["lam_plugin"]) * f})
        S["mspi1_apm30"] = MspiFast("mspi1_apm30", panel="outputs/role_panel_apm30.parquet", **APM)   # APM at penalty 30
        S["mspi1_apm_nodec"] = MspiFast("mspi1_apm_nodec", **{**APM, "decay": None, "decay_exposure": False})
        # the GBDT's regularisation on the noisier APM target
        for tag, prm in (("l2x5", {"l2_leaf_reg": 5.0}), ("l2x20", {"l2_leaf_reg": 20.0}), ("mcw20", {"min_child_weight": 20.0}),
                         ("d4", {"depth": 4}), ("d8", {"depth": 8}), ("lr05", {"learning_rate": 0.05})):
            S[f"mspi1_apm_{tag}"] = MspiFast(f"mspi1_apm_{tag}", **{**APM, "gbdt_params": {**FAST, **prm}})
        # cheaper GBDT trees: depth 4 (flat in loss in section 21.4), 64 bins
        S["mspi1_best_d4"] = MspiFast("mspi1_best_d4", **{**BEST, "gbdt_params": {**FAST, "depth": 4}})
        S["mspi1_best_mb64"] = MspiFast("mspi1_best_mb64", **{**BEST, "gbdt_params": {**FAST, "max_bins": 64}})
        S["mspi1_best_d4mb64"] = MspiFast("mspi1_best_d4mb64", **{**BEST, "gbdt_params": {**FAST, "depth": 4, "max_bins": 64}})
        # the two adjacent seasons weighted apart: the past one or the future one at 0.8
        for tag, sw in (("past08", {-2: 0.5, -1: 0.8, 1: 1.0}), ("fut08", {-2: 0.5, -1: 1.0, 1: 0.8}),
                        ("past07fut1", {-2: 0.4, -1: 0.7, 1: 1.0})):
            n = f"mspi1_best_{tag}"
            S[n] = MspiFast(n, lam=L05, gbdt_params=FAST, season_weights=sw, decay_exposure=True)
        # the defensive target with the season before the block in the shooters' rates
        S["mspi1_lam05_fast_dec05_p1"] = MspiFast("mspi1_lam05_fast_dec05_p1", lam=L05, gbdt_params=FAST, decay=0.5, def_target="x3def_p1")
        # one target for both sides (one solve): the opponent-3PM-replaced target on offense too
        S["mspi1_x3both"] = MspiFast("mspi1_x3both", off_target="x3def")
        # the APM-trained offense with the RAPM_1-trained defense (each side's calibrated version)
        S["mspi_mix"] = SplitSystem("mspi_mix", offense=S["mspi_apm"], defense=S["mspi"])
        # a replacement level for players the block never saw, on the board and on the chains
        for n in ("def3_p0", "mspi", "mspi_apm", "mspi_mix"):
            S[f"{n}_rep"] = ReplacementSystem(f"{n}_rep", S[n])

        # garbage time down-weighted in the FIT only (the held-out scoring never changes): the board and
        # the multi-stage chain at gt_weight 0.5 and 0, so the comparison holds the weighting fixed
        from .windows import build_window

        def gt_target(base: str, w: float):
            def target(seasons, cfg, wd_pts):
                if base == "x3def":
                    wd05 = build_window(seasons, cfg, gt_weight=w)
                    return xshoot.DEFENSE_TARGETS["x3def"](seasons, cfg, wd05)
                return build_window(seasons, cfg, gt_weight=w, target=base)
            target.__name__ = f"{base}_gt{w:g}"
            return target

        for w in (0.5, 0.0):
            tag = f"gt{w:g}".replace(".", "")
            o_t, d_t = gt_target("xpts_ft", w), gt_target("x3def", w)
            S[f"def3_{tag}"] = SplitSystem(f"def3_{tag}", offense=PluginSystem(f"hybrid_xft_{tag}", target=o_t, beta=beta_hybrid),
                                           defense=PluginSystem(f"hybrid_x3def_{tag}", target=d_t, beta=beta_hybrid))
            S[f"mspi_{tag}"] = SplitSystem(f"mspi_{tag}",
                                           offense=PluginSystem(f"mspi_{tag}_o", target=o_t, beta=beta_none,
                                                                offset=chain_offset(("O", "D"), "full")),
                                           defense=PluginSystem(f"mspi_{tag}_d", target=d_t, beta=beta_none,
                                                                offset=chain_offset(("O", "D"), "full")))
    except ImportError:
        pass
    if rankmap:
        rank_table = pd.read_parquet(rankmap)
        have = set(rank_table.system)
        for n in list(S):
            if n in have:
                S[f"rankmap_{n}"] = RankMappedSystem(f"rankmap_{n}", S[n], rank_table)
        if "rankmap_def3_p0" in S:                     # the board as it ships, plus the replacement level
            S["rankmap_def3_p0_rep"] = ReplacementSystem("rankmap_def3_p0_rep", S["rankmap_def3_p0"])
    if calmap:
        from .calmap import CalMappedSystem
        params = pd.read_parquet(calmap)
        for (name, base), _ in params.groupby(["system", "base"]):
            if base in S:
                S[name] = CalMappedSystem(name, S[base], params, name)
    return S
