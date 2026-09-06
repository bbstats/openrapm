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
        S["best"] = MspiFast("best", lam=L05, gbdt_params=FAST, decay=0.5, decay_exposure=True)
        BEST = dict(lam=L05, gbdt_params=FAST, decay=0.5, decay_exposure=True)
        # the padding of the box rates behind the prior: its constants halved / doubled, the league target
        S["mspi1_best_pad05"] = MspiFast("mspi1_best_pad05", pad_scale=0.5, **BEST)
        S["mspi1_best_pad2"] = MspiFast("mspi1_best_pad2", pad_scale=2.0, **BEST)
        S["mspi1_best_padleague"] = MspiFast("mspi1_best_padleague", pad_target="league", **BEST)
        # the GBDT prior trained on a role panel whose RAPM_1 used the halved ridge (scratch/panel_lam.py 0.5)
        S["mspi1_best_panel05"] = MspiFast("mspi1_best_panel05", panel="outputs/role_panel_lam0.5.parquet", **BEST)
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
