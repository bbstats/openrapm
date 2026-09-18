"""Feature selection for the boosted box prior: BorutaShap on chimeraboost, once per side and mode.

On the pooled training rows of outputs/role_panel.parquet (every window; the target is each
player's value over his OTHER windows), with chimeraboost's exact SHAP values as the importance
(src/eracoef/gbdt_prior.py: make_boruta).  Two modes, matching GBDTPrior:
    residual   target u (RAPM_1 beyond the role prior), candidates = 13 rates + season
    full       target rapm1, candidates = 13 rates + season + poss_pct, gs_pct, age
               the play-by-play block, and the only mode that can assess what the board actually uses
    sink       the shipped target per side on PAIR rows, candidates = everything (SINK, ~95 names): wide, the
               era-relative twins, career, bio, past plus-minus on both sides, teammate turnover
    sinknoagg  the sink without the ten pure linear aggregates (the form 23.10 says Boruta can read)
    single_year  the SINGLE-YEAR pipeline's prior (scripts/62_single_year_board.py).  Target = a
               LeaveSeasonOutRAPM over every season, joined on player_id, NOT pooled: it already IS the
               leave-one-out quantity and `training_rows` would pool it a second time.  Candidates = the
               sink minus PAST and the turnover feature (both need pair rows, and both are banned under
               "single year or bust") minus `season` (a row pooled over twelve of them has none), plus
               the four on-court columns.  Needs --panel=outputs/role_panel_season.parquet.
Prints accepted / tentative / rejected, writes outputs/csv/boruta_{mode}_{side}.csv (the importance
history) and the YAML lines for config.yaml -> gbdt.features_* / features_full_*.  `season` is kept
whether or not Boruta accepts it: it is the era term the prior exists for; the verdict is recorded.

BORUTA PRUNES, IT DOES NOT DECIDE (HANDOFF 3.1, FINDINGS 22.2).  It selects against the prior's OWN
offline target -- the same objective that ranked the career-experience block as the largest feature gain
ever measured here, immediately before it cost +0.054 on the criterion.  Read an acceptance as "this is
not noise against the offline target", never as "this belongs on the board".  The criterion is the gate.

usage: python scripts/50_boruta.py [--trials=50] [--sides=O,D] [--modes=residual,full,wide] [--threads=12]
                                   [--params=cheap|config] [--panel=outputs/role_panel_season.parquet]
                                   [--first=1997] [--last=2026]
"""
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eracoef import singleyear as sy  # noqa: E402
from eracoef.config import load_config  # noqa: E402
from eracoef.bio import PLAYER_INPUTS  # noqa: E402
from eracoef.singleyear import ONC  # noqa: E402
from eracoef.gbdt_prior import (BIO_BINS, CAREER, DEFAULT_FEATURES, DERIVED, FULL_FEATURES,  # noqa: E402
                                PAST, SHOT_FEATURES, TURN_FEATURE, interaction_features, run_boruta,
                                training_rows)

cfg = load_config()
OUT = Path(cfg["_root"]) / "outputs"
G = cfg.get("gbdt", {})


# The shared command line (scripts/_cli.py): `flag` records every name it is asked for so
# `check_flags` below can refuse one that was never asked for.  A misspelled flag used to be
# ignored silently, which is how a run looks right and is wrong.
from _cli import check_flags, flag as _flag, switch   # noqa: E402


trials = int(_flag("trials", G.get("boruta_trials", 50)))
sides = _flag("sides", "O,D").split(",")
modes = _flag("modes", "residual,full").split(",")
threads = int(_flag("threads", 12))
# `--panel=` overrides `paths.role_panel`: the single-year modes read the PER-SEASON panel
# (outputs/role_panel_season.parquet) while everything else reads the 3-season block one.
panel_path = Path(cfg["_root"]) / _flag("panel", cfg.get("paths", {}).get("role_panel",
                                                                         "outputs/role_panel.parquet"))
panel = pd.read_parquet(panel_path)
print(f"panel: {panel_path} ({len(panel):,} rows, {panel.window.nunique()} windows)", flush=True)
# The kitchen sink (the owner, 2026-09-07: "take the absolute fullest kitchen sink and run borutashap"): every
# column both paths can build -- the 55 of `wide`, the era-relative Dredge twins, the career block, who he is
# (bio.py, and the bins), his past plus-minus on both sides (PAST, which needs PAIR rows: the pooled target
# contains it) and the teammate turnover of the target window.  `sinknoagg` is the same without the ten pure
# linear aggregates, the form FINDINGS 23.10 says Boruta can actually read: given `stocks` a shadow `blk` is as
# good as `blk`, and which of a collinear pair survives is a coin toss.
SINK = [*SHOT_FEATURES, *CAREER, *PLAYER_INPUTS, *BIO_BINS, *PAST, TURN_FEATURE]
SINK_NOAGG = [f for f in SINK if f not in DERIVED]
# ...and the sink crossed with HOW MUCH HE PLAYED (the owner, 2026-09-10: "gs% * feature and poss played %
# x feature, for all available features"). `gs_pct_x_<f>` and `poss_pct_x_<f>` for every candidate that is not itself one
# of the two multipliers, so the booster can be handed "two blocks per 100 in 5,000 possessions" as one
# number instead of having to split on the rate and then again on the exposure inside every leaf.
INTERACTIONS = [*SINK, *interaction_features(SINK)]
# The single-year candidate set: the sink minus everything that needs a pair row or a second season of the
# same player.  PAST and the turnover feature are banned outright ("single year or bust"); `season` is
# dropped because a training row here is a player pooled over every season but one.  `singleyear.ONC` is
# added -- the board carries his own season's on-court record and Boruta should get to judge it.
SINGLE_YEAR = [f for f in SHOT_FEATURES if f != "season"] + [*CAREER, *PLAYER_INPUTS, *BIO_BINS, *ONC]
MODES = {"residual": ("u", DEFAULT_FEATURES, "features_{}"), "full": ("rapm1", FULL_FEATURES, "features_full_{}"),
         "sink": ("pairs", SINK, "features_full_{}"), "sinknoagg": ("pairs", SINK_NOAGG, "features_full_{}"),
         "interactions": ("pairs", INTERACTIONS, "features_full_{}"),
         "single_year": ("loso", SINGLE_YEAR, "features_sy_{}")}


def loso_training_rows(side, feats, panel):
    """One row per player: his other-seasons features, and a RAPM over EVERY season as the target.

    `training_rows` manufactures its target by pooling the player's other windows.  Here the target is
    already a leave-one-out quantity -- `LeaveSeasonOutRAPM` fits one rating per player over a set of
    seasons -- so pooling again would count it twice.  This joins it on `player_id` instead and takes
    the weight from the possessions behind it, which is what `scripts/62_single_year_board.py` does.

    The RAPM is fit over ALL seasons, not leaving one out: feature selection is a population-level
    decision made once, like every other feature list in this project, and there is no held-out season
    to protect at this stage.  Thirty seasons of designs are accumulated and discarded one at a time.
    """
    from eracoef.holdout import Context
    from eracoef.looseason import LeaveSeasonOutRAPM
    from eracoef.xshoot import DEFENSE_TARGETS
    name = sy.OFFENSE_TARGET if side == "O" else sy.DEFENSE_TARGET
    spec = DEFENSE_TARGETS[name] if name in DEFENSE_TARGETS else name
    ctx = Context.load(cfg)
    first = int(_flag("first", cfg.get("first_season", 1997)))
    last = int(_flag("last", cfg.get("last_season", 2026)))
    rapm = LeaveSeasonOutRAPM(min_possessions=sy.MIN_POSSESSIONS)
    for s in range(first, last + 1):
        rapm.add_season(s, ctx.design([s], spec))      # not held: ctx.design keeps only the last four
    target = rapm.ratings(offense_lambda=sy.RAPM_OFFENSE_LAMBDA, defense_lambda=sy.RAPM_DEFENSE_LAMBDA,
                          context_lambda=sy.RAPM_CONTEXT_LAMBDA)
    rows = sy.prior_rows(target, panel[(panel.side == side) & (panel.poss > 0)],
                         "offense" if side == "O" else "defense", feats)
    return rows.reset_index(), name


def pair_training_rows(side, feats):
    """The shipped prior's PAIR rows for one side with every sink column on them: the shipped target per side
    (blend0.7 on offense, rapm1 on defense), the shipped window discount, the turnover table for `turn`."""
    from eracoef.holdout import Context
    ctx = Context.load(cfg)
    P = cfg.get("ratings_prior", {})
    tgt = str(P.get("gbdt_target", "rapm1")) if side == "O" else str(P.get("gbdt_target_def") or P.get("gbdt_target", "rapm1"))
    wd = float(P.get("gbdt_win_decay", 1.0)) if side == "O" else float(P.get("gbdt_win_decay_def") or P.get("gbdt_win_decay", 1.0))
    col = tgt if (tgt == "apm" or tgt.startswith("blend")) else None
    prior = ctx.prior("full", col, {}, None, {"O": list(feats), "D": list(feats)}, wd, turn=True)
    return prior.rows(side, exclude=()), (col or "rapm1"), wd
# Boruta fits two models per trial, so the shipped offensive booster's five-member bag would cost five
# times over.  "cheap" is the unbagged shape the defensive side ships anyway.
PARAMS = {"cheap": {"linear_leaves": True, "cross_features": False},
          "config": dict(G.get("params", {}) or {})}[_flag("params", "cheap")]

check_flags()      # refuse a flag this script does not understand (scripts/_cli.py)

t0 = time.time()
yaml_lines = []
for mode in modes:
    target, feats, key = MODES[mode]
    for side in sides:
        if target == "loso":
            rows, tgt = loso_training_rows(side, feats, panel)
            print(f"\n=== mode {mode} (target: a LeaveSeasonOutRAPM on {tgt}, joined not pooled), side "
                  f"{side}: {len(rows)} rows, {len(feats)} candidates, {trials} trials", flush=True)
        elif target == "pairs":
            rows, tgt, wd = pair_training_rows(side, feats)
            print(f"\n=== mode {mode} (target {tgt}, PAIR rows, win_decay {wd:g}), side {side}: {len(rows)} rows, "
                  f"{len(feats)} candidates, {trials} trials", flush=True)
        else:
            rows = training_rows(panel, side, exclude=(), features=feats, target_col=target)
            print(f"\n=== mode {mode} (target {target}), side {side}: {len(rows)} training rows "
                  f"(players seen in 2+ windows), {trials} trials", flush=True)
        res = run_boruta(rows, feats, n_trials=trials, seed=int(G.get("seed", 0)), thread_count=threads,
                         verbose=False, **PARAMS)
        print(f"  accepted : {res['accepted']}")
        print(f"  tentative: {res['tentative']}")
        print(f"  rejected : {res['rejected']}   ({time.time() - t0:.0f}s)")
        if res["history"] is not None:
            res["history"].to_csv(OUT / "csv" / f"boruta_{mode}_{side}.csv", index=False)
            print("  EVERY candidate, verdict and mean importance over trials (z-scored; the shadow max "
                  "is the bar).  Never read this as a prose list of winners -- the rejections are the result.")
            imp = res["history"].mean().round(3).sort_values(ascending=False)
            verdict = {**{f: "accepted" for f in res["accepted"]}, **{f: "tentative" for f in res["tentative"]},
                       **{f: "REJECTED" for f in res["rejected"]}}
            tbl = pd.DataFrame({"importance": imp, "verdict": [verdict.get(f, "") for f in imp.index]})
            with pd.option_context("display.max_rows", None):
                print(tbl.to_string())
        # what the pipeline carries today, to read the verdicts against
        shipped = list(cfg["gbdt"].get(key.format(side))
                       or (sy.PRIOR_FEATURES if mode == "single_year" else []))
        if shipped:
            drop = [f for f in shipped if f in res["rejected"]]
            miss = [f for f in res["accepted"] if f not in shipped]
            print(f"  of the {len(shipped)} SHIPPED names on this side, Boruta rejects {len(drop)}: {drop}")
            print(f"  and accepts {len(miss)} the board does not carry: {miss}")
        keep = sorted(set(res["accepted"]) | {"season"})
        yaml_lines.append(f"  {key.format(side)}: [{', '.join(keep)}]"
                          + ("" if "season" in res["accepted"] else "   # season kept by design")
                          + (f"   # tentative: {', '.join(res['tentative'])}" if res["tentative"] else ""))

print("\n=== config.yaml -> gbdt:")
print("\n".join(yaml_lines))
