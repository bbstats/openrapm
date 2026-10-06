"""The swap adjustment: re-split each team's credit by what its lineup swaps say, judged on the swap test.

    python scripts/91_swap_adjust.py [--base=outputs/season_ratings_unshrinkdef.parquet]
                                     [--kappas=0.5,1] [--taus=1000,3000,10000,30000,100000]
                                     [--folds=5] [--ridge=1e-4] [--first=1998] [--last=2025] [--tag=swapadj]
                                     [--choose="type x0.5"] [--hold_spread=0|within|whole]
                                     [--exclude_near=1] [--score=1] [--giveback=minutes|flat|none]

What ships (adopted 2026-10-01, the owner's "team version"):

    python scripts/91_swap_adjust.py --base=outputs/season_ratings_product.parquet --kappas=0.5 --taus=
                                     --hold_spread=within --exclude_near=0 --score=0 --tag=product_swapadj

`--giveback` (experiment 41) is how each team's net type prediction is taken back so its total stays put:
`minutes` (shipped) in proportion to each player's share of the team's possessions, `flat` the same amount from every
player of the team, `none` not at all (a diagnostic: team totals move).  Only the type arms (no --taus) take it.

`--exclude_near=0` lets every other season train the type model (ruling 2); the tests need the default 1, which
keeps the two scored seasons out.  `--score=0` skips the swap test, which is only honest at `--exclude_near=1`.

`--hold_spread=within` gives back the width the adjustment adds: each player's distance from his team's mean is
shrunk by one factor per season and side until the spread inside teams equals the base's, so team totals and the
order inside every team stay the adjustment's.  `whole` scales each side's whole list back to the base's spread.

The adjustment is `src/eracoef/swapadjust.py`; the test is `src/eracoef/swaptest.py` through
`scripts/90_swap_test.py`'s `run`.  For every rated season 1997-2026:

  1. the base ranking's own prediction of the season (context "nofatigue": the full context without the
     time-on-court clocks, which measured an artifact) leaves a residual on every swap of the season;
  2. the TYPE model -- each swap's residual on the two swapped players' feature difference -- is learned from
     seasons at least two away and from the other four player folds;
  3. each team's corrections are solved from its own swaps, pulled toward the type prediction, with the
     team's possession-weighted total held at zero.

The grid, one arm each:
  type x<k>    the type prediction alone (times k), centred on each team         (--kappas)
  own t<tau>   his own swaps alone, shrunk toward zero by tau                     (--taus)
  both t<tau>  his own swaps, shrunk toward the type prediction by tau            (--taus)

Every arm is scored on the swap test of the seasons before and after (56 observations for 1998-2025); the
arm with the best net order score is written as outputs/season_ratings_<tag>.parquet in the base table's
schema (rating_* replaced, the corrections in c_off / c_def, positive-good), ready for 63_yoy.py, 64, 66, 70,
73 and 88.  Also: outputs/<tag>_grid.parquet (every arm's scores) and outputs/<tag>_types.csv (the type
model fitted on every season, for reading).
"""
import importlib.util
import os
import sys
import time
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _cli import check_flags, flag  # noqa: E402
from eracoef import swapadjust as sa  # noqa: E402
from eracoef import swaptest as sw  # noqa: E402
from eracoef.config import load_config  # noqa: E402
from eracoef.seasons import drop_untrainable  # noqa: E402

_spec = importlib.util.spec_from_file_location("_swap90", ROOT / "scripts" / "90_swap_test.py")
_t90 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_t90)

# The type model's inputs, all from the rated season: the season panel's padded box-score rates (league-centred,
# per 100), the role (share of the team's possessions, share of games started, share in garbage time), age,
# experience and size -- and the base ranking's box prior, so the model can say "the prior is too high for this
# kind of player" if the swaps do.
#
# NOT the base ranking's residual (`u_*`, the season's own on-court evidence after the ridge shrank it).  The swap
# residuals are read in the same season, so the part of a player's evidence the ridge held back is still in them,
# and a model given `u` learns to multiply it -- about 2x on both sides on the first run, a median move of 0.9 per
# 100 -- which is a lighter ridge penalty wearing a "player type" label, not a type.  The penalty is the
# year-over-year test's to choose (13,037).
PANEL = ["fg3m", "fg3_miss", "fg2m", "fg2_miss", "ftm", "ft_miss", "orb", "drb", "ast", "tov", "stl", "blk", "pf",
         "poss_pct", "gs_pct", "gt_share", "age", "exp_yrs", "height", "weight"]
OWN = ["prior_off", "prior_def"]
FEATURES = PANEL + OWN


def features(base: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, set]:
    """(player_id, season) -> FEATURES, each standardised over every player-season; and the seasons the type model
    may TRAIN on.  The trust boundary (src/eracoef/seasons.py): a season still being played may be loaded -- its
    players get a type prediction -- but never trained on, so its swaps stay out of every type model."""
    raw_panel = pd.read_parquet(ROOT / "outputs/role_panel_season.parquet")
    trainable, _ = drop_untrainable(raw_panel, cfg, what="the season panel (type model training)")
    panel = raw_panel[raw_panel.side == "O"][["player_id", "season", *PANEL]]
    f = panel.merge(base[["player_id", "season", *OWN]], on=["player_id", "season"], how="inner")
    x = f[FEATURES].astype(float)
    f[FEATURES] = (x - x.mean()) / x.std(ddof=0)
    return f.set_index(["player_id", "season"])[FEATURES], set(trainable.season.astype(int))


def main():
    check_flags()
    cfg = load_config(ROOT / "config.yaml")
    base_path = ROOT / flag("base", "outputs/season_ratings_unshrinkdef.parquet")
    kappas = [float(x) for x in flag("kappas", "0.5,1").split(",") if x]
    taus = [float(x) for x in flag("taus", "1000,3000,10000,30000,100000").split(",") if x]
    folds = int(flag("folds", "5"))
    ridge = float(flag("ridge", "1e-4"))
    first, last = int(flag("first", "1998")), int(flag("last", "2025"))
    tag = flag("tag", "swapadj")
    hold = flag("hold_spread", "0")            # "within" | "whole" | "0": give back the width (swapadjust.hold_spread)
    giveback = flag("giveback", "minutes")     # experiment 41: how a team's net type prediction is taken back
    assert giveback in sa.GIVEBACKS, f"--giveback={giveback}: one of {sa.GIVEBACKS}"
    assert giveback == "minutes" or not taus, "--giveback other than minutes is defined for the type arms only (--taus=)"
    assert hold in ("0", "within", "whole"), f"--hold_spread={hold}: 0, within or whole"
    # Seasons within this distance of the rated one stay out of its type model.  1 (the default) keeps both scored
    # seasons out, which the swap and year-over-year tests need; 0 is the published table's setting: every other
    # season trains it, which ruling 2 allows (coefficients from other seasons; the player folds keep his own out).
    exclude_near = int(flag("exclude_near", "1"))
    score = flag("score", "1") not in ("0", "no", "false")          # 0: skip the swap test, write the arm
    s0, s1 = int(cfg["first_season"]), int(cfg["last_season"])
    margin_clip = float(cfg.get("margin_clip", 25))

    base = pd.read_parquet(base_path)
    raw = _t90.load_table(base_path)
    feats, trainable = features(base, cfg)
    fold_of = pd.Series(np.unique(raw.player_id) % folds, index=np.unique(raw.player_id))
    print(f"base {base_path.name}: {len(base):,} player-seasons; type model on {len(FEATURES)} features, "
          f"{folds} player folds, seasons more than {exclude_near} away; kappas {kappas}, taus {taus}")

    # ---------------------------------------------------------------- 1. every season's swap residuals
    t0 = time.time()
    pairs, tposs, grams = {}, {}, {}
    for s in range(s0, s1 + 1):
        season = sw.prepare(sw.season_rows(_t90.load_stints(s, cfg), margin_clip=margin_clip), s)
        pairs[s] = sa.residual_pairs(season, raw[raw.season == s], context="nofatigue")
        tposs[s] = sa.team_possessions(season)
        fs = feats.xs(s, level="season") if s in feats.index.get_level_values("season") else feats.iloc[:0]
        grams[s] = {side: sa.pair_grams(pairs[s][side], fs, fold_of, folds) for side in sa.SIDES}
        print(f"  {s}: {len(pairs[s]['offense']):,} offensive and {len(pairs[s]['defense']):,} defensive swap pairs "
              f"({time.time() - t0:.0f}s)", flush=True)

    # ---------------------------------------------------------------- 2. the type model
    beta_all = {side: sa.type_coefficients([grams[s][side] for s in grams if s in trainable], None, ridge)
                for side in sa.SIDES}
    types = pd.DataFrame({"feature": FEATURES, "offense": beta_all["offense"],
                          "defense_points_allowed": beta_all["defense"]})
    types["defense_positive_good"] = -types.defense_points_allowed
    types.to_csv(ROOT / "outputs" / f"{tag}_types.csv", index=False)
    type_pred = {}                 # (season, side) -> player_id -> the type prediction, raw sign
    for H in range(s0, s1 + 1):
        fs = feats.xs(H, level="season") if H in feats.index.get_level_values("season") else feats.iloc[:0]
        train = [grams[s] for s in grams if abs(s - H) > exclude_near and s in trainable]
        for side in sa.SIDES:
            pred = pd.Series(np.nan, index=fs.index, dtype=float)
            for k in range(folds):
                beta = sa.type_coefficients([g[side] for g in train], k, ridge)
                mine = fold_of.reindex(fs.index).to_numpy() == k
                pred[mine] = fs[mine].to_numpy() @ beta
            type_pred[(H, side)] = pred
    print(f"type models fitted ({time.time() - t0:.0f}s)")

    # ---------------------------------------------------------------- 3. the arms
    arms = [(f"type x{k:g}", k, np.inf) for k in kappas]
    arms += [(f"own t{t:g}", 0.0, t) for t in taus]
    arms += [(f"both t{t:g}", 1.0, t) for t in taus]
    systems = {(s, side): sa.swap_system(pairs[s][side], tposs[s][side]) for s in pairs for side in sa.SIDES}
    tables, corr, factors = {"incumbent": raw}, {}, []
    main_team = _t90.main_teams().set_index(["player_id", "season"]).team_id
    w_off = base.set_index(["player_id", "season"]).poss_off
    w_def = base.set_index(["player_id", "season"]).poss_def
    for name, kappa, tau in arms:
        parts = []
        for s in pairs:
            c = {side: sa.solve(systems[(s, side)], kappa * type_pred[(s, side)], tau, giveback) for side in sa.SIDES}
            t = raw[raw.season == s].copy()
            t["c_o"] = t.player_id.map(c["offense"]).fillna(0.0).to_numpy()
            t["c_d"] = t.player_id.map(c["defense"]).fillna(0.0).to_numpy()
            parts.append(t)
        t = pd.concat(parts, ignore_index=True)
        key = pd.MultiIndex.from_arrays([t.player_id, t.season])
        wo, wd = w_off.reindex(key).fillna(0).to_numpy(), w_def.reindex(key).fillna(0).to_numpy()
        for col, w in (("c_o", wo), ("c_d", wd)):           # keep each season centred where the base was
            mean = pd.Series(t[col].to_numpy() * w).groupby(t.season.to_numpy()).sum() / \
                pd.Series(w).groupby(t.season.to_numpy()).sum()
            t[col] = t[col] - t.season.map(mean).to_numpy()
        if hold != "0":
            # give back the width the adjustment added: one factor per season and side (swapadjust.hold_spread)
            team = main_team.reindex(key).to_numpy(dtype=float)
            for side, col, w in (("offense", "c_o", wo), ("defense", "c_d", wd)):
                base_col = "o" if side == "offense" else "d"
                for s in sorted(t.season.unique()):
                    m = (t.season == s).to_numpy()
                    held, k = sa.hold_spread(t[base_col].to_numpy()[m] + t[col].to_numpy()[m],
                                             t[base_col].to_numpy()[m], w[m], team[m], hold)
                    t.loc[m, col] = held - t[base_col].to_numpy()[m]
                    factors.append(dict(arm=name, season=s, side=side, factor=k))
        corr[name] = t[["player_id", "season", "c_o", "c_d"]]
        tables[name] = t.assign(o=t.o + t.c_o, d=t.d + t.c_d)[["player_id", "season", "o", "d"]]
    print(f"{len(arms)} arms built ({time.time() - t0:.0f}s)")
    if factors:
        f = pd.DataFrame(factors).groupby(["arm", "side"]).factor.agg(["mean", "min", "max"])
        print(f"spread held ({hold}): the factor each season's adjusted spread was multiplied by\n{f.round(3)}")

    # ---------------------------------------------------------------- 4. the swap test, every arm
    if score:
        res, lev = _t90.run(tables, "incumbent", cfg, ["nofatigue", "home"], first, last, 3000.0, checks=False,
                            score_plain=False)
        res.to_parquet(ROOT / "outputs" / f"{tag}_grid.parquet", index=False)
        _t90.report(res, lev, "incumbent")

    # ---------------------------------------------------------------- 5. choose, write, describe
    if score:
        net = res[(res.context == "nofatigue") & (res.side == "net") & (res.group == "all")]
        piv = net.pivot_table(index=["scored", "rated"], columns="ranking", values="order")
        gain = (piv.drop(columns="incumbent").sub(piv["incumbent"], axis=0))
        summary = pd.DataFrame({"mean": gain.mean(), "z": gain.mean() / (gain.std(ddof=1) / np.sqrt(len(gain))),
                                "wins": (gain > 0).sum()}).sort_values("mean", ascending=False)
        print("\n=== net order score against the incumbent, nofatigue context, every arm (higher is better)")
        print(summary.to_string(float_format=lambda v: f"{v:+.4f}"))
        best = summary.index[0]
    else:
        best = arms[0][0]
    chosen = flag("choose", "") or best                      # --choose="type x0.5" overrides the order score
    assert chosen in corr, f"--choose={chosen!r} is not an arm: {sorted(corr)}"
    print(f"\nchosen: {chosen}")

    c = corr[chosen]
    out = base.merge(c, on=["player_id", "season"], how="left").fillna({"c_o": 0.0, "c_d": 0.0})
    out["rating_off"] = out.rating_off + out.c_o
    out["rating_def"] = out.rating_def - out.c_d                  # c_d is points allowed; rating_def is positive-good
    out["rating_total"] = out.rating_off + out.rating_def
    out["offense"], out["defense"] = out.rating_off, -out.rating_def
    out["c_off"], out["c_def"] = out.c_o, -out.c_d
    out = out.drop(columns=["c_o", "c_d"])
    path = ROOT / "outputs" / f"season_ratings_{tag}.parquet"
    out.to_parquet(path, index=False)
    print(f"wrote {path.relative_to(ROOT)} ({chosen})")

    pd.set_option("display.width", 220)
    print("\n=== the type model, fitted on every season (standardised features; what one standard deviation more of a")
    print("    feature than the teammate he swaps with is worth beyond what PI-RAPM credited, points per 100, positive good)")
    print(types[["feature", "offense", "defense_positive_good"]].to_string(index=False, float_format=lambda v: f"{v:+.3f}"))

    for season in (2026, 2025):
        o = out[out.season == season]
        move = (o.c_off + o.c_def).abs()
        print(f"\n=== {season}: how far the chosen adjustment moves each player's total, points per 100")
        print(f"    median {move.median():.3f}, 90th percentile {move.quantile(0.9):.3f}, largest {move.max():.3f}; "
              f"over 0.1: {(move > 0.1).mean():.0%} of {len(o)} players")
        big = o.assign(move=o.c_off + o.c_def).reindex((o.c_off + o.c_def).abs().sort_values(ascending=False).index)
        print(big[["player_name", "rating_total", "move", "c_off", "c_def", "poss_off"]].head(12)
              .to_string(index=False, float_format=lambda v: f"{v:+.2f}"))
    print(f"\ndone ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
