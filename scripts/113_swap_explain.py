"""One team's swap adjustment, step by step, reproducing the shipped numbers exactly.

    python scripts/113_swap_explain.py [--team=BOS] [--season=2026] [--player="Derrick White"] [--also=2024,2025]

The owner (2026-10-05): "Explain the swap adjustment very clearly for every BOS player in this example."  This re-runs
the shipped arm of scripts/91_swap_adjust.py ("type x0.5, team version": --kappas=0.5 --taus= --hold_spread=within
--exclude_near=0 on season_ratings_product_priorshrink_pre_swap) and keeps every intermediate number:

  type        the type model's prediction for the player (what players with his box-score rates, role, age, size
              and box prior beat or missed their ratings by in other seasons' lineup swaps), and its largest terms
  x 0.5       the shipped strength
  give-back   the team constraint: the team's possession-weighted total may not move, so the team's net type
              prediction is taken back from its players in proportion to their share of its possessions
  centering   the season's possession-weighted mean set back to the base's (one constant per side)
  spread      the width the adjustment added given back: each player's distance from his team's mean times one
              factor per season and side (swapadjust.hold_spread "within")
  final       the sum, checked against the shipped table (outputs/season_ratings_product.parquet)

Positive = good on both ends.  Writes outputs/swap_explain_<team>_<season>.csv.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _cli import check_flags, flag  # noqa: E402
from eracoef.config import load_config  # noqa: E402

NAMES = {"fg3m": "made threes", "fg3_miss": "missed threes", "fg2m": "made twos", "fg2_miss": "missed twos",
         "ftm": "made free throws", "ft_miss": "missed free throws", "orb": "offensive rebounds",
         "drb": "defensive rebounds", "ast": "assists", "tov": "turnovers", "stl": "steals", "blk": "blocks",
         "pf": "fouls", "poss_pct": "share of team possessions", "gs_pct": "share of games started",
         "gt_share": "garbage-time share", "age": "age", "exp_yrs": "seasons played", "height": "height",
         "weight": "weight", "prior_off": "box prior (offense)", "prior_def": "box prior (defense)"}


def _borrow(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    check_flags()
    team_code = flag("team", "BOS")
    H = int(flag("season", "2026"))
    S91 = _borrow("_swap91_for_113", "91_swap_adjust.py")
    sa, sw, t90 = S91.sa, S91.sw, S91._t90
    cfg = load_config(ROOT / "config.yaml")
    kappa, folds, ridge = 0.5, 5, 1e-4
    base_path = ROOT / "outputs" / "season_ratings_product_priorshrink_pre_swap.parquet"
    base = pd.read_parquet(base_path)
    raw = t90.load_table(base_path)
    feats, trainable = S91.features(base, cfg)
    fold_of = pd.Series(np.unique(raw.player_id) % folds, index=np.unique(raw.player_id))
    s0, s1 = int(cfg["first_season"]), int(cfg["last_season"])
    margin_clip = float(cfg.get("margin_clip", 25))
    pairs, tposs, grams = {}, {}, {}
    for s in range(s0, s1 + 1):
        season = sw.prepare(sw.season_rows(t90.load_stints(s, cfg), margin_clip=margin_clip), s)
        pairs[s] = sa.residual_pairs(season, raw[raw.season == s], context="nofatigue")
        tposs[s] = sa.team_possessions(season)
        fs = feats.xs(s, level="season") if s in feats.index.get_level_values("season") else feats.iloc[:0]
        grams[s] = {side: sa.pair_grams(pairs[s][side], fs, fold_of, folds) for side in sa.SIDES}
    print(f"swap pairs and grams built for {s0}-{s1}", flush=True)

    def decompose(H):
        """One season's shipped swap adjustment, every intermediate kept; checked against the shipped table."""
        fs = feats.xs(H, level="season")
        train = [grams[s] for s in grams if abs(s - H) > 0 and s in trainable]
        pred, contrib = {}, {}
        for side in sa.SIDES:
            p = pd.Series(np.nan, index=fs.index, dtype=float)
            cmat = pd.DataFrame(np.nan, index=fs.index, columns=fs.columns)
            for k in range(folds):
                beta = sa.type_coefficients([g[side] for g in train], k, ridge)
                mine = fold_of.reindex(fs.index).to_numpy() == k
                p[mine] = fs[mine].to_numpy() @ beta
                cmat[mine] = fs[mine].to_numpy() * beta
            pred[side], contrib[side] = p, cmat

        # the shipped arm, every intermediate kept (91's order: solve, centre the season, hold the spread)
        t = raw[raw.season == H].copy().reset_index(drop=True)
        key = pd.MultiIndex.from_arrays([t.player_id, t.season])
        w = {"offense": base.set_index(["player_id", "season"]).poss_off.reindex(key).fillna(0).to_numpy(),
             "defense": base.set_index(["player_id", "season"]).poss_def.reindex(key).fillna(0).to_numpy()}
        main_team = t90.main_teams().set_index(["player_id", "season"]).team_id.reindex(key).to_numpy(dtype=float)
        out = pd.DataFrame({"player_id": t.player_id})
        for side, bcol in (("offense", "o"), ("defense", "d")):
            system = sa.swap_system(pairs[H][side], tposs[H][side])
            m = kappa * pred[side]
            c1 = sa.solve(system, m, np.inf)
            typ = t.player_id.map(m).fillna(0.0).to_numpy()
            solved = t.player_id.map(c1).fillna(0.0).to_numpy()
            mean = float(np.sum(solved * w[side]) / np.sum(w[side]))
            centred = solved - mean
            held, k = sa.hold_spread(t[bcol].to_numpy() + centred, t[bcol].to_numpy(), w[side], main_team, "within")
            final = held - t[bcol].to_numpy()
            sign = 1.0 if side == "offense" else -1.0                       # positive = good on both ends
            tag = side[:3]
            out[f"{tag}_type"] = sign * typ
            out[f"{tag}_giveback"] = sign * (solved - typ)
            out[f"{tag}_centering"] = sign * (centred - solved)
            out[f"{tag}_spread"] = sign * (final - centred)
            out[f"{tag}_final"] = sign * final
            out[f"{tag}_spread_factor"] = k
            # the team's net type prediction and each player's share of the team's possessions (the give-back's base)
            top = contrib[side].reindex(t.player_id)
            out[f"{tag}_top_terms"] = [
                "; ".join(f"{NAMES[c]} {sign * kappa * v:+.2f}" for c, v in row.dropna().sort_values(key=np.abs, ascending=False)
                          .head(3).items()) if row.notna().any() else "" for _, row in top.iterrows()]
        shipped = pd.read_parquet(ROOT / "outputs" / "season_ratings_product.parquet")
        shipped = shipped[shipped.season == H].set_index("player_id")
        gap = max(float((out.set_index("player_id").off_final - shipped.c_off.reindex(out.player_id)).abs().max()),
                  float((out.set_index("player_id").def_final - shipped.c_def.reindex(out.player_id)).abs().max()))
        print(f"reproduces the shipped swap adjustment for every {H} player: max difference {gap:.2e}")
        assert gap < 1e-9, gap

        # every player of the season, for the league-wide view
        allp = out.merge(shipped[["player_name", "poss_off", "rating_total"]].reset_index(), on="player_id", how="left")
        allp.to_csv(ROOT / "outputs" / f"swap_explain_all_{H}.csv", index=False)

        return out, pred, contrib, fs, shipped

    for extra in [int(x) for x in flag("also", "").split(",") if x]:
        decompose(extra)
    out, pred, contrib, fs, shipped = decompose(H)

    who = flag("player", "")
    if who:
        pid = int(shipped.index[shipped.player_name == who][0])
        print(f"\n=== {who}: every term of the type prediction (half strength, positive good), with his standardised "
              f"feature value")
        for side in sa.SIDES:
            sign = 1.0 if side == "offense" else -1.0
            row = contrib[side].loc[pid]
            z = fs.loc[pid]
            terms = pd.DataFrame({"feature": [NAMES[c] for c in row.index], "z": z.to_numpy(),
                                  "term": sign * kappa * row.to_numpy()}).sort_values("term", key=np.abs, ascending=False)
            print(f"-- {side}: sum {terms.term.sum():+.3f}")
            print(terms.round(3).to_string(index=False))
        # his own swaps in the rated season, which the shipped arm does not read: the h-weighted mean residual of
        # the pairs he is in, signed so positive = his lineups beat the ratings' prediction against the same four
        # teammates with someone else
        print(f"\n=== {who}: his own {H} lineup swaps (not used by the shipped adjustment)")
        for side in sa.SIDES:
            pr = pairs[H][side]
            mine = (pr.fifth_a == pid) | (pr.fifth_b == pid)
            q = pr[mine]
            r = np.where(q.fifth_a == pid, q.r, -q.r) * (1.0 if side == "offense" else -1.0)
            h = q.h.to_numpy()
            mean = float(np.sum(h * r) / np.sum(h))
            sd = float(np.sqrt(np.sum(h * (r - mean) ** 2) / np.sum(h)))
            n_eff = float(np.sum(h) ** 2 / np.sum(h ** 2))
            print(f"  {side}: {mine.sum():,} pairs, information {np.sum(h):,.0f} possessions, mean residual "
                  f"{mean:+.2f} per 100 (naive se {sd / np.sqrt(n_eff):.2f}, effective pairs {n_eff:.0f})")
        # the own-swaps arm the shipped one is not: his own swaps pulled toward the type prediction ("both t30000")
        for tau in (30000.0, 3000.0):
            for side in sa.SIDES:
                system = sa.swap_system(pairs[H][side], tposs[H][side])
                c = sa.solve(system, kappa * pred[side], tau)
                sign = 1.0 if side == "offense" else -1.0
                print(f"  own swaps pulled toward the half-strength type, tau {tau:g}, {side}: {sign * c.get(pid, np.nan):+.2f}"
                      f" (before centering and the spread hold)")

    g = pd.read_parquet(ROOT / "data" / "raw" / "gamelog" / f"{H}_RS.parquet", columns=["PLAYER_ID", "TEAM_ID",
                                                                                         "TEAM_ABBREVIATION"])
    team_id = int(g[g.TEAM_ABBREVIATION == team_code].TEAM_ID.iloc[0])
    tp = pd.concat([tposs[H][s].assign(side=s) for s in sa.SIDES])
    share = (tp[tp.team == team_id].groupby("player_id").poss.sum() / tp[tp.team == team_id].poss.sum())
    ex = out[out.player_id.isin(share.index)].copy()
    ex["team_share"] = ex.player_id.map(share)
    ex = ex.merge(shipped[["player_name", "poss_off", "rating_total", "prior_off", "prior_def", "u_off", "u_def"]]
                  .reset_index(), on="player_id", how="left")
    ex["swap_total"] = ex.off_final + ex.def_final
    ex = ex.sort_values("rating_total", ascending=False)
    path = ROOT / "outputs" / f"swap_explain_{team_code}_{H}.csv"
    ex.to_csv(path, index=False)
    pd.set_option("display.width", 260, "display.max_columns", 40, "display.max_colwidth", 80)
    cols = ["player_name", "team_share", "off_type", "off_giveback", "off_centering", "off_spread", "off_final",
            "def_type", "def_giveback", "def_centering", "def_spread", "def_final", "swap_total"]
    print(ex[cols].round(3).to_string(index=False))
    print(ex[["player_name", "off_top_terms", "def_top_terms"]].to_string(index=False))
    for side in ("off", "def"):
        print(f"{side}: spread factor {ex[f'{side}_spread_factor'].iloc[0]:.3f}")
    print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
