"""Emit notebooks/single_year.ipynb.  Edit here, re-run, and the notebook is regenerated."""
import json
from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []


def md(s):
    C.append(nbf.v4.new_markdown_cell(s.strip("\n")))


def code(s):
    C.append(nbf.v4.new_code_cell(s.strip("\n")))


md(r"""
# Single year or bust — a hackable prior + PI-RAPM

Everything here is plain scikit-learn. `eracoef` is used to **load data and to score**, never to model.
The two model cells (the GBDT, the ridge) are yours to rewrite.

**The rule this notebook enforces.** A player's season-H rating uses H's games for the evidence and
*no games of his own from any other season*. Concretely:

* the panel is one row per **player-season** (not the shipped 3-season block),
* there are **no `past_*` features** — no `past_apm`, `past_poss`, `past_rapm`,
* there is **no career pooling** — the target for a row is that row's own season, so `gbdt_win_decay`
  has no job and does not exist here,
* the prior for season H is trained on every season **except H**. Set `DROP_PLAYER = True` to also
  drop that player's other seasons from the training rows, which is the strictest reading.

**What this costs, so you know going in.** The `past_*` block is where the defensive criterion gain
came from: `board_D_interactions_nopast` was +0.005 per 100, i.e. nothing. Expect single-year defense
to be weaker on the team-game number than the shipped board. Whether it is weaker on the *player*
losses is the open question, and the last cell is how you find out.
""")

code(r"""
import sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

ROOT = Path("A:/code/spmm")
sys.path.insert(0, str(ROOT / "src"))

from eracoef.config import load_config
from eracoef.holdout import Context, Ratings, player_scores, player_truth, predict_season, score
from eracoef.inseason import season_frac

cfg = load_config(ROOT / "config.yaml")
ctx = Context.load(cfg)          # loads stints lazily; the first design build takes ~20s
pd.set_option("display.width", 220, "display.max_columns", 50, "display.precision", 4)
""")

md(r"""
## 1. The panel — one row per player-season

`outputs/role_panel_season.parquet`, built by `python scripts/49_role_panel.py --season`.
One row per `(player_id, season, side)`; `side` is `O` or `D` and **both are raw sign**
(`o` adds points scored, `d` adds points *allowed*, so a good defender has a negative `d`).

Columns you care about:

| | |
|---|---|
| the 13 centred box rates | `fg3m`, `fg2_miss`, `ast`, `tov`, … per 100 possessions |
| `raw_*` | the same 13, uncentred |
| role | `poss_pct`, `gs_pct`, `age` |
| shot quality | `shot_*` — attempts, makes and expected points by zone |
| bio | `height`, `weight`, `draft_pick`, `exp_yrs`, `entry_age`, `tenure`, `n_teams` |
| on-court | `onc_o`, `onc_d` — his luck-adjusted on/off, with the possessions behind each |
| **targets** | `apm`, `spm`, `u`, `rapm1` — see the warning below |
""")

code(r"""
PANEL = pd.read_parquet(ROOT / "outputs/role_panel_season.parquet")

BOX   = ["fg3m", "fg3_miss", "fg2m", "fg2_miss", "ftm", "ft_miss", "orb", "drb", "ast", "tov", "stl", "blk", "pf"]
ROLE  = ["poss_pct", "gs_pct", "age"]
SHOT  = [c for c in PANEL.columns if c.startswith("shot_")]
BIO   = [c for c in ("height", "weight", "draft_pick", "exp_yrs", "entry_age", "tenure", "n_teams") if c in PANEL]
ONC   = [c for c in PANEL.columns if c.startswith("onc_")]

FEATURES = BOX + ROLE + SHOT + BIO + ONC        # <-- knob 1: what the prior gets to see

print(PANEL.shape, "seasons", PANEL.season.min(), "-", PANEL.season.max())
print(len(FEATURES), "features:", FEATURES)
assert not PANEL.duplicated(["player_id", "season", "side"]).any()
assert not any(c.startswith("past_") for c in FEATURES), "single year or bust"
""")

md(r"""
## 2. Read this before you pick a target

The panel carries four candidate targets and they are **not interchangeable**:

| target | what it is | sd (offense) | corr with `spm` |
|---|---|---|---|
| `apm` | raw adjusted plus-minus, almost no shrinkage | 4.19 | 0.41 |
| `spm` | the *linear* box-score prediction (7 role inputs), leave-season-out | 1.57 | 1.00 |
| `u` | the ridge residual — what the box score **missed** | 0.35 | 0.02 |
| `rapm1` | `spm + u`, the shipped target | 1.62 | **0.976** |

**The trap.** `rapm1` is 97.6% `spm`, and `spm` is a deterministic linear function of role inputs that
are sitting in `FEATURES`. So a GBDT will "predict" `rapm1` at r ≈ 0.99 and that number means nothing —
it has relearned an equation, not the game. The repo uses `rapm1` anyway, and that is fine, because the
GBDT's job there is to beat `spm`'s *linear form*; but **never read r² against `rapm1` as skill.**

If you want a number that moves when you get better, use `corr(pred, apm)` — printed below — or skip
straight to the scoring cell at the bottom, which is the only thing that actually decides anything.
""")

code(r"""
o = PANEL[PANEL.side == "O"]
print(pd.DataFrame({t: {"sd": o[t].std(), "corr_with_spm": np.corrcoef(o[t], o.spm)[0, 1]}
                    for t in ("apm", "spm", "u", "rapm1")}).T.round(3).to_string())
""")

md(r"""
## 3. The prior — plain `HistGradientBoostingRegressor`

Rewrite this cell freely. Swap in `chimeraboost`, `xgboost`, a `Pipeline`, whatever. The only contract
is: **`prior_for_season(h, side)` returns one prediction per player who played in season `h`, and the
model never saw season `h`.**
""")

code(r'''
GB = dict(max_iter=300, learning_rate=0.05, max_depth=4,
          min_samples_leaf=40, l2_regularization=1.0, random_state=0)   # <-- knob 2

TARGET      = "rapm1"     # <-- knob 3: "rapm1" | "apm" | "u"
DROP_PLAYER = False       # <-- knob 4: also drop this player's OTHER seasons from training


def prior_for_season(h, side, target=None, features=None, params=None, drop_player=None):
    """Train on every season but `h`, predict `h`.  Plain sklearn; nothing up my sleeve."""
    target      = TARGET      if target      is None else target
    features    = FEATURES    if features    is None else features
    drop_player = DROP_PLAYER if drop_player is None else drop_player

    p  = PANEL[PANEL.side == side]
    tr = p[p.season != h]
    te = p[p.season == h]
    if drop_player:
        tr = tr[~tr.player_id.isin(set(te.player_id))]

    gb = HistGradientBoostingRegressor(**(params or GB))
    gb.fit(tr[features], tr[target], sample_weight=tr.poss)

    out = pd.DataFrame({"player_id": te.player_id.to_numpy(),
                        "pred":      gb.predict(te[features]),
                        "poss":      te.poss.to_numpy(),
                        "target":    te[target].to_numpy(),
                        "apm":       te.apm.to_numpy()})
    return out, gb
''')

code(r"""
H = 2015          # <-- the season you are building

t0 = time.time()
pri = {s: prior_for_season(H, s)[0] for s in ("O", "D")}
print(f"fit in {time.time() - t0:.0f}s")

for s, d in pri.items():
    print(f"  {s}: n={len(d):4d}  sd(pred)={d.pred.std():.3f}  "
          f"corr(pred, {TARGET})={np.corrcoef(d.pred, d.target)[0, 1]:.3f}  "
          f"corr(pred, apm)={np.corrcoef(d.pred, d.apm)[0, 1]:.3f}   <-- read this one")
""")

md(r"""
## 4. The honest split

The board rates a season from that season's games, so the honest test is **in-season**: fit on the
first 75% of H's games, score the last 25%. That is the shipped estimand (`..._q75`) and it is the
only reason any number below means anything — fit and score the same games and everything looks great.

`season_frac` gives each game its position in the season, so the cut is by date, not at random.
""")

code(r"""
CUT = 0.75        # <-- knob 5

wd_full  = ctx.design([H], "pts")                 # sparse stint design: [Z_off | Z_def | fixed effects]
frac     = season_frac(wd_full.games)
early    = frac[wd_full.rows["game_idx"].to_numpy()] < CUT
wd_fit   = wd_full.subset(early)
wd_score = wd_full.subset(~early)

m, nfix = wd_full.spec.n_ps, len(wd_full.spec.f_names)
print(f"{m} players, fixed effects {wd_full.spec.f_names}")
print(f"fit on {wd_fit.X.shape[0]} stints, score on {wd_score.X.shape[0]}")
""")

md(r"""
## 5. PI-RAPM — plain `Ridge`

Prior-informed RAPM in four steps, all visible:

1. the prior's contribution to each stint is `Z @ offset`;
2. fit the fixed effects (intercept, home, playoff, garbage time, margin) by weighted least squares
   around it, and subtract — this is Frisch–Waugh, and it is what keeps the ridge penalty **off** the
   fixed effects, which sklearn would otherwise shrink along with everything else;
3. scale the defensive columns by `1/sqrt(lam_ratio)` so one `alpha` gives two effective penalties —
   that is the only trick in here;
4. `Ridge(alpha=LAM).fit(Z, resid, sample_weight=poss)`, then `rating = prior + residual`.
""")

code(r"""
LAM       = 5726.0                             # <-- knob 6: the shipped board's effective penalty
LAM_RATIO = float(cfg["lam_ratio_plugin"])     # <-- knob 7: defense penalty / offense penalty (0.62)


def pi_rapm(wd, prior_o=None, prior_d=None, lam=LAM, lam_ratio=LAM_RATIO):
    m, nfix = wd.spec.n_ps, len(wd.spec.f_names)
    ids = wd.spec.ps_table["player_id"].to_numpy()

    z   = np.zeros(m)
    po  = z if prior_o is None else pd.Series(prior_o).reindex(ids).fillna(0.0).to_numpy()
    pdd = z if prior_d is None else pd.Series(prior_d).reindex(ids).fillna(0.0).to_numpy()
    off = np.concatenate([po, pdd])

    Z = wd.X[:, :2 * m].tocsr()                                       # 5 ones per side per row
    F = np.asarray(wd.X[:, 2 * m:2 * m + nfix].todense())
    A = np.column_stack([np.ones(wd.X.shape[0]), F])                  # the nuisance block
    y, w = wd.y, wd.w                                                 # points per 100, possessions

    c     = Z @ off                                                   # 1. the prior on each stint
    AtW   = (A * w[:, None]).T                                        # 2. fixed effects around it
    resid = y - c - A @ np.linalg.lstsq(AtW @ A, AtW @ (y - c), rcond=None)[0]

    s   = np.concatenate([np.ones(m), np.full(m, 1.0 / np.sqrt(lam_ratio))])   # 3. two penalties, one alpha
    rid = Ridge(alpha=lam, fit_intercept=False, solver="lsqr")                 # 4. the ridge
    rid.fit(Z.multiply(s[None, :]).tocsr(), resid, sample_weight=w)
    u = rid.coef_ * s

    poss = np.asarray(Z[:, :m].multiply(wd.rows["poss"].to_numpy()[:, None]).sum(axis=0)).ravel()
    return Ratings(pd.DataFrame({"player_id": ids, "o": po + u[:m], "d": pdd + u[m:],
                                 "poss": poss, "prior_o": po, "prior_d": pdd}))
""")

md(r"""
## 6. Does it look like basketball?

The eyeball test before the arithmetic. `total = o - d` because `d` is raw sign.
""")

code(r"""
rat = pi_rapm(wd_fit,
              prior_o=dict(zip(pri["O"].player_id, pri["O"].pred)),
              prior_d=dict(zip(pri["D"].player_id, pri["D"].pred)))

names = (pd.read_parquet(ROOT / "artifacts/season_ratings.parquet", columns=["player_id", "player_name"])
           .drop_duplicates("player_id"))
top = (rat.df.assign(total=rat.df.o - rat.df.d).merge(names, on="player_id", how="left")
          .sort_values("total", ascending=False))
print(top[["player_name", "o", "d", "total", "poss"]].head(15).to_string(index=False))
""")

md(r"""
## 7. Score it — and this is the only cell that decides anything

Three numbers, and they answer different questions:

* **`tg`** — the team-game criterion, points per 100. Lower is better; `tg_base` is the same thing with
  no player ratings at all. This weights a player by how much he played, so a 200-possession player is a
  rounding error and **it cannot see the bottom of your board**.
* **`tau`** — Kendall tau over pairs of **teammates**. Higher is better. Every player counts once.
* **`money_skill`** — of the money a coin-flip board would misallocate on a cross-team trade, the share
  yours avoids. 0 is saying nothing, 1 is perfect.

The truth is a prior-free ridge fit of the scored games at `lam=100` — nearly unbiased, very noisy, and
containing no box prior, so it cannot flatter whichever prior you just built. **One season is one
sample.** Nothing here is significant; loop over seasons before you believe a difference.
""")

code(r"""
truth = player_truth(ctx, wd_score, lam=100.0)     # <-- knob 8: also read it at cfg["lam_plugin"]

def evaluate(name, **kw):
    r  = pi_rapm(wd_fit, **kw)
    p  = predict_season(r, wd_score, level="home")
    sc = score(p)
    pl = player_scores(r, truth).set_index("group").loc["all"]
    return dict(system=name, tg=sc["tg"], tg_base=sc["tg_base"],
                tau=pl.tau, tau_league=pl.tau_league, top_k=pl.top_k,
                money_skill=pl.money_skill, sd_o=r.df.o.std(), sd_d=r.df.d.std())

print(pd.DataFrame([
    evaluate("no prior (pure RAPM)"),
    evaluate("PI-RAPM, your prior",
             prior_o=dict(zip(pri["O"].player_id, pri["O"].pred)),
             prior_d=dict(zip(pri["D"].player_id, pri["D"].pred))),
]).round(4).to_string(index=False))
""")

md(r"""
## 8. The loop, when one season stops being enough

Every number above is one season. This runs the whole thing across seasons and gives you a paired
comparison — which is what the repo's own verdicts are based on. ~30s a season, so start with five.
""")

code(r"""
def run(seasons, **kw):
    out = []
    for h in seasons:
        pr = {s: prior_for_season(h, s)[0] for s in ("O", "D")}
        wf = ctx.design([h], "pts")
        e  = season_frac(wf.games)[wf.rows["game_idx"].to_numpy()] < CUT
        fit_, sc_ = wf.subset(e), wf.subset(~e)
        th = player_truth(ctx, sc_, lam=100.0)
        for name, pk in (("no prior", {}),
                         ("your prior", dict(prior_o=dict(zip(pr["O"].player_id, pr["O"].pred)),
                                             prior_d=dict(zip(pr["D"].player_id, pr["D"].pred))))):
            r = pi_rapm(fit_, **pk, **kw)
            p = predict_season(r, sc_, level="home")
            pl = player_scores(r, th).set_index("group").loc["all"]
            out.append(dict(season=h, system=name, tg=score(p)["tg"], tau=pl.tau,
                            tau_league=pl.tau_league, money_skill=pl.money_skill))
        print(f"  {h} done", flush=True)
    return pd.DataFrame(out)


# R = run(range(2015, 2020))
# print(R.groupby("system")[["tg", "tau", "tau_league", "money_skill"]].mean().round(4).to_string())
# d = R.pivot_table(index="season", columns="system", values="tau")
# print("tau, your prior minus no prior, per season:"); print((d["your prior"] - d["no prior"]).round(4).to_string())
""")

md(r"""
## The knobs, in one place

| # | name | what it changes |
|---|---|---|
| 1 | `FEATURES` | what the prior sees. Drop `ONC` to make it box-score-only; drop `BIO` to make it production-only |
| 2 | `GB` | the GBDT itself. Or replace the estimator entirely |
| 3 | `TARGET` | `rapm1` (shipped, 97.6% `spm`), `apm` (raw, noisy), `u` (what the box score missed) |
| 4 | `DROP_PLAYER` | strictest single-year: the player's own other seasons leave the training rows too |
| 5 | `CUT` | how much of the season the fit sees. 0.75 is shipped |
| 6 | `LAM` | the ridge penalty. The shipped board's effective value is ~5,726 |
| 7 | `LAM_RATIO` | defensive penalty relative to offensive. 0.62 shipped |
| 8 | `truth lam` | 100 = nearly unbiased and noisy; `cfg["lam_plugin"]` = low-variance but it shrinks bench players harder than starters and flatters a board that does the same. **A verdict that flips between the two has decided nothing.** |

**Things that do not exist here, on purpose:** `gbdt_win_decay`, `PAST_DECAY`, `past_apm`, `past_poss`,
`past_rapm`, the 3-season block panel, the calibration map. The map is worth knowing about — it is
fitted on team-game residuals and it *helps* the criterion while doing nothing clear to player rank
(z −6.8 against one truth, +1.4 against the other). `src/eracoef/calmap.py` if you want it back.
""")

nb["cells"] = C
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"}}
out = Path(__file__).resolve().parent / "single_year.ipynb"
nbf.write(nb, out)
print("wrote", out, len(C), "cells")
