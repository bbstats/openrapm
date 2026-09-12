"""Emit notebooks/single_year.ipynb.  Edit here, re-run, and the notebook is regenerated."""
from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()
CELLS = []


def md(text):
    CELLS.append(nbf.v4.new_markdown_cell(text.strip("\n")))


def code(text):
    CELLS.append(nbf.v4.new_code_cell(text.strip("\n")))


# kept out of line so its docstring's triple quotes never nest inside another literal
CELL_FIT_PRIOR = '\n'.join([
    'def fit_prior(held_out_season, side):',
    '    """Map a box score to the RAPM of every season except `held_out_season`."""',
    '    target = rapm.ratings(held_out_season=held_out_season,',
    '                          offense_lambda=RAPM_OFFENSE_LAMBDA,',
    '                          defense_lambda=RAPM_DEFENSE_LAMBDA,',
    '                          context_lambda=RAPM_CONTEXT_LAMBDA)',
    '    column = "offense" if side == "O" else "defense"',
    '',
    '    train = aggregate_features(held_out_season, side).join(',
    '        target.set_index("player_id")[[column, "possessions"]], how="inner").dropna()',
    '',
    '    model = ChimeraBoostRegressor(random_state=0,',
    '                                  **(chimera_offense if side == "O" else chimera_defense))',
    '    model.fit(train[features].to_numpy(dtype=float), train[column].to_numpy(dtype=float),',
    '              sample_weight=train.possessions.to_numpy(dtype=float))',
    '',
    '    held = panel[(panel.side == side) & (panel.season == held_out_season)]',
    '    return pd.DataFrame({"player_id": held.player_id.to_numpy(),',
    '                         "prediction": model.predict(held[features].to_numpy(dtype=float))}), model, train',
])


md(r"""
# Single year or bust

A player's rating for season H comes from H's games. Nothing that touches H is ever fit on H.

Three stages:

1. **The target** — `LeaveSeasonOutRAPM`. One rating per *player* from every season **except H**, so the
   thing the prior learns is measured on thousands of possessions instead of one noisy season. Its penalty
   is swept on seasons the fit has not seen.
2. **The prior** — `ChimeraBoostRegressor` maps a box score to that target. Trained on one row per player
   (his all-other-seasons averages → his all-other-seasons RAPM), then asked about season H's box score.
3. **The ratings** — `PriorRidgeCV`, a ridge that shrinks toward the prior instead of toward zero, fit on
   the first 75% of H's games and scored on the last 25%.

**The thing to keep an eye on.** Stage 2 trains on career averages and predicts on a single season, and a
single season is 10–25% wider in every feature. A booster does not extrapolate — it clamps at the edge of
what it saw — so extreme seasons get pulled toward the middle. That is a conservative failure rather than a
wild one, but it is a real cost of the trade and cell 6 measures it.
""")

code(r"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from chimeraboost import ChimeraBoostRegressor

ROOT = Path("A:/code/spmm")
sys.path.insert(0, str(ROOT / "src"))

from eracoef.config import load_config
from eracoef.holdout import Context, Ratings, predict_season, score
from eracoef.inseason import season_frac
from eracoef.looseason import LeaveSeasonOutRAPM
from eracoef.priorridge import PriorRidgeCV, armse

config = load_config(ROOT / "config.yaml")
context = Context.load(config)
pd.set_option("display.width", 220, "display.max_columns", 50, "display.precision", 4)
""")

md(r"""
## 1. The panel

One row per `(player_id, season, side)`. `side` is `"O"` or `"D"`, both in raw sign: offense adds points
scored, defense adds points **allowed**, so a good defender is negative.

Built by `python scripts/49_role_panel.py --season`.
""")

code(r"""
panel = pd.read_parquet(ROOT / "outputs/role_panel_season.parquet")
panel = panel[panel.poss > 0].reset_index(drop=True)

box_score    = ["fg3m", "fg3_miss", "fg2m", "fg2_miss", "ftm", "ft_miss",
                "orb", "drb", "ast", "tov", "stl", "blk", "pf"]
playing_time = ["poss_pct", "gs_pct", "age"]
shot_quality = [c for c in panel.columns if c.startswith("shot_")]
body         = ["height", "weight", "draft_pick", "exp_yrs", "entry_age", "tenure", "n_teams"]
on_court     = [c for c in panel.columns if c.startswith("onc_")]

features = box_score + playing_time + shot_quality + body + on_court

print(panel.shape, "seasons", panel.season.min(), "-", panel.season.max())
print(len(features), "features")
assert not any(c.startswith("past_") for c in features), "single year or bust"
""")

md(r"""
## 2. The target: a RAPM of every season except this one

Not `rapm1`, the old per-player-season target. That one is 97.6% correlated with `spm`, a linear box-score
fit built from the very features the booster is handed, so a model "predicts" it at r ≈ 0.99 by relearning
an equation. And one season of plus-minus for a bench player is mostly noise.

Instead: one rating per **player**, fit on every season except H, keeping players with at least
**100 possessions** (a possession is one offensive *and* one defensive trip, counted once).

With player units the normal equations are additive over seasons, so this is not thirty refits —
accumulate each season's gram once, then leaving H out is a subtraction. Each season's context
(intercept, home, playoff, garbage time, margin) is projected out *before* it is accumulated, so every
season keeps its own level.
""")

code(r"""
seasons = list(range(1997, 2027))
designs = {}


def design_for(s):
    if s not in designs:
        designs[s] = context.design([s], "pts")      # ~2s each; all 30 is a few hundred MB
    return designs[s]


rapm = LeaveSeasonOutRAPM(min_possessions=100.0)
for s in seasons:
    rapm.add_season(s, design_for(s))

print(f"{len(seasons)} seasons, {len(rapm.player_ids)} players")
""")

md(r"""
### Its penalty, swept on seasons it has not seen

The penalty matters more here than anywhere else: too heavy and the prior learns to predict a shrunken
thing, too light and it learns noise. So it is scored the way the board is scored — fit without season S,
predict S's games, take the team-game error weighted toward close games.

### Sweep these on the end-to-end score, not on the RAPM's own accuracy

Pooled over five seasons, building the prior and the board each time and scoring the held-out 25%.
**game ARMSE, points per 100 possessions — lower is better.** Baseline with no player ratings at all:
8.9261 points per 100 possessions.

| target defence λ → | 10,000 | 20,000 | **40,000** | 80,000 | 160,000 |
|---|---|---|---|---|---|
| offence 10,000 | 8.4675 | 8.4440 | 8.4311 | 8.4588 | 8.5235 |
| **offence 40,000** | 8.3854 | 8.3662 | **8.3503** | 8.3699 | 8.4358 |
| offence 160,000 | 8.4052 | 8.3825 | **8.3483** | 8.3751 | 8.4321 |
| offence 640,000 | 8.4523 | 8.4359 | 8.4118 | 8.4348 | 8.4901 |
| offence 2,560,000 | 8.4527 | 8.4284 | 8.4053 | 8.4267 | 8.4807 |
| offence 10,240,000 | 8.4583 | 8.4305 | 8.4153 | 8.4269 | 8.4884 |

**Both axes are interior now** — the surface rises in every direction from the middle, so neither is
sitting on a grid edge. **Defence is 40,000 and it is decisive**: moving one step either way costs
0.016 to 0.086 points per 100 possessions.

**Offence is bracketed but still not identified.** 40,000 and 160,000 sit 0.0020 points per 100
possessions apart, they split the five seasons 2–3, and the per-season swing between them is ±0.1 —
so that gap is noise. **40,000 is taken**, because 160,000 halves the board's own offensive spread
(sd 0.87 → 0.45 points per 100 possessions) for no measurable accuracy, and a board that compressed
says less about players.

**Context 0 wins again**: 8.3483 at 0, 8.3647 at 1,000, worse at 100,000. Three sweeps, same answer.

---

**Still open: the board's own penalties.** In this run `PriorRidgeCV` pinned its offensive penalty at
the grid's top in 24% of fits and its defensive penalty in 25%. With a multi-season prior this good and
only three quarters of one season of games, the board often wants to barely update the prior at all.
The default grids now run to 1,000,000,000, where the residual is numerically nil — so the top of the
grid *is* "keep the prior unchanged" — but how often it gets chosen has not been re-measured.

---

**Why the RAPM's own `sweep` is the wrong tool here---

**Why the RAPM's own `sweep` is the wrong tool here, kept as a cautionary record.** It
scores the penalty that makes this RAPM the best *direct predictor* of an unseen season. That is not the
penalty that makes it the best *training target for a prior*, and the two disagree in a predictable
direction: heavier shrinkage gives a lower-variance RAPM that predicts better and teaches the booster to
predict shrunken values, which makes a narrower and worse prior. Measured on 2015, taking this sweep's
answer (28,690 / 69,711) instead of 43,089 / 43,089 narrowed the defensive prior from sd 0.78 to 0.62 and
cost **0.017 game ARMSE**, with `PriorRidgeCV` responding by driving its own defensive penalty to the grid
ceiling — the ridge finding nothing useful to do with the data.

Choose these penalties on the end-to-end score instead (build the prior, fit the board, score the held-out
25%). What the RAPM-accuracy sweep found, for the record:

| | | |
|---|---|---|
| **offense** | 28,690 | interior — 11,808 reads 8.5311 and 69,711 reads 8.5225 |
| **defense** | 69,711 | **heavier than offense, ratio 2.43.** The hard-coded value was 0.625, i.e. the wrong side |
| **context** | 0 | unpenalised, and not close: 0 → 8.5134, 1e3 → 8.5145, 1e5 → 8.7037, 1e7 → 9.0879 |

Two caveats worth carrying. The defense axis is **flat** — at offense 28,690, a defense penalty of 28,690
reads 8.5137 against 69,711's 8.5134, so 2.43 is not really identified; what *is* clear is that anything
lighter than offense is worse, so 0.625 was pointing the wrong way. And "context = 0 on a grid whose edge
is 0" is not a boundary problem: you cannot penalise less than not at all, and the curve rises monotonically
from there.

Each triple is a ~5,900 × 5,900 solve per scoring season, so widen the grid only if you mean it.
""")

code(r"""
# Swept on the END-TO-END score -- build the prior, fit the board, score the held-out 25% -- pooled over
# five seasons.  40,000 / 40,000 is a dead heat with the grid's top offense value (8.3501 vs 8.3492) and
# is interior, so it is the one to hold.  See the note above before changing these.
RAPM_OFFENSE_LAMBDA = 40000.0
RAPM_DEFENSE_LAMBDA = 40000.0     # interior and clear: 10,000 reads 8.3881, 160,000 reads 8.4295
RAPM_CONTEXT_LAMBDA = 0.0         # unpenalised context wins on both objectives

# grid = np.logspace(np.log10(2000), 6, 8)
# sweep = rapm.sweep(design_for, offense_lambdas=grid, defense_lambdas=grid,
#                    context_lambdas=[0.0, 1e3, 1e5, 1e7], scoring_seasons=[2000, 2005, 2010, 2015, 2020])
# print(sweep.head(10).to_string(index=False))
""")

md(r"""
## 3. The prior

One training row per player: his possession-weighted feature averages over every season except H, mapped
to his RAPM over those same seasons. Then asked about season H's box score.

There is no season-rebalancing here any more — the training set has no season dimension left to rebalance.
""")

code(r"""
chimera_offense = dict(config["gbdt"]["params"])
chimera_defense = dict(config["gbdt"]["params_def"])


def aggregate_features(held_out_season, side):
    # one row per player: his features averaged over every season but held_out_season, by possessions
    rows = panel[(panel.side == side) & (panel.season != held_out_season)]
    totals = rows[features].mul(rows.poss, axis=0).groupby(rows.player_id).sum()
    return totals.div(rows.groupby("player_id").poss.sum(), axis=0)
""")

code(CELL_FIT_PRIOR)

code(r"""
season = 2015

prior, training = {}, {}
for side in ("O", "D"):
    prior[side], _, training[side] = fit_prior(season, side)
    print(f"{side}: trained on {len(training[side])} players, predicting {len(prior[side])}   "
          f"prediction sd {prior[side].prediction.std():.3f}")

prior_offense = dict(zip(prior["O"].player_id, prior["O"].prediction))
prior_defense = dict(zip(prior["D"].player_id, prior["D"].prediction))
""")

md(r"""
### The covariate shift, measured

Career averages are narrower than single seasons, and the booster will be asked about the wider thing.
A ratio much above 1.0 means the prior is being asked to extrapolate, which a tree cannot do.
""")

code(r"""
one_season = panel[(panel.side == "O") & (panel.season == season)][features].std()
career = training["O"][features].std()
shift = pd.DataFrame({"one season": one_season, "career average": career,
                      "ratio": one_season / career.replace(0, np.nan)})
print(shift.sort_values("ratio", ascending=False).head(10).round(3).to_string())
print(f"median ratio across all {len(features)} features: {shift.ratio.median():.2f}")
""")

md(r"""
## 4. The season, split

Fit on the first 75% of the games, score the last 25%.
""")

code(r"""
full_season = design_for(season)
game_position = season_frac(full_season.games)[full_season.rows["game_idx"].to_numpy()]
fit_games = full_season.subset(game_position < 0.75)
score_games = full_season.subset(game_position >= 0.75)

print(f"{full_season.spec.n_ps} players, {fit_games.X.shape[0]} stints to fit, "
      f"{score_games.X.shape[0]} to score")
""")

md(r"""
## 5. The ratings

`PriorRidgeCV` (`src/eracoef/priorridge.py`) is a ridge with three changes from `RidgeCV`:

1. it shrinks toward the **prior**, so a player with no minutes lands on his box score, not on zero;
2. **three penalties, swept independently** — offense, defense, and the context block (season intercept,
   home, playoff, garbage time, margin). Defense used to be pinned at 0.6245 × offense and the context at
   exactly 0; both are grids now, and 0 is still on the context grid so the old fit stays reachable.

The penalty is cross-validated over whole **games**, because the same ten players repeat across a game's
stints and splitting inside one leaks the answer.

**What the CV is minimising.** Not stint error — that is mostly binomial noise. It sums each team's points
over its rows in a game, compares that with what the ratings predicted, and weights the team-game by its
possessions divided by the game's **average |margin|**. A 30-point blowout says much less about who is
good than a game decided by two.

The number printed is **ARMSE** — the root of the weighted mean square, scaled by `sqrt(2/pi)` so it sits
on the mean-absolute scale. Read it as *"a typical team-game misses by this many points per 100"*.

**Read the whole curve, not just the argmax.** With a good prior and three quarters of one season, this
objective is nearly flat above ~1e5 — everything up there is within 0.002 ARMSE, and at those penalties the
rating essentially *is* the prior. The grid runs to 2e6 so the minimum is bracketed rather than sitting on
an edge, but a flat curve means the games cannot tell you much about this dial, not that a huge penalty won.
""")

code(r"""
ridge = PriorRidgeCV().fit(fit_games, prior_offense, prior_defense)

print(f"offense {ridge.offense_lambda_:>12,.0f}")
print(f"defense {ridge.defense_lambda_:>12,.0f}   (ratio to offense "
      f"{ridge.defense_lambda_ / ridge.offense_lambda_:.3f}; the old hard-coded value was 0.625)")
print(f"context {ridge.context_lambda_:>12,.0f}   (0 = the old unpenalised context)")
print(f"average |margin| per game: {ridge.average_margin_.mean():.2f} points")
print(ridge.cv_armse_.head(8).to_string(index=False))
""")

code(r"""
names = pd.read_parquet(ROOT / "artifacts/season_ratings.parquet",
                        columns=["player_id", "player_name"]).drop_duplicates("player_id")
board = (ridge.ratings_.assign(total=lambda d: d.offense - d.defense)
         .merge(names, on="player_id", how="left")
         .sort_values("total", ascending=False))
print(board[["player_name", "offense", "defense", "total", "possessions"]].head(15).to_string(index=False))
""")

md(r"""
## 6. The score

**`game_armse`** — what a typical team-game misses by, in points per 100. **Lower is better.**
`base_armse` is the same with no player ratings at all, so the gap is what your board is worth.

Scored on the last 25% of the season, which the fit never saw.

**One season is one sample.** Loop before you believe a difference.
""")

code(r"""
def evaluate(name, offense=None, defense=None):
    model = PriorRidgeCV().fit(fit_games, offense, defense)
    ratings = Ratings(model.as_ratings_frame())
    result = score(predict_season(ratings, score_games, level="home"))
    return dict(system=name, game_armse=armse(result["tg"]), base_armse=armse(result["tg_base"]),
                offense_lambda=model.offense_lambda_, defense_lambda=model.defense_lambda_,
                context_lambda=model.context_lambda_,
                sd_offense=ratings.df.o.std(), sd_defense=ratings.df.d.std())


print(pd.DataFrame([evaluate("no prior"),
                    evaluate("your prior", prior_offense, prior_defense)]).round(4).to_string(index=False))
""")

md(r"""
## 7. Every season

About a minute a season. Start with five.
""")

code(r'''
def run(season_list):
    results = []
    for s in season_list:
        priors = {side: fit_prior(s, side)[0] for side in ("O", "D")}
        design_for(s)
        offense = dict(zip(priors["O"].player_id, priors["O"].prediction))
        defense = dict(zip(priors["D"].player_id, priors["D"].prediction))

        design = design_for(s)
        position = season_frac(design.games)[design.rows["game_idx"].to_numpy()]
        early, late = design.subset(position < 0.75), design.subset(position >= 0.75)

        for name, args in (("no prior", (None, None)), ("your prior", (offense, defense))):
            model = PriorRidgeCV().fit(early, *args)
            ratings = Ratings(model.as_ratings_frame())
            result = score(predict_season(ratings, late, level="home"))
            results.append(dict(season=s, system=name, offense_lambda=model.offense_lambda_,
                                defense_lambda=model.defense_lambda_,
                                game_armse=armse(result["tg"]), base_armse=armse(result["tg_base"])))
        print(f"  {s} done", flush=True)
    return pd.DataFrame(results)


# results = run(range(2015, 2020))
# print(results.groupby("system")[["game_armse", "base_armse", "offense_lambda", "defense_lambda"]].mean().round(4))
# gap = results.pivot_table(index="season", columns="system", values="game_armse")
# print((gap["your prior"] - gap["no prior"]).round(4))
''')

md(r"""
## Knobs

| name | what it changes |
|---|---|
| `features` | what the prior sees. Drop `on_court` for box-score-only, `body` for production-only |
| `RAPM_OFFENSE_LAMBDA` / `RAPM_DEFENSE_LAMBDA` / `RAPM_CONTEXT_LAMBDA` | how hard the target is shrunk, per block. Re-sweep if you change the features, the seasons or the possession floor |
| `LeaveSeasonOutRAPM(min_possessions=)` | how many possessions a player needs to be in the target at all |
| `chimera_offense` / `chimera_defense` | the boosters, tuned values out of `config.yaml` |
| `PriorRidgeCV(offense_lambdas=, defense_lambdas=, context_lambdas=, n_folds=)` | the ridge's three grids |
| `PriorRidgeCV(weight_by_closeness=False)` | score every team-game by possessions alone, ignoring how close it was |
| `PriorRidgeCV(closeness_floor=)` | the smallest average margin the weighting will believe, in points |
| `0.75` in cell 4 | how much of the season the fit sees |

Deliberately absent: `gbdt_win_decay`, `PAST_DECAY`, `past_apm`, `past_poss`, `past_rapm`, the 3-season
block panel, the calibration map, and the `rapm1` target.
""")

nb["cells"] = CELLS
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"}}
out = Path(__file__).resolve().parent / "single_year.ipynb"
nbf.write(nb, out)
print("wrote", out, len(CELLS), "cells")
