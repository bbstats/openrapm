"""Emit notebooks/single_year.ipynb.  Edit here, re-run, and the notebook is regenerated."""
from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()
CELLS = []


def md(text):
    CELLS.append(nbf.v4.new_markdown_cell(text.strip("\n")))


def code(text):
    CELLS.append(nbf.v4.new_code_cell(text.strip("\n")))


md(r"""
# Single year or bust

A player's rating for season H comes from H's games and from no games of his own in any other season.
No `past_*` features, no career pooling, one row per player-season.

Two models, both yours to rewrite:

* the **prior** — `ChimeraBoostRegressor` on a player's season box score, trained on every season but H
  (and H's rebalancing partner season),
* the **ratings** — `PriorRidgeCV`, a ridge that shrinks toward the prior instead of toward zero and
  picks its own penalty.

Fit on the first 75% of H's games, score the last 25%.
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
from eracoef.holdout import Context, Ratings, player_scores, player_truth, predict_season, score
from eracoef.inseason import season_frac
from eracoef.priorridge import PriorRidgeCV
from eracoef.rloocv import rebalance_partners

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
target = "rapm1"

print(panel.shape, "seasons", panel.season.min(), "-", panel.season.max())
print(len(features), "features")
assert not any(c.startswith("past_") for c in features), "single year or bust"
""")

md(r"""
## 2. Pick a target with your eyes open

| target | what it is | sd | corr with `spm` |
|---|---|---|---|
| `apm` | raw adjusted plus-minus, barely shrunk | 4.19 | 0.41 |
| `spm` | the *linear* box-score prediction, leave-season-out | 1.57 | 1.00 |
| `u` | the ridge residual — what the box score **missed** | 0.35 | 0.02 |
| `rapm1` | `spm + u`, the shipped target | 1.62 | **0.976** |

`rapm1` is 97.6% `spm`, and `spm` is a linear function of features you are handing the model. So it will
"predict" `rapm1` at r ≈ 0.99 and **that number means nothing**. Read `corr(prediction, apm)` instead,
or go straight to the scoring cell.
""")

code(r"""
offense_rows = panel[panel.side == "O"]
print(pd.DataFrame({t: {"sd": offense_rows[t].std(),
                        "corr_with_spm": np.corrcoef(offense_rows[t], offense_rows.spm)[0, 1]}
                    for t in ("apm", "spm", "u", "rapm1")}).T.round(3).to_string())
""")

md(r"""
## 3. The prior

Trained on every season except H — and except H's **rebalancing partner**.

Why the partner: drop season H and the remaining mean shifts away from H, by arithmetic. Dropping one
more season on the other side puts it back (Austin, Pe'er & Korem 2025, `eracoef.rloocv`). Measured over
27 seasons it was worth +0.0009 within-roster tau at z +1.4 — consistently positive, never significant,
and it costs 3% of the training rows. It is on because it is the honest default, not because it is big.
""")

code(r"""
seasons = np.unique(panel.season)
partner_season = {}
for side in ("O", "D"):
    rows = panel[panel.side == side]
    partners = rebalance_partners(rows[target].to_numpy(), rows.poss.to_numpy(), rows.season.to_numpy())
    partner_season[side] = {int(s): (None if p < 0 else int(seasons[p])) for s, p in zip(seasons, partners)}

chimera_offense = dict(config["gbdt"]["params"])
chimera_defense = dict(config["gbdt"]["params_def"])
""")

code(r'''
def fit_prior(held_out_season, side, rebalance=True):
    """Predict every player's season from a booster that never saw that season."""
    rows = panel[(panel.side == side) & (panel.season != held_out_season)]
    if rebalance and partner_season[side][held_out_season] is not None:
        rows = rows[rows.season != partner_season[side][held_out_season]]

    model = ChimeraBoostRegressor(random_state=0,
                                  **(chimera_offense if side == "O" else chimera_defense))
    model.fit(rows[features].to_numpy(dtype=float),
              rows[target].to_numpy(dtype=float),
              sample_weight=rows.poss.to_numpy(dtype=float),
              groups=rows.player_id.to_numpy())

    held = panel[(panel.side == side) & (panel.season == held_out_season)]
    return pd.DataFrame({"player_id": held.player_id.to_numpy(),
                         "prediction": model.predict(held[features].to_numpy(dtype=float)),
                         "apm": held.apm.to_numpy()}), model
''')

code(r"""
season = 2015

prior = {}
for side in ("O", "D"):
    prior[side], _ = fit_prior(season, side)
    print(f"{side}: {len(prior[side])} players   sd {prior[side].prediction.std():.3f}   "
          f"corr with apm {np.corrcoef(prior[side].prediction, prior[side].apm)[0, 1]:.3f}")

prior_offense = dict(zip(prior["O"].player_id, prior["O"].prediction))
prior_defense = dict(zip(prior["D"].player_id, prior["D"].prediction))
""")

md(r"""
## 4. The season, split

Fit on the first 75% of the games, score the last 25%.
""")

code(r"""
full_season = context.design([season], "pts")
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
2. the context columns — intercept, home, playoff, garbage time, margin — are **not** penalised;
3. defense gets its own penalty, `defense_penalty_ratio` times offense's.

The penalty is cross-validated over whole **games**, because the same ten players repeat across a game's
stints and splitting inside one leaks the answer.
""")

code(r"""
ridge = PriorRidgeCV().fit(fit_games, prior_offense, prior_defense)

print(f"penalty chosen: {ridge.alpha_:,.0f}")
print(ridge.cv_error_.rename("cv error").to_frame().T.to_string())
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

* **`tg`** — team-game criterion, points per 100. **Lower is better.** Weights a player by how much he
  played, so it cannot see the bottom of the board.
* **`tau`** — Kendall tau over pairs of **teammates**. Higher is better. Every player counts once.
* **`money_skill`** — of the money a coin-flip board misallocates on a cross-team trade, the share yours
  avoids. 0 is saying nothing, 1 is perfect.

The truth is a prior-free ridge on the scored games at `lam=100` — noisy, nearly unbiased, and carrying
no box prior, so it cannot flatter whichever prior you just built.

**One season is one sample.** Loop before you believe a difference.
""")

code(r"""
truth = player_truth(context, score_games, lam=100.0)


def evaluate(name, offense=None, defense=None):
    model = PriorRidgeCV().fit(fit_games, offense, defense)
    ratings = Ratings(model.as_ratings_frame())
    stint = score(predict_season(ratings, score_games, level="home"))
    player = player_scores(ratings, truth).set_index("group").loc["all"]
    return dict(system=name, tg=stint["tg"], tg_base=stint["tg_base"], penalty=model.alpha_,
                tau=player.tau, tau_league=player.tau_league, money_skill=player.money_skill,
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
        offense = dict(zip(priors["O"].player_id, priors["O"].prediction))
        defense = dict(zip(priors["D"].player_id, priors["D"].prediction))

        design = context.design([s], "pts")
        position = season_frac(design.games)[design.rows["game_idx"].to_numpy()]
        early, late = design.subset(position < 0.75), design.subset(position >= 0.75)
        actual = player_truth(context, late, lam=100.0)

        for name, args in (("no prior", (None, None)), ("your prior", (offense, defense))):
            model = PriorRidgeCV().fit(early, *args)
            ratings = Ratings(model.as_ratings_frame())
            player = player_scores(ratings, actual).set_index("group").loc["all"]
            results.append(dict(season=s, system=name, penalty=model.alpha_,
                                tg=score(predict_season(ratings, late, level="home"))["tg"],
                                tau=player.tau, tau_league=player.tau_league,
                                money_skill=player.money_skill))
        print(f"  {s} done", flush=True)
    return pd.DataFrame(results)


# results = run(range(2015, 2020))
# print(results.groupby("system")[["tg", "tau", "tau_league", "money_skill", "penalty"]].mean().round(4))
# gap = results.pivot_table(index="season", columns="system", values="tau")
# print((gap["your prior"] - gap["no prior"]).round(4))
''')

md(r"""
## Knobs

| name | what it changes |
|---|---|
| `features` | what the prior sees. Drop `on_court` for box-score-only, `body` for production-only |
| `target` | `rapm1` (shipped), `apm` (raw, noisy), `u` (what the box score missed) |
| `chimera_offense` / `chimera_defense` | the boosters, tuned values out of `config.yaml` |
| `fit_prior(..., rebalance=False)` | plain leave-season-out, no partner dropped |
| `PriorRidgeCV(alphas=, defense_penalty_ratio=, n_folds=)` | the ridge |
| `0.75` in cell 4 | how much of the season the fit sees |
| `player_truth(..., lam=)` | 100 is nearly unbiased and noisy; `config["lam_plugin"]` is low-variance but shrinks bench players harder than starters and flatters a board that does the same. **A verdict that flips between the two has decided nothing.** |

Deliberately absent: `gbdt_win_decay`, `PAST_DECAY`, `past_apm`, `past_poss`, `past_rapm`, the 3-season
block panel, the calibration map.
""")

nb["cells"] = CELLS
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"}}
out = Path(__file__).resolve().parent / "single_year.ipynb"
nbf.write(nb, out)
print("wrote", out, len(CELLS), "cells")
