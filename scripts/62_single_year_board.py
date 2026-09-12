"""Build the whole board from the single-year pipeline: LOSO RAPM -> SPM -> PriorRidgeCV, season by season.

    python scripts/62_single_year_board.py [--first=1997] [--last=2026] [--out=season_ratings_sy]

For each season H:
  1. the TARGET is a RAPM over every season except H, one rating per player (looseason.LeaveSeasonOutRAPM),
     at the penalties swept on the end-to-end score (DECISIONS.md);
  2. the PRIOR is a chimeraboost mapping a player's all-other-seasons feature averages to that target,
     then asked about H's own box score -- the SPM step;
  3. the RATING is PriorRidgeCV on H's games, shrinking toward that prior, its three penalties chosen by
     cross-validation over whole games on a closeness-weighted team-game objective.

Nothing that touches season H is ever fit on season H, and no player's own other seasons reach his H
rating except through population-level model coefficients ("single year or bust", HANDOFF ruling 12).

Writes outputs/<out>.parquet with one row per player-season, which scripts/52_site.py reads.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from chimeraboost import ChimeraBoostRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eracoef.config import load_config  # noqa: E402
from eracoef.holdout import Context, Ratings  # noqa: E402
from eracoef.looseason import LeaveSeasonOutRAPM  # noqa: E402
from eracoef.priorridge import PriorRidgeCV  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

# swept on the end-to-end score over five seasons; the surface is in DECISIONS.md
RAPM_OFFENSE_LAMBDA = 40000.0
RAPM_DEFENSE_LAMBDA = 40000.0
RAPM_CONTEXT_LAMBDA = 0.0
MIN_POSSESSIONS = 100.0

BOARD_PLAYER_LAMBDAS = np.round(np.logspace(np.log10(2000.0), np.log10(1.0e9), 8), 0)
BOARD_CONTEXT_LAMBDAS = np.array([0.0, 1.0e3])


def _flag(name, default=None):
    hit = [a for a in sys.argv[1:] if a.startswith(f"--{name}=")]
    return hit[0].split("=", 1)[1] if hit else default


def main():
    cfg = load_config(ROOT / "config.yaml")
    ctx = Context.load(cfg)
    first, last = int(_flag("first", 1997)), int(_flag("last", 2026))
    out = ROOT / "outputs" / f"{_flag('out', 'season_ratings_sy')}.parquet"
    seasons = list(range(first, last + 1))

    panel = pd.read_parquet(ROOT / "outputs/role_panel_season.parquet")
    panel = panel[panel.poss > 0].reset_index(drop=True)
    box = ["fg3m", "fg3_miss", "fg2m", "fg2_miss", "ftm", "ft_miss", "orb", "drb", "ast", "tov",
           "stl", "blk", "pf"]
    features = (box + ["poss_pct", "gs_pct", "age"]
                + [c for c in panel.columns if c.startswith("shot_")]
                + ["height", "weight", "draft_pick", "exp_yrs", "entry_age", "tenure", "n_teams"]
                + [c for c in panel.columns if c.startswith("onc_")])
    assert not any(c.startswith("past_") for c in features), "single year or bust"

    t0 = time.time()
    designs = {}
    rapm = LeaveSeasonOutRAPM(min_possessions=MIN_POSSESSIONS)
    for s in seasons:
        designs[s] = ctx.design([s], "pts")
        rapm.add_season(s, designs[s])
    print(f"accumulated {len(seasons)} seasons, {len(rapm.player_ids)} players "
          f"({time.time() - t0:.0f}s)", flush=True)

    rows = []
    for season in seasons:
        target = rapm.ratings(held_out_season=season, offense_lambda=RAPM_OFFENSE_LAMBDA,
                              defense_lambda=RAPM_DEFENSE_LAMBDA, context_lambda=RAPM_CONTEXT_LAMBDA)
        prior = {}
        for side, params in (("O", cfg["gbdt"]["params"]), ("D", cfg["gbdt"]["params_def"])):
            column = "offense" if side == "O" else "defense"
            others = panel[(panel.side == side) & (panel.season != season)]
            aggregate = (others[features].mul(others.poss, axis=0).groupby(others.player_id).sum()
                         .div(others.groupby("player_id").poss.sum(), axis=0))
            train = aggregate.join(target.set_index("player_id")[[column, "possessions"]],
                                   how="inner").dropna()
            model = ChimeraBoostRegressor(random_state=0, **dict(params))
            model.fit(train[features].to_numpy(float), train[column].to_numpy(float),
                      sample_weight=train.possessions.to_numpy(float))
            held = panel[(panel.side == side) & (panel.season == season)]
            prior[side] = dict(zip(held.player_id.to_numpy(),
                                   model.predict(held[features].to_numpy(float))))

        ridge = PriorRidgeCV(offense_lambdas=BOARD_PLAYER_LAMBDAS, defense_lambdas=BOARD_PLAYER_LAMBDAS,
                             context_lambdas=BOARD_CONTEXT_LAMBDAS, n_folds=5).fit(
            designs[season], prior["O"], prior["D"])
        table = ridge.ratings_.assign(season=season, offense_lambda=ridge.offense_lambda_,
                                      defense_lambda=ridge.defense_lambda_,
                                      context_lambda=ridge.context_lambda_)
        rows.append(table)
        print(f"  {season}: {len(table)} players, lambdas "
              f"{ridge.offense_lambda_:,.0f} / {ridge.defense_lambda_:,.0f} / "
              f"{ridge.context_lambda_:,.0f}  ({time.time() - t0:.0f}s)", flush=True)
        pd.concat(rows, ignore_index=True).to_parquet(out, index=False)

    board = pd.concat(rows, ignore_index=True)
    # 52_site.py's schema: raw sign in, positive-good out, one row per player-season
    board = board.rename(columns={"possessions": "poss_off"})
    board["poss_season"] = board.poss_off
    board["poss_def"] = board.poss_off
    board["rating_off"] = board.offense
    board["rating_def"] = -board.defense                 # positive good on both ends
    board["rating_total"] = board.rating_off + board.rating_def
    board["prior_off"] = board.prior_offense
    board["prior_def"] = -board.prior_defense
    board["prior_total"] = board.prior_off + board.prior_def
    board["u_off"] = board.rating_off - board.prior_off
    board["u_def"] = board.rating_def - board.prior_def
    board["u_total"] = board.u_off + board.u_def
    board.to_parquet(out, index=False)
    print(f"\nwrote {out}: {len(board)} rows, {board.season.nunique()} seasons "
          f"({time.time() - t0:.0f}s)")

    names = pd.read_parquet(ROOT / "artifacts/season_ratings.parquet",
                            columns=["player_id", "player_name"]).drop_duplicates("player_id")
    pd.set_option("display.width", 200, "display.max_columns", 20, "display.precision", 3)
    for season in (seasons[-1], 2015 if 2015 in seasons else seasons[0]):
        top = (board[board.season == season].merge(names, on="player_id", how="left")
               .sort_values("rating_total", ascending=False))
        print(f"\n=== {season}, top 12 (points per 100 possessions, positive good on both ends)")
        print(top[["player_name", "rating_off", "rating_def", "rating_total",
                   "poss_off"]].head(12).to_string(index=False))


if __name__ == "__main__":
    main()
