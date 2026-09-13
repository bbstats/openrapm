"""Build the whole board from the single-year pipeline: LOSO RAPM -> SPM -> PriorRidgeCV, season by season.

    python scripts/62_single_year_board.py [--first=1997] [--last=2026] [--out=season_ratings_sy]
                                           [--score=1] [--target_off=xpts_ft] [--target_def=x3def_w0.25]
                                           [--free_scale=1] [--buckets=low_poss:2] [--boards=2015,2024]
                                           [--features=boruta_noonc|boruta|sy_noonc|sy]

`--first` / `--last` are the seasons the TARGET is accumulated over and must stay the full range -- a
leave-one-season-out RAPM with one season in it has nothing left.  `--boards=` restricts which seasons a
rating is produced for, which is how a change is measured on two seasons instead of thirty.

For each season H:
  1. the TARGET is a RAPM over every season except H, one rating per player (looseason.LeaveSeasonOutRAPM),
     at the penalties swept on the end-to-end score (DECISIONS.md);
  2. the PRIOR is a chimeraboost mapping a player's all-other-seasons feature averages to that target,
     then asked about H's own box score -- the SPM step.  What it sees is `singleyear.feature_set()`,
     by default the BorutaShap selection with the four `onc_*` columns removed (see `singleyear` for why
     the most important-looking feature in the set is a leak);
  3. the RATING is PriorRidgeCV on H's games, shrinking toward that prior, its three penalties chosen by
     cross-validation over whole games on a closeness-weighted team-game objective.

Nothing that touches season H is ever fit on season H, and no player's own other seasons reach his H
rating except through population-level model coefficients ("single year or bust", HANDOFF ruling 12).

**Two targets, one per side.**  The offensive fit explains `xpts_ft` -- points with made free throws
replaced by the shooter's padded expectation -- and the defensive fit explains `x3def_w0.25`, which also
replaces three quarters of every opponent three-point make by the shooter's own padded 3P%.  Both are
measured variance reductions on this project's own criterion (+0.26 and -0.39 to -0.47 per 100;
DECISIONS.md 37-52), which is one to two orders of magnitude more than the plus-minus stage is currently
worth.  So the board runs two RAPMs and two ridges and takes one side from each.  `--target_off=pts
--target_def=pts` reproduces the raw-points board.

**`--score` (on by default) is the diagnostic that can see compression.**  Per season it refits both sides
on the first 75% of the games and scores the last 25% AGAINST ACTUAL POINTS -- the luck adjustment is a
fitting target, never a scoring one -- reporting `game_armse` and `scale_off` / `scale_def`, what the unseen
quarter wants each side multiplied by.  A team-game total is linear in a per-player linear map, so
`game_armse` is nearly blind to a uniform rescale of the board; the scales are not.  Above 1.0 = too narrow.

Writes outputs/<out>.parquet with one row per player-season, which scripts/52_site.py reads, and
outputs/<out>_score.parquet with the per-season diagnostic.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from chimeraboost import ChimeraBoostRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eracoef import singleyear as sy  # noqa: E402
from eracoef.config import load_config  # noqa: E402
from eracoef.holdout import Context, Ratings, predict_season, score  # noqa: E402
from eracoef.inseason import season_frac  # noqa: E402
from eracoef.looseason import LeaveSeasonOutRAPM  # noqa: E402
from eracoef.priorridge import PriorRidgeCV, armse, calibration_miss  # noqa: E402
from eracoef.xshoot import DEFENSE_TARGETS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

# the RAPM penalties, the possession floor and the two targets live in `singleyear` so the Boruta run
# selects features against the same thing the board trains on
RAPM_OFFENSE_LAMBDA = sy.RAPM_OFFENSE_LAMBDA
RAPM_DEFENSE_LAMBDA = sy.RAPM_DEFENSE_LAMBDA
RAPM_CONTEXT_LAMBDA = sy.RAPM_CONTEXT_LAMBDA
MIN_POSSESSIONS = sy.MIN_POSSESSIONS

BOARD_PLAYER_LAMBDAS = np.round(np.logspace(np.log10(2000.0), np.log10(1.0e9), 8), 0)
BOARD_CONTEXT_LAMBDAS = np.array([0.0, 1.0e3])

SCORE_TARGET = "pts"        # the board is always SCORED on the points that were actually scored


def _flag(name, default=None):
    hit = [a for a in sys.argv[1:] if a.startswith(f"--{name}=")]
    return hit[0].split("=", 1)[1] if hit else default


def _target(name):
    """A `design.TARGETS` key, or the `xshoot.DEFENSE_TARGETS` callable of that name."""
    return DEFENSE_TARGETS[name] if name in DEFENSE_TARGETS else name


FREE_PRIOR_SCALE = True     # --free_scale=0 pins the prior at exactly the amplitude it came with
LAM_BUCKETS: dict = {}      # --buckets=low_poss:2 multiplies the bench's penalty


def _ridge(design, prior, free_prior_scale=None, lam_buckets=None):
    return PriorRidgeCV(offense_lambdas=BOARD_PLAYER_LAMBDAS, defense_lambdas=BOARD_PLAYER_LAMBDAS,
                        context_lambdas=BOARD_CONTEXT_LAMBDAS, n_folds=5,
                        free_prior_scale=FREE_PRIOR_SCALE if free_prior_scale is None else free_prior_scale,
                        lam_buckets=LAM_BUCKETS if lam_buckets is None else lam_buckets).fit(
        design, prior["O"], prior["D"])


def _first_75(design):
    position = season_frac(design.games)[design.rows["game_idx"].to_numpy()]
    return design.subset(position < 0.75), design.subset(position >= 0.75)


def main():
    cfg = load_config(ROOT / "config.yaml")
    ctx = Context.load(cfg)
    first, last = int(_flag("first", 1997)), int(_flag("last", 2026))
    out = ROOT / "outputs" / f"{_flag('out', 'season_ratings_sy')}.parquet"
    do_score = _flag("score", "1") not in ("0", "no", "false")
    global FREE_PRIOR_SCALE, LAM_BUCKETS
    FREE_PRIOR_SCALE = _flag("free_scale", "1") not in ("0", "no", "false")
    LAM_BUCKETS = {k: float(v) for k, v in
                   (part.split(":") for part in _flag("buckets", "").split(",") if part)}
    names = {"O": _flag("target_off", sy.OFFENSE_TARGET), "D": _flag("target_def", sy.DEFENSE_TARGET)}
    seasons = list(range(first, last + 1))
    boards = [int(x) for x in _flag("boards", "").split(",") if x] or seasons
    assert set(boards) <= set(seasons), f"--boards outside [{first}, {last}]"

    panel = pd.read_parquet(ROOT / "outputs/role_panel_season.parquet")
    panel = panel[panel.poss > 0].reset_index(drop=True)
    feature_set = _flag("features", "boruta")
    features = sy.feature_set(feature_set)
    for side, names_ in features.items():
        assert not any(c.startswith("past_") for c in names_), "single year or bust"
    print(f"targets: offense {names['O']}, defense {names['D']}; features {feature_set} "
          f"(O {len(features['O'])}, D {len(features['D'])}); "
          f"free_prior_scale {FREE_PRIOR_SCALE}; lam_buckets {LAM_BUCKETS or '{}'}", flush=True)

    t0 = time.time()
    # One accumulator per TARGET, because the two sides explain different things.  The designs are NOT
    # held: thirty of them is several gigabytes and the run pages, while `LeaveSeasonOutRAPM` only needs
    # each season's normal equations (a 1,000 x 1,000 gram, ~8 MB).  `ctx.design` caches the last four,
    # and a rebuild is a couple of seconds off the disk cache.
    def design_for(name, season):
        return ctx.design([season], _target(name))

    rapm = {}
    for name in dict.fromkeys(names.values()):
        rapm[name] = LeaveSeasonOutRAPM(min_possessions=MIN_POSSESSIONS)
        for s in seasons:
            rapm[name].add_season(s, design_for(name, s))
        print(f"  accumulated {len(seasons)} seasons on {name}, "
              f"{len(rapm[name].player_ids)} players ({time.time() - t0:.0f}s)", flush=True)

    rows, diagnostics = [], []
    for season in boards:
        prior, table, lam = {}, {}, {}
        for side, params in (("O", cfg["gbdt"]["params"]), ("D", cfg["gbdt"]["params_def"])):
            column = "offense" if side == "O" else "defense"
            target = rapm[names[side]].ratings(
                held_out_season=season, offense_lambda=RAPM_OFFENSE_LAMBDA,
                defense_lambda=RAPM_DEFENSE_LAMBDA, context_lambda=RAPM_CONTEXT_LAMBDA)
            feats = features[side]
            train = sy.prior_rows(target, panel[(panel.side == side) & (panel.season != season)],
                                  column, feats)
            model = ChimeraBoostRegressor(random_state=0, **dict(params))
            model.fit(train[feats].to_numpy(float), train.target.to_numpy(float),
                      sample_weight=train.weight.to_numpy(float))
            held = sy.season_frame(panel[(panel.side == side) & (panel.season == season)], feats)
            prior[side] = dict(zip(held.player_id.to_numpy(),
                                   model.predict(held[feats].to_numpy(float))))

        # one ridge per side's target; each contributes only its own half of the board
        scale = {}
        # one fit per DISTINCT target: when both sides explain the same thing (--target_off=pts
        # --target_def=pts) the second fit would be the first one over again
        fits = {name: _ridge(design_for(name, season), prior) for name in dict.fromkeys(names.values())}
        for side in ("O", "D"):
            ridge = fits[names[side]]
            table[side] = ridge.ratings_
            lam[side] = (ridge.offense_lambda_, ridge.defense_lambda_, ridge.context_lambda_)
            # how far the season's own games decided to trust the prior on this side.  Above 1 means the
            # prior was compressed and the games stretched it; 1.0 exactly means the lever is off.
            scale[side] = (ridge.prior_scale_[0 if side == "O" else 1]
                           if ridge.prior_scale_ is not None else 1.0)
        merged = (table["O"][["player_id", "offense", "prior_offense", "possessions"]]
                  .merge(table["D"][["player_id", "defense", "prior_defense"]], on="player_id"))
        rows.append(merged.assign(season=season, offense_lambda=lam["O"][0],
                                  defense_lambda=lam["D"][1], context_lambda=lam["O"][2],
                                  prior_scale_off=scale["O"], prior_scale_def=scale["D"]))
        print(f"  {season}: {len(merged)} players, lambdas O {lam['O'][0]:,.0f} / "
              f"D {lam['D'][1]:,.0f} / ctx {lam['O'][2]:,.0f},  prior_scale "
              f"{scale['O']:.2f} / {scale['D']:.2f}  ({time.time() - t0:.0f}s)", flush=True)

        if do_score:
            diagnostics.append(_diagnose(season, design_for, names, prior))
            d = diagnostics[-1]
            print(f"      75/25 on points: game_armse {d['game_armse']:.4f} (base {d['base_armse']:.4f})"
                  f"  scale {d['scale_off']:.2f} / {d['scale_def']:.2f}  miss {d['miss']:.3f}", flush=True)
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
    # `player_name`, because both scripts/52_site.py and tests/test_vs_consensus.py key on it and a board
    # of bare player_ids silently fails them both
    board = board.merge(_names(), on="player_id", how="left")
    board.to_parquet(out, index=False)
    print(f"\nwrote {out}: {len(board)} rows, {board.season.nunique()} seasons "
          f"({time.time() - t0:.0f}s)")

    pd.set_option("display.width", 200, "display.max_columns", 20, "display.precision", 3)
    if diagnostics:
        D = pd.DataFrame(diagnostics)
        D.to_parquet(out.with_name(out.stem + "_score.parquet"), index=False)
        print("\n=== 75/25 within-season diagnostic, scored on actual points "
              "(game_armse decides; miss breaks ties)")
        print(D.round(4).to_string(index=False))
        print(f"\npooled: game_armse {D.game_armse.mean():.4f}  base {D.base_armse.mean():.4f}  "
              f"scale_off {D.scale_off.mean():.3f}  scale_def {D.scale_def.mean():.3f}  "
              f"miss {D['miss'].mean():.3f}")

    wide = board[board.poss_off >= 1000]
    print("\n=== board spread, players with 1,000+ possessions (the shipped board reads 1.317 / 1.434)")
    print(f"  rating_off sd {wide.rating_off.std():.3f}   rating_def sd {wide.rating_def.std():.3f}")
    print(f"  prior_off  sd {wide.prior_off.std():.3f}   prior_def  sd {wide.prior_def.std():.3f}")
    print(f"  u_off      sd {wide.u_off.std():.3f}   u_def      sd {wide.u_def.std():.3f}"
          "        <- what the season's own games added")
    if "prior_scale_off" in board:
        by_season = board.groupby("season")[["prior_scale_off", "prior_scale_def"]].first()
        print(f"  prior_scale: offense {by_season.prior_scale_off.mean():.2f} "
              f"(min {by_season.prior_scale_off.min():.2f}, max {by_season.prior_scale_off.max():.2f}), "
              f"defense {by_season.prior_scale_def.mean():.2f} "
              f"(min {by_season.prior_scale_def.min():.2f}, max {by_season.prior_scale_def.max():.2f})")

    for season in dict.fromkeys([boards[-1], 2015 if 2015 in boards else boards[0]]):
        top = board[board.season == season].sort_values("rating_total", ascending=False)
        print(f"\n=== {season}, top 12 (points per 100 possessions, positive good on both ends)")
        print(top[["player_name", "rating_off", "rating_def", "rating_total",
                   "poss_off"]].head(12).to_string(index=False))


def _names() -> pd.DataFrame:
    return pd.read_parquet(ROOT / "artifacts/season_ratings.parquet",
                           columns=["player_id", "player_name"]).drop_duplicates("player_id")


def _diagnose(season, design_for, names, prior) -> dict:
    """Refit each side on the first 75% of its own target, score the last 25% on ACTUAL POINTS.

    The two sides come from two fits, so the board scored here is assembled the way the shipped one is:
    offense out of the offensive target's ridge, defense out of the defensive one's.
    """
    fits = {}
    for name in dict.fromkeys(names.values()):
        fit_games, _ = _first_75(design_for(name, season))
        fits[name] = _ridge(fit_games, prior).as_ratings_frame()
    frame = (fits[names["O"]][["player_id", "o", "poss", "prior_o"]]
             .merge(fits[names["D"]][["player_id", "d", "prior_d"]], on="player_id"))
    _, score_games = _first_75(design_for(SCORE_TARGET, season))
    result = score(predict_season(Ratings(frame), score_games, level="home"))
    return dict(season=season, game_armse=armse(result["tg"]), base_armse=armse(result["tg_base"]),
                scale_off=result["scale_off"], scale_def=result["scale_def"],
                calib_side=result["calib_side"],
                miss=calibration_miss(result["scale_off"], result["scale_def"]),
                sd_o=float(frame.o.std()), sd_d=float(frame.d.std()))


if __name__ == "__main__":
    main()
