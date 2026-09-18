"""Which box-score and play-by-play statistics predict what the trade set found the rating missing.

    python scripts/71_tradeset_features.py [--alpha=outputs/tradeset_alpha.parquet]
                                           [--panel=outputs/role_panel_season.parquet]
                                           [--features=boruta] [--player_folds=5]
                                           [--min_without=1] [--out=tradeset]

`scripts/70_tradeset.py` produces alpha: per player and season, what his team's games without him say
that his rating did not know.  This asks what a season's statistics can see of that.  A statistic that
predicts alpha is one the rating is not using well enough; a statistic that does not is not impactful,
however well it correlates with the rating itself.  That is the difference this is built to measure,
and it is the reason to have a trade set at all.

Two models, on the same rows, because they answer different halves of the question.  The booster
(chimeraboost, the same one the rankings' prior uses) says how MUCH is predictable, out of fold.  A
standardised ridge beside it says in which DIRECTION each statistic pushes, which importances cannot.

**The rating and the prior are features.**  If the only thing that predicts alpha is the rating itself,
the trade set is telling us the rating's amplitude is off and nothing more, and no statistic is being
missed.  Leaving them out would hide that and dress up a rescale as a discovery.

**Out-of-player folds.**  Every player is predicted by the fit that never saw a row of his, five folds
balanced on the label (`singleyear.stratified_player_folds`).  Without that a booster can return a
player's own alpha off his rate fingerprint and the R-squared means nothing.

Reads only.  Nothing is merged back into the panel.
Writes outputs/<out>_features.parquet (every feature, both models, per side) and
outputs/<out>_r2.parquet (how much of alpha is predictable, per side).
"""
import importlib.util
import os
import sys
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
os.environ.setdefault("NUMBA_NUM_THREADS", os.environ.get("OPENRAPM_NUMBA_THREADS", "4"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from eracoef import singleyear as sy  # noqa: E402
from eracoef.config import load_config
from eracoef.seasons import drop_untrainable  # noqa: E402

# `OutOfPlayerSPM` is the rankings' own prior fit, folds and all.  Imported rather than reimplemented so
# that "what predicts alpha" is measured by exactly the model that would have to use it.
_spec = importlib.util.spec_from_file_location("_board", ROOT / "scripts" / "62_single_year_board.py")
_board = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_board)
OutOfPlayerSPM = _board.OutOfPlayerSPM

SIDE_CODE = {"offense": "O", "defense": "D"}
EXTRA = ["rating", "rapm1_feature"]   # the rankings' own rating, and the panel's rapm1
                                      # (the role prior plus the season's own residual --
                                      # NOT the box prior the rankings are centred on)

# Features that measure HOW MUCH a player played rather than how well he played.  They have to be
# reported separately, because alpha is noisier and more shrunk for a player with less of it, so a
# feature naming those players can predict alpha's NOISE without knowing anything about basketball.
# This project has been caught by exactly that before: career experience scored the largest offline
# feature gain ever measured here and cost the criterion (DECISIONS.md, the measurement traps).
EXPOSURE = ["exp_poss", "exp_yrs", "poss_pct", "gs_pct", "tenure", "onc_poss_o", "onc_poss_d"]


# The shared command line (scripts/_cli.py): `flag` records every name it is asked for so
# `check_flags` below can refuse one that was never asked for.  A misspelled flag used to be
# ignored silently, which is how a run looks right and is wrong.
from _cli import check_flags, flag as _flag, switch   # noqa: E402


def weighted_r2(y, pred, weight) -> float:
    """Share of the weighted variance of alpha the prediction accounts for; 0 is the mean, below 0 is worse."""
    mean = float(np.average(y, weights=weight))
    residual = float(np.average((y - pred) ** 2, weights=weight))
    total = float(np.average((y - mean) ** 2, weights=weight))
    return 1.0 - residual / total if total > 0 else np.nan


def standardised_ridge(frame: pd.DataFrame, features: list, penalty: float = 1.0) -> np.ndarray:
    """One coefficient per feature, in points per 100 of alpha per standard deviation of the feature.

    Standardised so the coefficients are comparable across features on wildly different scales, and
    penalised lightly because several of these statistics are near-duplicates of each other.
    """
    X = frame[features].to_numpy(float)
    weight = frame.weight.to_numpy(float)
    mean = np.average(X, axis=0, weights=weight)
    sd = np.sqrt(np.average((X - mean) ** 2, axis=0, weights=weight))
    sd = np.where(sd > 0, sd, 1.0)
    Z = (X - mean) / sd
    y = frame.target.to_numpy(float) - float(np.average(frame.target, weights=weight))
    Zw = Z * weight[:, None]
    gram = Z.T @ Zw + penalty * np.eye(Z.shape[1]) * weight.sum() / len(weight)
    return np.linalg.solve(gram, Zw.T @ y)


def main():
    check_flags()      # refuse a flag this script does not understand (scripts/_cli.py)
    cfg = load_config(ROOT / "config.yaml")
    tag = _flag("out", "tradeset")
    # the alpha table follows the output tag unless it is named, so --out and --alpha
    # cannot silently disagree about which run is being read
    alpha = pd.read_parquet(ROOT / _flag("alpha", f"outputs/{tag}_alpha.parquet"))
    arm = str(alpha.team_effects.iloc[0]) if "team_effects" in alpha.columns else "unrecorded"
    print(f"alpha table: team effects {arm} (this script assumes 'none')")
    # The trust boundary (src/eracoef/seasons.py): rows whose unit reaches into a season still
    # being played may not reach a fit.  Gating at the read is what keeps every fit below honest;
    # it drops nothing while no season is in progress, and says so when it does.
    panel, _ = drop_untrainable(pd.read_parquet(ROOT / _flag("panel", "outputs/role_panel_season.parquet")),
                                cfg, what="the season panel")
    feature_sets = sy.feature_set(_flag("features", "boruta"))
    folds = int(_flag("player_folds", "5"))
    min_without = float(_flag("min_without", "1"))

    pd.set_option("display.width", 220, "display.max_columns", 30, "display.max_rows", 200,
                  "display.precision", 4)
    print(f"alpha: {len(alpha):,} rows, {alpha.season.nunique()} seasons, penalty "
          f"{alpha.penalty.iloc[0]:,.0f}")

    tables, scores = [], []
    for side, code in SIDE_CODE.items():
        features = list(feature_sets[code])
        rows = alpha[(alpha.side == side) & alpha.eligible & (alpha.without_poss >= min_without)]
        frame = sy.season_frame(panel[panel.side == code], features)
        frame = frame.merge(rows[["player_id", "season", "alpha_good", "rating", "with_poss",
                                  "without_poss"]], on=["player_id", "season"], how="inner")
        # The panel's `rapm1`, NOT the rankings' box prior (`prior_off`/`prior_def`): it is the
        # role prior plus the season's own residual, so it already contains what the season's
        # games said.  It was called `prior` here and reported as the rankings' prior, which it
        # is not.  A panel without it is one this script cannot read, so ask rather than zero it.
        if "rapm1" not in frame.columns:
            raise SystemExit("the panel has no `rapm1` column; rebuild it with scripts/49_role_panel.py")
        frame["rapm1_feature"] = frame["rapm1"]
        frame["target"] = frame.alpha_good
        # a with-and-without contrast knows as much as its smaller half: a player who missed two games
        # carries almost no information however many he played, and the reverse
        frame["weight"] = (frame.with_poss * frame.without_poss
                           / (frame.with_poss + frame.without_poss)).fillna(0.0)
        frame = frame[frame.weight > 0].set_index("player_id", drop=False)

        model_feats = features + EXTRA
        params = cfg["gbdt"]["params" if code == "O" else "params_def"]
        model = OutOfPlayerSPM(params, folds).fit(frame, model_feats)
        out_of_fold = model.predict(frame, model_feats)

        rating_only = OutOfPlayerSPM(params, folds).fit(frame, EXTRA)
        no_exposure = [f for f in model_feats if f not in EXPOSURE]
        without = OutOfPlayerSPM(params, folds).fit(frame, no_exposure)
        scores.append(dict(
            side=side, players=len(frame), seasons=frame.season.nunique(),
            alpha_sd=float(np.sqrt(np.average(frame.target ** 2, weights=frame.weight))),
            r2_everything=weighted_r2(frame.target, out_of_fold, frame.weight),
            r2_no_exposure=weighted_r2(frame.target, without.predict(frame, no_exposure), frame.weight),
            r2_rating_only=weighted_r2(frame.target, rating_only.predict(frame, EXTRA), frame.weight)))

        per_fold = (np.mean([m.feature_importances_ for m in model.fold_models_], axis=0)
                    if model.fold_models_ else model.full_.feature_importances_)
        tables.append(pd.DataFrame({
            "side": side, "feature": model_feats,
            "importance": np.asarray(model.full_.feature_importances_, dtype=float),
            "importance_folds": np.asarray(per_fold, dtype=float),
            "ridge_per_sd": standardised_ridge(frame, model_feats),
            "measures": ["how much he played" if f in EXPOSURE else
                         "the rankings themselves" if f in EXTRA else "how he played"
                         for f in model_feats],
        }))

    table = pd.concat(tables, ignore_index=True)
    table.to_parquet(ROOT / "outputs" / f"{tag}_features.parquet", index=False)
    score = pd.DataFrame(scores)
    score.to_parquet(ROOT / "outputs" / f"{tag}_r2.parquet", index=False)

    print("\n=== how much of what the rating missed a season's statistics can see, out of fold")
    print("    alpha_sd: the size of what is being predicted, points per 100.")
    print("    r2_everything: the share of it every feature accounts for together.")
    print("    r2_no_exposure: the same with the how-much-he-played features removed -- the honest number,")
    print("      because those can predict how NOISY a player's alpha is without knowing anything about him.")
    print("    r2_rating_only: what the rankings' own rating and prior manage on their own.")
    print(score.round(4).to_string(index=False))

    for side in SIDE_CODE:
        part = table[table.side == side].sort_values("importance", ascending=False)
        print(f"\n=== {side}: every feature, ranked by importance to the booster")
        print("    ridge_per_sd: points per 100 of alpha per standard deviation of the feature, sign and all.")
        print("    measures: what the feature is actually about.  A how-much-he-played feature near the top")
        print("    is a warning, not a discovery.")
        print(part[["feature", "importance", "importance_folds", "ridge_per_sd",
                    "measures"]].to_string(index=False))


if __name__ == "__main__":
    main()
