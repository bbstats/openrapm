"""Which statistics move a player's trade-set correction, and by how many points per 100.

    python scripts/72_tradeset_shap.py [--alpha=outputs/tradeset_team_alpha.parquet]
                                       [--panel=outputs/role_panel_season.parquet]
                                       [--features=boruta] [--player_folds=5] [--season=2026]
                                       [--min_without=1] [--players=15] [--out=tradeset_team]

`scripts/71_tradeset_features.py` says how MUCH of the correction the statistics can see.  This says
WHICH ones and BY HOW MUCH, per statistic and per player, using exact TreeSHAP in the model's own
additive space.  That space is alpha's own units, so every number on this page is **points per 100
possessions of correction**: a mean absolute contribution of 0.08 means that statistic moves a typical
player's correction by eight hundredths of a point per 100, up or down.

**Out of fold, like everything else here.**  Each player's contributions come from the fold model that
never saw a row of his, so a statistic cannot score by helping the booster recognise the player.

**Two readings, and they answer different questions.**

  `moves_typical`   the mean absolute contribution: how much this statistic moves a player at all.
  `low` / `high`    the mean SIGNED contribution among the bottom and top fifth of the statistic:
                    which way it pushes, and how far, for the players at each end of it.

Read the direction from `low` and `high` together.  A statistic with `low` -0.10 and `high` +0.12 says
the rating is under-crediting the players who do a lot of it.  One with both near zero moves nobody,
whatever its importance rank says.

**The how-much-he-played family is flagged, not dropped.**  Alpha is noisier and more shrunk for a
player with less exposure, so those statistics can move a player without saying anything about how he
plays.  They are labelled in the `measures` column; read them as a warning.

Writes outputs/<out>_shap.parquet (per statistic, per side) and outputs/<out>_shap_players.parquet
(per player, per statistic, for the season asked about).
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

_spec = importlib.util.spec_from_file_location("_board", ROOT / "scripts" / "62_single_year_board.py")
_board = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_board)
OutOfPlayerSPM = _board.OutOfPlayerSPM

_spec2 = importlib.util.spec_from_file_location("_feat", ROOT / "scripts" / "71_tradeset_features.py")
_feat = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(_feat)
SIDE_CODE, EXTRA, EXPOSURE = _feat.SIDE_CODE, _feat.EXTRA, _feat.EXPOSURE

TAIL = 0.2      # the bottom and top fifth of a statistic, for the direction


# The shared command line (scripts/_cli.py): `flag` records every name it is asked for so
# `check_flags` below can refuse one that was never asked for.  A misspelled flag used to be
# ignored silently, which is how a run looks right and is wrong.
from _cli import check_flags, flag as _flag, switch   # noqa: E402


def frame_for(side: str, code: str, alpha: pd.DataFrame, panel: pd.DataFrame,
              features: list, min_without: float) -> pd.DataFrame:
    """The same rows and weights scripts/71 fits, so the two reports are about one model."""
    rows = alpha[(alpha.side == side) & alpha.eligible & (alpha.without_poss >= min_without)]
    frame = sy.season_frame(panel[panel.side == code], features)
    frame = frame.merge(rows[["player_id", "season", "alpha_good", "rating", "with_poss",
                              "without_poss"]], on=["player_id", "season"], how="inner")
    # The panel's `rapm1`, NOT the rankings' box prior (`prior_off`/`prior_def`): it is the role
    # prior plus the season's own residual, so it already contains what the season's games said.
    # It was called `prior` here and reported as the rankings' prior, which it is not.  A panel
    # without it is a panel this script cannot read, so ask rather than filling a zero.
    if "rapm1" not in frame.columns:
        raise SystemExit("the panel has no `rapm1` column; rebuild it with scripts/49_role_panel.py")
    frame["rapm1_feature"] = frame["rapm1"]
    frame["target"] = frame.alpha_good
    frame["weight"] = (frame.with_poss * frame.without_poss
                       / (frame.with_poss + frame.without_poss)).fillna(0.0)
    return frame[frame.weight > 0].set_index("player_id", drop=False)


def out_of_fold_shap(model, frame: pd.DataFrame, feats: list) -> np.ndarray:
    """Every player's contributions from the fold model that never saw a row of his."""
    X = frame[feats].to_numpy(float)
    out = model.full_.shap_values(X)
    ids = frame.player_id.to_numpy()
    for fold, excluded in zip(model.fold_models_, model.excluded_):
        index = np.fromiter((p in excluded for p in ids), dtype=bool, count=len(ids))
        if index.any():
            out[index] = fold.shap_values(X[index])
    return out


def weighted_quantile(values, weights, q: float) -> float:
    order = np.argsort(values)
    values, weights = np.asarray(values)[order], np.asarray(weights)[order]
    total = np.cumsum(weights) - 0.5 * weights
    return float(np.interp(q * weights.sum(), total, values))


def main():
    check_flags()      # refuse a flag this script does not understand (scripts/_cli.py)
    cfg = load_config(ROOT / "config.yaml")
    tag = _flag("out", "tradeset_team")
    # the alpha table follows the output tag unless it is named, so --out and --alpha
    # cannot silently disagree about which run is being read
    alpha = pd.read_parquet(ROOT / _flag("alpha", f"outputs/{tag}_alpha.parquet"))
    arm = str(alpha.team_effects.iloc[0]) if "team_effects" in alpha.columns else "unrecorded"
    print(f"alpha table: team effects {arm} (this script assumes 'team')")
    # The trust boundary (src/eracoef/seasons.py): rows whose unit reaches into a season still
    # being played may not reach a fit.  Gating at the read is what keeps every fit below honest;
    # it drops nothing while no season is in progress, and says so when it does.
    panel, _ = drop_untrainable(pd.read_parquet(ROOT / _flag("panel", "outputs/role_panel_season.parquet")),
                                cfg, what="the season panel")
    feature_sets = sy.feature_set(_flag("features", "boruta"))
    folds = int(_flag("player_folds", "5"))
    min_without = float(_flag("min_without", "1"))
    season = int(_flag("season", "2026"))
    n_players = int(_flag("players", "15"))

    pd.set_option("display.width", 230, "display.max_columns", 30, "display.max_rows", 200,
                  "display.precision", 4)
    print(f"alpha: {len(alpha):,} rows, {alpha.season.nunique()} seasons, penalty "
          f"{alpha.penalty.iloc[0]:,.0f}")
    print("every number below is in points per 100 possessions OF CORRECTION -- how far this statistic")
    print("moves a player's rating away from what the season's own games said, out of fold")

    tables, player_rows, scale = [], [], []
    for side, code in SIDE_CODE.items():
        features = list(feature_sets[code])
        frame = frame_for(side, code, alpha, panel, features, min_without)
        feats = features + EXTRA
        params = cfg["gbdt"]["params" if code == "O" else "params_def"]
        model = OutOfPlayerSPM(params, folds).fit(frame, feats)
        shap = out_of_fold_shap(model, frame, feats)

        weight = frame.weight.to_numpy(float)
        X = frame[feats].to_numpy(float)
        record = []
        for j, name in enumerate(feats):
            column, contribution = X[:, j], shap[:, j]
            low_cut = weighted_quantile(column, weight, TAIL)
            high_cut = weighted_quantile(column, weight, 1.0 - TAIL)
            low, high = column <= low_cut, column >= high_cut
            # The signed average of a feature's contributions is ~0 by construction: contributions are
            # deviations from the model's own mean, so they cancel.  `mean_signed` is reported to show
            # that, and `slope_per_sd` is the number that carries direction AND size -- the weighted
            # least-squares slope of the contribution on the STANDARDISED feature, so it reads as
            # "one standard deviation more of this statistic moves the correction by this much".
            mean_x = float(np.average(column, weights=weight))
            sd_x = float(np.sqrt(np.average((column - mean_x) ** 2, weights=weight)))
            standardised = (column - mean_x) / (sd_x if sd_x > 0 else 1.0)
            mean_shap = float(np.average(contribution, weights=weight))
            slope = float(np.average((contribution - mean_shap) * standardised, weights=weight))
            record.append(dict(
                side=side, feature=name,
                mean_signed=mean_shap,
                slope_per_sd=slope,
                moves_typical=float(np.average(np.abs(contribution), weights=weight)),
                moves_sd=float(np.sqrt(np.average(contribution ** 2, weights=weight))),
                moves_p90=weighted_quantile(np.abs(contribution), weight, 0.9),
                biggest=float(np.abs(contribution).max()),
                low=float(np.average(contribution[low], weights=weight[low])) if low.any() else np.nan,
                high=float(np.average(contribution[high], weights=weight[high])) if high.any() else np.nan,
                measures=("how much he played" if name in EXPOSURE
                          else "the rankings themselves" if name in EXTRA else "how he played")))
        per_feature = pd.DataFrame(record)
        # each statistic's share of all the movement the statistics produce together
        per_feature["share_of_movement"] = per_feature.moves_typical / per_feature.moves_typical.sum()
        tables.append(per_feature)
        # the size of the WHOLE prediction against the size of the thing being predicted: the ceiling
        # every per-statistic number below sits under
        predicted = shap.sum(axis=1)
        scale.append(dict(
            side=side, players=len(frame),
            correction_sd=float(np.sqrt(np.average(frame.target.to_numpy() ** 2, weights=weight))),
            prediction_sd=float(np.sqrt(np.average(predicted ** 2, weights=weight))),
            prediction_p90=weighted_quantile(np.abs(predicted), weight, 0.9),
            prediction_biggest=float(np.abs(predicted).max()),
            correlation=float(np.corrcoef(frame.target.to_numpy(), predicted)[0, 1])))

        here = frame.season.to_numpy() == season
        if here.any():
            block = pd.DataFrame(shap[here], columns=feats)
            block["player_id"] = frame.player_id.to_numpy()[here]
            # the correction itself is ONE number per player per side; melting it alongside the
            # contributions would repeat it once per statistic and any later sum would multiply it
            actual = pd.DataFrame({"player_id": frame.player_id.to_numpy()[here], "side": side,
                                   "alpha": frame.target.to_numpy()[here],
                                   "predicted": shap[here].sum(axis=1)})
            player_rows.append((block.melt(id_vars="player_id", var_name="feature",
                                           value_name="contribution").assign(side=side), actual))

    table = pd.concat(tables, ignore_index=True)
    table.to_parquet(ROOT / "outputs" / f"{tag}_shap.parquet", index=False)
    scale_frame = pd.DataFrame(scale)
    scale_frame.to_parquet(ROOT / "outputs" / f"{tag}_shap_scale.parquet", index=False)
    print()
    print("=== the ceiling: the size of the whole prediction against the size of the correction")
    print("    correction_sd: how much a player's correction varies, points per 100.")
    print("    prediction_sd: how much the statistics TOGETHER move a player.  Every per-statistic")
    print("    number below is a piece of this, not of the correction itself.")
    print(scale_frame.round(4).to_string(index=False))
    names = alpha[["player_id", "player_name"]].drop_duplicates("player_id")

    for side in SIDE_CODE:
        part = table[table.side == side].sort_values("moves_typical", ascending=False)
        print(f"\n=== {side}: every statistic, by how much it moves a player's correction")
        print("    moves_typical: mean absolute contribution.  moves_p90: the ninth decile of it.")
        print("    low / high: the mean SIGNED contribution among the bottom and top fifth of the")
        print("    statistic -- which way it pushes the players at each end, and how far.")
        print(part[["feature", "moves_typical", "moves_p90", "biggest", "low", "high",
                    "measures"]].to_string(index=False))
        real = part[part.measures == "how he played"]
        print(f"    of the {len(part)} statistics, the {len(real)} about HOW he played move a typical "
              f"player {real.moves_typical.sum():.3f} in total; the how-much-he-played family moves him "
              f"{part[part.measures == 'how much he played'].moves_typical.sum():.3f}")

    if player_rows:
        players = pd.concat([p for p, _ in player_rows], ignore_index=True).merge(
            names, on="player_id", how="left")
        players.to_parquet(ROOT / "outputs" / f"{tag}_shap_players.parquet", index=False)
        total = (pd.concat([a for _, a in player_rows], ignore_index=True)
                 .groupby("player_id", as_index=False)
                 .agg(alpha=("alpha", "sum"), predicted=("predicted", "sum"))
                 .merge(names, on="player_id", how="left"))
        biggest = total.reindex(total.alpha.abs().sort_values(ascending=False).index).head(n_players)
        print(f"\n=== {season}: the {n_players} largest corrections, and the three statistics behind each")
        print("    correction: what the trade set actually moved him by, both sides added.")
        print("    predicted: how much of that the statistics account for -- small by construction, since")
        print("    they see only a few per cent of the correction.  The three named are the statistics")
        print("    pushing hardest inside that prediction, in points per 100.")
        for row in biggest.itertuples():
            mine = (players[players.player_id == row.player_id]
                    .groupby("feature", as_index=False).contribution.sum())
            top = mine.reindex(mine.contribution.abs().sort_values(ascending=False).index).head(3)
            parts = ", ".join(f"{t.feature} {t.contribution:+.3f}" for t in top.itertuples())
            print(f"  {row.player_name:<24} correction {row.alpha:+.2f}  statistics say "
                  f"{row.predicted:+.2f}   {parts}")


if __name__ == "__main__":
    main()
