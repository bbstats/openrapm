"""Why the blends lose: each prior's weighted squared miss on held-out players' long-run ratings, the prior's own
target, on 2016's training rows with the booster's five player folds; today's booster, the booster on the passed
inputs, and the line, each fitted without the fold it answers.  One bag, seed 0, today's booster settings."""
import os
import pickle
import sys

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(name, "1")
import numpy as np  # noqa: E402

sys.path.insert(0, "src")
from openrapm import box_prior  # noqa: E402
from openrapm.data.config import booster_params, load_config  # noqa: E402
from openrapm.features import EVERYTHING_121, FEATURE_SETS, PASSED_121, ROW_FEATURES  # noqa: E402

line = {side: pickle.load(open(f"outputs/screen/blend121/line_2016_{side}.pkl", "rb")) if os.path.exists(
    f"outputs/screen/blend121/line_2016_{side}.pkl") else None for side in ("O", "D")}
params = booster_params(load_config(), "params/booster_params_lgb1_tune2.json")
train_all = pickle.load(open("outputs/screen/screen120/train_2016.pkl", "rb"))["train"]
shipped = FEATURE_SETS["hustle_no_three_pct_self_creation"]
for side in ("O", "D"):
    train = train_all[side]
    fold = box_prior.balanced_player_folds(train, 5, 0)
    y = train.label.to_numpy(float)
    w = train.row_weight.to_numpy(float)
    out = {}
    for label, names in (("today's booster", shipped[side]), ("booster on the passed inputs", PASSED_121[side])):
        prior = box_prior.fit_out_of_player(train, names + ROW_FEATURES, params[side], 5, 0)
        guess = np.empty(len(train))
        matrix = train[names + ROW_FEATURES].to_numpy(float)
        for k, model in enumerate(prior.fold_models):
            guess[fold == k] = model.predict(matrix[fold == k])
        out[label] = guess
    from openrapm.linear_prior import fit_expansion, line_from_sums, row_sums  # noqa: E402
    inputs = list(EVERYTHING_121) + ROW_FEATURES
    matrix = train[inputs].to_numpy(float)
    wn = w / w.mean()
    expansion = fit_expansion(matrix, wn)
    parts = row_sums(expansion, matrix, y, wn, fold, 5)
    total = parts[0] + parts[1] + parts[2] + parts[3] + parts[4]
    import json
    penalty = json.load(open("outputs/screen/blend121/line_penalty.json"))[side]
    guess = np.empty(len(train))
    for k in range(5):
        model = line_from_sums(expansion, total - parts[k], penalty["alpha"], penalty["l1"])
        guess[fold == k] = model.predict(matrix[fold == k])
    out["line"] = guess
    base = float(w @ (y - (w @ y) / w.sum()) ** 2 / w.sum())
    for label, guess in out.items():
        miss = float(w @ (y - guess) ** 2 / w.sum())
        print(f"{side} {label}: held-out squared miss {miss:.4f}, share of the spread explained {1 - miss / base:.3f}")
    blend = 0.75 * out["booster on the passed inputs"] + 0.25 * out["line"]
    print(f"{side} 25% line blend: held-out squared miss {float(w @ (y - blend) ** 2 / w.sum()):.4f}")
