"""How many epochs and seconds the line's descent takes on real rows, at a few penalties."""
import os
import pickle
import sys
import time

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(name, "1")
import numpy as np  # noqa: E402

sys.path.insert(0, "src")
from openrapm import linear_prior as lp  # noqa: E402
from openrapm.features import EVERYTHING_121, ROW_FEATURES  # noqa: E402

train = pickle.load(open("outputs/screen/screen120/train_2016.pkl", "rb"))["train"]["O"]
inputs = list(EVERYTHING_121) + ROW_FEATURES
matrix, label = train[inputs].to_numpy(float), train.label.to_numpy(float)
weight = train.row_weight.to_numpy(float) / train.row_weight.mean()
started = time.time()
expansion = lp.fit_expansion(matrix, weight)
total = lp.row_sums(expansion, matrix, label, weight, np.zeros(len(train), dtype=int), 1)[0]
print("columns", expansion.width, "sums", round(time.time() - started), "s", flush=True)
for l1 in (0.5,):
    top = lp.alpha_max(total, l1)
    coef = None
    for factor in (0.3, 0.1, 0.03, 0.01, 0.003, 0.001):
        t = time.time()
        coef, intercept, epochs = lp.solve(total, top * factor, l1, coef)
        print(f"l1 {l1} alpha x{factor}: epochs {epochs}, {time.time() - t:.1f}s, columns {(coef != 0).sum()}",
              flush=True)
