"""Two things the box-score prior needs from Kumo Tabular, checked on synthetic rows: missing inputs (NaN) in the
training and the rated rows, and each ensemble member answering from its own set of training rows (a leading member
axis with num_estimators=None), plus whether running the members together is faster than one at a time."""
import time

import numpy as np
import sdm
import torch

rng = np.random.default_rng(0)
members, n_rows, n_query, n_cols = 4, 5000, 400, 20
x = rng.normal(size=(n_rows + n_query, n_cols)).astype(np.float32)
y = (x[:, 0] + 0.5 * np.nan_to_num(x[:, 5]) + rng.normal(size=n_rows + n_query) * 0.5).astype(np.float32)
x[: n_rows // 2, 12:18] = np.nan          # half the training rows lack six inputs, as the seasons before tracking do
x[n_rows: n_rows + n_query // 2, 12:18] = np.nan
cuda = torch.device("cuda")
model = sdm.models.KumoTabular(task="regression", size="small", device=cuda)

xc = torch.from_numpy(x[:n_rows]).to(cuda)
yc = torch.from_numpy(y[:n_rows, None]).to(cuda)
xq = torch.from_numpy(x[n_rows:]).to(cuda)
with torch.amp.autocast("cuda", torch.float16), torch.inference_mode():
    out = model(x_context=xc, y_context=yc, x_query=xq, num_estimators=members,
                generator=torch.Generator(cuda).manual_seed(1))
answer = out.numerical.float().mean(dim=-1).cpu().numpy()
print("NaN inputs: answers finite", bool(np.isfinite(answer).all()),
      "rmse", float(np.sqrt(np.mean((answer - y[n_rows:]) ** 2))))

# each member its own rows: a [members, rows, columns] context, the rated rows repeated per member
picks = np.stack([rng.choice(n_rows, size=3000, replace=False) for _ in range(members)])
xs = torch.from_numpy(x[picks]).to(cuda)
ys = torch.from_numpy(y[picks][..., None]).to(cuda)
xqs = xq.expand(members, *xq.shape)
for batch in (1, None):
    torch.cuda.synchronize()
    start = time.time()
    with torch.amp.autocast("cuda", torch.float16), torch.inference_mode():
        out = model(x_context=xs, y_context=ys, x_query=xqs, num_estimators=None, estimator_batch_size=batch,
                    generator=torch.Generator(cuda).manual_seed(1))
    torch.cuda.synchronize()
    answer = out.numerical.float().mean(dim=-1).cpu().numpy()
    print(f"own rows per member, batch {batch}: shape {tuple(out.numerical.shape)}, {time.time() - start:.2f}s, "
          f"rmse {float(np.sqrt(np.mean((answer - y[n_rows:]) ** 2))):.3f}")

# the same call twice with the same generator seed gives the same answers
with torch.amp.autocast("cuda", torch.float16), torch.inference_mode():
    a = model(x_context=xs, y_context=ys, x_query=xqs, num_estimators=None,
              generator=torch.Generator(cuda).manual_seed(5)).numerical.float().mean(dim=-1).cpu().numpy()
    b = model(x_context=xs, y_context=ys, x_query=xqs, num_estimators=None,
              generator=torch.Generator(cuda).manual_seed(5)).numerical.float().mean(dim=-1).cpu().numpy()
print("repeatable:", float(np.abs(a - b).max()))
# answers for a row do not depend on which other rows are asked with it
with torch.amp.autocast("cuda", torch.float16), torch.inference_mode():
    c = model(x_context=xs, y_context=ys, x_query=xqs[:, :50], num_estimators=None,
              generator=torch.Generator(cuda).manual_seed(5)).numerical.float().mean(dim=-1).cpu().numpy()
print("a row's answer alone vs with others:", float(np.abs(a[:50] - c).max()))
