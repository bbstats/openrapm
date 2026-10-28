"""Kumo Tabular on this machine's RTX 3070: seconds and peak GPU memory for one in-context answer at the box-score
prior's shape (20 inputs: the 18 of a side and row_possessions, row_seasons), by model size, training rows and
number of ensemble members.  Synthetic rows; only the shape matters here.

    .venv/Scripts/python scratch/2026-10-28_kumo/bench.py [--sizes small,medium] [--rows 2000,5000,...]
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import sdm
import torch


def one_answer(model, n_rows: int, n_query: int, n_cols: int, members: int, rng) -> tuple[float, float]:
    x = rng.normal(size=(n_rows + n_query, n_cols)).astype(np.float32)
    y = (x[:, :3].sum(axis=1) + rng.normal(size=n_rows + n_query)).astype(np.float32)
    device = torch.device("cuda")
    xc = torch.from_numpy(x[:n_rows]).to(device)
    yc = torch.from_numpy(y[:n_rows, None]).to(device)
    xq = torch.from_numpy(x[n_rows:]).to(device)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    with torch.amp.autocast("cuda", torch.float16), torch.inference_mode():
        out = model(x_context=xc, y_context=yc, x_query=xq, num_estimators=members)
    answer = out.numerical.float().mean(dim=-1).cpu().numpy()
    torch.cuda.synchronize()
    seconds = time.time() - start
    peak = torch.cuda.max_memory_allocated() / 2**30
    error = float(np.sqrt(np.mean((answer - y[n_rows:]) ** 2)))
    return seconds, peak, error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", default="small,medium")
    parser.add_argument("--rows", default="2000,5000,10000,20000,40000")
    parser.add_argument("--members", default="1,8")
    parser.add_argument("--cols", type=int, default=20)
    parser.add_argument("--query", type=int, default=600)
    args = parser.parse_args()
    rng = np.random.default_rng(0)
    print(f"torch {torch.__version__}, {torch.cuda.get_device_name(0)}, "
          f"free {torch.cuda.mem_get_info()[0] / 2**30:.2f} GB of {torch.cuda.mem_get_info()[1] / 2**30:.2f}",
          flush=True)
    for size in args.sizes.split(","):
        start = time.time()
        model = sdm.models.KumoTabular(task="regression", size=size, device="cuda")
        params = sum(p.numel() for p in model.parameters())
        print(f"\n{size}: {params / 1e6:.1f}M parameters, loaded in {time.time() - start:.1f}s, "
              f"weights {torch.cuda.memory_allocated() / 2**30:.2f} GB", flush=True)
        one_answer(model, 500, 100, args.cols, 1, rng)    # warm-up
        print(f"{'rows':>7} {'members':>7} {'seconds':>8} {'peak_GB':>8} {'rmse':>6}", flush=True)
        for members in (int(m) for m in args.members.split(",")):
            for n_rows in (int(r) for r in args.rows.split(",")):
                try:
                    seconds, peak, error = one_answer(model, n_rows, args.query, args.cols, members, rng)
                    print(f"{n_rows:>7} {members:>7} {seconds:>8.2f} {peak:>8.2f} {error:>6.3f}", flush=True)
                except torch.OutOfMemoryError:
                    print(f"{n_rows:>7} {members:>7} out of GPU memory", flush=True)
                    torch.cuda.empty_cache()
                    break
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
