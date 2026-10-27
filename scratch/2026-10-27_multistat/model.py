"""The multi-stat network: one model that reads every stat's history and predicts all 13 stats of the next game.

    link_j = baseline link_j + network_j(inputs)

    link       log of the rate per 100 for a rate, logit of the percentage for a percentage
    network    inputs -> 256 -> 256 -> 13, SiLU between layers (silu(x) = x / (1 + exp(-x))); its last layer starts at
               zero, so before training the model IS the per-stat baseline and training can only move away from it
               where the data pays for it

The loss is the joint negative log-likelihood (NLL) of all 13 stats of every training game:

    rate j, count y over p possessions:      m - y log m,     m = exp(link_j) x p / 100
    percentage j, y made of a attempts:      a softplus(link_j) - y link_j       (= binomial NLL on the logit scale)

Each stat is weighted by its own denominator automatically: a 3P% term is as large as the game's 3PA, so a 1-attempt
game moves the shooting estimate a tenth as much as a 10-attempt game, and a game with no attempts adds nothing.

Training: Adam (learning rate 2e-3, cosine decay to 0), batches of 4096 games, weight decay 1e-5 on the weights,
early stopping on the validation seasons' NLL (patience 4 passes).
"""
from __future__ import annotations

import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from sim_league import IS_PCT


def init_params(key, n_in, widths=(256, 256), n_out=13):
    params = []
    sizes = [n_in, *widths]
    for a, b in zip(sizes[:-1], sizes[1:]):
        key, sub = jax.random.split(key)
        params.append((jax.random.normal(sub, (a, b), jnp.float32) * jnp.sqrt(2.0 / a), jnp.zeros(b, jnp.float32)))
    params.append((jnp.zeros((sizes[-1], n_out), jnp.float32), jnp.zeros(n_out, jnp.float32)))
    return params


def forward(params, x, offset, key=None, drop=0.0):
    """drop > 0 with a key: dropout on each hidden layer (training only)."""
    h = x
    for W, b in params[:-1]:
        h = jax.nn.silu(h @ W + b)
        if key is not None and drop > 0.0:
            key, sub = jax.random.split(key)
            h = h * jax.random.bernoulli(sub, 1.0 - drop, h.shape) / (1.0 - drop)
    W, b = params[-1]
    return offset + h @ W + b


_IS_PCT = jnp.asarray(IS_PCT)

# The settings that won on the 15-season league's validation seasons (tune.py, 2026-10-27): 25.7% of the gap closed,
# against 19.2% for the first run (lr 2e-3, weight decay 1e-5, no dropout).
BEST = dict(widths=(256, 256), lr=5e-4, wd=1e-4, batch=4096, drop=0.1)


def nll_terms(link, y, att, log_scale):
    """(rows x 13) NLL per stat; att = attempts for the percentage columns, log_scale = log(p / 100)."""
    rate = jnp.exp(link + log_scale) - y * (link + log_scale)
    pct = att * jax.nn.softplus(link) - y * link
    return jnp.where(_IS_PCT, pct, rate)


def loss_fn(params, x, offset, y, att, log_scale, wd, key, drop):
    l = nll_terms(forward(params, x, offset, key, drop), y, att, log_scale).sum(1).mean()
    reg = sum((W ** 2).sum() for W, _ in params)
    return l + wd * reg


@partial(jax.jit, static_argnames=("drop",))
def _step(params, m, v, t, x, offset, y, att, log_scale, lr, wd, key, drop):
    loss, g = jax.value_and_grad(loss_fn)(params, x, offset, y, att, log_scale, wd, key, drop)
    b1, b2 = 0.9, 0.999
    m = jax.tree_util.tree_map(lambda a, b: b1 * a + (1 - b1) * b, m, g)
    v = jax.tree_util.tree_map(lambda a, b: b2 * a + (1 - b2) * b * b, v, g)
    mh = jax.tree_util.tree_map(lambda a: a / (1 - b1 ** t), m)
    vh = jax.tree_util.tree_map(lambda a: a / (1 - b2 ** t), v)
    params = jax.tree_util.tree_map(lambda p, a, b: p - lr * a / (jnp.sqrt(b) + 1e-8), params, mh, vh)
    return params, m, v, loss


@jax.jit
def _eval_nll(params, x, offset, y, att, log_scale):
    return nll_terms(forward(params, x, offset), y, att, log_scale)


def mean_nll(params, data, idx, chunk=65536):
    tot = np.zeros(13)
    for s in range(0, len(idx), chunk):
        j = idx[s:s + chunk]
        tot += np.asarray(_eval_nll(params, *(a[j] for a in data))).sum(0)
    return tot / len(idx)


def forward_numpy(weights, x, offset):
    """The trained network in plain numpy (no JAX at prediction time): weights = [(W, b), ...] as numpy arrays."""
    h = np.asarray(x, np.float64)
    for W, b in weights[:-1]:
        a = h @ W + b
        h = a / (1.0 + np.exp(-a))
    W, b = weights[-1]
    return offset + h @ W + b


def predict_link(params, x, offset, chunk=65536):
    f = jax.jit(forward)
    return np.concatenate([np.asarray(f(params, x[s:s + chunk], offset[s:s + chunk])) for s in range(0, len(x), chunk)])


def train(data, train_idx, val_idx, widths=(256, 256), lr=2e-3, wd=1e-5, batch=4096, max_epochs=40, patience=4,
          seed=0, verbose=True, drop=0.0):
    """data = (x, offset, y, att, log_scale) as float32 arrays over all rows; returns (best params, history)."""
    x = data[0]
    key = jax.random.PRNGKey(seed)
    params = init_params(key, x.shape[1], widths)
    m = jax.tree_util.tree_map(jnp.zeros_like, params)
    v = jax.tree_util.tree_map(jnp.zeros_like, params)
    rng = np.random.default_rng(seed)
    steps_per_epoch = int(np.ceil(len(train_idx) / batch))
    total = steps_per_epoch * max_epochs
    best, best_val, bad, t, hist = params, np.inf, 0, 0, []
    dev = [jnp.asarray(a) for a in data]
    for ep in range(max_epochs):
        t0 = time.perf_counter()
        perm = rng.permutation(train_idx)
        run = 0.0
        for s in range(0, len(perm), batch):
            j = jnp.asarray(perm[s:s + batch])
            t += 1
            cur_lr = lr * 0.5 * (1 + np.cos(np.pi * t / total))
            key, sub = jax.random.split(key)
            params, m, v, loss = _step(params, m, v, t, *(a[j] for a in dev), cur_lr, wd, sub, drop)
            run += float(loss)
        val = mean_nll(params, data, val_idx).sum()
        hist.append((ep, run / steps_per_epoch, val))
        if verbose:
            print(f"    pass {ep:2d}  train NLL {run / steps_per_epoch:9.5f}  validation NLL {val:9.5f}  "
                  f"{time.perf_counter() - t0:5.1f} s")
        if val < best_val - 1e-6:
            best, best_val, bad = params, val, 0
        else:
            bad += 1
            if bad >= patience:
                break
    return best, hist
