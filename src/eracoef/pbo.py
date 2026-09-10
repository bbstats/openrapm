"""The probability of backtest overfitting: how likely the winner of a search is a fluke.

The problem this exists for
---------------------------
A pull request says "this beats the shipped board by 0.06 per 100 over 28 held-out seasons."  That
number is worth almost nothing on its own, because it does not say how many things were tried before
one of them beat the board by 0.06.  Try forty variants against the same 28 seasons and one of them
will win by that much on noise alone.  The winner is then reported and the losers are not.

Combinatorially symmetric cross-validation (Bailey, Borwein, Lopez de Prado & Zhu, 2015) answers the
question directly.  Take the matrix of per-season scores, one column per configuration tried.  Split
the seasons into two halves in every balanced way there is.  On each split, find the configuration
that won the first half, and look at where it ranks on the second.  If the winners are real, they
keep winning.  If they were noise, they land around the middle.

    PBO = the fraction of splits on which the first-half winner scored below median on the second half

What the number means, measured rather than assumed
---------------------------------------------------
Calibrated on this project's shape -- 28 seasons, noise of about one point per 100, the planted edge
in units of that noise, median of five seeds (`tests/test_pbo.py::test_the_calibration_table`):

    planted edge |  5 candidates | 20 | 40 | 80
       0.0 sd    |     0.25      | 0.29 | 0.61 | 0.55
       0.25 sd   |     0.35      | 0.20 | 0.53 | 0.56
       0.5 sd    |     0.14      | 0.12 | 0.27 | 0.42
       1.0 sd    |     0.02      | 0.00 | 0.01 | 0.03
       2.0 sd    |     0.00      | 0.00 | 0.00 | 0.00

So: **at or below 0.25 the search found something; at or above 0.5 it did not.**  Read the number
beside the candidate count, because the same true edge reads worse the more things were tried -- an
edge of half a standard deviation is 0.12 after twenty attempts and 0.42 after eighty.  That is not
a defect in the statistic.  It is the statistic doing its job.

One subtlety worth knowing before quoting a number.  Two effects pull against each other.  A
configuration that is genuinely better keeps winning, which pushes PBO down.  But the two halves of
a split are complementary, so a configuration that happened to do well in one half must do
correspondingly worse in the other, which pushes PBO up.  With no signal at all the second effect
wins outright: if every configuration is centred so that none is better over the full sample, PBO
reads 1.00 for every candidate count.  Pure independent noise reads lower than that only because one
column gets a persistent edge by luck, and within this one sample that edge is real.

Why it costs nothing
--------------------
It reads a table of scores that already exist.  No refits.  C(16, 8) = 12,870 splits over sixteen
blocks takes under a second.

What it needs from a contributor
--------------------------------
Every configuration tried, not just the one being proposed.  A PR that logs one variant and wins has
a low PBO honestly; a PR that logs forty and reports the best gets the number it deserves.  This is
the whole point, and it is why the gate asks for the full candidate set rather than the winner.

Scores here are LOSSES -- points per 100 of squared error, lower is better.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PBOResult:
    pbo: float                  # P(the in-sample winner is below median out of sample)
    n_splits: int
    n_configs: int
    n_blocks: int
    median_logit: float         # the centre of the logit distribution; > 0 is a healthy search
    oos_rank: np.ndarray        # the winner's relative rank per split, in (0, 1); lower is better
    winners: np.ndarray         # which config index won the in-sample half, per split

    def __str__(self) -> str:
        verdict = ("the search found something" if self.pbo <= 0.25 else
                   "borderline: much of this gain is selection, read it beside the candidate count"
                   if self.pbo < 0.5 else
                   "the search found nothing: the in-sample winner does not carry out of sample")
        return (f"PBO {self.pbo:.3f} over {self.n_splits} splits of {self.n_blocks} blocks, "
                f"{self.n_configs} configurations; median logit {self.median_logit:+.2f} -- {verdict}")


def score_matrix(scores: pd.DataFrame, system: str = "system", unit: str = "season",
                 value: str = "score") -> tuple[pd.DataFrame, list]:
    """A (unit x configuration) matrix of losses from a long table, and the configuration names.

    Every configuration must have a score for every unit: a candidate scored on a different set of
    seasons than its rivals cannot be ranked against them, and quietly dropping the difference is
    how a comparison stops being paired.
    """
    M = scores.pivot_table(index=unit, columns=system, values=value, aggfunc="mean")
    missing = M.isna().sum()
    if missing.any():
        bad = missing[missing > 0]
        raise ValueError(
            f"every configuration needs a score on every {unit}, and these do not: "
            f"{dict(bad)}.  The comparison has to be paired.")
    return M.sort_index(), list(M.columns)


def _blocks(n_units: int, n_blocks: int) -> list[np.ndarray]:
    """Contiguous, near-equal blocks of unit positions.  Contiguous rather than interleaved because
    neighbouring seasons share players, and a split that puts 2024 in one half and 2025 in the other
    is not testing what it looks like it is testing."""
    return [b for b in np.array_split(np.arange(n_units), n_blocks) if len(b)]


def pbo(scores: pd.DataFrame, n_blocks: int = 16, system: str = "system", unit: str = "season",
        value: str = "score", max_splits: int | None = None, seed: int = 0) -> PBOResult:
    """The probability of backtest overfitting over the configurations in `scores`.

    `n_blocks` must be even; C(n_blocks, n_blocks/2) splits are enumerated exhaustively unless
    `max_splits` caps it, in which case that many are drawn without replacement.
    """
    M, names = score_matrix(scores, system=system, unit=unit, value=value)
    N = len(names)
    if N < 2:
        raise ValueError(f"PBO compares configurations against each other and needs at least two; got {N}.  "
                         f"A single candidate against the baseline is two -- log the baseline too.")
    n_blocks = int(n_blocks)
    if n_blocks % 2:
        raise ValueError(f"n_blocks must be even so the halves are balanced; got {n_blocks}")
    blocks = _blocks(len(M), n_blocks)
    if len(blocks) < 4:
        raise ValueError(f"{len(M)} {unit}s split into {len(blocks)} usable blocks; PBO needs at least 4")
    n_blocks = len(blocks)

    V = M.to_numpy(dtype=float)
    halves = list(combinations(range(n_blocks), n_blocks // 2))
    if max_splits is not None and len(halves) > max_splits:
        rng = np.random.default_rng(seed)
        halves = [halves[i] for i in rng.choice(len(halves), size=int(max_splits), replace=False)]

    ranks, winners = np.empty(len(halves)), np.empty(len(halves), dtype=int)
    for i, sel in enumerate(halves):
        ins = np.concatenate([blocks[b] for b in sel])
        oos = np.concatenate([blocks[b] for b in range(n_blocks) if b not in set(sel)])
        n_star = int(np.argmin(V[ins].mean(axis=0)))            # lower loss is better
        oos_mean = V[oos].mean(axis=0)
        # the winner's relative rank out of sample, in (0, 1), where 0 is best.  Ties share a rank so
        # that a set of identical configurations reads as the coin flip it is, not as a win.
        better = float((oos_mean < oos_mean[n_star]).sum())
        tied = float((oos_mean == oos_mean[n_star]).sum())
        ranks[i] = (better + (tied - 1.0) / 2.0 + 0.5) / N
        winners[i] = n_star

    lam = np.log(ranks / (1.0 - ranks))          # logit; > 0 means the winner beat the OOS median
    beat = (ranks > 0.5).astype(float) + 0.5 * (ranks == 0.5)
    return PBOResult(pbo=float(beat.mean()), n_splits=len(halves), n_configs=N,
                     n_blocks=n_blocks, median_logit=float(np.median(-lam)), oos_rank=ranks,
                     winners=winners)


def report(scores: pd.DataFrame, **kw) -> str:
    """One paragraph a pull request can paste, including who won how often."""
    r = pbo(scores, **kw)
    _, names = score_matrix(scores, system=kw.get("system", "system"), unit=kw.get("unit", "season"),
                            value=kw.get("value", "score"))
    won = pd.Series(r.winners).value_counts(normalize=True)
    top = ", ".join(f"{names[i]} {s:.0%}" for i, s in won.head(3).items())
    return f"{r}\n  in-sample winner was: {top}"
