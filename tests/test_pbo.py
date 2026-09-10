"""The probability of backtest overfitting, against cases where the answer is known.

What has to hold for the number to be worth putting in front of a pull request: a real edge reads
near 0, no edge at all reads 1, PBO falls as the edge grows, and the SAME edge reads worse the more
candidates were tried.  The last one is the property the merge gate rests on.

The absolute values here were measured, not assumed -- see test_the_calibration_table and the table
in the module docstring.  Note that pure independent noise does NOT read 0.5: one column gets a
persistent edge by luck and within a fixed sample that edge is real, so it reads lower.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eracoef import pbo as P  # noqa: E402

SEASONS = list(range(1998, 2026))          # the 28 the criterion holds out


def _long(V, names, seasons=SEASONS):
    """A (season x config) array as the long table the module takes."""
    return pd.DataFrame([{"season": s, "system": n, "score": V[i, j]}
                         for i, s in enumerate(seasons) for j, n in enumerate(names)])


def _noise(n_configs, seed, scale=1.0, seasons=SEASONS):
    rng = np.random.default_rng(seed)
    return rng.normal(110.0, scale, size=(len(seasons), n_configs))


# ------------------------------------------------------------------ known answers
def test_a_real_edge_reads_near_zero():
    """One configuration better by a full standard deviation: it wins in sample and keeps winning."""
    V = _noise(20, seed=1)
    V[:, 7] -= 1.0
    r = P.pbo(_long(V, [f"c{i:02d}" for i in range(20)]))
    assert r.pbo <= 0.05, r
    assert r.median_logit > 1.0, r


def test_no_edge_at_all_reads_one():
    """The unambiguous no-signal case: every column centred, so none is better over the full sample.
    The two halves of a split are complementary, so whatever won one half must lose the other."""
    V = _noise(20, seed=2)
    V -= V.mean(axis=0, keepdims=True)
    r = P.pbo(_long(V, [f"c{i:02d}" for i in range(20)]))
    assert r.pbo == 1.0, r


def test_the_same_true_edge_reads_worse_the_more_things_were_tried():
    """The property the gate rests on.  Half a standard deviation of real edge is trustworthy after
    twenty attempts and not after eighty -- which is the whole reason a PR must log every variant."""
    def at(n_configs):
        V = _noise(n_configs, seed=21)
        V[:, 0] -= 0.5
        return P.pbo(_long(V, [f"c{i:02d}" for i in range(n_configs)])).pbo
    few, many = at(20), at(80)
    assert few < many, f"PBO should rise with the candidate count: {few:.2f} at 20, {many:.2f} at 80"
    assert few <= 0.25 and many >= 0.35


def test_pbo_falls_monotonically_as_the_edge_grows():
    def at(effect):
        V = _noise(20, seed=31)
        V[:, 0] -= effect
        return P.pbo(_long(V, [f"c{i:02d}" for i in range(20)])).pbo
    got = [at(e) for e in (0.0, 0.5, 1.0, 2.0)]
    assert got == sorted(got, reverse=True), got
    # no absolute floor on the zero-edge end: uncentred noise is noisy across seeds (0.25 to 0.61 in
    # the calibration table).  What must hold is the separation.
    assert got[0] - got[-1] > 0.2 and got[-1] == 0.0, got


def test_identical_configurations_read_as_a_coin_flip_not_a_win():
    """The regression: exact ties sat at rank 0.5 and `rank > 0.5` counted them as a WIN, so eight
    clones read PBO 0.000 and certified nothing.  A tie is a coin flip and counts as half."""
    V = np.tile(_noise(1, seed=5), (1, 8))
    r = P.pbo(_long(V, [f"c{i:02d}" for i in range(8)]))
    assert r.pbo == 0.5, r


# ------------------------------------------------------------------ the guards
def test_an_unpaired_candidate_is_refused():
    """A candidate scored on different seasons than its rivals cannot be ranked against them."""
    t = _long(_noise(3, seed=6), ["a", "b", "c"])
    t = t[~((t.system == "b") & (t.season == 2000))]
    with pytest.raises(ValueError, match="paired"):
        P.pbo(t)


def test_a_single_candidate_is_refused_with_advice():
    t = _long(_noise(1, seed=7), ["only"])
    with pytest.raises(ValueError, match="log the baseline too"):
        P.pbo(t)


def test_an_odd_block_count_is_refused():
    with pytest.raises(ValueError, match="even"):
        P.pbo(_long(_noise(3, seed=8), ["a", "b", "c"]), n_blocks=15)


def test_too_few_seasons_to_split_is_refused():
    t = _long(_noise(3, seed=9, seasons=[2020, 2021]), ["a", "b", "c"], seasons=[2020, 2021])
    with pytest.raises(ValueError, match="at least 4"):
        P.pbo(t)


# ------------------------------------------------------------------ shape and cost
def test_sixteen_blocks_enumerate_every_balanced_split():
    r = P.pbo(_long(_noise(5, seed=10), list("abcde")))
    assert r.n_splits == 12870 and r.n_blocks == 16      # C(16, 8)


def test_max_splits_subsamples_deterministically():
    t = _long(_noise(5, seed=11), list("abcde"))
    a, b = P.pbo(t, max_splits=500), P.pbo(t, max_splits=500)
    assert a.n_splits == 500 and a.pbo == b.pbo


def test_blocks_are_contiguous_so_neighbouring_seasons_stay_together():
    """Interleaving would put 2024 in one half and 2025 in the other; they share most of their
    players, so that split does not test what it appears to."""
    blocks = P._blocks(28, 16)
    assert all(np.all(np.diff(b) == 1) for b in blocks)
    assert sorted(np.concatenate(blocks).tolist()) == list(range(28))


def test_report_names_the_winner():
    V = _noise(6, seed=12)
    V[:, 2] -= 1.0
    out = P.report(_long(V, list("abcdef")))
    assert "c 100%" in out and "PBO" in out


@pytest.mark.slow
def test_the_calibration_table():
    """The table quoted in the module docstring, recomputed.  If this drifts, the advice in the
    docstring and the threshold in the merge gate are both wrong and must be rewritten."""
    def at(n_configs, effect, seed):
        V = _noise(n_configs, seed=seed)
        V[:, 0] -= effect
        return P.pbo(_long(V, [f"c{i:02d}" for i in range(n_configs)])).pbo

    for effect, expected in ((0.0, 0.29), (0.5, 0.12), (1.0, 0.00)):
        got = float(np.median([at(20, effect, s) for s in range(5)]))
        assert abs(got - expected) < 0.10, f"edge {effect} sd at 20 candidates: {got:.2f}, docstring says {expected:.2f}"
