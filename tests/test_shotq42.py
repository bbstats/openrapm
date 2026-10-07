"""Experiment 42's quality plumbing: scripts/133 (per-attempt quality), scripts/134 (stint side tables), xshoot's
quality targets."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from eracoef import xshoot

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _script(name):
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), SCRIPTS / name)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _table(n=400, seed=0):
    r = np.random.default_rng(seed)
    value = np.where(np.arange(n) % 3 == 0, 3, 2)
    return pd.DataFrame(dict(season=2010, phase="RS", game_id=np.repeat([f"g{i}" for i in range(n // 40)], 40),
                             action_number=np.arange(n), value=value, made=(r.random(n) < 0.45).astype(int),
                             dist_xy=np.where(value == 3, 24.0, 8.0), corner3=False, noloc=False, heave=False,
                             sc_eff=12.0))


def test_the_recalibration_moves_threes_only_and_unlocated_threes_get_the_season_mean():
    m = _script("133_shot_q42.py")
    t = _table()
    r = np.random.default_rng(1)
    base = t[["game_id", "action_number"]].assign(eta_raw=r.normal(-0.4, 0.3, len(t)))
    arm = t[["game_id", "action_number"]].assign(eta_raw=base.eta_raw + r.normal(0, 0.2, len(t)),
                                                 q=1 / (1 + np.exp(-(base.eta_raw + 0.1))))
    q1 = m.quality(t, arm, base, alpha3=1.0)
    q0 = m.quality(t, arm, base, alpha3=0.0)
    two = t.value.to_numpy() == 2
    assert np.allclose(q1[two], arm.q[two]) and np.allclose(q0[two], arm.q[two])      # twos keep the arm's q
    assert not np.allclose(q1[~two], q0[~two])
    # alpha 0: the threes are exactly the base's logits relevelled per (season, band) from the season's makes
    from eracoef import shotmodel as sm, shotsearch as ss
    th = ~two
    g = t[th]
    lev = ss.relevel_fast(base.eta_raw.to_numpy()[th], g.season.to_numpy(), ss.band_codes(g), g.made.to_numpy(),
                          (g.phase == "RS").to_numpy())
    assert np.allclose(q0[th], sm.probability(lev))
    # unlocated threes: left out of the level fit, then priced at the season's located-three mean
    tn = t.assign(noloc=np.arange(len(t)) % 9 == 0)
    qn = m.quality(tn, arm, None, alpha3=None)
    un3 = ((tn.value == 3) & tn.noloc).to_numpy()
    loc3 = ((tn.value == 3) & ~tn.noloc).to_numpy()
    g = tn[th]
    lev = ss.relevel_fast(arm.eta_raw.to_numpy()[th], g.season.to_numpy(), ss.band_codes(g), g.made.to_numpy(),
                          ((g.phase == "RS") & ~g.noloc).to_numpy())
    want = sm.probability(lev)
    assert np.allclose(qn[loc3], want[~g.noloc.to_numpy()])
    assert np.allclose(qn[un3], qn[loc3].mean())
    assert np.allclose(qn[two], arm.q[two])


def test_the_side_table_sums_quality_by_stint_slot_and_reports_a_count_mismatch():
    m = _script("134_shot_sidecar.py")
    st = pd.DataFrame(dict(game_id=["g1", "g1", "g2"]))
    for kind in ("3", "2"):
        for c in m.cols(f"fg{kind}a"):
            st[c] = 0.0
    st.loc[0, "fg3a_s1_h"] = 2
    st.loc[1, "fg3a_sx_a"] = 1
    st.loc[2, "fg2a_s3_h"] = 1
    fr = pd.DataFrame(dict(game_id=["g1", "g1", "g1", "g2", "g2"], value=[3, 3, 3, 2, 3],
                           stint_no=[0, 0, 1, 0, -1], stint_side=["h", "h", "a", "h", ""],
                           stint_slot=["1", "1", "x", "3", ""], q_after=[0.3, 0.4, 0.5, 0.6, 0.9]))
    T, bad = m.side_table(st, fr)
    assert not bad
    assert T.loc[0, "xq3a_s1_h"] == 0.7 and T.loc[1, "xq3a_sx_a"] == 0.5 and T.loc[2, "xq2a_s3_h"] == 0.6
    assert T.drop(columns=[c for c in T.columns if c.startswith("fg")]).to_numpy().sum() == 0.7 + 0.5 + 0.6
    st.loc[0, "fg3a_s1_h"] = 3                                   # the cache says three, the frame placed two
    _, bad = m.side_table(st, fr)
    assert bad == {"fg3a_s1_h": 1}


def test_the_quality_targets_are_registered_beside_an_unchanged_incumbent():
    for tag in xshoot.QUALITY_ARMS:
        name = f"x3def_q{tag}_w0.25"
        assert xshoot.DEFENSE_TARGETS[name].__name__ == name
        assert xshoot.DEFENSE_TARGET_COLUMNS[name] == xshoot.DEFENSE_TARGET_COLUMNS["x3def"]
    assert xshoot.DEFENSE_TARGETS["x3def_w0.25"].__name__ == "x3def_w0.25"
    import inspect
    assert inspect.signature(xshoot.def_three_design).parameters["quality"].default is None
