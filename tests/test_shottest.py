"""The direct shot test (eracoef.shottest), on a synthetic league with a known answer, two eras in one frame."""
import numpy as np
import pandas as pd

from eracoef import shottest as st


def league(seed=0, seasons=(2000, 2001, 2002, 2020, 2021, 2022), n_shooters=60, shots=200):
    """Shots whose true make probability is a known `truth`; `noisy` adds pure noise to it; `flat` is the rate."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in seasons:
        skill = rng.normal(0, 0.04, n_shooters)
        for p in range(n_shooters):
            d = rng.uniform(0, 28, shots)
            truth = np.clip(0.65 - 0.012 * d + skill[p], 0.05, 0.95)
            made = rng.random(shots) < truth
            rows.append(pd.DataFrame(dict(season=s, shooter=p, team=p % 30, opp=(p + 7) % 30, arena=p % 30,
                                          home=1, neutral=False, game_id=[f"{s}{g:04d}" for g in rng.integers(0, 600, shots)],
                                          half=np.where(np.arange(shots) % 2 == 0, "A", "B"),
                                          value=np.where(d > 23.75, 3, 2), made=made.astype(int), dist=d,
                                          truth=np.clip(0.65 - 0.012 * d, 0.05, 0.95))))
    f = pd.concat(rows, ignore_index=True)
    f["noisy"] = np.clip(f["truth"] + rng.normal(0, 0.15, len(f)), 0.02, 0.98)
    f["flat"] = f.groupby(["season", "value"])["made"].transform("mean")
    return f


def test_the_true_curve_beats_flat_and_noise_on_log_loss():
    f = league()
    S = st.season_scores(f, ["truth", "flat", "noisy"])
    assert st.paired(S, "truth", "flat")["diff"] < 0 and st.paired(S, "truth", "flat")["z"] < -3
    assert st.paired(S, "noisy", "truth")["diff"] > 0


def test_calibration_slope_reads_too_wide():
    f = league()
    logit = np.log(f.truth / (1 - f.truth))
    f["wide"] = 1 / (1 + np.exp(-2.0 * logit))
    C = st.calibration(f, ["truth", "wide"], by=("value",)).set_index(["value", "arm"])
    assert abs(C.loc[(2, "truth"), "slope"] - 1.0) < 0.1
    assert C.loc[(2, "wide"), "slope"] < 0.65                  # a curve twice too wide wants about half


def test_other_half_prefers_the_true_expectation_and_chooses_k_elsewhere():
    f = league()
    a = st.other_half(f, "truth", unit="shooter", value=2, min_fga=50)
    b = st.other_half(f, "flat", unit="shooter", value=2, min_fga=50)
    assert len(a) == 6 and set(a["k"]) <= set(st.K_GRID)
    assert st.paired_seasons(a, b)["diff"] < 0


def test_loso_mse_ignores_a_global_rescale():
    rng = np.random.default_rng(1)
    D = pd.DataFrame(dict(season=np.repeat(np.arange(6), 30), x=rng.normal(size=180), w=1.0))
    D["truth"] = 2 * D["x"] + rng.normal(size=180)
    D["x_shrunk"] = 0.3 * D["x"]
    a, _ = st.loso_mse(D, ["x"])
    b, _ = st.loso_mse(D, ["x_shrunk"])
    assert abs(a - b) < 1e-9


def test_zone_matches_the_stints_rule():
    f = pd.DataFrame(dict(value=[2, 2, 2, 3, 2], dist=[0.0, 3.0, 4.0, 25.0, -1.0]))
    assert st.zone_of(f).tolist() == ["rim", "rim", "mid", "thr", "mid"]


def test_an_arm_that_saw_the_result_cannot_price_the_half_it_predicts():
    f = league()
    f["cheat_after"] = f["made"].clip(0.02, 0.98)              # the extreme 'after': the result itself
    leaked = st.other_half(f, "cheat_after", unit="shooter", value=2, min_fga=50)
    guarded = st.other_half(f, "cheat_after", unit="shooter", value=2, min_fga=50, dest_arm="truth")
    honest = st.other_half(f, "truth", unit="shooter", value=2, min_fga=50)
    assert leaked["err"].mean() < 0.5 * honest["err"].mean()   # what the guard prevents
    assert guarded["err"].mean() > 0.9 * honest["err"].mean()  # guarded, it has no edge from its own results


def test_contest_left_is_near_zero_for_an_arm_that_sees_the_contest():
    rng = np.random.default_rng(5)
    rows = []
    for g in range(3000):
        tight = rng.uniform(0.1, 0.9)
        n = 12
        is_tight = rng.random(n) < tight
        p_true = np.where(is_tight, 0.38, 0.52)
        rows.append(pd.DataFrame(dict(season=2015, game_date="2015-01-01", shooter=g, made=(rng.random(n) < p_true).astype(int),
                                      sees=p_true, blind=np.full(n, 0.45), tight_shot=is_tight)))
    f = pd.concat(rows, ignore_index=True)
    dash = f.groupby("shooter").agg(tight=("tight_shot", "sum"), fga=("made", "size")).reset_index()
    dash = dash.rename(columns={"shooter": "PLAYER_ID"}).assign(date="2015-01-01")
    C = st.contest_left(f, ["sees", "blind"], dash).set_index("arm")
    assert abs(C.loc["sees", "slope"]) < 3 * C.loc["sees", "se"]
    assert C.loc["blind", "slope"] < -0.08


def test_teacher_gap_prefers_the_arm_nearer_the_teacher():
    f = league(seasons=(2015,))
    f["action_number"] = np.arange(len(f))
    lab = f[["game_id", "action_number"]].assign(q_teacher=f["truth"])
    T = st.teacher_gap(f, ["truth", "noisy"], lab)
    p = st.paired_games(T, "noisy", "truth")
    assert p["diff"] > 0 and p["z"] > 3
