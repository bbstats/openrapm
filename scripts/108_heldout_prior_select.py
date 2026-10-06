"""Experiment 37, stages 2-4: input and model selection on the stored held-out quadratics (scripts/107).

    python scripts/108_heldout_prior_select.py --stage=2|3|4 [--base=<config name>] [--lockbox=0]

The protocol (decided with the owner, plan "one-thing-i-want-joyful-book"):
  * development seasons only: every season not in `heldoutprior.LOCKBOX` (1999, 2002, ..., 2026);
  * for each development season H, everything is chosen on development seasons outside {H-1, H, H+1}:
      - the SHAPE (input weights) on within-season held-out error at 2/3, 3/4 and 9/10 of a season, each size's
        pooled error normalized by its own possessions (sizes weigh equally), with the penalty pair and the
        corrections' penalty rho chosen by leave-one-season-out inside the training seasons;
      - the SCALE (year-over-year penalty pair and one multiplier per side, B3's four numbers) on development
        year-over-year pairs (both seasons development, neither in {H-1, H, H+1});
  * anchored configurations keep B1's prediction as a free input per side and penalize every other input (rho ->
    infinity is B3 exactly); unanchored ones penalize everything;
  * scored on season H: within-season error at all six sizes (the three fit sizes and three out-of-size checks), the
    same with every team's level removed (the within-team guardrail), and year over year on H's development neighbour.

Every configuration's per-season scores go to outputs/heldout/ledger.parquet; the summary compares each configuration
with the B3-equivalent (anchor only) and B2-equivalent (no inputs) arms run through the SAME procedure, paired by
season, and reports the probability of backtest overfitting over everything logged so far.
"""
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _cli import check_flags, flag  # noqa: E402
from eracoef import heldoutprior as hp  # noqa: E402
from eracoef.pbo import pbo  # noqa: E402

OUT = ROOT / "outputs" / "heldout"
SIZES = ["q1of4", "q1of3", "within2", "q2of3", "within", "within10"]
FIT_SIZES = ["q2of3", "within", "within10"]
RHOS = (1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)   # penalty on corrections (widened: 1.0 was an edge)
FIRST, LAST = 1997, 2026
ALL_GROUPS = [g for g in hp.GROUPS if g != "b1"]
PLUS_MINUS = ["on_court", "off_court", "same_games_rapm", "booster"]


@dataclass
class Config:
    name: str
    groups: list = field(default_factory=list)         # input groups on both sides (b1 excluded: that is the anchor)
    anchor: bool = True
    fit: str = "within"                                  # "yoy": the diagnostic arm, shape fit year over year
    columns: list = field(default_factory=list)          # single input columns on both sides (experiment 38's per-stat map)


class Data:
    def __init__(self):
        info = json.loads((OUT / "inputs.json").read_text())
        self.columns = info["columns"]
        self.grid = [tuple(g) for g in info["grid"]]
        self.levels = info["levels"]
        self.k1 = len(self.columns)
        self.seasons = list(range(FIRST, LAST + 1))
        self.dev = [s for s in self.seasons if s not in hp.LOCKBOX]
        self.within = {}
        for tag in SIZES:
            z = np.load(OUT / f"quad_{tag}.npz")
            G, sw = z["G"], z["sw"]
            with np.errstate(invalid="ignore", divide="ignore"):
                self.within[tag] = (G, sw)                     # per season: G (pairs x levels x K+1 x K+1), sw
        z = np.load(OUT / "quad_yoy.npz")
        self.yoy_G, self.yoy_sw, self.pairs = z["G"], z["sw"], [tuple(p) for p in z["pairs"]]

    def inputs(self, cfg: Config) -> tuple:
        """(input indices, penalty mask): anchor inputs carry mask 0 (free), the rest 1."""
        idx, mask = [], []
        if cfg.anchor:
            for side in (0, 1):
                idx.append(side * self.k1 + self.columns.index("b1"))
                mask.append(0.0)
        for col in [c for g in cfg.groups for c in hp.GROUPS[g]] + list(cfg.columns):
            for side in (0, 1):
                idx.append(side * self.k1 + self.columns.index(col))
                mask.append(1.0)
        return np.array(idx, dtype=int), np.array(mask)

    def within_norm(self, tag, seasons, g, level, idx):
        """Sum over `seasons` of tag's grams at pair g, normalized by their possessions (size-level weighting)."""
        G, sw = self.within[tag]
        lv = self.levels.index(level)
        si = [self.seasons.index(s) for s in seasons if sw[self.seasons.index(s)] > 0]
        S = G[si, g, lv].sum(axis=0)
        return hp.subset(S, idx) / sw[si].sum()


def fit_shape(D: Data, cfg: Config, train: list, idx: np.ndarray, mask: np.ndarray) -> tuple:
    """(pair index, rho, weights): the within-season shape, chosen by leave-one-season-out inside `train`."""
    best = None
    for g in range(len(D.grid)):
        per_season = {s: sum(D.within_norm(tag, [s], g, "home", idx) for tag in FIT_SIZES) for s in train}
        total = sum(per_season.values())
        for rho in (RHOS if mask.any() else (0.0,)):
            cv = 0.0
            for v in train:
                if not len(idx):                          # no inputs: B2, only the penalty pair is chosen
                    cv += per_season[v][0, 0]
                    continue
                inner = total - sum(per_season[s] for s in train if abs(s - v) <= 1)
                cv += hp.error_at(per_season[v], 1.0, hp.fit(inner, 1.0, rho * mask))
            if best is None or cv < best[0]:
                best = (cv, g, rho)
    _, g, rho = best
    total = sum(sum(D.within_norm(tag, [s], g, "home", idx) for tag in FIT_SIZES) for s in train)
    beta = hp.fit(total, 1.0, rho * mask) if len(idx) else np.zeros(0)
    return g, rho, beta


def fit_shape_yoy(D: Data, train: list, idx, mask) -> tuple:
    """The diagnostic: the shape fit year over year on development pairs inside `train` (rated and scored)."""
    pairs = [i for i, (S, T) in enumerate(D.pairs) if S in train and T in train]
    best = None
    for g in range(len(D.grid)):
        Gs = {i: hp.subset(D.yoy_G[i, g, 0], idx) / D.yoy_sw[i] for i in pairs}
        total = sum(Gs.values())
        for rho in RHOS:
            cv = 0.0
            for S in sorted({D.pairs[i][0] for i in pairs}):
                inner = total - sum(Gs[i] for i in pairs if abs(D.pairs[i][0] - S) <= 1 or abs(D.pairs[i][1] - S) <= 1)
                beta = hp.fit(inner, 1.0, rho * mask)
                cv += sum(hp.error_at(Gs[i], 1.0, beta) for i in pairs if D.pairs[i][0] == S)
            if best is None or cv < best[0]:
                best = (cv, g, rho)
    _, g, rho = best
    total = sum(hp.subset(D.yoy_G[i, g, 0], idx) / D.yoy_sw[i] for i in pairs)
    return g, rho, hp.fit(total, 1.0, rho * mask)


def side_directions(D: Data, idx: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """(len(idx) x 2): the shape split into its offensive and defensive directions, for the per-side scale."""
    T = np.zeros((len(idx), 2))
    off = idx < D.k1
    T[off, 0], T[~off, 1] = beta[off], beta[~off]
    return T


def fit_scale(D: Data, H: int, idx, beta) -> tuple:
    """(year-over-year pair index, m_o, m_d) on development pairs that touch no season in {H-1, H, H+1}."""
    near = {H - 1, H, H + 1}
    pairs = [i for i, (S, T) in enumerate(D.pairs) if S in D.dev and T in D.dev and not ({S, T} & near)]
    T = side_directions(D, idx, beta)
    M = np.zeros((1 + len(idx), 3))
    M[0, 0] = 1.0
    M[1:, 1:] = T
    best = None
    for g in range(len(D.grid)):
        Q = sum(M.T @ hp.subset(D.yoy_G[i, g, 0], idx) @ M for i in pairs)
        sw = sum(D.yoy_sw[i] for i in pairs)
        if not len(idx):
            e, m = Q[0, 0] / sw, np.zeros(2)
        else:
            m = np.linalg.lstsq(Q[1:, 1:], Q[0, 1:], rcond=None)[0]
            e = hp.error_at(Q, sw, m)
        if best is None or e < best[0]:
            best = (e, g, m)
    return best[1], best[2]


def evaluate(D: Data, cfg: Config, H: int, scored_yoy: list) -> dict:
    train = [s for s in D.dev if abs(s - H) > 1]
    idx, mask = D.inputs(cfg)
    if cfg.fit == "yoy":
        g, rho, beta = fit_shape_yoy(D, train, idx, mask)
    else:
        g, rho, beta = fit_shape(D, cfg, train, idx, mask)
    gy, m = fit_scale(D, H, idx, beta)
    row = dict(config=cfg.name, season=H, pair=g, lam_o=D.grid[g][0], ratio=D.grid[g][1], rho=rho,
               yoy_pair=gy, yoy_lam_o=D.grid[gy][0], m_o=m[0] if len(m) else 0.0, m_d=m[1] if len(m) else 0.0,
               n_inputs=len(idx))
    for tag in SIZES:
        G, sw = D.within[tag]
        si = D.seasons.index(H)
        if sw[si] > 0:
            for level in ("home", "team") if tag in FIT_SIZES else ("home",):
                Gs = hp.subset(G[si, g, D.levels.index(level)], idx)
                row[f"{tag}_{level}"] = hp.error_at(Gs, sw[si], beta) if len(idx) else Gs[0, 0] / sw[si]
    T = side_directions(D, idx, beta)
    for (S, Tt) in [(H, t) for t in scored_yoy]:
        if (S, Tt) not in D.pairs:
            continue
        i = D.pairs.index((S, Tt))
        Gs = hp.subset(D.yoy_G[i, gy, 0], idx)
        row[f"yoy_{'prev' if Tt == S + 1 else 'next'}"] = (hp.error_at(Gs, D.yoy_sw[i], T @ m) if len(idx)
                                                             else Gs[0, 0] / D.yoy_sw[i])
    return row


def configs_for(stage: int, base: str) -> list:
    b = dict(STAGE2_BASES)
    if stage == 2:
        return [Config("B2 (no inputs)", [], anchor=False), Config("B3 (anchor only)", []),
                Config("B3 + B1 inputs", ["b1_inputs"]), Config("B1 inputs, unanchored", ["b1_inputs"], anchor=False)]
    base_groups = b.get(base, ["b1_inputs"])
    out = []
    if stage == 3:
        for g in ALL_GROUPS:
            if g not in base_groups:
                out.append(Config(f"{base} + {g}", base_groups + [g]))
        full = ALL_GROUPS
        out.append(Config("all inputs", full))
        for g in ALL_GROUPS:
            out.append(Config(f"all inputs - {g}", [x for x in full if x != g]))
        out.append(Config("all inputs - every plus-minus input", [x for x in full if x not in PLUS_MINUS]))
        out.append(Config("all inputs, shape fit year over year (diagnostic)", full, fit="yoy"))
    return out


STAGE2_BASES = [("B3 + B1 inputs", ["b1_inputs"]), ("B3 (anchor only)", [])]


def summarise(ledger: pd.DataFrame) -> pd.DataFrame:
    rows = []
    ref = "B3 (anchor only)"
    L = ledger.copy()
    L["yoy"] = L[[c for c in ("yoy_prev", "yoy_next") if c in L.columns]].mean(axis=1)
    L["fit_sizes"] = L[[f"{t}_home" for t in FIT_SIZES]].mean(axis=1)
    L["fit_sizes_team"] = L[[f"{t}_team" for t in FIT_SIZES]].mean(axis=1)
    for cfg, g in L.groupby("config", sort=False):
        r = dict(config=cfg, seasons=g.season.nunique(), inputs=int(g.n_inputs.iloc[0]),
                 within_fit=np.sqrt(g.fit_sizes.mean() * 2 / np.pi), within_team=np.sqrt(g.fit_sizes_team.mean() * 2 / np.pi),
                 yoy=np.sqrt(g.yoy.mean() * 2 / np.pi), q1of4=np.sqrt(g.q1of4_home.mean() * 2 / np.pi),
                 lam_low_edge=float((g.lam_o == 2560.0).mean()), lam_high_edge=float((g.lam_o == 67290.0).mean()),
                 rho_edge=float(g.rho.isin([RHOS[0], RHOS[-1]]).mean()))
        if cfg != ref and ref in set(L.config):
            p = L[L.config.isin([ref, cfg])].pivot_table(index="season", columns="config", values=["yoy", "fit_sizes"])
            for metric in ("yoy", "fit_sizes"):
                d = (p[(metric, cfg)] - p[(metric, ref)]).dropna()
                se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else np.nan
                r[f"{metric}_vs_B3"] = d.mean()
                r[f"{metric}_z"] = d.mean() / se if se and se > 0 else np.nan
                early, late = d[d.index <= 2011], d[d.index > 2011]
                r[f"{metric}_halves_agree"] = bool(np.sign(early.mean()) == np.sign(late.mean()))
        rows.append(r)
    return pd.DataFrame(rows)


def main():
    check_flags()
    stage = int(flag("stage", "2"))
    base = flag("base", "B3 + B1 inputs")
    lockbox = flag("lockbox", "0") == "1"
    assert not lockbox, "the lockbox is scored once, by the freeze step, not by a selection stage"
    D = Data()
    t0 = time.time()
    ledger_path = OUT / "ledger.parquet"
    ledger = pd.read_parquet(ledger_path) if ledger_path.exists() else pd.DataFrame()
    cfgs = configs_for(stage, base)
    for cfg in cfgs:
        if len(ledger) and cfg.name in set(ledger.config):
            continue
        rows = []
        for H in D.dev:
            scored = [t for t in (H - 1, H + 1) if t in D.dev]
            rows.append(evaluate(D, cfg, H, scored))
        ledger = pd.concat([ledger, pd.DataFrame(rows)], ignore_index=True)
        ledger.to_parquet(ledger_path, index=False)
        print(f"  {cfg.name}: done ({time.time() - t0:.0f}s)", flush=True)
    summary = summarise(ledger)
    summary.to_csv(OUT / "summary.csv", index=False)
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    scores = ledger.assign(score=ledger[[c for c in ("yoy_prev", "yoy_next") if c in ledger.columns]].mean(axis=1))
    scores = scores[["config", "season", "score"]].dropna()
    if scores.config.nunique() > 1:
        r = pbo(scores, n_blocks=10, system="config", unit="season", value="score")
        print(f"\nprobability of backtest overfitting over {r.n_configs} configurations (development year over year): "
              f"{r.pbo:.2f}  (<= 0.25 found something; >= 0.5 did not)")


if __name__ == "__main__":
    main()
