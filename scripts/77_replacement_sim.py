"""Does 3-season RAPM overrate players with bad replacements?  A simulation where the truth is known.

    python scripts/77_replacement_sim.py [--penalty=40000] [--draws=200] [--sims=200] [--seed=0]

**Why this exists.**  The common belief is that RAPM overrates a player whose backups are bad: the drop when
he sits is credited to him.  Real data cannot show it, because nobody knows a player's true rating.  A
simulation can: take a truth, generate 2024-26 from it on the REAL lineups, fit plain RAPM, compare.  RAPM is
linear in the outcome, so its average over infinitely many simulated seasons is exact:

    expected RAPM = M @ truth,   M = (G + penalty)^-1 G,   G = X'WX, the fit's own normal equations

The with-or-without-you test is a controlled swap: keep the player's own truth, set only his REPLACEMENTS
(teammates on court more when he sits than when he plays, in games he played -- the rule of
`investigate.offcourt_rates`) to the league-typical replacement, and see how far his expected RAPM moves.
An unbiased RAPM would not move him at all.

**How to read it.**  Points per 100 possessions, positive good.
  RAPM inflation     how far the swap lowers his expected RAPM: what his replacements add to it
  on/off inflation   how far the same swap lowers his noise-free raw on/off
  share left in      slope of RAPM inflation on on/off inflation, through zero, every player once.
                     0% = RAPM removes the problem; 100% = RAPM is no better than raw on/off
  relative           the same slope after dividing by the fraction of his own true rating RAPM keeps
Three truths: OpenRAPM 2024-26; the consensus blend, with the players it lacks filled from OpenRAPM; and joint
draws from the plain RAPM's own posterior (the owner's first idea).  Under 0.1 per 100 is nothing.

**What it cannot tell you.**  Anything outside RAPM's additive model: bench units in garbage time, rotation
matching, fatigue.  That needs real data -- players who change teams.

Plain RAPM here is `looseason.LeaveSeasonOutRAPM` over the three seasons on raw points, regular season and
playoffs, at the project's plain-RAPM penalties (`singleyear.RAPM_*_LAMBDA`).  The run stops on any failed
check.  Writes outputs/replacement_sim.parquet (one row per player), outputs/replacement_sim_share.parquet
(truth x penalty x possessions tier), outputs/rapm_2024_26*.parquet (the rankings schema, for
scripts/66_compare.py) and outputs/replacement_sim_phone.html (the report as one static page).
"""
import html
import importlib.util
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Thread pinning before numpy, as in 62: BLAS gets one thread, numba four.
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
os.environ.setdefault("NUMBA_NUM_THREADS", os.environ.get("OPENRAPM_NUMBA_THREADS", "4"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.linalg as sla  # noqa: E402
import scipy.sparse as sp  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _cli import check_flags, flag  # noqa: E402
from eracoef import singleyear as sy  # noqa: E402
from eracoef.config import load_config  # noqa: E402
from eracoef.looseason import LeaveSeasonOutRAPM  # noqa: E402
from eracoef.windows import build_window  # noqa: E402

SEASONS = (2024, 2025, 2026)
BLOCK_LABEL = 2026              # the `season` 66_compare reads; every table here is the whole 2024-26 block
PENALTY_CURVE = (1e3, 3e3, 1e4, 2e4, 4e4, 8e4, 1.6e5)
TIERS = (("all", 0.0, np.inf), ("under 3,000", 0.0, 3000.0), ("3,000-10,000", 3000.0, 10000.0),
         ("over 10,000", 10000.0, np.inf))
LIST_MIN_POSS = 3000.0          # the most-inflated and most-deflated lists
SIGN_MIN_POSS = 3000.0          # the sign check's players
NOTHING = 0.1                   # the owner's rule: under 0.1 per 100 is nothing
TRUTH_WORDS = {"openrapm": "OpenRAPM", "consensus": "consensus", "draws": "your RAPM draws"}


class CheckFailed(SystemExit):
    pass


def check(ok: bool, what: str, log: list) -> None:
    """Record a check; stop the run on a failure (the plan's rule: no number is reported past a failed check)."""
    log.append(("PASS" if ok else "FAIL") + "  " + what)
    print(("  PASS  " if ok else "  FAIL  ") + what, flush=True)
    if not ok:
        raise CheckFailed(f"check failed: {what}")


def _script(name: str):
    """Another script's module body, for one helper it defines (76's name normaliser), as 70 borrows 63's."""
    spec = importlib.util.spec_from_file_location("_borrowed_" + name[:2], ROOT / "scripts" / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------------------------------------ the block
@dataclass
class Block:
    """The three seasons as one pooled plain-RAPM problem.

    Columns follow `LeaveSeasonOutRAPM._assemble`: [offense n | defense n | each season's own context].
    The defence columns are points ALLOWED (raw sign): a row's outcome is its context plus the five
    offensive ratings plus the five defensive ones.
    """
    rapm: LeaveSeasonOutRAPM
    designs: dict
    player_ids: np.ndarray
    n: int
    p: int
    gram: np.ndarray
    rhs: np.ndarray
    ywy: float
    X_parts: list               # each season's design in the global columns, rows in the design's order
    w_parts: list
    X: sp.csr_matrix            # the three stacked
    w: np.ndarray
    y: np.ndarray
    tg_off: np.ndarray          # per row: the team-game on offence, numbered base + 2 * game + (1 if home)
    tg_def: np.ndarray          # and the team-game defending
    poss: np.ndarray            # offensive possessions per player (looseason's count: a possession once)
    poss_def: np.ndarray
    active_o: np.ndarray        # players with at least one row on that side
    active_d: np.ndarray

    @property
    def n_rows(self) -> int:
        return len(self.w)


def load_block(cfg) -> Block:
    rapm = LeaveSeasonOutRAPM(min_possessions=0.0)
    designs, ywy = {}, 0.0
    for season in SEASONS:
        t0 = time.time()
        wd = build_window([season], cfg, target="pts")          # what Context.design's default loader calls
        rapm.add_season(season, wd)
        designs[season] = wd
        ywy += float(np.asarray(wd.w, dtype=float) @ np.asarray(wd.y, dtype=float) ** 2)
        print(f"  {season}: {len(wd.y):,} rows, {wd.spec.n_ps} players ({time.time() - t0:.0f}s)", flush=True)
    gram, rhs, poss, n_context = rapm._assemble(())
    n = poss.size
    p = 2 * n + n_context

    # Every season rebuilt in the global columns, so the rows themselves can be used: the on/off operators,
    # the simulated seasons and the sign check all need them, and G alone does not carry them.
    X_parts, w_parts, y_parts, tg_off, tg_def = [], [], [], [], []
    column, base = 2 * n, 0
    for season in rapm.seasons:                  # _assemble's order, so its context offsets are ours
        wd, stored = designs[season], rapm._season[season]
        slot, n_f = stored["slot"], stored["n_context"]
        F = np.asarray(wd.parts["F"], dtype=float)
        if F.shape[1] != n_f:
            raise SystemExit(f"{season}: {F.shape[1]} context columns in the design, {n_f} in the fit")
        rows = len(F)
        width = 10 + n_f
        indices = np.hstack([slot[np.asarray(wd.parts["lineup_o"])], slot[np.asarray(wd.parts["lineup_d"])] + n,
                             np.broadcast_to(column + np.arange(n_f), (rows, n_f))])
        data = np.hstack([np.ones((rows, 10)), F])
        X = sp.csr_matrix((data.ravel(), indices.ravel(), np.arange(0, rows * width + 1, width, dtype=np.int64)),
                          shape=(rows, p))
        X.sort_indices()
        X_parts.append(X)
        w_parts.append(np.asarray(wd.w, dtype=float))
        y_parts.append(np.asarray(wd.y, dtype=float))
        game = wd.rows["game_idx"].to_numpy(np.int64)
        home_off = wd.rows["is_home_off"].to_numpy(bool).astype(np.int64)
        tg_off.append(base + 2 * game + home_off)
        tg_def.append(base + 2 * game + 1 - home_off)
        base += 2 * (int(game.max()) + 1)        # even, so a game's two team-games stay an (even, odd) pair
        column += n_f
    X = sp.vstack(X_parts).tocsr()
    w, y = np.concatenate(w_parts), np.concatenate(y_parts)
    diag = np.diagonal(gram)
    return Block(rapm=rapm, designs=designs, player_ids=rapm.player_ids.copy(), n=n, p=p, gram=gram, rhs=rhs,
                 ywy=ywy, X_parts=X_parts, w_parts=w_parts, X=X, w=w, y=y, tg_off=np.concatenate(tg_off),
                 tg_def=np.concatenate(tg_def), poss=poss, poss_def=diag[n:2 * n].copy(),
                 active_o=diag[:n] > 0, active_d=diag[n:2 * n] > 0)


def weighted(X: sp.csr_matrix, w: np.ndarray) -> sp.csr_matrix:
    """diag(w) @ X, row-scaled in place of a product."""
    out = X.copy()
    out.data = out.data * np.repeat(w, np.diff(X.indptr))
    return out


def fit(block: Block, lam_o: float, lam_d: float):
    """Cholesky of G + penalty, the ridge fit and M = (G + penalty)^-1 G.

    Cholesky and nothing else: `priorridge.solve_diag` adds a jitter when a solve fails, which would break
    M @ context = context.  A failure here is a rank problem to fix, not to paper over.
    """
    n, p = block.n, block.p
    penalty = np.concatenate([np.full(n, float(lam_o)), np.full(n, float(lam_d)), np.zeros(p - 2 * n)])
    factor = sla.cho_factor(block.gram + np.diag(penalty))
    return factor, sla.cho_solve(factor, block.rhs), sla.cho_solve(factor, block.gram)


def centre(t: np.ndarray, block: Block) -> np.ndarray:
    """Unweighted mean zero per side over the players with rows on that side.

    That is how the ridge centres its own output: a constant on one side is invisible to the data (the
    season intercepts absorb it), so the penalty sets each side's plain sum to zero.  Centring the truth the
    same way keeps a constant shift between truth and fit from reading as a bias.
    """
    t = np.array(t, dtype=float, copy=True)
    n = block.n
    for part, active in ((t[:n], block.active_o), (t[n:2 * n], block.active_d)):
        part[active] -= part[active].mean(axis=0)
        part[~active] = 0.0
    return t


# ------------------------------------------------------------------------------------------ the truths
def openrapm_truth(block: Block):
    """OpenRAPM 2024-26: each player's seasons averaged, offence weighted by offensive possessions and defence
    by defensive ones; defence flipped to the points-allowed sign."""
    t = pd.read_parquet(ROOT / "outputs" / "season_ratings_product.parquet")
    t = t[t.season.isin(SEASONS)]
    w_o, w_d = t.poss_off.clip(lower=0.0), t.poss_def.clip(lower=0.0)
    off = (t.rating_off * w_o).groupby(t.player_id).sum() / w_o.groupby(t.player_id).sum()
    dfn = (t.rating_def * w_d).groupby(t.player_id).sum() / w_d.groupby(t.player_id).sum()
    names = t.sort_values("season").drop_duplicates("player_id", keep="last").set_index("player_id").player_name
    ids = block.player_ids
    o, d = off.reindex(ids).to_numpy(float), dfn.reindex(ids).to_numpy(float)
    if np.isnan(o).any() or np.isnan(d).any():
        raise SystemExit(f"OpenRAPM has no 2024-26 rating for {int(np.isnan(o + d).sum())} of the block's players")
    return np.concatenate([o, -d]), names.reindex(ids).fillna("").to_numpy(object)


def consensus_truth(block: Block, openrapm: np.ndarray, names: np.ndarray, norm):
    """The consensus blend (one row per player for 2024-26, positive good), joined on normalised names.

    A name that is not unique on either side is left unmatched rather than guessed.  Unmatched players --
    mostly the bench, i.e. mostly the replacements themselves -- are filled from OpenRAPM with one level
    shift per side, chosen so the two agree (possession-weighted) on the players they share.
    """
    n = block.n
    con = pd.read_csv(ROOT / "data" / "external" / "consensus.csv")
    for c in ("adj_offense", "adj_defense"):
        con[c] = pd.to_numeric(con[c], errors="coerce")
    con = con.dropna(subset=["player_name", "adj_offense", "adj_defense"]).copy()
    con["key"] = con.player_name.map(norm)
    ours = pd.Series(names).map(norm)
    ambiguous = set(con.key[con.key.duplicated(keep=False)]) | set(ours[ours.duplicated(keep=False)])
    lookup = con[~con.key.isin(ambiguous)].set_index("key")
    matched = np.array([(k in lookup.index) and (k not in ambiguous) and bool(k) for k in ours])
    o, d = openrapm[:n].copy(), openrapm[n:].copy()
    con_o = np.where(matched, lookup.adj_offense.reindex(ours).to_numpy(float), np.nan)
    con_d = np.where(matched, -lookup.adj_defense.reindex(ours).to_numpy(float), np.nan)
    shift_o = float(np.average(con_o[matched] - o[matched], weights=block.poss[matched]))
    shift_d = float(np.average(con_d[matched] - d[matched], weights=block.poss_def[matched]))
    o = np.where(matched, con_o, o + shift_o)
    d = np.where(matched, con_d, d + shift_d)
    info = dict(rows=len(con), matched=int(matched.sum()), ambiguous=len(ambiguous),
                unmatched_consensus=int((~con.key.isin(set(ours[matched]))).sum()),
                shift_off=shift_o, shift_def=-shift_d)
    return np.concatenate([o, d]), ~matched, info


# ------------------------------------------------------------------------------------------ on/off and replacements
def onoff_operators(block: Block) -> dict:
    """Per side, the linear map from the full truth (players and context) to each player's noise-free raw on/off.

    A team-game is (season, game, which team).  His team's rows are the rows of the team-games he appears in
    (on offence in its offensive rows or on defence in its defensive rows), so a mid-season trade needs no
    team ids.  Row i of the offensive operator is the possession-weighted mean of his team's offensive rows
    with him on court minus the same without him; applied to a truth it IS his offensive on/off.
    """
    X, w, n, p = block.X, block.w, block.n, block.p
    rows = np.arange(len(w))
    n_tg = int(max(block.tg_off.max(), block.tg_def.max())) + 1
    P_off = sp.csr_matrix((np.ones(len(w)), (rows, block.tg_off)), shape=(len(w), n_tg))
    P_def = sp.csr_matrix((np.ones(len(w)), (rows, block.tg_def)), shape=(len(w), n_tg))
    Z_o, Z_d = X[:, :n], X[:, n:2 * n]
    member = (Z_o.T @ P_off + Z_d.T @ P_def).tocsr()
    member.data[:] = 1.0
    both = member[:, 0::2].multiply(member[:, 1::2]).tocsr()
    both.eliminate_zeros()
    WX = weighted(X, w)
    out = {"both_teams": int(both.nnz)}
    for side, P, Z in (("O", P_off, Z_o), ("D", P_def, Z_d)):
        per_team_game = (P.T @ WX).tocsr()
        w_team_game = np.asarray(P.T @ w).ravel()
        total = (member @ per_team_game).toarray()
        w_total = np.asarray(member @ w_team_game).ravel()
        on = (Z.T @ WX).toarray()
        w_on = np.asarray(Z.T @ w).ravel()
        w_off = w_total - w_on
        ok = (w_on > 0) & (w_off > 1e-9)
        D = np.zeros((n, p))
        D[ok] = on[ok] / w_on[ok, None] - (total[ok] - on[ok]) / w_off[ok, None]
        out[side] = dict(D=D, ok=ok, w_on=w_on, w_off=w_off)
    return out


def replacement_structure(ops: dict, n: int) -> dict:
    """Replacement weights per side: e[i, j] = teammate j's share of his team's possessions when i sits minus
    when i plays.  They sum to exactly 1 (five teammates off court, four on); the positive ones are his
    replacements.  `a[j]` = the replacement possessions player j plays across the league, the weights of the
    typical replacement."""
    rep = {}
    for side, cols in (("O", slice(0, n)), ("D", slice(n, 2 * n))):
        D, ok = ops[side]["D"], ops[side]["ok"]
        own = np.diagonal(D[:, cols]).copy()
        e = -D[:, cols].copy()
        np.fill_diagonal(e, 0.0)
        e[~ok] = 0.0
        R = e > 1e-12
        a = (ops[side]["w_off"][:, None] * np.where(R, e, 0.0)).sum(0)
        rep[side] = dict(e=e, R=R, own=own, sums=e.sum(1), ok=ok, a=a)
    return rep


def inflation(T: np.ndarray, M: np.ndarray, ops: dict, rep: dict, n: int) -> dict:
    """For each player (rows) and truth (columns of T, 2n x K): what the swap takes off his expected RAPM and
    off his noise-free raw on/off.  Positive = his replacements make him look better than he is.

    The swap for player i: Delta_i = truth minus the truth with his replacements set to the typical
    replacement, per side, nothing else touched.  His RAPM moves by rows o_i and d_i of M @ Delta_i (the
    cross-side blocks included: a replacement's defence can leak into his offence); his on/off by the
    operators' rows.  The total is offence minus defence (defence in the points-allowed sign).
    """
    T = T.reshape(2 * n, -1)
    t_o, t_d = T[:n], T[n:2 * n]
    typ_o = rep["O"]["a"] @ t_o / rep["O"]["a"].sum()
    typ_d = rep["D"]["a"] @ t_d / rep["D"]["a"].sum()
    R_o, R_d = rep["O"]["R"], rep["D"]["R"]

    def swap(on_offence_cols, on_defence_cols):
        b_o = np.where(R_o, on_offence_cols, 0.0)
        b_d = np.where(R_d, on_defence_cols, 0.0)
        return (b_o @ t_o - b_o.sum(1)[:, None] * typ_o[None, :]) + (b_d @ t_d - b_d.sum(1)[:, None] * typ_d[None, :])

    r_o = swap(M[:n, :n], M[:n, n:2 * n])
    r_d = swap(M[n:2 * n, :n], M[n:2 * n, n:2 * n])
    D_O, D_D = ops["O"]["D"], ops["D"]["D"]
    q_o = swap(D_O[:, :n], D_O[:, n:2 * n])
    q_d = swap(D_D[:, :n], D_D[:, n:2 * n])
    keep_o, keep_d = np.diagonal(M)[:n], np.diagonal(M)[n:2 * n]
    e_o, e_d = np.where(R_o, rep["O"]["e"], 0.0), np.where(R_d, rep["D"]["e"], 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):       # players with no rows on a side; masked later
        gap_o = (e_o @ t_o) / e_o.sum(1)[:, None] - typ_o[None, :]
        gap_d = (e_d @ t_d) / e_d.sum(1)[:, None] - typ_d[None, :]
        rel = r_o / keep_o[:, None] - r_d / keep_d[:, None]
    return dict(rapm=r_o - r_d, rapm_off=r_o, rapm_def=-r_d, onoff=q_o - q_d, rel=rel, gap=gap_o - gap_d,
                typ_o=typ_o, typ_d=typ_d)


SHARED_GROUPS = (("backups", 0.0, 0.10), ("some shared", 0.10, 0.30), ("plays with him", 0.30, np.inf))


def shared_court_time(block: Block) -> np.ndarray:
    """Per pair (i, j): the share of i's possessions j is on court with him, offence and defence averaged."""
    n, d = block.n, np.diagonal(block.gram)
    with np.errstate(invalid="ignore", divide="ignore"):
        return 0.5 * (block.gram[:n, :n] / d[:n][:, None] + block.gram[n:2 * n, n:2 * n] / d[n:2 * n][:, None])


def inflation_by_group(t: np.ndarray, M: np.ndarray, rep: dict, shared: np.ndarray, n: int) -> dict:
    """RAPM inflation for one truth, split by how much court time each replacement shares with him.

    "On court more when he sits" also catches a staggered co-star (Shai Gilgeous-Alexander is one of Cason
    Wallace's), which is not what anyone means by a backup.  The groups partition his replacements -- backups
    share under 10% of his possessions, "plays with him" 30% or more -- so the parts sum to the inflation.
    """
    dev_o = t[:n] - rep["O"]["a"] @ t[:n] / rep["O"]["a"].sum()
    dev_d = t[n:2 * n] - rep["D"]["a"] @ t[n:2 * n] / rep["D"]["a"].sum()
    R_o, R_d = rep["O"]["R"], rep["D"]["R"]
    part = (np.where(R_o, M[:n, :n], 0.0) * dev_o + np.where(R_d, M[:n, n:2 * n], 0.0) * dev_d) \
        - (np.where(R_o, M[n:2 * n, :n], 0.0) * dev_o + np.where(R_d, M[n:2 * n, n:2 * n], 0.0) * dev_d)
    return {g: np.where((shared >= lo) & (shared < hi), part, 0.0).sum(1) for g, lo, hi in SHARED_GROUPS}


def col_corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a, b = a - a.mean(0), b - b.mean(0)
    return (a * b).sum(0) / np.sqrt((a ** 2).sum(0) * (b ** 2).sum(0))


def shares(inf: dict, poss: np.ndarray, valid: np.ndarray) -> list:
    """Per possessions tier: the slope through zero of RAPM inflation on on/off inflation, every player once
    (one value per truth column), the same for the relative version, and the correlation."""
    out = []
    for tier, lo, hi in TIERS:
        m = valid & (poss >= lo) & (poss < hi)
        x, y, y_rel = inf["onoff"][m], inf["rapm"][m], inf["rel"][m]
        den = (x ** 2).sum(0)
        out.append(dict(tier=tier, players=int(m.sum()), share=(x * y).sum(0) / den,
                        share_rel=(x * y_rel).sum(0) / den, correlation=col_corr(x, y)))
    return out


# ------------------------------------------------------------------------------------------ checks that need refits
def fresh_fit(block: Block, theta: np.ndarray, lam_o: float, lam_d: float) -> np.ndarray:
    """Noise-free outcomes from `theta` pushed through a NEW LeaveSeasonOutRAPM via `with_target`: the
    project's own fitting path, nothing of ours in it but the outcomes."""
    fresh = LeaveSeasonOutRAPM(min_possessions=0.0)
    for season, X_s in zip(block.rapm.seasons, block.X_parts):
        fresh.add_season(season, block.designs[season].with_target(X_s @ theta))
    if not np.array_equal(fresh.player_ids, block.player_ids):
        raise SystemExit("a fresh fit laid the players out differently")
    r = fresh.ratings(None, lam_o, lam_d, 0.0)
    return np.concatenate([r.offense.to_numpy(float), r.defense.to_numpy(float)])


def simulate(block: Block, factor, theta: np.ndarray, sigma2: float, n_sims: int, rng, chunk: int = 25) -> np.ndarray:
    """Whole 2024-26 seasons generated row by row -- outcome = lineup sum + context + noise with variance
    sigma2 / possessions -- and refitted.  The same G for every refit, so only the right-hand side changes."""
    fits = np.zeros((block.p, n_sims))
    for k0 in range(0, n_sims, chunk):
        k1 = min(n_sims, k0 + chunk)
        rhs = np.zeros((block.p, k1 - k0))
        for X_s, w_s in zip(block.X_parts, block.w_parts):
            mean = X_s @ theta
            y = mean[:, None] + np.sqrt(sigma2 / w_s)[:, None] * rng.standard_normal((len(w_s), k1 - k0))
            rhs += np.asarray(X_s.T @ (w_s[:, None] * y))
        fits[:, k0:k1] = sla.cho_solve(factor, rhs)
    return fits


def sign_slopes(block: Block, t: np.ndarray) -> tuple:
    """Weighted regression of the REAL outcomes on the truth's two lineup sums and every season's context.
    A truth with a flipped side gets a negative slope there."""
    n = block.n
    u_o = block.X[:, :n] @ t[:n]
    u_d = block.X[:, n:2 * n] @ t[n:2 * n]
    design = sp.hstack([sp.csr_matrix(u_o[:, None]), sp.csr_matrix(u_d[:, None]), block.X[:, 2 * n:]]).tocsr()
    W = weighted(design, block.w)
    coef = np.linalg.solve((design.T @ W).toarray(), np.asarray(W.T @ block.y).ravel())
    return float(coef[0]), float(coef[1])


# ------------------------------------------------------------------------------------------ report
def display_ratings(block: Block, o: np.ndarray, d: np.ndarray) -> tuple:
    """Offence and defence (points allowed) to the published convention: possession-weighted zero per side,
    defence positive-good."""
    keep = block.poss >= sy.MIN_POSSESSIONS
    o = o - np.average(o[keep], weights=block.poss[keep])
    d = d - np.average(d[keep], weights=block.poss_def[keep])
    return o, -d


def rankings_frame(block: Block, names, o, d) -> pd.DataFrame:
    keep = block.poss >= sy.MIN_POSSESSIONS
    off, dfn = display_ratings(block, o, d)
    return pd.DataFrame({"player_id": block.player_ids[keep], "player_name": names[keep], "season": BLOCK_LABEL,
                         "rating_off": off[keep], "rating_def": dfn[keep], "rating_total": off[keep] + dfn[keep],
                         "poss_off": block.poss[keep]})


def html_table(frame: pd.DataFrame, left=("player",)) -> str:
    """A static table for an email body: attributes rather than a style per cell, which kept a full report
    at a fifth of the size (mail clients strip scripts and most style sheets, so no classes either)."""
    def cell(tag, c, v):
        return f'<{tag}{" align=left" if c in left else ""}>{html.escape(str(v))}</{tag}>'

    head = "".join(cell("th", c, c) for c in frame.columns)
    body = "".join("<tr>" + "".join(cell("td", c, v) for c, v in zip(frame.columns, row)) + "</tr>"
                   for row in frame.itertuples(index=False))
    return ('<table border=1 cellpadding=3 cellspacing=0 align=left style="border-collapse:collapse;'
            'border-color:#e5e7eb;font-size:12px;white-space:nowrap;text-align:right;margin-bottom:8px">'
            f'<thead style="background:#f3f4f6"><tr>{head}</tr></thead><tbody>{body}</tbody></table>'
            '<br clear=all>')


def f2(x) -> str:
    return "" if x is None or not np.isfinite(x) else f"{x:+.2f}"


def pct(x) -> str:
    return "" if x is None or not np.isfinite(x) else f"{100 * x:.0f}%"


# ------------------------------------------------------------------------------------------ main
def main() -> None:
    check_flags()
    lam = float(flag("penalty", sy.RAPM_OFFENSE_LAMBDA))
    n_draws = int(flag("draws", 200))
    n_sims = int(flag("sims", 200))
    rng = np.random.default_rng(int(flag("seed", 0)))
    cfg = load_config()
    log: list = []
    started = time.time()

    print("loading 2024-26 (raw points, regular season and playoffs)", flush=True)
    block = load_block(cfg)
    n, p = block.n, block.p
    print(f"  {n} players, {p - 2 * n} context columns, {block.n_rows:,} rows", flush=True)

    print("checks on the design", flush=True)
    WX = weighted(block.X, block.w)
    gram2 = (block.X.T @ WX).toarray()
    rhs2 = np.asarray(block.X.T @ (block.w * block.y)).ravel()
    check(np.abs(gram2 - block.gram).max() <= 1e-9 * np.abs(block.gram).max()
          and np.abs(rhs2 - block.rhs).max() <= 1e-9 * np.abs(block.rhs).max(),
          "the rebuilt design reproduces LeaveSeasonOutRAPM's normal equations", log)
    del WX, gram2

    print(f"fitting plain RAPM at penalty {lam:,.0f} per side", flush=True)
    factor, beta, M = fit(block, lam, lam)
    ref = block.rapm.ratings(None, lam, lam, 0.0)
    diff = np.abs(beta[:2 * n] - np.concatenate([ref.offense, ref.defense])).max()
    check(diff < 1e-8, f"the fit equals LeaveSeasonOutRAPM.ratings() (max difference {diff:.1e})", log)
    ctx = slice(2 * n, p)
    diff = max(np.abs(M[ctx, ctx] - np.eye(p - 2 * n)).max(), np.abs(M[:2 * n, ctx]).max())
    check(diff < 1e-6, f"the exact expectation (the average RAPM over infinitely many simulated seasons) passes "
                       f"the context through unchanged, so no context leaks into a player (max difference "
                       f"{diff:.1e})", log)
    diff = max(np.abs(M[:n].sum(0)).max(), np.abs(M[n:2 * n].sum(0)).max())
    check(diff < 1e-6, f"the exact expectation sums to zero on each side, the ridge's own centring "
                       f"(max {diff:.1e})", log)
    rss = block.ywy - 2.0 * beta @ block.rhs + beta @ block.gram @ beta
    edf = float(np.trace(M))
    sigma2 = rss / (block.n_rows - edf)
    a_inv = sla.cho_solve(factor, np.eye(p))
    sampling = sigma2 * sla.cho_solve(factor, M.T)            # A^-1 G A^-1: the fit's sampling covariance
    posterior = sigma2 * a_inv
    so, sd_ = np.diagonal(sampling)[:n], np.diagonal(sampling)[n:2 * n]
    sod = np.diagonal(sampling[:n, n:2 * n])
    noise_sd = np.sqrt(np.maximum(so + sd_ - 2.0 * sod, 0.0))
    print(f"  residual variance per possession-weight {sigma2:,.0f} (sd of one possession "
          f"{np.sqrt(sigma2) / 100:.3f} points); effective parameters {edf:,.0f}", flush=True)

    print("truths", flush=True)
    norm = _script("76_bias_groups.py").norm
    t_open_raw, names = openrapm_truth(block)
    t_cons_raw, filled, cinfo = consensus_truth(block, t_open_raw, names, norm)
    print(f"  consensus: {cinfo['rows']} rows, {cinfo['matched']} matched to the block, {cinfo['ambiguous']} "
          f"ambiguous names left out, {cinfo['unmatched_consensus']} consensus rows unmatched; fill shift "
          f"offence {cinfo['shift_off']:+.2f}, defence {cinfo['shift_def']:+.2f} (positive good)", flush=True)
    t_open, t_cons = centre(t_open_raw, block), centre(t_cons_raw, block)
    L = np.linalg.cholesky(posterior[:2 * n, :2 * n])
    T_draws = centre(beta[:2 * n, None] + L @ rng.standard_normal((2 * n, n_draws)), block)
    t_fit = centre(beta[:2 * n], block)
    print(f"  spread (sd over players with 3,000+ possessions, offence / defence): "
          + "; ".join(f"{k} {np.std(v[:n][block.poss > 3000]):.2f} / {np.std(v[n:][block.poss > 3000]):.2f}"
                      for k, v in (("OpenRAPM", t_open), ("consensus", t_cons), ("plain RAPM fit", t_fit),
                                   ("one draw", T_draws[:, 0]))), flush=True)

    print("checks on the truths' signs", flush=True)
    heavy = block.poss > SIGN_MIN_POSS
    for key, t in (("openrapm", t_open), ("consensus", t_cons)):
        c_o = np.corrcoef(t[:n][heavy], beta[:n][heavy])[0, 1]
        c_d = np.corrcoef(t[n:2 * n][heavy], beta[n:2 * n][heavy])[0, 1]
        s_o, s_d = sign_slopes(block, t)
        check(c_o > 0.5 and c_d > 0.5 and s_o > 0 and s_d > 0,
              f"{TRUTH_WORDS[key]} truth: correlation with plain RAPM {c_o:.2f} offence / {c_d:.2f} defence; "
              f"real outcomes on its lineup sums, slope {s_o:.2f} offence / {s_d:.2f} defence", log)

    print("on/off operators and replacements", flush=True)
    ops = onoff_operators(block)
    rep = replacement_structure(ops, n)
    check(ops["both_teams"] == 0, "nobody is on both teams in one game", log)
    for side in ("O", "D"):
        ok = rep[side]["ok"]
        check(np.abs(rep[side]["own"][ok] - 1.0).max() < 1e-9 and np.abs(rep[side]["sums"][ok] - 1.0).max() < 1e-9,
              f"{'offence' if side == 'O' else 'defence'}: his own on/off weight is 1 and the replacement "
              f"weights sum to 1, for all {int(ok.sum())} players who both played and sat", log)
    valid = rep["O"]["ok"] & rep["D"]["ok"] & rep["O"]["R"].any(1) & rep["D"]["R"].any(1)
    filled_share = {s: float(rep[s]["a"][filled].sum() / rep[s]["a"].sum()) for s in ("O", "D")}
    print(f"  consensus truth: {100 * filled_share['O']:.0f}% of offensive and {100 * filled_share['D']:.0f}% of "
          f"defensive replacement possessions are players filled from OpenRAPM", flush=True)

    print("the exact expectation against real refits", flush=True)
    gamma = beta[2 * n:]
    theta_open = np.concatenate([t_open, gamma])
    expected_open = M @ theta_open
    fresh = fresh_fit(block, theta_open, lam, lam)
    check(np.abs(fresh - expected_open[:2 * n]).max() < 1e-8,
          f"noise-free OpenRAPM seasons refitted from scratch equal the exact expectation (max difference "
          f"{np.abs(fresh - expected_open[:2 * n]).max():.1e})", log)

    results = {}
    for key, T in (("openrapm", t_open), ("consensus", t_cons), ("draws", T_draws), ("fit", t_fit)):
        results[key] = inflation(T, M, ops, rep, n)
    inf_open = results["openrapm"]
    shared = shared_court_time(block)
    split = inflation_by_group(t_open, M, rep, shared, n)
    diff = np.abs(sum(split.values()) - inf_open["rapm"][:, 0])[valid].max()
    check(diff < 1e-10, f"the split by shared court time sums to the inflation (max difference {diff:.1e})", log)
    pick = np.flatnonzero(valid & (block.poss > LIST_MIN_POSS))
    pick = pick[np.argsort(-np.abs(inf_open["rapm"][pick, 0]))][:3]
    worst = 0.0
    for i in pick:
        swapped = t_open.copy()
        swapped[:n][rep["O"]["R"][i]] = inf_open["typ_o"][0]
        swapped[n:2 * n][rep["D"]["R"][i]] = inf_open["typ_d"][0]
        after = fresh_fit(block, np.concatenate([swapped, gamma]), lam, lam)
        moved = (fresh[i] - fresh[n + i]) - (after[i] - after[n + i])
        worst = max(worst, abs(moved - inf_open["rapm"][i, 0]))
        print(f"    {names[i]}: refit moves him {moved:+.4f}, the formula says {inf_open['rapm'][i, 0]:+.4f}", flush=True)
    check(worst < 1e-8, f"a real swap-and-refit on three players moves them exactly as the formula says "
                        f"(max difference {worst:.1e})", log)

    print(f"simulating {n_sims} seasons under the OpenRAPM truth", flush=True)
    fits = simulate(block, factor, theta_open, sigma2, n_sims, rng)
    active = np.concatenate([block.active_o, block.active_d])
    mean, spread = fits[:2 * n].mean(1), fits[:2 * n].std(1, ddof=1)
    z = np.abs(mean - expected_open[:2 * n]) / (spread / np.sqrt(n_sims))
    analytic = np.sqrt(np.diagonal(sampling)[:2 * n])
    close_mean = float(np.mean(z[active] < 4.0))
    close_sd = float(np.mean(np.abs(spread[active] / analytic[active] - 1.0) < 0.2))
    check(close_mean >= 0.99, f"simulated average within 4 Monte Carlo errors of the exact one for "
                              f"{100 * close_mean:.1f}% of player ratings", log)
    check(close_sd >= 0.99, f"simulated spread within 20% of the analytic noise for {100 * close_sd:.1f}% "
                            f"of player ratings", log)
    del fits

    print("shares and the penalty curve", flush=True)
    share_rows = []
    for key in ("openrapm", "consensus", "draws", "fit"):
        for row in shares(results[key], block.poss, valid):
            for stat in ("share", "share_rel", "correlation"):
                v = row[stat]
                row[stat + "_p10"], row[stat + "_p90"] = (float(np.percentile(v, 10)), float(np.percentile(v, 90))) \
                    if v.size > 1 else (np.nan, np.nan)
                row[stat] = float(np.median(v)) if v.size > 1 else float(v[0])
            share_rows.append(dict(truth=key, penalty=lam, **row))
    curve = []
    split_light = None
    for lam_c in PENALTY_CURVE:
        f_c, b_c, M_c = fit(block, lam_c, lam_c)
        if lam_c == PENALTY_CURVE[0]:          # the lightest penalty: where a backup's substitution shows most
            split_light = inflation_by_group(t_open, M_c, rep, shared, n)
        for key, T in (("openrapm", t_open), ("consensus", t_cons), ("fit", centre(b_c[:2 * n], block))):
            inf_c = inflation(T, M_c, ops, rep, n)
            for row in shares(inf_c, block.poss, valid):
                row = {k: (float(v[0]) if isinstance(v, np.ndarray) else v) for k, v in row.items()}
                curve.append(dict(truth=key, penalty=lam_c, **row))
                if lam_c != lam:
                    share_rows.append(dict(truth=key, penalty=lam_c, **row))
        keep_c = np.diagonal(M_c)[:n][block.poss > 10000]
        print(f"  penalty {lam_c:>9,.0f}: share left in (OpenRAPM, all players) "
              f"{100 * [c for c in curve if c['truth'] == 'openrapm' and c['penalty'] == lam_c and c['tier'] == 'all'][0]['share']:.0f}%;"
              f" a player with 10,000+ possessions keeps {100 * np.median(keep_c):.0f}% of his own offence", flush=True)
        del f_c, M_c
    share_table = pd.DataFrame(share_rows)
    curve_table = pd.DataFrame(curve)

    # ---------------------------------------------------------------------------------- tables
    o_disp, d_disp = display_ratings(block, beta[:n], beta[n:2 * n])
    keep_o, keep_d = np.diagonal(M)[:n], np.diagonal(M)[n:2 * n]
    player = pd.DataFrame({"player_id": block.player_ids, "player_name": names, "possessions": block.poss,
                           "rapm_off": o_disp, "rapm_def": d_disp, "rapm_total": o_disp + d_disp,
                           "keep_off": keep_o, "keep_def": keep_d, "noise_sd_total": noise_sd, "valid": valid,
                           "consensus_filled": filled})
    for key, t in (("openrapm", t_open), ("consensus", t_cons)):
        r = results[key]
        e = M[:2 * n, :2 * n] @ t
        player[f"truth_total_{key}"] = t[:n] - t[n:2 * n]
        player[f"expected_total_{key}"] = e[:n] - e[n:2 * n]
        player[f"error_total_{key}"] = (e[:n] - t[:n]) - (e[n:2 * n] - t[n:2 * n])
        for col in ("gap", "onoff", "rapm", "rapm_off", "rapm_def", "rel"):
            player[f"{col}_{key}"] = np.where(valid, r[col][:, 0], np.nan)
    for group, _, _ in SHARED_GROUPS:
        tag = group.replace(" ", "_")
        player[f"rapm_openrapm_{tag}"] = np.where(valid, split[group], np.nan)
        player[f"rapm_openrapm_{tag}_pen{PENALTY_CURVE[0]:.0f}"] = np.where(valid, split_light[group], np.nan)
    r = results["draws"]
    for col in ("gap", "onoff", "rapm"):
        player[f"{col}_draws"] = np.where(valid, r[col].mean(1), np.nan)
    player["rapm_draws_p10"] = np.where(valid, np.percentile(r["rapm"], 10, axis=1), np.nan)
    player["rapm_draws_p90"] = np.where(valid, np.percentile(r["rapm"], 90, axis=1), np.nan)
    tables = {"plain": rankings_frame(block, names, beta[:n], beta[n:2 * n])}
    for key, tag in (("openrapm", "openrapm_truth"), ("consensus", "consensus_truth"), ("draws", "your_draws")):
        r = results[key]
        r_off = np.where(valid, r["rapm_off"].mean(1), 0.0)
        r_def = np.where(valid, r["rapm_def"].mean(1), 0.0)       # positive good
        tables[tag] = rankings_frame(block, names, beta[:n] - r_off, beta[n:2 * n] + r_def)
    for tag, col in (("plain", "rank_plain"), ("openrapm_truth", "rank_without_inflation_openrapm")):
        ranks = tables[tag].set_index("player_id").rating_total.rank(ascending=False, method="min")
        player[col] = player.player_id.map(ranks)
    out = ROOT / "outputs"
    player.to_parquet(out / "replacement_sim.parquet", index=False)
    share_table.to_parquet(out / "replacement_sim_share.parquet", index=False)
    print("wrote outputs/replacement_sim.parquet, outputs/replacement_sim_share.parquet", flush=True)
    paths = {}
    for tag, frame in tables.items():
        path = out / ("rapm_2024_26.parquet" if tag == "plain" else f"rapm_2024_26_noinfl_{tag}.parquet")
        frame.to_parquet(path, index=False)
        paths[tag] = path.relative_to(ROOT).as_posix()
    print("wrote " + ", ".join(paths.values()), flush=True)

    # ---------------------------------------------------------------------------------- the report
    def share_line(key, tier="all"):
        row = share_table[(share_table.truth == key) & (share_table.penalty == lam) & (share_table.tier == tier)].iloc[0]
        extra = (f" (10th-90th percentile over draws {pct(row.share_p10)}-{pct(row.share_p90)})"
                 if key == "draws" else "")
        return row, extra

    lines = []
    for key in ("openrapm", "consensus", "draws"):
        row, extra = share_line(key)
        heavy_row, _ = share_line(key, "over 10,000")
        lines.append(f"{TRUTH_WORDS[key]} truth: of the distortion replacements put into raw on/off, RAPM keeps "
                     f"{pct(row.share)}{extra}; players with over 10,000 possessions {pct(heavy_row.share)}.  "
                     f"Divided by the share of his own rating RAPM keeps: {pct(row.share_rel)} "
                     f"({pct(heavy_row.share_rel)} over 10,000).  Negative means RAPM leans the other way.  "
                     f"Correlation {row['correlation']:.2f}.")
    top = player[player.possessions >= sy.MIN_POSSESSIONS].sort_values("rapm_total", ascending=False).head(20)
    big = top.loc[top.rapm_openrapm.abs().idxmax()]
    counts = {k: int((top[f"rapm_{k}"].abs() >= NOTHING).sum()) for k in ("openrapm", "consensus", "draws")}
    lines.append(f"Largest inflation in the top 20 (OpenRAPM truth): {big.player_name} {f2(big.rapm_openrapm)} per "
                 f"100.  Top-20 players at or above {NOTHING}: {counts['openrapm']} (OpenRAPM), "
                 f"{counts['consensus']} (consensus), {counts['draws']} (your draws).")
    light = f"rapm_openrapm_backups_pen{PENALTY_CURVE[0]:.0f}"
    heavy_p = player[valid & (player.possessions > 10000)]
    b_now, b_light = heavy_p["rapm_openrapm_backups"], heavy_p[light]
    w_big = heavy_p.loc[heavy_p["rapm_openrapm_plays_with_him"].abs().idxmax()]
    lines.append(f"True backups only (replacements who share under 10% of his court time), players with over "
                 f"10,000 possessions: at penalty {lam:,.0f} they add a median {f2(b_now.median())} "
                 f"({pct((b_now > 0).mean())} positive); at penalty {PENALTY_CURVE[0]:,.0f} a median "
                 f"{f2(b_light.median())} ({pct((b_light > 0).mean())} positive, the belief's direction), largest "
                 f"{f2(b_light.loc[b_light.abs().idxmax()])}.  The large effects come from teammates he also plays "
                 f"with (30%+ shared): largest {w_big.player_name} {f2(w_big['rapm_openrapm_plays_with_him'])}.")

    def source_rows(frame):
        return pd.DataFrame({"player": frame.player_name, "infl OpenRAPM": frame.rapm_openrapm.map(f2),
                             "backups": frame.rapm_openrapm_backups.map(f2),
                             "some shared": frame.rapm_openrapm_some_shared.map(f2),
                             "plays with him": frame.rapm_openrapm_plays_with_him.map(f2),
                             f"backups at {PENALTY_CURVE[0]:,.0f}": frame[light].map(f2)})

    def rank(v) -> str:
        return "" if not np.isfinite(v) else f"{v:.0f}"

    def top_rows(frame):
        return pd.DataFrame({
            "rank": frame.rank_plain.map(rank), "w/o infl": frame.rank_without_inflation_openrapm.map(rank),
            "player": frame.player_name, "off": frame.rapm_off.map(f2), "def": frame.rapm_def.map(f2),
            "total": frame.rapm_total.map(f2), "poss": frame.possessions.map(lambda v: f"{v:,.0f}"),
            "keeps": ((frame.keep_off + frame.keep_def) / 2).map(pct),
            "error": frame.error_total_openrapm.map(f2), "gap": frame.gap_openrapm.map(f2),
            "infl OpenRAPM": frame.rapm_openrapm.map(f2), "infl consensus": frame.rapm_consensus.map(f2),
            "infl draws": [f"{f2(a)} [{f2(b)}, {f2(c)}]" for a, b, c in
                           zip(frame.rapm_draws, frame.rapm_draws_p10, frame.rapm_draws_p90)],
            "noise sd": frame.noise_sd_total.map(lambda v: f"{v:.2f}")})

    listed = player[valid & (player.possessions >= LIST_MIN_POSS)]
    most = listed.sort_values("rapm_openrapm", ascending=False).head(10)
    least = listed.sort_values("rapm_openrapm").head(10)
    tier_view = share_table[(share_table.penalty == lam) & share_table.truth.isin(["openrapm", "consensus", "draws"])]
    tier_view = pd.DataFrame({"truth": tier_view.truth.map(TRUTH_WORDS), "possessions": tier_view.tier,
                              "players": tier_view.players, "share left in": tier_view.share.map(pct),
                              "relative": tier_view.share_rel.map(pct),
                              "correlation": tier_view["correlation"].map(lambda v: f"{v:.2f}")})
    curve_view = curve_table[curve_table.tier == "all"].pivot(index="penalty", columns="truth", values="share")
    rel_view = curve_table[curve_table.tier == "all"].pivot(index="penalty", columns="truth", values="share_rel")
    curve_view = pd.DataFrame({"penalty": [f"{v:,.0f}" for v in curve_view.index],
                               "OpenRAPM": curve_view.openrapm.map(pct), "relative": rel_view.openrapm.map(pct),
                               "consensus": curve_view.consensus.map(pct),
                               "RAPM fit as truth": curve_view.fit.map(pct)})

    print("\n" + "\n".join(lines))
    print("\nshare left in by possessions (penalty {:,.0f})\n".format(lam) + tier_view.to_string(index=False))
    print("\npenalty curve (all players)\n" + curve_view.to_string(index=False))
    print("\n2024-26 plain RAPM top 20\n" + top_rows(top).to_string(index=False))
    print("\nwhere the top 20's inflation comes from (OpenRAPM truth)\n" + source_rows(top).to_string(index=False))
    print(f"\nmost inflated ({LIST_MIN_POSS:,.0f}+ possessions)\n" + top_rows(most).to_string(index=False))
    print(f"\nmost deflated ({LIST_MIN_POSS:,.0f}+ possessions)\n" + top_rows(least).to_string(index=False))

    glossary = ("<b>w/o infl</b>: his plain-RAPM rank once his inflation (OpenRAPM truth) is taken out. "
                "<b>gap</b>: his replacements' true rating minus the league-typical replacement (negative = worse). "
                "<b>infl</b>: how far his expected RAPM falls when only his replacements are swapped for typical "
                "ones (positive = his replacements make him look better). <b>error</b>: expected RAPM minus truth, "
                "shrinkage included. <b>keeps</b>: the share of his own true rating RAPM keeps. <b>noise sd</b>: "
                "the fit's own sampling noise. <b>backups</b>: replacements who share under 10% of his court "
                "time; <b>plays with him</b>: 30% or more (a staggered co-star). Points per 100; under 0.1 is "
                "nothing.")
    page = ('<div style="font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;color:#1f2933">'
            '<h2 style="font-size:18px;margin:0 0 6px">Does 3-season RAPM overrate players with bad replacements?</h2>'
            f'<p style="font-size:13px;color:#6b7280;margin:0 0 8px">Plain RAPM on 2024-26, raw points, penalty '
            f'{lam:,.0f} per side, {n} players. Simulated with the truth known; every check passed.</p>'
            + "".join(f'<p style="font-size:14px;margin:0 0 6px">{html.escape(ln)}</p>' for ln in lines)
            + f'<p style="font-size:12px;color:#6b7280">{glossary}</p>'
            + '<h3 style="font-size:15px">2024-26 plain RAPM top 20</h3>' + html_table(top_rows(top))
            + '<h3 style="font-size:15px">Where the top 20\'s inflation comes from (OpenRAPM truth)</h3>'
            + html_table(source_rows(top))
            + f'<h3 style="font-size:15px">Most inflated ({LIST_MIN_POSS:,.0f}+ possessions)</h3>' + html_table(top_rows(most))
            + f'<h3 style="font-size:15px">Most deflated ({LIST_MIN_POSS:,.0f}+ possessions)</h3>' + html_table(top_rows(least))
            + '<h3 style="font-size:15px">Share left in, by possessions</h3>' + html_table(tier_view, left=("truth",))
            + '<h3 style="font-size:15px">Penalty curve (all players)</h3>' + html_table(curve_view, left=())
            + '<h3 style="font-size:15px">Checks</h3>'
            + "".join(f'<p style="font-size:12px;margin:0 0 3px">{html.escape(c)}</p>' for c in log)
            + "</div>")
    (out / "replacement_sim_phone.html").write_text(page, encoding="utf-8")
    print("wrote outputs/replacement_sim_phone.html", flush=True)

    subprocess.run([sys.executable, str(ROOT / "scripts" / "66_compare.py"),
                    *[f"{tag}={path}" for tag, path in paths.items()], "--season=2026", "--top=20", "--open=0",
                    "--title=Plain RAPM 2024-26, and with its replacement inflation removed under each truth"],
                   check=True, cwd=str(ROOT))
    print(f"\ndone in {time.time() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
