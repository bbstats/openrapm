"""Drop the bad replacements: does a star's plain RAPM fall when his worst backups' possessions are removed?

    python scripts/80_drop_replacements.py [--penalty=3000] [--floor=0.40] [--top=25] [--bad=-1]

**Why this exists.**  The owner's direct real-data test of "RAPM overrates players with bad replacements"
(scripts 77-79 tested it by simulation).  Every 2024-26 player is scored

    his RAPM x min%  -  largest(replacement's RAPM x min% while he is off the floor)

counting only replacements rated -1 or worse.  The top 25 with at least 40% of their teams' possessions are
taken, and for each one plain RAPM is refitted with EVERY possession of those bad replacements removed -- any
team, any season.  If the belief holds, he falls once the evidence of his bad backups is gone.

**Definitions.**  RAPM: plain pooled RAPM over 2024-26 (`LeaveSeasonOutRAPM`, raw points, regular season and
playoffs) at the owner's penalty, 3,000 per side, where the scale looks like public RAPM.  min%: his
possessions over his teams' possessions (data/cache/roles_RSPO.parquet).  A replacement: a teammate on the floor
more when he sits than when he plays, in games he played (77's rule), and on the floor for at least 5% of his
off-court possessions.  min% while he is off: the share of his off-court possessions with that replacement on.
largest(...): the most damaging one, the most negative RAPM x min% while off.

**How to read it.**  Points per 100 possessions, positive good.
  change        his total after the refit minus before, both centred on the same players
  luck alone    how far his total would move, one standard deviation, from dropping those possessions if only
                luck were at work -- exact under the fit's own noise, no extra refits
  model says    the move the lineup structure alone predicts if the players' true ratings were OpenRAPM's (77's
                truth) -- what the simulations of 77-79 expect here
The belief predicts negative changes larger than luck alone.  Under 0.1 per 100 is nothing.  The run stops on a
failed check.

Writes outputs/drop_replacements.parquet and outputs/drop_replacements_phone.html.
"""
import html
import importlib.util
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
os.environ.setdefault("NUMBA_NUM_THREADS", os.environ.get("OPENRAPM_NUMBA_THREADS", "4"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.linalg as sla  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _cli import check_flags, flag  # noqa: E402
from eracoef.config import load_config  # noqa: E402

# 77's block, fit, on/off operators, replacements and truth -- one definition of each, as 70 borrows 63's
_spec = importlib.util.spec_from_file_location("_borrowed_77", ROOT / "scripts" / "77_replacement_sim.py")
r77 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(r77)
check = r77.check

REAL_SEASONS = (2024, 2025, 2026)
MIN_OFF_SHARE = 0.05          # a replacement must be on the floor for 5% of his off-court possessions
NOTHING = 0.1
SIM_DRAWS = 200


def f2(x) -> str:
    return "" if x is None or not np.isfinite(x) else f"{x:+.2f}"


def pct(x) -> str:
    return "" if x is None or not np.isfinite(x) else f"{100 * x:.0f}%"


def setup(cfg, lam: float) -> SimpleNamespace:
    """The 2024-26 block, plain RAPM at penalty `lam`, the OpenRAPM truth, min%, the replacement shares and every
    row's ten players -- what a drop-and-refit needs, built once."""
    block = r77.load_block(cfg)
    n, p = block.n, block.p
    ops = r77.onoff_operators(block)
    rep = r77.replacement_structure(ops, n)
    t_raw, names = r77.openrapm_truth(block)
    factor, beta, M = r77.fit(block, lam, lam)
    o, d = r77.display_ratings(block, beta[:n], beta[n:2 * n])
    rss = block.ywy - 2.0 * beta @ block.rhs + beta @ block.gram @ beta
    sigma2 = rss / (block.n_rows - float(np.trace(M)))
    theta = np.concatenate([r77.centre(t_raw, block), beta[2 * n:]])   # OpenRAPM truth, the fit's own context
    roles = pd.read_parquet(ROOT / "data" / "cache" / "roles_RSPO.parquet")
    roles = roles[roles.season.isin(REAL_SEASONS)].groupby("player_id")[["poss_on", "team_poss"]].sum()
    share = (roles.poss_on / roles.team_poss).reindex(block.player_ids).fillna(0.0).to_numpy()
    G = block.gram
    with np.errstate(invalid="ignore", divide="ignore"):
        p_on = np.nan_to_num(G[:n, :n] / np.diagonal(G)[:n][:, None])   # share of his on-court possessions j is on for
    p_off = p_on - ops["O"]["D"][:, :n]                                    # ... and of his off-court ones (games he played)
    # every row's ten players (a row is one stint-side: 10 player columns, then its season's context columns)
    X = block.X
    width = np.diff(X.indptr)
    if not (width == width[0]).all():
        raise SystemExit("rows carry different numbers of stored entries; the lineup read below assumes they do not")
    cols = X.indices.reshape(-1, int(width[0]))[:, :10]
    if (cols >= 2 * n).any():
        raise SystemExit("a context column sits among a row's first ten entries")
    return SimpleNamespace(block=block, n=n, p=p, ops=ops, rep=rep, names=names, factor=factor, beta=beta, M=M,
                           off=o, dfn=d, total=o + d, sigma2=sigma2, theta=theta, expected_full=M @ theta,
                           penalty=np.concatenate([np.full(2 * n, lam), np.zeros(p - 2 * n)]), share=share,
                           p_off=p_off, on_floor=cols % n, lam=lam)


def replacement_mask(ctx: SimpleNamespace, bad=None, better=None) -> np.ndarray:
    """(player, teammate) pairs where the teammate is his replacement -- on the floor more when he sits than when he
    plays, and for at least MIN_OFF_SHARE of his off-court possessions -- rated `bad` or worse, or above `better`."""
    mask = ctx.rep["O"]["R"] & (ctx.p_off >= MIN_OFF_SHARE)
    if bad is not None:
        mask = mask & (ctx.total <= bad)[None, :]
    if better is not None:
        mask = mask & (ctx.total > better)[None, :]
    return mask


def refit(ctx: SimpleNamespace, i: int, removed: np.ndarray, drop: np.ndarray) -> SimpleNamespace:
    """Plain RAPM refitted without the rows in `drop`: player i's change on one zero point, its luck-alone standard
    deviation, and the model's move split into his own shrinkage and the `removed` players' leakage."""
    block, n, p = ctx.block, ctx.n, ctx.p
    G = block.gram
    X_d, w_d = block.X[drop], block.w[drop]
    G_kept = G - (X_d.T @ r77.weighted(X_d, w_d)).toarray()
    rhs_kept = block.rhs - np.asarray(X_d.T @ (w_d * block.y[drop])).ravel()
    kept_factor = sla.cho_factor(G_kept + np.diag(ctx.penalty))
    beta_kept = sla.cho_solve(kept_factor, rhs_kept)
    gone = float(np.diagonal(G_kept)[np.concatenate([removed, n + removed])].max()) if removed.size else 0.0
    # one zero point for both fits: possession-weighted zero per side over the players present in both
    common = block.active_o & block.active_d
    common[removed] = False
    w_o = np.where(common, block.poss, 0.0)
    w_dd = np.where(common, block.poss_def, 0.0)
    c = np.zeros(p)                                              # his centred total = c @ coefficients
    c[:n] -= w_o / w_o.sum()
    c[n:2 * n] += w_dd / w_dd.sum()
    c[i] += 1.0
    c[n + i] -= 1.0
    u = sla.cho_solve(ctx.factor, c)
    v = sla.cho_solve(kept_factor, c)
    # the model's move split: his own rating shrinking because the possessions that measured it are gone, the
    # removed players' leakage into him leaving with them (the belief's part, for bad replacements), and the rest
    parts = {}
    for part, entries in (("own", np.array([i, n + i])), ("replacements", np.concatenate([removed, n + removed]))):
        piece = np.zeros(p)
        piece[entries] = ctx.theta[entries]
        parts[part] = float(c @ (sla.cho_solve(kept_factor, G_kept @ piece) - ctx.M @ piece))
    return SimpleNamespace(
        change=float(c @ (beta_kept - ctx.beta)),
        luck=float(np.sqrt(ctx.sigma2 * (u @ G @ u - 2.0 * u @ G_kept @ v + v @ G_kept @ v))),
        model=float(c @ (sla.cho_solve(kept_factor, G_kept @ ctx.theta) - ctx.expected_full)),
        model_own=parts["own"], model_replacements=parts["replacements"], gone=gone, G_kept=G_kept,
        kept_factor=kept_factor, c=c, data_share=float(w_d.sum() / block.w.sum()))


def main() -> None:
    check_flags()
    lam = float(flag("penalty", 3000))
    floor = float(flag("floor", 0.40))
    top_n = int(flag("top", 25))
    bad = float(flag("bad", -1.0))
    cfg = load_config()
    log: list = []
    started = time.time()
    rng = np.random.default_rng(0)

    print(f"plain RAPM 2024-26 at penalty {lam:,.0f}", flush=True)
    ctx = setup(cfg, lam)
    block, n, p, names = ctx.block, ctx.n, ctx.p, ctx.names
    o, d, total, share, p_off = ctx.off, ctx.dfn, ctx.total, ctx.share, ctx.p_off
    G, X, factor, theta, sigma2 = block.gram, block.X, ctx.factor, ctx.theta, ctx.sigma2

    # ---------------------------------------------------------------------------------- score and pick
    subs = replacement_mask(ctx, bad=bad)
    damage = np.where(subs, total[None, :] * p_off, 0.0)
    worst = damage.min(1)
    worst_j = damage.argmin(1)
    score = total * share - worst
    eligible = (share >= floor) & (worst < 0) & ctx.rep["O"]["ok"]
    picks = np.argsort(-np.where(eligible, score, -np.inf))[:top_n]
    print(f"  {int(eligible.sum())} players have min% of {floor:.0%}+ and a replacement rated {bad:+.0f} or worse; "
          f"one possession's noise {np.sqrt(sigma2) / 100:.3f} points", flush=True)

    # ---------------------------------------------------------------------------------- the refits
    rows = []
    for rank, i in enumerate(picks, 1):
        t0 = time.time()
        removed = np.flatnonzero(subs[i])
        drop = np.isin(ctx.on_floor, removed).any(axis=1)
        r = refit(ctx, i, removed, drop)
        change, luck, model, kept_factor, c = r.change, r.luck, r.model, r.kept_factor, r.c
        parts = {"own": r.model_own, "replacements": r.model_replacements}
        if rank == 1:
            keep = ~drop
            X_k = X[keep]
            direct = (X_k.T @ r77.weighted(X_k, block.w[keep])).toarray()
            miss = float(np.abs(direct - r.G_kept).max() / np.abs(G).max())
            check(miss < 1e-10, f"subtracting the dropped rows gives the kept rows' normal equations (relative miss "
                                f"{miss:.1e})", log)
            nothing = sla.cho_solve(sla.cho_factor(G + np.diag(ctx.penalty)), block.rhs)
            check(np.abs(nothing - ctx.beta).max() < 1e-8, "a refit that drops nothing reproduces the baseline", log)
            check(r.gone < 1e-6, f"{names[i]}: his {removed.size} removed replacements have no possessions left in "
                                 f"the refit (largest {r.gone:.1e})", log)
        elif r.gone >= 1e-6:
            raise SystemExit(f"{names[i]}: removed players still have {r.gone:.1e} possessions in the refit")
        if rank == 1:
            # the luck-alone formula against simulated seasons: each draw goes through BOTH fits, the refit
            # seeing only the kept rows of the same draw
            drop_parts, start = [], 0
            for X_s in block.X_parts:
                drop_parts.append(drop[start:start + X_s.shape[0]])
                start += X_s.shape[0]
            changes = []
            for k0 in range(0, SIM_DRAWS, 25):
                k = min(25, SIM_DRAWS - k0)
                rhs_full, rhs_drop = np.zeros((p, k)), np.zeros((p, k))
                for X_s, w_s, dp in zip(block.X_parts, block.w_parts, drop_parts):
                    ys = (X_s @ theta)[:, None] + np.sqrt(sigma2 / w_s)[:, None] * rng.standard_normal((w_s.size, k))
                    rhs_full += np.asarray(X_s.T @ (w_s[:, None] * ys))
                    if dp.any():
                        rhs_drop += np.asarray(X_s[dp].T @ (w_s[dp][:, None] * ys[dp]))
                changes.append(c @ (sla.cho_solve(kept_factor, rhs_full - rhs_drop) - sla.cho_solve(factor, rhs_full)))
            simulated = float(np.std(np.concatenate(changes), ddof=1))
            check(abs(simulated / luck - 1.0) < 0.15, f"{names[i]}: luck alone {luck:.3f} against {SIM_DRAWS} simulated "
                                                      f"seasons {simulated:.3f}", log)
        j = worst_j[i]
        rows.append(dict(rank=rank, player_id=int(block.player_ids[i]), player=names[i], off=o[i], dfn=d[i],
                         total=total[i], min_share=share[i], score=score[i], worst=names[j], worst_rapm=total[j],
                         worst_off_share=p_off[i, j], removed=int(removed.size),
                         removed_names=", ".join(f"{names[k]} ({total[k]:+.1f})" for k in removed),
                         data_share=r.data_share, refit=total[i] + change,
                         change=change, luck=luck, model=model, model_own=parts["own"],
                         model_replacements=parts["replacements"],
                         model_rest=model - parts["own"] - parts["replacements"]))
        print(f"  {rank:>2} {names[i]:<24} {f2(total[i])}: without {removed.size} bad replacements "
              f"({pct(rows[-1]['data_share'])} of the data) {f2(total[i] + change)}, change {f2(change)} "
              f"(luck alone {luck:.2f}, model says {f2(model)}) ({time.time() - t0:.0f}s)", flush=True)

    tab = pd.DataFrame(rows)
    out = ROOT / "outputs"
    tab.to_parquet(out / "drop_replacements.parquet", index=False)
    print("wrote outputs/drop_replacements.parquet", flush=True)

    # ---------------------------------------------------------------------------------- the report
    fell = int((tab.change <= -NOTHING).sum())
    rose = int((tab.change >= NOTHING).sum())
    beyond = int((tab.change < -2.0 * tab.luck).sum())
    above = int((tab.change > 2.0 * tab.luck).sum())
    corr = float(np.corrcoef(tab.change, tab.model)[0, 1])
    tab["beyond"] = (tab.change - tab.model) / tab.luck
    lines = [
        f"Plain RAPM 2024-26 at penalty {lam:,.0f}. The {len(tab)} players whose replacements look worst by the owner's "
        f"score, each refitted with every possession of his replacements rated {bad:+.0f} or worse removed.",
        f"Median change {f2(tab.change.median())} per 100. {fell} of {len(tab)} fell by {NOTHING} or more, {rose} rose "
        f"by {NOTHING} or more. {beyond} fell by more than twice their luck alone, {above} rose by more than that.",
        f"The model expected a median {f2(tab.model.median())} (correlation with the real changes {corr:.2f}), made of: "
        f"his own rating shrinking because the possessions that measured it are gone {f2(tab.model_own.median())}, the "
        f"bad replacements' leakage into him leaving {f2(tab.model_replacements.median())} (the belief's part), and "
        f"everything else {f2(tab.model_rest.median())}.",
        f"Real minus model: median {f2((tab.change - tab.model).median())}; beyond twice luck alone below the model "
        f"{int((tab.beyond < -2).sum())} of {len(tab)}, above it {int((tab.beyond > 2).sum())}."]
    view = pd.DataFrame({
        "#": tab["rank"], "player": tab.player, "off": tab.off.map(f2), "def": tab.dfn.map(f2),
        "total": tab.total.map(f2), "min%": tab.min_share.map(pct),
        "worst replacement": [f"{w} ({f2(r)}, {pct(s)} of his off time)" for w, r, s in
                              zip(tab.worst, tab.worst_rapm, tab.worst_off_share)],
        "removed": tab.removed, "data removed": tab.data_share.map(lambda v: f"{100 * v:.1f}%"),
        "refit total": tab.refit.map(f2), "change": tab.change.map(f2), "luck alone": tab.luck.map(lambda v: f"{v:.2f}"),
        "change / luck": (tab.change / tab.luck).map(lambda v: f"{v:+.1f}"), "model says": tab.model.map(f2),
        "model: own shrinks": tab.model_own.map(f2), "model: replacements leave": tab.model_replacements.map(f2),
        "real beyond model / luck": tab.beyond.map(lambda v: f"{v:+.1f}")})
    removed_view = pd.DataFrame({"player": tab.player, "replacements removed (their RAPM)": tab.removed_names})
    print("\n" + "\n".join(lines))
    print("\n" + view.to_string(index=False))
    glossary = ("Points per 100 possessions, positive good. <b>min%</b>: his share of his teams' possessions. <b>worst "
                "replacement</b>: the one with the most negative RAPM x share of his off-court possessions. <b>change</b>: "
                "his total after the refit minus before. <b>luck alone</b>: one standard deviation of the move that "
                "dropping those possessions would cause by luck alone. <b>model says</b>: the move the lineup structure "
                "alone predicts if true ratings were OpenRAPM's, split into <b>own shrinks</b> (his rating shrinking "
                "further because the possessions that measured it are gone) and <b>replacements leave</b> (the bad "
                "replacements' leakage into him disappearing -- the belief's part). <b>real beyond model / luck</b>: "
                "how far the real change sits past the model, in units of luck alone. Under 0.1 is nothing.")
    page = ('<div style="font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;color:#1f2933">'
            '<h2 style="font-size:18px;margin:0 0 6px">Drop the bad replacements: do the stars fall?</h2>'
            + "".join(f'<p style="font-size:14px;margin:0 0 6px">{html.escape(ln)}</p>' for ln in lines)
            + f'<p style="font-size:12px;color:#6b7280">{glossary}</p>'
            + '<h3 style="font-size:15px">The 25 players</h3>' + r77.html_table(view, left=("player", "worst replacement"))
            + '<h3 style="font-size:15px">Who was removed for each</h3>'
            + r77.html_table(removed_view, left=("player", "replacements removed (their RAPM)"))
            + '<h3 style="font-size:15px">Checks</h3>'
            + "".join(f'<p style="font-size:12px;margin:0 0 3px">{html.escape(ln)}</p>' for ln in log) + "</div>")
    (out / "drop_replacements_phone.html").write_text(page, encoding="utf-8")
    print(f"wrote outputs/drop_replacements_phone.html\ndone in {time.time() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
