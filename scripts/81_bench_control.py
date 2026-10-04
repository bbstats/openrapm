"""Bench control: do stars with bad replacements fall further than stars with decent ones, at equal data loss?

    python scripts/81_bench_control.py [--penalty=3000] [--floor=0.40] [--k=3] [--draws=100] [--seed=0]

**Why this exists.**  Script 80 removed every possession of 25 stars' bad replacements (rated -1 or worse) and
19 of 25 fell, median -0.78 per 100.  But those replacements filled about 95% of most stars' bench time, so the
removal also took away nearly all the evidence that measures the star, and a ridge shrinks a star it can no longer
see.  This is the control the owner asked for: the same operation on stars whose replacements are NOT bad.

  treatment   80's 25 stars, their replacements rated -1 or worse removed (80's refits, recomputed and checked)
  control     every player with 40%+ of his teams' possessions whose bench is decent -- a -1-or-worse replacement
              on the floor for under a third of his bench time -- with his LESS-bad replacements (rated above -1)
              removed, every possession, any team, any season
  matched     each star against the 3 controls closest in starting rating, share of bench time removed and share
              of court time removed; the differences bootstrapped over the 25 stars
  regression  change ~ bad replacements + rating + bench share + court share + rating x each share, bootstrapped
              over players
  literal     for the stars who have enough less-bad bench time to match (at least as much as the removal took):
              the same amount of his bench time, court time and other data removed at random from rows WITHOUT his
              bad replacements, `--draws` times

**How to read it.**  Points per 100 possessions.  "Extra fall for having bad replacements" is a star's change minus
his matched controls' change: negative = the belief (bad replacements were propping him up).  Zero = the fall in 80
was only the lost data.  Bench time: his team's possessions in games he played while he sat.

Writes outputs/bench_control.parquet and outputs/bench_control_phone.html.
"""
import html
import importlib.util
import os
import sys
import time
from pathlib import Path

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

# 80's setup, replacement rule and refit (and through it 77's machinery) -- one definition of each
_spec = importlib.util.spec_from_file_location("_borrowed_80", ROOT / "scripts" / "80_drop_replacements.py")
s80 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s80)
r77 = s80.r77
check = r77.check

BAD = -1.0
DECENT = 1.0 / 3.0            # a control's bench: a bad replacement on the floor for under a third of it
BOOT = 10000


def f2(x) -> str:
    return "" if x is None or not np.isfinite(x) else f"{x:+.2f}"


def pct(x) -> str:
    return "" if x is None or not np.isfinite(x) else f"{100 * x:.0f}%"


def court_and_bench(ctx, i: int) -> tuple:
    """Rows where he is on the floor, and his bench time: his team's rows in games he played, him off the floor.
    His team in a game is read off the rows he played in (the offence's team-game when he attacks, the defence's
    when he defends), so a mid-season trade needs no team ids."""
    block, floor = ctx.block, ctx.on_floor
    on = (floor == i).any(axis=1)
    games = np.union1d(np.unique(block.tg_off[(floor[:, :5] == i).any(axis=1)]),
                       np.unique(block.tg_def[(floor[:, 5:] == i).any(axis=1)]))
    team = np.isin(block.tg_off, games) | np.isin(block.tg_def, games)
    return on, team & ~on, team


def change_only(ctx, i: int, drop: np.ndarray) -> float:
    """80's refit reduced to the one number the random draws need: his centred change."""
    block, n, p = ctx.block, ctx.n, ctx.p
    X_d, w_d = block.X[drop], block.w[drop]
    G_kept = block.gram - (X_d.T @ r77.weighted(X_d, w_d)).toarray()
    rhs_kept = block.rhs - np.asarray(X_d.T @ (w_d * block.y[drop])).ravel()
    beta_kept = sla.cho_solve(sla.cho_factor(G_kept + np.diag(ctx.penalty)), rhs_kept)
    common = block.active_o & block.active_d
    w_o, w_dd = np.where(common, block.poss, 0.0), np.where(common, block.poss_def, 0.0)
    c = np.zeros(p)
    c[:n] -= w_o / w_o.sum()
    c[n:2 * n] += w_dd / w_dd.sum()
    c[i] += 1.0
    c[n + i] -= 1.0
    return float(c @ (beta_kept - ctx.beta))


def take(pool: np.ndarray, w: np.ndarray, target: float, rng) -> np.ndarray:
    """A random set of rows from `pool` whose possessions first reach `target` (all of the pool if it is short)."""
    idx = rng.permutation(np.flatnonzero(pool))
    stop = int(np.searchsorted(np.cumsum(w[idx]), target)) + 1
    out = np.zeros(pool.size, dtype=bool)
    out[idx[:stop]] = True
    return out


def main() -> None:
    check_flags()
    lam = float(flag("penalty", 3000))
    floor = float(flag("floor", 0.40))
    k = int(flag("k", 3))
    draws = int(flag("draws", 100))
    rng = np.random.default_rng(int(flag("seed", 0)))
    cfg = load_config()
    log: list = []
    started = time.time()

    print(f"plain RAPM 2024-26 at penalty {lam:,.0f}", flush=True)
    ctx = s80.setup(cfg, lam)
    block, n, names, total, share, w = ctx.block, ctx.n, ctx.names, ctx.total, ctx.share, ctx.block.w
    bad = s80.replacement_mask(ctx, bad=BAD)
    less = s80.replacement_mask(ctx, better=BAD)
    stars = pd.read_parquet(ROOT / "outputs" / "drop_replacements.parquet")
    index = {int(pid): j for j, pid in enumerate(block.player_ids)}
    star_idx = [index[int(pid)] for pid in stars.player_id]

    def measure(i: int, group: str, removed: np.ndarray) -> dict:
        on, bench, _ = court_and_bench(ctx, i)
        drop = np.isin(ctx.on_floor, removed).any(axis=1)
        bad_rows = np.isin(ctx.on_floor, np.flatnonzero(bad[i])).any(axis=1)
        r = s80.refit(ctx, i, removed, drop)
        if r.gone >= 1e-6:
            raise SystemExit(f"{names[i]}: removed players still have possessions in the refit")
        weights = block.poss[removed]
        return dict(group=group, player_id=int(block.player_ids[i]), player=names[i], rating=total[i],
                    min_share=share[i], bad_bench=float(w[bench & bad_rows].sum() / w[bench].sum()),
                    bench_removed=float(w[drop & bench].sum() / w[bench].sum()),
                    court_removed=float(w[drop & on].sum() / w[on].sum()), data_removed=r.data_share,
                    removed=int(removed.size), removed_rating=float(np.average(total[removed], weights=weights)),
                    change=r.change, luck=r.luck, model=r.model, model_own=r.model_own,
                    model_replacements=r.model_replacements)

    # ---------------------------------------------------------------------------------- treatment and control
    rows = []
    print("treatment: 80's stars without their bad replacements", flush=True)
    for i, (_, star) in zip(star_idx, stars.iterrows()):
        rows.append(measure(i, "bad replacements", np.flatnonzero(bad[i])))
    tr = pd.DataFrame(rows)
    miss = float(np.abs(tr.change.to_numpy() - stars.change.to_numpy()).max())
    check(miss < 1e-10, f"the 25 treatment refits reproduce script 80 (max difference {miss:.1e})", log)

    candidates = [i for i in np.flatnonzero((share >= floor) & ctx.rep["O"]["ok"] & less.any(axis=1))
                  if i not in set(star_idx)]
    print(f"control: {len(candidates)} players with {floor:.0%}+ of possessions and a less-bad replacement; keeping "
          f"those whose bench is decent", flush=True)
    for i in candidates:
        on, bench, _ = court_and_bench(ctx, i)
        bad_rows = np.isin(ctx.on_floor, np.flatnonzero(bad[i])).any(axis=1)
        if w[bench & bad_rows].sum() / w[bench].sum() >= DECENT:
            continue
        rows.append(measure(i, "decent replacements", np.flatnonzero(less[i])))
    tab = pd.DataFrame(rows)
    ctl = tab[tab.group == "decent replacements"].reset_index(drop=True)
    tr = tab[tab.group == "bad replacements"].reset_index(drop=True)
    check(len(ctl) >= 3 * k, f"{len(ctl)} control players (decent bench, less-bad replacements removed)", log)
    print(f"  treatment removed replacements rated {f2(tr.removed_rating.median())} (median), bench time "
          f"{pct(tr.bench_removed.median())}; control removed {f2(ctl.removed_rating.median())}, bench time "
          f"{pct(ctl.bench_removed.median())}", flush=True)

    # ---------------------------------------------------------------------------------- matched comparison
    feats = ["rating", "bench_removed", "court_removed"]
    scale = tab[feats].std()
    zt, zc = (tr[feats] / scale).to_numpy(), (ctl[feats] / scale).to_numpy()
    dist = np.sqrt(((zt[:, None, :] - zc[None, :, :]) ** 2).sum(-1))
    nearest = np.argsort(dist, axis=1)[:, :k]
    tr["matched"] = [", ".join(ctl.player[j] for j in row) for row in nearest]
    tr["matched_change"] = [ctl.change[row].mean() for row in nearest]
    tr["matched_own"] = [ctl.model_own[row].mean() for row in nearest]
    tr["matched_rating"] = [ctl.rating[row].mean() for row in nearest]
    tr["matched_bench"] = [ctl.bench_removed[row].mean() for row in nearest]
    tr["extra"] = tr.change - tr.matched_change
    boot = rng.choice(tr.extra.to_numpy(), size=(BOOT, len(tr)), replace=True).mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    # the controls lost somewhat more data than their stars; take each player's expected shrinkage (the model's
    # own-shrinkage) out of his change before comparing (its own generator, so every draw above is unchanged)
    tr["extra_adj"] = (tr.change - tr.model_own) - [(ctl.change[row] - ctl.model_own[row]).mean() for row in nearest]
    boot_adj = np.random.default_rng(1).choice(tr.extra_adj.to_numpy(), size=(BOOT, len(tr)), replace=True).mean(axis=1)
    lo_adj, hi_adj = np.percentile(boot_adj, [2.5, 97.5])
    reused = pd.Series([ctl.player[j] for row in nearest for j in row]).value_counts()

    # ---------------------------------------------------------------------------------- regression, players bootstrapped
    def design(frame):
        r_, b_, c_ = frame.rating.to_numpy(), frame.bench_removed.to_numpy(), frame.court_removed.to_numpy()
        return np.column_stack([np.ones(len(frame)), (frame.group == "bad replacements").to_numpy(float), r_, b_, c_,
                                r_ * b_, r_ * c_])

    coef = np.linalg.lstsq(design(tab), tab.change.to_numpy(), rcond=None)[0][1]
    reg = []
    for _ in range(2000):
        sample = pd.concat([tr.sample(len(tr), replace=True, random_state=rng.integers(1 << 31)),
                            ctl.sample(len(ctl), replace=True, random_state=rng.integers(1 << 31))])
        reg.append(np.linalg.lstsq(design(sample), sample.change.to_numpy(), rcond=None)[0][1])
    reg_lo, reg_hi = np.percentile(reg, [2.5, 97.5])

    # ---------------------------------------------------------------------------------- the literal version
    literal = []
    for i, (_, star) in zip(star_idx, stars.iterrows()):
        on, bench, team = court_and_bench(ctx, i)
        removed = np.flatnonzero(bad[i])
        drop = np.isin(ctx.on_floor, removed).any(axis=1)
        bad_rows = drop
        targets = [(drop & on, on & ~bad_rows), (drop & bench, bench & ~bad_rows), (drop & ~team, ~team & ~bad_rows)]
        if w[bench & ~bad_rows].sum() < w[drop & bench].sum():
            continue
        t0 = time.time()
        moves = []
        for _ in range(draws):
            placebo = np.zeros(drop.size, dtype=bool)
            for taken, pool in targets:
                placebo |= take(pool, w, float(w[taken].sum()), rng)
            moves.append(change_only(ctx, i, placebo))
        moves = np.asarray(moves)
        real = float(star.change)
        literal.append(dict(player=names[i], rating=total[i], bad_change=real, placebo_median=float(np.median(moves)),
                            placebo_lo=float(np.percentile(moves, 5)), placebo_hi=float(np.percentile(moves, 95)),
                            below=float(np.mean(moves <= real))))
        print(f"  literal: {names[i]:<24} bad replacements removed {f2(real)}; same amount of less-bad data at random "
              f"{f2(np.median(moves))} [{f2(np.percentile(moves, 5))}, {f2(np.percentile(moves, 95))}] "
              f"({time.time() - t0:.0f}s)", flush=True)
    literal = pd.DataFrame(literal)

    out = ROOT / "outputs"
    tab.to_parquet(out / "bench_control.parquet", index=False)
    print("wrote outputs/bench_control.parquet", flush=True)

    # ---------------------------------------------------------------------------------- the report
    lines = [
        f"Plain RAPM 2024-26 at penalty {lam:,.0f}. Treatment: 80's {len(tr)} stars with their bad replacements removed "
        f"(rated {f2(tr.removed_rating.median())} on median; {pct(tr.bench_removed.median())} of their bench time went). "
        f"Control: {len(ctl)} players with decent benches and their less-bad replacements removed (rated "
        f"{f2(ctl.removed_rating.median())}; {pct(ctl.bench_removed.median())} of their bench time went).",
        f"Median change: treatment {f2(tr.change.median())}, control {f2(ctl.change.median())}.",
        f"Extra fall for having bad replacements, each star against his {k} closest controls: mean {f2(tr.extra.mean())} "
        f"per 100 (95% bootstrap interval {f2(lo)} to {f2(hi)}). Regression at equal data loss and rating: "
        f"{f2(coef)} ({f2(reg_lo)} to {f2(reg_hi)}). Negative would mean bad replacements were propping stars up.",
        f"The controls lost somewhat more data: the model's own-shrinkage is {f2(tr.matched_own.median())} for the matched "
        f"controls against {f2(tr.model_own.median())} for the stars. With each player's expected shrinkage taken out "
        f"first, the extra fall is {f2(tr.extra_adj.mean())} ({f2(lo_adj)} to {f2(hi_adj)}).",
        f"Only {len(ctl)} players have decent benches by this rule, so a few carry the matching: "
        + ", ".join(f"{name} {count}" for name, count in reused.head(4).items()) + f" of the {k * len(tr)} matches."]
    star_view = pd.DataFrame({
        "star": tr.player, "rating": tr.rating.map(f2), "bench time removed": tr.bench_removed.map(pct),
        "court time removed": tr.court_removed.map(pct), "change": tr.change.map(f2),
        "matched controls": tr.matched, "their rating": tr.matched_rating.map(f2),
        "their bench time removed": tr.matched_bench.map(pct), "their change": tr.matched_change.map(f2),
        "extra fall": tr.extra.map(f2)})
    ctl_view = ctl.sort_values("rating", ascending=False)
    ctl_view = pd.DataFrame({
        "control": ctl_view.player, "rating": ctl_view.rating.map(f2), "min%": ctl_view.min_share.map(pct),
        "bench with a bad replacement": ctl_view.bad_bench.map(pct), "removed": ctl_view.removed,
        "their rating": ctl_view.removed_rating.map(f2), "bench time removed": ctl_view.bench_removed.map(pct),
        "court time removed": ctl_view.court_removed.map(pct), "change": ctl_view.change.map(f2),
        "luck alone": ctl_view.luck.map(lambda v: f"{v:.2f}"), "model: own shrinks": ctl_view.model_own.map(f2)})
    literal_view = pd.DataFrame({
        "star": literal.player, "rating": literal.rating.map(f2), "bad replacements removed": literal.bad_change.map(f2),
        "same amount of less-bad data, median": literal.placebo_median.map(f2),
        "5th-95th percentile": [f"{f2(a)} to {f2(b)}" for a, b in zip(literal.placebo_lo, literal.placebo_hi)],
        "draws at or below the real fall": literal.below.map(pct)}) if len(literal) else pd.DataFrame()
    print("\n" + "\n".join(lines))
    print("\n" + star_view.to_string(index=False))
    if len(literal):
        print("\n" + literal_view.to_string(index=False))
    glossary = ("Points per 100 possessions. <b>bench time</b>: his team's possessions in games he played while he sat. "
                "<b>change</b>: his total after the refit minus before. <b>matched controls</b>: the 3 players with decent "
                "benches closest to him in rating, bench time removed and court time removed. <b>extra fall</b>: his "
                "change minus theirs; negative = the belief. <b>model: own shrinks</b>: how far the refit shrinks a "
                "player just because the possessions that measured him are gone.")
    page = ('<div style="font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;color:#1f2933">'
            '<h2 style="font-size:18px;margin:0 0 6px">Bench control: bad replacements against decent ones</h2>'
            + "".join(f'<p style="font-size:14px;margin:0 0 6px">{html.escape(ln)}</p>' for ln in lines)
            + f'<p style="font-size:12px;color:#6b7280">{glossary}</p>'
            + '<h3 style="font-size:15px">Each star against his matched controls</h3>'
            + r77.html_table(star_view, left=("star", "matched controls"))
            + ('<h3 style="font-size:15px">The literal version: stars with enough less-bad bench time</h3>'
               + r77.html_table(literal_view, left=("star",)) if len(literal) else "")
            + '<h3 style="font-size:15px">The control group</h3>' + r77.html_table(ctl_view, left=("control",))
            + '<h3 style="font-size:15px">Checks</h3>'
            + "".join(f'<p style="font-size:12px;margin:0 0 3px">{html.escape(ln)}</p>' for ln in log) + "</div>")
    (out / "bench_control_phone.html").write_text(page, encoding="utf-8")
    print(f"wrote outputs/bench_control_phone.html\ndone in {time.time() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
