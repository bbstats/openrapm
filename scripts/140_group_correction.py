"""The group correction (the Robustness pass, step 13, the owner 2026-10-09: "ok go ahead"): the prior shrink refitted
together with one straight line per group of correlated inputs that the season's own held-out games still say the
ratings lean on.

    python scripts/140_group_correction.py --base=season_ratings_<name>_raw --out=season_ratings_<name>_shrunk_raw
                                           [--rule=test|product] [--tag=weightfix_within] [--fold_seasons=2017-2026]
                                           [--from_audit=audit_weightfix] [--check=<table>] [--hold_multipliers=0]

Why.  Three experiments that went after the causes (47, 48, 49) left the leans where they were, and the checkpoint
(DECISIONS.md, "The checkpoint") found the largest of them starting in the box-score prior: big men underrated on
defence, steals overrated, and on offence big men overrated, three-point volume underrated, turnovers overrated.  The
plan's last resort is a small correction fitted on the yardstick that measures them -- other seasons' own held-out
games -- refitted together with the prior shrink, which is fitted on the same games.

What it fits.  99's weighted least squares on the held-out team-games of the within-season folds (points per 100 as
scored, the level and home edge free per fold, the games' parts and the stand-in ratings held at their own values, the
two prior parts each with a free multiplier), plus one column per eligible axis and side: the share-weighted sum of the
ten players' axis scores, signed so a positive line raises the rating.  An axis score is the audit's (138
`axis_scores`): a within-fold possession-weighted percentile and its normal score; a group's axis is the frozen first
principal component of its members' normal scores, ranked again (params/audit_groups.json); centred by possessions
within the fold and side.  The multipliers and the lines are fitted together, per rated season, on the folds outside
it (`--rule=test`: outside it and its two neighbours, as 99 ships; `product`: outside it alone, what the audit reads).
Applied as 99's shrink is -- the prior part times its multiplier -- plus each line times the player's axis score from
his season's own inputs, folded into the prior part, every rating re-centred per season and side.

Which axes.  The audit's partition at the cut of 0.5 (each group whose cuts include 0.5; every other input alone) that
COUNTS on thirty seasons in `--from_audit` (the checkpoint's `audit_weightfix`), and never one holding a shooting
percentage, shot quality, age or career, playing time, score state or team context (the plan's exclusions).  Every
other axis defaults to zero.  `--from_audit=none` fits no line, and must reproduce 99 exactly (`--check`).

Writes outputs/<out>.parquet in the base table's schema, outputs/csv/<out>_coefficients.csv (the rule's multipliers and
lines per rated season), outputs/csv/<out>_coefficients_product.csv (the same fitted outside the rated season alone,
for `138 --correction=<out>`), and outputs/<out>_correction.parquet (each player's correction per side).
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eracoef import singleyear as sy  # noqa: E402
from eracoef.config import load_config  # noqa: E402
from eracoef.seasons import drop_untrainable  # noqa: E402
from _cli import check_flags, flag  # noqa: E402


def _borrow(name: str, file: str):
    """Another script's functions, loaded the way 99 loads 98's: one definition, never a copy."""
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


C99 = _borrow("_shrink99", "99_prior_shrink.py")      # load_folds, parse_seasons, shrink
A138 = _borrow("_audit138", "138_heldout_audit.py")   # build_tests, axis_scores, normal_score, centre_columns, CLASS_OF

MAIN_CUT = 0.5
# never corrected (the plan, step 13): shooting percentages and shot quality, age and career, and what 138 measures only
NEVER = {"shooting percentage": ["efg", "ts", "fg3p", "fg2p", "ftp"],
         "shot quality": ["q2", "q3", "m2", "m3", "xps", "mpts"]}
NEVER_CLASSES = {"age and career", "playing time", "score state", "team context"}


def partition(tests: list) -> list:
    """The axes at the main cut: every group whose cuts include it, and every input that is in none of them."""
    groups = [t for t in tests if t["kind"] == "group" and MAIN_CUT in t["cuts"]]
    grouped = {m for g in groups for m in g["members"]}
    return [t for t in tests if t["kind"] == "input" and t["code"] not in grouped] + groups


def eligible(t: dict) -> bool:
    members = [t["code"]] if t["kind"] == "input" else t["members"]
    never = {c for v in NEVER.values() for c in v}
    return not any(m in never or A138.CLASS_OF[m] in NEVER_CLASSES for m in members)


def chosen_axes(audit: str) -> list:
    """(test, side) for every eligible axis of the main-cut partition that counts in the audit's summary."""
    if audit == "none":
        return []
    tests = A138.build_tests(json.loads(A138.GROUPS_FILE.read_text()))
    summary = pd.read_csv(ROOT / "outputs" / "csv" / f"{audit}_summary.csv")
    counts = set(zip(summary[summary.counts].test, summary[summary.counts].side))
    return [(t, side) for side in A138.SIDES for t in partition(tests) if eligible(t) and (t["id"], side) in counts]


def axis_columns(rows_of: dict, weight_of: dict, axes: list) -> dict:
    """Per side, (players x axes of that side) centred axis scores; positive = high on the axis."""
    out = {}
    for side in A138.SIDES:
        tests = [t for t, s in axes if s == side]
        if not tests:
            out[side] = np.zeros((len(rows_of[side]), 0))
            continue
        z = A138.normal_score(A138.axis_scores(rows_of[side], weight_of[side], tests))
        out[side] = A138.centre_columns(z, weight_of[side])
    return out


def fit_together(folds: list, axes: list, fixed: tuple | None = None) -> tuple:
    """(offence multiplier, defence multiplier, lines in the order of `axes`) from the folds' held-out team-games.
    `fixed` = (offence, defence) multipliers held at those values: only the lines are fitted (the attribution arm)."""
    ys, ws, Fs, cols = [], [], [], []
    for f in folds:
        o, d = f.rows["O"], f.rows["D"]
        prior_o, games_o = f.Z["O"] @ o.prior_off.to_numpy(float), f.Z["O"] @ o.u_off.to_numpy(float)
        prior_d, games_d = f.Z["D"] @ (-d.prior_def.to_numpy(float)), f.Z["D"] @ (-d.u_def.to_numpy(float))
        stand_in = f.base - (prior_o + games_o) - (prior_d + games_d)
        z = axis_columns(f.rows, {s: np.maximum(A138.poss_of(f.rows[s], s), 1e-9) for s in A138.SIDES}, axes)
        lines = [A138.SIGN[s] * (f.Z[s] @ z[s]) for s in A138.SIDES if z[s].shape[1]]
        if fixed is None:
            ys.append(f.y - stand_in - games_o - games_d)
            cols.append(np.column_stack([prior_o, prior_d, *lines]))
        else:
            ys.append(f.y - stand_in - games_o - games_d - fixed[0] * prior_o - fixed[1] * prior_d)
            cols.append(np.column_stack(lines))
        ws.append(f.w)
        Fs.append(f.F)
    level = sp.block_diag([sp.csr_matrix(F) for F in Fs], format="csr")
    M = sp.hstack([level, sp.csr_matrix(np.vstack(cols))], format="csr")
    y, w = np.concatenate(ys), np.concatenate(ws)
    gram = np.asarray((M.T @ M.multiply(w[:, None])).todense())
    rhs = np.asarray(M.T @ (w * y)).ravel()
    full = np.linalg.lstsq(gram, rhs, rcond=None)[0]
    coef = full[level.shape[1]:]
    # a naive standard error per line (team-games treated as independent, which they are not: a player's misfit
    # carries across his games, so it understates; the audit's career null is the honest one)
    resid = y - M @ full
    cov = np.linalg.pinv(gram) * float(np.sum(w * resid ** 2) / max(len(y) - gram.shape[0], 1))
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))[level.shape[1]:]
    if fixed is not None:
        return float(fixed[0]), float(fixed[1]), coef, se
    return float(coef[0]), float(coef[1]), coef[2:], se[2:]


def coefficients_for(seasons, folds: list, axes: list, rule: str, hold: bool = False) -> pd.DataFrame:
    """One row per rated season: the multipliers and lines, fitted on the folds outside the excluded seasons.  `hold`
    keeps 99's own multipliers (fitted on the same folds without the lines) and fits the lines on what they leave."""
    fold_seasons = sorted({f.season for f in folds})
    cache, rows = {}, []
    names = [f"line_{s}_{t['id']}" for t, s in axes]
    for season in sorted(seasons):
        excluded = frozenset({season} if rule == "product" else {season - 1, season, season + 1})
        if excluded not in cache:
            use = [f for f in folds if f.season not in excluded]
            cache[excluded] = fit_together(use, axes, C99.fit_prior_multipliers(use) if hold else None)
        m_off, m_def, lines, se = cache[excluded]
        row = dict(season=int(season), prior_off=m_off, prior_def=m_def,
                   fold_seasons=len([s for s in fold_seasons if s not in excluded]))
        row.update(dict(zip(names, map(float, lines))))
        row.update({f"se_{n[5:]}": float(e) for n, e in zip(names, se)})
        rows.append(row)
    return pd.DataFrame(rows)


def season_rows(panel: pd.DataFrame, table: pd.DataFrame, season: int) -> tuple:
    """The season's rated players' inputs per side (the panel's, derived), aligned to the table, and their weights."""
    t = table[table.season == season]
    rows, weights = {}, {}
    for side, poss in (("O", "poss_off"), ("D", "poss_def")):
        p = panel[(panel.season == season) & (panel.side == side)]
        frame = sy.season_frame(p.copy(), list(sy.PRIOR_FEATURES)).set_index("player_id")
        rows[side] = frame.reindex(t.player_id.to_numpy()).reset_index()
        weights[side] = np.maximum(t[poss].to_numpy(float), 1e-9)
    return rows, weights


def correct(table: pd.DataFrame, panel: pd.DataFrame, coefs: pd.DataFrame, axes: list) -> tuple:
    """99's shrink with the fitted multipliers, plus each line times the player's axis score, in the prior part."""
    m = coefs.set_index("season")
    shrunk = C99.shrink(table, coefs[["season", "prior_off", "prior_def"]])
    if not axes:
        return shrunk, pd.DataFrame(columns=["player_id", "season", "group_off", "group_def"])
    out, parts = [], []
    for season, t in shrunk.groupby("season", sort=True):
        t = t.copy()
        rows, weights = season_rows(panel, t, int(season))
        missing = {s: int(rows[s].isna().all(axis=1).sum()) for s in A138.SIDES}
        z = axis_columns({s: rows[s].fillna(rows[s].median(numeric_only=True)) for s in A138.SIDES}, weights, axes)
        add = {}
        for side in A138.SIDES:
            names = [f"line_{side}_{t_['id']}" for t_, s in axes if s == side]
            beta = np.array([m.at[season, n] for n in names], dtype=float)
            add[side] = z[side] @ beta if len(beta) else np.zeros(len(t))
        t["prior_off"] = t.prior_off + add["O"]
        t["prior_def"] = t.prior_def + add["D"]
        weight = t.poss_off.to_numpy(float)
        for side in ("off", "def"):
            level = np.average(t[f"prior_{side}"] + t[f"u_{side}"], weights=weight)
            t[f"prior_{side}"] -= level
            t[f"rating_{side}"] = t[f"prior_{side}"] + t[f"u_{side}"]
        t["rating_total"] = t.rating_off + t.rating_def
        t["prior_total"], t["u_total"] = t.prior_off + t.prior_def, t.u_off + t.u_def
        t["offense"], t["defense"] = t.rating_off, -t.rating_def
        t["prior_offense"], t["prior_defense"] = t.prior_off, -t.prior_def
        out.append(t)
        parts.append(pd.DataFrame({"player_id": t.player_id.to_numpy(), "season": int(season),
                                   "group_off": add["O"], "group_def": add["D"]}))
        if any(missing.values()):
            print(f"  {season}: players with no panel row (axis score at the median): {missing}", flush=True)
    return pd.concat(out, ignore_index=True)[list(table.columns)], pd.concat(parts, ignore_index=True)


def main() -> None:
    check_flags()
    base_name, out_name = flag("base"), flag("out")
    if not base_name or not out_name:
        raise SystemExit("--base=<steps 1-3 table> --out=<name>")
    rule = flag("rule", "test")
    assert rule in ("product", "test"), "--rule=product|test"
    tag = flag("tag", "weightfix_within")
    pinned = C99.parse_seasons(flag("fold_seasons", "2017-2026"))
    audit = flag("from_audit", "audit_weightfix")
    check_name = flag("check", "")
    # --hold_multipliers=1 (the attribution arm): 99's multipliers as they ship, the lines fitted on what they leave
    hold = flag("hold_multipliers", "0") not in ("0", "no", "false")
    cfg = load_config(ROOT / "config.yaml")
    folds = C99.load_folds(ROOT / "outputs" / "within" / tag, pinned)
    axes = chosen_axes(audit)
    print(f"{len(folds)} folds from {tag} ({sorted({f.season for f in folds})[0]}-{sorted({f.season for f in folds})[-1]}); "
          f"{len(axes)} lines: " + "; ".join(f"{'offence' if s == 'O' else 'defence'} {t['name']}" for t, s in axes),
          flush=True)

    base = pd.read_parquet(ROOT / "outputs" / f"{base_name}.parquet")
    missing = [c for c in ("prior_off", "prior_def", "u_off", "u_def", "poss_off", "poss_def") if c not in base.columns]
    assert not missing, f"{base_name} lacks {missing}"
    # gated like every reader that fits from the panel: a season still being played is never rated here
    panel, _ = drop_untrainable(pd.read_parquet(ROOT / "outputs" / "role_panel_season.parquet"), cfg,
                                what="the season panel")
    coefs = coefficients_for(base.season.unique(), folds, axes, rule, hold)
    product = coefficients_for(sorted({f.season for f in folds} | set(base.season.unique())), folds, axes, "product",
                               hold)
    for frame in (coefs, product):
        frame["fold_tag"], frame["from_audit"] = tag, audit
        frame["axes"] = json.dumps([[t["id"], s] for t, s in axes])
    table, correction = correct(base, panel, coefs, axes)
    if check_name:
        ref = pd.read_parquet(ROOT / "outputs" / f"{check_name}.parquet").set_index(["player_id", "season"])
        got = table.set_index(["player_id", "season"])
        assert got.index.sort_values().equals(ref.index.sort_values()), f"rows differ from {check_name}"
        cols = ["rating_off", "rating_def", "prior_off", "prior_def", "u_off", "u_def"]
        gap = float((got[cols] - ref.loc[got.index, cols]).abs().to_numpy().max())
        print(f"check against {check_name}: largest difference {gap:.2e}")
        assert gap <= 1e-10, f"does not reproduce {check_name}: largest difference {gap:.3e}"
    out = ROOT / "outputs" / f"{out_name}.parquet"
    table.to_parquet(out, index=False)
    correction.to_parquet(ROOT / "outputs" / f"{out_name}_correction.parquet", index=False)
    (ROOT / "outputs" / "csv").mkdir(exist_ok=True)
    coefs.to_csv(ROOT / "outputs" / "csv" / f"{out_name}_coefficients.csv", index=False)
    product.to_csv(ROOT / "outputs" / "csv" / f"{out_name}_coefficients_product.csv", index=False)
    pooled = fit_together(folds, axes, C99.fit_prior_multipliers(folds) if hold else None)
    print(f"multipliers on every fold together: offence {pooled[0]:.4f}, defence {pooled[1]:.4f}")
    for (t, s), b, e in zip(axes, pooled[2], pooled[3]):
        col = coefs["line_" + s + "_" + t["id"]]
        print(f"  {'offence' if s == 'O' else 'defence'} {t['name']}: {b:+.3f} per standard deviation of the axis "
              f"(naive se {e:.3f}); the rule's lines by season run {col.min():+.3f} to {col.max():+.3f}")
    moved = (table.set_index(["player_id", "season"]).rating_total
             - base.set_index(["player_id", "season"]).rating_total).abs()
    if len(correction):
        print(f"each player's correction: sd offence {correction.group_off.std():.3f}, defence "
              f"{correction.group_def.std():.3f}; largest {correction[['group_off', 'group_def']].abs().max().max():.2f}")
    print(f"{base_name} -> {out.relative_to(ROOT)}: {len(table)} rows, rule {rule}; multipliers offence "
          f"{coefs.prior_off.min():.3f}-{coefs.prior_off.max():.3f}, defence {coefs.prior_def.min():.3f}-"
          f"{coefs.prior_def.max():.3f}; total moved a median {moved.median():.3f} per 100")


if __name__ == "__main__":
    main()
