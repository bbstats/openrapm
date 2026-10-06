"""The prior shrink (experiment 33b, adopted 2026-10-03): the box-score prior part of every rating times one held-out
multiplier per side, the games' part unchanged.

    python scripts/99_prior_shrink.py --base=season_ratings_product_pre_swap --out=season_ratings_product_priorshrink_pre_swap
                                      [--rule=product|test] [--tag=within] [--fold_seasons=2017-2026|all]
                                      [--check=<existing table to reproduce>]

Why.  Rated from three quarters of a season and scored on the other quarter (scripts/97_within_season.py, the
within-season folds of 2017-2026), the two parts of a rating hold up differently: the games' own adjustment holds up
(held-out coefficient 1.21 on offence) and the box-score prior part, which the ridge stretches about 1.9x on offence,
does not (0.70; 0.64 for players with 2,500+ possessions).  Multiplying the prior part by the held-out coefficient --
about 0.72 on offence and 0.95 on defence -- beat the incumbent on the year-over-year test at z -4.1 (8.648 against
8.666), with the order unchanged, and fits the season's own held-out games better than shrinking the whole rating
(z -3.4, 9 of 10 seasons).  DECISIONS.md, experiment 33b.

How.  For each season, the multipliers are a weighted least squares on the held-out team-games of the within-season
folds: points per 100 as scored, on the share-weighted sums of the ten players' prior parts (offence and defence),
the games' parts and the stand-in ratings held at their own values, the level and home edge free per fold.  The
folds used are:

  --rule=product   every fold season but the rated one (ruling 2: never fitted on the rated season)
  --rule=test      every fold season outside the rated season and its two neighbours, so the year-over-year test,
                   which scores a season's rating on its neighbours' games, stays clean

`--fold_seasons` pins WHICH seasons' folds are read, 2017-2026 by default: the adopted multipliers were fitted when the
`within` folder held only those seasons, and on 2026-10-04 it grew to 1997-2026 (experiment 33c).  Without the pin
every rerun silently fitted on 27-28 fold seasons and shrank harder (offence 0.68-0.69 against 0.70-0.73) -- which is
what experiment 34's first run did.  `all` reads every season in the folder; only the 3/4-season folds (`rate_from`
rest, four folds a deal) are accepted, the size the adopted recipe was fitted at.  `--check=<table>` asserts the output
equals an existing table, so a rerun can prove it reproduces what ships.

Seasons outside the fold seasons borrow the multipliers fitted on all of them (they moved within +-0.015 across
2017-2026).  The table is re-centred per side as the rankings are (possession-weighted mean zero), and the prior and
games' columns are kept consistent (rating = prior part + games' part).  Writes outputs/<out>.parquet in the base
table's schema and outputs/csv/<out>_multipliers.csv.
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
from _cli import check_flags, flag  # noqa: E402

_spec = importlib.util.spec_from_file_location("_calib98", ROOT / "scripts" / "98_calibrator.py")
C98 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(C98)


def parse_seasons(spec: str) -> tuple | None:
    """`2017-2026` -> (2017, 2026); `all` -> None."""
    if spec == "all":
        return None
    first, last = (int(x) for x in spec.split("-"))
    assert first <= last, f"--fold_seasons={spec}"
    return first, last


def load_folds(directory: Path, seasons: tuple | None) -> list:
    """The 3/4-season folds of `directory` whose season is inside `seasons` (both ends included; None = all),
    after asserting their leak checks passed and every deal has its four folds."""
    stems = sorted(p.stem[len("fold_"):] for p in directory.glob("fold_*.json"))
    if seasons is not None:
        stems = [s for s in stems if seasons[0] <= int(s[:4]) <= seasons[1]]
    assert stems, f"no folds in {directory} for seasons {seasons}"
    deals = {}
    for s in stems:
        info = json.loads((directory / f"fold_{s}.json").read_text())
        mode = info["summary"].get("rate_from")
        if mode is None or (isinstance(mode, float) and np.isnan(mode)):
            mode = "rest"                                    # folds dealt before the flag existed are `rest`
        assert mode == "rest", f"fold {s} rates from {mode!r}, not the 3/4-season `rest` size"
        failed = [k for k, v in info["checks"].items() if isinstance(v, bool) and not v]
        assert not failed, f"fold {s}: leak checks failed {failed}"
        key = (int(info["summary"]["season"]), int(info["summary"]["repeat"]))
        deals.setdefault(key, set()).add(int(info["summary"]["fold"]))
    short = {k: v for k, v in deals.items() if v != {0, 1, 2, 3}}
    assert not short, f"deals without exactly four folds: {sorted(short)[:5]}"
    return [C98.Fold(directory, s) for s in stems]


def multipliers_for(seasons, folds: list, rule: str) -> pd.DataFrame:
    """One row per rated season: its prior-part multipliers, fitted on the folds outside the excluded seasons."""
    fold_seasons = sorted({f.season for f in folds})
    cache, rows = {}, []
    for season in sorted(seasons):
        excluded = frozenset({season} if rule == "product" else {season - 1, season, season + 1})
        if excluded not in cache:
            cache[excluded] = fit_prior_multipliers([f for f in folds if f.season not in excluded])
        m_off, m_def = cache[excluded]
        rows.append(dict(season=int(season), prior_off=m_off, prior_def=m_def,
                         fold_seasons=len([s for s in fold_seasons if s not in excluded])))
    return pd.DataFrame(rows)


def fit_prior_multipliers(folds: list) -> tuple:
    """(offence, defence) multipliers on the prior part, the games' part held at its own value."""
    ys, ws, Fs, cols = [], [], [], []
    for f in folds:
        o, d = f.rows["O"], f.rows["D"]
        prior_o, games_o = f.Z["O"] @ o.prior_off.to_numpy(float), f.Z["O"] @ o.u_off.to_numpy(float)
        prior_d, games_d = f.Z["D"] @ (-d.prior_def.to_numpy(float)), f.Z["D"] @ (-d.u_def.to_numpy(float))
        stand_in = f.base - (prior_o + games_o) - (prior_d + games_d)
        ys.append(f.y - stand_in - games_o - games_d)
        cols.append(np.column_stack([prior_o, prior_d]))
        ws.append(f.w)
        Fs.append(f.F)
    level = sp.block_diag([sp.csr_matrix(F) for F in Fs], format="csr")
    M = sp.hstack([level, sp.csr_matrix(np.vstack(cols))], format="csr")
    y, w = np.concatenate(ys), np.concatenate(ws)
    gram = np.asarray((M.T @ M.multiply(w[:, None])).todense())
    rhs = np.asarray(M.T @ (w * y)).ravel()
    coef = np.linalg.lstsq(gram, rhs, rcond=None)[0][level.shape[1]:]
    return float(coef[0]), float(coef[1])


def shrink(table: pd.DataFrame, multipliers: pd.DataFrame) -> pd.DataFrame:
    """The prior part times its season's multipliers, the rating re-centred per side."""
    out = []
    m = multipliers.set_index("season")
    for season, t in table.groupby("season", sort=True):
        t = t.copy()
        t["prior_off"] = m.at[season, "prior_off"] * t.prior_off
        t["prior_def"] = m.at[season, "prior_def"] * t.prior_def
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
    return pd.concat(out, ignore_index=True)[list(table.columns)]


def main() -> None:
    check_flags()
    base_name, out_name = flag("base"), flag("out")
    if not base_name or not out_name:
        raise SystemExit("--base=<steps 1-3 table> --out=<name>")
    rule = flag("rule", "product")
    assert rule in ("product", "test"), "--rule=product|test"
    tag = flag("tag", "within")
    pinned = parse_seasons(flag("fold_seasons", "2017-2026"))
    check_name = flag("check", "")
    directory = ROOT / "outputs" / "within" / tag
    folds = load_folds(directory, pinned)
    fold_seasons = sorted({f.season for f in folds})

    base = pd.read_parquet(ROOT / "outputs" / f"{base_name}.parquet")
    missing = [c for c in ("prior_off", "prior_def", "u_off", "u_def", "poss_off") if c not in base.columns]
    assert not missing, f"{base_name} lacks {missing}"
    multipliers = multipliers_for(base.season.unique(), folds, rule)
    multipliers["fold_tag"], multipliers["fold_first"], multipliers["fold_last"] = tag, fold_seasons[0], fold_seasons[-1]
    table = shrink(base, multipliers)
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
    (ROOT / "outputs" / "csv").mkdir(exist_ok=True)
    multipliers.to_csv(ROOT / "outputs" / "csv" / f"{out_name}_multipliers.csv", index=False)
    moved = (table.set_index(["player_id", "season"]).rating_total
             - base.set_index(["player_id", "season"]).rating_total).abs()
    print(f"{base_name} -> {out.relative_to(ROOT)}: {len(table)} rows, rule {rule}, folds {fold_seasons[0]}-"
          f"{fold_seasons[-1]} ({len(folds)}); prior-part multipliers offence {multipliers.prior_off.min():.3f}-"
          f"{multipliers.prior_off.max():.3f}, defence {multipliers.prior_def.min():.3f}-{multipliers.prior_def.max():.3f}; "
          f"total moved a median {moved.median():.3f} per 100")


if __name__ == "__main__":
    main()
