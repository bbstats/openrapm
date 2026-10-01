"""Write each player-season's RAPM pieces and his playoff share into the season panel, beside the on-court columns.

    python scripts/86_context_panel.py [--panel=outputs/role_panel_season.parquet] [--penalty=3000]

The owner's experiment 26 (2026-09-28): give the single-year prior the Decomposition page's pieces and some game
difficulty, beside a soft same-team measure it gets elsewhere (`singleyear.chunk_rows`).  What this writes, per
player-season, on both side rows (like `onc_*`):

  pc_<piece>_o   each of `pieces.PIECES` -- on_rtg, teammates, opponents, context, ridge (by player) and
  pc_<piece>_d   on_signal, off_adj_gp, off_adj_dnp, team_sos (by possession) -- plus pc_off_rtg, the actual
                 off-court rating.  `_o` comes from a vanilla RAPM on the OFFENSIVE design and `_d` from one on
                 the DEFENSIVE design, the two designs `onc_*` is built from (`xpts_ft`, `x3def`), so the pieces
                 measure what the prior's labels and its on-court columns measure.  Penalty 3,000 per side, the
                 page's own.  Positive = good on both sides (`onc_d` is the other way round).  Each is padded toward
                 0 over the possessions behind it with the same constant (`pieces.season_pieces`).
  po_share       the share of his possessions that came in the playoffs: `data/cache/roles.parquet` (regular
                 season) against `roles_RSPO.parquet` (both), on `poss_on`.  A share, like the closeness columns,
                 and like them not padded.

Game difficulty is also in `pc_opponents_*`, `pc_context_*`, `pc_team_sos_*` and the panel's closeness columns.

These are the rated season's own games, like `onc_*`: ruling 1 allows it, and the single-year rankings' cross-fit
rebuilds them from each game fold's training games (scripts/62_single_year_board.py) so the free prior scale is not
priced on the games it scores.

Checks, and the run stops on a failure: every per-side identity of the decomposition under 1e-9 on these designs;
the two designs have the same rows; the fold path (a subset of every row, read back off the matrix) reproduces the
whole-season pieces to 1e-12; the on-court columns recomputed from the same designs match the panel's to 1e-6, so
the pieces and `onc_*` come from the same games; under 0.1% of rows have a side the box scores cannot name.

The panel is backed up to `<panel>_pre_context.parquet` before it is overwritten.
"""
import importlib.util
import os
import shutil
import sys
import time
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
os.environ.setdefault("NUMBA_NUM_THREADS", os.environ.get("OPENRAPM_NUMBA_THREADS", "4"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _cli import check_flags, flag  # noqa: E402
from eracoef import pieces  # noqa: E402
from eracoef import singleyear as sy  # noqa: E402
from eracoef.config import load_config  # noqa: E402
from eracoef.holdout import Context  # noqa: E402
from eracoef.investigate import oncourt_rates  # noqa: E402
from eracoef.xshoot import DEFENSE_TARGETS  # noqa: E402

_spec = importlib.util.spec_from_file_location("_borrowed_83", ROOT / "scripts" / "83_decompose_site.py")
d83 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(d83)
check = d83.check

TOL = 1e-9
PO_SHARE = "po_share"


def playoff_share() -> pd.DataFrame:
    """player_id, season, po_share: 1 - regular-season possessions / all possessions, summed over his teams."""
    def total(name):
        r = pd.read_parquet(ROOT / "data" / "cache" / name, columns=["player_id", "season", "poss_on"])
        return r[r.poss_on > 0].groupby(["player_id", "season"]).poss_on.sum()
    both, regular = total("roles_RSPO.parquet"), total("roles.parquet")
    share = (1.0 - regular.reindex(both.index).fillna(0.0) / both).clip(0.0, 1.0)
    return share.rename(PO_SHARE).reset_index()


def same_rows(a, b) -> bool:
    """The two designs describe the same possessions in the same order: the fold masks apply to both.

    The defensive design is the points design with its response replaced, which drops the stored lineups, so
    they are read the way a fold reads them (`pieces.blocks`)."""
    if len(a.y) != len(b.y):
        return False
    _, lo_a, ld_a = pieces.blocks(a)
    _, lo_b, ld_b = pieces.blocks(b)
    return (np.array_equal(a.rows.game_idx.to_numpy(), b.rows.game_idx.to_numpy())
            and np.array_equal(a.rows.is_home_off.to_numpy(), b.rows.is_home_off.to_numpy())
            and np.array_equal(a.rows.poss.to_numpy(), b.rows.poss.to_numpy())
            and np.array_equal(lo_a, lo_b) and np.array_equal(ld_a, ld_b))


def main() -> None:
    check_flags()
    path = ROOT / flag("panel", "outputs/role_panel_season.parquet")
    lam = float(flag("penalty", pieces.PENALTY))
    cfg = load_config(ROOT / "config.yaml")
    ctx = Context.load(cfg)
    panel = pd.read_parquet(path)
    seasons = sorted(panel.season.unique())
    log, frames, started = [], [], time.time()
    worst = dict(identity=0.0, fold=0.0, onc=0.0, unplaced=0.0, rows_differ=0)
    for season in seasons:
        wd_o = ctx.design([season], "xpts_ft")
        wd_d = ctx.design([season], DEFENSE_TARGETS["x3def"])
        worst["rows_differ"] += int(not same_rows(wd_o, wd_d))
        teams = d83.box_teams(season)
        frame, misses = pieces.season_pieces(wd_o, wd_d, teams, lam)
        for tag in ("o", "d"):
            worst["identity"] = max(worst["identity"], *(misses[tag][k] for k in
                                                         ("by_player", "by_possession", "ridge", "weights")))
            worst["unplaced"] = max(worst["unplaced"], misses[tag]["unplaced"])
        # the cross-fit's path: a subset of every row has no stored parts and is read back off the matrix
        every = np.ones(len(wd_o.y), dtype=bool)
        again, _ = pieces.season_pieces(wd_o.subset(every), wd_d.subset(every), teams, lam)
        gap = again.set_index("player_id").reindex(frame.player_id).to_numpy() - frame.set_index("player_id").to_numpy()
        worst["fold"] = max(worst["fold"], float(np.nanmax(np.abs(gap))) if gap.size else 0.0)
        # the on-court columns from the same two designs must be the panel's, or the pieces are other games
        onc = oncourt_rates(wd_o, wd_d)
        onc = pd.DataFrame({"player_id": wd_o.spec.ps_table.player_id.to_numpy(),
                            **{c: onc[c].to_numpy(float) for c in sy.ONC}})
        mine = panel[(panel.season == season) & (panel.side == "O")][["player_id"] + sy.ONC]
        both = mine.merge(onc, on="player_id", suffixes=("", "_re"))
        worst["onc"] = max(worst["onc"], max(float((both[c] - both[f"{c}_re"]).abs().max()) for c in sy.ONC))
        frames.append(frame.assign(season=season))
        top = frame.assign(net=frame.pc_on_signal_o + frame.pc_on_signal_d).sort_values("net").iloc[-1]
        print(f"  {season}: {len(frame)} players, largest on-court signal {int(top.player_id)} {top.net:+.2f}, "
              f"worst identity {max(max(misses[t][k] for k in ('by_player', 'by_possession')) for t in 'od'):.1e} "
              f"({time.time() - started:.0f}s)", flush=True)
    new = pd.concat(frames, ignore_index=True)
    check(worst["rows_differ"] == 0, f"the offensive and defensive designs have the same rows in every season "
                                     f"({worst['rows_differ']} differ)", log)
    check(worst["identity"] < TOL, f"every per-side identity holds on the luck-adjusted designs (max miss "
                                   f"{worst['identity']:.1e})", log)
    check(worst["fold"] < 1e-12, f"a fold of every row reproduces the whole-season pieces (max gap "
                                 f"{worst['fold']:.1e})", log)
    check(worst["onc"] < 1e-6, f"the on-court columns from the same designs match the panel's (max gap "
                               f"{worst['onc']:.1e})", log)
    check(worst["unplaced"] < d83.UNPLACED_TOL, f"under {d83.UNPLACED_TOL:.1%} of rows have a side the box scores "
                                                f"cannot name (worst season {worst['unplaced']:.4%})", log)
    check(not new.duplicated(["player_id", "season"]).any(), "one row per player-season", log)

    share = playoff_share()
    backup = path.with_name(path.stem + "_pre_context.parquet")
    if not backup.exists():
        shutil.copy(path, backup)
    added = pieces.COLUMNS + [PO_SHARE]
    panel = panel.drop(columns=[c for c in added if c in panel.columns])
    before = len(panel)
    panel = (panel.merge(new, on=["player_id", "season"], how="left")
             .merge(share, on=["player_id", "season"], how="left"))
    check(len(panel) == before, "the merge added no rows", log)
    covered = panel[pieces.COLUMNS[0]].notna()
    print(f"  {int((~covered).sum())} of {len(panel)} panel rows have no piece (no rows on one side); set to 0")
    print(f"  {int(panel[PO_SHARE].isna().sum())} have no playoff share (no possessions in the roles table); set to 0")
    for c in added:
        panel[c] = panel[c].fillna(0.0)
    panel.to_parquet(path, index=False)
    wide = panel[(panel.side == "O") & (panel.poss >= 1000)]
    print(f"wrote {path.relative_to(ROOT)}: {len(panel)} rows, {len(added)} columns added")
    print("  among 1,000+ possession player-seasons (offence rows), sd of each padded piece:")
    for tag in ("o", "d"):
        print("   " + "  ".join(f"{c.replace('pc_', '')} {wide[c].std():.2f}"
                                for c in pieces.COLUMNS if c.endswith("_" + tag)))
    print(f"  corr(pc_on_rtg_o, onc_o) {np.corrcoef(wide.pc_on_rtg_o, wide.onc_o)[0, 1]:.3f}, "
          f"corr(pc_on_rtg_d, onc_d) {np.corrcoef(wide.pc_on_rtg_d, wide.onc_d)[0, 1]:.3f} (onc_d is negative-good)")
    print(f"  po_share: mean {wide[PO_SHARE].mean():.3f}, share with any playoff possessions "
          f"{(wide[PO_SHARE] > 0).mean():.1%}, max {wide[PO_SHARE].max():.3f}  ({time.time() - started:.0f}s)")


if __name__ == "__main__":
    main()
