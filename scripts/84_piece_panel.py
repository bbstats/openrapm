"""The pieces of every season's vanilla RAPM, kept per side: the panel for the owner's idea that a season's pieces
predict a player's neighbouring seasons better than their sum does (scripts/85_piece_rating.py tests it).

    python scripts/84_piece_panel.py [--penalty=3000] [--truth_penalty=100] [--first=1997] [--last=2026]
                                     [--out=outputs/piece_panel.parquet]

One row per player-season.  The fit and the pieces are the Decomposition page's (scripts/83_decompose_site.py,
loaded as a module and not edited): plain ridge on raw points, regular season and playoffs, one season at a time,
penalty 3,000 per side.  Since 2026-09-28 the fit and the per-side split are src/eracoef/pieces.py's -- the same
code the season panel's luck-adjusted pieces and the single-year rankings' cross-fit run -- and this panel came out
bit-identical through it.  The page shows only the net.  Here every piece is split into its offensive and defensive
halves, so a model can weigh the two sides separately, and the year-over-year test -- which needs an offensive
and a defensive rating -- can score what it builds.

**By player, per side.**  Offence, over his offensive possessions:

    on_rtg_off     his team's points per 100 while he is on the court, minus the season's league average
    teammates_off  minus the offensive RAPM of the four teammates on the court with him, added up
    opponents_off  the defensive RAPM of the five defenders he faced, added up
    context_off    minus what home court, playoffs, garbage time and the score margin predict for those
                   possessions, measured from the league average.  It also carries the published zero point's
                   constant, 5 x the two sides' shifts, which the defence carries with the opposite sign, so it
                   cancels in the net exactly as the page says.
    ridge_off      the ridge's pull toward zero: minus penalty x his offensive coefficient / his possessions

Defence mirrors it over his defensive possessions, positive = good.  Each side's five add up to that side's RAPM
exactly, and the offensive plus the defensive half of each piece is the page's piece exactly.

**By possession, per side.**  83's row weights h = W X A^-1 c with the contrast c split into its offensive and its
defensive half, summed over the same four groups of rows: his possessions (on_signal), his team's possessions
without him in games he played (off_adj_gp), his teams' possessions in games he missed (off_adj_dnp), and every
other possession (team_sos).  The context columns are unpenalised, so h is orthogonal to them for either half and
each side's four add up to that side's RAPM exactly.

Also per player-season: the actual off-court rating per side (beside the pieces, not one of them), possessions,
his team's possessions without him and in games he missed, his main team (most minutes -- the rule the
year-over-year test's movers split uses), and a second vanilla RAPM at a light penalty (100).  That second RAPM is
only a second truth to score against: a truth that is itself shrunk hard rewards a model that shrinks the same
way (DECISIONS.md, "The measurement traps").

The run stops on a failed check.
"""
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _cli import check_flags, flag  # noqa: E402
from eracoef.boxtable import season_box  # noqa: E402
from eracoef.config import load_config  # noqa: E402
from eracoef.pieces import BY_PLAYER, BY_POSSESSION, decompose_sides, fit as piece_fit  # noqa: E402
from eracoef.windows import build_window  # noqa: E402

_spec = importlib.util.spec_from_file_location("_borrowed_83", ROOT / "scripts" / "83_decompose_site.py")
d83 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(d83)
check = d83.check

SIDES = ("off", "def")
# each piece here -> the page's net piece it is the two halves of (83's column names)
NET_83 = dict(on_rtg="margin", teammates="teammates", opponents="opponents", context="context", ridge="ridge",
              on_signal="his_minutes", off_adj_gp="team_without_him", off_adj_dnp="games_missed",
              team_sos="everything_else")
TOL = 1e-9


def main_teams(season: int, cfg) -> pd.Series:
    """player_id -> the team he played the most minutes for (box scores; holdout.Context.main_team's rule)."""
    b = season_box([season], ["RS", "PO"], cfg).groupby(["player_id", "team_id"], as_index=False)["minutes"].sum()
    b = b.sort_values("minutes").drop_duplicates("player_id", keep="last")
    return pd.Series(b.team_id.to_numpy(np.int64), index=b.player_id.to_numpy(np.int64))


def main() -> None:
    check_flags()
    lam, truth_lam = float(flag("penalty", 3000)), float(flag("truth_penalty", 100))
    cfg = load_config()
    first, last = int(flag("first", cfg["first_season"])), int(flag("last", cfg["last_season"]))
    out_path = ROOT / flag("out", "outputs/piece_panel.parquet")
    names = d83.name_table()
    log, frames, started = [], [], time.time()
    worst = dict(by_player=0.0, by_possession=0.0, ridge=0.0, weights=0.0, vs_page=0.0, off_rtg=0.0, rows=0,
                 no_team=0, no_truth=0)
    for season in range(first, last + 1):
        # one season's raw-points design and the box scores' teams, as 83's span_fit builds them; the fit and the
        # decomposition are src/eracoef/pieces.py's, which the season panel and the rankings' cross-fit share
        wd, teams = build_window([season], cfg, target="pts"), d83.box_teams(season)
        fit = piece_fit(wd, teams, lam)
        tab, miss = decompose_sides(fit)
        for k in miss:
            worst[k] = max(worst[k], miss[k])
        # the page's own net pieces on the same fit: the two halves must add up to them
        page, _ = d83.decompose(fit)
        worst["rows"] += abs(len(page) - len(tab)) + int((~tab.i.isin(page.i)).sum())
        page = page.set_index("i").reindex(tab.i.to_numpy())
        for k, net in NET_83.items():
            worst["vs_page"] = max(worst["vs_page"],
                                   float(np.max(np.abs(tab[f"{k}_off"].to_numpy() + tab[f"{k}_def"].to_numpy()
                                                       - page[net].to_numpy()))))
        worst["vs_page"] = max(worst["vs_page"], float(np.max(np.abs(tab.rapm_off.to_numpy() + tab.rapm_def.to_numpy()
                                                                    - page.rapm.to_numpy()))))
        both = page.off_rtg.notna().to_numpy()
        worst["off_rtg"] = max(worst["off_rtg"], float(np.max(np.abs(
            (tab.off_rtg_off + tab.off_rtg_def).to_numpy()[both] - page.off_rtg.to_numpy()[both]), initial=0.0)))
        ids = fit["player_ids"][tab.i.to_numpy()].astype(np.int64)
        tab.insert(0, "season", season)
        tab.insert(0, "player_id", ids)
        tab.insert(2, "player_name", d83.span_names(names, ids, [season]))
        teams = main_teams(season, cfg)
        tab.insert(3, "team_id", teams.reindex(ids).fillna(-1).to_numpy(np.int64))
        worst["no_team"] += int((tab.team_id < 0).sum())
        # the second truth: the same fit at a light penalty, by player id
        light = piece_fit(wd, teams, truth_lam)
        light_ids = pd.Index(light["player_ids"].astype(np.int64))
        pos = light_ids.get_indexer(ids)
        worst["no_truth"] += int((pos < 0).sum())
        tab["rapm100_off"] = np.where(pos >= 0, light["off"][pos], np.nan)
        tab["rapm100_def"] = np.where(pos >= 0, light["dfn"][pos], np.nan)
        frames.append(tab.drop(columns="i"))
        top = tab.assign(net=tab.rapm_off + tab.rapm_def).sort_values("net").iloc[-1]
        print(f"  {season}: {len(tab)} players, top {top.player_name} {top.net:+.2f} "
              f"({time.time() - started:.0f}s)", flush=True)
    panel = pd.concat(frames, ignore_index=True)
    check(worst["by_player"] < TOL, f"each side's five by-player pieces add up to that side's RAPM (max miss "
                                    f"{worst['by_player']:.1e})", log)
    check(worst["by_possession"] < TOL, f"each side's four possession groups add up to that side's RAPM (max miss "
                                        f"{worst['by_possession']:.1e})", log)
    check(worst["vs_page"] < TOL, f"offence + defence of every piece equals the Decomposition page's net piece "
                                  f"(max miss {worst['vs_page']:.1e})", log)
    check(worst["off_rtg"] < TOL, f"the two halves of the off-court rating add up to the page's (max miss "
                                  f"{worst['off_rtg']:.1e})", log)
    check(worst["rows"] == 0, f"the same players as the page, season by season ({worst['rows']} differ)", log)
    check(worst["ridge"] < TOL, f"'ridge' equals penalty x coefficient / possessions on each side (max miss "
                                f"{worst['ridge']:.1e})", log)
    check(worst["weights"] < TOL, f"the fit's weights are his possessions (max relative miss "
                                  f"{worst['weights']:.1e})", log)
    check(worst["no_truth"] == 0, f"every player has a light-penalty RAPM ({worst['no_truth']} without)", log)
    check(panel.player_name.notna().all(), "every player has a name", log)
    check(not panel.duplicated(["player_id", "season"]).any(), "one row per player-season", log)
    print(f"  note: {worst['no_team']} player-seasons have no main team in the box scores (team_id -1)")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(out_path, index=False)
    print(f"wrote {out_path.relative_to(ROOT)}: {len(panel):,} player-seasons, {panel.season.nunique()} seasons, "
          f"{len(panel.columns)} columns ({time.time() - started:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
