"""The playoff DELTA: what a player's postseason says about him that his regular season did not.

    python scripts/63_playoff_delta.py [--mult=2] [--first=1997] [--last=2026] [--top=25] [--season]

Not a playoff rating.  The owner, 2026-09-09: *"let's surface the playoffs as a 'delta' rather than a
rating."*  That is also what the estimator produces: the playoff fit takes the regular-season rating as its
offset (`playoffs.playoff_system`), so its RESIDUAL is exactly the delta and the playoff rating is the sum
of the two.  Publishing the delta says the honest thing -- "the playoffs moved him this far" -- instead of
implying a standalone postseason number that eighty-odd games cannot support.

`--mult` is the penalty on the playoff residual as a multiple of the regular-season one.  **2 is the value
chosen on half the blocks and read on the other** (FINDINGS 38.2): on the confirm half it is worth -0.459
per 100 on held-out playoff games at z -2.16 and -0.508 on the attribution instrument at z -2.27.  Below 1
the delta predicts worse than not adjusting at all, so this is not a knob to open up.

Writes `outputs/playoff_delta.parquet` and `outputs/csv/playoff_delta.csv`: one row per player per window
with his regular-season rating, the delta per side, and the playoff possessions behind it.  A player the
playoffs never saw is not in the table at all rather than sitting at a delta of zero.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eracoef.boxtable import player_names, season_box    # noqa: E402
from eracoef.config import load_config                    # noqa: E402
from eracoef.holdout import Context                       # noqa: E402
from eracoef.playoffs import playoff_system               # noqa: E402
from eracoef.stints import season_names                   # noqa: E402
from eracoef.systems import registry                      # noqa: E402
from eracoef.windows import window_label, window_seasons  # noqa: E402

MIN_POSS = 200.0        # below this a delta is the prior and a rounding error; kept in the table, flagged


def flag(name, default=None):
    hit = [a for a in sys.argv[1:] if a.startswith(f"--{name}=")]
    return hit[-1].split("=", 1)[1] if hit else default


def main():
    cfg = load_config()
    S = registry(cfg)
    ctx = Context.load(cfg)
    base = S[flag("base", cfg.get("ratings_prior", {}).get("board") or "tune501_b7_pasto_pOD")]
    mult = float(flag("mult", 2.0))
    first, last = int(flag("first", cfg["first_season"])), int(flag("last", cfg["last_season"]))
    top = int(flag("top", 25))
    per_season = "--season" in sys.argv
    blocks = ([[s] for s in range(first, last + 1)] if per_season
              else [list(range(w[0], w[1] + 1)) for w in window_seasons(cfg)
                    if w[0] >= first and w[1] <= last])
    pd.set_option("display.width", 200)

    out, t0 = [], time.time()
    for seasons in blocks:
        lab = window_label(seasons) if not per_season else str(seasons[0])
        sysm = playoff_system(base, f"po_{lab}", lam_po=float(base.lam) * mult)
        try:
            po = sysm.fit(seasons, ctx).df
        except Exception as e:                      # a season with no playoff stints built yet
            print(f"  {lab}: skipped ({type(e).__name__}: {e})", flush=True)
            continue
        rs = sysm.prior_from.ratings(seasons, ctx).df
        nm = player_names(season_box(seasons, ["RS"], cfg),
                          pd.concat([season_names(x, "RS", cfg) for x in seasons], ignore_index=True))
        nm = (nm.sort_values("season").drop_duplicates("player_id", keep="last")
              [["player_id", "player_name"]].rename(columns={"player_name": "name"}))
        rs = rs.merge(nm, on="player_id", how="left")
        rs["name"] = rs["name"].fillna(rs["player_id"].astype(str))
        j = (rs[["player_id", "name", "o", "d", "poss"]]
             .rename(columns={"o": "rs_off", "d": "rs_def", "poss": "rs_poss"})
             .merge(po[["player_id", "o", "d", "poss"]]
                    .rename(columns={"o": "po_off", "d": "po_def", "poss": "po_poss"}),
                    on="player_id", how="inner"))
        # `Ratings.d` is the RAW-sign defensive rating -- points the opponent scored, so NEGATIVE is a good
        # defender -- while everything published (08_ratings' `rating_def`, the site, the season board) is
        # flipped so positive is good.  Flip here, once, before anything is added up.  Adding the raw sign
        # instead is what put Trae Young, Curry, Doncic and Lillard at the top of a column that was meant to
        # be a total: it was subtracting each of them their own defence and rewarding the worst defenders.
        for c in ("rs_def", "po_def"):
            j[c] = -j[c]
        # the playoff fit's residual IS the delta: its offset was the regular-season rating
        j["d_off"] = j.po_off - j.rs_off
        j["d_def"] = j.po_def - j.rs_def
        j["d_total"] = j.d_off + j.d_def
        j["rs_total"] = j.rs_off + j.rs_def
        j["window"] = lab
        out.append(j[j.po_poss > 0])
        print(f"  {lab}: {int((j.po_poss > 0).sum())} players with playoff possessions "
              f"({time.time() - t0:.0f}s)", flush=True)
    if not out:
        return 1
    D = pd.concat(out, ignore_index=True)
    D["thin"] = D.po_poss < MIN_POSS

    root = Path(cfg["_root"])
    D.to_parquet(root / "outputs" / "playoff_delta.parquet", index=False)
    (root / "outputs" / "csv").mkdir(parents=True, exist_ok=True)
    D.to_csv(root / "outputs" / "csv" / "playoff_delta.csv", index=False)

    hi = D[~D.thin]
    print(f"\n=== {len(D)} player-windows, {len(hi)} with {MIN_POSS:.0f}+ playoff possessions")
    print(f"    the delta's spread over those: offense {hi.d_off.std():.2f}, defense {hi.d_def.std():.2f}, "
          f"total {hi.d_total.std():.2f}   (the ratings' own spread: {hi.rs_total.std():.2f})")
    cols = ["window", "name", "rs_total", "d_off", "d_def", "d_total", "po_poss"]
    print(f"\n=== the playoffs raised these most (penalty x{mult:g})")
    print(hi.nlargest(top, "d_total")[cols].round(2).to_string(index=False))
    print("\n=== and lowered these most")
    print(hi.nsmallest(top, "d_total")[cols].round(2).to_string(index=False))
    print(f"\nwrote {root / 'outputs' / 'playoff_delta.parquet'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
