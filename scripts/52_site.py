"""OpenRAPM: the one-page site.  Exports the board to docs/data/ratings.json for docs/index.html.

Reads outputs/player_ratings.parquet (scripts/08_ratings.py).  One row per player per three-season
window: window, name, offense, defense, total (positive = good, points per 100 possessions), and the
regular-season possessions the rating rests on.  docs/index.html is static and reads this file.

If outputs/season_ratings.parquet exists (scripts/60_season_board.py) its rows go in beside them under
`seasons`, keyed by season instead of window, and the page offers both views.  A season's rating is fit
on that season and the two before it, so the latest one is the season in progress.

If outputs/playoff_delta.parquet exists (scripts/63_playoff_delta.py) its rows go in under `playoffs` as a
third view: the DELTA a postseason put on a player, not a playoff rating.
usage: python scripts/52_site.py
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eracoef.config import load_config  # noqa: E402

cfg = load_config()
root = Path(cfg["_root"])
rat = pd.read_parquet(root / "outputs" / "player_ratings.parquet")
cols = {"window": "w", "player_name": "n", "rating_off": "o", "rating_def": "d", "rating_total": "t", "poss_off": "p"}
d = rat[list(cols)].rename(columns=cols).copy()
d["n"] = d["n"].fillna("").astype(str)
d = d[d.p > 0].sort_values(["w", "t"], ascending=[True, False])
rows = [dict(w=r.w, n=r.n, o=round(float(r.o), 2), d=round(float(r.d), 2), t=round(float(r.t), 2), p=int(r.p))
        for r in d.itertuples(index=False)]
out = root / "docs" / "data"
out.mkdir(parents=True, exist_ok=True)
meta = dict(windows=sorted(d.w.unique().tolist()), built=pd.Timestamp.utcnow().strftime("%Y-%m-%d"),
            n_players=int(d.player_id.nunique()) if "player_id" in d.columns else int(rat.player_id.nunique()))
payload = dict(meta=meta, rows=rows)

# the season board, when it has been built: the same columns keyed by season, `p` the player's own
# regular-season possessions (the fit's kernel-weighted ones are a different quantity and would read oddly
# beside a block row)
sp = root / "outputs" / "season_ratings.parquet"
if sp.exists():
    sr = pd.read_parquet(sp)
    scols = {"season": "s", "player_name": "n", "rating_off": "o", "rating_def": "d", "rating_total": "t",
             "poss_season": "p"}
    e = sr[list(scols)].rename(columns=scols).copy()
    e["n"] = e["n"].fillna("").astype(str)
    e = e[e.p > 0].sort_values(["s", "t"], ascending=[True, False])
    payload["seasons"] = [dict(s=int(r.s), n=r.n, o=round(float(r.o), 2), d=round(float(r.d), 2),
                               t=round(float(r.t), 2), p=int(r.p)) for r in e.itertuples(index=False)]
    meta["seasons"] = sorted(int(x) for x in e.s.unique())
    print(f"  season board: {len(payload['seasons'])} rows, {len(meta['seasons'])} seasons")

# the PLAYOFF DELTA (scripts/63_playoff_delta.py), when it has been built: not a playoff rating but how far
# the postseason moved a player off his regular-season number, which is what the estimator actually produces
# (the playoff fit's offset IS the regular-season rating, so its residual IS the delta).  `r` is that
# regular-season total, `p` the playoff possessions the delta rests on.
pp = root / "outputs" / "playoff_delta.parquet"
if pp.exists():
    pl = pd.read_parquet(pp)
    # the Season column must be the number the OTHER views show for that player, not the playoff script's
    # own regular-season fit -- that one is unmapped, so it correlates 0.90 with the board rather than 1.00
    # and a reader switching views would see two different "season" ratings for the same man
    pl = pl.merge(rat[["player_id", "window", "rating_total"]], on=["player_id", "window"], how="left")
    pl["rs_total"] = pl["rating_total"].fillna(pl["rs_total"])
    pcols = {"window": "w", "name": "n", "rs_total": "r", "d_off": "o", "d_def": "d", "d_total": "t",
             "po_poss": "p"}
    f = pl[list(pcols)].rename(columns=pcols).copy()
    f["n"] = f["n"].fillna("").astype(str)
    f = f[f.p > 0].sort_values(["w", "t"], ascending=[True, False])
    payload["playoffs"] = [dict(w=r.w, n=r.n, r=round(float(r.r), 2), o=round(float(r.o), 2),
                                d=round(float(r.d), 2), t=round(float(r.t), 2), p=int(r.p))
                           for r in f.itertuples(index=False)]
    meta["playoff_windows"] = sorted(f.w.unique().tolist())
    print(f"  playoff delta: {len(payload['playoffs'])} rows, {len(meta['playoff_windows'])} windows")

(out / "ratings.json").write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
print(f"wrote docs/data/ratings.json: {len(rows)} rows, {len(meta['windows'])} windows")
