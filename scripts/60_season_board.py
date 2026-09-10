"""The season board: one rating per player per SEASON, from the rolling in-season kernel.

The shipped board (scripts/08_ratings.py) is one fit per disjoint three-season block.  This is the other
product: for every season `a`, the same estimator fit on `a` and the two seasons before it with the earlier
ones down-weighted (src/eracoef/inseason.py, config `ratings_prior.season_board`).  Nothing after `a` is
used, so the latest season's row is a rating of the season in progress and re-running the script updates it.

Per season, per player:
  prior_*   the boosted role-and-box prior, the fit's offset (defense flipped so positive = good)
  u_*       the ridge residual beyond it
  rating_*  prior + u, then the calibration map and a possession-weighted re-centring within the season
  poss_off  the KERNEL-WEIGHTED possessions the rating rests on (a season counts for its kernel weight),
            beside `poss_season`, the player's actual regular-season possessions in `season`

The map is the one fitted on the in-season dump (scratch/inseason_run.py), so the level and the exposure
term were chosen out of sample on fits shaped like these -- not on the block board's.
usage: python scripts/60_season_board.py [--system=ks55] [--first=1997] [--last=2026] [--out=<stem>]
       [--map=<parquet>] [--map-system=<name in it>] [--map-base=<the dumped system>] [--map-k=3]
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eracoef.boxtable import player_names, season_box  # noqa: E402
from eracoef.calmap import apply_params, params_row  # noqa: E402
from eracoef.config import load_config  # noqa: E402
from eracoef.holdout import Context  # noqa: E402
from eracoef.inseason import kernel_seasons  # noqa: E402
from eracoef.roles import build_roles, player_season_inputs  # noqa: E402
from eracoef.stints import season_names  # noqa: E402
from eracoef.systems import registry  # noqa: E402

pd.set_option("display.width", 250, "display.max_columns", 40, "display.precision", 2)
cfg = load_config()
OUT = Path(cfg["_root"]) / "outputs"
CSV = OUT / "csv"
CSV.mkdir(parents=True, exist_ok=True)


def flag(name, default=None):
    hit = [a for a in sys.argv[1:] if a.startswith(f"--{name}=")]
    return hit[0].split("=", 1)[1] if hit else default


SB = dict(cfg.get("ratings_prior", {}).get("season_board") or {})
name = flag("system", SB.get("system", "ks55"))
first = int(flag("first", cfg["first_season"]))
last = int(flag("last", cfg["last_season"]))
CM = dict(SB.get("cal_map") or {})
for _f, _key in (("map", "table"), ("map-system", "system"), ("map-base", "base"), ("map-k", "k")):
    if flag(_f):
        CM[_key] = flag(_f)          # a candidate kernel needs a map fitted on ITS OWN dump, not the shipped one

ctx = Context.load(cfg)
S = registry(cfg)
if name not in S:
    raise SystemExit(f"unknown system {name!r}; add it to src/eracoef/systems.py")
system = S[name]
if getattr(system, "kernel", None) is None:
    raise SystemExit(f"{name} has no kernel; the season board needs an in-season system (inseason.py)")

print(f"season board: {name}, kernel {system.kernel}, seasons {first}-{last}", flush=True)

roles = player_season_inputs(build_roles(cfg, verbose=False), cap=float(cfg.get("roles", {}).get("share_cap", 0.9)))
rows, t0 = [], time.time()
for a in range(first, last + 1):
    # `kernel_seasons`, not every offset the kernel names: a season the kernel weights ZERO must not enter
    # the list, or the inputs built from season tables (the shot-quality features, the role inputs) pool it
    # anyway.  See inseason.kernel_seasons.
    train = kernel_seasons(system.kernel, a, cfg["first_season"])
    rat = system.fit(train, ctx)                      # ctx.current_h stays None: this is a board, not a test
    d = rat.df.copy()
    d["season"] = a
    d["train"] = ",".join(map(str, train))
    # the player's own possessions in the season being rated, beside the kernel-weighted ones
    own = roles[(roles.season == a) & (roles.games > 0)].set_index("player_id")["poss_on"]
    d["poss_season"] = d.player_id.map(own).fillna(0.0).astype(float)
    # the map's `tshare` covariate: his share of his teams' possessions over the training seasons
    t = roles[roles.season.isin(train) & (roles.games > 0)].groupby("player_id")[["poss_on", "team_poss"]].sum()
    sh = (t.poss_on / t.team_poss.replace(0.0, np.nan)).fillna(0.0)
    d["tshare"] = d.player_id.map(sh).fillna(0.0).astype(float)
    names = player_names(season_box(train, ["RS"], cfg),
                         pd.concat([season_names(s, "RS", cfg) for s in train], ignore_index=True))
    nm = names.sort_values("season").drop_duplicates("player_id", keep="last")[["player_id", "player_name"]]
    d = d.merge(nm, on="player_id", how="left")
    rows.append(d)
    print(f"  {a}: {len(d)} players from {d.train.iloc[0]} ({time.time() - t0:.0f}s)", flush=True)

rat = pd.concat(rows, ignore_index=True)
# the model's raw sign in, the board's out: defense positive = good
rat["prior_off"], rat["prior_def"] = rat.prior_o, -rat.prior_d
rat["rating_off_raw"], rat["rating_def_raw"] = rat.o, -rat.d
rat["u_off"], rat["u_def"] = rat.o - rat.prior_o, -(rat.d - rat.prior_d)
rat = rat.rename(columns={"poss": "poss_off", "poss_d": "poss_def"})

if CM.get("table"):
    tbl = Path(cfg["_root"]) / CM["table"]
    if not tbl.exists():
        raise SystemExit(f"{tbl} is missing; run scratch/inseason_run.py for {CM.get('system')} first")
    row = params_row(pd.read_parquet(tbl), CM["system"], CM["base"], int(CM.get("k", 3)), None)
    EXTRA = pd.DataFrame({"tshare": rat.tshare.to_numpy()})
    o, d = apply_params(row, rat.rating_off_raw.to_numpy(), -rat.rating_def_raw.to_numpy(),   # the map is in raw sign
                        rat.poss_off.to_numpy(), prior_o=rat.prior_off.to_numpy(), prior_d=-rat.prior_def.to_numpy(),
                        extra=EXTRA)
    rat["rating_off"], rat["rating_def"] = o, -d
    print(f"calibration map applied from {CM['table']} ({CM['system']})")
else:
    rat["rating_off"], rat["rating_def"] = rat.rating_off_raw, rat.rating_def_raw
    print("no calibration map (ratings_prior.season_board.cal_map is unset): the raw kernel ratings")

# each side re-centred within the season, possession-weighted: 0 is the average player on the floor
w = rat.poss_off.to_numpy(dtype=float)
for col in ("rating_off", "rating_def"):
    v = rat[col]
    mean = (v * w).groupby(rat.season).transform("sum") / pd.Series(w, index=rat.index).groupby(rat.season).transform("sum")
    rat[col] = v - mean
for part in ("prior", "u", "rating"):
    rat[f"{part}_total"] = rat[f"{part}_off"] + rat[f"{part}_def"]
rat["rating_total_raw"] = rat.rating_off_raw + rat.rating_def_raw
rat["shrinkage"] = np.where(rat.poss_off > 0, rat.poss_off / (rat.poss_off + float(cfg["lam_plugin"])), 0.0)

cols = ["season", "train", "player_id", "player_name", "poss_off", "poss_def", "poss_season", "shrinkage",
        "prior_off", "prior_def", "prior_total", "u_off", "u_def", "u_total",
        "rating_off_raw", "rating_def_raw", "rating_total_raw", "rating_off", "rating_def", "rating_total"]
rat = rat[cols].sort_values(["season", "rating_total"], ascending=[True, False]).reset_index(drop=True)
# `--out=<stem>` writes beside the board instead of over it.  A partial run (one season, or a
# candidate system) that overwrote `season_ratings` used to break tests/test_vs_consensus.py
# silently, which reads this file as the shipped board.
stem = flag("out", "season_ratings")
rat.to_parquet(OUT / f"{stem}.parquet", index=False)
rat.round(4).to_csv(CSV / f"{stem}.csv", index=False)
print(f"\nwrote outputs/{stem}.parquet: {len(rat)} rows, {rat.season.nunique()} seasons "
      f"({time.time() - t0:.0f}s)")
top = rat[rat.season == rat.season.max()].head(15)
print(f"\ntop 15, {int(rat.season.max())} (kernel-weighted possessions, and his own in the season):\n")
print(top[["player_name", "rating_off", "rating_def", "rating_total", "poss_off", "poss_season"]].to_string(index=False))
