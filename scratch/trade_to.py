"""The trade-to question: a player's offensive prior and rating at his actual destination, at a league-average
one, and on every team's roster, ranked (context.py, FINDINGS 30).

    python scratch/trade_to.py "<player name>" ["<player name 2>" ...] [--window=2024-2026] [--system=tune501_b7_pasto_pOD_dest]

Fits the system on the window once per context (the actual rosters, the league average, then each team's
roster with the team's possessions as weights) and prints, per named player, the offensive prior and the
raw rating under each.  Writes outputs/csv/trade_to_<window>.csv with every player x every context.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, "src")
import numpy as np
import pandas as pd

from eracoef.config import load_config
from eracoef.context import team_roster
from eracoef.holdout import Context
from eracoef.roles import build_roles
from eracoef.systems import registry


def flag(name, default=None):
    hit = [a for a in sys.argv[1:] if a.startswith(f"--{name}=")]
    return hit[0].split("=", 1)[1] if hit else default


names = [a for a in sys.argv[1:] if not a.startswith("--")]
WIN = flag("window", "2024-2026")
SYS = flag("system", "tune501_b7_pasto_pOD_dest")
seasons = list(range(int(WIN[:4]), int(WIN[5:9]) + 1))
cfg = load_config()
S = registry(cfg)
ctx = Context.load(cfg)
ctx.current_h = None                          # no held-out season: the actual context is the block's own rosters, as the board builds it
sysm = S[SYS]
roles = build_roles(cfg, verbose=False)
bio = pd.concat([pd.read_parquet(f, columns=["PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "TEAM_ABBREVIATION"])
                 for f in sorted((Path(cfg["_root"]) / "data" / "raw" / "bio").glob("*_RS.parquet"))[-len(seasons):]])
abbr = bio.drop_duplicates("TEAM_ID").set_index("TEAM_ID").TEAM_ABBREVIATION.to_dict()
pname = bio.drop_duplicates("PLAYER_ID").set_index("PLAYER_ID").PLAYER_NAME.to_dict()
ids = {}
for n in names:
    hit = [k for k, v in pname.items() if v.lower() == n.lower()]
    if not hit:
        raise SystemExit(f"no player named {n!r} in {WIN}")
    ids[n] = int(hit[0])
teams = sorted(roles[roles.season.isin(seasons)].team_id.unique())

contexts = [("actual", None)]
t0 = time.time()
# the league average of the actual destinations: computed from the actual fit's own inputs below
rows = []
res = {}
for label, ov in contexts:
    ctx.dest_override = ov
    r = sysm.fit(seasons, ctx).df.set_index("player_id")
    res[label] = r
    print(f"  {label:8s} fitted ({time.time() - t0:.0f}s)", flush=True)
# league average: every player's own usage minutes kept, the destination set to the block's mean context
from eracoef.context import block_usage_apm  # noqa: E402
ctx.dest_override = {"dest_um": float("nan"), "dest_apm": float("nan")}   # placeholder, replaced below
# the mean context comes from the actual inputs; recompute them the way the fit does
wd = ctx.design(seasons, "pts", ("RS",), counter_cols=sysm.counter_columns())
from eracoef.cv import make_exposure  # noqa: E402
from eracoef.roles import window_inputs  # noqa: E402
from eracoef.turnover import cached_table  # noqa: E402
from eracoef.context import destination_inputs  # noqa: E402
exp = make_exposure(wd, mode="full", pad_target=cfg["pad_target"])
exp.parts = wd.parts
exp.fit(None, sample_weight=wd.w)
inputs = window_inputs(wd, ctx.role_inputs, cap=float(cfg.get("roles", {}).get("share_cap", 0.9)))
blk = block_usage_apm(wd, exp, inputs, cfg)
act = destination_inputs(blk, cached_table(ctx), seasons, wd.spec.ps_table["player_id"].to_numpy())
w = np.asarray(exp.season_poss_off_, dtype=float)
avg = {"dest_um": float(np.average(act.dest_um, weights=w)), "dest_apm": float(np.average(act.dest_apm, weights=w))}
ctx.dest_override = avg
res["average"] = sysm.fit(seasons, ctx).df.set_index("player_id")
print(f"  average  fitted: dest_um {avg['dest_um']:.1f}, dest_apm {avg['dest_apm']:+.2f} ({time.time() - t0:.0f}s)", flush=True)
for tid in teams:
    ro, wt = team_roster(roles, seasons, int(tid), tm=cached_table(ctx))
    if len(ro) < 8:
        continue
    ctx.dest_override = {"roster": ro, "weights": wt}
    res[abbr.get(int(tid), str(tid))] = sysm.fit(seasons, ctx).df.set_index("player_id")
print(f"  {len(teams)} team rosters fitted ({time.time() - t0:.0f}s)", flush=True)
ctx.dest_override = None

out = []
for label, r in res.items():
    for n, pid in ids.items():
        if pid in r.index:
            out.append(dict(player=n, context=label, prior_o=float(r.loc[pid, "prior_o"]), rating_o=float(r.loc[pid, "o"]),
                            prior_d=float(-r.loc[pid, "prior_d"]), rating_d=float(-r.loc[pid, "d"])))
T = pd.DataFrame(out)
path = Path("outputs/csv") / f"trade_to_{WIN}.csv"
T.to_csv(path, index=False)
pd.set_option("display.width", 200, "display.precision", 2)
for n in names:
    t = T[T.player == n].set_index("context")
    a, g = t.loc["actual"], t.loc["average"]
    print(f"\n{n}, {WIN}: offensive prior {a.prior_o:+.2f} on his actual rosters, {g.prior_o:+.2f} on a league-average one; "
          f"raw offensive rating {a.rating_o:+.2f} / {g.rating_o:+.2f}")
    teams_t = t.drop(index=["actual", "average"]).sort_values("prior_o", ascending=False)
    print("  best fits (offensive prior on that team's roster):  " + ", ".join(f"{k} {v:+.2f}" for k, v in teams_t.prior_o.head(6).items()))
    print("  worst fits:                                          " + ", ".join(f"{k} {v:+.2f}" for k, v in teams_t.prior_o.tail(6).items()))
print(f"\nwrote {path}")
