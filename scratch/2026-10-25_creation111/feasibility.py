"""Can 2016 tracking tell a missed shot that came off a pass?  Makes: the assist flag against dribbles and touch time;
then makes against misses in the same zone on those handling measures."""
import numpy as np
import pandas as pd

mv = pd.read_parquet("data/tracking/movement_shots_2016.parquet")
sf = pd.concat([pd.read_parquet(f"data/shotframe/2016_{p}.parquet") for p in ("RS",)], ignore_index=True)
d = sf.merge(mv[["game_id", "action_number", "dribbles", "touch_time"]], on=["game_id", "action_number"])
print(f"shots matched {len(d)} of {len(mv)} tracked; games {d.game_id.nunique()}")
d = d[d.at_heave == 0]
d["zone"] = np.select([d.value == 3, d.dist <= 6], ["three", "rim"], "mid")
d["putback"] = (d.att_no > 1) & (d.prev_event_clock - d.clock <= 3)
d["drib"] = pd.cut(d.dribbles, [-1, 0, 1, 2, 99], labels=["0", "1", "2", "3+"])
mk = d[d.made == 1]
print("\nmakes: share assisted by dribbles before the shot")
print(mk.groupby("drib", observed=True).post_assisted.agg(["mean", "size"]).round(3).to_string())
print("\nmakes: share assisted by touch time")
print(mk.groupby(pd.cut(mk.touch_time, [-1, 1, 2, 4, 99]), observed=True).post_assisted.agg(["mean", "size"]).round(3).to_string())
print("\nshare of shots with 0-1 dribbles and touch under 2 s, by zone, makes vs misses, putbacks out")
q = d[~d.putback].assign(catch=lambda t: (t.dribbles <= 1) & (t.touch_time < 2))
print(q.groupby(["zone", "made"]).catch.mean().unstack().round(3).to_string())
print("\nmakes: assisted share by zone, putbacks out")
print(q[q.made == 1].groupby("zone").post_assisted.mean().round(3).to_string())
print(f"\nputback share: makes {d[d.made==1].putback.mean():.3f}, misses {d[d.made==0].putback.mean():.3f}")
print("columns of poss_start:", d.poss_start.value_counts().head(8).to_dict())
