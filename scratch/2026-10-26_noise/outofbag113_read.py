"""Step 4's reading: the out-of-bag prior's four builds (itself and three noise controls, every bag's resample redrawn)
against today's four (today's fixed split and three random splits), season by season on the within-season WOWY test
after step 4, by era; and how far each set of four spreads.  Reads outputs/csv/wowy_outofbag113_all_seasons.csv."""
import numpy as np
import pandas as pd

ERAS = {"1997-2006": range(1997, 2007), "2007-2016": range(2007, 2017), "2017-2026": range(2017, 2027)}
TODAY = ["creation110", "noise_creation110", "noise2_creation110", "noise3_creation110"]
OOB = ["outofbag113", "noise_outofbag113", "noise2_outofbag113", "noise3_outofbag113"]

seasons = pd.read_csv("outputs/csv/wowy_outofbag113_all_seasons.csv")
for stage in ["step4", "step3"]:
    errors = seasons[(seasons.stage == stage) & (seasons.slice == "all")].pivot(index="season", columns="system",
                                                                                values="error")
    rows = []
    for name, diff in [*((b, errors[b] - errors[t]) for b, t in zip(OOB, TODAY)),
                       ("four_averaged", errors[OOB].mean(1) - errors[TODAY].mean(1))]:
        se = diff.std(ddof=1) / np.sqrt(len(diff))
        row = dict(build=name, change=diff.mean(), z=diff.mean() / se, seasons_better=int((diff < 0).sum()))
        for era, years in ERAS.items():
            row[era] = diff.reindex(years).mean()
        rows.append(row)
    print(f"\n{stage}: out of bag minus today's, matched build by build; below 0 is better")
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    spread = []
    for label, group in [("today", TODAY), ("out_of_bag", OOB)]:
        row = dict(set=label, sd_of_30_season_means=errors[group].mean().std(ddof=1),
                   per_season_sd=errors[group].std(axis=1, ddof=1).mean())
        for era, years in ERAS.items():
            row[f"sd_{era}"] = errors[group].reindex(years).mean().std(ddof=1)
        spread.append(row)
    print(f"{stage}: how far each set of four spreads")
    print(pd.DataFrame(spread).round(4).to_string(index=False))
