"""Step 1's reading: today's build (its fixed split) against three random splits of the box-score prior's players,
season by season on the within-season WOWY test after step 4, by era; and how far four builds that differ only in the
split spread.  Reads outputs/csv/wowy_noise3_seasons.csv (check 151, tag noise3)."""
import numpy as np
import pandas as pd

ERAS = {"1997-2006": range(1997, 2007), "2007-2016": range(2007, 2017), "2017-2026": range(2017, 2027)}
RANDOM = ["noise_creation110", "noise2_creation110", "noise3_creation110"]

seasons = pd.read_csv("outputs/csv/wowy_noise3_seasons.csv")
errors = seasons[(seasons.stage == "step4") & (seasons.slice == "all")].pivot(index="season", columns="system",
                                                                              values="error")
change = errors[RANDOM].sub(errors["creation110"], axis=0)
change["three_random_averaged"] = change[RANDOM].mean(axis=1)


def paired(diff: pd.Series) -> dict:
    se = diff.std(ddof=1) / np.sqrt(len(diff))
    return dict(change=diff.mean(), se=se, z=diff.mean() / se, seasons_better=int((diff < 0).sum()), seasons=len(diff))


rows = []
for column in change.columns:
    row = dict(build=column, **paired(change[column]))
    for era, years in ERAS.items():
        row[f"change_{era}"] = change[column].reindex(years).mean()
    rows.append(row)
print("each random split minus today's fixed split, step 4, all held-back possessions; below 0 is better")
print(pd.DataFrame(rows).round(4).to_string(index=False))

# the random splits against each other: what a change that means nothing looks like
pairs = []
for i, a in enumerate(RANDOM):
    for b in RANDOM[i + 1:]:
        diff = errors[a] - errors[b]
        row = dict(pair=f"{a} - {b}", **paired(diff))
        for era, years in ERAS.items():
            row[f"change_{era}"] = diff.reindex(years).mean()
        pairs.append(row)
print("\nrandom split against random split")
print(pd.DataFrame(pairs).round(4).to_string(index=False))

four = errors[["creation110", *RANDOM]]
spread = dict(per_season_sd_of_the_four=float(four.std(axis=1, ddof=1).mean()),
              sd_of_the_four_30_season_means=float(four.mean().std(ddof=1)),
              sd_of_the_three_random_30_season_means=float(four[RANDOM].mean().std(ddof=1)))
for era, years in ERAS.items():
    spread[f"sd_of_the_four_era_means_{era}"] = float(four.reindex(years).mean().std(ddof=1))
print("\nhow far the four builds spread")
for name, value in spread.items():
    print(f"  {name}: {value:.4f}")
change.round(4).to_csv("outputs/csv/noise3_by_season.csv")
print("\nwrote outputs/csv/noise3_by_season.csv")
