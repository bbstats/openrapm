---
name: experiment-comparison
description: Show two or more tables of player ratings side by side as an interactive HTML page (sortable, filterable, changes colored) with the year-over-year summary on top. Use after every experiment, and whenever the user asks to compare ratings, see a top 20, or "show me the tables".
---

# Experiment comparison

The owner reads the **2026 top 20** for every experiment; the eye test matters as much as the numbers.
This skill turns the parquet tables the build writes into one interactive page.

## Run it

`scripts/experiment_chain.sh` runs it for every candidate. By hand:

```
.venv/Scripts/python scripts/checks/66_compare.py incumbent=outputs/season_ratings_<incumbent>.parquet <name>=outputs/season_ratings_<name>.parquet --season=2026 --top=20 --yoy=outputs/yoy_<name>.parquet --ref=incumbent
```

- The incumbent is `INC_NAME` at the top of `scripts/experiment_chain.sh`. Name it **first**; changes are measured
  against the first table named.
- Use plain names on the page (`incumbent`, `off_court`); when they differ from the names the year-over-year run used
  (`--rankings=<name>=...` in `scripts/checks/63_yoy.py`), map them with `--alias=off_court:sy_offc` so the summary
  block can find them.
- The page is written to `outputs/compare_<names>_<season>.html` and opened in the browser (`--open=0` to skip). It is
  self-contained: no network, safe to attach or keep.

## What the page shows

1. **Summary** (when `--yoy=` is given): per table, the year-over-year error per team-game in points per 100, what the
   scored season wants each side multiplied by, and the paired comparison against the reference: mean difference in
   team-game MSE, z, seasons better out of 56.
2. **The main table**: every player in any table's top N for the season, with possessions, then offense, defense and
   total per table, and the change against the first table. Toggle "show every player" to see the whole season; type to
   filter by name; click headers to sort.

## Sending it

The owner reads on a phone and does not open local files. Never give a file path as the result. The script also writes
`outputs/compare_<names>_<season>_phone.html`: the same rows as one static table with inline styles. Send it as the
`htmlBody` of a Gmail message to the owner's own address, subject "Experiment comparison, <season>: <names>", or show
the rows in chat.

## How to report it in chat

Four parts: what changed, the result, the 2026 top 20, the owner's call. For the top 20: who moved in, who moved out,
and who moved by 0.5 points per 100 or more, in points per 100, never in rank places; the mechanism if it is known. One
quantity per column in any table, no parentheses, American English, no invented labels: say "the incumbent" and the
experiment's plain name. End with one question through the question tool: adopt, keep, or the next idea.

## Conventions this relies on

- A ratings table has `player_id`, `player_name`, `season`, `rating_off`, `rating_def`, `rating_total`, `poss_off`
  (positive good on both ends), as `python -m openrapm build` writes it.
- A year-over-year parquet comes from `scripts/checks/63_yoy.py`, systems named `<name>:prev` / `<name>:next`.
