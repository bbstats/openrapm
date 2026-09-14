---
name: experiment-comparison
description: Show two or more tables of player rankings side by side as an interactive HTML page (sortable, filterable, rank changes coloured) with the year-over-year summary on top. Use after every experiment, and whenever the user asks to compare rankings, see a top 20, or "show me the tables".
---

# Experiment comparison

The owner reads the **2026 top 20** for every experiment; the eye test matters as much as the numbers.
This skill turns the parquet tables the rankings script writes into one interactive page.

## Run it

```
.venv/Scripts/python scripts/66_compare.py incumbent=outputs/season_ratings_sy_chunks_cfs.parquet <name>=outputs/season_ratings_<name>.parquet --season=2026 --top=20 --yoy=outputs/yoy_<name>.parquet --ref=incumbent
```

- Name the **incumbent first**; rank changes are measured against the first table named.
- Use plain names on the page (`incumbent`, `off_court`); when they differ from the names the
  year-over-year run used (`--rankings=<name>=...` in `63_yoy.py`), map them with
  `--alias=off_court:sy_offc` so the summary block can find them.
- The page is written to `outputs/compare_<names>_<season>.html` and opened in the browser
  (`--open=0` to skip).  It is self-contained: no network, safe to attach or keep.

## What the page shows

1. **Summary** (when `--yoy=` is given): per table, the year-over-year error per team-game in points per
   100, what the scored season wants each side multiplied by, and the paired comparison against the
   reference: mean difference in team-game MSE, z, seasons better out of 56.
2. **The main table**: every player in any table's top N for the season, with possessions, then rank,
   offence, defence and total per table, and the rank change against the first table (green up, red
   down; hover for the two ranks).  Rows in one table's top N but not another's are highlighted.
   Toggle "show every player" to see the whole season; type to filter by name; click headers to sort.

## How to report it in chat

Give the path of the page, then a plain-words reading of the top 20: who moved in, who moved out, who
moved more than ten places, and the mechanism if it is known.  Then the two year-over-year numbers and
the consensus line.  No invented labels; say "the incumbent" and the experiment's plain name.
End with one question: adopt, keep, or the next idea.

## Conventions this relies on

- A rankings table has `player_id`, `player_name`, `season`, `rating_off`, `rating_def`, `rating_total`,
  `poss_off` (positive good on both ends), as `scripts/62_single_year_board.py` writes it.
- A year-over-year parquet comes from `scripts/63_yoy.py`, systems named `<name>:prev` / `<name>:next`.
