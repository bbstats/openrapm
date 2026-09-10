# Contributing

The project is a test with a model attached. If your change predicts held-out games better, it
ships, whoever wrote it and however strange it looks. The whole job of this document is to make that
claim cheap for you to make and cheap for a maintainer to believe.

## Setup

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"     # .venv/Scripts/pip on Windows
pytest -q
```

That passes on a fresh clone with no data. You do not need the scraped play-by-play to change the
model, run the unit tests or open a draft PR — only to run the criterion itself.

## The two rules

**1. Nothing that is fitted may see a season still being played.**

Every season may be loaded and rated. A season may be *fit* on only once its Finals are over.
`src/eracoef/seasons.py` decides which is which, by reading the game log and asking whether any team
has won the sixteen playoff games that win a title — so there is no year to bump and no flag to set.

This binds the box prior, the role prior, the calibration map, the padding constants, the feature
lists and every hyper-parameter. It does not bind the ridge: a rating *is* a ridge coefficient, and
solving it on the season in progress is the product rather than a fit.

If you need training rows, take them from a `Trainable`. It refuses to hold a season past the
cutoff, so you cannot be handed one by accident.

**2. Model code does not load its own data.**

```
data/   reads the disk and the network      -- scraping, stints, box tables, caches, loaders
model/  is handed data and returns numbers  -- the ridge, the priors, the padding, the calibration
```

`tests/test_layer_boundary.py` enforces this: a module on the model list may not call `read_parquet`,
`open`, `glob` or import `requests`. If your change needs data the loaders do not produce, add the
loader to the data layer and say so in the PR — that is a welcome kind of contribution, and it is
where a new source of data (tracking, a different feed) belongs.

## Making a change

Most model changes are a new entry in `src/eracoef/systems.py`. A system is anything with a name and
`fit(train_seasons, ctx) -> Ratings`.

```bash
python scripts/45_holdout.py --systems=ks52_lam05,my_idea --k=2,4 --workers=4 --tag=mine
python scripts/48_ladder.py outputs/holdout_mine.parquet --ref=ks52_lam05
```

## What a pull request has to clear

Four numbers. All four.

| | bar | why |
|---|---|---|
| **Δ points per 100** | negative, at team-game level, paired by season | the criterion |
| **Seasons won** | at least 60% of the held-out seasons | one lucky season is not a result |
| **PBO** | at or below 0.25 | the gain is not selection noise |
| **Runtime** | within 5x the baseline | somebody else has to be able to run this |

**PBO is the one people have not met before.** It asks how likely it is that the winner of your
search is a fluke. Split the held-out seasons into halves every balanced way there is, find what won
each first half, see where it ranks on the second. If your idea is real it keeps winning; if it was
noise it lands near the middle. `src/eracoef/pbo.py` has the measured thresholds.

**PBO only works if it sees everything you tried.** So:

> **Log every variant, not the one that won.**

This is the actual ask, and it is not a formality. Try forty things against the same 28 seasons and
one of them beats the board by 0.06 on noise alone. A PR that tried one thing and won gets a low PBO
honestly. A PR that tried forty and reports the best gets the number that deserves. Both are fine —
what is not fine is a candidate set that has been quietly pruned before the statistic sees it.

Paste the output of `pbo.report(...)` over your full candidate set, including the baseline.

## Three standing rules

**Accuracy wins, provided the testing is robust — and this has to stay runnable by strangers.**
A real accuracy gain is never traded away for seconds. But between two candidates the criterion
cannot separate, the simpler and faster one wins every time.

**The external consensus is a sanity check, never a fitting target.** `data/external/consensus.csv`
is a blend of public metrics, read once, to catch a board that has gone gross-wrong. Never choose a
constant because it clears a floor in `tests/test_vs_consensus.py`. A marginal miss is not a veto —
0.759 against a 0.76 floor is noise. A gross miss is, because that is what the check is for.

**Read the traps before trusting a number.** `DECISIONS.md` ends with the specific ways this
pipeline has produced confident wrong numbers: benchmarks that shared the estimate's blind spot,
an argmax sitting on a grid boundary, a feature that predicted how well-*measured* a row was rather
than how good the player was, a gain that the calibration map had already delivered. Every one of
them survived review at the time. They will cost you a day each if you meet them fresh.

## Things that will not be merged

- A change that only wins on the half of the seasons it was chosen on.
- A constant chosen because it cleared a consensus floor.
- A gain quoted before the calibration map was fitted. The map absorbs amplitude; a candidate that
  only rescales a side reads large unmapped and near zero mapped.
- Anything fitted on a season whose Finals are not over. The tests will catch this, but please do
  not make them.

## Reporting a bug

If a number looks wrong, say which number, what you expected and what command produced it. If it is
a rating that looks wrong, name the player and the season — that is usually faster to chase than a
summary statistic, and `scripts/57_investigate.py` exists for exactly that.
