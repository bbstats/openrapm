# OpenRAPM

Open player-impact ratings for the NBA, 1996-97 to now, in points per 100 possessions, one rating
per player per season. Live at <https://bbstats.github.io/openrapm/>.

The point of the project is the **test**, not any one model: every change is judged by whether it
predicts games it never saw. Anyone is welcome to propose one; the criterion decides.

## Start here

```bash
git clone https://github.com/bbstats/openrapm && cd openrapm
python -m venv .venv && .venv/bin/pip install -e ".[dev]"     # .venv/Scripts/pip on Windows
pytest -q
```

**That works on a fresh clone with no data**: 305 pass, 10 skip, 1 xfail, about 145 seconds
(measured 2026-09-17). Most of the suite runs against synthetic fixtures or is pure function; the
10 that need scraped play-by-play name the command that builds it in their skip message. Tests
cannot reach the network at all — `tests/conftest.py` blocks it, and `tests/test_network_block.py`
is the regression for that — so nothing is downloaded behind your back and the number above is
what you will see.

Eight of those passes, and the xfail, are the consensus checks on the shipped ratings. Until
2026-09-17 all eleven of them skipped on any machine without `data/raw` — including every CI run —
because one fixture reached for box scores that only one of them needed.

You can read, change and test the model without downloading anything.

Python 3.11+. The one unfamiliar dependency is [`chimeraboost`](https://github.com/bbstats/chimeraboost),
the gradient booster the box prior uses; it is first-party and Apache-2.0.

## The two rules

**1. The current season is never trained on.** Every season may be loaded and rated. A season may be
*fit* on only once its Finals are over — which `src/eracoef/seasons.py` reads from the data rather
than from a constant, by asking whether any team has won the sixteen playoff games that win a title.
That covers the box prior, the role prior, the calibration map and every hyper-parameter. The ridge
is exempt, because a rating *is* a ridge coefficient and solving it on the season in progress is the
product rather than a fit.

**2. Model code does not load its own data.** The data layer reads the disk and the network; the
model layer is handed data and returns numbers. `tests/test_layer_boundary.py` enforces it. Add a
data source freely — tracking data, a different feed, anything — but add it in the data layer.

## The test

Hold out one season. Fit on the seasons either side of it. Predict every stint of the held-out
season from the ten players on the floor, with only the level and home court refit on it. Score
possession-weighted squared error in points per 100, summed to team-games so possession noise
cancels. Twenty-eight held-out seasons, every system paired by season.

```bash
python scripts/45_holdout.py --systems=ks52_lam05,mspi --k=2,4 --workers=4 --tag=x
python scripts/48_ladder.py outputs/holdout_x.parquet --ref=ks52_lam05
```

A system is anything with a name and `fit(train_seasons, ctx) -> Ratings`
(`src/eracoef/holdout.py`); `src/eracoef/systems.py` names the ones built so far.

Winning that test is necessary and not sufficient. A gain that only shows up once, in one search,
against the same seasons everyone has been tuning against, is not a gain — see `CONTRIBUTING.md` for
what a pull request has to clear, and `src/eracoef/pbo.py` for the number that measures it.

## What ships

A box prior pulled toward what one season's possessions say. **One rating per player per season,
from that season's games only** — a player's other seasons may reach his rating's coefficients but
never his rating. `scripts/62_single_year_board.py` is the whole chain:

1. **The label** — one RAPM per player over every season *except* the one being rated
   (`looseason.py`). It is what the box prior is taught to predict, and it cannot contain the answer.
2. **The box prior** — chimeraboost on the player's padded box rates, his role and his size
   (`singleyear.py`, `gbdt_prior.py`). Trained on his career row plus contiguous 1-3 season chunks,
   and fitted in five player folds so no player's prior comes from a fit that saw his own rows.
3. **The rating** — a ridge of the season's own possessions with that prior as its centre
   (`priorridge.py`): `scale x prior + residual`, the scale priced on cross-fitted prior columns,
   the residual penalty fixed at 13,037 on both sides. Then centred at possession-weighted zero.

Positive is good on both ends. `PIPELINE.md` draws the older three-season-window board, which
`scripts/60_season_board.py` still builds and several tests still read.

Against an external consensus of public metrics, pooled over 2024-2026 and 475 matched players, rank
agreement is 0.772 total, 0.782 offense, 0.765 defense. That consensus is a **sanity check, read
once, never a fitting target** — it exists to catch a board that has gone gross-wrong.

## Building it

```bash
python scripts/01_ingest.py 1997 2026 RS,PO   # play-by-play and box scores. DAYS, but resumable.
python scripts/02_stints.py 1997 2026 RS,PO   # possessions with the same ten on the floor
python scripts/49_role_panel.py --season      # APM, role prior, prior-informed RAPM, one row per season
python scripts/62_single_year_board.py --out=season_ratings_product    # the ratings, ~3 min a season
python scripts/52_site.py                     # -> docs/data/ratings.json for the page
```

Step 1 hits stats.nba.com once per game for about 40,000 games and takes days. It is cached and
idempotent, so stopping and restarting it is fine. Everything under `data/` is rebuilt by steps 1
and 2 and none of it is in git.

Step 4 is the one that decides what ships: `outputs/season_ratings_product.parquet` is what step 5
publishes and what `tests/test_vs_consensus.py` scores. `scripts/60_season_board.py` builds the
older three-season-window board into `outputs/season_ratings.parquet`; its committed copy lives at
`artifacts/season_ratings.parquet` and several tests still read it, so do not overwrite that one.

The two artifacts that ship — the role panel and the calibration table — are committed under
`artifacts/`, so steps 3 onward work from a clone without refitting.

## Layout

    src/eracoef/
      seasons.py     which seasons may be trained on -- the trust boundary
      ingest.py      play-by-play and box scores from stats.nba.com
      stints.py      play-by-play to stints
      design.py      the sparse design: players, fixed effects, box exposures
      exposure.py    cross-fitted, empirical-Bayes padded per-100 rates
      estimator.py   the mixed model (Henderson), eigen lambda path
      fastfit.py     the shipped one-pass fit
      roles.py       playing-time share, starts, age per player-season
      spm.py         APM, the role prior, the chain's offset
      gbdt_prior.py  the boosted box prior, leave-one-out
      looseason.py   the label: one RAPM per player from every season but the rated one
      singleyear.py  the single-year prior's features and its per-player aggregation
      priorridge.py  the ratings ridge: the prior as the centre, the season's possessions as evidence
      calmap.py      the calibration map
      holdout.py     the out-of-season test: systems, runner, splits
      pbo.py         the probability that a search's winner is a fluke
      systems.py     every named system

`DECISIONS.md` is what the test has settled, what it rejected, and the measurement traps that have
produced confident wrong numbers here before. Read the traps before trusting a number.

## Contributing

See `CONTRIBUTING.md`. In one line: change the model layer, run the test against what ships, and
report every variant you tried — not just the one that won.

MIT licensed.
