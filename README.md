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

**That works on a fresh clone with no data.** Most of the suite runs against synthetic fixtures or
is pure function, and the handful that need the scraped play-by-play skip with a message telling you
which command builds it. You can read, change and test the model without downloading anything.

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

A multi-stage prior-informed RAPM. `PIPELINE.md` draws it stage by stage with the file behind each
box; the short version:

1. **APM** — a ridge with an almost-zero penalty on the on-court result, offense on a free-throw-
   adjusted target and defense on an opponent-three-point-adjusted one.
2. **A role prior** — a ridge of APM on the player's share of team possessions, his starts share and
   his age, fit leave-one-out. This is what carries a player the possessions cannot see.
3. **A prior-informed RAPM** — the shipped ridge pulled toward that role prior.
4. **A boosted box prior** — a gradient-boosted model of that RAPM from the player's padded box
   rates, his role and his size, trained on his *other* seasons so it can never memorise this one.
5. **The rating** — the ridge pulled toward the boosted prior. Positive is good on both ends.
6. **A calibration map** — the ridge over-shrinks, by an amount that depends on how many possessions
   it saw, and this corrects it. Fitted on the test itself, leave-one-season-out.

Against an external consensus of public metrics, pooled over 2024-2026 and 484 matched players, rank
agreement is 0.810 total, 0.822 offense, 0.758 defense. That consensus is a **sanity check, read
once, never a fitting target** — it exists to catch a board that has gone gross-wrong.

## Building it

```bash
python scripts/01_ingest.py 1997 2026 RS,PO   # play-by-play and box scores. DAYS, but resumable.
python scripts/02_stints.py 1997 2026 RS,PO   # possessions with the same ten on the floor
python scripts/49_role_panel.py --check       # APM, role prior, prior-informed RAPM per player
python scripts/60_season_board.py             # the board -> outputs/season_ratings.parquet
python scripts/52_site.py                     # -> docs/data/ratings.json for the page
```

Step 1 hits stats.nba.com once per game for about 40,000 games and takes days. It is cached and
idempotent, so stopping and restarting it is fine. Everything under `data/` is rebuilt by steps 1
and 2 and none of it is in git.

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
