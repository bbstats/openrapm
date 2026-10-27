# The multi-stat model: handoff, 2026-10-27

One neural network that predicts all 13 box-score stats of a player's next game from every stat's history and its
denominator; on simulated leagues it beats the per-stat padding on all 13 stats, and the 30-season test is the next
step.

**Where it stands.** Built and tested on simulated leagues only; no real data has been read.  On a 15-season
simulated league it beats the per-stat recursion on all 13 stats on held-out seasons.  The 30-season run, OpenRAPM's
span, has not been run: `python run_v2.py 0 5` is the next command.  Put this folder at
`scratch/2026-10-27_multistat/`, beside the other scratch work; nothing in `src/` depends on it.

## Words used

- **stat**: one of 13, in this order: 3PA, 3P%, 2PA, 2P%, FTA, FT%, ORB, DRB, AST, TOV, STL, BLK, PF.  Ten are rates
  per 100 possessions; three (3P%, 2P%, FT%) are make percentages.
- **exposure**: a stat's denominator in one game: possessions for a rate, attempts for a percentage (3P% over 3PA).
- **per-stat recursion**, the baseline: each stat on its own, `pred = pred + w x (observed - pred)` with
  `w = e / (e + k)`, e the game's exposure, k fitted per stat.  It is the game-by-game form of the season panel's
  padding `(n x rate + k x target) / (n + k)` (`src/openrapm/box_rates.py`).
- **NLL**: negative log-likelihood of one game's outcome under a prediction.  A rate: a Poisson count over his
  possessions, `m - y log m` with `m = rate x possessions / 100`.  A percentage: a binomial over his attempts,
  `-(y log p + (a - y) log(1 - p))`.  Lower is better.
- **truth**: the simulator's true rates.  The NLL under them is the lowest any predictor can expect.
- **gap closed**: `(baseline NLL - model NLL) / (baseline NLL - truth NLL)`, over the games scored.  0% is the
  baseline; 100% would be knowing the true rates.
- **ceiling**: a Kalman filter given the simulator's exact structure and each player's age.  It cannot be built on
  real data; it shows how much of the gap is reachable at all.

## The model

    link_j = baseline link_j + network_j(inputs)        log rate for a rate, logit for a percentage

- **Inputs, 124 columns, all from earlier games only** (`inputs.py`): every stat's padded rate and its exposure over
  six sums of his earlier games: possession-discounted with time scales of 100, 400, 1,500 and 5,000 possessions
  (about 1, 5, 20 and 70 games of a starter), this season so far, and his career.  Then playing time (average
  possessions per game, share of team games played, games since he last played), experience (career games, seasons)
  and the baseline's 13 predictions.  A column is 0 at league average: `log(padded rate / league rate)`, or
  `logit(padded) - logit(league)`.  Example: 9 assists in 120 possessions with a 50-possession pad toward 4.8 is
  `(900 + 50 x 4.8) / 170 = 6.7` per 100, column `log(6.7 / 4.8) = +0.33`.
- **Network** (`model.py`): 124 -> 256 -> 256 -> 13, SiLU, dropout 0.1, weight decay 1e-4, Adam at 5e-4 with cosine
  decay, batches of 4,096 games, early stopping on the validation seasons.  Its last layer starts at 0, so before
  training it is the per-stat recursion.
- **Loss**: the sum of the 13 NLLs, each over its own exposure.  The different denominators need no weights: the 3P%
  term of a game is `3PA x softplus(link) - 3PM x link`, so a 10-attempt game counts ten times a 1-attempt game and
  a game with no threes adds nothing to 3P% while its possessions still inform 3PA.  3PA per 100 and 3P% are
  separate outputs; made threes per 100 is their product.
- **Prediction needs only numpy**: `model.forward_numpy(weights, x, offset)`.  `run_v2.py` saves the layers, the
  input standardization and the baseline's k and starting values to `net_seed{seed}.npz`.

## Files

| file | what it does |
|---|---|
| `sim_league.py` | simulated league with known true rates: 6 shared talents, aging, drift, role changes, injuries; league means from `simulate.LEAGUE_MEAN` |
| `baseline.py` | the per-stat recursion and its k per stat; the NLLs |
| `inputs.py` | the 124 pre-game inputs; `python inputs.py` runs the leakage check |
| `model.py` | the network, its loss, training, `BEST` settings, numpy prediction |
| `kf_ceiling.py` | the ceiling filter |
| `run_experiment.py` | 15-season league: baseline against one network |
| `tune.py` | settings compared on the validation seasons only |
| `run_v2.py` | 30-season league: baseline, one network, an ensemble of 5, the ceiling |

## Results so far, simulated

**First run, 15-season league**, seasons 0-10 train, 11-12 validate, 13-14 test, 63,089 test games.  The first
settings: learning rate 2e-3, weight decay 1e-5, no dropout.  Error is the root of the exposure-weighted mean squared
difference from the true rate, per 100 possessions or in percentage points.

| stat | gap closed | baseline error | network error | error reduction |
|---|---|---|---|---|
| 3PA | 29.2% | 1.211 | 1.114 | 8.0% |
| 3P% | 4.8% | 2.797 | 2.780 | 0.6% |
| 2PA | 15.5% | 1.242 | 1.182 | 4.9% |
| 2P% | 1.5% | 2.919 | 2.954 | -1.2% |
| FTA | 9.7% | 0.642 | 0.661 | -2.9% |
| FT% | 17.2% | 3.317 | 3.123 | 5.8% |
| ORB | 24.2% | 0.426 | 0.397 | 6.8% |
| DRB | 21.3% | 0.803 | 0.758 | 5.6% |
| AST | 20.1% | 0.836 | 0.802 | 4.1% |
| TOV | 9.1% | 0.400 | 0.391 | 2.3% |
| STL | 0.7% | 0.264 | 0.264 | 0.3% |
| BLK | 24.5% | 0.259 | 0.244 | 6.0% |
| PF | 2.1% | 0.467 | 0.465 | 0.5% |
| all 13 | 17.3% | | | |

**Settings, scored on the validation seasons 11-12 only** (`tune.py`; the test seasons were not read again):

| learning rate | weight decay | dropout | layers | validation gap closed | best pass |
|---|---|---|---|---|---|
| 2e-3 | 1e-5 | 0 | 256, 256 | 19.2% | 2 |
| 5e-4 | 1e-5 | 0 | 256, 256 | 20.4% | 5 |
| 5e-4 | 1e-4 | 0 | 256, 256 | 22.6% | 16 |
| 5e-4 | 3e-4 | 0 | 256, 256 | 25.1% | 29 |
| 5e-4 | 1e-3 | 0 | 256, 256 | 23.3% | 51 |
| 5e-4 | 3e-3 | 0 | 256, 256 | 15.2% | 2 |
| 5e-4 | 1e-4 | 0.1 | 256, 256 | 25.7% | 32 |
| 5e-4 | 1e-4 | 0.3 | 256, 256 | 25.7% | 51 |
| 5e-4 | 1e-5 | 0 | 128, 128 | 20.6% | 17 |
| 5e-4 | 1e-5 | 0 | 512, 512 | 19.9% | 2 |
| 1e-3 | 1e-3 | 0 | 512, 512 | 23.0% | 29 |
| 2e-4 | 1e-5 | 0 | 256, 256, 256 | 19.9% | 7 |

The first run overfit within two passes; weight decay and dropout fixed most of it.  `model.BEST` is the dropout 0.1
row.  It has not been scored on any test seasons.

**The ceiling**, on a 5-season, 200-player league: 32.0% of the gap overall, from 4.7% for 3P% to 53.1% for blocks.
The network's 25.7% is against that, not against 100%.

**The baseline's k per stat**, 15-season league: 3PA 526, 2PA 807, FTA 1,136, ORB 1,283, DRB 990, AST 772,
TOV 1,792, STL 2,349, BLK 1,561, PF 1,872 possessions; 3P% 278, 2P% 266, FT% 133 attempts.

## Next steps, in order

1. `python run_v2.py 0 5`: 30 simulated seasons, 0-24 train, 25-27 validate, 28-29 test, one network, an ensemble of
   five and the ceiling.  About 30 minutes on 2 cores.  Then `python run_v2.py 1 5`, the same on a second simulated
   league, as the noise control.  It prints gap closed and error reduction per stat, the spread over network seeds,
   and gap closed by career possessions before the game.
2. **Where the remaining gap sits.**  In a 2-pass smoke test of `run_v2.py` on a 6-season league the ceiling gained
   most on players with 6,000+ career possessions, where the network gained least; check whether that holds once the
   network is fully trained.  In the first run STL, PF and 3P% barely moved.
3. **Real seasons.**  Build the same arrays from the game logs in the data layer: player, season, game, and per game
   the 13 counts from `config.yaml`'s box counts (3PA = fg3m + fg3_miss, its makes fg3m, and so on) with each stat's
   own possessions: offensive possessions for the offensive counts, defensive for the defensive ones, as
   `box_rates.py` uses.  The simulator gives every stat the same possessions; `baseline.exposures` is the one place
   to change.  Score by NLL on held-out seasons, per stat, against the per-stat recursion: real data has no truth.
4. **Into the ratings.**  The season panel's padded rates are the box-score prior's inputs; the network's
   predictions could replace them.  Three constraints from the repo: the current season is never trained on; model
   code does not load its own data; and whole-season inputs leak into the within-season folds (the z of -31 in
   `notes/HANDOFF.md`), so a season's inputs must be built out of fold.

## Traps already met

- The 15-season league's test seasons, 13-14 of seed 0, were scored once, in the first run.  Everything chosen since
  used the validation seasons.  The 30-season league's test seasons have never been scored.
- The ceiling knows each player's age; the network knows only his experience.  The ceiling overstates what a model
  of his own stats can reach.
- Simulated gains depend on how much the simulator ties the stats together: a true-talent correlation of 0.39
  between 3P% and FT%, and one size factor behind ORB, DRB, BLK and 3PA.  Only real seasons can say how much of the
  gain is real.
- `inputs.py` builds generated-code-free numba kernels; a name clash between a loop variable and an argument cost an
  hour twice in this work.  The leakage check (`python inputs.py`) edits one game's outcome and confirms that game's
  own inputs do not change.

## Dependencies

numpy and scipy are in the repo's dependencies; numba and jax are not (`pip install numba jax`).  JAX installs CPU
wheels on Windows; if it does not, run under WSL, or port the 40-line training loop in `model.py` to PyTorch.  JAX
uses every core while training.  Prediction alone needs only numpy.

Measured here, on 2 cores, for the 15-season league: simulating 8 seconds with calibration, the baseline's k 4
seconds, the inputs 3 seconds, one network with `BEST` about 3 minutes, the ceiling 15 seconds.
