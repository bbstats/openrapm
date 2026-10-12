# Contributing

The project is a test with a model attached. If your change predicts games it never saw better than what
ships, and none of the checks below vetoes it, it is adopted, whoever wrote it and however strange it looks.
The job of this document is to make that claim cheap for you to make and cheap for a maintainer to believe.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate     # .venv\Scripts\activate on Windows
pip install -e ".[dev]"
pytest
ruff check .
```

That passes on a fresh clone with no data (`README.md` gives the counts). The dev extra installs ruff 0.17.0,
the version CI pins, so `ruff check .` reads the code the way CI does; without the dev extra,
`uvx ruff@0.17.0 check .` runs the same version (uv's `uvx` fetches it once and keeps it). You do not need the scraped
play-by-play to change the model, run the unit tests or open a draft pull request; only to run the test that
decides. `README.md`, "Building the ratings from scratch", builds the data.

## What a change has to clear

**The year-over-year test decides.** Build the candidate at the test setting and score it with
`scripts/checks/63_yoy.py` against the incumbent, `outputs/season_ratings_weightfix.parquet`, which reads
8.5857 points per 100 per team-game (`README.md` defines the score). A change is adopted at a paired z of -2
or below: z is the mean of the 56 season-by-season differences in team-game mean squared error (28 scored
seasons, each predicted from the season before and the season after) divided by its standard error, a paired
t-statistic. Ties go to the simpler and faster version, every time.

**Three readings can veto it**, and none of them is ever fitted to:

- **the consensus** (`scripts/checks/64_consensus_report.py`): rank agreement with a blend of public metrics.
  A gross miss vetoes; a marginal one does not (0.759 against the minimum of the time, 0.76, is noise on a
  sanity check; the minimum is now 0.75, `MIN_RANK_AGREEMENT` in `src/openrapm/consensus.py`).
  Never choose a constant because it clears a minimum in `tests/test_vs_consensus.py`.
- **the trade loss** (`scripts/checks/70_tradeset.py`, then `73_tradeloss.py`): it counts every player once,
  so it sees the bench players the year-over-year test, an error per team-game, barely weighs.
- **the 2026 top 20** (`scripts/checks/66_compare.py`): the latest season's top of the ratings, side by side
  with the incumbent's.

**Report every variant you tried, not just the one that won.** Try forty things against the same 28 seasons
and one of them beats the incumbent on noise alone. A pull request that tried one thing and won is easy to
believe; one that tried forty and reports the best is not, unless the other thirty-nine are in it too.

**Read the pattern, not one number.** The owner (2026-10-08): "single player small moves shouldnt concern us
but - systemic improvements should be seen positive". `scripts/checks/135_robust_review.py` cuts every test
into slices (directions, eras, sides, quality tiers, players who changed teams, ages) and counts how many move each way, beside
the same counts for the noise control (the incumbent rebuilt with only the prior's player folds reassigned).
Report movement in points per 100, never in rank places; under 0.1 per 100 is nothing.

## How to run an experiment

1. Make the change in the model layer (`src/openrapm/`, outside `data/` and `commands/`). If it is a new
   setting, its default goes in `src/openrapm/settings.py` and it reaches the command line as an option of
   `openrapm rate`; `--defensive-target`, `--features`, `--booster-params` and `--fold-assignment` are there
   already.
2. Run the chain. It builds the candidate at the test setting (steps 1-5, all thirty seasons) and takes every
   standard reading against the incumbent:

   ```bash
   bash scripts/experiment_chain.sh NAME DEFENSIVE_TARGET ["EXTRA RATE FLAGS"]
   bash scripts/experiment_chain.sh all_threes points_with_expected_opponent_threes_all
   ```

   The second line is a target experiment: the defence ridge explains points with every opponent three
   repriced at the shooter's padded three-point percentage, instead of three quarters of them.

   A change to the box-score prior changes how far its part of a rating holds up, so give it its own
   within-season folds: `OWN_FOLDS=1` saves the candidate's boosters, builds its folds for 2017-2026, fits
   step 4 on them, keeps a second table on the incumbent's multipliers beside it, and runs the held-out audit
   (`scripts/checks/138_heldout_audit.py`: on held-back games, are players high on some input rated too high
   or too low). `FIRST_PASS=1` does only 2017-2026, its folds and the audit, and stops. The top of
   `scripts/experiment_chain.sh` lists the other switches.

   On a fresh build, make the incumbent's files first (`README.md`, "The test that decides") and run every
   chain with `INC_FOLDS=mine_within`. `OWN_FOLDS=1` then still stops at the audit: it reads a frozen
   threshold file, `outputs/audit_lgb_noonc_2017_thresholds.json`, that no documented command writes.
3. Read `outputs/NAME_chain.log`: the year-over-year test, the consensus, the 2026 top 20 (an HTML page), the
   swap test, the quality tiers and the trade loss. A reading that fails is logged and the chain goes on; the
   last line names every reading that failed.
4. Read the slices against the noise control:

   ```bash
   python scripts/checks/135_robust_review.py --cands=NAME --controls=noise_control --ref=weightfix --out=robust_NAME
   ```

5. Add the experiment's row to `experiments/runs.csv` (every number is read from the chain's outputs):

   ```bash
   python scripts/checks/101_log_run.py --name=NAME --exp=<number> --incumbent=weightfix --idea=owner --verdict=pending --changed="one plain sentence"
   ```

A candidate is never scored on its step-3 table alone: the incumbent has steps 4 and 5, so a candidate
without them is not compared like for like.

## The refactor rule

A change that should not move a number has to show that it moves none. Build the table before the change and
after it, with the same setting and seasons, and compare:

```bash
python -m openrapm build --name before                  # on the commit before the change
python -m openrapm build --name after                   # with the change
python -m openrapm compare outputs/season_ratings_before.parquet outputs/season_ratings_after.parquet
```

`openrapm compare` must print IDENTICAL-BYTES, for the `_step3` and `_step4` tables as well, and for
`--setting test`. A quicker first look is one season: `python -m openrapm rate --seasons 2026 --name before`,
the same with `--name after`, then `openrapm compare` on the two `_step3` tables. Every command holds the
linear-algebra libraries to one thread (unless the environment already sets `OMP_NUM_THREADS`,
`MKL_NUM_THREADS` or `OPENBLAS_NUM_THREADS`), so the same inputs give the same bits on any machine. One build at a
time: two at once share the processor and each takes far longer.

## Code rules

- **Complexity at most 10** for every function (McCabe, ruff's C901; tests are exempt from that rule only).
  `ruff check .` must be clean; CI runs it.
- **The layer rule.** The model layer (`src/openrapm/`, outside `data/` and `commands/`) is handed data and
  returns numbers: it may not open a file or import the data layer. `src/openrapm/data/` reads the disk and the
  network and may not import the commands. `src/openrapm/commands/` may import anything.
  `tests/test_layer_boundary.py` enforces it.
- **No `# noqa`**, no mutable module globals, no `global`, no `sys.path` changes, no importing one script from
  another.
- **Settings live in `src/openrapm/settings.py`**, each beside the experiment that set it. An experiment-only
  branch is deleted when the experiment ends: the shipped values are the defaults.
- **Comments say why**: the decision and the experiment that made it ("notes/DECISIONS.md, experiment 46"),
  never what the next line does.
- **Names are descriptive** (`gram`, `right_hand_side`, `penalty`, `prior_scale`), never single letters outside a
  defined equation. Identifiers keep the tables' spelling (`offense`, `defense`, `rating_off`); prose is British
  (offence, defence, centre).
- **A step module opens with** what the step is for, the words it uses, the equation it implements and an
  example with numbers; functions appear in the order they run.
- **Docs are tested.** `tests/test_config.py` fails when a doc names a script or command that does not exist,
  or a flag no parser declares.

## The writing rules

The owner wants the code to "almost read like a tutorial" and a repo people feel at home in. These rules bind
code, comments and docs:

- Never "board": say "the player ratings" or "the ratings table". Never "floor" for a threshold: say "minimum",
  "cap", or name the check. Never "thin" or "thick" players: say "a player with few possessions", "bench
  player", "starter".
- No invented terms or metaphors. Say the literal thing: "players are sorted by label and assigned to folds
  0,1,2,3,4,4,3,2,1,0,... so every fold gets the same mix of good and bad players".
- Every term is defined where it first appears. Statistics get their exact names (standard deviation, standard
  error, paired t-statistic, weighted least squares, ridge penalty).
- Short concrete examples beat paragraphs: "a 200-possession bench player's steals per 100 are mostly noise, so
  the rate is padded toward the league: (200 x 2.1 + 450 x 1.4) / (200 + 450) = 1.6".
- Movement is reported in points per 100, never in rank places.
- No "we" in text a reader sees. On the site the ratings are "OpenRAPM", and the published pages are American
  English.
- No drafted prose on the published site: `docs/*.html` is not edited, and the owner writes any wording it
  needs (the owner called a drafted footer "AI slop").

## Read the traps before trusting a number

`notes/DECISIONS.md`, "The measurement traps", lists the specific ways this pipeline has produced confident
wrong numbers: benchmarks that shared the estimate's blind spot, an argmax sitting on the edge of its grid, a
feature that predicted how well-*measured* a row was rather than how good the player was. Every one of them
survived review at the time. They will cost you a day each if you meet them fresh.

**`FINDINGS N.N` in a comment is not a dead link.** `FINDINGS.md` was removed from the tree in `0f7c0b2` and
kept whole under the tag `archive/research-2026-09`: `git show archive/research-2026-09:FINDINGS.md`.

## Things that will not be merged

- A change that wins only on the seasons it was tuned on. If a setting was chosen by trying values on some
  seasons, show that it also wins on seasons that played no part in choosing it.
- A constant chosen because it cleared a consensus minimum.
- A candidate scored without steps 4 and 5.
- Anything fitted on a season whose Finals are not over. `tests/test_no_current_season.py` fails when a
  change does this; do not rely on it to catch you.

## Reporting a bug

If a number looks wrong, say which number, what you expected and what command produced it. If it is a rating
that looks wrong, name the player and the season: that is usually faster to chase than a summary statistic.
`scripts/checks/113_swap_explain.py --base=season_ratings_product_step4 --team=BOS --player="Derrick White"`
rebuilds one team's step 5 player by player, from the step-4 table `openrapm build` writes: the way to see why
one rating moved.
