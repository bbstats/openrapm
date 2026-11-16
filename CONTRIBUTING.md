# Contributing

The project is a test with a model attached. If your change predicts games it never saw better than what ships, and
none of the checks below vetoes it, it is adopted, whoever wrote it and however strange it looks. The job of this
document is to make that claim cheap for you to make and cheap for a maintainer to believe.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate     # .venv\Scripts\activate on Windows
pip install -e ".[dev]"
pytest
ruff check .
```

That passes on a fresh clone with no data. The dev extra installs ruff 0.17.0, the version CI pins, so `ruff check .`
reads the code the way CI does; without the dev extra, `uvx ruff@0.17.0 check .` runs the same version. A few one-off
checks need Optuna: `pip install -e ".[checks]"`. You do not need the scraped play-by-play to change the model, run
the unit tests or open a draft pull request; only to run the test that decides. `README.md`, "Building the ratings",
builds the data.

## What a change has to clear

**The within-season WOWY test decides** (`scripts/checks/151_wowy.py`, `README.md` defines it): better in every era,
1997-2006, 2007-2016 and 2017-2026, read beside the noise control. **The year-over-year test guards it**
(`scripts/checks/63_yoy.py`): not worse at a paired z of +2, each side read with its spread matched to the incumbent's.
**The 2026 top 20** (`scripts/checks/66_compare.py`), side by side with the incumbent's, must not be worse in the
owner's reading. Ties go to the simpler and faster version, every time.

The incumbent and its noise control are the defaults `INC_NAME` and `CONTROL_NAME` at the top of
`scripts/experiment_chain.sh`; the chain scores every candidate against them.

**The consensus can veto, and is never fitted to** (`scripts/checks/64_consensus_report.py`, run with
`FULL_READINGS=1`): rank agreement with a blend of public metrics. A gross miss vetoes; a marginal one does not. The
minimum is `MIN_RANK_AGREEMENT` in `src/openrapm/consensus.py`. Never choose a constant because it clears a minimum
in `tests/test_vs_consensus.py`.

**Report every variant you tried, not just the one that won.** Try forty things against the same 30 seasons and one
of them beats the incumbent on noise alone. A pull request that tried one thing and won is easy to believe; one that
tried forty and reports the best is not, unless the other thirty-nine are in it too.

**Read the pattern, not one number.** The owner, 2026-10-08: "single player small moves shouldnt concern us but -
systemic improvements should be seen positive". `scripts/checks/135_robust_review.py` cuts every test into slices,
directions, eras, sides, quality tiers, players who changed teams, ages, and counts how many move each way, beside
the same counts for the noise control. Report movement in points per 100, never in rank places; under 0.1 per 100 is
nothing.

## How to run an experiment

1. Make the change in the model layer, `src/openrapm/` outside `data/` and `commands/`. If it is a new setting, its
   default goes in `src/openrapm/settings.py` and it reaches the command line as an option of `openrapm rate`.
2. Run the chain. It builds the candidate at the test setting for all thirty seasons, saves its boosters, builds its
   own held-back games for 1997-2016 and 2017-2026, fits the shrink on them, and takes the three readings against the
   incumbent:

   ```bash
   bash scripts/experiment_chain.sh NAME DEFENSIVE_TARGET ["EXTRA RATE FLAGS"]
   bash scripts/experiment_chain.sh halflife4 points_with_expected_opponent_threes_late_as_scored "--second-prior-halflife 4"
   ```

   The second line is an experiment that changes one setting: Box Adj's training seasons fade at a half-life of 4
   seasons instead of 8. The top of `scripts/experiment_chain.sh` lists every switch. `FULL_READINGS=1` adds the
   consensus, the swap test, the quality tiers, the trade loss, the held-out audit and a second table on the
   incumbent's multipliers. A name is built once: the chain stops on a name whose files exist, unless `REUSE=1` says
   to go on from them. It waits while the morning job holds its lock and between 5:50 and 7:30 AM.

   On a fresh build, make the incumbent's files first, `README.md`, "Building the ratings", and name them on every
   chain: `INC_NAME=mine INC_SYSTEM=mine INC_FOLDS=mine_early,mine_within CONTROL_NAME=...`. The held-out audit reads
   a frozen threshold file, `outputs/audit_lgb_noonc_2017_thresholds.json`, that no documented command writes.
3. Read `outputs/NAME_chain.log`. A reading that fails is logged and the chain goes on; the last line names every
   reading that failed.
4. Read the slices against the noise control:

   ```bash
   python scripts/checks/135_robust_review.py --cands=NAME --controls=noise_fade141 --ref=fade141 --out=robust_NAME
   ```

5. Add the experiment's row to `experiments/runs.csv`; every number is read from the chain's outputs:

   ```bash
   python scripts/checks/101_log_run.py --name=NAME --exp=<number> --idea=owner --verdict=pending --changed="one plain sentence"
   ```

A candidate is never scored on its step-3 table alone: the incumbent has the shrink, so a candidate without it is not
compared like for like.

## The refactor rule

A change that should not move a number has to show that it moves none. Build the tables before the change and after
it, with the same setting and seasons, and compare:

```bash
python -m openrapm build --name before                  # on the commit before the change
python -m openrapm build --name after                   # with the change
python -m openrapm compare outputs/season_ratings_before.parquet outputs/season_ratings_after.parquet
```

`openrapm compare` must print IDENTICAL-BYTES, for the `_step3` and `_step4` tables as well, and for `--setting test`.
Compare two builds, never a build with a file built weeks ago: a finished season's stints differ from a rebuild in the
last bits of the expected-makes columns. A quicker first look is one season: `python -m openrapm rate --seasons 2026
--name before`, the same with `--name after`, then `openrapm compare` on the two `_step3` tables. Every command holds
the linear-algebra libraries to one thread, unless the environment already sets `OMP_NUM_THREADS`, `MKL_NUM_THREADS`
or `OPENBLAS_NUM_THREADS`, so the same inputs give the same bits on any machine.

Several processes at once are fine while they fit the machine's memory: `--jobs` counts about 5.5 GB a process and
starts no more than fit. When a chain is stopped, stop its python children too.

## Code rules

- **Complexity at most 10** for every function, McCabe, ruff's C901; tests are exempt from that rule only.
  `ruff check .` must be clean; CI runs it.
- **The layer rule.** The model layer, `src/openrapm/` outside `data/` and `commands/`, is handed data and returns
  numbers: it may not open a file or import the data layer. `src/openrapm/data/` reads the disk and the network and
  may not import the commands. `src/openrapm/commands/` may import anything. `tests/test_layer_boundary.py` enforces
  it.
- **No `# noqa`, no `global`, no mutable module globals, no `sys.path` changes, no importing one script from
  another.** The exceptions are named in `tests/test_code_rules.py`, which fails when one is added: the checks that
  pin the thread variables before numpy loads carry `# noqa: E402` on their imports; the screens hand their workers'
  state over through `global`; the Power checks put `src/` on the path; and the checks that reuse another check's
  code, mostly check 151's scoring, load it by path. The package, the tests and the tutorial keep all four rules.
- **Settings live in `src/openrapm/settings.py`**, each beside the experiment that set it. An experiment-only branch
  is deleted when the experiment ends: the shipped values are the defaults. Code that lost its test is removed and
  kept readable at an archive tag, as `archive/research-2026-11` keeps the three box-score priors that lost.
- **Comments say why**: the decision and the experiment that made it, "notes/DECISIONS.md, experiment 46", never what
  the next line does, and never the list of every value a setting has had; the record keeps that.
- **Names are descriptive**: `gram`, `right_hand_side`, `penalty`, `prior_scale`, never single letters outside a
  defined equation. Identifiers keep the spelling the code already uses, `rating_off`, `defence_factors`,
  `context_types_team_zones_centred`; new prose is American English.
- **A step module opens with** what the step is for, the words it uses, the equation it implements and an example
  with numbers; functions appear in the order they run.
- **Docs are tested.** `tests/test_config.py` fails when a doc names a script or command that does not exist, or a
  flag no parser declares; `tests/test_docs.py` when a doc leaves out a check or a module, or names a setting's old
  value.

## The writing rules

The owner wants the code to "almost read like a tutorial" and a repo people feel at home in. These rules bind code,
comments, docs and every message to the owner:

- American English: defense, offense, center, color, analyze. Code names and file names already in the repository
  keep their spelling.
- Never "board": say "the player ratings" or "the ratings table". Never "floor" for a threshold: say "minimum", "cap",
  or name the check. Never "thin" or "thick" players: say "a player with few possessions", "bench player",
  "starter". Never "the decider" or "the guard": say "the within-season WOWY test" and "the year-over-year test".
- No invented terms or metaphors. Say the literal thing: "players are sorted by label and assigned to folds
  0,1,2,3,4,4,3,2,1,0,... so every fold gets the same mix of good and bad players".
- Every term is defined where it first appears. Statistics get their exact names: standard deviation, standard
  error, paired t-statistic, weighted least squares, ridge penalty.
- Short concrete examples beat paragraphs: "a 200-possession bench player's steals per 100 are mostly noise, so the
  rate is padded toward the league: (200 x 2.1 + 450 x 1.4) / (200 + 450) = 1.6".
- Movement is reported in points per 100, never in rank places.
- No "we" in text a reader sees. On the site the ratings are "OpenRAPM".
- The site's wording is the owner's. A change to a page adds data, a column or a control; any sentence on a page is
  the owner's, word for word.

## Read the traps before trusting a number

`notes/DECISIONS.md`, "The measurement traps", lists the specific ways this pipeline has produced confident wrong
numbers: benchmarks that shared the estimate's blind spot, an argmax sitting on the edge of its grid, a feature that
predicted how well-*measured* a row was rather than how good the player was. Every one of them survived review at the
time. They will cost you a day each if you meet them fresh.

**`FINDINGS N.N` in a comment is not a dead link.** `FINDINGS.md` was removed from the tree in `0f7c0b2` and kept
whole under the tag `archive/research-2026-09`: `git show archive/research-2026-09:FINDINGS.md`.

## Things that will not be merged

- A change that wins only on the seasons it was tuned on. If a setting was chosen by trying values on some seasons,
  show that it also wins on seasons that played no part in choosing it.
- A constant chosen because it cleared a consensus minimum.
- A candidate scored without the shrink.
- Anything fitted on a season whose Finals are not over. `tests/test_no_current_season.py` fails when a change does
  this; do not rely on it to catch you.

## Reporting a bug

If a number looks wrong, say which number, what you expected and what command produced it. If it is a rating that
looks wrong, name the player and the season: that is usually faster to chase than a summary statistic. The Detail
view on the site splits any published rating into its parts; `python -m openrapm site --openrapm-decomposition`
rebuilds that split from the tables.

## Using the code and the ratings

The code is MIT licensed. The published ratings carry their own terms, `LICENSE-RATINGS.md`: use and quote them with
credit to OpenRAPM by Nathan Walker and a link to openrapm.com; no bulk copying, no republishing the table, no AI
crawling.
