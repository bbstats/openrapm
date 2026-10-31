"""One-off edit for the bug-fix pass of 2026-10-30: README.md and CONTRIBUTING.md name today's incumbent and today's
rule (kept for the record)."""
from pathlib import Path


def edit(path: str, pairs: list) -> None:
    p = Path(path)
    s = p.read_text(encoding="utf-8")
    for old, new in pairs:
        assert s.count(old) == 1, (path, old[:80], s.count(old))
        s = s.replace(old, new)
    p.write_text(s, encoding="utf-8", newline="\n")


edit("README.md", [
    ("python -m openrapm build --folds mine_within\n",
     "python -m openrapm build --folds mine_within --extrapolate-folds=\n"),
    ("""   season, and stores the misses on the quarter held back: about 8 minutes. Rebuilt from the same data,
   these folds give step 4 exactly the multipliers the shipped folds give (`weightfix_within`, the
   default of `--folds`).""",
     """   season, and stores the misses on the quarter held back: about 8 minutes. The shipped folds cover every
   season, `creation110_early` for 1997-2016 and `creation110_within` for 2017-2026 (the default of
   `--folds`), and each multiplier is read to a full season with folds holding back a tenth of the games,
   `creation110_k10` (the default of `--extrapolate-folds`)."""),
    ("""   `build --setting test --folds mine_within --name weightfix` (874 seconds in the last run) writes the
   table the year-over-year test scores, under the incumbent's name, so the chain and the checks find it.
   Without `--folds`, `build` and `shrink` read `outputs/within/weightfix_within/`, which a fresh build
   does not have; they stop at step 4 with "no folds in ...", after steps 1-3 have run.""",
     """   `build --setting test --folds mine_within --extrapolate-folds= --name creation110` writes the table the
   tests score, under the incumbent's name, so the chain and the checks find it. Without `--folds`, `build`
   and `shrink` read the shipped folds, which a fresh build does not have; they stop at step 4 with "no
   folds in ...", after steps 1-3 have run."""),
    ("""## The test that decides, and how to run an experiment

**The year-over-year test** (`scripts/checks/63_yoy.py`): rate a season from its own games, then predict""",
     """## The test that decides, and how to run an experiment

**The within-season WOWY test decides** (`scripts/checks/151_wowy.py`; the owner, 2026-10-15). Each season
is rated from three quarters of its games, twelve times over, and scored stint by stint on the quarter held
back, with each team's own level in those games taken out, so what is scored is how the ratings split credit
among teammates. The paired z is the mean of the 30 season-by-season differences in error, candidate minus
incumbent, over its standard error. A change must be better in every era, 1997-2006, 2007-2016 and
2017-2026, read beside the noise control: today's ratings rebuilt with only the box-score prior's random split
of players changed, which moves this test by about 0.01 on its own.

**The year-over-year test guards it** (`scripts/checks/63_yoy.py`): rate a season from its own games, then predict"""),
    ("""observations. The incumbent, `outputs/season_ratings_weightfix.parquet` (experiment 46), reads
**8.5857**. The tables it scores are built at the test setting (`openrapm build --setting test`), so
nothing fitted for a season has seen the two seasons that score it.""",
     """observations. The incumbent, `outputs/season_ratings_creation110.parquet` (experiment 110), reads
**8.6943**. The tables it scores are built at the test setting (`openrapm build --setting test`), so
nothing fitted for a season has seen the two seasons that score it."""),
    ("""A change is adopted when all four hold; ties go to the simpler version.

- Its paired z is -2 or below. A bug fix, or a fix that removes a bias the held-out audit
  (`scripts/checks/138_heldout_audit.py`) counts, needs only a tie: z between -2 and +2
  (`notes/DECISIONS.md`, "The Robustness pass").""",
     """A change is adopted when all four hold, on the owner's word; ties go to the simpler version.

- The within-season WOWY test is better in every era, against the noise control's own movement. A bug fix
  needs only a tie.
- The year-over-year test is not worse at a paired z of +2."""),
    ("""- The trade loss is not clearly worse. There is no cutoff: experiment 33b shipped with its defensive trade
  loss worse at z +5.5, a difference of 0.0007 on a 0.248 scale.
- The 2026 top 20""",
     """- The 2026 top 20"""),
    ("""python -m openrapm build --setting test --folds mine_within --name weightfix
python scripts/checks/70_tradeset.py --rankings=outputs/season_ratings_weightfix.parquet --team_effects=team --out=tradeset_weightfix
INC_FOLDS=mine_within bash scripts/experiment_chain.sh NAME DEFENSIVE_TARGET""",
     """python -m openrapm build --setting test --folds mine_within --extrapolate-folds= --name creation110
INC_FOLDS=mine_within INC_SYSTEM=mine bash scripts/experiment_chain.sh NAME DEFENSIVE_TARGET"""),
    ("""The first writes the incumbent; the second writes its trade-set corrections, which every chain's trade
loss is paired against. On the project's machine both exist and `INC_FOLDS` can be left out.""",
     """The first writes the incumbent. On the project's machine it exists, with its folds and its noise
control, and `INC_FOLDS` and `INC_SYSTEM` can be left out."""),
    ("""The second line is the noise control: the incumbent rebuilt with only the players' assignment to the
prior's five folds changed, which shows how far the readings move when nothing real has changed. The
chain builds the candidate at the test setting, then reads it against the incumbent:

- the year-over-year test (`63_yoy.py`);
- the consensus: rank agreement with a blend of public metrics (`64_consensus_report.py`);
- the 2026 top 20, side by side, as an HTML page (`66_compare.py`);
- the swap test: does it order teammates the way their lineup swaps do (`90_swap_test.py`);
- the year-over-year difference by player-quality tier: top 30, 31-90, 91-150, 151-300, 301+
  (`88_yoy_by_player.py`);
- the trade loss: how much a player's comings and goings in the seasons either side correct his rating
  (`70_tradeset.py`, then `73_tradeloss.py`).""",
     """The second line is the noise control: the incumbent rebuilt with only the players' assignment to the
prior's five folds changed, which shows how far the readings move when nothing real has changed. The
chain builds the candidate at the test setting, its own folds for every season, then reads it against the
incumbent:

- the within-season WOWY test, by era, with the noise control beside it (`151_wowy.py`);
- the year-over-year test (`63_yoy.py`);
- the 2026 top 20, side by side, as an HTML page (`66_compare.py`).

`FULL_READINGS=1` adds the older readings: the consensus (`64_consensus_report.py`), the swap test
(`90_swap_test.py`), the year-over-year test by player-quality tier (`88_yoy_by_player.py`) and the trade
loss (`70_tradeset.py`, then `73_tradeloss.py`). A name is built once: the chain stops on a name whose
files exist, unless `REUSE=1` says to go on from them."""),
    ("""the last line names every reading that failed. The chain does not log the experiment: afterwards,
`scripts/checks/101_log_run.py --name=NAME --exp=<number> --incumbent=weightfix ...` adds its row to""",
     """the last line names every reading that failed. The chain does not log the experiment: afterwards,
`scripts/checks/101_log_run.py --name=NAME --exp=<number> ...` adds its row to"""),
])

edit("CONTRIBUTING.md", [
    ("""**The year-over-year test decides.** Build the candidate at the test setting and score it with
`scripts/checks/63_yoy.py` against the incumbent, `outputs/season_ratings_weightfix.parquet`, which reads
8.5857 points per 100 per team-game (`README.md` defines the score). A change is adopted at a paired z of -2
or below: z is the mean of the 56 season-by-season differences in team-game mean squared error (28 scored
seasons, each predicted from the season before and the season after) divided by its standard error, a paired
t-statistic. Ties go to the simpler and faster version, every time.""",
     """**The within-season WOWY test decides** (`scripts/checks/151_wowy.py`, `README.md` defines it). Build the
candidate with `scripts/experiment_chain.sh`, which scores it against the incumbent, experiment 110
(`creation110`), with the noise control `noise_creation110` beside it. A change must be better in every era,
1997-2006, 2007-2016 and 2017-2026; z is the mean of the 30 season-by-season differences in stint error over
its standard error, a paired t-statistic. **The year-over-year test guards it** (`scripts/checks/63_yoy.py`):
not worse at a paired z of +2, the mean of the 56 season-by-season differences in team-game mean squared error
(28 scored seasons, each predicted from the season before and the season after) over its standard error. Ties go
to the simpler and faster version, every time."""),
    ("""- **the trade loss** (`scripts/checks/70_tradeset.py`, then `73_tradeloss.py`): it counts every player once,
  so it sees the bench players the year-over-year test, an error per team-game, barely weighs.
""", ""),
    ("""   On a fresh build, make the incumbent's files first (`README.md`, "The test that decides") and run every
   chain with `INC_FOLDS=mine_within`.""",
     """   On a fresh build, make the incumbent's files first (`README.md`, "The test that decides") and run every
   chain with `INC_FOLDS=mine_within INC_SYSTEM=mine`."""),
    ("""   python scripts/checks/135_robust_review.py --cands=NAME --controls=noise_control --ref=weightfix --out=robust_NAME""",
     """   python scripts/checks/135_robust_review.py --cands=NAME --controls=noise_creation110 --ref=creation110 --out=robust_NAME"""),
    ("""   python scripts/checks/101_log_run.py --name=NAME --exp=<number> --incumbent=weightfix --idea=owner --verdict=pending --changed="one plain sentence\"""",
     """   python scripts/checks/101_log_run.py --name=NAME --exp=<number> --idea=owner --verdict=pending --changed="one plain sentence\""""),
])
print("ok")
