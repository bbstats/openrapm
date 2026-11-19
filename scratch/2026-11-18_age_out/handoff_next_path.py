"""notes/HANDOFF.md rewritten at its top for the next research path (the owner, 2026-11-19: "write a handoff for next
research path please"): everything above "## Waiting on the owner" is replaced, and that list brought to the day.

    .venv/Scripts/python -I scratch/2026-11-18_age_out/handoff_next_path.py
"""
from pathlib import Path

PLAN = r"C:\Users\Nathan\.claude\plans\what-are-some-possible-zippy-quill.md"

TOP = f"""# Handoff

**What this file is.**  Where the work stands, the next piece of work in full, what waits on the owner, and the
commands a session runs.  Its top was rewritten on 2026-11-19 for the next research path; the file before it is in git
at a77b019.  `DECISIONS.md` is the permanent record and carries every number quoted here; read its measurement traps
before trusting a number.  `README.md` says what OpenRAPM is, `PIPELINE.md` how each part works, `GLOSSARY.md` what
each word means, `CONTRIBUTING.md` the writing rules, `CLAUDE.md` the owner's standing rules.

## Talking to the owner

- American English in everything shared: defense, offense, color.  Code names keep their spelling.
- No step numbers in messages.  The names: **box-score prior** (the LightGBM); **Box Adj** (the line on his teammates'
  and opponents' priors and his team's shooting with him on and off the court, added before the games are weighed);
  **+/-** (what the season's games add on top of the prior); **the shrink** (both parts multiplied by what holds up on
  held-back games); **defense factor** (the shrink's extra factor on defense, 0.8682 before 2017 and 0.8704 from 2017;
  never "dial").  Define each the first time it appears.
- Experiments by what they changed, never by number or code name; numbers only in the record files.
- Answer exactly what is asked.  Results as tables, one quantity per column, no parentheses, one TL;DR sentence.
- Within-season WOWY is what counts, and a win must hold in every era: 1997-2006, 2007-2016, 2017-2026.
- **No jargon ever.**  "how the box-score prior splits players into its five training groups", never "folds";
  "10 averaged versions, each trained on a random resample of players", never "bags"; "random draws", never "seeds";
  "full build", never "chain"; no formulas or letters.
- Decisions go through the question tool, with the numbers in the question and one outcome per option.  The owner
  does not read every message: a question stands alone.
- I write no prose for the site.  Wording is the owner's; a page change of mine adds data, a column, or a line the
  owner has already approved elsewhere.
- **Age is not an input to any rating** (the owner, 2026-11-18: "I want age removed!").  Never offer age, an age
  curve or age bands to the prior, to Box Adj, to a screen's candidates or to a tree stage.
- The swap adjustment is off and the owner does not care about it (2026-11-19: "i dont care about the swap adjustment
  if we're not using it").  Do not raise it.  Box Adj is the swap adjustment's lesson moved into the prior.
- Before asking for a verdict, compare every season's top-rated player and WOSA leader in both tables, not 2026's
  alone: on 2026-11-19 two old seasons' WOSA leaders changed and the owner had answered on 2026.

## Start here: the next research path, the reads on Box Adj

**Approved by the owner on 2026-11-18** ("box adj is good ... the rest is fine"), not started.  The plan with the
whole ranked list: `{PLAN}`; the record:
DECISIONS.md, "The research paths of 2026-11-18".  Four reads, all from saved held-back games, no rating built, 3 to 4
hours.  **Report and stop**: a read that passes goes to the owner as a candidate for a full build, through the
question tool; nothing is built on a read alone.

**Why Box Adj.**  It is the only part of a rating fitted on the within-season test, about 45% of a regular's rating,
and the last three gains came from it: its weights fitted on that test, 0.488; the team's shots by five zones, 0.183;
its training seasons faded by distance, 0.133.  The season's games take back 57% to 60% of a prior change that moves a
team's total and 12% to 14% of one that moves credit between teammates, so only the second kind is worth trying, and
Box Adj holds team totals by construction.

**The rule, registered before any read** (experiment 140's, used for the zones and the fade): a read goes forward when
it is better than today's line in every era, 1997-2006, 2007-2016 and 2017-2026, and by more than 0.042 over the 30
seasons, with its shuffled control at nothing.  Between two that tie, the simpler one.  A read from saved sums gives the
sign in each era and overstates the size: the fade read 0.207 and built 0.133; age out read 0.031 and built 0.027.
Register each read in DECISIONS.md before its result, as "Experiment 142, read 1" was.

**What is on disk for it.**

| what | where |
|---|---|
| the held-back games the line is fitted on, rated without Box Adj | `outputs/within/rebuilt_base_early`, `rebuilt_base_within`; saved models `outputs/prior_models_rebuilt_base.pkl` |
| each of those sets' players, 239 columns rebuilt from its own games | `players_<season>_r<repeat>_f<fold>.parquet` inside those folders |
| today's line: its sums and each set's inputs | `outputs/second_prior/noage142_line/sums_<season>.npz`, `inputs_<season>.parquet`, `run.json` |
| the same with age, the line before it | `outputs/second_prior/rebuilt_line` |
| the incumbent's test tables, held-back games and noise control | `noage142`, `noage142_early`, `noage142_within`; `noise_noage142` |

A season's sums hold 114 columns: each end's fitted part, then 56 inputs for offense and 56 for defense, in the order
of `second_prior.INPUT_SETS["context_types_no_age_team_zones_centred"]`.  `rating_line._solve(total, columns, 1e-4)`
gives the weights on any subset of columns and `RatingSums.error(weights)` the within-season test's error at them.
Today's line for a rated season is solved on the seasons more than one from it, each counted
`second_prior.season_weight(other, rated, 8)`.  `scratch/2026-11-18_age_out/age_out_read.py` is the shortest working
example of such a read: 30 seasons, paired by season, by era, in seconds.

**The four reads.**

1. **More inputs** (`scratch/2026-11-03_direct/direct_screen.py`, `build` then `read`, about 8 minutes a pass).  On
   2026-11-03 every candidate added to the line at once read 0.830 better, 28 of 30 seasons, eras z -7.24, -4.73,
   -1.25; the team rates that were adopted were 0.218 of it; 172 shuffled candidates cost 0.244.  The add-one and
   drop-one tables were lost with the data.  To run it today:
   - `FOLDS_TAG` to `rebuilt_base_early,rebuilt_base_within`; `inputs_dir` to `outputs/second_prior/noage142_line`.
   - Its within-layer multipliers come from a csv of a full build, `season_ratings_ctx110_step4_multipliers.csv`,
     gone.  The line's own fit computes them from the same held-back games with
     `commands.build.within_layer_multipliers`, as `commands/second_prior.run_second_prior` does; use that.
   - It solves on every training season alike.  Add the fade: each training season's sums `.scaled(season_weight(...))`.
   - Take `age`, `age_at_first_season`, `age_at_debut` and every career column out of the candidates; `NOT_INPUTS`
     already holds the career ones.  `playoff_share` stays out: a coach's choice made after the held-back games.
   - The tracking columns are season totals and leak the held-back games; the screen drops any column equal across two
     sets of one season (`panel_candidates`).  Keep that check.
   - Two candidates to build, from each set's own games only: his share of his team's possessions in the fourth
     quarter and overtime with the margin within 5, set beside his share of all of them; and minutes per game.  The
     stint rows carry `period`, `margin_h` and `frac_rem`; `garbage_time_share` in `data/panel.py` is the pattern.
   - Then forward steps: the best candidate by the rule, added, and again, each beside its shuffled copy, until none
     passes.  Report the whole table, every candidate's verdict, never a prose list.
2. **What the LightGBM is worth beside the line.**  On 2026-11-03 dropping the box-score prior from one end of the
   line cost 0.060 on offense and 0.082 on defense; it is 92% to 94% a straight line on its own inputs.  Not read
   since the zones, the fade or age out, and never year over year.  The read: today's line without its `box_prior`
   and `box_prior_other_end` columns, with the LightGBM's own inputs offered to the line directly,
   `features.FEATURE_SETS["hustle_no_three_pct_self_creation"]`, 18 a side; those not already among the line's inputs
   are candidates of read 1's screen, so this is two more rows of the same pass.  The hustle inputs exist from 2017
   only: read 2017-2026 with them and every era without.  **A small cost is not a reason to drop the LightGBM by
   itself**: it alone sets each team's total in the prior, which the within-season test cannot see and the
   year-over-year test can.  A result here goes to the owner as a question, with what a build without it would have
   to show year over year.
3. **The box-score prior's weight by playing time.**  The line gives the box-score prior one weight for a starter and
   a tenth man alike.  The read: one more column an end, the box-score prior times his centered share of team
   possessions, scored like any candidate.  It is the backup-bigs question the tests keep visible,
   `tests/test_vs_consensus.py::test_backup_bigs_are_not_in_the_top_twenty`.  A product of two inputs is not in the
   saved sums: it needs the per-player sums of read 4, or a column in read 1's screen.
4. **Trees on what the line leaves** (`scratch/2026-11-10_boost_vs_line/player_sums.py`, 22 minutes, then
   `boost_vs_line.py`).  One step at ChimeraBoost's defaults read 0.100 better, 21 of 30, eras -0.111, -0.066, -0.124;
   a second random draw 0.115; shuffled targets +0.004.  DECISIONS.md, check 167: "A fit that repeats the step, or
   weighs teammates, could find more or the same; none of that was tried."  To run it today:
   - `player_sums.py`: `FOLDS` to `rebuilt_base_early,rebuilt_base_within`; `SCREEN` and `LINE` to `noage142_line`,
     whose folder holds both the inputs and the sums; `OUT` to `noage142_players`.  Its built-in check must pass: each
     season's per-player sums, projected on the line's columns, give back the line's own sums.
   - `boost_vs_line.py`: `NAMES` to the list without age, the folder to `noage142_players`, and the fade in `scores`
     and `block_columns`, where every training season counts alike today.
   - Then repeat the step: targets at the line plus the trees so far, a new column, weights solved again, up to five
     steps, with `BOOST_SHUFFLE=1` and a second `BOOST_SEED` at each.  The per-player sums hold the whole quadratic,
     so no ridge is solved again.  chimeraboost is installed in the environment and is not a dependency of the package.
   - Three seasons to a block, a block's models fitted on the seasons two or more away: keep it, and write down that
     a later step's targets come from columns whose models saw other blocks.  The shuffled control is what says
     whether that matters.

**Traps for these reads.**

- A candidate built from lineup results restates what the ridge already reads from the same games.  Fitted through
  the ridge it loses its weight by itself, which is why the direct screen exists; a candidate scored any other way
  must first pass `scratch/2026-11-03_team_rates/why_it_lost.py`'s check.
- 2017-2026 is the weak era for added inputs: more columns, more noise, and before the fade it wanted a stronger
  penalty on the line.  Read the line's penalty, 0.0001, again at 0.001 to 0.1 once inputs are added.
- In 1997-2006 the noise control alone reads 0.031 better, z -3.2.  A gain there of that size is not a gain.
- 2007-2016 is the era that fails most often.  Say which era failed and by how much; a tie is not noise.
- Every one of these is a read.  A full build is `bash scripts/experiment_chain.sh NAME
  points_with_expected_opponent_threes_late_as_scored "EXTRA RATE FLAGS"`, 70 to 120 minutes on this machine, and
  needs the owner's word.

**After these reads, the ranked paths none has started** (the plan has what the record says of each): three-point
luck out of the offense's target, as the defense's target has it; early-season defense, the hustle rates' padding
first; the ridge penalty, 13,037, never read on the within-season test, and the owner's EM idea; lineup terms from the
self-creation check; lineups against the league's official minutes.

## Where it stands, 2026-11-19

DECISIONS.md holds every number, from "The data deleted and rebuilt" to "The morning of 2026-11-19".

- **The data was deleted on 2026-11-16 and rebuilt on 2026-11-17.**  `git worktree remove --force` on old side
  checkouts followed their junctions into `data` and `outputs`.  **Never remove a side checkout before unlinking its
  junctions.**  Every earlier experiment's saved files are gone; what is on disk is the rebuild's and this week's.
  Backup of the rebuild: `D:\\openrapm_backup\\2026-11-17`.
- **Age is out of Box Adj: adopted and published 2026-11-19** (experiment 142; site 102e307).  The shipped line is
  `noage142_line`, 56 inputs an end; the shrink reads `noage142_early,noage142_within` and `noage142_k10`; the
  published tables are the full build `product142`'s, the ones before kept as `*_before_noage142*`.  It cost 0.027 on
  the within-season test, none of it since 2017, and its year-over-year cost is all a season's ratings asked about the
  season after it.  No 2026 total moved 0.1; two WOSA leaders changed by a hair, 1999 and 2003.
- **The incumbent is `noage142`, its noise control `noise_noage142`**, -0.013 on the within-season test; the chain
  names both.  Check 151's errors on the incumbent before it, `rebuilt`: 3,455.273, 3,595.240 and 3,932.011 by era.
- **The morning job waits for one game.**  0022600265, Hawks at Thunder of 2026-11-18, came from the league with every
  play-by-play description blank, so it builds no stints.  `config.yaml`'s `no_stints_accepted` names it (the owner:
  "skip the game for now but we should keep checking for it"): the season is rated without it, every morning asks the
  league again, and the day a morning's log says it has stints, take it off the list.  Basketball Reference's page for
  the game was not used: Sports Reference's terms bar feeding its pages to a model.
- **Main holds everything**, c29342d, GitHub's tests green.

"""

PAIRS = [
    ("""1. **Push the rest of the cleanup.**  The docs, the removals and the tests are on the branch `cleanup-2026-11-16`;
   `A:/code/spmm` runs them.  The site part went live on the owner's word.
""", """1. **Your word on any read of the next research path that passes its rule**, before a full build.
"""),
    ("""3. **Delete the old experiments' saved files?**  About 280 GB under `outputs/`: saved models 143 GB in 105 files, of
   which `fade141`, `noise_fade141` and `padbins127` are today's (Box Adj's line was fitted on `padbins127`'s
   held-back games); held-back games 61 GB in 255 folders, of which the `fade141_*` ones are today's; the input screens
   87 GB.  All of that is gone since the deletion of 2026-11-16; what is on disk is the rebuild's and this week's.
   And the old side checkouts `A:/code/spmm_gate`, `spmm_old`, `spmm_split`, `spmm_wosa`, `spmm_verify`, with
   the merged branches `split-robust`, `ui-tweaks`, `wosa-detail`.
""", """3. **The side checkout `A:/code/spmm_cleanup`** and its branch `cleanup-2026-11-16`, merged: remove them or keep them.
   Unlink its junctions first.
"""),
    ("""6. **A Changelog row for age out of Box Adj**, in the owner's words.
""", """6. **No Changelog row for age out of Box Adj for now** (the owner, 2026-11-19: "since we haven't explained box adj
   just leave it for now").
"""),
]


def main() -> None:
    path = Path("notes/HANDOFF.md")
    text = path.read_text(encoding="utf-8")
    text = TOP + text[text.index("## Waiting on the owner"):]
    for old, new in PAIRS:
        if text.count(old) != 1:
            raise SystemExit(f"found {text.count(old)} times, nothing written:\n{old}")
        text = text.replace(old, new)
    path.write_bytes(text.encode("utf-8"))
    print("wrote notes/HANDOFF.md,", len(text.splitlines()), "lines")


if __name__ == "__main__":
    main()
