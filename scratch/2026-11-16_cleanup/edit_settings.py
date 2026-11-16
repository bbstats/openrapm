"""settings.py's comments cut to the current value and the experiment that set it (the cleanup of 2026-11-16).
The list of every value a setting has had is in notes/DECISIONS.md and git; CONTRIBUTING.md, "Comments say why"."""
from pathlib import Path

p = Path("src/openrapm/settings.py")
raw = p.read_bytes()
crlf = b"\r\n" in raw
t = raw.decode("utf-8").replace("\r\n", "\n")
EDITS = [
    ('''"""The shipped settings: every number the five steps use, in one place, with where it was decided.

Two settings exist.  PRODUCT builds the published ratings.  TEST builds the table the year-over-year test
(defined below) scores: it rates each season from labels and training rows that have seen neither the season
before it nor the season after it, because those are the seasons the test predicts (notes/DECISIONS.md, "The
year-over-year test", 2026-09-13).
''', '''"""The shipped settings: every number the build uses, in one place, with where it was decided.

Two settings exist.  PRODUCT builds the published ratings.  TEST builds the table the year-over-year test
(defined below) scores: it rates each season from labels, training rows and a Box Adj line that have seen neither the
season before it nor the season after it, because those are the seasons the test predicts (notes/DECISIONS.md, "The
year-over-year test", 2026-09-13).
'''),
    ('''    year-over-year test   the deciding score (scripts/checks/63_yoy.py): each season's ratings predict the games of
                the season before and the season after from the ten players' ratings alone (only the scored
                season's intercept and home edge are refitted), scored against the points actually scored
''', '''    year-over-year test   the test that guards the within-season WOWY test, which decides (scripts/checks/63_yoy.py
                and 151_wowy.py): each season's ratings predict the games of the season before and the season after
                from the ten players' ratings alone (only the scored season's intercept and home edge are refitted),
                scored against the points actually scored
'''),
    ('''    # "hustle_no_three_pct_self_creation" (notes/DECISIONS.md, experiment 110, adopted 2026-10-25: "the rating changes
    # are good - we can adopt and publish"): the offensive list with the owner's self-creation appended.  Before it:
    # "hustle_no_three_pct" (notes/DECISIONS.md, experiment 77, adopted 2026-10-17: "Adopt and publish"): hustle with
    # three-point % off the offensive booster, the one input the cull on the decider found worse than its own
    # shuffled copy (check 152 `cull`).  Before it:
    # "hustle" (notes/DECISIONS.md, experiment 76, adopted 2026-10-16: "Hell yeah. Publish"): no_career_counts plus the
    # hustle box score's counts, each summed over the games a rating is built from, missing before 2017.  Before it:
    # "no_career_counts" (notes/DECISIONS.md, experiment 59, adopted 2026-10-14: "adopt anyway"): box_score_and_bio
    # without the career counts (career possessions before, seasons played before, seasons with his current team), so
    # the prior reads only the rated season's box score, role and body.  Before it:
    # "career_from_debut" (notes/DECISIONS.md, experiment 57, adopted 2026-10-13): box_score_and_bio with seasons played
    # before, career possessions before and seasons with his current team counted from each player's debut, his
    # seasons before 1997 included (data/careers.py), each in the place of the 1997-counted column.
    # "box_score_and_bio": the inputs Boruta selected (a feature-selection method: an input is kept when its importance
    # beats that of shuffled copies of the inputs), without the on-court columns (features.SHIPPED; experiment 45,
    # adopted 2026-10-08, where it is called boruta_noonc)
''', '''    # the season's box score, role and body, the hustle box score's counts summed over the games a rating is built from
    # (missing before 2017), three-point % off the offense, and on offense the owner's self-creation (notes/DECISIONS.md,
    # experiment 110, adopted 2026-10-25, on experiments 45, 59, 76 and 77 before it); features.FEATURE_SETS says
    # what each list holds
'''),
    ('''    # "": step 3 reads the box-score prior itself, as it did until experiment 82.  A tag: step 3 reads the second
    # prior instead, shipped as ctx82 since experiment 82 was adopted 2026-10-20, "adopt for now", its inputs centred
    # since experiment 93 (`second_prior_inputs`); ctx110 since experiment 126, adopted 2026-11-01, the same line
    # refitted on creation110's boosters and folds (ctx82's came from experiment 78's, before the parser fixes, the
    # body listing, the part-season rows and self-creation); experiment 127, adopted the same day, was tested and
    # adopted on this line, so it stayed until a refit passed a test of its own: rating134 since experiment 134, adopted
    # 2026-11-04 (the owner: "Adopt, then show me"), the weights fitted on the within-season test through the ridge
    # (`openrapm second-prior --target rating`) instead of on the swaps; zones139_line since experiment 139, adopted
    # 2026-11-06 (the owner: "Adopt but don't publish until tomorrows run"), the team's shots counted by five zones.
    # It is the box-score prior corrected for who a player played with and against, by a line fitted on the swaps of the
    # within-season folds' held-back games written under outputs/second_prior/<tag>/ (second_prior.py;
    # notes/DECISIONS.md, experiment 81, the owner, 2026-10-19: "take the handoff idea ... in as inputs to the prior")
''', '''    # Box Adj: step 3 reads the box-score prior corrected for who a player played with and against, by the line whose
    # sums are under outputs/second_prior/<tag>/ (second_prior.py, rating_line.py; notes/DECISIONS.md, experiment 81).
    # zones139_line since experiment 139, adopted 2026-11-06: weights fitted on the within-season test through the ridge
    # (experiment 134), the team's shots counted by five zones.  "" reads the box-score prior itself
'''),
    ('''    # what the line reads: "context", the box-score prior and the eight context numbers (experiment 81), or
    # "context_and_types", also the other end's box-score prior and step 5's box counts, role and body, so the swaps'
    # lesson enters before the ridge (experiment 82, the owner, 2026-10-19: "wouldnt it stand to reason to put the
    # swap adjustments into the priors before RAPMizing?"), or "context_and_types_centred", the same centred on the
    # season's possession-weighted average, so no player fold's line carries its own level (experiment 93)
    # shipped: "context_types_team_rates_centred", experiment 134, adopted 2026-11-04, today's 29 and the team's eight
    # box rates with the player on and off court ("context_and_types_centred", experiment 93, from 2026-10-21;
    # experiment 82's "context_and_types" from 2026-10-20); "context_types_team_zones_centred" since experiment 139,
    # adopted 2026-11-06: the team's shots made and missed by zone, rim, short mid, long mid, three and long three, in
    # place of threes and twos
''', '''    # what the line reads, a key of second_prior.INPUT_SETS: the box-score prior on both ends, the eight context
    # numbers, his box rates, role and body (experiments 81, 82 and 93), and the team's shots made and missed by zone,
    # free throws, offensive rebounds and turnovers with him on and off the court (experiments 134 and 139), every input
    # centered on the season, so no player fold's line carries its own level
'''),
    ('''    # the within-season folds the factors are fitted on, comma-separated folders: experiment 78's own folds, 1997-2016
    # and 2017-2026 (notes/DECISIONS.md, "Experiment 78", adopted 2026-10-17; before it experiment 77's,
    # nothreepct_early,nothreepct_within, experiment 76's, tune2_early,tune2_within, and experiment 63's,
    # cols06_early,cols06_within); the method is experiment 33b's, adopted 2026-10-03
    # since experiment 82, adopted 2026-10-20, its own folds, ctx82_early,ctx82_within, built with the second prior;
    # since experiment 93, adopted 2026-10-21, its own, ctx82c_early,ctx82c_within, built with the centred inputs;
    # since experiment 94, adopted 2026-10-22 (the bug-fix pass's parser fixes), its own on the rebuilt stints,
    # parser94_early,parser94_within; since experiment 95, adopted the same day (heaves counted as they happened), its
    # own, heave95_early,heave95_within; since experiment 99, adopted the same day (each season's own body listing, the
    # pad k around the bins), its own, inputs99_early,inputs99_within; since experiment 107, adopted 2026-10-23 (the
    # box-score prior's part-season rows), its own; since experiment 110, adopted 2026-10-25 (self-creation in the
    # offensive prior), its own; since experiment 126, adopted 2026-11-01 (the swap part of the prior refitted), its
    # own; since experiment 127, adopted the same day (the possession bins' targets padded toward the league rate),
    # its own; since experiment 134, adopted 2026-11-04 (the line fitted on the rating, with team rates), its own; since
    # experiment 139, adopted 2026-11-06 (the team's shots by zone), its own; since experiment 141, adopted 2026-11-10
    # (Box Adj's training seasons faded by distance), its own
''', '''    # the within-season folds the factors are fitted on, comma-separated folders, 1997-2016 and 2017-2026: every adopted
    # build brings its own, experiment 141's since it was adopted 2026-11-10; the method is experiment 33b's
'''),
    ('''    # 2026-10-22: "we are going to do it"); "" fits on folds_tag alone, as the experiment chain does.  inputs99_k10
    # until experiment 107 was adopted; its own since, parts107_k10; experiment 110's own since 2026-10-25; experiment
    # 126's own since 2026-11-01, then experiment 127's the same day; experiment 134's own since 2026-11-04; experiment
    # 139's own since 2026-11-06; experiment 141's own since 2026-11-10
''', '''    # 2026-10-22: "we are going to do it"); "" fits on folds_tag alone, as the experiment chain does.  Experiment 141's
    # own since 2026-11-10
'''),
    ('''    # one more factor on both defensive multipliers, by era: (first season, factor) pairs, each holding from its first
    # season until the next pair's (`factor_for`).  0.866 in every season from experiment 76 (adopted 2026-10-16,
    # check 152's guarded search: the held-back games set the defence's width within a season, and the next-season
    # check wants it narrower); since experiment 98 (adopted 2026-10-22) 0.89 before 2017 and 0.84 from 2017: the
    # within-season test wants the older seasons' defence wider in all 20 of them, and narrowing the newest era keeps
    # the year-over-year test even (notes/DECISIONS.md, "The defense factor by era"); since experiment 136, adopted
    # 2026-11-04, 0.8682 before 2017 and 0.8704 from 2017: halfway from those to what check 156 found for experiment
    # 134's line, 0.8464 and 0.9007, which went past the year-over-year limit.  ((0, 1.0),) is the old step
''', '''    # the defense factor: one more factor on both defensive multipliers, by era, (first season, factor) pairs, each
    # holding from its first season until the next pair's (`factor_for`).  The held-back games set the defense's width
    # within a season, and the year-over-year test wants it narrower (experiments 76 and 98).  Since experiment 136,
    # adopted 2026-11-04: halfway from experiment 98's 0.89 and 0.84 to what check 156 found for experiment 134's line,
    # 0.8464 and 0.9007, which went past the year-over-year limit.  ((0, 1.0),) is no factor
'''),
]
for old, new in EDITS:
    assert t.count(old) == 1, old[:90]
    t = t.replace(old, new)
p.write_bytes((t.replace("\n", "\r\n") if crlf else t).encode("utf-8"))
print("edited", p, "crlf" if crlf else "lf")
