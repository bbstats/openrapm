"""The command line's help texts that no longer said what the code does (the cleanup of 2026-11-16)."""
from pathlib import Path

p = Path("src/openrapm/commands/cli.py")
raw = p.read_bytes()
crlf = b"\r\n" in raw
t = raw.decode("utf-8").replace("\r\n", "\n")
EDITS = [
    ("""    openrapm build     all five steps in one run: the ratings tables after steps 3, 4 and 5
    openrapm rate      steps 1-3 for the seasons asked for: the ratings table before the prior shrink
    openrapm shrink    step 4 on a step-3 table: the prior part multiplied down
    openrapm swap      step 5 on a step-4 table: each team's credit re-split by its lineup swaps
""", """    openrapm build     every step in one run: the ratings tables after steps 3, 4 and 5, and WOSA
    openrapm rate      steps 1-3 with Box Adj for the seasons asked for: the ratings table before the shrink
    openrapm shrink    step 4 on a step-3 table: each part of every rating multiplied by what holds up
    openrapm swap      step 5 on a step-4 table: the swap adjustment, off as shipped
"""),
    ("""    openrapm second-prior   the swaps the second prior's line is fitted on, from the folds' held-back games
""", """    openrapm second-prior   the sums Box Adj's line is fitted on, from the folds' held-back games
"""),
    ("""(notes/HANDOFF.md, header: rebuilt on one thread, the published final table agrees to within 1.2e-12 per 100).
""", """(rebuilt on one thread in the 2026-10 cleanup, the published final table agreed to within 1.2e-12 per 100).
"""),
    ('''"nothing fitted for a season comes from the seasons either side of it (the labels and "
                             "the prior's training rows; in build also step 4's multipliers and step 5's type "
                             "model) (default %(default)s)")''',
     '''"nothing fitted for a season comes from the seasons either side of it (the labels, "
                             "the prior's training rows and Box Adj's line; in build also step 4's multipliers and "
                             "step 5's type model) (default %(default)s)")'''),
    ('''help="step 3 reads the second prior: the box-score prior corrected for who each player played "
                             "with and against, by the line fitted on outputs/second_prior/<TAG>/, which `openrapm "
                             "second-prior` writes.  Empty: the box-score prior itself (default, shipped)")''',
     '''help="Box Adj: step 3 reads the box-score prior corrected for who each player played with and "
                             "against, by the line fitted on outputs/second_prior/<TAG>/, which `openrapm "
                             "second-prior` writes.  Empty: the box-score prior itself (default %(default)s, "
                             "shipped)")'''),
    ('''help="context: the line reads the box-score prior and the eight context numbers; "
                             "context_and_types: also the other end's box-score prior and step 5's box counts, role "
                             "and body.  Must match what `openrapm second-prior --inputs` fitted (default %(default)s)")''',
     '''help="what Box Adj's line reads, a key of second_prior.INPUT_SETS: context_types_team_zones_centred, "
                             "shipped, is the box-score prior on both ends, the eight context numbers, his box "
                             "rates, role and body, and his team's shots by zone with him on and off the court.  "
                             "Must match what `openrapm second-prior --inputs` fitted (default %(default)s)")'''),
    ('''description="Step 4 on a step-3 table: each season's box-score prior part times two multipliers (offence, "
                    "defence) fitted on the within-season folds' held-out games, the ratings re-centred, the "
                    "games' part unchanged.  Writes outputs/season_ratings_<NAME>_step4.parquet, and the "
                    "multipliers of every season to outputs/csv/season_ratings_<NAME>_step4_multipliers.csv.")''',
     '''description="Step 4 on a step-3 table: each season's prior part and games' part, split into each team's "
                    "mean and each player's deviation from it, times multipliers fitted on the within-season folds' "
                    "held-back games, offense and defense, the ratings re-centered.  Writes "
                    "outputs/season_ratings_<NAME>_step4.parquet, and the multipliers of every season to "
                    "outputs/csv/season_ratings_<NAME>_step4_multipliers.csv.")'''),
    ('''description="Step 5 on a step-4 table: half of what a type model of other seasons' lineup swaps predicts "
                    "each player is under- or over-credited, given back inside his team so no team total moves, "
                    "and the spread inside teams held.  The type model''',
     '''description="Step 5 on a step-4 table, off as shipped (--applied-share 0): a share of what a type model of "
                    "other seasons' lineup swaps predicts each player is under- or over-credited, given back inside "
                    "his team so no team total moves, and the spread inside teams held.  The type model'''),
    ('''"the within-season folds, comma-separated folders under outputs/within/, e.g. "
                             "bags10_early,bags10_within")''',
     '''"the within-season folds, comma-separated folders under outputs/within/, e.g. "
                             "fade141_early,fade141_within")'''),
    ('''help="context: the box-score prior and the eight context numbers; context_and_types: also the "
                             "other end's box-score prior and step 5's 19 numbers; context_and_types_centred: the "
                             "same, centred on the season when applied; own_teammates_opponents: experiment 92's "
                             "three numbers; context_types_team_rates_centred: experiment 128's team rates too; "
                             "context_types_team_profile_centred: experiment 129's, without team makes "
                             "(default %(default)s)")''',
     '''help="the inputs, a key of second_prior.INPUT_SETS: context_types_team_zones_centred, shipped, is "
                             "the box-score prior on both ends, the eight context numbers, his box rates, role and "
                             "body, and his team's shots by zone with him on and off the court; "
                             "context_types_team_rates_zones_centred adds the team's twos and threes, the set the "
                             "shipped line was cut from (default %(default)s)")'''),
    ('''"--decomposition also writes docs/data/decompose_1y.json and "
                    "decompose_3y.json (vanilla RAPM split by player and by possession; every season and every "
                    "three-season window is fitted, 319 seconds in the last run); --openrapm-decomposition also "
                    "writes docs/data/decompose_openrapm.json (every published rating split into box score, on "
                    "court, off court, team SOS and swap adjustment, which solves step 3 again for every season; "
                    "run it after every `openrapm build`); --similarity also writes "
                    "docs/data/similarity/ (each player-season's nearest player-seasons of other players, by the "
                    "distance between their standardised inputs).  The pages' wording is not changed.")''',
     '''"--decomposition also writes docs/data/decompose_1y.json and "
                    "decompose_3y.json, the retired vanilla RAPM page's; --openrapm-decomposition also "
                    "writes docs/data/decompose_openrapm.json and ratings_list.json (every published rating split "
                    "into box score, Box Adj, on court, off court and team SOS, which solves step 3 again for every "
                    "season; run it after every `openrapm build`); --similarity also writes "
                    "docs/data/similarity/ (each player-season's nearest player-seasons of other players, by the "
                    "distance between their standardized inputs); --power also writes docs/data/power.json.  The "
                    "pages' wording is not changed.")'''),
    ('''help="the date every written file's meta.built shows (default: ratings.json the ratings "
                           "table's modification date, the decomposition files today)")''',
     '''help="the date every written file's meta.built shows (default: the ratings table's "
                           "modification date for ratings.json, the Detail view's files and power.json; today for "
                           "the Similarity and Bias files)")'''),
    ('''"(power_forecast.py); the morning job does not pass this")''',
     '''"(power_forecast.py); the morning job passes it")'''),
]
for old, new in EDITS:
    assert t.count(old) == 1, old[:90]
    t = t.replace(old, new)
p.write_bytes((t.replace("\n", "\r\n") if crlf else t).encode("utf-8"))
print("edited", p, "crlf" if crlf else "lf")
