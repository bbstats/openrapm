"""scripts/experiment_chain.sh's comments: the header to today's readings, the incumbent's lineage cut to today's."""
from pathlib import Path

p = Path("scripts/experiment_chain.sh")
raw = p.read_bytes()
crlf = b"\r\n" in raw
t = raw.decode("utf-8").replace("\r\n", "\n")

old_head = t[t.index("# One experiment's whole ratings chain"):t.index("# Log: outputs/NAME_chain.log.")]
new_head = """# One experiment's full build and its readings (README.md, "The test that decides"): steps 1-3 with Box Adj in one run
# (`openrapm rate --setting test`), the candidate's own within-season folds for all thirty seasons, step 4 (`openrapm
# shrink --setting test`), step 5 (`openrapm swap --setting test`, off as shipped), then the three readings the owner
# judges on (2026-10-15: "too many tests ... within-season is truly the best metric, particularly WOWY"): check 151,
# the within-season WOWY test, which decides, with the noise control beside it; check 63, the year-over-year test,
# which guards it; and check 66, the 2026 top 20.  FULL_READINGS=1 adds the older readings (138, the second arm, 64,
# 90, 88, 70 + 73).  Everything not named is the incumbent's.
#   bash scripts/experiment_chain.sh NAME DEFENSIVE_TARGET ["EXTRA RATE FLAGS"]   (INC_NAME, BASE_FLAGS, SHRINK_TAG, SKIP_BUILD: see below)
#   bash scripts/experiment_chain.sh noise_control points_with_expected_opponent_threes_late_as_scored "--fold-assignment 1"
"""
assert t.count(old_head) == 1
t = t.replace(old_head, new_head)

start = t.index("# The incumbent since 2026-10-09")
end = t.index("INC_NAME=${INC_NAME:-")
new_lineage = """# The incumbent: experiment 141, fade141, adopted 2026-11-10 (notes/DECISIONS.md, "Experiment 141"): Box Adj's training
# seasons counted half for every 8 seasons from the rated one, its own folds fade141_early and fade141_within.  Its
# noise control noise_fade141 (--fold-assignment 1): -0.013, z -1.41, 16 of 30, built 2026-11-10.  Each adoption
# names its build here; the earlier incumbents are in notes/DECISIONS.md and in git.  The incumbent's rate settings ride
# in BASE_FLAGS, BEFORE the experiment's own EXTRA flags, because the command line keeps the LAST occurrence of a flag:
# an experiment's --features or --booster-params wins.  The held-out audit pairs a candidate with the incumbent's
# 2017-2026 folds (AUDIT_REF).
"""
t = t[:start] + new_lineage + t[end:]
t = t.replace("# DECIDER_FLAGS: extra flags for the decider (151).", "# DECIDER_FLAGS: extra flags for the within-season WOWY test (151).")
t = t.replace("# (151 --shrink-flags), so it scores the step 4 the candidate was built with; before, 151 rebuilt the shipped one.",
              "# (151 --shrink-flags), so it scores the step 4 the candidate was built with.")
t = t.replace("# SHRINK_TAG=<tag>: the within-season folds step 4's multipliers are read from",
              "# SHRINK_TAG=<tag>: the within-season folds step 4's multipliers are read from")
p.write_bytes((t.replace("\n", "\r\n") if crlf else t).encode("utf-8"))
print("edited", p, "crlf" if crlf else "lf")
