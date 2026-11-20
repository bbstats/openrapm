"""The chain's incumbent after experiment 143 and its noise control (2026-11-19): scripts/experiment_chain.sh,
101_log_run.py's default, GLOSSARY.md, CONTRIBUTING.md and the handoff name rd143 and noise_rd143.  Every replacement
must find its text once, or nothing is written.  Run with no build running.  After
scratch/2026-11-18_age_out/incumbent142.py.

    .venv/Scripts/python -I scratch/2026-11-19_box_adj_reads/incumbent143.py [--dry]
"""
import sys
from pathlib import Path

CHANGES = {
    "scripts/experiment_chain.sh": [
        ("""# The incumbent: experiment 142, noage142, adopted 2026-11-18 (notes/DECISIONS.md, "Experiment 142"): Box Adj's line
# without age, its own folds noage142_early and noage142_within.  Its noise control noise_noage142
# (--fold-assignment 1): -0.013, z -1.39, 16 of 30, built 2026-11-19.  Each adoption""",
         """# The incumbent: experiment 143, rd143, adopted 2026-11-19 (notes/DECISIONS.md, "Experiment 143"): Box Adj's line
# with two defense inputs, its own folds rd143_early and rd143_within.  Its noise control noise_rd143
# (--fold-assignment 1): -0.012, z -1.37, 17 of 30, built 2026-11-19.  Each adoption"""),
        ("INC_NAME=${INC_NAME:-noage142}", "INC_NAME=${INC_NAME:-rd143}"),
        ("INC_SYSTEM=${INC_SYSTEM:-noage142}   ", "INC_SYSTEM=${INC_SYSTEM:-rd143}      "),
        ("CONTROL_NAME=${CONTROL_NAME:-noise_noage142}", "CONTROL_NAME=${CONTROL_NAME:-noise_rd143}"),
        ("INC_FOLDS=${INC_FOLDS:-noage142_early,noage142_within}", "INC_FOLDS=${INC_FOLDS:-rd143_early,rd143_within}"),
    ],
    "scripts/checks/101_log_run.py": [
        ("[--incumbent=noage142]", "[--incumbent=rd143]"),
        ('"--incumbent", default="noage142"', '"--incumbent", default="rd143"'),
    ],
    "CONTRIBUTING.md": [
        ("--controls=noise_noage142 --ref=noage142", "--controls=noise_rd143 --ref=rd143"),
    ],
    "GLOSSARY.md": [
        ("""today `noage142`, experiment 142 built at the test setting, with its held-back games `noage142_early` and
`noage142_within`.""",
         """today `rd143`, experiment 143 built at the test setting, with its held-back games `rd143_early` and
`rd143_within`."""),
        ("""`noise_noage142`. It shows how far every reading moves when nothing real changes: -0.013 on the within-season WOWY
test, paired z -1.39, better in 16 of 30 seasons.""",
         """`noise_rd143`. It shows how far every reading moves when nothing real changes: -0.012 on the within-season WOWY
test, paired z -1.37, better in 17 of 30 seasons."""),
    ],
    "notes/HANDOFF.md": [
        ("""- **The incumbent is `noage142`, its noise control `noise_noage142`**, -0.013 on the within-season test; the chain
  names both.""",
         """- **The incumbent is `rd143`, its noise control `noise_rd143`**, -0.012 on the within-season test; the chain
  names both since 2026-11-19.  Before it `noage142` and `noise_noage142`, -0.013."""),
        ("""- The incumbent is `noage142`: tables `outputs/season_ratings_noage142*`, folds `noage142_early`,
  `noage142_within`, year-over-year score 8.6901.  Its noise control is `noise_noage142`: -0.013 on the within-season
  test, paired z -1.39, 16 of 30, and -0.032, z -3.26 in 1997-2006 alone, so that era is read beside it.  The
  incumbent before it, `rebuilt`, and `noise_rebuilt` are still on disk.""",
         """- The incumbent is `rd143`: tables `outputs/season_ratings_rd143*`, folds `rd143_early`, `rd143_within`,
  year-over-year score 8.6900.  Its noise control is `noise_rd143`: -0.012 on the within-season test, paired z -1.37,
  17 of 30, and -0.031, z -3.18 in 1997-2006 alone, so that era is read beside it.  The incumbent before it,
  `noage142`, and `noise_noage142` are still on disk."""),
        ("--systems=noage142,noise_noage142,NAME --ref=noage142 --tag=NAME",
         "--systems=rd143,noise_rd143,NAME --ref=rd143 --tag=NAME"),
        ("- **Left to do:** the noise control `noise_rd143` (started 8:34 PM 2026-11-19, `scratch/2026-11-19_box_adj_reads/\n"
         "  noise_rd143.sh`, log `outputs/rd143/noise_rd143.out`); when it ends, switch the chain's incumbent to `rd143` and\n"
         "  `noise_rd143` as `scratch/2026-11-18_age_out/incumbent142.py` did for 142.  A Changelog row",
         "- **Done:** the noise control `noise_rd143`, -0.012, z -1.37, by era -0.031, -0.010, +0.004; the chain names\n"
         "  `rd143` and it (`scratch/2026-11-19_box_adj_reads/incumbent143.py`).  **Left to do:** a Changelog row"),
    ],
}


def main() -> None:
    texts = {}
    for path, pairs in CHANGES.items():
        text = Path(path).read_text(encoding="utf-8")
        for old, new in pairs:
            if text.count(old) != 1:
                raise SystemExit(f"{path}: found {text.count(old)} times, nothing written:\n{old}")
            text = text.replace(old, new)
        texts[path] = text
    if "--dry" in sys.argv:
        print(f"every replacement found once in {len(texts)} files; nothing written")
        return
    for path, text in texts.items():
        Path(path).write_bytes(text.encode("utf-8"))
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
