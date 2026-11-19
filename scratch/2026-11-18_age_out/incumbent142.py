"""The chain's incumbent after experiment 142 and its noise control (2026-11-19): scripts/experiment_chain.sh,
101_log_run.py's default, GLOSSARY.md, CONTRIBUTING.md and the handoff name noage142 and noise_noage142.  Every
replacement must find its text once, or nothing is written.  Run with no build running.

    .venv/Scripts/python -I scratch/2026-11-18_age_out/incumbent142.py [--dry]
"""
import sys
from pathlib import Path

CHANGES = {
    "scripts/experiment_chain.sh": [
        ("""# The incumbent: rebuilt, experiment 141's settings built again on 2026-11-17 after the data was deleted and fetched
# again (notes/DECISIONS.md, "The data deleted and rebuilt" and "The rebuild read on the two tests"), its own folds
# rebuilt_early and rebuilt_within.  Its noise control noise_rebuilt (--fold-assignment 1): -0.012, z -1.38, 16 of 30,
# built 2026-11-18.  Each adoption""",
         """# The incumbent: experiment 142, noage142, adopted 2026-11-18 (notes/DECISIONS.md, "Experiment 142"): Box Adj's line
# without age, its own folds noage142_early and noage142_within.  Its noise control noise_noage142
# (--fold-assignment 1): -0.013, z -1.39, 16 of 30, built 2026-11-19.  Each adoption"""),
        ("INC_NAME=${INC_NAME:-rebuilt}", "INC_NAME=${INC_NAME:-noage142}"),
        ("INC_SYSTEM=${INC_SYSTEM:-rebuilt} ", "INC_SYSTEM=${INC_SYSTEM:-noage142}"),
        ("CONTROL_NAME=${CONTROL_NAME:-noise_rebuilt}", "CONTROL_NAME=${CONTROL_NAME:-noise_noage142}"),
        ("INC_FOLDS=${INC_FOLDS:-rebuilt_early,rebuilt_within}", "INC_FOLDS=${INC_FOLDS:-noage142_early,noage142_within}"),
    ],
    "scripts/checks/101_log_run.py": [
        ("[--incumbent=rebuilt]", "[--incumbent=noage142]"),
        ('"--incumbent", default="rebuilt"', '"--incumbent", default="noage142"'),
    ],
    "CONTRIBUTING.md": [
        ("--controls=noise_rebuilt --ref=rebuilt", "--controls=noise_noage142 --ref=noage142"),
    ],
    "GLOSSARY.md": [
        ("""today `rebuilt`, experiment 141's settings built at the test setting on 2026-11-17, with its held-back games
`rebuilt_early` and `rebuilt_within`.""",
         """today `noage142`, experiment 142 built at the test setting, with its held-back games `noage142_early` and
`noage142_within`."""),
        ("""`noise_rebuilt`. It shows how far every reading moves when nothing real changes: -0.012 on the within-season WOWY
test, paired z -1.38, better in 16 of 30 seasons.""",
         """`noise_noage142`. It shows how far every reading moves when nothing real changes: -0.013 on the within-season WOWY
test, paired z -1.39, better in 16 of 30 seasons."""),
    ],
    "notes/HANDOFF.md": [
        ("""- **Still to do for the adoption:** a noise control on `noage142`, `--fold-assignment 1`, about 2 hours, then the
  chain's `INC_NAME`, `INC_SYSTEM`, `INC_FOLDS` and `CONTROL_NAME` to `noage142` and it, with GLOSSARY.md and this
  file; until then the chain scores against `rebuilt`, the incumbent before it.  Row 142 in the experiments sheet.
  The swap adjustment's unused model, `swap_adjust.PANEL_FEATURES`, still lists age: the owner's one-line choice.""",
         """- **The noise control on the new incumbent, `noise_noage142`:** -0.013 on the within-season test, by era -0.032,
  -0.010, +0.004.  The chain names `noage142` and it since 2026-11-19.
- **Still to do for the adoption:** row 142 in the experiments sheet.  The swap adjustment's unused model,
  `swap_adjust.PANEL_FEATURES`, still lists age: the owner's one-line choice."""),
        ("""- The incumbent is `rebuilt`: tables `outputs/season_ratings_rebuilt*`, folds `rebuilt_early`, `rebuilt_within`,
  year-over-year score 8.6891.  Its noise control is `noise_rebuilt`: -0.012 on the within-season test, paired z -1.38,
  16 of 30, and -0.031, z -3.18 in 1997-2006 alone, so that era is read beside it.""",
         """- The incumbent is `noage142`: tables `outputs/season_ratings_noage142*`, folds `noage142_early`,
  `noage142_within`, year-over-year score 8.6901.  Its noise control is `noise_noage142`: -0.013 on the within-season
  test, paired z -1.39, 16 of 30, and -0.032, z -3.26 in 1997-2006 alone, so that era is read beside it.  The
  incumbent before it, `rebuilt`, and `noise_rebuilt` are still on disk."""),
        ("--systems=rebuilt,noise_rebuilt,NAME --ref=rebuilt --tag=NAME",
         "--systems=noage142,noise_noage142,NAME --ref=noage142 --tag=NAME"),
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
