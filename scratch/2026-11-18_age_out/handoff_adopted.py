"""notes/HANDOFF.md brought to experiment 142's adoption (2026-11-19).  Every replacement must find its text once.

    .venv/Scripts/python -I scratch/2026-11-18_age_out/handoff_adopted.py
"""
from pathlib import Path

PLAN = r"C:\Users\Nathan\.claude\plans\what-are-some-possible-zippy-quill.md"
SECTION = f"""## Where it stands, 2026-11-19: age is out of Box Adj, adopted and published

DECISIONS.md holds every number, from "The data deleted and rebuilt" to "Experiment 142: adopted".

- **The research paths, ranked** (the owner, 2026-11-18): the plan
  `{PLAN}`, and DECISIONS.md, "The research paths of
  2026-11-18".  The owner's answer on the first: "box adj is good but guess what - I want age removed! :) the rest is
  fine."  **Age is not an input to any rating, and is offered to no screen or tree stage.**
- **Experiment 142, age out of Box Adj's line: adopted and published on the owner's word, "Adopt and publish".**  The
  shipped line is `noage142_line` on `context_types_no_age_team_zones_centred`, 56 inputs an end; the shrink reads
  `noage142_early,noage142_within` and `noage142_k10`; the published tables are the full build `product142`'s, the
  ones before them kept as `*_before_noage142*`.  Within-season +0.027, z +1.99, 10 of 30, by era +0.051, +0.041,
  -0.011; year over year at matched width z +2.18, all of it a season's ratings asked about the season after it.
  Published against the site before it: regulars' totals move a median 0.02 and 0.25 at most; no season's top-rated
  player changes; two WOSA leaders change by a hair, 1999 Duncan to Mutombo and 2003 Nowitzki to Garnett.  The Power
  table's weights stay.
- **The rebuild of 2026-11-17 read on the tests** before any of it: year over year 8.6891 against the earlier
  incumbent's 8.6908; check 151's errors on it 3,455.273, 3,595.240 and 3,932.011 by era; its noise control
  `noise_rebuilt` -0.012.
- **Next, approved by the owner:** the reads on Box Adj without age, from saved held-back games, no rating built:
  more inputs through the direct screen, close fourth-quarter share and minutes per game among them; what the LightGBM
  is worth beside the line; the box-score prior's weight by share of team possessions; trees on what the line leaves,
  several steps.  The tools must be pointed at `rebuilt_base_early`, `rebuilt_base_within` and the shipped line's
  folder, and given the fade's season weights: `scratch/2026-11-03_direct/direct_screen.py`,
  `scratch/2026-11-10_boost_vs_line/player_sums.py` and `boost_vs_line.py`.  Report and stop.
- **Still to do for the adoption:** a noise control on `noage142`, `--fold-assignment 1`, about 2 hours, then the
  chain's `INC_NAME`, `INC_SYSTEM`, `INC_FOLDS` and `CONTROL_NAME` to `noage142` and it, with GLOSSARY.md and this
  file; until then the chain scores against `rebuilt`, the incumbent before it.  Row 142 in the experiments sheet.
  The swap adjustment's unused model, `swap_adjust.PANEL_FEATURES`, still lists age: the owner's one-line choice.

"""
PAIRS = [
    ("""- **The shipped settings read the rebuild:** Box Adj's line `rebuilt_line`, the shrink's held-back games
  `rebuilt_early,rebuilt_within` and `rebuilt_k10`.""",
     """- **The shipped settings read the rebuild until 2026-11-19:** Box Adj's line `rebuilt_line`, the shrink's held-back
  games `rebuilt_early,rebuilt_within` and `rebuilt_k10`; since then experiment 142's, below."""),
    ("line `rebuilt_line` on `context_types_team_zones_centred`, 57 inputs an end,",
     "line `noage142_line` on `context_types_no_age_team_zones_centred`, 56 inputs an end, none of them his age,"),
    ("folds `rebuilt_early,rebuilt_within`, read to a full season with `rebuilt_k10`",
     "folds `noage142_early,noage142_within`, read to a full season with `noage142_k10`"),
    ("""6. **The verdict on experiment 142**, age out of Box Adj: adopt and publish, or leave built and unshipped.
""", """6. **A Changelog row for age out of Box Adj**, in the owner's words.
"""),
    ("""- **The published Box Adj line** is the rebuild's, fitted on `rebuilt_base`'s held-back games, pass 1 of README.md's
  recipe, so a fresh build repeats it.""",
     """- **The published Box Adj line** is fitted on `rebuilt_base`'s held-back games, pass 1 of README.md's recipe, on the
  default list of inputs, the one without age, so a fresh build repeats it."""),
]


def main() -> None:
    path = Path("notes/HANDOFF.md")
    text = path.read_text(encoding="utf-8")
    start = text.index("## Where it stands, 2026-11-18: the rebuild is the incumbent")
    end = text.index("## Where it stood, 2026-11-16")
    text = text[:start] + SECTION + text[end:]
    for old, new in PAIRS:
        if text.count(old) != 1:
            raise SystemExit(f"found {text.count(old)} times, nothing written:\n{old}")
        text = text.replace(old, new)
    path.write_bytes(text.encode("utf-8"))
    print("wrote notes/HANDOFF.md")


if __name__ == "__main__":
    main()
