"""The switch of experiment 142 (the owner, 2026-11-18, through the question tool: "Adopt and publish"): the shipped
settings, the tests that pin them and the docs name the line without age and its held-back games.  Run with no build
running, after scratch/2026-11-18_age_out/adopt142.sh has finished.  Every replacement must find its text exactly once,
or nothing is written.  The chain's incumbent is switched separately, once its noise control is built.

    .venv/Scripts/python -I scratch/2026-11-18_age_out/switch142.py [--dry]
"""
import sys
from pathlib import Path

LINE, LIST = "noage142_line", "context_types_no_age_team_zones_centred"
FOLDS, TENTH = "noage142_early,noage142_within", "noage142_k10"

CHANGES = {
    "src/openrapm/settings.py": [
        ("""    # rebuilt_line since 2026-11-17: experiment 139's line, weights fitted on the within-season test through the ridge
    # (experiment 134), the team's shots counted by five zones, fitted again by README.md's recipe after the data was
    # deleted and fetched again (notes/DECISIONS.md, "The cleanup of 2026-11-16").  "" reads the box-score prior itself
    second_prior: str = "rebuilt_line"
""", f"""    # {LINE} since 2026-11-18: the line without age (notes/DECISIONS.md, experiment 142; the owner: "I want age
    # removed!"), weights fitted on the within-season test through the ridge (experiment 134), the team's shots counted
    # by five zones (experiment 139), on the held-back games of the rebuild of 2026-11-17 without Box Adj.  "" reads the
    # box-score prior itself
    second_prior: str = "{LINE}"
"""),
        ("""    # centered on the season, so no player fold's line carries its own level
    second_prior_inputs: str = "context_types_team_zones_centred"
""", f"""    # centered on the season, so no player fold's line carries its own level.  Without age since experiment 142, adopted
    # 2026-11-18: no rating reads a player's age
    second_prior_inputs: str = "{LIST}"
"""),
        ("""    # build brings its own, the rebuild's of 2026-11-17 with experiment 141's settings; the method is experiment 33b's
    folds_tag: str = "rebuilt_early,rebuilt_within"
""", f"""    # build brings its own, experiment 142's since 2026-11-18; the method is experiment 33b's
    folds_tag: str = "{FOLDS}"
"""),
        ("""    # 2026-10-22: "we are going to do it"); "" fits on folds_tag alone, as the experiment chain does.  The rebuild's
    # own since 2026-11-17
    extrapolate_folds_tag: str = "rebuilt_k10"
""", f"""    # 2026-10-22: "we are going to do it"); "" fits on folds_tag alone, as the experiment chain does.  Experiment
    # 142's own since 2026-11-18
    extrapolate_folds_tag: str = "{TENTH}"
"""),
    ],
    "src/openrapm/commands/cli.py": [
        ("""                             "context_types_team_zones_centred, shipped, is the box-score prior on both ends, the "
                             "eight context numbers, his box rates, role and body, and his team's shots by zone with "
                             "him on and off the court; context_types_no_age_team_zones_centred is the same less "
                             "age.  "
""", """                             "context_types_no_age_team_zones_centred, shipped, is the box-score prior on both "
                             "ends, the eight context numbers, his box rates, role and body, and his team's shots by "
                             "zone with him on and off the court; context_types_team_zones_centred is the same with "
                             "his age.  "
"""),
        ("""                        help="the inputs, a key of second_prior.INPUT_SETS: context_types_team_zones_centred, shipped, "
                             "is "
                             "the box-score prior on both ends, the eight context numbers, his box rates, role and "
                             "body, and his team's shots by zone with him on and off the court; "
                             "context_types_team_rates_zones_centred adds the team's twos and threes, the set the "
                             "shipped line was cut from; context_types_no_age_team_zones_centred is the shipped "
                             "list less age (default %(default)s)")
""", """                        help="the inputs, a key of second_prior.INPUT_SETS: context_types_no_age_team_zones_centred, "
                             "shipped, is the box-score prior on both ends, the eight context numbers, his box rates, "
                             "role and body, and his team's shots by zone with him on and off the court; "
                             "context_types_team_zones_centred is the same with his age; "
                             "context_types_team_rates_zones_centred adds the team's twos and threes to that "
                             "(default %(default)s)")
"""),
        ("""                             "rebuilt_early,rebuilt_within")""", f"""                             "{FOLDS}")"""),
    ],
    "tests/test_cli.py": [
        ('["settings"].prior.second_prior == "rebuilt_line"', f'["settings"].prior.second_prior == "{LINE}"'),
        ('    assert default == "context_types_team_zones_centred"', f'    assert default == "{LIST}"'),
    ],
    "tests/test_settings.py": [
        ('    assert PRODUCT.prior.second_prior == "rebuilt_line"      # experiment 139\'s line, rebuilt 2026-11-17',
         f'    assert PRODUCT.prior.second_prior == "{LINE}"     # experiment 142, the line without age'),
        ('    assert PRODUCT.prior.second_prior_inputs == "context_types_team_zones_centred"   # experiment 139',
         f'    assert PRODUCT.prior.second_prior_inputs == "{LIST}"   # experiment 142'),
        ('    assert PRODUCT.shrink.folds_tag == "rebuilt_early,rebuilt_within"           # experiment 141\'s, rebuilt '
         '2026-11-17',
         f'    assert PRODUCT.shrink.folds_tag == "{FOLDS}"         # experiment 142\'s'),
        ('    assert PRODUCT.shrink.extrapolate_folds_tag == "rebuilt_k10"              # experiments 101 and 141, rebuilt',
         f'    assert PRODUCT.shrink.extrapolate_folds_tag == "{TENTH}"             # experiments 101 and 142'),
    ],
    "PIPELINE.md": [
        ("- **How:** one straight line per side over 57 inputs, every one from the rated season:",
         "- **How:** one straight line per side over 56 inputs, every one from the rated season:"),
        ("  - his 13 box rates, share of team possessions, share of games started, garbage-time share, age, height, "
         "weight;",
         "  - his 13 box rates, share of team possessions, share of games started, garbage-time share, height, weight; "
         "not his\n    age, which no rating reads (experiment 142);"),
        ("""- **Settings:** `PriorSettings.second_prior`, the sums `rebuilt_line`; `second_prior_inputs`,
  `context_types_team_zones_centred` (experiments 134 and 139); `second_prior_team_totals`, `held` (experiment 81);
""", f"""- **Settings:** `PriorSettings.second_prior`, the sums `{LINE}`; `second_prior_inputs`,
  `{LIST}` (experiments 134, 139 and 142); `second_prior_team_totals`, `held`
  (experiment 81);
"""),
        ("""- **The shipped sums:** since 2026-11-17, README.md's recipe on every season after the data was deleted and fetched
  again: `openrapm second-prior --models rebuilt_base --folds rebuilt_base_early,rebuilt_base_within --tag
  rebuilt_line --target rating`, on the held-back games of a build without Box Adj
  (`scratch/2026-11-16_cleanup/rebuild.sh`).  Before it, `zones139_line`, fitted on padbins127's folds and cut to the
  shipped inputs by `scratch/2026-11-05_second_stage/zones139_line.py`; the two agree within 0.05 points per 100 in
  finished seasons.
""", f"""- **The shipped sums:** since 2026-11-18, `openrapm second-prior --models rebuilt_base --folds
  rebuilt_base_early,rebuilt_base_within --tag {LINE} --target rating --inputs
  {LIST}`, on the held-back games of the rebuild of 2026-11-17 without Box Adj
  (`scratch/2026-11-18_age_out/noise_and_age_out.sh`).  They are the sums of the line before it, `rebuilt_line`, with
  age's two columns dropped, to 1e-16.
"""),
        ("  - the held-back games `rebuilt_early,rebuilt_within`, experiment 141's settings rebuilt 2026-11-17, seasons "
         "1997-2026 pinned",
         f"  - the held-back games `{FOLDS}`, experiment 142's, seasons 1997-2026 pinned"),
        ("fitted again on sets holding back a tenth of the games, `rebuilt_k10`, and",
         f"fitted again on sets holding back a tenth of the games, `{TENTH}`, and"),
    ],
    "GLOSSARY.md": [
        ("opponents': 57 inputs an end, every one from the rated season (`second_prior.py`).",
         "opponents': 56 inputs an end, every one from the rated season and none of them his age (`second_prior.py`)."),
    ],
}


def main() -> None:
    dry = "--dry" in sys.argv
    texts = {}
    for path, pairs in CHANGES.items():
        text = Path(path).read_text(encoding="utf-8")
        for old, new in pairs:
            if text.count(old) != 1:
                raise SystemExit(f"{path}: found {text.count(old)} times, nothing written:\n{old}")
            text = text.replace(old, new)
        texts[path] = text
    if dry:
        print(f"every replacement found once in {len(texts)} files; nothing written")
        return
    for path, text in texts.items():
        Path(path).write_bytes(text.encode("utf-8"))
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
