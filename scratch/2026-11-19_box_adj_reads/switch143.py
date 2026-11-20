"""The switch of experiment 143 (the owner, 2026-11-19, through the question tool: "Adopt and publish"): the shipped
settings, the tests that pin them and the docs name the line with the two defense inputs and its held-back games.  Run
with no build running, after scratch/2026-11-19_box_adj_reads/product143.sh has finished.  Every replacement must find
its text exactly once, or nothing is written.  The chain's incumbent is switched separately, once its noise control is
built.  After scratch/2026-11-18_age_out/switch142.py.

    .venv/Scripts/python -I scratch/2026-11-19_box_adj_reads/switch143.py [--dry]
"""
import sys
from pathlib import Path

LINE, LIST = "rd143_line", "context_types_no_age_team_zones_defence_only_centred"
OLD_LIST = "context_types_no_age_team_zones_centred"
FOLDS, TENTH = "rd143_early,rd143_within", "rd143_k10"

CHANGES = {
    "src/openrapm/settings.py": [
        ("""    # noage142_line since 2026-11-18: the line without age (notes/DECISIONS.md, experiment 142; the owner: "I want age
    # removed!"), weights fitted on the within-season test through the ridge (experiment 134), the team's shots counted
    # by five zones (experiment 139), on the held-back games of the rebuild of 2026-11-17 without Box Adj.  "" reads the
    # box-score prior itself
    second_prior: str = "noage142_line"
""", f"""    # {LINE} since 2026-11-19: the line without age (experiment 142) that also reads his offensive rebound share and
    # his close-game share of team possessions on the defense end (notes/DECISIONS.md, experiment 143), weights fitted
    # on the within-season test through the ridge (experiment 134), the team's shots counted by five zones (experiment
    # 139), on the held-back games of the rebuild of 2026-11-17 without Box Adj.  "" reads the box-score prior itself
    second_prior: str = "{LINE}"
"""),
        ("""    # centered on the season, so no player fold's line carries its own level.  Without age since experiment 142, adopted
    # 2026-11-18: no rating reads a player's age
    second_prior_inputs: str = "context_types_no_age_team_zones_centred"
""", f"""    # centered on the season, so no player fold's line carries its own level.  Without age since experiment 142, adopted
    # 2026-11-18: no rating reads a player's age.  With his offensive rebound share and close-game share of team
    # possessions on the defense end since experiment 143, adopted 2026-11-19
    second_prior_inputs: str = "{LIST}"
"""),
        ("""    # build brings its own, experiment 142's since 2026-11-18; the method is experiment 33b's
    folds_tag: str = "noage142_early,noage142_within"
""", f"""    # build brings its own, experiment 143's since 2026-11-19; the method is experiment 33b's
    folds_tag: str = "{FOLDS}"
"""),
        ("""    # 2026-10-22: "we are going to do it"); "" fits on folds_tag alone, as the experiment chain does.  Experiment
    # 142's own since 2026-11-18
    extrapolate_folds_tag: str = "noage142_k10"
""", f"""    # 2026-10-22: "we are going to do it"); "" fits on folds_tag alone, as the experiment chain does.  Experiment
    # 143's own since 2026-11-19
    extrapolate_folds_tag: str = "{TENTH}"
"""),
    ],
    "src/openrapm/commands/cli.py": [
        ("""                             "context_types_no_age_team_zones_centred, shipped, is the box-score prior on both "
                             "ends, the eight context numbers, his box rates, role and body, and his team's shots by "
                             "zone with him on and off the court; context_types_team_zones_centred is the same with "
                             "his age.  "
""", """                             "context_types_no_age_team_zones_defence_only_centred, shipped, is the box-score prior "
                             "on both ends, the eight context numbers, his box rates, role and body, his team's shots "
                             "by zone with him on and off the court, and on defense his offensive rebound share and "
                             "close-game share; context_types_no_age_team_zones_centred is the same without those "
                             "two.  "
"""),
        ("""                        help="the inputs, a key of second_prior.INPUT_SETS: context_types_no_age_team_zones_centred, "
                             "shipped, is the box-score prior on both ends, the eight context numbers, his box rates, "
                             "role and body, and his team's shots by zone with him on and off the court; "
                             "context_types_team_zones_centred is the same with his age; "
""", """                        help="the inputs, a key of second_prior.INPUT_SETS: "
                             "context_types_no_age_team_zones_defence_only_centred, shipped, is the box-score prior on "
                             "both ends, the eight context numbers, his box rates, role and body, his team's shots by "
                             "zone with him on and off the court, and on defense his offensive rebound share and "
                             "close-game share; context_types_no_age_team_zones_centred is the same without those two; "
                             "context_types_team_zones_centred is that with his age; "
"""),
        ("""                             "noage142_early,noage142_within")""", f"""                             "{FOLDS}")"""),
    ],
    "tests/test_cli.py": [
        ('["settings"].prior.second_prior == "noage142_line"', f'["settings"].prior.second_prior == "{LINE}"'),
        (f'    assert default == "{OLD_LIST}"', f'    assert default == "{LIST}"'),
    ],
    "tests/test_settings.py": [
        ('    assert PRODUCT.prior.second_prior == "noage142_line"     # experiment 142, the line without age',
         f'    assert PRODUCT.prior.second_prior == "{LINE}"     # experiment 143, the two defense inputs'),
        (f'    assert PRODUCT.prior.second_prior_inputs == "{OLD_LIST}"   # experiment 142',
         f'    assert PRODUCT.prior.second_prior_inputs == "{LIST}"   # experiment 143'),
        ('    assert PRODUCT.shrink.folds_tag == "noage142_early,noage142_within"         # experiment 142\'s',
         f'    assert PRODUCT.shrink.folds_tag == "{FOLDS}"         # experiment 143\'s'),
        ('    assert PRODUCT.shrink.extrapolate_folds_tag == "noage142_k10"             # experiments 101 and 142',
         f'    assert PRODUCT.shrink.extrapolate_folds_tag == "{TENTH}"             # experiments 101 and 143'),
    ],
    "PIPELINE.md": [
        ("- **How:** one straight line per side over 56 inputs, every one from the rated season:",
         "- **How:** one straight line per side over 58 inputs, every one from the rated season:"),
        ("""    feet or closer; short mid, to 13 feet; long mid, to the line; three, to 26 feet; long three, 27 feet or more;
    and free throws, offensive rebounds and turnovers.
""", """    feet or closer; short mid, to 13 feet; long mid, to the line; three, to 26 feet; long three, 27 feet or more;
    and free throws, offensive rebounds and turnovers;
  - on defense alone, his offensive rebound share of his rebounds, and his share of his team's possessions in the last
    five minutes of the fourth quarter or overtime with the margin within 5 less his share of all of them, padded by
    50 such team possessions (experiment 143); both read 0 on offense, a column the line weighs 0.
"""),
        (f"""- **Settings:** `PriorSettings.second_prior`, the sums `noage142_line`; `second_prior_inputs`,
  `{OLD_LIST}` (experiments 134, 139 and 142); `second_prior_team_totals`, `held`
""", f"""- **Settings:** `PriorSettings.second_prior`, the sums `{LINE}`; `second_prior_inputs`,
  `{LIST}` (experiments 134, 139, 142 and 143); `second_prior_team_totals`, `held`
"""),
        (f"""- **The shipped sums:** since 2026-11-18, `openrapm second-prior --models rebuilt_base --folds
  rebuilt_base_early,rebuilt_base_within --tag noage142_line --target rating --inputs
  {OLD_LIST}`, on the held-back games of the rebuild of 2026-11-17 without Box Adj
  (`scratch/2026-11-18_age_out/noise_and_age_out.sh`).  They are the sums of the line before it, `rebuilt_line`, with
  age's two columns dropped, to 1e-16.
""", f"""- **The shipped sums:** since 2026-11-19, `openrapm second-prior --models rebuilt_base --folds
  rebuilt_base_early,rebuilt_base_within --tag {LINE} --target rating --inputs
  {LIST}`, on the held-back games of the rebuild of 2026-11-17 without Box Adj
  (`scratch/2026-11-19_box_adj_reads/build143.sh`).  Over the line before it, `noage142_line`, they add four columns,
  the two inputs on each end, and agree with it on the others to 3e-15.
"""),
        ("  - the held-back games `noage142_early,noage142_within`, experiment 142's, seasons 1997-2026 pinned",
         f"  - the held-back games `{FOLDS}`, experiment 143's, seasons 1997-2026 pinned"),
        ("fitted again on sets holding back a tenth of the games, `noage142_k10`, and",
         f"fitted again on sets holding back a tenth of the games, `{TENTH}`, and"),
    ],
    "GLOSSARY.md": [
        ("opponents': 56 inputs an end, every one from the rated season and none of them his age (`second_prior.py`).",
         "opponents': 56 inputs an end and two more on defense, his offensive rebound share and his close-game share of "
         "team possessions, every one from the rated season and none of them his age (`second_prior.py`)."),
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
