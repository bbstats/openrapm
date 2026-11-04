"""2027 today under the incumbent and under experiment 134, like for like: the test setting, each arm's own line, folds and
held prior scales.  `openrapm build` reads its held scales from the published table, whose scales belong to the shipped
line: the line fitted on the rating has its own, about 0.69 / 0.69 against 1.09 / 0.97, so each arm holds the scales of
its own step-3 table.  Each arm's tables hold 2027 alone: season_ratings_<arm>now27*.parquet.

    .venv/Scripts/python scratch/2026-11-03_direct/build_2027.py
"""
import sys
from dataclasses import replace

from openrapm.commands.build import run_build
from openrapm.data.config import load_config
from openrapm.settings import TEST

ARMS = {"padbins127": dict(folds_tag="padbins127_early,padbins127_within"),
        "rating134": dict(folds_tag="rating134_early,rating134_within", second_prior="rating134",
                          second_prior_inputs="context_types_team_rates_centred")}


def main() -> int:
    cfg = load_config()
    for arm, chosen in ARMS.items():
        prior = TEST.prior
        if "second_prior" in chosen:
            prior = replace(prior, second_prior=chosen["second_prior"], second_prior_inputs=chosen["second_prior_inputs"])
        settings = replace(TEST, prior=prior,
                           shrink=replace(TEST.shrink, folds_tag=chosen["folds_tag"], extrapolate_folds_tag=""),
                           ridge=replace(TEST.ridge, held_scale_table=f"season_ratings_{arm}_step3"))
        run_build(cfg, settings, [2027], f"{arm}now27")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
