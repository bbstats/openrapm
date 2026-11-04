"""Write a Box Adj line fitted through the rating (direct_screen.py) in the form `openrapm rate --second-prior TAG` reads:
outputs/second_prior/<TAG>/grams_<season>.npz, one per season and side, so a full build can test the direct screen's weights.

The sums of direct_screen.py hold both ends in one regression; a line's normal equations hold one end each.  So each end's
right-hand side has the other end's inputs and the two fitted-part columns taken off at their weights from a fit on every
season: b_O = c_O - Q_O,D beta_D - Q_O,games g.  That fit reads the rated season too, through those small cross terms only,
which is acceptable for this check of the screen and would not be for a shipped line.  No player folds: 29 to 45 weights
over 30 seasons cannot learn one player, so G_fold and b_fold are zeros.

    .venv/Scripts/python scratch/2026-11-03_direct/write_line.py --sums ctx110_all --inputs context_and_types_centred --tag direct132
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

from openrapm import second_prior
from openrapm.data.config import load_config, resolve
from openrapm.data.second_prior_files import SIDES, grams_directory, grams_path, write_grams, write_run
from openrapm.swap_adjust import PairGrams, type_coefficients

spec = importlib.util.spec_from_file_location("direct_screen", Path(__file__).with_name("direct_screen.py"))
direct_screen = importlib.util.module_from_spec(spec)
sys.modules["direct_screen"] = direct_screen
spec.loader.exec_module(direct_screen)

N_FOLDS = 5


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sums", default="ctx110_all")
    parser.add_argument("--inputs", required=True, choices=sorted(second_prior.INPUT_SETS))
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    cfg = load_config()
    sums = direct_screen.Sums(resolve(cfg, "outputs") / "direct" / args.sums)
    names = list(second_prior.INPUT_SETS[args.inputs])
    games = direct_screen.GAMES
    columns = [*games, *[f"O:{n}" for n in names], *[f"D:{n}" for n in names]]
    fitted = dict(zip(columns, sums.fit(columns, sums.seasons) * sums.scale[[sums.position[c] for c in columns]]))
    out = grams_directory(cfg, args.tag)
    out.mkdir(parents=True, exist_ok=True)
    at = {prefix: [sums.position[f"{prefix}:{n}"] for n in names] for prefix in ("O", "D")}
    at_games = [sums.position[c] for c in games]
    beta = {prefix: np.array([fitted[f"{prefix}:{n}"] for n in names]) for prefix in ("O", "D")}
    g = np.array([fitted[c] for c in games])
    grams_all = {}
    for season in sums.seasons:
        Q, c = sums.store[season]["Q"], sums.store[season]["c"]
        grams = {}
        for side, prefix, other in zip(SIDES, ("O", "D"), ("D", "O")):
            mine = at[prefix]
            right = c[mine] - Q[np.ix_(mine, at[other])] @ beta[other] - Q[np.ix_(mine, at_games)] @ g
            grams[side] = PairGrams(G=Q[np.ix_(mine, mine)], b=right, G_fold=np.zeros((N_FOLDS, len(mine), len(mine))),
                                    b_fold=np.zeros((N_FOLDS, len(mine))), n=int(sums.sw[season]))
        write_grams(grams_path(out, season), grams)
        grams_all[season] = grams
    # the line the build will read against the screen's: the build's penalty is on the raw inputs
    for side, prefix in zip(SIDES, ("O", "D")):
        read = type_coefficients([grams_all[y][side] for y in sums.seasons], None, 1e-4)
        gap = np.abs(read - beta[prefix]) / np.maximum(np.abs(beta[prefix]), 1e-3)
        worst = int(np.argmax(gap))
        print(f"{side}: largest relative gap between the build's weights and the screen's {gap[worst]:.3f}, on "
              f"{names[worst]}: {read[worst]:+.3f} against {beta[prefix][worst]:+.3f}")
    base = json.loads((resolve(cfg, "outputs") / "second_prior" / "ctx110" / "run.json").read_text())
    write_run(out, dict(argv=sys.argv, models=base["models"], folds="ctx110_early,ctx110_within", seasons=sums.seasons,
                        n_folds=N_FOLDS, inputs=names, type_scale=base["type_scale"], padded=True, target="direct",
                        sums=args.sums, step4_rule=base.get("step4_rule"), prior_check=base.get("prior_check")))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
