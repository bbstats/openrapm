"""Multiply a finished set of rankings by one number per side, so the criterion can price the amplitude.

    python scripts/75_amplitude.py [--rankings=outputs/season_ratings_sy_lam13037.parquet]
                                   [--off=0.70,0.85,1.00,1.20,1.40] [--def=0.70,0.85,1.00,1.20,1.40]
                                   [--also=trade:0.749x0.919,bars:1.196x0.867] [--tag=amp]

Writes one table per pair, `outputs/season_ratings_<tag>_<name>.parquet`, and prints the `--rankings=`
string for `scripts/63_yoy.py`.  Nothing is fitted here: each table is the input table with `offense`,
`defense` and the three `rating_*` columns multiplied, and every other column left exactly as it was.

**Why this exists.**  Two instruments disagree about which side is mis-scaled.  The trade set asks for
x0.749 on offence and x0.919 on defence -- the rating is too WIDE across adjacent seasons.  The
consensus error bars, matched side by side, ask for x1.196 on offence and x0.867 on defence -- our
offensive spread is 20% NARROWER than the public metrics' and our defensive 15% wider.  They agree in
direction on defence and contradict each other on offence.  The year-over-year test refits only the
scored season's intercept and home edge, so it prices a uniform rescale directly, and it is the one
thing both instruments can be held against.

**Read the result as calibration, never as a ranking gain.**  Shrinking a noisy rating lowers a
squared-error score by construction, so a multiplier below 1 winning the team-game error is not evidence
that the ORDER improved: read `mse` for the amplitude question and the rescaled row for the order.  And
the input must be a table whose prior never saw the scored seasons (`--exclude_neighbours=1`), or the
leak flatters the wider arms.

**Do not ship a multiplier applied this way without carrying it into the fit.**  The standing rule from
the LRBoost branch is that a correction belongs inside the fit, not added afterwards; what this sweep
settles is which direction is right and how much is on the table.
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _cli import check_flags, flag  # noqa: E402

# Everything that is a rating in the published schema.  `prior_*` and `u_*` are the pieces the rating
# was built from and are deliberately left alone: scaling them would make `--columns=prior` read a
# table whose parts no longer add up to its whole.
OFFENCE = ["offense", "rating_off"]
DEFENCE = ["defense", "rating_def"]


def scaled(table: pd.DataFrame, a: float, b: float) -> pd.DataFrame:
    out = table.copy()
    for column in OFFENCE:
        out[column] = table[column] * a
    for column in DEFENCE:
        out[column] = table[column] * b
    out["rating_total"] = out.rating_off + out.rating_def
    return out


def main() -> None:
    check_flags()
    path = ROOT / flag("rankings", "outputs/season_ratings_sy_lam13037.parquet")
    tag = flag("tag", "amp")
    offs = [float(v) for v in flag("off", "0.70,0.85,1.00,1.20,1.40").split(",")]
    defs = [float(v) for v in flag("def", "0.70,0.85,1.00,1.20,1.40").split(",")]
    table = pd.read_parquet(path)
    missing = [c for c in OFFENCE + DEFENCE + ["rating_total"] if c not in table.columns]
    if missing:
        raise SystemExit(f"{path}: missing {missing}")

    pairs = [(f"o{a:g}d{b:g}", a, b) for a in offs for b in defs]
    for spec in (flag("also", "trade:0.749x0.919,bars:1.196x0.867") or "").split(","):
        if spec.strip():
            name, _, values = spec.partition(":")
            a, _, b = values.partition("x")
            pairs.append((name, float(a), float(b)))

    print(f"{path.name}: {len(table):,} rows, {table.season.nunique()} seasons -> {len(pairs)} arms")
    named = []
    for name, a, b in pairs:
        out = ROOT / "outputs" / f"season_ratings_{tag}_{name}.parquet"
        scaled(table, a, b).to_parquet(out, index=False)
        named.append(f"{name}={out.relative_to(ROOT).as_posix()}")
    print("\n--rankings=" + ",".join(named))
    print(f"\nThe arm named o1d1 is the input table unchanged; make it the --ref.")


if __name__ == "__main__":
    main()
