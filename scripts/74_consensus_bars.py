"""How far our rankings sit from the consensus, in units of the consensus's OWN error bar.

    python scripts/74_consensus_bars.py outputs/season_ratings_sy_lam13037.parquet [more.parquet ...]
                                        [--season=2026] [--min_poss=1000]

`data/external/consensus.csv` carries a standard deviation per player per side (`var_offense`,
`var_defense`, added by the owner on 2026-09-18; the name says variance and the numbers are standard
deviations).  They are not the spread of the raw metrics: the votes are de-duplicated first by a
regularised GLS weighting, `(R + lambda I)^-1 1`, so five collinear metrics split one vote and a lone
independent dissenter widens the bar instead of being averaged away.  A wide bar therefore means the
public metrics genuinely do not know where a player belongs.

**Why a rank correlation was the wrong instrument on its own.**  Spearman against the blend treats a
disagreement about Jokic's defence (bar 0.39) and one about Wembanyama's offence (bar 2.05) as the same
size of miss, and reports a single number that cannot say whether we are outside what the public metrics
themselves can resolve.  The owner's rule, 2026-09-18: read the error bars.

Three readings, each side scaled on its own (the owner, 2026-09-18: *"feel free to scale first when
comparing to the consensus, esp by O vs D"*).  Offence and defence never share a slope here:

* **as published** -- `(ours - consensus) / sd` per player, then the root mean square, the mean (which is
  a level offset, not a disagreement about players) and the share of players more than one and more than
  two bars out.  This answers "is the published number inside the public uncertainty".
* **level and spread matched** -- our side is shifted and stretched so its mean and standard deviation
  equal the consensus's, which removes the units difference and NOTHING else.  `sd_ratio` is the stretch:
  above 1 our published spread is narrower than theirs.  This is the reading to quote for "how far apart
  are we once the scale is agreed", and the one to quote for a spread comparison.
* **best-fit line** -- the same after the least-squares slope and intercept.  Read `ols_slope` as
  `correlation x sd_ratio`, never as a spread comparison: a low correlation drags it down on its own, and
  quoting it as a scale is how our defensive spread got reported as 1.4x the consensus's when the spreads
  differ by 1.15x.  It is here because it is the smallest the disagreement can be made by any straight
  line, so it bounds what a recalibration could buy.

Nothing here chooses anything: the consensus is a sanity check and never a fitting target.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _cli import check_flags, flag  # noqa: E402

CONSENSUS = ROOT / "data" / "external" / "consensus.csv"
SIDES = {"offense": ("rating_off", "adj_offense", "var_offense"),
         "defense": ("rating_def", "adj_defense", "var_defense")}


def norm(name) -> str:
    """The consensus file's join key, as tests/test_vs_consensus.py builds it."""
    import unicodedata
    text = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    return "".join(ch for ch in text.lower() if ch.isalnum())


def consensus() -> pd.DataFrame:
    table = pd.read_csv(CONSENSUS)
    missing = [c for c in ("var_offense", "var_defense") if c not in table.columns]
    if missing:
        raise SystemExit(f"{CONSENSUS} has no {missing}: this script needs the per-player standard "
                         f"deviations the owner added on 2026-09-18.")
    table["key"] = table.player_name.map(norm)
    return table.dropna(subset=["adj_overall"]).drop_duplicates("key").set_index("key")


def read(path: Path, season: int, min_poss: float) -> pd.DataFrame:
    table = pd.read_parquet(path)
    table = table[(table.season == season) & (table.poss_off >= min_poss)].copy()
    table["key"] = table.player_name.map(norm)
    return table.drop_duplicates("key").set_index("key")


def bars(ours: pd.Series, theirs: pd.Series, sd: pd.Series) -> dict:
    """The disagreement in bar units: as published, spread-matched, and on the best-fit line.

    The spread match is the honest way to remove a units difference -- it changes the scale and nothing
    else.  The least-squares slope is `correlation x sd_ratio`, so it is NOT a spread comparison.
    """
    keep = np.isfinite(ours) & np.isfinite(theirs) & np.isfinite(sd) & (sd > 0)
    o, t, s = ours[keep].to_numpy(), theirs[keep].to_numpy(), sd[keep].to_numpy()
    sd_ratio = float(t.std(ddof=1) / o.std(ddof=1))
    matched = t.mean() + (o - o.mean()) * sd_ratio
    ols_slope, ols_intercept = np.polyfit(o, t, 1)
    out = {"players": len(o), "our_sd": float(o.std(ddof=1)), "their_sd": float(t.std(ddof=1)),
           "corr": float(np.corrcoef(o, t)[0, 1]), "sd_ratio": sd_ratio,
           "ols_slope": float(ols_slope), "ols_intercept": float(ols_intercept)}
    for label, fitted in (("", o), ("_matched", matched), ("_ols", ols_slope * o + ols_intercept)):
        z = (fitted - t) / s
        out[f"rms{label}"] = float(np.sqrt(np.mean(z ** 2)))
        out[f"mean{label}"] = float(np.mean(z))
        out[f"over_1{label}"] = float(np.mean(np.abs(z) > 1))
        out[f"over_2{label}"] = float(np.mean(np.abs(z) > 2))
    return out


def main() -> None:
    check_flags()
    paths = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not paths:
        raise SystemExit(__doc__)
    season, min_poss = int(flag("season", 2026)), float(flag("min_poss", 1000))
    con = consensus()
    print(f"the consensus error bars, {season}, players over {min_poss:,.0f} possessions.  "
          f"A figure of 1.0 means we sit one standard deviation of the public metrics' own "
          f"disagreement away from their blend.")
    rows = []
    for path in paths:
        table = read(ROOT / path if not Path(path).is_absolute() else Path(path), season, min_poss)
        shared = table.index.intersection(con.index)
        for side, (mine, theirs, sd) in SIDES.items():
            rows.append({"rankings": Path(path).stem.replace("season_ratings_", ""), "side": side,
                         **bars(table.loc[shared, mine], con.loc[shared, theirs],
                                con.loc[shared, sd])})
    frame = pd.DataFrame(rows)
    print("as published:")
    print(frame[["rankings", "side", "players", "rms", "mean", "over_1", "over_2"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\nlevel and spread matched, each side on its own -- the units difference removed and "
          "nothing else:")
    print(frame[["rankings", "side", "our_sd", "their_sd", "sd_ratio", "rms_matched",
                 "over_1_matched", "over_2_matched"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\non the best-fit line, which bounds what any recalibration could buy "
          "(`ols_slope` = corr x sd_ratio, so it is not a spread comparison):")
    print(frame[["rankings", "side", "corr", "ols_slope", "ols_intercept", "rms_ols",
                 "over_1_ols", "over_2_ols"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\n`rms` is the root mean square disagreement in bar units, `mean` a level offset rather "
          "than a disagreement about players, `over_1` and `over_2` the share of players more than one "
          "and two bars out.  `sd_ratio` above 1 means our published spread is narrower than theirs.")


if __name__ == "__main__":
    main()
