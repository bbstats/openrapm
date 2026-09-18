"""The year-over-year test: one season's player rankings predict the games of the seasons either side of it.

    python scripts/63_yoy.py --rankings=sy=outputs/season_ratings_sy_yoy.parquet,ship=artifacts/season_ratings.parquet
                             [--ref=sy] [--splits=movers] [--tag=yoy] [--first=1998] [--last=2025]
                             [--columns=rating|prior] [--fill=name:500x0.5,name2:-1.6/-0.5]
                             [--offset=1]

For every scored season the previous season's rankings and the next season's rankings are each asked to
predict its games, from the ten players' offensive and defensive ratings alone; only the season's level
(intercept and home edge) is refit on the scored season.  Scored against the points actually scored, at
stint level and at team-game level (`holdout.score`).  The rankings for a season are read from a table:
nothing is fit here, so what is measured is exactly the table handed in.

**Why this test and not the within-season 75/25 split.**  The 75/25 split holds a quarter of a season's
games out of the ridge but not out of the prior's on-court features, which are averaged over the whole
season (DECISIONS.md, "the single-year board ... a leak").  Here the scored games are a different season
from the rated one, so nothing in the rated season's prior or evidence has seen them.  And a player who
changed teams between the two seasons carries his rating to new teammates, which is where a rating that
is really his old team's credit stops predicting -- the `movers` split reads that group on its own.

**What the rankings must satisfy to be tested honestly.**  A season's ratings may use that season's games
and may use model coefficients learned from OTHER seasons -- but not from the two seasons being scored.
`scripts/62_single_year_board.py --exclude_neighbours=1` builds a table that way.  A table whose prior was
trained on every other season (the shipped `--exclude_neighbours=0` default) has seen the scored season's
outcomes through the prior's coefficients and reads optimistically here.

**`--fill=` gives a table a replacement level for the players it has no row for.**  A season's rankings
cover only the players who played that season, so a rookie of the scored season -- 10% of its possessions
going forward, 6% going back -- is scored as the AVERAGE player (0), which is the one thing he is not.
`--fill=name:500` gives that table `holdout.ReplacementSystem`: the possession-weighted mean rating of its
own players under 500 possessions, read off the rating season and nothing else; `500x0.25` takes a
quarter of that level, which is the depth the test settled on (DECISIONS.md).  `--fill=name:-1.6/-0.5`
sets the two levels by hand, positive-good, offence first.  A name not listed keeps the 0 fill, so the
incumbent and the fill can be paired in one run by handing the same parquet in twice under two names.

**`--offset=N`** asks the rankings N seasons away instead of the adjacent one.  The trade set
(`scripts/70_tradeset.py`) is fitted on the games one season either side of the rated one, so scoring it
at the default offset would be scoring a fit on the very rows it used; at 2 the scored games are outside
its window.  Both arms of any comparison must use the same offset, and the absolute numbers move with it.

Each table needs `player_id`, `season`, `rating_off`, `rating_def` (both positive-good) and `poss_off`.
Writes outputs/yoy_<tag>.parquet with one row per scored season x direction x table x split x group.
"""
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eracoef.config import load_config  # noqa: E402
from eracoef.holdout import (SPLITS, Context, Holdout, Ratings, ReplacementSystem,  # noqa: E402
                             paired, pooled)

ROOT = Path(__file__).resolve().parents[1]
# which season's rankings predict the scored one.  `--offset=N` moves both directions out to N
# seasons: the trade set needs 2, because its alpha is fitted on the games one season either side and
# scoring it there would be scoring a fit on its own rows.
def directions(offset: int = 1) -> dict:
    return {"prev": -int(offset), "next": +int(offset)}



# The shared command line (scripts/_cli.py): `flag` records every name it is asked for so
# `check_flags` below can refuse one that was never asked for.  A misspelled flag used to be
# ignored silently, which is how a run looks right and is wrong.
from _cli import check_flags, flag as _flag, switch   # noqa: E402


@dataclass
class NeighbourTable:
    """The rankings of the season `offset` away from the scored one, read from a table, fit nothing."""
    name: str
    table: pd.DataFrame
    offset: int
    seasons: frozenset

    def train_for(self, h, ctx):
        s = int(h) + self.offset
        return [s] if s in self.seasons else None

    def fit(self, train, ctx):
        t = self.table[self.table.season == train[0]]
        return Ratings(t[["player_id", "o", "d", "poss"]].reset_index(drop=True))


def load_table(path: Path, columns: str = "rating") -> pd.DataFrame:
    """`columns="rating"` tests the finished rankings; `"prior"` tests the prior alone (`prior_off` / `prior_def`),
    which is how the two stages of a pipeline are told apart on this test."""
    t = pd.read_parquet(path)
    need = ["player_id", "season", f"{columns}_off", f"{columns}_def", "poss_off"]
    missing = [c for c in need if c not in t.columns]
    if missing:
        raise SystemExit(f"{path}: missing {missing}")
    # `Ratings` takes raw sign: offense positive-good, defense = points allowed, lower is better
    return pd.DataFrame({"player_id": t.player_id.astype(np.int64), "season": t.season.astype(int),
                         "o": t[f"{columns}_off"].astype(float), "d": -t[f"{columns}_def"].astype(float),
                         "poss": t.poss_off.astype(float)})


def _filled(system, spec):
    """`spec` None = the criterion's 0 fill; "<max_poss>" = ReplacementSystem; "<off>/<def>" = fixed,
    positive-good (a defensive rating is stored points-allowed, so its sign flips going in)."""
    if not spec:
        return system
    if "/" in spec:
        o, d = (float(x) for x in spec.split("/"))
        return _FixedFill(system.name, system, o, -d)
    poss, _, shrink = spec.partition("x")
    return ReplacementSystem(system.name, system, max_poss=float(poss), shrink=float(shrink or 1.0))


@dataclass
class _FixedFill:
    """`inner` with the absent player's rating set by hand, raw sign."""
    name: str
    inner: object
    fill_o: float
    fill_d: float

    def train_for(self, h, ctx):
        return self.inner.train_for(h, ctx)

    def fit(self, train, ctx):
        return Ratings(self.inner.fit(train, ctx).df, fill_o=self.fill_o, fill_d=self.fill_d)


def main():
    check_flags()      # refuse a flag this script does not understand (scripts/_cli.py)
    cfg = load_config(ROOT / "config.yaml")
    spec = _flag("rankings")
    if not spec:
        raise SystemExit("--rankings=name=path[,name=path...] is required")
    columns = _flag("columns", "rating")
    tables = {}
    for part in spec.split(","):
        name, path = part.split("=", 1)
        tables[name] = load_table(ROOT / path, columns)
    ref = _flag("ref", next(iter(tables)))
    split_names = [s for s in _flag("splits", "movers").split(",") if s]
    tag = _flag("tag", "yoy")
    first, last = int(_flag("first", cfg["holdout"]["first"])), int(_flag("last", cfg["holdout"]["last"]))

    fills = {}
    for part in (_flag("fill", "") or "").split(","):
        if part:
            name, spec = part.rsplit(":", 1)
            fills[name] = spec
    unknown = set(fills) - set(tables)
    assert not unknown, f"--fill names not in --rankings: {sorted(unknown)}"
    systems = []
    offset = int(_flag("offset", "1"))
    if offset != 1:
        print(f"offset {offset}: each season is predicted by the rankings {offset} seasons before and after it")
    for name, t in tables.items():
        for d, off in directions(offset).items():
            sysm = NeighbourTable(f"{name}:{d}", t, off, frozenset(t.season.unique()))
            systems.append(_filled(sysm, fills.get(name)))
    for name, spec in fills.items():
        print(f"fill {name}: {spec if '/' in spec else f'mean rating of the table players under {spec} possessions'}")
    ctx = Context.load(cfg)
    ho = Holdout.from_config(cfg, first=first, last=last, ks=[1], lam=[0.0])
    t0 = time.time()
    res = ho.run(systems, ctx, splits={s: SPLITS[s] for s in split_names},
                 out=ROOT / "outputs" / f"yoy_{tag}.parquet")
    print(f"scored {res.held_out.nunique()} seasons in {time.time() - t0:.0f}s\n")

    # one frame per direction, and one with both directions as separate observations
    res["direction"] = res.system.str.rsplit(":", n=1).str[1]
    res["system"] = res.system.str.rsplit(":", n=1).str[0]
    both = res.copy()
    both["held_out"] = both.held_out + both.direction.map({"prev": 0.0, "next": 0.5})
    frames = {"the PREVIOUS season's rankings predict this season": res[res.direction == "prev"],
              "the NEXT season's rankings predict this season": res[res.direction == "next"],
              "both directions together (two observations per scored season)": both}

    pd.set_option("display.width", 250, "display.max_columns", 40, "display.precision", 4)
    cols = ["system", "seasons", "armse", "vs_no_ratings", "calib_side", "scale_off", "scale_def", "covered",
            "game_armse", "game_vs_no_ratings"]
    print("=== year-over-year, pooled over scored seasons, LOWER IS BETTER.")
    print("    game_armse: what a typical team-game misses by, points per 100.  armse: the same per stint.")
    print("    scale_*: what the scored season wants each side multiplied by (1 = calibrated, below 1 = too wide).")
    print("    calib_side: stint MSE after each side is multiplied by that scale -- the error that is left once")
    print("    amplitude is taken out, i.e. the part that is ranking; a diagnostic, never a score.")
    for title, frame in frames.items():
        P = pooled(frame)
        print(f"\n--- {title}")
        for (split, group), d in P.groupby(["split", "group"], sort=True):
            head = "" if split == "all" else f"    [{split} = {group}, share of possessions {d.share.iloc[0]:.2f}]"
            if head:
                print(head)
            print(d[cols].to_string(index=False))
        if len(tables) > 1:
            print(f"\n    paired by scored season against {ref!r}; mean_diff below zero = better than {ref!r};")
            print("    se = standard error of the mean difference; z = mean_diff / se; wins = seasons better")
            for value, label in (("tg", "team-game level"), ("mse", "stint level"),
                                 ("calib_side", "stint level with each side rescaled to the scored season")):
                t = paired(frame, ref, value)
                if len(t):
                    print(f"    -- {label}")
                    print(t.drop(columns=["ref", "value", "k", "lam"]).to_string(index=False))


if __name__ == "__main__":
    main()
