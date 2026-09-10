"""Which seasons may be TRAINED on, and which may only be RATED.

This module is the trust boundary of the project.  The rule it enforces, in one sentence:

    Every season may be loaded and rated.  A season may be FIT on only once its Finals are over.

That covers every fitted quantity except the ridge itself -- the box prior, the role prior, the
calibration map, the padding constants, the feature lists, every hyper-parameter.  Those are the
things a contributor's pull request changes, and none of them may have seen a season in progress.
The ridge is exempt because a rating IS a ridge coefficient: solving it on the current season is the
product, not a fit.

Why the Finals and not the end of the regular season: a rating that is allowed to train on the first
three quarters of a season and then rate the fourth has seen the answer for every player whose team
is still playing.  The bright line has to be a point after which nothing more about the season can be
learned, and that is the last game of it.

Detecting it
------------
`playoffs_complete` counts each team's playoff wins and asks whether anyone reached the number that
wins a title: 16 from 2003 (four best-of-seven rounds) and 15 before it, when the first round was
best-of-five.  Verified against every season from 1997 to 2026 -- the maximum is exactly 15 through
2002 and exactly 16 from 2003, with no season in between.

This is a fact read from the data rather than a constant in `config.yaml`, so there is no year for
anyone to forget to bump.  In the 2026-27 season, `trainable_through` returns 2026 in October and
keeps returning it until the 2027 Finals end.

Using it
--------
    from .seasons import trainable, scoreable

    tr = trainable(cfg)                 # Trainable(seasons=(1997, ..., 2026), cutoff=2026)
    prior.fit(tr)                       # takes a Trainable and nothing else
    ratings = prior.rate(scoreable(cfg))    # every season, including the one in progress

`Trainable` refuses to hold a season past its own cutoff, so a fit cannot be handed one by accident.
That check is a courtesy, not the guarantee: the guarantee is `tests/test_no_current_season.py`,
which rebuilds every fitted artifact with the current season's data replaced by noise and asserts
that not one byte of them moves.
"""
from __future__ import annotations

import functools
from dataclasses import dataclass

import pandas as pd


class LeakageError(RuntimeError):
    """A fit was handed a season whose Finals are not over."""


def title_wins(season: int) -> int:
    """Playoff wins needed for the title: four best-of-seven rounds from 2003, and before that a
    best-of-five first round, so fifteen."""
    return 16 if int(season) >= 2003 else 15


@dataclass(frozen=True)
class Trainable:
    """Seasons a model may be fit on.  `cutoff` is the latest season whose Finals are over."""
    seasons: tuple[int, ...]
    cutoff: int

    def __post_init__(self):
        late = [s for s in self.seasons if int(s) > int(self.cutoff)]
        if late:
            raise LeakageError(
                f"seasons {late} are past the cutoff {self.cutoff}: their Finals are not over, so "
                f"nothing may be fit on them.  Rate them instead -- see src/eracoef/seasons.py.")

    def __iter__(self):
        return iter(self.seasons)

    def __len__(self):
        return len(self.seasons)

    def without(self, *held) -> "Trainable":
        """The same seasons minus `held` -- how a leave-one-season-out fit drops its held-out season."""
        drop = {int(h) for h in held}
        return Trainable(tuple(s for s in self.seasons if int(s) not in drop), self.cutoff)


@dataclass(frozen=True)
class Scoreable:
    """Seasons a fitted model may be applied to: all of them, including one in progress."""
    seasons: tuple[int, ...]

    def __iter__(self):
        return iter(self.seasons)

    def __len__(self):
        return len(self.seasons)


def playoffs_complete(season: int, cfg) -> bool:
    """Has any team won the number of playoff games that wins a title?"""
    from .ingest import raw_dir
    path = raw_dir(cfg) / "gamelog" / f"{int(season)}_PO.parquet"
    if not path.exists():
        return False
    gl = pd.read_parquet(path, columns=["TEAM_ID", "GAME_ID", "WL"])
    wins = gl[gl["WL"] == "W"].drop_duplicates(["TEAM_ID", "GAME_ID"]).groupby("TEAM_ID").size()
    return bool(len(wins) and int(wins.max()) >= title_wins(season))


def all_seasons(cfg) -> tuple[int, ...]:
    return tuple(range(int(cfg["first_season"]), int(cfg["last_season"]) + 1))


@functools.lru_cache(maxsize=8)
def _cutoff(first: int, last: int, root: str, raw: str) -> int:
    """Cached on the four things the answer depends on -- including the root, so a test that points
    at a temporary tree does not read the answer for the real one."""
    cfg = {"_root": root, "paths": {"raw": raw}}
    for s in range(int(last), int(first) - 1, -1):
        if playoffs_complete(s, cfg):
            return s
    raise LeakageError(
        f"no season between {first} and {last} has a completed playoffs under {root}, so there is "
        f"nothing to train on.  Run the ingest step first.")


def trainable_through(cfg) -> int:
    """The latest season whose Finals are over.  Read from the data, not from config."""
    return _cutoff(int(cfg["first_season"]), int(cfg["last_season"]), str(cfg["_root"]),
                   str(cfg.get("paths", {}).get("raw", "data/raw")))


def trainable(cfg) -> Trainable:
    """Every season a model may be fit on."""
    cut = trainable_through(cfg)
    return Trainable(tuple(s for s in all_seasons(cfg) if s <= cut), cut)


def scoreable(cfg) -> Scoreable:
    """Every season, including one in progress."""
    return Scoreable(all_seasons(cfg))


def in_progress(cfg) -> tuple[int, ...]:
    """The seasons that exist in the data but may not be trained on -- normally zero or one of them."""
    cut = trainable_through(cfg)
    return tuple(s for s in all_seasons(cfg) if s > cut)


def check_trainable(seasons, cfg, what: str = "this fit") -> Trainable:
    """Accept a `Trainable` unchanged; turn a bare sequence into one, raising if it reaches too far.

    The adapter for call sites that still pass lists.  Every one of them is a place the boundary is
    not yet a type, so `grep check_trainable` is the list of what is left to convert.
    """
    if isinstance(seasons, Trainable):
        return seasons
    cut = trainable_through(cfg)
    late = [int(s) for s in seasons if int(s) > cut]
    if late:
        raise LeakageError(
            f"{what} was given season(s) {late}, whose Finals are not over (the cutoff is {cut}).  "
            f"Only the ridge may see a season in progress; see src/eracoef/seasons.py.")
    return Trainable(tuple(int(s) for s in seasons), cut)


def unit_last_season(labels) -> "pd.Series":
    """The last season a panel unit covers, from its label.  "2024-2026" -> 2026, "2026" -> 2026."""
    lab = pd.Series(labels, dtype="object").astype(str)
    return lab.str.rsplit("-", n=1).str[-1].astype(int)


def drop_untrainable(panel: "pd.DataFrame", cfg, label_col: str = "window", what: str = "the panel"):
    """Panel rows a fit may use: those whose unit ENDS at or before the cutoff.

    A unit that reaches into a season in progress is dropped whole.  A three-season block ending in
    the current season carries that season's evidence in every one of its rows, so there is no
    honest way to keep part of it; a one-season row is simply that season.

    Returns (kept, dropped_labels).  The caller is expected to say what it dropped -- silence here
    is how a leak guard stops being noticed.
    """
    if panel is None or label_col not in getattr(panel, "columns", ()):
        return panel, ()
    cut = trainable_through(cfg)
    last = unit_last_season(panel[label_col])
    keep = last <= cut
    dropped = tuple(sorted(set(panel.loc[~keep, label_col].astype(str))))
    if dropped:
        print(f"  {what}: {int((~keep).sum())} rows dropped from {list(dropped)} -- past the "
              f"trainable cutoff {cut} (src/eracoef/seasons.py)", flush=True)
    return panel.loc[keep].reset_index(drop=True), dropped
