"""A rating as the prior for another rating: regular season -> playoffs.

The owner, 2026-09-09: *"SPM is the prior for reg season PI RAPM, then PI RAPM is the playoff prior for
playoff PI'' RAPM.  just not sure how to include playoff stats."*

The answer to the second half is that no playoff box score is needed at all.  The chain already ends in a
number per player per side, so that number becomes the OFFSET for a third fit that sees only playoff
possessions, and the ridge decides how far a run moves him:

    box score  ->  Simple SPM  ->  regular-season PI-RAPM  ->  playoff PI-RAPM
                                        (the prior)              (the offset + what the playoffs add)

Nothing else changes.  The offset enters exactly where the boosted box prior enters on the regular-season
fit (`fastfit.MspiFast.prior_from`), the same exposure and target machinery runs, and the playoff rating
is `regular-season rating + playoff residual` by construction.

**What the arithmetic means.**  A three-season block holds about 47,700 playoff possessions against
734,000 regular-season ones -- 6.5%.  So at any sane penalty most players will barely move, and the ones
who move are the ones who played playoff minutes.  That is the correct behaviour and it is also the
warning: a playoff rating is mostly its prior, and any claim that it has "found" a playoff riser has to
clear that.  `scratch/playoff_chain.py` is the test -- fit the update on one half of each series and
predict the other half, against the regular-season rating alone.

**The one free parameter** is the penalty on the playoff residual (`lam_po`).  It is what decides how much
a run is allowed to say.  Sweep it; do not pick it.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .holdout import Context, Ratings


def align_prior(rs: Ratings, wd, fill_o: float = 0.0, fill_d: float = 0.0) -> np.ndarray:
    """The regular-season ratings as a `2 * n_ps` offset on the PLAYOFF design's player layout.

    `player_unit: window` gives one row per player per block, so the join is on `player_id` alone and is
    exact.  A player with no regular-season rating -- traded in, or below the exposure floor -- gets the
    fill, which is 0 (the average player) unless the caller has a replacement level.
    """
    ps = wd.spec.ps_table
    if not ps["player_id"].is_unique:
        raise ValueError("align_prior needs one row per player in the design's ps_table "
                         "(config player_unit: window); got repeats, so the join would be ambiguous")
    r = rs.aligned(ps["player_id"].to_numpy())
    o = np.nan_to_num(r["o"].to_numpy(dtype=float), nan=float(fill_o))
    d = np.nan_to_num(r["d"].to_numpy(dtype=float), nan=float(fill_d))
    return np.concatenate([o, d])


@dataclass
class RegularSeasonPrior:
    """A `prior_from` callable: fit `inner` on the REGULAR season and hand its ratings over as the offset.

    The inner fit is cached per (train, id(ctx)) because a sweep over the playoff penalty re-uses the same
    regular-season ratings every time and they cost a minute each.
    """
    inner: object
    _cache: dict = None

    def __post_init__(self):
        if self._cache is None:
            self._cache = {}

    def ratings(self, train, ctx: Context) -> Ratings:
        key = (tuple(int(s) for s in train), id(ctx))
        if key not in self._cache:
            self._cache[key] = self.inner.fit(list(train), ctx)
        return self._cache[key]

    def __call__(self, train, ctx: Context, wd) -> np.ndarray:
        rs = self.ratings(train, ctx)
        return align_prior(rs, wd)


def playoff_system(inner, name: str, lam_po: float | None = None, lam_ratio: float | None = None,
                   phases=("PO",)):
    """`inner` fit on the regular season, then re-fit on playoff possessions with that as the prior.

    Everything about the estimator is `inner`'s -- target, exposure, calibration -- except the rows
    (`phases`) and the penalty on what the playoffs are allowed to add (`lam_po`, defaulting to `inner`'s
    own).  The prior chain of `inner` is REPLACED, not stacked: the regular-season rating already contains
    the box prior, so putting the box prior back would count it twice.
    """
    return replace(inner, name=name, phases=tuple(phases),
                   prior_from=RegularSeasonPrior(inner),
                   lam=float(lam_po) if lam_po is not None else inner.lam,
                   lam_ratio=float(lam_ratio) if lam_ratio is not None else inner.lam_ratio)
