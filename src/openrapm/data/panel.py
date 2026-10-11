"""The season panel, read through the trust boundary.

The season panel (`paths.panel`, outputs/role_panel_season.parquet) holds one row per player, season and side:
his padded box-score rates, role, career, body and on-court numbers, the inputs of the box-score prior.
`trainable_panel` is the one way a build reads it for the prior: every row of a season whose Finals are not over is
dropped first (`seasons.drop_untrainable`), so no prior can be trained on a season still being played.
`panel_and_trainable_seasons` is step 5's way: its type model describes every player of the table, the season in
progress included, but trains only on the seasons the same gate keeps.
"""
from __future__ import annotations

import pandas as pd

from .config import resolve
from .seasons import drop_untrainable


def trainable_panel(cfg: dict) -> pd.DataFrame:
    """The season panel without the seasons nothing may be fitted on (none while no season is in progress)."""
    panel, _ = drop_untrainable(pd.read_parquet(resolve(cfg, "panel")), cfg, what="the season panel")
    return panel


def panel_and_trainable_seasons(cfg: dict) -> tuple[pd.DataFrame, set]:
    """(the whole season panel, the seasons a fit may train on): every season is read, only finished ones train."""
    panel = pd.read_parquet(resolve(cfg, "panel"))
    trainable, _ = drop_untrainable(panel, cfg, what="the season panel (the swap type model's training)")
    return panel, set(trainable.season.astype(int))
