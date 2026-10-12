"""A season is its regular season and its playoffs, and nothing in the code may treat them as two.

Owner's rulings, 2026-09-10: the playoff delta is gone and "playoff games just join the fit like any
other games", then "RS + playoffs should be considered a single entity in our new version".

Before this, `phases=("RS",)` was the default in six places and each one was somewhere the playoffs
were silently dropped -- the training design, the box exposure's padded rates, the role inputs, the
shooter totals, and the rows the criterion scores.  notes/DECISIONS.md records that two of the three fixes
needed to make the old playoff delta work were bugs of exactly that shape in shared code.

The defaults are the thing worth pinning.  A caller who wants regular-season rows can always ask for
them; what goes wrong is nobody asking and a default quietly answering "RS".  So the first test below
reads the signatures and fails if any of them drifts back, and the second fails if a second phase
constant reappears -- there is one, `design.SEASON_PHASES`, and everything else imports it.

What is NOT a contradiction: the design keeps its `is_po` and `po_home` fixed columns.  Those are
level controls on the environment, the same kind of thing as the per-season intercepts, and dropping
them would make the playoffs' scoring level something the fit has to explain with the players on the
floor -- who in the playoffs are disproportionately the good ones.
"""
import inspect

import pytest

from openrapm import design as design_mod
from openrapm.data.config import load_config, resolve
from openrapm.data.designs import block_design
from openrapm.data.roles import ROLE_PHASES
from openrapm.data.shots import SHOT_PHASES
from openrapm.design import SEASON_PHASES

CFG = load_config()
_STINTS = resolve(CFG, "stints")


def _default(fn, name):
    p = inspect.signature(fn).parameters[name]
    assert p.default is not inspect.Parameter.empty, f"{fn.__qualname__}.{name} has no default"
    return tuple(p.default)


def test_every_phase_default_is_the_whole_season():
    assert SEASON_PHASES == ("RS", "PO")
    assert _default(block_design, "phases") == SEASON_PHASES, "block_design still defaults to regular season only"
    assert ROLE_PHASES == SEASON_PHASES      # minutes, starts and possessions behind the role prior
    assert SHOT_PHASES == SEASON_PHASES      # the shooter totals that price the luck-adjusted targets


def test_there_is_only_one_phase_constant():
    """A second one is how "single entity" comes apart: it starts as a deliberate exception and ends as
    two halves of the code disagreeing about what a season is."""
    extra = [n for n in dir(design_mod) if n.endswith("_PHASES") and n != "SEASON_PHASES"]
    assert not extra, f"a second phase constant appeared: {extra}"


@pytest.mark.needs_data("stints/*_PO.parquet")
def test_a_window_actually_contains_playoff_rows():
    season = max(int(p.stem.split("_")[0]) for p in _STINTS.glob("*_PO.parquet"))
    wd = block_design([season], CFG)
    phases = set(wd.games["phase"].unique())
    assert phases == {"RS", "PO"}, f"{season} built {phases}"
    n_po = int((wd.rows["phase"] == "PO").sum())
    assert n_po > 0, "playoff games are in the game table but no design row came from them"
    # asking for one phase is still possible; it is just never the default
    rs = block_design([season], CFG, phases=("RS",))
    assert set(rs.games["phase"].unique()) == {"RS"}
    assert len(rs.rows) < len(wd.rows)


@pytest.mark.needs_data("stints/*_PO.parquet")
def test_role_inputs_count_playoff_minutes():
    """A rotation player on a team that went deep played more games than his regular season says."""
    from openrapm.data.roles import season_roles
    season = max(int(p.stem.split("_")[0]) for p in _STINTS.glob("*_PO.parquet"))
    both = season_roles(season, CFG)
    rs = season_roles(season, CFG, phases=("RS",))
    j = rs.merge(both, on=["player_id", "team_id"], suffixes=("_rs", "_all"))
    assert (j.games_all >= j.games_rs).all()
    assert (j.games_all > j.games_rs).any()
    assert (j.poss_on_all >= j.poss_on_rs - 1e-6).all()
    # the share is a share: pooling both phases moves the league's mean poss_pct very little
    assert abs(both.poss_on.sum() / both.team_poss.sum() - rs.poss_on.sum() / rs.team_poss.sum()) < 0.02
