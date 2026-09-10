"""The playoffs are in the fit, and every default that feeds one says so.

Owner's ruling, 2026-09-10: the playoff delta is gone and "playoff games just join the fit like any
other games".  Before it, `phases=("RS",)` was the default in six places and each one was somewhere
the playoffs were silently dropped -- the training design, the box exposure's padded rates, the role
inputs, the shooter totals.  DECISIONS.md records that two of the three fixes needed to make the old
playoff delta work were bugs of exactly that shape in shared code.

The defaults are the thing worth pinning.  A caller who wants regular-season rows can always ask for
them; what goes wrong is nobody asking and a default quietly answering "RS".  So the first test
below reads the signatures and fails if any of them drifts back.

`SCORE_PHASES` is the other half of the ruling's edge.  The rows the criterion SCORES are still
regular season only, deliberately: the baselines in HANDOFF.md were measured on that estimand and
moving the yardstick in the same change that moves the fit would make the two indistinguishable.
It is pinned here so that stays a decision rather than an accident.
"""
import inspect
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eracoef.config import load_config  # noqa: E402
from eracoef.design import FIT_PHASES, SCORE_PHASES  # noqa: E402
from eracoef.exposure import BoxExposure  # noqa: E402
from eracoef.fastfit import MspiFast  # noqa: E402
from eracoef.holdout import Context, default_loader  # noqa: E402
from eracoef.roles import ROLE_PHASES  # noqa: E402
from eracoef.windows import build_window  # noqa: E402
from eracoef.xshoot import SHOT_PHASES  # noqa: E402

CFG = load_config()
_STINTS = Path(CFG["_root"]) / CFG.get("paths", {}).get("stints", "data/stints")
HAVE_DATA = _STINTS.is_dir() and any(_STINTS.glob("*_PO.parquet"))
needs_data = pytest.mark.skipif(not HAVE_DATA, reason="no stints; run scripts/02_stints.py")


def _default(fn, name):
    p = inspect.signature(fn).parameters[name]
    assert p.default is not inspect.Parameter.empty, f"{fn.__qualname__}.{name} has no default"
    return tuple(p.default)


def test_every_fit_default_carries_the_playoffs():
    from eracoef.designcache import build_window_cached
    assert FIT_PHASES == ("RS", "PO")
    for fn, arg in ((build_window, "phases"), (build_window_cached, "phases"),
                    (default_loader, "phases"), (Context.design, "phases"),
                    (BoxExposure.__init__, "phases")):
        assert _default(fn, arg) == FIT_PHASES, f"{fn.__qualname__} still defaults to regular season only"
    assert tuple(MspiFast("probe").phases) == FIT_PHASES
    assert ROLE_PHASES == FIT_PHASES        # minutes, starts and possessions behind the role prior
    assert SHOT_PHASES == FIT_PHASES        # the shooter totals that price the luck-adjusted targets


def test_the_scored_rows_are_still_regular_season():
    """Not an oversight: the criterion's estimand is the one HANDOFF.md's baselines were measured on."""
    assert SCORE_PHASES == ("RS",)
    assert SCORE_PHASES != FIT_PHASES


@needs_data
def test_a_window_actually_contains_playoff_rows():
    season = max(int(p.stem.split("_")[0]) for p in _STINTS.glob("*_PO.parquet"))
    wd = build_window([season], CFG)
    phases = set(wd.games["phase"].unique())
    assert phases == {"RS", "PO"}, f"{season} built {phases}"
    n_po = int((wd.rows["phase"] == "PO").sum())
    assert n_po > 0, "playoff games are in the game table but no design row came from them"
    # and asking for the scored estimand still gets regular season only
    rs = build_window([season], CFG, phases=SCORE_PHASES)
    assert set(rs.games["phase"].unique()) == {"RS"}
    assert len(rs.rows) < len(wd.rows)


@needs_data
def test_role_inputs_count_playoff_minutes():
    """A rotation player on a team that went deep played more games than his regular season says."""
    from eracoef.roles import season_roles
    season = max(int(p.stem.split("_")[0]) for p in _STINTS.glob("*_PO.parquet"))
    both = season_roles(season, CFG)
    rs = season_roles(season, CFG, phases=("RS",))
    j = rs.merge(both, on=["player_id", "team_id"], suffixes=("_rs", "_all"))
    assert (j.games_all >= j.games_rs).all()
    assert (j.games_all > j.games_rs).any()
    assert (j.poss_on_all >= j.poss_on_rs - 1e-6).all()
    # the share is a share: pooling both phases moves the league's mean poss_pct very little
    assert abs(both.poss_on.sum() / both.team_poss.sum() - rs.poss_on.sum() / rs.team_poss.sum()) < 0.02
