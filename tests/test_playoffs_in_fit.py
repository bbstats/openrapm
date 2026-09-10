"""A season is its regular season and its playoffs, and nothing in the code may treat them as two.

Owner's rulings, 2026-09-10: the playoff delta is gone and "playoff games just join the fit like any
other games", then "RS + playoffs should be considered a single entity in our new version".

Before this, `phases=("RS",)` was the default in six places and each one was somewhere the playoffs
were silently dropped -- the training design, the box exposure's padded rates, the role inputs, the
shooter totals, and the rows the criterion scores.  DECISIONS.md records that two of the three fixes
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
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eracoef import design as design_mod  # noqa: E402
from eracoef.config import load_config  # noqa: E402
from eracoef.design import SEASON_PHASES  # noqa: E402
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


def test_every_phase_default_is_the_whole_season():
    from eracoef.designcache import build_window_cached
    assert SEASON_PHASES == ("RS", "PO")
    for fn, arg in ((build_window, "phases"), (build_window_cached, "phases"),
                    (default_loader, "phases"), (Context.design, "phases"),
                    (BoxExposure.__init__, "phases")):
        assert _default(fn, arg) == SEASON_PHASES, f"{fn.__qualname__} still defaults to regular season only"
    assert tuple(MspiFast("probe").phases) == SEASON_PHASES
    assert ROLE_PHASES == SEASON_PHASES      # minutes, starts and possessions behind the role prior
    assert SHOT_PHASES == SEASON_PHASES      # the shooter totals that price the luck-adjusted targets


def test_there_is_only_one_phase_constant():
    """A second one is how "single entity" comes apart: it starts as a deliberate exception and ends as
    two halves of the code disagreeing about what a season is."""
    extra = [n for n in dir(design_mod) if n.endswith("_PHASES") and n != "SEASON_PHASES"]
    assert not extra, f"a second phase constant appeared: {extra}"


@needs_data
def test_a_window_actually_contains_playoff_rows():
    season = max(int(p.stem.split("_")[0]) for p in _STINTS.glob("*_PO.parquet"))
    wd = build_window([season], CFG)
    phases = set(wd.games["phase"].unique())
    assert phases == {"RS", "PO"}, f"{season} built {phases}"
    n_po = int((wd.rows["phase"] == "PO").sum())
    assert n_po > 0, "playoff games are in the game table but no design row came from them"
    # asking for one phase is still possible; it is just never the default
    rs = build_window([season], CFG, phases=("RS",))
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


@needs_data
def test_the_cut_trains_on_no_playoff_game_and_scores_every_one():
    """The two halves of "one entity", on the same real season.

    A cut q splits the anchor season into what the fit may see and what it is scored on.  Its playoff
    games belong on the scored side: they happen after every regular-season game, so a fit that saw the
    first q of the season has not seen them.  Until 2026-09-10 `season_frac` gave them -1, which put them
    below every cut -- excluded from training by a special case AND excluded from scoring by the same
    comparison, so they were in neither half.
    """
    from eracoef.holdout import cut_season
    from eracoef.inseason import kernel_game_mult, season_frac
    season = max(int(p.stem.split("_")[0]) for p in _STINTS.glob("*_PO.parquet"))
    wd = build_window([season], CFG)
    po_idx = wd.games.loc[wd.games["phase"] == "PO", "game_idx"].to_numpy()
    assert len(po_idx) > 0

    frac = season_frac(wd.games)
    assert (frac[po_idx] == 1.0).all()

    for q in (0.0, 0.25, 0.75):
        gm = kernel_game_mult(wd, season, {0: 1.0}, q)
        assert (gm[po_idx] == 0.0).all(), f"a playoff game carried training weight at q={q}"
        sub = cut_season(wd, season, q)
        assert (sub.rows["phase"] == "PO").sum() > 0, f"no playoff row survived the cut at q={q}"
    # and with no cut the playoffs train like any other game
    assert (kernel_game_mult(wd, season, {0: 1.0}, None)[po_idx] == 1.0).all()
