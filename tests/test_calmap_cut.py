"""The scoring frames must match the cut the fits were made at (calmap.frames_for_dump).

An in-season system is fit on the first `cut` of the held-out season.  Score it on the whole season
and it is handed back its own training games; the shorter the kernel, the larger that leak, so the
comparison it is in reads backwards.  scripts/53_calmap.py did exactly this until 2026-09-10 --
`54_track.py` and `57_investigate.py` always passed the cut, and `fit` never did.
"""
import numpy as np
import pandas as pd
import pytest

from eracoef import calmap


@pytest.fixture
def dump():
    """Two dumped systems, one fit at q75 and one uncut, in the schema `cut_of` reads."""
    rows = []
    for system, cut in (("insea_q75", 0.75), ("block", np.nan)):
        for h in (2002, 2003):
            rows.append(dict(system=system, k=3, held_out=h, player_id=1, o=0.0, d=0.0, poss=1000.0,
                             prior_o=0.0, prior_d=0.0, fill_o=0.0, fill_d=0.0, cut=cut))
    return pd.DataFrame(rows)


def _spy(monkeypatch):
    seen = []
    monkeypatch.setattr(calmap, "load_frames",
                        lambda ctx, seasons, level="home", verbose=True, cut=None: seen.append(cut) or {})
    return seen


def test_an_in_season_dump_is_scored_after_its_cut(dump, monkeypatch):
    seen = _spy(monkeypatch)
    calmap.frames_for_dump(None, [2002, 2003], dump, ["insea_q75"], level="home", verbose=False)
    assert seen == [0.75]


def test_a_dump_with_no_cut_scores_the_whole_season(dump, monkeypatch):
    seen = _spy(monkeypatch)
    calmap.frames_for_dump(None, [2002, 2003], dump, ["block"], level="home", verbose=False)
    assert seen == [None]


def test_mixing_cuts_in_one_run_is_an_error(dump, monkeypatch):
    """Different cuts mean different scored games, so the paired test would not be paired."""
    _spy(monkeypatch)
    with pytest.raises(ValueError, match="different cuts"):
        calmap.frames_for_dump(None, [2002, 2003], dump, ["insea_q75", "block"], verbose=False)


def test_the_fit_script_takes_its_frames_from_the_dump():
    """The regression itself: 53_calmap.py must not call the uncut loader."""
    from pathlib import Path
    src = Path(__file__).resolve().parents[1] / "scripts" / "53_calmap.py"
    text = src.read_text(encoding="utf-8")
    assert "frames_for_dump(" in text
    assert "load_frames(" not in text
