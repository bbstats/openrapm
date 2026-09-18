"""A registry that is missing systems must say so.

`systems.registry` built its 1,566 systems inside one `try: ... except ImportError: pass` that ran
for 880 lines and covered twenty imports.  It was written for the first of them -- `from . import
xshoot`, "the shooter-level targets, once built" -- and it swallowed the other nineteen too, so an
ImportError anywhere in the chain truncated the registry at that point and returned it.  The system
you then asked for came back "unknown systems [...]", which reads like a typo in your command rather
than a broken install, and nothing anywhere said an import had failed.

The chain is now `_shooter_chain`, and the guard is around a one-line probe of `xshoot` alone:

  * `xshoot` missing -- the case the guard was written for -- warns and returns the short registry;
  * anything else missing raises, because half a registry with no explanation is worse than a stack
    trace that names the module.

These two tests are that distinction.  They fake the import failure rather than requiring a broken
install, which is the only way to exercise a path that should never be reached.
"""
import builtins
import sys
import warnings
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eracoef.config import load_config  # noqa: E402

CFG = load_config()
_REAL_IMPORT = builtins.__import__


def _registry_without(monkeypatch, missing: str):
    """`systems.registry(cfg)` in a world where importing `missing` raises ImportError."""
    def blocked(name, globs=None, locs=None, fromlist=(), level=0):
        if name == missing or (level and fromlist and missing in fromlist):
            raise ImportError(f"no module named {missing} (simulated by {__name__})")
        return _REAL_IMPORT(name, globs, locs, fromlist, level)

    for module in [m for m in list(sys.modules) if m.startswith("eracoef")]:
        monkeypatch.delitem(sys.modules, module, raising=False)
    import eracoef.systems as systems
    monkeypatch.setattr(builtins, "__import__", blocked)
    return systems.registry(CFG)


def test_a_full_registry_is_built_without_warning():
    import eracoef.systems as systems
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        S = systems.registry(CFG)
    assert len(S) > 1000, f"only {len(S)} systems; something is truncating the registry"
    assert not [w for w in caught if issubclass(w.category, RuntimeWarning)]


def test_the_optional_import_warns_and_names_what_is_missing(monkeypatch):
    """xshoot is the one the guard exists for: a short registry is the right answer, silence is not."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        S = _registry_without(monkeypatch, "xshoot")
    assert len(S) < 100, "the chain should not have run without xshoot"
    said = [str(w.message) for w in caught if issubclass(w.category, RuntimeWarning)]
    assert said, "the registry was truncated and said nothing -- that is the whole defect"
    assert "xshoot" in said[0] and str(len(S)) in said[0]


def test_any_other_broken_import_raises_instead_of_truncating(monkeypatch):
    """The case the wide guard hid.  `context` is imported ~780 lines into the chain, so under the old
    guard this returned 1,509 of 1,566 systems and looked like a clean run."""
    with pytest.raises(ImportError, match="context"):
        _registry_without(monkeypatch, "context")
