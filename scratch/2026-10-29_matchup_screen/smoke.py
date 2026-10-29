"""Smoke check of check 155's new pieces: the screen seasons, the pool lists, the plans."""
import importlib.util
import sys

spec = importlib.util.spec_from_file_location("screen155", "scripts/checks/155_input_screen.py")
m = importlib.util.module_from_spec(spec)
sys.modules["screen155"] = m
spec.loader.exec_module(m)
print(m.screen_seasons("screen120"))
print(m.candidates("screen120"))
lists = m.trial_lists(m.Trial("pool", "-", "screen120", 0))
print(len(lists["O"]), len(lists["D"]), lists["O"][-2:], lists["D"][-7:])
print(len(m.plan("screen120", 3, 5)), len(m.plan("screen119", 3, 5, "tune120", True)),
      m.plan("screen119", 3, 5, "tune120", True)[7].key, m.plan("screen120", 3, 5)[7].key)
print(len(m.plan("screen112", 3, 5, "out_of_bag60")))
