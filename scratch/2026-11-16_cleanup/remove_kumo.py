"""The Kumo box-score prior out (experiment 118: worse than LightGBM in 10 of 10 seasons on the within-season WOWY
test; the owner, 2026-11-16: "Take all three out").  Kept at the tag archive/research-2026-11."""
from pathlib import Path


def load(path):
    raw = Path(path).read_bytes()
    return raw.decode("utf-8").replace("\r\n", "\n"), b"\r\n" in raw


def save(path, text, crlf):
    Path(path).write_bytes((text.replace("\n", "\r\n") if crlf else text).encode("utf-8"))
    print("edited", path)


def swap(text, old, new, count=1):
    assert text.count(old) == count, (old[:80], text.count(old))
    return text.replace(old, new)


def cut(text, start_marker, end_marker):
    start = text.index(start_marker)
    end = text.index(end_marker, start)
    return text[:start] + text[end:]


path = "src/openrapm/box_prior.py"
t, crlf = load(path)
t = swap(t, """A params file naming the Kumo learner (experiment 118) puts an in_context_prior.InContextPrior where each booster
stands: a pretrained network shown the same training rows at answer time instead of trees fitted to them
(in_context_prior.py says how the row weights pick the rows it is shown).

""", "")
t = swap(t, "from . import in_context_prior\n", "")
t = swap(t, '"ClippedBooster", "fit_lightgbm", "fit_booster",', '"ClippedBooster", "fit_lightgbm",')
t = cut(t, "def fit_booster(params: dict,", "@dataclass")
t = swap(t, "fit_booster(", "fit_lightgbm(", count=3)
save(path, t, crlf)

path = "src/openrapm/data/config.py"
t, crlf = load(path)
t = swap(t, """# the box-score prior's learners a params/ file may name: LightGBM, shipped, and the pretrained Kumo network
LEARNERS = ("lightgbm", "kumo_tabular")""", """# the box-score prior's learner a params/ file names: LightGBM (the Kumo network of experiment 118 was removed on
# 2026-11-16, at the tag archive/research-2026-11)
LEARNERS = ("lightgbm",)""")
t = swap(t, """    \"\"\"{"O": settings, "D": settings} from a tracked params/ file: each side's LightGBM settings as they are, or, for a
    side naming the Kumo learner (in_context_prior.py, experiment 118), its settings with `learner` added, which is
    how box_prior.fit_booster tells the two apart.
""", """    \"\"\"{"O": settings, "D": settings} from a tracked params/ file: each side's LightGBM settings as they are.
""")
t = swap(t, '        out[side] = params if learner == "lightgbm" else {**params, "learner": learner}\n',
         "        out[side] = params\n")
save(path, t, crlf)

path = "scripts/checks/155_input_screen.py"
t, crlf = load(path)
t = swap(t, """            # experiment 118 (2026-10-28): the pretrained Kumo network in place of LightGBM, no bags; a random draw
            # picks each member's training rows (in_context_prior.py)
            "kumo_small": dict(booster_file="params/booster_params_kumo_small.json", bags=0),
            "kumo_medium": dict(booster_file="params/booster_params_kumo_medium.json", bags=0),
            # its second look: every training row the same chance to be shown, whatever its weight
            "kumo_small_equal": dict(booster_file="params/booster_params_kumo_small_equal.json", bags=0),
""", "")
t = swap(t, """            for side in ("O", "D"):
                prime_in_context(saved[side], [ws.prior_rows(world.panel, side, saved[side])
                                               for world in self.run.worlds.values()])
""", "")
t = cut(t, "def prime_in_context(side_saved: dict, rows: list) -> None:", "_FIT = {}")
t = swap(t, "    from openrapm.in_context_prior import LEARNER\n", "")
t = swap(t, """    if params.get("learner") == LEARNER:
        # no bags to draw: the seed picks the Kumo members' training rows instead
        return fit_bags(train, names + ROW_FEATURES, {**params, "draw": trial.seed}, settings, 0)
""", "")
save(path, t, crlf)

path = "tests/test_check_155.py"
t, crlf = load(path)
t = swap(t, '    assert screen.same_inputs_per_tree({"learner": "kumo_tabular"}, 20, 21) == {"learner": "kumo_tabular"}',
         '    assert screen.same_inputs_per_tree({"learning_rate": 0.05}, 20, 21) == {"learning_rate": 0.05}')
save(path, t, crlf)

path = "tests/test_layer_boundary.py"
t, crlf = load(path)
lines = [line for line in t.split("\n") if not line.lstrip().startswith('"in_context_prior.py",')]
save(path, "\n".join(lines), crlf)

path = "pyproject.toml"
t, crlf = load(path)
t = swap(t, """# the box-score prior from NVIDIA's pretrained Kumo Tabular network (openrapm.in_context_prior, experiment 118), on a
# CUDA GPU with PyTorch's GPU build as above; the shipped LightGBM prior does not need it
kumo = ["structured-data-models>=0.2.1", "torch>=2.7"]
""", "")
save(path, t, crlf)

for gone in ("src/openrapm/in_context_prior.py", "tests/test_in_context_prior.py",
             "params/booster_params_kumo_small.json", "params/booster_params_kumo_medium.json",
             "params/booster_params_kumo_small_equal.json"):
    Path(gone).unlink()
    print("removed", gone)
