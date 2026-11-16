"""The elastic-net blend out of check 155 and the package (experiments 121-122: worse than today's prior in 4 of 4
seasons of the quick screen at every share; the owner, 2026-11-16: "Take all three out").  Kept at the tag
archive/research-2026-11.  The input lists of those screens stay in features.py: they are lists, not the blend."""
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


path = "scripts/checks/155_input_screen.py"
t, crlf = load(path)
t = swap(t, """# Experiment 121's quick screen (the owner, 2026-10-30: "let's just screen it on a smaller subset so it doesn't take as
# long"): four seasons, one an era and 2025, far enough apart that each season's step 4 reads the other three; one
# repeat of four held-back game sets a season; no next-season check
QUICK_SEASONS = (2001, 2010, 2019, 2025)
""", "")
t = swap(t, """           "screen120": ("screen120_pool", "SCREEN_CANDIDATES_MATCHUPS", MATCHUP_SEASONS),
           # experiment 121: the box-score prior blended with a line on every input and every product of two
           "blend121": ("everything121", None, QUICK_SEASONS),
           # experiment 122: the same with the booster on what Boruta keeps
           "blend122": ("everything121", None, QUICK_SEASONS)}
QUICK = {"blend121", "blend122"}                    # tags read on one repeat of held-back game sets, no year-over-year
CACHE_FROM = {"blend121": "screen120", "blend122": "screen120"}     # tags reading another tag's cache
# a blend tag: (the booster's feature set, the line's shares of the box-score prior, the tag whose lines it reads)
BLENDS = {"blend121": ("passed121", (0.0, 0.25, 0.5, 0.75, 1.0), "blend121"),
          "blend122": ("selected122", (0.0, 0.1, 0.25, 0.5), "blend121")}
PENALTY_SEASON = 2016                               # the season whose training rows choose the line's two penalties
L1_SHARES = (0.1, 0.5, 0.9)                         # the elastic net's lasso shares tried
N_ALPHAS = 10                                       # penalties tried for each, from the one that zeroes every column
""", """           "screen120": ("screen120_pool", "SCREEN_CANDIDATES_MATCHUPS", MATCHUP_SEASONS)}
""")
t = swap(t, '''    """{side: the inputs a tag screens, each appended alone to its side's shipped list}; none for a blend's tag."""
    from openrapm import features
    pool(tag)
    if SCREENS[tag][1] is None:
        return {"O": [], "D": []}
    return getattr(features, SCREENS[tag][1])''', '''    """{side: the inputs a tag screens, each appended alone to its side's shipped list}."""
    from openrapm import features
    pool(tag)
    return getattr(features, SCREENS[tag][1])''')
t = cut(t, "def cache_dir(tag: str) -> Path:", "def within_dir() -> Path:")
t = swap(t, "cache_dir(", "screen_dir(", count=t.count("cache_dir("))
t = swap(t, """    arm: str                     # base, add, add_shadow, shadow, pool, blend
    side: str                    # "O", "D", or "-" for base, pool and blend
    name: str                    # the input, or "-" for base; a pool trial's tag; a blend's weight, as "w25"
""", """    arm: str                     # base, add, add_shadow, shadow, pool
    side: str                    # "O", "D", or "-" for base and pool
    name: str                    # the input, or "-" for base; a pool trial's tag
""")
t = cut(t, "def weight_name(weight: float) -> str:", "def trial_lists(trial: Trial) -> dict:")
t = swap(t, '''    """{side: the boosters' inputs} of a trial; a blend's are every input, of which the booster reads its own."""
    from openrapm.features import EVERYTHING_121
    lists = shipped_lists()''', '''    """{side: the boosters' inputs} of a trial."""
    lists = shipped_lists()''')
t = swap(t, """    if trial.arm == "blend":
        lists = {side: list(EVERYTHING_121) for side in ("O", "D")}
    return lists""", """    return lists""")
t = swap(t, "        self.quick = tag in QUICK\n", "")
t = swap(t, """        file `fit_task` wrote it to}); return the whole season's step-3 table (raw sign), or nothing on a quick tag,
        which rates one repeat of held-back game sets and no whole season.  The boosters are let go after, so a worker
        holds one season's at a time.\"\"\"""", """        file `fit_task` wrote it to}); return the whole season's step-3 table (raw sign).  The boosters are let go
        after, so a worker holds one season's at a time.\"\"\"""")
t = swap(t, "            for repeat in range(1 if self.quick else self.run.repeats):\n",
         "            for repeat in range(self.run.repeats):\n")
t = swap(t, "            whole = None if self.quick else self.run.rate(set(self.run.game_ids))\n",
         "            whole = self.run.rate(set(self.run.game_ids))\n")
t = swap(t, "        return pd.DataFrame() if whole is None else whole.table.assign(season=self.season)\n",
         "        return whole.table.assign(season=self.season)\n")
t = swap(t, """    if trial.arm == "blend":
        return fit_blend(tag, season, side, trial, train, params, settings)
""", "")
t = cut(t, "# ------------------------------------------------------------------------------------------------- the blend",
        "class Fitters:")
t = swap(t, """    if SCREENS[tag][1] is None:
        # a blend's tag: today's prior and every weight at each seed, seed by seed
        return [trial for seed in range(seeds)
                for trial in [Trial("base", "-", "-", seed)] + [Trial("blend", "-", weight_name(weight), seed)
                                                                for weight in BLENDS[tag][1]]]
""", "")
t = swap(t, "    guard = None if args.tag in QUICK else Guard(args.tag, seasons)\n",
         "    guard = Guard(args.tag, seasons)\n")
t = cut(t, "def run_linear(args) -> int:", "def main(argv=None) -> int:")
t = swap(t, """    linear = commands.add_parser("linear", help="a blend tag's line: its penalties, then each season's; resumes")
    linear.add_argument("--tag", required=True)
    linear.add_argument("--processes", type=int, default=4)
    blends = commands.add_parser("blends", help="a blend tag's arms against today's prior")
    blends.add_argument("--tag", required=True)
""", "")
t = swap(t, """    if args.command == "linear":
        return run_linear(args)
    if args.command == "blends":
        return run_blends(args)
""", "")
for gone in ("QUICK", "BLENDS", "CACHE_FROM", "linear_prior", "fit_blend", "blend_weight", "EVERYTHING_121"):
    assert gone not in t, gone
save(path, t, crlf)

path = "tests/test_layer_boundary.py"
t, crlf = load(path)
lines = [line for line in t.split("\n") if not line.lstrip().startswith('"linear_prior.py",')]
save(path, "\n".join(lines), crlf)

for gone in ("src/openrapm/linear_prior.py", "tests/test_linear_prior.py"):
    Path(gone).unlink()
    print("removed", gone)
