"""`openrapm site --decomposition` out: it wrote docs/data/decompose_1y.json and decompose_3y.json for the vanilla RAPM
page, retired 2026-10-20, and tests/test_site.py forbids those files.  The code stays at the tag
archive/research-2026-11.  The cleanup of 2026-11-16."""
import re
from pathlib import Path


def load(path):
    raw = Path(path).read_bytes()
    return raw.decode("utf-8").replace("\r\n", "\n"), b"\r\n" in raw


def save(path, text, crlf):
    Path(path).write_bytes((text.replace("\n", "\r\n") if crlf else text).encode("utf-8"))
    print("edited", path)


def cut(text, start_marker, end_marker):
    """`text` without the span from `start_marker` up to, not including, `end_marker`."""
    start, end = text.index(start_marker), text.index(end_marker)
    assert start < end, (start_marker, end_marker)
    return text[:start] + text[end:]


def swap(text, old, new):
    assert text.count(old) == 1, old[:80]
    return text.replace(old, new)


# ------------------------------------------------------------------ commands/site.py
path = "src/openrapm/commands/site.py"
t, crlf = load(path)
t = swap(t, """    openrapm site --decomposition    also docs/data/decompose_1y.json and decompose_3y.json, the vanilla RAPM
                                     decomposition page
""", "")
t = swap(t, """                    so re-exporting the same table does not tell readers the ratings changed; the decomposition
                    files show the day they were computed.  --built sets it for every file written
""", """                    so re-exporting the same table does not tell readers the ratings changed; the Similarity and
                    Bias files show the day they were computed.  --built sets it for every file written
""")
t = swap(t, "    vanilla RAPM    plain ridge regression with no box-score prior: raw points, regular season and playoffs\n", "")
t = swap(t, "import scipy.linalg as sla\nimport scipy.sparse as sp\n", "")
t = swap(t, "from ..data.designs import DesignCache, block_design\n", "from ..data.designs import DesignCache\n")
t = swap(t, "from ..decomposition import lineups, refit, row_groups, row_keys, side_team, split_games_part, team_codes\n",
         "from ..decomposition import lineups, refit, row_keys, split_games_part\n")
t = swap(t, "from ..rapm import SeasonSums\n", "")
# the page's comment block and constants, kept: the tolerance the Detail view reads
start = t.index("# ===================================================================================== the decomposition page")
end = t.index("class BoxScoreFiles:")
t = t[:start] + ("# rows whose team the box scores cannot name: under this share of a season's rows (the Detail view's check)\n"
                 "UNPLACED_TOLERANCE = 0.001\n\n\n") + t[end:]
t = cut(t, "def span_label(seasons) -> str:", "def name_table(")
t = cut(t, "def decomposition_row(r) -> list:", "def check(ok: bool, what: str) -> None:")
t = cut(t, "def run_checks(span_words: str, worst: dict) -> None:",
        "# ===================================================================================== the OpenRAPM decomposition")
t = swap(t, """def run_site(cfg: dict, ratings=None, out_dir=None, built: str | None = None, bias: bool = False,
             decomposition: bool = False, similarity: bool = False, openrapm_decomposition: bool = False,
             power: bool = False, power_forecast: bool = False) -> int:
    \"\"\"`openrapm site`: ratings.json always; the decomposition files, the similarity files, the OpenRAPM decomposition
    file, the power ratings file and the Bias page's file when asked.""", """def run_site(cfg: dict, ratings=None, out_dir=None, built: str | None = None, bias: bool = False,
             similarity: bool = False, openrapm_decomposition: bool = False,
             power: bool = False, power_forecast: bool = False) -> int:
    \"\"\"`openrapm site`: ratings.json always; the similarity files, the OpenRAPM decomposition file, the power ratings
    file and the Bias page's file when asked.""")
t = swap(t, """    if decomposition:
        write_decomposition(cfg, ratings_path, out_dir / "data", built or today())
""", "")
assert "decompose_1y" not in t and "span_fit" not in t and "SeasonSums" not in t
save(path, t, crlf)

# ------------------------------------------------------------------ commands/cli.py
path = "src/openrapm/commands/cli.py"
t, crlf = load(path)
t = swap(t, """    openrapm site      the site's data: docs/data/ratings.json, and the decomposition (vanilla RAPM and OpenRAPM),
                       similarity, power and bias pages' files when asked
""", """    openrapm site      the site's data: docs/data/ratings.json, and the Detail view's, similarity, power and bias
                       pages' files when asked
""")
t = swap(t, "                    decomposition=args.decomposition, similarity=args.similarity,\n",
         "                    similarity=args.similarity,\n")
t = swap(t, '''        "site", help="the site's data: docs/data/ratings.json, and the decomposition, similarity, power and bias "''',
         '''        "site", help="the site's data: docs/data/ratings.json, and the Detail view's, similarity, power and bias "''')
t = swap(t, '''"--decomposition also writes docs/data/decompose_1y.json and "
                    "decompose_3y.json, the retired vanilla RAPM page's; --openrapm-decomposition also "''',
         '''"--openrapm-decomposition also "''')
t = swap(t, '''    site.add_argument("--decomposition", action="store_true", help="also the decomposition page's two files")\n''', "")
t = swap(t, '''help="also the OpenRAPM decomposition page's file (rerun after every build)")''',
         '''help="also the Detail view's file and the ratings page's list (rerun after every build)")''')
save(path, t, crlf)

# ------------------------------------------------------------------ rapm.py: two methods only the page read
path = "src/openrapm/rapm.py"
t, crlf = load(path)
t = cut(t, "    def player_columns(self, season: int) -> np.ndarray:", "    def context_penalty(self, season: int,")
save(path, t, crlf)

# ------------------------------------------------------------------ decomposition.py's docstring
path = "src/openrapm/decomposition.py"
t, crlf = load(path)
first = t.split("\n\n")[1]
print("decomposition.py's second paragraph:", first[:200].replace("\n", " "))
save(path, t, crlf)

# ------------------------------------------------------------------ config.yaml: the windows only the page read
path = "config.yaml"
t, crlf = load(path)
t = re.sub(r"windows:[^\n]*\n(  - \[\d{4}, \d{4}\]\n)+", "", t, count=1)
assert "windows" not in t
save(path, t, crlf)

# ------------------------------------------------------------------ tests
path = "tests/test_cli.py"
t, crlf = load(path)
t = swap(t, "    assert seen == dict(ratings=None, out_dir=None, built=None, bias=False, decomposition=False,\n",
         "    assert seen == dict(ratings=None, out_dir=None, built=None, bias=False,\n")
t = swap(t, '''                     "--decomposition", "--similarity", "--openrapm-decomposition", "--power"]) == 0''',
         '''                     "--similarity", "--openrapm-decomposition", "--power"]) == 0''')
t = swap(t, "    assert seen == dict(ratings=\"r.parquet\", out_dir=\"out\", built=\"2026-10-09\", bias=True, decomposition=True,\n",
         "    assert seen == dict(ratings=\"r.parquet\", out_dir=\"out\", built=\"2026-10-09\", bias=True,\n")
save(path, t, crlf)

path = "tests/test_site.py"
t, crlf = load(path)
start = t.index("def test_span_labels():")
end = t.index("\ndef ", start + 1) + 1
t = t[:start] + t[end:].lstrip("\n")
save(path, t, crlf)

path = "tests/test_decompose_site.py"
t, crlf = load(path)
start = t.index('@pytest.mark.parametrize("name", ["decompose_1y.json", "decompose_3y.json"])')
end = t.index("def test_ratings_carry_ids")
t = t[:start] + t[end:]
save(path, t, crlf)
