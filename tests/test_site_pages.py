"""The hand-written pages in docs/: what a visitor's typing, tapping and bookmarks rest on, read from the pages' own
text, since the tests run without a browser.  Each was a defect found on 2026-11-09 (notes/DECISIONS.md, "The
improvements round")."""
import re
import struct
import unicodedata
from pathlib import Path

DOCS = Path(__file__).resolve().parents[1] / "docs"
# how both pages reduce a name before comparing it
FOLD = ('const fold = s => s.normalize("NFD").replace(/[\\u0300-\\u036f]/g, "").toLowerCase()\n'
        '    .replace(/[\'\\u2018\\u2019.]/g, "").replace(/[-\\s]+/g, " ").trim();')


def _page(name: str) -> str:
    return (DOCS / name).read_text(encoding="utf-8").replace("\r\n", "\n")


def _fold(name: str) -> str:
    """FOLD's steps in Python: accents off, lower case, apostrophes and periods out, hyphens and runs of spaces to one
    space, no space at either end."""
    plain = re.sub("[̀-ͯ]", "", unicodedata.normalize("NFD", name)).lower()
    return re.sub(r"[-\s]+", " ", re.sub("['‘’.]", "", plain)).strip()


def test_find_and_the_similarity_box_reduce_a_name_the_same_forgiving_way():
    """A phone's keyboard leaves a space after a suggested word and turns ' into a curly mark, and nobody types the
    hyphen in Gilgeous-Alexander: each of those showed an empty table."""
    for page in ("index.html", "similarity.html"):
        assert FOLD in _page(page), page
    for typed, name in (("Jokic ", "Nikola Jokić"), ("gilgeous alexander", "Shai Gilgeous-Alexander"),
                        ("DeAaron", "De'Aaron Fox"), ("de’aaron fox", "De'Aaron Fox"),
                        ("pj wash", "P.J. Washington"), ("royce oneale", "Royce O'Neale"),
                        ("  karl-anthony", "Karl-Anthony Towns")):
        assert _fold(typed) in _fold(name), (typed, name)
    assert _fold("   ") == ""                              # nothing typed filters nothing


def test_the_power_tab_is_drawn_from_the_first_paint_once_a_browser_has_been_told_it_is_ready():
    """The answer is in power.json, which arrives after the page is drawn: the tab came in late on every page view and
    pushed the tabs after it sideways.  site.js now remembers the answer and sets it on <html> before anything paints,
    and site.css shows the tab under it."""
    script, style = _page("site.js"), _page("site.css")
    remembered = script.index('power = localStorage.getItem("power")')
    applied = script.index('if (power === "1") root.setAttribute("data-power", "1");')
    assert remembered < applied < script.index('document.addEventListener("DOMContentLoaded"')
    assert 'localStorage.setItem("power", ready ? "1" : "0")' in script and "tab.hidden = !ready;" in script
    assert ':root[data-power="1"] nav.site a[href="power.html"][hidden] { display: inline !important; }' in style
    for page in ("about.html", "index.html", "similarity.html", "bias.html", "changelog.html"):
        assert '<a href="power.html" hidden>Power</a>' in _page(page), page      # still hidden until told


def test_the_season_arrows_do_nothing_until_the_ratings_are_loaded():
    page = _page("index.html")
    for arrow in ("older", "newer"):
        assert re.search(rf'<button type="button" id="{arrow}" [^>]*\bdisabled>', page), arrow
    step = page[page.index("function step(by) {"):]
    assert step.index("if (!data) return;") < step.index("data.meta.seasons")


def test_the_csv_of_this_view_is_the_rows_on_screen_in_their_order():
    page = _page("index.html")
    assert "downloadCSV(csvName(state.season), shown.map(r => [state.season, byId.get(r.id)]));" in page
    assert "shown.map(r => [String(r.s), bySeason.get(String(r.s))])" in page   # a player's seasons, as sorted


def test_the_page_as_it_opens_keeps_a_bare_address():
    """With the season written into the address on every visit, a bookmark of the home page went on opening that
    season after the next one began."""
    page = _page("index.html")
    assert "opening() ? location.pathname + location.search : addr(state.season, state.team, state.player)" in page
    opening = page[page.index("const opening = () =>"):page.index("const writeAddress")]
    for part in ("state.player === null", "!state.team", 'state.view === "ratings"', 'state.side === "total"',
                 "data.meta.default_season"):
        assert part in opening, part


def test_the_tab_bars_two_buttons_and_a_phones_headers_are_forty_pixel_targets():
    """Every control is 40 px tall (site.css); the theme and colors buttons were 36, a phone's sort headers 32, and a
    phone's name link 18, the height of its text."""
    style, page = _page("site.css"), _page("index.html")
    assert "nav.site .theme, nav.site .colors { margin-left: auto; display: inline-flex; padding: 11px;" in style
    phone = page[page.index("@media (max-width: 600px) {"):page.index("</style>")]
    assert "th { padding-top: 11px; padding-bottom: 11px; }" in phone
    assert "td.name > a { margin: -7px -5px 0; padding: 7px 5px 2px; }" in phone
    assert "td.name .tm a { display: inline-block; margin: -2px 0 -7px; padding: 2px 0 7px; }" in phone


# ------------------------------------------------------------------ the ratings page's two files
def test_the_ratings_page_opens_on_the_list_file_and_asks_for_the_detail_file_only_when_detail_is_shown():
    """The page read the Detail file for everything, 1.0 MB compressed before its first row was drawn; the list file
    is 0.3 MB.  The Similarity pool, which only a player's own seasons read, waits for one to be drawn."""
    page = _page("index.html")
    load = page[page.index("async function load() {"):page.index("let detailed = null;")]
    assert 'fetch("data/ratings_list.json")' in load and "decompose_openrapm" not in load
    assert page.count('fetch("data/decompose_openrapm.json")') == 1
    render = page[page.index("function render() {"):page.index("// the CSV:")]
    waits = render.index("if (detail && !data.detail) {")
    assert waits < render.index("loadDetail().then(render") < render.index("let rows;")
    assert render.index("if (one) askSimilar();") < render.index("let rows;")
    assert page.count('fetch("data/similarity/player_seasons.json")') == 1
    assert 'for (const reached of ["pointerenter", "focus"])' in page          # on its way before the click


def test_the_list_of_the_season_in_progress_says_the_day_it_runs_through():
    """As the Power and Bias pages do: "through Nov 8, 2026", beside the byline, from the list file's meta.through."""
    page = _page("index.html")
    assert '<span id="through"></span></p>' in page
    assert "const through = !one && (data.meta.through || {})[state.season];" in page
    assert '$("through").textContent = through ? " \u00b7 through " + day(through) : "";' in page


def test_the_committed_list_file_carries_the_detail_files_ratings():
    """docs/data/ratings_list.json is cut from decompose_openrapm.json when both are written, and the morning job
    commits both: every player-season of one is in the other with the same name, teams, ratings and WOSA."""
    import json

    import pytest
    listed, detail = DOCS / "data" / "ratings_list.json", DOCS / "data" / "decompose_openrapm.json"
    if not listed.exists() or not detail.exists():
        pytest.skip("the ratings page's files are not built")
    listed, detail = (json.loads(path.read_text(encoding="utf-8")) for path in (listed, detail))
    assert listed["meta"]["fields"] == ["n", "id", "p", "tm", "o", "d", "t", "w"]
    for key in ("seasons", "default_season", "percentile_min", "rookies"):
        assert listed["meta"][key] == detail["meta"][key], key
    at = {name: detail["meta"]["fields"].index(name) for name in listed["meta"]["fields"]}
    assert list(listed["rows"]) == list(detail["rows"])
    for season, rows in detail["rows"].items():
        cut = [[row[at["n"]], row[at["id"]], row[at["p"]], row[at["tm"]], row[at["o"]][-1], row[at["d"]][-1],
                row[at["t"]][-1], row[at["w"]]] for row in rows]
        assert listed["rows"][season] == cut, season
    in_progress = listed["meta"]["through"]
    assert all(re.fullmatch(r"\d{4}-\d{2}-\d{2}", day) for day in in_progress.values())
    assert set(in_progress) <= {str(listed["meta"]["seasons"][0])}             # only the latest can be in progress


def test_a_similarity_file_that_fails_to_load_says_so_in_the_table():
    """Every redraw goes through one function that catches: a season's file that failed left the old table up and the
    failure in the console."""
    page = _page("similarity.html")
    assert "async function draw() {" in page and "async function render()" not in page
    catcher = page[page.index("const render = () => draw().catch(() => {"):]
    assert "Could not load the data." in catcher[:200]


# each page of the site, and its address after https://openrapm.com/
PAGES = {"index.html": "", "power.html": "power.html", "similarity.html": "similarity.html", "bias.html": "bias.html",
         "about.html": "about.html", "changelog.html": "changelog.html"}
SITE = "https://openrapm.com/"
PNG_START = bytes([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A])


def _png_size(name: str) -> tuple:
    """(width, height) of a PNG file in docs/, from its header."""
    data = (DOCS / name).read_bytes()
    assert data[:8] == PNG_START, name
    return struct.unpack(">II", data[16:24])


def test_every_page_names_the_tab_icon_and_the_picture_a_shared_link_shows():
    """Until 2026-11-10 a tab showed a blank page and a shared link a bare address.  Each page's head names the icon
    three ways, for any browser, for one that draws an SVG and for a phone's home screen; and it gives a shared link
    the page's own title, its address and one picture, by their full addresses, since whoever draws the preview reads
    the page from outside the site."""
    for page, address in PAGES.items():
        head = _page(page).split("</head>")[0]
        title = re.search(r"<title>(.*)</title>", head).group(1)
        for tag in ('<link rel="icon" href="favicon.ico" sizes="32x32">',
                    '<link rel="icon" href="favicon.svg" type="image/svg+xml">',
                    '<link rel="apple-touch-icon" href="apple-touch-icon.png">',
                    '<meta property="og:site_name" content="OpenRAPM">',
                    f'<meta property="og:title" content="{title}">',
                    f'<meta property="og:url" content="{SITE}{address}">',
                    f'<meta property="og:image" content="{SITE}preview.png">',
                    '<meta name="twitter:card" content="summary_large_image">'):
            assert head.count(tag) == 1, (page, tag)
        # the picture's size as the tags state it
        assert 'og:image:width" content="1200"' in head and 'og:image:height" content="630"' in head, page


def test_the_icon_and_the_picture_are_files_of_the_sizes_the_pages_state():
    assert _png_size("preview.png") == (1200, 630)
    assert _png_size("apple-touch-icon.png") == (180, 180)
    icon = (DOCS / "favicon.ico").read_bytes()
    assert icon[:4] == bytes([0, 0, 1, 0]) and struct.unpack("<H", icon[4:6])[0] == 3       # an .ico of three sizes
    assert {icon[6 + 16 * k] for k in range(3)} == {16, 32, 48}
    mark = (DOCS / "favicon.svg").read_text(encoding="utf-8")
    assert mark.startswith("<svg") and 'viewBox="0 0 32 32"' in mark
