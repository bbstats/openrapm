"""The hand-written pages in docs/: what a visitor's typing, tapping and bookmarks rest on, read from the pages' own
text, since the tests run without a browser.  Each was a defect found on 2026-11-09 (notes/DECISIONS.md, "The
improvements round")."""
import re
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
