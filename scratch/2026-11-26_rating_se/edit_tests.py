"""One-off: the published Detail file may end its rows with the standard errors (run once)."""
from edit_folds import edit

TEST = [
    ("from openrapm.commands.site import OPENRAPM_FIELDS, OPENRAPM_PIECES, WOSA_PARTS",
     "from openrapm.commands.site import ERRORS_FIELD, OPENRAPM_FIELDS, OPENRAPM_PIECES, WOSA_PARTS"),
    ('''    assert meta["fields"] == OPENRAPM_FIELDS and meta["pieces"] == OPENRAPM_PIECES''',
     '''    # with the ratings' standard errors written (openrapm errors), one more field at the end of every row
    assert meta["fields"] in (OPENRAPM_FIELDS, OPENRAPM_FIELDS + [ERRORS_FIELD])
    assert meta["pieces"] == OPENRAPM_PIECES'''),
]

if __name__ == "__main__":
    edit("tests/test_decompose_openrapm_site.py", TEST)
