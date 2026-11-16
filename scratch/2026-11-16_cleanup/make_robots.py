"""Writes docs/robots.txt from the list of AI crawlers kept at github.com/ai-robots-txt/ai.robots.txt, pinned to one
commit, less the two names the owner kept allowed on 2026-11-16: facebookexternalhit, which draws the preview of a
link pasted into Meta's apps, and Applebot, Apple's search crawler (Applebot-Extended, its AI use, stays barred).

    A:/code/spmm/.venv/Scripts/python scratch/2026-11-16_cleanup/make_robots.py [COMMIT]

To refresh the list: run it with the newest commit of that repository, read the diff, and run the site tests.
"""
import sys
from pathlib import Path

import requests

COMMIT = sys.argv[1] if len(sys.argv) > 1 else "be22abebcf663f7080766ad814829396dd19c659"
HOST = "raw." + "github" + "usercontent.com"
SOURCE = f"https://{HOST}/ai-robots-txt/ai.robots.txt/{COMMIT}/robots.txt"
KEPT_ALLOWED = {"facebookexternalhit", "Applebot"}
DAY = "2026-11-16"
OUT = Path(__file__).resolve().parents[2] / "docs" / "robots.txt"

HEAD = f"""\
# OpenRAPM, by Nathan Walker.  https://openrapm.com
# Ratings by Nathan Walker. Use them with credit and a link to openrapm.com. No bulk copying or scraping.
# The code is open: https://github.com/bbstats/openrapm.  The ratings' terms: LICENSE-RATINGS.md there.

# Search engines may read every page.  Nothing here is for AI answers or AI training.
# Content-Signal is defined at https://contentsignals.org
User-agent: *
Content-Signal: search=yes, ai-input=no, ai-train=no
Allow: /

# AI crawlers, barred from everything.  The names are the list kept at https://github.com/ai-robots-txt/ai.robots.txt,
# MIT license, at commit {COMMIT[:7]} of {DAY}, less facebookexternalhit, which draws link previews, and Applebot,
# Apple's search crawler: Applebot-Extended below turns its AI use off.
"""

text = requests.get(SOURCE, timeout=30).text
names = [line.split(":", 1)[1].strip() for line in text.splitlines() if line.lower().startswith("user-agent:")]
assert len(names) > 150 and text.rstrip().endswith("Disallow: /"), "the list's format changed"
kept = [name for name in names if name not in KEPT_ALLOWED]
assert len(kept) == len(names) - len(KEPT_ALLOWED), "a name kept allowed is no longer on the list"
body = "".join(f"User-agent: {name}\n" for name in kept) + "Disallow: /\n"
OUT.write_text(HEAD + body, encoding="utf-8", newline="\n")
print(f"wrote {OUT}: {len(kept)} AI crawlers barred, of {len(names)} on the list")
