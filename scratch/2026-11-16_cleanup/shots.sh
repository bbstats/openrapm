#!/usr/bin/env bash
# Screenshots of the site's pages from a local server, for the eye check before a push.
#     bash scratch/2026-11-16_cleanup/shots.sh OUTDIR PAGE...      (the server must be running on port 8765)
OUT=$1; shift
CHROME="/c/Program Files/Google/Chrome/Application/chrome.exe"
for page in "$@"; do
  for width in 1280 390; do
    "$CHROME" --headless=new --disable-gpu --hide-scrollbars --virtual-time-budget=6000 --window-size=$width,1100 \
      --screenshot="$OUT/shot_${page}_$width.png" --user-data-dir="$OUT/chrome_profile" \
      "http://localhost:8765/$page.html" 2>/dev/null
  done
done
ls "$OUT"/shot_*
