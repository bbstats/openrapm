#!/bin/bash
# Before the push of experiment 136: the working tree's code and settings give the published tables, every test passes,
# and the site's files are written into docs/.
cd "$(dirname "$0")/../.."
PY=.venv/Scripts/python
export PYTHONIOENCODING=utf8
(OPENRAPM_JOBS=1 OMP_NUM_THREADS=2 $PY -u -m openrapm build --seasons 2026,2027 --name check136 > outputs/check136.log 2>&1; echo "check build exit $?") &
(OMP_NUM_THREADS=2 $PY -m pytest -q > outputs/pytest136.log 2>&1; echo "pytest exit $?") &
(OMP_NUM_THREADS=4 $PY -u -m openrapm site --bias --openrapm-decomposition --similarity --power > outputs/site_publish_2026-11-04.log 2>&1; echo "site exit $?") &
wait
date
tail -n 3 outputs/pytest136.log | cut -c1-140
grep -c PASS outputs/site_publish_2026-11-04.log
grep -n "FAIL" outputs/site_publish_2026-11-04.log | cut -c1-140 | head -n 5
tail -n 2 outputs/site_publish_2026-11-04.log | cut -c1-140
