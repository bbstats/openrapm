"""One-off: the fold's booster check, for a season in progress (run once)."""
from edit_folds import edit

WITHIN = [
    ('''    checks["prior models: never trained on this season or its neighbours"] = (
        {run.season - 1, run.season, run.season + 1} <= unseen)''',
     '''    # a season in progress is rated by its own build's boosters, which leave out the season alone (--in-progress)
    near = {run.season} if run.in_progress_share is not None else {run.season - 1, run.season, run.season + 1}
    checks["prior models: never trained on this season or its neighbours"] = near <= unseen'''),
]

if __name__ == "__main__":
    edit("src/openrapm/commands/within_season.py", WITHIN)
