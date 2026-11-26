"""One-off: the reproduction check of a season in progress is held to a looser tolerance (run once)."""
from edit_folds import edit

WITHIN = [
    ('''REPRODUCE_TOLERANCE = 1e-6
''',
     '''REPRODUCE_TOLERANCE = 1e-6
# --in-progress: how far the season so far rebuilt game by game may sit from the published rows, in points per 100.
# The build reads the season in progress's hustle counts from the league's season-to-date dashboards, because a game's
# hustle box score fills in over the days after it (data/panel.season_panel), and a fold can only sum the games it
# keeps; a game still waited on (config.yaml, no_stints_accepted) has box scores and no lineups.  On 2026-11-26 the
# largest gap was 0.16 and the median 0.01.  A larger one is a rebuild that is not the build's
IN_PROGRESS_TOLERANCE = 0.5
'''),
    ('''def reproduce(rated: RatedWorld, base: pd.DataFrame, panel: pd.DataFrame, saved: dict) -> dict:
    """The whole-season world against the base table's rows for the season, its boosters' answers against the
    priors the build saved, and its rebuilt inputs against the stored season panel."""''',
     '''def reproduce(rated: RatedWorld, base: pd.DataFrame, panel: pd.DataFrame, saved: dict,
              tolerance: float = REPRODUCE_TOLERANCE) -> dict:
    """The whole-season world against the base table's rows for the season, its boosters' answers against the
    priors the build saved, and its rebuilt inputs against the stored season panel.  It passes when every rating is
    within `tolerance` of the base table's."""'''),
    ('''                         and max(out["max_diff_offense"], out["max_diff_defense"]) < REPRODUCE_TOLERANCE''',
     '''                         and max(out["max_diff_offense"], out["max_diff_defense"]) < tolerance'''),
    ('''    result = reproduce(rated, base, panel, run.saved)''',
     '''    tolerance = REPRODUCE_TOLERANCE if run.in_progress_share is None else IN_PROGRESS_TOLERANCE
    result = reproduce(rated, base, panel, run.saved, tolerance)'''),
]

if __name__ == "__main__":
    edit("src/openrapm/commands/within_season.py", WITHIN)
