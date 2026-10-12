"""The names the season panel and the saved files carried before 2026-10-11, and the plain names that replaced them.

Why this exists.  On 2026-10-11 every feature and panel column got a plain name: `fg3m` (made threes per 100)
became `threes_made_per100`, `poss_pct` became `team_possession_share`, the training rows' `chunk_poss` became
`row_possessions` (features.py defines every one).  The two ridge targets and the shipped input list were renamed
the same day.  Files written before then keep the old names, and none of them is rebuilt to be read: each is
translated as it is read, through the one table below.

    the season panel         outputs/role_panel_season.parquet becomes outputs/season_panel.parquet once, by
                             `openrapm data panel --migrate-from` (data/panel.py)
    the saved boosters       outputs/prior_models_<name>.pkl, on load (`saved_models`): each side's input list
                             keeps its order, so a booster, which reads its inputs by position, reads the same
                             columns
    the within-season folds  outputs/within/<tag>/players_*.parquet, on load (`translate_frame`, in
                             data/within_folds.py)
    a name typed by hand     an old target or input-list name on the command line (`target_name`,
                             `feature_set_name`)
    a config file            config.yaml's player_unit "window" or "season", on load (`player_unit_name`, in
                             data/config.py): today "player" and "player_season"

Raw per-game box counts keep their NBA abbreviations (the box table, the stint counters, config.yaml's
box_counts): `fg3m` is still the number of threes a player made in a game.  Only the per-100 rate built from it,
a panel column, was renamed.

Example: the old panel's `exp_poss` is read as `career_possessions_before_k`, and a saved booster's inputs
[..., "tenure", "ts", "weight", "chunk_poss", "chunk_seasons"] as
[..., "seasons_with_team", "true_shooting_pct", "weight_lb", "row_possessions", "row_seasons"], in that order.
"""
from __future__ import annotations

__all__ = ["COLUMNS", "DELETED", "FEATURE_SETS", "TARGETS", "PLAYER_UNITS", "column_name", "is_deleted",
           "translate_frame", "saved_models", "target_name", "feature_set_name", "player_unit_name"]

# The thirteen padded box-score rates per 100 possessions, in config.yaml's box_counts order.  Each had an
# uncentred twin, raw_<old> -> <new>_uncentred.
_RATES = {
    "fg3m": "threes_made_per100", "fg3_miss": "threes_missed_per100", "fg2m": "twos_made_per100",
    "fg2_miss": "twos_missed_per100", "ftm": "free_throws_made_per100", "ft_miss": "free_throws_missed_per100",
    "orb": "off_rebounds_per100", "drb": "def_rebounds_per100", "ast": "assists_per100",
    "tov": "turnovers_per100", "stl": "steals_per100", "blk": "blocks_per100", "pf": "fouls_per100",
}

# old name -> plain name.  `age`, `draft_pick`, `closeness`, `side`, `player_id` and `season` did not change.
COLUMNS = {
    **_RATES,
    **{f"raw_{old}": f"{new}_uncentred" for old, new in _RATES.items()},
    # sums of the rates (features.DERIVED)
    "pts": "points_per100", "fga": "shots_per100", "fta": "free_throw_attempts_per100",
    "fg3a": "three_point_attempts_per100", "usage": "used_possessions_per100", "reb": "rebounds_per100",
    "stocks": "steals_plus_blocks_per100", "creation": "assists_minus_turnovers_per100",
    "shotmix": "threes_minus_twos_attempted_per100",
    # efficiencies (features.RATIOS)
    "efg": "effective_fg_pct", "ts": "true_shooting_pct", "p3r": "three_point_attempt_share",
    "ftr": "free_throws_per_shot", "fg3p": "three_point_pct", "fg2p": "two_point_pct", "ftp": "free_throw_pct",
    "astr": "assists_per_used_possession", "tovr": "turnovers_per_used_possession", "orbsh": "off_rebound_share",
    # shot quality (features.SHOTQ)
    "q2": "two_point_expected_pct", "q3": "three_point_expected_pct", "m2": "two_point_pct_over_expected",
    "m3": "three_point_pct_over_expected", "xps": "expected_points_per_shot", "mpts": "points_per_shot_over_expected",
    # the shot totals and the league's levels beside them
    "shot_fg2a": "shot_twos_attempted", "shot_fg2m": "shot_twos_made", "shot_xl2": "shot_twos_expected_made",
    "shot_fg3a": "shot_threes_attempted", "shot_fg3m": "shot_threes_made", "shot_xl3": "shot_threes_expected_made",
    "shot_lg2": "league_two_point_pct", "shot_lg3": "league_three_point_pct", "shot_lgpps": "league_points_per_shot",
    # role, career, body
    "poss": "possessions", "poss_pct": "team_possession_share", "gs_pct": "games_started_share",
    "exp_yrs": "seasons_before", "exp_poss": "career_possessions_before_k", "entry_age": "age_at_first_season",
    "height": "height_in", "weight": "weight_lb", "tenure": "seasons_with_team", "n_teams": "teams_this_season",
    # his team with him on the court, off it, and the difference
    "onc_o": "on_court_scored_per100", "onc_d": "on_court_allowed_per100",
    "onc_poss_o": "on_court_off_possessions", "onc_poss_d": "on_court_def_possessions",
    "offc_o": "off_court_scored_per100", "offc_d": "off_court_allowed_per100",
    "offc_poss_o": "off_court_off_possessions", "offc_poss_d": "off_court_def_possessions",
    "net_o": "on_minus_off_scored_per100", "net_d": "on_minus_off_allowed_per100",
    # score state and playoffs
    "gt_share": "garbage_time_share", "abs_margin": "mean_abs_margin", "po_share": "playoff_share",
    # the evidence behind a training row (box_prior.py), and the held-out audit's team quality
    "chunk_poss": "row_possessions", "chunk_seasons": "row_seasons", "team_net": "team_net_per100",
}

# Columns no reader uses any more, dropped rather than renamed: the block label and the design's player index
# (the season gates the panel now), the APM and the Simple-SPM passes, the twenty possession-decomposition
# columns (pc_*), the shot-mix lift, the binned body columns and the "plays like a centre" sum.
DELETED = frozenset({"window", "ps_idx", "label_key", "apm", "spm", "u", "a", "rapm1", "mix_lift", "mix_n",
                     "height2", "weight15", "bigness"})
DELETED_PREFIX = "pc_"
INTERACTION_MARK = "_x_"          # height_x_weight and the like: products of two inputs

FEATURE_SETS = {"boruta_noonc": "box_score_and_bio"}
TARGETS = {
    "xpts_ft": "points_with_expected_free_throws",
    "x3def_w0.25": "points_with_expected_opponent_threes",
    "x3def": "points_with_expected_opponent_threes_all",
}
# config.yaml's player_unit (design.PLAYER_UNITS): one player column per player over the design's seasons, or one per
# player-season
PLAYER_UNITS = {"window": "player", "season": "player_season"}


def column_name(name: str) -> str:
    """The plain name of a column; a plain name, or one that never changed, comes back as it is."""
    return COLUMNS.get(name, name)


def is_deleted(name: str) -> bool:
    """Whether an old column is one no reader uses any more (`DELETED`, pc_*, the *_x_* products)."""
    return name in DELETED or name.startswith(DELETED_PREFIX) or INTERACTION_MARK in name


def translate_frame(frame):
    """`frame` (an old season panel, or a within-season players table) under the plain names: every deleted
    column dropped, every old one renamed, the order of the rest and of the rows untouched.  A frame already
    under the plain names comes back with the same columns."""
    kept = [column for column in frame.columns if not is_deleted(column)]
    return frame[kept].rename(columns=COLUMNS)


def target_name(name: str) -> str:
    """The plain name of a ridge target: "x3def_w0.25" -> "points_with_expected_opponent_threes"."""
    return TARGETS.get(name, name)


def feature_set_name(name: str) -> str:
    """The plain name of an input list: "boruta_noonc" -> "box_score_and_bio"."""
    return FEATURE_SETS.get(name, name)


def player_unit_name(name: str) -> str:
    """The plain name of a config player_unit: "window" -> "player", "season" -> "player_season"."""
    return PLAYER_UNITS.get(name, name)


def _side(boosters: dict) -> dict:
    """One side's saved boosters with `feats` (the input list) and `model_feats` (what the booster reads, in
    order) renamed, each list in its own order."""
    return {**boosters, "feats": [column_name(name) for name in boosters["feats"]],
            "model_feats": [column_name(name) for name in boosters["model_feats"]]}


def _season(entry):
    """A season's {"O": boosters, "D": boosters}, both sides renamed; anything else as it is."""
    if not isinstance(entry, dict):
        return entry
    return {side: _side(boosters) if isinstance(boosters, dict) and "feats" in boosters else boosters
            for side, boosters in entry.items()}


def _settings(settings: dict) -> dict:
    """A schema-2 file's settings with its target and input-list names plain."""
    out = dict(settings)
    if "targets" in out:
        out["targets"] = {side: target_name(name) for side, name in out["targets"].items()}
    if "feature_set" in out.get("prior", {}):
        out["prior"] = {**out["prior"], "feature_set": feature_set_name(out["prior"]["feature_set"])}
    return out


def _meta(meta: dict) -> dict:
    """The file's meta with the input list's name (`features`) and the settings' names plain."""
    out = dict(meta)
    if "features" in out:
        out["features"] = feature_set_name(out["features"])
    if out.get("settings"):
        out["settings"] = _settings(out["settings"])
    return out


def _names(saved: dict) -> list:
    """Every name a saved-boosters file records: the input lists, then the meta's input-list and target names."""
    names = [name for entry in saved.values() if isinstance(entry, dict) for boosters in entry.values()
             if isinstance(boosters, dict) and "feats" in boosters
             for name in (*boosters["feats"], *boosters["model_feats"])]
    meta = saved.get("meta", {})
    settings = meta.get("settings") or {}
    return names + [meta.get("features"), *settings.get("targets", {}).values()]


def saved_models(saved: dict) -> tuple[dict, bool]:
    """(a saved-boosters file under the plain names, whether any name in it was an old one).

    The file is {season: {"O": boosters, "D": boosters}, "meta": {...}}.  The boosters themselves are the same
    objects; only the names around them change.
    """
    translated = {key: _meta(entry) if key == "meta" else _season(entry) for key, entry in saved.items()}
    return translated, _names(translated) != _names(saved)
