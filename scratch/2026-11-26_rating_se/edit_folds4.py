"""One-off: the lint check's two findings on --in-progress (run once)."""
from edit_folds import edit

WITHIN = [
    ("    in_progress_panel,\n    trainable_panel,\n)", "    trainable_panel,\n)"),
    ("    fit_designs,\n", "    fit_designs,\n    in_progress_panel,\n"),
    ('''    if in_progress and (cut is not None or so_far is not None or hold_prior_scale is not None):
        raise SystemExit("--in-progress rates the season so far as the build does; it takes no --cut, --so-far or "
                         "--hold-prior-scale")
    if in_progress and any(int(season) <= trainable_through(cfg) for season in seasons):
        raise SystemExit(f"--in-progress is for seasons past {trainable_through(cfg)}, the last finished one")
    if cut is not None and so_far is not None:''',
     '''    neighbours, panel = fold_scope(cfg, seasons, in_progress, (cut, so_far, hold_prior_scale))
    if cut is not None and so_far is not None:'''),
    ('''    neighbours = 0 if in_progress else SETTINGS.exclude_neighbours
    check_models(meta, neighbours)''',
     '''    check_models(meta, neighbours)'''),
    ('''    panel = in_progress_panel(cfg) if in_progress else trainable_panel(cfg)
    roles = pd.read_parquet(roles_path(cfg))''',
     '''    roles = pd.read_parquet(roles_path(cfg))'''),
    ('''        if in_progress:
            in_progress_run(run, cfg)
        if whole_season_inputs:''',
     '''        in_progress_run(run, cfg, in_progress)
        if whole_season_inputs:'''),
    ('''def in_progress_run(run: "SeasonRun", cfg: dict) -> None:
    """--in-progress: `run` rates its season as the build rates a season in progress (commands/build.py,
    SeasonsInProgress and rate_one_season), said once: the boosters told the share of the regular season played, and
    short of settings.RidgeSettings.held_scale_through the prior scales held at the mean of the 8 finished seasons
    nearest it in the published step-3 table, blended toward each fit's own by that share."""
    ridge = PRODUCT.ridge''',
     '''def fold_scope(cfg: dict, seasons, in_progress: bool, other_designs: tuple) -> tuple[int, pd.DataFrame]:
    """(how many seasons either side of a season its boosters and its line must have left out, the stored season panel
    its rebuilt inputs are held against): the test setting's and the finished seasons' panel, or with --in-progress 0
    and the seasons in progress' panel, after checking the seasons are in progress and none of `other_designs`
    (--cut, --so-far, --hold-prior-scale) was asked for."""
    if not in_progress:
        return SETTINGS.exclude_neighbours, trainable_panel(cfg)
    if any(option is not None for option in other_designs):
        raise SystemExit("--in-progress rates the season so far as the build does; it takes no --cut, --so-far or "
                         "--hold-prior-scale")
    if any(int(season) <= trainable_through(cfg) for season in seasons):
        raise SystemExit(f"--in-progress is for seasons past {trainable_through(cfg)}, the last finished one")
    return 0, in_progress_panel(cfg)


def in_progress_run(run: "SeasonRun", cfg: dict, in_progress: bool = True) -> None:
    """--in-progress: `run` rates its season as the build rates a season in progress (commands/build.py,
    SeasonsInProgress and rate_one_season), said once: the boosters told the share of the regular season played, and
    short of settings.RidgeSettings.held_scale_through the prior scales held at the mean of the 8 finished seasons
    nearest it in the published step-3 table, blended toward each fit's own by that share.  Without `in_progress`,
    nothing."""
    if not in_progress:
        return
    ridge = PRODUCT.ridge'''),
]

if __name__ == "__main__":
    edit("src/openrapm/commands/within_season.py", WITHIN)
