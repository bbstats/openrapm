"""One-off: add --in-progress to `openrapm folds` (run once; kept as the record of the edit)."""
from pathlib import Path


def edit(path: str, pairs: list) -> None:
    file = Path(path)
    text = file.read_text(encoding="utf-8")
    for old, new in pairs:
        assert text.count(old) == 1, (path, old[:70], text.count(old))
        text = text.replace(old, new)
    file.write_text(text, encoding="utf-8", newline="\n")


WITHIN = [
    ("    trainable_panel,\n)", "    in_progress_panel,\n    trainable_panel,\n)"),
    ("from ..data.seasons import trainable, trainable_through",
     "from ..data.seasons import share_played, trainable, trainable_through"),
    ("from .build import season_lineup_rows", "from .build import held_scales, season_lineup_rows"),
    ("from ..settings import ROW_LABELS, TEST, UNSHRINK_LABELS, Targets",
     "from ..settings import PRODUCT, ROW_LABELS, TEST, UNSHRINK_LABELS, Targets"),
    ('''def check_models(meta: dict) -> None:
    """Refuse boosters that were not saved by a test-setting build of the shipped steps.
''', '''def check_models(meta: dict, neighbours: int = 1) -> None:
    """Refuse boosters that were not saved by a test-setting build of the shipped steps.  With `neighbours` 0, by a
    product-setting build instead: the season in progress's own boosters (--in-progress), which step 4 never reads.
'''),
    ('''    wrong = {key: (meta.get(key), value) for key, value in EXPECTED_META.items() if meta.get(key) != value}''',
     '''    expected = {**EXPECTED_META, "exclude_neighbours": neighbours}
    wrong = {key: (meta.get(key), value) for key, value in expected.items() if meta.get(key) != value}'''),
    ('''        raise SystemExit(f"the saved models were not built at the shipped test settings: {wrong}")''',
     '''        setting = "test" if neighbours else "product"
        raise SystemExit(f"the saved models were not built at the shipped {setting} settings: {wrong}")'''),
    ('''def season_models(saved_all: dict, season: int) -> dict:
    """{"O", "D"}: the season's saved boosters, after checking they never trained on it or a neighbor."""''',
     '''def season_models(saved_all: dict, season: int, neighbours: int = 1) -> dict:
    """{"O", "D"}: the season's saved boosters, after checking they never trained on it or a neighbor; with
    `neighbours` 0, on it alone (--in-progress)."""'''),
    ('''    if not {season - 1, season, season + 1} <= unseen:''',
     '''    if not set(range(season - neighbours, season + neighbours + 1)) <= unseen:'''),
    ('''    least_share: float = 1.0     # the smallest part-season share its boosters trained on (`least_share_of`)

    def __post_init__''',
     '''    least_share: float = 1.0     # the smallest part-season share its boosters trained on (`least_share_of`)
    # --in-progress: the share of its regular season the season in progress has played (data.seasons.share_played).
    # Every rating of it, the whole season so far and each fold, is then made as the build makes it: the boosters told
    # this share, the prior scales held and blended by it (commands/build.rate_one_season)
    in_progress_share: float | None = None

    def __post_init__'''),
    ('''        whole = {str(game) for game in fit_ids} == set(self.game_ids)
        if whole or self.cut is None or not self.part_season_rows:
            return 1.0
        return max(float(self.season_share), float(self.least_share))''',
     '''        if self.in_progress_share is not None:
            return max(float(self.in_progress_share), float(self.least_share))
        whole = {str(game) for game in fit_ids} == set(self.game_ids)
        if whole or self.cut is None or not self.part_season_rows:
            return 1.0
        return max(float(self.season_share), float(self.least_share))'''),
    ('''        whole = {str(game) for game in fit_ids} == set(self.game_ids)
        return rate_world(world, self.saved, self.prior_panel, context, self.settings, self.prior_share(fit_ids),
                          None if whole else self.held_scale, held_weight=self.held_weight)''',
     '''        whole = {str(game) for game in fit_ids} == set(self.game_ids)
        held = self.held_scale if self.in_progress_share is not None or not whole else None
        return rate_world(world, self.saved, self.prior_panel, context, self.settings, self.prior_share(fit_ids),
                          held, held_weight=self.held_weight)'''),
    ('''def attach_line(run: SeasonRun, lines: FoldLines, whole_season_context: bool) -> None:
    """The season's second-prior line and the stint rows its context numbers read, on `run`."""
    run.line = second_prior.season_coefficients(lines.grams, run.season, lines.trainable, SETTINGS.exclude_neighbours,
                                                lines.folds, lines.penalty, lines.nearest, lines.half_life)''',
     '''def attach_line(run: SeasonRun, lines: FoldLines, whole_season_context: bool,
                neighbours: int | None = None) -> None:
    """The season's second-prior line and the stint rows its context numbers read, on `run`.  `neighbours`: how many
    seasons either side of it the line leaves out, the test setting's unless given (0 with --in-progress)."""
    neighbours = SETTINGS.exclude_neighbours if neighbours is None else neighbours
    run.line = second_prior.season_coefficients(lines.grams, run.season, lines.trainable, neighbours,
                                                lines.folds, lines.penalty, lines.nearest, lines.half_life)'''),
    ('''              blend_through: float | None = None, blend_from: float = 0.0) -> int:
    """`openrapm folds`:''',
     '''              blend_through: float | None = None, blend_from: float = 0.0, in_progress: bool = False) -> int:
    """`openrapm folds`:'''),
    ('''    line from held at a season's first game to its fit's own at that share (prior_rapm.blend_weight)."""
    if cut is not None and so_far is not None:''',
     '''    line from held at a season's first game to its fit's own at that share (prior_rapm.blend_weight).

    With `in_progress`, the seasons are seasons in progress and `models` their own boosters, saved by the build that
    rated them (`openrapm build --save-models`, the product setting): the whole season so far and every fold are
    rated as that build rates the season (`in_progress_run`), so the whole-season rebuild must give back the published
    rows.  Step 4 never reads these folds; the ratings' standard errors do (commands/errors.py)."""
    if in_progress and (cut is not None or so_far is not None or hold_prior_scale is not None):
        raise SystemExit("--in-progress rates the season so far as the build does; it takes no --cut, --so-far or "
                         "--hold-prior-scale")
    if in_progress and any(int(season) <= trainable_through(cfg) for season in seasons):
        raise SystemExit(f"--in-progress is for seasons past {trainable_through(cfg)}, the last finished one")
    if cut is not None and so_far is not None:'''),
    ('''    check_models(meta)
    settings = fold_settings(meta)''',
     '''    neighbours = 0 if in_progress else SETTINGS.exclude_neighbours
    check_models(meta, neighbours)
    settings = fold_settings(meta)'''),
    ('''    panel = trainable_panel(cfg)
    roles = pd.read_parquet(roles_path(cfg))''',
     '''    panel = in_progress_panel(cfg) if in_progress else trainable_panel(cfg)
    roles = pd.read_parquet(roles_path(cfg))'''),
    ('''        saved = season_models(saved_all, season)
        full =''',
     '''        saved = season_models(saved_all, season, neighbours)
        full ='''),
    ('''        blend_run(run, blend_through, blend_from)
        if whole_season_inputs:''',
     '''        blend_run(run, blend_through, blend_from)
        if in_progress:
            in_progress_run(run, cfg)
        if whole_season_inputs:'''),
    ('''            attach_line(run, lines, whole_season_context)
        run_season(''',
     '''            attach_line(run, lines, whole_season_context, neighbours)
        run_season('''),
    ('''def boosters_identity(outputs: Path, models: str)''',
     '''def in_progress_run(run: "SeasonRun", cfg: dict) -> None:
    """--in-progress: `run` rates its season as the build rates a season in progress (commands/build.py,
    SeasonsInProgress and rate_one_season), said once: the boosters told the share of the regular season played, and
    short of settings.RidgeSettings.held_scale_through the prior scales held at the mean of the 8 finished seasons
    nearest it in the published step-3 table, blended toward each fit's own by that share."""
    ridge = PRODUCT.ridge
    share = share_played(run.season, cfg)
    run.in_progress_share = share
    if share < ridge.held_scale_through:
        run.held_scale = held_scales(cfg, ridge.held_scale_table, run.season)
        if run.held_scale is not None and ridge.held_scale_blend:
            run.held_weight = prior_rapm.blend_weight(share, ridge.held_scale_through, ridge.held_scale_blend_from)
    print(f"  {run.season} is in progress, {share:.1%} of its regular season played: every fold rated as the build "
          f"rates it, the prior scales {run.held_weight:.0%} of the way from held to each fit's own", flush=True)


def boosters_identity(outputs: Path, models: str)'''),
]

CLI = [
    ('''                     args.cut, args.so_far, args.hold_prior_scale, args.blend_through, args.blend_from)''',
     '''                     args.cut, args.so_far, args.hold_prior_scale, args.blend_through, args.blend_from,
                     args.in_progress)'''),
    ('''                            "(default %(default)s, as shipped; 0 is the line from the first game)")
    _add_jobs(folds)
''',
     '''                            "(default %(default)s, as shipped; 0 is the line from the first game)")
    folds.add_argument("--in-progress", action="store_true",
                       help="the seasons are in progress and --models their own boosters, saved by the build that "
                            "rated them (`openrapm build --save-models`): the season so far and every fold are rated "
                            "as that build rates the season.  For the ratings' standard errors (`openrapm errors`); "
                            "never for step 4")
    _add_jobs(folds)
'''),
]

if __name__ == "__main__":
    edit("src/openrapm/commands/within_season.py", WITHIN)
    edit("src/openrapm/commands/cli.py", CLI)
