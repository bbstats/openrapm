"""The game-by-game box-score prior out (experiments 114-117: the multi-stat network's prior split credit among
teammates worse than today's in every era, the closest version worse in 18 of 30 seasons; the owner, 2026-11-16: "Take
all three out").  Kept at the tag archive/research-2026-11.  A saved model file of a game-by-game build is refused by
name where saved models are read (within_season.load_models)."""
from pathlib import Path


def load(path):
    raw = Path(path).read_bytes()
    return raw.decode("utf-8").replace("\r\n", "\n"), b"\r\n" in raw


def save(path, text, crlf):
    Path(path).write_bytes((text.replace("\n", "\r\n") if crlf else text).encode("utf-8"))
    print("edited", path)


def swap(text, old, new, count=1):
    assert text.count(old) == count, (old[:90], text.count(old))
    return text.replace(old, new)


def cut(text, start_marker, end_marker):
    start = text.index(start_marker)
    end = text.index(end_marker, start)
    return text[:start] + text[end:]


# ------------------------------------------------------------------ settings.py
path = "src/openrapm/settings.py"
t, crlf = load(path)
t = cut(t, '    # "": one box-score prior per player and season, from the season panel (shipped).  A tag: the box-score prior game',
        "\n\nROW_LABELS = ")
assert "game_prior" not in t
save(path, t, crlf)

# ------------------------------------------------------------------ commands/cli.py
path = "src/openrapm/commands/cli.py"
t, crlf = load(path)
t = swap(t, "    openrapm game-rates     the multi-stat networks: one per rated season, trained on the GPU and scored\n", "")
t = swap(t, "from .game_rates import add_game_rates\n", "")
t = swap(t, """    if args.game_prior_whole_season and (not args.game_prior or args.game_prior_backward):
        raise SystemExit("--game-prior-whole-season reads the networks of --game-prior once a season, after each "
                         "player's last game, and takes no --game-prior-backward")
""", "")
t = swap(t, """                                 box_first=bool(args.box_first),
                                 game_prior=args.game_prior, game_prior_backward=args.game_prior_backward,
                                 game_prior_whole_season=bool(args.game_prior_whole_season)))""",
         """                                 box_first=bool(args.box_first)))""")
t = cut(t, '    parser.add_argument("--game-prior", metavar="TAG"', '    parser.add_argument("--second-prior-team-totals"')
t = swap(t, "    add_game_rates(commands, parse_seasons)\n", "")
assert "game_prior" not in t and "game-rates" not in t and "game_rates" not in t
save(path, t, crlf)

# ------------------------------------------------------------------ commands/build.py
path = "src/openrapm/commands/build.py"
t, crlf = load(path)
t = swap(t, "from .. import box_prior, game_prior, game_rates, prior_rapm, prior_shrink, rapm, second_prior, swap_adjust, swap_test\n",
         "from .. import box_prior, prior_rapm, prior_shrink, rapm, second_prior, swap_adjust, swap_test\n")
t = swap(t, "from ..data.game_rate_files import load_network\n", "")
t = swap(t, "from ..data.player_games import trainable_player_games\n", "")
t = swap(t, """def rated_second_prior(cfg: dict, season: int, panel: pd.DataFrame, priors: dict, lines: SecondPriorLines,
                       settings, share: float = 1.0, box: dict | None = None,
                       types_panel: pd.DataFrame | None = None, phases=SEASON_PHASES) -> SeasonSecondPrior:""",
         """def rated_second_prior(cfg: dict, season: int, panel: pd.DataFrame, priors: dict, lines: SecondPriorLines,
                       settings, share: float = 1.0, phases=SEASON_PHASES) -> SeasonSecondPrior:""")
t = swap(t, """    regular season played (box_prior.rated_rows).  With the box-score prior game by game, `box` is each player's
    season average ({"O", "D"}: {player_id: value}) and `types_panel` the season panel the line's step-5 numbers are
    read from, as without it.  `phases` are the stints read,""",
         """    regular season played (box_prior.rated_rows).  `phases` are the stints read,""")
t = swap(t, """    typed = panel if types_panel is None else types_panel
    types = (None if lines.type_scale is None else
             second_prior.standardised_types(typed[(typed.side == "O") & (typed.season == season)], lines.type_scale))
    box = {side: priors[side].values for side in SIDES} if box is None else box
""", """    types = (None if lines.type_scale is None else
             second_prior.standardised_types(panel[(panel.side == "O") & (panel.season == season)], lines.type_scale))
    box = {side: priors[side].values for side in SIDES}
""")
t = cut(t, "# what a game-by-game build keeps of each game row:", "def rate_seasons(")
t = swap(t, """                 part_seasons: pd.DataFrame | None = None,
                 progress: SeasonsInProgress | None = None, games: dict | None = None) -> tuple[list, dict, list]:
    \"\"\"`rate_one_season` for each season, or `rate_one_season_by_game` with the box-score prior game by game (`games`,
    each season's game rows): ([one centered table per season], {season: {side: saved boosters}}, [each season's game
    rows, when game by game]).
""", """                 part_seasons: pd.DataFrame | None = None,
                 progress: SeasonsInProgress | None = None) -> tuple[list, dict]:
    \"\"\"`rate_one_season` for each season: ([one centered table per season], {season: {side: saved boosters}}).
""")
t = swap(t, "    rated, models, label_cache, game_rows = {}, {}, {}, []\n", "    rated, models, label_cache = {}, {}, {}\n")
t = swap(t, """        if settings.prior.game_prior:
            table, priors, unseen, second, rows = rate_one_season_by_game(season, sums, panel, designs, settings,
                                                                          params, games, lines, label_cache)
            if rows is not None:
                game_rows.append(rows)
        else:
            table, priors, unseen, second = rate_one_season(season, sums, panel, designs, settings, params, lines,
                                                            label_cache, part_seasons, progress)
""", """        table, priors, unseen, second = rate_one_season(season, sums, panel, designs, settings, params, lines,
                                                        label_cache, part_seasons, progress)
""")
t = swap(t, "    return [rated[season] for season in seasons], models, game_rows\n",
         "    return [rated[season] for season in seasons], models\n")
t = cut(t, "def season_game_rows(cfg: dict, settings, seasons) -> dict | None:", "def rate_table(")
t = swap(t, """def rate_table(cfg: dict, settings, seasons, keep_models: bool = False) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    \"\"\"Steps 1-3 for `seasons`: (the step-3 table in the published schema, {season: saved boosters} or {}, every
    season's game rows with the box-score prior game by game, or None).\"\"\"""",
         """def rate_table(cfg: dict, settings, seasons, keep_models: bool = False) -> tuple[pd.DataFrame, dict]:
    \"\"\"Steps 1-3 for `seasons`: (the step-3 table in the published schema, {season: saved boosters} or {}).\"\"\"""")
t = swap(t, """    games = season_game_rows(cfg, settings, seasons)
    tables, models, game_rows = rate_seasons(seasons, sums, panel, designs, settings, params, keep_models=keep_models,
                                             lines=lines,
                                             part_seasons=None if games else part_season_panel(cfg, settings),
                                             progress=SeasonsInProgress.read(cfg, seasons, settings), games=games)
    table = prior_rapm.published_table(pd.concat(tables, ignore_index=True), player_names(cfg))
    return table, models, (pd.concat(game_rows, ignore_index=True) if game_rows else None)
""", """    tables, models = rate_seasons(seasons, sums, panel, designs, settings, params, keep_models=keep_models,
                                  lines=lines, part_seasons=part_season_panel(cfg, settings),
                                  progress=SeasonsInProgress.read(cfg, seasons, settings))
    table = prior_rapm.published_table(pd.concat(tables, ignore_index=True), player_names(cfg))
    return table, models
""")
t = cut(t, "def write_game_rows(rows: pd.DataFrame | None, cfg: dict, name: str) -> None:", "def write_table(")
t = swap(t, """    \"\"\"`rate_table` over `parallel.season_chunks`, one `openrapm rate` child per chunk: (the step-3 table, the
    boosters by season or {}, the game rows or None).  Each child writes season_ratings_<name>__part<k>_step3.parquet,
    when boosters are kept prior_models_<save_models>__part<k>.pkl, and game by game
    game_priors_<name>__part<k>.parquet; all are read back here, in season order, and removed.\"\"\"""",
         """    \"\"\"`rate_table` over `parallel.season_chunks`, one `openrapm rate` child per chunk: (the step-3 table, the
    boosters by season or {}).  Each child writes season_ratings_<name>__part<k>_step3.parquet and, when boosters are
    kept, prior_models_<save_models>__part<k>.pkl; all are read back here, in season order, and removed.\"\"\"""")
t = swap(t, "    tables, models, game_rows = [], {}, []\n", "    tables, models = [], {}\n")
t = swap(t, """        games = outputs / f"game_priors_{parallel.part_name(name, k)}.parquet"
        if games.exists():
            game_rows.append(pd.read_parquet(games))
            games.unlink()
""", "")
t = swap(t, """    rows = pd.concat(game_rows, ignore_index=True).sort_values("season", kind="mergesort") if game_rows else None
    return table, ({season: models[season] for season in seasons} if save_models else {}), rows
""", """    return table, ({season: models[season] for season in seasons} if save_models else {})
""")
t = swap(t, """        table, models, game_rows = rate_in_parts(cfg, seasons, name, save_models, argv, fits)
    else:
        table, models, game_rows = rate_table(cfg, settings, seasons, keep_models=bool(save_models))
""", """        table, models = rate_in_parts(cfg, seasons, name, save_models, argv, fits)
    else:
        table, models = rate_table(cfg, settings, seasons, keep_models=bool(save_models))
""")
t = swap(t, "    write_game_rows(game_rows, cfg, name)\n", "", count=2)
t = swap(t, "    step3, models, game_rows = rate_table(cfg, settings, seasons, keep_models=bool(save_models))\n",
         "    step3, models = rate_table(cfg, settings, seasons, keep_models=bool(save_models))\n")
for gone in ("game_prior", "game_rates", "game_rows", "load_network", "player_games", "rate_one_season_by_game"):
    assert gone not in t, gone
save(path, t, crlf)

# ------------------------------------------------------------------ commands/within_season.py
path = "src/openrapm/commands/within_season.py"
t, crlf = load(path)
t = swap(t, "from .. import box_prior, decomposition, game_prior, game_rates, prior_rapm, second_prior, trade_set\n",
         "from .. import box_prior, decomposition, prior_rapm, second_prior, trade_set\n")
t = swap(t, "from ..data.game_rate_files import load_network\n", "")
t = swap(t, "from ..data.player_games import season_games\n", "")
t = swap(t, "from .build import game_box_priors, line_weights, season_lineup_rows\n", "from .build import season_lineup_rows\n")
t = swap(t, """    saved, renamed = legacy_names.saved_models(pd.read_pickle(outputs / f"prior_models_{models}.pkl"))
""", """    saved, renamed = legacy_names.saved_models(pd.read_pickle(outputs / f"prior_models_{models}.pkl"))
    network = ((saved.get("meta", {}).get("settings") or {}).get("prior") or {}).get("game_prior")
    if network:
        raise SystemExit(f"prior_models_{models}.pkl was saved by a build whose box-score prior was read game by game, "
                         f"from the networks {network!r}: that code was removed on 2026-11-16 and is at the tag "
                         f"archive/research-2026-11")
""")
t = swap(t, """               scale=None, types_panel: pd.DataFrame | None = None, held_weight: float = 0.0) -> RatedWorld:""",
         """               scale=None, held_weight: float = 0.0) -> RatedWorld:""")
t = swap(t, """    the ridge's prior scales (prior_rapm.held_prior_scale, experiment 109); `types_panel`, the rows the second prior's
    step-5 numbers are read from when not the boosters' (the network's whole-season rows, `rate_world_by_game`);
    `held_weight`,""", """    the ridge's prior scales (prior_rapm.held_prior_scale, experiment 109); `held_weight`,""")
t = swap(t, """        typed = panel if types_panel is None else types_panel
        types = (None if context.type_scale is None else second_prior.standardised_types(
            typed[(typed.side == "O") & (typed.possessions > 0)], context.type_scale))""",
         """        types = (None if context.type_scale is None else second_prior.standardised_types(
            panel[(panel.side == "O") & (panel.possessions > 0)], context.type_scale))""")
t = cut(t, "def rate_world_by_game(world: SeasonWorld,", "def player_table(rated: RatedWorld)")
t = swap(t, """    network: object = None       # with the box-score prior game by game, the season's network (game_prior.py)
    backward: object = None      # and its backward network, when the first half of each player's games read the rest
    games: pd.DataFrame | None = None    # and the season's game rows (data/player_games.py)
    least_share: float = 1.0     # the smallest part-season share its boosters trained on (`least_share_of`)
    whole_season: bool = False   # and one prior a season from its state after the last game (game_prior_whole_season)
""", """    least_share: float = 1.0     # the smallest part-season share its boosters trained on (`least_share_of`)
""")
t = swap(t, """        if self.network is not None:
            return rate_world_by_game(world, self.saved, self.network, self.games, self.least_share, context,
                                      self.settings, None if whole else self.held_scale, self.backward,
                                      self.whole_season)
""", "")
t = cut(t, "def game_prior_tag(meta: dict,", "def held_scale_table(")
t = swap(t, "    game_tag = game_prior_tag(meta, pieces, whole_season_inputs, whole_season_context)\n", "")
t = swap(t, """        if game_tag:
            attach_networks(run, meta, game_tag)
""", "")
for gone in ("game_rates", "game_box_priors", "load_network", "season_games", "rate_world_by_game", "attach_networks"):
    assert gone not in t, gone
save(path, t, crlf)

# ------------------------------------------------------------------ prior_rapm.py
path = "src/openrapm/prior_rapm.py"
t, crlf = load(path)
t = swap(t, '"lineup_prior_sums", "game_lineup_sums", "game_averages", ', '"lineup_prior_sums", ')
t = cut(t, "def game_lineup_sums(players, game: np.ndarray, tables: tuple, n_players: int):", "@dataclass")
t = swap(t, """def ridge_system(design, prior_offense: dict, prior_defense: dict, penalty: float = 13037.0,
                 scale=None, game_tables: tuple | None = None) -> RidgeSystem:""",
         """def ridge_system(design, prior_offense: dict, prior_defense: dict, penalty: float = 13037.0,
                 scale=None) -> RidgeSystem:""")
t = swap(t, """    columns times their scales come off the target, and the system is [players | context].

    With `game_tables`, the priors change game by game (game_prior.py): each row's lineup prior reads its own game's
    (`game_lineup_sums`), `prior_offense` and `prior_defense` are not read, and the prior the system carries is each
    player's average over his rows (`game_averages`).\"\"\"""",
         """    columns times their scales come off the target, and the system is [players | context].\"\"\"""")
t = swap(t, """    if game_tables is None:
        prior = np.concatenate([aligned_prior(prior_offense, player_ids), aligned_prior(prior_defense, player_ids)])
        sums_o, sums_d = lineup_prior_sums(players, prior, n_players)
    else:
        game = design.rows["game_idx"].to_numpy()
        sums_o, sums_d = game_lineup_sums(players, game, game_tables, n_players)
        prior = game_averages(players, game, np.asarray(design.w, dtype=float), game_tables, n_players)
""", """    prior = np.concatenate([aligned_prior(prior_offense, player_ids), aligned_prior(prior_defense, player_ids)])
    sums_o, sums_d = lineup_prior_sums(players, prior, n_players)
""")
t = swap(t, """def fit(design, prior_offense: dict, prior_defense: dict, penalty: float = 13037.0, scale=None,
        game_tables: tuple | None = None) -> PriorRapmFit:
    \"\"\"The prior-informed ridge on one season's design (the equation in the module docstring), the system of
    `ridge_system` solved; with `scale`, the prior scales held at (s_O, s_D) rather than fitted; with `game_tables`,
    priors that change game by game, the reported prior the scale times each player's average.\"\"\"""",
         """def fit(design, prior_offense: dict, prior_defense: dict, penalty: float = 13037.0, scale=None) -> PriorRapmFit:
    \"\"\"The prior-informed ridge on one season's design (the equation in the module docstring), the system of
    `ridge_system` solved; with `scale`, the prior scales held at (s_O, s_D) rather than fitted.\"\"\"""")
t = swap(t, "    system = ridge_system(design, prior_offense, prior_defense, penalty, scale, game_tables)\n",
         "    system = ridge_system(design, prior_offense, prior_defense, penalty, scale)\n")
t = swap(t, """def rate_season(designs: dict, priors: dict, settings, season: int, scale=None,
                game_tables: dict | None = None) -> pd.DataFrame:""",
         """def rate_season(designs: dict, priors: dict, settings, season: int, scale=None) -> pd.DataFrame:""")
t = swap(t, """    (`held_prior_scale`); None fits them on the season's games, as shipped.  `game_tables`, {target: (table_O,
    table_D)} (game_prior.game_tables on that target's design), pulls each row toward its own game's priors.
    \"\"\"""", """    (`held_prior_scale`); None fits them on the season's games, as shipped.
    \"\"\"""")
t = swap(t, """    fits = {name: fit(designs[name], priors["O"], priors["D"], settings.ridge.penalty, scale,
                      None if game_tables is None else game_tables[name])
            for name in targets.distinct()}""",
         """    fits = {name: fit(designs[name], priors["O"], priors["D"], settings.ridge.penalty, scale)
            for name in targets.distinct()}""")
t = swap(t, """def rate_season_blended(designs: dict, priors: dict, settings, season: int, held=None, weight: float = 0.0,
                        game_tables: dict | None = None) -> pd.DataFrame:""",
         """def rate_season_blended(designs: dict, priors: dict, settings, season: int, held=None,
                        weight: float = 0.0) -> pd.DataFrame:""")
t = swap(t, """        return rate_season(designs, priors, settings, season, None, game_tables)
    if weight <= 0.0:
        return rate_season(designs, priors, settings, season, held, game_tables)
    free = rate_season(designs, priors, settings, season, None, game_tables)""",
         """        return rate_season(designs, priors, settings, season, None)
    if weight <= 0.0:
        return rate_season(designs, priors, settings, season, held)
    free = rate_season(designs, priors, settings, season, None)""")
t = swap(t, "    return rate_season(designs, priors, settings, season, blended_scale(held, fitted, weight), game_tables)\n",
         "    return rate_season(designs, priors, settings, season, blended_scale(held, fitted, weight))\n")
assert "game_tables" not in t and "game_prior" not in t
save(path, t, crlf)

# ------------------------------------------------------------------ box_prior.py
path = "src/openrapm/box_prior.py"
t, crlf = load(path)
t = swap(t, '"rated_rows", "predict", "game_rated_rows",\n           "predict_chosen", "SidePrior",',
         '"rated_rows", "predict",\n           "SidePrior",')
t = cut(t, "def game_rated_rows(rows: pd.DataFrame, feature_names,", "@dataclass\nclass SidePrior:")
save(path, t, crlf)

# ------------------------------------------------------------------ commands/data.py
path = "src/openrapm/commands/data.py"
t, crlf = load(path)
t = swap(t, """    openrapm data player-games [FIRST] [LAST]     each player's games, one row per player and game, from the stored
                                                  design pieces, game logs, box scores and hustle box scores
                                                  (paths.cache/player_games): what the multi-stat network reads
""", "")

def cut_function(text, header):
    """`text` without the top-level function that starts with `header`, up to the next top-level definition."""
    start = text.index(header)
    end = text.index("\ndef ", start + 1) + 1
    return text[:start] + text[end:]


t = cut_function(t, "def run_player_games(args: argparse.Namespace, cfg: dict) -> int:")
t = cut_function(t, "def _add_player_games(steps) -> None:")
t = swap(t, "    _add_player_games(steps)\n", "")
assert "player_games" not in t and "player-games" not in t
save(path, t, crlf)

# ------------------------------------------------------------------ the data layer
path = "src/openrapm/data/dashboards.py"
t, crlf = load(path)
t = cut_function(t, "def hustle_game_rows(cfg: dict, season: int) -> pd.DataFrame | None:")
save(path, t, crlf)

path = "src/openrapm/data/roles.py"
t, crlf = load(path)
t = swap(t, """    box), one row per player in each game's box score, played or not.  The per-game table of data/player_games.py reads
    it; `roles_from_boxes` keeps its own sums, so the season panel's numbers do not move.""",
         """    box), one row per player in each game's box score, played or not.  Check 157 reads it; `roles_from_boxes` keeps
    its own sums, so the season panel's numbers do not move.""")
save(path, t, crlf)

# ------------------------------------------------------------------ commands/second_prior.py: load_models refuses now
path = "src/openrapm/commands/second_prior.py"
t, crlf = load(path)
t = swap(t, """    if ((saved_all.get("meta", {}).get("settings") or {}).get("prior") or {}).get("game_prior"):
        raise SystemExit("these boosters' build read the box-score prior game by game; the second prior's swaps are "
                         "read from folds rated one prior per player-season")
""", "")
save(path, t, crlf)

# ------------------------------------------------------------------ tests
path = "tests/test_cli.py"
t, crlf = load(path)
start = t.index("def test_the_whole_season_game_prior_reads_the_forward_networks_alone():")
end = t.index("\ndef ", start + 1) + 1
t = t[:start] + '''def test_the_game_by_game_priors_flags_are_gone():
    """The game-by-game box-score prior was removed on 2026-11-16 (at the tag archive/research-2026-11): its flags and
    its command are refused, not read and ignored."""
    for flags in (("--game-prior", "net1"), ("--game-prior-backward", "net1_back"), ("--game-prior-whole-season",)):
        with pytest.raises(SystemExit):
            _parse("rate", *flags)
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["game-rates", "train", "--tag", "net1"])


''' + t[end:]
save(path, t, crlf)

path = "tests/test_layer_boundary.py"
t, crlf = load(path)
lines = [line for line in t.split("\n") if not line.lstrip().startswith(('"game_rates.py",', '"game_prior.py",'))]
save(path, "\n".join(lines), crlf)

path = "pyproject.toml"
t, crlf = load(path)
t = swap(t, """# the multi-stat network (openrapm.game_rates) trains with PyTorch; nothing else needs it, and the ratings read its
# saved weights with numpy.  The GPU build: --index-url https://download.pytorch.org/whl/cu130
network = ["torch>=2.4"]
""", "")
save(path, t, crlf)

for gone in ("src/openrapm/game_prior.py", "src/openrapm/game_rates.py", "src/openrapm/commands/game_rates.py",
             "src/openrapm/data/game_rate_files.py", "src/openrapm/data/player_games.py",
             "tests/test_game_prior.py", "tests/test_game_rates.py"):
    Path(gone).unlink()
    print("removed", gone)
