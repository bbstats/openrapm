"""`openrapm game-rates`: the multi-stat networks (game_rates.py), one per rated season, trained and scored.

    openrapm game-rates train --tag net1 [--seasons 1997-2026] [--no-opponent] [--reverse] [--seed 0]
    openrapm game-rates score --tag net1 [--against net1_no_opponent]

`train`, for each rated season S of `--seasons`:
    1. the seasons it may train on: every finished season but S - 1, S and S + 1 (the year-over-year setting's rule,
       so nothing in a season's rating has seen it or the seasons that score it), never a season in progress;
    2. a tenth of the players, those whose id ends in 3, kept out of training to stop it (`VALIDATION_DIGIT`);
    3. the padding's pads and the opponent pads fitted on the other players' rows of those seasons;
    4. the network trained on the GPU (PyTorch) and written to outputs/game_rates/<tag>/season_<S>.npz;
    5. season S forecast by it, which it never saw: each stat's log-loss per game, the network's against the
       padding's, written to outputs/game_rates/<tag>/score_<S>.csv (one file a season, so two processes can each
       train part of the seasons at once).
`score` prints that table: per stat, the network's log-loss per game against the padding's over every scored season;
with `--against`, a second tag's network beside it (e.g. the same networks trained with the opponent factor off).

Example: `openrapm game-rates train --tag net1 --seasons 2010` trains the network of 2010 on 1997-2008 and 2012-2026.
"""
from __future__ import annotations

import time

VALIDATION_DIGIT = 3        # players whose id ends in 3 stop each network's training and never train it


def _scored_rows(games, mask, forecasts, padding, factors, season: int, history: list):
    """One row per stat: the season's games, the padding's and the network's summed log-loss, the passes run."""
    import numpy as np
    import pandas as pd

    from .. import game_rates
    network = game_rates.nll(forecasts, games, factors)[mask].sum(axis=0)
    padding_loss = game_rates.nll(padding, games, factors)[mask].sum(axis=0)
    informed = games.available[mask].sum(axis=0)
    best = int(np.argmin([held for _, _, held in history]))
    return pd.DataFrame({"season": season, "stat": game_rates.NAMES, "games": int(mask.sum()), "informed": informed,
                         "padding_nll": padding_loss, "network_nll": network, "best_pass": best,
                         "passes": len(history)})


def _train_one(season: int, games, league_row, sums: tuple, args, cfg: dict):
    """Steps 1-5 of the module docstring for one rated season."""
    import numpy as np

    from .. import game_rates
    from ..data.game_rate_files import SavedNetwork, save_network
    started = time.time()
    real = ~games.frame["end"].to_numpy(dtype=bool)
    excluded = {season - 1, season, season + 1}
    trained = np.array(sorted(set(np.unique(games.season).tolist()) - excluded), dtype=np.int64)
    in_training = real & np.isin(games.season, trained)
    players = games.frame["player_id"].to_numpy(dtype=np.int64)
    validation = in_training & (players % 10 == VALIDATION_DIGIT)
    fitting = in_training & ~validation
    ks, pads = game_rates.fit_pads(games, league_row, fitting, opponent=not args.no_opponent)
    factors = game_rates.opponent_factors(sums, pads)
    padding = game_rates.padding_forecasts(games, league_row, factors, ks)
    inputs = game_rates.network_inputs(games, league_row, factors, padding)
    scaling = game_rates.standardise(inputs, fitting)
    weights, history = game_rates.train_network(scaling.apply(inputs), games, factors, padding, fitting, validation,
                                                seed=args.seed, verbose=args.verbose)
    save_network(cfg, args.tag, SavedNetwork(
        season=season, weights=weights, mean=scaling.mean, sd=scaling.sd, ks=ks, opponent_pads=pads, seasons=trained,
        validation_players=np.unique(players[validation]), input_names=game_rates.input_names(),
        stats=game_rates.NAMES, reverse=bool(args.reverse)))
    rated = real & (games.season == season)
    forecasts = np.zeros_like(padding)
    forecasts[rated] = game_rates.forecast(weights, scaling, inputs[rated], padding[rated], neutral=False)
    scored = _scored_rows(games, rated, forecasts, padding, factors, season, history)
    gain = (scored.padding_nll.sum() - scored.network_nll.sum()) / max(int(rated.sum()), 1)
    print(f"  {season}: trained on {len(trained)} seasons, best pass {scored.best_pass.iloc[0]} of "
          f"{len(history)}; log-loss per game {scored.padding_nll.sum() / max(int(rated.sum()), 1):.4f} padding, "
          f"{gain:+.4f} better with the network ({time.time() - started:.0f}s)", flush=True)
    return scored


def run_train(args, cfg: dict) -> int:
    """`openrapm game-rates train`: the module docstring's steps for every season asked for."""
    from .. import game_rates
    from ..data.game_rate_files import network_dir
    from ..data.player_games import trainable_player_games
    from ..data.seasons import trainable_through
    last = trainable_through(cfg)
    every = range(int(cfg["first_season"]), last + 1)
    seasons = [int(s) for s in (args.seasons or every)]
    started = time.time()
    rows = trainable_player_games(cfg, every)
    games = game_rates.games_from_rows(game_rates.reverse_rows(rows) if args.reverse else rows)
    league_row = game_rates.league_by_row(games, game_rates.league_levels(games))
    sums = game_rates.opponent_sums(games, league_row)
    print(f"game rows {int((~games.frame['end']).sum()):,} over {len(every)} seasons, {games.n_player_seasons:,} "
          f"player-seasons ({time.time() - started:.0f}s)", flush=True)
    folder = network_dir(cfg, args.tag)
    for season in seasons:
        out = folder / f"score_{season}.csv"
        if out.exists() and not args.force:
            print(f"  {season}: already trained, skipped (--force trains it again)", flush=True)
            continue
        scored = _train_one(season, games, league_row, sums, args, cfg)
        out.parent.mkdir(parents=True, exist_ok=True)
        scored.to_csv(out, index=False)
    print(f"done ({time.time() - started:.0f}s); scores in {folder}", flush=True)
    return 0


def score_table(scores):
    """Per stat over every scored season: games informed, the padding's and the network's log-loss per game, the
    network's gain per game, its share of the padding's loss, the seasons the network won."""
    import pandas as pd

    from .. import game_rates
    scores = scores.assign(gain=scores.padding_nll - scores.network_nll)
    table = scores.groupby("stat", sort=False).agg(informed=("informed", "sum"), padding=("padding_nll", "sum"),
                                                   network=("network_nll", "sum"), gain=("gain", "sum"),
                                                   seasons_won=("gain", lambda g: int((g > 0).sum())),
                                                   seasons=("gain", "size"))
    games = scores.drop_duplicates("season").games.sum()
    for column in ("padding", "network", "gain"):
        table[column] = table[column] / games
    table = table.reindex(game_rates.NAMES)
    total = pd.DataFrame({"informed": [table.informed.sum()], "padding": [table.padding.sum()],
                          "network": [table.network.sum()], "gain": [table.gain.sum()],
                          "seasons_won": [int((scores.groupby("season").gain.sum() > 0).sum())],
                          "seasons": [scores.season.nunique()]}, index=["all 21"])
    return pd.concat([table, total])


def run_score(args, cfg: dict) -> int:
    """`openrapm game-rates score`: the table of the module docstring."""
    import pandas as pd

    from ..data.game_rate_files import network_dir
    tables = {}
    for tag in [args.tag, *([args.against] if args.against else [])]:
        paths = sorted(network_dir(cfg, tag).glob("score_*.csv"))
        if not paths:
            raise SystemExit(f"no scores under {network_dir(cfg, tag)}; train them with "
                             f"`openrapm game-rates train --tag {tag}`")
        tables[tag] = score_table(pd.concat([pd.read_csv(path) for path in paths], ignore_index=True))
    pd.set_option("display.width", 200)
    for tag, table in tables.items():
        print(f"\n{tag}: log-loss per game, every season forecast by the network that never saw it")
        print(table.round(5).to_string(), flush=True)
    return 0


def add_game_rates(commands, parse_seasons) -> None:
    """`openrapm game-rates` and its two steps."""
    parser = commands.add_parser(
        "game-rates", help="the multi-stat networks: one per rated season, trained on the GPU and scored",
        description="Trains and scores the multi-stat networks (src/openrapm/game_rates.py): for each rated season, "
                    "a network that forecasts each player's 21 box-score rates before every game of the season "
                    "from his earlier games of it, trained on every finished season but that one and its two "
                    "neighbours.")
    steps = parser.add_subparsers(dest="game_rates_command", required=True, metavar="STEP")
    train = steps.add_parser("train", help="train one network per rated season, written to outputs/game_rates/<tag>/")
    train.add_argument("--tag", required=True, help="the folder under outputs/game_rates/")
    train.add_argument("--seasons", type=parse_seasons, default=None,
                       help="the rated seasons: 2026, 2017-2026 or 2017,2018 (default every finished season)")
    train.add_argument("--seed", type=int, default=0, help="the random draws' seed (default %(default)s)")
    train.add_argument("--reverse", action="store_true",
                       help="each season in reverse order: the network forecasts a game from the games after it, "
                            "the backward pass `openrapm rate --game-prior-backward` reads")
    train.add_argument("--no-opponent", action="store_true",
                       help="no opponent factor: every game's exposure as played (the comparison arm)")
    train.add_argument("--force", action="store_true", help="train seasons already trained again")
    train.add_argument("--verbose", action="store_true", help="print every pass")
    train.set_defaults(run=run_train)
    score = steps.add_parser("score", help="print the networks' log-loss per stat against the padding's")
    score.add_argument("--tag", required=True, help="the folder under outputs/game_rates/")
    score.add_argument("--against", default="", help="a second tag printed beside it")
    score.set_defaults(run=run_score)
