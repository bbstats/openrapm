"""One team's swap adjustment (step 5), player by player, rebuilt from its parts and checked against the shipped table.

The owner (2026-10-05): "Explain the swap adjustment very clearly for every BOS player in this example."  This
re-runs step 5 as `openrapm swap` ships it (src/openrapm/swap_adjust.py, the product setting) on the shipped step-4
table (`--base`) and keeps every intermediate number, so each player's move reads as a sum of five parts:

    type          the type model's prediction for him: what players with his box-score rates, role, age, size and
                  box prior beat or missed their ratings by in other seasons' lineup swaps (swap_adjust.py defines
                  the model), times the applied share, 0.5.  Its three largest terms are listed beside it
    give-back     the team constraint: the team's possession-weighted total may not move, so the team's net type
                  prediction is taken back from its players in proportion to their share of its possessions
    centring      the season's possession-weighted mean set back to the step-4 table's, one constant per side (the
                  `_centering` columns)
    spread        the width the adjustment added, given back: each player's distance from his main team's mean
                  scaled by one factor per season and side (swap_adjust.hold_spread_within)
    final         the sum of the four, which must equal the shipped table's c_off / c_def for every player of the
                  season to within 1e-9 (the run stops otherwise)

Every number is in points per 100 possessions, positive = good on both ends.  `team_share` is the player's share of
his team's possessions on both ends together, the weight of his give-back.

Example: 2026, Derrick White (DECISIONS.md, experiment 41): 2.60 before the swap step, 1.25 after, the league's
largest swap charge, -1.35, of which -0.99 is give-back.

Reads   the step-4 table (outputs/<base>.parquet), the shipped table (outputs/season_ratings_product.parquet), the
        season panel, the stints of every season (the type model trains on every season's swaps), the roles table
        (each player's main team) and the season's regular-season game log (the team's id from its abbreviation).
Writes  outputs/swap_explain_<team>_<season>.csv (the team's players) and outputs/swap_explain_all_<season>.csv
        (every player of the season, and of each `--also` season).

    python scripts/checks/113_swap_explain.py --team=BOS --player="Derrick White" --also=2024,2025
"""
import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from openrapm import swap_adjust as sa
from openrapm import swap_test as sw
from openrapm.commands.console import use_utf8
from openrapm.data.config import load_config, resolve
from openrapm.data.ingest import raw_dir
from openrapm.data.panel import panel_and_trainable_seasons
from openrapm.data.roles import main_teams
from openrapm.data.stints import load_season_stints
from openrapm.data.tables import load_ratings
from openrapm.settings import PRODUCT

NAMES = {"threes_made_per100": "made threes", "threes_missed_per100": "missed threes",
         "twos_made_per100": "made twos", "twos_missed_per100": "missed twos",
         "free_throws_made_per100": "made free throws", "free_throws_missed_per100": "missed free throws",
         "off_rebounds_per100": "offensive rebounds", "def_rebounds_per100": "defensive rebounds",
         "assists_per100": "assists", "turnovers_per100": "turnovers", "steals_per100": "steals",
         "blocks_per100": "blocks", "fouls_per100": "fouls", "team_possession_share": "share of team possessions",
         "games_started_share": "share of games started", "garbage_time_share": "garbage-time share", "age": "age",
         "seasons_before": "seasons played", "height_in": "height", "weight_lb": "weight",
         "prior_off": "box prior (offense)", "prior_def": "box prior (defense)"}
SHIPPED_TABLE = "season_ratings_product.parquet"
MAX_GAP = 1e-9                     # the rebuilt adjustment must match the shipped table's c_off / c_def this closely
TEAM_COLUMNS = ("player_name", "team_share", "off_type", "off_giveback", "off_centering", "off_spread", "off_final",
                "def_type", "def_giveback", "def_centering", "def_spread", "def_final", "swap_total")


@dataclass(frozen=True)
class Inputs:
    """What every season's rebuild reads, loaded once.

    base       the step-4 table (`--base`), positive = good
    raw        the same as (player_id, season, o, d) in raw sign (d = points allowed)
    features   the type features by (player_id, season) (swap_adjust.type_features)
    fold_of    player_id -> his player fold
    trainable  the seasons the type model may train on (never a season still being played)
    swaps      season -> swap_adjust.SeasonSwaps: the season's swaps, possessions per team and grams
    main_team  (player_id, season) -> his main team
    shipped    the shipped table
    """
    base: pd.DataFrame
    raw: pd.DataFrame
    features: pd.DataFrame
    fold_of: pd.Series
    trainable: set
    swaps: dict
    main_team: pd.Series
    shipped: pd.DataFrame


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="One team's swap adjustment, step by step.", allow_abbrev=False)
    parser.add_argument("--team", default="BOS", help="the team's abbreviation (default %(default)s)")
    parser.add_argument("--season", type=int, default=2026, help="default %(default)s")
    # the shipped table's step-4 input: experiment 46's since 2026-10-09 (`openrapm build` writes it as
    # season_ratings_product_step4; before, experiment 45's season_ratings_product_lgb_noonc_priorshrink_pre_swap)
    parser.add_argument("--base", default="season_ratings_product_weightfix_priorshrink_pre_swap",
                        help="outputs/<BASE>.parquet, the step-4 table (default %(default)s)")
    parser.add_argument("--also", default="", help="more seasons to rebuild, comma-separated")
    parser.add_argument("--player", default="", help="one player's every term, by name")
    return parser.parse_args(argv)


# ------------------------------------------------------------------------------------------- the inputs
def all_season_swaps(cfg: dict, raw: pd.DataFrame, features: pd.DataFrame, fold_of: pd.Series,
                     n_folds: int) -> dict:
    """{season: swap_adjust.SeasonSwaps} for every season of the config, against the step-4 ratings."""
    first, last = int(cfg["first_season"]), int(cfg["last_season"])
    margin_clip = float(cfg.get("margin_clip", 25))
    swaps = {}
    for season in range(first, last + 1):
        stints = load_season_stints(season, cfg, sw.STINT_COLUMNS)
        lineups = sw.prepare(sw.season_rows(stints, margin_clip=margin_clip), season)
        swaps[season] = sa.season_swaps(lineups, raw[raw.season == season], sa.season_features(features, season),
                                        fold_of, n_folds)
    print(f"swap pairs and grams built for {first}-{last}", flush=True)
    return swaps


def load_inputs(cfg: dict, base_name: str, settings) -> Inputs:
    """The step-4 table, its type features and every season's swaps; `settings` is settings.SwapSettings."""
    outputs = resolve(cfg, "outputs")
    base_path = outputs / f"{base_name}.parquet"
    base = pd.read_parquet(base_path)
    raw = load_ratings(base_path)
    panel, trainable = panel_and_trainable_seasons(cfg)
    features = sa.type_features(panel, base)
    fold_of = sa.player_folds(raw.player_id, settings.n_folds)
    swaps = all_season_swaps(cfg, raw, features, fold_of, settings.n_folds)
    return Inputs(base=base, raw=raw, features=features, fold_of=fold_of, trainable=trainable, swaps=swaps,
                  main_team=main_teams(cfg).set_index(["player_id", "season"]).team_id,
                  shipped=pd.read_parquet(outputs / SHIPPED_TABLE))


# ------------------------------------------------------------------------------------------- one season
def fold_terms(features: pd.DataFrame, grams: list, fold_of: pd.Series, settings) -> tuple[pd.Series, pd.DataFrame]:
    """(each player's type prediction m, each of its terms), from the model fitted without his fold's pairs.

        m_i = x_i' beta_fold(i),   term_ij = x_ij beta_fold(i)_j
    """
    prediction = pd.Series(np.nan, index=features.index, dtype=float)
    terms = pd.DataFrame(np.nan, index=features.index, columns=features.columns)
    for fold in range(settings.n_folds):
        beta = sa.type_coefficients(grams, fold, settings.type_model_penalty)
        mine = fold_of.reindex(features.index).to_numpy() == fold
        prediction[mine] = features[mine].to_numpy() @ beta
        terms[mine] = features[mine].to_numpy() * beta
    return prediction, terms


def type_model(inputs: Inputs, season: int, settings) -> tuple[dict, dict, pd.DataFrame]:
    """({side: type prediction}, {side: its terms}, the season's type features), trained on the other seasons."""
    features = inputs.features.xs(season, level="season")
    train = sa.training_seasons(season, list(inputs.swaps), inputs.trainable, settings.exclude_near)
    prediction, terms = {}, {}
    for side in sa.SIDES:
        prediction[side], terms[side] = fold_terms(features, [inputs.swaps[other].grams[side] for other in train],
                                                   inputs.fold_of, settings)
    return prediction, terms, features


def top_terms(row: pd.Series, sign: float, share: float) -> str:
    """A player's three largest terms, applied share included, in plain words: "made threes +0.41; ..."."""
    if not row.notna().any():
        return ""
    largest = row.dropna().sort_values(key=np.abs, ascending=False).head(3)
    return "; ".join(f"{NAMES[column]} {sign * share * value:+.2f}" for column, value in largest.items())


def side_steps(rows: pd.DataFrame, side: str, prediction: pd.Series, terms: pd.DataFrame, kept: sa.SeasonSwaps,
               weight: np.ndarray, team: np.ndarray, settings) -> dict:
    """One side's five parts for every player of the season, positive = good, as `<off|def>_<part>` columns.

    In swap_adjust's order: give back kappa m inside each team, centre the season (possession-weighted mean 0),
    hold the spread inside teams.  `weight` is the step-4 possessions on this end, `team` each player's main team.
    """
    share = settings.applied_share
    applied = share * prediction
    given_back = sa.give_back(*sa.possession_shares(kept.pairs[side], kept.team_possessions[side]), applied)
    typed = rows.player_id.map(applied).fillna(0.0).to_numpy()
    solved = rows.player_id.map(given_back).fillna(0.0).to_numpy()
    centred = solved - float(np.sum(solved * weight) / np.sum(weight))
    base = rows["o" if side == "offense" else "d"].to_numpy()
    held, factor = sa.hold_spread_within(base + centred, base, weight, team)
    final = held - base
    sign = 1.0 if side == "offense" else -1.0
    tag = side[:3]
    return {f"{tag}_type": sign * typed, f"{tag}_giveback": sign * (solved - typed),
            f"{tag}_centering": sign * (centred - solved), f"{tag}_spread": sign * (final - centred),
            f"{tag}_final": sign * final, f"{tag}_spread_factor": factor,
            f"{tag}_top_terms": [top_terms(row, sign, share) for _, row in terms.reindex(rows.player_id).iterrows()]}


def shipped_season(inputs: Inputs, season: int) -> pd.DataFrame:
    """The shipped table's rows of one season, by player_id."""
    return inputs.shipped[inputs.shipped.season == season].set_index("player_id")


def check_against_shipped(out: pd.DataFrame, shipped: pd.DataFrame, season: int) -> None:
    """The rebuilt final parts against the shipped c_off / c_def: print the largest difference, stop above MAX_GAP."""
    rebuilt = out.set_index("player_id")
    gap = max(float((rebuilt.off_final - shipped.c_off.reindex(out.player_id)).abs().max()),
              float((rebuilt.def_final - shipped.c_def.reindex(out.player_id)).abs().max()))
    print(f"reproduces the shipped swap adjustment for every {season} player: max difference {gap:.2e}")
    if not gap < MAX_GAP:
        raise SystemExit(f"the rebuilt swap adjustment differs from the shipped table by {gap}")


def rebuild(inputs: Inputs, season: int, settings, outputs: Path) -> tuple:
    """One season's shipped swap adjustment, every part kept, checked against the shipped table; writes
    outputs/swap_explain_all_<season>.csv.  Returns (the parts, the type terms, the season's features)."""
    prediction, terms, features = type_model(inputs, season, settings)
    rows = inputs.raw[inputs.raw.season == season].copy().reset_index(drop=True)
    key = pd.MultiIndex.from_arrays([rows.player_id, rows.season])
    indexed = inputs.base.set_index(["player_id", "season"])
    team = inputs.main_team.reindex(key).to_numpy(dtype=float)
    out = pd.DataFrame({"player_id": rows.player_id})
    for side, possessions in (("offense", "poss_off"), ("defense", "poss_def")):
        weight = indexed[possessions].reindex(key).fillna(0).to_numpy()
        steps = side_steps(rows, side, prediction[side], terms[side], inputs.swaps[season], weight, team, settings)
        for column, values in steps.items():
            out[column] = values
    shipped = shipped_season(inputs, season)
    check_against_shipped(out, shipped, season)
    league = out.merge(shipped[["player_name", "poss_off", "rating_total"]].reset_index(), on="player_id", how="left")
    league.to_csv(outputs / f"swap_explain_all_{season}.csv", index=False)
    return out, terms, features


# ------------------------------------------------------------------------------------------- one player
def own_swaps_line(pairs: pd.DataFrame, player: int, side: str) -> str:
    """His own swaps' information-weighted mean residual, signed so positive = his lineups beat the ratings'
    prediction against the same four teammates with someone else; with its naive standard error."""
    mine = (pairs.fifth_a == player) | (pairs.fifth_b == player)
    his = pairs[mine]
    residual = np.where(his.fifth_a == player, his.r, -his.r) * (1.0 if side == "offense" else -1.0)
    information = his.h.to_numpy()
    mean = float(np.sum(information * residual) / np.sum(information))
    sd = float(np.sqrt(np.sum(information * (residual - mean) ** 2) / np.sum(information)))
    effective = float(np.sum(information) ** 2 / np.sum(information ** 2))
    return (f"  {side}: {mine.sum():,} pairs, information {np.sum(information):,.0f} possessions, mean residual "
            f"{mean:+.2f} per 100 (naive se {sd / np.sqrt(effective):.2f}, effective pairs {effective:.0f})")


def print_player(who: str, season: int, terms: dict, features: pd.DataFrame, shipped: pd.DataFrame,
                 kept: sa.SeasonSwaps, share: float) -> None:
    """Every term of one player's type prediction, then his own swaps of the season (which the shipped
    adjustment does not read)."""
    player = int(shipped.index[shipped.player_name == who][0])
    print(f"\n=== {who}: every term of the type prediction (half strength, positive good), with his standardised "
          f"feature value")
    for side in sa.SIDES:
        sign = 1.0 if side == "offense" else -1.0
        row = terms[side].loc[player]
        table = pd.DataFrame({"feature": [NAMES[c] for c in row.index], "z": features.loc[player].to_numpy(),
                              "term": sign * share * row.to_numpy()}).sort_values("term", key=np.abs, ascending=False)
        print(f"-- {side}: sum {table.term.sum():+.3f}")
        print(table.round(3).to_string(index=False))
    print(f"\n=== {who}: his own {season} lineup swaps (not used by the shipped adjustment)")
    for side in sa.SIDES:
        print(own_swaps_line(kept.pairs[side], player, side))


# ------------------------------------------------------------------------------------------- one team
def team_id_of(cfg: dict, team_code: str, season: int) -> int:
    """The team's id, from its abbreviation in the season's regular-season game log."""
    games = pd.read_parquet(raw_dir(cfg) / "gamelog" / f"{season}_RS.parquet",
                            columns=["PLAYER_ID", "TEAM_ID", "TEAM_ABBREVIATION"])
    return int(games[games.TEAM_ABBREVIATION == team_code].TEAM_ID.iloc[0])


def team_table(out: pd.DataFrame, kept: sa.SeasonSwaps, shipped: pd.DataFrame, team: int) -> pd.DataFrame:
    """The team's players: their parts, their share of its possessions and their shipped numbers, best first."""
    played = pd.concat([kept.team_possessions[side].assign(side=side) for side in sa.SIDES])
    played = played[played.team == team]
    share = played.groupby("player_id").poss.sum() / played.poss.sum()
    table = out[out.player_id.isin(share.index)].copy()
    table["team_share"] = table.player_id.map(share)
    table = table.merge(shipped[["player_name", "poss_off", "rating_total", "prior_off", "prior_def", "u_off",
                                 "u_def"]].reset_index(), on="player_id", how="left")
    table["swap_total"] = table.off_final + table.def_final
    return table.sort_values("rating_total", ascending=False)


def print_team(table: pd.DataFrame) -> None:
    """One team's table, each player's largest terms per side, and each side's spread factor."""
    with pd.option_context("display.width", 260, "display.max_columns", 40, "display.max_colwidth", 80):
        print(table[list(TEAM_COLUMNS)].round(3).to_string(index=False))
        print(table[["player_name", "off_top_terms", "def_top_terms"]].to_string(index=False))
    for side in ("off", "def"):
        print(f"{side}: spread factor {table[f'{side}_spread_factor'].iloc[0]:.3f}")


def main() -> None:
    use_utf8()
    args = parse_args()
    cfg = load_config()
    settings = PRODUCT.swap
    outputs = resolve(cfg, "outputs")
    inputs = load_inputs(cfg, args.base, settings)
    for extra in [int(x) for x in args.also.split(",") if x]:
        rebuild(inputs, extra, settings, outputs)
    out, terms, features = rebuild(inputs, args.season, settings, outputs)
    shipped = shipped_season(inputs, args.season)
    kept = inputs.swaps[args.season]
    if args.player:
        print_player(args.player, args.season, terms, features, shipped, kept, settings.applied_share)
    table = team_table(out, kept, shipped, team_id_of(cfg, args.team, args.season))
    path = outputs / f"swap_explain_{args.team}_{args.season}.csv"
    table.to_csv(path, index=False)
    print_team(table)
    print(f"wrote outputs/{path.name}")


if __name__ == "__main__":
    main()
