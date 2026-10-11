"""Which players does a year-over-year difference come from?  The error difference between two ratings tables,
shared among the players on the court and added up by player group.

The year-over-year test (scripts/checks/63_yoy.py) scores a team-game, one team's points in one game, predicted
from the ten players' ratings of a neighbouring season.  Its splits label a stint or a team-game, so they cannot
say which PLAYERS a difference came from.  This check can, exactly:

    contribution of player i = sum over the team-games g he played in of  n_ig x (e_g^2 - r_g^2) / N

    e_g, r_g    the candidate's and the reference's miss on team-game g, points per 100 possessions
    n_ig        player i's possessions in team-game g, each of the ten on the court holding a tenth of a row's
                possessions, so the ten players' n_ig add up to the team-game's possessions
    N           the scored season's possessions

The contributions add up, season by season, to the paired team-game difference 63_yoy.py reports, so a group's
total is its part of that number and the groups recombine exactly (the run prints the largest gap, which is
rounding error).  The same is done at stint level.

The groups are fixed by the REFERENCE table or by facts, so every candidate is cut the same way:
    quality   the reference's rank by total rating in the RATED season (the season whose ratings are used): top 30,
              31-90, 91-150, 151-300, 301+, and "no rating" (not in that season's table, so rated 0 by both)
    movers    "changed teams" if his main team (most box-score minutes) in the scored season differs from the
              rated season's, "same team" if not, "not in both" if he has no main team in one of them
    age       his age in the scored season (the season panel's `age`): 23 and under, 24-26, 27-29, 30-32, 33+

Reads
    `--ref` and each `--cands` table: player_id, season, rating_off, rating_def, rating_total, poss_off;
    `name=prev_path|next_path` gives a candidate one file per direction (the retired scripts/89_stitch_by_move.py)
    the points design of every scored season, the box scores (main teams) and the season panel's ages
Writes
    outputs/yoy_by_player_<tag>.parquet: one row per scored season, direction, candidate, split and group, with
    tg (team-game), stint, poss_share and tg_abs.  `tg_abs` shares the candidate's OWN team-game squared error out
    the same way, so a group's tg_abs / poss_share is the mean squared miss of the team-games it played in: each
    group's accuracy, not only its part of a difference

How to read the printed tables
    Each group is read the way the test is read: per direction ("next" = the next season's ratings looking back,
    "prev" = the previous season's looking forward, "both" = the two as separate observations) the mean over
    scored seasons, z = mean / standard error, and wins = the seasons in which the group made the candidate
    better.  Below zero = the candidate is better in that group.
Example: the noise control (the incumbent rebuilt with the box prior's players assigned to other folds) against the
incumbent, team-game level: the top 30 play 11% of the possessions (poss_share 0.1128) and read both_mean -0.0103,
both_z -1.57, 30 wins of 56, which is within noise, as a noise control should be.

    python scripts/checks/88_yoy_by_player.py --cands=new=outputs/season_ratings_new.parquet \\
        --ref=incumbent=outputs/season_ratings_weightfix.parquet --tag=new
"""
import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from openrapm.commands.console import one_thread, use_utf8
from openrapm.commands.year_over_year import scored_seasons
from openrapm.data.box_scores import main_team_by_minutes
from openrapm.data.config import load_config, resolve, resolve_path
from openrapm.data.designs import DesignCache
from openrapm.data.seasons import drop_untrainable
from openrapm.data.tables import load_ratings
from openrapm.scoring import QUALITY_LABELS, Prediction, Ratings, predict_season, quality_tiers

AGE_EDGES = (0, 24, 27, 30, 33, 200)
AGE_LABELS = ("23 and under", "24-26", "27-29", "30-32", "33+")
SPLITS = ("quality", "movers", "age")
DIRECTIONS = (("prev", -1), ("next", +1))          # the rated season relative to the scored one
GROUP_ORDER = (("quality", (*QUALITY_LABELS, "no rating")), ("movers", ("same team", "changed teams", "not in both")),
               ("age", (*AGE_LABELS, "unknown")), ("all", ("all",)))
LEVELS = (("tg", "team-game level (what the test decides on)"), ("stint", "stint level"))
SHOWN = ("group", "poss_share", "next_mean", "next_z", "next_wins", "prev_mean", "prev_z", "prev_wins", "both_mean",
         "both_z", "both_wins")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Where a year-over-year difference comes from, by player group.",
                                     allow_abbrev=False)
    parser.add_argument("--ref", default="incumbent=outputs/season_ratings_unshrinkdef.parquet",
                        help="name=path of the reference rankings (default %(default)s)")
    parser.add_argument("--cands", default="", help="name=path[,name=path...], or name=prev_path|next_path")
    parser.add_argument("--splits", default="quality,movers,age", help="default %(default)s")
    parser.add_argument("--first", type=int, help="first scored season (default the second configured season)")
    parser.add_argument("--last", type=int, help="last scored season (default the second-last configured season)")
    parser.add_argument("--tag", help="outputs/yoy_by_player_<TAG>.parquet (default the candidates' names)")
    return parser.parse_args(argv)


# ------------------------------------------------------------------------------------------------ the inputs
def read_table(path, cfg: dict) -> pd.DataFrame:
    """player_id, season, o, d (raw sign: d is points allowed) and poss."""
    return load_ratings(resolve_path(cfg, path), possessions=True)


def read_by_direction(spec: str, cfg: dict) -> dict:
    """`a.parquet` for both directions, or `a.parquet|b.parquet`: the first for "prev" (its ratings look forward at
    the scored season), the second for "next"."""
    paths = spec.split("|")
    if len(paths) not in (1, 2):
        raise SystemExit(f"--cands {spec}: one path, or two joined by |")
    return {"prev": read_table(paths[0], cfg), "next": read_table(paths[-1], cfg)}


def read_ages(cfg: dict) -> pd.DataFrame:
    """player_id, season, age from the season panel's offence rows, read through the season gate like every other
    reader of the panel."""
    panel, _ = drop_untrainable(pd.read_parquet(resolve(cfg, "panel"),
                                                columns=["window", "player_id", "season", "side", "age"]),
                                cfg, what="the season panel")
    return panel[panel.side == "O"]


# ----------------------------------------------------------------------------------------- one scored season
@dataclass(frozen=True)
class ScoredSeason:
    """A scored season's design and what sharing a miss out needs: who is on the court in each row, and each row's
    team-game."""
    season: int
    design: object
    on_court: sp.csr_matrix     # player x row: 1 when the player is on the court, either end
    ids: np.ndarray             # the design's player ids, in column order
    game: np.ndarray            # each row's team-game, 0 .. n_games - 1
    n_games: int


def scored_season(designs: DesignCache, season: int) -> ScoredSeason:
    """The scored season's points design, its on-court matrix and its team-game index."""
    design = designs.season(season, "pts")
    m = design.spec.n_ps
    players = sp.csr_matrix(design.X[:, :2 * m])
    on_court = (players[:, :m] + players[:, m:]).T.tocsr()
    game_key = design.rows.game_idx.to_numpy() * 2 + design.rows.is_home_off.to_numpy().astype(int)
    game, games = pd.factorize(game_key)
    return ScoredSeason(season, design, on_court, design.spec.ps_table["player_id"].to_numpy(np.int64), game,
                        len(games))


def predict(table: pd.DataFrame, rated: int, design) -> Prediction:
    """The rated season's rows of `table` predicting the scored season's design (no fill: absent players are 0)."""
    return predict_season(Ratings(table[table.season == rated][["player_id", "o", "d", "poss"]]
                                  .reset_index(drop=True)), design)


def team_game_errors(p: Prediction, game: np.ndarray, n_games: int) -> np.ndarray:
    """Per team-game: (actual - predicted points) / possessions x 100, the unit 63_yoy.py scores."""
    actual = np.bincount(game, weights=p.y * p.poss / 100.0, minlength=n_games)
    predicted = np.bincount(game, weights=p.pred * p.poss / 100.0, minlength=n_games)
    poss = np.bincount(game, weights=p.poss, minlength=n_games)
    return (actual - predicted) / np.where(poss > 0, poss, 1.0) * 100.0


# ------------------------------------------------------------------------------------------------ the groups
def quality_groups(tiers: pd.Series, rated: int, ids: np.ndarray) -> np.ndarray:
    """Each player's quality tier in the reference's rated season, "no rating" when it has no row for him."""
    key = pd.MultiIndex.from_arrays([ids, np.full(len(ids), rated)])
    return tiers.reindex(key).fillna("no rating").to_numpy(dtype=object)


def mover_groups(cfg: dict, scored: int, rated: int, ids: np.ndarray) -> np.ndarray:
    """"changed teams", "same team" or "not in both", by each player's main team in the two seasons."""
    now, then = main_team_by_minutes(scored, cfg), main_team_by_minutes(rated, cfg)
    return np.array(["not in both" if (q not in now or q not in then) else
                     ("changed teams" if now[q] != then[q] else "same team") for q in map(int, ids)], dtype=object)


def age_groups(ages: pd.Series, ids: np.ndarray) -> np.ndarray:
    """Each player's age group in the scored season, "unknown" without an age."""
    age = ages.reindex(ids).to_numpy(dtype=float)
    out = np.full(len(ids), "unknown", dtype=object)
    known = ~np.isnan(age)
    out[known] = np.asarray(AGE_LABELS, dtype=object)[np.digitize(age[known], AGE_EDGES[1:-1])]
    return out


def group_labels(split: str, inputs: dict, scored: ScoredSeason, rated: int) -> np.ndarray:
    """Each player's group under `split` (one of SPLITS) for one scored season and rated season."""
    if split == "quality":
        return quality_groups(inputs["tiers"], rated, scored.ids)
    if split == "movers":
        return mover_groups(inputs["cfg"], scored.season, rated, scored.ids)
    ages = inputs["ages"]
    return age_groups(ages[ages.season == scored.season].set_index("player_id").age, scored.ids)


# ---------------------------------------------------------------------------------------- the contributions
@dataclass(frozen=True)
class Reference:
    """The reference's prediction of one scored season from one rated season, and the season's team-game sums."""
    prediction: Prediction
    poss_game: np.ndarray       # possessions per team-game
    share: np.ndarray           # each player's share of the season's possessions
    errors: np.ndarray          # the reference's miss per team-game


def reference_for(table: pd.DataFrame, rated: int, scored: ScoredSeason) -> Reference:
    """The reference's side of every candidate comparison in one scored season and direction."""
    p = predict(table, rated, scored.design)
    poss_game = np.bincount(scored.game, weights=p.poss, minlength=scored.n_games)
    share = np.asarray(scored.on_court @ (p.poss / 10.0)).ravel() / p.poss.sum()
    return Reference(p, poss_game, share, team_game_errors(p, scored.game, scored.n_games))


def contributions(p: Prediction, ref: Reference, scored: ScoredSeason) -> dict:
    """The season totals ("total_tg", "total_stint", "total_abs") and each player's share of them ("tg", "stint",
    "tg_abs"): the candidate-minus-reference squared error per team-game and per stint, and the candidate's own
    team-game squared error."""
    errors = team_game_errors(p, scored.game, scored.n_games)
    poss_row, poss_game, game = ref.prediction.poss, ref.poss_game, scored.game
    delta_game = errors ** 2 - ref.errors ** 2
    w = p.w
    delta_stint = (p.y - p.pred) ** 2 - (p.y - ref.prediction.pred) ** 2
    return dict(
        total_tg=float(np.sum(poss_game * delta_game) / poss_game.sum()),
        tg=np.asarray(scored.on_court @ (poss_row * delta_game[game] / (10.0 * poss_game.sum()))).ravel(),
        total_stint=float(np.sum(w * delta_stint) / w.sum()),
        stint=np.asarray(scored.on_court @ (w * delta_stint / (10.0 * w.sum()))).ravel(),
        total_abs=float(np.sum(poss_game * errors ** 2) / poss_game.sum()),
        tg_abs=np.asarray(scored.on_court @ (poss_row * (errors ** 2)[game] / (10.0 * poss_game.sum()))).ravel())


def gap(parts: dict) -> float:
    """How far the players' shares are from adding up to the season totals: should be rounding error."""
    return max(abs(parts["tg"].sum() - parts["total_tg"]), abs(parts["stint"].sum() - parts["total_stint"]))


def group_rows(key: dict, parts: dict, share: np.ndarray, labels: dict) -> list[dict]:
    """The result rows of one candidate in one scored season and direction: the whole season, then every group."""
    rows = [dict(**key, split="all", group="all", tg=parts["total_tg"], stint=parts["total_stint"], poss_share=1.0,
                 tg_abs=parts["total_abs"])]
    for split, label in labels.items():
        for group in pd.unique(label):
            k = label == group
            rows.append(dict(**key, split=split, group=str(group), tg=float(parts["tg"][k].sum()),
                             stint=float(parts["stint"][k].sum()), poss_share=float(share[k].sum()),
                             tg_abs=float(parts["tg_abs"][k].sum())))
    return rows


def direction_rows(scored: ScoredSeason, direction: str, rated: int, inputs: dict, splits: list) -> tuple:
    """(result rows, largest gap) of every candidate in one scored season and direction."""
    ref = reference_for(inputs["ref"], rated, scored)
    labels = {split: group_labels(split, inputs, scored, rated) for split in splits}
    rows, worst = [], 0.0
    for name, by_direction in inputs["tables"].items():
        table = by_direction[direction]
        if not (table.season == rated).any():
            continue
        parts = contributions(predict(table, rated, scored.design), ref, scored)
        worst = max(worst, gap(parts))
        rows += group_rows(dict(season=scored.season, direction=direction, candidate=name), parts, ref.share, labels)
    return rows, worst


def season_rows(scored: ScoredSeason, inputs: dict, splits: list) -> tuple:
    """(result rows, largest gap) of one scored season, both directions."""
    rows, worst = [], 0.0
    for direction, step in DIRECTIONS:
        rated = scored.season + step
        if not (inputs["ref"].season == rated).any():
            continue
        more, more_gap = direction_rows(scored, direction, rated, inputs, splits)
        rows += more
        worst = max(worst, more_gap)
    return rows, worst


# ------------------------------------------------------------------------------------------------ the report
def summarise(frame: pd.DataFrame, value: str) -> pd.DataFrame:
    """Per candidate, split, group: mean over observations, z and wins, per direction and both together."""
    out = []
    for (cand, split, group), g in frame.groupby(["candidate", "split", "group"], sort=False):
        row = dict(candidate=cand, split=split, group=group, poss_share=float(g.poss_share.mean()))
        for label, part in (("next", g[g.direction == "next"]), ("prev", g[g.direction == "prev"]), ("both", g)):
            v = part[value].to_numpy(float)
            se = float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else np.nan
            row.update({f"{label}_mean": float(v.mean()), f"{label}_z": float(v.mean() / se) if se else np.nan,
                        f"{label}_wins": int((v < 0).sum()), f"{label}_n": len(v)})
        out.append(row)
    return pd.DataFrame(out)


def print_group_table(summary: pd.DataFrame, cand: str, split: str) -> None:
    """One candidate's groups under one split, in the groups' natural order."""
    part = summary[(summary.candidate == cand) & (summary.split == split)].copy()
    part["o"] = part.group.map({g: i for i, g in enumerate(dict(GROUP_ORDER)[split])})
    part = part.sort_values("o")
    print(f"\n--- {cand}, split by {split} (next/prev: {int(part.next_n.iloc[0])} scored seasons each; "
          f"both: {int(part.both_n.iloc[0])})")
    print(part[list(SHOWN)].round(4).to_string(index=False))


def print_report(frame: pd.DataFrame, splits: list, ref_name: str) -> None:
    """Each level's summary, candidate by candidate, the whole season first and then each split."""
    for value, what in LEVELS:
        summary = summarise(frame, value)
        print(f"\n=== {what}: each group's part of the candidate-minus-{ref_name} difference in squared error; "
              f"below zero = the candidate is better there")
        for cand in summary.candidate.unique():
            for split in ["all", *splits]:
                print_group_table(summary, cand, split)


# ------------------------------------------------------------------------------------------------- the run
def read_inputs(args: argparse.Namespace, cfg: dict) -> dict:
    """The reference table and its quality tiers, the candidates by direction, the ages and the config."""
    cands = [part.split("=", 1) for part in args.cands.split(",") if part]
    if not cands:
        raise SystemExit("--cands=name=path[,name=path...] is required")
    ref_path = args.ref.split("=", 1)[1]
    return dict(ref=read_table(ref_path, cfg),
                tiers=quality_tiers(pd.read_parquet(resolve_path(cfg, ref_path),
                                                    columns=["player_id", "season", "rating_total"])),
                tables={name: read_by_direction(path, cfg) for name, path in cands}, ages=read_ages(cfg), cfg=cfg)


def read_splits(spec: str) -> list:
    """The splits named by `--splits`, each one of SPLITS."""
    splits = [s for s in spec.split(",") if s]
    unknown = sorted(set(splits) - set(SPLITS))
    if unknown:
        raise SystemExit(f"--splits {unknown}: this check has {', '.join(SPLITS)}")
    return splits


def shown_path(path: Path, cfg: dict) -> Path:
    """A written file as the run log names it: relative to the repository when it is inside it."""
    root = Path(cfg["_root"])
    return path.relative_to(root) if path.is_relative_to(root) else path


def main() -> None:
    use_utf8()
    args = parse_args()
    cfg = load_config()
    splits = read_splits(args.splits)
    inputs = read_inputs(args, cfg)
    tag = "_".join(inputs["tables"]) if args.tag is None else args.tag
    designs = DesignCache(cfg)
    rows, worst_gap, started = [], 0.0, time.time()
    with one_thread():
        for season in scored_seasons(cfg, args.first, args.last):
            more, more_gap = season_rows(scored_season(designs, season), inputs, splits)
            rows += more
            worst_gap = max(worst_gap, more_gap)
            print(f"  {season} ({time.time() - started:.0f}s)", flush=True)
    frame = pd.DataFrame(rows)
    out = resolve(cfg, "outputs") / f"yoy_by_player_{tag}.parquet"
    frame.to_parquet(out, index=False)
    print(f"\nthe groups add up to the whole in every season and direction: largest gap {worst_gap:.1e}")
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print_report(frame, splits, args.ref.split("=", 1)[0])
    print(f"\nwrote {shown_path(out, cfg)}")


if __name__ == "__main__":
    main()
