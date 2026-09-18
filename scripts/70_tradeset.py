"""The trade set: fit each season's with-and-without-you correction, choose its penalty, score it.

    python scripts/70_tradeset.py [--rankings=outputs/season_ratings_sy_lam13037.parquet]
                                  [--first=1997] [--last=2026] [--seasons=2015,2024,2025,2026]
                                  [--lambdas=1000,3000,10000,30000,100000,300000]
                                  [--fill=500x0.25] [--target_off=xpts_ft] [--target_def=x3def_w0.25]
                                  [--out=tradeset] [--score=1] [--block=1] [--free_scale=1]
                                  [--team_effects=none|team|team_season]

For every rated season the games of that season and the two either side are pooled into team-games,
each player's entry is his share of that team's possessions in that game, the rated season's rating is
subtracted from every row, and the remainder is ridged back onto the player columns.  That coefficient
is `alpha`: what a player's comings and goings say that his rating did not already know.  The reasoning
and the conventions are in `src/eracoef/tradeset.py`.

**Alpha never becomes a published rating.**  It reads the neighbouring seasons' games, and ruling 1 is
that a season's rating comes from that season's games alone.  What it is for is a loss that counts every
player once -- the year-over-year test is a team-game error, in which a 200-possession man is a rounding
error -- and a training target for the box prior, whose coefficients may learn from other seasons.

**Choosing the penalty.**  `--lambdas` is swept and each value is scored the only honest way available:
`rating + alpha` for season H is asked to predict the games of H-2 and H+2, which are outside the three
seasons alpha was fitted on.  The baseline arm is the rating alone.  The winner must be INTERIOR to the
grid -- an argmin at either end has chosen nothing -- and the level of both arms is optimistic, because
the incumbent's own prior was only held off H's immediate neighbours, so only the PAIRED difference
between the arms means anything.  If the grid runs to its weak end, the trade set is noise at this
sample size and that is the finding.

**Only players the rated season's rankings already carry get a row.**  The regression estimates a column
for everyone on the floor in the three seasons, but publishing alpha for a player the rated season has no
rating for would change who is covered as well as how well they are rated, and the two could not be told
apart in the score.

Writes outputs/<out>_alpha.parquet (one row per player, season and side), outputs/<out>_lambda.parquet
(the grid), outputs/<out>_loss.parquet (the trade loss by season, side and possession tier) and
outputs/<out>_yoy_input.parquet (rating + alpha in the schema scripts/63_yoy.py reads, so the paired
report is one more command:

    python scripts/63_yoy.py --offset=2 --ref=incumbent --tag=tradeset
        --rankings=alpha=outputs/tradeset_yoy_input.parquet,incumbent=outputs/season_ratings_sy_lam13037.parquet
        --fill=alpha:500x0.25,incumbent:500x0.25
"""
import importlib.util
import os
import sys
import time
from pathlib import Path

# Thread pinning BEFORE numpy, exactly as scripts/62_single_year_board.py does it: a BLAS that spins up
# twelve threads per small solve thrashes, and two builds at once can livelock.
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
os.environ.setdefault("NUMBA_NUM_THREADS", os.environ.get("OPENRAPM_NUMBA_THREADS", "4"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from eracoef import tradeset as ts_mod  # noqa: E402
from eracoef.config import load_config  # noqa: E402
from eracoef.holdout import Context, Ratings, predict_season, score  # noqa: E402
from eracoef.priorridge import armse  # noqa: E402
from eracoef.xshoot import DEFENSE_TARGETS  # noqa: E402

# `load_table` is the one place the positive-good rating columns are turned into the model's raw sign
# (defence = points allowed).  Importing it rather than copying it keeps a single definition of the flip:
# doing it twice is the pitfall that would double the defensive signal and look like a result.
_spec = importlib.util.spec_from_file_location("_yoy", ROOT / "scripts" / "63_yoy.py")
_yoy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_yoy)
load_table = _yoy.load_table

NO_ALPHA = "rating alone"


# The shared command line (scripts/_cli.py): `flag` records every name it is asked for so
# `check_flags` below can refuse one that was never asked for.  A misspelled flag used to be
# ignored silently, which is how a run looks right and is wrong.
from _cli import check_flags, flag as _flag, switch   # noqa: E402


def _target(name):
    """A `design.TARGETS` key, or the `xshoot.DEFENSE_TARGETS` callable of that name."""
    return DEFENSE_TARGETS[name] if name in DEFENSE_TARGETS else name


def team_of_rows(keys: pd.DataFrame, cfg) -> pd.DataFrame:
    """The team whose points each team-game row counts, and the team defending it.

    The stints carry no team id -- home and away are implied by the slot columns -- so it comes from the
    per-game diagnostic frame written beside them, one row per game.
    """
    root = Path(cfg["_root"])
    stints = root / cfg.get("paths", {}).get("stints", "data/stints")
    frames = []
    for (season, phase), _ in keys.groupby(["season", "phase"], sort=True):
        path = stints / f"{season}_{phase}_diag.parquet"
        frames.append(pd.read_parquet(path, columns=["game_id", "home_team_id", "away_team_id"]))
    diag = pd.concat(frames, ignore_index=True).drop_duplicates("game_id")
    diag["game_id"] = diag.game_id.astype(str)
    merged = keys.assign(game_id=keys.game_id.astype(str)).merge(diag, on="game_id", how="left")
    if merged.home_team_id.isna().any():
        raise SystemExit("a team-game has no row in the stint diagnostics; rebuild the stints")
    home, away = merged.home_team_id.to_numpy(np.int64), merged.away_team_id.to_numpy(np.int64)
    is_home_off = merged.is_home_off.to_numpy(bool)
    return pd.DataFrame({"team_off": np.where(is_home_off, home, away),
                         "team_def": np.where(is_home_off, away, home)})


def block_of(season: int, first: int, last: int, width: int = 1) -> list:
    """The rated season and `width` seasons either side, clipped to the seasons that exist.

    Clipped rather than merely skipped: handing the loader a season with no stints on disk makes it
    SCRAPE, which is the one failure mode here that costs an hour instead of raising.

    `width=0` is the control that matters.  It leaves the rated season on its own, so there is no
    with-and-without contrast across seasons at all and alpha can only be a team-game refit of the
    games the rating already saw.  Whatever that arm scores is the part of the gain that owes nothing
    to a player being absent, and the trade set's claim is only what it wins by on top.
    """
    return [s for s in range(season - int(width), season + int(width) + 1) if first <= s <= last]


def side_alpha(alpha: np.ndarray, n_players: int, side: str) -> np.ndarray:
    """The half of a fit's coefficients that side takes, raw sign."""
    return alpha[:n_players] if side == "offense" else alpha[n_players:]


def corrected(ratings_h: pd.DataFrame, player_ids, alpha_off, alpha_def, scale=(1.0, 1.0)) -> pd.DataFrame:
    """`scale x rating + alpha` for the rated season's own players, raw sign, ready for `Ratings`.

    The scale belongs here.  The fit says a team-game is explained by the rating multiplied by one free
    number per side PLUS the per-player correction, so the corrected rating is the two together; adding
    alpha to the unscaled rating would be reading half of a fit and reporting it as the whole.
    """
    add = pd.DataFrame({"player_id": np.asarray(player_ids, dtype=np.int64),
                        "alpha_o": alpha_off, "alpha_d": alpha_def})
    out = ratings_h.merge(add, on="player_id", how="left").fillna({"alpha_o": 0.0, "alpha_d": 0.0})
    return pd.DataFrame({"player_id": out.player_id, "o": scale[0] * out.o + out.alpha_o,
                         "d": scale[1] * out.d + out.alpha_d, "poss": out.poss})


def main():
    check_flags()      # refuse a flag this script does not understand (scripts/_cli.py)
    cfg = load_config(ROOT / "config.yaml")
    first, last = int(cfg["first_season"]), int(cfg["last_season"])
    rankings_path = ROOT / _flag("rankings", "outputs/season_ratings_sy_lam13037.parquet")
    lambdas = [float(x) for x in _flag("lambdas", "1000,3000,10000,30000,100000,300000").split(",")]
    max_poss, _, shrink = (_flag("fill", "500x0.25")).partition("x")
    max_poss, shrink = float(max_poss), float(shrink or 1.0)
    target = {"offense": _target(_flag("target_off", "xpts_ft")),
              "defense": _target(_flag("target_def", "x3def_w0.25"))}
    tag = _flag("out", "tradeset")
    do_score = _flag("score", "1") not in ("0", "no", "false")
    width = int(_flag("block", "1"))
    team_effects = _flag("team_effects", "none")
    assert team_effects in ("none", "team", "team_season"), team_effects
    free_scale = _flag("free_scale", "1") not in ("0", "no", "false")
    seasons = ([int(s) for s in _flag("seasons").split(",")] if _flag("seasons")
               else list(range(int(_flag("first", first)), int(_flag("last", last)) + 1)))

    raw = pd.read_parquet(rankings_path)
    names = raw[["player_id", "player_name"]].drop_duplicates("player_id")
    table = load_table(rankings_path, "rating")
    roles = pd.read_parquet(ROOT / "data/cache/roles_RSPO.parquet")

    ctx = Context.load(cfg)
    ctx.cache_size = 2
    print(f"rankings {rankings_path.name}: {len(table)} player-seasons, {table.season.nunique()} seasons")
    print(f"penalties {', '.join(f'{lam:,.0f}' for lam in lambdas)}"
          f"   fill {shrink:g} x the mean rating of players under {max_poss:,.0f} possessions")
    if team_effects != "none":
        print(f"team terms: {team_effects}. Each team's own level is free, per side, so a player is")
        print("identified by variation in who played inside his team, not by his team being good")
    print("the rating enters as two free columns per fit, one per side, so a pure rescale is read as a"
          if free_scale else "the rating enters at exactly the amplitude it came with (--free_scale=0):")
    print("scale and not written into every player's alpha (--free_scale=0 turns that off)"
          if free_scale else "any uniform tilt will appear in the alphas")

    t0 = time.time()
    alpha_of = {}          # (season, lam) -> {"offense": vector, "defense": vector}
    scale_of = {}          # (season, lam) -> (offence multiplier, defence multiplier)
    players_of = {}        # season -> the design's player ids
    exposure_of = {}       # season -> the with/without table
    grid_rows, one_sided = [], []

    for season in seasons:
        block = block_of(season, first, last, width)
        if len(block) < 3:
            one_sided.append(season)
        ratings_h = table[table.season == season]
        if len(ratings_h) == 0:
            print(f"  {season}: no rows in the rankings, skipped")
            continue
        fill_o, fill_d = ts_mod.replacement_fill(ratings_h, max_poss=max_poss, shrink=shrink)

        normal, exposure, player_ids, n_team_cols = {}, None, None, 0
        for side in ts_mod.SIDES:
            wd = ctx.design(block, target=target[side])
            # under per-season team terms each team-season column already carries its season's level,
            # so the season intercepts come out rather than sit there exactly collinear with them
            ts = ts_mod.team_game_design(wd, control_prefixes=(
                ("home", "is_po", "po_home") if team_effects == "team_season" else ts_mod.CONTROL_PREFIXES))
            if player_ids is None:
                player_ids = ts.player_ids
                teams = team_of_rows(ts.keys, cfg)
                eligible = ts_mod.eligibility(roles, season)
                exposure = ts_mod.exposure_table(
                    ts, teams, dict(zip(eligible.player_id.astype(int), eligible.team_id.astype(int))), season)
            elif not np.array_equal(player_ids, ts.player_ids):
                raise SystemExit("the two targets built different player columns")
            extra = None
            if team_effects != "none":
                extra, team_names = ts_mod.team_columns(ts, teams, per_season=(team_effects == "team_season"))
                n_team_cols = len(team_names)
            offset = ts_mod.offset_vector(ts.player_ids, ratings_h, fill_o, fill_d)
            normal[side] = (*ts_mod.normal_equations(ts, offset, free_scale=free_scale, extra=extra),
                            ts.n_players)
            if side == "offense":
                shares = np.asarray(ts.Z[:, :ts.n_players].sum(axis=1)).ravel()
                assert np.allclose(shares, 5.0, atol=1e-6), "a team-game's shares do not sum to five"
                n_team_games = ts.Z.shape[0]
                n_columns = ts.Z.shape[1] + len(ts.control_names) + n_team_cols
            del wd, ts
        ctx._cache.clear()

        for lam in lambdas:
            fits = {side: ts_mod.solve_alpha(g, r, n, n_free, lam)
                    for side, (g, r, n_free, n) in normal.items()}
            alpha_of[(season, lam)] = {side: side_alpha(fit[0], normal[side][3], side)
                                       for side, fit in fits.items()}
            if free_scale:
                # what each side's rating would have to be multiplied by for the three seasons to
                # agree with it; read it before the alphas, because a rescale is not attribution
                scale_of[(season, lam)] = (1.0 + fits["offense"][1][0], 1.0 + fits["defense"][1][1])
        players_of[season] = player_ids
        exposure_of[season] = exposure

        moved = exposure[(exposure.side == "offense") & (exposure.without_poss > 0)]
        scales = (f", rating scale {scale_of[(season, lambdas[0])][0]:.2f} / "
                  f"{scale_of[(season, lambdas[0])][1]:.2f}" if free_scale else "")
        print(f"  {season}: block {block[0]}-{block[-1]}, {n_team_games:,} team-games, {n_columns:,} columns, "
              f"{len(moved):,} players with games their team played without them{scales}"
              f"  ({time.time() - t0:.0f}s)", flush=True)

        if do_score:
            for neighbour in (season - 2, season + 2):
                if not (first <= neighbour <= last):
                    continue
                wd_h = ctx.design([neighbour], "pts")
                # three kinds of arm: the rating as it stands, the rating with only its amplitude
                # corrected, and the rating with the per-player correction on top of that.  Without
                # the middle one a rescale and a discovery about players cannot be told apart.
                zero = np.zeros(len(player_ids))
                arms = {NO_ALPHA: (zero, zero, (1.0, 1.0))}
                for lam in lambdas:
                    scale = scale_of.get((season, lam), (1.0, 1.0))
                    arms[f"{lam:,.0f} rescale only"] = (zero, zero, scale)
                    arms[f"{lam:,.0f} rescale + alpha"] = (alpha_of[(season, lam)]["offense"],
                                                           alpha_of[(season, lam)]["defense"], scale)
                for arm, (a_off, a_def, scale) in arms.items():
                    df = corrected(ratings_h, player_ids, a_off, a_def, scale)
                    sc = score(predict_season(Ratings(df, fill_o=fill_o, fill_d=fill_d), wd_h))
                    grid_rows.append(dict(season=season, scored=neighbour,
                                          direction="prev" if neighbour < season else "next",
                                          arm=arm, tg=sc["tg"], tg_n=sc["tg_n"], mse=sc["mse"],
                                          scale_off=sc["scale_off"], scale_def=sc["scale_def"]))
                del wd_h
            ctx._cache.clear()

    if one_sided:
        print(f"\none-sided blocks (no season on one side): {', '.join(str(s) for s in one_sided)}")

    # ------------------------------------------------------------------ the penalty
    chosen = lambdas[0]
    if do_score and grid_rows:
        grid = pd.DataFrame(grid_rows)
        grid.to_parquet(ROOT / "outputs" / f"{tag}_lambda.parquet", index=False)
        chosen = report_grid(grid, lambdas)
    elif do_score:
        print("\nnothing was scored")

    # ------------------------------------------------------------------ the alpha table
    rows = []
    for season, player_ids in players_of.items():
        ratings_h = table[table.season == season]
        eligible = ts_mod.eligibility(roles, season)
        alpha = alpha_of[(season, chosen)]
        for side in ts_mod.SIDES:
            column = "rating_off" if side == "offense" else "rating_def"
            sign = 1.0 if side == "offense" else -1.0     # positive-good, from the model's raw sign
            frame = pd.DataFrame({"player_id": player_ids, "season": season, "side": side,
                                  "alpha_raw": alpha[side], "alpha_good": sign * alpha[side],
                                  "rating_scale": scale_of.get((season, chosen), (1.0, 1.0))[
                                      0 if side == "offense" else 1]})
            frame = frame.merge(raw[raw.season == season][["player_id", column, "poss_off"]],
                                on="player_id", how="inner").rename(columns={column: "rating"})
            rows.append(frame)
    alpha_table = pd.concat(rows, ignore_index=True)
    exposure = pd.concat([e.assign(season=s) for s, e in exposure_of.items()], ignore_index=True)
    alpha_table = alpha_table.merge(exposure, on=["player_id", "season", "side"], how="left")
    eligible_all = pd.concat([ts_mod.eligibility(roles, s).assign(season=s) for s in players_of],
                             ignore_index=True)
    alpha_table = alpha_table.merge(eligible_all[["player_id", "season", "poss_on"]],
                                    on=["player_id", "season"], how="left")
    alpha_table["eligible"] = alpha_table.poss_on.notna()
    alpha_table["poss_on"] = alpha_table.poss_on.fillna(alpha_table.poss_off)
    alpha_table["one_sided"] = alpha_table.season.isin(one_sided)
    alpha_table["penalty"] = chosen
    # Which arm produced this.  Without it nothing downstream can tell an uncontrolled table from
    # a team-controlled one, and 71/72 default to different arms.
    alpha_table["team_effects"] = team_effects
    # the corrected rating the fit actually implies: the amplitude and the per-player part together
    alpha_table["corrected"] = alpha_table.rating_scale * alpha_table.rating + alpha_table.alpha_good
    alpha_table = alpha_table.merge(names, on="player_id", how="left")
    alpha_table.to_parquet(ROOT / "outputs" / f"{tag}_alpha.parquet", index=False)
    print(f"\nwrote outputs/{tag}_alpha.parquet: {len(alpha_table):,} rows at penalty {chosen:,.0f}")

    wide = to_wide(alpha_table)
    wide.to_parquet(ROOT / "outputs" / f"{tag}_yoy_input.parquet", index=False)
    print(f"wrote outputs/{tag}_yoy_input.parquet: the corrected rating, the schema scripts/63_yoy.py reads")
    # the same numbers in the full rankings schema, so scripts/66_compare.py can put the corrected list
    # beside the incumbent one and the owner can read the top twenty
    (wide.assign(rating_total=wide.rating_off + wide.rating_def).merge(names, on="player_id", how="left")
     .to_parquet(ROOT / "outputs" / f"{tag}_rankings.parquet", index=False))
    print(f"wrote outputs/{tag}_rankings.parquet: the same, in the schema scripts/66_compare.py reads")

    if free_scale:
        by_season = alpha_table.drop_duplicates(["season", "side"])
        print()
        print("=== the rating amplitude the three seasons ask for (1.00 = the rating as it stands)")
        for side, group in by_season.groupby("side").rating_scale:
            print(f"    {side}: {group.mean():.3f} (min {group.min():.3f}, max {group.max():.3f})")
        print("    Below 1 means the rating is too wide for the neighbouring seasons to bear out.")

    loss = ts_mod.trade_loss(alpha_table)
    loss.to_parquet(ROOT / "outputs" / f"{tag}_loss.parquet", index=False)
    report_tables(alpha_table, wide, loss, one_sided)


def report_grid(grid: pd.DataFrame, lambdas: list) -> float:
    """The penalty grid, scored two seasons out, with the paired difference against the rating alone."""
    pd.set_option("display.width", 220, "display.max_columns", 30, "display.precision", 4)
    base = grid[grid.arm == NO_ALPHA].set_index(["season", "scored"]).tg
    out = []
    for arm, part in grid.groupby("arm", sort=False):
        pooled = float(np.average(part.tg, weights=part.tg_n))
        diff = (part.set_index(["season", "scored"]).tg - base).dropna()
        se = float(diff.std(ddof=1) / np.sqrt(len(diff))) if len(diff) > 1 else np.nan
        out.append(dict(penalty=arm, observations=len(part), game_armse=armse(pooled),
                        vs_rating_alone=float(diff.mean()), z=float(diff.mean() / se) if se else np.nan,
                        wins=int((diff < 0).sum()), scale_off=part.scale_off.mean(),
                        scale_def=part.scale_def.mean()))
    frame = pd.DataFrame(out)
    print("\n=== the penalty, scored on the games TWO seasons away, LOWER IS BETTER")
    print("    game_armse: what a typical team-game misses by, points per 100.  Both arms read optimistically")
    print("    here, so only vs_rating_alone means anything; a negative difference is an improvement.")
    print(frame.round(4).to_string(index=False))

    swept = frame[frame.penalty.str.endswith("rescale + alpha")].reset_index(drop=True)
    rescale = frame[frame.penalty.str.endswith("rescale only")].reset_index(drop=True)
    best = int(swept.game_armse.idxmin())
    chosen = lambdas[best]
    interior = 0 < best < len(swept) - 1
    baseline = float(frame[frame.penalty == NO_ALPHA].game_armse.iloc[0])
    print(f"\n  best penalty {chosen:,.0f} ({'interior' if interior else 'AT THE EDGE OF THE GRID'}), "
          f"{swept.game_armse.iloc[best]:.4f} against {baseline:.4f} for the rating alone")
    if not interior:
        print("  An argmin on a boundary has chosen nothing: widen --lambdas before believing it.")

    # Amplitude or attribution.  A rescale says nothing about any individual player, so the two halves
    # have to be separated before this number is quoted anywhere.
    only = float(rescale.game_armse.iloc[best])
    beyond = paired_arms(grid, f"{chosen:,.0f} rescale + alpha", f"{chosen:,.0f} rescale only")
    print("\n  Splitting that improvement in two, in points per 100 of game_armse REMOVED "
          "(bigger is better here):")
    print(f"    {baseline - only:.4f} removed by the RESCALE alone, one number per side, "
          "no player told apart from another")
    print(f"    {only - float(swept.game_armse.iloc[best]):.4f} removed by the per-player correction "
          "on top of it")
    print(f"    paired on the per-player part only: {beyond[0]:+.4f} mse, paired statistic {beyond[1]:+.2f}, "
          f"{beyond[2]} of {beyond[3]} observations better")
    if beyond[0] >= 0:
        print("    The per-player correction adds nothing beyond the rescale.  That is the finding;")
        print("    the alpha table below is still written, as the diagnostic it is.")
    print("\n  Read this as a forecast and not as proof of attribution: the corrected rating carries three")
    print("  seasons of games where the plain one carries a single season, so part of any gap is simply")
    print("  the larger sample.  What the trade set is FOR is the per-player loss and the feature table.")
    return chosen


def paired_arms(grid: pd.DataFrame, arm: str, reference: str):
    """(mean difference, paired statistic, wins, observations) of one arm against another, same rows."""
    key = ["season", "scored"]
    difference = (grid[grid.arm == arm].set_index(key).tg
                  - grid[grid.arm == reference].set_index(key).tg).dropna()
    se = float(difference.std(ddof=1) / np.sqrt(len(difference))) if len(difference) > 1 else np.nan
    return (float(difference.mean()), float(difference.mean() / se) if se else np.nan,
            int((difference < 0).sum()), len(difference))


def to_wide(alpha_table: pd.DataFrame) -> pd.DataFrame:
    """The corrected rating per player-season, positive-good, in the schema scripts/63_yoy.py reads."""
    wide = alpha_table.pivot_table(index=["player_id", "season"], columns="side",
                                   values=["rating", "alpha_good", "corrected", "poss_off"]).reset_index()
    wide.columns = [c[0] if not c[1] else f"{c[0]}_{c[1][:3]}" for c in wide.columns]
    return pd.DataFrame({"player_id": wide.player_id.astype(np.int64), "season": wide.season.astype(int),
                         "rating_off": wide.corrected_off, "rating_def": wide.corrected_def,
                         "poss_off": wide.poss_off_off})


def report_tables(alpha_table: pd.DataFrame, wide: pd.DataFrame, loss: pd.DataFrame, one_sided: list):
    """The 2026 and 2015 top twenty with alpha beside the rating, the largest corrections, and the loss."""
    pd.set_option("display.width", 220, "display.max_columns", 30, "display.precision", 3)
    flat = (alpha_table.pivot_table(index=["player_id", "player_name", "season", "eligible"],
                                    columns="side",
                                    values=["rating", "alpha_good", "corrected", "with_poss",
                                            "without_poss", "poss_off"])
            .reset_index())
    flat.columns = [c[0] if not c[1] else f"{c[0]}_{c[1][:3]}" for c in flat.columns]
    flat["total"] = flat.rating_off + flat.rating_def
    flat["alpha_total"] = flat.alpha_good_off + flat.alpha_good_def
    flat["corrected"] = flat.corrected_off + flat.corrected_def

    seasons = [s for s in (flat.season.max(), 2015) if s in set(flat.season)]
    for season in dict.fromkeys(seasons):
        part = flat[flat.season == season].sort_values("total", ascending=False)
        cols = ["player_name", "rating_off", "rating_def", "total", "alpha_good_off", "alpha_good_def",
                "alpha_total", "corrected", "poss_off_off", "without_poss_off"]
        print(f"\n=== {season}, top 20 by the rating, with the trade set's correction beside it "
              "(points per 100, positive good)")
        print(part[cols].head(20).to_string(index=False))

        # the same filter the loss uses: one team all season, and games his team played without him.
        # A player whose team never took the floor without him has an alpha the other columns implied,
        # not one his own absence earned, and listing it as a finding would be a lie about the source.
        seen = part[part.eligible & (part.without_poss_off > 0) & (part.without_poss_def > 0)]
        big = seen.reindex(seen.alpha_total.abs().sort_values(ascending=False).index)
        print(f"\n=== {season}, the 20 largest corrections among players whose team played without them")
        print(big[cols].head(20).to_string(index=False))

    print("\n=== the trade loss: how far a season's ratings are from what the absences say, "
          "every player counted once")
    if loss.empty:
        print("    no player has a with-and-without contrast in this run, so there is no loss")
        print("    to read.  That is what --block=0 means: one season on its own has no")
        print("    neighbouring games in which a player's team played without him.")
        return
    print("    missed: the typical player's miss in points per 100.  missed_eff: the same, weighting a")
    print("    player by how much with-and-without evidence he has.  points: the season's total misprice.")
    pooled = loss.groupby(["side", "tier"], as_index=False).agg(
        seasons=("season", "nunique"), players=("players", "mean"), missed=("missed", "mean"),
        missed_eff=("missed_eff", "mean"), points=("points", "mean"))
    print(pooled.round(3).to_string(index=False))
    if one_sided:
        kept = loss[~loss.season.isin(one_sided)]
        both = kept.groupby(["side", "tier"], as_index=False).missed.mean()
        print(f"\n    dropping the {len(one_sided)} one-sided seasons: "
              + ", ".join(f"{r.side} {r.tier} {r.missed:.3f}" for r in both.itertuples()
                          if r.tier == "all"))


if __name__ == "__main__":
    main()
