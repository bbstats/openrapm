"""Check 151, the with-or-without-you test: on games the ratings never saw, do they get right how a team does when its
lineup changes, above all when a regular sits out?

The owner, 2026-10-15: "within-season is truly the best metric, particularly WOWY (eg deltas in the DNPs)".  A team's
margin is the sum of its players' ratings, so the year-over-year team-game test scores mostly each team's total and
is nearly blind to how the credit is split between teammates (notes/DECISIONS.md, "Why step 3 gives a better prior's
gain back").  This test takes each team's total out and scores only what is left: how a team's results move from
game to game as its lineup changes.  The biggest such change is a regular not playing.

Words used below
    fold            one season rated from three quarters of its games, the other quarter held back (`openrapm folds`;
                    data/within_folds.py); each season has three repeats of four folds
    team-game       one team's offence in one held-back game: points per 100 possessions as scored, weighted by its
                    possessions
    prediction      the fold's ratings summed over the ten players by their share of the team-game's possessions
                    (defence in points allowed), a stand-in rating for a player the fold did not rate.  Two stages:
                    `step3`, the rating as the fold built it; `step4`, its prior part times step 4's multiplier for the
                    season (from the 8 fold seasons nearest it, never the season or its neighbours: the test rule, as
                    the experiments ship), re-centred
    team totals out each fold's misses (points minus prediction) fitted with a free level for every attacking team, a
                    free level for every defending team, home and playoffs; what is left is the WOWY miss.  A rating
                    system gains here only by predicting how a team's games differ from each other, never by getting
                    a team's overall level right
    regular         a player with at least 40% of his team's fit-game possessions on that side (about 19 minutes a game)
    DNP team-game   a held-back team-game in which a regular of the attacking or the defending team did not play,
                    while he played for that team in a held-back game before it and one after it (so a traded player
                    or one who joined late is not counted absent)
    error           the possession-weighted mean of the squared WOWY miss, over a season's twelve folds; DNP error
                    the same over the DNP team-games only

Two units.  `stint` (the default and the decider since 2026-10-15): every stint of the held-back games, predicted from
the ten players on the court, so each substitution is a with-or-without comparison as well as each missed game.
`game`: the held-back team-games, as the folds store them (far weaker: experiment 63 reads z -1.1 by games and
z -7.5 by stints).  A check that
the stint unit measures players and not noise: on 2024's first four folds, today's ratings leave 13.6 less error than
ratings of zero once the team levels are out.

Read against a reference (the incumbent) season by season: the mean of the per-season differences, its standard error
across seasons, the z, and the seasons better.  The noise control (the incumbent rebuilt with its box-score prior's
players assigned to other folds) shows what chance alone does.  Below 0 is better.

Step 4 split into pieces (`--pieces source,certainty`, the retry plan's rank 3, 2026-10-15).  For a system whose
folds were built with `openrapm folds --pieces`, one more stage per family of check 145: the box-score part and each
piece of the games' part get their own multiplier, fitted as check 145 fits them on the 8 nearest fold seasons under
the test rule, never the season or its neighbours, then applied to the season's folds and re-centred.  The stage
The shipped step through the same code, `pieces_u_whole` since experiment 65 (`pieces_step4` before), must equal
`step4` to the last digits; the check stops if not.

Step 4 here is the shipped one, settings.ShrinkSettings: since experiment 65 (adopted 2026-10-15) it multiplies the
games' part too, so `step4` is m x prior + g x u, both factors fitted on the 8 nearest fold seasons.

Writes outputs/csv/wowy_<tag>_seasons.csv (every system, season, stage and slice) and wowy_<tag>.csv (the paired
summary).

    python scripts/checks/151_wowy.py --systems=nocareer,noise_nocareer,cols06 --ref=nocareer --tag=exp63
        (each name reads outputs/within/<name>_early and <name>_within; about 10 minutes)
    python scripts/checks/151_wowy.py --systems=cols06pieces --ref=cols06 --pieces=source,certainty --tag=pieces
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from openrapm.commands.console import use_utf8
from openrapm.settings import PRODUCT

REGULAR_SHARE = 0.40            # a regular's share of his team's fit-game possessions on that side
NEAREST = 8                     # step 4's fold seasons, as shipped
SLICES = ("all", "dnp")
PIECES_TOLERANCE = 1e-8         # the shipped step through check 145's code against `step4`, per player (145's own)
FREE_GAMES = PRODUCT.shrink.free_games_part     # step 4 multiplies the games' part too (experiment 65)
SHIPPED_FAMILY = "u_whole" if FREE_GAMES else "step4"      # check 145's family that is the shipped step 4


def _load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(file))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module          # its dataclasses look their module up while being defined
    spec.loader.exec_module(module)
    return module


def _audit():
    return _load("heldout_audit", "138_heldout_audit.py")


def _pieces():
    return _load("piece_multipliers", "145_piece_multipliers.py")


# ------------------------------------------------------------------------------------------------ one fold
def team_arrays(directory: Path, stem: str) -> dict:
    """The held-back team-games' attacking team, defending team, game id and phase, in the fold's row order."""
    test = np.load(directory / f"test_{stem}.npz", allow_pickle=True)
    return {name: test[name] for name in ("team_off", "team_def", "game_id", "phase", "is_home_off")}


def regulars_absent(fold, teams: dict) -> np.ndarray:
    """True for each team-game in which a regular of either team sat out (the module docstring)."""
    absent = np.zeros(len(fold.y), dtype=bool)
    order = np.argsort(teams["game_id"].astype(str), kind="stable")
    rank = np.empty(len(order), dtype=np.int64)
    rank[order] = np.arange(len(order))
    for side, team_column, poss_column in (("O", "team_off", "poss_off"), ("D", "team_def", "poss_def")):
        rows = fold.rows[side].reset_index(drop=True)
        shares = sp.csc_matrix(fold.Z[side])
        team_of = rows.team_id.to_numpy()
        poss = rows[poss_column].to_numpy(float) if poss_column in rows else rows.poss_off.to_numpy(float)
        team_total = pd.Series(poss).groupby(team_of).sum() / 5.0
        regular = poss >= REGULAR_SHARE * pd.Series(team_of).map(team_total).to_numpy(float)
        team_games = teams[team_column]
        for j in np.flatnonzero(regular):
            mine = np.flatnonzero(team_games == team_of[j])
            if len(mine) == 0:
                continue
            column = shares.getcol(j).toarray().ravel()
            played = mine[column[mine] > 0]
            if len(played) == 0:
                continue
            first, last = rank[played].min(), rank[played].max()
            sat = mine[(column[mine] == 0) & (rank[mine] > first) & (rank[mine] < last)]
            absent[sat] = True
    return absent


def team_levels(fold, teams: dict) -> sp.csr_matrix:
    """A free level for every attacking team and every defending team, home and playoffs (the fold's controls)."""
    off_codes, _ = pd.factorize(teams["team_off"])
    def_codes, _ = pd.factorize(teams["team_def"])
    n = len(fold.y)
    attack = sp.csr_matrix((np.ones(n), (np.arange(n), off_codes)), shape=(n, off_codes.max() + 1))
    defend = sp.csr_matrix((np.ones(n), (np.arange(n), def_codes)), shape=(n, def_codes.max() + 1))
    home = fold.F[:, [0]] if fold.F.shape[1] else np.zeros((n, 1))
    extra = fold.F[:, 2:] if fold.F.shape[1] > 2 else np.zeros((n, 0))
    return sp.hstack([attack, defend[:, 1:], sp.csr_matrix(home), sp.csr_matrix(extra)]).tocsr()


def wowy_miss(miss: np.ndarray, levels: sp.csr_matrix, w: np.ndarray) -> np.ndarray:
    """The miss with the team levels, home and playoffs fitted out (weighted least squares)."""
    root = np.sqrt(w)
    design = levels.multiply(root[:, None]).toarray()
    coef, *_ = np.linalg.lstsq(design, miss * root, rcond=None)
    return miss - levels @ coef


def step4_parts(audit, fold, parts, step4: tuple) -> dict:
    """{"O", "D"}: (the prior part, the games' part) after step 4 on this fold, re-centred as step 4 re-centres.
    `step4` is (m_O, m_D, g_O, g_D): the prior part's factors and the games' part's, 1 when step 4 holds it."""
    out = {}
    for side, prior, games, m, g in (("O", parts.prior_off, parts.u_off, step4[0], step4[2]),
                                     ("D", parts.prior_def, parts.u_def, step4[1], step4[3])):
        games = g * games
        out[side] = (audit.shrunk_prior(prior, games, m, np.zeros(len(games)),
                                        fold.rows[side].poss_off.to_numpy(float)), games)
    return out


def fold_errors(audit, fold, teams: dict, step4: tuple) -> dict:
    """{(stage, slice): (sum of w x WOWY miss squared, sum of w)} for one fold."""
    parts = audit.RatingParts.of(fold)
    shrunk = step4_parts(audit, fold, parts, step4)
    misses = audit.stage_misses(fold, parts, shrunk["O"][0], shrunk["D"][0],
                                {side: prior + games for side, (prior, games) in shrunk.items()})
    levels = team_levels(fold, teams)
    absent = regulars_absent(fold, teams)
    out = {}
    for stage, key in (("step3", "raw"), ("step4", "shrunk")):
        left = wowy_miss(misses[key], levels, fold.w)
        for slice_, keep in (("all", np.ones(len(left), dtype=bool)), ("dnp", absent)):
            out[(stage, slice_)] = (float(np.sum(fold.w[keep] * left[keep] ** 2)), float(fold.w[keep].sum()))
    out[("share", "dnp")] = (float(fold.w[absent].sum()), float(fold.w.sum()))
    return out


def stage_ratings(audit, fold, step4: tuple, extra: dict | None = None) -> dict:
    """{stage: ({player_id: rating_off}, {player_id: points allowed})} for the fold's rated players; `extra` adds
    stages given as {stage: {"O": rating_off, "D": rating_def}}, positive good, in the fold's row order."""
    parts = audit.RatingParts.of(fold)
    shrunk = step4_parts(audit, fold, parts, step4)
    ids_o, ids_d = fold.rows["O"].player_id.to_numpy(), fold.rows["D"].player_id.to_numpy()
    out = {"step3": (dict(zip(ids_o, parts.prior_off + parts.u_off)),
                     dict(zip(ids_d, -(parts.prior_def + parts.u_def)))),
           "step4": (dict(zip(ids_o, sum(shrunk["O"]))), dict(zip(ids_d, -sum(shrunk["D"]))))}
    for stage, ratings in (extra or {}).items():
        out[stage] = (dict(zip(ids_o, ratings["O"])), dict(zip(ids_d, -ratings["D"])))
    if f"pieces_{SHIPPED_FAMILY}" in out:
        for side in (0, 1):
            shipped, mine = out["step4"][side], out[f"pieces_{SHIPPED_FAMILY}"][side]
            worst = max(abs(shipped[key] - mine[key]) for key in shipped)
            if worst > PIECES_TOLERANCE:
                raise SystemExit(f"{fold.stem}: step 4 through the pieces code misses the shipped step by {worst:.1e}")
    return out


def stint_fold_errors(audit, fold, teams: dict, design, step4: tuple, extra: dict | None = None) -> dict:
    """{(stage, slice): (sum of w x WOWY miss squared, sum of w)} for one fold, scored stint by stint: every stint of
    the held-back games, predicted from the ten players on the court, the team levels fitted out as in `fold_errors`."""
    absent = regulars_absent(fold, teams)
    key = pd.DataFrame({"game_id": teams["game_id"].astype(str), "is_home_off": teams["is_home_off"].astype(bool),
                        "team_off": teams["team_off"], "team_def": teams["team_def"], "dnp": absent})
    games = design.games.set_index("game_idx").game_id.astype(str)
    rows = design.rows.assign(game_id=design.rows.game_idx.map(games).to_numpy())
    rows = rows.reset_index().merge(key, on=["game_id", "is_home_off"], how="inner")
    index = rows["index"].to_numpy()
    y, w = design.y[index], design.w[index]
    columns = design.spec.players.set_index("player_column").player_id
    lineup_o = columns.reindex(design.parts["lineup_o"][index].ravel()).to_numpy().reshape(-1, 5)
    lineup_d = columns.reindex(design.parts["lineup_d"][index].ravel()).to_numpy().reshape(-1, 5)
    off_codes, _ = pd.factorize(rows.team_off)
    def_codes, _ = pd.factorize(rows.team_def)
    n = len(rows)
    levels = sp.hstack([
        sp.csr_matrix((np.ones(n), (np.arange(n), off_codes)), shape=(n, off_codes.max() + 1)),
        sp.csr_matrix((np.ones(n), (np.arange(n), def_codes)), shape=(n, def_codes.max() + 1))[:, 1:],
        sp.csr_matrix(rows.is_home_off.to_numpy(float)[:, None]),
        sp.csr_matrix((rows.phase == "PO").to_numpy(float)[:, None])]).tocsr()
    keep = w > 0
    out = {}
    for stage, (rating_off, allowed) in stage_ratings(audit, fold, step4, extra).items():
        pred = (pd.DataFrame(lineup_o).apply(lambda c: c.map(rating_off)).fillna(fold.fill_o).to_numpy().sum(axis=1)
                + pd.DataFrame(lineup_d).apply(lambda c: c.map(allowed)).fillna(fold.fill_d).to_numpy().sum(axis=1))
        left = np.zeros(n)
        left[keep] = wowy_miss((y - pred)[keep], levels[keep], w[keep])
        for slice_, mask in (("all", keep), ("dnp", keep & rows.dnp.to_numpy())):
            out[(stage, slice_)] = (float(np.sum(w[mask] * left[mask] ** 2)), float(w[mask].sum()))
    out[("share", "dnp")] = (float(w[keep & rows.dnp.to_numpy()].sum()), float(w[keep].sum()))
    return out


# ------------------------------------------------------------------------------------------- step 4 in pieces
class PieceStages:
    """Check 145's families on every fold season of a system: each fold's normal equations, then for a scored season
    each family's multipliers from its 8 nearest fold seasons (the test rule) and the ratings they give."""

    def __init__(self, pieces_module, families: list, folds: list, where: dict):
        from openrapm.prior_shrink import fold_seasons_used
        self.module, self.where = pieces_module, where
        self.families = list(dict.fromkeys([SHIPPED_FAMILY, *families]))
        unknown = set(families) - set(pieces_module.FAMILIES)
        if unknown:
            raise SystemExit(f"unknown piece families {sorted(unknown)}; choose from {list(pieces_module.FAMILIES)}")
        self.sums: dict = {}
        for fold in folds:
            self.sums.setdefault(fold.season, pieces_module.Season()).add(fold, self.read(fold))
        fold_seasons = sorted(self.sums)
        self.multipliers = {}
        for season in fold_seasons:
            used = fold_seasons_used(season, fold_seasons, "test", NEAREST)
            self.multipliers[season] = {
                family: pieces_module.solve(sum(self.sums[s].sums[family]["gram"] for s in used),
                                            sum(self.sums[s].sums[family]["rhs"] for s in used))
                for family in self.families} if used else None

    def read(self, fold) -> pd.DataFrame:
        path = self.where[fold.stem] / f"pieces_{fold.stem}.parquet"
        if not path.exists():
            raise SystemExit(f"{path} is missing: build the folds with `openrapm folds --pieces`")
        return pd.read_parquet(path)

    def stages(self, fold) -> dict:
        """{"pieces_<family>": {"O", "D"}} for one fold, from its season's multipliers."""
        multipliers, pieces = self.multipliers[fold.season], self.read(fold)
        return {f"pieces_{family}": self.module.family_ratings(fold, pieces, family, multipliers[family])
                for family in self.families}

    def table(self, name: str) -> pd.DataFrame:
        """Every fold season's multipliers, one row per family, side and piece."""
        rows = []
        for season, by_family in self.multipliers.items():
            for family, values in (by_family or {}).items():
                columns = self.module.FAMILIES[family]["columns"]
                for side_index, side in enumerate(("O", "D")):
                    for k, piece in enumerate(columns):
                        rows.append(dict(system=name, season=season, family=family, side=side, piece=piece,
                                         multiplier=float(values[side_index * len(columns) + k])))
        return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------- one system
def system_folds(within, name: str) -> tuple[list, dict]:
    """(every fold on disk of the system, {stem: its folder}): step 4's multipliers for a scored season come from the
    seasons around it."""
    from openrapm.data.within_folds import load_folds
    folds, where = [], {}
    for folder in (within / f"{name}_early", within / f"{name}_within"):
        if not folder.exists():
            continue
        for fold in load_folds(folder, (1997, 2026)):
            folds.append(fold)
            where[fold.stem] = folder
    return folds, where


def piece_stages(name: str, folds: list, where: dict, unit: str, families: list | None,
                 multiplier_tables: list | None) -> PieceStages | None:
    """The system's step-4-in-pieces stages, when families are asked and every fold carries pieces; else None."""
    if not families or not all((where[fold.stem] / f"pieces_{fold.stem}.parquet").exists() for fold in folds):
        return None
    if unit != "stint":
        raise SystemExit("--pieces scores stint by stint only")
    pieces = PieceStages(_pieces(), families, folds, where)
    if multiplier_tables is not None:
        multiplier_tables.append(pieces.table(name))
    return pieces


def system_errors(audit, within, name: str, seasons: tuple, unit: str = "stint", families: list | None = None,
                  multiplier_tables: list | None = None) -> pd.DataFrame:
    """One row per season, stage and slice: the system's WOWY error over the season's folds.  With `families` (check
    145's), a system whose folds carry pieces also gets a stage per family; its multipliers go to
    `multiplier_tables`."""
    from openrapm.prior_shrink import multipliers_by_season
    folds, where = system_folds(within, name)
    scored = [fold for fold in folds if seasons[0] <= fold.season <= seasons[1]]
    multipliers = multipliers_by_season(sorted({fold.season for fold in scored}), folds, "test", NEAREST,
                                        free_games=FREE_GAMES)
    multipliers = multipliers.set_index("season")
    pieces = piece_stages(name, folds, where, unit, families, multiplier_tables)
    sums: dict = {}
    cache = None
    if unit == "stint":
        from openrapm.data.config import load_config
        from openrapm.data.designs import DesignCache
        cache = DesignCache(load_config(), keep=1)
    for fold in scored:
        row = multipliers.loc[fold.season]
        step4 = (float(row.multiplier_off), float(row.multiplier_def), float(row.get("games_multiplier_off", 1.0)),
                 float(row.get("games_multiplier_def", 1.0)))
        teams = team_arrays(where[fold.stem], fold.stem)
        if unit == "stint":
            errors = stint_fold_errors(audit, fold, teams, cache.season(fold.season, "pts"), step4,
                                       pieces.stages(fold) if pieces is not None else None)
        else:
            errors = fold_errors(audit, fold, teams, step4)
        for key, (top, bottom) in errors.items():
            acc = sums.setdefault((fold.season, *key), [0.0, 0.0])
            acc[0] += top
            acc[1] += bottom
    return pd.DataFrame([dict(system=name, season=season, stage=stage, slice=slice_, error=top / bottom)
                         for (season, stage, slice_), (top, bottom) in sums.items()])


def paired(seasons_table: pd.DataFrame, system: str, reference: str) -> list:
    """Each of the system's stages and slices against the reference season by season.  A `pieces_<family>` stage,
    which only a system built with pieces has, is read against the reference's `step4`: the shipped step it would
    replace."""
    rows = []
    table = seasons_table[seasons_table.stage != "share"]
    for (stage, slice_), group in table[table.system == system].groupby(["stage", "slice"]):
        against = "step4" if stage.startswith("pieces_") else stage
        mine = group.set_index("season").error
        theirs = table[(table.system == reference) & (table.stage == against) & (table.slice == slice_)]
        diff = (mine - theirs.set_index("season").error).dropna()
        if len(diff) < 2:
            continue
        se = float(diff.std(ddof=1) / np.sqrt(len(diff)))
        rows.append(dict(system=system, reference=reference, stage=stage, against=against, slice=slice_,
                         error=float(mine.reindex(diff.index).mean()), mean_change=float(diff.mean()), se=se,
                         z=float(diff.mean() / se) if se > 0 else np.nan, seasons_better=int((diff < 0).sum()),
                         seasons=len(diff)))
    return rows


def main(argv=None) -> int:
    use_utf8()
    parser = argparse.ArgumentParser(description="The with-or-without-you test on the within-season folds.")
    parser.add_argument("--systems", required=True, help="comma-separated names; each reads <name>_early/_within")
    parser.add_argument("--ref", required=True, help="the reference system (the incumbent)")
    parser.add_argument("--tag", default="wowy", help="names the output files (default %(default)s)")
    parser.add_argument("--unit", default="stint", choices=("stint", "game"),
                        help="score stint by stint (the decider) or team-game by team-game (default %(default)s)")
    parser.add_argument("--pieces", default="",
                        help="check 145's families, comma-separated (source, certainty, ...): a stage each for the "
                             "systems whose folds carry pieces (`openrapm folds --pieces`)")
    parser.add_argument("--first", type=int, default=1997)
    parser.add_argument("--last", type=int, default=2026)
    args = parser.parse_args(argv)
    from openrapm.data.config import load_config, resolve
    cfg = load_config()
    started, outputs = time.time(), resolve(cfg, "outputs")
    within = resolve(cfg, "within") if "within" in cfg.get("paths", {}) else outputs / "within"
    audit = _audit()
    # each name once, in order: the chain names the noise control beside the candidate, which may be the control itself
    names = list(dict.fromkeys(name.strip() for name in args.systems.split(",") if name.strip()))
    if args.ref not in names:
        names.insert(0, args.ref)
    tables, multiplier_tables = [], []
    families = [family for family in args.pieces.split(",") if family]
    for name in names:
        tables.append(system_errors(audit, within, name, (args.first, args.last), args.unit, families,
                                    multiplier_tables))
        print(f"  {name}: {tables[-1].season.nunique()} seasons ({time.time() - started:.0f}s)", flush=True)
    seasons_table = pd.concat(tables, ignore_index=True)
    (outputs / "csv").mkdir(exist_ok=True)
    seasons_table.to_csv(outputs / "csv" / f"wowy_{args.tag}_seasons.csv", index=False)
    share = seasons_table[(seasons_table.stage == "share") & (seasons_table.system == args.ref)].error.mean()
    summary = pd.DataFrame([row for name in names if name != args.ref for row in paired(seasons_table, name, args.ref)])
    summary.to_csv(outputs / "csv" / f"wowy_{args.tag}.csv", index=False)
    if multiplier_tables:
        pd.concat(multiplier_tables).to_csv(outputs / "csv" / f"wowy_{args.tag}_piece_multipliers.csv", index=False)
    pd.set_option("display.width", 200)
    print(f"\nDNP team-games: {share:.1%} of the held-back possessions")
    print("each system minus the reference, season by season; below 0 is better")
    print(summary.round(5).to_string(index=False))
    print(f"done in {time.time() - started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
