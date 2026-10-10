"""The pieces of one season's vanilla RAPM, per side: the Decomposition page's two exact splits, by player and by
possession, with every piece cut into its offensive and its defensive half.

Moved here from scripts/84_piece_panel.py (2026-09-28) so that the three callers run one piece of code: 84's
raw-points panel for the site, the season panel's luck-adjusted pieces (scripts/86_context_panel.py), and the
single-year rankings' cross-fit (scripts/62_single_year_board.py), which rebuilds them from each game fold's
training games the way it already rebuilds the on-court columns.

**By player, per side.**  Offence, over his offensive possessions:

    on_rtg     his team's points per 100 while he is on the court, minus the season's league average
    teammates  minus the offensive RAPM of the four teammates on the court with him, added up
    opponents  the defensive RAPM of the five defenders he faced, added up
    context    minus what home court, playoffs, garbage time and the score margin predict for those
               possessions, measured from the league average.  It also carries the published zero point's
               constant, 5 x the two sides' shifts, which the defence carries with the opposite sign, so it
               cancels in the net.
    ridge      the ridge's pull toward zero: minus penalty x his offensive coefficient / his possessions

Defence mirrors it over his defensive possessions, positive = good.  Each side's five add up to that side's RAPM
exactly.

**By possession, per side.**  His published rating is c'beta = sum over rows of h_r y_r with h = W X A^-1 c; the
contrast c is split into its offensive and its defensive half and h is summed over four groups of rows: his
possessions (on_signal), his team's possessions without him in games he played (off_adj_gp), his team's
possessions in games he missed (off_adj_dnp), and every other possession (team_sos).  The context columns are
unpenalised, so h is orthogonal to them and each side's four add up to that side's RAPM exactly.

Beside the pieces, not one of them: the actual off-court rating per side (his team's points per 100 without him,
games he played, minus the league average; defence positive = good) and the possessions behind each group.

**The fit.**  Plain ridge on one season's design, the given penalty on every player column and none on the
context columns (an empty context column -- the 2020 bubble's playoff home court -- gets a token 1.0, which makes
its coefficient exactly 0 and moves nothing else), published at possession-weighted zero per side over the players
with `MIN_POSSESSIONS`.  It is scripts/83_decompose_site.py's `span_fit` for a single season, operation for
operation, so 84's panel is bit-identical through either.  A row's team comes from `teams`, a (game_id, player_id)
-> team_id lookup the caller builds from the box scores: the first player of that side the box score lists.  This
module never opens a file.

A design that is a SUBSET of a season (`WindowData.subset`, a cross-fitting fold) has no stored parts; the context
block and the lineups are then read back off its matrix, which carries the same numbers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.linalg as sla
import scipy.sparse as sp

from . import pad
from .looseason import LeaveSeasonOutRAPM
from .singleyear import MIN_POSSESSIONS

__all__ = ["BY_PLAYER", "BY_POSSESSION", "PIECES", "PENALTY", "COLUMNS", "blocks", "fit", "decompose_sides",
           "season_pieces", "side_team"]

BY_PLAYER = ["on_rtg", "teammates", "opponents", "context", "ridge"]
BY_POSSESSION = ["on_signal", "off_adj_gp", "off_adj_dnp", "team_sos"]
PIECES = BY_PLAYER + BY_POSSESSION
PENALTY = 3000.0                  # the Decomposition page's penalty per side (the owner's, scripts 80-82)
# the prior's columns: every piece and the off-court rating, `_o` from the offensive design, `_d` from the defensive
COLUMNS = [f"pc_{k}_{tag}" for tag in ("o", "d") for k in PIECES + ["off_rtg"]]


def side_team(teams: pd.Series, game_ids: np.ndarray, players: np.ndarray) -> np.ndarray:
    """Each row's team on one side: the team of that side's first player the box score lists (-1 if none is)."""
    got = np.full(len(game_ids), np.nan)
    for k in range(players.shape[1]):
        todo = np.isnan(got)
        if not todo.any():
            break
        idx = pd.MultiIndex.from_arrays([game_ids[todo], players[todo, k]])
        got[todo] = teams.reindex(idx).to_numpy(dtype=float)
    return np.where(np.isnan(got), -1, got).astype(np.int64)


def _lineups(Z) -> np.ndarray:
    """The five column indices of each row of one side's player block, sorted, as an (n, 5) array."""
    Z = sp.csr_matrix(Z)
    Z.sort_indices()
    counts = np.diff(Z.indptr)
    if not (counts == 5).all():
        raise ValueError(f"{int((counts != 5).sum())} rows do not have five players on this side")
    return Z.indices.reshape(-1, 5).astype(np.int64)


def blocks(wd):
    """The context block and the two sides' lineups: stored on a whole season's design, read off a subset's X."""
    if wd.parts is not None:
        return (np.asarray(wd.parts["F"], dtype=float), np.asarray(wd.parts["lineup_o"]),
                np.asarray(wd.parts["lineup_d"]))
    n, n_f = wd.spec.n_ps, len(wd.spec.f_names)
    X = sp.csr_matrix(wd.X)
    return (np.asarray(X[:, 2 * n:2 * n + n_f].todense(), dtype=float), _lineups(X[:, :n]),
            _lineups(X[:, n:2 * n]))


def fit(wd, teams: pd.Series, lam: float = PENALTY) -> dict:
    """One season's vanilla RAPM on the design `wd`, in the layout `decompose_sides` (and 83's `decompose`) read.

    `teams` maps (game_id, player_id) to a team id; it names each row's two teams, which the by-possession groups
    need.  `wd` may be a whole season or a subset of its rows.
    """
    seasons = sorted(set(int(s) for s in wd.spec.seasons))
    if len(seasons) != 1:
        raise ValueError(f"one season at a time; this design spans {seasons}")
    season = seasons[0]
    rapm = LeaveSeasonOutRAPM(min_possessions=0.0)
    rapm.add_season(season, wd)
    gram, rhs, poss, n_context = rapm._assemble(())
    n = poss.size
    p = 2 * n + n_context
    stored = rapm._season[season]
    F, lineup_o, lineup_d = blocks(wd)
    rows, n_f = F.shape
    if n_f != stored["n_context"]:
        raise ValueError(f"{season}: {n_f} context columns in the design, {stored['n_context']} in the fit")
    lo, ld = stored["slot"][lineup_o], stored["slot"][lineup_d]
    column = 2 * n
    indices = np.hstack([lo, ld + n, np.broadcast_to(column + np.arange(n_f), (rows, n_f))])
    data = np.hstack([np.ones((rows, 10)), F])
    X = sp.csr_matrix((data.ravel(), indices.ravel(),
                       np.arange(0, rows * (10 + n_f) + 1, 10 + n_f, dtype=np.int64)), shape=(rows, p))
    gid = wd.games.set_index("game_idx").game_id.reindex(wd.rows.game_idx.to_numpy()).to_numpy().astype(str)
    team_att = side_team(teams, gid, rapm.player_ids[lo])
    team_def = side_team(teams, gid, rapm.player_ids[ld])
    season_row = np.full(rows, season, dtype=np.int64)
    diag = np.diagonal(gram)
    penalty = np.concatenate([np.full(2 * n, float(lam)), np.zeros(n_context)])
    penalty[(diag <= 0) & (penalty == 0)] = 1.0
    factor = sla.cho_factor(gram + np.diag(penalty))
    beta = sla.cho_solve(factor, rhs)
    keep = poss >= MIN_POSSESSIONS            # the published zero point, as the rankings pages set it
    w_off, w_def = np.where(keep, poss, 0.0), np.where(keep, diag[n:2 * n], 0.0)
    off = beta[:n] - w_off @ beta[:n] / w_off.sum()
    dfn = -(beta[n:2 * n] - w_def @ beta[n:2 * n] / w_def.sum())
    # every row's two teams as team-season and team-game codes; a side the box scores cannot name gets a code of
    # its own, so it never counts as anyone's team
    game_code = pd.factorize(gid)[0].astype(np.int64)
    unknown_att, unknown_def = -1 - np.arange(rows), -1 - rows - np.arange(rows)
    ts = np.concatenate([np.where(team_att >= 0, season_row * 10**10 + team_att, unknown_att),
                         np.where(team_def >= 0, season_row * 10**10 + team_def, unknown_def)])
    gt = np.concatenate([np.where(team_att >= 0, game_code * 10**10 + team_att, unknown_att),
                         np.where(team_def >= 0, game_code * 10**10 + team_def, unknown_def)])
    ts_code, gt_code = pd.factorize(ts)[0], pd.factorize(gt)[0]
    keys = dict(ts_att=ts_code[:rows], ts_def=ts_code[rows:], gt_att=gt_code[:rows], gt_def=gt_code[rows:],
                n_ts=int(ts_code.max()) + 1, n_gt=int(gt_code.max()) + 1,
                unplaced=float(((team_att < 0) | (team_def < 0)).mean()))
    return dict(n=n, X=X, lineup_o=lo, lineup_d=ld, w=np.asarray(wd.w, dtype=float), y=np.asarray(wd.y, dtype=float),
                beta=beta, diag=diag, poss=poss, off=off, dfn=dfn, player_ids=rapm.player_ids.copy(), lam=float(lam),
                factor=factor, w_off=w_off / w_off.sum(), w_def=w_def / w_def.sum(), keys=keys)


def decompose_sides(fit: dict) -> tuple:
    """Every player's nine pieces per side, the off-court rating per side, the possessions behind them, and the
    largest misses of the identities they rest on."""
    n, X, w, y, beta, off, dfn = (fit[k] for k in ("n", "X", "w", "y", "beta", "off", "dfn"))
    lam, diag, keys = fit["lam"], fit["diag"], fit["keys"]
    resid = y - X @ beta
    context_row = X[:, 2 * n:] @ beta[2 * n:]
    z = y - context_row
    league = float(np.average(y, weights=w))
    # the published zero point: off = beta_off - shift_off, dfn = -(beta_def - shift_def); a row's prediction is
    # then (the five attackers' off) - (the five defenders' dfn) + context + 5 x (shift_off + shift_def)
    shift = 5.0 * float(fit["w_off"] @ beta[:n] + fit["w_def"] @ beta[n:2 * n])
    sum_o = off[fit["lineup_o"]].sum(axis=1)
    sum_d = dfn[fit["lineup_d"]].sum(axis=1)
    rows = np.arange(len(w))
    at_o = sp.csc_matrix((np.ones(fit["lineup_o"].size), (np.repeat(rows, 5), fit["lineup_o"].ravel())),
                         shape=(len(w), n))
    at_d = sp.csc_matrix((np.ones(fit["lineup_d"].size), (np.repeat(rows, 5), fit["lineup_d"].ravel())),
                         shape=(len(w), n))
    p = X.shape[1]
    A_inv = sla.cho_solve(fit["factor"], np.eye(p))
    c_off, c_def = np.zeros(p), np.zeros(p)
    c_off[:n], c_def[n:2 * n] = -fit["w_off"], fit["w_def"]
    base_off, base_def = X @ (A_inv @ c_off), X @ (A_inv @ c_def)
    miss = dict(by_player=0.0, by_possession=0.0, ridge=0.0, weights=0.0)

    def mean(v, r):
        return float(np.average(v[r], weights=w[r]))

    out = []
    for i in np.flatnonzero((np.diff(at_o.indptr) > 0) & (np.diff(at_d.indptr) > 0)):
        ro, rd = at_o.indices[at_o.indptr[i]:at_o.indptr[i + 1]], at_d.indices[at_d.indptr[i]:at_d.indptr[i + 1]]
        rec = dict(i=i, poss_off=float(fit["poss"][i]), poss_def=float(diag[n + i]),
                   rapm_off=float(off[i]), rapm_def=float(dfn[i]),
                   on_rtg_off=mean(y, ro) - league,
                   teammates_off=-(mean(sum_o, ro) - off[i]),
                   opponents_off=mean(sum_d, ro),
                   context_off=-(mean(context_row, ro) - league) - shift,
                   ridge_off=-mean(resid, ro),
                   on_rtg_def=-(mean(y, rd) - league),
                   teammates_def=-(mean(sum_d, rd) - dfn[i]),
                   opponents_def=mean(sum_o, rd),
                   context_def=(mean(context_row, rd) - league) + shift,
                   ridge_def=mean(resid, rd))
        # by possession: 83's groups, each side's half of the contrast on its own
        on = np.zeros(len(w), dtype=bool)
        on[ro], on[rd] = True, True
        his_ts, his_gt = np.zeros(keys["n_ts"], dtype=bool), np.zeros(keys["n_gt"], dtype=bool)
        his_ts[keys["ts_att"][ro]], his_ts[keys["ts_def"][rd]] = True, True
        his_gt[keys["gt_att"][ro]], his_gt[keys["gt_def"][rd]] = True, True
        team_att, team_def = his_ts[keys["ts_att"]], his_ts[keys["ts_def"]]
        played_att, played_def = his_gt[keys["gt_att"]], his_gt[keys["gt_def"]]
        without = ((team_att & played_att) | (team_def & played_def)) & ~on
        missed = ((team_att & ~played_att) | (team_def & ~played_def)) & ~on & ~without
        rest = ~(on | without | missed)
        for side, h in (("off", w * (base_off + X @ A_inv[:, i])), ("def", w * (base_def - X @ A_inv[:, n + i]))):
            hz = h * z
            rec.update({f"on_signal_{side}": hz[on].sum(), f"off_adj_gp_{side}": hz[without].sum(),
                        f"off_adj_dnp_{side}": hz[missed].sum(), f"team_sos_{side}": hz[rest].sum()})
        # beside the pieces: the actual off-court rating, games he played, and the possessions behind each group
        off_att, off_def = team_att & played_att & ~on, team_def & played_def & ~on
        rec["off_rtg_off"] = mean(y, off_att) - league if off_att.any() else np.nan
        rec["off_rtg_def"] = -(mean(y, off_def) - league) if off_def.any() else np.nan
        rec["poss_gp_off"], rec["poss_gp_def"] = float(w[off_att].sum()), float(w[off_def].sum())
        rec["poss_dnp_off"] = float(w[missed & team_att].sum())
        rec["poss_dnp_def"] = float(w[missed & team_def].sum())
        for side, rapm in (("off", off[i]), ("def", dfn[i])):
            miss["by_player"] = max(miss["by_player"], abs(sum(rec[f"{k}_{side}"] for k in BY_PLAYER) - rapm))
            miss["by_possession"] = max(miss["by_possession"],
                                        abs(sum(rec[f"{k}_{side}"] for k in BY_POSSESSION) - rapm))
        miss["ridge"] = max(miss["ridge"], abs(mean(resid, ro) - lam * beta[i] / diag[i]),
                            abs(mean(resid, rd) - lam * beta[n + i] / diag[n + i]))
        miss["weights"] = max(miss["weights"], abs(w[ro].sum() - fit["poss"][i]) / fit["poss"][i])
        out.append(rec)
    tab = pd.DataFrame(out)
    tab["league"], tab["side_constant"] = league, shift
    return tab, miss


def season_pieces(wd_o, wd_d, teams: pd.Series, lam: float = PENALTY) -> tuple:
    """The prior's piece columns for one season (or one fold of it): one row per player, `COLUMNS`.

    Each side's pieces come from that side's own design -- the offensive ones from a fit on `wd_o`, the defensive
    ones from a fit on `wd_d` -- so on the luck-adjusted designs (`xpts_ft`, `x3def`) they measure what the
    on-court columns and the prior's labels measure.  Every piece is padded toward 0 over the possessions behind
    it (`pad.shrink(piece, n, lam, 0)`: the project's rule that no rate goes in unpadded, with the decomposition's
    own penalty as the constant); the off-court rating over his team's possessions without him.  A player with no
    rows on one side of a fit has nothing to pad and reads 0 there.

    Returns the frame and, per side, the largest misses of the identities (`decompose_sides`), which a caller
    should hold under 1e-9.
    """
    parts, misses = [], {}
    for tag, wd, side in (("o", wd_o, "off"), ("d", wd_d, "def")):
        f = fit(wd, teams, lam)
        tab, misses[tag] = decompose_sides(f)
        misses[tag]["unplaced"] = f["keys"]["unplaced"]
        frame = pd.DataFrame({"player_id": f["player_ids"][tab.i.to_numpy()].astype(np.int64)})
        n = tab[f"poss_{side}"].to_numpy(float)
        for k in PIECES:
            frame[f"pc_{k}_{tag}"] = pad.shrink(tab[f"{k}_{side}"].to_numpy(float), n, lam, 0.0)
        gp = tab[f"poss_gp_{side}"].to_numpy(float)
        frame[f"pc_off_rtg_{tag}"] = pad.shrink(np.nan_to_num(tab[f"off_rtg_{side}"].to_numpy(float)), gp, lam, 0.0)
        parts.append(frame.set_index("player_id"))
    out = parts[0].join(parts[1], how="outer").fillna(0.0).reset_index()
    return out[["player_id"] + COLUMNS], misses
