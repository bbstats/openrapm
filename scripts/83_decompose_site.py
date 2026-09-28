"""Plain RAPM, piece by piece, for the site: every player's RAPM over one season and over each three-season window,
split two ways that each add up to it exactly -- by player and by possession.  Writes docs/data/decompose_1y.json
and docs/data/decompose_3y.json for docs/decompose.html.

    python scripts/83_decompose_site.py [--penalty=3000]

**By player** -- who was on the court:

    RAPM = on-court rtg + teammates + opponents + context + ridge penalty

  on-court rtg    his on-court rating: his team's points per 100 possessions minus the opponents', while he is on
                  the court (only those possessions -- nothing from when he sits)
  teammates       minus the RAPM of the four teammates on the court with him, added up, weighted by time together
  opponents       the RAPM of the five opponents on the court with him, added up, weighted by time
  context         minus what home court, playoffs, garbage time, the score margin and the season predict for his
                  minutes
  ridge penalty   the ridge's pull toward zero: minus penalty x his coefficient / his possessions, on each side

**Why it is exact.**  It is the ridge's own first-order condition for his offensive and his defensive column: the
residuals of the possessions he played add up to penalty x his coefficient.  On the published zero point
(possession-weighted zero per side) each side picks up a league-wide constant, 5 x the two sides' shifts, and the
two cancel in the net, so the net needs no constant.  Each piece averages zero over the league, weighted by
possessions.  Teammates' and opponents' ratings come from the same fit, so the split explains his rating given
theirs.  scripts/82_decompose_rapm.py has the seven-piece version that starts from raw on/off instead.

**By possession** -- which possessions the rating came from:

    RAPM = on court + off court (GP) + off court (DNP) + other games

RAPM is linear in the outcomes: his published rating is c'beta = sum over rows of h_r y_r, h = W X A^-1 c.  The
context columns are unpenalised, so h is orthogonal to every one of them, and measuring each outcome against its
fitted context prediction leaves the total unchanged.  Each piece adds up h_r x (outcome - context prediction) over
one group of rows: the possessions he played; his team's possessions without him in games he played; his teams'
possessions in games he did not play, in seasons he played for them; and every other possession in the span.  A
row's team comes from the box scores (the team of the players on that side).  A season he missed entirely cannot be
placed -- injured players are not in the box scores -- so it lands in other games; for a traded player, his other
team's games before or after his stint count as games he missed.

Plain RAPM: raw points, regular season and playoffs, one fit per span -- every season alone, and each of
config.yaml's ten three-season windows -- at a penalty of 3,000 per side, the owner's choice in scripts 80-82.
Unlike the main page's ratings it has no box-score prior.  The run stops on a failed check.
"""
import json
import os
import sys
import time
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
os.environ.setdefault("NUMBA_NUM_THREADS", os.environ.get("OPENRAPM_NUMBA_THREADS", "4"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.linalg as sla  # noqa: E402
import scipy.sparse as sp  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _cli import check_flags, flag  # noqa: E402
from eracoef import singleyear as sy  # noqa: E402
from eracoef.config import load_config  # noqa: E402
from eracoef.looseason import LeaveSeasonOutRAPM  # noqa: E402
from eracoef.windows import build_window  # noqa: E402

FIELDS = ["n", "p", "m", "t", "o", "x", "r", "a", "h", "w", "g", "e"]   # name, possessions, the five pieces by
PIECES = ["margin", "teammates", "opponents", "context", "ridge"]        # player, RAPM, the four by possession
SOURCES = ["his_minutes", "team_without_him", "games_missed", "everything_else"]
UNPLACED_TOL = 0.001           # rows whose team the box scores cannot name: under this share of the span's rows
NAME_SOURCES = ("outputs/season_ratings_product.parquet", "outputs/season_ratings.parquet",
                "artifacts/season_ratings.parquet")  # scripts/52_site.py's order
CENTRE_TOL = 0.05              # a piece's possession-weighted league average must sit this close to zero


class CheckFailed(SystemExit):
    pass


def check(ok: bool, what: str, log: list) -> None:
    """Record a check; stop the run on a failure, so nothing is published past one."""
    log.append(("PASS  " if ok else "FAIL  ") + what)
    print(("  PASS  " if ok else "  FAIL  ") + what, flush=True)
    if not ok:
        raise CheckFailed(f"check failed: {what}")


def label(seasons) -> str:
    return str(seasons[0]) if len(seasons) == 1 else f"{seasons[0]}-{str(seasons[-1])[-2:]}"


_BOX_TEAMS: dict = {}


def box_teams(season: int) -> pd.Series:
    """Every player's team in every game of a season, from the box scores: (game_id, player_id) -> team id."""
    if season not in _BOX_TEAMS:
        frames = []
        for f in sorted((ROOT / "data" / "raw" / "box" / str(season)).glob("*.parquet")):
            b = pd.read_parquet(f, columns=["personId", "teamId"])
            frames.append(pd.DataFrame({"game_id": f.stem, "player_id": b.personId.to_numpy(np.int64),
                                        "team_id": b.teamId.to_numpy(np.int64)}))
        t = pd.concat(frames).drop_duplicates(["game_id", "player_id"]) if frames else \
            pd.DataFrame(columns=["game_id", "player_id", "team_id"])
        _BOX_TEAMS[season] = t.set_index(["game_id", "player_id"]).team_id
    return _BOX_TEAMS[season]


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


def span_fit(seasons, cfg, lam: float) -> dict:
    """One span's plain RAPM: its rows in global columns, every row's ten players, the fit, the published ratings."""
    rapm, designs = LeaveSeasonOutRAPM(min_possessions=0.0), {}
    for season in seasons:
        designs[season] = build_window([season], cfg, target="pts")
        rapm.add_season(season, designs[season])
    gram, rhs, poss, n_context = rapm._assemble(())
    n = poss.size
    p = 2 * n + n_context
    parts, lineup_o, lineup_d, w, y, column = [], [], [], [], [], 2 * n
    game_ids, team_att, team_def, season_row = [], [], [], []
    for season in rapm.seasons:                  # _assemble's order, so its context offsets are ours
        wd, stored = designs[season], rapm._season[season]
        F = np.asarray(wd.parts["F"], dtype=float)
        rows, n_f = F.shape
        if n_f != stored["n_context"]:
            raise SystemExit(f"{season}: {n_f} context columns in the design, {stored['n_context']} in the fit")
        lo, ld = stored["slot"][np.asarray(wd.parts["lineup_o"])], stored["slot"][np.asarray(wd.parts["lineup_d"])]
        indices = np.hstack([lo, ld + n, np.broadcast_to(column + np.arange(n_f), (rows, n_f))])
        data = np.hstack([np.ones((rows, 10)), F])
        parts.append(sp.csr_matrix((data.ravel(), indices.ravel(),
                                    np.arange(0, rows * (10 + n_f) + 1, 10 + n_f, dtype=np.int64)), shape=(rows, p)))
        lineup_o.append(lo)
        lineup_d.append(ld)
        w.append(np.asarray(wd.w, dtype=float))
        y.append(np.asarray(wd.y, dtype=float))
        gid = wd.games.set_index("game_idx").game_id.reindex(wd.rows.game_idx.to_numpy()).to_numpy().astype(str)
        teams = box_teams(season)
        game_ids.append(gid)
        team_att.append(side_team(teams, gid, rapm.player_ids[lo]))
        team_def.append(side_team(teams, gid, rapm.player_ids[ld]))
        season_row.append(np.full(rows, season, dtype=np.int64))
        column += n_f
    diag = np.diagonal(gram)
    penalty = np.concatenate([np.full(2 * n, float(lam)), np.zeros(n_context)])
    # an empty context column -- the 2020 bubble's playoff home court -- gets a token penalty: its coefficient is
    # then exactly 0 and nothing else moves, where unpenalized it makes the system singular
    penalty[(diag <= 0) & (penalty == 0)] = 1.0
    factor = sla.cho_factor(gram + np.diag(penalty))
    beta = sla.cho_solve(factor, rhs)
    keep = poss >= sy.MIN_POSSESSIONS            # the published zero point, as the rankings pages set it
    w_off, w_def = np.where(keep, poss, 0.0), np.where(keep, diag[n:2 * n], 0.0)
    off = beta[:n] - w_off @ beta[:n] / w_off.sum()
    dfn = -(beta[n:2 * n] - w_def @ beta[n:2 * n] / w_def.sum())
    # every row's two teams as team-season and team-game codes; a side the box scores cannot name gets a code of
    # its own, so it never counts as anyone's team
    team_att, team_def = np.concatenate(team_att), np.concatenate(team_def)
    season_row, game_code = np.concatenate(season_row), pd.factorize(np.concatenate(game_ids))[0].astype(np.int64)
    rows_all = len(team_att)
    unknown_att, unknown_def = -1 - np.arange(rows_all), -1 - rows_all - np.arange(rows_all)
    ts = np.concatenate([np.where(team_att >= 0, season_row * 10**10 + team_att, unknown_att),
                         np.where(team_def >= 0, season_row * 10**10 + team_def, unknown_def)])
    gt = np.concatenate([np.where(team_att >= 0, game_code * 10**10 + team_att, unknown_att),
                         np.where(team_def >= 0, game_code * 10**10 + team_def, unknown_def)])
    ts_code, gt_code = pd.factorize(ts)[0], pd.factorize(gt)[0]
    keys = dict(ts_att=ts_code[:rows_all], ts_def=ts_code[rows_all:], gt_att=gt_code[:rows_all],
                gt_def=gt_code[rows_all:], n_ts=int(ts_code.max()) + 1, n_gt=int(gt_code.max()) + 1,
                unplaced=float(((team_att < 0) | (team_def < 0)).mean()))
    return dict(n=n, X=sp.vstack(parts).tocsr(), lineup_o=np.vstack(lineup_o), lineup_d=np.vstack(lineup_d),
                w=np.concatenate(w), y=np.concatenate(y), beta=beta, diag=diag, poss=poss, off=off, dfn=dfn,
                player_ids=rapm.player_ids.copy(), lam=float(lam), factor=factor, w_off=w_off / w_off.sum(),
                w_def=w_def / w_def.sum(), keys=keys)


def decompose(fit: dict) -> tuple:
    """Every player's five pieces by player and four by possession, and the largest misses of the identities they
    rest on."""
    n, X, w, y, beta, off, dfn = (fit[k] for k in ("n", "X", "w", "y", "beta", "off", "dfn"))
    resid = y - X @ beta
    context_row = X[:, 2 * n:] @ beta[2 * n:]
    z = y - context_row                           # each outcome against what its situation predicts
    sum_o = off[fit["lineup_o"]].sum(axis=1)      # the attacking five's offence ratings, per row
    sum_d = dfn[fit["lineup_d"]].sum(axis=1)      # the defending five's defence ratings, per row
    rows = np.arange(len(w))
    at_o = sp.csc_matrix((np.ones(fit["lineup_o"].size), (np.repeat(rows, 5), fit["lineup_o"].ravel())),
                         shape=(len(w), n))
    at_d = sp.csc_matrix((np.ones(fit["lineup_d"].size), (np.repeat(rows, 5), fit["lineup_d"].ravel())),
                         shape=(len(w), n))
    # his published total is c'beta with c = e_off(i) - e_def(i) minus the zero point's weights; A^-1 c, a row
    # weight h = W X A^-1 c per row, and sum h y = c'beta
    p, keys = X.shape[1], fit["keys"]
    c0 = np.zeros(p)
    c0[:n], c0[n:2 * n] = -fit["w_off"], fit["w_def"]
    A_inv = sla.cho_solve(fit["factor"], np.eye(p))
    u0 = A_inv @ c0
    miss = dict(identity=0.0, ridge=0.0, weights=0.0, sources=0.0)

    def mean(v, r):
        return float(np.average(v[r], weights=w[r]))

    out = []
    for i in np.flatnonzero((np.diff(at_o.indptr) > 0) & (np.diff(at_d.indptr) > 0)):
        ro, rd = at_o.indices[at_o.indptr[i]:at_o.indptr[i + 1]], at_d.indices[at_d.indptr[i]:at_d.indptr[i + 1]]
        rec = dict(i=i, poss=fit["poss"][i],
                   margin=mean(y, ro) - mean(y, rd),
                   teammates=-((mean(sum_o, ro) - off[i]) + (mean(sum_d, rd) - dfn[i])),
                   opponents=mean(sum_d, ro) + mean(sum_o, rd),
                   context=-(mean(context_row, ro) - mean(context_row, rd)),
                   ridge=-(mean(resid, ro) - mean(resid, rd)),
                   rapm=off[i] + dfn[i])
        # by possession: his rows; his team's rows without him in games he played; his teams' rows in games he did
        # not play, in seasons he played for them; the rest
        hz = w * (X @ (u0 + A_inv[:, i] - A_inv[:, n + i])) * z
        on = np.zeros(len(w), dtype=bool)
        on[ro], on[rd] = True, True
        his_ts, his_gt = np.zeros(keys["n_ts"], dtype=bool), np.zeros(keys["n_gt"], dtype=bool)
        his_ts[keys["ts_att"][ro]], his_ts[keys["ts_def"][rd]] = True, True
        his_gt[keys["gt_att"][ro]], his_gt[keys["gt_def"][rd]] = True, True
        team_att, team_def = his_ts[keys["ts_att"]], his_ts[keys["ts_def"]]
        played_att, played_def = his_gt[keys["gt_att"]], his_gt[keys["gt_def"]]
        without = ((team_att & played_att) | (team_def & played_def)) & ~on
        missed = ((team_att & ~played_att) | (team_def & ~played_def)) & ~on & ~without
        rec.update(his_minutes=hz[on].sum(), team_without_him=hz[without].sum(), games_missed=hz[missed].sum(),
                   everything_else=hz[~(on | without | missed)].sum())
        miss["identity"] = max(miss["identity"], abs(sum(rec[k] for k in PIECES) - rec["rapm"]))
        miss["sources"] = max(miss["sources"], abs(sum(rec[k] for k in SOURCES) - rec["rapm"]))
        miss["ridge"] = max(miss["ridge"], abs(mean(resid, ro) - fit["lam"] * beta[i] / fit["diag"][i]),
                            abs(mean(resid, rd) - fit["lam"] * beta[n + i] / fit["diag"][n + i]))
        miss["weights"] = max(miss["weights"], abs(w[ro].sum() - fit["poss"][i]) / fit["poss"][i])
        out.append(rec)
    return pd.DataFrame(out), miss


def name_table() -> pd.DataFrame:
    """player_id, season, name: the rankings table first, then the bio files for the players it lacks."""
    src = next((ROOT / s for s in NAME_SOURCES if (ROOT / s).exists()), None)
    if src is None:
        raise SystemExit(f"no rankings table to take names from: {', '.join(NAME_SOURCES)}")
    t = pd.read_parquet(src, columns=["player_id", "season", "player_name"]).dropna(subset=["player_name"])
    bio = [pd.read_parquet(f, columns=["PLAYER_ID", "PLAYER_NAME"]).assign(season=int(f.stem[:4]))
           for f in sorted((ROOT / "data" / "raw" / "bio").glob("*_RS.parquet"))]
    if bio:
        bio = pd.concat(bio).rename(columns={"PLAYER_ID": "player_id", "PLAYER_NAME": "player_name"}).dropna()
        t = pd.concat([t, bio[~bio.player_id.isin(t.player_id)]], ignore_index=True)
    return t.sort_values("season")


def box_names(ids, seasons) -> pd.Series:
    """The last resort, for a player the rankings table and the bio files both lack (Luca Vildoza played only in
    the 2022 playoffs): his name as the span's box scores print it."""
    want, got = {int(i) for i in ids}, {}
    for season in seasons:
        for f in sorted((ROOT / "data" / "raw" / "box" / str(season)).glob("*.parquet")):
            d = pd.read_parquet(f, columns=["personId", "firstName", "familyName"])
            for r in d[d.personId.isin(want - set(got))].itertuples(index=False):
                got[int(r.personId)] = f"{r.firstName} {r.familyName}".strip()
            if want <= set(got):
                return pd.Series(got, dtype=object)
    return pd.Series(got, dtype=object)


def span_names(names: pd.DataFrame, ids: np.ndarray, seasons) -> np.ndarray:
    """Each player's name as of the span's last season he has one in; any season if the span has none; the box
    scores if no table has him."""
    inside = names[names.season.isin(seasons)].drop_duplicates("player_id", keep="last").set_index("player_id")
    anywhere = names.drop_duplicates("player_id", keep="last").set_index("player_id")
    got = inside.player_name.reindex(ids).fillna(anywhere.player_name.reindex(ids))
    if got.isna().any():
        got = got.fillna(box_names(ids[got.isna().to_numpy()], seasons).reindex(ids))
    return got.to_numpy(object)


def main() -> None:
    check_flags()
    lam = float(flag("penalty", 3000))
    cfg = load_config()
    names = name_table()
    spans = {"1y": ("1 season", [[s] for s in range(int(cfg["first_season"]), int(cfg["last_season"]) + 1)]),
             "3y": ("3 seasons", [list(range(int(a), int(b) + 1)) for a, b in cfg["windows"]])}
    out_dir = ROOT / "docs" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    built = pd.Timestamp.now("UTC").strftime("%Y-%m-%d")
    log: list = []
    for key, (span_words, chunks) in spans.items():
        started, rows = time.time(), {}
        worst = dict(identity=0.0, sources=0.0, ridge=0.0, weights=0.0, centre=0.0, unnamed=0, unplaced=0.0)
        for seasons in chunks:
            fit = span_fit(seasons, cfg, lam)
            tab, miss = decompose(fit)
            worst["unplaced"] = max(worst["unplaced"], fit["keys"]["unplaced"])
            for k in miss:
                worst[k] = max(worst[k], miss[k])
            for k in PIECES:
                worst["centre"] = max(worst["centre"], abs(float(np.average(tab[k], weights=tab.poss))))
            ids = fit["player_ids"][tab.i.to_numpy()]
            tab["name"] = span_names(names, ids, seasons)
            worst["unnamed"] += int(tab["name"].isna().sum())
            tab = tab.sort_values("rapm", ascending=False)
            rows[label(seasons)] = [[r.name, int(round(r.poss)), *(round(float(getattr(r, k)), 2) for k in PIECES),
                                     round(float(r.rapm), 2), *(round(float(getattr(r, k)), 2) for k in SOURCES)]
                                    for r in tab.itertuples(index=False)]
            top = tab.iloc[0]
            print(f"  {label(seasons)}: {len(tab)} players, top {top['name']} {top.rapm:+.2f} "
                  f"({time.time() - started:.0f}s)", flush=True)
        check(worst["identity"] < 1e-9, f"{span_words}: the five pieces add up to RAPM for every player of every "
                                        f"span (max miss {worst['identity']:.1e})", log)
        check(worst["sources"] < 1e-9, f"{span_words}: the four possession groups add up to RAPM for every player "
                                       f"of every span (max miss {worst['sources']:.1e})", log)
        check(worst["unplaced"] < UNPLACED_TOL, f"{span_words}: the box scores name both teams of nearly every row "
                                                f"(largest share without, in one span: {worst['unplaced']:.2%})", log)
        check(worst["ridge"] < 1e-9, f"{span_words}: 'ridge penalty' equals penalty x coefficient / possessions on "
                                     f"each side, the fit's own condition (max miss {worst['ridge']:.1e})", log)
        check(worst["weights"] < 1e-9, f"{span_words}: the fit's weights are his possessions, so 'per 100 "
                                       f"possessions' is literal (max relative miss {worst['weights']:.1e})", log)
        check(worst["centre"] < CENTRE_TOL, f"{span_words}: every piece averages zero over the league, weighted by "
                                            f"possessions (largest {worst['centre']:.3f})", log)
        check(worst["unnamed"] == 0, f"{span_words}: every player has a name ({worst['unnamed']} without)", log)
        periods = sorted(rows, reverse=True)
        doc = dict(meta=dict(span=span_words, penalty=lam, built=built, fields=FIELDS, periods=periods),
                   rows={k: rows[k] for k in periods})
        path = out_dir / f"decompose_{key}.json"
        path.write_text(json.dumps(doc, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        print(f"wrote {path.relative_to(ROOT)}: {sum(len(v) for v in rows.values()):,} rows over {len(periods)} "
              f"periods, {path.stat().st_size / 1e6:.2f} MB ({time.time() - started:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
