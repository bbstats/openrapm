"""The swap test: does a season's ranking order TEAMMATES the way the games do?

Any four of a lineup's five players are one of its CORES.  Two lineups that share a core differ by
exactly one player -- Podziemski in one, Spencer in the other, the same four teammates in both -- and
the difference between their results is the cleanest evidence the play-by-play holds about Podziemski
against Spencer: everything about the team that stays the same cancels.  The owner's idea (2026-10-01):
"every 5-man group that Brandin Podziemski is in, find the times those 4 played with someone else, and
find his delta vs that player, adjusting for game context."

A season's ratings are scored on the swaps of the seasons BEFORE and AFTER it, never on its own: the
rating was fitted on those.  Only pairs where both swapped players carry a rating in the rated season are
scored -- a rookie of the scored season has no rating to be right or wrong about.

**Team strength cannot move it.**  A constant added to every player on a team leaves every one of that
team's swap differences where it was, because both lineups of a pair carry five of its players.  What
moves it is the order, and the gaps, inside a team.

Each lineup's result is read against the ranking's own prediction of it: the ten players' ratings plus a
level refit on the scored season (the year-over-year convention), so the opponents a lineup happened to
face are accounted for by the same ratings being tested.  `context="full"` refits, as well as the
intercept and the home edge, the playoff, garbage-time and score-margin terms the ratings were fitted
with, and two terms they never see:

  * `fatigue_off` / `fatigue_def`: the minutes each five had been on the court without a break when the
    stint began.  A backup usually checks in when the other four have been out there a while, so
    without it a swap compares a fresh lineup with a tired one and calls it a player difference.
  * `clutch`: the last five minutes of the fourth quarter or overtime with the margin within five,
    where the stars close and the game is played differently.
  * `period_elapsed` (minutes into the period) and `period_end` (its last two minutes).  Without them the
    fatigue clocks, which restart every period, mostly measure how late in the period it is -- and late
    in a period teams are in the foul bonus and score more -- and came out as tired fives playing BETTER.

Those are a handful of numbers per scored season, the same for every player.

Two scores per side (offense, defense, and net -- points scored minus allowed), per scored season:

  order  the owner's choice (2026-10-01): a pair the ranking orders wrongly costs the size of the real
         gap.  The real gap is never seen, but each pair's observed swap difference is an unbiased reading
         of it once the opponents faced and the context are taken out, so `order` = the information-
         weighted mean of that adjusted difference, signed by the ranking's call -- and the call is the
         ranking's rating gap between the two swapped players and nothing else.  The adjustment is made
         ONCE per scored season, the same for every ranking, from that season's own plain RAPM (its
         opponents and its level refit), so two rankings that order every swapped pair the same way score
         exactly the same whatever their spreads.  Higher is better; a ranking with no idea scores 0.
         Between two rankings, half the difference in `order` is the difference in that expected
         misordering cost.
  gaps   the information-weighted squared error of the predicted swap difference against the observed
         one, each ranking predicting with its own ratings (opponents included).  Lower is better; it sees
         spread as well as order.

**What "teammates" means across two seasons.**  A pair is two players on the same team in the SCORED
season.  About half of a roster turns over between seasons, so many such pairs were on different teams
in the RATED season, and for them the test is grading the ranking's comparison of two players from two
teams -- whether a player's credit travels with him.  `score_season` reports the two kinds apart when it
is told each player's team in the rated season.

A pair's weight is its information, `h = n_a * n_b / (n_a + n_b)` possessions: a swap seen for 2,000
possessions on one side and 20 on the other knows about as much as 20 possessions.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .priorridge import solve_diag

__all__ = ["HOME", "AWAY", "OFF", "DEF", "SIDES", "CONTEXTS", "RAPM_CONTROLS", "PAIR_GROUPS", "STINT_COLUMNS",
           "order_stints", "fatigue", "season_rows", "swap_pairs", "Season", "prepare", "aligned", "predict",
           "side_scores", "score_season", "plain_rapm"]

HOME = [f"h{i}" for i in range(1, 6)]
AWAY = [f"a{i}" for i in range(1, 6)]
OFF = [f"o{i}" for i in range(1, 6)]
DEF = [f"d{i}" for i in range(1, 6)]
SIDES = ("offense", "defense", "net")

# The level refit on the scored season.  "none" is for the tests: with the intercept alone a lineup's
# prediction is its ratings and nothing else, so a uniform rescale of the ratings rescales every
# predicted gap exactly.
CONTEXTS = {
    "none": ["one"],
    "home": ["one", "home"],
    "full": ["one", "home", "is_po", "po_home", "is_gt", "margin", "margin_frem",
             "period_elapsed", "period_end", "fatigue_off", "fatigue_def", "clutch"],
    # the full context without the time-on-court clocks: on 1998-2025 they came out as "a five that has been on
    # longer plays BETTER" (+1.75 a minute on offense), an artifact and not tiredness (DECISIONS.md, the swap test)
    "nofatigue": ["one", "home", "is_po", "po_home", "is_gt", "margin", "margin_frem",
                  "period_elapsed", "period_end", "clutch"],
}
# what a plain single-season RAPM is fitted with: the design's own fixed block (design.build_design)
RAPM_CONTROLS = ["one", "home", "is_po", "po_home", "is_gt", "margin", "margin_frem"]

CLUTCH_SECONDS = 300.0       # the last five minutes of a period ...
CLUTCH_PERIOD = 4            # ... of the fourth quarter or overtime ...
CLUTCH_MARGIN = 5.0          # ... with the margin within five
FATIGUE_CAP = 15.0           # minutes; past it, a five's clock adds nothing
PERIOD_END_SECONDS = 120.0   # the last two minutes of any period: the foul bonus, two-for-ones, heaves

# what `season_rows` reads from a season's stints, plus `home_team_id` / `away_team_id` from the diagnostics
STINT_COLUMNS = ["game_id", "phase", "period", "start_clock", "frac_rem", "margin_h", "is_gt", "neutral",
                 "poss_h", "poss_a", "pts_h", "pts_a", *HOME, *AWAY]


# ------------------------------------------------------------------------------------------- the rows
def order_stints(stints: pd.DataFrame) -> pd.DataFrame:
    """Each game's stints together, in the order they were stored (time order); stable, so nothing else moves."""
    return stints.iloc[np.argsort(stints["game_id"].to_numpy(), kind="stable")].reset_index(drop=True)


def fatigue(stints: pd.DataFrame, cap: float = FATIGUE_CAP) -> tuple[np.ndarray, np.ndarray]:
    """Minutes each five had been on the court without a break when every stint began: (home, away).

    The stints must be in time order within a game, each game's together.  A player's clock runs while he
    stays on the court from one stint to the next within a period, and goes back to zero when he sits or
    a period starts.  A stint lasts from its own start to the next stint's start in the same period, or
    to the end of the period.  The five's number is the mean of its five clocks, capped at `cap`.

    Read at the START of the stint, never the middle: a stint's own length is partly its own result --
    a coach stops the game when the other team goes on a run -- so a clock that included it measured how
    well the stint went (it came out as tired offenses scoring MORE, +1.8 a minute, on 2024-25).
    """
    n = len(stints)
    game = stints["game_id"].to_numpy()
    period = stints["period"].to_numpy()
    clock = stints["start_clock"].to_numpy(dtype=float)
    same_next = np.zeros(n, dtype=bool)
    if n > 1:
        same_next[:-1] = (game[1:] == game[:-1]) & (period[1:] == period[:-1])
    end = np.zeros(n)
    end[:-1] = np.where(same_next[:-1], clock[1:], 0.0)
    duration = np.clip(clock - end, 0.0, None)

    home = stints[HOME].to_numpy(dtype=np.int64).tolist()
    away = stints[AWAY].to_numpy(dtype=np.int64).tolist()
    out_h, out_a = np.zeros(n), np.zeros(n)
    on: dict = {}
    for i in range(n):
        if i == 0 or not same_next[i - 1]:
            on = {}
        dur = float(duration[i])
        now: dict = {}
        total_h = 0.0
        for p in home[i]:
            before = on.get(p, 0.0)
            total_h += before
            now[p] = before + dur
        total_a = 0.0
        for p in away[i]:
            before = on.get(p, 0.0)
            total_a += before
            now[p] = before + dur
        on = now
        out_h[i], out_a[i] = total_h / 300.0, total_a / 300.0      # five clocks, seconds -> minutes
    return np.minimum(out_h, cap), np.minimum(out_a, cap)


def season_rows(stints: pd.DataFrame, margin_clip: float = 25.0) -> pd.DataFrame:
    """One row per stint and side that had possessions: the two fives, what happened, and the context.

    `stints` is a season's STINT_COLUMNS plus `home_team_id` / `away_team_id` (the script's loader joins
    them from the diagnostics); each game's stints in time order, as they are stored.
    Columns: team_off, team_def, o1..o5 (the offensive five, sorted ids), d1..d5, poss, pts, y (points
    per 100), and every column of CONTEXTS["full"], in the offense's point of view: `home` is +1 when the
    offense is at home, -1 away, 0 at a neutral site; `margin` is the offense's lead at the stint's start.
    """
    stints = order_stints(stints)
    fat_h, fat_a = fatigue(stints)
    neutral = stints["neutral"].fillna(False).to_numpy(dtype=bool)
    is_po = (stints["phase"].to_numpy() == "PO").astype(float)
    is_gt = stints["is_gt"].to_numpy().astype(float)
    margin_h = stints["margin_h"].to_numpy(dtype=float)
    frac = stints["frac_rem"].to_numpy(dtype=float)
    period = stints["period"].to_numpy()
    start = stints["start_clock"].to_numpy(dtype=float)
    clutch = ((period >= CLUTCH_PERIOD) & (start <= CLUTCH_SECONDS) & (np.abs(margin_h) <= CLUTCH_MARGIN)).astype(float)
    elapsed = (np.where(period <= 4, 720.0, 300.0) - start) / 60.0          # quarters are 12 minutes, overtime 5
    period_end = (start <= PERIOD_END_SECONDS).astype(float)
    five_h = np.sort(stints[HOME].to_numpy(dtype=np.int64), axis=1)
    five_a = np.sort(stints[AWAY].to_numpy(dtype=np.int64), axis=1)
    team_h = stints["home_team_id"].to_numpy(dtype=np.int64)
    team_a = stints["away_team_id"].to_numpy(dtype=np.int64)

    sides = []
    for side, sign in (("h", 1.0), ("a", -1.0)):
        off, de = (five_h, five_a) if side == "h" else (five_a, five_h)
        t_off, t_def = (team_h, team_a) if side == "h" else (team_a, team_h)
        f_off, f_def = (fat_h, fat_a) if side == "h" else (fat_a, fat_h)
        poss = stints[f"poss_{side}"].to_numpy(dtype=float)
        keep = poss > 0
        home = np.where(neutral, 0.0, sign)
        margin = np.clip(sign * margin_h, -margin_clip, margin_clip)
        frame = pd.DataFrame({
            "game_id": stints["game_id"].to_numpy()[keep],
            "team_off": t_off[keep], "team_def": t_def[keep],
            "poss": poss[keep], "pts": stints[f"pts_{side}"].to_numpy(dtype=float)[keep],
            "one": 1.0, "home": home[keep], "is_po": is_po[keep], "po_home": (is_po * home)[keep],
            "is_gt": is_gt[keep], "margin": margin[keep], "margin_frem": (margin * frac)[keep],
            "period_elapsed": elapsed[keep], "period_end": period_end[keep],
            "fatigue_off": f_off[keep], "fatigue_def": f_def[keep], "clutch": clutch[keep],
        })
        for k in range(5):
            frame[OFF[k]] = off[keep, k]
            frame[DEF[k]] = de[keep, k]
        sides.append(frame)
    rows = pd.concat(sides, ignore_index=True)
    rows["y"] = 100.0 * rows.pts / rows.poss
    return rows


# ------------------------------------------------------------------------------------------- the pairs
def swap_pairs(team: np.ndarray, five: np.ndarray) -> pd.DataFrame:
    """Every pair of lineups on the same team that share exactly four players: (team, lineup_a, lineup_b,
    fifth_a, fifth_b), lineup_a < lineup_b, each pair once.

    `five` is (lineups x 5) sorted player ids.  Two different lineups that share four players share
    exactly one core -- those four -- so meeting on a core finds every such pair once and nothing else.
    """
    n = len(team)
    frames = []
    for k in range(5):
        core = np.delete(five, k, axis=1)
        frames.append(pd.DataFrame({"lineup": np.arange(n), "team": team, "c1": core[:, 0], "c2": core[:, 1],
                                    "c3": core[:, 2], "c4": core[:, 3], "fifth": five[:, k]}))
    f = pd.concat(frames, ignore_index=True)
    keys = ["team", "c1", "c2", "c3", "c4"]
    f = f[f.duplicated(keys, keep=False)]                      # a core seen with one fifth makes no pair
    m = f.merge(f, on=keys, suffixes=("_a", "_b"))
    m = m[m.lineup_a < m.lineup_b]
    return m[["team", "lineup_a", "lineup_b", "fifth_a", "fifth_b"]].reset_index(drop=True)


@dataclass
class Season:
    """One scored season, made ready to score any number of rankings against.

    rows      season_rows(...)
    ids       every player id in the rows, sorted
    off_idx   (rows x 5) positions of the offensive five in `ids`; def_idx the same for the defense
    lineup    side -> (rows,) the lineup number of that side's five, into `tables[side]`
    tables    "offense" / "defense": team, p1..p5, poss, pts (scored by the five on offense, allowed on
              defense); "net": the lineups seen on both ends, with their row in each of the other two
    pairs     side -> swap_pairs(...) on that side's table
    """
    season: int
    rows: pd.DataFrame
    ids: np.ndarray
    off_idx: np.ndarray
    def_idx: np.ndarray
    lineup: dict
    tables: dict
    pairs: dict


def _lineup_table(team: np.ndarray, five: np.ndarray, poss: np.ndarray, pts: np.ndarray):
    key = np.column_stack([team, five])
    uniq, inverse = np.unique(key, axis=0, return_inverse=True)
    inverse = np.asarray(inverse).ravel()
    table = pd.DataFrame(uniq[:, 1:], columns=[f"p{k}" for k in range(1, 6)])
    table.insert(0, "team", uniq[:, 0])
    table["poss"] = np.bincount(inverse, weights=poss, minlength=len(uniq))
    table["pts"] = np.bincount(inverse, weights=pts, minlength=len(uniq))
    return table, inverse


def prepare(rows: pd.DataFrame, season: int = 0) -> Season:
    off = rows[OFF].to_numpy(dtype=np.int64)
    de = rows[DEF].to_numpy(dtype=np.int64)
    ids = np.unique(np.concatenate([off.ravel(), de.ravel()]))
    poss, pts = rows.poss.to_numpy(dtype=float), rows.pts.to_numpy(dtype=float)
    t_off, lu_off = _lineup_table(rows.team_off.to_numpy(dtype=np.int64), off, poss, pts)
    t_def, lu_def = _lineup_table(rows.team_def.to_numpy(dtype=np.int64), de, poss, pts)
    keys = ["team", "p1", "p2", "p3", "p4", "p5"]
    t_net = (t_off[keys].assign(row_off=np.arange(len(t_off)))
             .merge(t_def[keys].assign(row_def=np.arange(len(t_def))), on=keys, how="inner"))
    t_net["poss"] = t_off.poss.to_numpy()[t_net.row_off] + t_def.poss.to_numpy()[t_net.row_def]
    tables = {"offense": t_off, "defense": t_def, "net": t_net}
    pairs = {side: swap_pairs(t.team.to_numpy(dtype=np.int64), t[keys[1:]].to_numpy(dtype=np.int64))
             for side, t in tables.items()}
    return Season(season=int(season), rows=rows, ids=ids, off_idx=np.searchsorted(ids, off),
                  def_idx=np.searchsorted(ids, de), lineup={"offense": lu_off, "defense": lu_def},
                  tables=tables, pairs=pairs)


# ------------------------------------------------------------------------------------------- scoring
def aligned(ids: np.ndarray, ratings: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(o, d) in raw sign for every id in `ids`; a player the table has no row for is the average player, 0."""
    r = pd.DataFrame({"player_id": np.asarray(ids, dtype=np.int64)}).merge(
        ratings[["player_id", "o", "d"]], on="player_id", how="left")
    return r.o.fillna(0.0).to_numpy(dtype=float), r.d.fillna(0.0).to_numpy(dtype=float)


def _wls(A, y, w):
    AtW = (A * w[:, None]).T
    return np.linalg.lstsq(AtW @ A, AtW @ y, rcond=None)[0]


def predict(season: Season, o: np.ndarray, d: np.ndarray, context: str = "full") -> tuple[np.ndarray, np.ndarray]:
    """Every row's predicted points per 100 -- the ten players' ratings plus the level refit on the scored
    season with the ratings held as an offset -- and the level's coefficients, in CONTEXTS[context] order."""
    contrib = o[season.off_idx].sum(axis=1) + d[season.def_idx].sum(axis=1)
    A = season.rows[CONTEXTS[context]].to_numpy(dtype=float)
    w = season.rows.poss.to_numpy(dtype=float)
    gamma = _wls(A, season.rows.y.to_numpy(dtype=float) - contrib, w)
    return A @ gamma + contrib, gamma


def _side_rates(season: Season, pred: np.ndarray, o: np.ndarray, d: np.ndarray) -> dict:
    """side -> (observed rate, predicted rate, possessions, the five's own ratings) per lineup of that side.

    The last is the part of the prediction that is the lineup's own players -- the sum of the five's
    offensive ratings on offense, of their defensive ratings (points allowed) on defense, and offense minus
    defense for net -- so a pair's difference in it is exactly the ranking's gap between the two swapped
    players: the other four are in both lineups and cancel."""
    w = season.rows.poss.to_numpy(dtype=float)
    out = {}
    for side, own in (("offense", o), ("defense", d)):
        t = season.tables[side]
        P = t.poss.to_numpy(dtype=float)
        pred_rate = np.bincount(season.lineup[side], weights=pred * w, minlength=len(t)) / P
        five = np.searchsorted(season.ids, t[["p1", "p2", "p3", "p4", "p5"]].to_numpy(dtype=np.int64))
        out[side] = (100.0 * t.pts.to_numpy(dtype=float) / P, pred_rate, P, own[five].sum(axis=1))
    net = season.tables["net"]
    o_obs, o_pred, _, o_own = out["offense"]
    d_obs, d_pred, _, d_own = out["defense"]
    ro, rd = net.row_off.to_numpy(), net.row_def.to_numpy()
    out["net"] = (o_obs[ro] - d_obs[rd], o_pred[ro] - d_pred[rd], net.poss.to_numpy(dtype=float),
                  o_own[ro] - d_own[rd])
    return out


def side_scores(adjusted: np.ndarray, call: np.ndarray, observed: np.ndarray, predicted: np.ndarray,
                h: np.ndarray) -> dict:
    """The three numbers for one ranking on one set of pairs.

    adjusted   the observed swap difference with the opponents and context taken out (the same for every
               ranking); `call` the ranking's rating gap between the two swapped players; `observed` /
               `predicted` the raw swap difference and the ranking's whole prediction of it; `h` the weights.

    order  sum(h * adjusted * sign(call)) / sum(h): higher is better, 0 is no idea, spread-free.
    gaps   sum(h * (observed - predicted)^2) / sum(h): lower is better.
    slope  sum(h * adjusted * call) / sum(h * call^2): the multiplier the swaps ask the ranking's gaps
           inside a team for -- below 1, too wide.
    """
    hs = float(h.sum())
    cc = float(np.sum(h * call * call))
    return {"order": float(np.sum(h * adjusted * np.sign(call)) / hs),
            "gaps": float(np.sum(h * (observed - predicted) ** 2) / hs),
            "slope": float(np.sum(h * adjusted * call) / cc) if cc > 0 else np.nan}


PAIR_GROUPS = ("all", "teammates then", "not teammates then")


def score_season(season: Season, rankings: dict, rated: set, common: pd.DataFrame, contexts=("full",),
                 team_then: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score every ranking on one scored season's swaps.

    rankings   name -> a table (player_id, o, d) for the rated season, raw sign
    rated      the players every ranking carries.  A pair is scored only if both swapped players are in
               it, so every ranking is scored on exactly the same pairs.
    common     the scored season's own ratings (player_id, o, d) -- its plain RAPM -- used ONCE to take the
               opponents faced and the context out of each observed swap difference before `order` reads
               it, identically for every ranking
    team_then  player_id -> his team in the RATED season; with it each score is also reported for pairs who
               were teammates then and pairs who were not (PAIR_GROUPS)

    Returns (scores, levels): one row per ranking x context x side x pair group, and the level
    coefficients each ranking's refit chose.
    """
    rated_arr = np.fromiter(rated, dtype=np.int64, count=len(rated))
    pairs, group_of = {}, {}
    for side, pr in season.pairs.items():
        pr = pr[np.isin(pr.fifth_a.to_numpy(), rated_arr) & np.isin(pr.fifth_b.to_numpy(), rated_arr)]
        pairs[side] = pr
        if team_then is not None:
            ta = pr.fifth_a.map(team_then).to_numpy()
            tb = pr.fifth_b.map(team_then).to_numpy()
            same = pd.notna(ta) & pd.notna(tb) & (ta == tb)
            group_of[side] = np.where(same, PAIR_GROUPS[1], PAIR_GROUPS[2])

    def pair_arrays(side, rates):
        obs, prd, P, own = rates[side]
        a, b = pairs[side].lineup_a.to_numpy(), pairs[side].lineup_b.to_numpy()
        return obs[a] - obs[b], prd[a] - prd[b], own[a] - own[b], P[a] * P[b] / (P[a] + P[b])

    co, cd = aligned(season.ids, common)
    adjusted = {}
    for context in contexts:
        pred, _ = predict(season, co, cd, context)
        rates = _side_rates(season, pred, co, cd)
        for side in SIDES:
            observed, predicted, own, _ = pair_arrays(side, rates)
            adjusted[(context, side)] = observed - (predicted - own)      # what is left once all but the swap is out

    rows, levels = [], []
    for name, table in rankings.items():
        o, d = aligned(season.ids, table)
        for context in contexts:
            pred, gamma = predict(season, o, d, context)
            levels.append(dict(ranking=name, context=context, **dict(zip(CONTEXTS[context], gamma))))
            rates = _side_rates(season, pred, o, d)
            for side in SIDES:
                observed, predicted, call, h = pair_arrays(side, rates)
                adj = adjusted[(context, side)]
                groups = [("all", np.ones(len(h), dtype=bool))]
                if side in group_of:
                    groups += [(g, group_of[side] == g) for g in PAIR_GROUPS[1:]]
                for g, m in groups:
                    if not m.any():
                        continue
                    rows.append(dict(ranking=name, context=context, side=side, group=g, pairs=int(m.sum()),
                                     information=float(h[m].sum()),
                                     **side_scores(adj[m], call[m], observed[m], predicted[m], h[m])))
    return pd.DataFrame(rows), pd.DataFrame(levels)


# ------------------------------------------------------------------------------------------- a reference
def plain_rapm(season: Season, lam: float) -> pd.DataFrame:
    """Plain single-season RAPM on actual points: one penalty on every player column, nothing else pulled.

    Raw sign (`d` = points allowed), on the design's own controls (RAPM_CONTROLS), weighted by possessions.
    `poss` is each player's offensive possessions.
    """
    rows = season.rows
    n, m = len(rows), len(season.ids)
    r5 = np.repeat(np.arange(n), 5)
    Zo = sp.csr_matrix((np.ones(5 * n), (r5, season.off_idx.ravel())), shape=(n, m))
    Zd = sp.csr_matrix((np.ones(5 * n), (r5, season.def_idx.ravel())), shape=(n, m))
    F = rows[RAPM_CONTROLS].to_numpy(dtype=float)
    X = sp.hstack([Zo, Zd, sp.csr_matrix(F)], format="csr")
    w = rows.poss.to_numpy(dtype=float)
    gram = np.asarray((X.T @ (sp.diags(w) @ X)).todense(), dtype=float)
    rhs = np.asarray(X.T @ (w * rows.y.to_numpy(dtype=float))).ravel()
    penalty = np.concatenate([np.full(2 * m, float(lam)), np.zeros(F.shape[1])])
    sol = solve_diag(gram, rhs, penalty)
    return pd.DataFrame({"player_id": season.ids, "o": sol[:m], "d": sol[m:2 * m],
                         "poss": np.asarray(Zo.T @ w).ravel()})
