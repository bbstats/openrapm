"""A synthetic league: does RAPM overrate players whose replacements are truly bad?

    python scripts/78_synthetic_league.py [--leagues=20] [--seed=0]

**Why this exists.**  Script 77 kept the real 2024-26 lineups and asked what RAPM does with them.  This one
builds the whole league -- the owner's spec: random players with known true ratings, better players played
more at the real rate plus randomness, simulated substitutions, possession-level noise -- then fits plain RAPM
and asks whether truly bad replacements inflate a player.  Because the league is built, the way substitutions
happen can be varied, which real lineups cannot do:

  random          each stint's five drawn at random, better players more often (the owner's first spec)
  slots           every player has a position for the season; a backup only ever replaces the players at his
                  position, so a starter and his backup never share the court (the textbook case)
  slots + lines   the same, and the five tend to change together (starters with starters, bench units)
  mixed           half the stints by position, half at random, with some line structure

**What is calibrated to 2024-26** (data/cache/roles_RSPO.parquet, the OpenRAPM ratings, the real designs):
team talent (each team starts as a real team-season's top 16, true ratings from OpenRAPM plus jitter), minutes
by rank on the team, how strongly minutes follow quality (coaches rank players by true rating plus a
perception error chosen to match the real within-team rank correlation), games missed, roster churn between
seasons, stints per game, possessions per stint and the noise of one possession.

**How to read it.**  Every test makes some players one point worse (half at each end) with nothing else
changed, and reads how far a player's expected RAPM moves.  RAPM is linear, so this is exact and needs no
truth: it comes from the fit and the rotations alone.  Positive = bad replacements make him look better (the
belief).  Points per 100 possessions.
  share kept        the lineup that replaces him one point worse: each player who takes his minutes made worse in
                    proportion to how much of them he takes, scaled so his raw on/off rises by exactly 1.  RAPM's
                    move is the share of the replacement distortion it keeps: 0 = removes it, 1 = no better than
                    raw on/off
  team bench        every player outside the team's top five one point worse; the starters' average move
  own backups       (position leagues) the other players at his position one point worse; the starter's move
Medians over players with 10,000+ possessions (share kept) or over starters; brackets are the range over
leagues.  "real lineups" runs the same tests on the actual 2024-26 rotations.  The run stops on a failed check.

**What it cannot tell you.**  The league follows RAPM's own model: player values that add up, plus noise.

Writes outputs/synthetic_league_players.parquet, outputs/synthetic_league_groups.parquet,
outputs/synthetic_league_summary.parquet and outputs/synthetic_league_phone.html.
"""
import html
import importlib.util
import os
import sys
import time
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
os.environ.setdefault("NUMBA_NUM_THREADS", os.environ.get("OPENRAPM_NUMBA_THREADS", "4"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.sparse as sp  # noqa: E402
from scipy.special import ndtr  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _cli import check_flags, flag  # noqa: E402
from eracoef.config import load_config  # noqa: E402

# 77's block, fit, on/off operators and replacements -- one definition of each, as 70 borrows 63's
_spec = importlib.util.spec_from_file_location("_borrowed_77", ROOT / "scripts" / "77_replacement_sim.py")
r77 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(r77)
check = r77.check

REAL_SEASONS = (2024, 2025, 2026)
SEASONS, TEAMS, ROSTER, ROUNDS = 3, 30, 16, 41       # 41 rounds of home-and-away pairings = 82 games
STAY, MOVE = 0.50, 0.29        # of last season's players: same team / another team; the rest leave (real 2025-26)
LEVEL, HOME = 114.0, 1.5       # league points per 100 and the home edge; neither reaches a player's rating
SHARE_NOISE = 0.20             # game-to-game spread of a player's minutes, log scale
CAP = 0.95                     # nobody plays more than 95% of a game
MIN_AVAILABLE = 9
JITTER = 0.25                  # a synthetic player is a real rating plus this much noise per side
PERCEPTION_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0)
PENALTIES = (1e3, 1e4, 4e4, 1.6e5)
HEADLINE = 4e4
SETTINGS = {"random": (0.0, 0.0), "slots": (1.0, 0.0), "slots + lines": (1.0, 0.6), "mixed": (0.5, 0.3)}
SHARED_CUTS = (0.10, 0.30)
HEAVY = 10000.0
TABLE_MIN_POSS = 3000.0
# the league's levers, at the 2024-26 calibration; scripts/79_stack_the_deck.py draws them at random
DEFAULT_PARAMS = dict(seasons=SEASONS, stay=STAY, move=MOVE, injury=1.0, share_noise=SHARE_NOISE, starter_boost=1.0)


# ------------------------------------------------------------------------------------------ real targets
def real_targets() -> dict:
    """Minutes by rank, availability, how strongly minutes follow quality, team strength, rosters' talent, and
    each real team-season's starters and bench (for the team-bench test on the real lineups)."""
    roles = pd.read_parquet(ROOT / "data" / "cache" / "roles_RSPO.parquet")
    r = roles[roles.season.isin(REAL_SEASONS)].copy()
    r["share"] = r.poss_on / r.team_poss
    r["rank"] = r.groupby(["season", "team_id"]).share.rank(ascending=False, method="first").astype(int)
    template = r.groupby("rank").share.mean().reindex(range(1, ROSTER + 1)).to_numpy(float)
    team_games = r.groupby(["season", "team_id"]).games.transform("max")
    avail = (r.games / team_games)[r["rank"] <= 12].clip(0.02, 0.98)
    mean, var = float(avail.mean()), float(avail.var())
    k = mean * (1.0 - mean) / var - 1.0                               # Beta by the method of moments
    prod = pd.read_parquet(ROOT / "outputs" / "season_ratings_product.parquet")
    prod = prod[prod.season.isin(REAL_SEASONS)][["player_id", "season", "rating_off", "rating_def"]]
    m = r.merge(prod, on=["player_id", "season"], how="inner")
    m["total"] = m.rating_off + m.rating_def
    top = m[m["rank"] <= 12]
    rho = top.groupby(["season", "team_id"])[["total", "share"]].apply(
        lambda d: d.total.corr(d.share, method="spearman")).mean()
    strength = m.groupby(["season", "team_id"])[["share", "total"]].apply(lambda d: float((d.share * d.total).sum()))
    profiles = [np.column_stack([d.rating_off.to_numpy(float), -d.rating_def.to_numpy(float)])
                for _, d in m[m["rank"] <= ROSTER].sort_values("rank").groupby(["season", "team_id"])]
    teams = [(d.loc[d["rank"] <= 5, "player_id"].to_numpy(), d.loc[(d["rank"] > 5) & (d["rank"] <= ROSTER),
                                                                     "player_id"].to_numpy())
             for _, d in r.groupby(["season", "team_id"])]
    return dict(template=template, alpha=mean * k, beta=(1.0 - mean) * k, avail_mean=mean, spearman12=float(rho),
                strength_sd=float(strength.std()), profiles=profiles, teams=teams)


# ------------------------------------------------------------------------------------------ the tests
def replacement_split(ops: dict, rep: dict, shared: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Share of all replacement time (e+ x his off-court possessions) by how much court time the replacement
    shares with him: under 10%, 10-30%, 30%+.  The realism statistic for the substitution structure."""
    lo, hi = SHARED_CUTS
    out = []
    for side in ("O", "D"):
        wgt = np.where(rep[side]["R"], rep[side]["e"], 0.0) * ops[side]["w_off"][:, None]
        wgt[~valid] = 0.0
        total = wgt.sum()
        out.append([wgt[shared < lo].sum() / total, wgt[(shared >= lo) & (shared < hi)].sum() / total,
                    wgt[shared >= hi].sum() / total])
    return np.mean(out, axis=0)


def share_kept(rows_o: np.ndarray, rows_d: np.ndarray, rep: dict, n: int) -> np.ndarray:
    """Per player: how far his rating moves when the lineup that replaces him gets one point worse.

    Each player who takes his minutes is made worse in proportion to how much of them he takes (e+), scaled by
    the sum of e+ squared so that his raw on/off rises by exactly 1 (half at each end).  `rows_o` / `rows_d` are
    his offence / defence (points allowed) rows of a linear map: the fit's expectation, or the on/off operators
    (which must then give 1 -- a check).  A teammate who is on court only a hair more when he sits moves by a
    hair, so chance co-players cannot pose as replacements.
    """
    weights = {}
    for side in ("O", "D"):
        e = np.where(rep[side]["R"], rep[side]["e"], 0.0)
        norm = (e ** 2).sum(1)
        weights[side] = np.divide(e, norm[:, None], out=np.zeros_like(e), where=norm[:, None] > 0)
    # the change: offence of his offensive replacements -0.5 x weight, points allowed of his defensive ones +0.5 x
    off = -0.5 * (rows_o[:, :n] * weights["O"]).sum(1) + 0.5 * (rows_o[:, n:2 * n] * weights["D"]).sum(1)
    dfn = -0.5 * (rows_d[:, :n] * weights["O"]).sum(1) + 0.5 * (rows_d[:, n:2 * n] * weights["D"]).sum(1)
    return off - dfn


def group_response(M: np.ndarray, ops: dict, n: int, groups: list) -> tuple:
    """`groups`: (who, worse) pairs.  Make every player in `worse` one point worse (half each end) and read how
    far each player in `who` moves in expected RAPM and in raw on/off.  Positive = he rose."""
    change = np.zeros((2 * n, len(groups)))
    for c, (_, worse) in enumerate(groups):
        change[worse, c] = -0.5
        change[n + worse, c] = 0.5
    rapm = M[:2 * n, :2 * n] @ change
    on_o = ops["O"]["D"][:, :2 * n] @ change
    on_d = ops["D"]["D"][:, :2 * n] @ change
    return ([rapm[who, c] - rapm[n + who, c] for c, (who, _) in enumerate(groups)],
            [on_o[who, c] - on_d[who, c] for c, (who, _) in enumerate(groups)])


def structure(block, log, label: str, strict: bool):
    """77's on/off operators and replacements, with its team-structure checks."""
    n = block.n
    ops = r77.onoff_operators(block)
    rep = r77.replacement_structure(ops, n)
    valid = rep["O"]["ok"] & rep["D"]["ok"] & rep["O"]["R"].any(1) & rep["D"]["R"].any(1)
    if strict:
        check(ops["both_teams"] == 0, f"{label}: nobody is on both teams in one game", log)
        worst = max(max(np.abs(rep[s]["own"][rep[s]["ok"]] - 1.0).max(),
                        np.abs(rep[s]["sums"][rep[s]["ok"]] - 1.0).max()) for s in ("O", "D"))
        check(worst < 1e-9, f"{label}: own on/off weight 1 and replacement weights summing to 1 "
                            f"(max miss {worst:.1e})", log)
        unit = np.median(share_kept(ops["O"]["D"], ops["D"]["D"], rep, n)[valid])
        check(abs(unit - 1.0) < 0.02, f"{label}: the replacement lineup one point worse moves raw on/off by "
                                      f"{unit:.3f} (must be 1)", log)
    return ops, rep, r77.shared_court_time(block), valid


def measure(block, t, ops, rep, valid, groups: dict, realized: bool) -> tuple:
    """Per player and penalty: truth, expected and fitted RAPM, the share kept; per group test and penalty: each
    target's move in RAPM and raw on/off."""
    n = block.n
    frames, group_rows = [], []
    for lam in PENALTIES:
        _, beta, M = r77.fit(block, lam, lam)
        expected = M[:2 * n, :2 * n] @ t
        frames.append(pd.DataFrame({
            "player": np.arange(n), "penalty": lam, "poss": block.poss, "valid": valid,
            "truth": t[:n] - t[n:2 * n], "expected": expected[:n] - expected[n:2 * n],
            "fit": (beta[:n] - beta[n:2 * n]) if realized else np.nan,
            "keep": 0.5 * (np.diagonal(M)[:n] + np.diagonal(M)[n:2 * n]),
            "share_kept": share_kept(M[:n], M[n:2 * n], rep, n)}))
        for test, pairs in groups.items():
            if not pairs:
                continue
            rapm, onoff = group_response(M, ops, n, pairs)
            for (who, _), r, o in zip(pairs, rapm, onoff):
                group_rows.append(pd.DataFrame({"test": test, "penalty": lam, "player": who, "rapm": r, "onoff": o,
                                                "poss": block.poss[who]}))
    return pd.concat(frames, ignore_index=True), pd.concat(group_rows, ignore_index=True)


def real_reference(cfg, targets, log) -> dict:
    """77's machinery on the actual 2024-26 rotations: the sampling inputs for the synthetic stints and the
    'real lineups' row every setting is compared with."""
    block = r77.load_block(cfg)
    home = (block.tg_off % 2) == 1
    per_game = np.bincount(block.tg_off[home] // 2)
    _, beta, M = r77.fit(block, HEADLINE, HEADLINE)
    rss = block.ywy - 2.0 * beta @ block.rhs + beta @ block.gram @ beta
    sigma_pp = float(np.sqrt(rss / (block.n_rows - np.trace(M))) / 100.0)
    ops, rep, shared, valid = structure(block, log, "real lineups", strict=True)
    slot = {int(pid): i for i, pid in enumerate(block.player_ids)}
    pairs = []
    for starters, bench in targets["teams"]:
        who = np.array([slot[int(p)] for p in starters if int(p) in slot], dtype=np.int64)
        worse = np.array([slot[int(p)] for p in bench if int(p) in slot], dtype=np.int64)
        if who.size and worse.size:
            pairs.append((who, worse))
    t_raw, _ = r77.openrapm_truth(block)
    players, group_frame = measure(block, r77.centre(t_raw, block), ops, rep, valid, {"team bench": pairs},
                                   realized=False)
    return dict(poss_home=block.w[home], stints_per_game=per_game[per_game > 0], sigma_pp=sigma_pp,
                split=replacement_split(ops, rep, shared, valid), players=players, groups=group_frame)


# ------------------------------------------------------------------------------------------ the minutes model
def cap_normalize(s: np.ndarray, total: float = 5.0, cap: float = CAP) -> np.ndarray:
    """Scale to `total`, then pour anything above `cap` back into the others in proportion."""
    s = np.asarray(s, dtype=float) * total / s.sum()
    fixed = np.zeros(s.size, dtype=bool)
    for _ in range(50):
        over = (s > cap) & ~fixed
        if not over.any():
            break
        excess = float((s[over] - cap).sum())
        s[over] = cap
        fixed |= over
        free = ~fixed & (s > 0)
        s[free] += excess * s[free] / s[free].sum()
    return s


def ensure_available(available: np.ndarray, perceived: np.ndarray) -> np.ndarray:
    """At least MIN_AVAILABLE dressed: the best-rated absentees come back first."""
    if available.sum() >= MIN_AVAILABLE:
        return available
    out = available.copy()
    for i in np.argsort(-perceived):
        out[i] = True
        if out.sum() >= MIN_AVAILABLE:
            break
    return out


def game_shares(perceived, available, template, rng, noise: float = SHARE_NOISE):
    """One team-game: the dressed players in the coach's order, and each one's share of the game (sum 5)."""
    idx = np.flatnonzero(available)
    order = idx[np.argsort(-perceived[idx], kind="stable")]
    base = template[:order.size] if order.size <= template.size else \
        np.concatenate([template, template[-1] * 0.5 ** np.arange(1, order.size - template.size + 1)])
    return order, cap_normalize(base * np.exp(rng.normal(0.0, noise, order.size)))


def minutes_model(targets, template, sigma, rng, n_teams: int = 180):
    """Minutes only, no stints: season share by rank on the team, and the within-team rank correlation of true
    rating with minutes over each team's top 12."""
    by_rank, rhos = [], []
    games = 2 * ROUNDS
    for _ in range(n_teams):
        prof = targets["profiles"][rng.integers(len(targets["profiles"]))]
        total = prof[:, 0] - prof[:, 1] + rng.normal(0.0, JITTER * np.sqrt(2.0), len(prof))
        perceived = total + rng.normal(0.0, sigma, total.size)
        avail = rng.beta(targets["alpha"], targets["beta"], total.size)
        season = np.zeros(total.size)
        for _ in range(games):
            order, shares = game_shares(perceived, ensure_available(rng.random(total.size) < avail, perceived),
                                        template, rng)
            season[order] += shares
        season /= games
        order = np.argsort(-season)
        by_rank.append(np.pad(season[order], (0, max(0, ROSTER - season.size)))[:ROSTER])
        top = order[:12]
        rhos.append(pd.Series(total[top]).corr(pd.Series(season[top]), method="spearman"))
    return np.mean(by_rank, axis=0), float(np.nanmean(rhos))


def calibrate(targets, rng, log) -> dict:
    """The per-game minutes template and the coach's perception error, so that season minutes by rank and the
    within-team rank correlation of quality with minutes both match 2024-26."""
    real = targets["template"]
    best = None
    for sigma in PERCEPTION_GRID:
        template = cap_normalize(real / targets["avail_mean"])
        for _ in range(6):
            sim, _ = minutes_model(targets, template, sigma, rng)
            template = cap_normalize(template * real / np.maximum(sim, 1e-4))
        sim, rho = minutes_model(targets, template, sigma, rng)
        miss = abs(rho - targets["spearman12"])
        print(f"  perception error {sigma:.1f}: rank correlation {rho:.3f} (real {targets['spearman12']:.3f}), "
              f"minutes by rank off by at most {np.abs(sim - real).max():.3f}", flush=True)
        if best is None or miss < best["miss"]:
            best = dict(sigma=sigma, template=template, rho=rho, by_rank=sim, miss=miss)
    check(np.abs(best["by_rank"] - real).max() < 0.03,
          f"synthetic minutes by rank match 2024-26 within 0.03 of a game (max miss "
          f"{np.abs(best['by_rank'] - real).max():.3f})", log)
    check(best["miss"] < 0.05, f"minutes follow quality as in 2024-26: rank correlation {best['rho']:.3f} against "
                               f"{targets['spearman12']:.3f} (perception error {best['sigma']:.1f})", log)
    return best


# ------------------------------------------------------------------------------------------ the league
def assign_slots(shares: np.ndarray) -> np.ndarray:
    """Five positions: the five biggest shares start one each, and every next player backs up the position
    with the fewest minutes so far -- the sixth man backs up the fifth starter.  Returns each player's slot."""
    loads = np.zeros(5)
    slot_of = np.empty(shares.size, dtype=np.int64)
    for i in np.argsort(-shares, kind="stable"):
        k = int(np.argmin(loads))
        slot_of[i] = k
        loads[k] += shares[i]
    return slot_of


def fill_slots(slot_of: np.ndarray, shares: np.ndarray) -> np.ndarray:
    """A position with nobody dressed borrows the least-used player of the position with the most dressed."""
    slot_of = slot_of.copy()
    for _ in range(5):
        counts = np.bincount(slot_of, minlength=5)
        empty = np.flatnonzero(counts == 0)
        if not empty.size:
            break
        members = np.flatnonzero(slot_of == int(np.argmax(counts)))
        slot_of[members[np.argmin(shares[members])]] = empty[0]
    return slot_of


def draw_lineups(shares, slot_of, n_stints, slot_p, lines, rng) -> np.ndarray:
    """`n_stints` five-man lineups from one team's dressed players (indices into `shares`).

    At random: systematic sampling with inclusion probability = his share, so he plays his minutes and anyone
    can play with anyone.  By position: each position picks one of its players in proportion to their shares,
    the five positions' draws correlated by `lines`: toward 1, starters and bench units change as whole lines;
    below 0 (down to -1/4), staggered -- when one position goes to its bench the others tend to keep their
    starters, so a backup plays with the starter's own teammates.
    """
    m = shares.size
    out = np.empty((n_stints, 5), dtype=np.int64)
    by_slot = rng.random(n_stints) < slot_p
    free = np.flatnonzero(~by_slot)
    if free.size:
        perm = np.argsort(rng.random((free.size, m)), axis=1)
        cum = np.cumsum(shares[perm], axis=1)
        points = rng.random(free.size)[:, None] + np.arange(5)[None, :]
        pos = np.minimum((cum[:, None, :] <= points[:, :, None]).sum(-1), m - 1)
        out[free] = np.take_along_axis(perm, pos, axis=1)
    slotted = np.flatnonzero(by_slot)
    if slotted.size:
        if lines >= 0:
            common = rng.standard_normal((slotted.size, 1))
            u = ndtr(np.sqrt(lines) * common + np.sqrt(1.0 - lines) * rng.standard_normal((slotted.size, 5)))
        else:
            # equicorrelated at `lines` < 0: take a share `a` of the five draws' mean out of each, where
            # q = a^2 - 2a = 5 lines / (1 - lines) makes the correlation q / (5 + q) = lines (a = 1 at -1/4)
            q = 5.0 * lines / (1.0 - lines)
            a = 1.0 - np.sqrt(1.0 + q)
            eps = rng.standard_normal((slotted.size, 5))
            u = ndtr((eps - a * eps.mean(axis=1, keepdims=True)) / np.sqrt(1.0 + q / 5.0))
        for k in range(5):
            members = np.flatnonzero(slot_of == k)
            members = members[np.argsort(-shares[members], kind="stable")]
            q = np.cumsum(shares[members])
            q = q / q[-1]
            out[slotted, k] = members[np.minimum(np.searchsorted(q, u[:, k], side="right"), members.size - 1)]
    return out


def schedule(rng) -> list:
    games = []
    for _ in range(ROUNDS):
        perm = rng.permutation(TEAMS)
        for a, b in zip(perm[0::2], perm[1::2]):
            games += [(int(a), int(b)), (int(b), int(a))]
    return games


def churn(rosters, new_player, rng, stay: float = STAY, move: float = MOVE) -> list:
    movers, kept = [], []
    for ids in rosters:
        u = rng.random(len(ids))
        kept.append([pid for pid, x in zip(ids, u) if x < stay])
        movers += [pid for pid, x in zip(ids, u) if stay <= x < stay + move]
    rng.shuffle(movers)
    for ids in kept:
        while len(ids) < ROSTER:
            ids.append(movers.pop() if movers else new_player())
    return kept


def make_league(targets, real, cal, setting, rng, params=None):
    """A 30-team league over `params["seasons"]` seasons (DEFAULT_PARAMS otherwise).  Returns the truth (players x
    [offence, points allowed]), each season's rows, each season's realism numbers, and each team-season's
    roster, minutes and positions.  At the defaults it draws exactly what it drew before the levers existed."""
    p = dict(DEFAULT_PARAMS, **(params or {}))
    slot_p, lines = setting
    pool = np.vstack(targets["profiles"])
    truth = []

    def new_player(o=None, d=None):
        if o is None:
            o, d = pool[rng.integers(len(pool))]
        truth.append((o + rng.normal(0.0, JITTER), d + rng.normal(0.0, JITTER)))
        return len(truth) - 1

    rosters = [[new_player(o, d) for o, d in targets["profiles"][rng.integers(len(targets["profiles"]))]]
               for _ in range(TEAMS)]
    template = cal["template"]
    if p["starter_boost"] != 1.0:                        # starters' minutes scaled, the bench takes what is left
        template = template.copy()
        template[:5] *= p["starter_boost"]
        template = cap_normalize(template)
    seasons, stats, team_seasons = [], [], []
    for season in range(p["seasons"]):
        if season:
            rosters = churn(rosters, new_player, rng, p["stay"], p["move"])
        T = np.asarray(truth)
        total = T[:, 0] - T[:, 1]
        perceived = total + rng.normal(0.0, cal["sigma"], len(T))
        avail = rng.beta(targets["alpha"], targets["beta"], len(T))
        if p["injury"] != 1.0:                           # games missed scaled toward none
            avail = 1.0 - p["injury"] * (1.0 - avail)
        position = np.full(len(T), -1, dtype=np.int64)       # each player's position for the season
        for ids in rosters:
            ids = np.asarray(ids)
            ranked = np.argsort(-perceived[ids], kind="stable")
            expected = np.zeros(ids.size)
            expected[ranked] = template[:ids.size]
            position[ids] = assign_slots(expected)
        on_court, team_poss = np.zeros(len(T)), np.zeros(TEAMS)
        parts = {k: [] for k in ("game", "home_off", "lo", "ld", "poss")}
        for g, (home, away) in enumerate(schedule(rng)):
            n_stints = int(rng.choice(real["stints_per_game"]))
            p_home = rng.choice(real["poss_home"], n_stints)
            p_away = np.maximum(p_home + rng.integers(-1, 2, n_stints), 0.0)
            lineup = {}
            for team in (home, away):
                ids = np.asarray(rosters[team])
                present = ensure_available(rng.random(ids.size) < avail[ids], perceived[ids])
                order, shares = game_shares(perceived[ids], present, template, rng, p["share_noise"])
                slots = fill_slots(position[ids[order]], shares)
                lineup[team] = ids[order[draw_lineups(shares, slots, n_stints, slot_p, lines, rng)]]
            both = p_home + p_away
            for team in (home, away):
                np.add.at(on_court, lineup[team].ravel(), np.repeat(both, 5))
                team_poss[team] += both.sum()
            for lo, ld, poss, is_home in ((lineup[home], lineup[away], p_home, True),
                                          (lineup[away], lineup[home], p_away, False)):
                keep = poss > 0
                parts["game"].append(np.full(int(keep.sum()), g))
                parts["home_off"].append(np.full(int(keep.sum()), is_home))
                parts["lo"].append(lo[keep])
                parts["ld"].append(ld[keep])
                parts["poss"].append(poss[keep])
        seasons.append({k: np.concatenate(v) for k, v in parts.items()})
        by_rank, rhos, strength = [], [], []
        for team, ids in enumerate(rosters):
            ids = np.asarray(ids)
            share = on_court[ids] / team_poss[team]
            order = np.argsort(-share)
            by_rank.append(share[order][:ROSTER])
            top = order[:12]
            rhos.append(pd.Series(total[ids][top]).corr(pd.Series(share[top]), method="spearman"))
            strength.append(float((share * total[ids]).sum()))
            team_seasons.append(dict(season=season, team=team, ids=ids[order], share=share[order],
                                     position=position[ids[order]]))
        stats.append(dict(by_rank=np.mean(by_rank, axis=0), spearman12=float(np.nanmean(rhos)), strength=strength))
    return np.asarray(truth), seasons, stats, team_seasons


def group_tests(team_seasons: list, positions: bool) -> dict:
    """Team bench: each team-season's top five by minutes, against everyone else on the roster.  Own backups
    (position leagues): each position's most-used player, against the others at his position."""
    bench, backups = [], []
    for ts in team_seasons:
        bench.append((ts["ids"][:5], ts["ids"][5:]))
        if positions:
            for k in range(5):
                at = np.flatnonzero(ts["position"] == k)          # already in minutes order
                if at.size > 1:
                    backups.append((ts["ids"][at[:1]], ts["ids"][at[1:]]))
    return {"team bench": bench, "own backups": backups}


def outcomes(truth, rows, sigma_pp, rng) -> np.ndarray:
    """Points per 100 for each row: context + the five offences + the five defences + one possession's noise
    per possession (the real per-possession standard deviation)."""
    mean = (LEVEL + HOME * np.where(rows["home_off"], 1.0, -1.0) + truth[rows["lo"], 0].sum(1)
            + truth[rows["ld"], 1].sum(1))
    return mean + 100.0 * sigma_pp * rng.standard_normal(mean.size) / np.sqrt(rows["poss"])


def league_block(truth, seasons):
    """The league as 77's Block: [offence | defence | each season's home and intercept]."""
    P = len(truth)
    p = 2 * P + 2 * len(seasons)
    X_parts, w_parts, y_parts, tg_off, tg_def = [], [], [], [], []
    base = 0
    for s, rows in enumerate(seasons):
        N = rows["poss"].size
        F = np.column_stack([np.where(rows["home_off"], 1.0, -1.0), np.ones(N)])
        indices = np.hstack([rows["lo"], rows["ld"] + P, np.broadcast_to(2 * P + 2 * s + np.arange(2), (N, 2))])
        data = np.hstack([np.ones((N, 10)), F])
        X = sp.csr_matrix((data.ravel(), indices.ravel(), np.arange(0, N * 12 + 1, 12, dtype=np.int64)),
                          shape=(N, p))
        X.sort_indices()
        X_parts.append(X)
        w_parts.append(rows["poss"].astype(float))
        y_parts.append(rows["y"])
        h = rows["home_off"].astype(np.int64)
        tg_off.append(base + 2 * rows["game"] + h)
        tg_def.append(base + 2 * rows["game"] + 1 - h)
        base += 2 * (int(rows["game"].max()) + 1)
    X = sp.vstack(X_parts).tocsr()
    w, y = np.concatenate(w_parts), np.concatenate(y_parts)
    gram = (X.T @ r77.weighted(X, w)).toarray()
    diag = np.diagonal(gram)
    return r77.Block(rapm=None, designs={}, player_ids=np.arange(P), n=P, p=p, gram=gram,
                     rhs=np.asarray(X.T @ (w * y)).ravel(), ywy=float(w @ y ** 2), X_parts=X_parts, w_parts=w_parts,
                     X=X, w=w, y=y, tg_off=np.concatenate(tg_off), tg_def=np.concatenate(tg_def),
                     poss=diag[:P].copy(), poss_def=diag[P:2 * P].copy(), active_o=diag[:P] > 0,
                     active_d=diag[P:2 * P] > 0)


# ------------------------------------------------------------------------------------------ report helpers
def f2(x) -> str:
    return "" if x is None or not np.isfinite(x) else f"{x:+.2f}"


def pct(x) -> str:
    return "" if x is None or not np.isfinite(x) else f"{100 * x:.0f}%"


def main() -> None:
    check_flags()
    n_leagues = int(flag("leagues", 20))
    rng = np.random.default_rng(int(flag("seed", 0)))
    cfg = load_config()
    log: list = []
    started = time.time()

    print("real 2024-26 targets", flush=True)
    targets = real_targets()
    real = real_reference(cfg, targets, log)
    print(f"  one possession's noise {real['sigma_pp']:.3f} points; {np.median(real['stints_per_game']):.0f} "
          f"stints a game; replacement time from players sharing under 10% / 10-30% / 30%+ of his court time: "
          + " / ".join(pct(v) for v in real["split"]), flush=True)

    print("calibrating the minutes model", flush=True)
    cal = calibrate(targets, rng, log)

    players = [real["players"].assign(setting="real lineups", league=0)]
    groups = [real["groups"].assign(setting="real lineups", league=0)]
    realism = [dict(setting="real lineups", league=0, backups_time=real["split"][0], some_time=real["split"][1],
                    with_time=real["split"][2], strength_sd=targets["strength_sd"],
                    spearman12=targets["spearman12"], minutes_miss=0.0)]
    first = True
    for name, setting in SETTINGS.items():
        for league in range(n_leagues):
            t0 = time.time()
            truth, seasons, stats, team_seasons = make_league(targets, real, cal, setting, rng)
            for rows in seasons:
                rows["y"] = outcomes(truth, rows, real["sigma_pp"], rng)
            block = league_block(truth, seasons)
            n = block.n
            ops, rep, shared, valid = structure(block, log, f"{name} league {league}", strict=(league == 0))
            t = r77.centre(np.concatenate([truth[:, 0], truth[:, 1]]), block)
            if first:
                # the league's own truth must come back out of a nearly unpenalised fit, and noisy refits must
                # average to the exact expectation -- the block, the signs and the noise are built right
                _, _, M1 = r77.fit(block, 1.0, 1.0)
                heavy = np.concatenate([block.poss > 3000, block.poss_def > 3000])
                miss = float(np.abs((M1[:2 * n, :2 * n] @ t) - t)[heavy].max())
                check(miss < 0.05, f"a nearly unpenalised fit gives the league's truth back (max miss {miss:.3f}, "
                                   f"players with 3,000+ possessions)", log)
                factor, _, M = r77.fit(block, HEADLINE, HEADLINE)
                theta = np.concatenate([t, np.tile([HOME, LEVEL], SEASONS)])
                fits = r77.simulate(block, factor, theta, (100.0 * real["sigma_pp"]) ** 2, 100, rng)
                exact = (M @ theta)[:2 * n]
                z = np.abs(fits[:2 * n].mean(1) - exact) / (fits[:2 * n].std(1, ddof=1) / 10.0)
                act = np.concatenate([block.active_o, block.active_d])
                check(float(np.mean(z[act] < 4.0)) >= 0.99, f"100 noisy refits average to the exact expectation "
                                                             f"for {100 * np.mean(z[act] < 4.0):.1f}% of ratings", log)
                del M1, M, fits
                first = False
            p_frame, g_frame = measure(block, t, ops, rep, valid, group_tests(team_seasons, setting[0] > 0),
                                       realized=True)
            players.append(p_frame.assign(setting=name, league=league))
            groups.append(g_frame.assign(setting=name, league=league))
            split = replacement_split(ops, rep, shared, valid)
            realism.append(dict(setting=name, league=league, backups_time=split[0], some_time=split[1],
                                with_time=split[2], strength_sd=float(np.std(sum((s["strength"] for s in stats), []))),
                                spearman12=float(np.mean([s["spearman12"] for s in stats])),
                                minutes_miss=float(np.abs(np.mean([s["by_rank"] for s in stats], axis=0)
                                                          - targets["template"]).max())))
            now = p_frame[(p_frame.penalty == HEADLINE) & p_frame.valid & (p_frame.poss > HEAVY)]
            bench_now = g_frame[(g_frame.penalty == HEADLINE) & (g_frame.test == "team bench")]
            print(f"  {name:<14} league {league:>2}: {n} players; backups carry {pct(split[0])} of replacement "
                  f"time; at 40,000 share kept {f2(now.share_kept.median())}, team bench {f2(bench_now.rapm.median())} "
                  f"({time.time() - t0:.0f}s)", flush=True)

    players = pd.concat(players, ignore_index=True)
    groups = pd.concat(groups, ignore_index=True)
    realism = pd.DataFrame(realism)
    heavy = players[players.valid & (players.poss > HEAVY)]
    kept = heavy.groupby(["setting", "league", "penalty"]).share_kept.median().rename("value").reset_index() \
        .assign(test="share kept")
    moved = groups.groupby(["setting", "league", "penalty", "test"]).agg(value=("rapm", "median"),
                                                                          onoff=("onoff", "median")).reset_index()
    per_league = pd.concat([kept.assign(onoff=1.0), moved], ignore_index=True)
    summary = per_league.groupby(["setting", "test", "penalty"]).agg(
        value=("value", "median"), low=("value", "min"), high=("value", "max"), onoff=("onoff", "median"),
        leagues=("league", "nunique")).reset_index()
    real_view = realism.groupby("setting").mean(numeric_only=True).drop(columns="league")
    out = ROOT / "outputs"
    players.to_parquet(out / "synthetic_league_players.parquet", index=False)
    groups.to_parquet(out / "synthetic_league_groups.parquet", index=False)
    summary.merge(real_view.reset_index(), on="setting").to_parquet(out / "synthetic_league_summary.parquet",
                                                                    index=False)
    print("wrote outputs/synthetic_league_players.parquet, _groups.parquet, _summary.parquet", flush=True)

    # ---------------------------------------------------------------------------------- the report
    order = ["real lineups"] + list(SETTINGS)
    closest = min(SETTINGS, key=lambda s: abs(real_view.loc[s, "backups_time"] - real["split"][0]))
    realism_view = pd.DataFrame({
        "rotations": order,
        "replacement time from backups (<10% shared)": [pct(real_view.loc[s, "backups_time"]) for s in order],
        "from players he also plays with (30%+)": [pct(real_view.loc[s, "with_time"]) for s in order],
        "team strength spread": [f"{real_view.loc[s, 'strength_sd']:.1f}" for s in order],
        "minutes follow quality": [f"{real_view.loc[s, 'spearman12']:.2f}" for s in order]})

    def test_view(test: str) -> pd.DataFrame:
        rows = []
        for s in order:
            cells = {"rotations": s}
            for lam in PENALTIES:
                hit = summary[(summary.setting == s) & (summary.test == test) & (summary.penalty == lam)]
                if hit.empty:
                    cells[f"penalty {lam:,.0f}"] = "n/a"
                    continue
                r = hit.iloc[0]
                cells[f"penalty {lam:,.0f}"] = (f2(r.value) if r.leagues == 1
                                                else f"{f2(r.value)} [{f2(r.low)}, {f2(r.high)}]")
            hit = summary[(summary.setting == s) & (summary.test == test)]
            cells["raw on/off"] = f2(hit.onoff.iloc[0]) if not hit.empty else "n/a"
            rows.append(cells)
        return pd.DataFrame(rows)

    one = players[(players.setting == closest) & (players.league == 0) & (players.penalty == HEADLINE)
                  & (players.poss >= TABLE_MIN_POSS)].copy()
    one["rapm_rank"] = one.fit.rank(ascending=False, method="min").astype(int)
    one["truth_rank"] = one.truth.rank(ascending=False, method="min").astype(int)
    bench_one = groups[(groups.setting == closest) & (groups.league == 0) & (groups.penalty == HEADLINE)
                       & (groups.test == "team bench")].groupby("player").rapm.mean()
    top = one.sort_values("fit", ascending=False).head(20)
    top_view = pd.DataFrame({"RAPM rank": top.rapm_rank, "true rank": top.truth_rank, "truth": top.truth.map(f2),
                             "RAPM": top.fit.map(f2), "expected RAPM": top.expected.map(f2),
                             "poss": top.poss.map(lambda v: f"{v:,.0f}"), "keeps own rating": top.keep.map(pct),
                             "share kept": top.share_kept.map(f2),
                             "team bench 1 worse": top.player.map(bench_one).map(f2)})

    tables = [("How each league's rotations compare with 2024-26", realism_view, ("rotations",)),
              ("Share kept: the lineup that replaces him one point worse (raw on/off moves exactly 1)",
               test_view("share kept"), ("rotations",)),
              ("Team bench one point worse: how far the team's starters move", test_view("team bench"), ("rotations",)),
              ("His own position's backups one point worse: how far the starter moves", test_view("own backups"),
               ("rotations",)),
              (f"Top 20 by fitted RAPM, one '{closest}' league (closest to 2024-26), penalty 40,000", top_view, ())]
    for title, frame, _ in tables:
        print(f"\n{title}\n{frame.to_string(index=False)}")
    glossary = ("Points per 100 possessions. Every test makes some players one point worse (half at each end) and "
                "reads how far a player's expected RAPM moves; it needs no truth. <b>Positive = bad replacements "
                "make him look better (the belief).</b> Medians over players with 10,000+ possessions or over "
                f"starters; brackets are the range over {n_leagues} leagues. <b>raw on/off</b>: the same test on raw "
                "on/off, for scale. <b>real lineups</b>: the actual 2024-26 rotations. <b>keeps own rating</b>: the "
                "share of his own true rating RAPM keeps.")
    page = ('<div style="font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;color:#1f2933">'
            '<h2 style="font-size:18px;margin:0 0 6px">A synthetic league: do truly bad replacements inflate RAPM?</h2>'
            f'<p style="font-size:13px;color:#6b7280;margin:0 0 8px">{n_leagues} leagues per rotation style, 30 teams, '
            f'3 seasons, 82 games; plain RAPM; every check passed.</p>'
            f'<p style="font-size:12px;color:#6b7280">{glossary}</p>'
            + "".join(f'<h3 style="font-size:15px">{html.escape(title)}</h3>' + r77.html_table(frame, left=left)
                      for title, frame, left in tables)
            + '<h3 style="font-size:15px">Checks</h3>'
            + "".join(f'<p style="font-size:12px;margin:0 0 3px">{html.escape(c)}</p>' for c in log) + "</div>")
    (out / "synthetic_league_phone.html").write_text(page, encoding="utf-8")
    print(f"wrote outputs/synthetic_league_phone.html\ndone in {time.time() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
