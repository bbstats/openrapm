"""The window design assembled from cached per-season pieces.

`design.build_design` rebuilds everything from the raw stints every time a block is fitted: the game order,
the player-season keys, the slot indices, the per-game possession and box tables, the counters.  Every one of
those is a property of ONE season (and phase); only the concatenation and the block's player unit are a
property of the block.  `season_pieces` computes the per-season part once per process (`_PIECES`, a small
LRU: a worker's consecutive held-out seasons share most of their blocks) and `build_window_cached` assembles a
block from them.  The result is the same WindowData as `build_design` on the concatenated stints: the same
row order (home-offense rows of every season, then away-offense), the same column layout, the same tables
(`scratch/cmp_design.py` checks every array), so it can stand in for it wherever `margin_bins` is off.
"""
from __future__ import annotations

from collections import OrderedDict

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .design import AWAY_SLOTS, HOME_SLOTS, TARGETS, DesignSpec, WindowData, _order_games, ps_key
from .stints import POSS_COUNTERS, SLOT_COUNTERS

_PIECES: "OrderedDict[tuple, dict]" = OrderedDict()
MAX_PIECES = 8


def season_pieces(season: int, phase: str, cfg: dict) -> dict:
    """Everything about one season-phase that a block design needs, keyed by the player-season key."""
    key = (int(season), str(phase))
    if key in _PIECES:
        _PIECES.move_to_end(key)
        return _PIECES[key]
    from .boxtable import season_box
    from .windows import load_stints
    st = load_stints(int(season), phase, cfg).reset_index(drop=True).copy()
    for c in HOME_SLOTS + AWAY_SLOTS:
        st[c] = st[c].astype(np.int64)
    box = season_box([int(season)], [phase], cfg)
    games = _order_games(st)                                   # local game_idx 0..G-1
    st = st.merge(games[["game_id", "game_idx"]], on="game_id", how="left")
    season_arr = st["season"].to_numpy()
    slot_keys = np.stack([ps_key(st[c].to_numpy(), season_arr) for c in HOME_SLOTS + AWAY_SLOTS], axis=1)   # (n_st, 10)
    box_keys = ps_key(box["player_id"].to_numpy(), box["season"].to_numpy())
    keys_u = np.unique(np.concatenate([slot_keys.ravel(), box_keys]))
    slot_loc = np.searchsorted(keys_u, slot_keys)              # local psx index per slot
    n_st = len(st)
    is_po = (st["phase"].to_numpy() == "PO")
    # per (game, player-season) possessions, local indices, the groupby order of build_game_poss
    gi = st["game_idx"].to_numpy()
    g_all, x_all, po_all, pd_all = [], [], [], []
    for cols, own, opp in ((slice(0, 5), "poss_h", "poss_a"), (slice(5, 10), "poss_a", "poss_h")):
        po, pdd = st[own].to_numpy(), st[opp].to_numpy()
        for j in range(cols.start, cols.stop):
            g_all.append(gi); x_all.append(slot_loc[:, j]); po_all.append(po); pd_all.append(pdd)
    gp = pd.DataFrame({"game_idx": np.concatenate(g_all), "psx_idx": np.concatenate(x_all),
                       "poss_off": np.concatenate(po_all), "poss_def": np.concatenate(pd_all)})
    gp = gp.groupby(["game_idx", "psx_idx"], as_index=False)[["poss_off", "poss_def"]].sum()
    # the box table, local indices
    features = list(cfg["features"])
    bx = box.merge(games[["game_id", "game_idx"]], on="game_id", how="inner")
    bx["psx_idx"] = np.searchsorted(keys_u, ps_key(bx["player_id"].to_numpy(), bx["season"].to_numpy()))
    game_box = bx[["game_idx", "psx_idx", "phase"] + features].copy()
    game_box = game_box.groupby(["game_idx", "psx_idx", "phase"], as_index=False)[features].sum()
    # possessions per local psx over regular-season games
    if phase == "RS":
        psx_poss = np.bincount(gp["psx_idx"].to_numpy(), weights=gp["poss_off"].to_numpy(dtype=float), minlength=len(keys_u))
    else:
        psx_poss = np.zeros(len(keys_u))
    # the counters, per side, as one matrix in the column order build_design uses
    have = [c for c in POSS_COUNTERS + SLOT_COUNTERS if f"{c}_h" in st.columns and f"{c}_a" in st.columns]
    ccols = have + ["pts", "poss"]
    counters = {side: np.column_stack([st[f"{c}_{side}"].to_numpy(dtype=float) for c in ccols]) if ccols else None
                for side in ("h", "a")}
    pids = {"h": st[HOME_SLOTS].to_numpy(dtype=np.int64), "a": st[AWAY_SLOTS].to_numpy(dtype=np.int64)}
    series = st["series_id"].astype(str).to_numpy()
    gid = st["game_id"].astype(str).to_numpy()
    groups = np.where(is_po, np.char.add("S:", series.astype(str)), np.char.add("G:", gid.astype(str)))
    piece = dict(season=int(season), phase=str(phase), st=st, games=games, n_games=len(games), keys_u=keys_u,
                 slot_loc=slot_loc, n_st=n_st, is_po=is_po,
                 neutral=st["neutral"].to_numpy().astype(bool) if "neutral" in st.columns else np.zeros(n_st, bool),
                 is_gt=st["is_gt"].to_numpy().astype(bool), margin_h=st["margin_h"].to_numpy().astype(float),
                 frac_rem=st["frac_rem"].to_numpy().astype(float), game_idx=gi, groups=groups,
                 game_poss=gp, game_box=game_box, psx_poss=psx_poss, have=have, ccols=ccols, counters=counters, pids=pids)
    _PIECES[key] = piece
    while len(_PIECES) > MAX_PIECES:
        _PIECES.popitem(last=False)
    return piece


def build_window_cached(seasons, cfg, phases=("RS",), gt_weight=None, target="pts") -> WindowData:
    """`windows.build_window` from cached per-season pieces (no `margin_bins`)."""
    seasons = [int(s) for s in seasons]
    phases = list(phases)
    pieces = [season_pieces(s, p, cfg) for s in seasons for p in phases]
    if gt_weight is None:
        gt_weight = float(cfg.get("gt_weight", 1.0))
    margin_clip = float(cfg.get("margin_clip", 25))
    features = list(cfg["features"])

    # the block's player-season table: the union of the pieces' keys, sorted (= season-major)
    keys = np.unique(np.concatenate([p["keys_u"] for p in pieces]))
    psx_table = pd.DataFrame({"key": keys, "season": keys // 100_000_000, "player_id": keys % 100_000_000})
    psx_table["psx_idx"] = np.arange(len(psx_table), dtype=np.int64)
    ssn = sorted(psx_table["season"].unique().tolist())
    season_of_psx = np.searchsorted(np.asarray(ssn), psx_table["season"].to_numpy())
    player_unit = str(cfg.get("player_unit", "season"))
    if player_unit not in ("window", "season"):
        raise ValueError(f"player_unit must be 'window' or 'season', got {player_unit!r}")
    if player_unit == "window":
        players = np.unique(psx_table["player_id"].to_numpy())
        ps_table = pd.DataFrame({"player_id": players, "ps_idx": np.arange(len(players), dtype=np.int64)})
        ps_of_psx = pd.Index(players).get_indexer(psx_table["player_id"].to_numpy())
    else:
        ps_table = psx_table[["psx_idx", "player_id", "season"]].rename(columns={"psx_idx": "ps_idx"}).copy()
        ps_of_psx = psx_table["psx_idx"].to_numpy()
    psx_table["ps_idx"] = ps_of_psx
    n_ps = len(ps_table)

    # per piece: local -> block indices
    loc2blk = [np.searchsorted(keys, p["keys_u"]) for p in pieces]
    g_off = np.cumsum([0] + [p["n_games"] for p in pieces])[:-1]
    ps_slots = [ps_of_psx[m[p["slot_loc"]]] for p, m in zip(pieces, loc2blk)]        # (n_st, 10) Z units
    home_ps = np.concatenate([s[:, :5] for s in ps_slots]); away_ps = np.concatenate([s[:, 5:] for s in ps_slots])
    is_po = np.concatenate([p["is_po"] for p in pieces])
    neutral = np.concatenate([p["neutral"] for p in pieces])
    is_gt = np.concatenate([p["is_gt"] for p in pieces])
    margin_h = np.concatenate([p["margin_h"] for p in pieces])
    frac_rem = np.concatenate([p["frac_rem"] for p in pieces])
    r_game_st = np.concatenate([p["game_idx"] + o for p, o in zip(pieces, g_off)])
    season_st = np.concatenate([np.full(p["n_st"], p["season"]) for p in pieces])
    season_idx = np.searchsorted(np.asarray(ssn), season_st)
    groups_st = np.concatenate([p["groups"] for p in pieces])
    tg = TARGETS[target] if isinstance(target, str) else dict(target)

    def col(name):
        return np.concatenate([p["st"][name].to_numpy().astype(float) for p in pieces])

    sides = []
    for off_ps, def_ps, side, sign in ((home_ps, away_ps, "h", 1.0), (away_ps, home_ps, "a", -1.0)):
        poss = col(f"poss_{side}")
        den = col(f"{tg['den']}_{side}")
        num = sum(k * col(f"{c}_{side}") for c, k in tg["num"].items())
        keep = den > 0
        sides.append(dict(off=off_ps[keep], de=def_ps[keep], poss=poss[keep], den=den[keep], num=num[keep],
                          home=np.where(neutral[keep], 0.0, sign), margin=sign * margin_h[keep],
                          stint=np.flatnonzero(keep), is_home_off=sign > 0))
    off = np.concatenate([s["off"] for s in sides]); de = np.concatenate([s["de"] for s in sides])
    poss = np.concatenate([s["poss"] for s in sides]); den = np.concatenate([s["den"] for s in sides])
    num = np.concatenate([s["num"] for s in sides])
    home = np.concatenate([s["home"] for s in sides]); margin = np.concatenate([s["margin"] for s in sides])
    stint_i = np.concatenate([s["stint"] for s in sides])
    is_home_off = np.concatenate([np.full(len(s["stint"]), s["is_home_off"]) for s in sides])
    n = len(den)

    y = tg["scale"] * num / den
    w = den * np.where(is_gt[stint_i], gt_weight, 1.0)
    r_po = is_po[stint_i].astype(float)
    r_gt = is_gt[stint_i].astype(float)
    r_margin = np.clip(margin, -margin_clip, margin_clip)
    r_frem = frac_rem[stint_i]
    r_season = season_idx[stint_i]
    r_game = r_game_st[stint_i]

    f_names = ["home"] + [f"int_{s}" for s in ssn] + ["is_po", "po_home", "is_gt", "margin", "margin_frem"]
    f_cols = [home] + [(r_season == k).astype(float) for k in range(len(ssn))] + [r_po, r_po * home, r_gt, r_margin, r_margin * r_frem]
    F = sp.csr_matrix(np.column_stack(f_cols))
    rows_ar = np.repeat(np.arange(n), 5)
    Z_O = sp.csr_matrix((np.ones(n * 5), (rows_ar, off.ravel())), shape=(n, n_ps))
    Z_D = sp.csr_matrix((np.ones(n * 5), (rows_ar, de.ravel())), shape=(n, n_ps))
    assert Z_O.nnz == n * 5 and Z_D.nnz == n * 5, "duplicate player ids inside a lineup"
    G = sp.csr_matrix(r_game.astype(float)[:, None])
    X = sp.hstack([Z_O, Z_D, F, G], format="csr")
    X.sort_indices()
    groups = groups_st[stint_i]

    # the side tables, block indices
    game_poss = pd.concat([p["game_poss"].assign(game_idx=p["game_poss"]["game_idx"].to_numpy() + o,
                                                 psx_idx=m[p["game_poss"]["psx_idx"].to_numpy()])
                           for p, o, m in zip(pieces, g_off, loc2blk)], ignore_index=True)
    game_box = pd.concat([p["game_box"].assign(game_idx=p["game_box"]["game_idx"].to_numpy() + o,
                                               psx_idx=m[p["game_box"]["psx_idx"].to_numpy()])
                          for p, o, m in zip(pieces, g_off, loc2blk)], ignore_index=True)
    games = pd.concat([p["games"].assign(game_idx=p["games"]["game_idx"].to_numpy() + o)
                       for p, o in zip(pieces, g_off)], ignore_index=True)
    psx_poss = np.zeros(len(psx_table))
    for p, m in zip(pieces, loc2blk):
        psx_poss[m] += p["psx_poss"]
    ps_poss = np.bincount(ps_of_psx, weights=psx_poss, minlength=n_ps)
    if player_unit == "window":
        season_of_ps = np.zeros(n_ps, dtype=np.int64)
        best = np.full(n_ps, -1.0)
        for k in np.argsort(psx_poss):
            i = ps_of_psx[k]
            if psx_poss[k] >= best[i]:
                best[i], season_of_ps[i] = psx_poss[k], season_of_psx[k]
    else:
        season_of_ps = season_of_psx
    thr = float(cfg.get("low_poss_threshold", 500))
    thr_hi = float(cfg.get("starter_poss_threshold", 1500))
    low = np.flatnonzero(ps_poss < thr)
    high = np.flatnonzero(ps_poss >= thr_hi)
    col_groups = {"low_poss": np.concatenate([low, low + n_ps]), "high_poss": np.concatenate([high, high + n_ps]),
                  "high_poss_O": high, "high_poss_D": high + n_ps}
    spec = DesignSpec(n_ps=n_ps, seasons=ssn, season_of_ps=season_of_ps, ps_table=ps_table.copy(), f_names=f_names,
                      features=features, col_groups=col_groups,
                      psx_table=psx_table[["psx_idx", "player_id", "season", "ps_idx"]].copy(),
                      season_of_psx=season_of_psx, ps_of_psx=ps_of_psx, player_unit=player_unit)
    game_half = np.empty(len(games), dtype=object)
    game_half[games["game_idx"].to_numpy()] = games["half"].to_numpy()
    rows = pd.DataFrame({"game_idx": r_game, "season": np.asarray(ssn)[r_season], "phase": np.where(r_po > 0, "PO", "RS"),
                         "poss": poss, "den": den, "is_home_off": is_home_off, "is_gt": r_gt > 0,
                         "stint": stint_i, "half": game_half[r_game]})
    counters = None
    if pieces[0]["have"]:
        ccols = pieces[0]["ccols"]
        M = np.vstack([np.vstack([p["counters"][side] for p in pieces])[s["stint"]] for s, side in zip(sides, ("h", "a"))])
        counters = pd.DataFrame({c: M[:, j] for j, c in enumerate(ccols)})
        P = np.vstack([np.vstack([p["pids"][side] for p in pieces])[s["stint"]] for s, side in zip(sides, ("h", "a"))])
        for k in range(5):
            counters[f"pid_s{k + 1}"] = P[:, k]
        counters["half"] = rows["half"].to_numpy()
    parts = dict(Z=X[:, :2 * n_ps], F=np.asarray(F.todense()), lineup_o=np.sort(off, axis=1), lineup_d=np.sort(de, axis=1),
                 game_idx=r_game.astype(np.int64))
    return WindowData(X=X, y=y, w=w, groups=groups, spec=spec, game_box=game_box, game_poss=game_poss, rows=rows,
                      games=games, counters=counters, parts=parts)
