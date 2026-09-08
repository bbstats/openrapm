"""The team he is traded to: the destination's usage minutes and quality as inputs to the prior (FINDINGS 30).

The owner's finding (28.8): a high-usage player's value falls with new teammates because his USAGE falls
there -- his role does not transfer.  So the prior should see the destination.  Usage minutes = usage per
100 x possession share: his slice of his team's possessions.  Three columns:

  own_um     his usage minutes in the feature window
  dest_um    the usage minutes already spoken for beside him: four times the shared-possession-weighted mean
             of his TARGET-window teammates' usage minutes, each measured in the FEATURE window
  dest_apm   how good that group is: the same weighting of their offensive APM, measured in the feature window

Leak-free the way the turnover feature is: WHO he plays beside comes from the target window (the teammates
table, data/cache/teammates.parquet), but every number attached to a teammate is from the feature window --
nothing measured on the target window enters.  A teammate the feature window never saw takes that window's
possession-weighted league values.  At prediction time (`destination_inputs`) the feature window is the
training block itself: each teammate's usage minutes from the block's padded rates and share, his APM from
the block's own APM fit, and the roster from the target seasons (the held-out season for the criterion, the
block's own seasons for the shipped board).  `override` swaps the roster for any team's, or for league-average
values: the trade-to question.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DEST = ["own_um", "dest_um", "dest_apm"]
# the owner's extension: "block-minutes" and "rebound-minutes" -- the same construction on the defensive side, a
# rate x his possession share, his own and the destination's already spoken for, with the destination's
# DEFENSIVE APM beside them
DEST_D = ["own_bm", "own_rm", "dest_bm", "dest_rm", "dest_apm_d"]
DEST_ALL = [*DEST, *DEST_D]
FLOOR = 4.0          # the four teammates on the floor with him
_VALS = ["um", "bm", "rm", "apm_o", "apm_d"]                       # what a teammate carries into the destination
_OUT = {"um": "dest_um", "bm": "dest_bm", "rm": "dest_rm", "apm_o": "dest_apm", "apm_d": "dest_apm_d"}
_OWN = {"um": "own_um", "bm": "own_bm", "rm": "own_rm"}
_RATES = ("um", "bm", "rm")


def _rate(d: pd.DataFrame, stat: str, prefix: str = "raw_") -> np.ndarray:
    if stat == "um":
        return _usage(d, prefix)
    if stat == "bm":
        return d[f"{prefix}blk"].to_numpy(dtype=float)
    return d[f"{prefix}orb"].to_numpy(dtype=float) + d[f"{prefix}drb"].to_numpy(dtype=float)


def _fill(v: str, lg: pd.Series) -> float:
    return (FLOOR if v in _RATES else 1.0) * float(lg[f"{v}_lg"])


def _usage(d: pd.DataFrame, prefix: str = "raw_") -> np.ndarray:
    c = lambda n: d[f"{prefix}{n}"].to_numpy(dtype=float)
    return c("fg2m") + c("fg2_miss") + c("fg3m") + c("fg3_miss") + 0.44 * (c("ftm") + c("ft_miss")) + c("tov")


def usage_minutes_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Per (window, player): usage minutes and offensive APM from the panel's OFFENSIVE rows (raw padded rates,
    possession share, apm), with the possessions behind them."""
    o = panel[panel.side == "O"]
    d = panel[panel.side == "D"].set_index(["window", "player_id"]).apm
    share = o.poss_pct.to_numpy(dtype=float)
    key = pd.MultiIndex.from_arrays([o.window.to_numpy(), o.player_id.to_numpy(np.int64)])
    return pd.DataFrame({"window": o.window.to_numpy(), "player_id": o.player_id.to_numpy(np.int64),
                         "um": _rate(o, "um") * share, "bm": _rate(o, "bm") * share, "rm": _rate(o, "rm") * share,
                         "apm_o": o.apm.to_numpy(dtype=float), "apm_d": d.reindex(key).fillna(0.0).to_numpy(dtype=float),
                         "poss": o.poss.to_numpy(dtype=float)})


def _league(um: pd.DataFrame) -> pd.DataFrame:
    """Possession-weighted league values per window, columns <val>_lg, for every value column present."""
    w = um.poss.clip(lower=1.0)
    vals = [v for v in _VALS if v in um.columns]
    g = um.assign(_w=w, **{f"_{v}": um[v] * w for v in vals}).groupby("window")[[f"_{v}" for v in vals] + ["_w"]].sum()
    return pd.DataFrame({f"{v}_lg": g[f"_{v}"] / g._w for v in vals})


def _roster_features(um_feat: pd.DataFrame, lg: pd.Series, roster: pd.DataFrame) -> pd.DataFrame:
    """`roster`: rows (key, teammate_id, shared) for one feature-window frame `um_feat` (player_id, um, apm_o);
    returns per key dest_um, dest_apm with unseen teammates at the league values `lg` (um_lg, apm_lg)."""
    vals = [v for v in _VALS if v in um_feat.columns]
    m = roster.merge(um_feat[["player_id", *vals]].rename(columns={"player_id": "teammate_id"}), on="teammate_id", how="left")
    for v in vals:
        m[v] = m[v].fillna(float(lg[f"{v}_lg"]))
        m[f"_{v}"] = m[v] * m.shared
    g = m.groupby("key")[[f"_{v}" for v in vals] + ["shared"]].sum()
    return pd.DataFrame({_OUT[v]: (FLOOR if v in _RATES else 1.0) * g[f"_{v}"] / g.shared for v in vals})


def destination_features(um: pd.DataFrame, tm: pd.DataFrame, windows: dict, keys: pd.DataFrame) -> pd.DataFrame:
    """DEST for each row of `keys` (player_id, window, window_to): `um` from usage_minutes_panel, `tm` the
    teammates table, `windows` label -> list of seasons."""
    lg = _league(um)
    vals = [v for v in _VALS if v in um.columns]
    owns = [v for v in vals if v in _OWN]
    out = pd.DataFrame(index=keys.index, columns=[_OWN[v] for v in owns] + [_OUT[v] for v in vals], dtype=float)
    own = keys[["player_id", "window"]].merge(um[["player_id", "window", *owns]], on=["player_id", "window"], how="left")
    for v in owns:
        out[_OWN[v]] = own[v].fillna(own.window.map(lg[f"{v}_lg"])).to_numpy()
    k = keys.assign(key=np.arange(len(keys)))
    for (w, w_to), grp in k.groupby(["window", "window_to"]):
        seasons = list(windows[w_to])
        t = tm[tm.season.isin(seasons)].groupby(["player_id", "teammate_id"], as_index=False)["shared"].sum()
        roster = grp[["key", "player_id"]].merge(t, on="player_id", how="inner")[["key", "teammate_id", "shared"]]
        feat = um[um.window == w]
        r = _roster_features(feat, lg.loc[w], roster) if len(roster) else pd.DataFrame(columns=[_OUT[v] for v in vals])
        idx = grp.index
        r = r.reindex(grp.key.to_numpy()).astype(float)
        for v in vals:
            out.loc[idx, _OUT[v]] = r[_OUT[v]].fillna(_fill(v, lg.loc[w])).to_numpy()
    return out


def block_usage_apm(wd, exp, inputs: pd.DataFrame, cfg) -> pd.DataFrame:
    """The prediction-time feature frame: per player of the block (ps_idx order), usage minutes from the block's
    padded raw rates and possession share, offensive APM from the block's own APM fit (spm.apm_fit, the same
    definition the panel's `apm` column has), and the possessions."""
    from .design import FEATURES
    from .spm import apm_fit
    R = pd.DataFrame(np.asarray(exp.season_rates_, dtype=float), columns=[f"raw_{c}" for c in FEATURES])
    a = apm_fit(wd, cfg)
    share = inputs["poss_pct"].to_numpy(dtype=float)
    return pd.DataFrame({"player_id": wd.spec.ps_table["player_id"].to_numpy(np.int64),
                         "um": _rate(R, "um") * share, "bm": _rate(R, "bm") * share, "rm": _rate(R, "rm") * share,
                         "apm_o": a["u_o"], "apm_d": a["u_d"], "poss": np.asarray(exp.season_poss_off_, dtype=float)})


def destination_inputs(block: pd.DataFrame, tm: pd.DataFrame, target_seasons, player_ids, override=None) -> pd.DataFrame:
    """DEST aligned to `player_ids` from the block frame (`block_usage_apm`) and the rosters of `target_seasons`.

    `override`: None = the actual rosters;  {"dest_um": x, "dest_apm": y} = the same context for everyone (e.g.
    the league average);  {"roster": [ids], "weights": [...]} = one roster for everyone (a trade-to question:
    the teammates are the roster minus the player himself)."""
    b = block.assign(window="block")
    lg = _league(b).loc["block"]
    vals = [v for v in _VALS if v in b.columns]
    owns = [v for v in vals if v in _OWN]
    ids = np.asarray(player_ids, dtype=np.int64)
    own = pd.DataFrame({"player_id": ids}).merge(b[["player_id", *owns]], on="player_id", how="left")
    out = pd.DataFrame({_OWN[v]: own[v].fillna(float(lg[f"{v}_lg"])).to_numpy() for v in owns}, index=range(len(ids)))
    cols = [_OWN[v] for v in owns] + [_OUT[v] for v in vals]
    if isinstance(override, dict) and "dest_um" in override:
        for v in vals:                                  # the same context for everyone; unspecified = league
            out[_OUT[v]] = float(override.get(_OUT[v], _fill(v, lg)))
        return out[cols]
    if isinstance(override, dict) and "roster" in override:
        ro = np.asarray(override["roster"], dtype=np.int64)
        wt = np.asarray(override.get("weights", np.ones(len(ro))), dtype=float)
        rows = [(i, t, float(w_)) for i, p in enumerate(ids) for t, w_ in zip(ro, wt) if t != p]
        roster = pd.DataFrame(rows, columns=["key", "teammate_id", "shared"])
    else:
        t = tm[tm.season.isin(list(target_seasons))].groupby(["player_id", "teammate_id"], as_index=False)["shared"].sum()
        roster = pd.DataFrame({"key": np.arange(len(ids)), "player_id": ids}).merge(t, on="player_id", how="inner")[["key", "teammate_id", "shared"]]
    r = _roster_features(b, lg, roster) if len(roster) else pd.DataFrame(columns=[_OUT[v] for v in vals])
    r = r.reindex(range(len(ids))).astype(float)
    for v in vals:
        out[_OUT[v]] = r[_OUT[v]].fillna(_fill(v, lg)).to_numpy()
    return out[cols]


def team_roster(roles: pd.DataFrame, seasons, team_id: int, tm: pd.DataFrame | None = None,
                rotation: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """A team's players over `seasons` and the weight a newcomer would share the floor with each.

    With `tm` (the teammates table) the weight of player t is the mean shared possessions between t and the
    team's `rotation` highest-minute players -- the co-occurrence structure a rotation player actually has,
    which is starter-heavy the way `destination_inputs`' actual rosters are.  Without it, t's own on-court
    possessions, which weights the bench as if a newcomer shared the floor with everyone in proportion."""
    r = roles[roles.season.isin(list(seasons)) & (roles.team_id == int(team_id)) & (roles.poss_on > 0)]
    g = r.groupby("player_id").poss_on.sum()
    ids = g.index.to_numpy(np.int64)
    if tm is None or len(g) == 0:
        return ids, g.to_numpy(dtype=float)
    top = g.sort_values(ascending=False).index[:rotation].to_numpy(np.int64)
    t = tm[tm.season.isin(list(seasons)) & tm.player_id.isin(top) & tm.teammate_id.isin(ids)]
    sh = t.groupby("teammate_id").shared.sum() / float(len(top))
    w = pd.Series(ids, index=ids).map(sh).fillna(0.0).to_numpy(dtype=float)
    return ids, w
