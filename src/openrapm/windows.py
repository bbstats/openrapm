"""Windows of seasons: name a window, and the per-player ratings table of a plug-in fit.

A window is a run of seasons (the configured 3-season windows, or one season); its label is "first-last".
A window's design is built by `data.designs.build_window`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .cv import plugin_fit


def window_seasons(cfg, rolling=False):
    """[(first, last), ...] for the ten non-overlapping windows, or the rolling ones."""
    if not rolling:
        return [tuple(w) for w in cfg["windows"]]
    first, last = cfg["first_season"], cfg["last_season"]
    step = int(cfg.get("rolling_step", 1))
    return [(s, s + 2) for s in range(first, last - 1, step)]


def window_label(seasons) -> str:
    return f"{seasons[0]}-{seasons[-1]}"


def label_seasons(label: str) -> list:
    """The seasons a window label spans: "1997-1999" -> [1997, 1998, 1999], "2004-2004" -> [2004]."""
    a, b = str(label).split("-")
    return list(range(int(a), int(b) + 1))


def labels_covering(labels, seasons) -> set:
    """Every window label whose seasons intersect `seasons`.  On the configured windows this is the label of
    each season's own window; on a per-season panel it is the seasons themselves."""
    want = {int(s) for s in seasons}
    return {lab for lab in set(labels) if want.intersection(label_seasons(lab))}


def window_mid(seasons) -> float:
    return float(np.mean(seasons))


def player_ratings_table(wd, beta, cfg, seasons, beta_po=None, names=None,
                         prior_offset=None, prior_parts=None) -> pd.DataFrame:
    """Per player-season ratings from the plug-in fit at the fixed lambda.

    prior  = the player's padded full-season rates times the window's coefficients, centred on the
             average player (the exposure columns are centred on the sum of five, so divide by five)
    u      = what the ridge finds beyond the box score, and beyond `prior_offset` when one is given
    rapm_mm = prior + the residual fit with NO offset: what the model says without the boosted
             correction, kept so the with and without versions stay comparable
    Defense is flipped so positive = good; total is offense plus defense.

    `prior_offset` is the per-Z-column boosted correction in raw sign.  Passing it makes `u` the
    residual refit *given* the correction, which is what the retired scripts/10_boost.py shipped.  Adding a
    correction on top of a residual that was fit without it double-counts whatever it explains.
    `prior_parts` = (offense, defense) halves of an offset that IS the prior (the role-prior chain of
    spm.py / the retired gbdt_prior.py, raw sign): they are added into `prior_off` / `prior_def` so that
    `rating = prior + u` holds with no linear box term at all.
    """
    kw = dict(lam=float(cfg["lam_plugin"]), lam_ratio=float(cfg["lam_ratio_plugin"]),
              pad_target=cfg["pad_target"])
    plain = plugin_fit(wd, beta, **kw)
    pipe = plain if prior_offset is None else plugin_fit(wd, beta, prior_offset=prior_offset, **kw)
    exp, mm = pipe["exposure"], pipe["mm"]
    u_plain = plain["mm"].u_
    nf = mm.n_feat_
    m = wd.spec.n_ps
    ro = exp.season_rates_ - exp.means_o_ / 5.0
    rd = exp.season_rates_d_ - exp.means_d_ / 5.0
    df = wd.spec.ps_table.copy()
    df["window"] = window_label(seasons)
    df["window_mid"] = window_mid(seasons)
    df["poss_off"] = exp.season_poss_off_
    df["poss_def"] = exp.season_poss_def_
    df["prior_off"] = ro @ beta[:nf]
    df["prior_def"] = -(rd @ beta[nf:])
    if prior_parts is not None:
        df["prior_off"] = df["prior_off"] + np.asarray(prior_parts[0], dtype=float)
        df["prior_def"] = df["prior_def"] - np.asarray(prior_parts[1], dtype=float)     # raw sign in, flipped out
    df["u_off"] = mm.u_[:m]
    df["u_def"] = -mm.u_[m:]
    df["u_plain_off"] = u_plain[:m]
    df["u_plain_def"] = -u_plain[m:]
    if beta_po is not None:
        df["prior_off_po"] = ro @ beta_po[:nf]
        df["prior_def_po"] = -(rd @ beta_po[nf:])
    for part in ("prior", "u", "u_plain"):
        df[f"{part}_total"] = df[f"{part}_off"] + df[f"{part}_def"]
    for side in ("off", "def", "total"):
        df[f"rapm_mm_{side}"] = df[f"prior_{side}"] + df[f"u_plain_{side}"]
    if beta_po is not None:
        df["prior_total_po"] = df.prior_off_po + df.prior_def_po
        for side in ("off", "def", "total"):
            df[f"rapm_mm_{side}_po"] = df[f"prior_{side}_po"] + df[f"u_plain_{side}"]
    if names is not None:
        if "season" in df.columns and "season" in names.columns:
            df = df.merge(names, on=["player_id", "season"], how="left")
        else:
            # player-window units: take the name from the season he played most
            gp = wd.game_poss.merge(wd.spec.psx_table[["psx_idx", "player_id", "season"]], on="psx_idx", how="left")
            poss = gp.groupby(["player_id", "season"], as_index=False)["poss_off"].sum()
            pick = poss.sort_values("poss_off").drop_duplicates("player_id", keep="last")
            nm = pick.merge(names, on=["player_id", "season"], how="left")[["player_id", "player_name"]]
            df = df.merge(nm, on="player_id", how="left")
    df["shrinkage"] = np.where(df.poss_off > 0, df.poss_off / (df.poss_off + float(cfg["lam_plugin"])), 0.0)
    return df.drop(columns=["ps_idx", "psx_idx"], errors="ignore")
