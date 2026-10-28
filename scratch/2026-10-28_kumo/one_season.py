"""Experiment 118's first look, one test season: the Kumo prior against the shipped LightGBM prior on the 2025 rows.

For each side: the box-score prior from Kumo Tabular (params/booster_params_kumo_<size>.json, no bags) on the training
rows check 155 cached for 2025 (the test setting: 2024-2026 unseen), each 2025 player answered by the context that
never saw him; and creation110's saved prior.  Both are scored against the player's label (his RAPM over the seasons
outside 2024-2026, step 1), weighted by the possessions behind the label, over the 2025 players who have one.  Also:
seconds per side, and whether a row's answer is the same asked alone as asked with the others.

    .venv/Scripts/python scratch/2026-10-28_kumo/one_season.py [--size small] [--season 2025]
"""
from __future__ import annotations

import argparse
import copy
import pickle
import time

import numpy as np
import pandas as pd

from openrapm import box_prior
from openrapm.data.config import booster_params, load_config, resolve
from openrapm.features import FEATURE_SETS, ROW_FEATURES

SHIPPED = "hustle_no_three_pct_self_creation"


def weighted(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> dict:
    w = w / w.sum()
    mx, my = (w * x).sum(), (w * y).sum()
    cov = (w * (x - mx) * (y - my)).sum()
    return dict(rmse=float(np.sqrt((w * (x - y) ** 2).sum())), corr=float(cov / np.sqrt((w * (x - mx) ** 2).sum()
                                                                                         * (w * (y - my) ** 2).sum())),
                sd=float(np.sqrt((w * (x - mx) ** 2).sum())))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", default="small")
    parser.add_argument("--season", type=int, default=2025)
    args = parser.parse_args()
    cfg = load_config()
    outputs = resolve(cfg, "outputs")
    with open(outputs / "screen" / "screen112" / f"train_{args.season}.pkl", "rb") as handle:
        train = pickle.load(handle)["train"]
    panel = pd.read_parquet(resolve(cfg, "panel"))
    panel = panel[(panel.season == args.season) & (panel.possessions > 0)]
    saved = pd.read_pickle(outputs / "prior_models_creation110.pkl")[args.season]
    params = booster_params(cfg, f"params/booster_params_kumo_{args.size}.json")
    rows = []
    for side in ("O", "D"):
        names = list(FEATURE_SETS[SHIPPED][side])
        started = time.time()
        model = box_prior.fit_prior(train[side], names + ROW_FEATURES, params[side], n_folds=5, bags=0)
        rated = box_prior.rated_rows(panel[panel.side == side], names)
        kumo = pd.Series(box_prior.predict(model, rated), index=rated.player_id.to_numpy())
        seconds = time.time() - started
        shipped = pd.Series(saved[side]["prior"]).reindex(kumo.index)
        labels = train[side].groupby(level=0)[["label", "label_possessions"]].first()
        both = pd.DataFrame({"kumo": kumo, "shipped": shipped}).join(labels, how="inner").dropna()
        w = both.label_possessions.to_numpy(float)
        for name in ("kumo", "shipped"):
            score = weighted(both[name].to_numpy(float), both.label.to_numpy(float), w)
            rows.append(dict(side=side, prior=name, players=len(both), **score))
        agree = weighted(both.kumo.to_numpy(float), both.shipped.to_numpy(float), w)
        print(f"{side}: {len(kumo)} players rated in {seconds:.0f}s; Kumo vs shipped correlation {agree['corr']:.3f}",
              flush=True)
        # the same row asked alone and asked among the others
        matrix = rated[model.inputs].to_numpy(float)
        fresh = copy.copy(model.full)
        fresh.answers = {}
        alone = fresh.predict(matrix[:3])
        print(f"  a row asked alone vs among {len(matrix)}: largest difference "
              f"{float(np.abs(alone - model.full.predict(matrix[:3])).max()):.1e}", flush=True)
    out = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print("\nagainst the label, weighted by the possessions behind it, players with a label:")
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
