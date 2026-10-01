"""The trade-flag test's list: each player gets the rating that matches what he did next.

    python scripts/89_stitch_by_move.py --traded=outputs/season_ratings_flagtest.parquet
                                        --stayed=outputs/season_ratings_flagtest_stayed.parquet
                                        [--out=season_ratings_flagtest_stitched]

The owner, 2026-10-01: "are you testing the trade flag by turning it on for players who are traded?"  It was not.
Experiments 26 and 30 rate every player as if he changed teams (same_team 0), and the year-over-year test scores that
one rating for everyone -- the 64% of possessions whose player kept his team included.  `scripts/62_single_year_board.py
--rate_same_team=both` writes a second list from the same models, every player rated as if he stayed (same_team 1).
This script hands each player the one that matches what he did:

  forward   the rated season's rankings predict the season AFTER it (63_yoy.py's "prev" direction): the "traded"
            rating if his main team in the season after differs from the rated season's, the "stayed" one if not
  backward  they predict the season BEFORE it (the "next" direction): the same, against the season before

"Main team" is the one he played the most minutes for (`holdout.Context.main_team`, the rule the movers split of
scripts/88_yoy_by_player.py uses), read as a franchise (`singleyear.franchise`: Charlotte's 2002 move to New Orleans
is not a trade).  A player with no main team in the other season keeps the "traded" rating; he is either not on the
floor in the scored season or has no rating in the rated one, so the choice does not reach the test.

**Not a ranking.**  The list reads the neighbouring season's teams, which no published rating may.  It answers one
question: does the flag help when it is switched on only for the players it describes?

Writes outputs/<out>_fwd.parquet and outputs/<out>_bwd.parquet, the rankings schema with a `stayed` column added, for
`63_yoy.py --rankings=<name>=<fwd>|<bwd>` and `88_yoy_by_player.py --cands=<name>=<fwd>|<bwd>`.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _cli import check_flags, flag  # noqa: E402
from eracoef import singleyear as sy  # noqa: E402
from eracoef.config import load_config  # noqa: E402
from eracoef.holdout import Context  # noqa: E402

KEY = ["player_id", "season"]


def main_teams(ctx: Context, season: int, seasons: set) -> dict:
    """player_id -> franchise of his main team; empty for a season outside the data."""
    if season not in seasons:
        return {}
    m = ctx.main_team(season)
    ids = np.fromiter(m.keys(), dtype=np.int64, count=len(m))
    teams = sy.franchise(np.fromiter(m.values(), dtype=np.int64, count=len(m)), np.full(len(m), season))
    return dict(zip(ids.tolist(), teams.tolist()))


def kept_team(player_ids: np.ndarray, then: dict, now: dict) -> np.ndarray:
    """True where the player has the same main team in both seasons."""
    return np.array([p in then and p in now and then[p] == now[p] for p in player_ids.tolist()], dtype=bool)


def stitch(traded: pd.DataFrame, stayed: pd.DataFrame, kept: np.ndarray) -> pd.DataFrame:
    """`traded` with every column taken from `stayed` where `kept`; the two lists must cover the same rows."""
    out = traded.copy()
    for c in traded.columns:
        if c not in KEY + ["player_name"]:
            out[c] = np.where(kept, stayed[c].to_numpy(), traded[c].to_numpy())
    out["stayed"] = kept
    return out


def main() -> None:
    check_flags()
    cfg = load_config(ROOT / "config.yaml")
    traded_path, stayed_path = flag("traded"), flag("stayed")
    if not traded_path or not stayed_path:
        raise SystemExit("--traded=<rankings> and --stayed=<rankings> are required")
    name = flag("out", Path(traded_path).stem + "_stitched")
    traded = pd.read_parquet(ROOT / traded_path).sort_values(KEY, ignore_index=True)
    stayed = pd.read_parquet(ROOT / stayed_path).sort_values(KEY, ignore_index=True)
    assert traded[KEY].equals(stayed[KEY]), "the two lists must rate the same player-seasons"
    assert np.allclose(traded.poss_off, stayed.poss_off), "the two lists must come from the same games"
    seasons = set(traded.season.unique().tolist())
    ctx = Context.load(cfg)
    teams = {s: main_teams(ctx, s, seasons) for s in sorted(seasons)}
    ids, season = traded.player_id.to_numpy(np.int64), traded.season.to_numpy(int)
    for label, step in (("fwd", +1), ("bwd", -1)):
        kept = np.zeros(len(traded), dtype=bool)
        for s in sorted(seasons):
            rows = season == s
            kept[rows] = kept_team(ids[rows], teams[s], teams.get(s + step, {}))
        out = stitch(traded, stayed, kept)
        path = ROOT / "outputs" / f"{name}_{label}.parquet"
        out.to_parquet(path, index=False)
        w = traded.poss_off.to_numpy(float)
        moved = (stayed.rating_total - traded.rating_total).to_numpy(float)
        print(f"{label}: {kept.mean():.1%} of player-seasons and {w[kept].sum() / w.sum():.1%} of possessions "
              f"take the 'stayed' rating; their total moves {np.average(moved[kept], weights=w[kept]):+.3f} per 100 "
              f"on average (possession-weighted), median size {np.median(np.abs(moved[kept])):.3f}  -> "
              f"{path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
