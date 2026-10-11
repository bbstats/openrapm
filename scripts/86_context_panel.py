"""Write each player-season's playoff share into the season panel, beside the on-court columns.

    python scripts/86_context_panel.py [--panel=outputs/role_panel_season.parquet]

  po_share   the share of his possessions that came in the playoffs: `roles.parquet` (regular season) against
             `roles_RSPO.parquet` (both), both in paths.cache, on `poss_on`.  A share, like the closeness
             columns, and like them not padded.  Written on both side rows of a player-season.

This script also wrote the Decomposition page's RAPM pieces (`pc_*`, experiment 26) until the cleanup deleted the
research code that built them (phase 4): no shipped input reads them, and a panel that has them keeps them.

The panel is backed up to `<panel>_pre_context.parquet` before it is overwritten.
"""
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _cli import check_flags, flag  # noqa: E402
from openrapm.data.config import load_config  # noqa: E402
from openrapm.data.roles import roles_path  # noqa: E402

PO_SHARE = "po_share"


def playoff_share() -> pd.DataFrame:
    """player_id, season, po_share: 1 - regular-season possessions / all possessions, summed over his teams."""
    both_path = roles_path(load_config())
    # The regular-season table is the unsuffixed roles.parquet an older build wrote beside the pooled one, before
    # the phases went into the file name; nothing rebuilds it today.
    regular_path = both_path.with_name("roles.parquet")

    def total(path):
        r = pd.read_parquet(path, columns=["player_id", "season", "poss_on"])
        return r[r.poss_on > 0].groupby(["player_id", "season"]).poss_on.sum()
    both, regular = total(both_path), total(regular_path)
    share = (1.0 - regular.reindex(both.index).fillna(0.0) / both).clip(0.0, 1.0)
    return share.rename(PO_SHARE).reset_index()


def main() -> None:
    check_flags()
    path = ROOT / flag("panel", "outputs/role_panel_season.parquet")
    panel = pd.read_parquet(path)
    share = playoff_share()
    backup = path.with_name(path.stem + "_pre_context.parquet")
    if not backup.exists():
        shutil.copy(path, backup)
    panel = panel.drop(columns=[PO_SHARE], errors="ignore")
    before = len(panel)
    panel = panel.merge(share, on=["player_id", "season"], how="left")
    if len(panel) != before:
        raise SystemExit("the merge added rows: the playoff shares are not one per player-season")
    print(f"  {int(panel[PO_SHARE].isna().sum())} have no playoff share (no possessions in the roles table); set to 0")
    panel[PO_SHARE] = panel[PO_SHARE].fillna(0.0)
    panel.to_parquet(path, index=False)
    wide = panel[(panel.side == "O") & (panel.poss >= 1000)]
    print(f"wrote {path.relative_to(ROOT)}: {len(panel)} rows")
    print(f"  po_share among 1,000+ possession player-seasons (offence rows): mean {wide[PO_SHARE].mean():.3f}, "
          f"share with any playoff possessions {(wide[PO_SHARE] > 0).mean():.1%}, max {wide[PO_SHARE].max():.3f}")


if __name__ == "__main__":
    main()
