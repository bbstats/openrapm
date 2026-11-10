"""Read-only: is 2027's team Power, before any multiplier, as wide as the finished seasons' were at the same point?

Check 166 fits the Power table's multipliers on finished seasons rated from their first tenth and applies them to
the season in progress.  They carry over only if the two are built alike: if 2027's published ratings summed by team
were wider than the fit's, the multipliers would overshoot.  This prints the spread across teams of Power's offense
and defense, each season at a tenth and at 17.5% of its games, beside 2027's on the published table.

    .venv/Scripts/python scratch/2026-11-09_improvements/power_spread.py
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.stdout.reconfigure(encoding="utf-8")


def load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / "checks" / file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


C166 = load("check166", "166_power_forecast_refit.py")
C159 = C166.C159
from openrapm.data.config import load_config, resolve  # noqa: E402


def spreads(folder: str, cfg: dict, within: Path, wowy, stints: dict, teams: dict) -> pd.DataFrame:
    """One row a season: the standard deviation across teams of Power's offense and defense on the day `folder`."""
    rows = []
    for season, (rated, played) in sorted(C159.day_ratings(wowy, within, folder, step4="published").items()):
        power = C159.power_on_day(rated, played, stints[season], teams[season])
        share = len(played) / stints[season].game_id.nunique()
        rows.append(dict(season=season, share=share, teams=len(power), offense=power.power_o.std(),
                         defense=power.power_d.std(), total=(power.power_o + power.power_d).std()))
    return pd.DataFrame(rows).set_index("season")


def main() -> None:
    cfg = load_config()
    within = ROOT / "outputs" / "within"
    wowy = C159.load_wowy()
    stints = {season: C159.season_stints_ids(season, cfg) for season in C159.SEASONS}
    teams = {season: C166.with_unlogged_players(C159.regular_game_teams(season, cfg), stints[season])
             for season in C159.SEASONS}
    pd.set_option("display.width", 200)
    for cut, folder in (("10%", f"{C166.FAMILY}_cut10_held"), ("17.5%", f"{C166.FAMILY}_cut175_held")):
        table = spreads(folder, cfg, within, wowy, stints, teams)
        print(f"\nfinished seasons at the {cut} day, Power before any multiplier, standard deviation across teams")
        eras = {"1997-2006": table.loc[1997:2006], "2007-2016": table.loc[2007:2016], "2017-2026": table.loc[2017:2026],
                "all 30": table}
        print(pd.DataFrame({name: part[["share", "offense", "defense", "total"]].mean() for name, part in eras.items()})
              .T.round(3).to_string())
        print("  the seasons' own range: offense "
              f"{table.offense.min():.2f} to {table.offense.max():.2f}, defense {table.defense.min():.2f} to "
              f"{table.defense.max():.2f}, total {table.total.min():.2f} to {table.total.max():.2f}")

    # the season in progress, on the published table, through the games its stints hold
    published = pd.read_parquet(resolve(cfg, "outputs") / "season_ratings_product.parquet")
    rated = published[published.season == 2027][["player_id", "rating_off", "rating_def"]]
    now_stints = C159.season_stints_ids(2027, cfg)
    now_teams = C159.regular_game_teams(2027, cfg)
    played = set(now_stints.game_id.unique())
    power = C159.power_on_day(rated, played, now_stints, now_teams)
    print(f"\n2027 on the published table, {len(played)} games, {len(played) / 1230:.3f} of the season: offense "
          f"{power.power_o.std():.3f}, defense {power.power_d.std():.3f}, total "
          f"{(power.power_o + power.power_d).std():.3f}")
    np.set_printoptions(suppress=True)


if __name__ == "__main__":
    main()
