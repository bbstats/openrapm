"""Check a floor_<season>_<phase>.parquet file: every shooter is one of his team's five, every blocker one of the
other team's five, every fouler on the floor.

    .venv/Scripts/python scratch/2026-11-25_closest_defender/shared/check_floor.py floor_2015_RS
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
HOME = ["h1", "h2", "h3", "h4", "h5"]
AWAY = ["a1", "a2", "a3", "a4", "a5"]

table = pd.read_parquet(HERE / f"{sys.argv[1]}.parquet")
home = table[HOME].to_numpy()
away = table[AWAY].to_numpy()
person = table.person_id.to_numpy()[:, None]
in_home = (home == person).any(axis=1)
in_away = (away == person).any(axis=1)
team_is_home = (table.team_id == table.home_id).to_numpy()

shots = table.action_type.isin(["Made Shot", "Missed Shot"]).to_numpy() & table.valid.to_numpy()
own_five = np.where(team_is_home, in_home, in_away)
print(f"field-goal attempts with ten known players: {shots.sum()}")
print(f"  shooter among his team's five: {own_five[shots].mean():.4f}")

blocks = (table.action_type == "").to_numpy() & table.description.str.contains("BLOCK").to_numpy() & table.valid.to_numpy()
print(f"block rows: {blocks.sum()}")
print(f"  blocker on the floor: {(in_home | in_away)[blocks].mean():.4f}")
shot_rows = table[shots][["game_id", "action_number", "team_id", "person_id"]].rename(
    columns={"team_id": "shot_team", "person_id": "shooter"})
block_rows = table[blocks][["game_id", "action_number", "person_id"]].rename(columns={"person_id": "blocker"})
both = block_rows.merge(shot_rows, on=["game_id", "action_number"], how="left")
print(f"  block rows sharing action_number with a field-goal attempt: {both.shooter.notna().mean():.4f}")
missed = table[(table.action_type == "Missed Shot")]
print(f"  missed shots: {len(missed)}, blocked share: {len(block_rows) / max(len(missed), 1):.4f}")

fouls = (table.action_type == "Foul").to_numpy() & table.valid.to_numpy()
shooting = fouls & table.sub_type.str.contains("Shooting").to_numpy()
print(f"foul rows: {fouls.sum()}, shooting fouls: {shooting.sum()}")
print(f"  shooting fouler on the floor: {(in_home | in_away)[shooting].mean():.4f}")
print(table.sub_type[fouls].value_counts().head(12).to_string())
