"""The sheet's rows from coefs134.py's output: one table a side, Box-score prior on top, the rest by size of impact."""
import json
import sys

sys.stdout.reconfigure(encoding="utf-8")
d = json.load(open(sys.argv[1], encoding="utf-8"))
SAME = {"box_prior": "Box-score prior", "box_prior_other_end": "Box-score prior, opponents",
        "teammates_offence_on": "Teammates' offense, player on floor", "teammates_defence_on": "Teammates' defense, player on floor",
        "opponents_offence_on": "Opponents' offense, player on floor", "opponents_defence_on": "Opponents' defense, player on floor",
        "teammates_offence_bench": "Team's offense, player on bench", "teammates_defence_bench": "Team's defense, player on bench",
        "opponents_offence_bench": "Opponents' offense, player on bench", "opponents_defence_bench": "Opponents' defense, player on bench",
        "threes_made_per100": "Made threes per 100", "threes_missed_per100": "Missed threes per 100",
        "twos_made_per100": "Made twos per 100", "twos_missed_per100": "Missed twos per 100",
        "free_throws_made_per100": "Made free throws per 100", "free_throws_missed_per100": "Missed free throws per 100",
        "off_rebounds_per100": "Offensive rebounds per 100", "def_rebounds_per100": "Defensive rebounds per 100",
        "assists_per100": "Assists per 100", "turnovers_per100": "Turnovers per 100", "steals_per100": "Steals per 100",
        "blocks_per100": "Blocks per 100", "fouls_per100": "Personal fouls per 100",
        "team_possession_share": "Share of team possessions", "games_started_share": "Share of games started",
        "garbage_time_share": "Garbage-time share", "age": "Age, years", "height_in": "Height, inches", "weight_lb": "Weight, pounds"}
COUNT = {"threes_made": "made threes", "threes_missed": "missed threes", "twos_made": "made twos", "twos_missed": "missed twos",
         "free_throws_made": "made free throws", "free_throws_missed": "missed free throws",
         "off_rebounds": "offensive rebounds", "turnovers": "turnovers"}


def label(name, side):
    if name in SAME:
        return SAME[name]
    body = name.removeprefix("team_")
    count, where = body.rsplit("_", 1)
    who = "Team's" if side == "O" else "Opponents'"
    return f"{who} {COUNT[count]} per 100, player {'on floor' if where == 'on' else 'on bench'}"


tables = {}
for side in ("O", "D"):
    rows = sorted(d[side], key=lambda r: -abs(r["impact_inside"]))
    rows.sort(key=lambda r: r["input"] != "box_prior")
    tables[side] = [[label(r["input"], side), round(r["coefficient"], 3), round(r["impact_inside"], 2)] for r in rows]
out = [[*a, None, *b] for a, b in zip(tables["O"], tables["D"])]
print("rows", len(out))
print(json.dumps(out, ensure_ascii=False, separators=(",", ":")))
print()
for a, b in zip(tables["O"], tables["D"]):
    print(f"{a[0][:50]:50s} {a[1]:+8.3f} {a[2]:+6.2f}   {b[0][:50]:50s} {b[1]:+8.3f} {b[2]:+6.2f}")
