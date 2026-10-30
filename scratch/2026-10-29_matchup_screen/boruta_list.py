"""Every input screened against its shuffled copy, pass / maybe / fail by the three levels of the 2026-10-16 check:
pass when the real input beats its copy at z -2 or better, fail when it is worse at z +2 or more, maybe between."""
import pandas as pd

from openrapm.features import meaning


def level(z: float) -> str:
    return "Pass" if z <= -2 else ("Fail" if z >= 2 else "Maybe")


SCREENS = {"screen112": "Play-by-play, Oct 26", "screen119": "Fouls and turnovers, Oct 28",
           "screen120": "Matchups, Oct 29"}
rows = []
for tag, label in SCREENS.items():
    table = pd.read_csv(f"outputs/csv/screen_{tag}_table.csv")
    tuned = None
    try:
        tuned = pd.read_csv(f"outputs/csv/screen_{tag}_tune120_table.csv").set_index(["side", "input", "arm"])
    except FileNotFoundError:
        pass
    for _, x in table[table.arm == "add"].iterrows():
        row = dict(screen=label, input=x.input, name=meaning(x.input), side="Offense" if x.side == "O" else "Defense",
                   vs_copy=x.change_vs_shadow, z=x.z_vs_shadow, better=int(x.seasons_better_vs_shadow),
                   seasons=8 if tag == "screen120" else 10, verdict=level(x.z_vs_shadow), screen_rule=x.verdict)
        if tuned is not None:
            t = tuned.loc[(x.side, x.input, "add")]
            row.update(z_retuned=t.z_vs_shadow, verdict_retuned=level(t.z_vs_shadow))
        rows.append(row)
new = pd.DataFrame(rows)
order = {"Pass": 0, "Maybe": 1, "Fail": 2}
new = new.sort_values(["verdict", "z"], key=lambda c: c.map(order) if c.name == "verdict" else c)
new.to_csv("outputs/csv/boruta_list_screened.csv", index=False)
pd.set_option("display.width", 250)
print(new[["verdict", "name", "side", "vs_copy", "z", "better", "seasons", "screen", "z_retuned", "verdict_retuned",
           "screen_rule"]].round(4).to_string(index=False))

cull = pd.read_csv("outputs/csv/cull_lgb1_tune2_hustle_table.csv")
cull = cull[cull.side.isin(["offence", "defence"])].copy()
cull["vs_copy"] = -cull.shadow_change
cull["z"] = -cull.shadow_z
cull["verdict"] = cull.z.map(level)
cull["name"] = cull.input.map(meaning)
cull["side"] = cull.side.map({"offence": "Offense", "defence": "Defense"})
cull["ships"] = ~((cull.input == "three_point_pct") & (cull.side == "Offense"))
cull = cull.sort_values(["verdict", "z"], key=lambda c: c.map(order) if c.name == "verdict" else c)
cull.to_csv("outputs/csv/boruta_list_shipped.csv", index=False)
print(cull[["verdict", "name", "side", "vs_copy", "z", "shadow_seasons_better", "ships"]].round(4).to_string(index=False))
