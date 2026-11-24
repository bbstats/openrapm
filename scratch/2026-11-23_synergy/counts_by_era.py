"""Print chosen statistics' count tables by era: mean miss after the free terms by how many of the five sit in the
fold's top fifth, positive better for the lineup, with each bin's share of possessions."""
import sys
import pandas as pd
data = "outputs/synergy/rd143"
pick = [("off", "self_creation_per100"), ("off", "used_possessions_per100"), ("off", "assists_per100_uncentred"),
        ("off", "free_throw_attempts_per100"), ("off", "blocks_per100_uncentred"), ("off", "def_rebounds_per100_uncentred"),
        ("off", "three_point_attempts_per100"),
        ("def", "height_in"), ("def", "blocks_per100_uncentred"), ("def", "def_rebounds_per100_uncentred"),
        ("def", "off_rebounds_per100_uncentred"), ("def", "transition_attempts_per100_uncentred")]
c = pd.read_csv(f"{data}/scan_counts.csv")
c["bin"], c["kind"] = c.term.str.split(" ").str[0], c.term.str.split(" ").str[1]
pd.set_option("display.width", 250)
rows = []
for end, trait in pick:
    part = c[(c.end == end) & (c.trait == trait)]
    for era, g in part.groupby("era", sort=False):
        m = g[g.kind == "mean"].set_index("bin").value
        s = g[g.kind == "share"].set_index("bin").value
        rows.append(dict(end=end, trait=trait, era=era, **{f"{b} mean": m.get(b) for b in ("0", "1", "2", "3+")},
                         **{f"{b} share": s.get(b) for b in ("0", "1", "2", "3+")}))
print(pd.DataFrame(rows).round(2).to_string(index=False))
