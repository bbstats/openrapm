"""The NBA's potential assists against the play-by-play, league totals by season, regular season: the share of made
shots assisted, the share of missed shots potentially assisted, and free-throw assists per free-throw trip."""
import pandas as pd

rows = []
for season in range(2014, 2027):
    p = pd.read_parquet(f"data/raw/dashboards/tracking/pt_passing/{season}_RS.parquet")
    s = pd.read_parquet(f"data/shotframe/{season}_RS.parquet", columns=["made"])
    makes, misses = int(s.made.sum()), int((s.made == 0).sum())
    ast, pot, ft_ast = p.AST.sum(), p.POTENTIAL_AST.sum(), p.FT_AST.sum()
    rows.append({"season": season, "makes_assisted_share": ast / makes,
                 "misses_potentially_assisted_share": (pot - ast) / misses,
                 "ft_assists": ft_ast, "potential_assists": pot, "assists": ast, "makes": makes, "misses": misses})
print(pd.DataFrame(rows).round(3).to_string(index=False))
