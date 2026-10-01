"""The swap test: does a season's ranking order TEAMMATES the way the games of the seasons either side do?

    python scripts/90_swap_test.py --rankings=incumbent=outputs/season_ratings_unshrinkdef.parquet,
                                              prior=outputs/season_ratings_unshrinkdef.parquet:prior
                                   [--ref=incumbent] [--plain_rapm=3000] [--checks=1] [--shift_sd=2]
                                   [--first=1998] [--last=2025] [--contexts=full,home] [--seed=0] [--tag=swap]

The test is `src/eracoef/swaptest.py`: for every pair of lineups on the same team that share four players
-- Podziemski in one, Spencer in the other, the same four teammates in both -- compare the two lineups'
results in the scored season with what a NEIGHBOURING season's ranking says about the two swapped players.
Each scored season is scored twice, by the ranking of the season before it ("forward": a rating looking
ahead) and of the season after it ("back"), so 1998-2025 gives 56 observations, the year-over-year test's
convention.  A season's own swaps never grade its own ranking: the rating was fitted on them.

`--rankings=name=path[:columns]`: `columns` is `rating` (the default, the finished rankings) or `prior`
(the box prior alone: the `prior_off` / `prior_def` columns of the same table).
`--plain_rapm=<penalty>` is plain single-season RAPM on actual points, fitted here from the stints once per
season and kept in outputs/season_ratings_plainrapm<penalty>.parquet.  It is scored as a ranking, and the
SCORED season's own plain RAPM is what takes the opponents and the context out of each swap difference
before the `order` score reads it -- one adjustment, the same for every ranking.
`--checks=1` adds three copies of the reference that say whether the test sees what it claims to:
  shuffled   the reference's ratings dealt out again at random among each rated-season team's players: the
             order inside every team destroyed.  It must lose clearly.
  team_now   one random constant per team of the SCORED season (normal, sd `--shift_sd`, each side) added
             to every player who played for it: what the test claims to be blind to.  `order` must tie,
             apart from the few players who played for two teams.
  team_then  the same with the teams of the RATED season.  Rosters change between seasons, so this must tie
             on pairs who were teammates then too, and lose on pairs who were not: it measures how much
             of the test is a player's credit travelling with him to a new team.

Writes outputs/swaptest_<tag>.parquet (scored season x rated season x ranking x context x side x pair group)
and outputs/swaptest_<tag>_levels.parquet (the coefficients of the level refit on each scored season).
"""
import os
import sys
import time
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _cli import check_flags, flag  # noqa: E402
from eracoef import swaptest as sw  # noqa: E402
from eracoef.config import load_config  # noqa: E402


def load_stints(season: int, cfg: dict, phases=("RS", "PO")) -> pd.DataFrame:
    """A season's stints, regular season then playoffs, with the two teams' ids from the diagnostics."""
    folder = ROOT / cfg.get("paths", {}).get("stints", "data/stints")
    frames = []
    for phase in phases:
        path = folder / f"{season}_{phase}.parquet"
        if not path.exists():
            continue
        st = pd.read_parquet(path, columns=sw.STINT_COLUMNS)
        diag = pd.read_parquet(folder / f"{season}_{phase}_diag.parquet",
                               columns=["game_id", "home_team_id", "away_team_id"]).drop_duplicates("game_id")
        st = st.merge(diag, on="game_id", how="left", sort=False)
        if st.home_team_id.isna().any():
            raise SystemExit(f"{path.name}: a stint's game has no row in the diagnostics; rebuild the stints")
        frames.append(st)
    if not frames:
        raise SystemExit(f"no stints on disk for {season}")
    return pd.concat(frames, ignore_index=True)


def load_table(path: Path, columns: str = "rating") -> pd.DataFrame:
    """player_id, season, o, d in RAW sign (d = points allowed), from a table stored positive-good."""
    t = pd.read_parquet(path)
    need = ["player_id", "season", f"{columns}_off", f"{columns}_def"]
    missing = [c for c in need if c not in t.columns]
    if missing:
        raise SystemExit(f"{path}: missing {missing}")
    return pd.DataFrame({"player_id": t.player_id.astype(np.int64), "season": t.season.astype(int),
                         "o": t[f"{columns}_off"].astype(float), "d": -t[f"{columns}_def"].astype(float)})


def main_teams() -> pd.DataFrame:
    """player_id, season, team_id: the team he played the most possessions for that season."""
    roles = pd.read_parquet(ROOT / "data/cache/roles_RSPO.parquet", columns=["player_id", "season", "team_id", "poss_on"])
    return (roles[roles.poss_on > 0].groupby(["player_id", "season", "team_id"], as_index=False).poss_on.sum()
            .sort_values("poss_on").drop_duplicates(["player_id", "season"], keep="last")
            [["player_id", "season", "team_id"]].astype(np.int64))


def shuffled_within_team(table: pd.DataFrame, teams: pd.DataFrame, seed: int) -> pd.DataFrame:
    """The same numbers dealt out again at random among each team-season's players, (o, d) kept together."""
    rng = np.random.default_rng(seed)
    out = table.merge(teams, on=["player_id", "season"], how="left").reset_index(drop=True)
    for _, idx in out[out.team_id.notna()].groupby(["season", "team_id"]).groups.items():
        idx = np.asarray(idx)
        out.loc[idx, ["o", "d"]] = out.loc[rng.permutation(idx), ["o", "d"]].to_numpy()
    return out.drop(columns="team_id")


def shifted_by_team(table: pd.DataFrame, team_of: dict, seed: int, sd: float) -> pd.DataFrame:
    """One random constant per team and side added to every player `team_of` puts on that team."""
    rng = np.random.default_rng(seed)
    out = table.copy()
    team = out.player_id.map(team_of)
    for t in sorted(team.dropna().unique()):
        m = (team == t).to_numpy()
        out.loc[m, "o"] += rng.normal(0.0, sd)
        out.loc[m, "d"] += rng.normal(0.0, sd)
    return out


def paired_line(part: pd.DataFrame, ranking: str, ref: str, value: str, better: str) -> str:
    """mean difference ranking - ref over the observations, its paired z, and how many it wins."""
    piv = part.pivot_table(index=["scored", "rated"], columns="ranking", values=value)
    if ranking not in piv or ref not in piv:
        return "--"
    diff = (piv[ranking] - piv[ref]).dropna().to_numpy()
    se = diff.std(ddof=1) / np.sqrt(len(diff)) if len(diff) > 1 else np.nan
    z = diff.mean() / se if se and se > 0 else np.nan
    wins = int((diff > 0).sum() if better == "higher" else (diff < 0).sum())
    return f"{diff.mean():+.4f} z {z:+.1f}, {wins} of {len(diff)}"


def report(res: pd.DataFrame, levels: pd.DataFrame, ref: str) -> None:
    pd.set_option("display.width", 250, "display.max_columns", 30)
    names = list(dict.fromkeys(res.ranking))
    for context in dict.fromkeys(res.context):
        print(f"\n================ context: {context} ({', '.join(sw.CONTEXTS[context])})")
        for side in ("net", "offense", "defense"):
            for group in sw.PAIR_GROUPS:
                part = res[(res.context == context) & (res.side == side) & (res.group == group)]
                if part.empty or (group != "all" and (side != "net" or context != "full")):
                    continue
                print(f"\n--- {side}, {group} pairs: {part.pairs.mean():,.0f} swap pairs and {part.information.mean():,.0f} "
                      f"possessions of information per observation, {part.groupby(['scored', 'rated']).ngroups} observations")
                lines = []
                for name in names:
                    p = part[part.ranking == name]
                    lines.append(dict(ranking=name, order=p.order.mean(), gaps=p.gaps.mean(), slope=p.slope.mean(),
                                      order_vs_ref="" if name == ref else paired_line(part, name, ref, "order", "higher"),
                                      gaps_vs_ref="" if name == ref else paired_line(part, name, ref, "gaps", "lower")))
                print(pd.DataFrame(lines).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        net = res[(res.context == context) & (res.side == "net") & (res.group == "all")]
        for direction, label in (("forward", "a ranking predicting the season AFTER it"),
                                 ("back", "a ranking predicting the season BEFORE it")):
            part = net[net.direction == direction]
            text = "; ".join(f"{n} {paired_line(part, n, ref, 'order', 'higher')}" for n in names if n != ref)
            print(f"    net order vs {ref}, {label}: {text}")
    print("\n    order: the swap difference with opponents and context taken out, signed by the ranking's gap between")
    print("    the two swapped players, information-weighted, points per 100 -- HIGHER is better, 0 = no idea, spread-free.")
    print("    gaps: squared error of the ranking's predicted swap difference -- LOWER is better.  slope: what the swaps")
    print("    ask the ranking's gaps inside a team to be multiplied by (below 1 = too wide).  'x of n' = observations won.")
    lv = levels[(levels.ranking == ref) & (levels.context == "full")]
    if len(lv):
        print(f"\n    the level refit on the scored seasons ({ref}, full context), mean and spread over observations:")
        for c in ("fatigue_off", "fatigue_def", "period_elapsed", "period_end", "clutch", "is_gt", "home"):
            print(f"      {c:12s} {lv[c].mean():+.3f}  (sd {lv[c].std():.3f})  points per 100"
                  f"{' per minute on the court' if c.startswith('fatigue') else ''}")


def main():
    check_flags()
    cfg = load_config(ROOT / "config.yaml")
    spec = flag("rankings", "")
    if not spec:
        raise SystemExit(__doc__)
    tables = {}
    for part in spec.replace("\n", "").split(","):
        if not part.strip():
            continue
        name, _, rest = part.strip().partition("=")
        columns = "rating"
        if rest.endswith(":prior") or rest.endswith(":rating"):
            rest, columns = rest.rsplit(":", 1)
        tables[name] = load_table(ROOT / rest if not Path(rest).is_absolute() else Path(rest), columns)
    ref = flag("ref", next(iter(tables)))
    assert ref in tables, f"--ref={ref} is not one of {list(tables)}"
    lam = float(flag("plain_rapm", "3000"))
    assert lam > 0, "--plain_rapm must be positive: the scored season's plain RAPM is the swap adjustment"
    checks = flag("checks", "1") not in ("0", "no", "false")
    seed = int(flag("seed", "0"))
    shift_sd = float(flag("shift_sd", "2"))
    contexts = [c for c in flag("contexts", "full,home").split(",") if c]
    assert all(c in sw.CONTEXTS for c in contexts), contexts
    first, last = int(flag("first", "1998")), int(flag("last", "2025"))
    tag = flag("tag", "swap")

    print(f"rankings: {', '.join(tables)}, plain_rapm{', shuffled, team_now, team_then' if checks else ''}; "
          f"reference {ref}; scored seasons {first}-{last}; contexts {contexts}")
    t0 = time.time()
    res, lev = run(tables, ref, cfg, contexts, first, last, lam, checks=checks, seed=seed, shift_sd=shift_sd)
    res.to_parquet(ROOT / "outputs" / f"swaptest_{tag}.parquet", index=False)
    lev.to_parquet(ROOT / "outputs" / f"swaptest_{tag}_levels.parquet", index=False)
    print(f"wrote outputs/swaptest_{tag}.parquet and _levels ({time.time() - t0:.0f}s)")
    report(res, lev, ref)


def run(tables: dict, ref: str, cfg: dict, contexts, first: int, last: int, lam: float, checks: bool = False,
        seed: int = 0, shift_sd: float = 2.0, score_plain: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score every table on the swaps of every scored season in [first, last], both directions.

    `tables` maps a name to (player_id, season, o, d), raw sign.  The scored season's own plain RAPM at penalty
    `lam` is the swap adjustment; it is fitted here (or read from outputs/season_ratings_plainrapm<lam>.parquet)
    and, with `score_plain`, scored as a ranking too.  Returns (scores, levels) in score_season's shape plus
    scored, rated and direction."""
    tables = dict(tables)
    s0, s1 = int(cfg["first_season"]), int(cfg["last_season"])
    margin_clip = float(cfg.get("margin_clip", 25))
    teams = main_teams()
    team_of = {s: dict(zip(g.player_id, g.team_id)) for s, g in teams.groupby("season")}
    if checks:
        tables["shuffled"] = shuffled_within_team(tables[ref], teams, seed)
    rapm_path = ROOT / "outputs" / f"season_ratings_plainrapm{lam:.0f}.parquet"
    rapm_cached = load_table(rapm_path) if rapm_path.exists() else None
    cached_seasons = set() if rapm_cached is None else set(rapm_cached.season.unique())
    rapm_rows, rapm_fits = [], []

    t0 = time.time()
    held: dict = {}                 # season -> prepared Season, kept until both its neighbours are rated
    res, lev = [], []
    for s in range(max(s0, first - 1), min(s1, last + 1) + 1):
        season = sw.prepare(sw.season_rows(load_stints(s, cfg), margin_clip=margin_clip), s)
        if s in cached_seasons:
            fit = rapm_cached[rapm_cached.season == s]
        else:
            fit = sw.plain_rapm(season, lam).assign(season=s)
            rapm_rows.append(fit)
        rapm_fits.append(fit[["player_id", "season", "o", "d"]])
        plain = pd.concat(rapm_fits, ignore_index=True)
        held[s] = season
        scored = s - 1                                     # every neighbour of s - 1 is now rated
        if first <= scored <= last and scored in held:
            S = held[scored]
            common = plain[plain.season == scored]
            for rated, direction in ((scored - 1, "forward"), (scored + 1, "back")):
                if not (s0 <= rated <= s1):
                    continue
                arms = {n: t[t.season == rated] for n, t in tables.items()}
                if score_plain:
                    arms["plain_rapm"] = plain[plain.season == rated]
                if checks:
                    arms["team_now"] = shifted_by_team(arms[ref], team_of.get(scored, {}), seed + 1000 * scored, shift_sd)
                    arms["team_then"] = shifted_by_team(arms[ref], team_of.get(rated, {}), seed + 1000 * rated + 7, shift_sd)
                rated_set = set.intersection(*(set(a.player_id.astype(np.int64)) for a in arms.values()))
                sc, lv = sw.score_season(S, arms, rated_set, common, contexts=contexts,
                                         team_then=team_of.get(rated, {}))
                res.append(sc.assign(scored=scored, rated=rated, direction=direction))
                lev.append(lv.assign(scored=scored, rated=rated, direction=direction))
            print(f"  scored {scored} ({time.time() - t0:.0f}s)", flush=True)
            del held[scored]
        for old in [k for k in held if k < s - 1]:
            del held[old]

    if rapm_rows:
        new = pd.concat(rapm_rows, ignore_index=True)
        out = pd.DataFrame({"player_id": new.player_id.astype(np.int64), "season": new.season.astype(int),
                            "rating_off": new.o, "rating_def": -new.d, "poss_off": new.poss})
        out["rating_total"] = out.rating_off + out.rating_def
        if rapm_path.exists():
            old = pd.read_parquet(rapm_path)
            out = pd.concat([old[~old.season.isin(out.season.unique())], out], ignore_index=True)
        out.sort_values(["season", "player_id"]).to_parquet(rapm_path, index=False)
        print(f"wrote {rapm_path.relative_to(ROOT)}")
    return pd.concat(res, ignore_index=True), pd.concat(lev, ignore_index=True)


if __name__ == "__main__":
    main()
