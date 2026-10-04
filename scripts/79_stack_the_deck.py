"""Stack the deck: search for any league where truly bad backups inflate a starter's RAPM, and what causes it.

    python scripts/79_stack_the_deck.py [--configs=300] [--seed=0]

**Why this exists.**  78 built synthetic leagues calibrated to 2024-26 and found that making a starter's backups
one point worse raises his plain RAPM by at most +0.07 (raw on/off: +0.41), and a worse bench lowers the
starters.  This script leans on every lever that could make the belief true -- more collinearity, less data,
backups who barely play -- across hundreds of random leagues and the most extreme corners, and reports whether
it ever happens, how big it gets, and which levers do it.

**The levers**, each drawn at random per league (the 2024-26 value in brackets):
  position strictness  0 = anyone replaces anyone; 1 = a backup only ever replaces his own starter
  lines                -0.25 = staggered: one position rests at a time and the other starters stay on, so a
                       backup plays with the starter's own teammates (the textbook case); 0 = independent;
                       0.95 = hockey lines, starters together and bench units together
  seasons              1 to 3 [3] -- less data, fewer roster changes
  roster stability     share of players on the same team next season, 0.50 [0.50] to 0.95
  injuries             0 = nobody misses a game, 1 = the real rate [1]
  minutes noise        game-to-game randomness in a player's minutes, 0 to 0.4 [0.2]
  starter minutes      multiplier on the starters' minutes, 0.8 to 1.5 [1]; 1.5 puts them near 45 and leaves
                       the backups a few minutes -- the rotation truly bad backups get
  penalty              300 to 160,000 [40,000, the project's plain RAPM]

**Why "truly bad" is not a lever.**  Every test is linear in the backups' true ratings: backups 4 points worse
move a starter exactly 4 times as far as backups 1 point worse.  So the search looks for the largest pass-through
per point, and the report multiplies it out (Jokic's backups in 2024-26 were 2.0 worse than a typical replacement).

**The tests** (78's, at each league's own penalty; exact and truth-free; positive = the belief):
  own backups   the most-used player at a position: the position's other players one point worse
  share kept    the lineup that replaces a starter one point worse (raw on/off moves exactly 1)
  team bench    everyone outside the top five one point worse: the starters' move
Medians over starters, with the 90th percentile for the most-affected starters.

Writes outputs/stack_the_deck.parquet and outputs/stack_the_deck_phone.html.
"""
import html
import importlib.util
import os
import sys
import time
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
os.environ.setdefault("NUMBA_NUM_THREADS", os.environ.get("OPENRAPM_NUMBA_THREADS", "4"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.tree import DecisionTreeRegressor, export_text  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _cli import check_flags, flag  # noqa: E402
from eracoef.config import load_config  # noqa: E402

# 78's league builder and tests (and through it 77's fit) -- one definition of each
_spec = importlib.util.spec_from_file_location("_borrowed_78", ROOT / "scripts" / "78_synthetic_league.py")
s78 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s78)
r77 = s78.r77
check = r77.check

LEVERS = {"position strictness": "slot_p", "lines": "lines", "seasons": "seasons", "roster stability": "stay",
          "injuries": "injury", "minutes noise": "share_noise", "starter minutes": "starter_boost",
          "penalty (log10)": "log_penalty"}
MOVE_SHARE = s78.MOVE / (1.0 - s78.STAY)   # of the players who leave a team, the share who join another (real)
GAPS = (2.0, 4.0)                           # backups this many points worse than a typical replacement
NOTHING = 0.1
CORNERS = {
    "like 2024-26": dict(slot_p=0.5, lines=0.3, seasons=3, stay=0.50, injury=1.0, share_noise=0.20,
                         starter_boost=1.0),
    "textbook: staggered positions": dict(slot_p=1.0, lines=-0.25, seasons=1, stay=0.95, injury=0.0,
                                          share_noise=0.0, starter_boost=1.0),
    "textbook + backups barely play": dict(slot_p=1.0, lines=-0.25, seasons=1, stay=0.95, injury=0.0,
                                           share_noise=0.0, starter_boost=1.5),
    "hockey lines": dict(slot_p=1.0, lines=0.95, seasons=1, stay=0.95, injury=0.0, share_noise=0.0,
                         starter_boost=1.0),
    "hockey lines + backups barely play": dict(slot_p=1.0, lines=0.95, seasons=1, stay=0.95, injury=0.0,
                                               share_noise=0.0, starter_boost=1.5),
}
CORNER_PENALTIES = (300.0, 3000.0, 40000.0)
# one lever at a time, both ways: from the textbook corner back to its 2024-26 value (which lever kills the
# effect?), and from the 2024-26-like league to its textbook value (which lever makes it?)
ABLATE = {"position strictness": "slot_p", "lines": "lines", "seasons": "seasons", "roster stability": "stay",
          "injuries": "injury", "minutes noise": "share_noise"}
ABLATION_PENALTIES = (300.0, 3000.0)


def ablations() -> list:
    textbook, real = CORNERS["textbook: staggered positions"], CORNERS["like 2024-26"]
    out = [dict(kind=f"textbook, {word} as 2024-26", **{**textbook, key: real[key]}) for word, key in ABLATE.items()]
    out += [dict(kind=f"2024-26, {word} as textbook", **{**real, key: textbook[key]}) for word, key in ABLATE.items()]
    return out


def random_config(rng) -> dict:
    return dict(kind="random", slot_p=float(rng.uniform(0.0, 1.0)), lines=float(rng.uniform(-0.25, 0.95)),
                seasons=int(rng.integers(1, 4)), stay=float(rng.uniform(0.5, 0.95)),
                injury=float(rng.uniform(0.0, 1.0)), share_noise=float(rng.uniform(0.0, 0.4)),
                starter_boost=float(rng.uniform(0.8, 1.5)),
                penalty=float(np.exp(rng.uniform(np.log(300.0), np.log(160000.0)))))


def measure_league(block, team_seasons, ops, rep, valid, penalties) -> list:
    """78's three tests on one league at each penalty: medians over starters, and the 90th percentile."""
    n = block.n
    tests = s78.group_tests(team_seasons, positions=True)
    starters = np.unique(np.concatenate([ts["ids"][:5] for ts in team_seasons]))
    starters = starters[valid[starters]]
    out = []
    for lam in penalties:
        _, _, M = r77.fit(block, lam, lam)
        kept = s78.share_kept(M[:n], M[n:2 * n], rep, n)[starters]
        own_r, own_o = (np.concatenate(v) for v in s78.group_response(M, ops, n, tests["own backups"]))
        bench_r, bench_o = (np.concatenate(v) for v in s78.group_response(M, ops, n, tests["team bench"]))
        out.append(dict(penalty=lam, own=float(np.median(own_r)), own_p90=float(np.percentile(own_r, 90)),
                        own_onoff=float(np.median(own_o)), kept=float(np.median(kept)),
                        kept_p90=float(np.percentile(kept, 90)), bench=float(np.median(bench_r)),
                        bench_onoff=float(np.median(bench_o)),
                        keeps=float(np.median(0.5 * (np.diagonal(M)[starters] + np.diagonal(M)[n + starters])))))
    return out


def f2(x) -> str:
    return "" if x is None or not np.isfinite(x) else f"{x:+.2f}"


def pct(x) -> str:
    return "" if x is None or not np.isfinite(x) else f"{100 * x:.0f}%"


def main() -> None:
    check_flags()
    n_configs = int(flag("configs", 300))
    rng = np.random.default_rng(int(flag("seed", 0)))
    cfg = load_config()
    log: list = []
    started = time.time()

    print("real 2024-26 targets and the minutes calibration (as 78)", flush=True)
    targets = s78.real_targets()
    real = s78.real_reference(cfg, targets, log)
    cal = s78.calibrate(targets, rng, log)

    configs = ([dict(kind=name, group="corner", **c) for name, c in CORNERS.items()]
               + [dict(group="ablation", **c) for c in ablations()]
               + [dict(group="random", **random_config(rng)) for _ in range(n_configs)])
    rows = []
    for k, c in enumerate(configs):
        t0 = time.time()
        params = dict(seasons=c["seasons"], stay=c["stay"], move=(1.0 - c["stay"]) * MOVE_SHARE,
                      injury=c["injury"], share_noise=c["share_noise"], starter_boost=c["starter_boost"])
        truth, seasons, _, team_seasons = s78.make_league(targets, real, cal, (c["slot_p"], c["lines"]), rng, params)
        for season_rows in seasons:
            season_rows["y"] = s78.outcomes(truth, season_rows, real["sigma_pp"], rng)
        block = s78.league_block(truth, seasons)
        ops, rep, shared, valid = s78.structure(block, log, f"league {k} ({c['kind']})", strict=True)
        split = s78.replacement_split(ops, rep, shared, valid)
        minutes = 48.0 * float(np.mean([ts["share"][:5].mean() for ts in team_seasons]))
        penalties = {"random": (c.get("penalty"),), "corner": CORNER_PENALTIES,
                     "ablation": ABLATION_PENALTIES}[c["group"]]
        for m in measure_league(block, team_seasons, ops, rep, valid, penalties):
            rows.append({**{key: value for key, value in c.items() if key != "penalty"}, **m, "league": k,
                         "backups_time": float(split[0]), "starter_min": minutes, "players": block.n})
        last = rows[-1]
        print(f"  {k:>3} {c['kind'][:34]:<34} positions {c['slot_p']:.2f} lines {c['lines']:+.2f} seasons "
              f"{c['seasons']} stay {c['stay']:.2f} injuries {c['injury']:.2f} noise {c['share_noise']:.2f} "
              f"starters {minutes:4.1f} min, penalty {last['penalty']:>9,.0f}: own backups {f2(last['own'])} "
              f"(90th {f2(last['own_p90'])}), share kept {f2(last['kept'])} ({time.time() - t0:.0f}s)", flush=True)

    tab = pd.DataFrame(rows)
    tab["log_penalty"] = np.log10(tab.penalty)
    out = ROOT / "outputs"
    tab.to_parquet(out / "stack_the_deck.parquet", index=False)
    print("wrote outputs/stack_the_deck.parquet", flush=True)

    # ---------------------------------------------------------------------------------- what drives it
    rnd = tab[tab.kind == "random"].reset_index(drop=True)
    X = rnd[list(LEVERS.values())].astype(float)
    Z = ((X - X.mean()) / X.std()).to_numpy()
    effects = []
    for metric in ("own", "kept"):
        y = rnd[metric].to_numpy()
        A = np.column_stack([np.ones(len(Z)), Z])
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        r2 = 1.0 - np.sum((y - A @ coef) ** 2) / np.sum((y - y.mean()) ** 2)
        effects.append((metric, coef[1:], r2))
    lo, hi = X.quantile(1 / 3), X.quantile(2 / 3)
    effect_view = pd.DataFrame({
        "lever": list(LEVERS),
        "range": [f"{X[c].min():.2f} to {X[c].max():.2f}" for c in LEVERS.values()],
        "own backups: per +1 sd": [f2(v) for v in effects[0][1]],
        "own backups: bottom third": [f2(rnd.own[X[c] <= lo[c]].median()) for c in LEVERS.values()],
        "own backups: top third": [f2(rnd.own[X[c] >= hi[c]].median()) for c in LEVERS.values()],
        "share kept: per +1 sd": [f2(v) for v in effects[1][1]],
        "share kept: top third": [f2(rnd.kept[X[c] >= hi[c]].median()) for c in LEVERS.values()]})
    tree = DecisionTreeRegressor(max_depth=3, min_samples_leaf=15, random_state=0).fit(X, rnd.own)
    rules = export_text(tree, feature_names=list(LEVERS), decimals=2)

    def config_view(frame):
        return pd.DataFrame({
            "kind": frame.kind, "positions": frame.slot_p.map(lambda v: f"{v:.2f}"),
            "lines": frame.lines.map(lambda v: f"{v:+.2f}"), "seasons": frame.seasons,
            "stay": frame.stay.map(lambda v: f"{v:.2f}"), "injuries": frame.injury.map(lambda v: f"{v:.2f}"),
            "noise": frame.share_noise.map(lambda v: f"{v:.2f}"), "starter min": frame.starter_min.map(lambda v: f"{v:.0f}"),
            "penalty": frame.penalty.map(lambda v: f"{v:,.0f}"), "backup time <10% shared": frame.backups_time.map(pct),
            "own backups": frame.own.map(f2), "90th": frame.own_p90.map(f2), "raw on/off": frame.own_onoff.map(f2),
            **{f"if backups {g:.0f} worse": (frame.own * g).map(f2) for g in GAPS},
            "share kept": frame.kept.map(f2), "team bench": frame.bench.map(f2), "keeps own": frame.keeps.map(pct)})

    corners = tab[tab.group == "corner"]
    ablated = tab[tab.group == "ablation"]
    ablation_view = pd.DataFrame({
        "change": ablated.kind.unique(),
        **{f"own backups at {lam:,.0f}": [f2(ablated[(ablated.kind == k) & (ablated.penalty == lam)].own.iloc[0])
                                          for k in ablated.kind.unique()] for lam in ABLATION_PENALTIES},
        **{f"share kept at {lam:,.0f}": [f2(ablated[(ablated.kind == k) & (ablated.penalty == lam)].kept.iloc[0])
                                         for k in ablated.kind.unique()] for lam in ABLATION_PENALTIES},
        "backup time <10% shared": [pct(ablated[ablated.kind == k].backups_time.iloc[0]) for k in ablated.kind.unique()]})
    top = rnd.sort_values("own", ascending=False).head(10)
    share_pos = float((rnd.own >= NOTHING).mean())
    share_p90 = float((rnd.own_p90 >= NOTHING).mean())
    best = rnd.loc[rnd.own.idxmax()]
    best_corner = corners.loc[corners.own.idxmax()]
    lines = [
        f"{len(rnd)} random leagues: a starter's own backups one point worse raise his RAPM by a median "
        f"{f2(rnd.own.median())}; at least {NOTHING} per point in {pct(share_pos)} of leagues (the 90th-percentile "
        f"starter reaches it in {pct(share_p90)}).",
        f"Largest random league: {f2(best.own)} per point (90th percentile {f2(best.own_p90)}), with raw on/off "
        f"{f2(best.own_onoff)}; backups 4 points worse would add {f2(4 * best.own)}.",
        f"Largest corner: '{best_corner.kind}' at penalty {best_corner.penalty:,.0f}: {f2(best_corner.own)} per point "
        f"(90th {f2(best_corner.own_p90)}); backups 4 points worse would add {f2(4 * best_corner.own)}.",
        f"What the levers explain: own backups R-squared {effects[0][2]:.2f}, share kept {effects[1][2]:.2f}."]
    print("\n" + "\n".join(lines))
    print("\nthe corners\n" + config_view(corners).to_string(index=False))
    print("\none lever at a time\n" + ablation_view.to_string(index=False))
    print("\nthe ten random leagues where backups reach the starter most\n" + config_view(top).to_string(index=False))
    print("\nwhat each lever does (random leagues)\n" + effect_view.to_string(index=False))
    print("\nthe decision tree for own backups (random leagues)\n" + rules)

    glossary = ("Points per 100 possessions. <b>own backups</b>: the most-used player at a position, when the other "
                "players at his position get one point worse (median over starters; <b>90th</b> = the most-affected "
                "tenth). Positive = the belief. <b>raw on/off</b>: the same test on raw on/off. <b>if backups N worse</b>: "
                "the per-point number times N (Jokic's 2024-26 backups were 2.0 worse than a typical replacement). "
                "<b>share kept</b>: the lineup that replaces a starter one point worse, raw on/off moving exactly 1. "
                "<b>team bench</b>: the whole bench one point worse. <b>keeps own</b>: the share of his own true rating "
                "RAPM keeps. <b>backup time &lt;10% shared</b>: replacement time from players who share under 10% of his "
                "court time (2024-26: 55%). Levers: <b>positions</b> 1 = a backup only replaces his own starter; "
                "<b>lines</b> -0.25 = staggered rests, +0.95 = hockey lines; <b>stay</b> = roster stability; "
                "<b>injuries</b> 0 = none; <b>noise</b> = game-to-game minutes randomness; <b>starter min</b> = "
                "starters' minutes a game.")
    tables = [("The extreme corners", config_view(corners)),
              ("One lever at a time: from the textbook corner back to 2024-26, and from 2024-26 to the textbook",
               ablation_view),
              ("The ten random leagues where backups reach the starter most", config_view(top)),
              ("What each lever does (random leagues)", effect_view)]
    page = ('<div style="font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;color:#1f2933">'
            '<h2 style="font-size:18px;margin:0 0 6px">Stacking the deck: can truly bad backups inflate RAPM?</h2>'
            + "".join(f'<p style="font-size:14px;margin:0 0 6px">{html.escape(ln)}</p>' for ln in lines)
            + f'<p style="font-size:12px;color:#6b7280">{glossary}</p>'
            + "".join(f'<h3 style="font-size:15px">{html.escape(t)}</h3>'
                      + r77.html_table(f, left=("kind", "lever", "change")) for t, f in tables)
            + '<h3 style="font-size:15px">Decision tree for own backups</h3>'
            + f'<pre style="font-size:11px">{html.escape(rules)}</pre>'
            + f'<p style="font-size:12px;color:#6b7280">{sum(1 for c in log if c.startswith("PASS"))} checks passed; '
            f'every league was checked (nobody on both teams, replacement weights sum to 1, raw on/off moves exactly '
            f'1).</p></div>')
    (out / "stack_the_deck_phone.html").write_text(page, encoding="utf-8")
    print(f"wrote outputs/stack_the_deck_phone.html\ndone in {time.time() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
