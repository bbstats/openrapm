"""Split a player's raw on/off into his plain RAPM, his teammates on and off court, the opponents and the rest --
an exact identity, nothing left over.

    python scripts/82_decompose_rapm.py [--penalty=3000] [--players=Neemias Queta,Ajay Mitchell] [--top=25]

**Why this exists.**  The owner asked whether pure RAPM can be decomposed into self, teammates, opponents and the
off-court difference.  It can, exactly, once it is turned around.  His raw on/off (his team's net points per 100
with him on court minus with him off, in games he played) is a possession-weighted average of rows; every row's
outcome is the fit's prediction plus a residual; and the prediction is the ten players' ratings plus the context.
So, with every rating taken from the same plain RAPM fit:

    raw on/off = RAPM + On-Court + Off-Court + opponents + context + ridge penalty + unexplained off

  RAPM             his plain RAPM, the self part
  On-Court         the teammates who play more when he plays, weighted by how much more, at their RAPM
  Off-Court        minus the teammates who play more when he sits (his replacements), weighted by how much more, at
                   their RAPM; bad replacements make it positive
  opponents        how much weaker the opponents are, at their RAPM, when he plays than when he sits
  context          home, garbage time, score margin and season, on minus off
  ridge penalty    credit his lineups earned that the penalty withholds from his rating: the fit keeps
                   possessions / (possessions + penalty) of what his lineups' results say once teammates, opponents
                   and context are taken out, and the rest lands here (exactly penalty x rating / possessions, per
                   side, on the fit's own zero point)
  unexplained off  while he sits, how much worse his team did than the RAPM of the ten players on court (plus
                   context) adds up to.  It counts in his raw on/off but never reaches his RAPM, whose equation holds
                   only the possessions he played.  It is luck, the ridge penalty pulling the players on court then
                   toward zero, and whatever one rating per player cannot capture.

So RAPM = raw on/off minus the other six.  Points per 100 possessions, positive good, offence plus defence.
Ratings are on the published zero point (possession-weighted zero per side); that leaves the sum unchanged, because
his teammates' shares add up to exactly one more player on court when he sits than when he plays.  Plain RAPM:
2024-26 pooled, raw points, regular season and playoffs -- the same fit as 80 and 81.

Writes outputs/decompose_rapm.parquet (every player) and outputs/decompose_rapm_phone.html.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _cli import check_flags, flag  # noqa: E402
from eracoef.config import load_config  # noqa: E402

# 80's setup (and through it 77's block, fit and on/off operators) -- one definition of each
_spec = importlib.util.spec_from_file_location("_borrowed_80", ROOT / "scripts" / "80_drop_replacements.py")
s80 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s80)
r77 = s80.r77
check = r77.check
f2 = s80.f2
PARTS = ["self", "on_court", "off_court", "opponents", "context", "ridge_penalty", "unexplained_off"]
LABELS = dict(self="RAPM", on_court="On-Court", off_court="Off-Court", opponents="opponents", context="context",
              ridge_penalty="ridge penalty", unexplained_off="unexplained off")
MIN_POSS = 3000                 # the players the table and the summary lines rank


def decompose(ctx) -> tuple:
    """Every player's seven pieces, and the largest misses of the three identities the pieces rest on."""
    block, n, beta, w, y, G = ctx.block, ctx.n, ctx.beta, ctx.block.w, ctx.block.y, ctx.block.gram
    resid = y - block.X @ beta
    at_side = {"O": block.X[:, :n].tocsc(), "D": block.X[:, n:2 * n].tocsc()}
    rating = {"O": ctx.off, "D": ctx.dfn}                          # published zero point, positive good
    sign = {"O": 1.0, "D": -1.0}                                   # a defensive row's outcome is points allowed
    team_game = {"O": block.tg_off, "D": block.tg_def}
    own_col = {"O": 0, "D": n}
    opp = {"O": slice(n, 2 * n), "D": slice(0, n)}
    miss = dict(operator=0.0, ridge=0.0, shares=0.0)

    def mean(values, mask):
        return float(np.average(values[mask], weights=w[mask]))

    rows = []
    for i in np.flatnonzero(ctx.ops["O"]["ok"] & ctx.ops["D"]["ok"]):
        at = {s: Z.indices[Z.indptr[i]:Z.indptr[i + 1]] for s, Z in at_side.items()}
        games = np.union1d(block.tg_off[at["O"]], block.tg_def[at["D"]])   # 77's "his team": team-games he played
        rec = dict(i=i, raw=0.0, on_court_net=0.0, **{k: 0.0 for k in PARTS})
        for s in ("O", "D"):
            team = np.isin(team_game[s], games)
            on = np.zeros(len(w), dtype=bool)
            on[at[s]] = True
            off = team & ~on
            row, e, r, sg = ctx.ops[s]["D"][i], ctx.rep[s]["e"][i], rating[s], sign[s]
            raw = mean(y, on) - mean(y, off)
            miss["operator"] = max(miss["operator"], abs(row @ beta + mean(resid, on) - mean(resid, off) - raw))
            miss["ridge"] = max(miss["ridge"], abs(mean(resid, on) - ctx.lam * beta[own_col[s] + i]
                                                   / G[own_col[s] + i, own_col[s] + i]))
            miss["shares"] = max(miss["shares"], abs(row[own_col[s] + i] - 1.0), abs(e.sum() - 1.0))
            rec["raw"] += sg * raw
            rec["on_court_net"] += sg * mean(y, on)
            rec["self"] += r[i]
            rec["on_court"] += float(np.clip(-e, 0.0, None) @ r)
            rec["off_court"] -= float(np.clip(e, 0.0, None) @ r)
            rec["opponents"] += sg * float(row[opp[s]] @ beta[opp[s]])
            rec["context"] += sg * float(row[2 * n:] @ beta[2 * n:])
            rec["ridge_penalty"] += sg * mean(resid, on)
            rec["unexplained_off"] -= sg * mean(resid, off)
        rows.append(rec)
    return pd.DataFrame(rows), miss


def teammates_line(ctx, i: int) -> str:
    """His three biggest Off-Court and On-Court teammates, by what each adds to his on/off."""
    names = np.asarray(ctx.names)
    eo, ed = ctx.rep["O"]["e"][i], ctx.rep["D"]["e"][i]
    repl = -(np.clip(eo, 0.0, None) * ctx.off + np.clip(ed, 0.0, None) * ctx.dfn)
    mates = np.clip(-eo, 0.0, None) * ctx.off + np.clip(-ed, 0.0, None) * ctx.dfn

    def top(v, share, when):
        return "; ".join(f"{names[j]} ({share[j]:+.0%} of possessions when he {when}, RAPM {ctx.total[j]:+.2f}) "
                         f"{v[j]:+.2f}" for j in np.argsort(-np.abs(v))[:3])

    return f"{names[i]}. Off-Court: {top(repl, eo, 'sits')}. On-Court: {top(mates, -eo, 'plays')}."


def main() -> None:
    check_flags()
    lam = float(flag("penalty", 3000))
    wanted = [s.strip() for s in flag("players", "Neemias Queta,Ajay Mitchell").split(",") if s.strip()]
    top_n = int(flag("top", 25))
    cfg = load_config()
    log: list = []
    started = time.time()

    ctx = s80.setup(cfg, lam)
    tab, miss = decompose(ctx)
    check(miss["shares"] < 1e-9, f"he is on court in every row of his on/off and never in its off side, and his "
                                 f"teammates' extra shares when he sits add up to one player (max miss "
                                 f"{miss['shares']:.1e})", log)
    check(miss["operator"] < 1e-9, f"77's on/off operator plus the residuals reproduces raw on/off from the rows "
                                   f"(max miss {miss['operator']:.1e})", log)
    check(miss["ridge"] < 1e-9, f"'ridge penalty' equals penalty x rating / possessions on each side, the fit's own "
                                f"condition (max miss {miss['ridge']:.1e})", log)
    gap = float(np.abs(tab[PARTS].sum(axis=1) - tab.raw).max())
    check(gap < 1e-9, f"the seven pieces add up to raw on/off for all {len(tab)} players (max miss {gap:.1e})", log)

    block, names = ctx.block, np.asarray(ctx.names)
    idx = tab.i.to_numpy()
    tab.insert(0, "player", names[idx])
    tab.insert(0, "player_id", block.player_ids[idx])
    tab["poss"] = block.poss[idx]
    out = ROOT / "outputs"
    tab.drop(columns=["i"]).to_parquet(out / "decompose_rapm.parquet", index=False)

    big = tab[tab.poss >= MIN_POSS].sort_values("self", ascending=False).reset_index(drop=True)
    big["#"] = np.arange(1, len(big) + 1)
    show = pd.concat([big.head(top_n), big[big.player.isin(wanted) & (big["#"] > top_n)]])
    view = pd.DataFrame({"#": show["#"], "player": show.player, "raw on/off": show.raw.map(f2),
                         **{("= " if k == "self" else "+ ") + LABELS[k]: show[k].map(f2) for k in PARTS}})
    slope = {k: float(np.cov(big.raw, big[k])[0, 1] / big.raw.var()) for k in PARTS}
    head = (f"Plain RAPM 2024-26 at penalty {lam:,.0f}, the top {top_n} of the {len(big)} players with "
            f"{MIN_POSS:,}+ possessions: raw on/off = " + " + ".join(LABELS[k] for k in PARTS)
            + ", exact for every player, so RAPM is raw on/off minus the other six.")
    top = big.head(top_n)

    def listed(frame):
        return ", ".join(f"{who} {v:+.2f}" for who, v in zip(frame.player, frame.off_court))

    notes = [f"Bad replacements (Off-Court) lift raw on/off most for "
             f"{listed(top[top.off_court > 0].nlargest(4, 'off_court'))}; RAPM takes that back out.",
             f"Good replacements (Off-Court) hold raw on/off down most for "
             f"{listed(top[top.off_court < 0].nsmallest(4, 'off_court'))}; RAPM gives that back.",
             f"{int((top.opponents < 0).sum())} of the {top_n} face stronger opponents when they play (median "
             f"{top.opponents.median():+.2f}), which RAPM credits back."]
    spread = (f"Across all {len(big)} players with {MIN_POSS:,}+ possessions, each extra point of raw on/off splits on "
              f"average into " + ", ".join(f"{LABELS[k]} {slope[k]:+.2f}" for k in PARTS) + " (the seven add up to 1).")
    details = [teammates_line(ctx, int(tab.i[tab.player == who].iloc[0])) for who in wanted if (tab.player == who).any()]
    print("\n".join([head, *notes]) + "\n\n" + view.to_string(index=False) + "\n\n" + "\n".join([spread, *details]))

    glossary = ("Points per 100 possessions, positive good, offence plus defence. <b>raw on/off</b>: his team's net "
                "points per 100 with him on court minus with him off, in games he played. <b>RAPM</b>: his plain "
                "RAPM, the self part. <b>On-Court</b>: the teammates who play more when he plays, weighted by how "
                "much more, at their RAPM. <b>Off-Court</b>: minus the teammates who play more when he sits (his "
                "replacements), weighted the same way; bad replacements make it positive. <b>opponents</b>: how "
                "much weaker the opponents are when he plays. <b>context</b>: home, garbage time, score margin, "
                "season. <b>ridge penalty</b>: credit his lineups earned that the penalty withholds from his rating. "
                f"The fit keeps possessions / (possessions + {lam:,.0f}) of what his lineups' results say once "
                "teammates, opponents and context are taken out; the rest lands here. <b>unexplained off</b>: while "
                "he sits, how much worse his team did than the RAPM of the ten players on court (plus context) adds "
                "up to. It counts in his raw on/off but never reaches his RAPM, which is set by the possessions he "
                "played. It is luck, the ridge penalty pulling the players on court then toward zero, and whatever "
                "one rating per player cannot capture."
                + (" The % next to a teammate: his share of the possessions when the player sits minus when he plays."
                   if details else ""))
    para = '<p style="font-size:{}px;margin:0 0 6px">{}</p>'.format
    page = ('<div style="font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;color:#1f2933">'
            '<h2 style="font-size:18px;margin:0 0 6px">Where raw on/off goes: RAPM, On-Court, Off-Court, '
            'opponents</h2>' + para(14, html.escape(head))
            + f'<p style="font-size:12px;color:#6b7280">{glossary}</p>'
            + "".join(para(14, html.escape(ln)) for ln in notes) + r77.html_table(view, left=("player",))
            + "".join(para(13, html.escape(ln)) for ln in [spread, *details])
            + '<h3 style="font-size:15px">Checks</h3>' + "".join(para(12, html.escape(ln)) for ln in log) + "</div>")
    (out / "decompose_rapm_phone.html").write_text(page, encoding="utf-8")
    print(f"\nwrote outputs/decompose_rapm.parquet, outputs/decompose_rapm_phone.html ({time.time() - started:.0f}s)")


if __name__ == "__main__":
    main()
