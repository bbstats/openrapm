"""The shot-quality search's feature audits (DECISIONS.md, "The shot-quality search"; the spec's section 3).

    python scripts/128_feature_audit.py [--seasons=1998,2004,2010,2019,2022] [--blocks=late,scramble,ato,...]
                                        [--tol=0.005] [--clock=1] [--q=<arm>] [--tags=1] [--tag_games=300]
                                        [--table=_table] [--tag=run]

A new feature joins the search as a toggle only if it passes these, beside its held-out score.

(a) SAME SECOND.  The feed sometimes logs a foul before the made shot it belongs to (the and-one rate of mid and three
    shots whose previous row is within 0.5 s is 3.8% against 0.8% in 2002, 8.1% against 1.1% in 2018), and defensive
    goaltending sits at the shot's own clock reading.  A feature built from such an event knows the result.  So each
    block is rebuilt on a copy of the table whose events strings have every event at the shot's own clock reading
    removed (within --tol; an offensive rebound is kept, it really does come first), with the fields derived from the
    events recomputed on both copies alike (secs_since_oreb, secs_since_timeout, n_oreb, shotfeatures' event fields,
    and with --clock=1 the rebuilt shot clock, shotclock.rebuild with data/shotq/clock_lags.json).  A block passes
    when no column of any row changes.  The default --tol is shotfeatures.SAME_TOL, the rule itself; a wider one
    (--tol=0.5) is a stress test of how much rests on events just before the shot.
    Beside it, each column's link to the result it must not know: post_and1 on made shots and post_blocked on misses,
    active minus inactive within the same level cell (shotsearch.band_codes) and, with --q=<arm>, the same tenth of
    that arm's quality (data/shotq/<arm>/<season>.parquet).
(b) TAG PROXY (--tags=1).  How far each column predicts the scorer's own words for the shot (dunk, layup, tip,
    putback, hook, alley-oop, jump shot) within the same level cell: a feature that is a hidden shot-type tag shows up
    here.  The words are read from the CACHED raw play-by-play of --tag_games games a season (never fetched), held in
    memory only, and never written anywhere but as these aggregates.
(c) TEAM CONCENTRATION.  For each column, the share of the spread of its team-season means that is real rather than
    chance (1 - expected chance variance / observed variance of the means): a feature that is mostly a team's habit
    needs the owner's ruling.

Search seasons only (shotsearch.assert_phase): the audits feed a feature decision, and no decision reads a confirm
season's makes.  Reads the model tables, the shot frames' post_* columns and, with --tags, the cached raw play-by-play;
writes outputs/shotsearch/audit_<tag>_{same_second,link,tagproxy,team}.csv and nothing else.  A light job (a few
minutes a season), but still one job: check Get-Process python first.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _cli import check_flags, flag  # noqa: E402
from eracoef import shotfeatures as sf  # noqa: E402
from eracoef import shotmodel as sm  # noqa: E402
from eracoef import shotsearch as ss  # noqa: E402

DEFAULT_SEASONS = (1998, 2004, 2010, 2019, 2022)           # one from each search block
DEFAULT_BLOCKS = tuple(sf.BLOCKS) + ("putback", "clock")     # the new blocks, and the shipped ones that read events
CLOCK_COLS = ("sc", "sc_eff", "reset_kind", "since_reset", "clock_off")
TAGS = {"dunk": ("dunk",), "layup": ("layup", "lay up", "finger roll"), "tip": ("tip",),
        "putback": ("putback", "put back"), "hook": ("hook",), "alley_oop": ("alley oop", "alley-oop"),
        "jump": ("jump", "pullup", "pull-up", "step back", "fadeaway", "turnaround")}


# ------------------------------------------------------------------------------------------ (a) same second
def strip_same_second(t: pd.DataFrame, tol: float = sf.SAME_TOL) -> pd.DataFrame:
    """The table with every event at the shot's own clock reading (within `tol`) dropped from its events string,
    offensive rebounds kept.  Nothing else changes: `rederive` recomputes what is read from the events."""
    ev = t["events"].fillna("").to_numpy(object)
    clock = t["clock"].to_numpy(float)
    out = ev.copy()
    for i in np.flatnonzero(ev != ""):
        keep = []
        for tok in ev[i].split(";"):
            k, c = _split(tok)
            if k in sf.OREB_KINDS or c is None or abs(c - clock[i]) > tol:
                keep.append(tok)
        out[i] = ";".join(keep)
    return t.assign(events=out)


def _split(tok: str):
    k, _, c = tok.rpartition("@")
    try:
        return k, float(c)
    except ValueError:
        return k, None


def rederive(t: pd.DataFrame, rules=None) -> pd.DataFrame:
    """Every field the table derives from the events string, recomputed from it: shotframe.derive's three, the
    features' own event fields, and with `rules` (shotclock.ClockRules) the rebuilt shot clock."""
    from eracoef.shotframe import _last_event          # the definition the tables were built with
    clock = t["clock"].to_numpy(float)
    oreb = _last_event(t["events"].fillna(""), ("oreb", "oreb_team"))
    to = _last_event(t["events"].fillna(""), ("timeout",))
    out = t.assign(secs_since_oreb=np.clip(oreb - clock, 0.0, None), secs_since_timeout=np.clip(to - clock, 0.0, None),
                   n_oreb=t["events"].fillna("").str.count(r"(?:^|;)oreb").to_numpy())
    ef = sf.event_fields(out)
    out = out.assign(**{c: ef[c].to_numpy() for c in sf.EVENT_COLS})
    if rules is not None:
        from eracoef.shotclock import rebuild
        r = rebuild(out, rules)
        out = out.assign(**{c: r[c].to_numpy() for c in CLOCK_COLS})
    return out


def builders_for(blocks) -> dict:
    """name -> fn(f, sub) for block names (shotfeatures' or shotmodel's own)."""
    return {b: (lambda f, s, b=b: sm.block_columns(f, s, b)) for b in blocks}


def same_second_audit(t: pd.DataFrame, builders: dict, tol: float = sf.SAME_TOL, rules=None) -> pd.DataFrame:
    """Per block, sub-model and column: how many rows change when the block is rebuilt without the events at the
    shot's own second.  `t` is a prepared table (shotfeatures.add_previous_attempt done on the whole season)."""
    base = rederive(t, rules)
    strip = rederive(strip_same_second(t, tol), rules)
    sub = sm.submodel_of(base)
    rows = []
    for name, fn in builders.items():
        for s in sm.SUBMODELS:
            m = sub == s
            if not m.any():
                continue
            Xa, na = fn(base[m], s)
            Xb, nb = fn(strip[m], s)
            if list(na) != list(nb):
                raise AssertionError(f"block {name} built different columns without the same-second events: {na} / {nb}")
            Xa, Xb = np.asarray(Xa, float), np.asarray(Xb, float)
            for j, col in enumerate(na):
                same = np.isclose(Xa[:, j], Xb[:, j], rtol=0.0, atol=1e-12, equal_nan=True)
                diff = np.where(same, 0.0, np.abs(np.nan_to_num(Xa[:, j] - Xb[:, j], nan=np.inf)))
                rows.append(dict(block=name, sub=s, column=col, n=int(m.sum()), changed=int((~same).sum()),
                                 share=float((~same).mean()), max_abs=float(diff.max()) if len(diff) else 0.0))
    return pd.DataFrame(rows, columns=["block", "sub", "column", "n", "changed", "share", "max_abs"])


# ------------------------------------------------------------------------------------------ links and concentration
def stratified_diff(active: np.ndarray, outcome: np.ndarray, cells: np.ndarray) -> dict:
    """The outcome's mean among active rows minus among inactive rows, within cells, averaged over the cells holding
    both, weighted by their active rows; the standard error is the no-link one (each cell's pooled rate), so a rare
    column whose few active rows all read 0 is not given a standard error of 0."""
    active = np.asarray(active, bool)
    y = np.asarray(outcome, float)
    c = pd.factorize(pd.Series(cells))[0]
    G = c.max() + 1 if len(c) else 0
    n1 = np.bincount(c[active], minlength=G).astype(float)
    n0 = np.bincount(c[~active], minlength=G).astype(float)
    s1 = np.bincount(c[active], weights=y[active], minlength=G)
    s0 = np.bincount(c[~active], weights=y[~active], minlength=G)
    ok = (n1 > 0) & (n0 > 0)
    if not ok.any():
        return dict(n_active=int(active.sum()), diff=np.nan, se=np.nan)
    p1, p0 = s1[ok] / n1[ok], s0[ok] / n0[ok]
    w = n1[ok] / n1[ok].sum()
    pool = (s1[ok] + s0[ok]) / (n1[ok] + n0[ok])
    var = pool * (1 - pool) * (1.0 / n1[ok] + 1.0 / n0[ok])
    return dict(n_active=int(active.sum()), diff=float(np.sum(w * (p1 - p0))), se=float(np.sqrt(np.sum(w ** 2 * var))))


def team_concentration(x: np.ndarray, groups: np.ndarray, min_n: int = 30) -> dict:
    """The share of the spread of a column's group means (team-seasons) beyond chance: 1 - (mean within-group
    variance / n) / (variance of the means), over groups with `min_n` rows or more; and the means' sd."""
    x = np.asarray(x, float)
    g = pd.factorize(pd.Series(groups))[0]
    G = g.max() + 1 if len(g) else 0
    n = np.bincount(g, minlength=G).astype(float)
    s = np.bincount(g, weights=x, minlength=G)
    s2 = np.bincount(g, weights=x * x, minlength=G)
    ok = n >= min_n
    if ok.sum() < 3:
        return dict(groups=int(ok.sum()), sd_means=np.nan, real_share=np.nan)
    mean = s[ok] / n[ok]
    within = np.maximum(s2[ok] / n[ok] - mean ** 2, 0.0) * n[ok] / np.maximum(n[ok] - 1, 1)
    observed = mean.var(ddof=1)
    chance = float(np.mean(within / n[ok]))
    real = 1.0 - chance / observed if observed > 0 else np.nan
    return dict(groups=int(ok.sum()), sd_means=float(np.sqrt(observed)), real_share=float(max(real, 0.0)) if np.isfinite(real) else np.nan)


def tags_of(sub_type, description) -> pd.DataFrame:
    """The scorer's words for each shot as yes/no columns (TAGS), from subType and description, lower-cased.  For
    the tag-proxy audit only: never an input, never written to a table."""
    words = (pd.Series(sub_type, dtype=object).fillna("").astype(str) + " "
             + pd.Series(description, dtype=object).fillna("").astype(str)).str.lower().to_numpy(object)
    return pd.DataFrame({k: np.array([any(p in w for p in pats) for w in words], dtype=bool) for k, pats in TAGS.items()})


def column_table(f: pd.DataFrame, builders: dict):
    """(sub-model, block, column name, values) for every column of every block, on f's rows of that sub-model."""
    sub = sm.submodel_of(f)
    for name, fn in builders.items():
        for s in sm.SUBMODELS:
            m = sub == s
            if not m.any():
                continue
            X, names = fn(f[m], s)
            for j, col in enumerate(names):
                yield s, name, col, m, np.asarray(X, float)[:, j]


def audit_season(t: pd.DataFrame, builders: dict, season: int, tol: float = sf.SAME_TOL, rules=None, q=None,
                 tags: pd.DataFrame | None = None) -> dict:
    """Every audit of one season.  `t`: the season's prepared table (shotfeatures.prepare on all of it), heaves out,
    with post_and1 and post_blocked; `q` (optional): an arm's quality per row, for the level cells' tenths; `tags`
    (optional): TAGS columns per row, NaN where the play-by-play was not read.  Returns DataFrames same_second,
    link, team, tagproxy."""
    a = same_second_audit(t, builders, tol=tol, rules=rules).assign(season=season)
    f = rederive(t, rules)
    cells = ss.band_codes(f).astype(np.int64)
    if q is not None:
        cells = cells * 100 + np.nan_to_num(np.floor(np.asarray(q, float) * 10), nan=-1).astype(np.int64)
    made = f["made"].to_numpy() == 1
    team_season = f["team"].astype(str).to_numpy(object) + ":" + str(season)
    have_tags = None if tags is None else tags[list(TAGS)].notna().all(axis=1).to_numpy()
    link, team, tagp = [], [], []
    for s, name, col, m, v in column_table(f, builders):
        act = v != 0
        key = dict(season=season, block=name, sub=s, column=col)
        for outcome, rows in (("post_and1", made[m]), ("post_blocked", ~made[m])):
            link.append(dict(key, outcome=outcome, **stratified_diff(act[rows], f[outcome].to_numpy()[m][rows],
                                                                     cells[m][rows])))
        team.append(dict(key, **team_concentration(v, team_season[m])))
        if have_tags is not None:
            hv = have_tags[m]
            for k in TAGS:
                y = tags[k].to_numpy(object)[m][hv].astype(bool)
                tagp.append(dict(key, tag=k, **stratified_diff(act[hv], y, cells[m][hv])))
    return dict(same_second=a, link=pd.DataFrame(link), team=pd.DataFrame(team), tagproxy=pd.DataFrame(tagp))


# ------------------------------------------------------------------------------------------ the real tables (driver)
def _post_cols(cfg, season: int) -> pd.DataFrame:
    """post_and1 and post_blocked from the shot frames: for the audit's links and the previous attempt's blocked flag."""
    from eracoef.shotframe import frame_path
    parts = []
    for ph in ("RS", "PO"):
        p = frame_path(season, ph, cfg)
        if p.exists():
            parts.append(pd.read_parquet(p, columns=["game_id", "action_number", "post_and1", "post_blocked"]))
    if not parts:
        raise FileNotFoundError(f"no shot frame for {season} under {frame_path(season, 'RS', cfg).parent}")
    return pd.concat(parts, ignore_index=True).drop_duplicates(["game_id", "action_number"])


def _read_tags(cfg, game_ids) -> pd.DataFrame:
    """The scorer's words for the field-goal rows of these games, from the CACHED raw play-by-play only."""
    from eracoef.ingest import _pbp_path
    parts, missing = [], 0
    for gid in game_ids:
        p = _pbp_path(gid, cfg)
        if not p.exists():
            missing += 1
            continue
        r = pd.read_parquet(p, columns=["actionNumber", "actionType", "subType", "description"])
        r = r[r["actionType"].astype(str).str.strip().isin(["Made Shot", "Missed Shot"])]
        tg = tags_of(r["subType"].to_numpy(object), r["description"].to_numpy(object))
        tg["game_id"] = str(gid)
        tg["action_number"] = pd.to_numeric(r["actionNumber"], errors="coerce").fillna(-1).astype(np.int64).to_numpy()
        parts.append(tg)
    if missing:
        print(f"    tags: {missing} of {len(game_ids)} games have no cached play-by-play (skipped, never fetched)")
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["game_id", "action_number", *TAGS])


def main():
    check_flags()
    from eracoef.config import load_config
    cfg = load_config()
    root = Path(cfg["_root"])
    seasons = [int(s) for s in flag("seasons", ",".join(map(str, DEFAULT_SEASONS))).split(",") if s]
    ss.assert_phase(seasons, "search")
    blocks = [b for b in flag("blocks", ",".join(DEFAULT_BLOCKS)).split(",") if b]
    unknown = set(blocks) - set(sf.BLOCKS) - set(sm.BLOCKS) - set(sm.EXTRA_BLOCKS)
    if unknown:
        raise SystemExit(f"unknown blocks {sorted(unknown)}")
    tol = float(flag("tol", str(sf.SAME_TOL)))
    rules = None
    if flag("clock", "0") == "1":
        from eracoef.shotclock import ClockRules
        lags = json.loads((root / "data" / "shotq" / "clock_lags.json").read_text(encoding="utf-8"))
        rules = ClockRules(**lags["rules"], lag=lags["lags"])
    q_arm = flag("q", "")
    do_tags = flag("tags", "0") == "1"
    tag_games = int(flag("tag_games", "300"))
    table_dir = root / "data" / "shotq" / flag("table", "_table")
    tag = flag("tag", "run")
    out = root / "outputs" / "shotsearch"
    out.mkdir(parents=True, exist_ok=True)
    builders = builders_for(blocks)
    same, link, team, tagp = [], [], [], []
    for season in seasons:
        t0 = time.time()
        t = pd.read_parquet(table_dir / f"{season}.parquet")
        post = _post_cols(cfg, season)
        t = t.drop(columns=[c for c in ("post_and1", "post_blocked") if c in t.columns])
        t = t.merge(post, on=["game_id", "action_number"], how="left")
        t[["post_and1", "post_blocked"]] = t[["post_and1", "post_blocked"]].fillna(0).astype(np.int64)
        t = sf.prepare(t)                                   # the whole table: the previous attempt is another row
        t = t[~t["heave"].to_numpy(bool)].reset_index(drop=True)
        q = None
        if q_arm:
            qt = pd.read_parquet(root / "data" / "shotq" / q_arm / f"{season}.parquet", columns=["game_id", "action_number", "q"])
            q = t[["game_id", "action_number"]].merge(qt, on=["game_id", "action_number"], how="left")["q"].to_numpy(float)
        tags = None
        if do_tags:
            gids = np.random.default_rng(season).permutation(np.unique(t["game_id"].astype(str)))[:tag_games]
            tg = _read_tags(cfg, gids).drop_duplicates(["game_id", "action_number"])
            tags = (t[["game_id", "action_number"]].astype({"game_id": str})
                    .merge(tg, on=["game_id", "action_number"], how="left")[list(TAGS)])
        res = audit_season(t, builders, season, tol=tol, rules=rules, q=q, tags=tags)
        for k, sink in (("same_second", same), ("link", link), ("team", team), ("tagproxy", tagp)):
            sink.append(res[k])
        a = res["same_second"]
        bad = a[a["changed"] > 0]
        print(f"  {season}: {len(t):,} attempts, {time.time() - t0:.0f}s; same second: "
              + ("every column unchanged" if bad.empty else
                 "CHANGED " + ", ".join(sorted(set(bad["block"] + ":" + bad["column"])))), flush=True)
    for name, parts in (("same_second", same), ("link", link), ("team", team), ("tagproxy", tagp)):
        frame = pd.concat(parts, ignore_index=True)
        if len(frame):
            frame.to_csv(out / f"audit_{tag}_{name}.csv", index=False)
    S = pd.concat(same, ignore_index=True).groupby("block")["changed"].max()
    print("same-second verdict per block: "
          + ", ".join(f"{b} {'pass' if n == 0 else f'FAIL (up to {n} rows a season and column)'}" for b, n in S.items()))
    if rules is None and "clock" in blocks:
        print("  note: the clock block reads the table's rebuilt shot clock; without --clock=1 it is not rebuilt, so "
              "its pass says nothing")


if __name__ == "__main__":
    main()
