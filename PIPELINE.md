# How the model works

Two loops. The **build** turns play-by-play into a rating for every player-window. The **criterion** holds a
season out and scores the build against games it never saw; it is what every choice below was selected on.
Nothing here is a preference — each box is in the code at the file named beside it.

---

## 1. The build

```mermaid
flowchart TD
    RAW["Play-by-play, box scores, game logs<br/>1997-2026, data/raw/"]
    ST["<b>Stints</b> — one row per span with the same ten on the floor<br/>possessions, points, 118 per-possession and per-slot counters<br/><code>stints.py</code>, <code>scripts/02_stints.py</code>"]
    DES["<b>Design</b> — one row per stint per side, 195k rows for a 3-season window<br/>y = points per 100, weight = possessions<br/>columns: 5 offensive players, 5 defensive players, home, era,<br/>playoff, garbage time, score margin, margin x time<br/><code>designcache.py</code>, <code>design.py</code>"]
    TGO["<b>Offensive target</b> — made free throws replaced by expected ones<br/>xpts_ft = pts - ftm + xftm  ·  <code>design.TARGETS</code>"]
    TGD["<b>Defensive target</b> — every opponent three replaced by<br/>3 x that shooter's padded 3P%, from the other half of the block, k = 450<br/><code>xshoot.def_three_design</code>"]
    EXP["<b>Box exposure</b> — 13 per-100 rates per player-season,<br/>padded toward the league by method of moments, centred<br/><code>exposure.py</code>, <code>pad.py</code>"]

    RAW --> ST --> DES
    DES --> TGO
    DES --> TGD
    DES --> EXP

    subgraph PANEL["Role panel — built once, scripts/49_role_panel.py"]
        APM["<b>APM</b> — ridge at penalty 100 (lam_plugin is 18,352)<br/>what a player is worth on his own possessions, barely shrunk"]
        SPM["<b>Simple SPM</b> — ridge of APM on share of team possessions,<br/>its square, games-started share, its square, age, age², age³<br/>fitted leave-window-out"]
        R1["<b>RAPM_1</b> — the shipped ridge with SPM as its offset<br/>a player with no possessions gets exactly his role level"]
        APM --> SPM --> R1
    end

    EXP --> APM
    TGO --> APM
    TGD --> APM

    subgraph PRIOR["Box prior — one gradient-boosted model per side, leave-window-out"]
        PO["<b>Offense</b> — chimeraboost quality=4 (5 bagged members, auditions, cross features)<br/>37 features: 13 rates + season + role + 10 aggregations + 10 efficiency ratios<br/>target: 0.6 APM + 0.4 RAPM_1 over his OTHER windows, 0.3 discount per window away"]
        PD["<b>Defense</b> — chimeraboost, no auditions<br/>15 features: rates + season + role<br/>target: RAPM_1 over his other windows, every window alike"]
    end

    R1 --> PO
    R1 --> PD
    EXP --> PO
    EXP --> PD

    RIDGE["<b>The ridge</b> — one design, one exposure, one offset, two solves<br/>mixed model, per-season random effect, lambda 18,352, lambda_D/lambda_O 0.287<br/>rating = prior + what the possessions move it<br/><code>fastfit.MspiFast</code>, <code>estimator.py</code>"]
    PO --> RIDGE
    PD --> RIDGE
    TGO --> RIDGE
    TGD --> RIDGE

    MAP["<b>Calibration map</b> — fitted leave-one-season-out on the criterion<br/>offense: a·x + level in log exposure + b·x·log exposure + c·prior + d·role<br/>defense: the same without the prior and role terms<br/>then re-centred per window<br/><code>calmap.py</code>, table in <code>outputs/calmap_ship.parquet</code>"]
    RIDGE --> MAP

    OUT["<b>outputs/player_ratings.parquet</b> — ten 3-season windows, 1997-2026<br/>offense and defense per 100 possessions, positive is good on both sides<br/><code>scripts/08_ratings.py</code> → <code>scripts/52_site.py</code> → <code>docs/</code>"]
    MAP --> OUT
```

The two sides take **different priors on purpose**: the criterion wants the bagged, searched booster and the
richer feature set, and the consensus floors do not tolerate either on defense (FINDINGS 21.26). Only the
calibration map couples them, because its parameters are fitted jointly on the team-game residual.

---

## 2. The criterion

```mermaid
flowchart LR
    H["Hold out season H"] --> TR["Train on the K = 3 seasons<br/>nearest H, never including it"]
    TR --> FIT["The whole build above,<br/>on those seasons only"]
    FIT --> PRED["Predict every stint of H from<br/>the ten names on the floor.<br/>Only an intercept and a home term<br/>are refit on H"]
    PRED --> AGG["Sum each team's predicted and actual<br/>points within a game"]
    AGG --> SC["Possession-weighted squared error,<br/>pooled over 28 held-out seasons"]
    SC --> N["<b>109.85</b> for the criterion's best<br/><b>110.71</b> for what ships<br/><b>125.6</b> with no ratings at all"]
```

`holdout.py` is the runner, `scripts/54_track.py` the tracker that logs a row to `docs/progress.csv` and
redraws `docs/progress.png`. It is the one test in the project that is both external to the model (it scores
actual points in a season the ratings never saw) and legal to select on (it uses no outside data).

**Two terms live only here and cannot ship**, because a rating is a number attached to one player:

- the player's **age in H**, quadratic — a window has no single "age at H";
- a **cubic in the team-game total** and **the same cubic at stint level**, worth 0.28 per 100 together — a
  rating carries no team, and the criterion's prediction is the sum of the five on the floor.

The consensus (`data/external/consensus.csv`, the owner's blend of public metrics) is **validation only**: read
once, never selected on. Ten floors in `tests/test_vs_consensus.py` decide whether a board may ship at all.

---

## 3. What each stage is for

| stage | the problem it solves |
|---|---|
| stints | five players share every possession; the only unit that identifies them is the lineup change |
| luck adjustment | whether an opponent's three drops is almost pure noise, and free-throw shooting is one identified player with no defense |
| box exposure | the 13 rates are the only per-player description the box score gives, and a 200-possession player's rates need padding |
| APM → SPM → RAPM_1 | a target for the prior that is not itself the thing being predicted, with a role level for players the possessions cannot see |
| the boosted prior | what a player's box line is worth on court, learned from his *other* windows so it can never memorise the one being fitted |
| the ridge | ten collinear players per row; the penalty is what makes the system solvable, and the prior is where a low-minute player starts |
| the calibration map | the ridge over-shrinks, and by an amount that depends on how many possessions it saw |
| the criterion | everything above has a knob, and this is the only thing allowed to turn one |
