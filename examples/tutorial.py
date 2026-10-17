"""A tour of steps 1-3 on a made-up league whose true player impacts are known.

    python examples/tutorial.py            the full league, about 15 seconds
    python examples/tutorial.py --small    a small league in a few seconds (what tests/test_tutorial_golden.py runs)

Nothing is downloaded and nothing is written, unless --json or --write-golden asks for it.

The published ratings are built from thirty seasons of NBA play-by-play, and nobody knows any real player's true
impact, so a real season cannot say whether a rating is right.  A simulated league can: `openrapm.simulate` sets every
player's effect on points per 100 possessions before a game is played, then plays the games.  This script runs the
real code of the first three steps on such a league and prints, after each step, how close the ratings came to the
truth.

Words used below
    stint         a stretch of play with the same ten players on the court
    impact        a player's true effect on points per 100 possessions: on offence what he adds to the points his team
                  scores, on defence what he takes away from the points it allows (positive is good on both ends).
                  Both the impacts and the ratings are centred: the possession-weighted average player is 0
    rated season  the season whose ratings are built; here the last simulated one
    correlation   Pearson's correlation between the ratings and the true impacts, over the rated season's players
    error         the root mean square of (rating - true impact) over the same players, in points per 100
    bench player  one of the quarter of the rated season's players with the fewest possessions; a starter is one of
                  the quarter with the most

The league (`LEAGUES["full"]`)
    Four seasons, 2001-2004, of 30 teams with 13 players each: 1,230 games a season, 25 to 35 stints a game.  A
    player's impact is his true box-score rates times fixed weights plus a part no box count explains (simulate.py
    says how).  It persists from one season to the next with correlation 0.9, and 30% of the players change teams
    each season, so the same player is seen beside different teammates.  2004 is the rated season.

Step 1, plain RAPM (src/openrapm/rapm.py)
    One ridge regression of every stint's points per 100 possessions on the ten players on the court:

        minimise   sum over rows  w (y - Z theta - C gamma)^2  +  40,000 |theta|^2

    with y the row's points per 100, w its possessions, Z the players on the court, theta their ratings, C the
    context (season, home court, score margin) and gamma its coefficients.  Run on 2004 alone it is the plain
    one-season rating.  Run on 2001-2003 it is the label: one number per player for those seasons together, which
    step 2 learns to predict.  The penalty keeps about n / (n + 40,000) of what n possessions say, so a player with
    few possessions is pulled hard toward 0.
Step 2, the box-score prior (src/openrapm/box_prior.py)
    LightGBM learns each player's label from his box score averaged over 2001-2003 (his career row and every run of
    1, 2 and 3 seasons), then predicts every player's 2004 value from his 2004 box score alone.  The box score is the
    season panel's: padded rates per 100 possessions, role, career and body (`simulate.simulated_panel` builds it
    with the same padding data/panel.py uses for real seasons).
Step 3, prior-informed RAPM (src/openrapm/prior_rapm.py)
    The 2004 ridge again, with each player pulled toward s x his prior instead of toward 0, and penalty 13,037:

        minimise   sum over rows  w (y - Z r - C gamma - s_O P_O - s_D P_D)^2  +  13,037 |r|^2

    P_O and P_D are the sums of the five offensive and five defensive priors on the court; the prior scales s_O and
    s_D are set by the games, unpenalised.  A rating is s x prior + r.

What differs from the published build, and why
    The shipped targets take free-throw and three-point luck out of the points (settings.Targets); the simulated
    stints carry plain points only, so both sides fit the plain points target, "pts".  The simulated league has no
    shot locations, on-court record or birthdays, so the panel holds the thirteen rates and the role, career and body
    columns the shipped inputs (features.SHIPPED) are made from, and not the rest; heights and weights are drawn at
    random and say nothing about impact.
    Everything else is the shipped setting: step 1's penalty of 40,000 with the defensive label un-shrunk, the
    booster settings in params/booster_params_lgb1.json, five player folds, and step 3's penalty of 13,037.

What the full league prints, in short (2004's 390 players; error in points per 100)
    one-season RAPM          correlation 0.74 on offence, 0.70 on defence; error 4.2 and 2.0
    the box-score prior      correlation 0.88 on offence, 0.52 on defence, from the box score alone
    prior-informed RAPM      correlation 0.93 on offence, 0.75 on defence; error 1.7 and 1.5
"""
from __future__ import annotations

import argparse
import json
import textwrap
import time
from dataclasses import replace
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

from openrapm import box_prior, prior_rapm, rapm
from openrapm.commands.console import one_thread, use_utf8
from openrapm.data.config import booster_params, load_config
from openrapm.design import FEATURES, assemble, pieces_from_frames
from openrapm.settings import PRODUCT, Targets
from openrapm.simulate import player_profiles, simulate, simulated_panel

GOLDEN = Path(__file__).resolve().parents[1] / "tests" / "golden" / "tutorial_small.json"

# The two leagues.  Possessions per stint are simulate's (1 + a Poisson draw with mean 6 a side), so a team-game holds
# about twice the NBA's possessions; the full league is otherwise the NBA's size.
LEAGUES = {
    "full": dict(n_seasons=4, n_teams=30, players_per_team=13, games_per_season=1230, stints_per_game=(25, 35)),
    "small": dict(n_seasons=4, n_teams=10, players_per_team=10, games_per_season=100, stints_per_game=(20, 30)),
}
# Talent persists (rho), 30% of players move each season, and a stint's points come from the ten impacts, the context
# and noise alone (leak=False: simulate.py's "leak" is off)
TALENT = dict(rho=0.9, turnover=0.3, leak=False, seed=1)
# The shipped setting with the plain points target on both sides (the simulated stints carry no free-throw counters)
# and the inputs before the hustle counts (the simulated league has no hustle box score)
SETTINGS = replace(PRODUCT, targets=Targets(offence="pts", defence="pts"),
                   prior=replace(PRODUCT.prior, feature_set="no_career_counts"))


class Side(NamedTuple):
    """The names one side goes by in each table the tour reads."""
    name: str           # "offence" or "defence": the suffix of the tour's own columns
    adjective: str      # "offensive" or "defensive"
    key: str           # "O" or "D": the side key of the labels and the priors
    column: str         # "offense" or "defense": the ratings tables' column
    impact: str         # "impact_O" or "impact_D": the truth's column
    unexplained: str    # "u_O" or "u_D": the truth's part of the impact no box count explains
    sign: float         # +1 or -1: a raw defensive number is points allowed, so -1 makes it positive-good


SIDES = (Side("offence", "offensive", "O", "offense", "impact_O", "u_O", 1.0),
         Side("defence", "defensive", "D", "defense", "impact_D", "u_D", -1.0))


def say(text: str) -> None:
    """Print one paragraph, wrapped at 100 characters and indented under its step's heading."""
    print(textwrap.fill(" ".join(text.split()), width=100, initial_indent="  ", subsequent_indent="  "), flush=True)


# ------------------------------------------------------------------------------------------------- the league
class League(NamedTuple):
    """A simulated league made ready for the steps: each season's design and the season panel, and the truth."""
    shape: dict
    seasons: list            # ascending
    rated: int               # the rated season, the last one
    designs: dict            # {season: its design on the plain points target}
    panel: pd.DataFrame      # the season panel (simulate.simulated_panel)
    truth: pd.DataFrame      # one row per player-season: impact_O, impact_D, u_O, u_D, rate_<count>, ...
    weights: dict            # {"O", "D"}: the fixed weights on the true box-score rates (simulate.BETA_O, BETA_D)
    n_stints: int


def make_league(shape: dict) -> League:
    """Simulate the league, then build each season's design with the one design assembly (design.assemble) and the
    season panel from the designs' box scores."""
    sim = simulate(**shape, **TALENT)
    stints, box = sim["stints"], sim["box"]
    seasons = sorted(int(season) for season in stints["season"].unique())
    cfg = load_config()
    designs = {season: assemble(pieces_from_frames(stints[stints.season == season], box[box.season == season],
                                                   FEATURES), cfg, "pts")
               for season in seasons}
    panel = simulated_panel(designs, player_profiles(sim, seed=TALENT["seed"]))
    weights = {"O": sim["truth"]["beta_O"], "D": sim["truth"]["beta_D"]}
    return League(shape, seasons, seasons[-1], designs, panel, sim["truth"]["ps"], weights, len(stints))


def centred(values, possessions) -> np.ndarray:
    """Values minus their possession-weighted mean: 0 is the average possession's player, as the ratings are centred."""
    values = np.asarray(values, dtype=float)
    return values - np.average(values, weights=possessions)


def rated_players(league: League, table: pd.DataFrame) -> pd.DataFrame:
    """One row per rated-season player of step 3's table: his possessions and his true impact on each side, centred,
    positive good."""
    truth = league.truth[league.truth.season == league.rated].set_index("player_id")
    players = pd.DataFrame({"player_id": table.player_id.to_numpy(), "possessions": table.poss_off.to_numpy()})
    for side in SIDES:
        raw = truth[side.impact].reindex(players.player_id).to_numpy()
        players[f"true_{side.name}"] = side.sign * centred(raw, players.possessions)
    return players


def agreement(rating, truth) -> dict:
    """{"correlation", "error"} of a rating against the true impact (both centred, positive good)."""
    rating, truth = np.asarray(rating, dtype=float), np.asarray(truth, dtype=float)
    return {"correlation": float(np.corrcoef(rating, truth)[0, 1]),
            "error": float(np.sqrt(np.mean((rating - truth) ** 2)))}


# ------------------------------------------------------------------------------------------------- step 1
def plain_rapm(league: League) -> tuple[dict, dict]:
    """(sums, labels): every season's normal equations on the plain points target, and step 1's labels for the rated
    season (rapm.career_labels: every season but the rated one, the defensive label un-shrunk)."""
    sums = {"pts": rapm.SeasonSums(SETTINGS.labels.min_possessions)}
    for season in league.seasons:
        sums["pts"].add(season, league.designs[season])
    return sums, rapm.career_labels(sums, [league.rated], SETTINGS.labels, SETTINGS.targets)


def one_season_rapm(league: League, sums: dict, players: pd.DataFrame) -> pd.DataFrame:
    """Plain RAPM of the rated season alone, with step 1's penalties, centred, positive good, one row per player of
    `players` (0 for a player under the 100 possessions a rating needs)."""
    labels = SETTINGS.labels
    others = [season for season in league.seasons if season != league.rated]
    alone = sums["pts"].ratings(others, (labels.offense_penalty, labels.defense_penalty, labels.context_penalty))
    alone = alone.set_index("player_id").reindex(players.player_id).fillna(0.0)
    out = pd.DataFrame(index=players.index)
    for side in SIDES:
        out[f"rapm_{side.name}"] = side.sign * centred(alone[side.column].to_numpy(), players.possessions)
    return out


def label_agreement(league: League, labels: dict) -> dict:
    """Each side's label against the player's true impact averaged over the label's seasons: the correlation only,
    since the offensive label keeps the ridge's shrinkage and the defensive one is un-shrunk (rapm.unshrink)."""
    seen = league.truth[league.truth.season != league.rated]
    career = seen.groupby("player_id")[["impact_O", "impact_D"]].mean()
    out = {}
    for side in SIDES:
        label = labels[side.key].set_index("player_id")
        truth = career[side.impact].reindex(label.index)
        out[side.name] = {"correlation": agreement(label[side.column], truth)["correlation"], "players": len(label)}
    return out


def say_league(league: League, players: pd.DataFrame) -> None:
    """Print the league: its size, how talent persists, the rated season's players and how far apart they are."""
    shape = league.shape
    print(f"\nThe league: {len(league.seasons)} seasons ({league.seasons[0]}-{league.seasons[-1]}), {shape['n_teams']} "
          f"teams of {shape['players_per_team']} players, {league.n_stints:,} stints", flush=True)
    say(f"Every player's true impact is set before the games are played, and persists from one season to the next "
        f"with correlation {TALENT['rho']}; {TALENT['turnover']:.0%} of the players change teams each season.  "
        f"{league.rated} is the rated season: {len(players)} players, {players.possessions.median():,.0f} possessions "
        f"for the median one (from {players.possessions.min():,.0f} to {players.possessions.max():,.0f}).  Their "
        f"true impacts spread by a standard deviation of {players.true_offence.std():.1f} points per 100 on offence "
        f"and {players.true_defence.std():.1f} on defence.")


def say_step1(results: dict, players: pd.DataFrame) -> None:
    """Print step 1: the one-season RAPM against the truth, the pull toward 0, and the label."""
    step, penalty = results["step1"], SETTINGS.labels.offense_penalty
    bench, starters = tiers(players)
    print(f"\nStep 1: plain RAPM (rapm.py), penalty {penalty:,.0f}", flush=True)
    say(f"On {results['league']['rated']} alone, the correlation with the true impact is "
        f"{step['one_season']['offence']['correlation']:.2f} on offence and "
        f"{step['one_season']['defence']['correlation']:.2f} on defence; the root mean square error is "
        f"{step['one_season']['offence']['error']:.1f} and {step['one_season']['defence']['error']:.1f} points per "
        f"100.")
    n_bench, n_starter = players.possessions[bench].median(), players.possessions[starters].median()
    say(f"Players with few possessions are pulled hard toward zero: a ridge keeps about n / (n + {penalty:,.0f}) of "
        f"what n possessions say, so the median bench player ({n_bench:,.0f} possessions) keeps "
        f"{n_bench / (n_bench + penalty):.0%} and the median starter ({n_starter:,.0f}) keeps "
        f"{n_starter / (n_starter + penalty):.0%}.  The bench players' offensive ratings spread by a standard "
        f"deviation of {players.rapm_offence[bench].std():.2f} points per 100; their true impacts spread by "
        f"{players.true_offence[bench].std():.2f}.")
    say(f"On the other seasons together, the label: {step['label']['offence']['players']} players with "
        f"{SETTINGS.labels.min_possessions:.0f} possessions or more, correlation "
        f"{step['label']['offence']['correlation']:.2f} on offence and {step['label']['defence']['correlation']:.2f} "
        f"on defence with their true impact over those seasons.  More seasons, more possessions, a better estimate; "
        f"but it is one number for several seasons, and a player has one only if he played in them.")


# ------------------------------------------------------------------------------------------------- step 2
def box_score_prior(league: League, labels: dict) -> dict:
    """{"O", "D"}: each side's box_prior.SidePrior for the rated season, with the shipped booster settings."""
    params = booster_params(load_config(), SETTINGS.prior.booster_file)
    return box_prior.season_priors(labels, league.panel, league.rated, [league.rated], SETTINGS, params)


def prior_agreement(priors: dict, players: pd.DataFrame) -> dict:
    """Each side's prior against the true impact: the correlation only, since the prior is on the label's scale
    (step 3 sets the scale)."""
    out = {}
    for side in SIDES:
        prior = side.sign * pd.Series(priors[side.key].values).reindex(players.player_id).to_numpy(dtype=float)
        out[side.name] = {"correlation": agreement(prior, players[f"true_{side.name}"])["correlation"]}
    return out


def impact_parts(league: League) -> dict:
    """Per side, the standard deviation over the rated season's players of the two parts of the true impact: the
    box score's part (his true rates times the fixed weights) and the part no box count explains; and the
    ceiling, the correlation of the box score's part with the whole true impact: what a model that knew every
    player's true rates and the weights would reach."""
    truth = league.truth[league.truth.season == league.rated]
    rates = truth[[f"rate_{count}" for count in FEATURES]].to_numpy()
    return {side.name: {"box_score": float(np.std(rates @ league.weights[side.key])),
                        "unexplained": float(np.std(truth[side.unexplained])),
                        "ceiling": float(np.corrcoef(rates @ league.weights[side.key], truth[side.impact])[0, 1])}
            for side in SIDES}


def say_step2(results: dict, priors: dict, league: League) -> None:
    """Print step 2: what the booster learns from, how well its prior alone predicts the truth, and why defence is
    the weaker side."""
    step, parts = results["step2"], impact_parts(league)
    print("\nStep 2: the box-score prior (box_prior.py)", flush=True)
    say(f"LightGBM learns the label from {results['step1']['label']['offence']['players']} players' box scores over "
        f"{len(league.seasons) - 1} seasons (each player's career row and every run of 1, 2 and 3 seasons), reading "
        f"{len(priors['O'].feature_names)} inputs on offence and {len(priors['D'].feature_names)} on defence.  "
        f"Asked about {results['league']['rated']}'s box scores alone, it predicts the true impact with correlation "
        f"{step['offence']['correlation']:.2f} on offence and {step['defence']['correlation']:.2f} on defence.")
    say(f"Defence is the weaker side, partly because the box score sees less of it.  In this league most of the "
        f"defensive weight sits on steals, blocks, defensive rebounds and fouls (simulate.BETA_D).  The box score's "
        f"part of a player's true defensive impact spreads by a standard deviation of "
        f"{parts['defence']['box_score']:.1f} points per 100 and the part no box count explains by "
        f"{parts['defence']['unexplained']:.1f}; on offence the two are {parts['offence']['box_score']:.1f} and "
        f"{parts['offence']['unexplained']:.1f}.  So a model that knew every player's true rates would reach a "
        f"correlation of {parts['defence']['ceiling']:.2f} on defence and {parts['offence']['ceiling']:.2f} on "
        f"offence; what the prior falls short of that is the booster's own: it learns the weights from the labels "
        f"and reads one season's counts, not the true rates.")


# ------------------------------------------------------------------------------------------------- step 3
def prior_informed_rapm(league: League, priors: dict) -> pd.DataFrame:
    """Step 3's table for the rated season in the published schema (positive good on both ends)."""
    values = {key: priors[key].values for key in ("O", "D")}
    table = prior_rapm.rate_season({"pts": league.designs[league.rated]}, values, SETTINGS, league.rated)
    names = pd.DataFrame({"player_id": table.player_id, "player_name": "player " + table.player_id.astype(str)})
    return prior_rapm.published_table(table, names)


def tiers(players: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(bench, starters): the quarter of the players with the fewest possessions and the quarter with the most."""
    low, high = np.quantile(players.possessions, [0.25, 0.75])
    return (players.possessions <= low).to_numpy(), (players.possessions >= high).to_numpy()


def by_tier(players: pd.DataFrame) -> dict:
    """Per tier and side: the average size of a step-3 rating's two parts, its prior part s x prior and its games'
    part r, and the error of step 1 and of step 3."""
    out = {}
    for tier, members in zip(("bench", "starters"), tiers(players)):
        chosen = players[members]
        out[tier] = {side: {"prior_part": float(np.mean(np.abs(chosen[f"prior_{side}"]))),
                            "games_part": float(np.mean(np.abs(chosen[f"rating_{side}"] - chosen[f"prior_{side}"]))),
                            "error_step1": agreement(chosen[f"rapm_{side}"], chosen[f"true_{side}"])["error"],
                            "error_step3": agreement(chosen[f"rating_{side}"], chosen[f"true_{side}"])["error"]}
                     for side in ("offence", "defence")}
    return out


def say_negative_scale(results: dict) -> None:
    """Print what a prior scale below 0 means, when one is (the small league's defence)."""
    for side in SIDES:
        if results["step3"]["prior_scale"][side.name] < 0:
            say(f"A prior scale below 0 on {side.name} means that in this season's games the players the "
                f"{side.adjective} prior calls better did worse, so the fit uses the prior with its sign reversed.  "
                f"Here the {side.adjective} prior matches the true impact with correlation only "
                f"{results['step2'][side.name]['correlation']:.2f}: the shipped booster settings are tuned on thirty "
                f"NBA seasons and keep hundreds of training rows in every leaf (params/booster_params_lgb1.json), and "
                f"a league this small has few rows to split.")


def say_step3(results: dict) -> None:
    """Print step 3: the prior scales, the ratings against the truth, and where a rating comes from, by tier."""
    step, before = results["step3"], results["step1"]["one_season"]
    bench, starters = step["by_tier"]["bench"], step["by_tier"]["starters"]
    print(f"\nStep 3: prior-informed RAPM (prior_rapm.py), penalty {SETTINGS.ridge.penalty:,.0f}", flush=True)
    say(f"The games set the prior scale at {step['prior_scale']['offence']:.2f} on offence and "
        f"{step['prior_scale']['defence']:.2f} on defence: each player now starts from that multiple of his prior "
        f"rather than from 0.")
    say_negative_scale(results)
    say(f"The correlation with the true impact is {step['offence']['correlation']:.2f} on offence and "
        f"{step['defence']['correlation']:.2f} on defence (step 1 on the same season: "
        f"{before['offence']['correlation']:.2f} and {before['defence']['correlation']:.2f}); the root mean square "
        f"error is "
        f"{step['offence']['error']:.1f} and {step['defence']['error']:.1f} points per 100 (step 1: "
        f"{before['offence']['error']:.1f} and {before['defence']['error']:.1f}).")
    say(f"A bench player's rating is mostly his prior: on offence its prior part averages "
        f"{bench['offence']['prior_part']:.1f} points per 100 in size and the part his games add "
        f"{bench['offence']['games_part']:.1f}; a starter's games add {starters['offence']['games_part']:.1f} to a "
        f"prior part of {starters['offence']['prior_part']:.1f}.  A player with few possessions has little evidence "
        f"of his own, so the box score says most of what is known about him; a starter's games can move him away "
        f"from it.")
    say(f"The offensive root mean square error fell from {bench['offence']['error_step1']:.1f} to "
        f"{bench['offence']['error_step3']:.1f} for the bench players and from "
        f"{starters['offence']['error_step1']:.1f} to {starters['offence']['error_step3']:.1f} for the starters; on "
        f"defence from {bench['defence']['error_step1']:.1f} to {bench['defence']['error_step3']:.1f} and from "
        f"{starters['defence']['error_step1']:.1f} to {starters['defence']['error_step3']:.1f}.")


# ------------------------------------------------------------------------------------------------- the tour
def player_table(players: pd.DataFrame, labels: dict, priors: dict, table: pd.DataFrame) -> pd.DataFrame:
    """Every number the tour computed per rated-season player: the truth, step 1 (one season, label), step 2 (the
    booster's prior, raw sign), step 3 (rating and its prior part, published sign)."""
    out = players.copy()
    for side in SIDES:
        label = labels[side.key].set_index("player_id")[side.column]
        out[f"label_{side.name}"] = label.reindex(out.player_id).to_numpy()
    for side in SIDES:
        out[f"booster_{side.name}"] = pd.Series(priors[side.key].values).reindex(out.player_id).to_numpy(dtype=float)
    for kind in ("prior", "rating"):
        out[f"{kind}_offence"] = table[f"{kind}_off"].to_numpy()
        out[f"{kind}_defence"] = table[f"{kind}_def"].to_numpy()
    return out


def tour(shape: dict) -> tuple[dict, pd.DataFrame]:
    """Steps 1-3 on the league `shape`, printing each step's story; returns (the summary, the player table)."""
    started = time.time()
    league = make_league(shape)
    sums, labels = plain_rapm(league)
    priors = box_score_prior(league, labels)
    table = prior_informed_rapm(league, priors)
    players = rated_players(league, table)
    players = players.join(one_season_rapm(league, sums, players))
    players = player_table(players, labels, priors, table)
    results = {"league": {**shape, **TALENT, "rated": league.rated, "stints": league.n_stints},
               "step1": {"one_season": {side: agreement(players[f"rapm_{side}"], players[f"true_{side}"])
                                        for side in ("offence", "defence")},
                         "label": label_agreement(league, labels)},
               "step2": prior_agreement(priors, players),
               "step3": {**{side: agreement(players[f"rating_{side}"], players[f"true_{side}"])
                            for side in ("offence", "defence")},
                         "prior_scale": {"offence": float(table.prior_scale_off.iloc[0]),
                                         "defence": float(table.prior_scale_def.iloc[0])},
                         "by_tier": by_tier(players)}}
    say_league(league, players)
    say_step1(results, players)
    say_step2(results, priors, league)
    say_step3(results)
    print(f"\n({time.time() - started:.0f} seconds)", flush=True)
    return results, players


def as_json(results: dict, players: pd.DataFrame) -> dict:
    """The summary and every player's numbers, as plain JSON types."""
    return {"summary": results, "players": {column: players[column].tolist() for column in players.columns}}


def main(argv=None) -> int:
    use_utf8()
    parser = argparse.ArgumentParser(description="Steps 1-3 on a simulated league whose true impacts are known.")
    parser.add_argument("--small", action="store_true", help="the small league, a few seconds")
    parser.add_argument("--json", metavar="FILE", help="also write every number computed to FILE")
    parser.add_argument("--write-golden", action="store_true",
                        help=f"rewrite {GOLDEN.relative_to(GOLDEN.parents[2]).as_posix()} (needs --small), after a "
                             f"change that is meant to move the numbers")
    args = parser.parse_args(argv)
    if args.write_golden and not args.small:
        parser.error("--write-golden records the small league: add --small")
    with one_thread():
        results, players = tour(LEAGUES["small" if args.small else "full"])
    for path in map(Path, filter(None, (args.json, GOLDEN if args.write_golden else None))):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(as_json(results, players), indent=1) + "\n", encoding="utf-8", newline="\n")
        print(f"wrote {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
