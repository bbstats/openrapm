"""The second prior (src/openrapm/second_prior.py): the eight context numbers, the boosters that never saw a player,
and the line fitted on swaps."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from openrapm import box_prior
from openrapm import second_prior as sp
from openrapm import swap_adjust as sa
from openrapm import swap_test as sw


def _answers(values: dict) -> sp.BoosterAnswers:
    """One booster that answered `values` ({player_id: prior}) for every player."""
    ids = np.array(sorted(values))
    return sp.BoosterAnswers(player_ids=ids, values=np.array([[values[i]] for i in ids], dtype=float),
                             choice=np.zeros((len(ids), 1), dtype=int))


def _stints(rows) -> pd.DataFrame:
    """Stint rows from (game, team on offence, team on defence, possessions, offence's five, defence's five)."""
    return pd.DataFrame([dict(game_id=game, team_off=team_off, team_def=team_def, poss=float(poss),
                              **{f"o{k + 1}": player for k, player in enumerate(offence)},
                              **{f"d{k + 1}": player for k, player in enumerate(defence)})
                         for game, team_off, team_def, poss, offence, defence in rows])


# team 1: player 7 plays the first stint, sits the second; team 2 fields 11-15, then 16-20
TWO_STINTS = _stints([("g1", 1, 2, 10, [7, 1, 2, 3, 4], [11, 12, 13, 14, 15]),
                      ("g1", 1, 2, 5, [1, 2, 3, 4, 5], [16, 17, 18, 19, 20])])
OFFENCE = {7: 3.0, 1: 1.0, 2: 0.0, 3: 0.0, 4: -1.0, 5: 2.0, **{p: 0.5 for p in range(11, 16)},
           **{p: -0.5 for p in range(16, 21)}}
DEFENCE = {**{p: 0.0 for p in OFFENCE}, 1: 1.0}


def test_the_docstring_example_on_the_court_and_on_the_bench():
    numbers = sp.context_numbers(TWO_STINTS, _answers(OFFENCE), _answers(DEFENCE), pad=False).loc[7]
    assert numbers.teammates_offence_on == 0.0                        # (1 + 0 + 0 - 1) / 4
    assert np.isclose(numbers.teammates_offence_bench, 0.4)           # (1 + 0 + 0 - 1 + 2) / 5
    assert np.isclose(numbers.opponents_offence_on, 0.5) and np.isclose(numbers.opponents_offence_bench, -0.5)
    assert np.isclose(numbers.teammates_defence_on, 0.25) and np.isclose(numbers.teammates_defence_bench, 0.2)


def test_a_player_who_never_sat_gets_his_teams_level_as_his_bench_numbers():
    numbers = sp.context_numbers(TWO_STINTS, _answers(OFFENCE), _answers(DEFENCE), pad=False).loc[1]
    # 10 possessions beside 7, 2, 3, 4 and 5 beside 2, 3, 4, 5
    assert np.isclose(numbers.teammates_offence_on, (10 * (3 + 0 + 0 - 1) + 5 * (0 + 0 - 1 + 2)) / (4 * 15))
    # team 1's fives averaged 3 / 5 for 10 possessions and 2 / 5 for 5; the fives it faced +0.5 and -0.5
    assert np.isclose(numbers.teammates_offence_bench, (10 * 0.6 + 5 * 0.4) / 15)
    assert np.isclose(numbers.opponents_offence_bench, (10 * 0.5 - 5 * 0.5) / 15)
    padded = sp.context_numbers(TWO_STINTS, _answers(OFFENCE), _answers(DEFENCE)).loc[1]
    assert np.isclose(padded.teammates_offence_bench, numbers.teammates_offence_bench)


def test_the_pad_halves_two_numbers_whose_spread_is_half_noise():
    padded = sp.pad_toward_level([1.0, -1.0, 3.0], [100.0, 100.0, 0.0], [0.0, 0.0, 0.5], 50.0)
    assert np.allclose(padded, [0.5, -0.5, 0.5])        # the docstring example; no possessions: the level


def test_padding_moves_every_number_toward_its_level_and_never_past_it(lineup_league):
    stints, truth, _ = lineup_league(1, games=60)
    rows = sw.season_rows(stints)
    prior = {side: truth.set_index("player_id")[column] for side, column in (("O", "o"), ("D", "d"))}
    answers = {side: _answers(prior[side].to_dict()) for side in prior}
    raw = sp.context_numbers(rows, answers["O"], answers["D"], pad=False)
    padded = sp.context_numbers(rows, answers["O"], answers["D"])
    ids = np.sort(truth.player_id.to_numpy())
    for side, word in (("O", "offence"), ("D", "defence")):
        _, levels = sp._Entries(rows, ids).numbers(prior[side].reindex(ids).to_numpy()[:, None])
        for kind in sp.KINDS:
            name = kind.format(word)
            level = pd.Series(levels[name.split("_")[0]][:, 0], index=ids).reindex(raw.index)
            assert ((padded[name] - level) * (raw[name] - level) >= -1e-12).all(), name     # never past the level
            assert ((padded[name] - level).abs() <= (raw[name] - level).abs() + 1e-12).all(), name
            assert not np.allclose(padded[name], raw[name]), name                          # and it does pad


def test_every_teammate_is_read_on_the_booster_that_never_saw_the_player():
    ids = np.array(sorted(OFFENCE))
    # column 0 answers 1.0 for everyone, column 1 answers 2.0; only player 7 is answered by column 1
    choice = np.where(ids == 7, 1, 0)[:, None]
    offence = sp.BoosterAnswers(player_ids=ids, values=np.column_stack([np.ones(len(ids)), np.full(len(ids), 2.0)]),
                                choice=choice)
    numbers = sp.context_numbers(TWO_STINTS, offence, _answers(DEFENCE), pad=False)
    assert numbers.loc[7].teammates_offence_on == 2.0      # his teammates read on HIS booster, not their own
    assert numbers.loc[1].teammates_offence_on == 1.0      # 7 read on 1's booster, which answers 1.0


@dataclass
class _Times:
    """A stand-in booster: answers k x the first input."""
    k: float

    def predict(self, matrix):
        return self.k * np.asarray(matrix, dtype=float)[:, 0]


def test_the_answers_give_back_the_box_score_prior_exactly():
    members = [box_prior.OutOfPlayerPrior(full=_Times(1.0), fold_models=[_Times(2.0), _Times(3.0)],
                                          held_out_players=[{1}, {2}], inputs=["x"]),
               box_prior.OutOfPlayerPrior(full=_Times(5.0), fold_models=[_Times(7.0), _Times(11.0)],
                                          held_out_players=[{2, 3}, {4}], inputs=["x"])]
    prior = box_prior.BaggedPrior(members=members, inputs=["x"])
    rows = pd.DataFrame({"player_id": [1, 2, 3, 4, 5], "x": [1.0, -2.0, 0.5, 3.0, 1.5]})
    answers = sp.booster_answers(prior, rows)
    assert answers.values.shape == (5, 6) and answers.choice.tolist() == [[1, 3], [2, 4], [0, 4], [0, 5], [0, 3]]
    assert np.allclose(sp.own_priors(answers).to_numpy(), box_prior.predict(prior, rows), rtol=0, atol=1e-12)


def test_the_target_adds_back_the_two_swapped_players_ratings():
    pairs = {side: pd.DataFrame({"team": [1], "fifth_a": [7], "fifth_b": [8], "h": [2.0], "r": [0.5]})
             for side in sa.SIDES}
    ratings = pd.DataFrame({"player_id": [7, 8], "o": [2.0, 1.0], "d": [-1.0, 0.5]})
    targets = sp.swap_targets(pairs, ratings)
    assert targets["offense"].r.iloc[0] == 1.5 and targets["defense"].r.iloc[0] == -1.0


def test_each_player_reads_the_line_fitted_without_his_fold():
    inputs = pd.DataFrame(np.ones((2, len(sp.INPUTS))), index=[10, 11], columns=sp.INPUTS)
    coefficients = np.arange(1, 7, dtype=float)[:, None] * np.ones(len(sp.INPUTS)) / len(sp.INPUTS)
    values = sp.second_priors(inputs, coefficients, 5)
    assert np.isclose(values.loc[10], 1.0) and np.isclose(values.loc[11], 2.0)   # player 10 is fold 0, 11 fold 1


def test_the_line_learns_from_seasons_away_from_the_rated_one(lineup_league):
    stints, truth, _ = lineup_league(0, games=400, noise=0.3)
    rows = sw.season_rows(stints)
    box = {"O": truth.set_index("player_id").o, "D": truth.set_index("player_id").d}
    answers = {side: _answers(box[side].to_dict()) for side in ("O", "D")}
    context = sp.context_numbers(rows, answers["O"], answers["D"])
    inputs = {side: sp.line_inputs(box[side], context) for side in ("O", "D")}
    zero = truth.assign(o=0.0, d=0.0)
    grams = sp.held_back_grams(sw.prepare(rows), zero, inputs, 5)
    by_season = {2000: grams, 2001: grams}
    assert sp.season_coefficients(by_season, 2001, {2000, 2001}, 1, 5, 1e-4) is None       # 2000 is a neighbour
    line = sp.season_coefficients(by_season, 2002, {2000, 2001}, 1, 5, 1e-4)
    assert set(line) == {"O", "D"} and line["O"].shape == (6, len(sp.INPUTS))
    # The lineups' points were drawn from the true ratings, so inside each team the line must give back the truth's
    # gaps.  Its single weights are not pinned down here: with lineups drawn at random, a player's teammates on the
    # court are his team minus himself, so inside a team his teammates' numbers repeat his own prior.
    team = pd.Series({player: player // 100 for player in truth.player_id})
    for side, column in (("O", "o"), ("D", "d")):
        line_value = sp.second_priors(inputs[side], np.tile(line[side][-1], (6, 1)), 5)
        frame = pd.DataFrame({"line": line_value, "truth": truth.set_index("player_id")[column], "team": team})
        within = frame[["line", "truth"]] - frame.groupby("team")[["line", "truth"]].transform("mean")
        slope = float((within.line * within.truth).sum() / (within.truth ** 2).sum())
        assert within.line.corr(within.truth) > 0.95 and abs(slope - 1.0) < 0.15, side


def test_held_team_totals_keep_every_teams_total_and_the_lines_gaps_inside_it(lineup_league):
    stints, truth, _ = lineup_league(2, games=40)
    rows = sw.season_rows(stints)
    box = truth.set_index("player_id").o
    line = box * 1.5 + 0.3 * (box.index.to_numpy() // 100)        # wider, and a level per team the swaps never saw
    team_poss = sp.team_possessions(rows, "O")
    held = sp.hold_team_totals(line, box, team_poss)
    joined = team_poss.assign(held=team_poss.player_id.map(held), box=team_poss.player_id.map(box),
                              line=team_poss.player_id.map(line))
    totals = joined.groupby("team").apply(lambda team: np.sum(team.poss * (team.held - team.box)), include_groups=False)
    assert np.allclose(totals.to_numpy(), 0.0, atol=1e-8)
    # one team per player here, so inside a team the gaps are the line's exactly
    gaps = joined.assign(team_mean=joined.groupby("team").held.transform("mean"),
                         line_mean=joined.groupby("team").line.transform("mean"))
    assert np.allclose(gaps.held - gaps.team_mean, gaps.line - gaps.line_mean, atol=1e-8)


def test_the_held_totals_docstring_example():
    held = sp.hold_team_totals(pd.Series({1: 1.0, 2: 0.0}), pd.Series({1: 0.0, 2: 0.0}),
                               pd.DataFrame({"player_id": [1, 2], "team": [9, 9], "poss": [600.0, 400.0]}))
    assert np.allclose(held.to_numpy(), [0.4, -0.6])


def test_the_held_spread_docstring_example():
    held, factor = sp.hold_spread(pd.Series({1: 3.0, 2: -1.0}), pd.Series({1: 1.0, 2: -1.0}),
                                  pd.DataFrame({"player_id": [1, 2], "team": [9, 9], "poss": [500.0, 500.0]}))
    assert np.isclose(factor, 0.5) and np.allclose(held.to_numpy(), [2.0, 0.0])


def test_the_held_spread_keeps_team_means_and_order_and_takes_the_anchors_spread_inside_teams(lineup_league):
    # experiment 90: after the team totals, the line's spread inside teams pulled back to b x box-score prior's
    stints, truth, _ = lineup_league(2, games=40)
    rows = sw.season_rows(stints)
    box = truth.set_index("player_id").o
    line = box * 1.5 + 0.2 * np.sin(box.index.to_numpy())          # wider than the anchor, and reordered a little
    team_poss = sp.team_possessions(rows, "O")
    totals = sp.hold_team_totals(line, box, team_poss)
    held, factor = sp.hold_spread(totals, box, team_poss)
    assert 0.0 < factor < 1.0
    joined = team_poss.assign(held=team_poss.player_id.map(held), box=team_poss.player_id.map(box),
                              totals=team_poss.player_id.map(totals))

    def within(column):
        mean = joined.groupby("team")[column].transform(lambda x: np.average(x, weights=joined.loc[x.index, "poss"]))
        return joined[column] - mean

    spread = {column: np.average(within(column) ** 2, weights=joined.poss) for column in ("held", "box")}
    assert np.isclose(spread["held"], spread["box"], rtol=1e-9)
    for _, team in joined.groupby("team"):
        assert np.isclose(np.average(team.held, weights=team.poss), np.average(team.totals, weights=team.poss))
        assert (team.held.rank().to_numpy() == team.totals.rank().to_numpy()).all()


def test_the_spread_held_second_prior_is_the_held_one_when_the_line_is_b_x_box(lineup_league):
    stints, truth, _ = lineup_league(3, games=40)
    rows = sw.season_rows(stints)
    box = {side: truth.set_index("player_id")[column].to_dict() for side, column in (("O", "o"), ("D", "d"))}
    answers = {side: _answers(box[side]) for side in box}
    weights = np.zeros((6, len(sp.INPUTS)))
    weights[:, 0] = 3.5
    held, _ = sp.season_second_priors(rows, answers, box, {"O": weights, "D": weights}, 5, "held")
    spread, _ = sp.season_second_priors(rows, answers, box, {"O": weights, "D": weights}, 5, "spread_held")
    for side in ("O", "D"):
        assert np.allclose(pd.Series(spread[side]).sort_index(), pd.Series(held[side]).sort_index(), atol=1e-9)


def test_step_5s_numbers_are_standardised_on_one_scale_and_join_the_line():
    rows = pd.DataFrame({"player_id": [1, 2, 3], **{name: [1.0, 2.0, 3.0] for name in sp.TYPES}})
    scale = sp.type_scale(rows)
    assert np.isclose(scale["age"][0], 2.0) and np.isclose(scale["age"][1], np.sqrt(2.0 / 3.0))
    types = sp.standardised_types(rows, scale)
    assert np.allclose(types.loc[3].to_numpy(), np.sqrt(1.5))
    context = pd.DataFrame(0.0, index=pd.Index([1, 2, 3], name="player_id"), columns=sp.CONTEXT)
    box = {"O": {1: 0.5, 2: 0.0, 3: -0.5}, "D": {1: -1.0, 2: 0.0, 3: 1.0}}
    inputs = sp.side_inputs(box, context, types)
    assert list(inputs["O"].columns) == sp.INPUT_SETS["context_and_types"]
    assert inputs["O"].loc[1, sp.OTHER_BOX] == -1.0 and inputs["D"].loc[1, sp.OTHER_BOX] == 0.5
    assert list(sp.side_inputs(box, context)["D"].columns) == sp.INPUTS


def test_with_no_context_weight_the_held_second_prior_is_the_line_itself(lineup_league):
    # experiment 81's first run held team totals at the box-score prior's own scale, a third of the line's; with the
    # context weighing nothing the line is b x box-score prior, and holding its team totals must change nothing
    stints, truth, _ = lineup_league(3, games=40)
    rows = sw.season_rows(stints)
    box = {side: truth.set_index("player_id")[column].to_dict() for side, column in (("O", "o"), ("D", "d"))}
    answers = {side: _answers(box[side]) for side in box}
    weights = np.zeros((6, len(sp.INPUTS)))
    weights[:, 0] = 3.5
    held, _ = sp.season_second_priors(rows, answers, box, {"O": weights, "D": weights}, 5, "held")
    for side in ("O", "D"):
        expected = pd.Series(box[side]) * 3.5
        got = pd.Series(held[side])
        assert np.allclose(got.to_numpy(), expected.reindex(got.index).to_numpy(), atol=1e-9), side


def test_the_line_can_train_on_the_nearest_seasons_only(lineup_league):
    stints, truth, _ = lineup_league(4, games=40)
    rows = sw.season_rows(stints)
    box = {side: truth.set_index("player_id")[column] for side, column in (("O", "o"), ("D", "d"))}
    answers = {side: _answers(box[side].to_dict()) for side in box}
    context = sp.context_numbers(rows, answers["O"], answers["D"])
    grams = sp.held_back_grams(sw.prepare(rows), truth.assign(o=0.0, d=0.0),
                               {side: sp.line_inputs(box[side], context) for side in box}, 5)
    scaled = {side: sa.PairGrams(G=2 * g.G, b=-g.b, G_fold=2 * g.G_fold, b_fold=-g.b_fold, n=g.n)
              for side, g in grams.items()}
    by_season = {2010: grams, 2011: grams, 2014: scaled, 2018: grams, 2019: grams}
    trainable = set(by_season)
    every = sp.season_coefficients(by_season, 2016, trainable, 1, 5, 1e-4)
    near = sp.season_coefficients(by_season, 2016, trainable, 1, 5, 1e-4, nearest=1)
    only_2014 = sp.season_coefficients({2014: scaled}, 2016, trainable, 1, 5, 1e-4)
    assert np.allclose(near["O"], only_2014["O"]) and not np.allclose(every["O"], near["O"])


def test_the_three_input_prior_reads_own_prior_teammates_same_side_and_opponents_other_side_as_sums():
    context = pd.DataFrame({name: [0.0, 0.0] for name in sp.CONTEXT}, index=pd.Index([1, 2], name="player_id"))
    context["teammates_defence_on"] = [0.5, -0.25]
    context["opponents_offence_on"] = [0.2, 0.0]
    context["teammates_offence_on"] = [1.0, 1.0]
    frame = sp.three_inputs(pd.Series({1: -0.4, 2: 0.3}), context, "D")
    assert list(frame.columns) == sp.INPUT_SETS["own_teammates_opponents"]
    assert np.allclose(frame.loc[1].to_numpy(), [-0.4, 2.0, 1.0])       # the docstring's sums: 4 x 0.5, 5 x 0.2
    assert np.allclose(sp.three_inputs(pd.Series({1: 0.0}), context, "O").loc[1, sp.SAME_TEAMMATES], 4.0)


def test_centred_inputs_average_zero_by_possessions():
    out = sp.centred(pd.DataFrame({"x": [1.0, 5.0]}, index=[7, 8]), pd.Series({7: 300.0, 8: 100.0}))
    assert np.allclose(out.x.to_numpy(), [-1.0, 3.0])                    # the docstring example


def test_centred_inputs_take_away_the_level_each_player_folds_line_carries(lineup_league):
    # experiment 82's lines differ by player fold, and on raw inputs each fold's line had its own level; centred, two
    # folds whose lines differ by a constant on every weight give the average player the same second prior
    stints, truth, _ = lineup_league(3, games=40)
    rows = sw.season_rows(stints)
    box = {side: truth.set_index("player_id")[column].to_dict() for side, column in (("O", "o"), ("D", "d"))}
    answers = {side: _answers(box[side]) for side in box}
    weights = np.tile([1.0, 0.3, -0.2], (6, 1))
    weights[1] += 0.5                                       # fold 1's line differs from the others' on every weight
    values, inputs = sp.season_second_priors(rows, answers, box, {"O": weights, "D": weights}, 5, "free",
                                             input_set="own_teammates_opponents")
    for side in ("O", "D"):
        possessions = sp.team_possessions(rows, side).groupby("player_id").poss.sum()
        weight = possessions.reindex(inputs[side].index).fillna(0.0)
        assert np.allclose(np.average(inputs[side].to_numpy(), axis=0, weights=weight), 0.0, atol=1e-9)


def test_experiment_82s_inputs_centred_give_the_average_player_0_on_every_folds_line(lineup_league):
    # the ID-group bug (notes/DECISIONS.md, "Experiment 82's defensive prior read through"): on 82's raw inputs, fold
    # 1's line reads the average player differently from fold 0's; on the same inputs centred, both read him 0, and
    # every player moves by exactly the level his own fold's line carried
    stints, truth, _ = lineup_league(3, games=40)
    rows = sw.season_rows(stints)
    box = {side: truth.set_index("player_id")[column].to_dict() for side, column in (("O", "o"), ("D", "d"))}
    answers = {side: _answers(box[side]) for side in box}
    rng = np.random.default_rng(3)
    types = pd.DataFrame(rng.normal(0.5, 1.0, (len(truth), len(sp.TYPES))), columns=sp.TYPES,
                         index=pd.Index(truth.player_id.to_numpy(), name="player_id"))
    assert sp.INPUT_SETS["context_and_types_centred"] == sp.INPUT_SETS["context_and_types"]
    weights = np.tile(np.linspace(1.0, -0.5, len(sp.INPUT_SETS["context_and_types"])), (6, 1))
    weights[1] += 0.5                                       # fold 1's line differs from the others' on every weight
    lines = {"O": weights, "D": weights}
    raw, raw_inputs = sp.season_second_priors(rows, answers, box, lines, 5, "free", types, "context_and_types")
    fixed, fixed_inputs = sp.season_second_priors(rows, answers, box, lines, 5, "free", types,
                                                  "context_and_types_centred")
    for side in ("O", "D"):
        possessions = sp.team_possessions(rows, side).groupby("player_id").poss.sum()
        weight = possessions.reindex(raw_inputs[side].index).fillna(0.0)
        average_raw = np.average(raw_inputs[side].to_numpy(), axis=0, weights=weight)
        average_fixed = np.average(fixed_inputs[side].to_numpy(), axis=0, weights=weight)
        assert not np.isclose(average_raw @ weights[0], average_raw @ weights[1])
        assert np.allclose(average_fixed @ weights[:2].T, 0.0, atol=1e-9)
        ids = raw_inputs[side].index.to_numpy(dtype=np.int64)
        level = (weights[ids % 5] @ average_raw)
        gap = pd.Series(raw[side]).reindex(ids).to_numpy() - pd.Series(fixed[side]).reindex(ids).to_numpy()
        assert np.allclose(gap, level)
