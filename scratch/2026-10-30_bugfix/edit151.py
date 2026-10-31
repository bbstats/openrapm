"""One-off edit of scripts/checks/151_wowy.py for the bug-fix pass of 2026-10-30 (kept for the record)."""
from pathlib import Path

p = Path("scripts/checks/151_wowy.py")
s = p.read_text(encoding="utf-8")


def sub(old, new, count=1):
    global s
    assert s.count(old) == count, (old[:70], s.count(old))
    s = s.replace(old, new)


sub('''def within_layer(folds: list, where: dict, seasons, defence_factors: tuple,
                 within_factors: tuple = WITHIN_FACTORS) -> pd.DataFrame | None:
    """The shipped step 4's within layer for `seasons`, indexed by season: its four multipliers fitted on the system's
    own held-back stints within games, test rule, 8 nearest fold seasons, times the within factors and the defence
    factor (commands/build.within_layer_multipliers); None when the shipped step 4 has one layer."""
    if not PRODUCT.shrink.within_team_layer:
        return None
    from dataclasses import replace

    from openrapm.commands.build import within_layer_multipliers
    from openrapm.data.config import load_config
    shrink = replace(PRODUCT.shrink, rule="test", nearest=NEAREST, extrapolate_folds_tag="",
                     defence_factors=tuple(defence_factors), within_factors=tuple(within_factors))
    return within_layer_multipliers(load_config(), seasons, folds, where, shrink).set_index("season")


def step4_tables(within, folds: list, where: dict, scored_seasons: list, defence_factors: tuple,
                 step4_from: str | None, within_factors: tuple = WITHIN_FACTORS
                 ) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """(step 4's multipliers by season, its within layer by season or None), fitted on the system's own folds, or on
    the folds of `step4_from` when given."""
    from openrapm.prior_shrink import multipliers_by_season
    if step4_from:
        folds, where = system_folds(within, step4_from)
    multipliers = multipliers_by_season(scored_seasons, folds, "test", NEAREST, free_games=FREE_GAMES)
    return multipliers.set_index("season"), within_layer(folds, where, scored_seasons, defence_factors, within_factors)''',
    '''def shrink_settings(flags: str = ""):
    """The step 4 a system was built with, from the flags its `openrapm shrink` was given (the chain's SHRINK_FLAGS),
    read by the same options `openrapm shrink` reads, so a flag means the same in both; at the test rule and without
    the full-season read, as every system here is scored.  "" is the shipped step 4.

    Why (2026-10-30): this check rebuilt every system's step 4 at the shipped settings, so a candidate that changed
    the within factors, the nearest seasons or the layers was read unchanged on the test that decides, while the
    year-over-year test saw the change."""
    import shlex
    from dataclasses import replace

    from openrapm.commands.cli import _add_folds_tag, shrink_options
    parser = argparse.ArgumentParser(prog="the step-4 flags", add_help=False, allow_abbrev=False)
    _add_folds_tag(parser)
    return replace(shrink_options(PRODUCT.shrink, parser.parse_args(shlex.split(flags))), rule="test",
                   extrapolate_folds_tag="")


def within_layer(folds: list, where: dict, seasons, defence_factors: tuple,
                 within_factors: tuple = WITHIN_FACTORS, shrink=None) -> pd.DataFrame | None:
    """The shipped step 4's within layer for `seasons`, indexed by season: its four multipliers fitted on the system's
    own held-back stints within games, test rule, 8 nearest fold seasons, times the within factors and the defence
    factor (commands/build.within_layer_multipliers); None when the shipped step 4 has one layer.  With `shrink`
    (`shrink_settings`), that step 4's layer instead, and `defence_factors` and `within_factors` are not read."""
    from dataclasses import replace

    from openrapm.commands.build import within_layer_multipliers
    from openrapm.data.config import load_config
    if shrink is None:
        if not PRODUCT.shrink.within_team_layer:
            return None
        shrink = replace(PRODUCT.shrink, rule="test", nearest=NEAREST, extrapolate_folds_tag="",
                         defence_factors=tuple(defence_factors), within_factors=tuple(within_factors))
    elif not shrink.within_team_layer:
        return None
    return within_layer_multipliers(load_config(), seasons, folds, where, shrink).set_index("season")


def step4_tables(within, folds: list, where: dict, scored_seasons: list, defence_factors: tuple,
                 step4_from: str | None, within_factors: tuple = WITHIN_FACTORS, shrink=None
                 ) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """(step 4's multipliers by season, its within layer by season or None), fitted on the system's own folds, or on
    the folds of `step4_from` when given; with `shrink` (`shrink_settings`), by that step 4's rule: its nearest count,
    games' part, fold seasons and layers."""
    from openrapm.prior_shrink import multipliers_by_season
    if step4_from:
        folds, where = system_folds(within, step4_from)
    nearest, free_games = (NEAREST, FREE_GAMES) if shrink is None else (shrink.nearest, shrink.free_games_part)
    if shrink is not None:
        folds = [fold for fold in folds if shrink.fold_seasons[0] <= fold.season <= shrink.fold_seasons[1]]
    multipliers = multipliers_by_season(scored_seasons, folds, "test", nearest, free_games=free_games)
    return multipliers.set_index("season"), within_layer(folds, where, scored_seasons, defence_factors, within_factors,
                                                         shrink)''')

sub('''                  label: str | None = None, step4_from: str | None = None, within_factors: tuple = WITHIN_FACTORS,
                  only=None) -> pd.DataFrame:''',
    '''                  label: str | None = None, step4_from: str | None = None, within_factors: tuple = WITHIN_FACTORS,
                  only=None, shrink=None) -> pd.DataFrame:''')

sub('''    `only`, a set of seasons, keeps the folds of those seasons alone (check 156 scores check 152's tune seasons)."""
    from openrapm.prior_shrink import WITHIN_COLUMNS
    folds, where = system_folds(within, name)
    scored = [fold for fold in folds if seasons[0] <= fold.season <= seasons[1]
              and (only is None or fold.season in only)]
    multipliers, layer = step4_tables(within, folds, where, sorted({fold.season for fold in scored}), defence_factors,
                                      step4_from, within_factors)''',
    '''    `only`, a set of seasons, keeps the folds of those seasons alone (check 156 scores check 152's tune seasons).
    `shrink` (`shrink_settings`), the system's own step 4, in place of `defence_factors` and `within_factors`."""
    from openrapm.prior_shrink import WITHIN_COLUMNS
    if shrink is not None:
        defence_factors = shrink.defence_factors
    folds, where = system_folds(within, name)
    scored = [fold for fold in folds if seasons[0] <= fold.season <= seasons[1]
              and (only is None or fold.season in only)]
    multipliers, layer = step4_tables(within, folds, where, sorted({fold.season for fold in scored}), defence_factors,
                                      step4_from, within_factors, shrink)''')

sub('''    to the usual rule, 4 x (n / 100)^(2/9) rounded down: 3 for 30 seasons.  With lags 0 it is the plain standard
    error, sd / sqrt(n), exactly."""''',
    '''    to the usual rule, 4 x (n / 100)^(2/9) rounded down: 3 for 30 seasons.  With lags 0 it is the plain standard
    error, sd / sqrt(n), exactly.  `paired` never reads it below the plain one."""''')

sub('''    replace.  Beside the plain standard error and z, `se_newey_west` and `z_newey_west` (`newey_west_se`)."""''',
    '''    replace.  Beside the plain standard error and z, `se_newey_west` and `z_newey_west` (`newey_west_se`), never below
    the plain standard error.

    Why the floor (2026-10-30): the Newey-West correction is there for neighbours that move together, which widens the
    standard error.  Among four builds of the same ratings, each with the box-score prior's players split another way,
    every pair's season differences leaned the other way, lag-one correlation -0.03 to -0.23, so the correction
    narrowed it: a noise control against today's ratings read plain z -1.83 and Newey-West z -2.44.  The four builds'
    30-season means have standard deviation 0.0074, which the plain standard error, 0.0094 for two of them, matches."""''')
sub('''        se_nw = newey_west_se(diff)''', '''        se_nw = float(np.fmax(newey_west_se(diff), se))''')

sub('''def main(argv=None) -> int:''',
    '''ERAS = ((1997, 2006), (2007, 2016), (2017, 2026))


def era_rows(seasons_table: pd.DataFrame, system: str, reference: str) -> list:
    """The shipped stage on every stint, the system against the reference, in each era and over all seasons: a win
    must hold in every era (the owner, 2026-10-20)."""
    rows = []
    for first, last in (*ERAS, (ERAS[0][0], ERAS[-1][1])):
        inside = seasons_table[seasons_table.season.between(first, last)]
        for row in paired(inside, system, reference):
            if row["stage"] == "step4" and row["slice"] == "all":
                rows.append(dict(era=f"{first}-{last}" if (first, last) in ERAS else "all seasons", **row))
    return rows


def parse_defence_factors(text: str) -> dict:
    """"cand:0.9,other:0.89,2017:0.84" -> {cand: ((0, 0.9),), other: ((0, 0.89), (2017, 0.84))}: a name starts a
    system's factors, and a season-numbered part adds an era to the system before it (check 156 prints that form;
    until 2026-10-30 its "2017:0.84" read as a system named 2017 and the 0.84 was lost)."""
    factors: dict = {}
    name = None
    for part in (part.strip() for part in text.split(",") if part.strip()):
        key, _, value = part.rpartition(":")
        if key.isdigit() and name is not None:
            factors[name] += ((int(key), float(value)),)
        else:
            name = key
            factors[name] = ((0, float(value)),)
    return factors


def main(argv=None) -> int:''')

sub('''    parser.add_argument("--defence-factors", default="",
                        help="name:K,... : step 4's defensive multipliers times K in every season for that "
                             "system instead of the shipped settings.ShrinkSettings.defence_factors (openrapm shrink "
                             "--defence-factor); a system built before experiment 76 is 1")''',
    '''    parser.add_argument("--defence-factors", default="",
                        help="name:K[,FIRST:K...],... : step 4's defensive multipliers times K for that system "
                             "instead of the shipped settings.ShrinkSettings.defence_factors, one factor or one by "
                             "era from its first season (openrapm shrink --defence-factor); a system built before "
                             "experiment 76 is 1")
    parser.add_argument("--shrink-flags", action="append", default=[], metavar="NAME:FLAGS",
                        help="the flags NAME's `openrapm shrink` was given, e.g. 'cand:--within-factors 0.8,1': its "
                             "step 4 is scored as built (the chain passes its SHRINK_FLAGS); once for each system")''')

sub('''    factors = {key: ((0, float(value)),) for key, value in (part.split(":") for part in args.defence_factors.split(",")
                                                            if part)}
    for name in names:
        tables.append(system_errors(audit, within, folders[name], (args.first, args.last), args.unit, families,
                                    multiplier_tables, factors.get(name, DEFENCE_FACTORS), label=name,
                                    step4_from=step4_from.get(name)))''',
    '''    factors = parse_defence_factors(args.defence_factors)
    shrinks = {name: shrink_settings(flags) for name, _, flags in (item.partition(":") for item in args.shrink_flags)}
    if set(factors) & set(shrinks):
        raise SystemExit(f"{sorted(set(factors) & set(shrinks))}: give a system's defence factor in its --shrink-flags")
    for name in names:
        if name in shrinks:
            print(f"  {name}: step 4 as built with {args.shrink_flags}", flush=True)
        tables.append(system_errors(audit, within, folders[name], (args.first, args.last), args.unit, families,
                                    multiplier_tables, factors.get(name, DEFENCE_FACTORS), label=name,
                                    step4_from=step4_from.get(name), shrink=shrinks.get(name)))''')

sub('''    print(summary.round(5).to_string(index=False))
    print(f"done in {time.time() - started:.0f}s")''',
    '''    print(summary.round(5).to_string(index=False))
    eras = pd.DataFrame([row for name in names if name != args.ref for row in era_rows(seasons_table, name, args.ref)])
    if len(eras):
        eras.to_csv(outputs / "csv" / f"wowy_{args.tag}_eras.csv", index=False)
        print("\\nthe shipped stage, every stint, by era; below 0 is better")
        print(eras[["system", "era", "mean_change", "se", "z", "z_newey_west", "seasons_better", "seasons"]]
              .round(5).to_string(index=False))
    print(f"done in {time.time() - started:.0f}s")''')

p.write_text(s, encoding="utf-8", newline="\n")
print("ok")
