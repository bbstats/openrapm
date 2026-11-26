"""One-off: the standard errors in the site's two ratings files (run once)."""
from edit_folds import edit

SITE = [
    ('''OPENRAPM_FILE = "decompose_openrapm.json"
OPENRAPM_FIELDS = ["n", "id", "p", "o", "d", "t", "tm", "w", "m", "wo", "wd", "wt"]
''',
     '''OPENRAPM_FILE = "decompose_openrapm.json"
OPENRAPM_FIELDS = ["n", "id", "p", "o", "d", "t", "tm", "w", "m", "wo", "wd", "wt"]
# The ratings' standard errors (rating_error.py), one more field at the end of each row when `openrapm errors` has
# written outputs/rating_errors_<name>.parquet beside the ratings table: [offense, defense, total] at two decimals,
# null for a player without one (every game of his in one fold).  Without that file the two site files are as they
# were, so a fresh clone and a build that never ran the errors write the same documents as before 2026-11-26.
ERRORS_FIELD = "se"
'''),
    ('''def openrapm_rows(rows: pd.DataFrame, pieces: pd.DataFrame, names: np.ndarray, teams: dict | None = None) -> list:
    """One season's rows in OPENRAPM_FIELDS order, best total first, players under 60 possessions left out (as
    ratings.json leaves them): name, NBA id, possessions (whole, as ratings.json writes them), each side's pieces at
    two decimals, then his teams ({player_id: "SAC/IND"}; "" for a player it lacks), then his wins over starter average
    at two decimals, his regular-season minutes, whole, and each side's WOSA_PARTS at two decimals."""
    teams = teams or {}''',
     '''def error_cell(errors: dict, player_id: int) -> list | None:
    """A row's `ERRORS_FIELD`: [offense, defense, total] at two decimals from {player_id: (se_off, se_def, se_total)},
    None for a player it lacks or whose standard errors are missing."""
    mine = errors.get(int(player_id))
    if mine is None or any(pd.isna(value) for value in mine):
        return None
    return [round(float(value), 2) for value in mine]


def openrapm_rows(rows: pd.DataFrame, pieces: pd.DataFrame, names: np.ndarray, teams: dict | None = None,
                  errors: dict | None = None) -> list:
    """One season's rows in OPENRAPM_FIELDS order, best total first, players under 60 possessions left out (as
    ratings.json leaves them): name, NBA id, possessions (whole, as ratings.json writes them), each side's pieces at
    two decimals, then his teams ({player_id: "SAC/IND"}; "" for a player it lacks), then his wins over starter average
    at two decimals, his regular-season minutes, whole, and each side's WOSA_PARTS at two decimals.  With `errors`,
    {player_id: (se_off, se_def, se_total)}, each row ends with its `ERRORS_FIELD` (`error_cell`)."""
    teams = teams or {}'''),
    ('''             *([round(float(r[column]), 2) for column in wosa[field]] for field in ("wo", "wd", "wt"))]
            for r in frame.to_dict("records")]''',
     '''             *([round(float(r[column]), 2) for column in wosa[field]] for field in ("wo", "wd", "wt")),
             *([] if errors is None else [error_cell(errors, r["player_id"])])]
            for r in frame.to_dict("records")]'''),
    ('''def write_openrapm_file(path: Path, rows: dict, built: str, penalty: float, opens_on: int | None = None,
                        percentile_min: dict | None = None, rookie_ids: dict | None = None) -> dict:''',
     '''def write_openrapm_file(path: Path, rows: dict, built: str, penalty: float, opens_on: int | None = None,
                        percentile_min: dict | None = None, rookie_ids: dict | None = None,
                        with_errors: bool = False) -> dict:'''),
    ('''    `rookie_ids`, each season's rookies (data/panel.py `rookies`), kept to the players the season's rows show, none
    when not given."""''',
     '''    `rookie_ids`, each season's rookies (data/panel.py `rookies`), kept to the players the season's rows show, none
    when not given.  `with_errors`: the rows end with their standard errors, and the fields with `ERRORS_FIELD`."""'''),
    ('''                              percentile_min=minimums, rookies=rookie_lists, fields=OPENRAPM_FIELDS,''',
     '''                              percentile_min=minimums, rookies=rookie_lists,
                              fields=OPENRAPM_FIELDS + ([ERRORS_FIELD] if with_errors else []),'''),
    ('''#    "rows": {season: [[name, id, possessions, teams, offense, defense, total, WOSA], ...]}}     the Detail file's order
''',
     '''#    "rows": {season: [[name, id, possessions, teams, offense, defense, total, WOSA], ...]}}     the Detail file's order
# When the Detail file carries the standard errors (`ERRORS_FIELD`), each row ends with the total's, null without one,
# and the fields with "se".
'''),
    ('''    Example: the Detail row ["Victor Wembanyama", 1641705, 620, [3.9, 0.3, 0.2, 0.0, 0.66, 5.06], [...], [...], "SAS",
    1.9, ...] gives ["Victor Wembanyama", 1641705, 620, "SAS", 5.06, 4.27, 9.33, 1.9]."""
    fields = detail["meta"]["fields"]
    at = {name: fields.index(name) for name in RATINGS_LIST_FIELDS}
    rating = detail["meta"]["pieces"].index("rating")
    rows = {season: [[row[at["n"]], row[at["id"]], row[at["p"]], row[at["tm"]], row[at["o"]][rating],
                      row[at["d"]][rating], row[at["t"]][rating], row[at["w"]]] for row in season_rows]
            for season, season_rows in detail["rows"].items()}
    meta = {key: detail["meta"][key] for key in RATINGS_LIST_META}
    meta.update(through={str(season): str(day) for season, day in sorted((through or {}).items())},
                fields=RATINGS_LIST_FIELDS)''',
     '''    Example: the Detail row ["Victor Wembanyama", 1641705, 620, [3.9, 0.3, 0.2, 0.0, 0.66, 5.06], [...], [...], "SAS",
    1.9, ...] gives ["Victor Wembanyama", 1641705, 620, "SAS", 5.06, 4.27, 9.33, 1.9]; ending [0.67, 0.7, 0.93], his
    standard errors, it gives the same with 0.93 after it."""
    fields = detail["meta"]["fields"]
    at = {name: fields.index(name) for name in RATINGS_LIST_FIELDS}
    rating = detail["meta"]["pieces"].index("rating")
    errors = fields.index(ERRORS_FIELD) if ERRORS_FIELD in fields else None

    def total_error(row) -> list:
        return [] if errors is None else [None if row[errors] is None else row[errors][-1]]

    rows = {season: [[row[at["n"]], row[at["id"]], row[at["p"]], row[at["tm"]], row[at["o"]][rating],
                      row[at["d"]][rating], row[at["t"]][rating], row[at["w"]], *total_error(row)]
                     for row in season_rows]
            for season, season_rows in detail["rows"].items()}
    meta = {key: detail["meta"][key] for key in RATINGS_LIST_META}
    meta.update(through={str(season): str(day) for season, day in sorted((through or {}).items())},
                fields=RATINGS_LIST_FIELDS + ([] if errors is None else [ERRORS_FIELD]))'''),
    ('''def write_openrapm_decomposition(cfg: dict, ratings_path: Path, data_dir: Path, built: str) -> None:
    """decompose_openrapm.json in `data_dir`, after its checks pass.  Solves step 3 again for every season and both
    sides, so it must run after `openrapm build`, whenever the ratings table changes."""
    table, step3, factors = published_parts(ratings_path, cfg)''',
     '''def errors_path(ratings_path: Path) -> Path:
    """The standard errors `openrapm errors` writes beside a ratings table: season_ratings_product.parquet's are
    rating_errors_product.parquet."""
    return ratings_path.with_name(f"rating_errors_{ratings_path.stem.removeprefix('season_ratings_')}.parquet")


def rating_errors(ratings_path: Path) -> dict | None:
    """{season: {player_id: (se_off, se_def, se_total)}} from `errors_path`, None when the file is not there."""
    path = errors_path(ratings_path)
    if not path.exists():
        return None
    table = pd.read_parquet(path)
    return {int(season): {int(row.player_id): (row.se_off, row.se_def, row.se_total)
                          for row in rows.itertuples(index=False)}
            for season, rows in table.groupby("season")}


def write_openrapm_decomposition(cfg: dict, ratings_path: Path, data_dir: Path, built: str) -> None:
    """decompose_openrapm.json in `data_dir`, after its checks pass.  Solves step 3 again for every season and both
    sides, so it must run after `openrapm build`, whenever the ratings table changes.  The rows carry the ratings'
    standard errors when `errors_path` is there."""
    table, step3, factors = published_parts(ratings_path, cfg)
    errors = rating_errors(ratings_path)'''),
    ('''        rows[season] = openrapm_rows(season_rows, pieces, named, box_files.season_teams(season))''',
     '''        rows[season] = openrapm_rows(season_rows, pieces, named, box_files.season_teams(season),
                                     None if errors is None else errors.get(season, {}))'''),
    ('''    document = write_openrapm_file(path, rows, built, penalty, default_season(table), percentile_minimums(cfg),
                                   rookies(cfg))''',
     '''    document = write_openrapm_file(path, rows, built, penalty, default_season(table), percentile_minimums(cfg),
                                   rookies(cfg), with_errors=errors is not None)
    if errors is not None:
        without = {season: sum(1 for row in season_rows if row[-1] is None) for season, season_rows in rows.items()}
        print(f"  standard errors from {errors_path(ratings_path).name}: {sum(without.values())} of "
              f"{sum(len(v) for v in rows.values()):,} rows without one"
              + "".join(f"; {season}: {count}" for season, count in without.items() if count > 20), flush=True)'''),
]

if __name__ == "__main__":
    edit("src/openrapm/commands/site.py", SITE)
