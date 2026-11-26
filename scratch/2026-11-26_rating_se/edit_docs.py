"""One-off: the ratings' standard errors in PIPELINE.md, README.md and GLOSSARY.md (run once)."""
from edit_folds import edit

PIPELINE = [
    ('''## Wins over starter average (`wins.py`)
''',
     '''## The ratings' standard errors (`rating_error.py`)

- **For:** how far a rating would land from the one shown had the season's games come out as another draw of the same
  schedule. No rating moves.
- **Reads:** the held-back games with a tenth held back, `outputs/within/rd143_k10/`, and each season's published
  step-4 multipliers; for the season in progress, its own set, `outputs/within/daily_k10/`.
- **Writes:** `outputs/rating_errors_product.parquet`: player_id, season, se_off, se_def, se_total, repeats.
- **How:** the delete-a-group jackknife over games. Each of a season's 30 folds is the season rated without a tenth of
  its games, every input rebuilt from the games kept, then shrunk with the season's published multipliers. For each of
  the three repeats, se^2 = 9/10 x the sum of squares of the ten ratings about their mean; the three are averaged. A
  player whose every game fell in one fold has no rating there and that repeat is left out for him; with no complete
  repeat he has no standard error.
- **What it holds and leaves out:** it holds the games' noise in everything a rating reads from them, the box score,
  Box Adj and the ridge. It leaves out how far the box-score prior can miss a player whatever games are played, so the
  distance from a player's true value is larger.
- **Sizes, 1997-2026:** a player with 1,500 or more offensive possessions reads about 0.55 on offense, 0.55 on defense
  and 0.80 on the total; under 250 possessions the total reads 1.5. More of it comes from the prior's part than from
  the games' part: 0.47 against 0.40 on offense, 0.49 against 0.30 on defense.
- **The check:** the same jackknife on the quarter-held-back sets, four ratings a repeat, reads 6 to 8% smaller in
  every era, because a rating from three quarters of the games leans more on its prior.
- **The season in progress:** `openrapm folds --in-progress` rates it 30 more times as its build rates it, by that
  build's own boosters (`openrapm build --save-models`): the boosters told the share of the regular season played, the
  prior scales held and blended by it. The season so far rebuilt game by game sits within 0.16 of the published rows
  (2026-11-26; the median 0.01), because the build reads the season's hustle counts from the league's season-to-date
  dashboards and a fold can only sum the games it keeps; the command stops past 0.5.
- **Commands:**

  ```bash
  python -m openrapm errors                                    # 1997-2026 from rd143_k10, 22 seconds
  python -m openrapm folds --models daily --base season_ratings_product_step3 --seasons 2027 --tag daily_k10 \\
      --folds-per-repeat 10 --in-progress                      # 2 minutes on 2026-11-26
  python -m openrapm errors --seasons 2027 --folds daily_k10
  ```

- **On the site:** `openrapm site --openrapm-decomposition` reads the file when it is beside the ratings table and
  ends each row of `decompose_openrapm.json` with [offense, defense, total] and of `ratings_list.json` with the
  total's; without the file the two are as before. The Ratings page shows the total's beside Total and, in Detail,
  the shown side's beside its rating.
- **Code:** `src/openrapm/rating_error.py`, `commands/errors.py`, `commands/within_season.py` (`in_progress_run`),
  `commands/site.py`, `docs/index.html`.

## Wins over starter average (`wins.py`)
'''),
    ('''python -m openrapm build --seasons 2027 --replace-seasons    # 2027 in place of yesterday's 2027, nothing else moves
''',
     '''python -m openrapm build --seasons 2027 --replace-seasons --save-models daily    # 2027 in place of yesterday's
python -m openrapm folds --models daily --base season_ratings_product_step3 --seasons 2027 --tag daily_k10 \\
    --folds-per-repeat 10 --in-progress              # not a step: 2027 rated 30 more times, for its standard errors
python -m openrapm errors --seasons 2027 --folds daily_k10           # not a step either
'''),
    ('''| `wins.py` | WOSA |
''',
     '''| `wins.py` | WOSA |
| `rating_error.py` | the ratings' standard errors |
'''),
    ('''| `commands/second_prior.py` | `openrapm second-prior` |
''',
     '''| `commands/second_prior.py` | `openrapm second-prior` |
| `commands/errors.py` | `openrapm errors` |
'''),
]

README = [
    ('''Read from the finished ratings, not part of them: WOSA, wins over starter average (`wins.py`); the Power page''',
     '''Read from the finished ratings, not part of them: each rating's standard error (`rating_error.py`); WOSA, wins
over starter average (`wins.py`); the Power page'''),
    ('''python -m openrapm site --openrapm-decomposition --similarity --power --bias
''',
     '''python -m openrapm errors                             # every rating's standard error, from the tenth-held-back sets
python -m openrapm site --openrapm-decomposition --similarity --power --bias
'''),
    ('''      wins.py           WOSA: the win model, leverage, the pace ridge, the bar, the wins
''',
     '''      wins.py           WOSA: the win model, leverage, the pace ridge, the bar, the wins
      rating_error.py   a rating's standard error: the jackknife over the sets that hold back a tenth of the games
'''),
]

GLOSSARY = [
    ('''**The shrink.** Step 4 (`prior_shrink.py`)''',
     '''**Standard error of a rating.** How far a rating would land from the one shown had the season's games come out as
another draw of the same schedule (`rating_error.py`): the season is rated 30 more times, each without a tenth of its
games, and the spread of those ratings is scaled up to a whole season's. A regular reads about 0.8 on his total. It
holds the games' noise in the box score, Box Adj and the ridge; it leaves out how far the box-score prior can miss a
player whatever games are played.

**The shrink.** Step 4 (`prior_shrink.py`)'''),
]

if __name__ == "__main__":
    edit("PIPELINE.md", PIPELINE)
    edit("README.md", README)
    edit("GLOSSARY.md", GLOSSARY)
