"""Canonical team abbreviations.

The three upstream feeds disagree about how to spell a relocated franchise, and they
disagree in different directions:

  * `load_schedules` keeps the code in use during that season -- OAK, SD, STL.
  * `load_pbp` normalises every season to the franchise's modern code, so even 2015
    plays say LV, LAC and LA.
  * `load_ngs` also uses modern codes but spells the Rams LAR where schedules and
    play-by-play both say LA.

Nothing downstream notices a disagreement, because every join onto a game is a left
merge and `build_game_features` ends with a blanket `fillna(0.0)`. A miss therefore
produces a silently zero-filled feature rather than an error. Measured on the real
2016-2025 dataset before this module existed:

  * 78 of 2639 games (OAK/SD, 2016-2019) lost all five rating features.
  * 165 of 2639 games -- every Rams game in all ten seasons, 85 of them inside the
    2021-2025 backtest window -- lost all three NGS features to the LA/LAR mismatch.
    For comparison, all-zero NGS diffs outside week 1 numbered 0 of 2252 among the
    unaffected teams, so this was not imputation, it was a failed join.

Normalising at ingestion is what keeps that from recurring: the codes are made
consistent once, at the boundary, before any layer builds a key out of them.
"""

import pandas as pd

# Franchise codes as the modern feeds spell them. LAR (not LA) is canonical because
# that is what NGS uses and NGS is the only source that cannot be re-derived.
CANONICAL_TEAMS = frozenset(
    {
        "ARI",
        "ATL",
        "BAL",
        "BUF",
        "CAR",
        "CHI",
        "CIN",
        "CLE",
        "DAL",
        "DEN",
        "DET",
        "GB",
        "HOU",
        "IND",
        "JAX",
        "KC",
        "LAC",
        "LAR",
        "LV",
        "MIA",
        "MIN",
        "NE",
        "NO",
        "NYG",
        "NYJ",
        "PHI",
        "PIT",
        "SEA",
        "SF",
        "TB",
        "TEN",
        "WAS",
    }
)

# Historical or variant spellings -> canonical. Each maps a franchise to itself across
# its relocation, which is what makes a rating built from pre-move seasons apply to the
# post-move team.
TEAM_CODE_MAP = {
    "OAK": "LV",  # Raiders, Oakland -> Las Vegas (2020)
    "SD": "LAC",  # Chargers, San Diego -> Los Angeles (2017)
    "STL": "LAR",  # Rams, St. Louis -> Los Angeles (2016)
    "LA": "LAR",  # Rams, schedules/pbp spelling -> NGS spelling
}


def normalize_team_codes(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Rewrite the named columns to canonical team codes.

    Only the named columns are touched -- this must not guess which columns hold team
    codes, or it would rewrite unrelated strings. Absent columns are skipped so one
    call site can name the union of what the sources carry. Nulls are preserved (pbp
    has null posteam/defteam on timeouts and end-of-quarter rows).

    Raises ValueError on a code that is neither canonical nor a known alias: an
    unrecognised spelling means the upstream feed changed, and passing it through is
    exactly how the LA/LAR mismatch went unnoticed for the life of this project.
    """
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            continue
        values = out[col]
        present = set(values.dropna().unique())
        unknown = present - CANONICAL_TEAMS - set(TEAM_CODE_MAP)
        if unknown:
            raise ValueError(
                f"unrecognised team code(s) in column {col!r}: {sorted(unknown)}. "
                "Add them to TEAM_CODE_MAP (alias) or CANONICAL_TEAMS (new franchise)."
            )
        out[col] = values.map(lambda v: TEAM_CODE_MAP.get(v, v) if pd.notna(v) else v)
    return out
