# QB Starter Advisory (Track A) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Surface the expected starting quarterback, and how much of a change it is in EPA/dropback, on the live slate and web dashboard — without altering any model prediction, `edge_flag`, or tracker record.

**Architecture:** The depth-chart resolution machinery already exists in `src/nfl_game/ratings/depth.py` and `src/nfl_game/ratings/qb.py` but is pinned to kickoff time and consumed only by the Ridge-v2 research build. This plan parameterizes that cutoff, adds a game-level advisory frame, wraps a bounded-cache live provider mirroring `market/live.py`, and overlays the result onto the slate *after* prediction. Nothing enters `FEATURE_COLS`.

**Tech Stack:** Python 3.12, pandas, `nflreadpy`, pytest, ruff. Windows; the interpreter is `./.venv/Scripts/python.exe` from Git Bash.

## Global Constraints

- **`FEATURE_COLS` is not touched.** Track A adds no model input. `src/nfl_game/model/features.py:13` must be byte-identical at the end of this plan.
- **`edge_flag` keeps its exact current meaning** — `(spread_gap.abs() >= edge_threshold)`, `market/compare.py:54`. The advisory never changes it.
- **`model_margin` and `model_total` are never affected** by advisory presence, absence, or staleness.
- **Regression baseline holds.** `scripts/backtest.py --test-seasons 2021-2025` must still print margin MAE `10.274`, total MAE `10.684`, ATS `0.4977` (n=1326), O/U `0.5022` (n=1348), `model_coef` `-0.0218`, r² `0.2083`.
- **The tracker is untouched.** No change to `src/nfl_game/tracking/`, and `data/processed/tracker_ledger.parquet` keeps its 1,359 historical rows.
- **`web/` stays read-only.** The advisory is never written to a packaged artifact.
- **Team codes normalize at ingestion** via `src/nfl_game/data/teams.py`, which raises on an unrecognized code. No per-join fixups — a left merge ending in `fillna(0.0)` is how a `LA`/`LAR` mismatch once silently zeroed every Rams game across ten seasons.
- **`qb_features_for_targets`' existing numeric output must not move.** Its default behavior is what the Ridge-v2 artifacts were measured with; every new parameter defaults to the current behavior.
- Run `./.venv/Scripts/python.exe -m ruff check .` before every commit.

---

### Task 1: Parameterize the depth-chart cutoff

`_targets_from_schedule` (`src/nfl_game/ratings/qb.py:101`) hardcodes the depth-chart cutoff to kickoff time. That is why Ridge-v2 candidate C2 was measured with starter knowledge no published pick could have. Everything else in this plan depends on being able to ask "who was QB1 at *this* instant".

**Files:**
- Modify: `src/nfl_game/ratings/qb.py:101-117` (`_targets_from_schedule`), and `qb_features_for_targets` at `:139`
- Test: `tests/test_qb.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `nfl_game.ratings.qb.CutoffPolicy` — type alias `pd.Timestamp | pd.Timedelta | None`
  - `_targets_from_schedule(schedules: pd.DataFrame, targets: list[tuple[int, int]], cutoff: CutoffPolicy = None) -> pd.DataFrame`
  - `qb_features_for_targets(qb_weeks, depth_history, schedules, targets, cutoff: CutoffPolicy = None) -> pd.DataFrame` — same columns as today.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_qb.py`. This tests the cutoff in **both directions** — a single-cutoff test would leave "the argument is ignored" invisible.

```python
import pandas as pd
import pytest

from nfl_game.ratings.qb import qb_features_for_targets


def _cutoff_fixture():
    """A team whose QB1 changes between Tuesday and Sunday kickoff."""
    schedules = pd.DataFrame(
        {
            "season": [2025],
            "week": [5],
            "home_team": ["BAL"],
            "away_team": ["CIN"],
            "kickoff_at": [pd.Timestamp("2025-10-05 17:00", tz="UTC")],
        }
    )
    # 2025-era feed: daily `dt` snapshots, `pos_rank`, `club_code`, `pos_abb`.
    depth = pd.DataFrame(
        {
            "club_code": ["BAL", "BAL", "BAL", "BAL", "CIN", "CIN"],
            "gsis_id": ["LAMAR", "HUNT", "LAMAR", "HUNT", "BURROW", "BURROW"],
            "pos_abb": ["QB", "QB", "QB", "QB", "QB", "QB"],
            "pos_rank": [1.0, 2.0, 2.0, 1.0, 1.0, 1.0],
            "dt": pd.to_datetime(
                [
                    "2025-09-30 12:00",  # Tuesday: Lamar QB1
                    "2025-09-30 12:00",
                    "2025-10-04 12:00",  # Saturday: Huntley QB1
                    "2025-10-04 12:00",
                    "2025-09-30 12:00",
                    "2025-10-04 12:00",
                ],
                utc=True,
            ),
        }
    )
    stats = pd.DataFrame(
        {
            "season": [2025] * 3,
            "week": [4, 4, 4],
            "team": ["BAL", "BAL", "CIN"],
            "player_id": ["LAMAR", "HUNT", "BURROW"],
            "position": ["QB", "QB", "QB"],
            "season_type": ["REG"] * 3,
            "attempts": [30.0, 0.0, 30.0],
            "sacks_suffered": [2.0, 0.0, 2.0],
            "passing_epa": [12.0, 0.0, 8.0],
            "passing_cpoe": [3.0, 0.0, 1.0],
            "passing_interceptions": [0.0, 0.0, 1.0],
        }
    )
    return stats, depth, schedules


def test_cutoff_at_kickoff_sees_the_late_change():
    stats, depth, schedules = _cutoff_fixture()
    out = qb_features_for_targets(
        qb_week_stats(stats), depth, schedules, [(2025, 5)], cutoff=None
    ).set_index("team")
    assert out.loc["BAL", "expected_starter_id"] == "HUNT"


def test_cutoff_at_publication_lead_sees_the_earlier_chart():
    stats, depth, schedules = _cutoff_fixture()
    out = qb_features_for_targets(
        qb_week_stats(stats),
        depth,
        schedules,
        [(2025, 5)],
        cutoff=pd.Timedelta(days=5),
    ).set_index("team")
    assert out.loc["BAL", "expected_starter_id"] == "LAMAR"


def test_cutoff_accepts_an_absolute_instant():
    stats, depth, schedules = _cutoff_fixture()
    out = qb_features_for_targets(
        qb_week_stats(stats),
        depth,
        schedules,
        [(2025, 5)],
        cutoff=pd.Timestamp("2025-10-04 18:00", tz="UTC"),
    ).set_index("team")
    assert out.loc["BAL", "expected_starter_id"] == "HUNT"


def test_naive_absolute_cutoff_raises():
    stats, depth, schedules = _cutoff_fixture()
    with pytest.raises(ValueError, match="timezone-aware"):
        qb_features_for_targets(
            qb_week_stats(stats),
            depth,
            schedules,
            [(2025, 5)],
            cutoff=pd.Timestamp("2025-10-04 18:00"),
        )


def _pre2025_era_fixture():
    """The OTHER feed era: week-labelled charts, no `dt`, `team`/`position`/`depth_team`.

    The two eras share almost no columns, and `chart_as_of` takes a different branch for
    each -- the timestamped branch filters on `dt <= cutoff`, the labelled branch matches
    season/week and cannot see a cutoff at all. A test that exercises only the 2025-era
    feed leaves the entire labelled branch unpinned.
    """
    _, _, schedules = _cutoff_fixture()
    depth = pd.DataFrame(
        {
            "season": [2019, 2019, 2019, 2019],
            "week": [5, 5, 5, 5],
            "team": ["BAL", "BAL", "CIN", "CIN"],
            "player_id": ["LAMAR", "HUNT", "BURROW", "OTHER"],
            "position": ["QB", "QB", "QB", "QB"],
            "depth_team": ["1", "2", "1", "2"],
        }
    )
    schedules = schedules.assign(
        season=2019, kickoff_at=[pd.Timestamp("2019-10-06 17:00", tz="UTC")]
    )
    return depth, schedules


def test_labelled_era_resolves_a_starter_at_both_cutoffs():
    stats, _, _ = _cutoff_fixture()
    depth, schedules = _pre2025_era_fixture()
    weeks = qb_week_stats(stats.assign(season=2019, week=4))
    for cutoff in (None, pd.Timedelta(days=5)):
        out = qb_features_for_targets(
            weeks, depth, schedules, [(2019, 5)], cutoff=cutoff
        ).set_index("team")
        # The labelled era carries no timestamp, so the chart is the same at both
        # cutoffs -- but it must still RESOLVE rather than falling through to null.
        assert out.loc["BAL", "expected_starter_id"] == "LAMAR", cutoff
        assert out.loc["BAL", "qb_uncertain"] == 0, cutoff
```

Add `qb_week_stats` to the existing import from `nfl_game.ratings.qb` at the top of the file.

- [ ] **Step 2: Run the tests to verify they fail**

```bash
./.venv/Scripts/python.exe -m pytest tests/test_qb.py -k cutoff -v
```

Expected: FAIL — `TypeError: qb_features_for_targets() got an unexpected keyword argument 'cutoff'`.

- [ ] **Step 3: Implement the cutoff policy**

In `src/nfl_game/ratings/qb.py`, add the type alias next to the existing module constants:

```python
from typing import TypeAlias

CutoffPolicy: TypeAlias = "pd.Timestamp | pd.Timedelta | None"
```

Add the resolver above `_targets_from_schedule`:

```python
def _cutoff_for(kickoff: pd.Series, cutoff: CutoffPolicy) -> pd.Series:
    """Resolve the depth-chart cutoff for each game.

    `None` is kickoff itself -- the original behavior, kept verbatim so the Ridge-v2
    research output stays reproducible. A `Timedelta` is kickoff minus that lead, per
    game, which is what a published pick actually has. A `Timestamp` is one absolute
    instant for every game, which is what a live advisory has.

    The lead is subtracted from EACH GAME's own kickoff, never from the week's first
    kickoff. Anchoring to the week is how a cache named for a five-day lead came to have
    a 7.51-day mean.
    """
    if cutoff is None:
        return kickoff
    if isinstance(cutoff, pd.Timedelta):
        return kickoff - cutoff
    stamp = pd.Timestamp(cutoff)
    if stamp.tzinfo is None:
        raise ValueError("an absolute depth-chart cutoff must be timezone-aware")
    return pd.Series(stamp, index=kickoff.index)
```

Change the `_targets_from_schedule` signature and its final cutoff assignment:

```python
def _targets_from_schedule(
    schedules: pd.DataFrame, targets: list[tuple[int, int]], cutoff: CutoffPolicy = None
) -> pd.DataFrame:
```

and replace the line `out["cutoff"] = cutoff` with:

```python
    out["cutoff"] = _cutoff_for(cutoff_series, cutoff)
```

renaming the locally computed kickoff series from `cutoff` to `cutoff_series` in the two branches above it so the parameter is not shadowed.

Then thread the parameter through `qb_features_for_targets`:

```python
def qb_features_for_targets(
    qb_weeks: pd.DataFrame,
    depth_history: pd.DataFrame,
    schedules: pd.DataFrame,
    targets: list[tuple[int, int]],
    cutoff: CutoffPolicy = None,
) -> pd.DataFrame:
```

and change its first body line to `games = _targets_from_schedule(schedules, targets, cutoff)`.

- [ ] **Step 4: Run the new tests, then the whole QB and v2 suites**

```bash
./.venv/Scripts/python.exe -m pytest tests/test_qb.py -v
./.venv/Scripts/python.exe -m pytest tests/test_v2_features.py tests/test_depth.py -v
```

Expected: all PASS. The pre-existing tests passing unchanged is the guard that `cutoff=None` reproduces today's behavior exactly.

- [ ] **Step 5: Lint and commit**

```bash
./.venv/Scripts/python.exe -m ruff check .
git add src/nfl_game/ratings/qb.py tests/test_qb.py
git commit -m "feat: parameterize the depth-chart cutoff instead of pinning it to kickoff"
```

---

### Task 2: Expose the prior starter

The advisory has to say what changed *from*, and `qb_features_for_targets` computes `recent_starter` internally but discards it. `qb_change_epa` alone cannot tell you the model's ratings were built on Lamar.

**Files:**
- Modify: `src/nfl_game/ratings/qb.py:139-193` (`qb_features_for_targets`)
- Test: `tests/test_qb.py`

**Interfaces:**
- Consumes: `qb_features_for_targets(..., cutoff)` from Task 1.
- Produces: `qb_features_for_targets` output gains a `recent_starter_id` column, positioned immediately after `expected_starter_id`. Full column order becomes `["season", "week", "team", "expected_starter_id", "recent_starter_id", *QB_FEATURE_COLS]`.

`QB_FEATURE_COLS` itself is **unchanged**, so `build_v2.py:354`'s `_block_coverage("C2", qb, expected, QB_FEATURE_COLS, ...)` and every `v2_features.py` mapping keep working against the same eight names.

- [ ] **Step 1: Write the failing test**

```python
def test_recent_starter_id_is_reported():
    stats, depth, schedules = _cutoff_fixture()
    out = qb_features_for_targets(
        qb_week_stats(stats), depth, schedules, [(2025, 5)], cutoff=None
    ).set_index("team")
    assert out.loc["BAL", "expected_starter_id"] == "HUNT"
    assert out.loc["BAL", "recent_starter_id"] == "LAMAR"
    assert out.loc["BAL", "qb_new_starter"] == 1
    # An unchanged team names the same quarterback on both sides.
    assert out.loc["CIN", "recent_starter_id"] == "BURROW"
    assert out.loc["CIN", "qb_new_starter"] == 0


def test_recent_starter_id_is_null_when_no_prior_game_exists():
    stats, depth, schedules = _cutoff_fixture()
    empty = qb_week_stats(stats.iloc[0:0])
    out = qb_features_for_targets(
        empty, depth, schedules, [(2025, 5)], cutoff=None
    ).set_index("team")
    assert pd.isna(out.loc["BAL", "recent_starter_id"])
```

- [ ] **Step 2: Run to verify failure**

```bash
./.venv/Scripts/python.exe -m pytest tests/test_qb.py -k recent_starter -v
```

Expected: FAIL with `KeyError: 'recent_starter_id'`.

- [ ] **Step 3: Implement**

In `qb_features_for_targets`, change the `columns` line:

```python
    columns = [
        "season",
        "week",
        "team",
        "expected_starter_id",
        "recent_starter_id",
        *QB_FEATURE_COLS,
    ]
```

and add one key to the appended result dict, immediately after `"expected_starter_id": expected,`:

```python
                "recent_starter_id": recent_starter,
```

`recent_starter` is already `None` when no prior game exists, which becomes a null in the frame.

- [ ] **Step 4: Update the existing column-order assertion and run**

`tests/test_qb.py:171` asserts the exact column list. Change it to:

```python
    assert list(out.columns) == [
        "season",
        "week",
        "team",
        "expected_starter_id",
        "recent_starter_id",
        *QB_FEATURE_COLS,
    ]
```

```bash
./.venv/Scripts/python.exe -m pytest tests/test_qb.py tests/test_v2_features.py -v
```

Expected: all PASS.

- [ ] **Step 5: Lint and commit**

```bash
./.venv/Scripts/python.exe -m ruff check .
git add src/nfl_game/ratings/qb.py tests/test_qb.py
git commit -m "feat: report the prior starter alongside the expected one"
```

---

### Task 3: The game-level advisory frame

Turn per-team starter rows into one row per game, with display names and the two markers the slate needs.

**Files:**
- Create: `src/nfl_game/ratings/starters.py`
- Test: `tests/test_starters.py`

**Interfaces:**
- Consumes: `qb_features_for_targets(qb_weeks, depth_history, schedules, targets, cutoff)` and its `recent_starter_id` column from Tasks 1-2; `qb_week_stats` from `nfl_game.ratings.qb`.
- Produces:
  - `ADVISORY_COLS: list[str]` — `["game_id", "home_qb", "away_qb", "qb_change_epa_home", "qb_change_epa_away", "qb_watch", "qb_inferred"]`
  - `player_display_names(players: pd.DataFrame) -> dict[str, str]`
  - `starter_advisory(qb_weeks, depth_history, schedules, players, targets, cutoff) -> pd.DataFrame` with exactly `ADVISORY_COLS`.

**Column semantics — each column means one thing:**

| column | dtype | meaning |
| --- | --- | --- |
| `home_qb`, `away_qb` | `string` | expected starter display name, or `<NA>` if unresolved |
| `qb_change_epa_home/away` | `float64` | expected starter EPA/dropback minus prior starter's; `0.0` when unchanged |
| `qb_watch` | `Int64` | 1 if either side has a new starter, 0 if neither; `<NA>` if the advisory could not be built |
| `qb_inferred` | `Int64` | 1 if either side's starter was inferred from last week rather than read from a chart |

`qb_watch` is **nullable and never defaulted to 0**. A null means "we do not know"; a zero asserts "no change", which is a different and false claim.

- [ ] **Step 1: Write the failing test**

Create `tests/test_starters.py`:

```python
import pandas as pd

from nfl_game.ratings.qb import qb_week_stats
from nfl_game.ratings.starters import (
    ADVISORY_COLS,
    player_display_names,
    starter_advisory,
)


def _players():
    return pd.DataFrame(
        {
            "gsis_id": ["LAMAR", "HUNT", "BURROW"],
            "display_name": ["Lamar Jackson", "Tyler Huntley", "Joe Burrow"],
        }
    )


def _fixture():
    schedules = pd.DataFrame(
        {
            "game_id": ["2025_05_CIN_BAL"],
            "season": [2025],
            "week": [5],
            "home_team": ["BAL"],
            "away_team": ["CIN"],
            "kickoff_at": [pd.Timestamp("2025-10-05 17:00", tz="UTC")],
        }
    )
    depth = pd.DataFrame(
        {
            "club_code": ["BAL", "BAL", "CIN"],
            "gsis_id": ["HUNT", "LAMAR", "BURROW"],
            "pos_abb": ["QB", "QB", "QB"],
            "pos_rank": [1.0, 2.0, 1.0],
            "dt": pd.to_datetime(["2025-10-04 12:00"] * 3, utc=True),
        }
    )
    stats = pd.DataFrame(
        {
            "season": [2025] * 3,
            "week": [4, 4, 4],
            "team": ["BAL", "BAL", "CIN"],
            "player_id": ["LAMAR", "HUNT", "BURROW"],
            "position": ["QB"] * 3,
            "season_type": ["REG"] * 3,
            "attempts": [30.0, 0.0, 30.0],
            "sacks_suffered": [2.0, 0.0, 2.0],
            "passing_epa": [12.0, 0.0, 8.0],
            "passing_cpoe": [3.0, 0.0, 1.0],
            "passing_interceptions": [0.0, 0.0, 1.0],
        }
    )
    return qb_week_stats(stats), depth, schedules


def test_display_names_map_ids_to_names():
    assert player_display_names(_players())["HUNT"] == "Tyler Huntley"


def test_advisory_reports_one_row_per_game_with_the_expected_columns():
    weeks, depth, schedules = _fixture()
    out = starter_advisory(weeks, depth, schedules, _players(), [(2025, 5)], cutoff=None)
    assert list(out.columns) == ADVISORY_COLS
    assert len(out) == 1


def test_advisory_names_the_changed_starter_and_flags_the_game():
    weeks, depth, schedules = _fixture()
    out = starter_advisory(
        weeks, depth, schedules, _players(), [(2025, 5)], cutoff=None
    ).iloc[0]
    assert out["home_qb"] == "Tyler Huntley"
    assert out["away_qb"] == "Joe Burrow"
    assert out["qb_watch"] == 1
    # Huntley has no dropbacks, so he shrinks fully to the league prior, which is
    # below Lamar's own rate. The delta must therefore be negative, not zero.
    assert out["qb_change_epa_home"] < 0.0
    assert out["qb_change_epa_away"] == 0.0


def test_qb_watch_is_zero_when_neither_side_changed():
    weeks, depth, schedules = _fixture()
    unchanged = depth.copy()
    unchanged.loc[unchanged["gsis_id"].eq("HUNT"), "pos_rank"] = 2.0
    unchanged.loc[unchanged["gsis_id"].eq("LAMAR"), "pos_rank"] = 1.0
    out = starter_advisory(
        weeks, unchanged, schedules, _players(), [(2025, 5)], cutoff=None
    ).iloc[0]
    assert out["qb_watch"] == 0
    assert out["qb_inferred"] == 0


def test_missing_chart_marks_the_row_inferred_rather_than_guessing_silently():
    weeks, depth, schedules = _fixture()
    out = starter_advisory(
        weeks, depth.iloc[0:0], schedules, _players(), [(2025, 5)], cutoff=None
    ).iloc[0]
    assert out["qb_inferred"] == 1
    # The fallback still names last week's starter -- it is labelled, not suppressed.
    assert out["home_qb"] == "Lamar Jackson"


def test_unknown_player_id_yields_a_null_name_not_the_raw_id():
    weeks, depth, schedules = _fixture()
    out = starter_advisory(
        weeks, depth, schedules, _players().iloc[0:0], [(2025, 5)], cutoff=None
    ).iloc[0]
    assert pd.isna(out["home_qb"])


def test_empty_targets_yield_an_empty_frame_with_the_full_schema():
    weeks, depth, schedules = _fixture()
    out = starter_advisory(weeks, depth, schedules, _players(), [], cutoff=None)
    assert out.empty
    assert list(out.columns) == ADVISORY_COLS
```

- [ ] **Step 2: Run to verify failure**

```bash
./.venv/Scripts/python.exe -m pytest tests/test_starters.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'nfl_game.ratings.starters'`.

- [ ] **Step 3: Implement**

Create `src/nfl_game/ratings/starters.py`:

```python
"""Game-level expected-starter advisory.

This is presentation, not a model input. Nothing here reaches `FEATURE_COLS`, and the
advisory is joined onto a slate AFTER prediction, so its presence, absence or staleness
can never move `model_margin` or `model_total`.
"""

from __future__ import annotations

import pandas as pd

from nfl_game.ratings.qb import CutoffPolicy, qb_features_for_targets

ADVISORY_COLS = [
    "game_id",
    "home_qb",
    "away_qb",
    "qb_change_epa_home",
    "qb_change_epa_away",
    "qb_watch",
    "qb_inferred",
]

_NAME_SOURCES = ("gsis_id", "player_id")
_DISPLAY_SOURCES = ("display_name", "football_name", "full_name")


def _empty() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": pd.Series(dtype="string"),
            "home_qb": pd.Series(dtype="string"),
            "away_qb": pd.Series(dtype="string"),
            "qb_change_epa_home": pd.Series(dtype="float64"),
            "qb_change_epa_away": pd.Series(dtype="float64"),
            "qb_watch": pd.Series(dtype="Int64"),
            "qb_inferred": pd.Series(dtype="Int64"),
        }
    )


def player_display_names(players: pd.DataFrame) -> dict[str, str]:
    """Map player id to display name, tolerating the crosswalk's column spellings.

    An id with no matching row yields no entry rather than falling back to the raw id:
    a slate cell reading "00-0034796" looks like a rendering bug, while an empty cell
    correctly reads as "not resolved".
    """
    if players.empty:
        return {}
    id_col = next((name for name in _NAME_SOURCES if name in players), None)
    name_col = next((name for name in _DISPLAY_SOURCES if name in players), None)
    if id_col is None or name_col is None:
        return {}
    rows = players[[id_col, name_col]].dropna()
    return {str(pid): str(name) for pid, name in rows.itertuples(index=False)}


def starter_advisory(
    qb_weeks: pd.DataFrame,
    depth_history: pd.DataFrame,
    schedules: pd.DataFrame,
    players: pd.DataFrame,
    targets: list[tuple[int, int]],
    cutoff: CutoffPolicy = None,
) -> pd.DataFrame:
    """One advisory row per scheduled game in `targets`."""
    if not targets:
        return _empty()
    per_team = qb_features_for_targets(qb_weeks, depth_history, schedules, targets, cutoff)
    if per_team.empty:
        return _empty()

    requested = pd.DataFrame(sorted(set(targets)), columns=["season", "week"])
    games = schedules.merge(requested, on=["season", "week"], how="inner")
    games = games[["game_id", "season", "week", "home_team", "away_team"]].drop_duplicates(
        "game_id"
    )
    if games.empty:
        return _empty()

    names = player_display_names(players)
    out = games[["game_id"]].copy()
    out["game_id"] = out["game_id"].astype("string")
    for side in ("home", "away"):
        merged = games.merge(
            per_team,
            left_on=["season", "week", f"{side}_team"],
            right_on=["season", "week", "team"],
            how="left",
        )
        out[f"{side}_qb"] = (
            merged["expected_starter_id"].map(names).astype("string").to_numpy()
        )
        out[f"qb_change_epa_{side}"] = pd.to_numeric(
            merged["qb_change_epa"], errors="coerce"
        ).to_numpy()
        out[f"_new_{side}"] = pd.to_numeric(
            merged["qb_new_starter"], errors="coerce"
        ).to_numpy()
        out[f"_unc_{side}"] = pd.to_numeric(
            merged["qb_uncertain"], errors="coerce"
        ).to_numpy()

    # A side whose starter never resolved leaves the game unknown rather than "no change".
    known = out[["_new_home", "_new_away"]].notna().all(axis=1)
    watch = out[["_new_home", "_new_away"]].max(axis=1)
    out["qb_watch"] = watch.where(known).astype("Int64")
    out["qb_inferred"] = (
        out[["_unc_home", "_unc_away"]].max(axis=1).where(known).astype("Int64")
    )
    return out[ADVISORY_COLS].reset_index(drop=True)
```

- [ ] **Step 4: Run to verify pass**

```bash
./.venv/Scripts/python.exe -m pytest tests/test_starters.py -v
```

Expected: all PASS.

- [ ] **Step 5: Lint and commit**

```bash
./.venv/Scripts/python.exe -m ruff check .
git add src/nfl_game/ratings/starters.py tests/test_starters.py
git commit -m "feat: build a game-level expected-starter advisory frame"
```

---

### Task 4: The live starter provider

A bounded-cache live depth-chart read that fails soft, mirroring `market/live.py`'s snapshot, TTL, staleness and fallback contract.

**Files:**
- Create: `src/nfl_game/market/live_starters.py`
- Test: `tests/test_live_starters.py`

**Placement note for the implementer:** a depth chart is not market data, and the name `market/` is a compromise. It goes here because `market/` is the package defined as owning "the bounded five-minute live cache/stale-snapshot behavior" and overlaying independently nullable data after prediction, which is exactly this component. Do not add a new top-level package for one module.

**Interfaces:**
- Consumes: `starter_advisory` and `ADVISORY_COLS` from Task 3.
- Produces:
  - `StarterSnapshot` — frozen dataclass with `rows: pd.DataFrame`, `observed_at: datetime`, `source: str = "nflverse"`, `stale: bool = False`
  - `StartersUnavailableError(RuntimeError)`
  - `NflverseStarterProvider(depth_loader=..., stats_loader=..., players_loader=..., schedule_loader=..., clock=..., ttl=timedelta(minutes=30), timeout_seconds=20.0)` with `.snapshot(season: int, week: int) -> StarterSnapshot`

**Two scope decisions the implementer must not silently change:**

1. **TTL is 30 minutes, not the market's 5.** Depth charts publish daily at most; a 5-minute TTL would re-download a large feed for no new information.
2. **Player stats load `[season - 1, season]` only.** The full 2016-onward corpus is a heavy download to hold behind a live request. `qb_epa_per_db` shrinks toward the league prior at 200 dropbacks, so a quarterback with little recent history lands near league average rather than at a wild value. This makes the advisory's `qb_change_epa` **not** numerically identical to the Ridge-v2 C2 block's, which trains on the full history. That is acceptable for an advisory and must stay documented; Track B must not quote advisory numbers as research numbers.

- [ ] **Step 1: Write the failing test**

Create `tests/test_live_starters.py`:

```python
from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from nfl_game.market.live_starters import (
    NflverseStarterProvider,
    StarterSnapshot,
    StartersUnavailableError,
)


def _loaders():
    schedules = pd.DataFrame(
        {
            "game_id": ["2025_05_CIN_BAL"],
            "season": [2025],
            "week": [5],
            "home_team": ["BAL"],
            "away_team": ["CIN"],
            "kickoff_at": [pd.Timestamp("2025-10-05 17:00", tz="UTC")],
        }
    )
    depth = pd.DataFrame(
        {
            "club_code": ["BAL", "CIN"],
            "gsis_id": ["HUNT", "BURROW"],
            "pos_abb": ["QB", "QB"],
            "pos_rank": [1.0, 1.0],
            "dt": pd.to_datetime(["2025-10-04 12:00"] * 2, utc=True),
        }
    )
    stats = pd.DataFrame(
        {
            "season": [2025, 2025],
            "week": [4, 4],
            "team": ["BAL", "CIN"],
            "player_id": ["LAMAR", "BURROW"],
            "position": ["QB", "QB"],
            "season_type": ["REG", "REG"],
            "attempts": [30.0, 30.0],
            "sacks_suffered": [2.0, 2.0],
            "passing_epa": [12.0, 8.0],
            "passing_cpoe": [3.0, 1.0],
            "passing_interceptions": [0.0, 1.0],
        }
    )
    players = pd.DataFrame(
        {
            "gsis_id": ["LAMAR", "HUNT", "BURROW"],
            "display_name": ["Lamar Jackson", "Tyler Huntley", "Joe Burrow"],
        }
    )
    return schedules, depth, stats, players


def _provider(clock=None, **overrides):
    schedules, depth, stats, players = _loaders()
    kwargs = dict(
        depth_loader=lambda seasons, save=False: depth,
        stats_loader=lambda seasons, save=False: stats,
        players_loader=lambda save=False: players,
        schedule_loader=lambda seasons=None, save=False: schedules,
        clock=clock or (lambda: datetime(2025, 10, 4, 13, tzinfo=UTC)),
    )
    kwargs.update(overrides)
    return NflverseStarterProvider(**kwargs)


def test_snapshot_returns_the_advisory_rows():
    snap = _provider().snapshot(2025, 5)
    assert isinstance(snap, StarterSnapshot)
    assert snap.stale is False
    assert snap.rows.iloc[0]["home_qb"] == "Tyler Huntley"
    assert snap.rows.iloc[0]["qb_watch"] == 1


def test_second_call_within_the_ttl_does_not_reload():
    calls = []

    def counting_depth(seasons, save=False):
        calls.append(seasons)
        return _loaders()[1]

    provider = _provider(depth_loader=counting_depth)
    provider.snapshot(2025, 5)
    provider.snapshot(2025, 5)
    assert len(calls) == 1


def test_a_failing_load_with_no_cache_raises():
    def boom(seasons, save=False):
        raise RuntimeError("feed down")

    with pytest.raises(StartersUnavailableError):
        _provider(depth_loader=boom).snapshot(2025, 5)


def test_a_failing_load_after_a_good_one_returns_the_cache_marked_stale():
    state = {"fail": False}

    def flaky(seasons, save=False):
        if state["fail"]:
            raise RuntimeError("feed down")
        return _loaders()[1]

    times = iter(
        [
            datetime(2025, 10, 4, 13, tzinfo=UTC),
            datetime(2025, 10, 4, 13, tzinfo=UTC),
            datetime(2025, 10, 4, 23, tzinfo=UTC),
        ]
    )
    last = {"t": datetime(2025, 10, 4, 13, tzinfo=UTC)}

    def clock():
        last["t"] = next(times, last["t"])
        return last["t"]

    provider = _provider(clock=clock, depth_loader=flaky)
    provider.snapshot(2025, 5)
    state["fail"] = True
    snap = provider.snapshot(2025, 5)
    assert snap.stale is True
    assert snap.rows.iloc[0]["home_qb"] == "Tyler Huntley"


def test_stats_loader_is_asked_for_the_prior_and_current_season_only():
    seen = []

    def recording_stats(seasons, save=False):
        seen.append(list(seasons))
        return _loaders()[2]

    _provider(stats_loader=recording_stats).snapshot(2025, 5)
    assert seen == [[2024, 2025]]
```

- [ ] **Step 2: Run to verify failure**

```bash
./.venv/Scripts/python.exe -m pytest tests/test_live_starters.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'nfl_game.market.live_starters'`.

- [ ] **Step 3: Implement**

Create `src/nfl_game/market/live_starters.py`:

```python
"""Bounded-cache live expected-starter snapshots.

Mirrors `market/live.py`'s snapshot/TTL/stale/fallback contract deliberately: the two
overlays have the same failure modes and should behave the same way under them. The TTL
is longer because depth charts publish daily at most.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta

import pandas as pd

from nfl_game.data.nfl import (
    load_depth_charts,
    load_player_stats,
    load_players,
    load_schedules,
)
from nfl_game.ratings.qb import qb_week_stats
from nfl_game.ratings.starters import starter_advisory


class StartersUnavailableError(RuntimeError):
    """No advisory could be produced and no cached one exists."""


@dataclass(frozen=True)
class StarterSnapshot:
    rows: pd.DataFrame
    observed_at: datetime
    source: str = "nflverse"
    stale: bool = False


class NflverseStarterProvider:
    def __init__(
        self,
        depth_loader=load_depth_charts,
        stats_loader=load_player_stats,
        players_loader=load_players,
        schedule_loader=load_schedules,
        clock=lambda: datetime.now(UTC),
        ttl=timedelta(minutes=30),
        timeout_seconds=20.0,
    ):
        self._depth_loader = depth_loader
        self._stats_loader = stats_loader
        self._players_loader = players_loader
        self._schedule_loader = schedule_loader
        self._clock = clock
        self._ttl = ttl
        self._timeout_seconds = timeout_seconds
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._snapshots = {}
        self._futures = {}
        self._latest_futures = {}

    def snapshot(self, season: int, week: int) -> StarterSnapshot:
        key = (int(season), int(week))
        now = self._clock()
        with self._lock:
            cached = self._snapshots.get(key)
            if cached is not None and now - cached.observed_at < self._ttl:
                return cached
            future = self._futures.get(key)
            if future is None:
                future = self._executor.submit(self._load_snapshot, key)
                self._futures[key] = future
                self._latest_futures[key] = future
        try:
            refreshed = future.result(timeout=self._timeout_seconds)
        except Exception as exc:  # noqa: BLE001 - the injected upstream may fail arbitrarily
            # FutureTimeoutError is itself an Exception, so this one clause covers both a
            # slow feed and a broken one.
            return self._stale_or_raise(key, future, exc)
        return self._store_refresh(key, future, refreshed)

    def _load_snapshot(self, key) -> StarterSnapshot:
        season, week = key
        # The prior season carries the "recent starter" for an early-season week; the
        # full corpus is deliberately not loaded behind a live request. See the plan.
        seasons = [season - 1, season]
        depth = self._depth_loader(seasons, save=False)
        stats = self._stats_loader(seasons, save=False)
        players = self._players_loader(save=False)
        schedules = self._schedule_loader(seasons, save=False)
        rows = starter_advisory(
            qb_week_stats(stats),
            depth,
            schedules,
            players,
            [(season, week)],
            cutoff=self._clock(),
        )
        return StarterSnapshot(rows=rows.copy(deep=True), observed_at=self._clock())

    def _store_refresh(self, key, future, refreshed):
        with self._lock:
            if self._latest_futures.get(key) is not future:
                return refreshed
            if self._futures.get(key) is future:
                self._futures.pop(key)
            self._snapshots[key] = refreshed
        return refreshed

    def _stale_or_raise(self, key, future, exc):
        # Deliberate divergence from market/live.py: that provider has a
        # `_consume_completed` path that adopts a future which finished just after the
        # timeout. Here the TTL is 30 minutes against a 20-second timeout, so discarding
        # a late result costs at most one refresh cycle and is not worth the extra state.
        with self._lock:
            if self._futures.get(key) is future:
                self._futures.pop(key)
            cached = self._snapshots.get(key)
        if cached is not None:
            return replace(cached, rows=cached.rows.copy(deep=True), stale=True)
        raise StartersUnavailableError("expected-starter feed unavailable") from exc
```

Note the `cutoff=self._clock()` — the live advisory reads the chart as of *now*, which is the whole point of Task 1.

- [ ] **Step 4: Run to verify pass**

```bash
./.venv/Scripts/python.exe -m pytest tests/test_live_starters.py -v
```

Expected: all PASS.

- [ ] **Step 5: Lint and commit**

```bash
./.venv/Scripts/python.exe -m ruff check .
git add src/nfl_game/market/live_starters.py tests/test_live_starters.py
git commit -m "feat: add a bounded-cache live expected-starter provider"
```

---

### Task 5: Overlay the advisory onto the slate

**Files:**
- Modify: `src/nfl_game/market/compare.py:13-27` (`SLATE_COLS`), `:31-58` (`build_slate`), `:73-94` (`slate_markdown`)
- Test: `tests/test_compare.py`

**Interfaces:**
- Consumes: `ADVISORY_COLS` from Task 3.
- Produces: `build_slate(features_df, preds, probs, edge_threshold=2.0, starters=None) -> pd.DataFrame`. `SLATE_COLS` gains the six advisory columns after `edge_flag`, in `ADVISORY_COLS` order minus `game_id`.

The **inert-overlay test is the most important test in this plan.** It is the mechanical guard that Track A cannot move a prediction.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_compare.py`:

```python
def _advisory(game_id, watch=1):
    return pd.DataFrame(
        {
            "game_id": pd.Series([game_id], dtype="string"),
            "home_qb": pd.Series(["Tyler Huntley"], dtype="string"),
            "away_qb": pd.Series(["Joe Burrow"], dtype="string"),
            "qb_change_epa_home": [-0.31],
            "qb_change_epa_away": [0.0],
            "qb_watch": pd.Series([watch], dtype="Int64"),
            "qb_inferred": pd.Series([0], dtype="Int64"),
        }
    )


def test_slate_without_starters_has_null_advisory_columns():
    out = build_slate(*_inputs())
    assert list(out.columns) == SLATE_COLS
    assert out["qb_watch"].isna().all()
    assert out["home_qb"].isna().all()


def test_the_overlay_never_moves_a_prediction_or_a_flag():
    feats, preds, probs = _inputs()
    without = build_slate(feats, preds, probs)
    game_id = without["game_id"].iloc[0]
    with_advisory = build_slate(feats, preds, probs, starters=_advisory(game_id))
    for column in ("model_spread", "model_total", "spread_gap", "total_gap", "edge_flag"):
        pd.testing.assert_series_equal(
            without[column], with_advisory[column], check_exact=True
        )


def test_the_overlay_attaches_to_the_right_game():
    feats, preds, probs = _inputs()
    game_id = build_slate(feats, preds, probs)["game_id"].iloc[0]
    out = build_slate(feats, preds, probs, starters=_advisory(game_id)).set_index("game_id")
    assert out.loc[game_id, "home_qb"] == "Tyler Huntley"
    assert out.loc[game_id, "qb_watch"] == 1


def test_an_advisory_for_an_unknown_game_leaves_every_row_null():
    feats, preds, probs = _inputs()
    out = build_slate(feats, preds, probs, starters=_advisory("2099_01_XXX_YYY"))
    assert out["qb_watch"].isna().all()


def test_markdown_renders_the_qb_column_and_marks_a_watch():
    feats, preds, probs = _inputs()
    game_id = build_slate(feats, preds, probs)["game_id"].iloc[0]
    md = slate_markdown(build_slate(feats, preds, probs, starters=_advisory(game_id)))
    assert "| QB |" in md
    assert "Tyler Huntley" in md
    assert "-0.31" in md


def test_markdown_renders_a_missing_advisory_as_not_available():
    md = slate_markdown(build_slate(*_inputs()))
    assert "| QB |" in md
    assert "nan" not in md.lower()
```

- [ ] **Step 2: Run to verify failure**

```bash
./.venv/Scripts/python.exe -m pytest tests/test_compare.py -k "overlay or advisory or qb" -v
```

Expected: FAIL — `TypeError: build_slate() got an unexpected keyword argument 'starters'`.

- [ ] **Step 3: Implement**

In `src/nfl_game/market/compare.py`, extend the module docstring with a paragraph:

```
The QB advisory columns are presentation only. They are joined AFTER prediction and
after edge_flag is computed, so their presence, absence or staleness can never move
model_spread, model_total or edge_flag -- a property tests/test_compare.py pins
directly. qb_watch is a separate marker from edge_flag on purpose: suppressing an edge
on a starter change would make one flag mean two things and silently hide edges.
```

Append to `SLATE_COLS`:

```python
    "home_qb",
    "away_qb",
    "qb_change_epa_home",
    "qb_change_epa_away",
    "qb_watch",
    "qb_inferred",
]

_ADVISORY_DTYPES = {
    "home_qb": "string",
    "away_qb": "string",
    "qb_change_epa_home": "float64",
    "qb_change_epa_away": "float64",
    "qb_watch": "Int64",
    "qb_inferred": "Int64",
}
```

Change the signature and add the overlay immediately before `out = df[SLATE_COLS].copy()`:

```python
def build_slate(
    features_df: pd.DataFrame,
    preds: pd.DataFrame,
    probs: pd.DataFrame,
    edge_threshold: float = 2.0,
    starters: pd.DataFrame | None = None,
) -> pd.DataFrame:
```

```python
    # Joined after edge_flag on purpose: every column above this line is computed from
    # the model and the market alone, so no advisory failure mode can reach them.
    if starters is not None and not starters.empty:
        advisory = starters.drop_duplicates("game_id").copy()
        advisory["game_id"] = advisory["game_id"].astype("string")
        df["game_id"] = df["game_id"].astype("string")
        df = df.merge(advisory, on="game_id", how="left", validate="one_to_one")
    for name, dtype in _ADVISORY_DTYPES.items():
        if name not in df:
            df[name] = pd.Series(pd.NA, index=df.index, dtype=dtype)
        else:
            df[name] = df[name].astype(dtype)
```

In `slate_markdown`, add a `QB` column. Change the header to:

```python
    header = (
        "| Game | Model | Market | Gap | Cover% | Model O/U | Market O/U | Gap | Over% "
        "| Edge | QB |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|\n"
    )
```

Add a renderer above `slate_markdown`:

```python
def _qb_cell(row) -> str:
    """Render the advisory for one game, or "n/a" when it could not be built.

    A blank cell would be indistinguishable from "no change"; the advisory being
    unavailable is a different fact and reads as one.
    """
    if pd.isna(row.qb_watch):
        return "n/a"
    if row.qb_watch != 1:
        return ""
    parts = []
    for name, delta in (
        (row.home_qb, row.qb_change_epa_home),
        (row.away_qb, row.qb_change_epa_away),
    ):
        if pd.isna(delta) or delta == 0.0:
            continue
        label = "unknown" if pd.isna(name) else name
        parts.append(f"{label} {delta:+.2f}")
    suffix = " (inferred)" if row.qb_inferred == 1 else ""
    return ("; ".join(parts) or "change") + suffix
```

and append `f"| {_qb_cell(r)} |"` to the row f-string.

- [ ] **Step 4: Run the full compare suite**

```bash
./.venv/Scripts/python.exe -m pytest tests/test_compare.py -v
```

Expected: all PASS, including every pre-existing test. The pre-existing `test_slate_columns` assertion compares against `SLATE_COLS`, so it follows the constant automatically.

- [ ] **Step 5: Lint and commit**

```bash
./.venv/Scripts/python.exe -m ruff check .
git add src/nfl_game/market/compare.py tests/test_compare.py
git commit -m "feat: overlay the QB starter advisory onto the slate after prediction"
```

---

### Task 6: Wire the advisory into `scripts/slate.py`

**Files:**
- Modify: `scripts/slate.py:8` (import), `:56` (the `build_slate` call), and the argument parser
- Create: `tests/test_slate_script.py` — no test currently covers `scripts/slate.py`

**Interfaces:**
- Consumes: `NflverseStarterProvider` from Task 4, `build_slate(..., starters=)` from Task 5.
- Produces: a `--no-starters` flag on `scripts/slate.py`. Default is starters **on**.

- [ ] **Step 1: Write the failing test**

```python
import subprocess
import sys


def test_slate_help_documents_the_no_starters_flag():
    out = subprocess.run(
        [sys.executable, "scripts/slate.py", "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--no-starters" in out.stdout
```

- [ ] **Step 2: Run to verify failure**

```bash
./.venv/Scripts/python.exe -m pytest tests/test_slate_script.py -v
```

Expected: FAIL — `--no-starters` not in help output.

- [ ] **Step 3: Implement**

Add the import to `scripts/slate.py`:

```python
from nfl_game.market.live_starters import NflverseStarterProvider, StartersUnavailableError
```

Add the argument next to the existing ones:

```python
    parser.add_argument(
        "--no-starters",
        action="store_true",
        help="skip the expected-starter advisory (it needs a live depth-chart fetch)",
    )
```

Replace the `build_slate` call at `:56`:

```python
    starters = None
    if not args.no_starters:
        try:
            starters = NflverseStarterProvider().snapshot(args.season, args.week).rows
        except StartersUnavailableError as exc:
            # Advisory only. A slate that prints without it is still correct; a slate
            # that refuses to print because a depth chart was unreachable is not.
            print(f"warning: expected-starter advisory unavailable ({exc})")
    slate = build_slate(
        target, preds, probs, edge_threshold=args.edge_threshold, starters=starters
    )
```

- [ ] **Step 4: Run to verify pass**

```bash
./.venv/Scripts/python.exe -m pytest tests/test_slate_script.py -v
./.venv/Scripts/python.exe scripts/slate.py --season 2026 --week 1 --no-starters
```

Expected: test PASSES; the offline slate still renders with a `QB` column reading `n/a`.

- [ ] **Step 5: Lint and commit**

```bash
./.venv/Scripts/python.exe -m ruff check .
git add scripts/slate.py tests/test_slate_script.py
git commit -m "feat: show the expected-starter advisory on the CLI slate"
```

---

### Task 7: Wire the advisory into the web dashboard

**Files:**
- Modify: `src/nfl_game/web/service.py:128-161` (constructor), `:328-353` (`_slate_result`), `:344-352` (status columns), and `payload`
- Modify: `src/nfl_game/web/app.py:134-165` (`renderGames`)
- Test: `tests/test_web_service.py` — it already provides the `feature_rows()` and
  `feature_rows_with_2026_weeks()` helpers and constructs `SlateService(rows, ...)`
  positionally. There is no `_service(tmp_path)` helper; do not invent one.

**Interfaces:**
- Consumes: `NflverseStarterProvider`, `StarterSnapshot`, `StartersUnavailableError` from Task 4.
- Produces: `SlateService(..., starter_provider=None)`; `payload()` gains a `"starters"` metadata key shaped like the existing `"market"` key — `{"source", "observed_at", "stale"}` — or `None` when no provider is configured.

`web/` stays read-only. A `StartersUnavailableError` must degrade to null advisory columns, never to a failed request — the market overlay raises `SlateUnavailableError` because a slate without lines is not a slate, but a slate without a QB advisory is perfectly usable.

- [ ] **Step 1: Write the failing test**

```python
from datetime import UTC, datetime

from nfl_game.market.live_starters import StarterSnapshot, StartersUnavailableError


def _advisory_rows(game_id="2026_01_AAA_BBB"):
    return pd.DataFrame(
        {
            "game_id": pd.Series([game_id], dtype="string"),
            "home_qb": pd.Series(["Tyler Huntley"], dtype="string"),
            "away_qb": pd.Series(["Joe Burrow"], dtype="string"),
            "qb_change_epa_home": [-0.31],
            "qb_change_epa_away": [0.0],
            "qb_watch": pd.Series([1], dtype="Int64"),
            "qb_inferred": pd.Series([0], dtype="Int64"),
        }
    )


class _StubStarterProvider:
    def __init__(self, rows=None):
        self._rows = _advisory_rows() if rows is None else rows

    def snapshot(self, season, week):
        return StarterSnapshot(
            rows=self._rows,
            observed_at=datetime(2026, 9, 8, 12, tzinfo=UTC),
        )


class _BrokenStarterProvider:
    def snapshot(self, season, week):
        raise StartersUnavailableError("down")


def test_payload_carries_starter_metadata_when_a_provider_is_configured():
    service = SlateService(
        feature_rows_with_2026_weeks(), starter_provider=_StubStarterProvider()
    )
    payload = service.payload(2026, 1, "ridge", 2.0)
    assert payload["starters"]["stale"] is False
    assert payload["games"][0]["qb_watch"] == 1
    assert payload["games"][0]["home_qb"] == "Tyler Huntley"


def test_payload_starters_key_is_none_without_a_provider():
    payload = SlateService(feature_rows_with_2026_weeks()).payload(2026, 1, "ridge", 2.0)
    assert payload["starters"] is None
    assert payload["games"][0]["qb_watch"] is None


def test_an_unavailable_starter_feed_still_returns_a_slate():
    service = SlateService(
        feature_rows_with_2026_weeks(), starter_provider=_BrokenStarterProvider()
    )
    payload = service.payload(2026, 1, "ridge", 2.0)
    assert payload["games"]
    assert payload["starters"] is None
    assert payload["games"][0]["qb_watch"] is None


def test_the_advisory_never_moves_a_web_prediction():
    rows = feature_rows_with_2026_weeks()
    without = SlateService(rows).payload(2026, 1, "ridge", 2.0)["games"][0]
    with_advisory = SlateService(
        rows, starter_provider=_StubStarterProvider()
    ).payload(2026, 1, "ridge", 2.0)["games"][0]
    for key in ("model_spread", "model_total", "spread_gap", "total_gap", "edge_flag"):
        assert without[key] == with_advisory[key], key
```

`SlateService`'s existing positional parameters are `(rows, packaged_schedule, market_provider, clock)`, so `starter_provider` must be added as a **keyword-capable parameter after** them to avoid breaking the positional call sites at `tests/test_web_service.py:106` and throughout.

- [ ] **Step 2: Run to verify failure**

```bash
./.venv/Scripts/python.exe -m pytest tests/test_web_service.py -k starter -v
```

Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'starter_provider'`.

- [ ] **Step 3: Implement**

Add `starter_provider=None` to `SlateService.__init__` and to the `from_path` classmethod, storing it as `self._starter_provider`. Add:

```python
    def _starter_snapshot(self, season: int, week: int):
        if self._starter_provider is None:
            return None
        try:
            return self._starter_provider.snapshot(season, week)
        except StartersUnavailableError:
            # Advisory only -- a slate without it is still a correct slate.
            return None

    @staticmethod
    def _starter_metadata(snapshot):
        if snapshot is None:
            return None
        observed_at = pd.Timestamp(snapshot.observed_at)
        observed_at = (
            observed_at.tz_localize(UTC)
            if observed_at.tzinfo is None
            else observed_at.tz_convert(UTC)
        )
        return {
            "source": snapshot.source,
            "observed_at": observed_at.isoformat(),
            "stale": bool(snapshot.stale),
        }
```

In `_slate_result`, add before the `build_slate` call:

```python
        starter_snapshot = self._starter_snapshot(season, week)
        starters = None if starter_snapshot is None else starter_snapshot.rows
```

pass `starters=starters` to `build_slate`, and change the return to a three-tuple carrying `self._starter_metadata(starter_snapshot)`. Update `slate()` to ignore the extra element and `payload()` to:

```python
        slate, metadata, starter_metadata = self._slate_result(
            season, week, estimator, edge_threshold
        )
        return {
            "games": self._json_records(slate),
            "market": metadata,
            "starters": starter_metadata,
        }
```

In `src/nfl_game/web/app.py`, add two columns to `renderGames`:

```javascript
    ['QB', game => qbCell(game)],
```

and a helper next to the other formatters:

```javascript
function qbCell(game) {
  if (game.qb_watch === null || game.qb_watch === undefined) return 'n/a';
  if (game.qb_watch !== 1) return '';
  const parts = [];
  for (const [name, delta] of [[game.home_qb, game.qb_change_epa_home],
                               [game.away_qb, game.qb_change_epa_away]]) {
    if (delta === null || delta === undefined || delta === 0) continue;
    parts.push(`${name === null ? 'unknown' : name} ${delta > 0 ? '+' : ''}${delta.toFixed(2)}`);
  }
  return (parts.join('; ') || 'change') + (game.qb_inferred === 1 ? ' (inferred)' : '');
}
```

Add a `qb-watch` class to the row when `game.qb_watch === 1`, kept distinct from the existing `edge` class so a game can show both.

- [ ] **Step 4: Run the web suite**

```bash
./.venv/Scripts/python.exe -m pytest tests/test_web_service.py tests/test_web_schedule_page.py tests/test_webapp.py -v
```

Expected: all PASS.

- [ ] **Step 5: Lint and commit**

```bash
./.venv/Scripts/python.exe -m ruff check .
git add src/nfl_game/web/service.py src/nfl_game/web/app.py tests/test_web_service.py
git commit -m "feat: show the expected-starter advisory on the web dashboard"
```

---

### Task 8: Prove the fixtures match a real feed era

Every fixture in Tasks 1, 3 and 4 was **written by hand**. The spec requires fixtures drawn from live data, and for good reason: this project has already shipped a block that emitted a constant because its fixtures invented a schema the feed never had. An invented fixture that matches neither era passes every test and proves nothing.

**Files:**
- Create: `tests/test_depth_fixture_schema.py`
- Modify: `docs/superpowers/plans/2026-09-03-qb-starter-advisory.md` (record the observed column sets)

**Interfaces:**
- Consumes: `_TEAM_SOURCES`, `_POSITION_SOURCES`, `_PLAYER_SOURCES`, `_RANK_SOURCES` from `nfl_game.ratings.depth`.
- Produces: nothing other tasks depend on. This is a verification task.

- [ ] **Step 1: Observe the live feed's real columns for both eras**

```bash
./.venv/Scripts/python.exe -c "
from nfl_game.data.nfl import load_depth_charts
for season in (2019, 2025):
    df = load_depth_charts([season], save=False)
    present = sorted(c for c in df.columns if df[c].notna().any())
    print(season, len(df), present)
"
```

Record the two printed column lists in this plan file under this step, verbatim. Then confirm by inspection that:

- the 2019 list contains `season`, `week`, `team`, `position`, `depth_team` and **not** a populated `dt`;
- the 2025 list contains `club_code`, `pos_abb`, `pos_rank`, `dt` and **not** a populated `season`/`week`/`team`.

**If either fixture used a column the live feed does not populate for that era, fix the fixture before continuing** — the tests it feeds are worthless until it matches.

- [ ] **Step 2: Write the offline drift test**

This runs in CI without a network. It pins the hand-written fixtures against the column spellings `depth.py` actually coalesces, so a future feed rename that updates `depth.py` but not the fixtures fails loudly.

```python
from nfl_game.ratings.depth import (
    _PLAYER_SOURCES,
    _POSITION_SOURCES,
    _RANK_SOURCES,
    _TEAM_SOURCES,
)

# The exact column spellings the hand-written fixtures in tests/test_qb.py,
# tests/test_starters.py and tests/test_live_starters.py use for each era.
TIMESTAMPED_FIXTURE_COLUMNS = {"club_code", "gsis_id", "pos_abb", "pos_rank", "dt"}
LABELLED_FIXTURE_COLUMNS = {"season", "week", "team", "player_id", "position", "depth_team"}


def test_each_fixture_column_is_one_depth_py_actually_reads():
    known = set(_TEAM_SOURCES) | set(_POSITION_SOURCES) | set(_PLAYER_SOURCES)
    known |= set(_RANK_SOURCES) | {"dt", "season", "week"}
    for columns in (TIMESTAMPED_FIXTURE_COLUMNS, LABELLED_FIXTURE_COLUMNS):
        assert columns <= known, sorted(columns - known)


def test_the_two_fixture_eras_are_genuinely_disjoint_on_identity():
    # If these overlap, one "era" fixture is really the other wearing a different hat,
    # and the era branch it claims to exercise is not being exercised at all.
    identity = {"club_code", "team", "gsis_id", "player_id", "pos_abb", "position"}
    assert not (
        TIMESTAMPED_FIXTURE_COLUMNS & LABELLED_FIXTURE_COLUMNS & identity
    )
```

- [ ] **Step 3: Run**

```bash
./.venv/Scripts/python.exe -m pytest tests/test_depth_fixture_schema.py -v
```

Expected: both PASS.

- [ ] **Step 4: Lint and commit**

```bash
./.venv/Scripts/python.exe -m ruff check .
git add tests/test_depth_fixture_schema.py docs/superpowers/plans/2026-09-03-qb-starter-advisory.md
git commit -m "test: pin the depth-chart fixtures to columns the live feed really carries"
```

---

### Task 9: Verify the invariants and document

The whole point of Track A is that it changed nothing it was not supposed to change. This task proves it.

**Files:**
- Modify: `CLAUDE.md` (new "QB starter advisory" subsection under the architecture package list), `README.md` (web dashboard operations)
- Test: the full suite plus the backtest

- [ ] **Step 1: Run the complete test suite**

```bash
./.venv/Scripts/python.exe -m pytest
```

Expected: all PASS, no skips that were not skipping before.

- [ ] **Step 2: Run the regression baseline and compare against CLAUDE.md**

```bash
./.venv/Scripts/python.exe scripts/backtest.py --test-seasons 2021-2025
```

Expected, exactly: margin MAE `10.274`, market `9.752`; total MAE `10.684`, market `10.309`; ATS `0.4977` (n=1326); O/U `0.5022` (n=1348); `model_coef` `-0.0218`, `market_coef` `1.0755`; r² `0.2083`.

**If any figure moved, stop.** Track A touches no feature and no estimator; a moved baseline means something leaked into the model path and must be found before continuing. Do not adjust the expected values to match.

- [ ] **Step 3: Confirm `FEATURE_COLS` is untouched**

```bash
git diff master -- src/nfl_game/model/features.py
```

Expected: empty output.

- [ ] **Step 4: Document**

Add to `CLAUDE.md`'s package-ownership list, after the `market/` bullet:

```
- `market/live_starters.py` — the expected-starter advisory. It is PRESENTATION, joined
  onto the slate after prediction and after `edge_flag`, so no advisory failure can move
  `model_margin`, `model_total` or `edge_flag`; `tests/test_compare.py` pins that
  directly. `qb_watch` is a separate marker from `edge_flag` on purpose. Its
  `qb_change_epa` is NOT the Ridge-v2 C2 figure: the provider loads player stats for the
  prior and current season only, while C2 trains on the full corpus, so advisory numbers
  must never be quoted as research numbers. The advisory is never written to a packaged
  artifact and never reaches the tracker.
```

Add a paragraph to `README.md`'s "Web dashboard operations" recording that the `QB` column reads `n/a` when the depth-chart feed is unreachable, that this is expected rather than a fault, and that `scripts/slate.py --no-starters` skips the fetch entirely.

- [ ] **Step 5: Commit**

```bash
./.venv/Scripts/python.exe -m ruff check .
git add CLAUDE.md README.md
git commit -m "docs: record the QB starter advisory and its presentation-only contract"
```

---

## Not in this plan

- **Track B** — the conditional re-measurement of the QB block, its Gate 0 publication-cutoff visibility check, and any change to `FEATURE_COLS`. Separate branch, separate plan. See the spec.
- **`load_injuries`** — practice participation and game status. Deferred; the depth chart alone is a weak proxy that routinely lists an injured starter as QB1.
- **The CLAUDE.md 4-vs-5-day discrepancy** — `PUBLISH_BEFORE` is 5 days (`tracking/live.py:16`) but the "Immutable and market-blind facts" section says 4. A separate correction, not a side effect of this work.
