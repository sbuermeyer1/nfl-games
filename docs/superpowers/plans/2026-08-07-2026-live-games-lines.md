# 2026 Live Games, Betting Lines, and Tracker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add current and next-week 2026 Ridge predictions with five-minute nflverse line refreshes, a full 2026 schedule view, leak-free scheduled model-data refreshes, and an auditable automated live tracker with published and closing lines.

**Spec:** `docs/superpowers/specs/2026-08-07-2026-live-games-lines-design.md`

**Architecture:** Keep the web process read-only and split the delivery into three boundaries: deterministic schedule-driven model artifacts, a concurrency-safe runtime market overlay with packaged fallback, and a pure live-ledger state machine driven by GitHub Actions. Preserve the immutable 2021-2025 historical corpus, derive 2026 prediction weeks from schedules rather than play-by-play, and enable tracker writes only after a dry-run rollout passes review.

**Tech Stack:** Python 3.11+, pandas, NumPy, scikit-learn Ridge through the existing `GameModel`, nflreadpy/nflverse, PyArrow/Parquet, FastAPI, vanilla HTML/CSS/JavaScript, pytest, QuickJS, Ruff, Docker, GitHub Actions.

## Global Constraints

- Use `nflreadpy.load_schedules()` as the 2026 schedule and market source.
- Cache runtime market observations for exactly five minutes per application process.
- Keep a short bounded upstream wait; a slow or failed refresh must not block the site indefinitely.
- Use packaged market rows as visibly stale fallback; never silently relabel fallback as live.
- Treat spread and total independently; one missing market must not suppress the other.
- Zero is a valid line. Missing values remain null and never become zero, `NaN`, or infinity in API output.
- Keep market lines out of `FEATURE_COLS`, rating inputs, and estimator training.
- Build 2026 ratings from completed games in weeks strictly before the target week; never use same-week or future results.
- Package predictions only for the earliest unplayed 2026 regular-season week and the next distinct unplayed week.
- Keep the complete 2026 regular-season schedule separately browseable.
- Official tracking remains Ridge-only under `ridge-v1` unless a separately reviewed model version is introduced.
- Freeze model predictions once at the first eligible run within 24 hours of kickoff.
- Freeze spread and total independently; retry a missing market until one hour before kickoff, then exclude that market permanently.
- Official grades use frozen published lines. Closing grades and signed CLV remain separate.
- Positive CLV always means the published number was better for the frozen pick.
- Qualified picks use absolute edge `>= 2.0`; spread cohorts are cumulative absolute `5.0+`, `10.0+`, and `15.0+`.
- Pushes remain displayed and excluded from win-rate denominators.
- The deployed web process never writes official artifacts.
- Preserve the exact 2021-2025 acceptance baseline: 1,359 games, ATS `660-666-33` (`n=1326`, `0.497737556561086`), and O/U `677-671-11` (`n=1348`, `0.5022255192878339`).
- Generated writes are atomic, deterministic, and no-op when content is unchanged.
- Automation never amends or force-pushes. A moved remote causes the run to fail safely.
- Stage 1 keeps official tracker writes disabled. Stage 2 requires explicit review before enabling repository writes.

---

## File structure

### New files

- `src/nfl_game/data/schedule.py` — normalized schedule schema, kickoff parsing, finalization guard, and current/next-week selection.
- `src/nfl_game/pipeline/__init__.py` — pipeline package boundary.
- `src/nfl_game/pipeline/refresh_2026.py` — deterministic 2026 schedule and feature artifact construction.
- `src/nfl_game/market/live.py` — nflverse market snapshots, cache, timeout, concurrency, and freshness state.
- `src/nfl_game/tracking/live.py` — pure live publication/deadline/closing lifecycle.
- `src/nfl_game/web/schedule_page.py` — framework-free full 2026 schedule page.
- `scripts/refresh_2026.py` — CLI for dry-run or atomic 2026 artifact refresh.
- `scripts/update_live_tracker.py` — CLI for dry-run or atomic live-ledger advancement.
- `tests/test_schedule.py` — normalization, kickoff, finalization, and active-week tests.
- `tests/test_refresh_2026.py` — fixed-history preservation, scheduled target rows, and atomic output tests.
- `tests/test_live_market.py` — cache, timeout, concurrency, validation, and stale-state tests.
- `tests/test_live_tracking.py` — publication, independent markets, exclusions, closing, postponement, and idempotency tests.
- `tests/test_update_live_tracker.py` — CLI dry-run, baseline, no-op, and atomic write tests.
- `tests/test_operations_cli.py` ? documented command-line contract tests.
- `tests/test_web_schedule_page.py` — schedule page/API rendering and stale-response tests.
- `.github/workflows/refresh-2026-model.yml` — daily/manual schedule and model artifact workflow.
- `.github/workflows/update-2026-tracker.yml` — 15-minute/manual tracker lifecycle workflow with dry-run gate.
- `data/processed/schedule_2026.parquet` — packaged full-schedule and fallback-market artifact.

### Modified files

- `src/nfl_game/data/nfl.py` — pass requested seasons to nflreadpy instead of downloading every season.
- `src/nfl_game/ratings/build.py` — build ratings for explicit schedule target weeks.
- `src/nfl_game/model/features.py` — accept normalized schedule metadata without changing model columns.
- `src/nfl_game/web/service.py` — current/next defaults, live market overlay, payload metadata, and raw model predictions.
- `src/nfl_game/web/app.py` — market-aware slate response, schedule routes, navigation, and freshness rendering.
- `src/nfl_game/web/runtime.py` — require packaged schedule and construct the live market provider.
- `src/nfl_game/web/tracker_service.py` — live seasons, publication state, and expanded audit fields.
- `src/nfl_game/web/tracker_page.py` — live season/overall selection and publication/closing audit output.
- `src/nfl_game/tracking/ledger.py` — live publication facts, exclusions, voids, derived grades, and validation.
- `src/nfl_game/tracking/summary.py` — exclude unpublished markets and expose live audit facts.
- `scripts/game_app.py` — pass `schedule_2026.parquet` at startup.
- `tests/test_data_nfl.py` — requested-season loader contract.
- `tests/test_build_ratings.py` — explicit future schedule week coverage.
- `tests/test_features.py` — 2026 rows retain market nulls without feature nulls.
- `tests/test_web_service.py` — active defaults, overlay, fallback, and payload metadata.
- `tests/test_webapp.py` — API shape, freshness UI, schedule navigation, and auth.
- `tests/test_web_runtime.py` — required/corrupt schedule artifact and provider construction.
- `tests/test_tracking_ledger.py` — expanded live fact schema and validation.
- `tests/test_tracking_summary.py` — excluded market denominators and live closing summaries.
- `tests/test_web_tracker_service.py` — separate historical/live season options and audit facts.
- `tests/test_web_tracker_page.py` — live selection and official publication display.
- `.gitignore` — allow exactly `data/processed/schedule_2026.parquet`.
- `.dockerignore` — include the packaged 2026 schedule artifact.
- `Dockerfile` — copy the packaged 2026 schedule artifact.
- `README.md` — live data semantics, automation, dry-run enablement, and recovery.
- `CLAUDE.md` — 2026 data boundaries and automation invariants.

---

### Task 1: Normalize the 2026 schedule and select active weeks

**Files:**
- Create: `src/nfl_game/data/schedule.py`
- Create: `tests/test_schedule.py`
- Modify: `src/nfl_game/data/nfl.py`
- Modify: `tests/test_data_nfl.py`

**Interfaces:**
- Consumes: nflverse schedule frames with game identity, `gameday`, `gametime`, teams, results, and line fields.
- Produces: `ScheduleSchemaError`, `normalize_schedule(rows: pd.DataFrame, season: int) -> pd.DataFrame`, `is_final_game(row: pd.Series, now: datetime) -> bool`, and `active_prediction_weeks(schedule: pd.DataFrame, now: datetime) -> list[int]`.

- [ ] **Step 1: Write failing normalization and active-week tests**

Create `tests/test_schedule.py` with literal 2026 fixtures:

```python
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from nfl_game.data.schedule import (
    ScheduleSchemaError,
    active_prediction_weeks,
    is_final_game,
    normalize_schedule,
)


NOW = datetime(2026, 9, 10, 16, tzinfo=timezone.utc)


def raw_schedule():
    return pd.DataFrame(
        [
            {
                "game_id": "2026_01_LA_SF",
                "season": 2026,
                "game_type": "REG",
                "week": 1,
                "gameday": "2026-09-09",
                "gametime": "20:20",
                "away_team": "LA",
                "home_team": "SF",
                "result": 3.0,
                "total": 47.0,
                "spread_line": 2.5,
                "total_line": 45.5,
            },
            {
                "game_id": "2026_01_BUF_NYJ",
                "season": 2026,
                "game_type": "REG",
                "week": 1,
                "gameday": "2026-09-13",
                "gametime": "13:00",
                "away_team": "BUF",
                "home_team": "NYJ",
                "result": np.nan,
                "total": np.nan,
                "spread_line": -3.0,
                "total_line": np.nan,
            },
            {
                "game_id": "2026_02_KC_DEN",
                "season": 2026,
                "game_type": "REG",
                "week": 2,
                "gameday": "2026-09-20",
                "gametime": "16:25",
                "away_team": "KC",
                "home_team": "DEN",
                "result": np.nan,
                "total": np.nan,
                "spread_line": np.nan,
                "total_line": 46.0,
            },
            {
                "game_id": "2026_03_PRE_X_Y",
                "season": 2026,
                "game_type": "PRE",
                "week": 3,
                "gameday": "2026-08-20",
                "gametime": "20:00",
                "away_team": "X",
                "home_team": "Y",
                "result": 1.0,
                "total": 30.0,
                "spread_line": 1.0,
                "total_line": 33.0,
            },
        ]
    )


def test_normalize_schedule_filters_regular_season_normalizes_teams_and_kickoff():
    out = normalize_schedule(raw_schedule(), 2026)
    assert list(out["game_id"]) == [
        "2026_01_LA_SF",
        "2026_01_BUF_NYJ",
        "2026_02_KC_DEN",
    ]
    assert out.iloc[0]["away_team"] == "LAR"
    assert str(out["kickoff_at"].dtype) == "datetime64[ns, UTC]"
    assert out["game_id"].is_unique


def test_active_prediction_weeks_returns_earliest_two_weeks_with_unplayed_games():
    out = normalize_schedule(raw_schedule(), 2026)
    assert active_prediction_weeks(out, NOW) == [1, 2]


def test_finalization_requires_scores_and_six_hours_after_kickoff():
    row = normalize_schedule(raw_schedule(), 2026).iloc[0]
    assert not is_final_game(row, datetime(2026, 9, 10, 1, tzinfo=timezone.utc))
    assert is_final_game(row, NOW)


def test_normalize_schedule_rejects_duplicate_ids_and_invalid_lines():
    duplicate = pd.concat([raw_schedule(), raw_schedule().iloc[[0]]], ignore_index=True)
    with pytest.raises(ScheduleSchemaError, match="duplicate game_id"):
        normalize_schedule(duplicate, 2026)

    invalid = raw_schedule()
    invalid.loc[0, "spread_line"] = float("inf")
    with pytest.raises(ScheduleSchemaError, match="spread_line"):
        normalize_schedule(invalid, 2026)
```

Add a loader-contract test to `tests/test_data_nfl.py`:

```python
def test_load_schedules_forwards_requested_seasons(monkeypatch):
    calls = []

    class FakePolars:
        def to_pandas(self):
            return pd.DataFrame({"season": [2026], "home_team": ["KC"], "away_team": ["BUF"]})

    monkeypatch.setattr(
        nfl.nflreadpy,
        "load_schedules",
        lambda seasons: calls.append(seasons) or FakePolars(),
    )
    nfl.load_schedules([2026], save=False)
    assert calls == [[2026]]
```

- [ ] **Step 2: Run the tests to verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_schedule.py tests\test_data_nfl.py -q
```

Expected: collection fails because `nfl_game.data.schedule` does not exist and the existing loader calls `load_schedules()` without the requested seasons.

- [ ] **Step 3: Implement the normalized schedule boundary**

Create `src/nfl_game/data/schedule.py` with:

```python
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from nfl_game.data.teams import normalize_team_codes

FINALIZATION_DELAY = timedelta(hours=6)
EASTERN = ZoneInfo("America/New_York")
REQUIRED_COLUMNS = {
    "game_id", "season", "game_type", "week", "gameday", "gametime",
    "away_team", "home_team", "result", "total", "spread_line", "total_line",
}


class ScheduleSchemaError(ValueError):
    pass


def _kickoffs(rows):
    text = rows["gameday"].astype(str) + " " + rows["gametime"].astype(str)
    parsed = pd.to_datetime(text, errors="coerce")
    if parsed.isna().any():
        raise ScheduleSchemaError("schedule contains invalid kickoff date or time")
    return parsed.dt.tz_localize(EASTERN, ambiguous="raise", nonexistent="raise").dt.tz_convert("UTC")


def normalize_schedule(rows, season):
    missing = sorted(REQUIRED_COLUMNS - set(rows.columns))
    if missing:
        raise ScheduleSchemaError(f"schedule missing columns: {missing}")
    out = rows.loc[(rows["season"] == season) & (rows["game_type"] == "REG")].copy()
    out = normalize_team_codes(out, ["home_team", "away_team"])
    out["kickoff_at"] = _kickoffs(out)
    if out["game_id"].duplicated().any():
        raise ScheduleSchemaError("schedule contains duplicate game_id values")
    for column in ("spread_line", "total_line", "result", "total"):
        numeric = pd.to_numeric(out[column], errors="coerce")
        if np.isinf(numeric.dropna().to_numpy(dtype=float)).any():
            raise ScheduleSchemaError(f"schedule column {column} contains infinite values")
        out[column] = numeric
    return out.sort_values(["kickoff_at", "game_id"]).reset_index(drop=True)


def is_final_game(row, now):
    kickoff = row["kickoff_at"].to_pydatetime()
    return pd.notna(row["result"]) and pd.notna(row["total"]) and now >= kickoff + FINALIZATION_DELAY


def active_prediction_weeks(schedule, now):
    unplayed = schedule.loc[
        [not is_final_game(row, now) for _, row in schedule.iterrows()]
    ]
    return sorted(int(week) for week in unplayed["week"].unique())[:2]
```

Modify `load_schedules()` in `src/nfl_game/data/nfl.py` so the upstream call is:

```python
requested = True if seasons is None else seasons
df = nflreadpy.load_schedules(requested).to_pandas()
```

Keep the existing normalization, local filter, and save behavior as defense in depth.

- [ ] **Step 4: Run focused tests to verify GREEN**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_schedule.py tests\test_data_nfl.py -q
.\.venv\Scripts\python.exe -m ruff check src\nfl_game\data tests\test_schedule.py tests\test_data_nfl.py
```

Expected: all focused tests and Ruff checks pass.

- [ ] **Step 5: Commit**

```powershell
git add src/nfl_game/data/schedule.py src/nfl_game/data/nfl.py tests/test_schedule.py tests/test_data_nfl.py
git commit -m "feat: normalize 2026 schedule data"
```

---

### Task 2: Build leak-free ratings for explicit scheduled weeks

**Files:**
- Modify: `src/nfl_game/ratings/build.py`
- Modify: `tests/test_build_ratings.py`
- Modify: `tests/test_features.py`

**Interfaces:**
- Consumes: team-game EPA history and explicit `(season, week)` target pairs from Task 1.
- Produces: `ratings_for_targets(team_games: pd.DataFrame, targets: list[tuple[int, int]], **kwargs) -> pd.DataFrame`.

- [ ] **Step 1: Write failing future-week and same-week exclusion tests**

Append to `tests/test_build_ratings.py`:

```python
from nfl_game.ratings.build import ratings_for_targets


def test_ratings_for_targets_builds_future_week_without_same_season_pbp():
    history = _games().query("season < 2024").copy()
    out = ratings_for_targets(history, [(2024, 1), (2024, 2)])
    assert sorted(out[["season", "week"]].drop_duplicates().itertuples(index=False, name=None)) == [
        (2024, 1),
        (2024, 2),
    ]
    assert out.groupby(["season", "week"])["team"].nunique().eq(4).all()


def test_ratings_for_targets_excludes_every_game_in_the_target_week(monkeypatch):
    seen = []

    def fake_build(team_games, asof_season, asof_week, **kwargs):
        cutoff = team_games.loc[
            (team_games["season"] < asof_season)
            | ((team_games["season"] == asof_season) & (team_games["week"] < asof_week))
        ]
        seen.append(set(zip(cutoff["season"], cutoff["week"], strict=True)))
        return pd.DataFrame({"team": ["AAA"], "off_rating": [0.0], "def_rating": [0.0]})

    monkeypatch.setattr("nfl_game.ratings.build.build_ratings", fake_build)
    ratings_for_targets(_games(), [(2024, 3)])
    assert all(week < 3 for season, week in seen[0] if season == 2024)
```

Add a feature assembly test to `tests/test_features.py` proving a future 2026 row with null results and lines retains finite `FEATURE_COLS` and null targets.

- [ ] **Step 2: Run the tests to verify RED**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_build_ratings.py tests\test_features.py -q
```

Expected: import fails because `ratings_for_targets` is not defined.

- [ ] **Step 3: Implement explicit target ratings**

Add to `src/nfl_game/ratings/build.py`:

```python
def ratings_for_targets(
    team_games: pd.DataFrame,
    targets: list[tuple[int, int]],
    **kwargs,
) -> pd.DataFrame:
    frames = []
    for season, week in sorted(set(targets)):
        ratings = build_ratings(
            team_games,
            asof_season=int(season),
            asof_week=int(week),
            **kwargs,
        )
        ratings.insert(0, "week", int(week))
        ratings.insert(0, "season", int(season))
        frames.append(ratings)
    if not frames:
        columns = ["season", "week", "team"]
        return pd.DataFrame(columns=columns)
    return pd.concat(frames, ignore_index=True)
```

Refactor `ratings_by_week()` to construct its existing target list and delegate to this function. Do not change weight calculations or week-level cutoff semantics.

- [ ] **Step 4: Run focused and regression tests**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_build_ratings.py tests\test_fit_ratings.py tests\test_features.py -q
.\.venv\Scripts\python.exe -m ruff check src\nfl_game\ratings src\nfl_game\model\features.py tests\test_build_ratings.py tests\test_features.py
```

Expected: all pass.

- [ ] **Step 5: Commit**

```powershell
git add src/nfl_game/ratings/build.py tests/test_build_ratings.py tests/test_features.py
git commit -m "feat: rate explicit scheduled weeks"
```

---

### Task 3: Build deterministic 2026 schedule and feature artifacts

**Files:**
- Create: `src/nfl_game/pipeline/__init__.py`
- Create: `src/nfl_game/pipeline/refresh_2026.py`
- Create: `scripts/refresh_2026.py`
- Create: `tests/test_refresh_2026.py`
- Create: `data/processed/schedule_2026.parquet`
- Modify: `scripts/build_dataset.py`
- Modify: `data/processed/game_features.parquet`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: frozen historical `game_features.parquet`, normalized 2026 schedule rows, team-game EPA history, available 2026 NGS rows, and an explicit UTC clock.
- Produces: `RefreshArtifacts(features: pd.DataFrame, schedule: pd.DataFrame)`, `build_refresh_artifacts(historical_features, schedules, team_games, ngs, now) -> RefreshArtifacts`, `write_artifacts_atomic(artifacts, feature_path, schedule_path) -> None`, and a CLI with mutually exclusive `--dry-run` and `--write`.

- [ ] **Step 1: Write failing fixed-history and target-week tests**

Create `tests/test_refresh_2026.py`:

```python
from datetime import datetime, timezone

import pandas as pd

from nfl_game.model.features import FEATURE_COLS
from nfl_game.pipeline.refresh_2026 import build_refresh_artifacts


def test_refresh_preserves_historical_rows_byte_for_value_and_adds_only_active_2026_weeks(
    monkeypatch,
):
    historical = historical_feature_fixture()
    schedule = normalized_2026_schedule_fixture(weeks=(1, 2, 3))
    now = datetime(2026, 9, 1, 12, tzinfo=timezone.utc)

    monkeypatch.setattr(
        "nfl_game.pipeline.refresh_2026.ratings_for_targets",
        lambda team_games, targets: rating_fixture(targets),
    )
    monkeypatch.setattr(
        "nfl_game.pipeline.refresh_2026.build_game_features",
        lambda schedules, ratings, ngs: feature_fixture_for_schedule(schedules),
    )

    result = build_refresh_artifacts(
        historical_features=historical,
        schedules=schedule,
        team_games=pd.DataFrame(),
        ngs=pd.DataFrame(),
        now=now,
    )

    pd.testing.assert_frame_equal(
        result.features.query("season <= 2025").reset_index(drop=True),
        historical.reset_index(drop=True),
        check_exact=True,
    )
    assert sorted(result.features.query("season == 2026")["week"].unique()) == [1, 2]
    assert result.features["game_id"].is_unique
    assert result.features[FEATURE_COLS].notna().all().all()
    assert sorted(result.schedule["week"].unique()) == [1, 2, 3]
```

Add tests that:

- no active weeks produces a clear no-op result;
- the historical acceptance gate is called before any write;
- a failed second parquet write leaves both original artifacts unchanged;
- `--dry-run` never changes output files;
- `--write` replaces both files only when their content digest changes.

- [ ] **Step 2: Run the tests to verify RED**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_refresh_2026.py -q
```

Expected: collection fails because the pipeline module does not exist.

- [ ] **Step 3: Implement the pure artifact builder**

Create `src/nfl_game/pipeline/refresh_2026.py` around this boundary:

```python
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from nfl_game.data.schedule import active_prediction_weeks
from nfl_game.model.features import FEATURE_COLS, build_game_features
from nfl_game.ratings.build import ratings_for_targets


@dataclass(frozen=True)
class RefreshArtifacts:
    features: pd.DataFrame
    schedule: pd.DataFrame


def build_refresh_artifacts(
    historical_features,
    schedules,
    team_games,
    ngs,
    now,
):
    weeks = active_prediction_weeks(schedules, now)
    targets = [(2026, week) for week in weeks]
    historical = historical_features.loc[historical_features["season"] <= 2025].copy()
    if not targets:
        return RefreshArtifacts(historical.reset_index(drop=True), schedules.copy())
    ratings = ratings_for_targets(team_games, targets)
    target_schedule = schedules.loc[schedules["week"].isin(weeks)]
    live = build_game_features(target_schedule, ratings, ngs)
    live = live.loc[live["season"].eq(2026)]
    combined = pd.concat([historical, live], ignore_index=True)
    if combined["game_id"].duplicated().any():
        raise ValueError("refreshed game features contain duplicate game_id values")
    if combined[FEATURE_COLS].isna().any().any():
        raise ValueError("refreshed game features contain null model features")
    return RefreshArtifacts(combined, schedules.copy())
```

Implement atomic paired writes by writing both parquet files into a temporary directory under the destination directory, reading and validating both temporary files, then replacing `schedule_2026.parquet` first and `game_features.parquet` second only after every validation passes. Keep backups until both replacements succeed and restore on any exception.

- [ ] **Step 4: Implement the injected CLI**

`scripts/refresh_2026.py` must:

1. load the existing feature artifact;
2. call `load_schedules([2026], save=False)` and `normalize_schedule`;
3. determine whether 2026 has completed regular-season games before requesting 2026 PBP/NGS;
4. load historical PBP needed for ratings without replacing historical feature rows;
5. call `build_refresh_artifacts`;
6. run `build_historical_ledger` and `assert_acceptance_baseline` in memory;
7. print old/new SHA-256 digests and row counts;
8. write only with `--write`; default to `--dry-run`.

Use an importable `main(argv=None, loaders=None, now=None)` so tests inject fixed loaders and clocks. Update `scripts/build_dataset.py` help text to direct 2026 production refreshes to this CLI while preserving the historical rebuild command.

When no 2026 NGS rows are available, construct a typed empty NGS frame with the exact columns consumed by `build_game_features` (`season`, `week`, `team`, all NGS metrics, and availability flags). Do not pass a zero-column frame into the feature builder.

- [ ] **Step 5: Run focused tests and deterministic double-build**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_refresh_2026.py tests\test_build_tracker.py -q
.\.venv\Scripts\python.exe -m ruff check src\nfl_game\pipeline scripts\refresh_2026.py scripts\build_dataset.py tests\test_refresh_2026.py
.\.venv\Scripts\python.exe scripts\refresh_2026.py --dry-run
.\.venv\Scripts\python.exe scripts\refresh_2026.py --dry-run
```

Expected: tests pass and the two dry runs report identical digests without modifying tracked files.

- [ ] **Step 6: Generate and review the first packaged artifacts**

```powershell
.\.venv\Scripts\python.exe scripts\refresh_2026.py --write
.\.venv\Scripts\python.exe scripts\refresh_2026.py --dry-run
git diff --stat -- data/processed/game_features.parquet data/processed/schedule_2026.parquet
```

Expected: the write creates the complete normalized 2026 schedule plus only the active two prediction weeks; the following dry run reports identical digests. Record schedule rows, unique game IDs, active weeks, feature rows by season, and null market counts for the task review. Add `!data/processed/schedule_2026.parquet` to `.gitignore` before staging so the new artifact is tracked explicitly.

- [ ] **Step 7: Commit**

```powershell
git add src/nfl_game/pipeline scripts/refresh_2026.py scripts/build_dataset.py tests/test_refresh_2026.py data/processed/game_features.parquet data/processed/schedule_2026.parquet .gitignore
git commit -m "feat: build deterministic 2026 artifacts"
```

---

### Task 4: Add the concurrency-safe live nflverse market provider

**Files:**
- Create: `src/nfl_game/market/live.py`
- Create: `tests/test_live_market.py`

**Interfaces:**
- Consumes: an injected `loader(seasons, save=False) -> pd.DataFrame`, clock, TTL, and timeout.
- Produces: `MarketSnapshot(rows, observed_at, source, stale)`, `MarketUnavailableError`, and `NflverseMarketProvider.snapshot(season: int) -> MarketSnapshot`.

- [ ] **Step 1: Write failing cache, timeout, and validation tests**

Create `tests/test_live_market.py` with a fake clock and blocking loader:

```python
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from threading import Event

import pandas as pd
import pytest

from nfl_game.market.live import MarketUnavailableError, NflverseMarketProvider


def test_snapshot_reuses_one_observation_for_five_minutes(schedule_fixture):
    calls = []
    clock = FakeClock(datetime(2026, 9, 1, tzinfo=timezone.utc))
    provider = NflverseMarketProvider(
        loader=lambda seasons, save=False: calls.append(seasons) or schedule_fixture,
        clock=clock,
        ttl=timedelta(minutes=5),
        timeout_seconds=0.2,
    )
    first = provider.snapshot(2026)
    clock.advance(minutes=4, seconds=59)
    second = provider.snapshot(2026)
    assert calls == [[2026]]
    assert second is first


def test_concurrent_cold_requests_share_one_loader_call(schedule_fixture):
    release = Event()
    calls = []

    def loader(seasons, save=False):
        calls.append(seasons)
        release.wait(timeout=1)
        return schedule_fixture

    provider = NflverseMarketProvider(loader=loader, timeout_seconds=1)
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(provider.snapshot, 2026) for _ in range(8)]
        release.set()
        snapshots = [future.result() for future in futures]
    assert calls == [[2026]]
    assert all(snapshot.observed_at == snapshots[0].observed_at for snapshot in snapshots)


def test_timeout_returns_stale_cache_but_cold_timeout_raises(schedule_fixture):
    clock = FakeClock(datetime(2026, 9, 1, tzinfo=timezone.utc))
    release = Event()
    calls = []

    def loader(seasons, save=False):
        calls.append(seasons)
        if len(calls) == 1:
            return schedule_fixture
        release.wait(timeout=1)
        return schedule_fixture

    provider = NflverseMarketProvider(
        loader=loader,
        clock=clock,
        timeout_seconds=0.01,
    )
    provider.snapshot(2026)
    clock.advance(minutes=6)
    assert provider.snapshot(2026).stale is True

    cold = NflverseMarketProvider(loader=blocking_loader, timeout_seconds=0.01)
    with pytest.raises(MarketUnavailableError, match="market feed unavailable"):
        cold.snapshot(2026)
```

Add tests for:

- successful rows use normalized team codes and UTC kickoff;
- duplicate IDs, team mismatches, invalid timestamps, and infinite lines reject the refresh;
- a successful snapshot with a null spread retains its valid total and null spread;
- a failed refresh never overwrites a previously valid cache.

- [ ] **Step 2: Run the tests to verify RED**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_live_market.py -q
```

Expected: collection fails because `nfl_game.market.live` does not exist.

- [ ] **Step 3: Implement the provider**

Use:

```python
@dataclass(frozen=True)
class MarketSnapshot:
    rows: pd.DataFrame
    observed_at: datetime
    source: str = "nflverse"
    stale: bool = False


class NflverseMarketProvider:
    def __init__(
        self,
        loader=load_schedules,
        clock=lambda: datetime.now(timezone.utc),
        ttl=timedelta(minutes=5),
        timeout_seconds=5.0,
    ):
        self._loader = loader
        self._clock = clock
        self._ttl = ttl
        self._timeout_seconds = timeout_seconds
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._snapshots = {}
        self._futures = {}

    def snapshot(self, season):
        now = self._clock()
        with self._lock:
            cached = self._snapshots.get(season)
            if cached is not None and now - cached.observed_at < self._ttl:
                return cached
            future = self._futures.get(season)
            if future is None:
                future = self._executor.submit(self._load_snapshot, season)
                self._futures[season] = future

        try:
            refreshed = future.result(timeout=self._timeout_seconds)
        except FutureTimeoutError as exc:
            return self._stale_or_raise(season, future, exc)
        except Exception as exc:
            return self._stale_or_raise(season, future, exc)

        with self._lock:
            self._futures.pop(season, None)
            self._snapshots[season] = refreshed
        return refreshed

    def _load_snapshot(self, season):
        raw = self._loader([season], save=False)
        rows = normalize_schedule(raw, season)
        return MarketSnapshot(
            rows=rows.copy(deep=True),
            observed_at=self._clock(),
        )

    def _stale_or_raise(self, season, future, exc):
        with self._lock:
            if future.done():
                self._futures.pop(season, None)
            cached = self._snapshots.get(season)
        if cached is not None:
            return replace(cached, rows=cached.rows.copy(deep=True), stale=True)
        raise MarketUnavailableError("market feed unavailable") from exc
```

Import `TimeoutError as FutureTimeoutError` from `concurrent.futures`, `replace` from `dataclasses`, and the locking/executor dependencies used above. Never mutate `MarketSnapshot.rows`; copy normalized frames before storing. Keep a timed-out future registered so later callers reuse or consume it rather than starting unbounded background threads.

- [ ] **Step 4: Run focused tests and Ruff**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_live_market.py tests\test_schedule.py -q
.\.venv\Scripts\python.exe -m ruff check src\nfl_game\market\live.py tests\test_live_market.py
```

Expected: pass.

- [ ] **Step 5: Commit**

```powershell
git add src/nfl_game/market/live.py tests/test_live_market.py
git commit -m "feat: cache live nflverse markets"
```

---

### Task 5: Overlay live lines in the slate service and API

**Files:**
- Modify: `src/nfl_game/web/service.py`
- Modify: `src/nfl_game/web/app.py`
- Modify: `tests/test_web_service.py`
- Modify: `tests/test_webapp.py`

**Interfaces:**
- Consumes: packaged features, packaged 2026 schedule, and optional `NflverseMarketProvider`.
- Produces: `SlateService.model_predictions(season, week, estimator="ridge") -> pd.DataFrame`, `SlateService.payload(season, week, estimator, edge_threshold) -> dict`, `SlateService.schedule_records(season) -> dict`, and market-aware `/api/slate`.

- [ ] **Step 1: Write failing service tests**

Add tests to `tests/test_web_service.py`:

```python
def test_options_default_to_earliest_unplayed_2026_week(monkeypatch):
    features = feature_rows_with_2026_weeks((1, 2))
    schedule = packaged_schedule_with_week_one_unplayed()
    service = SlateService(features, packaged_schedule=schedule, clock=fixed_now)
    options = service.options()
    assert options["latest"] == {"season": 2026, "week": 1}
    assert options["weeks"] == [1, 2]


def test_payload_overlays_live_markets_without_changing_model_predictions(monkeypatch):
    provider = FakeProvider(
        market_snapshot(
            spread_line=4.5,
            total_line=47.0,
            observed_at="2026-09-01T12:00:00Z",
        )
    )
    service = fake_fitted_2026_service(monkeypatch, provider=provider)
    body = service.payload(2026, 1, "ridge", 2.0)
    game = body["games"][0]
    assert game["model_spread"] == 4.0
    assert game["market_spread"] == 4.5
    assert game["market_total"] == 47.0
    assert body["market"] == {
        "source": "nflverse",
        "observed_at": "2026-09-01T12:00:00+00:00",
        "stale": False,
    }


def test_successful_feed_missing_one_market_does_not_use_packaged_value(monkeypatch):
    provider = FakeProvider(market_snapshot(spread_line=None, total_line=47.0))
    service = fake_fitted_2026_service(monkeypatch, provider=provider)
    game = service.payload(2026, 1, "ridge", 2.0)["games"][0]
    assert game["market_spread"] is None
    assert game["spread_market_status"] == "missing"
    assert game["market_total"] == 47.0
    assert game["total_market_status"] == "live"


def test_cold_feed_failure_uses_packaged_lines_as_stale(monkeypatch):
    provider = FailingProvider()
    service = fake_fitted_2026_service(monkeypatch, provider=provider)
    body = service.payload(2026, 1, "ridge", 2.0)
    assert body["market"]["source"] == "packaged"
    assert body["market"]["stale"] is True
    assert body["games"][0]["spread_market_status"] == "stale"
```

Add route tests to `tests/test_webapp.py` asserting `/api/slate` returns both `games` and `market`, while `/api/slate.csv` uses the exact same overlaid rows and blanks missing values.

- [ ] **Step 2: Run tests to verify RED**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_web_service.py tests\test_webapp.py -q
```

Expected: constructor and payload assertions fail because market injection and payload metadata do not exist.

- [ ] **Step 3: Implement market overlay and raw prediction boundary**

Change the constructor to:

```python
def __init__(
    self,
    features: pd.DataFrame,
    packaged_schedule: pd.DataFrame | None = None,
    market_provider=None,
    clock=lambda: datetime.now(timezone.utc),
):
```

Keep old historical tests supported when `packaged_schedule` and `market_provider` are omitted. Add:

```python
def model_predictions(self, season, week, estimator="ridge"):
    self._validate(season, week, estimator, DEFAULT_EDGE_THRESHOLD)
    target = self._target(season, week)
    return self._bundle(season, estimator).model.predict(target)


def _market_snapshot(self, season):
    if self._market_provider is not None:
        try:
            return self._market_provider.snapshot(season)
        except MarketUnavailableError:
            pass
    rows = self._packaged_schedule.loc[self._packaged_schedule["season"].eq(season)].copy()
    return MarketSnapshot(
        rows=rows,
        observed_at=self._packaged_observed_at,
        source="packaged",
        stale=True,
    )
```

In one internal `_slate_result(season, week, estimator, edge_threshold)` call:

1. select the target features;
2. obtain one market snapshot;
3. validate `game_id`, home team, and away team matches;
4. replace target `spread_line` and `total_line` with the snapshot values, preserving null;
5. generate predictions, calibration, and `build_slate`;
6. add `spread_market_status` and `total_market_status`;
7. return the frame and one response-level market metadata dictionary.

`payload()` converts the frame to JSON-safe records and returns:

```python
{"games": records, "market": metadata}
```

Keep `records()` returning only the list for internal compatibility. Make `csv()` call the same internal result once.

- [ ] **Step 4: Update the API route**

Change `/api/slate` in `src/nfl_game/web/app.py` to:

```python
return service.payload(season, week, estimator, edge_threshold)
```

Do not fetch market data separately in the route.

- [ ] **Step 5: Run focused tests**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_web_service.py tests\test_webapp.py tests\test_compare.py -q
.\.venv\Scripts\python.exe -m ruff check src\nfl_game\web\service.py src\nfl_game\web\app.py tests\test_web_service.py tests\test_webapp.py
```

Expected: pass.

- [ ] **Step 6: Commit**

```powershell
git add src/nfl_game/web/service.py src/nfl_game/web/app.py tests/test_web_service.py tests/test_webapp.py
git commit -m "feat: overlay live lines on 2026 slates"
```

---

### Task 6: Add freshness UI and a full 2026 schedule page

**Files:**
- Create: `src/nfl_game/web/schedule_page.py`
- Create: `tests/test_web_schedule_page.py`
- Modify: `src/nfl_game/web/app.py`
- Modify: `tests/test_webapp.py`

**Interfaces:**
- Consumes: `SlateService.schedule_records(2026)` and market-aware slate payloads from Task 5.
- Produces: `/schedule`, `/api/schedule?season=2026`, visible market freshness state, and null-safe schedule/slate rendering.

- [ ] **Step 1: Write failing markup and JavaScript behavior tests**

Create `tests/test_web_schedule_page.py` using the existing QuickJS harness style. Pin:

- navigation between `/`, `/schedule`, and `/tracker`;
- schedule rows render every 2026 regular-season game;
- successful current market data renders “Lines updated” with the timestamp;
- stale data renders a warning containing “stale”;
- missing spread or total renders an em dash and never `0`, `NaN`, or `Infinity`;
- a late response from an older request cannot overwrite the current selection.

Add `tests/test_webapp.py` assertions:

```python
def test_schedule_page_and_api_are_protected_and_use_service():
    http_client = client()
    assert http_client.get("/schedule").status_code == 200
    body = http_client.get("/api/schedule", params={"season": 2026}).json()
    assert body["season"] == 2026
    assert body["games"][0]["game_id"].startswith("2026_")
    assert body["market"]["source"] in {"nflverse", "packaged"}
```

- [ ] **Step 2: Run tests to verify RED**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_web_schedule_page.py tests\test_webapp.py -q
```

Expected: 404s and missing freshness elements.

- [ ] **Step 3: Create the schedule page**

Create `SCHEDULE_PAGE` in `src/nfl_game/web/schedule_page.py` with:

```html
<nav aria-label="Site navigation">
  <a href="/">Weekly predictions</a>
  <a href="/tracker">Performance tracker</a>
</nav>
<h1>2026 NFL Schedule</h1>
<p id="schedule-message" role="status" aria-live="polite"></p>
<div class="table-wrap"><table id="schedule-games"></table></div>
```

The script fetches `/api/schedule?season=2026`, renders week, kickoff, matchup, spread, and total with `textContent`, and owns requests with a monotonically increasing request token. Use `—` for null values.

- [ ] **Step 4: Add schedule routes and slate freshness rendering**

In `create_app()` add:

```python
@app.get("/schedule", response_class=HTMLResponse)
def schedule_page():
    return SCHEDULE_PAGE


@app.get("/api/schedule")
def schedule(season: int = 2026):
    return service.schedule_records(season)
```

Update `PAGE` to:

- link to `/schedule`;
- render `body.market.observed_at`;
- show a stale warning when `body.market.stale` is true;
- display `—` for null market fields;
- leave existing race-ownership and CSV protections intact.

- [ ] **Step 5: Run focused tests and browser-script tests**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_web_schedule_page.py tests\test_webapp.py tests\test_web_login.py -q
.\.venv\Scripts\python.exe -m ruff check src\nfl_game\web tests\test_web_schedule_page.py tests\test_webapp.py
```

Expected: pass.

- [ ] **Step 6: Commit**

```powershell
git add src/nfl_game/web/schedule_page.py src/nfl_game/web/app.py tests/test_web_schedule_page.py tests/test_webapp.py
git commit -m "feat: show 2026 schedule and line freshness"
```

---

### Task 7: Extend the immutable ledger for live publication state

**Files:**
- Modify: `src/nfl_game/tracking/ledger.py`
- Modify: `tests/test_tracking_ledger.py`
- Modify: `src/nfl_game/tracking/summary.py`
- Modify: `tests/test_tracking_summary.py`

**Interfaces:**
- Consumes: existing historical facts and new live lifecycle facts.
- Produces: validated publication statuses, exclusion reasons, observed timestamps, revised kickoff, void state, and deterministic live grades/CLV.

- [ ] **Step 1: Write failing schema and derivation tests**

Extend the fact helper in `tests/test_tracking_ledger.py` with:

```python
"spread_publication_status": "published",
"total_publication_status": "published",
"spread_exclusion_reason": pd.NA,
"total_exclusion_reason": pd.NA,
"published_spread_observed_at": pd.Timestamp("2026-09-01T12:00:00Z"),
"published_total_observed_at": pd.Timestamp("2026-09-01T12:00:00Z"),
"closing_spread_observed_at": pd.Timestamp("2026-09-06T17:01:00Z"),
"closing_total_observed_at": pd.Timestamp("2026-09-06T17:01:00Z"),
"current_kickoff_at": pd.Timestamp("2026-09-06T17:00:00Z"),
"void_reason": pd.NA,
```

Add:

```python
def test_excluded_market_is_no_pick_and_has_no_edge_clv_or_denominator():
    ledger = grade_ledger(
        facts(
            {
                "official_spread_line": np.nan,
                "published_spread_line": np.nan,
                "closing_spread_line": np.nan,
                "spread_publication_status": "excluded",
                "spread_exclusion_reason": "missing_line_at_deadline",
            }
        )
    )
    row = ledger.iloc[0]
    assert pd.isna(row["spread_pick"])
    assert pd.isna(row["spread_edge"])
    assert row["spread_grade"] == "no_pick"
    assert pd.isna(row["spread_clv"])
    validate_ledger(ledger)


def test_live_published_line_is_immutable_fact_and_clv_direction_stays_positive():
    out = grade_ledger(facts({})).iloc[0]
    assert out["official_spread_line"] == out["published_spread_line"] == 3.0
    assert out["spread_clv"] == 2.0


def test_void_game_is_no_pick_for_both_markets():
    out = grade_ledger(facts({"void_reason": "cancelled"})).iloc[0]
    assert out["spread_grade"] == "no_pick"
    assert out["total_grade"] == "no_pick"
```

Add summary tests showing excluded markets do not change all, qualified, threshold, CLV, or close-grade denominators.

- [ ] **Step 2: Run tests to verify RED**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_tracking_ledger.py tests\test_tracking_summary.py -q
```

Expected: missing columns and incorrect pending grades.

- [ ] **Step 3: Extend fact columns and validation**

Add these columns to `LEDGER_COLUMNS` before derived fields:

```python
"spread_publication_status",
"total_publication_status",
"spread_exclusion_reason",
"total_exclusion_reason",
"published_spread_observed_at",
"published_total_observed_at",
"closing_spread_observed_at",
"closing_total_observed_at",
"current_kickoff_at",
"void_reason",
```

Use allowed statuses:

```python
PUBLICATION_STATUSES = frozenset({"pending", "published", "excluded"})
```

Validation rules:

- backtest rows have all new live-only fields null;
- a live `published` market has non-null official/published line, null exclusion reason, and a source observation time;
- a live `pending` market has null official/published line and null exclusion reason;
- a live `excluded` market has null official/published line and a nonblank exclusion reason;
- official and published lines match for every live row;
- `current_kickoff_at` is non-null for live rows;
- void reason is null or a nonblank string;
- all timestamp fields parse as timezone-aware UTC values.

Update grading so `excluded` and void markets derive `no_pick`; published markets retain existing pick, grade, close-grade, and CLV formulas. Backtest conversion fills new fields with null without changing any existing historical derived value.

- [ ] **Step 4: Run focused tests and exact baseline**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_tracking_ledger.py tests\test_tracking_summary.py tests\test_build_tracker.py -q
$trackerScratch = Join-Path ([System.IO.Path]::GetTempPath()) "nfl-game-model-task7-tracker.parquet"
.\.venv\Scripts\python.exe scripts\build_tracker.py --output $trackerScratch
Remove-Item -LiteralPath $trackerScratch
.\.venv\Scripts\python.exe -m ruff check src\nfl_game\tracking tests\test_tracking_ledger.py tests\test_tracking_summary.py
```

Expected: all tests pass and the printed historical acceptance record remains exact.

- [ ] **Step 5: Commit**

```powershell
git add src/nfl_game/tracking/ledger.py src/nfl_game/tracking/summary.py tests/test_tracking_ledger.py tests/test_tracking_summary.py
git commit -m "feat: represent live tracker publication state"
```

---

### Task 8: Implement the pure live tracker lifecycle

**Files:**
- Create: `src/nfl_game/tracking/live.py`
- Create: `tests/test_live_tracking.py`

**Interfaces:**
- Consumes: existing live facts, normalized schedule rows, frozen model predictions, current UTC time, and model version.
- Produces: `advance_live_ledger(existing_live, schedule, predictions, now, model_version="ridge-v1") -> pd.DataFrame`.

- [ ] **Step 1: Write failing lifecycle boundary tests**

Create `tests/test_live_tracking.py` with fixed UTC kickoffs and assert:

```python
def test_first_run_inside_24_hours_freezes_prediction_and_available_markets():
    out = advance_live_ledger(
        empty_live_ledger(),
        schedule_fixture(hours_to_kickoff=23, spread=3.0, total=44.0),
        predictions_fixture(model_margin=6.0, model_total=47.0),
        NOW,
    )
    row = out.iloc[0]
    assert row["published_at"] == pd.Timestamp(NOW)
    assert row["model_margin"] == 6.0
    assert row["spread_publication_status"] == "published"
    assert row["published_spread_line"] == 3.0


def test_missing_spread_retries_but_total_freezes_independently():
    first = advance_live_ledger(
        empty_live_ledger(),
        schedule_fixture(hours_to_kickoff=23, spread=None, total=44.0),
        predictions_fixture(),
        NOW,
    )
    assert first.iloc[0]["spread_publication_status"] == "pending"
    assert first.iloc[0]["total_publication_status"] == "published"

    second = advance_live_ledger(
        first,
        schedule_fixture(hours_to_kickoff=2, spread=2.5, total=46.0),
        changed_predictions_fixture(),
        NOW + pd.Timedelta(hours=21),
    )
    assert second.iloc[0]["published_spread_line"] == 2.5
    assert second.iloc[0]["published_total_line"] == 44.0
    assert second.iloc[0]["model_margin"] == first.iloc[0]["model_margin"]


def test_missing_market_at_one_hour_is_excluded_forever():
    pending = pending_fixture(hours_to_kickoff=2)
    excluded = advance_live_ledger(
        pending,
        schedule_fixture(hours_to_kickoff=1, spread=None, total=44.0),
        predictions_fixture(),
        NOW,
    )
    later = advance_live_ledger(
        excluded,
        schedule_fixture(hours_to_kickoff=0.5, spread=2.0, total=44.0),
        changed_predictions_fixture(),
        NOW + pd.Timedelta(minutes=30),
    )
    assert later.iloc[0]["spread_publication_status"] == "excluded"
    assert later.iloc[0]["spread_exclusion_reason"] == "missing_line_at_deadline"
    assert pd.isna(later.iloc[0]["published_spread_line"])


def test_finalization_waits_six_hours_then_captures_results_and_close():
    published = published_fixture()
    too_early = advance_live_ledger(
        published,
        final_score_schedule(hours_since_kickoff=4),
        pd.DataFrame(),
        NOW,
    )
    assert pd.isna(too_early.iloc[0]["actual_margin"])

    final = advance_live_ledger(
        published,
        final_score_schedule(hours_since_kickoff=6, close_spread=5.0, close_total=46.0),
        pd.DataFrame(),
        NOW + pd.Timedelta(hours=2),
    )
    assert final.iloc[0]["actual_margin"] == 6.0
    assert final.iloc[0]["closing_spread_line"] == 5.0
    assert final.iloc[0]["spread_clv"] == 2.0
```

Also test:

- before 24 hours creates no record;
- a workflow outage that misses the one-hour deadline creates only excluded markets with reason `publication_window_missed`;
- repeated identical calls are frame-equal;
- a changed feature artifact cannot change a frozen model prediction;
- a postponed kickoff updates `current_kickoff_at` but not `kickoff_at` or publication facts;
- final rows with missing closing lines retry;
- incomplete rows older than seven days raise a visible lifecycle error;
- an explicitly voided game grades both markets `no_pick`.

- [ ] **Step 2: Run tests to verify RED**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_live_tracking.py -q
```

Expected: module does not exist.

- [ ] **Step 3: Implement the state machine**

Create constants and pure entry point:

```python
PUBLISH_BEFORE = pd.Timedelta(hours=24)
LINE_DEADLINE = pd.Timedelta(hours=1)
FINALIZATION_DELAY = pd.Timedelta(hours=6)
FINAL_RETRY_LIMIT = pd.Timedelta(days=7)


class LiveTrackerLifecycleError(RuntimeError):
    pass


def advance_live_ledger(
    existing_live,
    schedule,
    predictions,
    now,
    model_version=HISTORICAL_MODEL_VERSION,
):
    now = pd.Timestamp(now)
    schedule_rows = {
        str(row.game_id): row._asdict()
        for row in schedule.copy(deep=True).itertuples(index=False)
    }
    prediction_rows = {
        str(row.game_id): row._asdict()
        for row in predictions.copy(deep=True).itertuples(index=False)
    }
    records = {
        str(row.game_id): row._asdict()
        for row in existing_live.copy(deep=True).itertuples(index=False)
    }
    advanced = []

    for game_id in sorted(schedule_rows):
        game = schedule_rows[game_id]
        record = records.pop(game_id, None)
        if record is None:
            if now < pd.Timestamp(game["kickoff_at"]) - PUBLISH_BEFORE:
                continue
            prediction = prediction_rows.get(game_id)
            if prediction is None:
                raise LiveTrackerLifecycleError(
                    f"eligible game {game_id} has no Ridge prediction"
                )
            record = _new_record(game, prediction, now, model_version)
        else:
            record = _apply_schedule_change(record, game)

        record = _advance_market(record, game, "spread", now)
        record = _advance_market(record, game, "total", now)
        record = _capture_final(record, game, now)
        advanced.append(record)

    advanced.extend(records[game_id] for game_id in sorted(records))
    result = pd.DataFrame.from_records(advanced, columns=LIVE_LEDGER_COLUMNS)
    result = grade_ledger(result)
    validate_ledger(result)
    return result.sort_values("game_id", kind="stable").reset_index(drop=True)
```

Replace the comment block with small helpers:

- `_new_record(game, prediction, now, model_version)`;
- `_advance_market(record, game, kind, now)`;
- `_capture_final(record, game, now)`;
- `_apply_schedule_change(record, game)`.

Every helper returns a new dictionary. Finish with `grade_ledger`, `validate_ledger`, and stable sorting. Never mutate the caller’s frame.

`_new_record` freezes `model_margin`, `model_total`, `model_version`, original `kickoff_at`, and `published_at`. When the first run occurs at or inside the one-hour deadline, initialize both markets as `excluded` with `publication_window_missed`; otherwise initialize each market independently as `published` when its line exists or `pending` when it does not.

`_advance_market` changes only a `pending` market: publish the current finite line before the deadline, or mark it `excluded` with `missing_line_at_deadline` at the deadline. `_apply_schedule_change` changes only `current_kickoff_at`. `_capture_final` waits for scores and the six-hour guard, captures available closing lines independently, retries missing closes for seven days, then raises `LiveTrackerLifecycleError` unless the game was manually voided.

- [ ] **Step 4: Run lifecycle, ledger, and summary tests**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_live_tracking.py tests\test_tracking_ledger.py tests\test_tracking_summary.py -q
.\.venv\Scripts\python.exe -m ruff check src\nfl_game\tracking\live.py tests\test_live_tracking.py
```

Expected: pass.

- [ ] **Step 5: Commit**

```powershell
git add src/nfl_game/tracking/live.py tests/test_live_tracking.py
git commit -m "feat: advance official live tracker records"
```

---

### Task 9: Add the atomic live tracker update CLI

**Files:**
- Create: `scripts/update_live_tracker.py`
- Create: `tests/test_update_live_tracker.py`
- Modify: `src/nfl_game/web/service.py`
- Modify: `tests/test_web_service.py`

**Interfaces:**
- Consumes: packaged features, full tracker ledger, current normalized schedule, and injected clock/loaders.
- Produces: `main(argv=None, loader=None, now=None)`, dry-run change summaries, and optional atomic ledger writes.

- [ ] **Step 1: Write failing CLI tests**

Create `tests/test_update_live_tracker.py` to assert:

```python
def test_dry_run_reports_change_without_writing(tmp_path, monkeypatch, capsys):
    ledger_path = write_historical_ledger(tmp_path)
    original = ledger_path.read_bytes()
    result = run_cli(
        tmp_path,
        "--dry-run",
        now="2026-09-05T17:00:00Z",
        schedule=schedule_inside_publish_window(),
    )
    assert result == 0
    assert ledger_path.read_bytes() == original
    assert '"new_live_records": 1' in capsys.readouterr().out


def test_write_combines_unchanged_history_with_valid_live_rows(tmp_path):
    ledger_path = write_historical_ledger(tmp_path)
    run_cli(tmp_path, "--write", schedule=schedule_inside_publish_window())
    ledger = pd.read_parquet(ledger_path)
    assert len(ledger.query("record_type == 'backtest'")) == 1359
    assert len(ledger.query("record_type == 'live'")) == 1
    assert_acceptance_baseline(
        ledger.query("record_type == 'backtest'"),
        EXPECTED_BASELINE,
    )


def test_identical_write_is_no_op_with_identical_digest(tmp_path):
    first = run_cli(tmp_path, "--write", schedule=schedule_inside_publish_window())
    digest = sha256_file(tmp_path / "tracker_ledger.parquet")
    second = run_cli(tmp_path, "--write", schedule=schedule_inside_publish_window())
    assert first == 0
    assert second == 0
    assert sha256_file(tmp_path / "tracker_ledger.parquet") == digest
```

Also pin corrupt inputs, lifecycle failure, and temporary write cleanup.

- [ ] **Step 2: Run tests to verify RED**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_update_live_tracker.py -q
```

Expected: script does not exist.

- [ ] **Step 3: Expose raw model predictions**

Complete `SlateService.model_predictions()` from Task 5 so it returns only:

```python
["game_id", "model_margin", "model_total"]
```

It must use the cached Ridge bundle and packaged feature rows without applying live market lines or recalculating official picks.

- [ ] **Step 4: Implement the CLI**

`scripts/update_live_tracker.py` must:

1. parse `--features`, `--ledger`, `--season`, `--now`, and mutually exclusive `--dry-run`/`--write`;
2. load and validate both artifacts;
3. fetch and normalize the current season schedule;
4. select games that may publish or finalize;
5. generate Ridge predictions only for unpublished eligible games;
6. call `advance_live_ledger`;
7. concatenate unchanged historical rows with the returned live rows;
8. validate the full ledger and exact historical baseline;
9. print a deterministic JSON summary;
10. atomically replace the ledger only for `--write` and only when the digest changes.

Default to `--dry-run`. A manual void uses repeatable:

```text
--void-game 2026_01_AAA_BBB=cancelled
```

and is applied before lifecycle validation.

- [ ] **Step 5: Run focused tests and dry-run**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_update_live_tracker.py tests\test_live_tracking.py tests\test_web_service.py -q
.\.venv\Scripts\python.exe -m ruff check scripts\update_live_tracker.py tests\test_update_live_tracker.py src\nfl_game\web\service.py
.\.venv\Scripts\python.exe scripts\update_live_tracker.py --dry-run
```

Expected: tests pass and the real dry run does not modify `tracker_ledger.parquet`.

- [ ] **Step 6: Commit**

```powershell
git add scripts/update_live_tracker.py tests/test_update_live_tracker.py src/nfl_game/web/service.py tests/test_web_service.py
git commit -m "feat: update live tracker atomically"
```

---

### Task 10: Expose live seasons and publication facts on the tracker

**Files:**
- Modify: `src/nfl_game/web/tracker_service.py`
- Modify: `src/nfl_game/tracking/summary.py`
- Modify: `src/nfl_game/web/tracker_page.py`
- Modify: `tests/test_web_tracker_service.py`
- Modify: `tests/test_web_tracker_page.py`
- Modify: `tests/test_webapp.py`

**Interfaces:**
- Consumes: the expanded combined ledger.
- Produces: historical and live season options, overall/live summaries, and game audit rows containing publication status, published lines, close, grade, and CLV.

- [ ] **Step 1: Write failing live selection and audit tests**

Add service tests:

```python
def test_options_keep_historical_and_live_seasons_separate():
    service = TrackerService(combined_ledger_fixture())
    options = service.options()
    assert options["seasons"] == {
        "backtest": [2021, 2022, 2023, 2024, 2025],
        "live": [2026],
    }
    assert options["live_available"] is True


def test_live_audit_exposes_publication_closing_and_exclusion_facts():
    row = TrackerService(combined_ledger_fixture()).records("live", 2026)[0]
    assert row["published_at"] == "2026-09-05T17:00:00+00:00"
    assert row["spread_publication_status"] == "published"
    assert row["published_spread_line"] == 3.0
    assert row["closing_spread_line"] == 5.0
    assert row["spread_clv"] == 2.0
```

Add QuickJS page tests that switch the live tab, populate `Overall` and `2026`, request concrete live game rows, and render excluded markets as `excluded` rather than `pending`.

- [ ] **Step 2: Run tests to verify RED**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_web_tracker_service.py tests\test_web_tracker_page.py tests\test_webapp.py -q
```

Expected: options shape and live audit fields fail.

- [ ] **Step 3: Expand summary audit fields and JSON conversion**

Add to `_AUDIT_COLUMNS`:

```python
"published_at",
"kickoff_at",
"current_kickoff_at",
"spread_publication_status",
"total_publication_status",
"spread_exclusion_reason",
"total_exclusion_reason",
"published_spread_line",
"published_total_line",
"closing_spread_line",
"closing_total_line",
"spread_clv",
"total_clv",
"spread_close_grade",
"total_close_grade",
"void_reason",
```

Serialize timestamps as ISO 8601 strings in `_json_value`.

Change `TrackerService.options()` to return:

```python
"seasons": {
    record_type: sorted(
        int(value)
        for value in self._ledger.loc[
            self._ledger["record_type"].eq(record_type), "season"
        ].unique()
    )
    for record_type in ("backtest", "live")
},
```

Keep `default_record_type="backtest"` and `default_season="all"`.

- [ ] **Step 4: Update tracker page behavior**

Remove the rule that disables season selection for live records. Populate the selector from `options.seasons[activeRecordType]`, always include `Overall`, and fetch `/api/tracker/games` for any concrete historical or live season.

Add audit columns for publication status, published line, closing line, CLV, and close grade. Render null as `n/a` and exclusion reason as plain text. Keep all DOM writes on `textContent`.

- [ ] **Step 5: Run focused tests**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_web_tracker_service.py tests\test_web_tracker_page.py tests\test_webapp.py tests\test_tracking_summary.py -q
.\.venv\Scripts\python.exe -m ruff check src\nfl_game\web\tracker_service.py src\nfl_game\tracking\summary.py tests\test_web_tracker_service.py
```

Expected: pass.

- [ ] **Step 6: Commit**

```powershell
git add src/nfl_game/web/tracker_service.py src/nfl_game/tracking/summary.py src/nfl_game/web/tracker_page.py tests/test_web_tracker_service.py tests/test_web_tracker_page.py tests/test_webapp.py
git commit -m "feat: show official 2026 live records"
```

---

### Task 11: Wire runtime artifacts and GitHub Actions automation

**Files:**
- Create: `.github/workflows/refresh-2026-model.yml`
- Create: `.github/workflows/update-2026-tracker.yml`
- Modify: `src/nfl_game/web/runtime.py`
- Modify: `scripts/game_app.py`
- Modify: `tests/test_web_runtime.py`
- Modify: `.dockerignore`
- Modify: `Dockerfile`

**Interfaces:**
- Consumes: all three packaged artifacts and the two tested CLIs.
- Produces: fail-closed startup, daily/manual model refresh, 15-minute/manual tracker lifecycle, shared writer concurrency, and dry-run gating.

- [ ] **Step 1: Write failing runtime artifact tests**

Add to `tests/test_web_runtime.py`:

```python
def test_load_app_rejects_missing_packaged_2026_schedule(tmp_path):
    dataset = write_feature_artifact(tmp_path)
    tracker = write_tracker_artifact(tmp_path)
    with pytest.raises(RuntimeConfigError, match="packaged 2026 schedule not found"):
        load_app(
            protected_config(),
            dataset,
            tracker,
            tmp_path / "missing-schedule.parquet",
        )


def test_load_app_wraps_invalid_schedule_schema(tmp_path, monkeypatch):
    dataset = write_feature_artifact(tmp_path)
    tracker = write_tracker_artifact(tmp_path)
    schedule = tmp_path / "schedule.parquet"
    pd.DataFrame({"bad": [1]}).to_parquet(schedule)
    with pytest.raises(RuntimeConfigError, match="cannot load packaged 2026 schedule"):
        load_app(protected_config(), dataset, tracker, schedule)
```

Update entrypoint tests to require:

```python
PROCESSED_DIR / "schedule_2026.parquet"
```

- [ ] **Step 2: Run runtime tests to verify RED**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_web_runtime.py -q
```

Expected: signature and missing-artifact assertions fail.

- [ ] **Step 3: Wire fail-closed runtime loading**

Change:

```python
def load_app(config, dataset_path, tracker_path, schedule_path):
```

Require all three files. Read and normalize the schedule, construct `NflverseMarketProvider`, and pass both to `SlateService`. Wrap schedule/provider construction errors in `RuntimeConfigError` without exposing unsafe internals to web clients.

Update `scripts/game_app.py` to pass the schedule path.

- [ ] **Step 4: Package the schedule artifact**

Add the exact Docker build-context exception (the repository ignore exception was already added in Task 3):

```text
# .dockerignore
!data/processed/schedule_2026.parquet
```

Add to `Dockerfile`:

```dockerfile
COPY data/processed/schedule_2026.parquet ./data/processed/schedule_2026.parquet
```

Place it beside the other two explicit artifact copies.

- [ ] **Step 5: Create the model refresh workflow**

Create `.github/workflows/refresh-2026-model.yml` with:

```yaml
name: Refresh 2026 model data

on:
  workflow_dispatch:
  schedule:
    - cron: "30 10 * * *"

permissions:
  contents: write

concurrency:
  group: nfl-generated-data-writer
  cancel-in-progress: false

jobs:
  refresh:
    runs-on: ubuntu-latest
    timeout-minutes: 45
    steps:
      - uses: actions/checkout@11d5960a326750d5838078e36cf38b85af677262 # v4
      - uses: actions/setup-python@ece7cb06caefa5fff74198d8649806c4678c61a1 # v6
        with:
          python-version: "3.12"
          cache: pip
      - run: python -m pip install -e ".[dev]"
      - run: python -m pytest -q
      - run: python scripts/refresh_2026.py --write
      - run: docker build -t nfl-game-model:refresh .
      - name: Commit changed artifacts
        shell: bash
        run: |
          git add data/processed/game_features.parquet data/processed/schedule_2026.parquet
          git diff --cached --quiet && exit 0
          git config user.name "github-actions[bot]"
          git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
          git commit -m "data: refresh 2026 model artifacts"
          git fetch origin master
          git merge-base --is-ancestor origin/master HEAD
          git push origin HEAD:master
```

Use these reviewed full commit SHAs in both workflows; do not replace them with mutable tags.

- [ ] **Step 6: Create the dry-run-gated tracker workflow**

Create `.github/workflows/update-2026-tracker.yml` with the same permissions, concurrency group, pinning rule, Python setup, and safe push block. Use:

```yaml
on:
  workflow_dispatch:
  schedule:
    - cron: "0,15,30,45 * * 8-12,1-2 *"
```

Run one of:

```yaml
- name: Dry-run tracker update
  if: vars.ENABLE_OFFICIAL_TRACKER != 'true'
  run: python scripts/update_live_tracker.py --dry-run

- name: Write tracker update
  if: vars.ENABLE_OFFICIAL_TRACKER == 'true'
  run: python scripts/update_live_tracker.py --write
```

Only the write branch stages `data/processed/tracker_ledger.parquet`. No-op runs exit without a commit.

- [ ] **Step 7: Validate runtime, workflows, and build context**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_web_runtime.py tests\test_webapp.py tests\test_update_live_tracker.py tests\test_refresh_2026.py -q
.\.venv\Scripts\python.exe -m ruff check src scripts tests
go run github.com/rhysd/actionlint/cmd/actionlint@v1.7.12 .github/workflows/refresh-2026-model.yml .github/workflows/update-2026-tracker.yml
git diff --check
docker build -t nfl-game-model:2026 .
```

Expected: the pinned actionlint release reports no findings, and Docker succeeds with all three artifacts in `/app/data/processed`.

- [ ] **Step 8: Commit**

```powershell
git add .github/workflows/refresh-2026-model.yml .github/workflows/update-2026-tracker.yml src/nfl_game/web/runtime.py scripts/game_app.py tests/test_web_runtime.py .dockerignore Dockerfile
git commit -m "ci: automate 2026 model and tracker data"
```

---

### Task 12: Generate reviewed artifacts, document operations, and complete staged rollout

**Files:**
- Create: `tests/test_operations_cli.py`
- Modify: `data/processed/schedule_2026.parquet`
- Modify: `data/processed/game_features.parquet`
- Modify: `data/processed/tracker_ledger.parquet` because the live-schema migration changes its deterministic bytes.
- Modify: `README.md`
- Modify: `CLAUDE.md`

**Interfaces:**
- Consumes: all completed code, current nflverse feed, production Docker, and GitHub workflows.
- Produces: reviewed 2026 Stage 1 artifacts, operations runbook, release evidence, and an explicit Stage 2 enablement checkpoint.

- [ ] **Step 1: Write documentation acceptance tests before prose**

Create `tests/test_operations_cli.py`:

```python
from pathlib import Path
import subprocess
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLI_CASES = [
    ("scripts/refresh_2026.py", ["--dry-run", "--write"]),
    (
        "scripts/update_live_tracker.py",
        ["--dry-run", "--write", "--now", "--void-game"],
    ),
]


@pytest.mark.parametrize(("script", "expected_flags"), CLI_CASES)
def test_documented_cli_help_matches_supported_flags(script, expected_flags):
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / script), "--help"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    for flag in expected_flags:
        assert flag in result.stdout


@pytest.mark.parametrize(("script", "expected_flags"), CLI_CASES)
def test_dry_run_and_write_are_mutually_exclusive(script, expected_flags):
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / script), "--dry-run", "--write"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "not allowed with argument" in result.stderr
```

Keep the fixed-loader default-invocation tests in `tests/test_refresh_2026.py` and `tests/test_update_live_tracker.py`; each must assert that no output artifact changes when neither mode flag is supplied. Do not test README wording.

- [ ] **Step 2: Run tests to verify RED if any CLI contract is missing**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_operations_cli.py tests\test_refresh_2026.py tests\test_update_live_tracker.py -q
```

Expected: any missing CLI contract fails before documentation is written.

- [ ] **Step 3: Generate current 2026 artifacts**

Run:

```powershell
.\.venv\Scripts\python.exe scripts\refresh_2026.py --dry-run
.\.venv\Scripts\python.exe scripts\refresh_2026.py --write
.\.venv\Scripts\python.exe scripts\update_live_tracker.py --dry-run
```

Review and record:

- full schedule row count and unique game IDs;
- active prediction weeks;
- counts of null spread and total markets;
- historical and 2026 feature row counts;
- exact historical acceptance metrics;
- SHA-256 digest of all three packaged artifacts.

At the current preseason date, the tracker dry run should create no official record more than 24 hours before kickoff.

- [ ] **Step 4: Document exact operations**

Update `README.md` with:

- live line source, five-minute cache, and stale fallback behavior;
- current/next-week selection;
- schedule route and API;
- artifact refresh and tracker dry-run/write commands;
- publication, one-hour exclusion, finalization, CLV, push, postponement, and void semantics;
- workflow schedules, shared concurrency, no-op commits, and safe push rejection;
- Stage 1 `ENABLE_OFFICIAL_TRACKER` default-off behavior;
- manual workflow dispatch, upstream outage recovery, artifact rollback, and seven-day incomplete-record alert;
- all release verification commands.

Update `CLAUDE.md` with package boundaries, immutable facts, market-blind invariants, and the exact three packaged artifacts.

- [ ] **Step 5: Run the complete release gate**

Run fresh:

```powershell
.\.venv\Scripts\python.exe -m ruff format --check src tests scripts
.\.venv\Scripts\python.exe -m ruff check src tests scripts
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts\backtest.py --test-seasons 2021-2025
.\.venv\Scripts\python.exe scripts\refresh_2026.py --dry-run
.\.venv\Scripts\python.exe scripts\update_live_tracker.py --dry-run
docker build -t nfl-game-model:2026 .
```

The pytest gate above covers the authenticated-cookie boundary. Smoke the built image internally in no-auth loopback mode so the checks do not weaken production cookie security:

```powershell
function Wait-NflContainer {
    param([string]$ContainerName)
    $probe = "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).read()"
    $deadline = (Get-Date).AddSeconds(30)
    do {
        docker exec $ContainerName python -c $probe
        if ($LASTEXITCODE -eq 0) { return }
        Start-Sleep -Seconds 1
    } while ((Get-Date) -lt $deadline)
    docker logs $ContainerName
    throw "$ContainerName did not become healthy"
}

$endpointProbe = @"
import json
import urllib.request
for path in (
    "/health",
    "/api/options",
    "/api/schedule?season=2026",
    "/api/slate",
    "/api/tracker/options",
):
    with urllib.request.urlopen("http://127.0.0.1:8000" + path, timeout=10) as response:
        json.load(response)
    print(path)
"@

docker rm -f nfl-game-smoke nfl-game-stale 2>$null | Out-Null
docker run -d --name nfl-game-smoke nfl-game-model:2026 python scripts/game_app.py --no-auth
try {
    Wait-NflContainer "nfl-game-smoke"
    docker exec nfl-game-smoke python -c $endpointProbe
    if ($LASTEXITCODE -ne 0) { throw "connected container smoke failed" }
} finally {
    docker rm -f nfl-game-smoke | Out-Null
}

$staleProbe = @"
import json
import urllib.request
with urllib.request.urlopen("http://127.0.0.1:8000/api/slate", timeout=10) as response:
    payload = json.load(response)
assert payload["market"]["source"] == "packaged"
assert payload["market"]["stale"] is True
"@

docker run -d --name nfl-game-stale --network none nfl-game-model:2026 python scripts/game_app.py --no-auth
try {
    Wait-NflContainer "nfl-game-stale"
    docker exec nfl-game-stale python -c $staleProbe
    if ($LASTEXITCODE -ne 0) { throw "offline fallback smoke failed" }
} finally {
    docker rm -f nfl-game-stale | Out-Null
}
```

After Stage 1 deploys behind HTTPS, sign in with the private access code and repeat `/api/options`, `/api/schedule?season=2026`, `/api/slate`, and `/api/tracker/options` in the deployed browser session.

Expected:

- formatting, Ruff, and all tests pass;
- historical acceptance metrics remain exact;
- dry runs are no-op or show only expected deterministic changes;
- Docker contains all artifacts and starts;
- current live lines or clearly marked packaged fallback appear;
- tracker remains read-only and live writes remain disabled.

- [ ] **Step 6: Commit Stage 1 artifacts and documentation**

```powershell
git add data/processed/game_features.parquet data/processed/schedule_2026.parquet data/processed/tracker_ledger.parquet README.md CLAUDE.md tests/test_operations_cli.py
git commit -m "data: package reviewed 2026 schedule and predictions"
```

Before committing, remove `tracker_ledger.parquet` from staging if its digest did not change.

- [ ] **Step 7: Push Stage 1 and inspect workflow dry runs**

After explicit integration approval, push the feature branch or merge through the selected finishing workflow. Manually dispatch both workflows. Confirm:

- model workflow is green and no-ops when artifacts match;
- tracker workflow logs the expected publication candidates but creates no commit while `ENABLE_OFFICIAL_TRACKER` is absent or false;
- the deployed site passes the same authenticated smoke checks.

- [ ] **Step 8: Obtain explicit Stage 2 approval**

Present the dry-run publication rows, market timestamps, model version, edge calculations, and expected excluded markets to the user. Do not enable writes in the same step.

- [ ] **Step 9: Enable official tracker writes after approval**

With explicit approval and GitHub CLI authentication:

```powershell
gh variable set ENABLE_OFFICIAL_TRACKER --body true
$previousTrackerRunId = gh run list --workflow update-2026-tracker.yml --limit 1 --json databaseId --jq ".[0].databaseId"
gh workflow run update-2026-tracker.yml
$trackerRunDeadline = (Get-Date).AddMinutes(1)
do {
    Start-Sleep -Seconds 2
    $trackerRunId = gh run list --workflow update-2026-tracker.yml --limit 1 --json databaseId --jq ".[0].databaseId"
} while ($trackerRunId -eq $previousTrackerRunId -and (Get-Date) -lt $trackerRunDeadline)
if (-not $trackerRunId -or $trackerRunId -eq $previousTrackerRunId) { throw "new tracker workflow run was not found" }
gh run watch $trackerRunId --exit-status
```

Verify the resulting commit changes only `data/processed/tracker_ledger.parquet`, preserves 1,359 historical rows and exact baselines, adds valid `record_type=live` rows, deploys successfully, and appears under the live tracker’s 2026/overall selections.

- [ ] **Step 10: Final operations commit if review found documentation corrections**

```powershell
git add README.md CLAUDE.md
git commit -m "docs: finalize 2026 live data operations"
```

Skip this commit when no documentation correction exists.

---

## Plan completion checklist

- [ ] Every production behavior begins with a failing test and observed expected failure.
- [ ] Each task receives a fresh requirements review and code-quality review before the next task.
- [ ] Schedule schema and real-feed smoke checks are separate from deterministic fixture tests.
- [ ] Same-week and future game results cannot enter a target week’s ratings.
- [ ] Live market refresh cannot mutate model predictions or official tracker facts.
- [ ] Missing spread and total markets progress independently.
- [ ] Publication facts remain immutable across reruns, data refreshes, postponements, and deployments.
- [ ] Historical baselines remain exact after ledger schema migration.
- [ ] All generated writes are atomic and digest-aware.
- [ ] Both GitHub workflows share one writer concurrency group and reject moved remotes.
- [ ] Docker explicitly packages features, tracker ledger, and the 2026 schedule.
- [ ] Stage 1 is deployed and observed with tracker writes disabled.
- [ ] Stage 2 is enabled only after explicit review of dry-run records.
