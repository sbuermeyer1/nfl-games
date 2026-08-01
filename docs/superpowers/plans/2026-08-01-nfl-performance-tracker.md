# NFL Performance Tracker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an auditable Ridge-only website tracker for historical ATS and O/U performance, with a separate live-record contract ready for the 2026 data phase.

**Architecture:** Build and grade an immutable game-level ledger offline, package it beside the existing game-feature artifact, and let a focused read-only tracker service calculate season/overall summaries without retraining the model. Serve those summaries through protected FastAPI routes and a separate framework-free tracker page, while preserving the current slate service and its statistical baseline.

**Tech Stack:** Python 3.11+, pandas, NumPy, scikit-learn Ridge through the existing `GameModel`, PyArrow/Parquet, FastAPI, vanilla HTML/CSS/JavaScript, pytest, QuickJS, Ruff, Docker.

## Global Constraints

- Track only the official Ridge estimator with the established default alpha; do not expose GBM in tracker options.
- Use `ridge-v1` as the initial immutable `model_version`.
- Historical tracker scope is the validated 2021–2025 walk-forward window.
- Keep `backtest` and `live` records separate in every query and aggregate.
- A qualified pick has an absolute official edge of at least `2.0` points; exact boundaries are included.
- Spread-edge cohorts are cumulative `5.0+`, `10.0+`, and `15.0+`, not exclusive bands.
- Win rate is `wins / (wins + losses)`; pushes are displayed and excluded from the denominator.
- Positive internal spread values mean home-team margin/favored points, matching `spread_line` throughout the repository.
- Historical official lines are closing lines. Future live official lines are frozen published lines and never change.
- Closing-line metrics retain the originally frozen pick and never change qualification or cohort membership.
- The web process remains read-only: no database, refresh endpoint, scheduled job, or request-time model fitting.
- Existing access-code middleware protects `/tracker` and every `/api/tracker/*` route.
- Do not change the established 2021–2025 baseline: 1,359 games, ATS `660-666-33` (`n=1326`, `0.4977`), and O/U `677-671-11` (`n=1348`, `0.5022`).
- Keep the current one-way dependency direction; `nfl_game.tracking` must not import from `nfl_game.web`.

---

## File structure

### New files

- `src/nfl_game/tracking/__init__.py` — package boundary only.
- `src/nfl_game/tracking/ledger.py` — ledger schema, deterministic grading, CLV derivation, historical conversion, and validation.
- `src/nfl_game/tracking/summary.py` — qualified/all/threshold/season/closing-line aggregation.
- `src/nfl_game/web/tracker_service.py` — tracker selection validation and JSON-safe service boundary.
- `src/nfl_game/web/tracker_page.py` — tracker HTML, CSS, and JavaScript only.
- `scripts/build_tracker.py` — offline Ridge walk-forward ledger builder and acceptance gate.
- `tests/test_tracking_ledger.py` — direction, grade, CLV, schema, and derived-field validation.
- `tests/test_tracking_summary.py` — record arithmetic, thresholds, isolation, and empty states.
- `tests/test_build_tracker.py` — Ridge-only builder, baseline gate, CLI, and output behavior.
- `tests/test_web_tracker_service.py` — tracker service options, selections, live state, and audit rows.
- `tests/test_web_tracker_page.py` — tracker markup, safe rendering, selection behavior, and stale-response handling.
- `data/processed/tracker_ledger.parquet` — reviewed historical `ridge-v1` deployment artifact.

### Modified files

- `src/nfl_game/web/app.py` — navigation, tracker routes, dependency injection, and tracker error mapping.
- `src/nfl_game/web/runtime.py` — require and load both packaged artifacts.
- `scripts/game_app.py` — pass the tracker ledger path at startup.
- `tests/test_webapp.py` — supply a fake tracker service and pin auth/routing integration.
- `tests/test_web_runtime.py` — pin missing/corrupt tracker startup failures and entrypoint arguments.
- `.gitignore` — allow exactly `data/processed/tracker_ledger.parquet`.
- `Dockerfile` — copy the tracker artifact into the image.
- `README.md` — document tracker semantics, routes, artifact rebuild, and release checks.
- `CLAUDE.md` — document tracking architecture, model governance, and immutable baselines.

---

### Task 1: Deterministic ledger grading and validation

**Files:**
- Create: `src/nfl_game/tracking/__init__.py`
- Create: `src/nfl_game/tracking/ledger.py`
- Create: `tests/test_tracking_ledger.py`

**Interfaces:**
- Consumes: walk-forward prediction frames containing `game_id`, `season`, `week`, teams, `model_margin`, `model_total`, `spread_line`, `total_line`, `margin`, and `total_points`.
- Produces: `LEDGER_COLUMNS`, `HISTORICAL_MODEL_VERSION`, `build_backtest_ledger(predictions, model_version) -> pd.DataFrame`, `grade_ledger(facts) -> pd.DataFrame`, and `validate_ledger(ledger) -> None`.

- [ ] **Step 1: Write failing direction, grading, pending, and CLV tests**

Create `tests/test_tracking_ledger.py` with a complete fact-row helper and explicit assertions:

```python
import numpy as np
import pandas as pd
import pytest

from nfl_game.tracking.ledger import grade_ledger, validate_ledger


def facts(*overrides):
    base = {
        "record_type": "live",
        "model_version": "ridge-v1",
        "estimator": "ridge",
        "game_id": "2026_01_AAA_BBB",
        "season": 2026,
        "week": 1,
        "away_team": "AAA",
        "home_team": "BBB",
        "model_margin": 7.0,
        "model_total": 48.0,
        "official_spread_line": 3.0,
        "official_total_line": 44.0,
        "published_spread_line": 3.0,
        "published_total_line": 44.0,
        "closing_spread_line": 5.0,
        "closing_total_line": 46.0,
        "published_at": pd.Timestamp("2026-09-01T12:00:00Z"),
        "kickoff_at": pd.Timestamp("2026-09-06T17:00:00Z"),
        "actual_margin": 6.0,
        "actual_total": 47.0,
    }
    return pd.DataFrame([{**base, **change} for change in overrides])


def test_grade_ledger_pins_direction_grades_and_positive_clv():
    out = grade_ledger(
        facts(
            {},
            {
                "game_id": "2026_01_CCC_DDD",
                "model_margin": -2.0,
                "official_spread_line": 1.0,
                "published_spread_line": 1.0,
                "closing_spread_line": -1.0,
                "actual_margin": -3.0,
                "model_total": 41.0,
                "official_total_line": 44.0,
                "published_total_line": 44.0,
                "closing_total_line": 42.0,
                "actual_total": 40.0,
            },
        )
    ).set_index("game_id")

    home = out.loc["2026_01_AAA_BBB"]
    assert (home.spread_pick, home.spread_grade, home.spread_clv) == ("home", "win", 2.0)
    assert (home.total_pick, home.total_grade, home.total_clv) == ("over", "win", 2.0)

    away = out.loc["2026_01_CCC_DDD"]
    assert (away.spread_pick, away.spread_grade, away.spread_clv) == ("away", "win", 2.0)
    assert (away.total_pick, away.total_grade, away.total_clv) == ("under", "win", 2.0)


def test_push_no_pick_and_pending_are_distinct():
    out = grade_ledger(
        facts(
            {"game_id": "push", "actual_margin": 3.0, "actual_total": 44.0},
            {
                "game_id": "no-pick",
                "model_margin": 3.0,
                "model_total": 44.0,
            },
            {"game_id": "pending", "actual_margin": np.nan, "actual_total": np.nan},
            {
                "game_id": "missing-line",
                "official_spread_line": np.nan,
                "official_total_line": np.nan,
            },
        )
    ).set_index("game_id")

    assert out.loc["push", ["spread_grade", "total_grade"]].tolist() == ["push", "push"]
    assert out.loc["no-pick", ["spread_grade", "total_grade"]].tolist() == [
        "no_pick",
        "no_pick",
    ]
    assert out.loc["pending", ["spread_grade", "total_grade"]].tolist() == [
        "pending",
        "pending",
    ]
    assert out.loc["missing-line", ["spread_grade", "total_grade"]].tolist() == [
        "pending",
        "pending",
    ]


def test_validate_ledger_rejects_duplicates_non_ridge_and_stale_grades():
    valid = grade_ledger(facts({}))
    validate_ledger(valid)

    with pytest.raises(ValueError, match="duplicate"):
        validate_ledger(pd.concat([valid, valid], ignore_index=True))

    non_ridge = valid.copy()
    non_ridge.loc[0, "estimator"] = "gbm"
    with pytest.raises(ValueError, match="ridge"):
        validate_ledger(non_ridge)

    stale = valid.copy()
    stale.loc[0, "spread_grade"] = "loss"
    with pytest.raises(ValueError, match="derived"):
        validate_ledger(stale)
```

- [ ] **Step 2: Run the ledger tests to verify they fail**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_tracking_ledger.py -q
```

Expected: collection fails because `nfl_game.tracking.ledger` does not exist.

- [ ] **Step 3: Implement the schema and grading functions**

Create `src/nfl_game/tracking/__init__.py` as an empty package marker. In
`src/nfl_game/tracking/ledger.py`, define the complete persisted column order and helpers:

```python
HISTORICAL_MODEL_VERSION = "ridge-v1"
OFFICIAL_ESTIMATOR = "ridge"
RECORD_TYPES = frozenset({"backtest", "live"})
GRADE_VALUES = frozenset({"win", "loss", "push", "pending", "no_pick"})
PICK_VALUES = {
    "spread_pick": frozenset({"home", "away"}),
    "total_pick": frozenset({"over", "under"}),
}

LEDGER_COLUMNS = [
    "record_type", "model_version", "estimator", "game_id", "season", "week",
    "away_team", "home_team", "model_margin", "model_total",
    "official_spread_line", "official_total_line",
    "published_spread_line", "published_total_line",
    "closing_spread_line", "closing_total_line", "published_at", "kickoff_at",
    "actual_margin", "actual_total", "spread_pick", "total_pick",
    "spread_edge", "total_edge", "spread_grade", "total_grade",
    "spread_clv", "total_clv", "spread_close_grade", "total_close_grade",
]


def _pick(model, line, high_label, low_label):
    if pd.isna(model) or pd.isna(line) or model == line:
        return pd.NA
    return high_label if model > line else low_label


def _grade(pick, actual, line, high_label):
    if pd.isna(line) or pd.isna(actual):
        return "pending"
    if pd.isna(pick):
        return "no_pick"
    if actual == line:
        return "push"
    high_won = actual > line
    return "win" if high_won == (pick == high_label) else "loss"


def _clv(record_type, pick, published, closing, high_label):
    if record_type != "live" or pd.isna(pick) or pd.isna(published) or pd.isna(closing):
        return np.nan
    movement = closing - published
    return float(movement if pick == high_label else -movement)
```

Implement `grade_ledger(facts)` by copying the input, deriving signed `spread_edge` and
`total_edge`, freezing pick directions from official lines, grading official outcomes, and
calculating CLV. Calculate close grades only for live rows, using the original official pick;
use `pending` when the close is unavailable. Reindex to `LEDGER_COLUMNS` before returning.

Implement `build_backtest_ledger(predictions, model_version=HISTORICAL_MODEL_VERSION)` by
renaming `margin -> actual_margin` and `total_points -> actual_total`, setting both official
and closing lines from `spread_line`/`total_line`, setting published lines/timestamps to null,
setting `record_type="backtest"` and `estimator="ridge"`, then calling `grade_ledger` and
`validate_ledger`.

Implement `validate_ledger` to reject missing columns, empty frames, duplicate
`(record_type, model_version, game_id)` keys, blank identity strings, record types outside
`RECORD_TYPES`, non-Ridge official rows, non-positive/fractional season or week values,
non-finite non-null numeric fields, invalid grades/picks, backtest rows whose official and
closing lines differ, backtest rows with a published timestamp, live rows whose official and
published lines differ, and any derived field that differs from a fresh `grade_ledger` result.
Use exact comparison for labels and
`np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12, equal_nan=True)` for
numeric derived columns.

- [ ] **Step 4: Run ledger tests and existing backtest tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_tracking_ledger.py tests\test_backtest.py -q
```

Expected: all tests pass; existing backtest behavior is unchanged.

- [ ] **Step 5: Commit the ledger domain**

```powershell
git add src/nfl_game/tracking tests/test_tracking_ledger.py
git commit -m "feat: add immutable tracker ledger grading"
```

---

### Task 2: Tracker summaries and cumulative edge cohorts

**Files:**
- Create: `src/nfl_game/tracking/summary.py`
- Create: `tests/test_tracking_summary.py`

**Interfaces:**
- Consumes: validated ledger frames from `grade_ledger`/`build_backtest_ledger`.
- Produces: `QUALIFIED_EDGE`, `SPREAD_EDGE_THRESHOLDS`, `record_summary(grades) -> dict`, `summarize_selection(ledger, record_type, season) -> dict`, and `audit_rows(ledger, record_type, season) -> list[dict]`.

- [ ] **Step 1: Write failing summary tests with exact threshold arithmetic**

Create a test fixture whose spread edges land exactly on 2, 5, 10, and 15 points, then pin
the record math and isolation:

```python
import pandas as pd
import pytest

from nfl_game.tracking.ledger import grade_ledger
from nfl_game.tracking.summary import audit_rows, summarize_selection


def tracker_ledger():
    rows = []
    for index, (record_type, season, edge, actual_margin, total_edge, actual_total) in enumerate(
        [
            ("backtest", 2024, 2.0, 6.0, 2.0, 48.0),
            ("backtest", 2024, 5.0, 1.0, -2.0, 40.0),
            ("backtest", 2025, 10.0, 20.0, 1.0, 48.0),
            ("backtest", 2025, 15.0, 3.0, -5.0, 44.0),
            ("live", 2026, 5.0, 9.0, 3.0, 49.0),
        ]
    ):
        line = 3.0
        total_line = 44.0
        published_spread = line if record_type == "live" else None
        published_total = total_line if record_type == "live" else None
        rows.append(
            {
                "record_type": record_type,
                "model_version": "ridge-v1",
                "estimator": "ridge",
                "game_id": f"game-{index}",
                "season": season,
                "week": 1,
                "away_team": "AAA",
                "home_team": "BBB",
                "model_margin": line + edge,
                "model_total": total_line + total_edge,
                "official_spread_line": line,
                "official_total_line": total_line,
                "published_spread_line": published_spread,
                "published_total_line": published_total,
                "closing_spread_line": line + (1.0 if record_type == "live" else 0.0),
                "closing_total_line": total_line + (1.0 if record_type == "live" else 0.0),
                "published_at": (
                    pd.Timestamp("2026-09-01T12:00:00Z") if record_type == "live" else None
                ),
                "kickoff_at": None,
                "actual_margin": actual_margin,
                "actual_total": actual_total,
            }
        )
    return grade_ledger(pd.DataFrame(rows))


def test_summary_separates_live_and_backtest_and_counts_pushes():
    summary = summarize_selection(tracker_ledger(), "backtest", "all")
    assert summary["available"] is True
    assert summary["qualified"]["spread"] == {
        "wins": 2,
        "losses": 1,
        "pushes": 1,
        "n_graded": 3,
        "win_rate": pytest.approx(2 / 3),
    }
    assert summary["all_predictions"]["total"]["n_graded"] == 3
    assert [row["min_edge"] for row in summary["spread_edges"]] == [5.0, 10.0, 15.0]
    assert [row["record"]["n_graded"] for row in summary["spread_edges"]] == [2, 1, 0]
    assert {row["season"] for row in summary["by_season"]} == {2024, 2025}


def test_live_clv_uses_the_frozen_qualified_cohort():
    summary = summarize_selection(tracker_ledger(), "live", "all")
    assert summary["closing_line"]["spread"] == {
        "average_clv": 1.0,
        "beat_close_rate": 1.0,
        "n_clv": 1,
        "record": {"wins": 1, "losses": 0, "pushes": 0, "n_graded": 1, "win_rate": 1.0},
    }


def test_empty_live_selection_is_unavailable_and_audit_requires_one_season():
    historical_only = tracker_ledger().query("record_type == 'backtest'")
    assert summarize_selection(historical_only, "live", "all") == {
        "available": False,
        "record_type": "live",
        "message": "Live tracking begins with the 2026 season.",
    }
    with pytest.raises(ValueError, match="concrete season"):
        audit_rows(historical_only, "backtest", "all")
```

- [ ] **Step 2: Run summary tests to verify they fail**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_tracking_summary.py -q
```

Expected: collection fails because `nfl_game.tracking.summary` does not exist.

- [ ] **Step 3: Implement records, cohorts, by-season rows, and CLV summaries**

Create `summary.py` with these constants and shapes:

```python
QUALIFIED_EDGE = 2.0
SPREAD_EDGE_THRESHOLDS = (5.0, 10.0, 15.0)
LIVE_UNAVAILABLE_MESSAGE = "Live tracking begins with the 2026 season."


def record_summary(grades: pd.Series) -> dict:
    wins = int((grades == "win").sum())
    losses = int((grades == "loss").sum())
    pushes = int((grades == "push").sum())
    n_graded = wins + losses
    return {
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "n_graded": n_graded,
        "win_rate": wins / n_graded if n_graded else None,
    }
```

Implement a private `_core_summary(selected)` that returns qualified spread/total records,
all non-zero-edge records, cumulative spread-edge records, and qualified-cohort live closing
metrics. `summarize_selection(ledger, record_type, season)` must validate `record_type`, parse
`"all"` or an integer season, return the explicit live-unavailable object for no live rows,
and include `by_season` only for an overall selection. Do not recursively include
`by_season` inside each season row.

`audit_rows` must reject `"all"`, filter exactly one record type/season, sort by week and
game ID, select only the browser audit columns, convert non-finite/missing pandas values to
`None`, and return `list[dict]`.

- [ ] **Step 4: Run tracker-domain tests**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_tracking_ledger.py tests\test_tracking_summary.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit summary behavior**

```powershell
git add src/nfl_game/tracking/summary.py tests/test_tracking_summary.py
git commit -m "feat: summarize tracker records and edge cohorts"
```

---

### Task 3: Offline historical ledger builder and acceptance gate

**Files:**
- Create: `scripts/build_tracker.py`
- Create: `tests/test_build_tracker.py`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: `game_features.parquet`, `walk_forward`, `build_backtest_ledger`, and summary functions.
- Produces: `build_historical_ledger(features, test_seasons, model_version, expected_baseline) -> pd.DataFrame`, `acceptance_metrics(ledger) -> dict`, `assert_acceptance_baseline(ledger, expected) -> None`, and CLI output at `data/processed/tracker_ledger.parquet`.

- [ ] **Step 1: Write failing builder and baseline-gate tests**

Create `tests/test_build_tracker.py`:

```python
import pandas as pd
import pytest

from scripts import build_tracker


def predictions():
    return pd.DataFrame(
        {
            "game_id": ["2025_01_AAA_BBB", "2025_01_CCC_DDD"],
            "season": [2025, 2025],
            "week": [1, 1],
            "away_team": ["AAA", "CCC"],
            "home_team": ["BBB", "DDD"],
            "model_margin": [7.0, -2.0],
            "model_total": [48.0, 40.0],
            "spread_line": [3.0, 1.0],
            "total_line": [44.0, 44.0],
            "margin": [8.0, -3.0],
            "total_points": [50.0, 38.0],
        }
    )


def test_builder_is_ridge_only_and_forwards_the_requested_seasons(monkeypatch):
    calls = []

    def fake_walk_forward(features, test_seasons, estimator, alpha):
        calls.append((features.copy(), test_seasons, estimator, alpha))
        return predictions()

    monkeypatch.setattr(build_tracker, "walk_forward", fake_walk_forward)
    expected = {
        "games": 2,
        "ats_n": 2,
        "ats_hit_rate": 1.0,
        "ou_n": 2,
        "ou_hit_rate": 1.0,
    }
    features = pd.DataFrame({"sentinel": [1]})
    ledger = build_tracker.build_historical_ledger(
        features,
        test_seasons=[2025],
        model_version="ridge-v1",
        expected_baseline=expected,
    )

    assert calls[0][1:] == ([2025], "ridge", 1.0)
    assert set(ledger["record_type"]) == {"backtest"}
    assert set(ledger["model_version"]) == {"ridge-v1"}


def test_acceptance_gate_rejects_any_corpus_or_hit_rate_drift():
    ledger = build_tracker.build_backtest_ledger(predictions())
    wrong = {
        "games": 3,
        "ats_n": 2,
        "ats_hit_rate": 1.0,
        "ou_n": 2,
        "ou_hit_rate": 1.0,
    }
    with pytest.raises(RuntimeError, match="acceptance baseline changed"):
        build_tracker.assert_acceptance_baseline(ledger, wrong)


def test_cli_writes_the_validated_ledger(tmp_path, monkeypatch):
    features_path = tmp_path / "features.parquet"
    output_path = tmp_path / "tracker.parquet"
    pd.DataFrame({"sentinel": [1]}).to_parquet(features_path)
    monkeypatch.setattr(build_tracker, "build_historical_ledger", lambda *args, **kwargs: build_tracker.build_backtest_ledger(predictions()))

    build_tracker.main(["--features", str(features_path), "--output", str(output_path)])

    written = pd.read_parquet(output_path)
    assert written["game_id"].tolist() == predictions()["game_id"].tolist()
```

- [ ] **Step 2: Run the builder tests to verify they fail**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_build_tracker.py -q
```

Expected: import fails because `scripts.build_tracker` does not exist.

- [ ] **Step 3: Implement the fixed Ridge builder and CLI**

Use these production constants and signatures:

```python
HISTORICAL_SEASONS = tuple(range(2021, 2026))
EXPECTED_BASELINE = {
    "games": 1359,
    "ats_n": 1326,
    "ats_hit_rate": 0.497737556561086,
    "ou_n": 1348,
    "ou_hit_rate": 0.5022255192878339,
}


def build_historical_ledger(
    features: pd.DataFrame,
    test_seasons=HISTORICAL_SEASONS,
    model_version=HISTORICAL_MODEL_VERSION,
    expected_baseline=EXPECTED_BASELINE,
) -> pd.DataFrame:
    predictions = walk_forward(
        features,
        list(test_seasons),
        estimator="ridge",
        alpha=DEFAULT_ALPHA,
    )
    ledger = build_backtest_ledger(predictions, model_version=model_version)
    assert_acceptance_baseline(ledger, expected_baseline)
    return ledger
```

`acceptance_metrics` must calculate all-prediction official records from the ledger and
return exactly the five `EXPECTED_BASELINE` keys. Compare integer values exactly and floats
with `math.isclose(rel_tol=0, abs_tol=5e-13)`. The CLI accepts `--features`, `--output`, and
`--model-version` only; it does not accept an estimator or threshold override. Default paths
are the two packaged artifacts under `PROCESSED_DIR`. Write the Parquet file only after the
full ledger and acceptance checks pass, then print row count and ATS/O/U record summaries.

Add this narrow ignore exception after the existing game-feature exception:

```gitignore
!data/processed/tracker_ledger.parquet
```

- [ ] **Step 4: Run builder and tracker-domain tests**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_build_tracker.py tests\test_tracking_ledger.py tests\test_tracking_summary.py -q
```

Expected: all tests pass and the CLI test writes only inside its temporary directory.

- [ ] **Step 5: Commit the offline builder**

```powershell
git add scripts/build_tracker.py tests/test_build_tracker.py .gitignore
git commit -m "feat: build reviewed historical tracker ledger"
```

---

### Task 4: Read-only tracker web service

**Files:**
- Create: `src/nfl_game/web/tracker_service.py`
- Create: `tests/test_web_tracker_service.py`

**Interfaces:**
- Consumes: validated ledger artifact plus `summarize_selection` and `audit_rows`.
- Produces: `TrackerInputError`, `TrackerService.from_parquet(path)`, `TrackerService.options()`, `TrackerService.summary(record_type, season)`, and `TrackerService.records(record_type, season)`.

- [ ] **Step 1: Write failing service tests**

```python
import pandas as pd
import pytest

from nfl_game.tracking.ledger import grade_ledger
from nfl_game.web.tracker_service import TrackerInputError, TrackerService


def service_ledger():
    facts = pd.DataFrame(
        [
            {
                "record_type": "backtest",
                "model_version": "ridge-v1",
                "estimator": "ridge",
                "game_id": f"{season}_01_AAA_BBB",
                "season": season,
                "week": 1,
                "away_team": "AAA",
                "home_team": "BBB",
                "model_margin": 7.0,
                "model_total": 48.0,
                "official_spread_line": 3.0,
                "official_total_line": 44.0,
                "published_spread_line": None,
                "published_total_line": None,
                "closing_spread_line": 3.0,
                "closing_total_line": 44.0,
                "published_at": None,
                "kickoff_at": None,
                "actual_margin": 8.0,
                "actual_total": 50.0,
            }
            for season in (2024, 2025)
        ]
    )
    return grade_ledger(facts)


def test_options_are_fixed_to_the_official_model_and_thresholds():
    options = TrackerService(service_ledger()).options()
    assert options == {
        "record_types": ["backtest", "live"],
        "historical_seasons": [2024, 2025],
        "default_record_type": "backtest",
        "default_season": "all",
        "model_version": "ridge-v1",
        "qualified_edge": 2.0,
        "spread_edge_thresholds": [5.0, 10.0, 15.0],
        "live_available": False,
    }


def test_summary_and_records_validate_selections():
    service = TrackerService(service_ledger())
    assert service.summary("backtest", "all")["available"] is True
    assert service.summary("live", "all")["available"] is False
    assert len(service.records("backtest", 2024)) == 1

    with pytest.raises(TrackerInputError, match="record type"):
        service.summary("research", "all")
    with pytest.raises(TrackerInputError, match="season 2023"):
        service.summary("backtest", "2023")
    with pytest.raises(TrackerInputError, match="concrete season"):
        service.records("backtest", "all")
```

- [ ] **Step 2: Run service tests to verify they fail**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_web_tracker_service.py -q
```

Expected: collection fails because `nfl_game.web.tracker_service` does not exist.

- [ ] **Step 3: Implement the service boundary**

Use this class boundary:

```python
class TrackerInputError(ValueError):
    """A tracker record type or season selection is invalid."""


class TrackerService:
    def __init__(self, ledger: pd.DataFrame):
        validate_ledger(ledger)
        self._ledger = ledger.copy()
        versions = sorted(self._ledger["model_version"].unique())
        if versions != [HISTORICAL_MODEL_VERSION]:
            raise ValueError(f"official tracker requires only {HISTORICAL_MODEL_VERSION!r}")

    @classmethod
    def from_parquet(cls, path: str | Path) -> "TrackerService":
        return cls(pd.read_parquet(path))
```

Implement the exact options object from the test. Convert string seasons to integers only
after rejecting blank, fractional, unavailable, or non-numeric values. `summary` translates
domain `ValueError` into `TrackerInputError`; `records` requires one concrete season. Convert
pandas missing values to JSON `None` before returning records.

- [ ] **Step 4: Run service and tracker-domain tests**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_web_tracker_service.py tests\test_tracking_summary.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit the tracker service**

```powershell
git add src/nfl_game/web/tracker_service.py tests/test_web_tracker_service.py
git commit -m "feat: serve tracker summaries from immutable data"
```

---

### Task 5: Framework-free tracker page and browser behavior

**Files:**
- Create: `src/nfl_game/web/tracker_page.py`
- Create: `tests/test_web_tracker_page.py`

**Interfaces:**
- Consumes: `/api/tracker/options`, `/api/tracker/summary`, and `/api/tracker/games` response shapes from Tasks 4 and 6.
- Produces: `TRACKER_PAGE`, a standalone authenticated page with historical/live tabs, season selection, cards, threshold and season tables, audit table, safe formatting, and stale-request invalidation.

- [ ] **Step 1: Write failing page structure and safe-rendering tests**

Create tests that extract the tracker script and run it in QuickJS with a small fake DOM:

```python
import json
import re

import quickjs

from nfl_game.web.tracker_page import TRACKER_PAGE

SCRIPT = re.compile(r"<script>(.*?)</script>", re.DOTALL)


def test_tracker_page_has_required_navigation_controls_and_regions():
    assert 'href="/"' in TRACKER_PAGE
    for element_id in (
        "historical-tab",
        "live-tab",
        "tracker-season",
        "tracker-message",
        "qualified-cards",
        "all-records",
        "spread-edges",
        "season-breakdown",
        "audit-games",
        "closing-line",
    ):
        assert f'id="{element_id}"' in TRACKER_PAGE
    assert "52.4%" in TRACKER_PAGE
    assert "not betting advice" in TRACKER_PAGE


def test_tracker_script_uses_text_content_for_matchup_values():
    script = SCRIPT.search(TRACKER_PAGE)
    assert script is not None
    assert ".innerHTML" not in script.group(1)
    assert ".textContent" in script.group(1)


def test_tracker_script_declares_separate_stale_request_tokens():
    script = SCRIPT.search(TRACKER_PAGE)
    assert script is not None
    source = script.group(1)
    assert "latestSummaryRequest" in source
    assert "latestGamesRequest" in source
    assert "request !== latestSummaryRequest" in source
    assert "request !== latestGamesRequest" in source
```

Add a `tracker_state(responses, actions)` QuickJS helper that implements the exact page IDs
above, `document.createElement`, `textContent`, `appendChild`, `replaceChildren`, event
listeners, and a deferred `fetch`. Add executable tests that:

1. resolve options and overall historical summary, then assert qualified card text, the
   5+/10+/15+ rows, and the season table;
2. select a concrete season and assert `/api/tracker/games` is requested and matchup strings
   are placed through `textContent`;
3. switch to live and assert the explicit 2026 unavailable message with no zero-percent cards;
4. start two season requests, resolve the newer one first and the older one last, and assert
   the older response never replaces the current summary or audit table;
5. return `401` and assert `window.location == "/login"`.

- [ ] **Step 2: Run page tests to verify they fail**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_web_tracker_page.py -q
```

Expected: collection fails because `nfl_game.web.tracker_page` does not exist.

- [ ] **Step 3: Implement the page markup and request-state machine**

Create `TRACKER_PAGE` with the required IDs and these client-side invariants:

```javascript
let activeRecordType = 'backtest';
let latestSummaryRequest = 0;
let latestGamesRequest = 0;

function trackerQuery() {
  return new URLSearchParams({
    record_type: activeRecordType,
    season: activeRecordType === 'live' ? 'all' : season.value,
  }).toString();
}

function invalidateTracker() {
  latestSummaryRequest += 1;
  latestGamesRequest += 1;
  qualifiedCards.replaceChildren();
  allRecords.replaceChildren();
  spreadEdges.replaceChildren();
  seasonBreakdown.replaceChildren();
  auditGames.replaceChildren();
  closingLine.replaceChildren();
}
```

Use DOM creation plus `textContent` for every server value. Render record text as
`W-L-P · XX.X% · n=N`, using `n/a` when `win_rate` is null. On historical overall, render
the season table and do not request audit rows. On a concrete historical season, request and
render audit rows. On live unavailable, render only the service message. On future live data,
render the qualified cards plus spread/total CLV, beat-close rate, `n_clv`, and close record.

Every async request captures its counter and query. Before any DOM change and in `finally`,
require both the counter and current query to match. Increment both counters on record-type or
season changes so late historical responses cannot overwrite live state and late audit rows
cannot appear under another season.

Use compact responsive cards and horizontally scrolling tables; do not introduce a frontend
framework or build step.

- [ ] **Step 4: Run tracker page tests**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_web_tracker_page.py -q
```

Expected: all page and QuickJS tests pass.

- [ ] **Step 5: Commit the tracker page**

```powershell
git add src/nfl_game/web/tracker_page.py tests/test_web_tracker_page.py
git commit -m "feat: add performance tracker page"
```

---

### Task 6: FastAPI routes, authentication, and fail-closed runtime wiring

**Files:**
- Modify: `src/nfl_game/web/app.py`
- Modify: `src/nfl_game/web/runtime.py`
- Modify: `scripts/game_app.py`
- Modify: `tests/test_webapp.py`
- Modify: `tests/test_web_runtime.py`

**Interfaces:**
- Consumes: `TrackerService`, `TrackerInputError`, and `TRACKER_PAGE`.
- Produces: protected `/tracker`, `/api/tracker/options`, `/api/tracker/summary`, and `/api/tracker/games`; runtime startup that requires both Parquet artifacts.

- [ ] **Step 1: Write failing tracker route and runtime tests**

Add a minimal `FakeTrackerService` to `tests/test_webapp.py` and pass it from the existing
`client()` helper:

```python
class FakeTrackerService:
    def __init__(self):
        self.calls = []

    def options(self):
        self.calls.append(("options",))
        return {"model_version": "ridge-v1", "historical_seasons": [2024, 2025]}

    def summary(self, record_type, season):
        self.calls.append(("summary", record_type, season))
        return {"available": True, "record_type": record_type, "season": season}

    def records(self, record_type, season):
        self.calls.append(("records", record_type, season))
        return [{"game_id": "2025_01_AAA_BBB"}]


def client(service=None, tracker_service=None, access_code=None):
    return TestClient(
        create_app(
            service or FakeService(),
            tracker_service or FakeTrackerService(),
            access_code=access_code,
        ),
        base_url="https://testserver",
        follow_redirects=False,
    )
```

Add route assertions:

```python
def test_tracker_routes_forward_exact_selections():
    tracker = FakeTrackerService()
    http_client = client(tracker_service=tracker)

    assert http_client.get("/tracker").status_code == 200
    assert http_client.get("/api/tracker/options").status_code == 200
    assert http_client.get(
        "/api/tracker/summary", params={"record_type": "backtest", "season": "all"}
    ).json()["available"] is True
    assert http_client.get(
        "/api/tracker/games", params={"record_type": "backtest", "season": 2025}
    ).json() == {"games": [{"game_id": "2025_01_AAA_BBB"}]}
    assert tracker.calls == [
        ("options",),
        ("summary", "backtest", "all"),
        ("records", "backtest", 2025),
    ]


def test_auth_protects_tracker_page_and_api():
    http_client = client(access_code="letmein")
    assert http_client.get("/tracker").status_code == 303
    assert http_client.get("/api/tracker/options").json() == {"error": "session expired"}
```

In `tests/test_web_runtime.py`, update every `load_app` call to provide a tracker path and add
tests for a missing and corrupt tracker artifact. Update the entrypoint spy to accept
`(config, dataset_path, tracker_path)` and assert the third argument is
`PROCESSED_DIR / "tracker_ledger.parquet"`.

- [ ] **Step 2: Run route/runtime tests to verify they fail**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_webapp.py tests\test_web_runtime.py -q
```

Expected: failures show that `create_app` and `load_app` do not yet accept the tracker service/path and tracker routes do not exist.

- [ ] **Step 3: Wire the app factory and tracker routes**

Change the factory signature to:

```python
def create_app(
    service: SlateService,
    tracker_service: TrackerService,
    access_code: str | None,
) -> FastAPI:
```

Register `TrackerInputError` as a client-safe `422`, serve `TRACKER_PAGE` at `/tracker`, and
add these exact route calls:

```python
@app.get("/api/tracker/options")
def tracker_options():
    return tracker_service.options()


@app.get("/api/tracker/summary")
def tracker_summary(record_type: str = "backtest", season: str = "all"):
    return tracker_service.summary(record_type, season)


@app.get("/api/tracker/games")
def tracker_games(season: int, record_type: str = "backtest"):
    return {"games": tracker_service.records(record_type, season)}
```

Add a `Performance tracker` link to the existing slate page. Update every direct
`create_app(FakeService(), FakeTrackerService(), access_code=None)` pattern in
`tests/test_webapp.py`, with each test's existing service and access-code values preserved.
Do not add tracker routes to `PUBLIC_PATHS`; the existing middleware will redirect
the page and return `401` for its API.

- [ ] **Step 4: Require the tracker artifact at runtime**

Change runtime loading to:

```python
def load_app(
    config: RuntimeConfig,
    dataset_path: str | Path,
    tracker_path: str | Path,
):
    dataset = Path(dataset_path)
    tracker = Path(tracker_path)
    if not dataset.is_file():
        raise RuntimeConfigError(f"packaged dataset not found: {dataset}")
    if not tracker.is_file():
        raise RuntimeConfigError(f"packaged tracker ledger not found: {tracker}")
    try:
        slate_service = SlateService.from_parquet(dataset)
    except Exception as exc:
        raise RuntimeConfigError(f"cannot load packaged dataset {dataset}: {exc}") from exc
    try:
        tracker_service = TrackerService.from_parquet(tracker)
    except Exception as exc:
        raise RuntimeConfigError(f"cannot load packaged tracker ledger {tracker}: {exc}") from exc
    return create_app(slate_service, tracker_service, access_code=config.access_code)
```

Pass `PROCESSED_DIR / "tracker_ledger.parquet"` from `scripts/game_app.py`.

- [ ] **Step 5: Run all web tests**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_webapp.py tests\test_web_runtime.py tests\test_web_tracker_service.py tests\test_web_tracker_page.py tests\test_web_auth.py tests\test_web_login.py -q
```

Expected: all tests pass, including the existing slate browser state-machine tests.

- [ ] **Step 6: Commit web and runtime integration**

```powershell
git add src/nfl_game/web/app.py src/nfl_game/web/runtime.py scripts/game_app.py tests/test_webapp.py tests/test_web_runtime.py
git commit -m "feat: wire protected tracker routes and runtime"
```

---

### Task 7: Generate and verify the reviewed historical artifact

**Files:**
- Create: `data/processed/tracker_ledger.parquet`
- Modify: `Dockerfile`

**Interfaces:**
- Consumes: the checked-in `game_features.parquet` and completed builder.
- Produces: the exact packaged `ridge-v1` 2021–2025 ledger used by the website and container.

- [ ] **Step 1: Run the full historical build**

```powershell
.\.venv\Scripts\python.exe scripts\build_tracker.py
```

Expected: it writes 1,359 rows to `data/processed/tracker_ledger.parquet` and prints the
accepted all-prediction records without a baseline error.

- [ ] **Step 2: Inspect the new tracker summaries through production code**

```powershell
.\.venv\Scripts\python.exe -c "import pandas as pd; from nfl_game.tracking.summary import summarize_selection; d=pd.read_parquet('data/processed/tracker_ledger.parquet'); s=summarize_selection(d,'backtest','all'); print(s['qualified']); print(s['all_predictions']); print(s['spread_edges'])"
```

Expected current `ridge-v1` results:

```text
Qualified ATS 2+: 336-371-16, n=707, win rate 0.475248
Qualified O/U 2+: 396-407-6, n=803, win rate 0.493151
All ATS:          660-666-33, n=1326, win rate 0.497738
All O/U:          677-671-11, n=1348, win rate 0.502226
Spread 5+:        102-102-4, n=204, win rate 0.500000
Spread 10+:       9-11-0, n=20, win rate 0.450000
Spread 15+:       1-1-0, n=2, win rate 0.500000
```

If any value differs, stop and inspect the ledger/grading or upstream artifact; do not update
the expected numbers to make the check pass.

- [ ] **Step 3: Run artifact-backed service and baseline tests**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_build_tracker.py tests\test_web_tracker_service.py tests\test_backtest.py -q
.\.venv\Scripts\python.exe scripts\backtest.py --test-seasons 2021-2025
```

Expected: tests pass and the printed existing statistical baseline is unchanged.

- [ ] **Step 4: Commit the reviewed artifact**

Add this Docker copy directly after the existing feature artifact copy:

```dockerfile
COPY data/processed/tracker_ledger.parquet ./data/processed/tracker_ledger.parquet
```

```powershell
git add data/processed/tracker_ledger.parquet Dockerfile
git commit -m "data: package ridge v1 historical tracker ledger"
```

---

### Task 8: Operational documentation and full release verification

**Files:**
- Modify: `README.md`
- Modify: `CLAUDE.md`

**Interfaces:**
- Consumes: completed tracker behavior and commands.
- Produces: operator-facing rebuild/deployment instructions and maintainer invariants.

- [ ] **Step 1: Update README operations and API documentation**

Add a `Performance tracker` subsection that states:

- `/tracker` separates historical walk-forward backtests from live published picks;
- historical records cover Ridge `ridge-v1`, 2021–2025, against closing lines;
- qualified picks use 2+ points and spread groups are cumulative 5+/10+/15+;
- pushes do not enter win-rate denominators;
- the live section starts in 2026 and remains unavailable until the separate live workflow is built;
- future official live grades use frozen published lines, while CLV and the close record are secondary;
- rebuild command is `.\.venv\Scripts\python.exe scripts\build_tracker.py` after rebuilding features;
- both Parquet artifacts must be reviewed, committed, and deployed together when upstream results change.

Document all four tracker routes beside the existing slate routes and add the tracker artifact
to missing-artifact troubleshooting and Docker release verification.

- [ ] **Step 2: Update maintainer architecture and invariants**

In `CLAUDE.md`, add `tracking` after `market` in the one-way flow and document:

```text
data -> ratings -> model -> market -> tracking -> web
```

State that tracker artifacts are offline/read-only, only Ridge `ridge-v1` is official,
historical/live records must never aggregate together, thresholds are fixed at 2 and
5/10/15 cumulative, and model-version changes must not rewrite published live history. Add
the exact tracker acceptance records from Task 7 beside the existing model baseline.

- [ ] **Step 3: Run formatting, complete tests, and regression gates**

```powershell
.\.venv\Scripts\python.exe -m ruff format --check src tests scripts
.\.venv\Scripts\python.exe -m ruff check src tests scripts
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe scripts\backtest.py --test-seasons 2021-2025
git diff --check
git status --short
```

Expected: formatting, Ruff, and every test pass; the backtest prints the unchanged accepted
baseline; `git diff --check` is silent; status lists only the two documentation edits.

- [ ] **Step 4: Run the packaged application smoke**

Run locally with explicit loopback-only no-auth mode:

```powershell
.\.venv\Scripts\python.exe scripts\game_app.py --no-auth
```

From another terminal, verify:

```powershell
curl.exe -s -o NUL -w "%{http_code}" http://127.0.0.1:8000/health
curl.exe -s -o NUL -w "%{http_code}" http://127.0.0.1:8000/tracker
curl.exe -s -o NUL -w "%{http_code}" "http://127.0.0.1:8000/api/tracker/summary?record_type=backtest&season=all"
```

Expected: `200`, `200`, `200`. Stop the local server cleanly after the smoke.

- [ ] **Step 5: Run the existing container gate when Docker is available**

```powershell
$acceptedCommit = git rev-parse HEAD
docker build --tag "ashburn-nfl:$acceptedCommit" .
docker run --rm --name ashburn-nfl-missing-code "ashburn-nfl:$acceptedCommit"
```

Expected: the build includes both Parquet files and the no-code container exits nonzero with
the established `ACCESS_CODE` error. Then run the protected-port smoke from the README and
verify `/tracker` redirects to login and `/api/tracker/options` returns `401` without a
session. If Docker is unavailable, record that as an unpassed release gate instead of claiming
container verification.

- [ ] **Step 6: Commit documentation and final verified state**

```powershell
git add README.md CLAUDE.md
git commit -m "docs: document NFL tracker operations"
git status --short --branch
```

Expected: the worktree is clean and the branch contains the tracker code, reviewed artifact,
tests, and documentation as separate reviewable commits.

