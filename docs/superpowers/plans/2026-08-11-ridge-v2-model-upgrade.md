# Ridge v2 Model Upgrade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and evaluate a free-data, market-independent Ridge-v2 challenger with separate margin and total feature sets, then promote it only if every approved evidence gate passes.

**Architecture:** Preserve Ridge v1 unchanged while adding source-specific caches, as-of rating modules, a union Ridge-v2 feature artifact, target-specific Ridge pipelines, and a nested walk-forward experiment layer. Research artifacts remain separate from production; website, tracker, and deployment work is conditional on an explicit promotion approval after the complete report.

**Tech Stack:** Python 3.11+, pandas, NumPy, SciPy, scikit-learn Ridge/logistic regression, nflreadpy/nflverse, PyArrow, FastAPI, pytest, Ruff, GitHub Actions, Docker.

## Global Constraints

- Ridge is the only Ridge-v2 estimator. Do not add boosting, neural nets, stacking, or market-derived inputs.
- `spread_line`, `total_line`, moneylines, and market probabilities must never appear in either Ridge-v2 feature list.
- Ridge v1 remains byte/metric compatible: 1,359 games; margin MAE 10.274; total MAE 10.684; ATS 660-666-33; O/U 677-671-11; margin model coefficient -0.0218.
- Margin and total use separate feature schemas and independently selected Ridge penalties.
- Use only free public data. nflverse injury data after 2024 is unavailable and cannot be a production dependency.
- Outer promotion seasons are exactly 2021–2025. Generate 2019–2020 out-of-sample predictions only as calibration seeds.
- Ridge alphas are exactly `0.1`, `1.0`, `10.0`, and `100.0`.
- Rating half-life pairs are exactly `(4, 16)`, `(8, 24)`, and `(12, 32)` games.
- Prior-season weights are exactly `0.4`, `0.6`, and `0.8`.
- A selectable candidate needs at least two inner validation seasons and 400 inner validation games.
- Choose the lowest-numbered candidate within 0.05 MAE points of the best eligible candidate; serialize configurations for deterministic tie-breaking.
- A production block needs at least 90% team-week numeric coverage in every included season before imputation.
- Roster/depth-chart snapshots used for a new prediction may be at most 72 hours old.
- Performance feeds must cover every game whose result was available at least 48 hours before the prediction cutoff.
- Official predictions freeze 24 hours before kickoff.
- Default CLI execution is dry-run. Only explicit `--write` modes may atomically replace Ridge-v2 artifacts.
- Do not alter `data/processed/game_features.parquet` or `data/processed/tracker_ledger.parquet` while building the challenger.
- Keep these Ridge-v2 artifacts separate:
  `game_features_ridge_v2.parquet`, `ridge_v2_manifest.json`,
  `ridge_v2_outer_predictions.parquet`, `ridge_v2_evaluation.json`,
  `ridge_v2_ablation.parquet`, `ridge_v2_calibration.json`, and—only after promotion—
  `tracker_ledger_ridge_v2.parquet`.
- Every data transformation must have an as-of leak test. Every task ends with focused tests, the full suite, Ruff, and a focused commit.
- Execute in an isolated worktree created with `superpowers:using-git-worktrees`. Create a fresh untracked virtual environment there if the inherited `.venv` interpreter is stale.

## File Structure

New focused modules:

```text
src/nfl_game/
  data/
    source_manifest.py       source schema, coverage, freshness, and manifest records
  ratings/
    v2_team.py               situational team-game aggregation and short/long ratings
    qb.py                    as-of quarterback ratings and expected-starter context
    style.py                 pace, pass tendency, turnover, field-position, special teams
    personnel.py             roster/depth/snap continuity
    pfr.py                   team-week PFR advanced aggregates
  model/
    v2_config.py             candidate IDs, tuning grid, manifests, artifact paths
    v2_features.py           target-specific game features and candidate manifests
    v2.py                    separate margin/total Ridge pipelines
  experiments/
    __init__.py
    v2_selection.py          nested walk-forward selection and prediction
    v2_evaluation.py         calibration, bootstrap intervals, gates, reports
  pipeline/
    build_v2.py              historical Ridge-v2 artifact builder
    refresh_v2_2026.py       conditional post-promotion 2026 shadow/production refresh
  web/
    v2_service.py            conditional promoted Ridge-v2 slate service
scripts/
  build_v2_dataset.py
  backtest_v2.py
  refresh_v2_2026.py
  build_v2_tracker.py
```

Existing modules remain the Ridge-v1 source of truth. New modules may import stable v1 helpers,
but v1 model behavior and artifacts must not be routed through v2 code.

---

### Task 1: Freeze Ridge-v1 and define Ridge-v2 configuration contracts

**Files:**
- Create: `src/nfl_game/model/v2_config.py`
- Modify: `src/nfl_game/paths.py`
- Create: `tests/test_v2_config.py`
- Modify: `tests/test_backtest.py`

**Interfaces:**
- Produces: `CandidateId`, `TargetConfig`, `V2ModelConfig`, `FeatureManifest`,
  `target_tuning_grid()`, `V2_*_PATH` constants.
- Preserves: the exact existing Ridge-v1 2021–2025 metrics.

- [ ] **Step 1: Add a failing exact Ridge-v1 regression test**

Add a real-artifact test that loads `data/processed/game_features.parquet`, runs the existing
`walk_forward(features, list(range(2021, 2026)), estimator="ridge", alpha=1.0)`, and asserts the
exact documented values within `5e-4` for displayed MAEs and `5e-7` for hit rates.

```python
def test_ridge_v1_real_artifact_remains_frozen():
    features = pd.read_parquet(PROCESSED_DIR / "game_features.parquet")
    preds = walk_forward(features, list(range(2021, 2026)), estimator="ridge", alpha=1.0)
    metrics = evaluate(preds)
    assert metrics["n_games"] == 1359
    assert metrics["margin_mae"] == pytest.approx(10.274, abs=5e-4)
    assert metrics["total_mae"] == pytest.approx(10.684, abs=5e-4)
    assert metrics["ats_hit_rate"] == pytest.approx(0.4977375566, abs=5e-7)
    assert metrics["ou_hit_rate"] == pytest.approx(0.5022255193, abs=5e-7)
```

- [ ] **Step 2: Write failing configuration tests**

Assert the exact candidates, grids, deterministic key, market-column rejection, JSON
round-trip, and artifact filenames.

```python
def test_target_grid_is_fixed_and_deterministic():
    grid = target_tuning_grid("C2")
    assert len(grid) == 4 * 3 * 3
    assert grid[0].candidate == "C2"
    assert {c.alpha for c in grid} == {0.1, 1.0, 10.0, 100.0}
    assert {(c.short_halflife, c.long_halflife) for c in grid} == {
        (4, 16), (8, 24), (12, 32)
    }
    assert {c.prior_season_weight for c in grid} == {0.4, 0.6, 0.8}

def test_manifest_rejects_market_features():
    with pytest.raises(ValueError, match="market column"):
        FeatureManifest(
            version="ridge-v2-test",
            margin_by_candidate={"C1": ("spread_line",)},
            total_by_candidate={"C1": ("pace_sum",)},
            sources={},
            constants={},
        )
```

- [ ] **Step 3: Run focused tests and confirm the new imports fail**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_v2_config.py tests/test_backtest.py::test_ridge_v1_real_artifact_remains_frozen -v
```

Expected: Ridge-v1 test passes; v2 tests fail because `nfl_game.model.v2_config` is absent.

- [ ] **Step 4: Create the configuration types and path constants**

Use immutable dataclasses and stable JSON-compatible dictionaries:

```python
CANDIDATES = ("C0", "C1", "C2", "C3", "C4", "C5")
ALPHAS = (0.1, 1.0, 10.0, 100.0)
RATING_WINDOWS = ((4, 16), (8, 24), (12, 32))
PRIOR_SEASON_WEIGHTS = (0.4, 0.6, 0.8)
MARKET_COLUMNS = frozenset({"spread_line", "total_line", "away_moneyline", "home_moneyline"})

@dataclass(frozen=True, order=True)
class TargetConfig:
    candidate: str
    alpha: float
    short_halflife: int
    long_halflife: int
    prior_season_weight: float

    def key(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

@dataclass(frozen=True)
class V2ModelConfig:
    margin: TargetConfig
    total: TargetConfig

@dataclass(frozen=True)
class FeatureManifest:
    version: str
    margin_by_candidate: dict[str, tuple[str, ...]]
    total_by_candidate: dict[str, tuple[str, ...]]
    sources: dict[str, str]
    constants: dict[str, object]

    def __post_init__(self) -> None:
        for target, mapping in (
            ("margin", self.margin_by_candidate),
            ("total", self.total_by_candidate),
        ):
            for candidate, columns in mapping.items():
                overlap = MARKET_COLUMNS.intersection(columns)
                if overlap:
                    raise ValueError(
                        f"market column in {target}/{candidate}: {sorted(overlap)}"
                    )

    def columns(self, target: str, candidate: str) -> tuple[str, ...]:
        mappings = {
            "margin": self.margin_by_candidate,
            "total": self.total_by_candidate,
        }
        return tuple(mappings[target][candidate])

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "margin_by_candidate": {
                key: list(value) for key, value in self.margin_by_candidate.items()
            },
            "total_by_candidate": {
                key: list(value) for key, value in self.total_by_candidate.items()
            },
            "sources": dict(self.sources),
            "constants": dict(self.constants),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "FeatureManifest":
        return cls(
            version=str(payload["version"]),
            margin_by_candidate={
                key: tuple(value) for key, value in payload["margin_by_candidate"].items()
            },
            total_by_candidate={
                key: tuple(value) for key, value in payload["total_by_candidate"].items()
            },
            sources=dict(payload["sources"]),
            constants=dict(payload["constants"]),
        )
```

Add exact `PROCESSED_DIR / <approved filename>` constants in `paths.py`.

- [ ] **Step 5: Run focused and full verification**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_v2_config.py tests/test_backtest.py -v
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
```

Expected: all tests pass; Ridge-v1 metrics remain exact; Ruff is clean.

- [ ] **Step 6: Commit**

```powershell
git add src/nfl_game/model/v2_config.py src/nfl_game/paths.py tests/test_v2_config.py tests/test_backtest.py
git commit -m "test: freeze ridge v1 and define v2 contracts"
```

---

### Task 2: Add source manifests, coverage checks, and atomic JSON writes

**Files:**
- Create: `src/nfl_game/data/source_manifest.py`
- Create: `tests/test_source_manifest.py`

**Interfaces:**
- Produces: `SourceSnapshot`, `schema_fingerprint()`, `numeric_coverage()`,
  `require_coverage()`, `write_json_atomic()`, `read_source_manifest()`.
- Consumed by: Tasks 3, 9, 13, and 15.

- [ ] **Step 1: Write failing manifest tests**

Cover column-order-independent schema hashing, non-finite values, 90% coverage, UTC timestamps,
JSON round-trip, and rollback when replacement fails.

```python
def test_schema_fingerprint_is_column_order_independent():
    a = pd.DataFrame({"team": pd.Series(["BUF"], dtype="string"), "value": [1.0]})
    b = a[["value", "team"]]
    assert schema_fingerprint(a) == schema_fingerprint(b)

def test_require_coverage_rejects_below_ninety_percent():
    frame = pd.DataFrame({"season": [2024] * 10, "week": [1] * 10,
                          "team": list("ABCDEFGHIJ"), "rating": [1.0] * 8 + [np.nan] * 2})
    with pytest.raises(SourceContractError, match="0.8000"):
        require_coverage(frame, ["rating"], minimum=0.90)
```

- [ ] **Step 2: Run and confirm import failure**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_source_manifest.py -v
```

Expected: FAIL because the module does not exist.

- [ ] **Step 3: Create the source contract module**

```python
@dataclass(frozen=True)
class SourceSnapshot:
    name: str
    seasons: tuple[int, ...]
    retrieved_at: datetime
    schema_sha256: str
    rows: int
    coverage: dict[str, float]
    latest_event_at: datetime | None

def schema_fingerprint(frame: pd.DataFrame) -> str:
    pairs = sorted((name, str(dtype)) for name, dtype in frame.dtypes.items())
    return hashlib.sha256(json.dumps(pairs, separators=(",", ":")).encode()).hexdigest()

def numeric_coverage(frame: pd.DataFrame, columns: Sequence[str]) -> dict[str, float]:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise SourceContractError(f"missing source columns: {missing}")
    coverage = {}
    for column in columns:
        numeric = pd.to_numeric(frame[column], errors="coerce")
        invalid = frame[column].notna() & numeric.isna()
        if invalid.any() or not np.isfinite(numeric.dropna()).all():
            raise SourceContractError(f"non-numeric or non-finite values in {column}")
        coverage[column] = float(numeric.notna().mean()) if len(frame) else 0.0
    return coverage

def require_coverage(
    frame: pd.DataFrame, columns: Sequence[str], minimum: float = 0.90
) -> None:
    coverage = numeric_coverage(frame, columns)
    below = {name: value for name, value in coverage.items() if value < minimum}
    if below:
        raise SourceContractError(f"coverage below {minimum:.4f}: {below}")

def write_json_atomic(payload: Mapping[str, object], path: Path) -> None:
    staged = path.with_suffix(path.suffix + ".tmp")
    staged.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    staged.replace(path)
```

Reject blank source names, naive datetimes, duplicate snapshot names, missing columns,
non-numeric non-null values, and non-finite numbers. Write UTF-8, sorted-key JSON through a
temporary sibling followed by `Path.replace()`. `read_source_manifest()` must reject any payload
that does not round-trip through `SourceSnapshot` with unique names and UTC timestamps.

- [ ] **Step 4: Verify focused and full suites**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_source_manifest.py -v
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
```

- [ ] **Step 5: Commit**

```powershell
git add src/nfl_game/data/source_manifest.py tests/test_source_manifest.py
git commit -m "feat: validate ridge v2 source manifests"
```

---

### Task 3: Add normalized nflreadpy loaders for free Ridge-v2 sources

**Files:**
- Modify: `src/nfl_game/data/nfl.py`
- Modify: `tests/test_data_nfl.py`

**Interfaces:**
- Produces: `load_player_stats()`, `load_players()`, `load_rosters_weekly()`,
  `load_depth_charts()`, `load_snap_counts()`, `load_pfr_advstats()`,
  `load_ftn_charting()`.
- All season-scoped loaders accept `seasons: list[int]` and `save: bool = True`.

- [ ] **Step 1: Write parameterized failing loader tests**

For each wrapper, monkeypatch the matching nflreadpy call, return a fake Polars object, assert
pandas conversion, arguments, team-code normalization, and exact cache filename. Include:

```python
@pytest.mark.parametrize("stat_type", ["pass", "rush", "rec", "def"])
def test_load_pfr_advstats_forwards_week_level(monkeypatch, tmp_path, stat_type):
    calls = []
    monkeypatch.setattr(nfl, "RAW_DIR", tmp_path)
    monkeypatch.setattr(
        nfl.nflreadpy,
        "load_pfr_advstats",
        lambda seasons, stat_type, summary_level: calls.append(
            (seasons, stat_type, summary_level)
        ) or FakePolars(pd.DataFrame({"team": ["OAK"], "opponent": ["SD"]})),
    )
    out = nfl.load_pfr_advstats([2024], stat_type, save=True)
    assert calls == [([2024], stat_type, "week")]
    assert out.loc[0, "team"] == "LV"
    assert (tmp_path / f"pfr_{stat_type}_2024.parquet").exists()
```

Also assert PFR rejects a fifth stat type, `load_players(save=False)` calls without a season,
and `save=False` creates no files.

- [ ] **Step 2: Run and confirm missing-wrapper failures**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_data_nfl.py -v
```

- [ ] **Step 3: Add thin wrappers and normalization boundaries**

Use these source columns:

```python
PLAYER_STATS_TEAM_COLS = ["team", "opponent_team"]
ROSTER_TEAM_COLS = ["team"]
DEPTH_CHART_TEAM_COLS = ["team"]
SNAP_TEAM_COLS = ["team", "opponent"]
PFR_TEAM_COLS = ["team", "opponent"]
PFR_STAT_TYPES = ("pass", "rush", "rec", "def")
```

Call nflreadpy with `summary_level="week"` for player stats and PFR. Convert depth-chart `dt`
to UTC with `pd.to_datetime(frame["dt"], utc=True, errors="raise")` when present. Preserve
source identity fields; Tasks 5 and 7 create their as-of normalized views.

- [ ] **Step 4: Probe the installed nflreadpy API without writing data**

```powershell
.\.venv\Scripts\python.exe -c "import inspect,nflreadpy as n; names=['load_player_stats','load_players','load_rosters_weekly','load_depth_charts','load_snap_counts','load_pfr_advstats','load_ftn_charting']; [print(x, inspect.signature(getattr(n,x))) for x in names]"
```

Expected: all seven functions exist and their call signatures match the wrappers. If the
installed version lacks one, update `nflreadpy` in `pyproject.toml` and regenerate the exact
production wheel lock in the same commit before continuing.

- [ ] **Step 5: Verify**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_data_nfl.py -v
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
```

- [ ] **Step 6: Commit**

```powershell
git add src/nfl_game/data/nfl.py tests/test_data_nfl.py pyproject.toml requirements-prod.txt
git commit -m "feat: load free ridge v2 data sources"
```

---

### Task 4: Build situational team-game inputs and short/long opponent-adjusted ratings

**Files:**
- Create: `src/nfl_game/ratings/v2_team.py`
- Create: `tests/test_v2_team.py`

**Interfaces:**
- Produces: `team_game_v2(pbp) -> pd.DataFrame`,
  `v2_team_ratings(team_games, targets, short_halflife, long_halflife,
  prior_season_weight) -> pd.DataFrame`.
- Output key: `season, week, team`.

- [ ] **Step 1: Write failing aggregation tests**

Build a two-game PBP fixture containing early/late downs, a fourth-quarter play, a nine-point
score difference, sacks, 20-yard passes, and 10-yard rushes. Assert:

```python
def test_neutral_and_early_down_filters_are_fixed():
    out = team_game_v2(pbp_fixture()).set_index("team")
    assert out.loc["BUF", "early_down_epa"] == pytest.approx(0.20)
    assert out.loc["BUF", "neutral_epa"] == pytest.approx(0.15)

def test_explosive_and_sack_rates_use_declared_denominators():
    out = team_game_v2(pbp_fixture()).set_index("team")
    assert out.loc["BUF", "explosive_pass_rate"] == pytest.approx(1 / 3)
    assert out.loc["BUF", "sack_rate"] == pytest.approx(1 / 4)
```

Neutral plays are scrimmage plays in quarters 1–3 with absolute posteam score differential at
most 8. Early downs are downs 1–2. Explosive passes gain at least 20 yards; explosive rushes at
least 10. Sack rate denominator is QB dropbacks.

- [ ] **Step 2: Write failing rating/as-of tests**

Assert short ratings react more strongly to a recent game, every output column carries
`short_` or `long_`, and changing a target/future game cannot alter an earlier target.

```python
def test_future_rows_cannot_change_prior_rating():
    base = v2_team_ratings(team_games(), [(2024, 4)], 4, 16, 0.6)
    poisoned = pd.concat([team_games(), future_game_with_extreme_epa()])
    actual = v2_team_ratings(poisoned, [(2024, 4)], 4, 16, 0.6)
    pd.testing.assert_frame_equal(base, actual)
```

- [ ] **Step 3: Run and confirm module failure**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_v2_team.py -v
```

- [ ] **Step 4: Create aggregation and rating code**

Define these rating targets exactly:

```python
V2_RATING_TARGETS = (
    "epa_play", "epa_pass", "epa_rush", "success_rate",
    "early_down_epa", "neutral_epa", "explosive_pass_rate",
    "explosive_rush_rate", "sack_rate",
)
```

Aggregate one offensive row per game/team. For every target and both half-lives, call the
existing `fit_ratings()` with `decay_weights()` and rename outputs to
`{window}_off_{target}` and `{window}_def_{target}`. Negated defensive orientation remains
“higher is better.” For each target week, call
`decay_weights(team_games, asof_season, asof_week, halflife_games=halflife,
season_penalty=prior_season_weight)` and pass the positive-weight slice to `fit_ratings()`.

- [ ] **Step 5: Verify**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_v2_team.py tests/test_epa.py tests/test_build_ratings.py -v
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
```

- [ ] **Step 6: Commit**

```powershell
git add src/nfl_game/ratings/v2_team.py tests/test_v2_team.py
git commit -m "feat: add situational short and long team ratings"
```

---

### Task 5: Build leak-free quarterback and expected-starter features

**Files:**
- Create: `src/nfl_game/ratings/qb.py`
- Create: `tests/test_qb.py`

**Interfaces:**
- Produces: `qb_week_stats(player_stats)`, `normalize_depth_chart_history(depth_charts,
  schedules)`, `qb_features_for_targets(qb_weeks, depth_history, schedules, targets)`.
- Output key: `season, week, team`; player key is `player_id`/GSIS ID.

- [ ] **Step 1: Write failing quarterback-rate tests**

Use official weekly stat fields `attempts`, `sacks_suffered`, `passing_epa`, `passing_cpoe`, and
`passing_interceptions`. Assert dropback denominators and 200-dropback empirical-Bayes shrinkage:

```python
def test_qb_week_stats_uses_attempts_plus_sacks_as_dropbacks():
    out = qb_week_stats(player_stats_fixture()).set_index("player_id")
    assert out.loc["qb-a", "dropbacks"] == 44
    assert out.loc["qb-a", "epa_per_db"] == pytest.approx(8.8 / 44)

def test_small_sample_rates_shrink_toward_league():
    out = qb_features_for_targets(**qb_feature_inputs())
    assert league_rate < out.loc[0, "qb_int_rate"] < raw_one_game_rate
```

- [ ] **Step 2: Write failing expected-starter and leak tests**

Test 2025+ depth rows by `dt`; pre-2025 rows by their assigned season/week. Rank-1 QB wins.
When no eligible chart exists, choose the prior game's team QB with the most dropbacks and set
`qb_uncertain=1`. Never use schedule `home_qb_id`/`away_qb_id` to establish historical starters.

```python
def test_future_depth_snapshot_cannot_change_expected_starter():
    before = qb_features_for_targets(qb_weeks(), depth_history(), schedules(), [(2025, 4)])
    future = pd.concat([depth_history(), depth_row(dt="2025-10-01T12:00:00Z", player="qb-z")])
    after = qb_features_for_targets(qb_weeks(), future, schedules(), [(2025, 4)])
    pd.testing.assert_frame_equal(before, after)
```

- [ ] **Step 3: Run and confirm failure**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_qb.py -v
```

- [ ] **Step 4: Create the QB module**

Emit these numeric fields:

```python
QB_FEATURE_COLS = (
    "qb_epa_per_db", "qb_cpoe", "qb_sack_rate", "qb_int_rate",
    "qb_change_epa", "qb_new_starter", "qb_rookie", "qb_uncertain",
)
QB_PRIOR_DROPBACKS = 200
ROOKIE_DROPBACK_LIMIT = 100
```

`qb_change_epa` is expected-starter prior EPA/dropback minus the prior team's most-used recent
QB EPA/dropback. `qb_new_starter` is 1 when those player IDs differ. A QB with fewer than 100
career prior dropbacks is a rookie/unproven QB for this feature. Impute league-prior rates only
after calculating and retaining `qb_uncertain`.

- [ ] **Step 5: Verify**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_qb.py -v
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
```

- [ ] **Step 6: Commit**

```powershell
git add src/nfl_game/ratings/qb.py tests/test_qb.py
git commit -m "feat: add as-of quarterback context"
```

---

### Task 6: Build style, turnover, field-position, and special-teams features

**Files:**
- Create: `src/nfl_game/ratings/style.py`
- Create: `tests/test_style.py`

**Interfaces:**
- Produces: `team_game_style(pbp)`, `style_features_for_targets(team_games, targets,
  halflife=8.0)`.
- Output key: `season, week, team`.

- [ ] **Step 1: Write failing formula tests**

Pin the exact definitions:

- neutral pass rate: pass attempts/dropbacks plus designed rushes, quarters 1–3, score within 8;
- pace: median elapsed seconds between consecutive same-drive offensive snaps, keeping `0 < dt <= 60`;
- turnover rate: interceptions plus lost fumbles per scrimmage play;
- starting field position: `100 - yardline_100` on the first valid offensive play of each drive;
- special-teams EPA/play: mean EPA on `special_teams_play == 1`, assigned to `posteam`.

```python
def test_pace_excludes_quarter_break_and_timeout_gaps():
    out = team_game_style(style_pbp()).set_index("team")
    assert out.loc["BUF", "pace_seconds"] == pytest.approx(27.0)

def test_turnover_rate_is_shrunk_over_trailing_history():
    out = style_features_for_targets(style_games(), [(2024, 5)])
    assert 0 < out.loc[0, "turnover_rate"] < out.loc[0, "raw_turnover_rate"]
```

- [ ] **Step 2: Run and confirm failure**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_style.py -v
```

- [ ] **Step 3: Create the style module**

```python
STYLE_FEATURE_COLS = (
    "neutral_pass_rate", "pace_seconds", "turnover_rate",
    "explosive_play_rate", "starting_field_position", "special_teams_epa",
    "style_imputed",
)
TURNOVER_PRIOR_PLAYS = 200
```

Use trailing eight-game exponential weights. Shrink turnover rate with the league rate and 200
prior plays. Carry a `style_imputed` flag when a team has no eligible prior observations. Drop
all rows at or after each target cutoff before aggregation.

- [ ] **Step 4: Verify**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_style.py -v
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
```

- [ ] **Step 5: Commit**

```powershell
git add src/nfl_game/ratings/style.py tests/test_style.py
git commit -m "feat: add game style and hidden-yardage features"
```

---

### Task 7: Build roster, depth-chart, and snap-share continuity features

**Files:**
- Create: `src/nfl_game/ratings/personnel.py`
- Create: `tests/test_personnel.py`

**Interfaces:**
- Produces: `player_id_map(players)`, `normalize_snap_counts(snaps, players)`,
  `personnel_features_for_targets(snaps, rosters, depth_charts, players, schedules, targets)`.
- Output key: `season, week, team`.

- [ ] **Step 1: Write failing identity and continuity tests**

Join snap `pfr_player_id` to roster/depth `gsis_id` through the players table; never use player
name as a primary join. Assert returning shares and concentration:

```python
def test_returning_snap_share_uses_current_roster_and_prior_season_snaps():
    out = personnel_features_for_targets(**personnel_inputs(targets=[(2025, 1)])).set_index("team")
    assert out.loc["BUF", "off_returning_share"] == pytest.approx(0.75)
    assert out.loc["BUF", "roster_churn"] == pytest.approx(0.25)

def test_snap_hhi_is_sum_of_squared_player_shares():
    out = personnel_features_for_targets(**personnel_inputs())
    assert out.loc[0, "off_snap_hhi"] == pytest.approx(0.5**2 + 0.3**2 + 0.2**2)
```

- [ ] **Step 2: Write failing as-of and missingness tests**

Add a depth chart and roster snapshot after the target kickoff; assert the earlier output is
byte-identical. If fewer than 90% of prior snaps map to a player ID, set `personnel_imputed=1`
and report `id_coverage`; do not treat unmapped players as departures.

- [ ] **Step 3: Run and confirm failure**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_personnel.py -v
```

- [ ] **Step 4: Create the personnel module**

```python
PERSONNEL_FEATURE_COLS = (
    "off_returning_share", "def_returning_share", "off_snap_hhi", "def_snap_hhi",
    "depth_chart_change_rate", "roster_churn", "id_coverage", "personnel_imputed",
)
```

For Week 1, returning shares compare the current as-of roster with prior-season unit snaps. For
later weeks, snap HHI uses only current-season games before the target. Depth-chart change rate
compares the last two eligible snapshots by player ID and position slot. All snapshot selection
uses `dt <= prediction_cutoff`; a pre-2025 weekly snapshot is eligible only for its labeled week.

- [ ] **Step 5: Verify**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_personnel.py -v
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
```

- [ ] **Step 6: Commit**

```powershell
git add src/nfl_game/ratings/personnel.py tests/test_personnel.py
git commit -m "feat: add personnel continuity features"
```

---

### Task 8: Build the PFR advanced-stat candidate block

**Files:**
- Create: `src/nfl_game/ratings/pfr.py`
- Create: `tests/test_pfr.py`
- Modify: `README.md`

**Interfaces:**
- Produces: `PFR_REQUIRED_COLUMNS`, `team_week_pfr(frames)`,
  `pfr_features_for_targets(team_weeks, targets, halflife=8.0)`.
- Input `frames` keys: `pass`, `rush`, `rec`, `def`.

- [ ] **Step 1: Pin the live source contract before transformations**

Run all four 2025 weekly loaders and save only their sorted column names in the task report.
Assert these identity columns for every frame: `game_id`, `season`, `week`, `game_type`, `team`,
`opponent`, `pfr_player_id`. Assert the passing frame includes:

```python
PFR_PASS_INPUTS = (
    "passing_drops", "passing_bad_throws", "times_sacked", "times_blitzed",
    "times_hurried", "times_hit", "times_pressured", "times_pressured_pct",
    "def_times_blitzed", "def_times_hurried", "def_times_hitqb",
)
```

Run:

```powershell
.\.venv\Scripts\python.exe -c "from nfl_game.data.nfl import load_pfr_advstats; [print(k, sorted(load_pfr_advstats([2025], k, save=False).columns)) for k in ('pass','rush','rec','def')]"
```

If an asserted column is absent, fail the block as unavailable rather than renaming an unrelated
field. Record nflverse/PFR attribution in `README.md` before this block can be production-eligible.

- [ ] **Step 2: Write failing aggregation tests**

Fixtures cover every required field and assert volume-weighted team-week values. The initial
output columns are:

```python
PFR_FEATURE_COLS = (
    "pfr_pressure_rate", "pfr_hurry_rate", "pfr_hit_rate", "pfr_bad_throw_rate",
    "pfr_drop_rate", "pfr_sack_rate", "pfr_rush_ybc", "pfr_rush_yac",
    "pfr_broken_tackle_rate", "pfr_rec_drop_rate", "pfr_def_missed_tackle_rate",
    "pfr_def_pressure_rate", "pfr_imputed",
)
```

The rush/receiving/defense source-contract dictionaries must map their real 2025 field names to
these outputs in one constant; tests instantiate fixtures from that constant so a schema drift
fails at the boundary.

- [ ] **Step 3: Run and confirm failure**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_pfr.py -v
```

- [ ] **Step 4: Create aggregation, trailing features, and coverage gate**

Aggregate counts before forming rates; weight player averages by attempts/targets/snaps. Use an
eight-game trailing decay and only weeks before the target. `pfr_imputed=1` when any requested
team-week aggregate is absent. Call `require_coverage(team_weeks, PFR_REQUIRED_NUMERIC_COLUMNS, minimum=0.90)` before making C5
production-eligible.

```python
def _safe_rate(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator.div(denominator.where(denominator.gt(0)))

def team_week_pfr(frames: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    missing = sorted(set(PFR_STAT_TYPES).difference(frames))
    if missing:
        raise SourceContractError(f"missing PFR stat types: {missing}")
    normalized = [
        normalize_pfr_frame(stat_type, frames[stat_type])
        for stat_type in PFR_STAT_TYPES
    ]
    return reduce(
        lambda left, right: left.merge(
            right, on=["season", "week", "team"], how="outer", validate="one_to_one"
        ),
        normalized,
    )

def pfr_features_for_targets(
    team_weeks: pd.DataFrame,
    targets: Sequence[tuple[int, int]],
    halflife: float = 8.0,
) -> pd.DataFrame:
    return trailing_pfr_features(team_weeks, targets, PFR_OUTPUT_COLS, halflife)
```

- [ ] **Step 5: Verify**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_pfr.py tests/test_source_manifest.py -v
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
```

- [ ] **Step 6: Commit**

```powershell
git add src/nfl_game/ratings/pfr.py tests/test_pfr.py README.md
git commit -m "feat: add pfr advanced candidate features"
```

---

### Task 9: Assemble the union Ridge-v2 game feature artifact and frozen manifests

**Files:**
- Create: `src/nfl_game/model/v2_features.py`
- Create: `tests/test_v2_features.py`

**Interfaces:**
- Consumes: Ridge-v1 base features plus Tasks 4–8 team-week frames.
- Produces: `V2FeatureBundle(frame, manifest)`, `build_v2_game_features()`.

- [ ] **Step 1: Write failing schema and sign tests**

Construct two games and assert home-minus-away versus combined-total semantics. Pin these core
families in `MARGIN_FEATURES_BY_BLOCK` and `TOTAL_FEATURES_BY_BLOCK`:

```python
MARGIN_FEATURES_BY_BLOCK = {
    "C0": tuple(FEATURE_COLS),
    "C1": ("rating_net_diff_short", "rating_net_diff_long",
           "pass_matchup_diff_short", "pass_matchup_diff_long",
           "rush_matchup_diff_short", "rush_matchup_diff_long",
           "success_diff_short", "success_diff_long",
           "early_down_diff_short", "neutral_diff_short",
           "explosive_pass_diff", "explosive_rush_diff",
           "rest_diff", "home_indicator", "div_game"),
    "C2": ("qb_epa_diff", "qb_cpoe_diff", "qb_sack_rate_diff",
           "qb_int_rate_diff", "qb_change_epa_diff", "qb_new_starter_any",
           "qb_rookie_any", "qb_uncertain_any"),
    "C3": ("neutral_pass_rate_diff", "pace_diff", "turnover_rate_diff",
           "explosive_play_diff", "field_position_diff", "special_teams_diff",
           "style_imputed_any"),
    "C4": ("off_returning_share_diff", "def_returning_share_diff",
           "off_snap_hhi_diff", "def_snap_hhi_diff", "depth_change_diff",
           "roster_churn_diff", "personnel_imputed_any"),
    "C5": ("pfr_pressure_edge_diff", "pfr_accuracy_diff", "pfr_drop_diff",
           "pfr_rush_contact_diff", "pfr_tackle_diff", "pfr_imputed_any"),
}
```

The total block uses matchup sums, combined QB quality/uncertainty, mean pace/pass tendency,
combined turnover/explosive/special-teams environment, the weaker returning-share value, dome,
temperature, wind, and combined PFR pressure/accuracy/contact values. Test that every candidate
is cumulative: C3 columns equal C1 + C2 + C3 block columns without duplicates.

- [ ] **Step 2: Write failing join, coverage, and leak tests**

Assert one-to-one game keys, many-to-one team-week joins, normalized team identity, numeric
finite output, explicit imputation flags, and that changing any post-cutoff input row cannot
change an earlier game. Assert `spread_line` and `total_line` remain output metadata but are
absent from both manifest schemas.

- [ ] **Step 3: Run and confirm failure**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_v2_features.py -v
```

- [ ] **Step 4: Create the feature assembler**

```python
@dataclass(frozen=True)
class V2FeatureBundle:
    frame: pd.DataFrame
    manifest: FeatureManifest

def merge_v2_blocks(
    base_features: pd.DataFrame, blocks: Mapping[str, pd.DataFrame]
) -> pd.DataFrame:
    frame = base_features.copy()
    for name, block in blocks.items():
        sided = team_block_to_game_features(block, name=name)
        frame = frame.merge(
            sided, on=["game_id", "season", "week"], how="left", validate="one_to_one"
        )
    return frame
```

Merge each team-week frame twice, once by home and once by away. Derive all differences/sums in
one named helper per block. Fill only documented block-neutral values, then set the corresponding
`*_imputed_any` field. `build_v2_game_features()` calls `merge_v2_blocks()`, validates game-key
uniqueness and finite manifested values, and returns `V2FeatureBundle(frame, manifest)`. Create the
manifest version from the SHA-256 of sorted feature formulas, source versions, and constants.

- [ ] **Step 5: Verify**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_v2_features.py tests/test_features.py -v
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
```

- [ ] **Step 6: Commit**

```powershell
git add src/nfl_game/model/v2_features.py tests/test_v2_features.py
git commit -m "feat: assemble target-specific ridge v2 features"
```

---

### Task 10: Fit separate margin and total Ridge-v2 pipelines

**Files:**
- Create: `src/nfl_game/model/v2.py`
- Create: `tests/test_v2_predict.py`

**Interfaces:**
- Produces: `RidgeV2Model(config, manifest).fit(frame).predict(frame)` and
  `fit_target_ridge(frame, target, config, manifest)`.
- Prediction output: exactly `game_id, model_margin, model_total`.

- [ ] **Step 1: Write failing separate-schema tests**

Use synthetic features where only a margin column explains margin and only a total column
explains total. Assert each prediction changes only when its own schema changes.

```python
def test_margin_and_total_use_different_columns():
    model = RidgeV2Model(config(), manifest()).fit(training_rows())
    changed = probe_rows().copy()
    changed["margin_signal"] += 10
    before = model.predict(probe_rows())
    after = model.predict(changed)
    assert not np.allclose(before["model_margin"], after["model_margin"])
    np.testing.assert_allclose(before["model_total"], after["model_total"])
```

- [ ] **Step 2: Write failing guards**

Assert fit rejects a manifest containing a market column, missing/NaN/non-finite features,
degenerate non-binary inputs, an empty target, and prediction before fit. Assert binary
imputation flags are allowed. Assert the two configured alpha values reach their respective
Ridge instances.

- [ ] **Step 3: Run and confirm failure**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_v2_predict.py -v
```

- [ ] **Step 4: Create the model**

Reuse `RobustStandardScaler` but keep v2 state isolated:

```python
def fit_target_ridge(
    train: pd.DataFrame,
    target: Literal["margin", "total_points"],
    config: TargetConfig,
    manifest: FeatureManifest,
) -> Pipeline:
    manifest_target = "margin" if target == "margin" else "total"
    columns = list(manifest.columns(manifest_target, config.candidate))
    valid = train[target].notna()
    validate_model_matrix(train.loc[valid, columns], target=target)
    pipeline = make_pipeline(
        RobustStandardScaler(),
        Ridge(alpha=config.alpha),
    )
    pipeline.fit(train.loc[valid, columns], train.loc[valid, target])
    return pipeline
```

`RidgeV2Model.fit()` calls `fit_target_ridge()` separately for margin and total, stores both
pipelines, and returns `self`. `predict()` requires fitted pipelines and returns exactly
`game_id`, `model_margin`, and `model_total` while selecting each target's own manifested columns.
The margin pipeline reads only `manifest.columns("margin", candidate)`; total reads only
`manifest.columns("total", candidate)`. Validate each selected column set independently.

- [ ] **Step 5: Verify**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_v2_predict.py tests/test_predict.py -v
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
```

- [ ] **Step 6: Commit**

```powershell
git add src/nfl_game/model/v2.py tests/test_v2_predict.py
git commit -m "feat: fit separate ridge v2 targets"
```

---

### Task 11: Implement deterministic nested walk-forward selection

**Files:**
- Create: `src/nfl_game/experiments/__init__.py`
- Create: `src/nfl_game/experiments/v2_selection.py`
- Create: `tests/test_v2_selection.py`

**Interfaces:**
- Produces: `TargetSelection`, `OuterSelection`, `NestedBacktestResult`,
  `select_target_config()`, `nested_walk_forward_v2()`.

- [ ] **Step 1: Write failing inner-boundary tests**

Inject a spy target fitter and assert an outer 2024 call never receives a 2024+ row; each 2023
inner validation fit sees only `<2023`. Add a poisoned future season and assert selections and
predictions for earlier seasons are identical.

```python
def test_inner_selection_never_reads_outer_season():
    seen = []
    nested_walk_forward_v2(features(), [2024], manifest(),
                           target_fitter=record_training_seasons(seen))
    assert seen
    assert all(max(seasons) < validation_season for seasons, validation_season in seen)
```

- [ ] **Step 2: Write failing eligibility and simplicity tests**

Assert a candidate with one validation season or 399 games is ineligible. When C4 beats C2 by
0.03 MAE, select C2; when it beats by 0.06, select C4. Assert margin and total can select
different candidates/alphas, and exact ties resolve by `TargetConfig.key()`.

- [ ] **Step 3: Run and confirm failure**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_v2_selection.py -v
```

- [ ] **Step 4: Create the selection engine**

```python
@dataclass(frozen=True)
class TargetSelection:
    target: str
    config: TargetConfig
    mean_inner_mae: float
    validation_seasons: tuple[int, ...]
    validation_games: int

@dataclass(frozen=True)
class OuterSelection:
    season: int
    margin: TargetSelection
    total: TargetSelection

@dataclass(frozen=True)
class NestedBacktestResult:
    predictions: pd.DataFrame
    selections: tuple[OuterSelection, ...]

MIN_INNER_SEASONS = 2
MIN_INNER_GAMES = 400
ONE_STANDARD_ERROR_TOLERANCE = 0.05
```

`nested_walk_forward_v2(features, test_seasons, manifest)` evaluates every fixed grid entry using
only seasons before each outer season, filters using the constants above, and applies the exact
deterministic `TargetConfig.key()` tie-break. Score inner seasons equally. Use all eligible prior
seasons for refit after selection. For a
calibration-seed year where no v2 candidate satisfies two-season/400-game eligibility, use C0
without pretending it was selected by v2 evidence.

- [ ] **Step 5: Verify**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_v2_selection.py -v
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
```

- [ ] **Step 6: Commit**

```powershell
git add src/nfl_game/experiments/__init__.py src/nfl_game/experiments/v2_selection.py tests/test_v2_selection.py
git commit -m "feat: select ridge v2 with nested walk-forward"
```

---

### Task 12: Add walk-forward calibration, uncertainty, and promotion gates

**Files:**
- Create: `src/nfl_game/experiments/v2_evaluation.py`
- Create: `tests/test_v2_evaluation.py`

**Interfaces:**
- Produces: `BootstrapInterval`, `walk_forward_probabilities()`,
  `block_bootstrap_mean()`, `joint_market_regression()`, `evaluate_v2()`,
  `research_gate_decision()`, and `promotion_decision()`.

- [ ] **Step 1: Write failing calibration-order tests**

Generate predictions for 2019–2025. Assert 2021 probabilities are fit only on 2019–2020 rows,
pushes are excluded per target, and 2021–2025 report metrics ignore the seed seasons.

```python
def test_walk_forward_calibration_uses_only_prior_oos_predictions(monkeypatch):
    seen = []
    out = walk_forward_probabilities(predictions_2019_2025(), fit_observer=seen.append)
    assert seen_for(seen, 2021) == {2019, 2020}
    assert set(out.query("season >= 2021")["season"]) == set(range(2021, 2026))
```

- [ ] **Step 2: Write failing bootstrap and gate tests**

Resample `(season, week)` clusters with `np.random.default_rng(0)`, 10,000 draws, and use the
10th percentile as the one-sided 90% lower bound. Synthetic reports independently fail each exact
gate:

1. Margin MAE is below `10.274` on paired games.
2. Total MAE is below `10.684` on paired games.
3. Margin MAE improves in at least three of the five outer seasons.
4. Total MAE improves in at least three of the five outer seasons.
5. Both targets' paired absolute-error improvement lower-90 bounds are positive.
6. Both joint closing-line/model regressions have positive model coefficients and lower-90 bounds.
7. ATS hit rate is at least `0.497737556561086 - 0.01`.
8. O/U hit rate is at least `0.5022255192878339 - 0.01`.
9. Cover and over Brier scores are each no worse than Ridge v1 on identical rows.
10. Correctness, availability, determinism, and source-reliability checks all pass.
11. The 2026 shadow-production rebuild succeeds without changing Ridge-v1 artifacts or ledgers.

Edge cohorts are report-only and cannot override these gates. Pin the numeric floors and the
two-stage shadow decision:

```python
ATS_FLOOR = 0.497737556561086 - 0.01
OU_FLOOR = 0.5022255192878339 - 0.01

def test_research_gate_rejects_nonpositive_market_increment():
    report = passing_report()
    report["margin_market_model_coef_lower90"] = 0.0
    decision = research_gate_decision(report)
    assert not decision.approved
    assert "margin market contribution" in decision.failures

def test_full_promotion_requires_successful_shadow_rebuild():
    decision = promotion_decision(passing_report(), shadow_rebuild_passed=None)
    assert not decision.approved
    assert "shadow production rebuild" in decision.pending
```

- [ ] **Step 3: Run and confirm failure**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_v2_evaluation.py -v
```

- [ ] **Step 4: Create evaluation code**

```python
@dataclass(frozen=True)
class BootstrapInterval:
    estimate: float
    lower90: float
    upper90: float

@dataclass(frozen=True)
class PromotionDecision:
    approved: bool
    failures: tuple[str, ...]
    pending: tuple[str, ...]

def block_bootstrap_mean(
    frame: pd.DataFrame, value_col: str, draws: int = 10_000, seed: int = 0
) -> BootstrapInterval:
    blocks = [
        group[value_col].to_numpy(dtype=float)
        for _, group in frame.groupby(["season", "week"], sort=True)
    ]
    rng = np.random.default_rng(seed)
    samples = np.empty(draws)
    for draw in range(draws):
        chosen = rng.integers(0, len(blocks), size=len(blocks))
        samples[draw] = np.concatenate([blocks[index] for index in chosen]).mean()
    return BootstrapInterval(
        estimate=float(frame[value_col].mean()),
        lower90=float(np.quantile(samples, 0.10)),
        upper90=float(np.quantile(samples, 0.90)),
    )
```

`joint_market_regression()` fits actual outcome on an intercept, closing line, and model output;
bootstrap the model coefficient with the same week blocks. `evaluate_v2()` constructs the complete
metric dictionary from one-to-one paired rows. `research_gate_decision()` evaluates gates 1–10.
`promotion_decision(report, shadow_rebuild_passed)` adds gate 11, never treats `None` as success,
and returns every failure and pending gate without rounding.

Join v1/v2 by `game_id` with `validate="one_to_one"` and compare identical valid rows. Report
per-season MAEs, RMSEs, paired errors, joint line/model coefficients, Brier scores, all-pick
records, and cumulative 2+/5+/10+/15+ cohorts with wins/losses/pushes/n/intervals.

- [ ] **Step 5: Verify**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_v2_evaluation.py tests/test_calibrate.py tests/test_backtest.py -v
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
```

- [ ] **Step 6: Commit**

```powershell
git add src/nfl_game/experiments/v2_evaluation.py tests/test_v2_evaluation.py
git commit -m "feat: evaluate ridge v2 promotion evidence"
```

---

### Task 13: Build reproducible historical Ridge-v2 features and manifests

**Files:**
- Create: `src/nfl_game/pipeline/build_v2.py`
- Create: `scripts/build_v2_dataset.py`
- Create: `tests/test_build_v2.py`
- Modify: `README.md`
- Modify: `CLAUDE.md`

**Interfaces:**
- Produces: `V2BuildInputs`, `V2BuildArtifacts`, `build_v2_artifacts()`,
  `write_v2_artifacts_atomic()` and a dry-run/write CLI.

- [ ] **Step 1: Write failing injectable-builder tests**

Use tiny fake loader frames and a fixed UTC clock. Assert the builder calls the declared season
ranges, produces one unique row per regular-season game, creates all source snapshots, preserves
C0 columns/targets/lines exactly, and refuses sub-90% production block coverage.

```python
def test_builder_preserves_c0_values_exactly(fake_inputs):
    out = build_v2_artifacts(fake_inputs, retrieved_at=FIXED_UTC)
    pd.testing.assert_frame_equal(
        out.features[["game_id", *FEATURE_COLS]],
        fake_inputs.base_features[["game_id", *FEATURE_COLS]],
    )
```

- [ ] **Step 2: Write failing CLI and atomicity tests**

Assert default/dry-run writes nothing, `--write` replaces both feature and manifest artifacts,
and a second identical write preserves digests. Inject a failure after the first staged file and
assert both original destinations are restored.

- [ ] **Step 3: Run and confirm failure**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_build_v2.py -v
```

- [ ] **Step 4: Create the builder and CLI**

```python
@dataclass(frozen=True)
class V2BuildInputs:
    schedules: pd.DataFrame
    pbp: pd.DataFrame
    ngs: pd.DataFrame
    player_stats: pd.DataFrame
    players: pd.DataFrame
    rosters: pd.DataFrame
    depth_charts: pd.DataFrame
    snap_counts: pd.DataFrame
    pfr: dict[str, pd.DataFrame]
    base_features: pd.DataFrame

@dataclass(frozen=True)
class V2BuildArtifacts:
    features: pd.DataFrame
    manifest: dict[str, object]

V2_HISTORICAL_SEASONS = tuple(range(2015, 2026))
V2_EVALUATION_SEASONS = tuple(range(2021, 2026))
```

`build_v2_artifacts()` invokes Tasks 2–9 in dependency order and returns both payloads without
writing. `write_v2_artifacts_atomic()` stages both files, validates both staged payloads, replaces
both destinations, and restores both originals if either replacement fails.

The CLI loads 2015–2025 PBP for warm-up, 2016–2025 NGS/player/roster/snap data where available,
2018–2025 PFR, and the existing frozen C0 feature artifact. Print every source row count,
coverage value, schema digest, and output digest. Use `--dry-run` by default and `--write`
explicitly.

- [ ] **Step 5: Run focused tests, full tests, and a real dry run**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_build_v2.py -v
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe scripts/build_v2_dataset.py --dry-run
```

Expected: the real dry run reports source/coverage/schema/output details and changes no tracked
artifact. Review any block below 90% before permitting the write step.

- [ ] **Step 6: Write reviewed artifacts and verify idempotence**

```powershell
.\.venv\Scripts\python.exe scripts/build_v2_dataset.py --write
.\.venv\Scripts\python.exe scripts/build_v2_dataset.py --write
git status --short
```

Expected: only the approved Ridge-v2 artifact files and documentation are changed; the second
run reports identical semantic content/digests.

- [ ] **Step 7: Commit**

```powershell
git add src/nfl_game/pipeline/build_v2.py scripts/build_v2_dataset.py tests/test_build_v2.py README.md CLAUDE.md data/processed/game_features_ridge_v2.parquet data/processed/ridge_v2_manifest.json
git commit -m "data: build reproducible ridge v2 features"
```

---

### Task 14: Run the full Ridge-v2 experiment and stop for shadow-build approval

**Files:**
- Create: `scripts/backtest_v2.py`
- Create: `tests/test_backtest_v2_cli.py`
- Modify: `README.md`
- Modify: `CLAUDE.md`

**Interfaces:**
- Produces the four research artifacts: outer predictions, evaluation JSON, ablation Parquet,
  calibration JSON.
- Hard stop: no tracker, web, workflow, Docker, or public artifact changes in this task.

- [ ] **Step 1: Write failing CLI tests**

Inject feature reads and experiment functions. Assert default dry-run writes nothing; `--write`
atomically writes all four research artifacts; a failed research gate exits nonzero only when
`--require-research-gates` is supplied; no flag can update a v1 artifact.

```python
def test_cli_never_targets_v1_artifacts():
    args = backtest_v2._parser().parse_args([])
    outputs = {args.predictions, args.evaluation, args.ablation, args.calibration}
    assert PROCESSED_DIR / "game_features.parquet" not in outputs
    assert PROCESSED_DIR / "tracker_ledger.parquet" not in outputs
```

- [ ] **Step 2: Run and confirm failure**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_backtest_v2_cli.py -v
```

- [ ] **Step 3: Create the experiment CLI**

The CLI must:

1. Reproduce Ridge-v1 2021–2025 predictions and exact baseline.
2. Run Ridge-v2 nested predictions for 2019–2025.
3. Use 2019–2020 only as calibration seeds.
4. Evaluate promotion on 2021–2025 identical rows.
5. Run remove-one-block ablations for every eligible selected candidate.
6. Print each outer season's selected margin/total configuration.
7. Print gates 1–10 as `PASS` or `FAIL` and gate 11 as `PENDING`.
8. Write only under explicit `--write`.

```python
def main(argv=None, dependencies=None) -> int:
    args = _parser().parse_args(argv)
    result = nested_walk_forward_v2(features, range(2019, 2026), manifest)
    probabilities = walk_forward_probabilities(result.predictions)
    report = evaluate_v2(v1_predictions, result.predictions, v1_probs, probabilities)
    decision = research_gate_decision(report)
    artifacts = build_experiment_artifacts(result, probabilities, report, decision)
    print_gate_report(report, decision, shadow_status="PENDING")
    if args.write:
        write_experiment_artifacts_atomic(artifacts, output_paths(args))
    return int(args.require_research_gates and not decision.approved)
```

- [ ] **Step 4: Verify focused/full tests and dry run**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_backtest_v2_cli.py -v
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe scripts/backtest_v2.py --dry-run
```

- [ ] **Step 5: Run and write the locked experiment once**

```powershell
.\.venv\Scripts\python.exe scripts/backtest_v2.py --write
```

Record runtime, selected configurations, exact metrics, bootstrap intervals, gates 1–10, the
pending shadow gate, artifact row counts, and SHA-256 digests in the task report. Do not modify
formulas, manifests, or grids in response to the results inside this experiment name.

- [ ] **Step 6: Verify written artifacts and Ridge-v1 preservation**

```powershell
.\.venv\Scripts\python.exe scripts/backtest.py --test-seasons 2021-2025
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
git diff --check
git status --short
```

- [ ] **Step 7: Commit the experiment outputs and report documentation**

```powershell
git add scripts/backtest_v2.py tests/test_backtest_v2_cli.py README.md CLAUDE.md data/processed/ridge_v2_outer_predictions.parquet data/processed/ridge_v2_evaluation.json data/processed/ridge_v2_ablation.parquet data/processed/ridge_v2_calibration.json
git commit -m "data: evaluate locked ridge v2 challenger"
```

- [ ] **Step 8: Mandatory user shadow-build checkpoint**

Present the complete research report. If any gate 1–10 fails, state that Ridge v1 remains official
and stop. If gates 1–10 pass, request explicit approval to run Task 15's shadow-production rebuild.
Approval of this plan does not count as approval to run the shadow rebuild or promote Ridge v2.

---

### Task 15: Conditionally build and shadow-refresh 2026 Ridge-v2 artifacts

**Precondition:** Task 14 gates 1–10 passed and the user explicitly approved shadow-build work.

**Files:**
- Create: `src/nfl_game/pipeline/refresh_v2_2026.py`
- Create: `scripts/refresh_v2_2026.py`
- Create: `tests/test_refresh_v2_2026.py`

**Interfaces:**
- Produces: `V2RefreshArtifacts`, `validate_source_freshness()`,
  `build_v2_refresh_artifacts()`, `write_v2_refresh_atomic()`, and a dry-run
  `--require-promotion` mode that combines the locked research report with gate 11.

- [ ] **Step 1: Write failing freshness tests**

Use a fixed prediction cutoff. Assert 73-hour roster/depth snapshots fail, 72-hour snapshots
pass, and a performance feed missing a game final for 49 hours fails. Assert last-good reuse is
allowed only when those same checks pass.

- [ ] **Step 2: Write failing refresh tests**

Assert historical rows/digests are preserved; only active 2026 prediction weeks are appended;
selected production configs come from the locked evaluation; all model inputs are finite; no
market column is consumed; default mode is shadow/dry-run; atomic rollback covers feature and
manifest files.

- [ ] **Step 3: Run and confirm failure**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_refresh_v2_2026.py -v
```

- [ ] **Step 4: Create refresh module and CLI**

```python
@dataclass(frozen=True)
class V2RefreshArtifacts:
    features: pd.DataFrame
    manifest: dict[str, object]

MAX_SNAPSHOT_AGE = timedelta(hours=72)
MAX_MISSING_FINAL_AGE = timedelta(hours=48)

def snapshot_age(snapshot: SourceSnapshot, cutoff: datetime) -> timedelta:
    if cutoff.tzinfo is None or snapshot.retrieved_at.tzinfo is None:
        raise SourceContractError("freshness timestamps must be timezone-aware")
    return cutoff - snapshot.retrieved_at
```

`validate_source_freshness(snapshots, schedule, cutoff)` rejects negative ages, required snapshots
older than `MAX_SNAPSHOT_AGE`, and any scheduled final absent from the performance feeds for more
than `MAX_MISSING_FINAL_AGE`. `build_v2_refresh_artifacts()` freezes historical rows, assembles
only active 2026 prediction weeks against `locked_manifest`, and revalidates keys, columns, and
finite values. Fail closed on stale required blocks. Do not fall back to Ridge v1 or remove a
selected block. Print that output is shadow-only until Task 17 activates it.

- [ ] **Step 5: Verify and perform the gate-11 shadow run**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_refresh_v2_2026.py -v
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe scripts/refresh_v2_2026.py --dry-run --require-promotion
```

Review source timestamps, coverage, output rows, selected configs, prediction values, and the
combined eleven-gate decision. The dry run must prove Ridge-v1 artifacts and ledgers are unchanged.

- [ ] **Step 6: Commit shadow capability**

```powershell
git add src/nfl_game/pipeline/refresh_v2_2026.py scripts/refresh_v2_2026.py tests/test_refresh_v2_2026.py
git commit -m "feat: shadow refresh 2026 ridge v2 data"
```

- [ ] **Step 7: Mandatory final promotion checkpoint**

Present gate 11 with the locked Task 14 report. If it fails, Ridge v1 remains official and stop.
If all eleven gates pass, request explicit approval for Task 16 public model wiring. Approval to
run the shadow build does not count as promotion approval.

---

### Task 16: Conditionally add Ridge-v2 tracker and website version isolation

**Precondition:** All eleven gates passed and the user explicitly approved public model wiring.

**Files:**
- Create: `scripts/build_v2_tracker.py`
- Create: `src/nfl_game/web/v2_service.py`
- Modify: `src/nfl_game/tracking/ledger.py`
- Modify: `src/nfl_game/web/tracker_service.py`
- Modify: `src/nfl_game/web/tracker_page.py`
- Modify: `src/nfl_game/web/app.py`
- Modify: `src/nfl_game/web/runtime.py`
- Modify: `scripts/game_app.py`
- Create: `tests/test_build_v2_tracker.py`
- Create: `tests/test_web_v2_service.py`
- Modify: `tests/test_web_tracker_service.py`
- Modify: `tests/test_web_tracker_page.py`
- Modify: `tests/test_web_runtime.py`
- Modify: `tests/test_webapp.py`

**Interfaces:**
- Produces a separate Ridge-v2 historical ledger and a promoted Ridge-v2 slate service.
- Tracker API adds required `model_version` filtering whenever more than one version is loaded.

- [ ] **Step 1: Write failing v2-ledger isolation tests**

Build the ledger from the frozen v2 outer predictions. Assert `model_version == "ridge-v2"`,
historical v1 bytes are untouched, and concatenating service inputs cannot produce a summary
without an explicit single version filter.

```python
def test_tracker_never_aggregates_model_versions():
    service = TrackerService(pd.concat([v1_ledger(), v2_ledger()]))
    with pytest.raises(ValueError, match="model_version"):
        service.summary("backtest", "all")
```

- [ ] **Step 2: Write failing dashboard/service tests**

Assert Ridge-v2 service loads its own features, manifest, configuration, and calibration; emits
`model_version="ridge-v2"`; accepts only Ridge; trains on seasons `< requested season`; never
uses line columns; and preserves live-line comparison after predictions exist.

- [ ] **Step 3: Write failing UI/runtime tests**

Assert the dashboard visibly labels Ridge v2, tracker provides a model-version selector, every
summary/games request sends the selected version, v1 and v2 records remain separate, and runtime
fails closed when any promoted artifact is missing or invalid.

- [ ] **Step 4: Run and confirm failures**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_build_v2_tracker.py tests/test_web_v2_service.py tests/test_web_tracker_service.py tests/test_web_tracker_page.py tests/test_web_runtime.py tests/test_webapp.py -v
```

- [ ] **Step 5: Create the ledger builder and v2 service**

```python
RIDGE_V2_MODEL_VERSION = "ridge-v2"

def require_v2_artifacts(paths: Mapping[str, Path]) -> None:
    missing = sorted(name for name, path in paths.items() if not path.is_file())
    if missing:
        raise FileNotFoundError(f"missing Ridge-v2 artifacts: {missing}")
```

`RidgeV2SlateService.from_artifacts()` calls `require_v2_artifacts()`, verifies every digest and
the approved evaluation decision, then constructs a Ridge-only service with the locked manifest
and calibrators. `payload(season, week, edge_threshold=2.0)` trains on seasons strictly before the
requested season, predicts without line inputs, then adds live-line comparisons and the explicit
`model_version="ridge-v2"` label.

`build_v2_tracker.py` accepts only the frozen Task 14 prediction artifact and exact evaluation
digest, creates `tracker_ledger_ridge_v2.parquet`, and never opens v1 output for writing.

- [ ] **Step 6: Add explicit tracker version filtering and UI state**

Change service keys to `(model_version, record_type, season)`. Require one selected version for
every summary and records call. Default to Ridge v2 only after promotion; retain Ridge v1 as a
separate selectable historical model. Keep historical/live tabs separate inside each version.

- [ ] **Step 7: Verify**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_build_v2_tracker.py tests/test_web_v2_service.py tests/test_web_tracker_service.py tests/test_web_tracker_page.py tests/test_web_runtime.py tests/test_webapp.py -v
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
```

- [ ] **Step 8: Build the reviewed v2 historical ledger and commit**

```powershell
.\.venv\Scripts\python.exe scripts/build_v2_tracker.py --write
git add scripts/build_v2_tracker.py src/nfl_game/tracking/ledger.py src/nfl_game/web/v2_service.py src/nfl_game/web/tracker_service.py src/nfl_game/web/tracker_page.py src/nfl_game/web/app.py src/nfl_game/web/runtime.py scripts/game_app.py tests/test_build_v2_tracker.py tests/test_web_v2_service.py tests/test_web_tracker_service.py tests/test_web_tracker_page.py tests/test_web_runtime.py tests/test_webapp.py data/processed/tracker_ledger_ridge_v2.parquet
git commit -m "feat: publish isolated ridge v2 model views"
```

---

### Task 17: Conditionally package, automate, deploy, and smoke-test Ridge v2

**Precondition:** Task 16 passed and the user explicitly approved deployment.

**Files:**
- Modify: `Dockerfile`
- Modify: `.github/workflows/refresh-2026-model.yml`
- Modify: `.github/workflows/update-2026-tracker.yml`
- Modify: `README.md`
- Modify: `CLAUDE.md`
- Modify: `tests/test_operations_cli.py`
- Modify: `tests/test_smoke.py`

**Interfaces:**
- Packages all promoted v2 artifacts and keeps tracker write gating disabled unless separately
  approved near the publication window.

- [ ] **Step 1: Write failing packaging and workflow tests**

Assert Docker includes every required v2 artifact, the model workflow runs the v2 dry-run and
tests before committing generated files, tracker workflow targets the correct versioned ledger,
and `ENABLE_OFFICIAL_TRACKER` remains the only write gate.

- [ ] **Step 2: Run and confirm failures**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_operations_cli.py tests/test_smoke.py -v
```

- [ ] **Step 3: Update packaging, workflows, and operator documentation**

The model refresh workflow must stage only the approved v2 feature/manifest artifacts plus the
existing schedule. The tracker workflow must not write v2 live rows until the existing Stage-2
review supplies a separate explicit approval. Document rollback as selecting Ridge v1 and
restoring the last reviewed v2 artifacts; do not rewrite ledger history.

- [ ] **Step 4: Run local release verification**

```powershell
.\.venv\Scripts\python.exe -m ruff format --check .
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts/backtest.py --test-seasons 2021-2025
.\.venv\Scripts\python.exe scripts/backtest_v2.py --require-research-gates
.\.venv\Scripts\python.exe scripts/refresh_v2_2026.py --dry-run --require-promotion
git diff --check
```

Expected: Ridge v1 is exact; every v2 promotion gate passes; refresh is valid; all tests and
format/lint checks pass.

- [ ] **Step 5: Commit release wiring**

```powershell
git add Dockerfile .github/workflows/refresh-2026-model.yml .github/workflows/update-2026-tracker.yml README.md CLAUDE.md tests/test_operations_cli.py tests/test_smoke.py
git commit -m "ci: package promoted ridge v2 artifacts"
```

- [ ] **Step 6: Request merge/push/deploy approval**

Present branch commits, exact tests, v1/v2 metrics, artifact digests, workflow diff, and rollback
procedure. Merge, push, workflow dispatch, and deployment require the user's explicit approval.

- [ ] **Step 7: After approval, monitor and smoke-test**

Verify workflow completion, remote commit identity, deployment health, authenticated dashboard
Ridge-v2 label/predictions, tracker version separation, v1 exact historical records, v2 exact
historical records, and disabled live writes unless Stage 2 has separately been approved.

---

### Task 18: Evaluate FTN charting as a separate research-only experiment

**Precondition:** Complete Task 14 first. This task cannot delay the core promotion decision.

**Files:**
- Create: `src/nfl_game/ratings/ftn.py`
- Create: `tests/test_ftn.py`
- Create: `scripts/backtest_v2_ftn.py`
- Create: `tests/test_backtest_v2_ftn.py`
- Modify: `README.md`

**Interfaces:**
- Produces candidate `E1` research features and report; never produces production artifacts.

- [ ] **Step 1: Write failing FTN aggregation and as-of tests**

Aggregate 2022+ charting fields by offense/team-week: motion, play action, RPO, screens,
out-of-pocket, interception-worthy throws, catchable balls, drops, blitzers, pass rushers, and
QB-fault sacks. Assert future games cannot change prior target features and include an
`ftn_imputed` flag.

- [ ] **Step 2: Run and confirm failure**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_ftn.py tests/test_backtest_v2_ftn.py -v
```

- [ ] **Step 3: Create research-only aggregation and CLI**

```python
FTN_FEATURE_COLS = (
    "ftn_motion_rate", "ftn_play_action_rate", "ftn_rpo_rate", "ftn_screen_rate",
    "ftn_out_of_pocket_rate", "ftn_int_worthy_rate", "ftn_catchable_rate",
    "ftn_drop_rate", "ftn_blitzers_mean", "ftn_pass_rushers_mean",
    "ftn_qb_fault_sack_rate", "ftn_imputed",
)
```

Train only where the 2022+ history supports the declared inner eligibility rule. Label every
result `E1-research`; prohibit output paths used by Tasks 13–17.

- [ ] **Step 4: Verify and run the experiment**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_ftn.py tests/test_backtest_v2_ftn.py -v
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe scripts/backtest_v2_ftn.py --dry-run
```

- [ ] **Step 5: Commit code and documented findings, not generated production artifacts**

```powershell
git add src/nfl_game/ratings/ftn.py tests/test_ftn.py scripts/backtest_v2_ftn.py tests/test_backtest_v2_ftn.py README.md
git commit -m "research: evaluate ftn ridge v2 signals"
```

---

## Execution Checkpoints

- Tasks 1–3 establish immutable contracts and new data boundaries.
- Tasks 4–8 add one independently reviewable signal block at a time.
- Tasks 9–12 assemble and evaluate the model without production changes.
- Tasks 13–14 build the locked artifacts and stop after research gates 1–10.
- Task 15 is the separately approved shadow build and final gate; Tasks 16–17 require new approvals.
- Task 18 is research-only and can run after Task 14 without affecting the core decision.

## Final Verification

Before calling the project complete:

```powershell
.\.venv\Scripts\python.exe -m ruff format --check .
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts/backtest.py --test-seasons 2021-2025
.\.venv\Scripts\python.exe scripts/backtest_v2.py --require-research-gates
.\.venv\Scripts\python.exe scripts/refresh_v2_2026.py --dry-run --require-promotion
git diff --check
git status --short --branch
```

The completion report must include Ridge-v1 exact preservation, Ridge-v2 selected configs,
every promotion gate, all artifact digests, source freshness/coverage, full test count, workflow
results, deployment health, and authenticated v1/v2 tracker separation.
