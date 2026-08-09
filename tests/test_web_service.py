import csv
import io
import math
from datetime import UTC, datetime

import pandas as pd
import pytest

from nfl_game.market.live import MarketSnapshot, MarketUnavailableError
from nfl_game.model.features import FEATURE_COLS
from nfl_game.model.predict import DEFAULT_ALPHA
from nfl_game.web.service import (
    SlateInputError,
    SlateNotFoundError,
    SlateService,
    SlateUnavailableError,
)


def feature_rows() -> pd.DataFrame:
    rows = []
    for season, weeks in ((2024, (1, 2)), (2025, (1, 3))):
        for week in weeks:
            row = {column: 0.1 for column in FEATURE_COLS}
            row.update(
                game_id=f"{season}_{week:02d}_AAA_BBB",
                season=season,
                week=week,
                away_team="AAA",
                home_team="BBB",
                spread_line=2.5,
                total_line=44.5,
                margin=3.0,
                total_points=45.0,
            )
            rows.append(row)
    return pd.DataFrame(rows)


def feature_rows_with_2026_weeks(weeks=(1, 2)) -> pd.DataFrame:
    rows = feature_rows()
    additions = []
    for week in weeks:
        row = {column: 0.1 for column in FEATURE_COLS}
        row.update(
            game_id=f"2026_{week:02d}_AAA_BBB",
            season=2026,
            week=week,
            away_team="AAA",
            home_team="BBB",
            spread_line=2.5,
            total_line=44.5,
            margin=float("nan"),
            total_points=float("nan"),
        )
        additions.append(row)
    return pd.concat([rows, pd.DataFrame(additions)], ignore_index=True)


FIXED_NOW = datetime(2026, 9, 1, 12, tzinfo=UTC)


def packaged_schedule(weeks=(1, 2)) -> pd.DataFrame:
    rows = []
    for week in weeks:
        rows.append(
            {
                "game_id": f"2026_{week:02d}_AAA_BBB",
                "season": 2026,
                "week": week,
                "away_team": "AAA",
                "home_team": "BBB",
                "spread_line": 2.5,
                "total_line": 44.5,
                "kickoff_at": pd.Timestamp(f"2026-09-{10 + week:02d}T17:00:00Z"),
                "result": float("nan"),
                "total": float("nan"),
            }
        )
    return pd.DataFrame(rows)


def market_snapshot(
    *, spread_line=4.5, total_line=47.0, observed_at="2026-09-01T12:00:00Z"
):
    rows = packaged_schedule((1,))
    rows.loc[0, "spread_line"] = spread_line
    rows.loc[0, "total_line"] = total_line
    return MarketSnapshot(rows=rows, observed_at=pd.Timestamp(observed_at).to_pydatetime())


class FakeProvider:
    def __init__(self, snapshot):
        self.value = snapshot
        self.calls = []

    def snapshot(self, season):
        self.calls.append(season)
        return self.value


class FailingProvider:
    def snapshot(self, season):
        raise MarketUnavailableError("market feed unavailable")


def test_options_default_to_latest_packaged_week():
    service = SlateService(feature_rows())
    assert service.options() == {
        "seasons": [2024, 2025],
        "weeks": [1, 3],
        "estimators": ["gbm", "ridge"],
        "default_estimator": "ridge",
        "default_edge_threshold": 2.0,
        "latest": {"season": 2025, "week": 3},
    }


@pytest.mark.parametrize("threshold", [-0.1, math.inf, -math.inf, math.nan])
def test_threshold_must_be_finite_and_non_negative(threshold):
    with pytest.raises(SlateInputError, match="edge threshold"):
        SlateService(feature_rows()).slate(2025, 1, "ridge", threshold)


@pytest.mark.parametrize("threshold", [None, "2", True])
def test_threshold_must_be_a_non_boolean_number(threshold):
    with pytest.raises(SlateInputError, match="edge threshold"):
        SlateService(feature_rows()).slate(2025, 1, "ridge", threshold)


def test_season_week_pair_must_exist():
    with pytest.raises(SlateInputError, match="week 2 is not available for season 2025"):
        SlateService(feature_rows()).slate(2025, 2, "ridge", 2.0)


def test_rejects_a_dataset_missing_required_columns():
    with pytest.raises(ValueError, match="game features missing required columns"):
        SlateService(feature_rows().drop(columns="margin"))


def test_rejects_an_empty_dataset_with_the_required_schema():
    empty = feature_rows().iloc[0:0]
    with pytest.raises(ValueError, match="game features dataset is empty"):
        SlateService(empty)


def test_rejects_duplicate_game_ids():
    rows = feature_rows()
    duplicate = pd.concat([rows, rows.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate game_id"):
        SlateService(duplicate)


@pytest.mark.parametrize("column", ["game_id", "away_team", "home_team"])
@pytest.mark.parametrize("value", [None, "", "   "])
def test_rejects_null_or_blank_game_identity_values(column, value):
    """Catch incomplete identifiers or matchups being accepted as a healthy dataset."""
    rows = feature_rows()
    rows.loc[0, column] = value

    with pytest.raises(
        ValueError,
        match=rf"^game features column '{column}' contains null or blank values$",
    ):
        SlateService(rows)


@pytest.mark.parametrize(
    ("column", "value", "expected"),
    [
        ("season", None, "contains null values"),
        ("week", "one", "must contain numeric values"),
        ("season", math.inf, "contains non-finite values"),
        ("week", 1.5, "contains fractional values"),
        ("season", 0, "contains non-positive values"),
        ("week", -1, "contains non-positive values"),
    ],
)
def test_rejects_invalid_season_and_week_values(column, value, expected):
    """Catch selectors that cannot safely become positive integer route options."""
    rows = feature_rows()
    rows[column] = rows[column].astype(object)
    rows.loc[0, column] = value

    with pytest.raises(
        ValueError,
        match=rf"^game features column '{column}' {expected}$",
    ):
        SlateService(rows)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, "contains null values"),
        ("not-a-number", "must contain numeric values"),
        (math.inf, "contains non-finite values"),
        (-math.inf, "contains non-finite values"),
    ],
)
def test_rejects_invalid_model_feature_values(value, expected):
    """Catch feature values that the fitted estimators cannot safely consume."""
    column = FEATURE_COLS[0]
    rows = feature_rows()
    rows[column] = rows[column].astype(object)
    rows.loc[0, column] = value

    with pytest.raises(
        ValueError,
        match=rf"^game feature column '{column}' {expected}$",
    ):
        SlateService(rows)


@pytest.mark.parametrize("column", ["spread_line", "total_line", "margin", "total_points"])
def test_line_and_target_columns_allow_missing_values(column):
    """Catch startup validation that rejects legitimate missing market or target values."""
    rows = feature_rows()
    rows[column] = rows[column].astype(object)
    rows.loc[0, column] = None

    SlateService(rows)


@pytest.mark.parametrize("column", ["spread_line", "total_line", "margin", "total_points"])
@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("not-a-number", "must contain numeric values or null"),
        (math.inf, "contains infinite values"),
        (-math.inf, "contains infinite values"),
    ],
)
def test_rejects_invalid_line_and_target_values(column, value, expected):
    """Catch invalid market/target values while preserving the missing-value contract."""
    rows = feature_rows()
    rows[column] = rows[column].astype(object)
    rows.loc[0, column] = value

    with pytest.raises(
        ValueError,
        match=rf"^game line/target column '{column}' {expected}$",
    ):
        SlateService(rows)


def test_requires_prior_season_data_to_build_a_bundle():
    with pytest.raises(SlateUnavailableError, match="no calibration data is available"):
        SlateService(feature_rows()).slate(2024, 1, "ridge", 2.0)


def fake_fitted_service(monkeypatch, spread_line=2.5, total_line=44.5):
    rows = feature_rows()
    rows.loc[rows["season"] == 2025, "spread_line"] = spread_line
    rows.loc[rows["season"] == 2025, "total_line"] = total_line
    service = SlateService(rows)
    calls = {"fit": 0}

    class FakeModel:
        def __init__(self, estimator, alpha):
            self.estimator = estimator

        def fit(self, train):
            calls["fit"] += 1
            return self

        def predict(self, target):
            return pd.DataFrame(
                {
                    "game_id": target["game_id"].to_numpy(),
                    "model_margin": [4.0] * len(target),
                    "model_total": [46.0] * len(target),
                }
            )

    class FakeCalibrator:
        def fit(self, oos):
            return self

        def predict(self, merged):
            cover = [float("nan") if pd.isna(value) else 0.6 for value in merged["spread_line"]]
            over = [float("nan") if pd.isna(value) else 0.55 for value in merged["total_line"]]
            return pd.DataFrame(
                {
                    "game_id": merged["game_id"].to_numpy(),
                    "cover_prob": cover,
                    "over_prob": over,
                }
            )

    monkeypatch.setattr("nfl_game.web.service.GameModel", FakeModel)
    monkeypatch.setattr("nfl_game.web.service.Calibrator", FakeCalibrator)
    monkeypatch.setattr(
        "nfl_game.web.service.walk_forward",
        lambda features, seasons, estimator, alpha: features.assign(
            model_margin=3.0, model_total=45.0
        ),
    )
    return service, calls


def fake_fitted_2026_service(monkeypatch, provider=None):
    _, calls = fake_fitted_service(monkeypatch)
    service = SlateService(
        feature_rows_with_2026_weeks(),
        packaged_schedule=packaged_schedule(),
        market_provider=provider,
        clock=lambda: FIXED_NOW,
    )
    return service, calls


def test_reuses_bundle_across_weeks(monkeypatch):
    service = SlateService(feature_rows())
    calls = {"fit": 0}

    class FakeModel:
        def __init__(self, estimator, alpha):
            self.estimator = estimator

        def fit(self, train):
            calls["fit"] += 1
            return self

        def predict(self, target):
            return pd.DataFrame(
                {
                    "game_id": target["game_id"],
                    "model_margin": 4.0,
                    "model_total": 46.0,
                }
            )

    class FakeCalibrator:
        def fit(self, oos):
            return self

        def predict(self, merged):
            return pd.DataFrame(
                {
                    "game_id": merged["game_id"],
                    "cover_prob": 0.6,
                    "over_prob": 0.55,
                }
            )

    monkeypatch.setattr("nfl_game.web.service.GameModel", FakeModel)
    monkeypatch.setattr("nfl_game.web.service.Calibrator", FakeCalibrator)
    monkeypatch.setattr(
        "nfl_game.web.service.walk_forward",
        lambda features, seasons, estimator, alpha: features.assign(
            model_margin=3.0, model_total=45.0
        ),
    )

    service.slate(2025, 1, "ridge", 2.0)
    service.slate(2025, 3, "ridge", 3.0)
    assert calls["fit"] == 1


def test_estimator_has_its_own_cache_key(monkeypatch):
    service, calls = fake_fitted_service(monkeypatch)
    service.slate(2025, 1, "ridge", 2.0)
    service.slate(2025, 1, "gbm", 2.0)
    assert calls["fit"] == 2


def test_concurrent_requests_fit_one_bundle(monkeypatch):
    import time
    from concurrent.futures import ThreadPoolExecutor

    service = SlateService(feature_rows())
    calls = {"fit": 0}
    bundle = object()

    def slow_fit(season, estimator):
        calls["fit"] += 1
        time.sleep(0.05)
        return bundle

    monkeypatch.setattr(service, "_fit_bundle", slow_fit)
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(lambda _: service._bundle(2025, "ridge"), range(8)))
    assert calls["fit"] == 1
    assert all(result is bundle for result in results)


def test_records_convert_nan_to_none(monkeypatch):
    service, _ = fake_fitted_service(monkeypatch, spread_line=float("nan"))
    row = service.records(2025, 1, "ridge", 2.0)[0]
    assert row["market_spread"] is None
    assert row["spread_gap"] is None
    assert row["cover_prob"] is None


def test_csv_uses_same_rows_and_blanks_missing_values(monkeypatch):
    service, _ = fake_fitted_service(monkeypatch, total_line=float("nan"))
    records = service.records(2025, 1, "ridge", 2.0)
    csv_text = service.csv(2025, 1, "ridge", 2.0)
    csv_rows = list(csv.DictReader(io.StringIO(csv_text)))

    assert list(csv_rows[0]) == list(records[0])
    assert csv_rows == [
        {key: "" if value is None else str(value) for key, value in row.items()} for row in records
    ]
    assert csv_rows[0]["market_total"] == ""
    assert csv_rows[0]["total_gap"] == ""
    assert csv_rows[0]["over_prob"] == ""


def test_bundle_uses_prior_seasons_selected_estimator_and_default_alpha(monkeypatch):
    service = SlateService(feature_rows())
    calls = {"model_init": [], "walk_forward": []}

    class FakeModel:
        def __init__(self, estimator, alpha):
            calls["model_init"].append((estimator, alpha))

        def fit(self, train):
            return self

        def predict(self, target):
            return pd.DataFrame(
                {
                    "game_id": target["game_id"].to_numpy(),
                    "model_margin": [4.0] * len(target),
                    "model_total": [46.0] * len(target),
                }
            )

    class FakeCalibrator:
        def fit(self, oos):
            return self

        def predict(self, merged):
            return pd.DataFrame(
                {
                    "game_id": merged["game_id"].to_numpy(),
                    "cover_prob": [0.6] * len(merged),
                    "over_prob": [0.55] * len(merged),
                }
            )

    def fake_walk_forward(features, seasons, estimator, alpha):
        calls["walk_forward"].append((list(seasons), estimator, alpha))
        return features.assign(model_margin=3.0, model_total=45.0)

    monkeypatch.setattr("nfl_game.web.service.GameModel", FakeModel)
    monkeypatch.setattr("nfl_game.web.service.Calibrator", FakeCalibrator)
    monkeypatch.setattr("nfl_game.web.service.walk_forward", fake_walk_forward)

    service.slate(2025, 1, "gbm", 2.0)

    assert calls["walk_forward"] == [([2024], "gbm", DEFAULT_ALPHA)]
    assert calls["model_init"] == [("gbm", DEFAULT_ALPHA)]


def test_raises_not_found_when_a_valid_slate_has_no_model_rows(monkeypatch):
    service = SlateService(feature_rows())

    class EmptyModel:
        def predict(self, target):
            return pd.DataFrame(columns=["game_id", "model_margin", "model_total"])

    class UnusedCalibrator:
        def predict(self, merged):
            return pd.DataFrame(columns=["game_id", "cover_prob", "over_prob"])

    class EmptyBundle:
        model = EmptyModel()
        calibrator = UnusedCalibrator()

    monkeypatch.setattr(service, "_bundle", lambda season, estimator: EmptyBundle())
    with pytest.raises(SlateNotFoundError, match="no games are available"):
        service.slate(2025, 1, "ridge", 2.0)


def test_options_default_to_earliest_unplayed_2026_week():
    service = SlateService(
        feature_rows_with_2026_weeks(),
        packaged_schedule=packaged_schedule(),
        clock=lambda: FIXED_NOW,
    )

    options = service.options()

    assert options["latest"] == {"season": 2026, "week": 1}
    assert options["weeks"] == [1, 2]


def test_payload_overlays_live_markets_without_changing_model_predictions(monkeypatch):
    provider = FakeProvider(market_snapshot(spread_line=4.5, total_line=47.0))
    service, _ = fake_fitted_2026_service(monkeypatch, provider)

    raw_predictions = service.model_predictions(2026, 1, "ridge")
    body = service.payload(2026, 1, "ridge", 2.0)

    assert raw_predictions.loc[0, "model_margin"] == 4.0
    assert provider.calls == [2026]
    assert body["games"][0]["model_spread"] == 4.0
    assert body["games"][0]["market_spread"] == 4.5
    assert body["games"][0]["market_total"] == 47.0
    assert body["market"] == {
        "source": "nflverse",
        "observed_at": "2026-09-01T12:00:00+00:00",
        "stale": False,
    }


def test_model_predictions_exposes_only_raw_model_columns(monkeypatch):
    """Catch market or pick fields leaking into the lifecycle prediction boundary."""
    service, _ = fake_fitted_2026_service(monkeypatch)

    class ExtraColumnModel:
        def predict(self, target):
            return pd.DataFrame(
                {
                    "game_id": target["game_id"].to_numpy(),
                    "model_margin": [4.0] * len(target),
                    "model_total": [46.0] * len(target),
                    "market_spread": [99.0] * len(target),
                    "spread_pick": ["home"] * len(target),
                }
            )

    bundle = service._bundle(2026, "ridge")
    monkeypatch.setattr(bundle.model, "predict", ExtraColumnModel().predict)
    predictions = service.model_predictions(2026, 1)

    assert predictions.columns.tolist() == ["game_id", "model_margin", "model_total"]


def test_successful_feed_missing_one_market_does_not_use_packaged_value(monkeypatch):
    provider = FakeProvider(market_snapshot(spread_line=None, total_line=47.0))
    service, _ = fake_fitted_2026_service(monkeypatch, provider)

    game = service.payload(2026, 1, "ridge", 2.0)["games"][0]

    assert game["market_spread"] is None
    assert game["spread_market_status"] == "missing"
    assert game["market_total"] == 47.0
    assert game["total_market_status"] == "live"


def test_cold_feed_failure_uses_packaged_lines_as_stale(monkeypatch):
    service, _ = fake_fitted_2026_service(monkeypatch, FailingProvider())

    body = service.payload(2026, 1, "ridge", 2.0)

    assert body["market"]["source"] == "packaged"
    assert body["market"]["stale"] is True
    assert body["games"][0]["market_spread"] == 2.5
    assert body["games"][0]["spread_market_status"] == "stale"


def test_csv_uses_one_market_snapshot_and_blanks_missing_live_values(monkeypatch):
    provider = FakeProvider(market_snapshot(spread_line=None, total_line=47.0))
    service, _ = fake_fitted_2026_service(monkeypatch, provider)

    csv_rows = list(csv.DictReader(io.StringIO(service.csv(2026, 1, "ridge", 2.0))))

    assert provider.calls == [2026]
    assert csv_rows[0]["market_spread"] == ""
    assert csv_rows[0]["spread_market_status"] == "missing"
    assert csv_rows[0]["market_total"] == "47.0"


def test_market_snapshot_team_identity_must_match_features(monkeypatch):
    snapshot = market_snapshot()
    snapshot.rows.loc[0, "home_team"] = "CCC"
    service, _ = fake_fitted_2026_service(monkeypatch, FakeProvider(snapshot))

    with pytest.raises(SlateUnavailableError, match="identity"):
        service.payload(2026, 1, "ridge", 2.0)


@pytest.mark.parametrize("column", ["away_team", "home_team"])
def test_market_snapshot_team_identity_must_not_be_missing(monkeypatch, column):
    snapshot = market_snapshot()
    snapshot.rows[column] = snapshot.rows[column].astype("string")
    snapshot.rows.loc[0, column] = pd.NA
    service, _ = fake_fitted_2026_service(monkeypatch, FakeProvider(snapshot))

    with pytest.raises(SlateUnavailableError, match="identity"):
        service.payload(2026, 1, "ridge", 2.0)


def test_schedule_records_uses_one_snapshot_and_json_safe_lines(monkeypatch):
    provider = FakeProvider(market_snapshot(spread_line=None))
    service, _ = fake_fitted_2026_service(monkeypatch, provider)

    body = service.schedule_records(2026)

    assert provider.calls == [2026]
    assert body["season"] == 2026
    assert body["games"][0]["spread_line"] is None
    assert body["market"]["source"] == "nflverse"
