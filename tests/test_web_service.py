import math

import pandas as pd
import pytest

from nfl_game.model.features import FEATURE_COLS
from nfl_game.web.service import SlateInputError, SlateService


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


def test_season_week_pair_must_exist():
    with pytest.raises(SlateInputError, match="week 2 is not available for season 2025"):
        SlateService(feature_rows()).slate(2025, 2, "ridge", 2.0)


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


def test_csv_uses_same_rows_and_never_writes_nan(monkeypatch):
    service, _ = fake_fitted_service(monkeypatch, total_line=float("nan"))
    csv_text = service.csv(2025, 1, "ridge", 2.0)
    assert csv_text.startswith("game_id,season,week,away_team,home_team")
    assert "nan" not in csv_text.lower()
