from fastapi.testclient import TestClient

from nfl_game.web.app import create_app
from nfl_game.web.service import (
    SlateInputError,
    SlateNotFoundError,
    SlateUnavailableError,
)


class FakeService:
    def options(self):
        return {
            "seasons": [2024, 2025],
            "weeks": [1, 3],
            "estimators": ["gbm", "ridge"],
            "default_estimator": "ridge",
            "default_edge_threshold": 2.0,
            "latest": {"season": 2025, "week": 3},
        }

    def weeks(self, season):
        if season != 2025:
            raise SlateInputError(f"season {season} is not available")
        return [1, 3]

    def records(self, season, week, estimator, edge_threshold):
        if week == 2:
            raise SlateInputError("week 2 is not available for season 2025")
        if week == 3 and estimator == "gbm":
            raise SlateUnavailableError("cannot train gbm for season 2025")
        if week == 4:
            raise SlateNotFoundError("no games are available for season 2025 week 4")
        if week == 5:
            raise RuntimeError("database password leaked in internal trace")
        return [
            {
                "game_id": "2025_01_AAA_BBB",
                "season": season,
                "week": week,
                "away_team": "AAA",
                "home_team": "BBB",
                "model_spread": 4.0,
                "market_spread": 2.5,
                "spread_gap": 1.5,
                "cover_prob": 0.55,
                "model_total": 46.0,
                "market_total": None,
                "total_gap": None,
                "over_prob": None,
                "edge_flag": 0,
            }
        ]

    def csv(self, season, week, estimator, edge_threshold):
        return "game_id,season,week\n2025_01_AAA_BBB,2025,1\n"


def client():
    return TestClient(create_app(FakeService(), access_code=None))


def test_page_contains_all_controls_and_disclaimer():
    html = client().get("/").text
    for control_id in ("season", "week", "estimator", "edge", "run", "download"):
        assert f'id="{control_id}"' in html
    assert "home-team margins" in html
    assert "not betting advice" in html


def test_options_and_weeks_routes():
    assert client().get("/api/options").json()["latest"] == {"season": 2025, "week": 3}
    assert client().get("/api/weeks", params={"season": 2025}).json() == {"weeks": [1, 3]}


def test_slate_json_preserves_nulls():
    response = client().get(
        "/api/slate",
        params={"season": 2025, "week": 1, "estimator": "ridge", "edge_threshold": 2.0},
    )
    assert response.status_code == 200
    assert response.json()["games"][0]["market_total"] is None


def test_csv_has_matching_filename_and_content_type():
    response = client().get(
        "/api/slate.csv",
        params={"season": 2025, "week": 1, "estimator": "ridge", "edge_threshold": 2.0},
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    assert 'filename="slate_2025_wk01_ridge.csv"' in response.headers["content-disposition"]


def test_input_unavailable_and_empty_errors_are_client_safe():
    bad_week = client().get(
        "/api/slate",
        params={"season": 2025, "week": 2, "estimator": "ridge", "edge_threshold": 2.0},
    )
    unavailable = client().get(
        "/api/slate",
        params={"season": 2025, "week": 3, "estimator": "gbm", "edge_threshold": 2.0},
    )
    empty = client().get(
        "/api/slate",
        params={"season": 2025, "week": 4, "estimator": "ridge", "edge_threshold": 2.0},
    )
    assert bad_week.status_code == 422
    assert bad_week.json() == {"error": "week 2 is not available for season 2025"}
    assert unavailable.status_code == 409
    assert empty.status_code == 404
    assert "traceback" not in unavailable.text.lower()


def test_unexpected_error_is_generic_and_hides_internal_message():
    safe_client = TestClient(
        create_app(FakeService(), access_code=None),
        raise_server_exceptions=False,
    )
    response = safe_client.get(
        "/api/slate",
        params={"season": 2025, "week": 5, "estimator": "ridge", "edge_threshold": 2.0},
    )
    assert response.status_code == 500
    assert response.json() == {"error": "Unexpected server error"}
    assert "database password" not in response.text
