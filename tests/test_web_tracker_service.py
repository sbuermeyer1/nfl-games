import json
from decimal import Decimal

import pandas as pd
import pytest

from nfl_game.tracking.ledger import grade_ledger
from nfl_game.web.tracker_service import TrackerInputError, TrackerService


def service_ledger(*, include_live=False, missing_actual_total=False, model_version="ridge-v1"):
    rows = [
        {
            "record_type": "backtest",
            "model_version": model_version,
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
            "actual_total": None if missing_actual_total and season == 2024 else 50.0,
        }
        for season in (2024, 2025)
    ]
    if include_live:
        rows.append(
            {
                "record_type": "live",
                "model_version": model_version,
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
                "closing_spread_line": 3.0,
                "closing_total_line": 44.0,
                "published_at": pd.Timestamp("2026-09-01T12:00:00Z"),
                "kickoff_at": None,
                "actual_margin": 8.0,
                "actual_total": 50.0,
            }
        )
    return grade_ledger(pd.DataFrame(rows))


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


def test_options_keep_historical_seasons_separate_from_live_availability():
    options = TrackerService(service_ledger(include_live=True)).options()

    assert options["historical_seasons"] == [2024, 2025]
    assert options["live_available"] is True


@pytest.mark.parametrize("season", ["", "   ", pd.NA, "2024.0", "2024.5", "twenty-twenty-four"])
def test_summary_rejects_blank_fractional_and_non_numeric_seasons(season):
    with pytest.raises(TrackerInputError, match="season"):
        TrackerService(service_ledger()).summary("backtest", season)


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


def test_advertised_historical_season_returns_its_summary_and_audit_row():
    service = TrackerService(service_ledger())
    season = service.options()["historical_seasons"][0]

    summary = service.summary("backtest", season)
    records = service.records("backtest", season)

    assert summary["season"] == season
    assert summary["qualified"]["spread"]["n_graded"] == 1
    assert [row["season"] for row in records] == [season]


def test_records_convert_every_pandas_missing_value_to_json_none():
    row = TrackerService(service_ledger(missing_actual_total=True)).records("backtest", "2024")[0]

    assert row["actual_total"] is None


def test_records_normalize_decimal_scalars_for_strict_json():
    ledger = service_ledger()
    ledger["actual_total"] = ledger["actual_total"].astype(object)
    ledger.loc[ledger["season"] == 2024, "actual_total"] = Decimal("50.5")

    records = TrackerService(ledger).records("backtest", 2024)

    assert type(records[0]["actual_total"]) is float
    assert records[0]["actual_total"] == 50.5
    assert json.loads(json.dumps(records, allow_nan=False))[0]["actual_total"] == 50.5


def test_rejects_any_model_version_other_than_the_official_historical_version():
    with pytest.raises(ValueError, match="official tracker requires only 'ridge-v1'"):
        TrackerService(service_ledger(model_version="ridge-v2"))


def test_summary_translates_domain_value_errors(monkeypatch):
    service = TrackerService(service_ledger())

    def unavailable_summary(*_args):
        raise ValueError("domain details")

    monkeypatch.setattr("nfl_game.web.tracker_service.summarize_selection", unavailable_summary)
    with pytest.raises(TrackerInputError, match="domain details"):
        service.summary("backtest", "all")
