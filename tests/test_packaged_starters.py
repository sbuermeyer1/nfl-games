from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from nfl_game.market.live_starters import StartersUnavailableError
from nfl_game.market.packaged_starters import STALE_AFTER, PackagedStarterProvider
from nfl_game.ratings.starters import ADVISORY_COLS


def _artifact_frame(computed_at=datetime(2026, 9, 8, 10, 30, tzinfo=UTC)):
    return pd.DataFrame(
        {
            "game_id": pd.Series(["2026_01_CIN_BAL", "2026_02_KC_DEN"], dtype="string"),
            "home_qb": pd.Series(["Tyler Huntley", "Patrick Mahomes"], dtype="string"),
            "away_qb": pd.Series(["Joe Burrow", None], dtype="string"),
            "qb_change_epa_home": [-0.31, 0.0],
            "qb_change_epa_away": [0.0, 0.12],
            "qb_watch": pd.Series([1, 0], dtype="Int64"),
            "qb_inferred": pd.Series([0, 1], dtype="Int64"),
            "season": pd.Series([2026, 2026], dtype="int64"),
            "week": pd.Series([1, 2], dtype="int64"),
            "computed_at": pd.Series([computed_at, computed_at]),
        }
    )


def _write_artifact(tmp_path, frame=None, name="starter_advisory.parquet"):
    path = tmp_path / name
    (frame if frame is not None else _artifact_frame()).to_parquet(path, index=False)
    return path


def test_missing_file_raises_unavailable(tmp_path):
    with pytest.raises(StartersUnavailableError, match="not found"):
        PackagedStarterProvider(tmp_path / "missing.parquet")


def test_unreadable_file_raises_unavailable(tmp_path):
    path = tmp_path / "starter_advisory.parquet"
    path.write_bytes(b"not a parquet file")
    with pytest.raises(StartersUnavailableError, match="cannot read"):
        PackagedStarterProvider(path)


def test_wrong_columns_raises_unavailable(tmp_path):
    path = tmp_path / "starter_advisory.parquet"
    pd.DataFrame({"game_id": ["x"]}).to_parquet(path, index=False)
    with pytest.raises(StartersUnavailableError, match="unexpected columns"):
        PackagedStarterProvider(path)


def test_no_rows_for_requested_season_week_raises_unavailable(tmp_path):
    path = _write_artifact(tmp_path)
    provider = PackagedStarterProvider(path)
    with pytest.raises(StartersUnavailableError, match="no packaged starter advisory"):
        provider.snapshot(2026, 3)


def test_empty_artifact_raises_unavailable_on_any_snapshot(tmp_path):
    empty = _artifact_frame().iloc[0:0]
    path = _write_artifact(tmp_path, frame=empty)
    provider = PackagedStarterProvider(path)
    with pytest.raises(StartersUnavailableError):
        provider.snapshot(2026, 1)


def test_snapshot_returns_advisory_cols_only_with_packaged_source(tmp_path):
    path = _write_artifact(tmp_path)
    provider = PackagedStarterProvider(
        path, clock=lambda: datetime(2026, 9, 8, 11, tzinfo=UTC)
    )
    snap = provider.snapshot(2026, 1)
    assert list(snap.rows.columns) == list(ADVISORY_COLS)
    assert snap.rows.iloc[0]["home_qb"] == "Tyler Huntley"
    assert snap.source == "packaged"
    assert snap.stale is False
    assert snap.observed_at == datetime(2026, 9, 8, 10, 30, tzinfo=UTC)


def test_snapshot_filters_to_the_requested_season_and_week(tmp_path):
    path = _write_artifact(tmp_path)
    provider = PackagedStarterProvider(
        path, clock=lambda: datetime(2026, 9, 8, 11, tzinfo=UTC)
    )
    snap = provider.snapshot(2026, 2)
    assert len(snap.rows) == 1
    assert snap.rows.iloc[0]["home_qb"] == "Patrick Mahomes"


def test_stale_is_false_within_the_bound(tmp_path):
    computed_at = datetime(2026, 9, 8, 10, 30, tzinfo=UTC)
    path = _write_artifact(tmp_path, frame=_artifact_frame(computed_at))
    provider = PackagedStarterProvider(
        path, clock=lambda: computed_at + STALE_AFTER - timedelta(minutes=1)
    )
    assert provider.snapshot(2026, 1).stale is False


def test_stale_is_true_past_the_bound(tmp_path):
    computed_at = datetime(2026, 9, 8, 10, 30, tzinfo=UTC)
    path = _write_artifact(tmp_path, frame=_artifact_frame(computed_at))
    provider = PackagedStarterProvider(
        path, clock=lambda: computed_at + STALE_AFTER + timedelta(minutes=1)
    )
    assert provider.snapshot(2026, 1).stale is True


def test_file_is_read_once_at_construction_not_per_request(tmp_path, monkeypatch):
    path = _write_artifact(tmp_path)
    calls = []
    original_read_parquet = pd.read_parquet

    def counting_read_parquet(given_path, *args, **kwargs):
        calls.append(given_path)
        return original_read_parquet(given_path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", counting_read_parquet)
    provider = PackagedStarterProvider(path, clock=lambda: datetime(2026, 9, 8, 11, tzinfo=UTC))
    provider.snapshot(2026, 1)
    provider.snapshot(2026, 1)
    provider.snapshot(2026, 2)
    assert len(calls) == 1
