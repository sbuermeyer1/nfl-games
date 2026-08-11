import json
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from nfl_game.data.source_manifest import (
    SourceContractError,
    SourceSnapshot,
    numeric_coverage,
    read_source_manifest,
    require_coverage,
    schema_fingerprint,
    write_json_atomic,
)


def snapshot(name="weekly_rosters", retrieved_at=datetime(2024, 9, 1, tzinfo=UTC)):
    return SourceSnapshot(
        name=name,
        seasons=(2023, 2024),
        retrieved_at=retrieved_at,
        schema_sha256="a" * 64,
        rows=32,
        coverage={"rating": 1.0},
        latest_event_at=datetime(2024, 9, 1, tzinfo=UTC),
    )


def snapshot_payload(*snapshots):
    return {
        "snapshots": [
            {
                "name": item.name,
                "seasons": list(item.seasons),
                "retrieved_at": item.retrieved_at.isoformat(),
                "schema_sha256": item.schema_sha256,
                "rows": item.rows,
                "coverage": item.coverage,
                "latest_event_at": item.latest_event_at.isoformat()
                if item.latest_event_at is not None
                else None,
            }
            for item in snapshots
        ]
    }


def test_schema_fingerprint_is_column_order_independent():
    a = pd.DataFrame({"team": pd.Series(["BUF"], dtype="string"), "value": [1.0]})
    b = a[["value", "team"]]

    assert schema_fingerprint(a) == schema_fingerprint(b)


@pytest.mark.parametrize("value", [np.inf, -np.inf, "not-a-number"])
def test_numeric_coverage_rejects_non_finite_or_non_numeric_non_null_values(value):
    frame = pd.DataFrame({"rating": [1.0, value]})

    with pytest.raises(SourceContractError, match="non-numeric or non-finite values in rating"):
        numeric_coverage(frame, ["rating"])


def test_require_coverage_rejects_below_ninety_percent():
    frame = pd.DataFrame(
        {
            "season": [2024] * 10,
            "week": [1] * 10,
            "team": list("ABCDEFGHIJ"),
            "rating": [1.0] * 8 + [np.nan] * 2,
        }
    )

    with pytest.raises(SourceContractError, match="0.8000"):
        require_coverage(frame, ["rating"], minimum=0.90)


def test_require_coverage_rejects_a_season_masked_by_aggregate_coverage():
    frame = pd.DataFrame(
        {
            "season": [2023] * 10 + [2024] * 10,
            "rating": [1.0] * 18 + [np.nan] * 2,
        }
    )

    with pytest.raises(SourceContractError, match=r"season 2024.*0\.8000"):
        require_coverage(frame, ["rating"], minimum=0.90)

def test_numeric_coverage_rejects_missing_columns():
    with pytest.raises(SourceContractError, match=r"missing source columns: \['rating'\]"):
        numeric_coverage(pd.DataFrame({"team": ["BUF"]}), ["rating"])


def test_source_snapshot_rejects_blank_name_and_naive_timestamps():
    with pytest.raises(SourceContractError, match="source snapshot name must not be blank"):
        snapshot(name="  ")

    with pytest.raises(SourceContractError, match="must be UTC"):
        snapshot(retrieved_at=datetime(2024, 9, 1, tzinfo=UTC).replace(tzinfo=None))


def test_read_source_manifest_round_trips_utc_snapshots_and_rejects_duplicate_names(tmp_path):
    path = tmp_path / "sources.json"
    expected = (snapshot(), snapshot("snap_counts"))
    write_json_atomic(snapshot_payload(*expected), path)

    assert read_source_manifest(path) == expected

    write_json_atomic(snapshot_payload(snapshot(), snapshot()), path)
    with pytest.raises(SourceContractError, match="duplicate source snapshot names"):
        read_source_manifest(path)


def test_read_source_manifest_rejects_non_utc_timestamp(tmp_path):
    path = tmp_path / "sources.json"
    payload = snapshot_payload(snapshot())
    payload["snapshots"][0]["retrieved_at"] = "2024-09-01T00:00:00+01:00"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(SourceContractError, match="must be UTC"):
        read_source_manifest(path)


def test_read_source_manifest_rejects_snapshot_values_that_cannot_round_trip(tmp_path):
    path = tmp_path / "sources.json"
    payload = snapshot_payload(snapshot())
    payload["snapshots"][0]["schema_sha256"] = []
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(SourceContractError, match="schema_sha256 must be a string"):
        read_source_manifest(path)

def test_read_source_manifest_rejects_list_shaped_coverage(tmp_path):
    path = tmp_path / "sources.json"
    payload = snapshot_payload(snapshot())
    payload["snapshots"][0]["coverage"] = [["rating", 1.0]]
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(SourceContractError, match="coverage must be a dictionary"):
        read_source_manifest(path)

def test_write_json_atomic_preserves_existing_file_when_replacement_fails(tmp_path, monkeypatch):
    path = tmp_path / "sources.json"
    path.write_text('{"generation":"old"}', encoding="utf-8")

    def fail_replace(self, target):
        raise OSError("replacement failed")

    monkeypatch.setattr(type(path), "replace", fail_replace)

    with pytest.raises(OSError, match="replacement failed"):
        write_json_atomic({"generation": "new"}, path)

    assert path.read_text(encoding="utf-8") == '{"generation":"old"}'
    assert not path.with_suffix(".json.tmp").exists()
