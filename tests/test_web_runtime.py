import math

import pandas as pd
import pytest

from nfl_game.model.features import FEATURE_COLS
from nfl_game.paths import PROCESSED_DIR
from nfl_game.web.runtime import RuntimeConfig, RuntimeConfigError, load_app, resolve_runtime


def test_default_startup_requires_access_code():
    """Catch a deployment that starts without an access-code gate."""
    with pytest.raises(RuntimeConfigError, match="ACCESS_CODE is required"):
        resolve_runtime(no_auth=False, environ={})


def test_protected_runtime_binds_all_interfaces():
    """Catch protected deployments that do not use their configured public bind."""
    config = resolve_runtime(no_auth=False, environ={"ACCESS_CODE": "letmein", "PORT": "9000"})

    assert config.access_code == "letmein"
    assert config.host == "0.0.0.0"
    assert config.port == 9000


def test_explicit_no_auth_binds_loopback_only():
    """Catch local no-auth startup that is reachable beyond the local machine."""
    config = resolve_runtime(no_auth=True, environ={})

    assert config.access_code is None
    assert config.host == "127.0.0.1"
    assert config.port == 8000


@pytest.mark.parametrize("host", ["127.0.0.1", "::1"])
def test_unprotected_runtime_config_accepts_numeric_loopback_hosts(host):
    """Catch local-only validation that incorrectly rejects a standard loopback family."""
    config = RuntimeConfig(access_code=None, host=host, port=8000)

    assert config.host == host


@pytest.mark.parametrize("host", ["0.0.0.0", "localhost", "::"])
def test_unprotected_runtime_config_rejects_public_or_non_numeric_hosts(host):
    """Catch no-auth validation that permits a wildcard bind or hostname instead of loopback."""
    with pytest.raises(RuntimeConfigError, match="numeric loopback"):
        RuntimeConfig(access_code=None, host=host, port=8000)


def test_no_auth_rejects_access_code_to_avoid_ambiguous_intent():
    """Catch conflicting auth configuration that could select the wrong security mode."""
    with pytest.raises(RuntimeConfigError, match="cannot be combined"):
        resolve_runtime(no_auth=True, environ={"ACCESS_CODE": "letmein"})


def test_blank_access_code_is_missing():
    """Catch whitespace-only deployment secrets being accepted as authentication."""
    with pytest.raises(RuntimeConfigError, match="ACCESS_CODE is required"):
        resolve_runtime(no_auth=False, environ={"ACCESS_CODE": "   "})


def test_invalid_port_is_configuration_error():
    """Catch malformed ports that would otherwise fail unclearly during server startup."""
    with pytest.raises(RuntimeConfigError, match="PORT"):
        resolve_runtime(no_auth=False, environ={"ACCESS_CODE": "letmein", "PORT": "abc"})


@pytest.mark.parametrize("port", ["0", "65536"])
def test_runtime_rejects_ports_outside_tcp_range(port):
    """Catch an off-by-one port check that passes an invalid TCP port to Uvicorn."""
    with pytest.raises(RuntimeConfigError, match="PORT must be between 1 and 65535"):
        resolve_runtime(no_auth=False, environ={"ACCESS_CODE": "letmein", "PORT": port})


def test_load_app_rejects_missing_dataset(tmp_path):
    """Catch startup that attempts to run without the packaged read-only dataset."""
    config = resolve_runtime(no_auth=True, environ={})

    with pytest.raises(RuntimeConfigError, match="packaged dataset not found"):
        load_app(config, tmp_path / "missing.parquet")


def test_load_app_wraps_parquet_read_failure(tmp_path, monkeypatch):
    """Catch corrupt packaged data escaping as an unhelpful startup exception."""
    dataset = tmp_path / "broken.parquet"
    dataset.write_bytes(b"not parquet")

    def fail(path):
        raise ValueError("invalid parquet footer")

    monkeypatch.setattr("nfl_game.web.runtime.SlateService.from_parquet", fail)
    config = resolve_runtime(no_auth=True, environ={})

    with pytest.raises(RuntimeConfigError, match="cannot load packaged dataset") as caught:
        load_app(config, dataset)
    assert "invalid parquet footer" in str(caught.value)


def startup_feature_rows() -> pd.DataFrame:
    rows = []
    for season in (2024, 2025):
        row = {column: 0.1 for column in FEATURE_COLS}
        row.update(
            game_id=f"{season}_01_AAA_BBB",
            season=season,
            week=1,
            away_team="AAA",
            home_team="BBB",
            spread_line=2.5,
            total_line=44.5,
            margin=3.0,
            total_points=45.0,
        )
        rows.append(row)
    return pd.DataFrame(rows)


@pytest.mark.parametrize(
    ("column", "value", "expected"),
    [
        ("game_id", None, "contains null or blank values"),
        ("week", 1.5, "contains fractional values"),
        (FEATURE_COLS[0], "not-a-number", "must contain numeric values"),
        ("total_line", math.inf, "contains infinite values"),
    ],
)
def test_load_app_wraps_invalid_dataset_schema(tmp_path, column, value, expected):
    """Catch schema-invalid packaged data passing startup and breaking a later request."""
    rows = startup_feature_rows()
    if isinstance(value, str):
        rows[column] = value
    elif column == "week":
        rows[column] = rows[column].astype(float)
        rows.loc[0, column] = value
    else:
        rows.loc[0, column] = value
    dataset = tmp_path / "invalid.parquet"
    rows.to_parquet(dataset)
    config = resolve_runtime(no_auth=True, environ={})

    with pytest.raises(RuntimeConfigError, match="cannot load packaged dataset") as caught:
        load_app(config, dataset)

    assert expected in str(caught.value)


def test_entrypoint_refuses_to_start_without_access_code(monkeypatch, capsys):
    """Catch the command-line entry point bypassing the fail-closed runtime guard."""
    monkeypatch.delenv("ACCESS_CODE", raising=False)

    from scripts.game_app import main

    with pytest.raises(SystemExit) as caught:
        main([])

    assert caught.value.code == 2
    assert "ACCESS_CODE is required" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("argv", "environ", "expected_config"),
    [
        (
            [],
            {"ACCESS_CODE": "launch-code", "PORT": "9123"},
            RuntimeConfig(access_code="launch-code", host="0.0.0.0", port=9123),
        ),
        (
            ["--no-auth"],
            {"PORT": "9124"},
            RuntimeConfig(access_code=None, host="127.0.0.1", port=9124),
        ),
    ],
)
def test_entrypoint_passes_resolved_runtime_to_loader_and_server(
    monkeypatch, argv, environ, expected_config
):
    """Catch launcher wiring that serves the wrong app, dataset, host, or port."""
    from scripts import game_app

    monkeypatch.delenv("ACCESS_CODE", raising=False)
    monkeypatch.delenv("PORT", raising=False)
    for name, value in environ.items():
        monkeypatch.setenv(name, value)

    app = object()
    loader_calls = []
    server_calls = []

    def load_app_without_starting_server(config, dataset_path):
        loader_calls.append((config, dataset_path))
        return app

    def record_server_start(server_app, *, host, port):
        server_calls.append((server_app, host, port))

    monkeypatch.setattr(game_app, "load_app", load_app_without_starting_server)
    monkeypatch.setattr(game_app.uvicorn, "run", record_server_start)

    game_app.main(argv)

    assert loader_calls == [(expected_config, PROCESSED_DIR / "game_features.parquet")]
    assert server_calls == [(app, expected_config.host, expected_config.port)]
