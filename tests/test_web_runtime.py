
import pytest

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


def test_unprotected_runtime_config_rejects_non_loopback_host():
    """Catch a manually constructed no-auth config exposing the dashboard publicly."""
    with pytest.raises(RuntimeConfigError, match="numeric loopback"):
        RuntimeConfig(access_code=None, host="0.0.0.0", port=8000)


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


def test_entrypoint_refuses_to_start_without_access_code(monkeypatch, capsys):
    """Catch the command-line entry point bypassing the fail-closed runtime guard."""
    monkeypatch.delenv("ACCESS_CODE", raising=False)

    from scripts.game_app import main

    with pytest.raises(SystemExit) as caught:
        main([])

    assert caught.value.code == 2
    assert "ACCESS_CODE is required" in capsys.readouterr().err
