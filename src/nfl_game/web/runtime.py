"""Fail-closed runtime configuration for the NFL slate dashboard."""

from collections.abc import Mapping
from dataclasses import dataclass
from ipaddress import ip_address
from pathlib import Path

from nfl_game.web.app import create_app
from nfl_game.web.service import SlateService


class RuntimeConfigError(RuntimeError):
    """The dashboard cannot safely start with the supplied runtime settings."""


@dataclass(frozen=True)
class RuntimeConfig:
    access_code: str | None
    host: str
    port: int

    def __post_init__(self) -> None:
        if self.access_code is None:
            try:
                is_loopback = ip_address(self.host).is_loopback
            except ValueError as exc:
                raise RuntimeConfigError("--no-auth requires a numeric loopback bind host") from exc
            if not is_loopback:
                raise RuntimeConfigError("--no-auth requires a numeric loopback bind host")


def resolve_runtime(no_auth: bool, environ: Mapping[str, str]) -> RuntimeConfig:
    """Resolve fail-closed web settings from the environment and local-only flag."""
    raw_code = environ.get("ACCESS_CODE", "")
    access_code = raw_code.strip() or None
    if no_auth and access_code is not None:
        raise RuntimeConfigError("--no-auth cannot be combined with ACCESS_CODE")
    if not no_auth and access_code is None:
        raise RuntimeConfigError(
            "ACCESS_CODE is required; use --no-auth only for loopback local development"
        )
    try:
        port = int(environ.get("PORT", "8000"))
    except ValueError as exc:
        raise RuntimeConfigError("PORT must be an integer") from exc
    if not 1 <= port <= 65535:
        raise RuntimeConfigError("PORT must be between 1 and 65535")
    return RuntimeConfig(
        access_code=None if no_auth else access_code,
        host="127.0.0.1" if no_auth else "0.0.0.0",
        port=port,
    )


def load_app(config: RuntimeConfig, dataset_path: str | Path):
    """Load the packaged dataset through the read-only slate service."""
    path = Path(dataset_path)
    if not path.is_file():
        raise RuntimeConfigError(f"packaged dataset not found: {path}")
    try:
        service = SlateService.from_parquet(path)
    except Exception as exc:
        raise RuntimeConfigError(f"cannot load packaged dataset {path}: {exc}") from exc
    return create_app(service, access_code=config.access_code)
