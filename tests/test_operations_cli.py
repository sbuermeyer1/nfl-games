import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLI_CASES = [
    ("scripts/refresh_2026.py", ["--dry-run", "--write"]),
    (
        "scripts/update_live_tracker.py",
        ["--dry-run", "--write", "--now", "--void-game"],
    ),
]


@pytest.mark.parametrize(("script", "expected_flags"), CLI_CASES)
def test_documented_cli_help_matches_supported_flags(script, expected_flags):
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / script), "--help"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    for flag in expected_flags:
        assert flag in result.stdout


@pytest.mark.parametrize(("script", "expected_flags"), CLI_CASES)
def test_dry_run_and_write_are_mutually_exclusive(script, expected_flags):
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / script), "--dry-run", "--write"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "not allowed with argument" in result.stderr
