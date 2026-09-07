import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_slate_help_documents_the_no_starters_flag():
    out = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "slate.py"), "--help"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--no-starters" in out.stdout
