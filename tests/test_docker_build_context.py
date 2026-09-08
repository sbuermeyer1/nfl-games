"""Every data file the Dockerfile COPYs must survive .dockerignore.

`.dockerignore` here is an ALLOWLIST -- it starts with `*` and re-includes named paths.
A file can therefore be committed to git, present on disk, and named in a COPY line, and
still be absent from the build context, because nothing re-included it. Docker then fails
at build time with "failed to compute cache key: ... not found", which is a deploy-time
failure, not a test-time one.

That happened on 2026-09-08 with `starter_advisory.parquet`: it was tracked, pushed, and
COPYed, and the deploy still failed. `git ls-files` and the COPY line were both checked;
the build context was not. This test closes that gap by comparing the two lists directly.
"""

import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _copied_data_paths() -> set[str]:
    lines = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8").splitlines()
    return {
        match.group(1)
        for line in lines
        if (match := re.match(r"COPY\s+(data/\S+)", line.strip()))
    }


def _dockerignore_allowlist() -> set[str]:
    lines = (PROJECT_ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
    return {line[1:].strip() for line in lines if line.startswith("!")}


def test_every_copied_data_file_is_allowlisted_in_dockerignore():
    copied = _copied_data_paths()
    assert copied, "expected the Dockerfile to COPY at least one data artifact"
    missing = sorted(copied - _dockerignore_allowlist())
    assert not missing, (
        "these paths are COPYed by the Dockerfile but excluded from the build context "
        f"by .dockerignore, so the image build will fail: {missing}"
    )


def test_every_copied_data_file_exists_on_disk():
    """A COPY of a file that is not committed fails the build just as loudly."""
    missing = sorted(p for p in _copied_data_paths() if not (PROJECT_ROOT / p).is_file())
    assert not missing, f"Dockerfile COPYs files that do not exist: {missing}"
