import subprocess
import sys
from pathlib import Path

import pandas as pd
from scripts import slate

from nfl_game.market.live_starters import NflverseStarterProvider, StartersUnavailableError
from nfl_game.paths import PROCESSED_DIR

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


def test_no_starters_flag_defaults_to_false_and_flips_when_passed():
    """I6: the --help substring check above would pass with `action="store_false"`
    (silently inverting the default for every real run) or with the flag parsed and
    never read. Pin the actual parsed value on both sides instead. `_parser()` is
    extracted from `main()` specifically so this can import and call it directly."""
    base = ["--season", "2024", "--week", "1"]
    assert slate._parser().parse_args(base).no_starters is False
    assert slate._parser().parse_args([*base, "--no-starters"]).no_starters is True


def test_an_unavailable_starter_feed_still_prints_the_slate(monkeypatch, tmp_path, capsys):
    """I6: the `StartersUnavailableError` catch in `main()` -- the entire point of the
    --no-starters/fail-soft task -- had no test. Mirrors
    test_web_service.py::test_an_unavailable_starter_feed_still_returns_a_slate: the
    provider is monkeypatched to raise, never invoking any network-calling loader, and
    the slate must still print. Reads/writes are redirected to tmp_path so this does
    not touch the repo's packaged data/processed files."""
    tmp_path.joinpath("game_features.parquet").write_bytes(
        (PROCESSED_DIR / "game_features.parquet").read_bytes()
    )
    monkeypatch.setattr(slate, "PROCESSED_DIR", tmp_path)

    def raise_unavailable(self, season, week):
        raise StartersUnavailableError("feed down")

    monkeypatch.setattr(NflverseStarterProvider, "snapshot", raise_unavailable)

    feats = pd.read_parquet(PROCESSED_DIR / "game_features.parquet")
    season = int(feats["season"].max())
    week = int(feats.loc[feats["season"].eq(season), "week"].min())

    slate.main(["--season", str(season), "--week", str(week)])

    out = capsys.readouterr().out
    assert "warning: expected-starter advisory unavailable" in out
    assert "| QB |" in out
