"""Build the reviewed historical tracker ledger from fixed walk-forward predictions."""

import argparse
import math
import tempfile
from pathlib import Path

import pandas as pd

from nfl_game.backtest import walk_forward
from nfl_game.model.predict import DEFAULT_ALPHA
from nfl_game.paths import PROCESSED_DIR, RAW_DIR
from nfl_game.tracking.ledger import (
    HISTORICAL_MODEL_VERSION,
    build_backtest_ledger,
    validate_ledger,
)
from nfl_game.tracking.summary import summarize_selection

HISTORICAL_SEASONS = tuple(range(2021, 2026))
EXPECTED_BASELINE = {
    "games": 1359,
    "ats_wins": 660,
    "ats_losses": 666,
    "ats_pushes": 33,
    "ats_n": 1326,
    "ats_hit_rate": 0.497737556561086,
    "ou_wins": 677,
    "ou_losses": 671,
    "ou_pushes": 11,
    "ou_n": 1348,
    "ou_hit_rate": 0.5022255192878339,
}


DEFAULT_EARLY_LINES = RAW_DIR / "line_history" / "line_history_combined_g05.parquet"

EXPECTED_EARLY_BASELINE = {
    "games": 1359,
    "ats_wins": 651,
    "ats_losses": 665,
    "ats_pushes": 29,
    "ats_n": 1316,
    "ats_hit_rate": 0.4946808510638298,
    "ou_wins": 681,
    "ou_losses": 645,
    "ou_pushes": 19,
    "ou_n": 1326,
    "ou_hit_rate": 0.5135746606334841,
    "excluded_spread": 14,
    "excluded_total": 14,
}


def early_acceptance_metrics(ledger: pd.DataFrame) -> dict:
    """Acceptance records for the SHIPPED early-line ledger, plus the unpriced-game counts.

    The excluded counts are pinned because they are the quietest way this artifact could move:
    a line-history rebuild that lost coverage would shrink the graded population and shift every
    rate, with nothing else in the ledger looking wrong.
    """
    return {
        **acceptance_metrics(ledger),
        "excluded_spread": int(ledger["spread_publication_status"].eq("excluded").sum()),
        "excluded_total": int(ledger["total_publication_status"].eq("excluded").sum()),
    }


def acceptance_metrics(ledger: pd.DataFrame) -> dict:
    """Return the fixed all-prediction historical acceptance records."""
    summary = summarize_selection(ledger, "backtest", "all")
    all_predictions = summary["all_predictions"]
    ats = all_predictions["spread"]
    ou = all_predictions["total"]
    return {
        "games": len(ledger),
        "ats_wins": ats["wins"],
        "ats_losses": ats["losses"],
        "ats_pushes": ats["pushes"],
        "ats_n": ats["n_graded"],
        "ats_hit_rate": ats["win_rate"],
        "ou_wins": ou["wins"],
        "ou_losses": ou["losses"],
        "ou_pushes": ou["pushes"],
        "ou_n": ou["n_graded"],
        "ou_hit_rate": ou["win_rate"],
    }


def assert_acceptance_baseline(ledger: pd.DataFrame, expected: dict, metrics=None) -> None:
    """Reject any change to the fixed historical corpus or all-prediction records.

    `metrics` selects which record is being pinned: `acceptance_metrics` for the legacy
    closing-line build, `early_acceptance_metrics` for the shipped early-line one. The key sets
    differ, so the shape check is against whatever the chosen function returns.
    """
    metrics = acceptance_metrics if metrics is None else metrics
    actual = metrics(ledger)
    if set(actual) != set(expected):
        raise RuntimeError("acceptance baseline changed")

    for key, value in actual.items():
        target = expected[key]
        matches = (
            math.isclose(value, target, rel_tol=0.0, abs_tol=5e-13)
            if isinstance(value, float)
            else value == target
        )
        if not matches:
            raise RuntimeError("acceptance baseline changed")


def build_historical_ledger(
    features: pd.DataFrame,
    test_seasons=HISTORICAL_SEASONS,
    model_version=HISTORICAL_MODEL_VERSION,
    expected_baseline=EXPECTED_BASELINE,
    early_lines=None,
    expected_early_baseline=EXPECTED_EARLY_BASELINE,
) -> pd.DataFrame:
    """Build and accept the Ridge-only historical tracker ledger.

    With `early_lines` the shipped ledger is graded at the line available at publication time,
    so a backtest row means the same thing as a live one. The legacy closing-line ledger is
    still built and still asserted against the original baseline: grading is cheap next to the
    walk-forward fit, which both share, and it keeps the guard protecting exactly the facts it
    always did rather than re-basing the constant that exists to catch changes.
    """
    predictions = walk_forward(
        features,
        list(test_seasons),
        estimator="ridge",
        alpha=DEFAULT_ALPHA,
    )
    legacy = build_backtest_ledger(predictions, model_version=model_version)
    assert_acceptance_baseline(legacy, expected_baseline)
    if early_lines is None:
        return legacy

    ledger = build_backtest_ledger(
        predictions, model_version=model_version, early_lines=early_lines
    )
    assert_acceptance_baseline(ledger, expected_early_baseline, metrics=early_acceptance_metrics)
    return ledger


def _write_atomic_parquet(ledger: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".parquet", delete=False) as temp:
        temporary_path = Path(temp.name)
    try:
        ledger.to_parquet(temporary_path, index=False)
        temporary_path.replace(output)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=PROCESSED_DIR / "game_features.parquet")
    parser.add_argument("--output", type=Path, default=PROCESSED_DIR / "tracker_ledger.parquet")
    parser.add_argument("--model-version", default=HISTORICAL_MODEL_VERSION)
    parser.add_argument(
        "--early-lines",
        type=Path,
        default=DEFAULT_EARLY_LINES,
        help=(
            "game-anchored line history used to grade each game at the number available when it "
            "would have been published. Must be a _g<lead> file: a _d<lead> file is anchored to "
            "the week's first kickoff and is NOT the lead the live tracker publishes at."
        ),
    )
    parser.add_argument(
        "--no-early-lines",
        action="store_true",
        help="grade the historical corpus at the closing line, as it was before the backfill",
    )
    args = parser.parse_args(argv)

    features = pd.read_parquet(args.features)
    early_lines = None
    if not args.no_early_lines:
        early = pd.read_parquet(args.early_lines)
        early_lines = early[
            ["game_id", "early_spread_line", "early_total_line", "snapshot_at"]
        ]
    ledger = build_historical_ledger(
        features, model_version=args.model_version, early_lines=early_lines
    )
    validate_ledger(ledger)
    _write_atomic_parquet(ledger, args.output)

    metrics = acceptance_metrics(ledger)
    records = summarize_selection(ledger, "backtest", "all")["all_predictions"]
    ats = records["spread"]
    ou = records["total"]
    graded_at = "closing line" if early_lines is None else f"early line ({args.early_lines.name})"
    print(f"wrote {len(ledger)} tracker ledger rows to {args.output}, graded at the {graded_at}")
    print(f"ATS: {ats['wins']}-{ats['losses']}-{ats['pushes']} (n={metrics['ats_n']})")
    print(f"O/U: {ou['wins']}-{ou['losses']}-{ou['pushes']} (n={metrics['ou_n']})")
    if early_lines is not None:
        excluded = early_acceptance_metrics(ledger)["excluded_spread"]
        print(f"excluded (no line at publication): {excluded}")


if __name__ == "__main__":
    main()
