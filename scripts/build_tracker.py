"""Build the reviewed historical tracker ledger from fixed walk-forward predictions."""

import argparse
import math
import tempfile
from pathlib import Path

import pandas as pd

from nfl_game.backtest import walk_forward
from nfl_game.model.predict import DEFAULT_ALPHA
from nfl_game.paths import PROCESSED_DIR
from nfl_game.tracking.ledger import (
    HISTORICAL_MODEL_VERSION,
    build_backtest_ledger,
    validate_ledger,
)
from nfl_game.tracking.summary import summarize_selection

HISTORICAL_SEASONS = tuple(range(2021, 2026))
EXPECTED_BASELINE = {
    "games": 1359,
    "ats_n": 1326,
    "ats_hit_rate": 0.497737556561086,
    "ou_n": 1348,
    "ou_hit_rate": 0.5022255192878339,
}


def acceptance_metrics(ledger: pd.DataFrame) -> dict:
    """Return the fixed all-prediction historical acceptance records."""
    summary = summarize_selection(ledger, "backtest", "all")
    all_predictions = summary["all_predictions"]
    ats = all_predictions["spread"]
    ou = all_predictions["total"]
    return {
        "games": len(ledger),
        "ats_n": ats["n_graded"],
        "ats_hit_rate": ats["win_rate"],
        "ou_n": ou["n_graded"],
        "ou_hit_rate": ou["win_rate"],
    }


def assert_acceptance_baseline(ledger: pd.DataFrame, expected: dict) -> None:
    """Reject any change to the fixed historical corpus or all-prediction records."""
    actual = acceptance_metrics(ledger)
    if set(expected) != set(EXPECTED_BASELINE):
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
) -> pd.DataFrame:
    """Build and accept the Ridge-only historical tracker ledger."""
    predictions = walk_forward(
        features,
        list(test_seasons),
        estimator="ridge",
        alpha=DEFAULT_ALPHA,
    )
    ledger = build_backtest_ledger(predictions, model_version=model_version)
    assert_acceptance_baseline(ledger, expected_baseline)
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
    args = parser.parse_args(argv)

    features = pd.read_parquet(args.features)
    ledger = build_historical_ledger(features, model_version=args.model_version)
    validate_ledger(ledger)
    _write_atomic_parquet(ledger, args.output)

    metrics = acceptance_metrics(ledger)
    records = summarize_selection(ledger, "backtest", "all")["all_predictions"]
    ats = records["spread"]
    ou = records["total"]
    print(f"wrote {len(ledger)} tracker ledger rows to {args.output}")
    print(f"ATS: {ats['wins']}-{ats['losses']}-{ats['pushes']} (n={metrics['ats_n']})")
    print(f"O/U: {ou['wins']}-{ou['losses']}-{ou['pushes']} (n={metrics['ou_n']})")


if __name__ == "__main__":
    main()
