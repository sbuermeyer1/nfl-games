"""Leak-free calibration, paired evaluation, uncertainty, and Ridge-v2 gates."""

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

REPORT_SEASONS = tuple(range(2021, 2026))
CALIBRATION_SEASONS = tuple(range(2019, 2026))
EDGE_THRESHOLDS = (0.0, 2.0, 5.0, 10.0, 15.0)
ATS_FLOOR = 0.497737556561086 - 0.01
OU_FLOOR = 0.5022255192878339 - 0.01

_PREDICTION_COLS = (
    "game_id",
    "season",
    "week",
    "margin",
    "total_points",
    "spread_line",
    "total_line",
    "model_margin",
    "model_total",
)
_PROBABILITY_COLS = ("game_id", "cover_prob", "over_prob")
_QUALITY_KEYS = ("correctness", "availability", "determinism", "source_reliability")


@dataclass(frozen=True)
class BootstrapInterval:
    estimate: float
    lower90: float
    upper90: float


@dataclass(frozen=True)
class PromotionDecision:
    approved: bool
    failures: tuple[str, ...]
    pending: tuple[str, ...]


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...], label: str) -> None:
    if frame.columns.duplicated().any():
        duplicates = frame.columns[frame.columns.duplicated()].unique().tolist()
        raise ValueError(f"{label} contains duplicate column label(s) {duplicates}")
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required column(s) {missing}")


def _validate_ids(frame: pd.DataFrame, label: str) -> None:
    if frame["game_id"].isna().any() or frame["game_id"].duplicated().any():
        raise ValueError(f"{label} requires unique, non-null game_id values")


def _as_finite_numeric(frame: pd.DataFrame, columns: tuple[str, ...], label: str) -> None:
    for column in columns:
        converted = pd.to_numeric(frame[column], errors="coerce")
        if converted.isna().any() or not np.isfinite(converted.to_numpy(dtype=float)).all():
            raise ValueError(f"{label} column {column!r} must contain finite numeric values")


def _validated_time_keys(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    """Validate raw season/week scalars strictly and return normalized integer keys."""
    _require_columns(frame, ("season", "week"), label)
    normalized: dict[str, pd.Series] = {}
    for column in ("season", "week"):
        values: list[int] = []
        for raw in frame[column].tolist():
            if isinstance(raw, (bool, np.bool_)) or not isinstance(
                raw, (int, float, np.integer, np.floating)
            ):
                requirement = "positive integer" if column == "week" else "integer"
                raise ValueError(f"{label} {column} values must be finite numeric {requirement}s")  # noqa: TRY004
            numeric = float(raw)
            if not np.isfinite(numeric) or not numeric.is_integer():
                requirement = "positive integer" if column == "week" else "integer"
                raise ValueError(f"{label} {column} values must be finite numeric {requirement}s")
            value = int(numeric)
            if column == "week" and value <= 0:
                raise ValueError(f"{label} week values must be positive integers")
            values.append(value)
        normalized[column] = pd.Series(values, index=frame.index, dtype="int64")
    result = frame.copy()
    for column, values in normalized.items():
        result[column] = values
    return result


def block_bootstrap_mean(
    frame: pd.DataFrame,
    value_col: str,
    draws: int = 10_000,
    seed: int = 0,
) -> BootstrapInterval:
    """Bootstrap a mean by resampling sorted season-week blocks."""
    if type(draws) is not int or draws <= 0:
        raise ValueError("draws must be a positive integer")
    _require_columns(frame, ("season", "week", value_col), "bootstrap frame")
    if frame.empty:
        raise ValueError("bootstrap frame cannot be empty")
    frame = _validated_time_keys(frame, "bootstrap frame")
    _as_finite_numeric(frame, (value_col,), "bootstrap frame")
    blocks = [
        group[value_col].to_numpy(dtype=float)
        for _, group in frame.groupby(["season", "week"], sort=True, dropna=False)
    ]
    if not blocks or any(len(block) == 0 for block in blocks):
        raise ValueError("bootstrap frame has no valid season-week blocks")

    rng = np.random.default_rng(seed)
    samples = np.empty(draws, dtype=float)
    for draw in range(draws):
        chosen = rng.integers(0, len(blocks), size=len(blocks))
        samples[draw] = np.concatenate([blocks[index] for index in chosen]).mean()
    values = frame[value_col].to_numpy(dtype=float)
    return BootstrapInterval(
        estimate=float(values.mean()),
        lower90=float(np.quantile(samples, 0.10)),
        upper90=float(np.quantile(samples, 0.90)),
    )


def _fit_probability_model(edge: np.ndarray, outcome: np.ndarray, target: str):
    if len(edge) == 0 or len(np.unique(outcome)) < 2:
        raise ValueError(f"{target} calibration requires both outcome classes")
    return LogisticRegression(random_state=0).fit(edge.reshape(-1, 1), outcome)


def walk_forward_probabilities(
    predictions: pd.DataFrame,
    *,
    fit_observer: Callable[[dict[str, object]], None] | None = None,
) -> pd.DataFrame:
    """Fit target-specific calibrators only on earlier out-of-sample seasons."""
    _require_columns(predictions, _PREDICTION_COLS, "calibration predictions")
    _validate_ids(predictions, "calibration predictions")
    predictions = _validated_time_keys(predictions, "calibration predictions")
    _as_finite_numeric(
        predictions,
        (
            "margin",
            "total_points",
            "spread_line",
            "total_line",
            "model_margin",
            "model_total",
        ),
        "calibration predictions",
    )
    seasons = predictions["season"]
    present_seasons = {int(value) for value in seasons.unique()}
    missing_calibration = sorted(set(CALIBRATION_SEASONS).difference(present_seasons))
    if missing_calibration:
        raise ValueError(f"missing calibration season(s) {missing_calibration}")
    if not set(REPORT_SEASONS).issubset(set(seasons)):
        missing = sorted(set(REPORT_SEASONS).difference(seasons))
        raise ValueError(f"calibration predictions are missing report season(s) {missing}")

    output: list[pd.DataFrame] = []
    for prediction_season in REPORT_SEASONS:
        train = predictions.loc[(seasons >= CALIBRATION_SEASONS[0]) & (seasons < prediction_season)]
        target = predictions.loc[seasons == prediction_season]
        if train.empty or target.empty:
            raise ValueError(f"no calibration evidence for prediction season {prediction_season}")
        predicted = target.loc[:, ["game_id", "season", "week"]].copy()
        for name, actual_col, line_col, model_col, probability_col in (
            ("cover", "margin", "spread_line", "model_margin", "cover_prob"),
            ("over", "total_points", "total_line", "model_total", "over_prob"),
        ):
            non_push = train.loc[train[actual_col] != train[line_col]]
            required_prior = set(range(CALIBRATION_SEASONS[0], prediction_season))
            observed_prior = {int(value) for value in non_push["season"].unique()}
            missing_prior = sorted(required_prior.difference(observed_prior))
            if missing_prior:
                raise ValueError(
                    f"{name} calibration is missing non-push evidence for prior season(s) "
                    f"{missing_prior}"
                )
            edge = (non_push[model_col] - non_push[line_col]).to_numpy(dtype=float)
            outcome = (non_push[actual_col] > non_push[line_col]).astype(int).to_numpy()
            if fit_observer is not None:
                fit_observer(
                    {
                        "target": name,
                        "prediction_season": prediction_season,
                        "training_seasons": tuple(
                            sorted(int(value) for value in non_push["season"].unique())
                        ),
                        "n_training_games": len(non_push),
                    }
                )
            calibrator = _fit_probability_model(edge, outcome, name)
            target_edge = (target[model_col] - target[line_col]).to_numpy(dtype=float)
            predicted[probability_col] = calibrator.predict_proba(target_edge.reshape(-1, 1))[:, 1]
        output.append(predicted)
    return pd.concat(output, ignore_index=True)


def _regression_coefficients(
    frame: pd.DataFrame, actual_col: str, market_col: str, model_col: str
) -> np.ndarray:
    x = frame.loc[:, [market_col, model_col]].to_numpy(dtype=float)
    y = frame[actual_col].to_numpy(dtype=float)
    matrix = np.column_stack([np.ones(len(x)), x])
    if len(frame) < 3 or np.linalg.matrix_rank(matrix) < 3:
        raise ValueError("joint market regression is degenerate")
    coefficients, _, _, _ = np.linalg.lstsq(matrix, y, rcond=None)
    return coefficients


def joint_market_regression(
    frame: pd.DataFrame,
    actual_col: str,
    market_col: str,
    model_col: str,
    *,
    draws: int = 10_000,
    seed: int = 0,
) -> dict[str, float | int]:
    """Fit actual on closing line and model and block-bootstrap the model coefficient."""
    if type(draws) is not int or draws <= 0:
        raise ValueError("draws must be a positive integer")
    columns = ("season", "week", actual_col, market_col, model_col)
    _require_columns(frame, columns, "joint market regression frame")
    if frame.empty:
        raise ValueError("joint market regression frame cannot be empty")
    frame = _validated_time_keys(frame, "joint market regression frame")
    _as_finite_numeric(frame, (actual_col, market_col, model_col), "joint market regression frame")
    coefficients = _regression_coefficients(frame, actual_col, market_col, model_col)
    grouped = [group for _, group in frame.groupby(["season", "week"], sort=True)]
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(draws, dtype=float)
    for draw in range(draws):
        chosen = rng.integers(0, len(grouped), size=len(grouped))
        sample = pd.concat([grouped[index] for index in chosen], ignore_index=True)
        try:
            bootstrap[draw] = _regression_coefficients(sample, actual_col, market_col, model_col)[2]
        except ValueError as exc:
            raise ValueError(f"joint market bootstrap draw {draw} is degenerate") from exc
    predictions = (
        coefficients[0]
        + coefficients[1] * frame[market_col].to_numpy(dtype=float)
        + coefficients[2] * frame[model_col].to_numpy(dtype=float)
    )
    actual = frame[actual_col].to_numpy(dtype=float)
    residual_sum = float(np.square(actual - predictions).sum())
    total_sum = float(np.square(actual - actual.mean()).sum())
    return {
        "intercept": float(coefficients[0]),
        "market_coef": float(coefficients[1]),
        "model_coef": float(coefficients[2]),
        "model_coef_lower90": float(np.quantile(bootstrap, 0.10)),
        "model_coef_upper90": float(np.quantile(bootstrap, 0.90)),
        "r2": float(1.0 - residual_sum / total_sum) if total_sum > 0 else 1.0,
        "n": len(frame),
    }


def _record(
    frame: pd.DataFrame,
    *,
    actual_col: str,
    line_col: str,
    model_col: str,
    threshold: float,
    draws: int,
    seed: int,
) -> dict[str, float | int | None]:
    subset = frame.loc[(frame[model_col] - frame[line_col]).abs() >= threshold].copy()
    pick_high = subset[model_col] > subset[line_col]
    push = subset[actual_col] == subset[line_col]
    high_won = subset[actual_col] > subset[line_col]
    wins = int((~push & (pick_high == high_won)).sum())
    losses = int((~push & (pick_high != high_won)).sum())
    pushes = int(push.sum())
    non_push = subset.loc[~push].copy()
    result: dict[str, float | int | None] = {
        "min_edge": threshold,
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "n": wins + losses,
        "total_picks": len(subset),
        "hit_rate": float(wins / (wins + losses)) if wins + losses else None,
        "lower90": None,
        "upper90": None,
    }
    if not non_push.empty:
        non_push["won"] = (pick_high.loc[~push] == high_won.loc[~push]).astype(float)
        interval = block_bootstrap_mean(non_push, "won", draws=draws, seed=seed)
        result["lower90"] = interval.lower90
        result["upper90"] = interval.upper90
    return result


def _probability_pair(
    pairs: pd.DataFrame,
    probabilities: pd.DataFrame,
    prefix: str,
) -> pd.DataFrame:
    _require_columns(probabilities, _PROBABILITY_COLS, f"{prefix} probabilities")
    _validate_ids(probabilities, f"{prefix} probabilities")
    _as_finite_numeric(probabilities, ("cover_prob", "over_prob"), f"{prefix} probabilities")
    if (
        not probabilities["cover_prob"].between(0, 1).all()
        or not probabilities["over_prob"].between(0, 1).all()
    ):
        raise ValueError(f"{prefix} probabilities must be within [0, 1]")
    expected_ids = set(pairs["game_id"])
    probability_ids = set(probabilities["game_id"])
    if probability_ids != expected_ids:
        missing = sorted(expected_ids.difference(probability_ids))
        extra = sorted(probability_ids.difference(expected_ids))
        raise ValueError(
            f"{prefix} probabilities require exact game_id coverage; "
            f"missing={missing}, extra={extra}"
        )
    joined = pairs.loc[:, ["game_id"]].merge(
        probabilities, on="game_id", how="left", validate="one_to_one"
    )
    if joined[["cover_prob", "over_prob"]].isna().any(axis=None):
        raise ValueError(f"{prefix} probabilities do not cover every paired report row")
    return joined


def _quality_report(quality_checks: Mapping[str, object] | None) -> dict[str, bool | None]:
    evidence = quality_checks or {}
    return {
        f"{key}_passed": evidence.get(key) if type(evidence.get(key)) is bool else None
        for key in _QUALITY_KEYS
    }


def _calibration_report(probabilities: pd.Series, outcomes: pd.Series) -> dict[str, object]:
    probability = probabilities.to_numpy(dtype=float)
    outcome = outcomes.to_numpy(dtype=int)
    clipped = np.clip(probability, 1e-12, 1 - 1e-12)
    logit = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
    if len(np.unique(outcome)) < 2 or np.ptp(logit) == 0:
        intercept: float | None = None
        slope: float | None = None
    else:
        fitted = LogisticRegression(C=np.inf, random_state=0).fit(logit, outcome)
        intercept = float(fitted.intercept_[0])
        slope = float(fitted.coef_[0, 0])
    table = pd.DataFrame({"probability": probability, "outcome": outcome})
    table["bin"] = pd.cut(table["probability"], bins=np.linspace(0.0, 1.0, 11), include_lowest=True)
    reliability = []
    for bucket, group in table.groupby("bin", observed=True, sort=True):
        reliability.append(
            {
                "lower": float(bucket.left),
                "upper": float(bucket.right),
                "n": len(group),
                "mean_probability": float(group["probability"].mean()),
                "observed_rate": float(group["outcome"].mean()),
            }
        )
    return {"intercept": intercept, "slope": slope, "reliability": reliability}


def evaluate_v2(
    v1_predictions: pd.DataFrame,
    v2_predictions: pd.DataFrame,
    v1_probabilities: pd.DataFrame,
    v2_probabilities: pd.DataFrame,
    *,
    quality_checks: Mapping[str, object] | None = None,
    bootstrap_draws: int = 10_000,
    seed: int = 0,
) -> dict[str, Any]:
    """Build complete paired Ridge-v1/v2 evidence for 2021-2025."""
    validated_predictions: list[pd.DataFrame] = []
    for frame, label in ((v1_predictions, "v1 predictions"), (v2_predictions, "v2 predictions")):
        _require_columns(frame, _PREDICTION_COLS, label)
        _validate_ids(frame, label)
        validated_predictions.append(_validated_time_keys(frame, label))
    v1_predictions, v2_predictions = validated_predictions
    v1 = v1_predictions.loc[v1_predictions["season"].isin(REPORT_SEASONS)].copy()
    v2 = v2_predictions.loc[v2_predictions["season"].isin(REPORT_SEASONS)].copy()
    report_numeric = tuple(
        column for column in _PREDICTION_COLS if column not in ("game_id", "season", "week")
    )
    _as_finite_numeric(v1, report_numeric, "v1 report predictions")
    _as_finite_numeric(v2, report_numeric, "v2 report predictions")
    v1_ids = set(v1["game_id"])
    v2_ids = set(v2["game_id"])
    if v1_ids != v2_ids:
        missing = sorted(v1_ids.difference(v2_ids))
        extra = sorted(v2_ids.difference(v1_ids))
        raise ValueError(
            "Ridge-v1/v2 report predictions require exact game_id coverage; "
            f"missing_from_v2={missing}, extra_in_v2={extra}"
        )
    pairs = v1.merge(v2, on="game_id", suffixes=("_v1", "_v2"), validate="one_to_one")
    if pairs.empty:
        raise ValueError("no valid paired Ridge-v1/v2 report rows")
    for shared in ("season", "week", "margin", "total_points", "spread_line", "total_line"):
        left = pd.to_numeric(pairs[f"{shared}_v1"], errors="coerce")
        right = pd.to_numeric(pairs[f"{shared}_v2"], errors="coerce")
        if (
            left.isna().any()
            or right.isna().any()
            or not np.array_equal(left.to_numpy(dtype=float), right.to_numpy(dtype=float))
        ):
            raise ValueError(f"paired report rows disagree on {shared!r}")
        pairs[shared] = left
    _as_finite_numeric(
        pairs,
        (
            "season",
            "week",
            "margin",
            "total_points",
            "spread_line",
            "total_line",
            "model_margin_v1",
            "model_total_v1",
            "model_margin_v2",
            "model_total_v2",
        ),
        "paired report rows",
    )
    present = set(pairs["season"].astype(int))
    if present != set(REPORT_SEASONS):
        missing = sorted(set(REPORT_SEASONS).difference(present))
        raise ValueError(f"paired report rows are missing season(s) {missing}")

    v1_probs = _probability_pair(pairs, v1_probabilities, "v1")
    v2_probs = _probability_pair(pairs, v2_probabilities, "v2")
    pairs["v1_cover_prob"] = v1_probs["cover_prob"].to_numpy(dtype=float)
    pairs["v1_over_prob"] = v1_probs["over_prob"].to_numpy(dtype=float)
    pairs["cover_prob"] = v2_probs["cover_prob"].to_numpy(dtype=float)
    pairs["over_prob"] = v2_probs["over_prob"].to_numpy(dtype=float)
    pairs["margin_v1_abs_error"] = (pairs["model_margin_v1"] - pairs["margin"]).abs()
    pairs["margin_v2_abs_error"] = (pairs["model_margin_v2"] - pairs["margin"]).abs()
    pairs["total_v1_abs_error"] = (pairs["model_total_v1"] - pairs["total_points"]).abs()
    pairs["total_v2_abs_error"] = (pairs["model_total_v2"] - pairs["total_points"]).abs()
    pairs["margin_improvement"] = pairs["margin_v1_abs_error"] - pairs["margin_v2_abs_error"]
    pairs["total_improvement"] = pairs["total_v1_abs_error"] - pairs["total_v2_abs_error"]

    margin_interval = block_bootstrap_mean(
        pairs, "margin_improvement", draws=bootstrap_draws, seed=seed
    )
    total_interval = block_bootstrap_mean(
        pairs, "total_improvement", draws=bootstrap_draws, seed=seed
    )
    margin_market = joint_market_regression(
        pairs,
        "margin",
        "spread_line",
        "model_margin_v2",
        draws=bootstrap_draws,
        seed=seed,
    )
    total_market = joint_market_regression(
        pairs,
        "total_points",
        "total_line",
        "model_total_v2",
        draws=bootstrap_draws,
        seed=seed,
    )

    per_season: dict[str, dict[str, float | int]] = {}
    for season, group in pairs.groupby("season", sort=True):
        margin_v1 = float(group["margin_v1_abs_error"].mean())
        margin_v2 = float(group["margin_v2_abs_error"].mean())
        total_v1 = float(group["total_v1_abs_error"].mean())
        total_v2 = float(group["total_v2_abs_error"].mean())
        per_season[str(int(season))] = {
            "n_games": len(group),
            "margin_mae": margin_v2,
            "margin_rmse": float(
                np.sqrt(np.square(group["model_margin_v2"] - group["margin"]).mean())
            ),
            "v1_margin_mae": margin_v1,
            "v1_margin_rmse": float(
                np.sqrt(np.square(group["model_margin_v1"] - group["margin"]).mean())
            ),
            "total_mae": total_v2,
            "total_rmse": float(
                np.sqrt(np.square(group["model_total_v2"] - group["total_points"]).mean())
            ),
            "v1_total_mae": total_v1,
            "v1_total_rmse": float(
                np.sqrt(np.square(group["model_total_v1"] - group["total_points"]).mean())
            ),
            "margin_improvement": margin_v1 - margin_v2,
            "total_improvement": total_v1 - total_v2,
            "market_margin_mae": float((group["spread_line"] - group["margin"]).abs().mean()),
            "market_total_mae": float((group["total_line"] - group["total_points"]).abs().mean()),
        }

    cover_non_push = pairs["margin"] != pairs["spread_line"]
    over_non_push = pairs["total_points"] != pairs["total_line"]
    cover_outcome = pairs.loc[cover_non_push, "margin"] > pairs.loc[cover_non_push, "spread_line"]
    over_outcome = pairs.loc[over_non_push, "total_points"] > pairs.loc[over_non_push, "total_line"]
    if not cover_non_push.any() or not over_non_push.any():
        raise ValueError("paired report rows have no valid non-push betting outcomes")
    ats_pick = (
        pairs.loc[cover_non_push, "model_margin_v2"] > pairs.loc[cover_non_push, "spread_line"]
    )
    ou_pick = pairs.loc[over_non_push, "model_total_v2"] > pairs.loc[over_non_push, "total_line"]

    report: dict[str, Any] = {
        "report_seasons": list(REPORT_SEASONS),
        "n_games": len(pairs),
        "per_season": per_season,
        "margin_mae": float(pairs["margin_v2_abs_error"].mean()),
        "margin_rmse": float(np.sqrt(np.square(pairs["model_margin_v2"] - pairs["margin"]).mean())),
        "v1_margin_mae": float(pairs["margin_v1_abs_error"].mean()),
        "v1_margin_rmse": float(
            np.sqrt(np.square(pairs["model_margin_v1"] - pairs["margin"]).mean())
        ),
        "market_margin_mae": float((pairs["spread_line"] - pairs["margin"]).abs().mean()),
        "total_mae": float(pairs["total_v2_abs_error"].mean()),
        "total_rmse": float(
            np.sqrt(np.square(pairs["model_total_v2"] - pairs["total_points"]).mean())
        ),
        "v1_total_mae": float(pairs["total_v1_abs_error"].mean()),
        "v1_total_rmse": float(
            np.sqrt(np.square(pairs["model_total_v1"] - pairs["total_points"]).mean())
        ),
        "market_total_mae": float((pairs["total_line"] - pairs["total_points"]).abs().mean()),
        "margin_paired_improvement": margin_interval.estimate,
        "margin_paired_improvement_lower90": margin_interval.lower90,
        "margin_paired_improvement_upper90": margin_interval.upper90,
        "total_paired_improvement": total_interval.estimate,
        "total_paired_improvement_lower90": total_interval.lower90,
        "total_paired_improvement_upper90": total_interval.upper90,
        "margin_seasons_improved": sum(
            value["margin_improvement"] > 0 for value in per_season.values()
        ),
        "total_seasons_improved": sum(
            value["total_improvement"] > 0 for value in per_season.values()
        ),
        "margin_market_regression": margin_market,
        "margin_market_model_coef": margin_market["model_coef"],
        "margin_market_model_coef_lower90": margin_market["model_coef_lower90"],
        "total_market_regression": total_market,
        "total_market_model_coef": total_market["model_coef"],
        "total_market_model_coef_lower90": total_market["model_coef_lower90"],
        "ats_hit_rate": float((ats_pick == cover_outcome).mean()),
        "ats_n": int(cover_non_push.sum()),
        "ou_hit_rate": float((ou_pick == over_outcome).mean()),
        "ou_n": int(over_non_push.sum()),
        "cover_brier": float(
            np.square(pairs.loc[cover_non_push, "cover_prob"] - cover_outcome.astype(float)).mean()
        ),
        "v1_cover_brier": float(
            np.square(
                pairs.loc[cover_non_push, "v1_cover_prob"] - cover_outcome.astype(float)
            ).mean()
        ),
        "over_brier": float(
            np.square(pairs.loc[over_non_push, "over_prob"] - over_outcome.astype(float)).mean()
        ),
        "v1_over_brier": float(
            np.square(pairs.loc[over_non_push, "v1_over_prob"] - over_outcome.astype(float)).mean()
        ),
        "cover_calibration": _calibration_report(
            pairs.loc[cover_non_push, "cover_prob"], cover_outcome
        ),
        "v1_cover_calibration": _calibration_report(
            pairs.loc[cover_non_push, "v1_cover_prob"], cover_outcome
        ),
        "over_calibration": _calibration_report(
            pairs.loc[over_non_push, "over_prob"], over_outcome
        ),
        "v1_over_calibration": _calibration_report(
            pairs.loc[over_non_push, "v1_over_prob"], over_outcome
        ),
        "ats_cohorts": [
            _record(
                pairs,
                actual_col="margin",
                line_col="spread_line",
                model_col="model_margin_v2",
                threshold=threshold,
                draws=bootstrap_draws,
                seed=seed,
            )
            for threshold in EDGE_THRESHOLDS
        ],
        "ou_cohorts": [
            _record(
                pairs,
                actual_col="total_points",
                line_col="total_line",
                model_col="model_total_v2",
                threshold=threshold,
                draws=bootstrap_draws,
                seed=seed,
            )
            for threshold in EDGE_THRESHOLDS
        ],
        **_quality_report(quality_checks),
    }
    return report


def _finite_number(report: Mapping[str, object], key: str) -> float | None:
    value = report.get(key)
    if type(value) not in (int, float):
        return None
    numeric = float(value)
    return numeric if np.isfinite(numeric) else None


def research_gate_decision(report: Mapping[str, object]) -> PromotionDecision:
    """Evaluate gates 1-10, collecting every failure and missing evidence."""
    failures: list[str] = []
    pending: list[str] = []
    derived_improvements: tuple[int, int] | None = None
    derived_n_games: int | None = None
    if "report_seasons" not in report or "per_season" not in report:
        pending.append("outer season evidence")
    elif report.get("report_seasons") != list(REPORT_SEASONS):
        failures.append("outer season evidence")
    else:
        per_season = report.get("per_season")
        expected_keys = {str(season) for season in REPORT_SEASONS}
        if not isinstance(per_season, Mapping) or set(per_season) != expected_keys:
            failures.append("outer season evidence")
        else:
            margin_improved = 0
            total_improved = 0
            total_games = 0
            valid_seasons = True
            for season in REPORT_SEASONS:
                evidence = per_season[str(season)]
                if not isinstance(evidence, Mapping):
                    valid_seasons = False
                    break
                n_games = evidence.get("n_games")
                margin_value = _finite_number(evidence, "margin_improvement")
                total_value = _finite_number(evidence, "total_improvement")
                if type(n_games) is not int or n_games <= 0:
                    valid_seasons = False
                    break
                if margin_value is None or total_value is None:
                    valid_seasons = False
                    break
                margin_improved += int(margin_value > 0)
                total_improved += int(total_value > 0)
                total_games += n_games
            if valid_seasons:
                derived_improvements = (margin_improved, total_improved)
                derived_n_games = total_games
            else:
                failures.append("outer season evidence")

    if "n_games" not in report or report.get("n_games") is None:
        pending.append("game count evidence")
    elif (
        type(report["n_games"]) is not int
        or report["n_games"] <= 0
        or derived_n_games is not None
        and report["n_games"] != derived_n_games
    ):
        failures.append("game count evidence")

    def numeric_gate(key: str, label: str, passes: Callable[[float], bool]) -> None:
        raw = report.get(key)
        value = _finite_number(report, key)
        if key not in report or raw is None:
            pending.append(label)
        elif value is None or not passes(value):
            failures.append(label)

    numeric_gate("margin_mae", "margin MAE", lambda value: value < 10.274)
    numeric_gate("total_mae", "total MAE", lambda value: value < 10.684)
    numeric_gate("margin_seasons_improved", "margin season improvement", lambda value: value >= 3)
    numeric_gate("total_seasons_improved", "total season improvement", lambda value: value >= 3)
    if "margin_seasons_improved" in report and type(report["margin_seasons_improved"]) is not int:
        failures.append("margin season improvement")
    if "total_seasons_improved" in report and type(report["total_seasons_improved"]) is not int:
        failures.append("total season improvement")
    if derived_improvements is not None:
        margin_count = _finite_number(report, "margin_seasons_improved")
        total_count = _finite_number(report, "total_seasons_improved")
        if margin_count is not None and margin_count != derived_improvements[0]:
            failures.append("margin season improvement")
        if total_count is not None and total_count != derived_improvements[1]:
            failures.append("total season improvement")
    numeric_gate(
        "margin_paired_improvement_lower90",
        "margin paired improvement",
        lambda value: value > 0,
    )
    numeric_gate(
        "total_paired_improvement_lower90",
        "total paired improvement",
        lambda value: value > 0,
    )
    numeric_gate("margin_market_model_coef", "margin market contribution", lambda value: value > 0)
    numeric_gate(
        "margin_market_model_coef_lower90",
        "margin market contribution",
        lambda value: value > 0,
    )
    numeric_gate("total_market_model_coef", "total market contribution", lambda value: value > 0)
    numeric_gate(
        "total_market_model_coef_lower90",
        "total market contribution",
        lambda value: value > 0,
    )
    numeric_gate("ats_hit_rate", "ATS hit rate", lambda value: value >= ATS_FLOOR)
    numeric_gate("ou_hit_rate", "O/U hit rate", lambda value: value >= OU_FLOOR)
    for candidate_key, baseline_key, label in (
        ("cover_brier", "v1_cover_brier", "cover Brier"),
        ("over_brier", "v1_over_brier", "over Brier"),
    ):
        candidate = _finite_number(report, candidate_key)
        baseline = _finite_number(report, baseline_key)
        candidate_missing = candidate_key not in report or report.get(candidate_key) is None
        baseline_missing = baseline_key not in report or report.get(baseline_key) is None
        malformed = (not candidate_missing and candidate is None) or (
            not baseline_missing and baseline is None
        )
        if malformed:
            failures.append(label)
        if candidate_missing or baseline_missing:
            pending.append(label)
        elif not malformed and candidate > baseline:
            failures.append(label)
    for key, label in (
        ("correctness_passed", "correctness"),
        ("availability_passed", "availability"),
        ("determinism_passed", "determinism"),
        ("source_reliability_passed", "source reliability"),
    ):
        value = report.get(key)
        if type(value) is not bool:
            pending.append(label)
        elif value is not True:
            failures.append(label)
    failures = list(dict.fromkeys(failures))
    pending = list(dict.fromkeys(pending))
    return PromotionDecision(not failures and not pending, tuple(failures), tuple(pending))


def promotion_decision(
    report: Mapping[str, object], shadow_rebuild_passed: bool | None
) -> PromotionDecision:
    """Evaluate all eleven gates, keeping a missing shadow result pending."""
    research = research_gate_decision(report)
    failures = list(research.failures)
    pending = list(research.pending)
    if shadow_rebuild_passed is None:
        pending.append("shadow production rebuild")
    elif type(shadow_rebuild_passed) is not bool:
        raise TypeError("shadow_rebuild_passed must be bool or None")
    elif not shadow_rebuild_passed:
        failures.append("shadow production rebuild")
    return PromotionDecision(not failures and not pending, tuple(failures), tuple(pending))


def bootstrap_interval_dict(interval: BootstrapInterval) -> dict[str, float]:
    """Return an explicitly JSON-safe representation for artifact builders."""
    return asdict(interval)
