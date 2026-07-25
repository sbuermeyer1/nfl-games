"""Walk-forward evaluation against the market.

The market is the benchmark, not a strawman. Every accuracy number is reported next to
the closing line's own error on the same games, and market_comparison_regression answers
the only question that really matters: does the model add anything the line doesn't
already contain?
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from nfl_game.model.predict import DegenerateFeatureError, GameModel

# Columns that must all be present for a game to enter any evaluate()-family metric.
# Model and market are always compared on the identical game set, so a missing
# spread_line (an unplayed game, or a season with no posted line) drops the row from
# every metric rather than only from the ones that happen to touch that column.
_REQUIRED_COLS = [
    "margin",
    "total_points",
    "spread_line",
    "total_line",
    "model_margin",
    "model_total",
]


def _valid_games(preds: pd.DataFrame) -> pd.DataFrame:
    """Rows where model and market both have something to say, and the outcome is known."""
    mask = preds[_REQUIRED_COLS].notna().all(axis=1)
    return preds[mask]


def walk_forward(
    features_df: pd.DataFrame,
    test_seasons: list[int],
    estimator: str = "ridge",
    alpha: float = 1.0,
) -> pd.DataFrame:
    """Predict each test season using a model trained only on strictly earlier seasons."""
    frames = []
    for season in sorted(test_seasons):
        train = features_df[features_df["season"] < season]
        test = features_df[features_df["season"] == season]
        if train.empty or test.empty:
            continue
        try:
            model = GameModel(estimator=estimator, alpha=alpha).fit(train)
        except DegenerateFeatureError:
            # A training slice that can't support a stable coefficient for some
            # feature shouldn't contribute predictions at all -- same treatment as a
            # season with no prior data.
            continue
        preds = model.predict(test)
        merged = test.merge(preds, on="game_id", how="left", validate="one_to_one")
        frames.append(merged)
    if not frames:
        return pd.DataFrame(columns=[*features_df.columns, "model_margin", "model_total"])
    return pd.concat(frames, ignore_index=True)


def evaluate(preds: pd.DataFrame) -> dict:
    """Accuracy and ATS metrics, each paired with the market's own performance.

    ATS: the model takes the home side when it predicts a bigger home margin than the
    line. Exact pushes are excluded, which is why ats_n is reported alongside the rate.
    Break-even at standard -110 juice is 52.4%.
    """
    d = _valid_games(preds)

    out = {
        "n_games": len(d),
        "margin_mae": float((d["model_margin"] - d["margin"]).abs().mean()),
        "market_margin_mae": float((d["spread_line"] - d["margin"]).abs().mean()),
        "total_mae": float((d["model_total"] - d["total_points"]).abs().mean()),
        "market_total_mae": float((d["total_line"] - d["total_points"]).abs().mean()),
    }

    played = d[d["margin"] != d["spread_line"]]
    if len(played):
        picks_home = played["model_margin"] > played["spread_line"]
        home_covered = played["margin"] > played["spread_line"]
        out["ats_hit_rate"] = float((picks_home == home_covered).mean())
        out["ats_n"] = len(played)
    else:
        out["ats_hit_rate"] = float("nan")
        out["ats_n"] = 0

    ou = d[d["total_points"] != d["total_line"]]
    if len(ou):
        picks_over = ou["model_total"] > ou["total_line"]
        went_over = ou["total_points"] > ou["total_line"]
        out["ou_hit_rate"] = float((picks_over == went_over).mean())
        out["ou_n"] = len(ou)
    else:
        out["ou_hit_rate"] = float("nan")
        out["ou_n"] = 0

    out["ats_breakeven"] = 0.524
    return out


def market_comparison_regression(preds: pd.DataFrame) -> dict:
    """Regress actual margin on both the market line and the model line.

    If model_coef is indistinguishable from zero, the model contributes nothing beyond
    what the closing line already knows. This is the decisive test.
    """
    d = _valid_games(preds)
    X = d[["spread_line", "model_margin"]].to_numpy(dtype=float)
    y = d["margin"].to_numpy(dtype=float)
    fit = LinearRegression().fit(X, y)
    return {
        "market_coef": float(fit.coef_[0]),
        "model_coef": float(fit.coef_[1]),
        "intercept": float(fit.intercept_),
        "r2": float(fit.score(X, y)),
        "n": len(d),
    }


def ats_by_threshold(preds: pd.DataFrame, thresholds=(0, 1, 2, 3, 4, 6)) -> pd.DataFrame:
    """ATS hit rate bucketed by how far the model disagrees with the line."""
    d = _valid_games(preds).copy()
    d["edge"] = (d["model_margin"] - d["spread_line"]).abs()
    rows = []
    for t in thresholds:
        sub = d[(d["edge"] >= t) & (d["margin"] != d["spread_line"])]
        if sub.empty:
            rows.append({"min_edge": t, "n": 0, "hit_rate": np.nan})
            continue
        picks_home = sub["model_margin"] > sub["spread_line"]
        home_covered = sub["margin"] > sub["spread_line"]
        rows.append(
            {"min_edge": t, "n": len(sub), "hit_rate": float((picks_home == home_covered).mean())}
        )
    return pd.DataFrame(rows)
