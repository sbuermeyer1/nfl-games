"""Turn model-vs-market point gaps into probabilities.

A 4-point disagreement means nothing on its own. Fitting the gap against historical
cover outcomes gives it an empirical hit rate.

Fit this on walk-forward predictions only. In-sample gaps are systematically
overconfident, and a calibrator trained on them produces probabilities that look sharp
and are wrong.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Columns each target needs a real value in before a row can teach that half of the
# calibrator anything. The cover and over models are trained independently: a row
# missing total_line still has a perfectly good margin/spread_line/model_margin and
# should not be thrown out of cover training just because the total side is unusable.
# A NaN in any of a target's own columns propagates into its edge (NaN - x is NaN) and
# LogisticRegression.fit raises on NaN input, so rows failing a target's own filter
# would crash fit() for that target specifically rather than merely being silently
# wrong.
_COVER_COLS = ("margin", "spread_line", "model_margin")
_OVER_COLS = ("total_points", "total_line", "model_total")


def brier_score(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """Mean squared error of probabilistic forecasts. Lower is better; 0.25 is a coin flip."""
    return float(np.mean((np.asarray(probs) - np.asarray(outcomes)) ** 2))


def reliability_table(probs: np.ndarray, outcomes: np.ndarray, bins: int = 5) -> pd.DataFrame:
    """Predicted vs observed frequency per probability bucket. Well-calibrated means they match."""
    df = pd.DataFrame({"p": np.asarray(probs), "y": np.asarray(outcomes)})
    df["bin"] = pd.cut(df["p"], bins=np.linspace(0, 1, bins + 1), include_lowest=True)
    grouped = df.groupby("bin", observed=True).agg(
        n=("y", "size"), mean_pred=("p", "mean"), observed=("y", "mean")
    )
    return grouped.reset_index()


class Calibrator:
    """Maps (model - market) disagreement to cover and over probabilities."""

    def __init__(self):
        self._cover = None
        self._over = None

    def fit(self, preds: pd.DataFrame) -> "Calibrator":
        # A push (outcome exactly equal to the line) returns the stake -- it is not a
        # loss for the home/over side, so it must not train as one. backtest.evaluate
        # already excludes exact pushes from ats_hit_rate/ou_hit_rate on this same
        # reasoning; matching that here keeps "cover"/"over" meaning the same thing
        # everywhere in the codebase instead of two contradictory definitions.
        cover_mask = preds[list(_COVER_COLS)].notna().all(axis=1)
        d_cover = preds[cover_mask]
        d_cover = d_cover[d_cover["margin"] != d_cover["spread_line"]]
        spread_edge = (
            (d_cover["model_margin"] - d_cover["spread_line"]).to_numpy(dtype=float).reshape(-1, 1)
        )
        covered = (d_cover["margin"] > d_cover["spread_line"]).astype(int).to_numpy()
        self._cover = LogisticRegression().fit(spread_edge, covered)

        over_mask = preds[list(_OVER_COLS)].notna().all(axis=1)
        d_over = preds[over_mask]
        d_over = d_over[d_over["total_points"] != d_over["total_line"]]
        total_edge = (
            (d_over["model_total"] - d_over["total_line"]).to_numpy(dtype=float).reshape(-1, 1)
        )
        went_over = (d_over["total_points"] > d_over["total_line"]).astype(int).to_numpy()
        self._over = LogisticRegression().fit(total_edge, went_over)
        return self

    def predict(self, preds: pd.DataFrame) -> pd.DataFrame:
        if self._cover is None or self._over is None:
            raise RuntimeError("call fit() before predict()")

        cover_prob = np.full(len(preds), np.nan)
        cover_ok = (preds["model_margin"].notna() & preds["spread_line"].notna()).to_numpy()
        if cover_ok.any():
            spread_edge = (
                (preds.loc[cover_ok, "model_margin"] - preds.loc[cover_ok, "spread_line"])
                .to_numpy(dtype=float)
                .reshape(-1, 1)
            )
            cover_prob[cover_ok] = self._cover.predict_proba(spread_edge)[:, 1]

        over_prob = np.full(len(preds), np.nan)
        over_ok = (preds["model_total"].notna() & preds["total_line"].notna()).to_numpy()
        if over_ok.any():
            total_edge = (
                (preds.loc[over_ok, "model_total"] - preds.loc[over_ok, "total_line"])
                .to_numpy(dtype=float)
                .reshape(-1, 1)
            )
            over_prob[over_ok] = self._over.predict_proba(total_edge)[:, 1]

        return pd.DataFrame(
            {
                "game_id": preds["game_id"].to_numpy(),
                "cover_prob": cover_prob,
                "over_prob": over_prob,
            }
        )
