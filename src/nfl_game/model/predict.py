"""Margin and total regressors behind a single interface.

Ridge is the honest baseline; gradient boosting is the challenger. Task 9's backtest
picks between them on evidence. Everything downstream consumes GameModel and never
needs to know which one is in use.
"""

import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from nfl_game.model.features import FEATURE_COLS

ESTIMATORS = {
    # Ridge penalises raw coefficient size, so it is only meaningful on standardised
    # inputs. FEATURE_COLS mixes 0/1 flags, EPA rating diffs near 0.1, and temperatures
    # near 60; unscaled, the penalty would fall hardest on the rating features that carry
    # the signal and barely touch temperature. Standardising also keeps the ridge-vs-gbm
    # comparison honest, since trees are scale-invariant either way.
    "ridge": lambda alpha: make_pipeline(StandardScaler(), Ridge(alpha=alpha)),
    "gbm": lambda alpha: HistGradientBoostingRegressor(
        max_depth=3, learning_rate=0.05, max_iter=300, random_state=0
    ),
}


class GameModel:
    """Fits one regressor for game margin and one for total points."""

    def __init__(self, estimator: str = "ridge", alpha: float = 1.0):
        if estimator not in ESTIMATORS:
            raise ValueError(f"estimator must be one of {sorted(ESTIMATORS)}, got {estimator!r}")
        self.estimator = estimator
        self.alpha = alpha
        self._margin = None
        self._total = None
        self.n_train_margin_ = 0
        self.n_train_total_ = 0

    def fit(self, train: pd.DataFrame) -> "GameModel":
        m = train[train["margin"].notna()]
        t = train[train["total_points"].notna()]
        self.n_train_margin_ = len(m)
        self.n_train_total_ = len(t)

        self._margin = ESTIMATORS[self.estimator](self.alpha)
        self._margin.fit(m[FEATURE_COLS].to_numpy(dtype=float), m["margin"].to_numpy(dtype=float))

        self._total = ESTIMATORS[self.estimator](self.alpha)
        self._total.fit(
            t[FEATURE_COLS].to_numpy(dtype=float), t["total_points"].to_numpy(dtype=float)
        )
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._margin is None or self._total is None:
            raise RuntimeError("call fit() before predict()")
        X = df[FEATURE_COLS].to_numpy(dtype=float)
        return pd.DataFrame(
            {
                "game_id": df["game_id"].to_numpy(),
                "model_margin": self._margin.predict(X),
                "model_total": self._total.predict(X),
            }
        )
