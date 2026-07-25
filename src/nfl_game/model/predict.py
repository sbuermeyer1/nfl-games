"""Margin and total regressors behind a single interface.

Ridge is the honest baseline; gradient boosting is the challenger. Task 9's backtest
picks between them on evidence. Everything downstream consumes GameModel and never
needs to know which one is in use.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from nfl_game.model.features import FEATURE_COLS


class RobustStandardScaler(StandardScaler):
    """StandardScaler with an absolute floor under sklearn's constant-feature guard.

    StandardScaler already refuses to divide by an exactly-zero variance, but its
    constant-feature test (``sklearn.preprocessing._data._is_constant_feature``) checks
    variance *relative to the feature's mean*. A feature that is constant to
    floating-point noise but happens to have a mean near zero -- e.g. ryoe_diff in an
    early season, where sparse NGS data means nearly every row gets the same imputed
    value -- slips through that test with a ``scale_`` on the order of 1e-17 rather than
    being flagged as constant. Dividing by it then blows up every downstream prediction.

    sklearn's own ``_handle_zeros_in_scale`` helper already contains the fix for this:
    an absolute fallback of ``scale < 10 * eps`` that catches near-zero scales
    regardless of mean. That fallback only runs when no ``constant_mask`` is supplied,
    but ``StandardScaler`` always supplies one (the mean-relative test), so the
    absolute check is effectively dead code from ``StandardScaler``'s own call site.
    This subclass reinstates it as a second, OR'd condition after every fit, which is
    exactly the behaviour sklearn intends but does not reach for this case. It is safe
    for a feature that is constant within one training slice but varies in others (e.g.
    walk-forward folds on different seasons), since it is recomputed from scratch on
    every fit call and never persists across folds.
    """

    def partial_fit(self, X, y=None, sample_weight=None):
        super().partial_fit(X, y, sample_weight=sample_weight)
        if self.scale_ is not None:
            eps = np.finfo(self.scale_.dtype).eps
            near_zero = self.scale_ < 10 * eps
            self.scale_ = np.where(near_zero, 1.0, self.scale_)
        return self


MIN_DISTINCT_VALUES = 10
# Threshold for the degenerate-feature guard below. Picked from the real walk-forward
# training slices in the calibration corpus (seasons 2017-2025): the smallest healthy
# non-flag feature, rest_diff, holds at least 15 distinct values from the 2019 fold
# onward. The one fold where it goes lower is 2017, at 13 -- and that fold is already
# rejected on ryoe_diff regardless, so 13 is the true floor across folds this guard is
# meant to let through. The two poisoned folds (ryoe_diff trained on 2016 alone: 3
# distinct values; trained on 2016+2017: 5) sit well below. 10 sits with margin on both
# sides and needs no retuning as more seasons accumulate -- it is not relative to fold
# size. Keep this in step with the same account in CLAUDE.md.


class DegenerateFeatureError(ValueError):
    """A FEATURE_COLS column in this training slice can't support a coefficient."""


def _degenerate_features(df: pd.DataFrame) -> list[str]:
    """FEATURE_COLS columns with too few distinct values to fit a stable coefficient.

    Skips columns whose only values are 0/1 -- deliberate indicator flags (is_dome,
    div_game, ngs_imputed_any) legitimately take just two values in every fold; that is
    normal, not degenerate. Every other FEATURE_COLS column is a continuous rating,
    edge, or difference that is expected to vary case by case, so fewer than
    MIN_DISTINCT_VALUES distinct values in a training slice means it has collapsed to a
    handful of imputed defaults plus a few stray real observations. That is exactly the
    2018 walk-forward fold in the real feature set: ryoe_diff there has std ~1.05e-2 --
    far above RobustStandardScaler's ~2.22e-15 floor, so the scaler lets it through --
    but only 5 distinct values across 512 rows (416 rows at 0.0, 94 at float noise
    around it, and only 2 rows carrying an actual signal). A coefficient fit on that is
    noise, not signal, regardless of how "non-zero" its variance looks.
    """
    bad = []
    for col in FEATURE_COLS:
        values = df[col].dropna()
        if values.empty:
            # An all-NaN column is the most degenerate case there is, so it must not be
            # the one case exempted. Falling through to Ridge.fit would raise a raw
            # sklearn "Input X contains NaN" that walk_forward does not catch, crashing
            # the whole backtest instead of skipping the one bad fold.
            bad.append(col)
            continue
        if set(values.unique()) <= {0, 1}:
            continue
        if values.nunique() < MIN_DISTINCT_VALUES:
            bad.append(col)
    return bad


ESTIMATORS = {
    # Ridge penalises raw coefficient size, so it is only meaningful on standardised
    # inputs. FEATURE_COLS mixes 0/1 flags, EPA rating diffs near 0.1, and temperatures
    # near 60; unscaled, the penalty would fall hardest on the rating features that carry
    # the signal and barely touch temperature. Standardising also keeps the ridge-vs-gbm
    # comparison honest, since trees are scale-invariant either way.
    "ridge": lambda alpha: make_pipeline(RobustStandardScaler(), Ridge(alpha=alpha)),
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

        for label, data in (("margin", m), ("total_points", t)):
            bad = _degenerate_features(data)
            if bad:
                raise DegenerateFeatureError(
                    f"training slice for target {label!r} ({len(data)} rows) has "
                    f"degenerate feature(s) {bad}: fewer than {MIN_DISTINCT_VALUES} "
                    "distinct non-binary values. Refusing to fit -- this slice cannot "
                    "support a stable coefficient for it."
                )

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
