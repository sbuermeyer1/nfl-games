### Task 8: Margin and total models

**Files:**
- Create: `src/nfl_game/model/predict.py`
- Test: `tests/test_predict.py`

**Interfaces:**
- Consumes: `features.FEATURE_COLS`, `features.TARGET_COLS`.
- Produces:
  - `GameModel(estimator: str = "ridge", alpha: float = 1.0)` with `.fit(train_df) -> GameModel` and `.predict(df) -> pd.DataFrame` returning columns `game_id, model_margin, model_total`.
  - `ESTIMATORS: dict[str, callable]` mapping `"ridge"` and `"gbm"` to factory functions.

**Context for the implementer:** One class wrapping two fitted regressors — one for margin, one for total. Keeping both behind a single object means the backtest, calibration, and slate code never care which estimator won. Rows with a null target are dropped at fit time; rows with null targets are still predictable.

Ridge is the baseline and default. GBM is the challenger. Task 9 decides between them with evidence.

- [ ] **Step 1: Write the failing test**

`tests/test_predict.py`:

```python
import numpy as np
import pandas as pd
import pytest

from nfl_game.model.features import FEATURE_COLS
from nfl_game.model.predict import ESTIMATORS, GameModel


def _train(n=300, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.normal(size=n) for c in FEATURE_COLS})
    df["game_id"] = [f"g{i}" for i in range(n)]
    # margin is a known linear function of two features plus noise
    df["margin"] = 3.0 * df["net_rating_diff"] + 1.5 * df["rest_diff"] + rng.normal(scale=0.5, size=n)
    df["total_points"] = 44.0 + 2.0 * df["off_pass_edge_home"] + rng.normal(scale=0.5, size=n)
    return df


def test_predict_returns_expected_columns():
    m = GameModel().fit(_train())
    out = m.predict(_train(n=10, seed=1))
    assert list(out.columns) == ["game_id", "model_margin", "model_total"]
    assert len(out) == 10


def test_recovers_a_known_linear_signal():
    train = _train()
    m = GameModel(estimator="ridge", alpha=0.01).fit(train)
    pred = m.predict(train)
    corr = np.corrcoef(pred["model_margin"], train["margin"])[0, 1]
    assert corr > 0.95


def test_total_model_is_separate_from_margin():
    train = _train()
    m = GameModel(estimator="ridge", alpha=0.01).fit(train)
    pred = m.predict(train)
    assert pred["model_total"].mean() == pytest.approx(44.0, abs=1.0)


def test_gbm_estimator_also_fits():
    train = _train()
    m = GameModel(estimator="gbm").fit(train)
    pred = m.predict(train)
    assert pred["model_margin"].notna().all()


def test_rejects_unknown_estimator():
    with pytest.raises(ValueError, match="estimator"):
        GameModel(estimator="magic")


def test_rows_with_null_targets_are_dropped_at_fit():
    train = _train()
    train.loc[:50, "margin"] = np.nan
    m = GameModel(estimator="ridge", alpha=0.01).fit(train)
    assert m.n_train_margin_ == len(train) - 51


def test_can_predict_rows_with_null_targets():
    train = _train()
    future = _train(n=5, seed=2)
    future[["margin", "total_points"]] = np.nan
    m = GameModel().fit(train)
    out = m.predict(future)
    assert out["model_margin"].notna().all()


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError, match="fit"):
        GameModel().predict(_train(n=3))


def test_estimators_registry_exposes_both():
    assert set(ESTIMATORS) == {"ridge", "gbm"}
```

- [ ] **Step 2: Run it and confirm it fails**

```
.\.venv\Scripts\python.exe -m pytest tests/test_predict.py -v
```

Expected: `ModuleNotFoundError: No module named 'nfl_game.model.predict'`.

- [ ] **Step 3: Write `src/nfl_game/model/predict.py`**

```python
"""Margin and total regressors behind a single interface.

Ridge is the honest baseline; gradient boosting is the challenger. Task 9's backtest
picks between them on evidence. Everything downstream consumes GameModel and never
needs to know which one is in use.
"""

import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from nfl_game.model.features import FEATURE_COLS

ESTIMATORS = {
    "ridge": lambda alpha: Ridge(alpha=alpha),
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
```

- [ ] **Step 4: Run the tests and confirm they pass**

```
.\.venv\Scripts\python.exe -m pytest tests/test_predict.py -v
```

Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add src/nfl_game/model/predict.py tests/test_predict.py
git commit -m "feat: margin and total regressors with ridge and gbm estimators"
```

---

