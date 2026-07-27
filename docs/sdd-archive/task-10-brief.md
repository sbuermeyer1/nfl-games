### Task 10: Probability calibration

**Files:**
- Create: `src/nfl_game/model/calibrate.py`
- Test: `tests/test_calibrate.py`

**Interfaces:**
- Consumes: `walk_forward` output.
- Produces:
  - `Calibrator()` with `.fit(preds) -> Calibrator` and `.predict(preds) -> pd.DataFrame` returning `game_id, cover_prob, over_prob`.
  - `brier_score(probs: np.ndarray, outcomes: np.ndarray) -> float`
  - `reliability_table(probs, outcomes, bins=5) -> pd.DataFrame`

**Context for the implementer:** This converts a raw point disagreement into a probability with an empirical basis. Fit logistic regression on a single feature — `model_margin - spread_line` — against whether the home team actually covered. Same shape for totals.

Critically, the calibrator must be fit on **walk-forward (out-of-sample) predictions**, not in-sample ones. In-sample gaps are systematically overconfident and would produce probabilities that look sharp and are wrong.

`cover_prob` is always the probability that the **home team covers**.

- [ ] **Step 1: Write the failing test**

`tests/test_calibrate.py`:

```python
import numpy as np
import pandas as pd
import pytest

from nfl_game.model.calibrate import Calibrator, brier_score, reliability_table


def _preds(n=800, seed=0):
    rng = np.random.default_rng(seed)
    spread = rng.normal(scale=6.0, size=n)
    edge = rng.normal(scale=3.0, size=n)
    margin = spread + edge * 0.5 + rng.normal(scale=10.0, size=n)
    total_line = rng.normal(loc=45, scale=4.0, size=n)
    t_edge = rng.normal(scale=3.0, size=n)
    return pd.DataFrame(
        {
            "game_id": [f"g{i}" for i in range(n)],
            "spread_line": spread,
            "model_margin": spread + edge,
            "margin": margin,
            "total_line": total_line,
            "model_total": total_line + t_edge,
            "total_points": total_line + t_edge * 0.5 + rng.normal(scale=9.0, size=n),
        }
    )


def test_predict_returns_expected_columns():
    c = Calibrator().fit(_preds())
    out = c.predict(_preds(n=20, seed=1))
    assert list(out.columns) == ["game_id", "cover_prob", "over_prob"]


def test_probabilities_are_in_range():
    c = Calibrator().fit(_preds())
    out = c.predict(_preds(n=100, seed=1))
    assert out["cover_prob"].between(0, 1).all()
    assert out["over_prob"].between(0, 1).all()


def test_bigger_edge_means_higher_cover_probability():
    c = Calibrator().fit(_preds())
    df = _preds(n=2, seed=3)
    df["spread_line"] = [0.0, 0.0]
    df["model_margin"] = [1.0, 7.0]
    out = c.predict(df)
    assert out.iloc[1]["cover_prob"] > out.iloc[0]["cover_prob"]


def test_zero_edge_is_near_a_coin_flip():
    c = Calibrator().fit(_preds())
    df = _preds(n=1, seed=4)
    df["spread_line"] = [0.0]
    df["model_margin"] = [0.0]
    assert c.predict(df).iloc[0]["cover_prob"] == pytest.approx(0.5, abs=0.08)


def test_brier_score_rewards_accuracy():
    outcomes = np.array([1, 1, 0, 0])
    good = np.array([0.9, 0.8, 0.2, 0.1])
    bad = np.array([0.1, 0.2, 0.8, 0.9])
    assert brier_score(good, outcomes) < brier_score(bad, outcomes)


def test_reliability_table_shape():
    c = Calibrator().fit(_preds())
    p = _preds(n=400, seed=5)
    out = c.predict(p)
    covered = (p["margin"] > p["spread_line"]).astype(int).to_numpy()
    table = reliability_table(out["cover_prob"].to_numpy(), covered, bins=4)
    assert set(table.columns) == {"bin", "n", "mean_pred", "observed"}
    assert len(table) <= 4


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError, match="fit"):
        Calibrator().predict(_preds(n=3))
```

- [ ] **Step 2: Run it and confirm it fails**

```
.\.venv\Scripts\python.exe -m pytest tests/test_calibrate.py -v
```

Expected: `ModuleNotFoundError: No module named 'nfl_game.model.calibrate'`.

- [ ] **Step 3: Write `src/nfl_game/model/calibrate.py`**

```python
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
        d = preds[preds["margin"].notna() & preds["total_points"].notna()]

        spread_edge = (d["model_margin"] - d["spread_line"]).to_numpy(dtype=float).reshape(-1, 1)
        covered = (d["margin"] > d["spread_line"]).astype(int).to_numpy()
        self._cover = LogisticRegression().fit(spread_edge, covered)

        total_edge = (d["model_total"] - d["total_line"]).to_numpy(dtype=float).reshape(-1, 1)
        went_over = (d["total_points"] > d["total_line"]).astype(int).to_numpy()
        self._over = LogisticRegression().fit(total_edge, went_over)
        return self

    def predict(self, preds: pd.DataFrame) -> pd.DataFrame:
        if self._cover is None or self._over is None:
            raise RuntimeError("call fit() before predict()")
        spread_edge = (
            (preds["model_margin"] - preds["spread_line"]).to_numpy(dtype=float).reshape(-1, 1)
        )
        total_edge = (
            (preds["model_total"] - preds["total_line"]).to_numpy(dtype=float).reshape(-1, 1)
        )
        return pd.DataFrame(
            {
                "game_id": preds["game_id"].to_numpy(),
                "cover_prob": self._cover.predict_proba(spread_edge)[:, 1],
                "over_prob": self._over.predict_proba(total_edge)[:, 1],
            }
        )
```

- [ ] **Step 4: Run the tests and confirm they pass**

```
.\.venv\Scripts\python.exe -m pytest tests/test_calibrate.py -v
```

Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add src/nfl_game/model/calibrate.py tests/test_calibrate.py
git commit -m "feat: calibrate model-market gaps into cover and over probabilities"
```

---

