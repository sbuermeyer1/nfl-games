### Task 2: Data layer

**Files:**
- Create: `src/nfl_game/data/nfl.py`
- Test: `tests/test_data_nfl.py`

**Interfaces:**
- Consumes: `nfl_game.paths.RAW_DIR`.
- Produces:
  - `load_schedules(seasons: list[int] | None = None, save: bool = True) -> pd.DataFrame`
  - `load_pbp(seasons: list[int], save: bool = True) -> pd.DataFrame`
  - `load_ngs(seasons: list[int], stat_type: str, save: bool = True) -> pd.DataFrame`
  - `_seasons_label(seasons: list[int]) -> str`

**Context for the implementer:** `nflreadpy` returns Polars DataFrames. Convert immediately. `load_schedules()` with no seasons returns all seasons 1999–2026 including future games with null results — that is intentional and needed for predicting upcoming slates.

- [ ] **Step 1: Write the failing test**

`tests/test_data_nfl.py`:

```python
import pandas as pd
import pytest

from nfl_game.data import nfl


def test_seasons_label_single():
    assert nfl._seasons_label([2024]) == "2024"


def test_seasons_label_range():
    assert nfl._seasons_label([2024, 2016, 2020]) == "2016-2024"


def test_load_pbp_converts_and_saves(monkeypatch, tmp_path):
    class FakePolars:
        def to_pandas(self):
            return pd.DataFrame({"game_id": ["2024_01_ARI_BUF"], "epa": [0.5]})

    monkeypatch.setattr(nfl, "RAW_DIR", tmp_path)
    monkeypatch.setattr(nfl.nflreadpy, "load_pbp", lambda seasons: FakePolars())

    out = nfl.load_pbp([2024])

    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == ["game_id", "epa"]
    assert (tmp_path / "pbp_2024.parquet").exists()


def test_load_pbp_can_skip_save(monkeypatch, tmp_path):
    class FakePolars:
        def to_pandas(self):
            return pd.DataFrame({"game_id": ["x"], "epa": [0.1]})

    monkeypatch.setattr(nfl, "RAW_DIR", tmp_path)
    monkeypatch.setattr(nfl.nflreadpy, "load_pbp", lambda seasons: FakePolars())

    nfl.load_pbp([2024], save=False)

    assert list(tmp_path.iterdir()) == []


def test_load_ngs_rejects_bad_stat_type():
    with pytest.raises(ValueError, match="stat_type"):
        nfl.load_ngs([2024], stat_type="kicking")
```

- [ ] **Step 2: Run it and confirm it fails**

```
.\.venv\Scripts\python.exe -m pytest tests/test_data_nfl.py -v
```

Expected: `ModuleNotFoundError: No module named 'nfl_game.data.nfl'`.

- [ ] **Step 3: Write `src/nfl_game/data/nfl.py`**

```python
"""NFL data ingestion via nflreadpy (nflverse public data releases).

nflreadpy returns Polars DataFrames; everything here converts to pandas so nothing
downstream of this module has to know Polars exists.
"""

import nflreadpy
import pandas as pd

from nfl_game.paths import RAW_DIR

NGS_STAT_TYPES = ("passing", "rushing", "receiving")


def _seasons_label(seasons: list[int]) -> str:
    seasons = sorted(seasons)
    return f"{seasons[0]}-{seasons[-1]}" if len(seasons) > 1 else str(seasons[0])


def load_schedules(seasons: list[int] | None = None, save: bool = True) -> pd.DataFrame:
    """Game schedule, results, and closing betting lines.

    Passing seasons=None loads every season (1999+), including future games whose
    result/total are null but whose spread_line/total_line may already be posted.
    """
    df = nflreadpy.load_schedules().to_pandas()
    if seasons is not None:
        df = df[df["season"].isin(seasons)].reset_index(drop=True)
    if save:
        label = _seasons_label(seasons) if seasons else "all"
        df.to_parquet(RAW_DIR / f"schedules_{label}.parquet")
    return df


def load_pbp(seasons: list[int], save: bool = True) -> pd.DataFrame:
    """Play-by-play with EPA. Large: roughly 50k rows and 372 columns per season."""
    df = nflreadpy.load_pbp(seasons).to_pandas()
    if save:
        df.to_parquet(RAW_DIR / f"pbp_{_seasons_label(seasons)}.parquet")
    return df


def load_ngs(seasons: list[int], stat_type: str, save: bool = True) -> pd.DataFrame:
    """Next Gen Stats, 2016+ only. stat_type is one of passing/rushing/receiving.

    Note: rows with week == 0 are season aggregates, not week-zero games. Callers
    doing weekly joins must filter them out.
    """
    if stat_type not in NGS_STAT_TYPES:
        raise ValueError(f"stat_type must be one of {NGS_STAT_TYPES}, got {stat_type!r}")
    df = nflreadpy.load_nextgen_stats(seasons=seasons, stat_type=stat_type).to_pandas()
    if save:
        df.to_parquet(RAW_DIR / f"ngs_{stat_type}_{_seasons_label(seasons)}.parquet")
    return df
```

- [ ] **Step 4: Run the tests and confirm they pass**

```
.\.venv\Scripts\python.exe -m pytest tests/test_data_nfl.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Verify against real data once, by hand**

```
.\.venv\Scripts\python.exe -c "from nfl_game.data.nfl import load_schedules; d=load_schedules(save=False); print(d.shape); print(d[['spread_line','total_line','result','total']].notna().sum())"
```

Expected: shape around `(7548, 46)`; all four columns show thousands of non-null values.

- [ ] **Step 6: Commit**

```bash
git add src/nfl_game/data/nfl.py tests/test_data_nfl.py
git commit -m "feat: add nflreadpy data loaders for schedules, pbp, and NGS"
```

---

