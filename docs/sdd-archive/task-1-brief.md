### Task 1: Repo scaffold and packaging

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `README.md`, `src/nfl_game/__init__.py`, `src/nfl_game/paths.py`, `data/raw/.gitkeep`, `data/processed/.gitkeep`
- Test: `tests/test_smoke.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `nfl_game.paths.PROJECT_ROOT`, `RAW_DIR`, `PROCESSED_DIR` — all `pathlib.Path`.

- [ ] **Step 1: Create the directory skeleton**

```bash
cd ~/Documents/"NFL Game Model"
mkdir -p src/nfl_game/{data,ratings,model,market} tests scripts data/raw data/processed
touch data/raw/.gitkeep data/processed/.gitkeep
touch src/nfl_game/__init__.py
touch src/nfl_game/data/__init__.py src/nfl_game/ratings/__init__.py
touch src/nfl_game/model/__init__.py src/nfl_game/market/__init__.py
```

- [ ] **Step 2: Write `pyproject.toml`**

```toml
[project]
name = "nfl-game"
version = "0.1.0"
description = "NFL game model for spreads and totals"
requires-python = ">=3.11"
dependencies = [
    "pandas>=2.2",
    "numpy>=1.26",
    "scikit-learn>=1.4",
    "scipy>=1.13",
    "nflreadpy>=0.1",
    "pyarrow>=16.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "ruff>=0.5",
]

[tool.pytest.ini_options]
testpaths = ["tests"]

[tool.ruff]
line-length = 100
src = ["src"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]
```

- [ ] **Step 3: Write `.gitignore`**

```
__pycache__/
*.pyc
.venv/
venv/
*.egg-info/
.pytest_cache/
.ruff_cache/

# raw/processed data can be large and regeneratable — keep out of git by default
data/raw/*
data/processed/*
!data/raw/.gitkeep
!data/processed/.gitkeep

.ipynb_checkpoints/
```

- [ ] **Step 4: Write the failing test**

`tests/test_smoke.py`:

```python
from nfl_game import paths


def test_paths_resolve_under_project_root():
    assert paths.RAW_DIR == paths.PROJECT_ROOT / "data" / "raw"
    assert paths.PROCESSED_DIR == paths.PROJECT_ROOT / "data" / "processed"


def test_data_dirs_exist():
    assert paths.RAW_DIR.is_dir()
    assert paths.PROCESSED_DIR.is_dir()
```

- [ ] **Step 5: Run it and confirm it fails**

```
.\.venv\Scripts\python.exe -m pytest tests/test_smoke.py -v
```

Expected: `ModuleNotFoundError: No module named 'nfl_game'` (venv not built yet) or `ImportError` on `paths`.

- [ ] **Step 6: Write `src/nfl_game/paths.py`**

```python
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
```

- [ ] **Step 7: Create the venv and install**

```
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

- [ ] **Step 8: Run the test and confirm it passes**

```
.\.venv\Scripts\python.exe -m pytest tests/test_smoke.py -v
```

Expected: 2 passed.

- [ ] **Step 9: Write `README.md`**

```markdown
# NFL Game Model

Predicts NFL game margins and totals from EPA + Next Gen Stats team ratings, then compares
those predictions against the closing spread and total.

The model is **market-blind**: it never sees the betting line when predicting. A separate
layer compares model output to the market and reports calibrated cover probabilities.

Data source: [`nflreadpy`](https://nflreadpy.nflverse.com/). No API key required.

Design: `docs/superpowers/specs/2026-07-23-nfl-game-model-design.md`

## Setup

    python -m venv .venv
    .\.venv\Scripts\python.exe -m pip install -e ".[dev]"

## Tests

    .\.venv\Scripts\python.exe -m pytest
```

- [ ] **Step 10: Commit**

```bash
git add pyproject.toml .gitignore README.md src/ tests/ data/
git commit -m "feat: scaffold nfl_game package and paths"
```

---

