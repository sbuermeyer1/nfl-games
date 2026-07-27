# Task 1 Report: Repo scaffold and packaging

## Summary

Implemented the full scaffold exactly as specified in `task-1-brief.md`, following the brief's
TDD ordering: directory skeleton -> pyproject.toml -> .gitignore -> failing test -> confirm RED
-> paths.py -> venv/install -> confirm GREEN -> README -> commit.

## What was implemented

- `pyproject.toml` — package metadata, `requires-python = ">=3.11"`, runtime deps (pandas,
  numpy, scikit-learn, scipy, nflreadpy, pyarrow), dev deps (pytest, ruff), pytest `testpaths`,
  ruff `line-length = 100` / `src = ["src"]`, setuptools src-layout build config. Verbatim from
  brief.
- `.gitignore` — standard Python/venv ignores plus `data/raw/*` and `data/processed/*` with
  `.gitkeep` exceptions. Verbatim from brief.
- `src/nfl_game/__init__.py` — empty package marker.
- `src/nfl_game/paths.py` — `PROJECT_ROOT`, `RAW_DIR`, `PROCESSED_DIR` as `pathlib.Path`.
  Verbatim from brief.
- `src/nfl_game/{data,ratings,model,market}/__init__.py` — empty package markers per Step 1
  (subpackages for later tasks; no logic added, consistent with YAGNI).
- `data/raw/.gitkeep`, `data/processed/.gitkeep` — placeholders so the (gitignored) data dirs
  exist and are tracked as empty.
- `tests/test_smoke.py` — two tests: path composition (`RAW_DIR`/`PROCESSED_DIR` relative to
  `PROJECT_ROOT`) and directory existence. Verbatim from brief.
- `README.md` — project description, market-blind framing, data source, design doc pointer,
  setup/test instructions. Verbatim from brief. Confirmed the referenced design doc
  (`docs/superpowers/specs/2026-07-23-nfl-game-model-design.md`) actually exists in the repo.
- `scripts/`, `tests/` directories created per Step 1 (empty; not tracked by git since they
  contain no files — this is expected Git behavior, not an omission).
- Created `.venv` and installed the package with `pip install -e ".[dev]"`.

## TDD evidence

**RED** — run before `paths.py` existed and before the venv was built:

```
PS> cd "C:\Users\sbuer\Documents\NFL Game Model"; .\.venv\Scripts\python.exe -m pytest tests/test_smoke.py -v
```

Output:
```
.\.venv\Scripts\python.exe : The term '.\.venv\Scripts\python.exe' is not recognized as the name
of a cmdlet, function, script file, or operable program. ...
```

This is expected per the brief's own framing ("venv not built yet") — the interpreter didn't
exist yet, so the command failed before it could even attempt the import. This confirms the
test could not possibly pass at this point, which is the RED state the brief calls for at this
step (the brief allows either `ModuleNotFoundError` or this venv-not-present failure).

**GREEN** — after writing `paths.py`, creating the venv, and installing the package:

```
PS> cd "C:\Users\sbuer\Documents\NFL Game Model"; .\.venv\Scripts\python.exe -m pytest tests/test_smoke.py -v
```

Output:
```
============================= test session starts =============================
platform win32 -- Python 3.12.10, pytest-9.1.1, pluggy-1.6.0 -- ...\.venv\Scripts\python.exe
rootdir: C:\Users\sbuer\Documents\NFL Game Model
configfile: pyproject.toml
collecting ... collected 2 items

tests/test_smoke.py::test_paths_resolve_under_project_root PASSED        [ 50%]
tests/test_smoke.py::test_data_dirs_exist PASSED                         [100%]

============================== 2 passed in 0.04s ==============================
```

## Additional verification

- Full test suite (`-m pytest -v`, no path filter): 2 passed, same two tests (only test file
  in repo).
- `ruff check src tests`: "All checks passed!" — no lint issues in the scaffold.
- `pip install -e ".[dev]"` succeeded, pulling pandas 3.0.5, numpy 2.5.1, scikit-learn 1.9.0,
  scipy 1.18.0, nflreadpy 0.1.5, pyarrow 25.0.0, pytest 9.1.1, ruff 0.16.0, plus transitive deps
  (polars, pydantic, requests, etc. via nflreadpy). No install errors.
- `git status` before staging showed only the intended new files as untracked — `.venv/`,
  `*.egg-info/`, `.pytest_cache/` were correctly excluded by `.gitignore`, confirming the
  ignore rules work as intended before commit.
- Confirmed no code imports from `nfl_ffm` (only `pathlib` is imported in the scaffold).
- Confirmed no test hits the network (the only test module imports `nfl_game.paths` and checks
  local filesystem state).

## Files changed (all new)

- `C:\Users\sbuer\Documents\NFL Game Model\pyproject.toml`
- `C:\Users\sbuer\Documents\NFL Game Model\.gitignore`
- `C:\Users\sbuer\Documents\NFL Game Model\README.md`
- `C:\Users\sbuer\Documents\NFL Game Model\src\nfl_game\__init__.py`
- `C:\Users\sbuer\Documents\NFL Game Model\src\nfl_game\paths.py`
- `C:\Users\sbuer\Documents\NFL Game Model\src\nfl_game\data\__init__.py`
- `C:\Users\sbuer\Documents\NFL Game Model\src\nfl_game\ratings\__init__.py`
- `C:\Users\sbuer\Documents\NFL Game Model\src\nfl_game\model\__init__.py`
- `C:\Users\sbuer\Documents\NFL Game Model\src\nfl_game\market\__init__.py`
- `C:\Users\sbuer\Documents\NFL Game Model\data\raw\.gitkeep`
- `C:\Users\sbuer\Documents\NFL Game Model\data\processed\.gitkeep`
- `C:\Users\sbuer\Documents\NFL Game Model\tests\test_smoke.py`

Not committed (correctly gitignored, present locally only): `.venv/`, `nfl_game.egg-info/`,
`.pytest_cache/`.

## Commit

```
bad39fd feat: scaffold nfl_game package and paths
```
12 files changed, 84 insertions(+), matching exactly the file list `git add pyproject.toml
.gitignore README.md src/ tests/ data/` from the brief.

## Self-review

- **Completeness against brief**: every file and content block in the brief was created
  verbatim, in the order specified. All 10 steps executed.
- **Naming**: `PROJECT_ROOT`, `RAW_DIR`, `PROCESSED_DIR` match the interface spec exactly
  (`nfl_game.paths.PROJECT_ROOT`, etc.), all `pathlib.Path`.
- **YAGNI**: no extra code, no extra dependencies, no extra config beyond the brief. The four
  empty subpackages (`data`, `ratings`, `model`, `market`) were created only because Step 1's
  literal `mkdir`/`touch` commands specify them — they hold no logic, just `__init__.py`
  markers for later tasks to fill in.
- **Does the test verify real behavior?** Yes — `test_paths_resolve_under_project_root` checks
  that `RAW_DIR`/`PROCESSED_DIR` are actually computed as `PROJECT_ROOT / "data" / <name>`
  (not just any path), and `test_data_dirs_exist` checks the directories are real and present
  on disk, not just importable names. Both would fail if `paths.py` were missing, wrong, or if
  the data directories weren't created — this is a genuine smoke test, not a tautology.

## Concerns

None. Everything specified in the brief was implemented, tests pass, ruff is clean, and the
commit matches the requested message and file scope exactly.
