FROM python:3.12-slim@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de

WORKDIR /app

COPY requirements-build.txt requirements-prod.txt ./
RUN python -m pip install --no-cache-dir --require-hashes -r requirements-build.txt \
    && python -m pip install --no-cache-dir --only-binary=:all: \
        --require-hashes -r requirements-prod.txt

COPY pyproject.toml README.md ./
COPY src ./src
# Editable install is load-bearing, not a convenience: it keeps the package at
# /app/src so nfl_game.paths.PROJECT_ROOT (Path(__file__).parents[2]) resolves to
# /app and the app finds /app/data/processed at runtime. An ordinary install moves
# the package under site-packages, where parents[2] is the interpreter's lib dir and
# startup dies with "packaged dataset not found". Deps stay hashed and pinned above;
# --no-deps keeps this step from reaching PyPI.
RUN python -m pip install --no-cache-dir -e . --no-deps --no-build-isolation

COPY scripts ./scripts
COPY data/processed/game_features.parquet ./data/processed/game_features.parquet
COPY data/processed/tracker_ledger.parquet ./data/processed/tracker_ledger.parquet

EXPOSE 8000
CMD ["python", "scripts/game_app.py"]
