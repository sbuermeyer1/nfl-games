FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir -e .

COPY scripts ./scripts
COPY data/processed/game_features.parquet ./data/processed/game_features.parquet

EXPOSE 8000
CMD ["python", "scripts/game_app.py"]
