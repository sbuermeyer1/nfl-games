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
