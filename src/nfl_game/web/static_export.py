"""Pre-render the dashboard into a static site for Firebase Hosting.

Every response a page can request is computed through the same services the served
dashboard uses and written under `data/` as `{"status": ..., "body": ...}`, so an
error the server would return (a season with no prior calibration data, say) is
exported and shown the same way. `static_shim.py` maps the pages' `/api/...` calls onto
these files; the page HTML is the served HTML with that shim added.

Freshness is set at export time: market lines, the starter advisory, and the week the
page opens on are whatever they were when this ran, so the site is only as fresh as its
last deploy. Unexpected exceptions are NOT exported -- the server would answer them with
a 500, but here they mean the build is wrong, so the export fails and nothing deploys.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from pathlib import Path

from fastapi.encoders import jsonable_encoder

from nfl_game.web.app import PAGE
from nfl_game.web.schedule_page import SCHEDULE_PAGE
from nfl_game.web.service import (
    SlateInputError,
    SlateNotFoundError,
    SlateService,
    SlateUnavailableError,
)
from nfl_game.web.static_shim import STATIC_SHIM
from nfl_game.web.tracker_page import TRACKER_PAGE
from nfl_game.web.tracker_service import TrackerInputError, TrackerService

SHIM_PATH = "static-shim.js"
SHIM_TAG = f'<script src="/{SHIM_PATH}"></script>'
#: The schedule page only ever requests this season (`/api/schedule?season=2026`).
SCHEDULE_SEASON = 2026
PAGES = {"index.html": PAGE, "schedule.html": SCHEDULE_PAGE, "tracker.html": TRACKER_PAGE}

# Mirrors the exception handlers in app.py::create_app.
_STATUS = (
    (SlateNotFoundError, 404),
    (SlateUnavailableError, 409),
    (SlateInputError, 422),
    (TrackerInputError, 422),
)


def _envelope(call: Callable[[], object]) -> dict:
    try:
        return {"status": 200, "body": jsonable_encoder(call())}
    except tuple(error for error, _ in _STATUS) as exc:
        status = next(code for error, code in _STATUS if isinstance(exc, error))
        return {"status": status, "body": {"error": str(exc)}}


def _dumps(value) -> str:
    # Same encoding as Starlette's JSONResponse; allow_nan=False fails on a stray NaN.
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def with_shim(page: str) -> str:
    """Load the shim ahead of the page's single inline script."""
    if page.count("<script>") != 1:
        raise ValueError("expected exactly one inline <script> in the page")
    return page.replace("<script>", f"{SHIM_TAG}\n<script>", 1)


def export_site(
    out_dir: str | Path,
    slate_service: SlateService,
    tracker_service: TrackerService,
) -> dict[str, int]:
    """Write the complete static site to `out_dir`, replacing anything there."""
    out = Path(out_dir)
    if out.exists():
        shutil.rmtree(out)
    counts = {"pages": 0, "data": 0, "csv": 0}

    def write(relative: str, text: str) -> None:
        path = out / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        # newline="" keeps pandas' own line endings in the CSVs byte-for-byte.
        path.write_text(text, encoding="utf-8", newline="")

    def write_data(relative: str, call: Callable[[], object]) -> dict:
        envelope = _envelope(call)
        write(f"data/{relative}", _dumps(envelope))
        counts["data"] += 1
        return envelope

    for name, page in PAGES.items():
        write(name, with_shim(page))
        counts["pages"] += 1
    write(SHIM_PATH, STATIC_SHIM)

    options = write_data("options.json", slate_service.options)["body"]
    for season in options["seasons"]:
        weeks = write_data(
            f"weeks/{season}.json", lambda s=season: {"weeks": slate_service.weeks(s)}
        )["body"]["weeks"]
        for week in weeks:
            for estimator in options["estimators"]:
                stem = f"slate/{season}/{week}/{estimator}"
                result: list[str] = []

                def slate_payload(s=season, w=week, e=estimator, keep=result):
                    payload, csv = slate_service.payload_and_csv(s, w, e)
                    keep.append(csv)
                    return payload

                write_data(f"{stem}.json", slate_payload)
                if result:
                    write(f"data/{stem}.csv", result[0])
                    counts["csv"] += 1

    write_data(
        f"schedule/{SCHEDULE_SEASON}.json",
        lambda: slate_service.schedule_records(SCHEDULE_SEASON),
    )

    tracker_options = write_data("tracker/options.json", tracker_service.options)["body"]
    for record_type in tracker_options["record_types"]:
        seasons = tracker_options["seasons"][record_type]
        for season in ["all", *seasons]:
            write_data(
                f"tracker/summary/{record_type}/{season}.json",
                lambda r=record_type, s=season: tracker_service.summary(r, s),
            )
        for season in seasons:
            write_data(
                f"tracker/games/{record_type}/{season}.json",
                lambda r=record_type, s=season: {"games": tracker_service.records(r, s)},
            )
    return counts
