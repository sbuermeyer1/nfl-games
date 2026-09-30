"""Export the dashboard as a static site for Firebase Hosting (`firebase.json` serves `site/`).

Builds the slate and tracker services exactly as `scripts/game_app.py` does -- including
the live nflverse market overlay for 2026 lines -- and pre-renders every page and API
response into `--out`. The result is only as fresh as this run; see
`src/nfl_game/web/static_export.py`.

    python scripts/export_static_site.py
    npx firebase-tools deploy --only hosting
"""

from __future__ import annotations

import argparse
import time
import warnings

from nfl_game.paths import PROCESSED_DIR, PROJECT_ROOT
from nfl_game.web.runtime import load_services
from nfl_game.web.static_export import export_site


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(PROJECT_ROOT / "site"), help="output directory")
    args = parser.parse_args(argv)
    started = time.monotonic()
    slate_service, tracker_service = load_services(
        PROCESSED_DIR / "game_features.parquet",
        PROCESSED_DIR / "tracker_ledger.parquet",
        PROCESSED_DIR / "schedule_2026.parquet",
    )
    # walk_forward warns once per skipped early season on every fit; the served app
    # emits the same warnings on first request. They are expected and drown the output.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        counts = export_site(args.out, slate_service, tracker_service)
    elapsed = time.monotonic() - started
    print(
        f"wrote {counts['pages']} pages, {counts['data']} data files and {counts['csv']} CSVs "
        f"to {args.out} in {elapsed:.0f}s"
    )


if __name__ == "__main__":
    main()
