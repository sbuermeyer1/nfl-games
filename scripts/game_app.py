"""Serve the packaged NFL game model dashboard."""

import argparse
import os

import uvicorn

from nfl_game.paths import PROCESSED_DIR
from nfl_game.web.runtime import RuntimeConfigError, load_app, resolve_runtime


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-auth",
        action="store_true",
        help="local-only: disable login and bind to 127.0.0.1",
    )
    args = parser.parse_args(argv)
    try:
        config = resolve_runtime(args.no_auth, os.environ)
        app = load_app(config, PROCESSED_DIR / "game_features.parquet")
    except RuntimeConfigError as exc:
        parser.error(str(exc))
    uvicorn.run(app, host=config.host, port=config.port)


if __name__ == "__main__":
    main()
