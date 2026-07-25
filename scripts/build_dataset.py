"""Build and cache the full game-features dataset. Run before backtesting."""

import argparse

from nfl_game.data.nfl import load_ngs, load_pbp, load_schedules
from nfl_game.model.features import build_game_features
from nfl_game.paths import PROCESSED_DIR
from nfl_game.ratings.build import ratings_by_week
from nfl_game.ratings.epa import team_game_epa
from nfl_game.ratings.ngs import team_week_ngs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-season", type=int, default=2016)
    ap.add_argument("--end-season", type=int, default=2025)
    args = ap.parse_args()

    seasons = list(range(args.start_season, args.end_season + 1))
    # One extra season of pbp history feeds ratings_by_week so week 1 of the first
    # requested season has strictly-prior games to rate off of. build_ratings raises
    # rather than silently rating off nothing, so without this warm-up season the
    # very first (season, week) in the range fails outright. It is used only to seed
    # ratings history; the final dataset is still filtered down to `seasons` below.
    pbp_seasons = list(range(args.start_season - 1, args.end_season + 1))
    print(f"loading pbp for {pbp_seasons[0]}-{pbp_seasons[-1]} (this takes a few minutes)...")
    pbp = load_pbp(pbp_seasons)
    team_games = team_game_epa(pbp)

    print("building as-of ratings...")
    ratings = ratings_by_week(team_games, seasons=seasons)

    print("building NGS team-weeks...")
    ngs = team_week_ngs(
        load_ngs(seasons, "passing"),
        load_ngs(seasons, "rushing"),
        load_ngs(seasons, "receiving"),
    )

    print("assembling features...")
    feats = build_game_features(load_schedules(), ratings, ngs)
    feats = feats[feats["season"].isin(seasons)]

    path = PROCESSED_DIR / "game_features.parquet"
    feats.to_parquet(path)
    print(f"wrote {len(feats)} games to {path}")


if __name__ == "__main__":
    main()
