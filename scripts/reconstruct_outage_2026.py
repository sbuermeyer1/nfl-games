"""Reconstruct the 2026 week 2-3 tracker records lost to the Sept 16-29 CI outage.

From 2026-09-16 to 2026-09-29 every scheduled refresh and tracker run failed at its pytest
step (seven tests read week 1 from the shipped features artifact; fixed in 87948ad). The
model kept no record of weeks 2 and 3. This script replays what the tracker WOULD have
published, using only information that existed at each moment, and writes the result as a
separate, clearly labelled artifact -- never into the official live ledger, whose records
must be genuinely published before kickoff.

What is replayed, and from where:

* **When.** The real run times of the failed workflows, from the GitHub Actions API. Each
  tracker run is a publication opportunity; each refresh run (commit landing
  `REFRESH_COMMIT_DELAY` after it started, as the last successful refresh did) replaces the
  features artifact the next tracker run checks out.
* **The schedule and lines at a moment.** `data/games.csv` in nflverse/nfldata at the newest
  commit at or before that moment (`nfl_game.data.line_history`). This carries the weather
  and lines as they stood then, not the post-game values the live feed now holds.
* **Play-by-play and NGS at a refresh.** Today's feeds truncated to the 2026 weeks whose
  every game was final in that moment's schedule. Measured on 2026-10-01: rebuilding the
  2026-09-15 refresh this way reproduces the real b2ff001 artifact to within 1e-15 (all 32
  rows, every model feature), and `--verify` re-checks that on every run. NGS timing cannot move a
  prediction: an upcoming week has no NGS rows of its own, so its NGS features are the
  imputed defaults at any cutoff before it is played.
* **The pick itself.** The updater's own `_select_schedule` / `_new_predictions` and
  `advance_live_ledger`, unchanged, so a reconstructed record is exactly the record the
  tracker would have written at that run, then advanced to its closing line and final score.

Only weeks 2 and 3 are kept; `record_type` is "reconstructed". Dry-run by default.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import subprocess
import urllib.request
import warnings
from collections.abc import Callable

import pandas as pd

from nfl_game.data.line_history import games_at
from nfl_game.data.nfl import SCHEDULE_TEAM_COLS, load_ngs, load_pbp
from nfl_game.data.schedule import is_final_game, normalize_schedule
from nfl_game.data.teams import normalize_team_codes
from nfl_game.model.features import FEATURE_COLS
from nfl_game.paths import PROCESSED_DIR, PROJECT_ROOT, RAW_DIR
from nfl_game.pipeline.refresh_2026 import build_refresh_artifacts
from nfl_game.ratings.epa import team_game_epa
from nfl_game.ratings.ngs import team_week_ngs
from nfl_game.tracking.ledger import LEDGER_COLUMNS, validate_ledger
from nfl_game.tracking.live import advance_live_ledger
from nfl_game.web.service import SlateService

try:
    from scripts.refresh_2026 import HISTORICAL_PBP_SEASONS, empty_ngs_frame
    from scripts.update_live_tracker import (
        _first_publishable_week,
        _new_predictions,
        _select_schedule,
        _validate_current_schedule,
    )
except ModuleNotFoundError:  # run as `python scripts/reconstruct_outage_2026.py`
    from refresh_2026 import HISTORICAL_PBP_SEASONS, empty_ngs_frame
    from update_live_tracker import (
        _first_publishable_week,
        _new_predictions,
        _select_schedule,
        _validate_current_schedule,
    )

SEASON = 2026
WEEKS = (2, 3)
RECORD_TYPE = "reconstructed"
REPO = "sbuermeyer1/nfl-games"
#: The last refresh before the outage: it succeeded and committed b2ff001, the features
#: artifact every tracker run in the outage checked out until the next refresh would have.
FIRST_REFRESH_STARTED = pd.Timestamp("2026-09-15T15:08:35Z")
VERIFY_COMMIT = "b2ff001"
#: Measured on that run: started 15:08:35, committed b2ff001 at 15:11:03.
REFRESH_COMMIT_DELAY = pd.Timedelta(seconds=148)
#: The fix (87948ad) was pushed at 01:16 and the next refresh started at 01:22; from then on
#: the real workflows ran, so the replay stops before it.
OUTAGE_END = pd.Timestamp("2026-09-30T01:20:00Z")
#: Float-rounding headroom only: order of operations shifts results by ~1e-16.
VERIFY_TOLERANCE = 1e-12
OUTPUT_PATH = PROCESSED_DIR / "tracker_reconstructed.parquet"
MANIFEST_PATH = PROCESSED_DIR / "tracker_reconstructed_manifest.json"
CACHE_DIR = RAW_DIR / "reconstruct_2026"


def _utc(value) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _get_json(url: str):
    """GET a GitHub API URL, cached on disk: unauthenticated calls are capped at 60/hour."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"api_{hashlib.sha256(url.encode()).hexdigest()[:16]}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    request = urllib.request.Request(url, headers={"User-Agent": "nfl-game-reconstruct"})
    with urllib.request.urlopen(request, timeout=120) as response:
        body = response.read()
    cache.write_bytes(body)
    return json.loads(body)


def workflow_run_times(workflow: str, *, fetch=_get_json) -> list[pd.Timestamp]:
    """Start times of every scheduled run of one workflow between Sept 14 and the fix."""
    times: list[pd.Timestamp] = []
    for page in range(1, 10):
        payload = fetch(
            f"https://api.github.com/repos/{REPO}/actions/workflows/{workflow}/runs"
            f"?per_page=100&page={page}&created=2026-09-14..2026-09-30"
        )
        runs = payload.get("workflow_runs", [])
        times += [_utc(run["created_at"]) for run in runs if run["event"] == "schedule"]
        if len(runs) < 100:
            break
    return sorted(times)


def nfldata_commits(*, fetch=_get_json) -> list[tuple[pd.Timestamp, str]]:
    """Every games.csv commit in nflverse/nfldata from Sept 13 to Oct 1, oldest first."""
    commits: list[tuple[pd.Timestamp, str]] = []
    for page in range(1, 30):
        payload = fetch(
            "https://api.github.com/repos/nflverse/nfldata/commits?path=data/games.csv"
            f"&since=2026-09-13T00:00:00Z&until=2026-10-01T00:00:00Z&per_page=100&page={page}"
        )
        commits += [(_utc(c["commit"]["committer"]["date"]), c["sha"]) for c in payload]
        if len(payload) < 100:
            break
    return sorted(commits)


def commit_at_or_before(commits: list[tuple[pd.Timestamp, str]], moment) -> str:
    stamps = [stamp for stamp, _ in commits]
    index = bisect.bisect_right(stamps, _utc(moment)) - 1
    if index < 0:
        raise ValueError(f"no nfldata commit at or before {moment}")
    return commits[index][1]


def schedule_from_games(games: pd.DataFrame) -> pd.DataFrame:
    """A games.csv snapshot shaped exactly as `load_schedules` + `normalize_schedule` shape it."""
    rows = games.loc[games["season"].eq(SEASON)].reset_index(drop=True)
    return normalize_schedule(normalize_team_codes(rows, SCHEDULE_TEAM_COLS), SEASON)


def final_weeks(schedule: pd.DataFrame, now) -> set[int]:
    """2026 weeks whose every game was final at `now` -- the weeks whose data existed then."""
    moment = _utc(now).to_pydatetime()
    final = {}
    for _, row in schedule.iterrows():
        week = int(row["week"])
        final[week] = final.get(week, True) and is_final_game(row, moment)
    return {week for week, done in final.items() if done}


def truncate_to_weeks(frame: pd.DataFrame, weeks: set[int]) -> pd.DataFrame:
    """Drop 2026 rows from any week not in `weeks`; other seasons are untouched."""
    later = frame["season"].eq(SEASON) & ~frame["week"].astype(int).isin(weeks)
    return frame.loc[~later].reset_index(drop=True)


def features_as_of(
    history: pd.DataFrame,
    schedule: pd.DataFrame,
    team_games: pd.DataFrame,
    ngs: pd.DataFrame,
    now,
) -> pd.DataFrame:
    """The features artifact a refresh running at `now` would have built."""
    known = final_weeks(schedule, now)
    known_ngs = truncate_to_weeks(ngs, known)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        artifacts = build_refresh_artifacts(
            historical_features=history,
            schedules=schedule,
            team_games=truncate_to_weeks(team_games, known),
            ngs=known_ngs if not known_ngs.empty else empty_ngs_frame(),
            now=_utc(now).to_pydatetime(),
        )
    return artifacts.features


def replay(
    tracker_times: list[pd.Timestamp],
    refresh_commits: list[pd.Timestamp],
    schedule_at: Callable[[pd.Timestamp], pd.DataFrame],
    features_at: Callable[[pd.Timestamp], pd.DataFrame],
) -> tuple[pd.DataFrame, list[dict]]:
    """Run the tracker's own publication step at every real tracker run time.

    Each run sees the features of the latest refresh that had committed before it and the
    schedule as it stood at that instant. Returns the advanced live-shaped records and one
    audit entry per run that touched a week-2/3 record.
    """
    live = pd.DataFrame(columns=LEDGER_COLUMNS)
    services: dict[pd.Timestamp, SlateService] = {}
    audit: list[dict] = []
    for now in tracker_times:
        prior = [commit for commit in refresh_commits if commit <= now]
        if not prior:
            continue
        refreshed = prior[-1]
        features = features_at(refreshed)
        service = services.setdefault(refreshed, SlateService(features))
        floor = _first_publishable_week(features, SEASON)
        schedule = schedule_at(now)
        _validate_current_schedule(schedule)
        selected = _select_schedule(schedule, live, now, floor)
        predictions = _new_predictions(service, features, selected, live, SEASON)
        before = set(live["game_id"].astype(str))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            live = advance_live_ledger(
                live, selected, predictions, now, first_publishable_week=floor
            )
        published = sorted(
            game_id
            for game_id, week in zip(live["game_id"].astype(str), live["week"], strict=True)
            if game_id not in before and int(week) in WEEKS
        )
        if published:
            audit.append(
                {
                    "tracker_run": now.isoformat(),
                    "features_from_refresh": refreshed.isoformat(),
                    "first_publishable_week": floor,
                    "published": published,
                }
            )
    return live, audit


def reconstructed_records(live: pd.DataFrame) -> pd.DataFrame:
    """Keep weeks 2-3, relabelled so they can never be read as official live records."""
    kept = live.loc[live["week"].astype(int).isin(WEEKS)].copy()
    kept["record_type"] = RECORD_TYPE
    kept = kept.reindex(columns=LEDGER_COLUMNS).sort_values("game_id").reset_index(drop=True)
    validate_ledger(kept)
    return kept


def _verify_first_refresh(features: pd.DataFrame) -> float:
    """Max |difference| between the rebuilt 2026-09-15 features and the real b2ff001 file."""
    blob = subprocess.run(
        ["git", "show", f"{VERIFY_COMMIT}:data/processed/game_features.parquet"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        check=True,
    ).stdout
    cache = CACHE_DIR / f"game_features_{VERIFY_COMMIT}.parquet"
    cache.write_bytes(blob)
    real = pd.read_parquet(cache)
    real = real.loc[real["season"].eq(SEASON)].sort_values("game_id").reset_index(drop=True)
    rebuilt = features.loc[features["season"].eq(SEASON)].sort_values("game_id")
    rebuilt = rebuilt.reset_index(drop=True)
    if list(real["game_id"]) != list(rebuilt["game_id"]):
        raise AssertionError("rebuilt 2026-09-15 features cover different games than b2ff001")
    diff = (rebuilt[FEATURE_COLS].astype(float) - real[FEATURE_COLS].astype(float)).abs()
    return float(diff.to_numpy().max())


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write", action="store_true", help="replace the artifact and manifest")
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    tracker_runs = workflow_run_times("update-2026-tracker.yml")
    refresh_runs = workflow_run_times("refresh-2026-model.yml")
    refresh_starts = [t for t in refresh_runs if FIRST_REFRESH_STARTED <= t < OUTAGE_END]
    refresh_commits = [t + REFRESH_COMMIT_DELAY for t in refresh_starts]
    tracker_times = [t for t in tracker_runs if refresh_commits[0] <= t < OUTAGE_END]
    commits = nfldata_commits()

    games_cache: dict[str, pd.DataFrame] = {}
    used_shas: dict[str, str] = {}

    def schedule_at(moment: pd.Timestamp) -> pd.DataFrame:
        sha = commit_at_or_before(commits, moment)
        used_shas[moment.isoformat()] = sha
        if sha not in games_cache:
            path = CACHE_DIR / f"games_{sha}.parquet"
            if not path.exists():
                games_at(sha).to_parquet(path)
            games_cache[sha] = schedule_from_games(pd.read_parquet(path))
        return games_cache[sha].copy()

    history = pd.read_parquet(PROCESSED_DIR / "game_features.parquet")
    history = history.loc[history["season"].le(SEASON - 1)].reset_index(drop=True)
    team_games = team_game_epa(load_pbp([*HISTORICAL_PBP_SEASONS, SEASON], save=False))
    ngs = team_week_ngs(
        load_ngs([SEASON], "passing", save=False),
        load_ngs([SEASON], "rushing", save=False),
        load_ngs([SEASON], "receiving", save=False),
    )
    features_cache: dict[pd.Timestamp, pd.DataFrame] = {}

    def features_at(committed: pd.Timestamp) -> pd.DataFrame:
        if committed not in features_cache:
            started = committed - REFRESH_COMMIT_DELAY
            features_cache[committed] = features_as_of(
                history, schedule_at(started), team_games, ngs, started
            )
        return features_cache[committed]

    verify_diff = _verify_first_refresh(features_at(refresh_commits[0]))
    if verify_diff > VERIFY_TOLERANCE:
        raise AssertionError(
            f"rebuilding the 2026-09-15 refresh no longer reproduces {VERIFY_COMMIT} "
            f"(max diff {verify_diff}); the as-of method cannot be trusted"
        )

    live, audit = replay(tracker_times, refresh_commits, schedule_at, features_at)
    records = reconstructed_records(live)
    expected = set(schedule_at(OUTAGE_END).query("week in @WEEKS")["game_id"].astype(str))
    missing = sorted(expected - set(records["game_id"]))
    if missing:
        raise AssertionError(f"replay left week 2-3 games without a record: {missing}")

    manifest = {
        "purpose": "2026 weeks 2-3 tracker records reconstructed after the Sept 16-29 outage",
        "record_type": RECORD_TYPE,
        "records": len(records),
        "verification": {
            "rebuilt_refresh": refresh_starts[0].isoformat(),
            "compared_with": VERIFY_COMMIT,
            "max_abs_feature_diff": verify_diff,
        },
        "refresh_runs_started": [t.isoformat() for t in refresh_starts],
        "refresh_commit_delay_seconds": REFRESH_COMMIT_DELAY.total_seconds(),
        "tracker_runs_replayed": len(tracker_times),
        "publications": audit,
        "nfldata_commit_by_moment": used_shas,
    }
    summary = {
        "mode": "write" if args.write else "dry-run",
        "records": len(records),
        "verification_max_diff": verify_diff,
        "tracker_runs_replayed": len(tracker_times),
        "refresh_runs": len(refresh_starts),
        "spread_grades": records["spread_grade"].value_counts(dropna=False).to_dict(),
        "total_grades": records["total_grade"].value_counts(dropna=False).to_dict(),
    }
    print(json.dumps(summary, sort_keys=True, default=str))
    if args.write:
        records.to_parquet(OUTPUT_PATH, index=False)
        MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
