"""Dry-run or atomically advance the official live tracker ledger."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from nfl_game.data.nfl import load_schedules
from nfl_game.data.schedule import ScheduleSchemaError, normalize_schedule
from nfl_game.data.teams import normalize_team_codes
from nfl_game.paths import PROCESSED_DIR
from nfl_game.tracking.ledger import LEDGER_COLUMNS, validate_ledger
from nfl_game.tracking.live import PUBLISH_BEFORE, advance_live_ledger
from nfl_game.web.service import SlateService

if __package__:
    from scripts.build_tracker import assert_historical_baseline
else:
    from build_tracker import assert_historical_baseline

PREDICTION_COLUMNS = ["game_id", "model_margin", "model_total"]
LEGACY_BACKTEST_MISSING_COLUMNS = {
    "spread_publication_status",
    "total_publication_status",
    "spread_exclusion_reason",
    "total_exclusion_reason",
    "published_spread_observed_at",
    "published_total_observed_at",
    "closing_spread_observed_at",
    "closing_total_observed_at",
    "current_kickoff_at",
    "void_reason",
}
CANONICAL_LEDGER_SCHEMA = tuple(LEDGER_COLUMNS)
LEGACY_BACKTEST_SCHEMA = tuple(
    column for column in LEDGER_COLUMNS if column not in LEGACY_BACKTEST_MISSING_COLUMNS
)
CANONICAL_TIMESTAMP_COLUMNS = {
    "published_at",
    "kickoff_at",
    "published_spread_observed_at",
    "published_total_observed_at",
    "closing_spread_observed_at",
    "closing_total_observed_at",
    "current_kickoff_at",
}
CANONICAL_NUMERIC_COLUMNS = {
    "model_margin",
    "model_total",
    "official_spread_line",
    "official_total_line",
    "published_spread_line",
    "published_total_line",
    "closing_spread_line",
    "closing_total_line",
    "actual_margin",
    "actual_total",
    "spread_edge",
    "total_edge",
    "spread_clv",
    "total_clv",
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Advance the official live tracker from the current nflverse schedule."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="validate and report without writing")
    mode.add_argument("--write", action="store_true", help="atomically replace a changed ledger")
    parser.add_argument("--features", type=Path, default=PROCESSED_DIR / "game_features.parquet")
    parser.add_argument("--ledger", type=Path, default=PROCESSED_DIR / "tracker_ledger.parquet")
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--now", help="UTC lifecycle timestamp (defaults to the current time)")
    parser.add_argument(
        "--void-game",
        action="append",
        default=[],
        metavar="GAME_ID=REASON",
        help="manually void an existing live record; may be repeated",
    )
    return parser


def _utc_timestamp(value, label: str) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a timezone-aware timestamp") from error
    if pd.isna(timestamp) or timestamp.tzinfo is None:
        raise ValueError(f"{label} must be a timezone-aware timestamp")
    return timestamp.tz_convert(UTC)


def _parse_voids(values: list[str]) -> dict[str, str]:
    voids: dict[str, str] = {}
    for value in values:
        game_id, separator, reason = value.partition("=")
        game_id = game_id.strip()
        reason = reason.strip()
        if not separator or not game_id or not reason:
            raise ValueError("--void-game must use GAME_ID=REASON with nonblank values")
        previous = voids.get(game_id)
        if previous is not None and previous != reason:
            raise ValueError(f"conflicting void reasons for {game_id}")
        voids[game_id] = reason
    return voids


def _apply_voids(live: pd.DataFrame, voids: dict[str, str]) -> pd.DataFrame:
    updated = live.copy(deep=True)
    known = set(updated["game_id"].astype(str))
    missing = sorted(set(voids) - known)
    if missing:
        raise ValueError(f"cannot void unknown live game: {missing[0]}")
    for game_id, reason in voids.items():
        row = updated["game_id"].astype(str).eq(game_id)
        existing = updated.loc[row, "void_reason"].iloc[0]
        if pd.notna(existing) and existing != reason:
            raise ValueError(f"live game {game_id} already has a different void reason")
        updated.loc[row, "void_reason"] = reason
    return updated


def _load_ledger(path: Path) -> pd.DataFrame:
    ledger = pd.read_parquet(path)
    schema = tuple(ledger.columns)
    legacy_backtest = (
        schema == LEGACY_BACKTEST_SCHEMA and ledger["record_type"].eq("backtest").all()
    )
    if legacy_backtest:
        ledger = ledger.reindex(columns=LEDGER_COLUMNS)
    elif schema != CANONICAL_LEDGER_SCHEMA:
        raise ValueError("ledger schema must be exactly canonical or approved legacy backtest")
    validate_ledger(ledger)
    return ledger


def _validate_current_schedule(schedule: pd.DataFrame) -> None:
    if schedule.empty:
        raise ScheduleSchemaError("schedule contains no regular-season games for season")

    identity = (
        schedule["game_id"]
        .astype("string")
        .str.extract(
            r"^(?P<season>\d{4})_(?P<week>\d{2})_(?P<away_team>[A-Z]+)_(?P<home_team>[A-Z]+)$"
        )
    )
    if identity.isna().any(axis=None):
        raise ScheduleSchemaError("schedule contains a team mismatch with game_id")
    identity = normalize_team_codes(identity, ["away_team", "home_team"])
    mismatched = (
        pd.to_numeric(identity["season"]).ne(schedule["season"].to_numpy())
        | pd.to_numeric(identity["week"]).ne(schedule["week"].to_numpy())
        | identity["away_team"].ne(schedule["away_team"].to_numpy())
        | identity["home_team"].ne(schedule["home_team"].to_numpy())
    )
    if mismatched.any() or schedule["away_team"].eq(schedule["home_team"]).any():
        raise ScheduleSchemaError("schedule contains a team mismatch with game_id")

    appearances = pd.concat(
        [
            schedule[["week", "away_team"]].rename(columns={"away_team": "team"}),
            schedule[["week", "home_team"]].rename(columns={"home_team": "team"}),
        ],
        ignore_index=True,
    )
    if appearances.duplicated(["week", "team"]).any():
        raise ScheduleSchemaError("schedule contains a team mismatch within a week")


def _validate_feature_schedule_identity(features: pd.DataFrame, schedule: pd.DataFrame) -> None:
    if schedule.empty:
        return
    game_ids = schedule["game_id"].astype(str).tolist()
    selected = features.loc[features["game_id"].astype(str).isin(game_ids)]
    if len(selected) != len(game_ids) or set(selected["game_id"].astype(str)) != set(game_ids):
        raise ValueError("feature identity does not match schedule")

    feature_identity = (
        selected.set_index(selected["game_id"].astype(str))
        .loc[game_ids, ["season", "week", "away_team", "home_team"]]
        .reset_index(drop=True)
    )
    schedule_identity = schedule[["season", "week", "away_team", "home_team"]].reset_index(
        drop=True
    )
    mismatched = (
        feature_identity["season"].astype(int).ne(schedule_identity["season"].astype(int))
        | feature_identity["week"].astype(int).ne(schedule_identity["week"].astype(int))
        | feature_identity["away_team"].astype(str).ne(schedule_identity["away_team"].astype(str))
        | feature_identity["home_team"].astype(str).ne(schedule_identity["home_team"].astype(str))
    )
    if mismatched.any():
        raise ValueError("feature identity does not match schedule")


def _first_publishable_week(features: pd.DataFrame, season: int) -> int | None:
    """The earliest week whose features were built from a complete prior week.

    refresh_2026 appends only `active_prediction_weeks` -- the first two unplayed
    weeks -- so the minimum week present for the season is the one whose predecessors
    were all final at build time. The week after it was built without the current
    week's results, so publishing from it would freeze a stale prediction.
    """
    weeks = features.loc[features["season"].eq(season), "week"]
    if weeks.empty:
        return None
    if weeks.isna().any():
        raise ValueError(
            f"features for season {season} contain {int(weeks.isna().sum())} null week "
            "value(s); the publication floor cannot be derived safely"
        )
    return int(weeks.min())


def _select_schedule(
    schedule: pd.DataFrame,
    live: pd.DataFrame,
    now: pd.Timestamp,
    first_publishable_week: int | None,
) -> pd.DataFrame:
    existing_ids = set(live["game_id"].astype(str))
    if first_publishable_week is None:
        eligible = pd.Series(False, index=schedule.index)
    else:
        eligible = schedule["kickoff_at"].le(now + PUBLISH_BEFORE) & schedule["week"].astype(
            int
        ).eq(int(first_publishable_week))
    existing = schedule["game_id"].astype(str).isin(existing_ids)
    return schedule.loc[eligible | existing].copy()


def _floor_blocked_count(
    schedule: pd.DataFrame,
    live: pd.DataFrame,
    now: pd.Timestamp,
    first_publishable_week: int | None,
) -> int:
    """Games the publication window would take but the vintage floor is holding back.

    A value that stays nonzero across days means the floor is stuck -- typically a game
    whose result never arrived, which pins active_prediction_weeks permanently. Without
    this the stoppage is indistinguishable from an ordinary quiet run.
    """
    existing_ids = set(live["game_id"].astype(str))
    in_window = schedule["kickoff_at"].le(now + PUBLISH_BEFORE)
    unpublished = ~schedule["game_id"].astype(str).isin(existing_ids)
    if first_publishable_week is None:
        blocked = in_window & unpublished
    else:
        blocked = (
            in_window & unpublished & schedule["week"].astype(int).ne(int(first_publishable_week))
        )
    return int(blocked.sum())


def _new_predictions(
    service: SlateService,
    features: pd.DataFrame,
    schedule: pd.DataFrame,
    live: pd.DataFrame,
    season: int,
) -> pd.DataFrame:
    existing_ids = set(live["game_id"].astype(str))
    unpublished = schedule.loc[~schedule["game_id"].astype(str).isin(existing_ids)]
    if unpublished.empty:
        return pd.DataFrame(columns=PREDICTION_COLUMNS)
    _validate_feature_schedule_identity(features, unpublished)

    predictions = []
    for week in sorted(int(value) for value in unpublished["week"].unique()):
        weekly = service.model_predictions(season, week, "ridge")
        predictions.append(weekly.loc[:, PREDICTION_COLUMNS])
    combined = pd.concat(predictions, ignore_index=True)
    combined = combined.loc[
        combined["game_id"].astype(str).isin(set(unpublished["game_id"].astype(str)))
    ].copy()
    if combined["game_id"].duplicated().any():
        raise ValueError("Ridge predictions contain duplicate game_id values")
    expected_ids = set(unpublished["game_id"].astype(str))
    actual_ids = set(combined["game_id"].astype(str))
    if actual_ids != expected_ids:
        raise ValueError("Ridge predictions do not match unpublished eligible games")
    return combined.sort_values("game_id", kind="stable").reset_index(drop=True)


def _canonicalize_ledger(ledger: pd.DataFrame) -> pd.DataFrame:
    canonical = ledger.copy()
    for column in LEDGER_COLUMNS:
        if column in {"season", "week"}:
            canonical[column] = canonical[column].astype("int64")
        elif column in CANONICAL_NUMERIC_COLUMNS:
            canonical[column] = pd.to_numeric(canonical[column], errors="raise").astype("float64")
        elif column in CANONICAL_TIMESTAMP_COLUMNS:
            canonical[column] = pd.to_datetime(canonical[column], utc=True).astype(
                "datetime64[ns, UTC]"
            )
        else:
            canonical[column] = canonical[column].astype(object)
    return canonical


def _validated_parquet_bytes(ledger: pd.DataFrame) -> bytes:
    stream = io.BytesIO()
    _canonicalize_ledger(ledger).to_parquet(stream, index=False)
    payload = stream.getvalue()
    persisted = pd.read_parquet(io.BytesIO(payload))
    validate_ledger(persisted)
    historical = persisted.loc[persisted["record_type"].eq("backtest")]
    assert_historical_baseline(historical)
    return payload


def _atomic_replace(payload: bytes, destination: Path) -> None:
    destination = Path(destination)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.name}.update-",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(payload)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def main(argv=None, loader=None, now=None) -> int:
    """Advance live state with injectable schedule loader and clock; default to dry-run."""
    args = _parser().parse_args(argv)
    current = _utc_timestamp(
        args.now if args.now is not None else (datetime.now(UTC) if now is None else now),
        "tracker clock",
    )
    schedule_loader = load_schedules if loader is None else loader

    features = pd.read_parquet(args.features)
    service = SlateService(features)
    first_publishable_week = _first_publishable_week(features, args.season)
    ledger = _load_ledger(args.ledger)
    historical = ledger.loc[ledger["record_type"].eq("backtest")].copy()
    existing_live = ledger.loc[ledger["record_type"].eq("live")].copy()
    assert_historical_baseline(historical)

    voids = _parse_voids(args.void_game)
    existing_live = _apply_voids(existing_live, voids)
    raw_schedule = schedule_loader([args.season], save=False)
    schedule = normalize_schedule(raw_schedule, args.season)
    _validate_current_schedule(schedule)
    selected_schedule = _select_schedule(schedule, existing_live, current, first_publishable_week)
    floor_blocked = _floor_blocked_count(schedule, existing_live, current, first_publishable_week)
    predictions = _new_predictions(service, features, selected_schedule, existing_live, args.season)
    advanced_live = advance_live_ledger(
        existing_live,
        selected_schedule,
        predictions,
        current,
        first_publishable_week=first_publishable_week,
    )

    combined = (
        historical.copy()
        if advanced_live.empty
        else pd.concat([historical, advanced_live], ignore_index=True)
    )
    combined = combined.reindex(columns=LEDGER_COLUMNS)
    validate_ledger(combined)
    assert_historical_baseline(combined.loc[combined["record_type"].eq("backtest")])
    payload = _validated_parquet_bytes(combined)
    original = args.ledger.read_bytes()
    changed = _digest(payload) != _digest(original)
    existing_ids = set(ledger.loc[ledger["record_type"].eq("live"), "game_id"].astype(str))
    new_records = int((~advanced_live["game_id"].astype(str).isin(existing_ids)).sum())
    voided_records = int(advanced_live["void_reason"].notna().sum())

    if args.write and changed:
        _atomic_replace(payload, args.ledger)

    summary = {
        "changed": changed,
        "first_publishable_week": first_publishable_week,
        "floor_blocked_games": floor_blocked,
        "historical_records": len(historical),
        "live_records": len(advanced_live),
        "mode": "write" if args.write else "dry-run",
        "new_live_records": new_records,
        "voided_records": voided_records,
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
