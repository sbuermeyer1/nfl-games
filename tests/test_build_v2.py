from __future__ import annotations

import copy
import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nfl_game.model.features import FEATURE_COLS
from nfl_game.model.v2_config import (
    PRIOR_SEASON_WEIGHTS,
    RATING_WINDOWS,
    FeatureManifest,
    TargetConfig,
)
from nfl_game.paths import V2_FEATURES_PATH, V2_MANIFEST_PATH
from nfl_game.pipeline.build_v2 import (
    V2BuildInputs,
    build_v2_artifacts,
    semantic_frame_digest,
    write_v2_artifacts_atomic,
)

FIXED_UTC = datetime(2026, 8, 13, 12, 0, tzinfo=UTC)
TEAMS = ("BUF", "KC", "MIA", "NE")


def _base_features() -> pd.DataFrame:
    rows = []
    for game_id, home, away, margin, total in (
        ("2021_01_KC_BUF", "BUF", "KC", 3.0, 51.0),
        ("2021_01_NE_MIA", "MIA", "NE", -7.0, 37.0),
    ):
        row = {
            "game_id": game_id,
            "season": np.int32(2021),
            "week": np.int32(1),
            "home_team": home,
            "away_team": away,
            "spread_line": -2.5,
            "total_line": 47.5,
            "margin": margin,
            "total_points": total,
        }
        row.update({column: 0.0 for column in FEATURE_COLS})
        row.update(
            {
                "rest_diff": np.int32(2),
                "is_dome": np.int64(0),
                "temp_outdoor": 62.0,
                "wind_outdoor": 7.0,
                "div_game": np.int64(game_id.endswith("NE_MIA")),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _schedules() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "game_id": row.game_id,
                "season": 2021,
                "week": 1,
                "game_type": "REG",
                "home_team": row.home_team,
                "away_team": row.away_team,
                "gameday": "2021-09-12",
                "gametime": "13:00",
                "kickoff_at": "2021-09-12T17:00:00Z",
            }
            for row in _base_features().itertuples(index=False)
        ]
        + [
            {
                "game_id": "postseason-poison",
                "season": 2021,
                "week": 19,
                "game_type": "POST",
                "home_team": "BUF",
                "away_team": "KC",
                "gameday": "2022-01-15",
                "gametime": "20:00",
                "kickoff_at": "2022-01-16T01:00:00Z",
            }
        ]
    )


def _pbp() -> pd.DataFrame:
    rows = []
    matchups = (("BUF", "KC"), ("MIA", "NE"))
    strengths = {"BUF": 0.35, "KC": 0.20, "MIA": -0.05, "NE": -0.20}
    for season in (2019, 2020):
        for week in (1, 9, 17):
            for game_number, (home, away) in enumerate(matchups):
                game_id = f"{season}_{week:02d}_{away}_{home}"
                for team, opponent in ((home, away), (away, home)):
                    trend = (week / 100) * (1 if team in {"BUF", "MIA"} else -1)
                    season_shift = (season - 2019) * (0.08 if team in {"BUF", "NE"} else -0.04)
                    for play, (is_pass, is_rush, yards) in enumerate(((1, 0, 22), (0, 1, 11))):
                        epa = strengths[team] + trend + season_shift + play * 0.03
                        rows.append(
                            {
                                "game_id": game_id,
                                "season": season,
                                "week": week,
                                "season_type": "REG",
                                "posteam": team,
                                "defteam": opponent,
                                "home_team": home,
                                "away_team": away,
                                "pass": is_pass,
                                "rush": is_rush,
                                "qb_dropback": is_pass,
                                "sack": 0,
                                "down": play + 1,
                                "qtr": 1,
                                "posteam_score_differential": 0,
                                "epa": epa,
                                "success": int(epa > 0),
                                "yards_gained": yards,
                                "game_seconds_remaining": 900 - play * 28,
                                "yardline_100": 75 - play * 5,
                                "drive": game_number + 1,
                                "play_id": play + 1,
                                "interception": 0,
                                "fumble_lost": 0,
                                "special_teams_play": 0,
                            }
                        )
    return pd.DataFrame(rows)


def _player_stats() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "season": season,
                "week": week,
                "season_type": "REG",
                "team": team,
                "opponent_team": TEAMS[(index + 1) % len(TEAMS)],
                "player_id": f"qb-{team.lower()}",
                "position": "QB",
                "attempts": 30,
                "sacks_suffered": 2,
                "passing_epa": 2.0 + index + (season - 2019),
                "passing_cpoe": float(index),
                "passing_interceptions": index % 2,
            }
            for season in (2019, 2020)
            for week in (1, 9, 17)
            for index, team in enumerate(TEAMS)
        ]
    )


def _players() -> pd.DataFrame:
    return pd.DataFrame(
        [{"pfr_id": f"pfr-{team.lower()}", "gsis_id": f"qb-{team.lower()}"} for team in TEAMS]
    )


def _depth_charts() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "season": 2021,
                "week": week,
                "club_code": team,
                "position": "QB",
                "depth_team": "1",
                "player_id": f"qb-{team.lower()}",
                "gsis_id": f"qb-{team.lower()}",
                "dt": pd.NaT,
            }
            for team in TEAMS
            for week in (1,)
        ]
    )


def _rosters() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "season": 2021,
                "week": 1,
                "team": team,
                "gsis_id": f"qb-{team.lower()}",
                "dt": pd.NaT,
            }
            for team in TEAMS
        ]
    )


def _snap_counts() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "season": 2020,
                "week": 17,
                "team": team,
                "opponent": TEAMS[(index + 1) % len(TEAMS)],
                "pfr_player_id": f"pfr-{team.lower()}",
                "offense_snaps": 60,
                "defense_snaps": 60,
            }
            for index, team in enumerate(TEAMS)
        ]
    )


def _pfr_identity(team: str, index: int) -> dict[str, object]:
    return {
        "game_id": f"2020-17-{team}",
        "season": 2020,
        "week": 17,
        "game_type": "REG",
        "team": team,
        "opponent": TEAMS[(index + 1) % len(TEAMS)],
        "pfr_player_id": f"pfr-{team.lower()}",
    }


def _pfr_frames() -> dict[str, pd.DataFrame]:
    frames = {name: [] for name in ("pass", "rush", "rec", "def")}
    for index, team in enumerate(TEAMS):
        identity = _pfr_identity(team, index)
        frames["pass"].append(
            {
                **identity,
                "passing_drops": 1.0,
                "passing_bad_throws": 2.0,
                "times_sacked": 1.0,
                "times_blitzed": 8.0,
                "times_hurried": 3.0,
                "times_hit": 2.0,
                "times_pressured": 6.0,
                "times_pressured_pct": 0.2,
                "def_times_blitzed": 0.0,
                "def_times_hurried": 0.0,
                "def_times_hitqb": 0.0,
            }
        )
        frames["rush"].append(
            {
                **identity,
                "carries": 20.0,
                "rushing_yards_before_contact": 50.0,
                "rushing_yards_after_contact": 30.0,
                "rushing_broken_tackles": 2.0,
            }
        )
        frames["rec"].append({**identity, "receiving_drop": 2.0, "receiving_drop_pct": 0.1})
        frames["def"].append(
            {
                **identity,
                "def_targets": 30.0,
                "def_pressures": 3.0,
                "def_missed_tackles": 2.0,
                "def_missed_tackle_pct": 0.1,
            }
        )
    return {name: pd.DataFrame(rows) for name, rows in frames.items()}


def fake_inputs() -> V2BuildInputs:
    ngs = pd.DataFrame(
        {
            "season": [2020],
            "week": [17],
            "team_abbr": ["BUF"],
            "stat_type": ["passing"],
        }
    )
    return V2BuildInputs(
        schedules=_schedules(),
        pbp=_pbp(),
        ngs=ngs,
        player_stats=_player_stats(),
        players=_players(),
        rosters=_rosters(),
        depth_charts=_depth_charts(),
        snap_counts=_snap_counts(),
        pfr=_pfr_frames(),
        base_features=_base_features(),
    )


def _cli_loaders(inputs: V2BuildInputs, builder) -> dict[str, object]:
    """Loader injections for the CLI that return the in-memory fake frames."""
    return {
        "load_schedules": lambda *args, **kwargs: inputs.schedules,
        "load_pbp": lambda *args, **kwargs: inputs.pbp,
        "load_ngs": lambda *args, **kwargs: inputs.ngs,
        "load_player_stats": lambda *args, **kwargs: inputs.player_stats,
        "load_players": lambda *args, **kwargs: inputs.players,
        "load_rosters_weekly": lambda *args, **kwargs: inputs.rosters,
        "load_depth_charts": lambda *args, **kwargs: inputs.depth_charts,
        "load_snap_counts": lambda *args, **kwargs: inputs.snap_counts,
        "load_pfr_advstats": lambda seasons, stat_type, save=False: inputs.pfr[stat_type],
        "read_parquet": lambda path: inputs.base_features,
        "build_v2_artifacts": builder,
    }


@pytest.fixture(scope="module")
def built():
    return build_v2_artifacts(fake_inputs(), retrieved_at=FIXED_UTC, evaluation_seasons=(2021,))


def test_builder_preserves_c0_values_dtypes_lines_and_targets_exactly(built):
    expected = _base_features().reset_index(drop=True)

    pd.testing.assert_frame_equal(built.features[expected.columns], expected)
    assert len(built.features) == 2
    assert not built.features["game_id"].duplicated().any()


def test_builder_materializes_real_distinct_rating_variants_and_selection_contract(built):
    manifest = FeatureManifest.from_dict(built.manifest["feature_manifest"])
    manifest.validate_selection_contract()
    physical = []
    for short, long in RATING_WINDOWS:
        for prior in PRIOR_SEASON_WEIGHTS:
            config = TargetConfig("C1", 1.0, short, long, prior)
            physical.append(
                manifest.rating_variant_columns("margin", config)["rating_net_diff_short"]
            )

    assert len(physical) == 9
    assert built.features[physical].nunique(axis=1).gt(1).all()
    assert built.manifest["selection_contract_valid"] is True


def test_builder_records_all_sources_coverage_schema_and_semantic_digests(built):
    names = {snapshot["name"] for snapshot in built.manifest["source_snapshots"]}
    assert names == {
        "schedules",
        "pbp",
        "ngs",
        "player_stats",
        "players",
        "rosters",
        "depth_charts",
        "snap_counts",
        "pfr_pass",
        "pfr_rush",
        "pfr_rec",
        "pfr_def",
        "base_features",
    }
    assert all(snapshot["schema_sha256"] for snapshot in built.manifest["source_snapshots"])
    inputs = fake_inputs()
    assert {
        snapshot["name"]: snapshot["rows"] for snapshot in built.manifest["source_snapshots"]
    } == {
        "schedules": len(inputs.schedules),
        "pbp": len(inputs.pbp),
        "ngs": len(inputs.ngs),
        "player_stats": len(inputs.player_stats),
        "players": len(inputs.players),
        "rosters": len(inputs.rosters),
        "depth_charts": len(inputs.depth_charts),
        "snap_counts": len(inputs.snap_counts),
        "pfr_pass": len(inputs.pfr["pass"]),
        "pfr_rush": len(inputs.pfr["rush"]),
        "pfr_rec": len(inputs.pfr["rec"]),
        "pfr_def": len(inputs.pfr["def"]),
        "base_features": len(inputs.base_features),
    }
    assert built.manifest["block_coverage"]["C5"]["production_eligible"] is False
    assert built.manifest["output"]["features_semantic_sha256"] == semantic_frame_digest(
        built.features
    )
    assert built.manifest["build_timestamp"] == FIXED_UTC.isoformat()


def test_builder_fails_closed_when_production_block_team_week_coverage_is_below_ninety_percent():
    inputs = fake_inputs()
    schedules = inputs.schedules.loc[inputs.schedules["game_id"].ne("postseason-poison")].copy()
    schedules = pd.concat(
        [
            schedules,
            pd.DataFrame(
                [
                    {
                        "game_id": "2021_01_BUF_DAL",
                        "season": 2021,
                        "week": 1,
                        "game_type": "REG",
                        "home_team": "DAL",
                        "away_team": "BUF",
                        "gameday": "2021-09-12",
                        "gametime": "13:00",
                        "kickoff_at": "2021-09-12T17:00:00Z",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    base = inputs.base_features.copy()
    extra = base.iloc[[0]].assign(game_id="2021_01_BUF_DAL", home_team="DAL", away_team="BUF")
    base = pd.concat([base, extra], ignore_index=True)
    low_coverage = V2BuildInputs(
        **{
            **inputs.__dict__,
            "schedules": schedules,
            "base_features": base,
        }
    )

    with pytest.raises(ValueError, match=r"C1.*coverage below 0\.9000"):
        build_v2_artifacts(low_coverage, retrieved_at=FIXED_UTC, evaluation_seasons=(2021,))


def test_fixed_clock_and_semantic_digest_make_build_reproducible(built):
    rebuilt = build_v2_artifacts(fake_inputs(), retrieved_at=FIXED_UTC, evaluation_seasons=(2021,))

    pd.testing.assert_frame_equal(built.features, rebuilt.features)
    assert built.manifest == rebuilt.manifest


def _manifest_without_clock_fields(manifest: dict) -> dict:
    """Strip every clock-derived leaf, computed independently of the production helper."""
    payload = copy.deepcopy(manifest)
    payload.pop("build_timestamp", None)
    for snapshot in payload.get("source_snapshots", []):
        snapshot.pop("retrieved_at", None)
    return payload


def test_semantic_manifest_digests_are_stable_across_a_moving_build_clock():
    later = FIXED_UTC + timedelta(days=9, hours=3, minutes=17)
    first = build_v2_artifacts(fake_inputs(), retrieved_at=FIXED_UTC, evaluation_seasons=(2021,))
    second = build_v2_artifacts(fake_inputs(), retrieved_at=later, evaluation_seasons=(2021,))

    # The clock genuinely moved; without this the digests could match for the wrong reason.
    assert first.manifest["build_timestamp"] != second.manifest["build_timestamp"]
    assert [snapshot["retrieved_at"] for snapshot in first.manifest["source_snapshots"]] != [
        snapshot["retrieved_at"] for snapshot in second.manifest["source_snapshots"]
    ]

    assert first.manifest["source_manifest_sha256"] == second.manifest["source_manifest_sha256"]
    assert (
        first.manifest["output"]["manifest_semantic_sha256"]
        == second.manifest["output"]["manifest_semantic_sha256"]
    )
    assert _manifest_without_clock_fields(first.manifest) == _manifest_without_clock_fields(
        second.manifest
    )


def test_semantic_manifest_digests_still_move_when_a_source_changes():
    inputs = fake_inputs()
    extra_player = pd.DataFrame([{"pfr_id": "pfr-unused", "gsis_id": "qb-unused"}])
    changed = replace(inputs, players=pd.concat([inputs.players, extra_player], ignore_index=True))

    baseline = build_v2_artifacts(inputs, retrieved_at=FIXED_UTC, evaluation_seasons=(2021,))
    moved = build_v2_artifacts(changed, retrieved_at=FIXED_UTC, evaluation_seasons=(2021,))

    assert baseline.manifest["source_row_counts"]["players"] == len(inputs.players)
    assert moved.manifest["source_row_counts"]["players"] == len(inputs.players) + 1
    assert baseline.manifest["source_manifest_sha256"] != moved.manifest["source_manifest_sha256"]
    assert (
        baseline.manifest["output"]["manifest_semantic_sha256"]
        != moved.manifest["output"]["manifest_semantic_sha256"]
    )


def test_atomic_writer_round_trips_pair_and_identical_second_write(built, tmp_path):
    features = tmp_path / "features.parquet"
    manifest = tmp_path / "manifest.json"

    write_v2_artifacts_atomic(built, features, manifest)
    first_bytes = manifest.read_bytes()
    first_digest = semantic_frame_digest(pd.read_parquet(features))
    write_v2_artifacts_atomic(built, features, manifest)

    assert manifest.read_bytes() == first_bytes
    assert semantic_frame_digest(pd.read_parquet(features)) == first_digest
    assert json.loads(manifest.read_text(encoding="utf-8")) == built.manifest


def test_atomic_writer_restores_both_originals_after_second_replacement_fails(
    built, tmp_path, monkeypatch
):
    features = tmp_path / "features.parquet"
    manifest = tmp_path / "manifest.json"
    features.write_bytes(b"old-feature-bytes")
    manifest.write_bytes(b"old-manifest-bytes")
    original_replace = Path.replace
    failed = False

    def fail_manifest_publication(self, target):
        nonlocal failed
        if not failed and Path(target) == manifest and ".publish-" in self.name:
            failed = True
            raise OSError("second replacement failed")
        return original_replace(self, target)

    monkeypatch.setattr(Path, "replace", fail_manifest_publication)

    with pytest.raises(OSError, match="second replacement failed"):
        write_v2_artifacts_atomic(built, features, manifest)

    assert features.read_bytes() == b"old-feature-bytes"
    assert manifest.read_bytes() == b"old-manifest-bytes"
    assert not list(tmp_path.glob(".*.tmp"))


def test_atomic_writer_replaces_malformed_existing_destinations(built, tmp_path):
    features = tmp_path / "features.parquet"
    manifest = tmp_path / "manifest.json"
    features.write_bytes(b"not parquet")
    manifest.write_text("not json", encoding="utf-8")

    write_v2_artifacts_atomic(built, features, manifest)

    assert semantic_frame_digest(pd.read_parquet(features)) == semantic_frame_digest(built.features)
    assert json.loads(manifest.read_text(encoding="utf-8")) == built.manifest


def test_cli_default_dry_run_declares_loader_ranges_and_writes_nothing(built, tmp_path, capsys):
    from scripts import build_v2_dataset

    calls = []
    inputs = fake_inputs()

    def record(name, value):
        def loader(*args, **kwargs):
            calls.append((name, args, kwargs))
            return copy.deepcopy(value)

        return loader

    loaders = {
        "load_schedules": record("schedules", inputs.schedules),
        "load_pbp": record("pbp", inputs.pbp),
        "load_ngs": record("ngs", inputs.ngs),
        "load_player_stats": record("player_stats", inputs.player_stats),
        "load_players": record("players", inputs.players),
        "load_rosters_weekly": record("rosters", inputs.rosters),
        "load_depth_charts": record("depth_charts", inputs.depth_charts),
        "load_snap_counts": record("snap_counts", inputs.snap_counts),
        "load_pfr_advstats": lambda seasons, stat_type, save=False: (
            calls.append((f"pfr_{stat_type}", (seasons, stat_type), {"save": save}))
            or inputs.pfr[stat_type].copy()
        ),
        "read_parquet": record("read_parquet", inputs.base_features),
        "build_v2_artifacts": lambda loaded, retrieved_at: built,
    }
    feature_path = tmp_path / "features.parquet"
    manifest_path = tmp_path / "manifest.json"

    result = build_v2_dataset.main(
        ["--features", str(feature_path), "--manifest", str(manifest_path)],
        loaders=loaders,
        retrieved_at=FIXED_UTC,
    )

    assert result == 0
    assert not feature_path.exists()
    assert not manifest_path.exists()
    assert ("pbp", (list(range(2015, 2026)),), {"save": False}) in calls
    assert ("player_stats", (list(range(2016, 2026)),), {"save": False}) in calls
    assert ("pfr_pass", (list(range(2018, 2026)), "pass"), {"save": False}) in calls
    assert "dry-run: no artifacts changed" in capsys.readouterr().out


def test_cli_write_targets_only_v2_destinations_and_preserves_v1_files(built, tmp_path):
    from scripts import build_v2_dataset

    inputs = fake_inputs()
    v1_features = tmp_path / "game_features.parquet"
    v1_ledger = tmp_path / "tracker_ledger.parquet"
    v1_features.write_bytes(b"v1-features")
    v1_ledger.write_bytes(b"v1-ledger")
    loaders = {
        "load_schedules": lambda *args, **kwargs: inputs.schedules,
        "load_pbp": lambda *args, **kwargs: inputs.pbp,
        "load_ngs": lambda *args, **kwargs: inputs.ngs,
        "load_player_stats": lambda *args, **kwargs: inputs.player_stats,
        "load_players": lambda *args, **kwargs: inputs.players,
        "load_rosters_weekly": lambda *args, **kwargs: inputs.rosters,
        "load_depth_charts": lambda *args, **kwargs: inputs.depth_charts,
        "load_snap_counts": lambda *args, **kwargs: inputs.snap_counts,
        "load_pfr_advstats": lambda seasons, stat_type, save=False: inputs.pfr[stat_type],
        "read_parquet": lambda path: inputs.base_features,
        "build_v2_artifacts": lambda loaded, retrieved_at: built,
    }
    features = tmp_path / V2_FEATURES_PATH.name
    manifest = tmp_path / V2_MANIFEST_PATH.name

    result = build_v2_dataset.main(
        ["--write", "--features", str(features), "--manifest", str(manifest)],
        loaders=loaders,
        retrieved_at=FIXED_UTC,
    )

    assert result == 0
    assert features.exists() and manifest.exists()
    assert v1_features.read_bytes() == b"v1-features"
    assert v1_ledger.read_bytes() == b"v1-ledger"


def test_cli_parser_rejects_combined_dry_run_and_write():
    from scripts import build_v2_dataset

    with pytest.raises(SystemExit):
        build_v2_dataset._parser().parse_args(["--dry-run", "--write"])


def test_atomic_writer_refuses_a_ridge_v1_destination_and_leaves_it_untouched(built, tmp_path):
    """Nothing in the v2 pipeline may republish a frozen Ridge-v1 artifact."""
    v1_features = tmp_path / "game_features.parquet"
    v1_features.write_bytes(b"frozen-v1")

    with pytest.raises(ValueError, match="Ridge-v1"):
        write_v2_artifacts_atomic(built, v1_features, tmp_path / V2_MANIFEST_PATH.name)

    assert v1_features.read_bytes() == b"frozen-v1"


def test_atomic_writer_refuses_a_ridge_v1_manifest_destination(built, tmp_path):
    """The manifest side of the pair needs the same guard as the feature side."""
    v1_ledger = tmp_path / "tracker_ledger.parquet"
    v1_ledger.write_bytes(b"frozen-v1-ledger")

    with pytest.raises(ValueError, match="Ridge-v1"):
        write_v2_artifacts_atomic(built, tmp_path / V2_FEATURES_PATH.name, v1_ledger)

    assert v1_ledger.read_bytes() == b"frozen-v1-ledger"


def test_cli_write_refuses_a_ridge_v1_destination_and_exits_nonzero(built, tmp_path, capsys):
    from scripts import build_v2_dataset

    inputs = fake_inputs()
    v1_features = tmp_path / "game_features.parquet"
    v1_features.write_bytes(b"frozen-v1")
    loaders = {
        "load_schedules": lambda *args, **kwargs: inputs.schedules,
        "load_pbp": lambda *args, **kwargs: inputs.pbp,
        "load_ngs": lambda *args, **kwargs: inputs.ngs,
        "load_player_stats": lambda *args, **kwargs: inputs.player_stats,
        "load_players": lambda *args, **kwargs: inputs.players,
        "load_rosters_weekly": lambda *args, **kwargs: inputs.rosters,
        "load_depth_charts": lambda *args, **kwargs: inputs.depth_charts,
        "load_snap_counts": lambda *args, **kwargs: inputs.snap_counts,
        "load_pfr_advstats": lambda seasons, stat_type, save=False: inputs.pfr[stat_type],
        "read_parquet": lambda path: inputs.base_features,
        "build_v2_artifacts": lambda loaded, retrieved_at: built,
    }

    result = build_v2_dataset.main(
        [
            "--write",
            "--features",
            str(v1_features),
            "--manifest",
            str(tmp_path / V2_MANIFEST_PATH.name),
        ],
        loaders=loaders,
        retrieved_at=FIXED_UTC,
    )

    assert result == 1
    assert v1_features.read_bytes() == b"frozen-v1"


def test_builder_fails_when_a_required_evaluation_season_is_absent_from_the_corpus():
    """A truncated corpus must stop the build, not silently drop the season's gate."""
    with pytest.raises(ValueError, match=r"C1.*2022"):
        build_v2_artifacts(fake_inputs(), retrieved_at=FIXED_UTC, evaluation_seasons=(2021, 2022))


def test_c5_drop_rate_coverage_is_measured_from_the_build_not_a_frozen_literal(built):
    """A literal beside computed coverage records a measurement the build never made."""
    key = "pfr_rec_drop_rate_source_coverage"
    complete = built.manifest["block_coverage"]["C5"][key]

    inputs = fake_inputs()
    rec = inputs.pfr["rec"].copy()
    rec.loc[rec["team"].isin(("MIA", "NE")), ["receiving_drop", "receiving_drop_pct"]] = np.nan
    degraded = build_v2_artifacts(
        replace(inputs, pfr={**inputs.pfr, "rec": rec}),
        retrieved_at=FIXED_UTC,
        evaluation_seasons=(2021,),
    ).manifest["block_coverage"]["C5"][key]

    assert complete == {"2020": pytest.approx(1.0)}
    assert degraded == {"2020": pytest.approx(0.5)}


def test_cli_traceback_flag_reports_the_failing_stack(built, tmp_path, capsys):
    """The bare CLI boundary printed only str(exc); a 30-minute build failed on one line."""
    from scripts import build_v2_dataset

    inputs = fake_inputs()

    def explode(loaded, retrieved_at):
        raise ValueError("deliberate build failure")

    loaders = _cli_loaders(inputs, builder=explode)

    quiet = build_v2_dataset.main(["--dry-run"], loaders=loaders, retrieved_at=FIXED_UTC)
    quiet_err = capsys.readouterr().err
    loud = build_v2_dataset.main(
        ["--dry-run", "--traceback"], loaders=loaders, retrieved_at=FIXED_UTC
    )
    loud_err = capsys.readouterr().err

    assert quiet == 1 and loud == 1
    assert "Traceback (most recent call last)" not in quiet_err
    assert "Traceback (most recent call last)" in loud_err
    assert "deliberate build failure" in loud_err
