import importlib

import numpy as np
import pandas as pd
import pytest

from nfl_game.data.source_manifest import SourceContractError


def _pfr():
    return importlib.import_module("nfl_game.ratings.pfr")


def _identity(season=2024, week=1, team="A", opponent="B", player="player-1"):
    return {
        "game_id": f"{season}-{week}-{team}-{opponent}",
        "season": season,
        "week": week,
        "game_type": "REG",
        "team": team,
        "opponent": opponent,
        "pfr_player_id": player,
    }


def _pass_row(**overrides):
    row = {
        **_identity(),
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
    row.update(overrides)
    return row


def _contract_row(stat_type, values=None, **overrides):
    contract = _pfr().PFR_SOURCE_CONTRACT[stat_type]
    defaults = {
        "rush": {
            "attempts": 20.0,
            "yards_before_contact": 50.0,
            "yards_after_contact": 30.0,
            "broken_tackles": 2.0,
        },
        "rec": {"drops": 2.0, "drop_rate": 0.1},
        "def": {
            "pressure_opportunities": 30.0,
            "pressures": 3.0,
            "missed_tackles": 2.0,
            "missed_tackle_rate": 0.1,
        },
    }[stat_type]
    logical = {**defaults, **(values or {})}
    row = {**_identity(), **{source: logical[name] for name, source in contract.items()}}
    row.update(overrides)
    return row


def _frames():
    return {
        "pass": pd.DataFrame([_pass_row()]),
        "rush": pd.DataFrame([_contract_row("rush")]),
        "rec": pd.DataFrame([_contract_row("rec")]),
        "def": pd.DataFrame([_contract_row("def")]),
    }


def _feature_row(season, week, team, value=1.0, missing=None):
    row = {"season": season, "week": week, "team": team}
    row.update({column: value for column in _pfr().PFR_REQUIRED_NUMERIC_COLUMNS})
    if missing is not None:
        row[missing] = np.nan
    return row


def test_team_week_pfr_aggregates_counts_before_forming_rates_and_weighted_averages():
    pfr = _pfr()
    frames = _frames()
    frames["pass"] = pd.DataFrame(
        [
            _pass_row(),
            _pass_row(
                pfr_player_id="player-2",
                passing_drops=2.0,
                passing_bad_throws=4.0,
                times_sacked=2.0,
                times_hurried=1.0,
                times_hit=1.0,
                times_pressured=4.0,
                times_pressured_pct=0.2,
            ),
            _pass_row(
                pfr_player_id="postseason-poison",
                game_type="POST",
                passing_drops=999.0,
                times_pressured=1.0,
                times_pressured_pct=1.0,
            ),
        ]
    )
    frames["rush"] = pd.DataFrame(
        [
            _contract_row(
                "rush",
                {
                    "attempts": 10.0,
                    "yards_before_contact": 20.0,
                    "yards_after_contact": 30.0,
                    "broken_tackles": 1.0,
                },
            ),
            _contract_row(
                "rush",
                {
                    "attempts": 30.0,
                    "yards_before_contact": 90.0,
                    "yards_after_contact": 60.0,
                    "broken_tackles": 3.0,
                },
                pfr_player_id="player-2",
            ),
        ]
    )
    frames["rec"] = pd.DataFrame(
        [
            _contract_row("rec", {"drops": 1.0, "drop_rate": 0.1}),
            _contract_row(
                "rec",
                {"drops": 3.0, "drop_rate": 0.15},
                pfr_player_id="player-2",
            ),
        ]
    )
    frames["def"] = pd.DataFrame(
        [
            _contract_row(
                "def",
                {
                    "pressure_opportunities": 20.0,
                    "pressures": 2.0,
                    "missed_tackles": 1.0,
                    "missed_tackle_rate": 0.1,
                },
            ),
            _contract_row(
                "def",
                {
                    "pressure_opportunities": 40.0,
                    "pressures": 4.0,
                    "missed_tackles": 2.0,
                    "missed_tackle_rate": 0.1,
                },
                pfr_player_id="player-2",
            ),
        ]
    )

    row = pfr.team_week_pfr(frames).iloc[0]

    assert row["pfr_pressure_rate"] == pytest.approx(10 / 50)
    assert row["pfr_hurry_rate"] == pytest.approx(4 / 50)
    assert row["pfr_hit_rate"] == pytest.approx(3 / 50)
    assert row["pfr_bad_throw_rate"] == pytest.approx(6 / 50)
    assert row["pfr_drop_rate"] == pytest.approx(3 / 50)
    assert row["pfr_sack_rate"] == pytest.approx(3 / 50)
    assert row["pfr_rush_ybc"] == pytest.approx(110 / 40)
    assert row["pfr_rush_yac"] == pytest.approx(90 / 40)
    assert row["pfr_broken_tackle_rate"] == pytest.approx(4 / 40)
    assert row["pfr_rec_drop_rate"] == pytest.approx(4 / 30)
    assert row["pfr_def_missed_tackle_rate"] == pytest.approx(3 / 30)
    assert row["pfr_def_pressure_rate"] == pytest.approx(6 / 60)
    assert list(pfr.PFR_FEATURE_COLS) == [
        "pfr_pressure_rate",
        "pfr_hurry_rate",
        "pfr_hit_rate",
        "pfr_bad_throw_rate",
        "pfr_drop_rate",
        "pfr_sack_rate",
        "pfr_rush_ybc",
        "pfr_rush_yac",
        "pfr_broken_tackle_rate",
        "pfr_rec_drop_rate",
        "pfr_def_missed_tackle_rate",
        "pfr_def_pressure_rate",
        "pfr_imputed",
    ]


@pytest.mark.parametrize("stat_type", ["pass", "rush", "rec", "def"])
def test_schema_drift_in_any_pfr_frame_fails_at_the_source_boundary(stat_type):
    pfr = _pfr()
    frames = _frames()
    missing = next(
        column
        for column in pfr.PFR_REQUIRED_COLUMNS[stat_type]
        if column not in pfr.PFR_IDENTITY_COLUMNS
    )
    frames[stat_type] = frames[stat_type].drop(columns=missing)

    with pytest.raises(SourceContractError, match=rf"missing PFR {stat_type} columns.*{missing}"):
        pfr.team_week_pfr(frames)


def test_missing_stat_type_fails_closed():
    pfr = _pfr()
    frames = _frames()
    del frames["rec"]

    with pytest.raises(SourceContractError, match=r"missing PFR stat types: \['rec'\]"):
        pfr.team_week_pfr(frames)


def test_outer_merge_preserves_a_team_week_missing_one_stat_frame():
    pfr = _pfr()
    frames = _frames()
    for stat_type in ("pass", "rec", "def"):
        extra = (
            _pass_row(team="B", opponent="A", pfr_player_id="player-b")
            if stat_type == "pass"
            else _contract_row(
                stat_type,
                team="B",
                opponent="A",
                pfr_player_id="player-b",
            )
        )
        frames[stat_type] = pd.concat([frames[stat_type], pd.DataFrame([extra])], ignore_index=True)

    out = pfr.team_week_pfr(frames).set_index("team")

    assert list(out.index) == ["A", "B"]
    assert out.loc["B", ["pfr_rush_ybc", "pfr_rush_yac"]].isna().all()
    assert pd.notna(out.loc["B", "pfr_pressure_rate"])


def test_zero_denominators_produce_missing_rates_instead_of_infinite_values():
    pfr = _pfr()
    frames = _frames()
    frames["pass"] = pd.DataFrame(
        [
            _pass_row(
                passing_drops=0.0,
                passing_bad_throws=0.0,
                times_sacked=0.0,
                times_hurried=0.0,
                times_hit=0.0,
                times_pressured=0.0,
                times_pressured_pct=0.0,
            )
        ]
    )
    frames["rush"] = pd.DataFrame(
        [
            _contract_row(
                "rush",
                {
                    "attempts": 0.0,
                    "yards_before_contact": 0.0,
                    "yards_after_contact": 0.0,
                    "broken_tackles": 0.0,
                },
            )
        ]
    )
    frames["rec"] = pd.DataFrame([_contract_row("rec", {"drops": 0.0, "drop_rate": 0.0})])
    frames["def"] = pd.DataFrame(
        [
            _contract_row(
                "def",
                {
                    "pressure_opportunities": 0.0,
                    "pressures": 0.0,
                    "missed_tackles": 0.0,
                    "missed_tackle_rate": 0.0,
                },
            )
        ]
    )

    row = pfr.team_week_pfr(frames).iloc[0]

    assert row[list(pfr.PFR_REQUIRED_NUMERIC_COLUMNS)].isna().all()
    assert not np.isinf(row[list(pfr.PFR_REQUIRED_NUMERIC_COLUMNS)].to_numpy(float)).any()


def test_trailing_features_use_only_eight_actual_games_across_byes_and_seasons():
    pfr = _pfr()
    season_weeks = [
        (2023, 10),
        (2023, 12),
        (2023, 15),
        (2023, 18),
        (2024, 1),
        (2024, 3),
        (2024, 7),
        (2024, 10),
        (2024, 14),
    ]
    rows = [
        _feature_row(season, week, "A", value=999.0 if index == 0 else float(index))
        for index, (season, week) in enumerate(season_weeks)
    ]

    out = pfr.pfr_features_for_targets(pd.DataFrame(rows), [(2025, 1)]).set_index("team")
    weights = np.array([0.5 ** (age / 8.0) for age in range(8, 0, -1)])
    expected = np.average(np.arange(1.0, 9.0), weights=weights)

    assert out.loc["A", "pfr_pressure_rate"] == pytest.approx(expected)
    assert out.loc["A", "pfr_imputed"] == 0


def test_target_and_future_rows_cannot_change_as_of_features():
    pfr = _pfr()
    base_rows = [
        _feature_row(2024, 1, "A", 1.0),
        _feature_row(2024, 3, "A", 3.0),
    ]
    baseline = pfr.pfr_features_for_targets(pd.DataFrame(base_rows), [(2024, 5)])
    poisoned_rows = [
        *base_rows,
        _feature_row(2024, 5, "A", 999.0),
        _feature_row(2025, 1, "A", 9999.0),
    ]
    poisoned = pfr.pfr_features_for_targets(pd.DataFrame(poisoned_rows), [(2024, 5)])

    pd.testing.assert_frame_equal(baseline, poisoned)


def test_missing_team_aggregate_is_league_imputed_and_flagged():
    pfr = _pfr()
    metric = "pfr_pressure_rate"
    rows = [
        _feature_row(2024, 1, f"T{index}", float(index + 1), metric if index == 0 else None)
        for index in range(10)
    ]

    out = pfr.pfr_features_for_targets(pd.DataFrame(rows), [(2024, 2)]).set_index("team")

    assert out.loc["T0", metric] == pytest.approx(np.mean(np.arange(2.0, 11.0)))
    assert out.loc["T0", "pfr_imputed"] == 1
    assert out.loc["T1", "pfr_imputed"] == 0
    assert out[list(pfr.PFR_REQUIRED_NUMERIC_COLUMNS)].notna().all().all()


def test_no_pre_target_history_uses_missing_aggregate_imputation():
    pfr = _pfr()
    target_rows = pd.DataFrame([_feature_row(2024, 1, team, value=99.0) for team in ("A", "B")])

    out = pfr.pfr_features_for_targets(target_rows, [(2024, 1)]).set_index("team")

    assert out["pfr_imputed"].eq(1).all()
    assert out[list(pfr.PFR_REQUIRED_NUMERIC_COLUMNS)].eq(0.0).all().all()


def test_coverage_gate_rejects_one_sparse_season_even_when_combined_coverage_passes():
    pfr = _pfr()
    rows = []
    for season in (2023, 2024):
        for index in range(10):
            missing = "pfr_pressure_rate" if season == 2024 and index < 2 else None
            rows.append(_feature_row(season, 1, f"T{index}", 1.0, missing))

    with pytest.raises(SourceContractError, match=r"season 2024.*0\.8000"):
        pfr.pfr_features_for_targets(pd.DataFrame(rows), [(2025, 1)])
