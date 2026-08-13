import importlib

import numpy as np
import pandas as pd
import pytest

from nfl_game.model.features import FEATURE_COLS
from nfl_game.model.v2_config import CANDIDATES, MARKET_COLUMNS, MARKET_PROBABILITY_COLUMNS


def _v2_features():
    return importlib.import_module("nfl_game.model.v2_features")


def _base_features() -> pd.DataFrame:
    rows = []
    for game_id, home, away, margin, total in (
        ("g1", "BUF", "KC", 7.0, 51.0),
        ("g2", "OAK", "SD", -3.0, 41.0),
    ):
        row = {
            "game_id": game_id,
            "season": 2024,
            "week": 2,
            "home_team": home,
            "away_team": away,
            "spread_line": -2.5,
            "total_line": 47.5,
            "home_moneyline": -135.0,
            "away_moneyline": 115.0,
            "cover_prob": 0.55,
            "over_prob": 0.51,
            "margin": margin,
            "total_points": total,
        }
        row.update({column: 0.0 for column in FEATURE_COLS})
        row.update(
            {
                "rest_diff": 2.0,
                "is_dome": float(game_id == "g2"),
                "temp_outdoor": 52.0 if game_id == "g1" else 0.0,
                "wind_outdoor": 8.0 if game_id == "g1" else 0.0,
                "div_game": float(game_id == "g2"),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _rating_row(team: str, off: float, defense: float, week: int = 2) -> dict[str, object]:
    row: dict[str, object] = {"season": 2024, "week": week, "team": team}
    targets = (
        "epa_play",
        "epa_pass",
        "epa_rush",
        "success_rate",
        "early_down_epa",
        "neutral_epa",
        "explosive_pass_rate",
        "explosive_rush_rate",
    )
    for window in ("short", "long"):
        for target in targets:
            row[f"{window}_off_{target}"] = off
            row[f"{window}_def_{target}"] = defense
    return row


def _blocks() -> dict[str, pd.DataFrame]:
    teams = ("BUF", "KC", "LV", "LAC")
    ratings = pd.DataFrame(
        [
            _rating_row("BUF", 10.0, 1.0),
            _rating_row("KC", 4.0, 2.0),
            _rating_row("LV", 3.0, 0.5),
            _rating_row("LAC", 2.0, 1.5),
        ]
    )
    qb = pd.DataFrame(
        [
            {
                "season": 2024,
                "week": 2,
                "team": team,
                "qb_epa_per_db": value,
                "qb_cpoe": value / 10,
                "qb_sack_rate": value / 100,
                "qb_int_rate": value / 200,
                "qb_change_epa": value / 20,
                "qb_new_starter": int(team == "KC"),
                "qb_rookie": int(team == "LAC"),
                "qb_uncertain": int(team == "LV"),
            }
            for team, value in zip(teams, (10.0, 4.0, 3.0, 2.0), strict=True)
        ]
    )
    style = pd.DataFrame(
        [
            {
                "season": 2024,
                "week": 2,
                "team": team,
                "neutral_pass_rate": pass_rate,
                "pace_seconds": pace,
                "turnover_rate": turnover,
                "explosive_play_rate": explosive,
                "starting_field_position": field_position,
                "special_teams_epa": special_teams,
                "style_imputed": 0,
            }
            for team, pass_rate, pace, turnover, explosive, field_position, special_teams in (
                ("BUF", 0.60, 24.0, 0.01, 0.12, 31.0, 0.20),
                ("KC", 0.40, 30.0, 0.03, 0.08, 27.0, -0.10),
                ("LV", 0.55, 27.0, 0.02, 0.09, 29.0, 0.00),
                ("LAC", 0.45, 29.0, 0.04, 0.07, 26.0, 0.10),
            )
        ]
    )
    personnel = pd.DataFrame(
        [
            {
                "season": 2024,
                "week": 2,
                "team": team,
                "off_returning_share": returning,
                "def_returning_share": returning - 0.1,
                "off_snap_hhi": hhi,
                "def_snap_hhi": hhi + 0.05,
                "depth_chart_change_rate": change,
                "roster_churn": churn,
                "personnel_imputed": 0,
            }
            for team, returning, hhi, change, churn in (
                ("BUF", 0.80, 0.20, 0.10, 0.15),
                ("KC", 0.60, 0.30, 0.20, 0.25),
                ("LV", 0.70, 0.25, 0.15, 0.20),
                ("LAC", 0.65, 0.28, 0.12, 0.22),
            )
        ]
    )
    pfr = pd.DataFrame(
        [
            {
                "season": 2024,
                "week": 2,
                "team": team,
                "pfr_pressure_rate": pressure,
                "pfr_bad_throw_rate": bad_throw,
                "pfr_drop_rate": drop,
                "pfr_rec_drop_rate": rec_drop,
                "pfr_rush_ybc": ybc,
                "pfr_rush_yac": yac,
                "pfr_broken_tackle_rate": broken,
                "pfr_def_missed_tackle_rate": missed,
                "pfr_def_pressure_rate": created,
                "pfr_imputed": 0,
            }
            for team, pressure, bad_throw, drop, rec_drop, ybc, yac, broken, missed, created in (
                ("BUF", 0.20, 0.10, 0.04, 0.05, 2.0, 3.0, 0.10, 0.08, 0.30),
                ("KC", 0.25, 0.12, 0.06, 0.07, 1.5, 2.5, 0.08, 0.10, 0.20),
                ("LV", 0.22, 0.11, 0.05, 0.06, 1.8, 2.8, 0.09, 0.09, 0.25),
                ("LAC", 0.24, 0.13, 0.07, 0.08, 1.6, 2.6, 0.07, 0.11, 0.23),
            )
        ]
    )
    return {"C1": ratings, "C2": qb, "C3": style, "C4": personnel, "C5": pfr}


def _build(**kwargs):
    module = _v2_features()
    return module.build_v2_game_features(_base_features(), _blocks(), **kwargs)


def test_margin_feature_blocks_are_pinned_and_candidate_schemas_are_cumulative():
    module = _v2_features()
    assert module.MARGIN_FEATURES_BY_BLOCK == {
        "C0": tuple(FEATURE_COLS),
        "C1": (
            "rating_net_diff_short",
            "rating_net_diff_long",
            "pass_matchup_diff_short",
            "pass_matchup_diff_long",
            "rush_matchup_diff_short",
            "rush_matchup_diff_long",
            "success_diff_short",
            "success_diff_long",
            "early_down_diff_short",
            "neutral_diff_short",
            "explosive_pass_diff",
            "explosive_rush_diff",
            "rest_diff",
            "home_indicator",
            "div_game",
        ),
        "C2": (
            "qb_epa_diff",
            "qb_cpoe_diff",
            "qb_sack_rate_diff",
            "qb_int_rate_diff",
            "qb_change_epa_diff",
            "qb_new_starter_any",
            "qb_rookie_any",
            "qb_uncertain_any",
        ),
        "C3": (
            "neutral_pass_rate_diff",
            "pace_diff",
            "turnover_rate_diff",
            "explosive_play_diff",
            "field_position_diff",
            "special_teams_diff",
            "style_imputed_any",
        ),
        "C4": (
            "off_returning_share_diff",
            "def_returning_share_diff",
            "off_snap_hhi_diff",
            "def_snap_hhi_diff",
            "depth_change_diff",
            "roster_churn_diff",
            "personnel_imputed_any",
        ),
        "C5": (
            "pfr_pressure_edge_diff",
            "pfr_accuracy_diff",
            "pfr_drop_diff",
            "pfr_rush_contact_diff",
            "pfr_tackle_diff",
            "pfr_imputed_any",
        ),
    }
    manifest = _build().manifest
    expected = tuple(
        dict.fromkeys(
            module.MARGIN_FEATURES_BY_BLOCK["C0"]
            + module.MARGIN_FEATURES_BY_BLOCK["C1"]
            + module.MARGIN_FEATURES_BY_BLOCK["C2"]
            + module.MARGIN_FEATURES_BY_BLOCK["C3"]
        )
    )
    assert manifest.columns("margin", "C3") == expected
    for target in ("margin", "total"):
        previous: set[str] = set()
        for candidate in CANDIDATES:
            columns = manifest.columns(target, candidate)
            assert len(columns) == len(set(columns))
            assert previous.issubset(columns)
            previous = set(columns)


def test_target_specific_formulas_use_margin_differences_and_total_combinations():
    module = _v2_features()
    bundle = _build()
    game = bundle.frame.set_index("game_id").loc["g1"]
    assert game["rating_net_diff_short"] == pytest.approx(5.0)
    assert game["rating_matchup_sum_short"] == pytest.approx(11.0)
    assert game["qb_epa_diff"] == pytest.approx(6.0)
    assert game["qb_epa_sum"] == pytest.approx(14.0)
    assert game["neutral_pass_rate_diff"] == pytest.approx(0.20)
    assert game["neutral_pass_rate_mean"] == pytest.approx(0.50)
    assert game["pace_diff"] == pytest.approx(-6.0)
    assert game["pace_mean"] == pytest.approx(27.0)
    assert game["off_returning_share_diff"] == pytest.approx(0.20)
    assert game["off_returning_share_min"] == pytest.approx(0.60)
    assert "rating_net_diff_short" not in module.TOTAL_FEATURES_BY_BLOCK["C1"]
    assert "rating_matchup_sum_short" not in module.MARGIN_FEATURES_BY_BLOCK["C1"]


def test_game_and_team_week_join_cardinality_is_enforced():
    module = _v2_features()
    duplicate_game = pd.concat([_base_features(), _base_features().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="game key"):
        module.merge_v2_blocks(duplicate_game, _blocks())

    missing_key = _base_features().assign(
        game_id=lambda frame: frame["game_id"].mask(frame.index == 0)
    )
    with pytest.raises(ValueError, match="game key"):
        module.merge_v2_blocks(missing_key, _blocks())

    blocks = _blocks()
    blocks["C1"] = pd.concat([blocks["C1"], blocks["C1"].iloc[[0]]], ignore_index=True)
    with pytest.raises(pd.errors.MergeError):
        module.merge_v2_blocks(_base_features(), blocks)


def test_team_aliases_are_normalized_before_many_to_one_joins():
    bundle = _build()
    game = bundle.frame.set_index("game_id").loc["g2"]
    assert game["home_team"] == "LV"
    assert game["away_team"] == "LAC"
    assert game["rating_net_diff_short"] == pytest.approx(0.0)
    assert game["qb_epa_diff"] == pytest.approx(1.0)


def test_manifested_values_are_numeric_finite_and_missing_sides_are_explicitly_imputed():
    module = _v2_features()
    blocks = _blocks()
    blocks["C3"] = blocks["C3"].loc[blocks["C3"]["team"].ne("BUF")].copy()
    bundle = module.build_v2_game_features(_base_features(), blocks)
    game = bundle.frame.set_index("game_id").loc["g1"]
    assert game["style_imputed_any"] == 1
    assert game["neutral_pass_rate_diff"] == pytest.approx(0.15)
    all_columns = tuple(
        dict.fromkeys(
            column
            for target in ("margin", "total")
            for column in bundle.manifest.columns(target, "C5")
        )
    )
    assert all(pd.api.types.is_numeric_dtype(bundle.frame[column]) for column in all_columns)
    assert np.isfinite(bundle.frame[list(all_columns)].to_numpy(dtype=float)).all()

    invalid = _base_features()
    invalid.loc[0, FEATURE_COLS[0]] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        module.build_v2_game_features(invalid, _blocks())


def test_missing_noncentered_inputs_use_frozen_feature_specific_neutral_priors():
    module = _v2_features()
    blocks = _blocks()
    blocks["C3"].loc[
        blocks["C3"]["team"].eq("BUF"), ["pace_seconds", "starting_field_position"]
    ] = np.nan
    blocks["C4"].loc[blocks["C4"]["team"].eq("BUF"), ["off_returning_share", "off_snap_hhi"]] = (
        np.nan
    )
    blocks["C5"].loc[blocks["C5"]["team"].eq("BUF"), ["pfr_rush_ybc", "pfr_rush_yac"]] = np.nan

    bundle = module.build_v2_game_features(_base_features(), blocks)
    game = bundle.frame.set_index("game_id").loc["g1"]

    assert game["pace_diff"] == pytest.approx(-2.0)  # fixed 28.0 prior - KC 30.0
    assert game["pace_mean"] == pytest.approx(29.0)
    assert game["field_position_diff"] == pytest.approx(-2.0)  # fixed own-25 prior
    assert game["field_position_sum"] == pytest.approx(52.0)
    assert game["off_returning_share_diff"] == pytest.approx(0.10)  # fixed .70 prior
    assert game["off_returning_share_min"] == pytest.approx(0.60)
    assert game["off_snap_hhi_diff"] == pytest.approx(-0.20)  # fixed .10 prior
    assert game["off_snap_hhi_sum"] == pytest.approx(0.40)
    assert game["pfr_rush_contact_diff"] == pytest.approx(1.02)
    assert game["pfr_rush_contact_sum"] == pytest.approx(9.18)
    assert game[["style_imputed_any", "personnel_imputed_any", "pfr_imputed_any"]].eq(1).all()

    priors = bundle.manifest.constants["raw_neutral_priors"]
    assert priors["pace_seconds"] == 28.0
    assert priors["starting_field_position"] == 25.0
    assert priors["off_returning_share"] == 0.70
    assert priors["off_snap_hhi"] == 0.10
    assert priors["pfr_rush_ybc"] == 2.5
    assert priors["pfr_rush_yac"] == 2.5


def test_missing_qb_uncertainty_value_sets_the_explicit_uncertainty_flag():
    module = _v2_features()
    blocks = _blocks()
    blocks["C2"].loc[blocks["C2"]["team"].eq("BUF"), "qb_uncertain"] = np.nan

    bundle = module.build_v2_game_features(_base_features(), blocks)

    game = bundle.frame.set_index("game_id").loc["g1"]
    assert game["qb_uncertain_any"] == 1


def test_market_values_remain_metadata_and_never_enter_either_schema():
    bundle = _build()
    assert {"spread_line", "total_line"}.issubset(bundle.frame.columns)
    excluded = MARKET_COLUMNS | MARKET_PROBABILITY_COLUMNS
    for target in ("margin", "total"):
        for candidate in CANDIDATES:
            assert excluded.isdisjoint(bundle.manifest.columns(target, candidate))


def test_manifest_sha256_is_deterministic_over_sorted_inputs():
    first = _build(
        sources={"style": "nflverse-pbp-style@v2", "ratings": "team-ratings@v1"},
        constants={"z": 2, "a": 1},
    ).manifest
    reordered = _build(
        sources={"ratings": "team-ratings@v1", "style": "nflverse-pbp-style@v2"},
        constants={"a": 1, "z": 2},
    ).manifest
    changed = _build(
        sources={"ratings": "team-ratings@v3", "style": "nflverse-pbp-style@v2"},
        constants={"a": 1, "z": 2},
    ).manifest
    assert len(first.version) == 64
    assert set(first.version) <= set("0123456789abcdef")
    assert first.version == reordered.version
    assert first.version != changed.version


def test_post_cutoff_team_week_row_cannot_change_earlier_game():
    module = _v2_features()
    baseline = _build().frame.set_index("game_id").loc["g1"]
    blocks = _blocks()
    future = (
        blocks["C3"]
        .iloc[[0]]
        .assign(
            week=3,
            neutral_pass_rate=999.0,
            pace_seconds=999.0,
            turnover_rate=999.0,
            explosive_play_rate=999.0,
            starting_field_position=999.0,
            special_teams_epa=999.0,
        )
    )
    blocks["C3"] = pd.concat([blocks["C3"], future], ignore_index=True)
    poisoned = (
        module.build_v2_game_features(_base_features(), blocks).frame.set_index("game_id").loc["g1"]
    )
    manifested = tuple(
        dict.fromkeys(
            module.build_v2_game_features(_base_features(), blocks).manifest.columns("margin", "C5")
            + module.build_v2_game_features(_base_features(), blocks).manifest.columns(
                "total", "C5"
            )
        )
    )
    pd.testing.assert_series_equal(baseline[list(manifested)], poisoned[list(manifested)])


def test_c5_schema_is_research_visible_but_explicitly_production_ineligible():
    manifest = _build().manifest
    assert manifest.columns("margin", "C5")
    assert manifest.columns("total", "C5")
    assert manifest.constants["c5_production_eligible"] is False
    assert manifest.constants["pfr_rec_drop_rate_coverage_2025"] == pytest.approx(0.6912)


def test_frozen_raw_neutral_prior_map_cannot_be_overridden():
    with pytest.raises(ValueError, match="neutral-prior map"):
        _build(constants={"raw_neutral_priors": {"pace_seconds": 1.0}})


def test_c5_production_ineligibility_has_no_manifest_bypass():
    with pytest.raises(ValueError, match="production-ineligible"):
        _build(constants={"c5_production_eligible": True})


def test_every_non_c0_output_has_an_exact_hashed_formula_entry():
    module = _v2_features()
    expected = set()
    for blocks in (module.MARGIN_FEATURES_BY_BLOCK, module.TOTAL_FEATURES_BY_BLOCK):
        for candidate in CANDIDATES[1:]:
            expected.update(blocks[candidate])
    assert set(module.FEATURE_FORMULAS) == expected
    assert module.FEATURE_FORMULAS["qb_sack_rate_diff"] == ("home.qb_sack_rate - away.qb_sack_rate")
    assert module.FEATURE_FORMULAS["pfr_pressure_environment_sum"] == (
        "home.pfr_pressure_rate + away.pfr_def_pressure_rate "
        "+ away.pfr_pressure_rate + home.pfr_def_pressure_rate"
    )


def test_default_and_override_sources_are_concrete_versioned_contracts():
    manifest = _build().manifest
    assert manifest.sources["C1"] == "nflverse-pbp-team-ratings@v1|nflreadpy@0.1.5"
    assert all(
        part.count("@") == 1 and all(part.split("@", 1))
        for contract in manifest.sources.values()
        for part in contract.split("|")
    )

    with pytest.raises(ValueError, match="versioned source contract"):
        _build(sources={"C3": "   "})
    with pytest.raises(ValueError, match="versioned source contract"):
        _build(sources={"C3": "nflverse-pbp-style"})


def test_manifest_hash_changes_when_an_exact_formula_changes(monkeypatch):
    module = _v2_features()
    before = _build().manifest.version
    monkeypatch.setitem(
        module.FEATURE_FORMULAS,
        "turnover_rate_diff",
        "away.turnover_rate - home.turnover_rate",
    )
    after = _build().manifest.version
    assert after != before


def test_all_distinct_qb_style_and_pfr_families_pin_signs_and_operators():
    game = _build().frame.set_index("game_id").loc["g1"]

    assert game["qb_sack_rate_diff"] == pytest.approx(0.06)
    assert game["qb_sack_rate_sum"] == pytest.approx(0.14)
    assert game["qb_int_rate_diff"] == pytest.approx(0.03)
    assert game["qb_int_rate_sum"] == pytest.approx(0.07)

    assert game["turnover_rate_diff"] == pytest.approx(-0.02)
    assert game["turnover_rate_sum"] == pytest.approx(0.04)
    assert game["field_position_diff"] == pytest.approx(4.0)
    assert game["field_position_sum"] == pytest.approx(58.0)
    assert game["special_teams_diff"] == pytest.approx(0.30)
    assert game["special_teams_sum"] == pytest.approx(0.10)

    assert game["pfr_pressure_edge_diff"] == pytest.approx(0.15)
    assert game["pfr_pressure_environment_sum"] == pytest.approx(0.95)
    assert game["pfr_accuracy_diff"] == pytest.approx(0.02)
    assert game["pfr_accuracy_sum"] == pytest.approx(-0.22)
    assert game["pfr_drop_diff"] == pytest.approx(0.04)
    assert game["pfr_drop_sum"] == pytest.approx(0.22)
    assert game["pfr_rush_contact_diff"] == pytest.approx(1.02)
    assert game["pfr_rush_contact_sum"] == pytest.approx(9.18)
    assert game["pfr_tackle_diff"] == pytest.approx(0.02)
    assert game["pfr_tackle_environment_sum"] == pytest.approx(0.18)
