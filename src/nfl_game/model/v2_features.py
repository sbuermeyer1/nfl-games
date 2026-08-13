"""Target-specific Ridge-v2 game-feature assembly and frozen manifests."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from nfl_game.data.teams import normalize_team_codes
from nfl_game.model.features import FEATURE_COLS
from nfl_game.model.v2_config import CANDIDATES, FeatureManifest

GAME_KEYS = ("game_id", "season", "week")
TEAM_WEEK_KEYS = ("season", "week", "team")

MARGIN_FEATURES_BY_BLOCK = {
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

TOTAL_FEATURES_BY_BLOCK = {
    "C0": tuple(FEATURE_COLS),
    "C1": (
        "rating_matchup_sum_short",
        "rating_matchup_sum_long",
        "pass_matchup_sum_short",
        "pass_matchup_sum_long",
        "rush_matchup_sum_short",
        "rush_matchup_sum_long",
        "success_matchup_sum_short",
        "success_matchup_sum_long",
        "early_down_matchup_sum_short",
        "neutral_matchup_sum_short",
        "explosive_pass_matchup_sum",
        "explosive_rush_matchup_sum",
        "is_dome",
        "temp_outdoor",
        "wind_outdoor",
    ),
    "C2": (
        "qb_epa_sum",
        "qb_cpoe_sum",
        "qb_sack_rate_sum",
        "qb_int_rate_sum",
        "qb_change_epa_sum",
        "qb_new_starter_any",
        "qb_rookie_any",
        "qb_uncertain_any",
    ),
    "C3": (
        "neutral_pass_rate_mean",
        "pace_mean",
        "turnover_rate_sum",
        "explosive_play_sum",
        "field_position_sum",
        "special_teams_sum",
        "style_imputed_any",
    ),
    "C4": (
        "off_returning_share_min",
        "def_returning_share_min",
        "off_snap_hhi_sum",
        "def_snap_hhi_sum",
        "depth_change_sum",
        "roster_churn_sum",
        "personnel_imputed_any",
    ),
    "C5": (
        "pfr_pressure_environment_sum",
        "pfr_accuracy_sum",
        "pfr_drop_sum",
        "pfr_rush_contact_sum",
        "pfr_tackle_environment_sum",
        "pfr_imputed_any",
    ),
}

DEFAULT_SOURCES = {
    "C0": "ridge-v1-feature-cols",
    "C1": "nflverse-play-by-play-team-ratings",
    "C2": "nflverse-player-stats-and-depth-charts",
    "C3": "nflverse-play-by-play-style",
    "C4": "nflverse-rosters-depth-charts-and-snap-counts",
    "C5": "nflverse-pfr-advanced-weekly",
}
DEFAULT_CONSTANTS: dict[str, object] = {
    "block_neutral_fill": 0.0,
    "c5_production_eligible": False,
    "pfr_rec_drop_rate_coverage_2025": 0.6912,
}

FEATURE_FORMULAS = {
    **{f"C0.{column}": "unchanged Ridge-v1 feature" for column in FEATURE_COLS},
    "rating_net_diff": "(home_off + home_def) - (away_off + away_def)",
    "rating_matchup_diff": "(home_off-away_def) - (away_off-home_def)",
    "rating_matchup_sum": "(home_off-away_def) + (away_off-home_def)",
    "difference": "home - away",
    "sum": "home + away",
    "mean": "(home + away) / 2",
    "minimum": "min(home, away)",
    "any": "max(home, away, missing-side-indicator)",
    "pfr_pressure_edge_diff": (
        "(away_pressure+home_def_pressure) - (home_pressure+away_def_pressure)"
    ),
    "pfr_accuracy_diff": "away_bad_throw_rate - home_bad_throw_rate",
    "pfr_drop_diff": "(away_drop+away_rec_drop) - (home_drop+home_rec_drop)",
    "pfr_rush_contact_diff": "home_ybc+home_yac+home_broken - away equivalents",
    "pfr_tackle_diff": "away_def_missed_tackle_rate - home_def_missed_tackle_rate",
    "pfr_pressure_environment_sum": (
        "home_pressure+away_pressure+home_def_pressure+away_def_pressure"
    ),
    "pfr_accuracy_sum": "-(home_bad_throw_rate + away_bad_throw_rate)",
    "pfr_drop_sum": "home_drop+home_rec_drop+away_drop+away_rec_drop",
    "pfr_rush_contact_sum": "home_ybc+home_yac+home_broken + away equivalents",
    "pfr_tackle_environment_sum": ("home_def_missed_tackle_rate + away_def_missed_tackle_rate"),
}


@dataclass(frozen=True)
class V2FeatureBundle:
    frame: pd.DataFrame
    manifest: FeatureManifest


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], *, label: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{label} missing columns: {missing}")


def _validate_game_keys(frame: pd.DataFrame) -> None:
    _require_columns(frame, (*GAME_KEYS, "home_team", "away_team"), label="base features")
    if frame[list(GAME_KEYS)].isna().any(axis=None):
        raise ValueError("base features contain a missing game key")
    if frame[["home_team", "away_team"]].isna().any(axis=None):
        raise ValueError("base features contain missing team identity")
    if frame.duplicated(list(GAME_KEYS)).any() or frame["game_id"].duplicated().any():
        raise ValueError("base features contain a duplicate game key")


def _merge_team_sides(
    games: pd.DataFrame, block: pd.DataFrame, columns: Sequence[str], *, name: str
) -> pd.DataFrame:
    _require_columns(block, (*TEAM_WEEK_KEYS, *columns), label=f"{name} block")
    normalized = normalize_team_codes(block, ["team"])
    sided = games[[*GAME_KEYS, "home_team", "away_team"]].copy()
    for side in ("home", "away"):
        team_column = f"{side}_team"
        renamed = normalized[[*TEAM_WEEK_KEYS, *columns]].rename(
            columns={
                "team": team_column,
                **{column: f"{side}_{column}" for column in columns},
            }
        )
        sided = sided.merge(
            renamed,
            on=["season", "week", team_column],
            how="left",
            validate="many_to_one",
        )
    return sided


def _fill_numeric_sides(
    sided: pd.DataFrame,
    columns: Sequence[str],
    *,
    source_flag: str | None = None,
) -> pd.Series:
    value_columns = [f"{side}_{column}" for side in ("home", "away") for column in columns]
    numeric = sided[value_columns].apply(pd.to_numeric, errors="coerce")
    invalid = numeric.isna().any(axis=1)
    non_finite = ~np.isfinite(numeric.fillna(0.0).to_numpy(dtype=float)).all(axis=1)
    if non_finite.any():
        raise ValueError("non-finite value in team-week feature block")
    sided[value_columns] = numeric.fillna(float(DEFAULT_CONSTANTS["block_neutral_fill"]))
    if source_flag is None:
        return invalid.astype(int)
    flag_columns = [f"home_{source_flag}", f"away_{source_flag}"]
    flags = sided[flag_columns].apply(pd.to_numeric, errors="coerce")
    invalid |= flags.isna().any(axis=1)
    sided[flag_columns] = flags.fillna(1.0)
    return pd.concat([sided[flag_columns], invalid.rename("missing")], axis=1).max(axis=1)


def _difference(sided: pd.DataFrame, column: str) -> pd.Series:
    return sided[f"home_{column}"] - sided[f"away_{column}"]


def _sum(sided: pd.DataFrame, column: str) -> pd.Series:
    return sided[f"home_{column}"] + sided[f"away_{column}"]


def _mean(sided: pd.DataFrame, column: str) -> pd.Series:
    return _sum(sided, column) / 2.0


def _minimum(sided: pd.DataFrame, column: str) -> pd.Series:
    return sided[[f"home_{column}", f"away_{column}"]].min(axis=1)


def _rating_features(games: pd.DataFrame, block: pd.DataFrame) -> pd.DataFrame:
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
    columns = tuple(
        f"{window}_{unit}_{target}"
        for window in ("short", "long")
        for unit in ("off", "def")
        for target in targets
    )
    sided = _merge_team_sides(games, block, columns, name="C1")
    imputed = _fill_numeric_sides(sided, columns)
    out = sided[list(GAME_KEYS)].copy()

    def matchup(target: str, window: str, operation: str) -> pd.Series:
        home_edge = sided[f"home_{window}_off_{target}"] - sided[f"away_{window}_def_{target}"]
        away_edge = sided[f"away_{window}_off_{target}"] - sided[f"home_{window}_def_{target}"]
        return home_edge - away_edge if operation == "diff" else home_edge + away_edge

    for window in ("short", "long"):
        out[f"rating_net_diff_{window}"] = (
            sided[f"home_{window}_off_epa_play"]
            + sided[f"home_{window}_def_epa_play"]
            - sided[f"away_{window}_off_epa_play"]
            - sided[f"away_{window}_def_epa_play"]
        )
        out[f"rating_matchup_sum_{window}"] = matchup("epa_play", window, "sum")
        for prefix, target in (("pass", "epa_pass"), ("rush", "epa_rush")):
            out[f"{prefix}_matchup_diff_{window}"] = matchup(target, window, "diff")
            out[f"{prefix}_matchup_sum_{window}"] = matchup(target, window, "sum")
        out[f"success_diff_{window}"] = matchup("success_rate", window, "diff")
        out[f"success_matchup_sum_{window}"] = matchup("success_rate", window, "sum")
    for prefix, target in (("early_down", "early_down_epa"), ("neutral", "neutral_epa")):
        out[f"{prefix}_diff_short"] = matchup(target, "short", "diff")
        out[f"{prefix}_matchup_sum_short"] = matchup(target, "short", "sum")
    for prefix, target in (
        ("explosive_pass", "explosive_pass_rate"),
        ("explosive_rush", "explosive_rush_rate"),
    ):
        out[f"{prefix}_diff"] = matchup(target, "short", "diff")
        out[f"{prefix}_matchup_sum"] = matchup(target, "short", "sum")
    out["home_indicator"] = 1.0
    out["rating_imputed_any"] = imputed
    return out


def _qb_features(games: pd.DataFrame, block: pd.DataFrame) -> pd.DataFrame:
    values = ("qb_epa_per_db", "qb_cpoe", "qb_sack_rate", "qb_int_rate", "qb_change_epa")
    flags = ("qb_new_starter", "qb_rookie", "qb_uncertain")
    sided = _merge_team_sides(games, block, (*values, *flags), name="C2")
    missing = _fill_numeric_sides(sided, values)
    flag_missing = _fill_numeric_sides(sided, flags)
    missing = pd.concat([missing, flag_missing], axis=1).max(axis=1)
    out = sided[list(GAME_KEYS)].copy()
    for source, feature in (
        ("qb_epa_per_db", "qb_epa"),
        ("qb_cpoe", "qb_cpoe"),
        ("qb_sack_rate", "qb_sack_rate"),
        ("qb_int_rate", "qb_int_rate"),
        ("qb_change_epa", "qb_change_epa"),
    ):
        out[f"{feature}_diff"] = _difference(sided, source)
        out[f"{feature}_sum"] = _sum(sided, source)
    for flag in flags:
        out[f"{flag}_any"] = sided[[f"home_{flag}", f"away_{flag}"]].max(axis=1)
    out["qb_uncertain_any"] = pd.concat(
        [out["qb_uncertain_any"], missing.rename("missing")], axis=1
    ).max(axis=1)
    return out


def _style_features(games: pd.DataFrame, block: pd.DataFrame) -> pd.DataFrame:
    values = (
        "neutral_pass_rate",
        "pace_seconds",
        "turnover_rate",
        "explosive_play_rate",
        "starting_field_position",
        "special_teams_epa",
    )
    sided = _merge_team_sides(games, block, (*values, "style_imputed"), name="C3")
    imputed = _fill_numeric_sides(sided, values, source_flag="style_imputed")
    out = sided[list(GAME_KEYS)].copy()
    for source, feature in (
        ("neutral_pass_rate", "neutral_pass_rate"),
        ("pace_seconds", "pace"),
        ("turnover_rate", "turnover_rate"),
        ("explosive_play_rate", "explosive_play"),
        ("starting_field_position", "field_position"),
        ("special_teams_epa", "special_teams"),
    ):
        out[f"{feature}_diff"] = _difference(sided, source)
    out["neutral_pass_rate_mean"] = _mean(sided, "neutral_pass_rate")
    out["pace_mean"] = _mean(sided, "pace_seconds")
    for source, feature in (
        ("turnover_rate", "turnover_rate"),
        ("explosive_play_rate", "explosive_play"),
        ("starting_field_position", "field_position"),
        ("special_teams_epa", "special_teams"),
    ):
        out[f"{feature}_sum"] = _sum(sided, source)
    out["style_imputed_any"] = imputed
    return out


def _personnel_features(games: pd.DataFrame, block: pd.DataFrame) -> pd.DataFrame:
    values = (
        "off_returning_share",
        "def_returning_share",
        "off_snap_hhi",
        "def_snap_hhi",
        "depth_chart_change_rate",
        "roster_churn",
    )
    sided = _merge_team_sides(games, block, (*values, "personnel_imputed"), name="C4")
    imputed = _fill_numeric_sides(sided, values, source_flag="personnel_imputed")
    out = sided[list(GAME_KEYS)].copy()
    for source, feature in (
        ("off_returning_share", "off_returning_share"),
        ("def_returning_share", "def_returning_share"),
        ("off_snap_hhi", "off_snap_hhi"),
        ("def_snap_hhi", "def_snap_hhi"),
        ("depth_chart_change_rate", "depth_change"),
        ("roster_churn", "roster_churn"),
    ):
        out[f"{feature}_diff"] = _difference(sided, source)
    for source, feature in (
        ("off_snap_hhi", "off_snap_hhi"),
        ("def_snap_hhi", "def_snap_hhi"),
        ("depth_chart_change_rate", "depth_change"),
        ("roster_churn", "roster_churn"),
    ):
        out[f"{feature}_sum"] = _sum(sided, source)
    out["off_returning_share_min"] = _minimum(sided, "off_returning_share")
    out["def_returning_share_min"] = _minimum(sided, "def_returning_share")
    out["personnel_imputed_any"] = imputed
    return out


def _pfr_features(games: pd.DataFrame, block: pd.DataFrame) -> pd.DataFrame:
    values = (
        "pfr_pressure_rate",
        "pfr_bad_throw_rate",
        "pfr_drop_rate",
        "pfr_rec_drop_rate",
        "pfr_rush_ybc",
        "pfr_rush_yac",
        "pfr_broken_tackle_rate",
        "pfr_def_missed_tackle_rate",
        "pfr_def_pressure_rate",
    )
    sided = _merge_team_sides(games, block, (*values, "pfr_imputed"), name="C5")
    imputed = _fill_numeric_sides(sided, values, source_flag="pfr_imputed")
    out = sided[list(GAME_KEYS)].copy()
    home_pressure = sided["home_pfr_pressure_rate"] + sided["away_pfr_def_pressure_rate"]
    away_pressure = sided["away_pfr_pressure_rate"] + sided["home_pfr_def_pressure_rate"]
    out["pfr_pressure_edge_diff"] = away_pressure - home_pressure
    out["pfr_accuracy_diff"] = sided["away_pfr_bad_throw_rate"] - sided["home_pfr_bad_throw_rate"]
    home_drops = sided["home_pfr_drop_rate"] + sided["home_pfr_rec_drop_rate"]
    away_drops = sided["away_pfr_drop_rate"] + sided["away_pfr_rec_drop_rate"]
    out["pfr_drop_diff"] = away_drops - home_drops
    home_contact = (
        sided["home_pfr_rush_ybc"]
        + sided["home_pfr_rush_yac"]
        + sided["home_pfr_broken_tackle_rate"]
    )
    away_contact = (
        sided["away_pfr_rush_ybc"]
        + sided["away_pfr_rush_yac"]
        + sided["away_pfr_broken_tackle_rate"]
    )
    out["pfr_rush_contact_diff"] = home_contact - away_contact
    out["pfr_tackle_diff"] = (
        sided["away_pfr_def_missed_tackle_rate"] - sided["home_pfr_def_missed_tackle_rate"]
    )
    out["pfr_pressure_environment_sum"] = home_pressure + away_pressure
    out["pfr_accuracy_sum"] = -(sided["home_pfr_bad_throw_rate"] + sided["away_pfr_bad_throw_rate"])
    out["pfr_drop_sum"] = home_drops + away_drops
    out["pfr_rush_contact_sum"] = home_contact + away_contact
    out["pfr_tackle_environment_sum"] = (
        sided["home_pfr_def_missed_tackle_rate"] + sided["away_pfr_def_missed_tackle_rate"]
    )
    out["pfr_imputed_any"] = imputed
    return out


_BLOCK_BUILDERS: dict[str, Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame]] = {
    "C1": _rating_features,
    "C2": _qb_features,
    "C3": _style_features,
    "C4": _personnel_features,
    "C5": _pfr_features,
}


def team_block_to_game_features(
    games: pd.DataFrame, block: pd.DataFrame, *, name: str
) -> pd.DataFrame:
    """Convert one normalized team-week block to one row per scheduled game."""
    try:
        builder = _BLOCK_BUILDERS[name]
    except KeyError as exc:
        raise ValueError(f"unknown Ridge-v2 feature block: {name!r}") from exc
    return builder(games, block)


def merge_v2_blocks(
    base_features: pd.DataFrame, blocks: Mapping[str, pd.DataFrame]
) -> pd.DataFrame:
    """Merge target-week blocks onto unique games with cardinality validation."""
    _validate_game_keys(base_features)
    unknown = sorted(set(blocks).difference(_BLOCK_BUILDERS))
    if unknown:
        raise ValueError(f"unknown Ridge-v2 feature blocks: {unknown}")
    frame = normalize_team_codes(base_features, ["home_team", "away_team"])
    for name in CANDIDATES[1:]:
        if name not in blocks:
            continue
        sided = team_block_to_game_features(frame, blocks[name], name=name)
        frame = frame.merge(
            sided,
            on=list(GAME_KEYS),
            how="left",
            validate="one_to_one",
        )
    return frame


def _cumulative_columns(blocks: Mapping[str, tuple[str, ...]]) -> dict[str, tuple[str, ...]]:
    cumulative: list[str] = []
    result = {}
    for candidate in CANDIDATES:
        cumulative = list(dict.fromkeys([*cumulative, *blocks[candidate]]))
        result[candidate] = tuple(cumulative)
    return result


def _manifest(
    *, sources: Mapping[str, str] | None, constants: Mapping[str, object] | None
) -> FeatureManifest:
    source_versions = {**DEFAULT_SOURCES, **dict(sources or {})}
    frozen_constants = {**DEFAULT_CONSTANTS, **dict(constants or {})}
    if frozen_constants["block_neutral_fill"] != 0.0:
        raise ValueError("the frozen block-neutral fill is 0.0")
    if frozen_constants["c5_production_eligible"] is not False:
        raise ValueError("C5 is production-ineligible at 69.12% receiving drop-rate coverage")
    if frozen_constants["pfr_rec_drop_rate_coverage_2025"] != 0.6912:
        raise ValueError("the frozen 2025 PFR receiving drop-rate coverage is 0.6912")
    payload = {
        "formulas": FEATURE_FORMULAS,
        "sources": source_versions,
        "constants": frozen_constants,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    version = hashlib.sha256(encoded).hexdigest()
    return FeatureManifest(
        version=version,
        margin_by_candidate=_cumulative_columns(MARGIN_FEATURES_BY_BLOCK),
        total_by_candidate=_cumulative_columns(TOTAL_FEATURES_BY_BLOCK),
        sources=source_versions,
        constants=frozen_constants,
    )


def build_v2_game_features(
    base_features: pd.DataFrame,
    blocks: Mapping[str, pd.DataFrame],
    *,
    sources: Mapping[str, str] | None = None,
    constants: Mapping[str, object] | None = None,
) -> V2FeatureBundle:
    """Build the union game artifact and its deterministic target-specific manifest."""
    missing_blocks = sorted(set(CANDIDATES[1:]).difference(blocks))
    if missing_blocks:
        raise ValueError(f"missing Ridge-v2 feature blocks: {missing_blocks}")
    frame = merge_v2_blocks(base_features, blocks)
    manifest = _manifest(sources=sources, constants=constants)
    manifested = tuple(
        dict.fromkeys(manifest.columns("margin", "C5") + manifest.columns("total", "C5"))
    )
    _require_columns(frame, manifested, label="assembled Ridge-v2 features")
    numeric = frame[list(manifested)].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any(axis=None) or not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("assembled Ridge-v2 features contain missing or non-finite values")
    frame[list(manifested)] = numeric
    _validate_game_keys(frame)
    return V2FeatureBundle(frame=frame.reset_index(drop=True), manifest=manifest)
