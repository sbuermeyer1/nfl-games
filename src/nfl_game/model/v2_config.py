import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Literal, TypeAlias

CandidateId: TypeAlias = Literal["C0", "C1", "C2", "C3", "C4", "C5"]

CANDIDATES = ("C0", "C1", "C2", "C3", "C4", "C5")
ALPHAS = (0.1, 1.0, 10.0, 100.0)
RATING_WINDOWS = ((4, 16), (8, 24), (12, 32))
PRIOR_SEASON_WEIGHTS = (0.4, 0.6, 0.8)
MARKET_COLUMNS = frozenset({"spread_line", "total_line", "away_moneyline", "home_moneyline"})
MARKET_PROBABILITY_COLUMNS = frozenset({"cover_prob", "over_prob"})


def _deep_freeze(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _deep_freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_deep_freeze(item) for item in value)
    return value


def _mutable_json_copy(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _mutable_json_copy(item) for key, item in value.items()}
    if isinstance(value, (tuple, set, frozenset)):
        return [_mutable_json_copy(item) for item in value]
    return value


def rating_setting_key(short_halflife: int, long_halflife: int, prior_season_weight: float) -> str:
    """Stable manifest key for one predefined rating-recency setting."""
    return json.dumps(
        {
            "long_halflife": long_halflife,
            "prior_season_weight": prior_season_weight,
            "short_halflife": short_halflife,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def rating_variant_physical_column(
    canonical: str, short_halflife: int, long_halflife: int, prior_season_weight: float
) -> str:
    """Exact persisted column name for one canonical rating feature and setting."""
    weight = round(prior_season_weight * 10)
    return f"{canonical}__s{short_halflife}_l{long_halflife}_p{weight:02d}"


@dataclass(frozen=True, order=True)
class TargetConfig:
    candidate: str
    alpha: float
    short_halflife: int
    long_halflife: int
    prior_season_weight: float

    def key(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    def rating_key(self) -> str:
        return rating_setting_key(
            self.short_halflife,
            self.long_halflife,
            self.prior_season_weight,
        )


@dataclass(frozen=True)
class V2ModelConfig:
    margin: TargetConfig
    total: TargetConfig


@dataclass(frozen=True)
class FeatureManifest:
    version: str
    margin_by_candidate: Mapping[str, tuple[str, ...]]
    total_by_candidate: Mapping[str, tuple[str, ...]]
    sources: Mapping[str, str]
    constants: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(self, "margin_by_candidate", _deep_freeze(self.margin_by_candidate))
        object.__setattr__(self, "total_by_candidate", _deep_freeze(self.total_by_candidate))
        object.__setattr__(self, "sources", _deep_freeze(self.sources))
        object.__setattr__(self, "constants", _deep_freeze(self.constants))

        for target, mapping in (
            ("margin", self.margin_by_candidate),
            ("total", self.total_by_candidate),
        ):
            for candidate, columns in mapping.items():
                overlap = MARKET_COLUMNS.intersection(columns)
                if overlap:
                    raise ValueError(f"market column in {target}/{candidate}: {sorted(overlap)}")
                probability_overlap = MARKET_PROBABILITY_COLUMNS.intersection(columns)
                if probability_overlap:
                    raise ValueError(
                        "market probability column "
                        f"in {target}/{candidate}: {sorted(probability_overlap)}"
                    )

    def columns(self, target: str, candidate: str) -> tuple[str, ...]:
        mappings = {
            "margin": self.margin_by_candidate,
            "total": self.total_by_candidate,
        }
        return tuple(mappings[target][candidate])

    def _candidate_mapping(self, target: str) -> Mapping[str, tuple[str, ...]]:
        if target == "margin":
            return self.margin_by_candidate
        if target in {"total", "total_points"}:
            return self.total_by_candidate
        raise ValueError(f"unsupported manifest target {target!r}")

    def rating_variant_canonical_columns(self, target: str) -> tuple[str, ...]:
        """The exact cumulative C1-minus-C0 schema for one target."""
        mapping = self._candidate_mapping(target)
        if "C0" not in mapping or "C1" not in mapping:
            raise ValueError(f"non-C0 candidate requires a declared C1 contract for {target}")
        c0 = tuple(mapping["C0"])
        c1 = tuple(mapping["C1"])
        if not set(c0).issubset(c1):
            raise ValueError(f"{target}/C1 must contain the complete C0 schema")
        c0_set = set(c0)
        expected = tuple(column for column in c1 if column not in c0_set)
        if not expected:
            raise ValueError(f"{target}/C1 has no rating-variant canonical columns")
        return expected

    def _raw_rating_variant_mapping(
        self, target: str, config: TargetConfig
    ) -> Mapping[object, object]:
        variants = self.constants.get("rating_variant_columns")
        if not isinstance(variants, Mapping):
            raise TypeError("manifest has no declared rating variant column contract")
        setting = variants.get(config.rating_key())
        if not isinstance(setting, Mapping):
            raise TypeError(
                f"manifest has no declared rating variant for setting {config.rating_key()}"
            )
        target_key = "margin" if target == "margin" else "total"
        mapping = setting.get(target_key)
        if not isinstance(mapping, Mapping):
            raise TypeError(
                f"manifest rating variant {config.rating_key()} has no {target_key} mapping"
            )
        return mapping

    def rating_variant_columns(self, target: str, config: TargetConfig) -> Mapping[str, str]:
        """Return an exact canonical-to-physical mapping for one predefined setting."""
        if config.candidate == "C0":
            return MappingProxyType({})
        expected_canonicals = self.rating_variant_canonical_columns(target)
        mapping = self._raw_rating_variant_mapping(target, config)
        actual_canonicals = set(mapping)
        if actual_canonicals != set(expected_canonicals):
            raise ValueError(
                f"manifest rating variant canonical set must be exactly "
                f"{list(expected_canonicals)} for {target}; got {sorted(actual_canonicals)}"
            )

        normalized: dict[str, str] = {}
        for canonical in expected_canonicals:
            physical = mapping[canonical]
            if not isinstance(physical, str) or not physical:
                raise ValueError("manifest rating variant has an invalid physical column")
            expected_physical = rating_variant_physical_column(
                canonical,
                config.short_halflife,
                config.long_halflife,
                config.prior_season_weight,
            )
            if physical != expected_physical:
                raise ValueError(
                    f"rating variant {canonical!r} must use exact derived physical column "
                    f"{expected_physical!r}; arbitrary, forbidden, reused, or wrong-setting "
                    f"source {physical!r} is not permitted"
                )
            normalized[canonical] = physical
        if len(set(normalized.values())) != len(normalized):
            raise ValueError("manifest rating variant contains physical column reuse")
        return MappingProxyType(normalized)

    def validate_selection_contract(self) -> None:
        """Fail closed on candidate, schema, grid, C5, and rating-variant structure."""
        mappings = {
            "margin": self.margin_by_candidate,
            "total": self.total_by_candidate,
        }
        declared = set().union(*(set(mapping) for mapping in mappings.values()))
        unknown = sorted(declared.difference(CANDIDATES))
        if unknown:
            raise ValueError(f"manifest contains unsupported candidate(s) {unknown}")

        for target, mapping in mappings.items():
            for candidate, columns in mapping.items():
                duplicates = [
                    column for index, column in enumerate(columns) if column in columns[:index]
                ]
                if duplicates:
                    raise ValueError(
                        f"duplicate manifest feature label(s) in {target}/{candidate}: "
                        f"{list(dict.fromkeys(duplicates))}"
                    )

            non_c0 = [candidate for candidate in mapping if candidate != "C0"]
            if not non_c0:
                continue
            expected_c1 = tuple(mapping.get("C1", ()))
            expected_variants = self.rating_variant_canonical_columns(target)
            for candidate in non_c0:
                missing_c1 = sorted(set(expected_c1).difference(mapping[candidate]))
                if missing_c1:
                    raise ValueError(
                        f"{target}/{candidate} must contain the complete C1 contract; "
                        f"missing {missing_c1}"
                    )
            if len(expected_variants) != len(set(expected_variants)):
                raise ValueError(f"duplicate C1 rating-variant canonical in {target}")

        if "C5" in declared:
            flag = self.constants.get("c5_production_eligible")
            if type(flag) is not bool:
                raise TypeError(
                    "c5_production_eligible must be a real bool whenever C5 is declared"
                )

        if any(candidate != "C0" for candidate in declared):
            variants = self.constants.get("rating_variant_columns")
            if not isinstance(variants, Mapping):
                raise TypeError("manifest has no declared rating variant column contract")
            expected_keys = {
                rating_setting_key(short, long, prior)
                for short, long in RATING_WINDOWS
                for prior in PRIOR_SEASON_WEIGHTS
            }
            if set(variants) != expected_keys:
                raise ValueError(
                    "manifest rating variant settings must equal the complete fixed nine-setting "
                    f"grid; missing={sorted(expected_keys.difference(variants))}, "
                    f"extra={sorted(set(variants).difference(expected_keys))}"
                )
            for short, long in RATING_WINDOWS:
                for prior in PRIOR_SEASON_WEIGHTS:
                    for target in mappings:
                        config = TargetConfig("C1", 1.0, short, long, prior)
                        self.rating_variant_columns(target, config)

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "margin_by_candidate": {
                key: list(value) for key, value in self.margin_by_candidate.items()
            },
            "total_by_candidate": {
                key: list(value) for key, value in self.total_by_candidate.items()
            },
            "sources": _mutable_json_copy(self.sources),
            "constants": _mutable_json_copy(self.constants),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "FeatureManifest":
        return cls(
            version=str(payload["version"]),
            margin_by_candidate={
                key: tuple(value) for key, value in payload["margin_by_candidate"].items()
            },
            total_by_candidate={
                key: tuple(value) for key, value in payload["total_by_candidate"].items()
            },
            sources=dict(payload["sources"]),
            constants=dict(payload["constants"]),
        )


def target_tuning_grid(candidate: CandidateId) -> list[TargetConfig]:
    return [
        TargetConfig(candidate, alpha, short_halflife, long_halflife, prior_season_weight)
        for alpha in ALPHAS
        for short_halflife, long_halflife in RATING_WINDOWS
        for prior_season_weight in PRIOR_SEASON_WEIGHTS
    ]
