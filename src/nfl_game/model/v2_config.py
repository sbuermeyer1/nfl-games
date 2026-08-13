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

    def rating_variant_columns(self, target: str, config: TargetConfig) -> Mapping[str, str]:
        """Return the frozen canonical-to-physical columns for one rating setting."""
        if config.candidate == "C0":
            return MappingProxyType({})
        target_key = "margin" if target == "margin" else "total"
        if target not in {"margin", "total", "total_points"}:
            raise ValueError(f"unsupported rating-variant target {target!r}")
        variants = self.constants.get("rating_variant_columns")
        if not isinstance(variants, Mapping):
            raise TypeError("manifest has no declared rating variant column contract")
        setting = variants.get(config.rating_key())
        if not isinstance(setting, Mapping):
            raise TypeError(
                f"manifest has no declared rating variant for setting {config.rating_key()}"
            )
        mapping = setting.get(target_key)
        if not isinstance(mapping, Mapping) or not mapping:
            raise ValueError(
                f"manifest rating variant {config.rating_key()} has no {target_key} mapping"
            )
        normalized: dict[str, str] = {}
        for canonical, physical in mapping.items():
            if not isinstance(canonical, str) or not canonical:
                raise ValueError("manifest rating variant has an invalid canonical column")
            if not isinstance(physical, str) or not physical:
                raise ValueError("manifest rating variant has an invalid physical column")
            normalized[canonical] = physical
        if len(set(normalized.values())) != len(normalized):
            raise ValueError(
                "manifest rating variant maps multiple features to one physical column"
            )
        return MappingProxyType(normalized)

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
