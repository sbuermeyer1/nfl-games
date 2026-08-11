import json
from dataclasses import asdict, dataclass
from typing import Literal, TypeAlias

CandidateId: TypeAlias = Literal["C0", "C1", "C2", "C3", "C4", "C5"]

CANDIDATES = ("C0", "C1", "C2", "C3", "C4", "C5")
ALPHAS = (0.1, 1.0, 10.0, 100.0)
RATING_WINDOWS = ((4, 16), (8, 24), (12, 32))
PRIOR_SEASON_WEIGHTS = (0.4, 0.6, 0.8)
MARKET_COLUMNS = frozenset({"spread_line", "total_line", "away_moneyline", "home_moneyline"})


@dataclass(frozen=True, order=True)
class TargetConfig:
    candidate: str
    alpha: float
    short_halflife: int
    long_halflife: int
    prior_season_weight: float

    def key(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class V2ModelConfig:
    margin: TargetConfig
    total: TargetConfig


@dataclass(frozen=True)
class FeatureManifest:
    version: str
    margin_by_candidate: dict[str, tuple[str, ...]]
    total_by_candidate: dict[str, tuple[str, ...]]
    sources: dict[str, str]
    constants: dict[str, object]

    def __post_init__(self) -> None:
        for target, mapping in (
            ("margin", self.margin_by_candidate),
            ("total", self.total_by_candidate),
        ):
            for candidate, columns in mapping.items():
                overlap = MARKET_COLUMNS.intersection(columns)
                if overlap:
                    raise ValueError(
                        f"market column in {target}/{candidate}: {sorted(overlap)}"
                    )

    def columns(self, target: str, candidate: str) -> tuple[str, ...]:
        mappings = {
            "margin": self.margin_by_candidate,
            "total": self.total_by_candidate,
        }
        return tuple(mappings[target][candidate])

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "margin_by_candidate": {
                key: list(value) for key, value in self.margin_by_candidate.items()
            },
            "total_by_candidate": {
                key: list(value) for key, value in self.total_by_candidate.items()
            },
            "sources": dict(self.sources),
            "constants": dict(self.constants),
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
