"""Pre-registered test: does the totals edge survive out of sample in 2026?

Written and committed 2026-08-31, when `schedule_2026.parquet` held **0 of 272** games with a
result and the tracker had published **0 live records**. Nothing in this file was chosen with
knowledge of a 2026 outcome, because no 2026 outcome existed.

## Why this exists

The historical signal below was found by EXPLORATORY analysis. The 5-point threshold was chosen
after looking at disjoint edge buckets and noticing the top one was much stronger than the rest.
That is a fitted parameter, and the repository has already been burned once by a lead-time result
that looked decisive and turned out to be an artefact of how the data was cut. A number found
that way is a hypothesis, not a finding, and the only thing that settles it is a test specified
before the data exists.

The 2026 season is that test, and it is already running: the live tracker publishes each game at
a true 5-day lead, which is exactly the lead the historical estimate was measured at.

## The hypothesis

At a true 5-day publication lead, the model's TOTALS predictions carry closing-line value that
INCREASES with the size of the model's disagreement, and is large where that disagreement is at
least 5.0 points. Spreads do not show this: their CLV is flat at ~+0.14 across every edge bucket.

## What would falsify it

A 2026 mean CLV at or below zero on qualifying totals, or one indistinguishable from zero.
Regression to the mean is expected, so the replication bar is deliberately set at HALF the
historical point estimate rather than at the estimate itself.

## What this test cannot do

It cannot establish profitability. At ~49 qualifying games a season the O/U hit rate has a 95%
interval roughly +/-0.13 wide, so a single season cannot separate 0.52 from 0.55. The hit rate is
recorded as a secondary endpoint and is NOT a success criterion; treating it as one would be the
same mistake as judging the model by its ATS record, which this repository already documents as
requiring ~27 seasons.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

# --- The registration. Changing any value here invalidates the test. ---------------------------

SEASON = 2026
MARKET = "total"
LEAD_DAYS = 5
MIN_ABS_EDGE = 5.0

#: Primary endpoint. CLV is continuous, so it has power a win rate does not at this sample size.
PRIMARY_ENDPOINT = "mean_clv"
#: Replication is claimed only if BOTH hold. The floor is half the historical estimate.
PRIMARY_MIN_MEAN_CLV = 0.29
PRIMARY_MIN_Z = 2.0

#: Secondary endpoints: recorded and reported, never decisive.
SECONDARY_ENDPOINTS = ("ou_hit_rate", "clv_by_edge_bucket", "clv_by_direction")
BREAK_EVEN_HIT_RATE = 0.5238

#: Disjoint buckets, so the shape of the curve can be checked rather than just the chosen cut.
#: If the 5+ threshold was real, the curve should rise across these in 2026 as it did historically.
EDGE_BUCKETS = ((0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 5.0), (5.0, float("inf")))

#: What was measured historically, on 2021-2025 at a true 5-day lead. Recorded so the comparison
#: is fixed and cannot be re-derived more favourably later.
HISTORICAL = {
    "n": 245,
    "mean_clv": 0.5796,
    "z": 6.37,
    "ou_hit_rate": 0.5496,
    "seasons_positive_clv": 5,
    "per_season_clv": {"2021": 0.8214, "2022": 0.3833, "2023": 0.5867, "2024": 0.7439,
                       "2025": 0.3704},
    "over_picks": 197,
    "under_picks": 48,
    "spread_comparison_clv_flat_at": 0.14,
}

#: Written when 0 of 272 games in schedule_2026.parquet had a result and the ledger held 0 live
#: records. Both are checkable after the fact from git history.
REGISTERED_AT = "2026-08-31"
REGISTERED_STATE = {"games_2026_with_result": 0, "live_records_at_registration": 0}


def registration() -> dict[str, Any]:
    """The frozen registration as data, in a stable order for digesting."""
    return {
        "season": SEASON,
        "market": MARKET,
        "lead_days": LEAD_DAYS,
        "min_abs_edge": MIN_ABS_EDGE,
        "primary_endpoint": PRIMARY_ENDPOINT,
        "primary_min_mean_clv": PRIMARY_MIN_MEAN_CLV,
        "primary_min_z": PRIMARY_MIN_Z,
        "secondary_endpoints": list(SECONDARY_ENDPOINTS),
        "break_even_hit_rate": BREAK_EVEN_HIT_RATE,
        "edge_buckets": [[lo, None if hi == float("inf") else hi] for lo, hi in EDGE_BUCKETS],
        "historical": HISTORICAL,
        "registered_at": REGISTERED_AT,
        "registered_state": REGISTERED_STATE,
    }


def registration_digest() -> str:
    """Digest of the registration, pinned by a test so a silent edit cannot pass review."""
    payload = json.dumps(registration(), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()
