"""Evaluate the pre-registered 2026 totals test against the live tracker ledger.

The registration is `nfl_game.experiments.prereg_totals_2026`; read its docstring first.

This script deliberately WITHHOLDS the primary endpoint until the 2026 regular season has
finished AND every qualifying game has settled. A pre-registered test that lets you watch its
primary endpoint accumulate is an optional-stopping test wearing a disguise, and the difference
has to be enforced in code -- operator discipline has already failed once in this repository,
when a sizing probe wrote its decisive deltas to disk before N was fixed and got through three
reviews.

AMENDED 2026-09-14, after the gate was found not to do what this docstring already claimed.
Completion was `pending == 0` alone, which means "every qualifying game published SO FAR has
settled" -- true on any quiet day between weeks. Run in week 1 it reported `complete: true`,
`n: 1`, `verdict: "not replicated"`, and it would have emitted a fresh verdict every week of
the season. The registration itself is untouched (its digest still pins the 2026-08-31 payload);
this is machinery, and the new condition is outcome-independent and strictly more conservative
than the one it replaces.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from nfl_game.experiments import prereg_totals_2026 as prereg
from nfl_game.paths import PROCESSED_DIR

#: The 2026 regular season, as recorded in the registration's docstring on the day it was
#: written ("0 of 272"). Checked as a count, not just for nulls: a truncated schedule artefact
#: has no unfinished games in it either, and would read as a finished season.
EXPECTED_REG_GAMES = 272


def qualifying(ledger: pd.DataFrame) -> pd.DataFrame:
    """Live, published, in-season totals whose edge magnitude meets the registered threshold."""
    if ledger.empty:
        return ledger
    edge = pd.to_numeric(ledger.get("total_edge"), errors="coerce")
    return ledger.loc[
        ledger["record_type"].eq("live")
        & ledger["season"].eq(prereg.SEASON)
        & ledger["total_publication_status"].eq("published")
        & edge.abs().ge(prereg.MIN_ABS_EDGE)
    ]


def season_progress(schedule: pd.DataFrame) -> tuple[bool, int]:
    """Whether the registered season is over, and how many of its games are still unplayed."""
    if schedule.empty:
        return False, 0
    rows = schedule.loc[
        schedule["season"].eq(prereg.SEASON) & schedule["game_type"].astype(str).eq("REG")
    ]
    without_result = int(pd.to_numeric(rows.get("result"), errors="coerce").isna().sum())
    complete = len(rows) == EXPECTED_REG_GAMES and without_result == 0
    return complete, without_result


def _bucket_curve(rows: pd.DataFrame) -> list[dict]:
    edge = pd.to_numeric(rows["total_edge"], errors="coerce").abs()
    out = []
    for lo, hi in prereg.EDGE_BUCKETS:
        sub = rows.loc[edge.ge(lo) & edge.lt(hi)]
        clv = pd.to_numeric(sub["total_clv"], errors="coerce").dropna()
        out.append({
            "bucket": f"{lo}+" if hi == float("inf") else f"{lo}-{hi}",
            "n": len(clv),
            "mean_clv": float(clv.mean()) if len(clv) else None,
        })
    return out


def evaluate(ledger: pd.DataFrame, schedule: pd.DataFrame) -> dict:
    """Report progress while the season runs; the primary endpoint only once it is done."""
    rows = qualifying(ledger)
    finished, without_result = season_progress(schedule)
    clv = pd.to_numeric(rows["total_clv"], errors="coerce") if len(rows) else pd.Series(dtype=float)
    settled = int(clv.notna().sum())
    pending = int(len(rows) - settled)
    report = {
        "registration_digest": prereg.registration_digest(),
        "season": prereg.SEASON,
        "qualifying": len(rows),
        "settled": settled,
        "pending": pending,
        "season_complete": finished,
        "games_without_result": without_result,
        "complete": bool(finished and len(rows) > 0 and pending == 0),
    }
    if not report["complete"]:
        report["withheld"] = (
            "primary endpoint withheld until the regular season has finished and every "
            "qualifying game has settled; see the module docstring"
        )
        return report

    values = clv.dropna().to_numpy(dtype=float)
    mean = float(values.mean())
    se = float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else float("nan")
    z = mean / se if se and np.isfinite(se) and se > 0 else float("nan")
    graded = rows["total_close_grade"].isin(["win", "loss"])
    wins = int(rows.loc[graded, "total_close_grade"].eq("win").sum())
    report.update({
        "n": len(values),
        "mean_clv": mean,
        "se": se,
        "z": z,
        "verdict": (
            "replicated"
            if mean >= prereg.PRIMARY_MIN_MEAN_CLV and np.isfinite(z) and z >= prereg.PRIMARY_MIN_Z
            else "not replicated"
        ),
        "historical_mean_clv": prereg.HISTORICAL["mean_clv"],
        "ou_hit_rate": wins / int(graded.sum()) if int(graded.sum()) else None,
        "ou_break_even": prereg.BREAK_EVEN_HIT_RATE,
        "clv_by_edge_bucket": _bucket_curve(rows),
    })
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=PROCESSED_DIR / "tracker_ledger.parquet")
    parser.add_argument("--schedule", type=Path, default=PROCESSED_DIR / "schedule_2026.parquet")
    args = parser.parse_args(argv)
    report = evaluate(pd.read_parquet(args.ledger), pd.read_parquet(args.schedule))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
