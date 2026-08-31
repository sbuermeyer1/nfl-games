"""Evaluate the pre-registered 2026 totals test against the live tracker ledger.

The registration is `nfl_game.experiments.prereg_totals_2026`; read its docstring first.

This script deliberately WITHHOLDS the primary endpoint until every qualifying game has settled.
A pre-registered test that lets you watch its primary endpoint accumulate is an optional-stopping
test wearing a disguise, and the difference has to be enforced in code -- operator discipline has
already failed once in this repository, when a sizing probe wrote its decisive deltas to disk
before N was fixed and got through three reviews.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from nfl_game.experiments import prereg_totals_2026 as prereg
from nfl_game.paths import PROCESSED_DIR


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


def evaluate(ledger: pd.DataFrame) -> dict:
    """Report progress while the season runs; the primary endpoint only once it is done."""
    rows = qualifying(ledger)
    clv = pd.to_numeric(rows["total_clv"], errors="coerce") if len(rows) else pd.Series(dtype=float)
    settled = int(clv.notna().sum())
    pending = int(len(rows) - settled)
    report = {
        "registration_digest": prereg.registration_digest(),
        "season": prereg.SEASON,
        "qualifying": len(rows),
        "settled": settled,
        "pending": pending,
        "complete": bool(len(rows) > 0 and pending == 0),
    }
    if not report["complete"]:
        report["withheld"] = (
            "primary endpoint withheld until every qualifying game has settled; "
            "see the module docstring"
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
    args = parser.parse_args(argv)
    print(json.dumps(evaluate(pd.read_parquet(args.ledger)), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
