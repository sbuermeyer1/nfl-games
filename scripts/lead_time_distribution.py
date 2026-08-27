"""Realized publication lead times on a real schedule, under the full publication rule.

`PUBLISH_BEFORE` alone does not determine when a pick publishes. The vintage floor holds a
game until the features artifact has been rebuilt from a complete prior week, so a game's
realized lead is `min(PUBLISH_BEFORE, time from the floor release to kickoff)`. Widening the
lock therefore does NOT widen every lead -- it moves games into the floor-bound bucket, where
their lead is set by the refresh cadence and does not change at all.

This script reports that distribution so a lead-time decision is made on realized leads rather
than on the nominal constant. It is the reproducible form of the table in
`docs/superpowers/specs/2026-08-26-publication-lead-time-design.md`; `--assert-baseline`
re-derives that table's published 4-day figures as a self-check on the method.

    python scripts/lead_time_distribution.py --season 2025 --days 4 5 --assert-baseline
"""

import argparse
import re
from pathlib import Path

import pandas as pd

from nfl_game.data.schedule import EASTERN, FINALIZATION_DELAY
from nfl_game.tracking.live import PUBLISH_BEFORE

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEDULES = REPO_ROOT / "data" / "raw" / "schedules_all.parquet"
REFRESH_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "refresh-2026-model.yml"


def refresh_time_utc() -> tuple[int, int]:
    """Read the refresh cadence from the workflow that actually runs it.

    The floor releases a week at the first refresh after the prior week finalizes, so the
    cron schedule is an input to every number this script prints. Parsing it keeps the two
    from drifting apart silently.
    """
    text = REFRESH_WORKFLOW.read_text(encoding="utf-8")
    match = re.search(r'cron:\s*"(\d+)\s+(\d+)\s+\*\s+\*\s+\*"', text)
    if match is None:
        raise SystemExit(f"could not read a daily cron from {REFRESH_WORKFLOW}")
    return int(match.group(2)), int(match.group(1))


# Published in the design spec for the 4-day lock, from the 2025 regular season.
BASELINE_4D = {"games": 272, "full_lead": 251, "floor_bound": 21, "min_lead": 2.31}


def load_regular_season(season: int) -> pd.DataFrame:
    sched = pd.read_parquet(SCHEDULES)
    sched = sched[(sched["season"] == season) & (sched["game_type"] == "REG")].copy()
    naive = pd.to_datetime(sched["gameday"].astype(str) + " " + sched["gametime"].astype(str))
    sched["kickoff_at"] = naive.dt.tz_localize(
        EASTERN, ambiguous="raise", nonexistent="raise"
    ).dt.tz_convert("UTC")
    return sched.sort_values("kickoff_at").reset_index(drop=True)


def floor_release(sched: pd.DataFrame) -> dict[int, pd.Timestamp]:
    """First daily refresh at or after the prior week's last game is final.

    Week 1 has no prior week, so its floor is vacuous and it is absent from the mapping.
    """
    hour, minute = refresh_time_utc()
    last_kickoff = sched.groupby("week")["kickoff_at"].max()
    releases = {}
    for week in sorted(sched["week"].unique()):
        if week == 1:
            continue
        final_at = last_kickoff[week - 1] + FINALIZATION_DELAY
        refresh = final_at.normalize() + pd.Timedelta(hours=hour, minutes=minute)
        if refresh < final_at:  # today's refresh already ran; wait for tomorrow's
            refresh += pd.Timedelta(days=1)
        releases[week] = refresh
    return releases


def leads(sched: pd.DataFrame, publish_before: pd.Timedelta) -> pd.DataFrame:
    releases = floor_release(sched)
    out = sched.copy()
    window_open = out["kickoff_at"] - publish_before
    floor = out["week"].map(releases)
    # A vacuous floor (week 1) never binds.
    out["publish_at"] = window_open.where(floor.isna(), window_open.combine(floor, max))
    out["lead_days"] = (out["kickoff_at"] - out["publish_at"]) / pd.Timedelta(days=1)
    out["floor_bound"] = out["publish_at"] > window_open
    # %-I / %#I are platform-specific; strip the zero pad by hand so this is portable.
    local = out["kickoff_at"].dt.tz_convert(EASTERN)
    out["slot"] = local.dt.strftime("%a ") + local.dt.strftime("%I:%M%p").str.lstrip("0")
    return out


def summarize(out: pd.DataFrame, publish_before: pd.Timedelta) -> dict:
    nominal = publish_before / pd.Timedelta(days=1)
    return {
        "games": len(out),
        "full_lead": int((~out["floor_bound"]).sum()),
        "floor_bound": int(out["floor_bound"].sum()),
        "min_lead": round(out["lead_days"].min(), 2),
        "mean_lead": round(out["lead_days"].mean(), 2),
        "nominal": nominal,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument(
        "--days",
        type=int,
        nargs="+",
        default=[int(PUBLISH_BEFORE / pd.Timedelta(days=1))],
    )
    ap.add_argument("--assert-baseline", action="store_true")
    ap.add_argument("--slots", action="store_true", help="per-slot min-max breakdown")
    args = ap.parse_args()

    sched = load_regular_season(args.season)
    print(f"{args.season} regular season: {len(sched)} games\n")
    print(f"{'lock':>6}{'games':>7}{'full lead':>11}{'floor-bound':>13}{'min':>7}{'mean':>7}")

    summaries = {}
    for days in args.days:
        out = leads(sched, pd.Timedelta(days=days))
        s = summarize(out, pd.Timedelta(days=days))
        summaries[days] = (s, out)
        pct = 100 * s["floor_bound"] / s["games"]
        print(
            f"{days:>5}d{s['games']:>7}{s['full_lead']:>11}"
            f"{s['floor_bound']:>8} ({pct:>4.1f}%){s['min_lead']:>7.2f}{s['mean_lead']:>7.2f}"
        )

    if args.slots:
        for days in args.days:
            _, out = summaries[days]
            print(f"\n--- {days}-day lock, per slot ---")
            print(f"{'slot':>22}{'games':>6}{'lead':>13}{'floor':>6}")
            g = out.groupby("slot").agg(
                games=("lead_days", "size"),
                lo=("lead_days", "min"),
                hi=("lead_days", "max"),
                floor_bound=("floor_bound", "sum"),
            )
            for slot, r in g.sort_values("games", ascending=False).iterrows():
                span = f"{r.lo:.2f}" if abs(r.hi - r.lo) < 5e-3 else f"{r.lo:.2f}-{r.hi:.2f}"
                print(f"{slot:>22}{int(r.games):>6}{span:>13}{int(r.floor_bound):>6}")

    if args.assert_baseline:
        if 4 not in summaries:
            print("\n--assert-baseline needs 4 in --days")
            return 2
        s = summaries[4][0]
        bad = {k: (v, s[k]) for k, v in BASELINE_4D.items() if s[k] != v}
        if bad:
            print(f"\nBASELINE MISMATCH (spec, computed): {bad}")
            return 1
        print(f"\nBaseline OK: 4-day figures reproduce the design spec exactly {BASELINE_4D}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
