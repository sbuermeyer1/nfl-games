"""Tests for the realized-lead-time analysis in `scripts/lead_time_distribution.py`.

The script's numbers decide the publication lock, so the properties pinned here are the ones
a wrong answer would hide: that the floor rollover works, that week 1 is exempt, and that
widening the lock can never shorten a game's realized lead.
"""

import pandas as pd
import pytest
from scripts import lead_time_distribution as ltd
from scripts.lead_time_distribution import (
    REFRESH_WORKFLOW,
    floor_release,
    leads,
    refresh_time_utc,
    summarize,
)


def schedule(rows):
    """rows: (week, kickoff ISO in UTC). game_id is synthesized per row."""
    return pd.DataFrame(
        [
            {"game_id": f"g{i}", "week": week, "kickoff_at": pd.Timestamp(kickoff)}
            for i, (week, kickoff) in enumerate(rows)
        ]
    )


def test_floor_releases_at_the_first_refresh_after_the_prior_week_finalizes():
    # Week 1's last kickoff 00:15Z -> final at 06:15Z, before that day's 10:30Z refresh,
    # so week 2 releases the same day.
    sched = schedule([(1, "2026-09-14T00:15:00Z"), (2, "2026-09-20T17:00:00Z")])

    assert floor_release(sched)[2] == pd.Timestamp("2026-09-14T10:30:00Z")


def test_floor_waits_for_the_next_day_when_finalization_lands_after_the_refresh():
    # Week 1's last kickoff 06:00Z -> final at 12:00Z, PAST that day's 10:30Z refresh.
    sched = schedule([(1, "2026-09-15T06:00:00Z"), (2, "2026-09-20T17:00:00Z")])

    assert floor_release(sched)[2] == pd.Timestamp("2026-09-16T10:30:00Z")


def test_week_one_has_no_floor_because_it_has_no_prior_week():
    sched = schedule([(1, "2026-09-13T17:00:00Z"), (2, "2026-09-20T17:00:00Z")])

    assert 1 not in floor_release(sched)


def test_week_one_takes_the_full_nominal_lead_and_is_never_floor_bound():
    sched = schedule([(1, "2026-09-13T17:00:00Z"), (2, "2026-09-20T17:00:00Z")])

    out = leads(sched, pd.Timedelta(days=5)).set_index("game_id")

    assert out.loc["g0", "lead_days"] == 5.0
    assert not out.loc["g0", "floor_bound"]


def test_a_game_held_by_the_floor_reports_the_floor_lead_not_the_nominal_one():
    # Week 2 Thursday kickoff whose 5-day mark precedes the week-2 release.
    sched = schedule([(1, "2026-09-14T00:15:00Z"), (2, "2026-09-17T00:15:00Z")])

    out = leads(sched, pd.Timedelta(days=5)).set_index("game_id")

    assert out.loc["g1", "floor_bound"]
    # released 2026-09-14T10:30Z, kickoff 2026-09-17T00:15Z
    assert out.loc["g1", "lead_days"] == pytest.approx(2.5729166, abs=1e-6)


def test_widening_the_lock_never_shortens_a_realized_lead():
    """The property that makes a longer lock safe to ship.

    A wider window can only move a game's publication earlier or leave it pinned at the
    floor, which does not move. If this ever fails, a lock increase is silently costing
    some slot lead time and the slot table must be read before shipping it.
    """
    sunday = pd.Timestamp("2026-09-13T17:00:00Z")
    thursday = pd.Timestamp("2026-09-17T00:15:00Z")
    week = pd.Timedelta(days=7)
    sched = schedule(
        [(w, (sunday + (w - 1) * week).isoformat()) for w in range(1, 6)]
        + [(w, (thursday + (w - 2) * week).isoformat()) for w in range(2, 6)]
    )

    narrow = leads(sched, pd.Timedelta(days=4)).set_index("game_id")["lead_days"]
    wide = leads(sched, pd.Timedelta(days=5)).set_index("game_id")["lead_days"]

    assert (wide >= narrow - 1e-9).all()
    assert (wide > narrow + 1e-9).any()  # and it does help somebody


def test_summarize_counts_floor_bound_games_separately_from_full_lead_ones():
    sched = schedule([(1, "2026-09-14T00:15:00Z"), (2, "2026-09-17T00:15:00Z")])

    s = summarize(leads(sched, pd.Timedelta(days=5)), pd.Timedelta(days=5))

    assert s["games"] == 2
    assert s["full_lead"] == 1
    assert s["floor_bound"] == 1
    assert s["nominal"] == 5.0


def test_refresh_time_follows_the_workflow_cron_rather_than_a_constant(tmp_path, monkeypatch):
    """Pointed at a DIFFERENT cron, the parser must report that one.

    Asserting only the shipped (10, 30) cannot distinguish parsing from a hardcoded return --
    both pass. Changing the file underneath it is what lets this fail for its actual reason.
    """
    workflow = tmp_path / "refresh.yml"
    workflow.write_text('on:\n  schedule:\n    - cron: "45 3 * * *"\n', encoding="utf-8")
    monkeypatch.setattr(ltd, "REFRESH_WORKFLOW", workflow)

    assert refresh_time_utc() == (3, 45)


def test_refresh_time_matches_the_cron_actually_shipped():
    assert refresh_time_utc() == (10, 30)
    assert 'cron: "30 10 * * *"' in REFRESH_WORKFLOW.read_text(encoding="utf-8")


def test_refresh_time_refuses_a_workflow_with_no_daily_cron(tmp_path, monkeypatch):
    """Guessing a cadence here would silently invent every lead time downstream."""
    workflow = tmp_path / "refresh.yml"
    workflow.write_text("on:\n  workflow_dispatch:\n", encoding="utf-8")
    monkeypatch.setattr(ltd, "REFRESH_WORKFLOW", workflow)

    with pytest.raises(SystemExit):
        refresh_time_utc()
