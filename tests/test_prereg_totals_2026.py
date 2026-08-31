from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scripts import evaluate_prereg

from nfl_game.experiments import prereg_totals_2026 as prereg


def live(game_id, edge, clv, *, status="published", season=2026, hit="win"):
    return {
        "record_type": "live",
        "season": season,
        "game_id": game_id,
        "total_edge": edge,
        "total_clv": clv,
        "total_publication_status": status,
        "total_close_grade": hit,
        "total_grade": hit,
    }


def ledger(rows):
    return pd.DataFrame(rows)


def test_registration_digest_is_pinned():
    """A silent edit to the registration must fail here rather than pass review.

    If this test fails, the registration changed. That invalidates the test it describes -- do
    not update the digest to match; work out who changed it and why.
    """
    assert prereg.registration_digest() == (
        "450d369d0ca82462950540c36c603cba25e6294a732903d71b88eaa1fab427bc"
    )


def test_registration_was_made_before_any_2026_outcome_existed():
    assert prereg.REGISTERED_STATE == {
        "games_2026_with_result": 0,
        "live_records_at_registration": 0,
    }


def test_qualifying_selects_only_published_2026_totals_over_the_edge():
    frame = ledger([
        live("a", 6.0, 0.5),
        live("b", 4.9, 0.5),                       # under the threshold
        live("c", -7.0, 0.5),                      # negative edge still qualifies on magnitude
        live("d", 6.0, 0.5, status="excluded"),    # never published
        live("e", 6.0, 0.5, season=2025),          # wrong season
        {**live("f", 6.0, 0.5), "record_type": "backtest"},
    ])

    got = evaluate_prereg.qualifying(frame)

    assert sorted(got["game_id"]) == ["a", "c"]


def test_primary_endpoint_is_withheld_while_the_season_is_incomplete():
    """Enforced in code, not by operator discipline.

    Letting someone watch the primary endpoint accumulate is how a pre-registered test turns
    into an optional-stopping one. An incomplete season reports progress and nothing else.
    """
    frame = ledger([live("a", 6.0, 0.5), live("b", 6.0, np.nan)])

    report = evaluate_prereg.evaluate(frame)

    assert report["complete"] is False
    assert "mean_clv" not in report
    assert "verdict" not in report
    assert report["settled"] == 1
    assert report["pending"] == 1


def test_primary_endpoint_is_computed_once_every_qualifying_game_has_settled():
    frame = ledger([live("a", 6.0, 1.0), live("b", 6.0, 0.0), live("c", 6.0, 2.0)])

    report = evaluate_prereg.evaluate(frame)

    assert report["complete"] is True
    assert report["n"] == 3
    assert report["mean_clv"] == pytest.approx(1.0)


def test_verdict_requires_both_the_clv_floor_and_the_z_floor():
    """Either alone must not be enough, or the conjunction is decoration.

    Every arm carries real spread. A constant-CLV fixture has zero variance, which makes z nan
    and hands two of these three arms the right answer for the wrong reason.
    """
    jitter = [0.5 if i % 2 else -0.5 for i in range(40)]

    strong = [live(str(i), 6.0, 0.60 + j) for i, j in enumerate(jitter)]
    report = evaluate_prereg.evaluate(ledger(strong))
    assert report["mean_clv"] == pytest.approx(0.60)
    assert report["z"] > prereg.PRIMARY_MIN_Z
    assert report["verdict"] == "replicated"

    # Mean clears the floor but the spread is far too wide to distinguish it from zero.
    noisy = [live(str(i), 6.0, 0.60 + j * 60.0) for i, j in enumerate(jitter)]
    report = evaluate_prereg.evaluate(ledger(noisy))
    assert report["mean_clv"] == pytest.approx(0.60)
    assert report["z"] < prereg.PRIMARY_MIN_Z
    assert report["verdict"] == "not replicated"

    # Tight and highly significant, but below the replication floor.
    small = [live(str(i), 6.0, 0.10 + j * 0.2) for i, j in enumerate(jitter)]
    report = evaluate_prereg.evaluate(ledger(small))
    assert report["mean_clv"] == pytest.approx(0.10)
    assert report["z"] > prereg.PRIMARY_MIN_Z
    assert report["verdict"] == "not replicated"


def test_empty_season_is_incomplete_rather_than_a_verdict():
    assert evaluate_prereg.evaluate(ledger([]))["complete"] is False
