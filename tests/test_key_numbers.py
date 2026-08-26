"""The half-point analysis turns entirely on its sign convention.

`analyze_line_value.py` settles a home pick when `margin > line` and an away pick when
`margin < line`, with `margin == line` a push. If the "better number" is applied in the wrong
direction the script still runs and still prints a plausible table -- it just reports the value
of a WORSE number, which would invert the conclusion. These tests pin the direction on hand-built
cases where the right answer is obvious by inspection.
"""

import numpy as np
import pandas as pd
from scripts.analyze_key_numbers import (
    _outcome,
    _returns,
    half_point_value,
    margin_frequency,
)


def _frame(rows):
    """rows: (margin, spread_line, edge_close)."""
    return pd.DataFrame(
        [{"margin": m, "spread_line": line, "edge_close": edge} for m, line, edge in rows]
    )


def test_outcome_settles_home_and_away_picks_in_opposite_directions():
    margin = np.array([7.0, 7.0, 3.0])
    line = np.array([3.0, 3.0, 3.0])
    home_pick = np.array([True, False, True])

    assert list(_outcome(margin, line, home_pick)) == ["WIN", "LOSS", "PUSH"]


def test_returns_price_a_push_between_a_loss_and_a_win():
    out = _returns(np.array(["WIN", "PUSH", "LOSS"]))

    assert out[0] > 0
    assert out[1] == 0.0
    assert out[2] == -1.0
    assert out[0] < 1.0, "a -110 win pays less than the stake risked"


def test_a_better_number_converts_a_home_loss_into_a_push():
    # Home favoured by 3, game lands on exactly 3: at the 3 they bet, this is a push
    # already -- so use a line of 3.5, where the home pick LOSES by a half point.
    frame = _frame([(3.0, 3.5, +2.0)])

    row = half_point_value(frame, threshold=0.0, step=0.5)

    assert row["n"] == 1
    assert row["loss_to_push"] == 1
    assert row["ev_gain_per_bet"] > 0


def test_a_better_number_converts_an_away_loss_into_a_push():
    # Away pick (negative edge) needs margin < line. Margin 3 against a line of 2.5 loses;
    # half a point better for an away bettor RAISES the line to 3.0, making it a push.
    frame = _frame([(3.0, 2.5, -2.0)])

    row = half_point_value(frame, threshold=0.0, step=0.5)

    assert row["n"] == 1
    assert row["loss_to_push"] == 1
    assert row["ev_gain_per_bet"] > 0


def test_a_better_number_never_makes_a_bet_worse():
    """The direction pin: applying the step the wrong way would produce negative gain."""
    frame = _frame(
        [
            (7.0, 3.5, +2.0),  # home already winning comfortably
            (3.0, 2.5, -2.0),  # away loss that a better number rescues
            (-10.0, 3.5, +2.0),  # home loss a half point cannot rescue
            (14.0, 7.0, -2.0),  # away loss a half point cannot rescue
        ]
    )

    row = half_point_value(frame, threshold=0.0, step=0.5)

    assert row["ev_gain_per_bet"] >= 0.0
    assert row["loss_to_win"] == 0, "a half point cannot span a win and a loss on integer margins"


def test_threshold_filters_on_absolute_edge_so_away_picks_survive():
    frame = _frame([(3.0, 2.5, -2.0), (3.0, 2.5, -0.5)])

    row = half_point_value(frame, threshold=2.0, step=0.5)

    assert row["n"] == 1, "a strong AWAY pick has edge -2.0 and must not be filtered out"


def test_margin_frequency_is_computed_on_absolute_margins():
    frame = _frame([(3.0, 0.0, 1.0), (-3.0, 0.0, 1.0), (7.0, 0.0, 1.0)])

    freq = margin_frequency(frame)
    row = freq.loc[freq["abs_margin"] == 3.0].iloc[0]

    assert row["games"] == 2
    assert row["share"] == 2 / 3
