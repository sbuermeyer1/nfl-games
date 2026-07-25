import pandas as pd

from nfl_game.market.compare import SLATE_COLS, build_slate, slate_markdown


def _inputs():
    feats = pd.DataFrame(
        {
            "game_id": ["2026_01_KC_BUF", "2026_01_MIA_NYJ"],
            "season": [2026, 2026],
            "week": [1, 1],
            "home_team": ["BUF", "NYJ"],
            "away_team": ["KC", "MIA"],
            "spread_line": [2.5, -1.0],
            "total_line": [48.5, 43.0],
        }
    )
    preds = pd.DataFrame(
        {
            "game_id": ["2026_01_KC_BUF", "2026_01_MIA_NYJ"],
            "model_margin": [6.0, -1.5],
            "model_total": [51.0, 43.2],
        }
    )
    probs = pd.DataFrame(
        {
            "game_id": ["2026_01_KC_BUF", "2026_01_MIA_NYJ"],
            "cover_prob": [0.58, 0.49],
            "over_prob": [0.55, 0.51],
        }
    )
    return feats, preds, probs


def test_slate_has_fixed_schema():
    out = build_slate(*_inputs())
    assert list(out.columns) == SLATE_COLS


def test_gap_is_model_minus_market():
    out = build_slate(*_inputs()).set_index("game_id")
    assert out.loc["2026_01_KC_BUF", "spread_gap"] == 3.5  # 6.0 - 2.5
    assert out.loc["2026_01_KC_BUF", "total_gap"] == 2.5  # 51.0 - 48.5


def test_edge_flag_respects_threshold():
    out = build_slate(*_inputs(), edge_threshold=2.0).set_index("game_id")
    assert out.loc["2026_01_KC_BUF", "edge_flag"] == 1  # gap 3.5
    assert out.loc["2026_01_MIA_NYJ", "edge_flag"] == 0  # gap 0.5


def test_higher_threshold_flags_fewer_games():
    feats, preds, probs = _inputs()
    assert build_slate(feats, preds, probs, edge_threshold=10.0)["edge_flag"].sum() == 0


def test_sorted_by_absolute_edge():
    out = build_slate(*_inputs())
    assert out.iloc[0]["game_id"] == "2026_01_KC_BUF"


def test_sort_order_independent_of_input_row_order():
    # In `_inputs()`, the largest-gap game (KC_BUF, gap 3.5) is already first in the
    # input frames, so `test_sorted_by_absolute_edge` above would pass even with no
    # sorting at all (verified: removing the sort still passes it). This fixture puts
    # the smaller-gap game first in every input frame instead, so only an actual sort
    # by absolute edge — not input order — produces KC_BUF first in the output.
    feats = pd.DataFrame(
        {
            "game_id": ["2026_01_MIA_NYJ", "2026_01_KC_BUF"],
            "season": [2026, 2026],
            "week": [1, 1],
            "home_team": ["NYJ", "BUF"],
            "away_team": ["MIA", "KC"],
            "spread_line": [-1.0, 2.5],
            "total_line": [43.0, 48.5],
        }
    )
    preds = pd.DataFrame(
        {
            "game_id": ["2026_01_MIA_NYJ", "2026_01_KC_BUF"],
            "model_margin": [-1.5, 6.0],
            "model_total": [43.2, 51.0],
        }
    )
    probs = pd.DataFrame(
        {
            "game_id": ["2026_01_MIA_NYJ", "2026_01_KC_BUF"],
            "cover_prob": [0.49, 0.58],
            "over_prob": [0.51, 0.55],
        }
    )
    out = build_slate(feats, preds, probs)
    assert list(out["game_id"]) == ["2026_01_KC_BUF", "2026_01_MIA_NYJ"]


def test_markdown_renders_every_game():
    md = slate_markdown(build_slate(*_inputs()))
    assert "KC" in md and "BUF" in md and "NYJ" in md
    assert md.startswith("|")


def test_markdown_game_column_is_away_at_home_not_flipped():
    # slate_markdown must render "{away} @ {home}", never the reverse. The brief-mandated
    # test above only checks substring presence, so "BUF @ KC" would pass just as readily
    # as the correct "KC @ BUF" -- pin the exact substring for both games.
    md = slate_markdown(build_slate(*_inputs()))
    assert "KC @ BUF" in md
    assert "BUF @ KC" not in md
    assert "MIA @ NYJ" in md
    assert "NYJ @ MIA" not in md


def test_markdown_handles_missing_cover_prob_without_rendering_nan():
    # Calibrator.predict returns NaN cover_prob for a row missing spread_line (e.g. a
    # not-yet-posted line on an upcoming slate) while still returning a real over_prob.
    # A bare f"{nan:.1%}" formats as the literal string "nan%", which reads like a data
    # bug rather than "unavailable." slate_markdown must not emit that.
    feats, preds, probs = _inputs()
    probs = probs.copy()
    probs.loc[0, "cover_prob"] = float("nan")
    md = slate_markdown(build_slate(feats, preds, probs))
    assert "nan%" not in md


def test_markdown_handles_missing_spread_line_without_rendering_nan_anywhere():
    # A missing spread_line propagates NaN into market_spread, spread_gap, and
    # cover_prob alike -- the "nan reads as a data bug" rationale applies identically
    # to all of them, not just the probability column. The previous test only checked
    # "nan%" not in md, which passed even though market_spread/spread_gap still
    # rendered "nan"/"+nan" literally; this pins the full row instead.
    feats, preds, probs = _inputs()
    feats = feats.copy()
    probs = probs.copy()
    feats.loc[0, "spread_line"] = float("nan")
    probs.loc[0, "cover_prob"] = float("nan")
    out = build_slate(feats, preds, probs)
    md = slate_markdown(out)

    assert "nan" not in md.lower()
    row = out.set_index("game_id").loc["2026_01_KC_BUF"]
    assert pd.isna(row["market_spread"]) and pd.isna(row["spread_gap"])
    lines = [line for line in md.splitlines() if "KC @ BUF" in line]
    assert len(lines) == 1
    cells = [c.strip() for c in lines[0].strip("|").split("|")]
    # Game, Model, Market, Gap, Cover%, Model O/U, Market O/U, Gap, Over%, Edge
    assert cells[2] == "n/a"  # market spread
    assert cells[3] == "n/a"  # spread gap
    assert cells[4] == "n/a"  # cover%


def test_markdown_handles_missing_total_line_without_rendering_nan_anywhere():
    # The over/under mirror of the test above. Both halves of the "n/a" guard were
    # fixed together, but only the spread half was pinned: reverting market_total and
    # total_gap to a raw format() left the whole suite green, because no test nulled
    # total_line. An upcoming slate can have a posted spread and no posted total just
    # as easily as the reverse.
    feats, preds, probs = _inputs()
    feats = feats.copy()
    probs = probs.copy()
    feats.loc[0, "total_line"] = float("nan")
    probs.loc[0, "over_prob"] = float("nan")
    out = build_slate(feats, preds, probs)
    md = slate_markdown(out)

    assert "nan" not in md.lower()
    row = out.set_index("game_id").loc["2026_01_KC_BUF"]
    assert pd.isna(row["market_total"]) and pd.isna(row["total_gap"])
    lines = [line for line in md.splitlines() if "KC @ BUF" in line]
    assert len(lines) == 1
    cells = [c.strip() for c in lines[0].strip("|").split("|")]
    # Game, Model, Market, Gap, Cover%, Model O/U, Market O/U, Gap, Over%, Edge
    assert cells[6] == "n/a"  # market total
    assert cells[7] == "n/a"  # total gap
    assert cells[8] == "n/a"  # over%
    # the spread side still carries real values -- only the total side is missing
    assert cells[2] == "+2.5"


def test_model_and_market_spread_columns_are_not_swapped():
    # Nothing in the brief-mandated tests pins model_spread/market_spread individually
    # -- test_gap_is_model_minus_market only checks spread_gap, which is computed
    # independently and stays correct even if the two source columns were swapped.
    # Swapping them would invert every displayed pick.
    row = build_slate(*_inputs()).set_index("game_id").loc["2026_01_KC_BUF"]
    assert (row["model_spread"], row["market_spread"]) == (6.0, 2.5)


def test_cover_and_over_prob_columns_are_not_swapped():
    row = build_slate(*_inputs()).set_index("game_id").loc["2026_01_KC_BUF"]
    assert (row["cover_prob"], row["over_prob"]) == (0.58, 0.55)


def _inputs_edge_cases():
    """Fixture for the sort/edge_flag mutants the plain two-game _inputs() fixture
    can't distinguish: a negative gap bigger in magnitude than any positive gap (so
    sorting by signed value gives a different order than sorting by absolute value), a
    gap exactly at the default edge_threshold (so `>` vs `>=` differ), and a game whose
    total_gap is large while its spread_gap is small (so edge_flag driven by total_gap
    instead of spread_gap would flag it incorrectly)."""
    feats = pd.DataFrame(
        {
            "game_id": ["g_pos", "g_neg", "g_at_threshold", "g_total_only"],
            "season": [2026, 2026, 2026, 2026],
            "week": [2, 2, 2, 2],
            "home_team": ["BBB", "DDD", "FFF", "HHH"],
            "away_team": ["AAA", "CCC", "EEE", "GGG"],
            "spread_line": [1.0, 5.0, 3.0, 0.0],
            "total_line": [40.0, 44.0, 45.0, 40.0],
        }
    )
    preds = pd.DataFrame(
        {
            "game_id": ["g_pos", "g_neg", "g_at_threshold", "g_total_only"],
            "model_margin": [4.0, -1.0, 5.0, 0.5],
            "model_total": [41.0, 44.5, 45.0, 48.0],
        }
    )
    probs = pd.DataFrame(
        {
            "game_id": ["g_pos", "g_neg", "g_at_threshold", "g_total_only"],
            "cover_prob": [0.55, 0.40, 0.60, 0.51],
            "over_prob": [0.52, 0.51, 0.50, 0.75],
        }
    )
    return feats, preds, probs


# spread_gap: g_pos=+3.0, g_neg=-6.0, g_at_threshold=+2.0, g_total_only=+0.5
# total_gap:  g_pos=+1.0, g_neg=+0.5, g_at_threshold=+0.0, g_total_only=+8.0


def test_sorted_by_absolute_edge_not_signed_edge():
    # g_neg's gap (-6.0) is the largest by absolute value but the smallest (most
    # negative) by signed value. Sorting by signed value descending would put it last;
    # sorting by absolute value descending puts it first.
    out = build_slate(*_inputs_edge_cases())
    assert list(out["game_id"])[:2] == ["g_neg", "g_pos"]


def test_edge_flag_threshold_is_inclusive():
    # g_at_threshold's spread_gap is exactly 2.0, equal to the default edge_threshold.
    # The brief specifies `>=`, so this must flag; a `>` mutant would not.
    out = build_slate(*_inputs_edge_cases()).set_index("game_id")
    assert out.loc["g_at_threshold", "edge_flag"] == 1


def test_edge_flag_is_driven_by_spread_gap_not_total_gap():
    # g_total_only has a small spread_gap (0.5, below threshold) but a large total_gap
    # (8.0, above threshold). edge_flag must stay 0: it flags spread disagreement only.
    out = build_slate(*_inputs_edge_cases()).set_index("game_id")
    assert out.loc["g_total_only", "edge_flag"] == 0
    assert out.loc["g_total_only", "total_gap"] == 8.0
