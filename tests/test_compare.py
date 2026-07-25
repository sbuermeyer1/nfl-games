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
