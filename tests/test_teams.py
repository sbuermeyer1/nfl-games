import pandas as pd
import pytest

from nfl_game.data.teams import CANONICAL_TEAMS, TEAM_CODE_MAP, normalize_team_codes


def test_relocated_franchises_map_to_their_canonical_code():
    # The three mismatches measured across the real sources: schedules keep the
    # historical code for a relocated franchise while play-by-play always uses the
    # modern one, and NGS spells the Rams LAR where schedules and pbp both say LA.
    assert TEAM_CODE_MAP["OAK"] == "LV"
    assert TEAM_CODE_MAP["SD"] == "LAC"
    assert TEAM_CODE_MAP["STL"] == "LAR"
    assert TEAM_CODE_MAP["LA"] == "LAR"


def test_canonical_codes_are_fixed_points():
    # Normalisation must be idempotent: applying it to already-canonical data is a
    # no-op, so it is safe to apply at every ingestion point without tracking whether
    # a frame has been through it before.
    for code in CANONICAL_TEAMS:
        assert TEAM_CODE_MAP.get(code, code) == code


def test_there_are_thirty_two_canonical_teams():
    assert len(CANONICAL_TEAMS) == 32


def test_normalize_rewrites_every_named_column():
    df = pd.DataFrame(
        {
            "home_team": ["OAK", "LA", "KC"],
            "away_team": ["SD", "STL", "BUF"],
            "unrelated": ["OAK", "LA", "SD"],
        }
    )
    out = normalize_team_codes(df, ["home_team", "away_team"])

    assert list(out["home_team"]) == ["LV", "LAR", "KC"]
    assert list(out["away_team"]) == ["LAC", "LAR", "BUF"]
    # columns not named are left alone -- this function must not guess which columns
    # hold team codes, or it would rewrite unrelated string data
    assert list(out["unrelated"]) == ["OAK", "LA", "SD"]


def test_normalize_does_not_mutate_the_input():
    df = pd.DataFrame({"team": ["OAK"]})
    normalize_team_codes(df, ["team"])
    assert list(df["team"]) == ["OAK"]


def test_normalize_is_idempotent():
    df = pd.DataFrame({"team": ["OAK", "LA", "KC"]})
    once = normalize_team_codes(df, ["team"])
    twice = normalize_team_codes(once, ["team"])
    pd.testing.assert_frame_equal(once, twice)


def test_normalize_preserves_nulls():
    # pbp has null posteam/defteam on timeouts, end-of-quarter rows, etc. Those must
    # stay null rather than becoming the string "nan" or being dropped.
    df = pd.DataFrame({"team": ["OAK", None, "KC"]})
    out = normalize_team_codes(df, ["team"])
    assert out["team"].isna().sum() == 1
    assert list(out["team"].dropna()) == ["LV", "KC"]


def test_normalize_ignores_absent_columns():
    # NGS has team_abbr but no home_team; one call site should not need to know which
    # columns a given source happens to carry.
    df = pd.DataFrame({"team_abbr": ["OAK"]})
    out = normalize_team_codes(df, ["team_abbr", "home_team"])
    assert list(out["team_abbr"]) == ["LV"]


def test_normalize_rejects_an_unrecognised_code():
    # A code that is neither canonical nor a known alias means the upstream feed
    # changed its spelling. Silently passing it through is how the LA/LAR mismatch
    # survived: the join simply missed and features.py zero-filled the result with no
    # error anywhere. Fail loudly instead.
    df = pd.DataFrame({"team": ["KC", "XYZ"]})
    with pytest.raises(ValueError, match="XYZ"):
        normalize_team_codes(df, ["team"])
