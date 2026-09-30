"""The static site must publish exactly what the served dashboard answers.

Every oracle here is the served FastAPI app over the same services, never the export
code itself: each API URL a page can request is mapped to its file by the shim's own
`staticPath`, and that file is compared with the live response.
"""

import json
import warnings

import pandas as pd
import pytest
import quickjs
from fastapi.testclient import TestClient

from nfl_game.paths import PROCESSED_DIR
from nfl_game.web.app import create_app
from nfl_game.web.service import SlateService
from nfl_game.web.static_export import SHIM_TAG, export_site, with_shim
from nfl_game.web.static_shim import STATIC_SHIM
from nfl_game.web.tracker_service import TrackerService

# Real packaged data, trimmed so the fits stay fast. 2021 has no prior season inside
# the trim, so its slates exercise the exported 409 path.
SEASONS = (2021, 2022, 2023)
THRESHOLDS = (0, 0.5, 1.5, 2, 3.5, 100)


@pytest.fixture(scope="module")
def services():
    features = pd.read_parquet(PROCESSED_DIR / "game_features.parquet")
    trimmed = features[features["season"].isin(SEASONS)].reset_index(drop=True)
    slate = SlateService(trimmed)
    tracker = TrackerService.from_parquet(
        PROCESSED_DIR / "tracker_ledger.parquet", PROCESSED_DIR / "tracker_reconstructed.parquet"
    )
    return slate, tracker


@pytest.fixture(scope="module")
def site(services, tmp_path_factory):
    out = tmp_path_factory.mktemp("site")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        counts = export_site(out, *services)
    return out, counts


@pytest.fixture(scope="module")
def served(services):
    return TestClient(create_app(*services, access_code=None))


@pytest.fixture(scope="module")
def shim():
    context = quickjs.Context()
    context.eval(STATIC_SHIM)
    return context


def call(shim, name, *args):
    encoded = ", ".join(json.dumps(arg) for arg in args)
    return json.loads(shim.eval(f"JSON.stringify(__staticSite.{name}({encoded}))"))


def exported(site_dir, shim, url):
    path = call(shim, "staticPath", url)
    return json.loads((site_dir / path.lstrip("/")).read_text(encoding="utf-8"))


def page_urls(served):
    """Every API URL the three pages can request, enumerated from the served options."""
    options = served.get("/api/options").json()
    urls = ["/api/options", "/api/schedule?season=2026", "/api/tracker/options"]
    slates = []
    for season in options["seasons"]:
        urls.append(f"/api/weeks?season={season}")
        for week in served.get(f"/api/weeks?season={season}").json()["weeks"]:
            for estimator in options["estimators"]:
                slates.append(f"season={season}&week={week}&estimator={estimator}")
    tracker = served.get("/api/tracker/options").json()
    for record_type in tracker["record_types"]:
        seasons = tracker["seasons"][record_type]
        for season in ["all", *seasons]:
            urls.append(f"/api/tracker/summary?record_type={record_type}&season={season}")
        for season in seasons:
            urls.append(f"/api/tracker/games?season={season}&record_type={record_type}")
    return urls, slates


def test_every_page_request_is_exported_identically(site, served, shim):
    site_dir, _ = site
    urls, slates = page_urls(served)
    urls += [f"/api/slate?{query}&edge_threshold=2" for query in slates]
    statuses = set()
    for url in urls:
        live = served.get(url)
        envelope = exported(site_dir, shim, url)
        assert envelope == {"status": live.status_code, "body": live.json()}, url
        statuses.add(live.status_code)
    # The comparison covered both published slates and exported errors.
    assert statuses >= {200, 409}


def test_slate_edges_match_the_server_at_every_threshold(site, served, shim):
    site_dir, _ = site
    _, slates = page_urls(served)
    checked = 0
    for query in slates:
        for threshold in THRESHOLDS:
            url = f"/api/slate?{query}&edge_threshold={threshold}"
            live = served.get(url)
            if live.status_code != 200:
                continue
            games = exported(site_dir, shim, url)["body"]["games"]
            assert call(shim, "withEdge", games, threshold) == live.json()["games"], url
            checked += 1
    assert checked


def test_csv_download_matches_the_server_at_every_threshold(site, served, shim):
    site_dir, _ = site
    _, slates = page_urls(served)
    flipped = 0
    for query in slates:
        base = exported(site_dir, shim, f"/api/slate?{query}")
        if base["status"] != 200:
            continue
        csv_path = site_dir / call(shim, "staticPath", f"/api/slate.csv?{query}").lstrip("/")
        csv = csv_path.read_bytes().decode("utf-8")
        for threshold in THRESHOLDS:
            games = call(shim, "withEdge", base["body"]["games"], threshold)
            flags = [game["edge_flag"] for game in games]
            patched = call(shim, "patchCsvEdge", csv, flags)
            live = served.get(f"/api/slate.csv?{query}&edge_threshold={threshold}")
            assert patched == live.content.decode("utf-8"), (query, threshold)
            flipped += patched != csv
    # Non-default thresholds really did rewrite some flags.
    assert flipped


def test_missing_spread_gap_is_never_flagged_even_at_zero(shim):
    games = [{"spread_gap": None}, {"spread_gap": 0.0}, {"spread_gap": -2.5}]
    flags = [game["edge_flag"] for game in call(shim, "withEdge", games, 0)]
    assert flags == [0, 1, 1]
    flags = [game["edge_flag"] for game in call(shim, "withEdge", games, 2.5)]
    assert flags == [0, 0, 1]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("2", {"value": 2}),
        (" 1.5 ", {"value": 1.5}),
        ("", {"error": "Invalid request"}),
        ("abc", {"error": "Invalid request"}),
        ("-1", {"error": "edge threshold must be a finite non-negative number"}),
        ("Infinity", {"error": "edge threshold must be a finite non-negative number"}),
    ],
)
def test_threshold_parsing_mirrors_the_server(shim, served, raw, expected):
    assert call(shim, "parseThreshold", raw) == expected
    live = served.get(f"/api/slate?season=2023&week=1&estimator=ridge&edge_threshold={raw.strip()}")
    if "error" in expected:
        assert live.status_code == 422
        assert live.json() == {"error": expected["error"]}
    else:
        assert live.status_code == 200


def test_patch_csv_keeps_quoted_fields_and_line_endings(shim):
    text = 'game,edge_flag,qb\r\n"A, B",0,"say ""hi"""\r\nC,1,\r\n'
    patched = call(shim, "patchCsvEdge", text, [1, 0])
    assert patched == 'game,edge_flag,qb\r\n"A, B",1,"say ""hi"""\r\nC,0,\r\n'


def test_patch_csv_refuses_a_row_count_mismatch(shim):
    with pytest.raises(quickjs.JSException, match="does not match"):
        call(shim, "patchCsvEdge", "edge_flag\n0\n", [1, 0])


def test_unknown_urls_are_left_to_the_network(shim):
    assert call(shim, "staticPath", "/static-shim.js") is None
    assert call(shim, "staticPath", "/api/slate?season=2023&week=1&estimator=ridge") == (
        "/data/slate/2023/1/ridge.json"
    )


def test_pages_load_the_shim_before_their_own_script(site):
    site_dir, counts = site
    assert counts["pages"] == 3
    for name in ("index.html", "schedule.html", "tracker.html"):
        html = (site_dir / name).read_text(encoding="utf-8")
        assert html.count(SHIM_TAG) == 1
        assert html.index(SHIM_TAG) < html.index("<script>")
    assert (site_dir / "static-shim.js").read_text(encoding="utf-8") == STATIC_SHIM


def test_with_shim_rejects_a_page_with_more_than_one_script():
    with pytest.raises(ValueError):
        with_shim("<script>a</script><script>b</script>")


def test_export_replaces_a_stale_site(services, tmp_path):
    stale = tmp_path / "data" / "slate" / "1999" / "1" / "ridge.json"
    stale.parent.mkdir(parents=True)
    stale.write_text("{}")
    slate, tracker = services
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        export_site(tmp_path, slate, tracker)
    assert not stale.exists()
