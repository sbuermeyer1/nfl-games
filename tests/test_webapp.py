import csv
import io
import json
import math
import re

import pytest
import quickjs
from fastapi.testclient import TestClient

from nfl_game.web.app import create_app
from nfl_game.web.service import (
    SlateInputError,
    SlateNotFoundError,
    SlateUnavailableError,
)
from nfl_game.web.tracker_service import TrackerInputError

PAGE_SCRIPT = re.compile(r"<script>(.*?)</script>", re.DOTALL)


class FakeService:
    def __init__(self):
        self.calls: list[tuple] = []

    def options(self):
        self.calls.append(("options",))
        return {
            "seasons": [2024, 2025],
            "weeks": [1, 3],
            "estimators": ["gbm", "ridge"],
            "default_estimator": "ridge",
            "default_edge_threshold": 2.0,
            "latest": {"season": 2025, "week": 3},
        }

    def weeks(self, season):
        self.calls.append(("weeks", season))
        if season == 2024:
            return [2]
        if season == 2025:
            return [1, 3]
        raise SlateInputError(f"season {season} is not available")

    @staticmethod
    def _validate(estimator, edge_threshold):
        if estimator not in {"gbm", "ridge"}:
            raise SlateInputError(f"estimator {estimator!r} is not available")
        if not math.isfinite(edge_threshold) or edge_threshold < 0:
            raise SlateInputError("edge threshold must be a finite non-negative number")

    def payload(self, season, week, estimator, edge_threshold):
        self.calls.append(("payload", season, week, estimator, edge_threshold))
        return {
            "games": self.records(season, week, estimator, edge_threshold),
            "market": {
                "source": "nflverse",
                "observed_at": "2026-09-01T12:00:00+00:00",
                "stale": False,
            },
        }

    def records(self, season, week, estimator, edge_threshold):
        self.calls.append(("records", season, week, estimator, edge_threshold))
        self._validate(estimator, edge_threshold)
        if week == 2 and season == 2025:
            raise SlateInputError("week 2 is not available for season 2025")
        if week == 3 and estimator == "gbm":
            raise SlateUnavailableError("cannot train gbm for season 2025")
        if week == 4:
            raise SlateNotFoundError(f"no games are available for season {season} week 4")
        if week == 5:
            raise RuntimeError("database password leaked in internal trace")
        return [
            {
                "game_id": "2025_01_AAA_BBB",
                "season": season,
                "week": week,
                "away_team": "AAA",
                "home_team": "BBB",
                "model_spread": 4.0,
                "market_spread": 2.5,
                "spread_gap": 1.5,
                "cover_prob": 0.55,
                "model_total": 46.0,
                "market_total": None,
                "total_gap": None,
                "over_prob": None,
                "edge_flag": 0,
            }
        ]

    def csv(self, season, week, estimator, edge_threshold):
        self.calls.append(("csv", season, week, estimator, edge_threshold))
        rows = self.records(season, week, estimator, edge_threshold)
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        return output.getvalue()

    def schedule_records(self, season):
        self.calls.append(("schedule_records", season))
        return {
            "season": season,
            "games": [
                {
                    "game_id": f"{season}_01_AAA_BBB",
                    "season": season,
                    "week": 1,
                    "kickoff_at": "2026-09-01T17:00:00+00:00",
                    "away_team": "AAA",
                    "home_team": "BBB",
                    "spread_line": 2.5,
                    "total_line": None,
                }
            ],
            "market": {
                "source": "nflverse",
                "observed_at": "2026-09-01T12:00:00+00:00",
                "stale": False,
            },
        }


class FakeTrackerService:
    def __init__(self):
        self.calls: list[tuple] = []

    def options(self):
        self.calls.append(("options",))
        return {
            "record_types": ["backtest", "live"],
            "seasons": {"backtest": [2024, 2025], "live": [2026]},
            "default_record_type": "backtest",
            "default_season": "all",
            "model_version": "ridge-v1",
            "qualified_edge": 2.0,
            "spread_edge_thresholds": [5.0, 10.0, 15.0],
            "live_available": True,
        }

    def summary(self, record_type, season):
        self.calls.append(("summary", record_type, season))
        if record_type == "research":
            raise TrackerInputError("invalid record type")
        return {"available": True, "record_type": record_type, "season": season}

    def records(self, record_type, season):
        self.calls.append(("records", record_type, season))
        return [{"game_id": f"{season}_01_AAA_BBB"}]


def client(service=None, tracker_service=None, access_code=None):
    return TestClient(
        create_app(
            service or FakeService(),
            tracker_service or FakeTrackerService(),
            access_code=access_code,
        ),
        base_url="https://testserver",
        follow_redirects=False,
    )


def response(status=200, body=None):
    return {"status": status, "body": body if body is not None else {}}


def dashboard_state(http_client, responses, actions):
    """Run the returned dashboard script with DOM/fetch behavior in QuickJS."""
    page = http_client.get("/").text
    script = PAGE_SCRIPT.search(page)
    assert script, "dashboard page must contain an executable script"
    payload = json.dumps({"script": script.group(1), "responses": responses, "actions": actions})
    context = quickjs.Context()
    context.eval(f"const input = JSON.parse({json.dumps(payload)});")
    context.eval(DASHBOARD_HARNESS)
    while context.execute_pending_job():
        pass
    return json.loads(context.eval("globalThis.__state"))


DASHBOARD_HARNESS = r"""
const listeners = {};
const pending = {};
const calls = [];
const unhandled = [];

class Element {
  constructor(tagName) {
    this.tagName = tagName;
    this.children = [];
    this.listeners = {};
    this.textContent = '';
    this.className = '';
    this.disabled = false;
    this.selected = false;
    this._value = undefined;
  }
  appendChild(child) { this.children.push(child); return child; }
  replaceChildren(...children) { this.children = children; }
  addEventListener(name, callback) { (this.listeners[name] ||= []).push(callback); }
  get value() {
    if (this._value !== undefined) return this._value;
    const selected = this.children.find(child => child.selected) || this.children[0];
    return selected ? selected.value : '';
  }
  set value(value) {
    this._value = String(value);
    for (const child of this.children) child.selected = child.value === this._value;
  }
}

const nodes = Object.fromEntries(
  ['season', 'week', 'estimator', 'edge', 'run', 'download', 'message', 'market-message', 'results']
    .map(id => [id, new Element(id === 'edge' ? 'input' : 'select')])
);
nodes.run.tagName = 'button';
nodes.download.tagName = 'button';
nodes.message.tagName = 'p';
nodes['market-message'].tagName = 'p';
nodes.results.tagName = 'table';

globalThis.document = {
  getElementById: id => nodes[id],
  createElement: tagName => new Element(tagName),
  addEventListener: (name, callback) => { (listeners[name] ||= []).push(callback); },
};
globalThis.window = { location: '' };
class URLSearchParams {
  constructor(values) { this.values = values; }
  toString() {
    return Object.entries(this.values)
      .map(([key, value]) => `${encodeURIComponent(key)}=${encodeURIComponent(value).replace(/%20/g, '+')}`)
      .join('&');
  }
}
globalThis.URLSearchParams = URLSearchParams;

function makeResponse(spec) {
  return {
    status: spec.status ?? 200,
    ok: (spec.status ?? 200) >= 200 && (spec.status ?? 200) < 300,
    json: async () => spec.body,
  };
}

globalThis.fetch = url => {
  calls.push(url);
  const spec = input.responses[url];
  if (!spec) return Promise.reject(new Error(`Unexpected URL: ${url}`));
  if (spec.network) return Promise.reject(new Error('network unavailable'));
  if (spec.deferred) return new Promise((resolve, reject) => { pending[url] = { resolve, reject }; });
  return Promise.resolve(makeResponse(spec));
};

async function settle() {
  for (let index = 0; index < 12; index += 1) await Promise.resolve();
}

function trigger(target, name, wait) {
  const callbacks = target === 'document' ? listeners[name] || [] : nodes[target].listeners[name] || [];
  const promises = callbacks.map(callback => Promise.resolve().then(callback).catch(error => {
    unhandled.push(error.message);
  }));
  return wait ? Promise.all(promises) : Promise.resolve();
}

eval(input.script);

(async () => {
  for (const action of input.actions) {
    if (action.type === 'fire') await trigger(action.target, action.event, action.wait);
    if (action.type === 'set') nodes[action.target].value = action.value;
    if (action.type === 'resolve') pending[action.url].resolve(makeResponse(action.response));
    if (action.type === 'settle') await settle();
  }
  await settle();
  const rows = nodes.results.children.map(row => ({
    className: row.className,
    cells: row.children.map(cell => cell.textContent),
  }));
  const options = id => nodes[id].children.map(option => ({ value: option.value, selected: option.selected }));
  globalThis.__state = JSON.stringify({
    calls,
    unhandled,
    location: window.location,
    message: nodes.message.textContent,
    marketMessage: nodes['market-message'].textContent,
    rows,
    season: { value: nodes.season.value, options: options('season') },
    week: { value: nodes.week.value, options: options('week') },
    estimator: { value: nodes.estimator.value, options: options('estimator') },
    edge: nodes.edge.value,
    runDisabled: nodes.run.disabled,
    downloadDisabled: nodes.download.disabled,
  });
})().catch(error => { globalThis.__state = JSON.stringify({error: error.stack}); });
"""


def dashboard_game(away_team="AAA", home_team="BBB"):
    return {
        "away_team": away_team,
        "home_team": home_team,
        "model_spread": 4.0,
        "market_spread": 2.5,
        "spread_gap": 1.5,
        "cover_prob": 0.55,
        "model_total": 46.0,
        "market_total": 44.5,
        "total_gap": 1.5,
        "over_prob": 0.55,
        "edge_flag": 0,
    }


def standard_responses():
    return {
        "/api/options": response(
            body={
                "seasons": [2024, 2025],
                "weeks": [1, 3],
                "estimators": ["gbm", "ridge"],
                "default_estimator": "ridge",
                "default_edge_threshold": 2.0,
                "latest": {"season": 2025, "week": 3},
            }
        ),
        "/api/slate?season=2025&week=3&estimator=ridge&edge_threshold=2": response(
            body={
                "games": [
                    {
                        "away_team": "AAA",
                        "home_team": "BBB",
                        "model_spread": 4.0,
                        "market_spread": 2.5,
                        "spread_gap": 1.5,
                        "cover_prob": 0.55,
                        "model_total": 46.0,
                        "market_total": None,
                        "total_gap": None,
                        "over_prob": None,
                        "edge_flag": 1,
                    }
                ],
                "market": {
                    "source": "nflverse",
                    "observed_at": "2026-09-01T12:00:00+00:00",
                    "stale": False,
                },
            }
        ),
    }


def initialize_actions():
    return [{"type": "fire", "target": "document", "event": "DOMContentLoaded", "wait": True}]


def test_dashboard_initializes_selectors_and_renders_safe_game_values():
    """Catch dashboard code that fails to populate controls or format returned slate values."""
    state = dashboard_state(client(), standard_responses(), initialize_actions())

    assert state["season"]["value"] == "2025"
    assert state["week"]["value"] == "3"
    assert state["estimator"]["value"] == "ridge"
    assert state["edge"] == "2"
    assert state["rows"][1] == {
        "className": "edge",
        "cells": [
            "AAA @ BBB",
            "+4.0",
            "+2.5",
            "+1.5",
            "55.0%",
            "46.0",
            "\N{EM DASH}",
            "\N{EM DASH}",
            "\N{EM DASH}",
            "*",
        ],
    }
    assert state["marketMessage"] == "Lines updated 2026-09-01T12:00:00+00:00"


def test_dashboard_warns_when_market_data_is_stale():
    """Catch packaged fallback lines being presented as current market data."""
    responses = standard_responses()
    slate_url = "/api/slate?season=2025&week=3&estimator=ridge&edge_threshold=2"
    responses[slate_url]["body"]["market"] = {
        "source": "packaged",
        "observed_at": "2026-09-01T12:00:00+00:00",
        "stale": True,
    }

    state = dashboard_state(client(), responses, initialize_actions())

    assert "stale" in state["marketMessage"].lower()
    assert "2026-09-01T12:00:00+00:00" in state["marketMessage"]


def test_dashboard_loading_message_is_ascii_safe():
    """Catch a loading state whose visible text depends on broken page encoding."""
    responses = standard_responses()
    responses["/api/slate?season=2025&week=3&estimator=ridge&edge_threshold=2"] = {"deferred": True}

    state = dashboard_state(
        client(),
        responses,
        [
            {"type": "fire", "target": "document", "event": "DOMContentLoaded", "wait": False},
            {"type": "settle"},
        ],
    )

    assert state["message"] == "Loading..."


@pytest.mark.parametrize(
    "weeks_response",
    [
        response(status=503, body={"error": "upstream details should not be displayed"}),
        {"network": True},
    ],
)
def test_dashboard_week_change_error_clears_stale_slate(weeks_response):
    """Catch a failed season change that leaves stale results or an unhandled rejection."""
    responses = standard_responses()
    responses["/api/weeks?season=2024"] = weeks_response
    actions = initialize_actions() + [
        {"type": "set", "target": "season", "value": "2024"},
        {"type": "fire", "target": "season", "event": "change", "wait": False},
        {"type": "settle"},
    ]

    state = dashboard_state(client(), responses, actions)

    assert state["rows"] == []
    assert state["message"] == "Unable to load weeks."
    assert state["unhandled"] == []


def test_dashboard_ignores_late_week_response_after_rapid_season_changes():
    """Catch an older week response overwriting the newest season selection."""
    responses = standard_responses()
    responses["/api/weeks?season=2024"] = {"deferred": True}
    responses["/api/weeks?season=2025"] = {"deferred": True}
    actions = initialize_actions() + [
        {"type": "set", "target": "season", "value": "2024"},
        {"type": "fire", "target": "season", "event": "change", "wait": False},
        {"type": "set", "target": "season", "value": "2025"},
        {"type": "fire", "target": "season", "event": "change", "wait": False},
        {"type": "settle"},
        {
            "type": "resolve",
            "url": "/api/weeks?season=2025",
            "response": response(body={"weeks": [1, 3]}),
        },
        {"type": "settle"},
        {
            "type": "resolve",
            "url": "/api/weeks?season=2024",
            "response": response(body={"weeks": [2]}),
        },
        {"type": "settle"},
    ]

    state = dashboard_state(client(), responses, actions)

    assert state["week"] == {
        "value": "3",
        "options": [{"value": "1", "selected": False}, {"value": "3", "selected": True}],
    }


@pytest.mark.parametrize(
    "old_response",
    [
        response(body={"games": [dashboard_game("OLD", "TEAM")]}),
        response(status=503, body={"error": "old request failed"}),
    ],
    ids=["late-success", "late-error"],
)
def test_dashboard_ignores_out_of_order_slate_completion(old_response):
    """Catch a late slate success or error taking ownership from the current selection."""
    old_url = "/api/slate?season=2025&week=1&estimator=ridge&edge_threshold=2"
    current_url = "/api/slate?season=2025&week=1&estimator=gbm&edge_threshold=2"
    responses = standard_responses()
    responses[old_url] = {"deferred": True}
    responses[current_url] = {"deferred": True}
    actions = initialize_actions() + [
        {"type": "set", "target": "week", "value": "1"},
        {"type": "fire", "target": "week", "event": "change", "wait": True},
        {"type": "fire", "target": "run", "event": "click", "wait": False},
        {"type": "set", "target": "estimator", "value": "gbm"},
        {"type": "fire", "target": "estimator", "event": "change", "wait": True},
        {"type": "fire", "target": "run", "event": "click", "wait": False},
        {"type": "settle"},
        {
            "type": "resolve",
            "url": current_url,
            "response": response(body={"games": [dashboard_game("NEW", "TEAM")]}),
        },
        {"type": "settle"},
        {"type": "resolve", "url": old_url, "response": old_response},
        {"type": "settle"},
        {"type": "fire", "target": "download", "event": "click", "wait": True},
    ]

    state = dashboard_state(client(), responses, actions)

    assert state["rows"][1]["cells"][0] == "NEW @ TEAM"
    assert state["message"] == "1 games"
    assert state["runDisabled"] is False
    assert state["downloadDisabled"] is False
    assert state["location"] == f"/api/slate.csv?{current_url.split('?', 1)[1]}"
    assert state["unhandled"] == []


def test_stale_slate_finally_does_not_enable_actions_for_a_pending_current_request():
    """Catch an old request completion re-enabling actions while the current slate is pending."""
    old_url = "/api/slate?season=2025&week=1&estimator=ridge&edge_threshold=2"
    current_url = "/api/slate?season=2025&week=1&estimator=gbm&edge_threshold=2"
    responses = standard_responses()
    responses[old_url] = {"deferred": True}
    responses[current_url] = {"deferred": True}
    actions = initialize_actions() + [
        {"type": "set", "target": "week", "value": "1"},
        {"type": "fire", "target": "week", "event": "change", "wait": True},
        {"type": "fire", "target": "run", "event": "click", "wait": False},
        {"type": "set", "target": "estimator", "value": "gbm"},
        {"type": "fire", "target": "estimator", "event": "change", "wait": True},
        {"type": "fire", "target": "run", "event": "click", "wait": False},
        {"type": "settle"},
        {
            "type": "resolve",
            "url": old_url,
            "response": response(body={"games": [dashboard_game("OLD", "TEAM")]}),
        },
        {"type": "settle"},
    ]

    state = dashboard_state(client(), responses, actions)

    assert state["rows"] == []
    assert state["message"] == "Loading..."
    assert state["runDisabled"] is True
    assert state["downloadDisabled"] is True


@pytest.mark.parametrize(
    ("target", "value", "extra_responses", "expected_run_disabled"),
    [
        ("season", "2024", {"/api/weeks?season=2024": {"deferred": True}}, True),
        ("week", "1", {}, False),
        ("estimator", "gbm", {}, False),
        ("edge", "2.5", {}, False),
    ],
)
def test_selector_change_invalidates_rendered_slate_and_csv(
    target, value, extra_responses, expected_run_disabled
):
    """Catch any selector leaving stale rows or a stale CSV action available."""
    responses = standard_responses()
    responses.update(extra_responses)
    actions = initialize_actions() + [
        {"type": "set", "target": target, "value": value},
        {"type": "fire", "target": target, "event": "change", "wait": False},
        {"type": "settle"},
        {"type": "fire", "target": "download", "event": "click", "wait": True},
    ]

    state = dashboard_state(client(), responses, actions)

    assert state["rows"] == []
    assert state["message"] == ""
    assert state["runDisabled"] is expected_run_disabled
    assert state["downloadDisabled"] is True
    assert state["location"] == ""


def test_dashboard_download_encodes_the_successfully_rendered_query():
    """Catch CSV navigation that drops, mis-encodes, or bypasses rendered slate ownership."""
    query = "season=2025&week=3&estimator=ridge+%26+test&edge_threshold=2.5"
    responses = standard_responses()
    responses[f"/api/slate?{query}"] = response(body={"games": []})
    actions = initialize_actions() + [
        {"type": "set", "target": "estimator", "value": "ridge & test"},
        {"type": "fire", "target": "estimator", "event": "change", "wait": True},
        {"type": "set", "target": "edge", "value": "2.5"},
        {"type": "fire", "target": "edge", "event": "change", "wait": True},
        {"type": "fire", "target": "run", "event": "click", "wait": True},
        {"type": "fire", "target": "download", "event": "click", "wait": True},
    ]

    state = dashboard_state(client(), responses, actions)

    assert state["downloadDisabled"] is False
    assert state["location"] == f"/api/slate.csv?{query}"


def test_options_and_weeks_routes_forward_arguments_and_errors():
    """Catch options/weeks routes that skip the service or leak unsafe failures."""
    service = FakeService()
    http_client = client(service)

    assert http_client.get("/api/options").json()["latest"] == {"season": 2025, "week": 3}
    assert http_client.get("/api/weeks", params={"season": 2025}).json() == {"weeks": [1, 3]}
    invalid = http_client.get("/api/weeks", params={"season": 1999})

    assert service.calls[:3] == [("options",), ("weeks", 2025), ("weeks", 1999)]
    assert invalid.status_code == 422
    assert invalid.json() == {"error": "season 1999 is not available"}


def test_tracker_routes_forward_exact_selections_and_link_from_slate():
    """Catch missing tracker navigation or routes that coerce or reorder service selections."""
    tracker = FakeTrackerService()
    http_client = client(tracker_service=tracker)

    slate_page = http_client.get("/")
    tracker_page = http_client.get("/tracker")
    options = http_client.get("/api/tracker/options")
    summary = http_client.get(
        "/api/tracker/summary", params={"record_type": "backtest", "season": "all"}
    )
    games = http_client.get(
        "/api/tracker/games", params={"record_type": "backtest", "season": 2025}
    )
    live_games = http_client.get(
        "/api/tracker/games", params={"record_type": "live", "season": 2026}
    )

    assert '<a href="/tracker">Performance tracker</a>' in slate_page.text
    assert tracker_page.status_code == 200
    assert "NFL Performance Tracker" in tracker_page.text
    assert options.json()["model_version"] == "ridge-v1"
    assert summary.json() == {
        "available": True,
        "record_type": "backtest",
        "season": "all",
    }
    assert games.json() == {"games": [{"game_id": "2025_01_AAA_BBB"}]}
    assert live_games.json() == {"games": [{"game_id": "2026_01_AAA_BBB"}]}
    assert tracker.calls == [
        ("options",),
        ("summary", "backtest", "all"),
        ("records", "backtest", 2025),
        ("records", "live", 2026),
    ]


def test_schedule_page_and_api_are_protected_and_use_service():
    """Catch an unprotected or disconnected 2026 schedule page/API pair."""
    service = FakeService()
    http_client = client(service)

    assert http_client.get("/schedule").status_code == 200
    body = http_client.get("/api/schedule", params={"season": 2026}).json()

    assert body["season"] == 2026
    assert body["games"][0]["game_id"].startswith("2026_")
    assert body["market"]["source"] in {"nflverse", "packaged"}
    assert service.calls == [("schedule_records", 2026)]

    protected = client(access_code="letmein")
    page = protected.get("/schedule")
    api = protected.get("/api/schedule", params={"season": 2026})

    assert page.status_code == 303
    assert page.headers["location"] == "/login"
    assert api.status_code == 401
    assert api.json() == {"error": "session expired"}


def test_tracker_input_errors_are_client_safe_422s():
    """Catch tracker selection errors becoming a 500 or exposing framework internals."""
    response = client().get(
        "/api/tracker/summary", params={"record_type": "research", "season": "all"}
    )

    assert response.status_code == 422
    assert response.json() == {"error": "invalid record type"}
    assert "traceback" not in response.text.lower()


def test_auth_protects_tracker_page_and_api():
    """Catch tracker endpoints being accidentally added to the public auth allowlist."""
    http_client = client(access_code="letmein")

    page = http_client.get("/tracker")
    api = http_client.get("/api/tracker/options")

    assert page.status_code == 303
    assert page.headers["location"] == "/login"
    assert api.status_code == 401
    assert api.json() == {"error": "session expired"}


def test_slate_defaults_and_csv_forward_matching_arguments_and_content():
    """Catch slate routes that use different defaults, arguments, or CSV payloads."""
    service = FakeService()
    http_client = client(service)

    slate = http_client.get("/api/slate", params={"season": 2025, "week": 1})
    csv_response = http_client.get("/api/slate.csv", params={"season": 2025, "week": 1})

    assert slate.status_code == 200
    assert csv_response.status_code == 200
    assert set(slate.json()) == {"games", "market"}
    assert slate.json()["market"] == {
        "source": "nflverse",
        "observed_at": "2026-09-01T12:00:00+00:00",
        "stale": False,
    }
    assert service.calls == [
        ("payload", 2025, 1, "ridge", 2.0),
        ("records", 2025, 1, "ridge", 2.0),
        ("csv", 2025, 1, "ridge", 2.0),
        ("records", 2025, 1, "ridge", 2.0),
    ]
    csv_game = next(csv.DictReader(io.StringIO(csv_response.text)))
    json_game = slate.json()["games"][0]
    assert csv_game == {
        key: "" if value is None else str(value) for key, value in json_game.items()
    }
    assert csv_game["market_total"] == ""
    assert csv_response.headers["content-type"].startswith("text/csv")
    assert csv_response.headers["content-disposition"] == 'attachment; filename="slate_2025_wk01_ridge.csv"'


@pytest.mark.parametrize("path", ["/api/slate", "/api/slate.csv"])
@pytest.mark.parametrize(
    ("params", "expected_error"),
    [
        ({"estimator": "unknown"}, "estimator 'unknown' is not available"),
        ({"edge_threshold": "-0.5"}, "edge threshold must be a finite non-negative number"),
        ({"edge_threshold": "nan"}, "edge threshold must be a finite non-negative number"),
        ({"edge_threshold": "inf"}, "edge threshold must be a finite non-negative number"),
    ],
)
def test_slate_routes_map_service_input_errors_to_safe_422s(path, params, expected_error):
    """Catch JSON or CSV validation failures that use the wrong status or expose internals."""
    base = {"season": 2025, "week": 1}
    base.update(params)

    response = client().get(path, params=base)

    assert response.status_code == 422
    assert response.json() == {"error": expected_error}
    assert "traceback" not in response.text.lower()


@pytest.mark.parametrize("path", ["/api/slate", "/api/slate.csv"])
def test_malformed_threshold_is_a_generic_client_safe_422(path):
    """Catch framework validation details leaking through malformed query parameters."""
    response = client().get(path, params={"season": 2025, "week": 1, "edge_threshold": "nope"})

    assert response.status_code == 422
    assert response.json() == {"error": "Invalid request"}


@pytest.mark.parametrize(
    ("path", "params", "status"),
    [
        ("/api/slate", {"season": 2025, "week": 3, "estimator": "gbm"}, 409),
        ("/api/slate.csv", {"season": 2025, "week": 3, "estimator": "gbm"}, 409),
        ("/api/slate", {"season": 2025, "week": 4}, 404),
        ("/api/slate.csv", {"season": 2025, "week": 4}, 404),
    ],
)
def test_slate_and_csv_share_unavailable_and_empty_error_mappings(path, params, status):
    """Catch CSV errors diverging from the corresponding JSON slate errors."""
    response = client().get(path, params=params)

    assert response.status_code == status
    assert "traceback" not in response.text.lower()


def test_unexpected_error_is_generic_and_hides_internal_message():
    """Catch unexpected route errors that expose exception details in the response."""
    safe_client = TestClient(
        create_app(FakeService(), FakeTrackerService(), access_code=None),
        raise_server_exceptions=False,
    )
    response = safe_client.get(
        "/api/slate",
        params={"season": 2025, "week": 5, "estimator": "ridge", "edge_threshold": 2.0},
    )

    assert response.status_code == 500
    assert response.json() == {"error": "Unexpected server error"}
    assert "database password" not in response.text


def test_factory_integrates_access_code_middleware_login_and_public_routes():
    """Catch an app factory that wires protected and public routes inconsistently."""
    http_client = client(access_code="letmein")

    assert http_client.get("/").status_code == 303
    assert http_client.get("/api/options").json() == {"error": "session expired"}
    assert http_client.get("/health").json() == {"ok": True}
    assert http_client.get("/login").status_code == 200
    assert http_client.post("/login", json={"code": "letmein"}).json() == {"ok": True}
    assert http_client.get("/").status_code == 200
    assert http_client.get("/api/options").status_code == 200
