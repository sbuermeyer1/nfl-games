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
        self.records(season, week, estimator, edge_threshold)
        return f"game_id,season,week\n2025_01_AAA_BBB,{season},{week}\n"


def client(service=None, access_code=None):
    return TestClient(
        create_app(service or FakeService(), access_code=access_code),
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
  ['season', 'week', 'estimator', 'edge', 'run', 'download', 'message', 'results']
    .map(id => [id, new Element(id === 'edge' ? 'input' : 'select')])
);
nodes.run.tagName = 'button';
nodes.download.tagName = 'button';
nodes.message.tagName = 'p';
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
    rows,
    season: { value: nodes.season.value, options: options('season') },
    week: { value: nodes.week.value, options: options('week') },
    estimator: { value: nodes.estimator.value, options: options('estimator') },
    edge: nodes.edge.value,
  });
})().catch(error => { globalThis.__state = JSON.stringify({error: error.stack}); });
"""


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
                ]
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
            "n/a",
            "n/a",
            "n/a",
            "*",
        ],
    }


def test_dashboard_loading_message_is_ascii_safe():
    """Catch a loading state whose visible text depends on broken page encoding."""
    responses = standard_responses()
    responses["/api/slate?season=2025&week=3&estimator=ridge&edge_threshold=2"] = {"deferred": True}

    state = dashboard_state(
        client(),
        responses,
        [{"type": "fire", "target": "document", "event": "DOMContentLoaded", "wait": False}, {"type": "settle"}],
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
        {"type": "resolve", "url": "/api/weeks?season=2025", "response": response(body={"weeks": [1, 3]})},
        {"type": "settle"},
        {"type": "resolve", "url": "/api/weeks?season=2024", "response": response(body={"weeks": [2]})},
        {"type": "settle"},
    ]

    state = dashboard_state(client(), responses, actions)

    assert state["week"] == {
        "value": "3",
        "options": [{"value": "1", "selected": False}, {"value": "3", "selected": True}],
    }


def test_dashboard_download_encodes_the_active_query():
    """Catch CSV navigation that drops or mis-encodes active selector values."""
    responses = standard_responses()
    actions = initialize_actions() + [
        {"type": "set", "target": "season", "value": "2025"},
        {"type": "set", "target": "week", "value": "3"},
        {"type": "set", "target": "estimator", "value": "ridge & test"},
        {"type": "set", "target": "edge", "value": "2.5"},
        {"type": "fire", "target": "download", "event": "click", "wait": True},
    ]

    state = dashboard_state(client(), responses, actions)

    assert state["location"] == "/api/slate.csv?season=2025&week=3&estimator=ridge+%26+test&edge_threshold=2.5"


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


def test_slate_defaults_and_csv_forward_matching_arguments_and_content():
    """Catch slate routes that use different defaults, arguments, or CSV payloads."""
    service = FakeService()
    http_client = client(service)

    slate = http_client.get("/api/slate", params={"season": 2025, "week": 1})
    csv = http_client.get("/api/slate.csv", params={"season": 2025, "week": 1})

    assert slate.status_code == 200
    assert csv.status_code == 200
    assert service.calls == [
        ("records", 2025, 1, "ridge", 2.0),
        ("csv", 2025, 1, "ridge", 2.0),
        ("records", 2025, 1, "ridge", 2.0),
    ]
    assert csv.text == "game_id,season,week\n2025_01_AAA_BBB,2025,1\n"
    assert csv.headers["content-type"].startswith("text/csv")
    assert csv.headers["content-disposition"] == 'attachment; filename="slate_2025_wk01_ridge.csv"'


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
        create_app(FakeService(), access_code=None),
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
