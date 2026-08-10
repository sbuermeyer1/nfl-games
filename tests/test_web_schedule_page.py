import json
import re

import quickjs
from fastapi.testclient import TestClient
from tests.test_webapp import FakeService, FakeTrackerService

from nfl_game.web.app import create_app

SCRIPT = re.compile(r"<script>(.*?)</script>", re.DOTALL)


def client():
    return TestClient(
        create_app(FakeService(), FakeTrackerService(), access_code=None),
        base_url="https://testserver",
        follow_redirects=False,
    )


def response(status=200, body=None):
    return {"status": status, "body": body if body is not None else {}}


def schedule_game(index, *, spread_line=2.5, total_line=44.5):
    week = index % 18 + 1
    return {
        "game_id": f"2026_{week:02d}_A{index:03d}_H{index:03d}",
        "season": 2026,
        "week": week,
        "kickoff_at": f"2026-09-{week:02d}T17:00:00+00:00",
        "away_team": f"A{index:03d}",
        "home_team": f"H{index:03d}",
        "spread_line": spread_line,
        "total_line": total_line,
    }


def schedule_body(*, stale=False, source="nflverse", games=None):
    return {
        "season": 2026,
        "games": games if games is not None else [schedule_game(0)],
        "market": {
            "source": source,
            "observed_at": "2026-09-01T12:00:00+00:00",
            "stale": stale,
        },
    }


def schedule_state(responses, actions):
    page = client().get("/schedule")
    script = SCRIPT.search(page.text)
    assert script, "schedule page must contain an executable script"
    payload = json.dumps({"script": script.group(1), "responses": responses, "actions": actions})
    context = quickjs.Context()
    context.eval(f"const input = JSON.parse({json.dumps(payload)});")
    context.eval(SCHEDULE_HARNESS)
    while context.execute_pending_job():
        pass
    return json.loads(context.eval("globalThis.__state"))


SCHEDULE_HARNESS = r"""
const listeners = {};
const pending = [];
const calls = [];
const textWrites = [];
const unhandled = [];

class Element {
  constructor(tagName) {
    this.tagName = tagName;
    this.children = [];
    this.listeners = {};
    this._textContent = '';
  }
  appendChild(child) { this.children.push(child); return child; }
  replaceChildren(...children) { this.children = children; this._textContent = ''; }
  addEventListener(name, callback) { (this.listeners[name] ||= []).push(callback); }
  get textContent() { return this._textContent; }
  set textContent(value) {
    this._textContent = String(value);
    this.children = [];
    textWrites.push(this._textContent);
  }
}

const nodes = {
  'schedule-message': new Element('p'),
  'schedule-games': new Element('table'),
};
globalThis.document = {
  getElementById: id => nodes[id],
  createElement: tagName => new Element(tagName),
  addEventListener: (name, callback) => { (listeners[name] ||= []).push(callback); },
};
globalThis.window = { location: '' };

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
  if (spec.deferred) {
    return new Promise((resolve, reject) => pending.push({ url, resolve, reject }));
  }
  return Promise.resolve(makeResponse(spec));
};

async function settle() {
  for (let index = 0; index < 16; index += 1) await Promise.resolve();
}

async function fireDocument(name, wait) {
  const promises = (listeners[name] || []).map(callback =>
    Promise.resolve().then(callback).catch(error => unhandled.push(error.message))
  );
  if (wait) await Promise.all(promises);
}

eval(input.script);

(async () => {
  for (const action of input.actions) {
    if (action.type === 'fire') await fireDocument(action.event, action.wait);
    if (action.type === 'resolve') {
      pending[action.index].resolve(makeResponse(action.response));
    }
    if (action.type === 'settle') await settle();
  }
  await settle();
  const rows = nodes['schedule-games'].children.map(row =>
    row.children.map(cell => cell.textContent)
  );
  globalThis.__state = JSON.stringify({
    calls,
    unhandled,
    textWrites,
    location: window.location,
    message: nodes['schedule-message'].textContent,
    rows,
  });
})().catch(error => {
  globalThis.__state = JSON.stringify({ fatal: error.message, calls, unhandled });
});
"""


def initialize_actions(wait=True):
    return [{"type": "fire", "event": "DOMContentLoaded", "wait": wait}]


def test_schedule_page_links_all_site_sections():
    """Catch the new page becoming a navigation dead end or being unreachable from home."""
    http_client = client()

    home = http_client.get("/").text
    schedule = http_client.get("/schedule").text

    assert '<a href="/schedule">2026 schedule</a>' in home
    assert '<a href="/">Weekly predictions</a>' in schedule
    assert '<a href="/tracker">Performance tracker</a>' in schedule
    assert "2026 NFL Schedule" in schedule


def test_schedule_renders_all_regular_season_games_current_freshness_and_safe_nulls():
    """Catch truncation, unsafe matchup rendering, or missing lines becoming fake numbers."""
    games = [schedule_game(index) for index in range(272)]
    games[17] = schedule_game(17, spread_line=None, total_line=None)
    games[17]["away_team"] = "A<script>alert(1)</script>"
    body = schedule_body(games=games)

    state = schedule_state(
        {"/api/schedule?season=2026": response(body=body)},
        initialize_actions(),
    )

    assert state["calls"] == ["/api/schedule?season=2026"]
    assert len(state["rows"]) == 273
    assert state["rows"][0] == ["Week", "Kickoff", "Matchup", "Spread", "Total"]
    assert state["rows"][18] == [
        "18",
        "2026-09-18T17:00:00+00:00",
        "A<script>alert(1)</script> @ H017",
        "—",
        "—",
    ]
    assert "A<script>alert(1)</script> @ H017" in state["textWrites"]
    assert "Lines updated" in state["message"]
    assert "2026-09-01T12:00:00+00:00" in state["message"]
    assert not any(value in {"0", "NaN", "Infinity"} for value in state["rows"][18][3:])
    assert state["unhandled"] == []


def test_schedule_marks_packaged_market_data_stale():
    """Catch fallback lines being presented with the same confidence as current lines."""
    state = schedule_state(
        {"/api/schedule?season=2026": response(body=schedule_body(stale=True, source="packaged"))},
        initialize_actions(),
    )

    assert "stale" in state["message"].lower()
    assert "2026-09-01T12:00:00+00:00" in state["message"]


def test_schedule_ignores_a_late_older_response():
    """Catch an earlier schedule request overwriting the most recently requested data."""
    url = "/api/schedule?season=2026"
    responses = {url: {"deferred": True}}
    actions = initialize_actions(wait=False) + [
        {"type": "settle"},
        {"type": "fire", "event": "DOMContentLoaded", "wait": False},
        {"type": "settle"},
        {
            "type": "resolve",
            "index": 1,
            "response": response(body=schedule_body(games=[schedule_game(1)])),
        },
        {"type": "settle"},
        {
            "type": "resolve",
            "index": 0,
            "response": response(
                body=schedule_body(
                    stale=True,
                    source="packaged",
                    games=[schedule_game(0)],
                )
            ),
        },
        {"type": "settle"},
    ]

    state = schedule_state(responses, actions)

    assert state["rows"][1][2] == "A001 @ H001"
    assert "Lines updated" in state["message"]
    assert "stale" not in state["message"].lower()
    assert state["unhandled"] == []


def test_schedule_redirects_to_login_when_the_session_expires():
    """Catch an expired session leaving the schedule page in a broken state."""
    state = schedule_state(
        {"/api/schedule?season=2026": response(status=401, body={"error": "session expired"})},
        initialize_actions(),
    )

    assert state["location"] == "/login"
    assert state["unhandled"] == []
