import json
import re

import quickjs

from nfl_game.web.tracker_page import TRACKER_PAGE

SCRIPT = re.compile(r"<script>(.*?)</script>", re.DOTALL)
TRACKER_IDS = (
    "historical-tab",
    "live-tab",
    "tracker-season",
    "tracker-message",
    "qualified-cards",
    "all-records",
    "spread-edges",
    "season-breakdown",
    "audit-games",
    "closing-line",
)


def response(status=200, body=None):
    return {"status": status, "body": body if body is not None else {}}


def record(wins, losses, pushes, win_rate):
    return {
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "n_graded": wins + losses,
        "win_rate": win_rate,
    }


def core_summary(label, *, closing_line=None):
    offset = {"overall": 0, "2024": 10, "2025": 20, "live": 30}[label]
    return {
        "qualified": {
            "spread": record(offset + 6, 4, 1, (offset + 6) / (offset + 10)),
            "total": record(offset + 7, 3, 2, (offset + 7) / (offset + 10)),
        },
        "all_predictions": {
            "spread": record(offset + 8, 6, 1, (offset + 8) / (offset + 14)),
            "total": record(offset + 9, 5, 0, (offset + 9) / (offset + 14)),
        },
        "spread_edges": [
            {"min_edge": 5.0, "record": record(offset + 5, 3, 1, (offset + 5) / (offset + 8))},
            {"min_edge": 10.0, "record": record(offset + 2, 2, 0, (offset + 2) / (offset + 4))},
            {"min_edge": 15.0, "record": record(offset + 1, 1, 0, (offset + 1) / (offset + 2))},
        ],
        "closing_line": closing_line,
    }


def overall_summary():
    return {
        "available": True,
        "record_type": "backtest",
        "season": "all",
        **core_summary("overall"),
        "by_season": [
            {"season": 2024, **core_summary("2024")},
            {"season": 2025, **core_summary("2025")},
        ],
    }


def season_summary(season):
    return {
        "available": True,
        "record_type": "backtest",
        "season": season,
        **core_summary(str(season)),
    }


def live_summary():
    closing = {
        "spread": {
            "average_clv": 1.25,
            "beat_close_rate": 0.75,
            "n_clv": 8,
            "record": record(5, 2, 1, 5 / 7),
        },
        "total": {
            "average_clv": None,
            "beat_close_rate": None,
            "n_clv": 0,
            "record": record(0, 0, 0, None),
        },
    }
    return {
        "available": True,
        "record_type": "live",
        "season": "all",
        **core_summary("live", closing_line=closing),
        "by_season": [{"season": 2026, **core_summary("live", closing_line=closing)}],
    }


def options():
    return {
        "record_types": ["backtest", "live"],
        "historical_seasons": [2024, 2025],
        "default_record_type": "backtest",
        "default_season": "all",
        "model_version": "ridge-v1",
        "qualified_edge": 2.0,
        "spread_edge_thresholds": [5.0, 10.0, 15.0],
        "live_available": False,
    }


def audit_game(season, away_team, home_team):
    return {
        "game_id": f"{season}_01_{away_team}_{home_team}",
        "season": season,
        "week": 1,
        "away_team": away_team,
        "home_team": home_team,
        "model_margin": 7.5,
        "official_spread_line": 3.0,
        "spread_pick": "home",
        "spread_edge": 4.5,
        "actual_margin": 8.0,
        "spread_grade": "win",
        "model_total": 47.5,
        "official_total_line": 44.0,
        "total_pick": "over",
        "total_edge": 3.5,
        "actual_total": 50.0,
        "total_grade": "win",
    }


def standard_responses():
    return {
        "/api/tracker/options": response(body=options()),
        "/api/tracker/summary?record_type=backtest&season=all": response(body=overall_summary()),
    }


def tracker_state(responses, actions):
    """Run the tracker script with observable DOM and deferred fetch behavior in QuickJS."""
    script = SCRIPT.search(TRACKER_PAGE)
    assert script, "tracker page must contain an executable script"
    payload = json.dumps(
        {
            "script": script.group(1),
            "responses": responses,
            "actions": actions,
            "ids": TRACKER_IDS,
        }
    )
    context = quickjs.Context()
    context.eval(f"const input = JSON.parse({json.dumps(payload)});")
    context.eval(TRACKER_HARNESS)
    while context.execute_pending_job():
        pass
    return json.loads(context.eval("globalThis.__state"))


TRACKER_HARNESS = r"""
const documentListeners = {};
const pending = {};
const calls = [];
const textWrites = [];
const unhandled = [];

class Element {
  constructor(tagName, id = '') {
    this.tagName = tagName;
    this.id = id;
    this.children = [];
    this.listeners = {};
    this.className = '';
    this.disabled = false;
    this.hidden = false;
    this.selected = false;
    this._textContent = '';
    this._value = undefined;
  }
  appendChild(child) { this.children.push(child); return child; }
  replaceChildren(...children) { this.children = children; this._textContent = ''; }
  addEventListener(name, callback) { (this.listeners[name] ||= []).push(callback); }
  setAttribute(name, value) { this[name] = String(value); }
  get textContent() { return this._textContent; }
  set textContent(value) {
    this._textContent = String(value);
    this.children = [];
    textWrites.push(this._textContent);
  }
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

const nodes = Object.fromEntries(input.ids.map(id => [id, new Element('section', id)]));
nodes['historical-tab'].tagName = 'button';
nodes['live-tab'].tagName = 'button';
nodes['tracker-season'].tagName = 'select';
nodes['tracker-message'].tagName = 'p';
nodes['spread-edges'].tagName = 'table';
nodes['season-breakdown'].tagName = 'table';
nodes['audit-games'].tagName = 'table';

globalThis.document = {
  getElementById: id => nodes[id],
  createElement: tagName => new Element(tagName),
  addEventListener: (name, callback) => { (documentListeners[name] ||= []).push(callback); },
};
globalThis.window = { location: '' };
class URLSearchParams {
  constructor(values) { this.values = values; }
  toString() {
    return Object.entries(this.values)
      .map(([key, value]) => `${encodeURIComponent(key)}=${encodeURIComponent(value)}`)
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
  if (spec.deferred) {
    return new Promise((resolve, reject) => { pending[url] = { resolve, reject }; });
  }
  return Promise.resolve(makeResponse(spec));
};

async function settle() {
  for (let index = 0; index < 16; index += 1) await Promise.resolve();
}

function trigger(target, name, wait) {
  const callbacks = target === 'document'
    ? documentListeners[name] || []
    : nodes[target].listeners[name] || [];
  const promises = callbacks.map(callback => Promise.resolve().then(callback).catch(error => {
    unhandled.push(error.message);
  }));
  return wait ? Promise.all(promises) : Promise.resolve();
}

function nodeText(node) {
  return [node._textContent, ...node.children.map(nodeText)].filter(Boolean).join(' | ');
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
  const regions = Object.fromEntries(input.ids.map(id => [id, nodeText(nodes[id])]));
  globalThis.__state = JSON.stringify({
    calls,
    textWrites,
    unhandled,
    location: window.location,
    regions,
    season: {
      value: nodes['tracker-season'].value,
      options: nodes['tracker-season'].children.map(option => ({
        value: option.value,
        text: option.textContent,
        selected: option.selected,
      })),
    },
  });
})().catch(error => {
  globalThis.__state = JSON.stringify({ fatal: error.message, calls, unhandled });
});
"""


def initialize_actions():
    return [{"type": "fire", "target": "document", "event": "DOMContentLoaded", "wait": True}]


def test_tracker_page_has_required_navigation_controls_and_regions():
    """Catch page markup that drops a required control, disclosure, or rendering region."""
    assert 'href="/"' in TRACKER_PAGE
    for element_id in TRACKER_IDS:
        assert f'id="{element_id}"' in TRACKER_PAGE
    assert "52.4%" in TRACKER_PAGE
    assert "not betting advice" in TRACKER_PAGE


def test_tracker_script_uses_text_content_for_server_values():
    """Catch unsafe HTML insertion that could execute team names or other ledger values."""
    script = SCRIPT.search(TRACKER_PAGE)
    assert script is not None
    assert ".innerHTML" not in script.group(1)
    assert ".textContent" in script.group(1)


def test_tracker_script_declares_separate_stale_request_tokens():
    """Catch summary and audit fetches sharing a token that cannot invalidate independently."""
    script = SCRIPT.search(TRACKER_PAGE)
    assert script is not None
    source = script.group(1)
    assert "latestSummaryRequest" in source
    assert "latestGamesRequest" in source
    assert "request !== latestSummaryRequest" in source
    assert "request !== latestGamesRequest" in source


def test_overall_historical_renders_records_thresholds_and_seasons():
    """Catch the overall branch omitting qualified, cumulative, or per-season evidence."""
    state = tracker_state(standard_responses(), initialize_actions())

    assert state["season"]["value"] == "all"
    assert [option["text"] for option in state["season"]["options"]] == [
        "Overall (2024-2025)",
        "2024",
        "2025",
    ]
    assert "6-4-1 · 60.0% · n=10" in state["regions"]["qualified-cards"]
    assert "8-6-1 · 57.1% · n=14" in state["regions"]["all-records"]
    assert all(threshold in state["regions"]["spread-edges"] for threshold in ("5+", "10+", "15+"))
    assert "2024" in state["regions"]["season-breakdown"]
    assert "2025" in state["regions"]["season-breakdown"]
    assert not any(url.startswith("/api/tracker/games") for url in state["calls"])
    assert state["unhandled"] == []


def test_concrete_season_requests_and_safely_renders_audit_rows():
    """Catch season selection skipping its audit request or treating matchup text as markup."""
    responses = standard_responses()
    summary_url = "/api/tracker/summary?record_type=backtest&season=2024"
    games_url = "/api/tracker/games?record_type=backtest&season=2024"
    matchup = "A<script>alert(1)</script> @ H&OME"
    responses[summary_url] = response(body=season_summary(2024))
    responses[games_url] = response(
        body={"games": [audit_game(2024, "A<script>alert(1)</script>", "H&OME")]}
    )
    actions = initialize_actions() + [
        {"type": "set", "target": "tracker-season", "value": "2024"},
        {"type": "fire", "target": "tracker-season", "event": "change", "wait": True},
    ]

    state = tracker_state(responses, actions)

    assert summary_url in state["calls"]
    assert games_url in state["calls"]
    assert matchup in state["regions"]["audit-games"]
    assert matchup in state["textWrites"]
    assert "7.5" in state["regions"]["audit-games"]
    assert "win" in state["regions"]["audit-games"]


def test_live_unavailable_shows_only_the_service_message():
    """Catch unavailable live data being misrepresented as empty zero-percent performance."""
    responses = standard_responses()
    responses["/api/tracker/summary?record_type=live&season=all"] = response(
        body={
            "available": False,
            "record_type": "live",
            "message": "Live tracking begins with the 2026 season.",
        }
    )
    actions = initialize_actions() + [
        {"type": "fire", "target": "live-tab", "event": "click", "wait": True}
    ]

    state = tracker_state(responses, actions)

    assert state["regions"]["tracker-message"] == "Live tracking begins with the 2026 season."
    for region in (
        "qualified-cards",
        "all-records",
        "spread-edges",
        "season-breakdown",
        "audit-games",
        "closing-line",
    ):
        assert state["regions"][region] == ""
    assert "0.0%" not in " ".join(state["regions"].values())


def test_future_live_summary_renders_official_and_closing_line_metrics():
    """Catch populated live data omitting secondary CLV metrics or labeling nulls as zero."""
    responses = standard_responses()
    responses["/api/tracker/summary?record_type=live&season=all"] = response(body=live_summary())
    actions = initialize_actions() + [
        {"type": "fire", "target": "live-tab", "event": "click", "wait": True}
    ]

    state = tracker_state(responses, actions)

    assert "36-4-1 · 90.0% · n=40" in state["regions"]["qualified-cards"]
    closing = state["regions"]["closing-line"]
    assert "1.3" in closing
    assert "75.0%" in closing
    assert "n=8" in closing
    assert "5-2-1 · 71.4% · n=7" in closing
    assert "n/a" in closing
    assert state["regions"]["audit-games"] == ""


def test_out_of_order_season_responses_cannot_replace_current_selection():
    """Catch late summary or audit completions overwriting the newer season's DOM."""
    responses = standard_responses()
    old_summary = "/api/tracker/summary?record_type=backtest&season=2024"
    old_games = "/api/tracker/games?record_type=backtest&season=2024"
    new_summary = "/api/tracker/summary?record_type=backtest&season=2025"
    new_games = "/api/tracker/games?record_type=backtest&season=2025"
    responses.update(
        {
            old_summary: {"deferred": True},
            old_games: {"deferred": True},
            new_summary: {"deferred": True},
            new_games: {"deferred": True},
        }
    )
    actions = initialize_actions() + [
        {"type": "set", "target": "tracker-season", "value": "2024"},
        {"type": "fire", "target": "tracker-season", "event": "change", "wait": False},
        {"type": "settle"},
        {"type": "set", "target": "tracker-season", "value": "2025"},
        {"type": "fire", "target": "tracker-season", "event": "change", "wait": False},
        {"type": "settle"},
        {"type": "resolve", "url": new_summary, "response": response(body=season_summary(2025))},
        {
            "type": "resolve",
            "url": new_games,
            "response": response(body={"games": [audit_game(2025, "NEW", "TEAM")]}),
        },
        {"type": "settle"},
        {"type": "resolve", "url": old_summary, "response": response(body=season_summary(2024))},
        {
            "type": "resolve",
            "url": old_games,
            "response": response(body={"games": [audit_game(2024, "OLD", "TEAM")]}),
        },
        {"type": "settle"},
    ]

    state = tracker_state(responses, actions)

    assert "26-4-1 · 86.7% · n=30" in state["regions"]["qualified-cards"]
    assert "NEW @ TEAM" in state["regions"]["audit-games"]
    assert "OLD @ TEAM" not in state["regions"]["audit-games"]
    assert state["season"]["value"] == "2025"
    assert state["unhandled"] == []


def test_unauthorized_response_redirects_to_login():
    """Catch expired authenticated sessions leaving the tracker in a broken page state."""
    responses = {"/api/tracker/options": response(status=401, body={"error": "session expired"})}

    state = tracker_state(responses, initialize_actions())

    assert state["location"] == "/login"
    assert state["calls"] == ["/api/tracker/options"]
    assert state["unhandled"] == []
