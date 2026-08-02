"""FastAPI application factory and embedded NFL slate dashboard."""

import logging

from fastapi import FastAPI, Query
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse, Response

from nfl_game.web.auth import AccessCodeMiddleware
from nfl_game.web.login import add_login_routes
from nfl_game.web.service import (
    DEFAULT_EDGE_THRESHOLD,
    SlateInputError,
    SlateNotFoundError,
    SlateService,
    SlateUnavailableError,
)
from nfl_game.web.session import SessionStore
from nfl_game.web.tracker_page import TRACKER_PAGE
from nfl_game.web.tracker_service import TrackerInputError, TrackerService

logger = logging.getLogger(__name__)


PAGE = """<!doctype html>
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NFL Game Model</title>
<style>
  body { font-family: system-ui, sans-serif; margin: 1rem; color: #191919; }
  main { max-width: 72rem; margin: auto; }
  .controls { display: flex; flex-wrap: wrap; gap: .75rem; align-items: end; }
  label { display: grid; gap: .25rem; }
  select, input, button { font: inherit; padding: .45rem; }
  button { cursor: pointer; }
  #message { min-height: 1.4rem; color: #a00; }
  .table-wrap { overflow-x: auto; }
  table { border-collapse: collapse; width: 100%; white-space: nowrap; }
  th, td { border-bottom: 1px solid #ddd; padding: .5rem; text-align: right; }
  th:first-child, td:first-child { text-align: left; }
  .edge { font-weight: 700; color: #087443; }
  .note { color: #555; font-size: .9rem; }
</style>
<main>
  <nav aria-label="Site navigation"><a href="/tracker">Performance tracker</a></nav>
  <h1>NFL Game Model</h1>
  <p>Weekly model-versus-market margins and totals.</p>
  <div class="controls">
    <label>Season <select id="season"></select></label>
    <label>Week <select id="week"></select></label>
    <label>Estimator <select id="estimator"></select></label>
    <label>Edge threshold <input id="edge" type="number" min="0" step="0.5"></label>
    <button id="run" type="button">Run slate</button>
    <button id="download" type="button">Download CSV</button>
  </div>
  <p id="message" role="status"></p>
  <div class="table-wrap"><table id="results"></table></div>
  <p class="note">Spreads are home-team margins. An edge flag shows model/market
  disagreement and is not betting advice.</p>
</main>
<script>
const season = document.getElementById('season');
const week = document.getElementById('week');
const estimator = document.getElementById('estimator');
const edge = document.getElementById('edge');
const runButton = document.getElementById('run');
const downloadButton = document.getElementById('download');
const message = document.getElementById('message');
const results = document.getElementById('results');
let latestWeekRequest = 0;
let latestSlateRequest = 0;
let renderedSlateQuery = null;
runButton.disabled = true;
downloadButton.disabled = true;

function replaceOptions(select, values, selected) {
  select.replaceChildren();
  for (const value of values) {
    const option = document.createElement('option');
    option.value = String(value);
    option.textContent = String(value);
    option.selected = value === selected;
    select.appendChild(option);
  }
}

function queryString() {
  return new URLSearchParams({
    season: season.value,
    week: week.value,
    estimator: estimator.value,
    edge_threshold: edge.value,
  }).toString();
}

async function jsonOrError(url) {
  const response = await fetch(url);
  if (response.status === 401) {
    window.location = '/login';
    throw new Error('Session expired');
  }
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.error || `Request failed (${response.status})`);
  return body;
}

function formatted(value, kind) {
  if (value === null || value === undefined) return 'n/a';
  if (kind === 'probability') return `${(Number(value) * 100).toFixed(1)}%`;
  if (kind === 'signed') {
    const number = Number(value);
    return `${number >= 0 ? '+' : ''}${number.toFixed(1)}`;
  }
  return Number(value).toFixed(1);
}

function renderGames(games) {
  const columns = [
    ['Game', game => `${game.away_team} @ ${game.home_team}`],
    ['Model', game => formatted(game.model_spread, 'signed')],
    ['Market', game => formatted(game.market_spread, 'signed')],
    ['Gap', game => formatted(game.spread_gap, 'signed')],
    ['Cover%', game => formatted(game.cover_prob, 'probability')],
    ['Model O/U', game => formatted(game.model_total, 'number')],
    ['Market O/U', game => formatted(game.market_total, 'number')],
    ['Gap', game => formatted(game.total_gap, 'signed')],
    ['Over%', game => formatted(game.over_prob, 'probability')],
    ['Edge', game => game.edge_flag === 1 ? '*' : ''],
  ];
  results.replaceChildren();
  const header = document.createElement('tr');
  for (const [label] of columns) {
    const th = document.createElement('th');
    th.textContent = label;
    header.appendChild(th);
  }
  results.appendChild(header);
  for (const game of games) {
    const row = document.createElement('tr');
    for (const [, value] of columns) {
      const cell = document.createElement('td');
      cell.textContent = value(game);
      row.appendChild(cell);
    }
    if (game.edge_flag === 1) row.className = 'edge';
    results.appendChild(row);
  }
}

function invalidateSlate(runAvailable = true) {
  latestSlateRequest += 1;
  renderedSlateQuery = null;
  results.replaceChildren();
  message.textContent = '';
  downloadButton.disabled = true;
  runButton.disabled = !runAvailable;
}

async function loadWeeks(runAfter = false) {
  const request = ++latestWeekRequest;
  try {
    const body = await jsonOrError(`/api/weeks?season=${encodeURIComponent(season.value)}`);
    if (request !== latestWeekRequest) return;
    const latest = body.weeks[body.weeks.length - 1];
    replaceOptions(week, body.weeks, latest);
    if (runAfter) await runSlate();
  } catch (error) {
    if (request !== latestWeekRequest) return;
    results.replaceChildren();
    message.textContent = 'Unable to load weeks.';
  }
}

async function runSlate() {
  const request = ++latestSlateRequest;
  const query = queryString();
  renderedSlateQuery = null;
  downloadButton.disabled = true;
  runButton.disabled = true;
  message.textContent = 'Loading...';
  try {
    const body = await jsonOrError(`/api/slate?${query}`);
    if (request !== latestSlateRequest || query !== queryString()) return;
    renderGames(body.games);
    renderedSlateQuery = query;
    downloadButton.disabled = false;
    message.textContent = `${body.games.length} games`;
  } catch (error) {
    if (request !== latestSlateRequest || query !== queryString()) return;
    results.replaceChildren();
    message.textContent = error.message;
  } finally {
    if (request === latestSlateRequest && query === queryString()) {
      runButton.disabled = false;
    }
  }
}

async function initialize() {
  try {
    const options = await jsonOrError('/api/options');
    replaceOptions(season, options.seasons, options.latest.season);
    replaceOptions(week, options.weeks, options.latest.week);
    replaceOptions(estimator, options.estimators, options.default_estimator);
    edge.value = String(options.default_edge_threshold);
    await runSlate();
  } catch (error) {
    message.textContent = error.message;
  }
}

season.addEventListener('change', () => {
  invalidateSlate(false);
  loadWeeks(true);
});
for (const selector of [week, estimator, edge]) {
  selector.addEventListener('change', () => invalidateSlate());
}
runButton.addEventListener('click', runSlate);
downloadButton.addEventListener('click', () => {
  if (downloadButton.disabled) return;
  if (renderedSlateQuery === null || renderedSlateQuery !== queryString()) {
    invalidateSlate();
    return;
  }
  window.location = `/api/slate.csv?${renderedSlateQuery}`;
});
document.addEventListener('DOMContentLoaded', initialize);
</script>
"""


def create_app(
    service: SlateService,
    tracker_service: TrackerService,
    access_code: str | None,
) -> FastAPI:
    """Create the protected web dashboard around a slate service."""
    app = FastAPI(title="NFL game model")
    store = SessionStore()

    @app.exception_handler(RequestValidationError)
    async def request_validation_error(request, exc):
        return JSONResponse({"error": "Invalid request"}, status_code=422)

    @app.exception_handler(SlateInputError)
    async def input_error(request, exc):
        return JSONResponse({"error": str(exc)}, status_code=422)

    @app.exception_handler(TrackerInputError)
    async def tracker_input_error(request, exc):
        return JSONResponse({"error": str(exc)}, status_code=422)

    @app.exception_handler(SlateNotFoundError)
    async def not_found_error(request, exc):
        return JSONResponse({"error": str(exc)}, status_code=404)

    @app.exception_handler(SlateUnavailableError)
    async def unavailable_error(request, exc):
        return JSONResponse({"error": str(exc)}, status_code=409)

    @app.exception_handler(Exception)
    async def unexpected_error(request, exc):
        logger.exception("Unhandled web request failure")
        return JSONResponse({"error": "Unexpected server error"}, status_code=500)

    @app.get("/", response_class=HTMLResponse)
    def index():
        return PAGE

    @app.get("/tracker", response_class=HTMLResponse)
    def tracker():
        return TRACKER_PAGE

    @app.get("/health")
    def health():
        return {"ok": True}

    @app.get("/api/options")
    def options():
        return service.options()

    @app.get("/api/weeks")
    def weeks(season: int):
        return {"weeks": service.weeks(season)}

    @app.get("/api/tracker/options")
    def tracker_options():
        return tracker_service.options()

    @app.get("/api/tracker/summary")
    def tracker_summary(record_type: str = "backtest", season: str = "all"):
        return tracker_service.summary(record_type, season)

    @app.get("/api/tracker/games")
    def tracker_games(season: int, record_type: str = "backtest"):
        return {"games": tracker_service.records(record_type, season)}

    @app.get("/api/slate")
    def slate(
        season: int,
        week: int,
        estimator: str = "ridge",
        edge_threshold: float = Query(DEFAULT_EDGE_THRESHOLD),
    ):
        return {"games": service.records(season, week, estimator, edge_threshold)}

    @app.get("/api/slate.csv")
    def slate_csv(
        season: int,
        week: int,
        estimator: str = "ridge",
        edge_threshold: float = Query(DEFAULT_EDGE_THRESHOLD),
    ):
        content = service.csv(season, week, estimator, edge_threshold)
        filename = f"slate_{season}_wk{week:02d}_{estimator}.csv"
        return Response(
            content,
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    if access_code is not None:
        add_login_routes(app, store, access_code)
    app.add_middleware(
        AccessCodeMiddleware,
        store=store,
        enabled=access_code is not None,
    )
    return app
