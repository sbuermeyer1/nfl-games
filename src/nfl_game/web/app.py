"""FastAPI application factory and embedded NFL slate dashboard."""

import logging

from fastapi import FastAPI, Query
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse, Response

from nfl_game.web.auth import AccessCodeMiddleware
from nfl_game.web.login import add_login_routes
from nfl_game.web.schedule_page import SCHEDULE_PAGE
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
<title>Beat the book this week | NFL Game Model</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=JetBrains+Mono:wght@400;700&display=swap" rel="stylesheet">
<style>
  :root { --ink: #111111; --muted: #555555; --rule: #e4e4e4; --accent: #c4142f; --qb: #8a4b00; }
  body { margin: 0; background: #ffffff; color: var(--ink); font-family: 'Space Grotesk', system-ui, sans-serif; }
  main { max-width: 76rem; margin: auto; padding: 1.5rem 1rem 4rem; display: grid; gap: 2rem; }
  /* Grid items default to min-width:auto, so the wide table would stretch the whole page
     sideways on a phone instead of scrolling inside .table-wrap. */
  main > * { min-width: 0; }
  a { color: var(--ink); }
  a:hover { color: var(--accent); }
  .mono { font-family: 'JetBrains Mono', ui-monospace, monospace; }
  .masthead { display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 1rem; border-bottom: 3px solid var(--ink); padding-bottom: 1rem; }
  .brand { font-weight: 700; font-size: 1.25rem; letter-spacing: -0.01em; }
  .masthead nav { display: flex; flex-wrap: wrap; gap: 1.5rem; font-weight: 500; }
  .masthead nav a { text-decoration: none; }
  .masthead nav a[aria-current="page"] { text-decoration: underline; text-decoration-thickness: 2px; text-underline-offset: 6px; text-decoration-color: var(--accent); }
  .hero { display: flex; flex-wrap: wrap; justify-content: space-between; align-items: flex-end; gap: 1.5rem; }
  .kicker { margin: 0 0 .5rem; font-size: .875rem; letter-spacing: .06em; color: var(--accent); text-transform: uppercase; }
  h1 { margin: 0; font-size: clamp(2.5rem, 6vw, 4rem); line-height: 1; letter-spacing: -0.03em; }
  h2 { margin: 0; font-size: 1.375rem; }
  .status { margin: .5rem 0 0; font-size: .875rem; color: var(--muted); min-height: 1.2rem; }
  .status:empty { display: none; }
  .controls { display: flex; flex-wrap: wrap; gap: .625rem; align-items: flex-end; }
  label { display: grid; gap: .25rem; font-size: .8125rem; font-weight: 500; }
  select, input, button { font: inherit; font-size: .9375rem; height: 2.75rem; box-sizing: border-box; border: 2px solid var(--ink); border-radius: 0; background: #ffffff; color: var(--ink); padding: 0 .625rem; }
  input { width: 6rem; }
  button { cursor: pointer; font-weight: 700; padding: 0 1.125rem; }
  button.primary { background: var(--ink); color: #ffffff; }
  button:disabled { cursor: not-allowed; opacity: .45; }
  .callouts { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); border: 2px solid var(--ink); }
  .callouts:empty { display: none; }
  .callouts article { padding: 1.5rem; display: grid; gap: .75rem; align-content: start; border-right: 2px solid var(--ink); }
  .callouts article:last-child { border-right: 0; }
  .callouts .rank { font-family: 'JetBrains Mono', ui-monospace, monospace; font-size: .8125rem; color: var(--muted); }
  .callouts .matchup { font-size: 1.75rem; font-weight: 700; letter-spacing: -0.01em; }
  .callouts .gap { font-size: 4.5rem; line-height: .9; font-weight: 700; letter-spacing: -0.04em; }
  .callouts .gap.hot { color: var(--accent); }
  .callouts .unit { font-size: 1rem; color: var(--muted); }
  .callouts .story { margin: 0; font-size: 1.0625rem; line-height: 1.45; }
  .callouts .qb { font-size: .875rem; font-weight: 700; color: var(--qb); }
  .section-head { display: flex; flex-wrap: wrap; justify-content: space-between; align-items: baseline; gap: .5rem; }
  .table-wrap { overflow-x: auto; }
  table { border-collapse: collapse; width: 100%; min-width: 56rem; white-space: nowrap; }
  th { text-align: left; padding: .625rem .5rem; border-bottom: 2px solid var(--ink); font-family: 'JetBrains Mono', ui-monospace, monospace; font-size: .75rem; font-weight: 400; letter-spacing: .06em; text-transform: uppercase; color: var(--muted); }
  td { padding: .875rem .5rem; border-bottom: 1px solid var(--rule); font-family: 'JetBrains Mono', ui-monospace, monospace; font-size: .9375rem; }
  td:first-child { font-family: 'Space Grotesk', system-ui, sans-serif; font-size: 1.0625rem; }
  tr.edge td:first-child, tr.edge td:nth-child(3) { font-weight: 700; }
  td.lean { padding-left: calc(.5rem + 136px); color: var(--muted); background: linear-gradient(#cfcfcf, #cfcfcf) no-repeat .5rem 50% / var(--bar, 0px) 12px; }
  td.lean.hot { color: var(--accent); font-weight: 700; background-image: linear-gradient(var(--accent), var(--accent)); }
  tr.qb-watch td:last-child { color: var(--qb); font-weight: 700; }
  .note { margin: 0; color: var(--muted); font-size: .875rem; line-height: 1.5; }
  @media (max-width: 820px) {
    .callouts { grid-template-columns: 1fr; }
    .callouts article { border-right: 0; border-bottom: 2px solid var(--ink); }
    .callouts article:last-child { border-bottom: 0; }
    .callouts .gap { font-size: 3.5rem; }
  }
</style>
<main>
  <header class="masthead">
    <span class="brand">THE EDGE REPORT</span>
    <nav aria-label="Site navigation">
      <a href="/" aria-current="page">This week</a>
      <a href="/schedule">Schedule</a>
      <a href="/tracker">Track record</a>
    </nav>
  </header>

  <section class="hero" aria-labelledby="page-title">
    <div>
      <p class="kicker mono" id="slate-label"></p>
      <h1 id="page-title">Beat the book this week</h1>
      <p id="market-message" class="status mono" role="status" aria-live="polite"></p>
      <p id="starter-message" class="status mono" role="status" aria-live="polite"></p>
      <p id="message" class="status mono" role="status"></p>
    </div>
    <div class="controls">
      <label>Season <select id="season"></select></label>
      <label>Week <select id="week"></select></label>
      <label>Model <select id="estimator"></select></label>
      <label>Edge at <input id="edge" type="number" min="0" step="0.5"></label>
      <button id="run" type="button" class="primary">Run slate</button>
      <button id="download" type="button">CSV</button>
    </div>
  </section>

  <section aria-labelledby="spread-edges-title" class="section">
    <div class="section-head"><h2 id="spread-edges-title">Biggest spread edges</h2></div>
    <div id="top-spreads" class="callouts"></div>
  </section>

  <section aria-labelledby="total-edges-title" class="section">
    <div class="section-head"><h2 id="total-edges-title">Biggest over/under edges</h2></div>
    <div id="top-totals" class="callouts"></div>
  </section>

  <section aria-labelledby="slate-title">
    <div class="section-head">
      <h2 id="slate-title">The full slate</h2>
      <span class="mono note">EDGE = DISAGREEMENT AT OR ABOVE THE EDGE SETTING · NOT BETTING ADVICE</span>
    </div>
    <div class="table-wrap"><table id="results"></table></div>
  </section>

  <p class="note">Spreads are shown from each favorite's side. A lean is the side of the market line the model prefers; its bar grows with the gap, up to 5 points. Spread edges come from the model's published edge flag; over/under edges use the same setting. This is model tracking, not betting advice.</p>
</main>
<script>
const season = document.getElementById('season');
const week = document.getElementById('week');
const estimator = document.getElementById('estimator');
const edge = document.getElementById('edge');
const runButton = document.getElementById('run');
const downloadButton = document.getElementById('download');
const message = document.getElementById('message');
const marketMessage = document.getElementById('market-message');
const starterMessage = document.getElementById('starter-message');
const results = document.getElementById('results');
const slateLabel = document.getElementById('slate-label');
const topSpreads = document.getElementById('top-spreads');
const topTotals = document.getElementById('top-totals');
const BAR_MAX_PX = 120;
const BAR_FULL_GAP = 5;
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

function finite(value) {
  if (value === null || value === undefined || value === '') return null;
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function signedLine(value) {
  if (value === 0) return 'PK';
  return `${value > 0 ? '+' : ''}${value.toFixed(1)}`;
}

// Spreads are home-team margins; a reader expects the favorite's side ("CHI -3.5").
function favoriteLine(game, margin) {
  const value = finite(margin);
  if (value === null) return '—';
  if (value === 0) return 'PK';
  return value > 0
    ? `${game.home_team} -${value.toFixed(1)}`
    : `${game.away_team} -${(-value).toFixed(1)}`;
}

// The side of the MARKET line the model prefers: a positive gap means the model rates
// the home team higher than the market does.
function spreadLean(game) {
  const gap = finite(game.spread_gap);
  const market = finite(game.market_spread);
  if (gap === null || market === null) return null;
  if (gap === 0) return { gap, text: 'No lean' };
  const pick = gap > 0
    ? `${game.home_team} ${signedLine(-market)}`
    : `${game.away_team} ${signedLine(market)}`;
  return { gap, pick, text: `${pick} · ${Math.abs(gap).toFixed(1)} pts` };
}

function totalLean(game) {
  const gap = finite(game.total_gap);
  const market = finite(game.market_total);
  const model = finite(game.model_total);
  if (gap === null || market === null || model === null) return null;
  if (gap === 0) return { gap, text: `NO LEAN · model ${model.toFixed(1)}` };
  const side = gap > 0 ? 'over' : 'under';
  return {
    gap,
    side,
    market,
    model,
    text: `${side.toUpperCase()} ${market.toFixed(1)} · model ${model.toFixed(1)}`,
  };
}

function barWidth(gap) {
  return `${Math.round(Math.min(Math.abs(gap), BAR_FULL_GAP) / BAR_FULL_GAP * BAR_MAX_PX)}px`;
}

function edgeThreshold() {
  const value = finite(edge.value);
  return value === null ? Infinity : value;
}

function easternTime(value) {
  if (!value) return '—';
  try {
    return `${new Intl.DateTimeFormat('en-US', {
      timeZone: 'America/New_York',
      month: 'short',
      day: 'numeric',
      hour: 'numeric',
      minute: '2-digit',
    }).format(new Date(value))} ET`;
  } catch (error) {
    return value;
  }
}

function qbCell(game) {
  // Mirrors market/compare.py::_qb_cell exactly -- a reader comparing the CLI and
  // the dashboard for the same game must see the same thing. In particular,
  // qb_watch === 0 with qb_inferred === 1 (no chart published yet) must render
  // "unconfirmed", not the same blank cell as a genuine, chart-confirmed no-change.
  if (game.qb_watch === null || game.qb_watch === undefined) return 'n/a';
  const inferred = game.qb_inferred === 1;
  if (game.qb_watch === 0) return inferred ? 'unconfirmed' : '';
  const parts = [];
  for (const [name, delta] of [[game.home_qb, game.qb_change_epa_home],
                               [game.away_qb, game.qb_change_epa_away]]) {
    if (delta === null || delta === undefined || delta === 0) continue;
    parts.push(`${name === null ? 'unknown' : name} ${delta > 0 ? '+' : ''}${delta.toFixed(2)}`);
  }
  return (parts.join('; ') || 'change') + (inferred ? ' (inferred)' : '');
}

function renderMarket(market) {
  if (!market) {
    marketMessage.textContent = '';
    return;
  }
  const observedAt = market.observed_at ? easternTime(market.observed_at) : '—';
  marketMessage.textContent = market.stale
    ? `Warning: market lines are stale. Last observed ${observedAt}`
    : `Lines updated ${observedAt}`;
}

function renderStarters(starters) {
  // Mirrors renderMarket: a stale starter snapshot is not bounded by a short TTL --
  // the provider keeps returning it, marked stale, for as long as refreshes keep
  // failing -- and a depth chart's whole value is timeliness.
  if (!starters) {
    starterMessage.textContent = '';
    return;
  }
  const observedAt = starters.observed_at ? easternTime(starters.observed_at) : '—';
  starterMessage.textContent = starters.stale
    ? `Warning: starter advisory is stale. Last observed ${observedAt}`
    : '';
}

function appendText(parent, tagName, className, text) {
  const element = document.createElement(tagName);
  element.className = className;
  element.textContent = text;
  parent.appendChild(element);
  return element;
}

function callout(rankLabel, game, gap, hot, story, qbNote) {
  const article = document.createElement('article');
  appendText(article, 'span', 'rank', rankLabel);
  appendText(article, 'span', 'matchup', `${game.away_team} @ ${game.home_team}`);
  appendText(article, 'span', hot ? 'gap hot' : 'gap', Math.abs(gap).toFixed(1));
  appendText(article, 'span', 'unit', 'points apart');
  appendText(article, 'p', 'story', story);
  if (qbNote) appendText(article, 'span', 'qb', `QB watch: ${qbNote}`);
  return article;
}

function renderCallouts(games, threshold) {
  const spreads = games
    .map(game => ({ game, lean: spreadLean(game) }))
    .filter(item => item.lean && item.lean.gap !== 0)
    .sort((a, b) => Math.abs(b.lean.gap) - Math.abs(a.lean.gap))
    .slice(0, 3);
  topSpreads.replaceChildren(...spreads.map(({ game, lean }, index) => callout(
    `#${index + 1} SPREAD ${game.edge_flag === 1 ? 'EDGE' : 'GAP'}`,
    game,
    lean.gap,
    game.edge_flag === 1,
    `Market has ${favoriteLine(game, game.market_spread)}. The model makes it `
      + `${favoriteLine(game, game.model_spread)}, so it leans ${lean.pick}.`,
    game.qb_watch === 1 ? qbCell(game) : '',
  )));

  const totals = games
    .map(game => ({ game, lean: totalLean(game) }))
    .filter(item => item.lean && item.lean.gap !== 0)
    .sort((a, b) => Math.abs(b.lean.gap) - Math.abs(a.lean.gap))
    .slice(0, 3);
  topTotals.replaceChildren(...totals.map(({ game, lean }, index) => callout(
    `#${index + 1} TOTAL ${Math.abs(lean.gap) >= threshold ? 'EDGE' : 'GAP'}`,
    game,
    lean.gap,
    Math.abs(lean.gap) >= threshold,
    `Market total is ${lean.market.toFixed(1)}. The model projects ${lean.model.toFixed(1)}, `
      + `so it leans the ${lean.side} ${lean.market.toFixed(1)}.`,
    '',
  )));
}

function renderGames(games, threshold) {
  const columns = [
    ['Game', game => ({ text: `${game.away_team} @ ${game.home_team}` })],
    ['Market', game => ({ text: favoriteLine(game, game.market_spread) })],
    ['Model', game => ({ text: favoriteLine(game, game.model_spread) })],
    ['Spread lean', game => {
      const lean = spreadLean(game);
      if (!lean) return { text: '—' };
      return { text: lean.text, gap: lean.gap, hot: game.edge_flag === 1 };
    }],
    ['Over/under lean', game => {
      const lean = totalLean(game);
      if (!lean) return { text: '—' };
      return { text: lean.text, gap: lean.gap, hot: Math.abs(lean.gap) >= threshold };
    }],
    ['QB', game => ({ text: qbCell(game) })],
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
      const content = value(game);
      cell.textContent = content.text;
      if (content.gap !== undefined) {
        cell.className = content.hot ? 'lean hot' : 'lean';
        cell.style.setProperty('--bar', barWidth(content.gap));
      }
      row.appendChild(cell);
    }
    const classes = [];
    if (game.edge_flag === 1) classes.push('edge');
    if (game.qb_watch === 1) classes.push('qb-watch');
    if (classes.length) row.className = classes.join(' ');
    results.appendChild(row);
  }
}

function clearSlate() {
  results.replaceChildren();
  topSpreads.replaceChildren();
  topTotals.replaceChildren();
  slateLabel.textContent = '';
}

function invalidateSlate(runAvailable = true) {
  latestSlateRequest += 1;
  renderedSlateQuery = null;
  clearSlate();
  message.textContent = '';
  marketMessage.textContent = '';
  starterMessage.textContent = '';
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
    clearSlate();
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
  marketMessage.textContent = '';
  starterMessage.textContent = '';
  try {
    const body = await jsonOrError(`/api/slate?${query}`);
    if (request !== latestSlateRequest || query !== queryString()) return;
    const threshold = edgeThreshold();
    renderGames(body.games, threshold);
    renderCallouts(body.games, threshold);
    slateLabel.textContent = `${season.value} · Week ${week.value} · ${body.games.length} games`;
    renderMarket(body.market);
    renderStarters(body.starters);
    renderedSlateQuery = query;
    downloadButton.disabled = false;
    message.textContent = `${body.games.length} games`;
  } catch (error) {
    if (request !== latestSlateRequest || query !== queryString()) return;
    clearSlate();
    marketMessage.textContent = '';
    starterMessage.textContent = '';
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

    @app.get("/schedule", response_class=HTMLResponse)
    def schedule_page():
        return SCHEDULE_PAGE

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

    @app.get("/api/schedule")
    def schedule(season: int = 2026):
        return service.schedule_records(season)

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
        return service.payload(season, week, estimator, edge_threshold)

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
