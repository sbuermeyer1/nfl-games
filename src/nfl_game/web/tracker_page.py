"""Framework-free performance tracker page."""

TRACKER_PAGE = """<!doctype html>
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NFL Performance Tracker</title>
<style>
  :root { color-scheme: light; }
  * { box-sizing: border-box; }
  body { font-family: system-ui, sans-serif; margin: 1rem; color: #191919; background: #fafafa; }
  main { max-width: 72rem; margin: auto; }
  a { color: #075f3b; }
  .controls { display: flex; flex-wrap: wrap; gap: .75rem; align-items: end; }
  .tabs { display: flex; gap: .4rem; }
  button, select { font: inherit; padding: .5rem .7rem; }
  button { cursor: pointer; border: 1px solid #aaa; border-radius: .35rem; background: white; }
  button[aria-selected="true"] { color: white; background: #075f3b; border-color: #075f3b; }
  label { display: grid; gap: .25rem; font-weight: 600; }
  #tracker-message { min-height: 1.4rem; }
  .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(14rem, 1fr)); gap: .75rem; }
  .card { padding: .9rem; border: 1px solid #ddd; border-radius: .5rem; background: white; }
  .card h3 { margin: 0 0 .35rem; font-size: 1rem; }
  .record { margin: 0; font-variant-numeric: tabular-nums; }
  section { margin-top: 1.5rem; }
  .table-wrap { overflow-x: auto; background: white; }
  table { border-collapse: collapse; width: 100%; white-space: nowrap; }
  th, td { border-bottom: 1px solid #ddd; padding: .5rem; text-align: right; }
  th:first-child, td:first-child { text-align: left; }
  .note { color: #555; font-size: .9rem; }
  @media (max-width: 36rem) {
    body { margin: .65rem; }
    .controls, .tabs { align-items: stretch; flex-direction: column; }
    button, select { width: 100%; }
  }
</style>
<main>
  <nav aria-label="Site navigation"><a href="/">Weekly slate</a></nav>
  <h1>NFL Performance Tracker</h1>
  <p>Official Ridge walk-forward model records, kept separate from published live picks.</p>
  <div class="controls">
    <div class="tabs" role="tablist" aria-label="Record type">
      <button id="historical-tab" type="button" role="tab" aria-selected="true">
        Historical backtest
      </button>
      <button id="live-tab" type="button" role="tab" aria-selected="false">Live record</button>
    </div>
    <label>Season <select id="tracker-season"></select></label>
  </div>
  <p id="tracker-message" role="status" aria-live="polite"></p>

  <section aria-labelledby="qualified-heading">
    <h2 id="qualified-heading">Qualified 2+ point picks</h2>
    <div id="qualified-cards" class="cards"></div>
  </section>
  <section aria-labelledby="all-heading">
    <h2 id="all-heading">All predictions</h2>
    <div id="all-records" class="cards"></div>
  </section>
  <section aria-labelledby="edges-heading">
    <h2 id="edges-heading">Cumulative spread edges</h2>
    <div class="table-wrap"><table id="spread-edges"></table></div>
  </section>
  <section aria-labelledby="seasons-heading">
    <h2 id="seasons-heading">Season breakdown</h2>
    <div class="table-wrap"><table id="season-breakdown"></table></div>
  </section>
  <section aria-labelledby="audit-heading">
    <h2 id="audit-heading">Game audit</h2>
    <div class="table-wrap"><table id="audit-games"></table></div>
  </section>
  <section aria-labelledby="closing-heading">
    <h2 id="closing-heading">Closing-line metrics</h2>
    <div id="closing-line" class="cards"></div>
  </section>

  <p class="note">Historical results are walk-forward backtests graded against closing lines.
  Live tracking begins in 2026. A 52.4% win rate is the standard -110 break-even reference.
  This is model tracking, not betting advice.</p>
</main>
<script>
const historicalTab = document.getElementById('historical-tab');
const liveTab = document.getElementById('live-tab');
const season = document.getElementById('tracker-season');
const trackerMessage = document.getElementById('tracker-message');
const qualifiedCards = document.getElementById('qualified-cards');
const allRecords = document.getElementById('all-records');
const spreadEdges = document.getElementById('spread-edges');
const seasonBreakdown = document.getElementById('season-breakdown');
const auditGames = document.getElementById('audit-games');
const closingLine = document.getElementById('closing-line');

let activeRecordType = 'backtest';
let latestSummaryRequest = 0;
let latestGamesRequest = 0;

function trackerQuery() {
  return new URLSearchParams({
    record_type: activeRecordType,
    season: activeRecordType === 'live' ? 'all' : season.value,
  }).toString();
}

function invalidateTracker() {
  latestSummaryRequest += 1;
  latestGamesRequest += 1;
  qualifiedCards.replaceChildren();
  allRecords.replaceChildren();
  spreadEdges.replaceChildren();
  seasonBreakdown.replaceChildren();
  auditGames.replaceChildren();
  closingLine.replaceChildren();
  trackerMessage.textContent = '';
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

function formatValue(value) {
  if (value === null || value === undefined) return 'n/a';
  return Number(value).toFixed(1);
}

function formatRate(value) {
  if (value === null || value === undefined) return 'n/a';
  return `${(Number(value) * 100).toFixed(1)}%`;
}

function recordText(record) {
  return `${record.wins}-${record.losses}-${record.pushes} · ${formatRate(record.win_rate)} · n=${record.n_graded}`;
}

function appendText(parent, tagName, text, className = '') {
  const node = document.createElement(tagName);
  node.textContent = text;
  node.className = className;
  parent.appendChild(node);
  return node;
}

function renderRecordCards(parent, records, prefix = '') {
  parent.replaceChildren();
  for (const [kind, label] of [['spread', 'ATS'], ['total', 'O/U']]) {
    const card = document.createElement('article');
    card.className = 'card';
    appendText(card, 'h3', `${prefix}${label}`);
    appendText(card, 'p', recordText(records[kind]), 'record');
    parent.appendChild(card);
  }
}

function renderTable(table, columns, rows) {
  table.replaceChildren();
  const header = document.createElement('tr');
  for (const [label] of columns) appendText(header, 'th', label);
  table.appendChild(header);
  for (const rowData of rows) {
    const row = document.createElement('tr');
    for (const [, value] of columns) appendText(row, 'td', value(rowData));
    table.appendChild(row);
  }
}

function renderSpreadEdges(edges) {
  renderTable(
    spreadEdges,
    [
      ['Minimum edge', row => `${formatValue(row.min_edge).replace('.0', '')}+`],
      ['Record', row => recordText(row.record)],
    ],
    edges,
  );
}

function renderSeasons(rows) {
  renderTable(
    seasonBreakdown,
    [
      ['Season', row => String(row.season)],
      ['Qualified ATS', row => recordText(row.qualified.spread)],
      ['Qualified O/U', row => recordText(row.qualified.total)],
      ['All ATS', row => recordText(row.all_predictions.spread)],
      ['All O/U', row => recordText(row.all_predictions.total)],
    ],
    rows,
  );
}

function renderClosingMetrics(metrics) {
  closingLine.replaceChildren();
  for (const [kind, label] of [['spread', 'Spread'], ['total', 'Total']]) {
    const values = metrics[kind];
    const card = document.createElement('article');
    card.className = 'card';
    appendText(card, 'h3', label);
    appendText(card, 'p', `Average CLV: ${formatValue(values.average_clv)} points`);
    appendText(card, 'p', `Beat close: ${formatRate(values.beat_close_rate)} · n=${values.n_clv}`);
    appendText(card, 'p', `Close record: ${recordText(values.record)}`);
    closingLine.appendChild(card);
  }
}

function renderSummary(body) {
  if (!body.available) {
    trackerMessage.textContent = body.message;
    return;
  }
  trackerMessage.textContent = activeRecordType === 'backtest'
    ? 'Walk-forward backtest against closing lines.'
    : 'Official record uses frozen published lines.';
  renderRecordCards(qualifiedCards, body.qualified);
  renderRecordCards(allRecords, body.all_predictions);
  renderSpreadEdges(body.spread_edges);
  if (activeRecordType === 'backtest' && season.value === 'all' && body.by_season) {
    renderSeasons(body.by_season);
  }
  if (body.closing_line) renderClosingMetrics(body.closing_line);
}

function renderGames(games) {
  renderTable(
    auditGames,
    [
      ['Matchup', game => `${game.away_team} @ ${game.home_team}`],
      ['Week', game => String(game.week)],
      ['Model margin', game => formatValue(game.model_margin)],
      ['Spread line', game => formatValue(game.official_spread_line)],
      ['ATS pick', game => game.spread_pick],
      ['ATS edge', game => formatValue(game.spread_edge)],
      ['Final margin', game => formatValue(game.actual_margin)],
      ['ATS grade', game => game.spread_grade],
      ['Model total', game => formatValue(game.model_total)],
      ['Total line', game => formatValue(game.official_total_line)],
      ['O/U pick', game => game.total_pick],
      ['O/U edge', game => formatValue(game.total_edge)],
      ['Final total', game => formatValue(game.actual_total)],
      ['O/U grade', game => game.total_grade],
    ],
    games,
  );
}

async function loadSummary() {
  const request = ++latestSummaryRequest;
  const query = trackerQuery();
  trackerMessage.textContent = 'Loading...';
  try {
    const body = await jsonOrError(`/api/tracker/summary?${query}`);
    if (request !== latestSummaryRequest || query !== trackerQuery()) return;
    renderSummary(body);
  } catch (error) {
    if (request !== latestSummaryRequest || query !== trackerQuery()) return;
    trackerMessage.textContent = error.message;
  } finally {
    if (request !== latestSummaryRequest || query !== trackerQuery()) return;
    season.disabled = activeRecordType === 'live';
  }
}

async function loadGames() {
  const request = ++latestGamesRequest;
  const query = trackerQuery();
  try {
    const body = await jsonOrError(`/api/tracker/games?${query}`);
    if (request !== latestGamesRequest || query !== trackerQuery()) return;
    renderGames(body.games);
  } catch (error) {
    if (request !== latestGamesRequest || query !== trackerQuery()) return;
    auditGames.replaceChildren();
    trackerMessage.textContent = error.message;
  } finally {
    if (request !== latestGamesRequest || query !== trackerQuery()) return;
  }
}

async function loadSelection() {
  const requests = [loadSummary()];
  if (activeRecordType === 'backtest' && season.value !== 'all') requests.push(loadGames());
  await Promise.all(requests);
}

function selectRecordType(recordType) {
  activeRecordType = recordType;
  historicalTab.setAttribute('aria-selected', recordType === 'backtest' ? 'true' : 'false');
  liveTab.setAttribute('aria-selected', recordType === 'live' ? 'true' : 'false');
  season.disabled = recordType === 'live';
  invalidateTracker();
  return loadSelection();
}

function replaceSeasonOptions(options) {
  season.replaceChildren();
  const years = options.historical_seasons;
  const overall = document.createElement('option');
  overall.value = 'all';
  overall.textContent = years.length
    ? `Overall (${years[0]}-${years[years.length - 1]})`
    : 'Overall';
  overall.selected = options.default_season === 'all';
  season.appendChild(overall);
  for (const year of years) {
    const option = document.createElement('option');
    option.value = String(year);
    option.textContent = String(year);
    option.selected = String(year) === String(options.default_season);
    season.appendChild(option);
  }
  season.value = String(options.default_season);
}

async function initialize() {
  try {
    const options = await jsonOrError('/api/tracker/options');
    replaceSeasonOptions(options);
    activeRecordType = options.default_record_type;
    historicalTab.setAttribute('aria-selected', activeRecordType === 'backtest' ? 'true' : 'false');
    liveTab.setAttribute('aria-selected', activeRecordType === 'live' ? 'true' : 'false');
    season.disabled = activeRecordType === 'live';
    invalidateTracker();
    await loadSelection();
  } catch (error) {
    trackerMessage.textContent = error.message;
  }
}

historicalTab.addEventListener('click', () => selectRecordType('backtest'));
liveTab.addEventListener('click', () => selectRecordType('live'));
season.addEventListener('change', () => {
  invalidateTracker();
  return loadSelection();
});
document.addEventListener('DOMContentLoaded', initialize);
</script>
"""
