"""Framework-free performance tracker page."""

TRACKER_PAGE = """<!doctype html>
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Track record | NFL Performance Tracker</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=JetBrains+Mono:wght@400;700&display=swap" rel="stylesheet">
<style>
  :root { color-scheme: light; --ink: #111111; --muted: #555555; --rule: #e4e4e4; --accent: #c4142f; }
  * { box-sizing: border-box; }
  body { margin: 0; background: #ffffff; color: var(--ink); font-family: 'Space Grotesk', system-ui, sans-serif; }
  main { max-width: 76rem; margin: auto; padding: 1.5rem 1rem 4rem; display: grid; gap: 1.75rem; }
  main > * { min-width: 0; }
  a { color: var(--ink); }
  a:hover { color: var(--accent); }
  .masthead { display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 1rem; border-bottom: 3px solid var(--ink); padding-bottom: 1rem; }
  .brand { font-weight: 700; font-size: 1.25rem; letter-spacing: -0.01em; }
  .masthead nav { display: flex; flex-wrap: wrap; gap: 1.5rem; font-weight: 500; }
  .masthead nav a { text-decoration: none; }
  .masthead nav a[aria-current="page"] { text-decoration: underline; text-decoration-thickness: 2px; text-underline-offset: 6px; text-decoration-color: var(--accent); }
  .kicker { margin: 0 0 .5rem; font-family: 'JetBrains Mono', ui-monospace, monospace; font-size: .875rem; letter-spacing: .06em; color: var(--accent); }
  h1 { margin: 0; font-size: clamp(2.5rem, 6vw, 4rem); line-height: 1; letter-spacing: -0.03em; }
  .lede { margin: .75rem 0 0; font-size: 1.0625rem; line-height: 1.45; max-width: 48rem; }
  h2 { margin: 0 0 .75rem; font-size: 1.375rem; }
  .controls { display: flex; flex-wrap: wrap; gap: 1rem; align-items: end; }
  .tabs { display: flex; flex-wrap: wrap; gap: 0; border: 2px solid var(--ink); }
  button, select { font: inherit; font-size: .9375rem; min-height: 2.75rem; border-radius: 0; color: var(--ink); background: #ffffff; }
  button { cursor: pointer; border: 0; border-right: 2px solid var(--ink); padding: 0 1rem; font-weight: 700; }
  .tabs button:last-of-type, .tabs button:has(+ button[hidden]) { border-right: 0; }
  button[aria-selected="true"] { color: #ffffff; background: var(--ink); }
  select { border: 2px solid var(--ink); padding: 0 .625rem; }
  label { display: grid; gap: .25rem; font-size: .8125rem; font-weight: 500; }
  #tracker-message { margin: 0; min-height: 1.2rem; font-family: 'JetBrains Mono', ui-monospace, monospace; font-size: .875rem; color: var(--muted); }
  .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(16rem, 1fr)); border: 2px solid var(--ink); }
  .cards:empty { display: none; }
  .card { padding: 1.25rem 1.5rem; border-right: 2px solid var(--ink); }
  .card:last-child { border-right: 0; }
  .card h3 { margin: 0 0 .5rem; font-family: 'JetBrains Mono', ui-monospace, monospace; font-size: .8125rem; font-weight: 400; letter-spacing: .06em; color: var(--muted); }
  .card p { margin: .25rem 0 0; }
  .record { font-size: 1.5rem; font-weight: 700; letter-spacing: -0.01em; font-variant-numeric: tabular-nums; }
  .table-wrap { overflow-x: auto; }
  table { border-collapse: collapse; width: 100%; white-space: nowrap; }
  th { text-align: right; padding: .625rem .5rem; border-bottom: 2px solid var(--ink); font-family: 'JetBrains Mono', ui-monospace, monospace; font-size: .75rem; font-weight: 400; letter-spacing: .06em; text-transform: uppercase; color: var(--muted); }
  td { text-align: right; padding: .75rem .5rem; border-bottom: 1px solid var(--rule); font-family: 'JetBrains Mono', ui-monospace, monospace; font-size: .9375rem; }
  th:first-child, td:first-child { text-align: left; }
  td:first-child { font-family: 'Space Grotesk', system-ui, sans-serif; font-weight: 700; }
  .note { margin: 0; color: var(--muted); font-size: .875rem; line-height: 1.5; }
  @media (max-width: 36rem) {
    .controls, .tabs { align-items: stretch; flex-direction: column; }
    .tabs button { border-right: 0; border-bottom: 2px solid var(--ink); }
    .tabs button:last-of-type { border-bottom: 0; }
    button, select { width: 100%; }
    .card { border-right: 0; border-bottom: 2px solid var(--ink); }
    .card:last-child { border-bottom: 0; }
  }
</style>
<main>
  <header class="masthead">
    <span class="brand">THE EDGE REPORT</span>
    <nav aria-label="Site navigation">
      <a href="/">This week</a>
      <a href="/schedule">Schedule</a>
      <a href="/tracker" aria-current="page">Track record</a>
    </nav>
  </header>
  <section>
    <p class="kicker">OFFICIAL RIDGE MODEL &middot; RIDGE-V1</p>
    <h1>Track record</h1>
    <p class="lede">How the model's picks have graded: the walk-forward backtest and the live 2026 record, always kept separate.</p>
  </section>
  <div class="controls">
    <div class="tabs" role="tablist" aria-label="Record type">
      <button id="historical-tab" type="button" role="tab" aria-selected="true">
        Historical backtest
      </button>
      <button id="live-tab" type="button" role="tab" aria-selected="false">Live record</button>
      <button id="reconstructed-tab" type="button" role="tab" aria-selected="false" hidden>
        Reconstructed (outage)
      </button>
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
  <section aria-labelledby="total-edges-heading">
    <h2 id="total-edges-heading">Cumulative over/under edges</h2>
    <div class="table-wrap"><table id="total-edges"></table></div>
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

  <p class="note">Historical results are walk-forward backtests graded at the line available five days before kickoff, the same lock the live tracker publishes under. The closing-line columns show the same picks settled at the close instead.
  Live tracking begins in 2026. 2026 weeks 2-3 were never published because an automation outage (Sept 16-29) blocked every tracker run; the Reconstructed tab replays them from data available at each would-be publication and is kept out of the live record.
  A 52.4% win rate is the standard -110 break-even reference.
  This is model tracking, not betting advice.</p>
</main>
<script>
const historicalTab = document.getElementById('historical-tab');
const liveTab = document.getElementById('live-tab');
const reconstructedTab = document.getElementById('reconstructed-tab');
const season = document.getElementById('tracker-season');
const trackerMessage = document.getElementById('tracker-message');
const qualifiedCards = document.getElementById('qualified-cards');
const allRecords = document.getElementById('all-records');
const spreadEdges = document.getElementById('spread-edges');
const totalEdges = document.getElementById('total-edges');
const seasonBreakdown = document.getElementById('season-breakdown');
const auditGames = document.getElementById('audit-games');
const closingLine = document.getElementById('closing-line');

let activeRecordType = 'backtest';
let trackerOptions = null;
let queuedRecordType = null;
let latestSummaryRequest = 0;
let latestGamesRequest = 0;

function trackerQuery() {
  return new URLSearchParams({
    record_type: activeRecordType,
    season: season.value,
  }).toString();
}

function invalidateTracker() {
  latestSummaryRequest += 1;
  latestGamesRequest += 1;
  qualifiedCards.replaceChildren();
  allRecords.replaceChildren();
  spreadEdges.replaceChildren();
  totalEdges.replaceChildren();
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

function formatText(value) {
  if (value === null || value === undefined) return 'n/a';
  return String(value);
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

function renderEdges(table, edges) {
  renderTable(
    table,
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
  trackerMessage.textContent = {
    backtest: 'Walk-forward backtest, graded at the 5-day publication line.',
    live: 'Official record uses frozen published lines.',
    reconstructed: 'Reconstructed after the Sept 16-29 outage: each pick replayed at the '
      + 'tracker run that would have published it, using only data available then. '
      + 'Not part of the live record.',
  }[activeRecordType];
  renderRecordCards(qualifiedCards, body.qualified);
  renderRecordCards(allRecords, body.all_predictions);
  renderEdges(spreadEdges, body.spread_edges);
  if (body.total_edges) renderEdges(totalEdges, body.total_edges);
  if (season.value === 'all' && body.by_season) {
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
      ['Published at', game => formatText(game.published_at)],
      ['Kickoff', game => formatText(game.kickoff_at)],
      ['Current kickoff', game => formatText(game.current_kickoff_at)],
      ['Void reason', game => formatText(game.void_reason)],
      ['Model margin', game => formatValue(game.model_margin)],
      ['Spread publication', game => formatText(game.spread_publication_status)],
      ['Spread exclusion', game => formatText(game.spread_exclusion_reason)],
      ['Published spread', game => formatValue(game.published_spread_line)],
      ['Closing spread', game => formatValue(game.closing_spread_line)],
      ['Official spread', game => formatValue(game.official_spread_line)],
      ['ATS pick', game => formatText(game.spread_pick)],
      ['ATS edge', game => formatValue(game.spread_edge)],
      ['Final margin', game => formatValue(game.actual_margin)],
      ['ATS grade', game => formatText(game.spread_grade)],
      ['Spread CLV', game => formatValue(game.spread_clv)],
      ['Spread close grade', game => formatText(game.spread_close_grade)],
      ['Model total', game => formatValue(game.model_total)],
      ['Total publication', game => formatText(game.total_publication_status)],
      ['Total exclusion', game => formatText(game.total_exclusion_reason)],
      ['Published total', game => formatValue(game.published_total_line)],
      ['Closing total', game => formatValue(game.closing_total_line)],
      ['Official total', game => formatValue(game.official_total_line)],
      ['O/U pick', game => formatText(game.total_pick)],
      ['O/U edge', game => formatValue(game.total_edge)],
      ['Final total', game => formatValue(game.actual_total)],
      ['O/U grade', game => formatText(game.total_grade)],
      ['Total CLV', game => formatValue(game.total_clv)],
      ['Total close grade', game => formatText(game.total_close_grade)],
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
  if (season.value !== 'all') requests.push(loadGames());
  await Promise.all(requests);
}

function selectRecordType(recordType) {
  activeRecordType = recordType;
  historicalTab.setAttribute('aria-selected', recordType === 'backtest' ? 'true' : 'false');
  liveTab.setAttribute('aria-selected', recordType === 'live' ? 'true' : 'false');
  reconstructedTab.setAttribute('aria-selected', recordType === 'reconstructed' ? 'true' : 'false');
  invalidateTracker();
  if (!trackerOptions) {
    queuedRecordType = recordType;
    return Promise.resolve();
  }
  queuedRecordType = null;
  replaceSeasonOptions(trackerOptions);
  return loadSelection();
}

function replaceSeasonOptions(options) {
  season.replaceChildren();
  const years = options.seasons[activeRecordType];
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
    trackerOptions = options;
    activeRecordType = queuedRecordType || options.default_record_type;
    queuedRecordType = null;
    replaceSeasonOptions(options);
    historicalTab.setAttribute('aria-selected', activeRecordType === 'backtest' ? 'true' : 'false');
    liveTab.setAttribute('aria-selected', activeRecordType === 'live' ? 'true' : 'false');
    reconstructedTab.setAttribute(
      'aria-selected', activeRecordType === 'reconstructed' ? 'true' : 'false',
    );
    reconstructedTab.hidden = !options.reconstructed_available;
    invalidateTracker();
    await loadSelection();
  } catch (error) {
    trackerMessage.textContent = error.message;
  }
}

historicalTab.addEventListener('click', () => selectRecordType('backtest'));
liveTab.addEventListener('click', () => selectRecordType('live'));
reconstructedTab.addEventListener('click', () => selectRecordType('reconstructed'));
season.addEventListener('change', () => {
  invalidateTracker();
  return loadSelection();
});
document.addEventListener('DOMContentLoaded', initialize);
</script>
"""
