"""Embedded full-season schedule page."""


SCHEDULE_PAGE = """<!doctype html>
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>2026 NFL Schedule</title>
<style>
  body { font-family: system-ui, sans-serif; margin: 1rem; color: #191919; }
  main { max-width: 72rem; margin: auto; }
  nav { display: flex; flex-wrap: wrap; gap: 1rem; }
  #schedule-message { min-height: 1.4rem; }
  .table-wrap { overflow-x: auto; }
  table { border-collapse: collapse; width: 100%; white-space: nowrap; }
  th, td { border-bottom: 1px solid #ddd; padding: .5rem; text-align: right; }
  th:nth-child(3), td:nth-child(3) { text-align: left; }
</style>
<main>
  <nav aria-label="Site navigation">
    <a href="/">Weekly predictions</a>
    <a href="/tracker">Performance tracker</a>
  </nav>
  <h1>2026 NFL Schedule</h1>
  <p id="schedule-message" role="status" aria-live="polite"></p>
  <div class="table-wrap"><table id="schedule-games"></table></div>
</main>
<script>
const scheduleMessage = document.getElementById('schedule-message');
const scheduleGames = document.getElementById('schedule-games');
let latestScheduleRequest = 0;

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

function displayValue(value, kind = 'text') {
  if (value === null || value === undefined || value === '') return '\u2014';
  if (kind === 'text') return String(value);
  const number = Number(value);
  if (!Number.isFinite(number)) return '\u2014';
  if (kind === 'signed') return `${number >= 0 ? '+' : ''}${number.toFixed(1)}`;
  return number.toFixed(1);
}

function renderGames(games) {
  const columns = [
    ['Week', game => displayValue(game.week)],
    ['Kickoff', game => displayValue(game.kickoff_at)],
    ['Matchup', game => `${displayValue(game.away_team)} @ ${displayValue(game.home_team)}`],
    ['Spread', game => displayValue(game.spread_line, 'signed')],
    ['Total', game => displayValue(game.total_line, 'number')],
  ];
  scheduleGames.replaceChildren();
  const header = document.createElement('tr');
  for (const [label] of columns) {
    const cell = document.createElement('th');
    cell.textContent = label;
    header.appendChild(cell);
  }
  scheduleGames.appendChild(header);
  for (const game of games) {
    const row = document.createElement('tr');
    for (const [, value] of columns) {
      const cell = document.createElement('td');
      cell.textContent = value(game);
      row.appendChild(cell);
    }
    scheduleGames.appendChild(row);
  }
}

function freshnessMessage(market, gameCount) {
  const observedAt = displayValue(market && market.observed_at);
  if (market && market.stale) {
    return `Warning: market lines are stale. Last observed ${observedAt} · ${gameCount} games`;
  }
  return `Lines updated ${observedAt} · ${gameCount} games`;
}

async function loadSchedule() {
  const request = ++latestScheduleRequest;
  scheduleGames.replaceChildren();
  scheduleMessage.textContent = 'Loading schedule...';
  try {
    const body = await jsonOrError('/api/schedule?season=2026');
    if (request !== latestScheduleRequest) return;
    renderGames(body.games);
    scheduleMessage.textContent = freshnessMessage(body.market, body.games.length);
  } catch (error) {
    if (request !== latestScheduleRequest) return;
    scheduleGames.replaceChildren();
    scheduleMessage.textContent = error.message;
  }
}

document.addEventListener('DOMContentLoaded', loadSchedule);
</script>
"""
