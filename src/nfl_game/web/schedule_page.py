"""Embedded full-season schedule page."""

SCHEDULE_PAGE = """<!doctype html>
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>2026 NFL Schedule | The Edge Report</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=JetBrains+Mono:wght@400;700&display=swap" rel="stylesheet">
<style>
  :root { --ink: #111111; --muted: #555555; --rule: #e4e4e4; --accent: #c4142f; }
  body { margin: 0; background: #ffffff; color: var(--ink); font-family: 'Space Grotesk', system-ui, sans-serif; }
  main { max-width: 76rem; margin: auto; padding: 1.5rem 1rem 4rem; display: grid; gap: 1.5rem; }
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
  #schedule-message { margin: .5rem 0 0; min-height: 1.2rem; font-family: 'JetBrains Mono', ui-monospace, monospace; font-size: .875rem; color: var(--muted); }
  .table-wrap { overflow-x: auto; }
  table { border-collapse: collapse; width: 100%; min-width: 40rem; white-space: nowrap; }
  th { text-align: left; padding: .625rem .5rem; border-bottom: 2px solid var(--ink); font-family: 'JetBrains Mono', ui-monospace, monospace; font-size: .75rem; font-weight: 400; letter-spacing: .06em; text-transform: uppercase; color: var(--muted); }
  td { padding: .75rem .5rem; border-bottom: 1px solid var(--rule); font-family: 'JetBrains Mono', ui-monospace, monospace; font-size: .9375rem; }
  td:nth-child(3) { font-family: 'Space Grotesk', system-ui, sans-serif; font-size: 1.0625rem; font-weight: 700; }
  .note { margin: 0; color: var(--muted); font-size: .875rem; line-height: 1.5; }
</style>
<main>
  <header class="masthead">
    <span class="brand">THE EDGE REPORT</span>
    <nav aria-label="Site navigation">
      <a href="/">This week</a>
      <a href="/schedule" aria-current="page">Schedule</a>
      <a href="/tracker">Track record</a>
    </nav>
  </header>
  <section>
    <p class="kicker">2026 REGULAR SEASON &middot; ALL 272 GAMES</p>
    <h1>2026 NFL Schedule</h1>
    <p id="schedule-message" role="status" aria-live="polite"></p>
  </section>
  <div class="table-wrap"><table id="schedule-games"></table></div>
  <p class="note">Kickoffs in Eastern time. Spreads are shown from each favorite's side; totals are the market over/under.</p>
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
  if (value === null || value === undefined || value === '') return '\\u2014';
  if (kind === 'text') return String(value);
  const number = Number(value);
  if (!Number.isFinite(number)) return '\\u2014';
  if (kind === 'signed') return `${number >= 0 ? '+' : ''}${number.toFixed(1)}`;
  return number.toFixed(1);
}

// spread_line is the home team's expected margin; read it from the favorite's side.
function favoriteSpread(game) {
  const value = game.spread_line;
  if (value === null || value === undefined || value === '') return '\\u2014';
  const number = Number(value);
  if (!Number.isFinite(number)) return '\\u2014';
  if (number === 0) return 'PK';
  return number > 0
    ? `${displayValue(game.home_team)} -${number.toFixed(1)}`
    : `${displayValue(game.away_team)} -${(-number).toFixed(1)}`;
}

function easternTime(value) {
  if (value === null || value === undefined || value === '') return '\\u2014';
  try {
    return `${new Intl.DateTimeFormat('en-US', {
      timeZone: 'America/New_York',
      weekday: 'short',
      month: 'short',
      day: 'numeric',
      hour: 'numeric',
      minute: '2-digit',
    }).format(new Date(value))} ET`;
  } catch (error) {
    return String(value);
  }
}

function renderGames(games) {
  const columns = [
    ['Week', game => displayValue(game.week)],
    ['Kickoff', game => easternTime(game.kickoff_at)],
    ['Matchup', game => `${displayValue(game.away_team)} @ ${displayValue(game.home_team)}`],
    ['Spread', game => favoriteSpread(game)],
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
  const observedAt = easternTime(market && market.observed_at);
  if (market && market.stale) {
    return `Warning: market lines are stale. Last observed ${observedAt} \\u00b7 ${gameCount} games`;
  }
  return `Lines updated ${observedAt} \\u00b7 ${gameCount} games`;
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
