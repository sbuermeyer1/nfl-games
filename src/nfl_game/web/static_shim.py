"""Browser shim that lets the unchanged dashboard pages run as a static site.

The static export (`static_export.py`) pre-renders every API response the pages can
request into `/data/...` files. This script loads before each page's own script and
replaces `window.fetch` so the page's `/api/...?query` calls read those files instead.
The pages themselves are not modified, so the served dashboard and the static site run
the same page code.

Two things cannot be pre-rendered for every input, because the edge threshold is a
free numeric field. Both are recomputed here from the default-threshold export using
the exact rule in `market/compare.py::build_slate`: `edge_flag = |spread_gap| >= t`,
with a missing gap never flagged.

- `/api/slate` payloads get `edge_flag` recomputed per game.
- The CSV download is intercepted (the page navigates to `/api/slate.csv`, which no
  fetch override can catch) and served from the exported CSV with only the
  `edge_flag` field rewritten, leaving every other byte as pandas wrote it.
"""

STATIC_SHIM = r"""
(() => {
const SERVER_THRESHOLD_ERROR = 'edge threshold must be a finite non-negative number';
const realFetch = globalThis.fetch ? globalThis.fetch.bind(globalThis) : null;

// Self-contained so the pure functions below run in QuickJS, which has no URL class.
function parseUrl(url) {
  const text = String(url);
  const mark = text.indexOf('?');
  const pathname = mark < 0 ? text : text.slice(0, mark);
  const params = {};
  if (mark >= 0) {
    for (const pair of text.slice(mark + 1).split('&')) {
      if (!pair) continue;
      const eq = pair.indexOf('=');
      const decode = (value) => decodeURIComponent(value.replace(/\+/g, ' '));
      const key = decode(eq < 0 ? pair : pair.slice(0, eq));
      if (!(key in params)) params[key] = eq < 0 ? '' : decode(pair.slice(eq + 1));
    }
  }
  return { pathname, get: (name) => (name in params ? params[name] : null) };
}

function staticPath(url) {
  const parsed = parseUrl(url);
  const part = (name) => encodeURIComponent(parsed.get(name) || '');
  switch (parsed.pathname) {
    case '/api/options': return '/data/options.json';
    case '/api/weeks': return `/data/weeks/${part('season')}.json`;
    case '/api/slate':
      return `/data/slate/${part('season')}/${part('week')}/${part('estimator')}.json`;
    case '/api/slate.csv':
      return `/data/slate/${part('season')}/${part('week')}/${part('estimator')}.csv`;
    case '/api/schedule': return `/data/schedule/${part('season')}.json`;
    case '/api/tracker/options': return '/data/tracker/options.json';
    case '/api/tracker/summary':
      return `/data/tracker/summary/${part('record_type')}/${part('season')}.json`;
    case '/api/tracker/games':
      return `/data/tracker/games/${part('record_type')}/${part('season')}.json`;
    default: return null;
  }
}

// Mirrors the server: an unparseable value is a request validation error, and a
// parseable but negative or non-finite one is the service's own input error.
function parseThreshold(raw) {
  const text = raw === null || raw === undefined ? '' : String(raw).trim();
  const value = Number(text);
  if (text === '' || Number.isNaN(value)) return { error: 'Invalid request' };
  if (!Number.isFinite(value) || value < 0) return { error: SERVER_THRESHOLD_ERROR };
  return { value };
}

function edgeFlag(game, threshold) {
  const gap = game.spread_gap;
  if (gap === null || gap === undefined) return 0;
  return Math.abs(gap) >= threshold ? 1 : 0;
}

function withEdge(games, threshold) {
  return games.map(game => ({ ...game, edge_flag: edgeFlag(game, threshold) }));
}

// Split CSV text into rows of [start, end) field spans, honouring quoted fields.
function csvFieldSpans(text) {
  const rows = [];
  let row = [];
  let start = 0;
  let quoted = false;
  let i = 0;
  while (i < text.length) {
    const ch = text[i];
    if (quoted) {
      if (ch === '"' && text[i + 1] === '"') { i += 2; continue; }
      if (ch === '"') quoted = false;
      i += 1;
      continue;
    }
    if (ch === '"') { quoted = true; i += 1; continue; }
    if (ch === ',') { row.push([start, i]); start = i + 1; i += 1; continue; }
    if (ch === '\r' || ch === '\n') {
      row.push([start, i]);
      rows.push(row);
      row = [];
      i += ch === '\r' && text[i + 1] === '\n' ? 2 : 1;
      start = i;
      continue;
    }
    i += 1;
  }
  if (start < text.length) {
    row.push([start, text.length]);
    rows.push(row);
  }
  return rows;
}

function patchCsvEdge(text, flags) {
  const rows = csvFieldSpans(text);
  if (!rows.length) throw new Error('Slate CSV is empty');
  const header = rows[0].map(([a, b]) => text.slice(a, b));
  const column = header.indexOf('edge_flag');
  if (column < 0) throw new Error('Slate CSV has no edge_flag column');
  if (rows.length - 1 !== flags.length) throw new Error('Slate CSV does not match the slate');
  let out = '';
  let cursor = 0;
  rows.slice(1).forEach((row, index) => {
    const [a, b] = row[column];
    out += text.slice(cursor, a) + String(flags[index]);
    cursor = b;
  });
  return out + text.slice(cursor);
}

function jsonResponse(status, body) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  });
}

async function readEnvelope(path) {
  const response = await realFetch(path, { cache: 'no-cache' });
  if (!response.ok) return { status: 404, body: { error: 'Not available' } };
  return response.json();
}

async function staticFetch(url, init) {
  const path = staticPath(String(url));
  if (path === null) return realFetch(url, init);
  const envelope = await readEnvelope(path);
  let body = envelope.body;
  if (envelope.status === 200 && path.startsWith('/data/slate/')) {
    const threshold = parseThreshold(parseUrl(url).get('edge_threshold'));
    if (threshold.error) return jsonResponse(422, { error: threshold.error });
    body = { ...body, games: withEdge(body.games, threshold.value) };
  }
  return jsonResponse(envelope.status, body);
}

async function downloadSlateCsv(query) {
  const params = parseUrl(`?${query}`);
  const threshold = parseThreshold(params.get('edge_threshold'));
  if (threshold.error) throw new Error(threshold.error);
  const payload = await readEnvelope(staticPath(`/api/slate?${query}`));
  if (payload.status !== 200) throw new Error(payload.body.error || 'Slate not available');
  const csvResponse = await realFetch(staticPath(`/api/slate.csv?${query}`), { cache: 'no-cache' });
  if (!csvResponse.ok) throw new Error('Slate CSV not available');
  const flags = payload.body.games.map(game => edgeFlag(game, threshold.value));
  const csv = patchCsvEdge(await csvResponse.text(), flags);
  const week = String(params.get('week')).padStart(2, '0');
  const link = document.createElement('a');
  link.href = URL.createObjectURL(new Blob([csv], { type: 'text/csv' }));
  link.download = `slate_${params.get('season')}_wk${week}_${params.get('estimator')}.csv`;
  document.body.appendChild(link);
  link.click();
  link.remove();
}

// The dashboard's own download handler assigns window.location, which a static host
// cannot answer. This capture-phase listener runs first and replaces it, keeping the
// page's guard: only the slate currently on screen may be downloaded. It reads the
// page's top-level bindings (renderedSlateQuery, queryString, invalidateSlate) at
// click time, after the page script has declared them.
function onDownloadClick(event) {
  const target = event.target;
  if (!target || target.id !== 'download') return;
  event.stopImmediatePropagation();
  event.preventDefault();
  if (target.disabled) return;
  if (renderedSlateQuery === null || renderedSlateQuery !== queryString()) {
    invalidateSlate();
    return;
  }
  downloadSlateCsv(renderedSlateQuery).catch(error => {
    document.getElementById('message').textContent = error.message;
  });
}

globalThis.fetch = staticFetch;
if (globalThis.document) document.addEventListener('click', onDownloadClick, true);
globalThis.__staticSite = { parseUrl, staticPath, parseThreshold, withEdge, patchCsvEdge };
})();
"""
