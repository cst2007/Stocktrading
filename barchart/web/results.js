const statusElement = document.getElementById('results-status');
const overviewSection = document.getElementById('result-overview');
const tabsSection = document.getElementById('result-tabs');
const tabListElement = tabsSection.querySelector('.tabs');
const tabPanelsElement = tabsSection.querySelector('.tab-panels');
const backButton = document.getElementById('back-button');

const pairElement = document.getElementById('result-pair');
const timestampElement = document.getElementById('result-timestamp');
const spotElement = document.getElementById('result-spot');
const ivDirectionElement = document.getElementById('result-iv-direction');
const marketStateElement = document.getElementById('result-market-state');
const marketDescriptionElement = document.getElementById('result-market-description');
const marketStateContainer = document.getElementById('result-spx-market-state');
const marketDescriptionContainer = document.getElementById('result-spx-market-description');
const combinedElement = document.getElementById('result-combined');
const derivedElement = document.getElementById('result-derived');
const sideElement = document.getElementById('result-side');
const reactivityElement = document.getElementById('result-reactivity');
const insightJsonElement = document.getElementById('result-insight-json');
const movedListElement = document.getElementById('result-moved-list');
const insightSection = document.getElementById('result-insight');
const insightModelElement = document.getElementById('result-insight-model');
const insightLatencyElement = document.getElementById('result-insight-latency');
const insightPromptElement = document.getElementById('result-insight-prompt');
const insightResponseElement = document.getElementById('result-insight-response');
function normalizeResultPayload(rawPayload) {
  if (!rawPayload || typeof rawPayload !== 'object') {
    return null;
  }

  if (
    rawPayload.result &&
    !rawPayload.pairDisplay &&
    !rawPayload.processedAt &&
    rawPayload.spotPrice === undefined &&
    rawPayload.ivDirection === undefined
  ) {
    return rawPayload.result;
  }

  return rawPayload;
}

function setStatus(message, isError = false) {
  statusElement.textContent = message;
  statusElement.classList.toggle('error', Boolean(isError));
}

function renderFileLink(element, filePath, fileUrl) {
  element.innerHTML = '';

  if (!filePath) {
    element.textContent = 'Unavailable';
    return;
  }

  if (fileUrl) {
    const link = document.createElement('a');
    link.href = fileUrl;
    link.textContent = filePath;
    link.target = '_blank';
    link.rel = 'noopener noreferrer';
    element.appendChild(link);
    return;
  }

  element.textContent = filePath;
}

function formatTimestamp(value) {
  if (!value) {
    return 'Unknown';
  }

  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }
  return date.toLocaleString();
}

function buildRelativeUrl(targetFile) {
  const { origin, pathname } = window.location;
  let basePath = pathname;

  if (!basePath.endsWith('/')) {
    const lastSlashIndex = basePath.lastIndexOf('/');
    const trailingSegment = basePath.slice(lastSlashIndex + 1);

    if (trailingSegment && !trailingSegment.includes('.')) {
      basePath = `${basePath}/`;
    } else {
      basePath = basePath.slice(0, lastSlashIndex + 1);
    }
  }

  if (!basePath.endsWith('/')) {
    basePath += '/';
  }

  return `${origin}${basePath}${targetFile}`;
}

function normalizeTicker(value) {
  return String(value || '').trim().toLowerCase();
}

async function renderOverview(data) {
  const result = data.result || {};
  overviewSection.hidden = false;
  pairElement.textContent = data.pairDisplay || 'Unknown pair';
  timestampElement.textContent = formatTimestamp(data.processedAt);
  spotElement.textContent = Number.isFinite(data.spotPrice) ? data.spotPrice : 'Unknown';
  const directionValue = String(data.ivDirection || result.iv_direction || '').toLowerCase();
  const directionLabels = {
    up: 'Up',
    down: 'Down',
    unknown: 'Unknown',
  };
  ivDirectionElement.textContent = directionLabels[directionValue] || 'Unknown';

  const tickerValue = result.pair?.ticker || '';
  const normalizedTicker = normalizeTicker(tickerValue);
  const isSpx = normalizedTicker === 'spx' || normalizedTicker.startsWith('spxw');

  const marketState = result.market_state || '';
  const marketDescription = result.market_state_description || '';

  if (isSpx && marketState) {
    marketStateContainer.hidden = false;
    marketDescriptionContainer.hidden = false;
    marketStateElement.textContent = marketState;
    marketDescriptionElement.textContent =
      marketDescription || 'No description available.';
  } else {
    marketStateContainer.hidden = true;
    marketDescriptionContainer.hidden = true;
    marketStateElement.textContent = '';
    marketDescriptionElement.textContent = '';
  }

  renderFileLink(combinedElement, result.combined_csv, result.combined_csv_url);
  renderFileLink(derivedElement, result.derived_csv, result.derived_csv_url);
  renderFileLink(sideElement, result.side_csv, result.side_csv_url);
  renderFileLink(reactivityElement, result.reactivity_csv, result.reactivity_csv_url);
  renderFileLink(
    insightJsonElement,
    result.insights?.insight_json,
    result.insights?.insight_json_url,
  );

  if (result.insights) {
    insightSection.hidden = false;
    insightModelElement.textContent = result.insights.model || 'Unknown';
    const latency = Number(result.insights.latency_ms);
    insightLatencyElement.textContent = Number.isFinite(latency) ? latency.toFixed(2) : 'Unknown';
    renderFileLink(insightPromptElement, result.insights.prompt, result.insights.prompt_url);
    renderFileLink(
      insightResponseElement,
      result.insights.response,
      result.insights.response_url,
    );
  } else {
    insightSection.hidden = true;
    insightModelElement.textContent = '';
    insightLatencyElement.textContent = '';
    insightPromptElement.textContent = '';
    insightResponseElement.textContent = '';
  }

  movedListElement.innerHTML = '';
  let movedFiles = [];
  if (Array.isArray(result.moved_files_info)) {
    movedFiles = result.moved_files_info;
  } else if (Array.isArray(result.moved_files)) {
    movedFiles = result.moved_files.map((path) => ({ path, url: null }));
  }
  if (movedFiles.length === 0) {
    const item = document.createElement('li');
    item.textContent = 'No files were moved.';
    movedListElement.appendChild(item);
  } else {
    movedFiles.forEach((entry) => {
      const { path: filePath, url } = entry;
      const item = document.createElement('li');
      if (filePath && url) {
        const link = document.createElement('a');
        link.href = url;
        link.textContent = filePath;
        link.target = '_blank';
        link.rel = 'noopener noreferrer';
        item.appendChild(link);
      } else {
        const code = document.createElement('code');
        code.textContent = filePath || 'Unavailable';
        item.appendChild(code);
      }
      movedListElement.appendChild(item);
    });
  }
}

function renderTabs(data) {
  const summaries = Array.isArray(data.result?.summaries) ? data.result.summaries : [];

  if (summaries.length === 0) {
    tabsSection.hidden = true;
    return;
  }

  tabsSection.hidden = false;
  tabListElement.innerHTML = '';
  tabPanelsElement.innerHTML = '';

  const activateTab = (index) => {
    const tabButtons = tabListElement.querySelectorAll('[role="tab"]');
    const panels = tabPanelsElement.querySelectorAll('[role="tabpanel"]');

    tabButtons.forEach((button, buttonIndex) => {
      const isSelected = buttonIndex === index;
      button.setAttribute('aria-selected', isSelected ? 'true' : 'false');
      button.tabIndex = isSelected ? 0 : -1;
      panels[buttonIndex].hidden = !isSelected;
    });
  };

  summaries.forEach((summary, index) => {
    const tabId = `tab-${index}`;
    const panelId = `panel-${index}`;

    const tabButton = document.createElement('button');
    tabButton.type = 'button';
    tabButton.role = 'tab';
    tabButton.id = tabId;
    tabButton.className = 'tab-button';
    tabButton.setAttribute('aria-controls', panelId);
    tabButton.textContent = `${summary.ticker || 'Unknown'} ${summary.expiry || ''}`.trim();
    if (!tabButton.textContent) {
      tabButton.textContent = `Set ${index + 1}`;
    }

    tabButton.addEventListener('click', () => {
      activateTab(index);
      tabButton.focus();
    });

    tabListElement.appendChild(tabButton);

    const panel = document.createElement('div');
    panel.role = 'tabpanel';
    panel.id = panelId;
    panel.className = 'tab-panel';
    panel.setAttribute('aria-labelledby', tabId);
    panel.hidden = true;

    const heading = document.createElement('h3');
    heading.textContent = tabButton.textContent;
    panel.appendChild(heading);

    const list = document.createElement('ul');
    list.className = 'summary-list';

    const addListItem = (label, value, url) => {
      const item = document.createElement('li');
      const labelElement = document.createElement('span');
      labelElement.className = 'summary-label';
      labelElement.textContent = `${label}:`;
      item.appendChild(labelElement);

      if (value && url) {
        const link = document.createElement('a');
        link.href = url;
        link.textContent = value;
        link.target = '_blank';
        link.rel = 'noopener noreferrer';
        item.appendChild(link);
      } else {
        const valueElement = document.createElement('code');
        valueElement.textContent = value || 'Unavailable';
        item.appendChild(valueElement);
      }

      list.appendChild(item);
    };

    addListItem('Summary JSON', summary.summary_json, summary.summary_json_url);
    addListItem(
      'Per-strike CSV',
      summary.per_strike_csv,
      summary.per_strike_csv_url,
    );
    if (summary.chart) {
      addListItem('Chart', summary.chart, summary.chart_url);
    }

    panel.appendChild(list);
    tabPanelsElement.appendChild(panel);
  });

  activateTab(0);
}

async function loadResult() {
  const raw = sessionStorage.getItem('latestProcessResult');
  if (raw) {
    try {
      const stored = JSON.parse(raw);
      const data = normalizeResultPayload(stored);
      if (!data) {
        throw new Error('Stored results were invalid.');
      }
      setStatus('Processing run loaded successfully.');
      await renderOverview(data);
      renderTabs(data);
      return;
    } catch (error) {
      console.error(error);
      sessionStorage.removeItem('latestProcessResult');
      setStatus('Stored results were corrupted. Attempting to load the most recent run…', true);
    }
  } else {
    setStatus('Looking for the most recent processing run…');
  }

  overviewSection.hidden = true;
  tabsSection.hidden = true;

  try {
    const response = await fetch('/api/results/latest');
    const payload = await response.json();

    if (!response.ok) {
      const message = payload?.error || 'Unable to load the most recent processing run.';
      throw new Error(message);
    }

    const data = normalizeResultPayload(payload?.result ?? payload);
    if (!data || typeof data !== 'object') {
      throw new Error('Latest processing payload was malformed.');
    }

    sessionStorage.setItem('latestProcessResult', JSON.stringify(data));

    setStatus('Processing run loaded successfully.');
    await renderOverview(data);
    renderTabs(data);
  } catch (error) {
    console.error(error);
    setStatus(
      error.message ||
        'No recent processing results were found. Process a pair to view its outputs.',
      true,
    );
  }
}

backButton.addEventListener('click', () => {
  const indexUrl = buildRelativeUrl('index.html');
  window.location.assign(indexUrl);
});

window.addEventListener('DOMContentLoaded', () => {
  loadResult();
});
