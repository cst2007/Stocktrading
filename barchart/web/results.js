const statusElement = document.getElementById('results-status');
const overviewSection = document.getElementById('result-overview');
const tabsSection = document.getElementById('result-tabs');
const tabListElement = tabsSection.querySelector('.tabs');
const tabPanelsElement = tabsSection.querySelector('.tab-panels');
const backButton = document.getElementById('back-button');

const pairElement = document.getElementById('result-pair');
const timestampElement = document.getElementById('result-timestamp');
const spotElement = document.getElementById('result-spot');
const combinedElement = document.getElementById('result-combined');
const derivedElement = document.getElementById('result-derived');
const insightJsonElement = document.getElementById('result-insight-json');
const movedListElement = document.getElementById('result-moved-list');
const insightSection = document.getElementById('result-insight');
const insightModelElement = document.getElementById('result-insight-model');
const insightLatencyElement = document.getElementById('result-insight-latency');
const insightPromptElement = document.getElementById('result-insight-prompt');
const insightResponseElement = document.getElementById('result-insight-response');

function setStatus(message, isError = false) {
  statusElement.textContent = message;
  statusElement.classList.toggle('error', Boolean(isError));
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

function renderOverview(data) {
  overviewSection.hidden = false;
  pairElement.textContent = data.pairDisplay || 'Unknown pair';
  timestampElement.textContent = formatTimestamp(data.processedAt);
  spotElement.textContent = Number.isFinite(data.spotPrice) ? data.spotPrice : 'Unknown';

  const result = data.result || {};
  combinedElement.textContent = result.combined_csv || 'Unavailable';
  derivedElement.textContent = result.derived_csv || 'Unavailable';
  insightJsonElement.textContent = result.insights?.insight_json || 'Unavailable';

  if (result.insights) {
    insightSection.hidden = false;
    insightModelElement.textContent = result.insights.model || 'Unknown';
    const latency = Number(result.insights.latency_ms);
    insightLatencyElement.textContent = Number.isFinite(latency) ? latency.toFixed(2) : 'Unknown';
    insightPromptElement.textContent = result.insights.prompt || 'Unavailable';
    insightResponseElement.textContent = result.insights.response || 'Unavailable';
  } else {
    insightSection.hidden = true;
    insightModelElement.textContent = '';
    insightLatencyElement.textContent = '';
    insightPromptElement.textContent = '';
    insightResponseElement.textContent = '';
  }

  movedListElement.innerHTML = '';
  const movedFiles = Array.isArray(result.moved_files) ? result.moved_files : [];
  if (movedFiles.length === 0) {
    const item = document.createElement('li');
    item.textContent = 'No files were moved.';
    movedListElement.appendChild(item);
  } else {
    movedFiles.forEach((filePath) => {
      const item = document.createElement('li');
      const code = document.createElement('code');
      code.textContent = filePath;
      item.appendChild(code);
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

    const addListItem = (label, value) => {
      const item = document.createElement('li');
      const labelElement = document.createElement('span');
      labelElement.className = 'summary-label';
      labelElement.textContent = `${label}:`;
      item.appendChild(labelElement);

      const valueElement = document.createElement('code');
      valueElement.textContent = value || 'Unavailable';
      item.appendChild(valueElement);

      list.appendChild(item);
    };

    addListItem('Summary JSON', summary.summary_json);
    addListItem('Per-strike CSV', summary.per_strike_csv);
    if (summary.chart) {
      addListItem('Chart', summary.chart);
    }

    panel.appendChild(list);
    tabPanelsElement.appendChild(panel);
  });

  activateTab(0);
}

function loadResult() {
  const raw = sessionStorage.getItem('latestProcessResult');
  if (!raw) {
    setStatus('No recent processing results were found. Process a pair to view its outputs.', true);
    overviewSection.hidden = true;
    tabsSection.hidden = true;
    return;
  }

  try {
    const data = JSON.parse(raw);
    setStatus('Processing run loaded successfully.');
    renderOverview(data);
    renderTabs(data);
  } catch (error) {
    console.error(error);
    setStatus('Unable to read the stored results. Try processing the files again.', true);
    overviewSection.hidden = true;
    tabsSection.hidden = true;
  }
}

backButton.addEventListener('click', () => {
  const indexUrl = buildRelativeUrl('index.html');
  window.location.assign(indexUrl);
});

window.addEventListener('DOMContentLoaded', () => {
  loadResult();
});
