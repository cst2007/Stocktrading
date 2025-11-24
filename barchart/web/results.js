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
const marketStructureSection = document.getElementById('result-market-structure');
const marketStructureNameElement = document.getElementById('result-market-structure-name');
const marketStructurePlainElement = document.getElementById('result-market-structure-plain');
const marketStructureBehaviorElement = document.getElementById('result-market-structure-behavior');
const marketStructureNextElement = document.getElementById('result-market-structure-next');
const marketStructureEquationElement = document.getElementById('result-market-structure-equation');
const marketStructureFileContainer = document.getElementById('result-market-structure-file');
const marketStructureFileLinkElement = document.getElementById('result-market-structure-file-link');
const marketStructureFileContentElement = document.getElementById(
  'result-market-structure-file-content',
);
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

const MARKET_STRUCTURE_ENTRIES = [
  {
    names: ['Best Bullish (Long Adam)'],
    displayName: 'Best Bullish (Long Adam)',
    plainEnglish: 'Market wants to go up. Very stable. Dips get immediately bought.',
    expectedBehavior: 'Slow grind up, shallow pullbacks, strong dip buying.',
    nextStep: 'Find Gamma Box low → prepare long.',
    equation: 'Long_Setup = (Spot_Position < 0.25) AND (VEX_dir >= 0)',
  },
  {
    names: [
      'Dip Acceleration → Magnet Up (Conditional Adam)',
      'Dip-Acceleration → Magnet Up (Conditional Long Adam)',
    ],
    displayName: 'Dip Acceleration → Magnet Up (Conditional Adam)',
    plainEnglish: 'Dips accelerate downward but bounce even harder.',
    expectedBehavior: 'Quick flush → strong bounce → magnet pull upward.',
    nextStep: 'Wait for dip to reach Gamma Box bottom → time long.',
    equation: 'Dip_Buy_Zone = (spot <= Gamma_Box_Low + 0.1 * Gamma_Box_Width)\nLong_Setup = Dip_Buy_Zone AND (VEX_dir >= -1)',
  },
  {
    names: ['Upside Stall (No Adam)'],
    displayName: 'Upside Stall (No Adam)',
    plainEnglish: 'Market tries to rise but keeps hitting invisible ceiling.',
    expectedBehavior: 'Slow, choppy grind. Upside attempts get absorbed.',
    nextStep: 'Check box top for fade OR check breakout if VEX positive.',
    equation: 'Fade_Short = (Spot_Position > 0.75) AND (VEX_dir < 0)\nBreakout_Long = Valid_Breakout_Up',
  },
  {
    names: ['Low-Volatility Stall (Avoid Adam)'],
    displayName: 'Low-Volatility Stall (Avoid Adam)',
    plainEnglish: 'Market is stuck in glue. No trend, tiny candles.',
    expectedBehavior: 'Chop → small ranges → no direction.',
    nextStep: 'Only act when price reaches Gamma Box edges.',
    equation:
      'If 0.25 < Spot_Position < 0.75:\n    Avoid = True\nElse:\n    Trade_Edge = True',
  },
  {
    names: ['Support + Weak Down Magnet (Weak Long Scalp)'],
    displayName: 'Support + Weak Down Magnet (Weak Long Scalp)',
    plainEnglish: 'Market has some support but still drifts downward slightly.',
    expectedBehavior: 'Small bounces → weak upward corrections.',
    nextStep: 'Scalp long only if VEX positive.',
    equation: 'Long_Scalp = (Spot_Position < 0.25) AND (VEX_dir > 0)',
  },
  {
    names: ['Very Bearish (Strong Short Adam)'],
    displayName: 'Very Bearish (Strong Short Adam)',
    plainEnglish: 'Market naturally wants to fall. Rallies fail violently.',
    expectedBehavior: 'Slide → bounce → deeper slide → breakdown.',
    nextStep: 'Short at Gamma Box top when price pops.',
    equation: 'Short_Setup = (Spot_Position > 0.75) AND (VEX_dir < 0)',
  },
  {
    names: ['Fade Rises (No Adam)'],
    displayName: 'Fade Rises (No Adam)',
    plainEnglish: 'Every rise is sold. Controlled grind down.',
    expectedBehavior: 'Weak upside → consistent selling.',
    nextStep: 'Fade only at top of box.',
    equation: 'Fade_Short = (Spot_Position > 0.75) AND (GEX_state == "Pinning")',
  },
  {
    names: ['Pop → Slam Down (Short Adam)'],
    displayName: 'Pop → Slam Down (Short Adam)',
    plainEnglish: 'Market pops up to trick longs then slams down hard.',
    expectedBehavior: 'Small up-move → instant reversal → big drop.',
    nextStep: 'Wait for the pop into box top → short.',
    equation: 'Short_Setup = (spot >= Gamma_Box_High) AND (VEX_dir < 0)',
  },
  {
    names: ['Bullish Explosion (Fast Long Adam)'],
    displayName: 'Bullish Explosion (Fast Long Adam)',
    plainEnglish: 'Market can explode upward fast. Dealers forced to buy.',
    expectedBehavior: 'Vertical upside moves.',
    nextStep: 'Confirm breakout + positive VEX → long immediately.',
    equation: 'Long_Setup = Valid_Breakout_Up AND (VEX_dir > 0)',
  },
  {
    names: ['Volatility Whipsaw (Avoid Adam)'],
    displayName: 'Volatility Whipsaw (Avoid Adam)',
    plainEnglish: 'Chaotic. Both up and down moves accelerate. No pattern.',
    expectedBehavior: 'Whipsaws, fake-outs, unpredictable movement.',
    nextStep: 'Avoid. Only touch at extreme edges.',
    equation: 'Avoid = True\nUnless Spot_Position < 0.15 OR Spot_Position > 0.85',
  },
  {
    names: ['Uptrend With Brake (No Adam)', 'Uptrend + Brake (No Adam)'],
    displayName: 'Uptrend With Brake (No Adam)',
    plainEnglish: 'Uptrend exists but is capped. Slow and hesitant.',
    expectedBehavior: 'Slow grind → fails at resistance → might break eventually.',
    nextStep: 'Wait for breakout above box.',
    equation: 'Break_Brake = Valid_Breakout_Up',
  },
  {
    names: ['Short-Squeeze Blowout (Not Adam)'],
    displayName: 'Short-Squeeze Blowout (Not Adam)',
    plainEnglish: 'Unpredictable upside spikes. No safe risk control.',
    expectedBehavior: 'Vertical candles + violent jumps.',
    nextStep: 'Only fade at extreme extensions.',
    equation: 'Fade_Short = (IV > IV_threshold) AND (spot >> Gamma_Box_High)',
  },
  {
    names: ['Tug-of-War (No Adam)'],
    displayName: 'Tug-of-War (No Adam)',
    plainEnglish: 'Mixed structure. Nothing dominates yet.',
    expectedBehavior: 'Choppy drift → direction not clear.',
    nextStep: 'Whichever is stronger — VEX or TEX — determines tilt.',
    equation: 'If abs(VEX) > abs(TEX): Direction = sign(VEX_dir)\nElse: Direction = sign(TEX_dir)',
  },
  {
    names: ['Super-Magnet Down (Strongest Short Adam)'],
    displayName: 'Super-Magnet Down (Strongest Short Adam)',
    plainEnglish: 'Market is magnetically dragged downward. Strongest bearish setup.',
    expectedBehavior: 'Fast downtrend → overshoot → continuation.',
    nextStep: 'Short at Gamma Box upper boundary.',
    equation: 'Short_Setup = (Spot_Position > 0.75) AND (VEX_dir < 0)',
  },
  {
    names: ['Fake Up → Drop (Short Scalp)'],
    displayName: 'Fake Up → Drop (Short Scalp)',
    plainEnglish: 'Market lifts slightly (bait) then dumps.',
    expectedBehavior: 'Fake up → immediate rejection → drop.',
    nextStep: 'Wait for fake-up above box → short.',
    equation:
      'Fake_Up = (spot > Gamma_Box_High) AND (IV rising)\nShort_Setup = Fake_Up AND (VEX_dir < 0)',
  },
  {
    names: ['Volatile Downtrend (Short Adam Possible)'],
    displayName: 'Volatile Downtrend (Short Adam Possible)',
    plainEnglish: 'Market is moving down with big volatile spikes.',
    expectedBehavior: 'Pop → slam → continuation.',
    nextStep: 'Short all pullbacks to Gamma Box high.',
    equation: 'Short_Setup = (Spot_Position > 0.75) AND (VEX_dir < 0)',
  },
  {
    names: ['Volatility Box (Avoid ALL Setups)', 'Volatility Box (Avoid)'],
    displayName: 'Volatility Box (Avoid ALL Setups)',
    plainEnglish:
      'Market is wild and directionless. Both up and down moves accelerate.',
    expectedBehavior: 'Whipsaw → fake breakout → reverse → fake breakout again.',
    nextStep: 'Adam: Do NOT trade here.',
    equation: 'Avoid = True',
  },
  {
    names: ['Dream Bullish (Perfect Long Adam)'],
    displayName: 'Dream Bullish (Perfect Long Adam)',
    plainEnglish: 'Everything is aligned for a smooth, stable uptrend.',
    expectedBehavior: 'Dip → bounce → grind up for hours.',
    nextStep: 'Adam: The single most reliable long setup.',
    equation: 'Adam: The single most reliable long setup.',
  },
  {
    names: [
      'Super-Magnet Down (Perfect Short Adam)',
      'Negative–Negative Same Strike (Perfect Short Adam)',
    ],
    displayName: 'Super-Magnet Down (Perfect Short Adam)',
    plainEnglish: 'There is no support. Gamma + DEX combine to pull price straight down.',
    expectedBehavior: 'Pull → overshoot → failed base → continuation.',
    nextStep: 'Adam: One of the best short entries possible.',
    equation: 'Adam: One of the best short entries possible.',
  },
];

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

function normalizeStructureName(value) {
  return String(value || '').trim().toLowerCase();
}

function findMarketStructureDetails(marketState) {
  const normalized = normalizeStructureName(marketState);
  if (!normalized) {
    return null;
  }

  return (
    MARKET_STRUCTURE_ENTRIES.find((entry) =>
      entry.names.some((name) => normalizeStructureName(name) === normalized),
    ) || null
  );
}

function renderMarketStructure(ticker, marketState) {
  const normalizedTicker = normalizeStructureName(ticker);
  const isSpx = normalizedTicker === 'spx' || normalizedTicker.startsWith('spxw');
  const structure = findMarketStructureDetails(marketState);

  if (!isSpx || !structure) {
    marketStructureSection.hidden = true;
    marketStructureNameElement.textContent = '';
    marketStructurePlainElement.textContent = '';
    marketStructureBehaviorElement.textContent = '';
    marketStructureNextElement.textContent = '';
    marketStructureEquationElement.textContent = '';
    return;
  }

  marketStructureSection.hidden = false;
  marketStructureNameElement.textContent = structure.displayName;
  marketStructurePlainElement.textContent = structure.plainEnglish;
  marketStructureBehaviorElement.textContent = structure.expectedBehavior;
  marketStructureNextElement.textContent = structure.nextStep;
  marketStructureEquationElement.textContent = structure.equation;
}

async function renderMarketStructureFile(filePath, fileUrl) {
  marketStructureFileLinkElement.textContent = '';
  marketStructureFileContentElement.textContent = '';

  if (!filePath) {
    marketStructureFileContainer.hidden = true;
    return;
  }

  marketStructureFileContainer.hidden = false;
  renderFileLink(marketStructureFileLinkElement, filePath, fileUrl);

  if (!fileUrl) {
    marketStructureFileContentElement.textContent = 'Preview unavailable (no file URL).';
    return;
  }

  marketStructureFileContentElement.textContent = 'Loading preview…';

  try {
    const response = await fetch(fileUrl);
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    const text = await response.text();
    const trimmed = text.trim();
    marketStructureFileContentElement.textContent = trimmed || 'File is empty.';
  } catch (error) {
    console.error('Unable to load market_structure.txt', error);
    marketStructureFileContentElement.textContent = 'Unable to load file contents.';
  }
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
  const normalizedTicker = normalizeStructureName(tickerValue);
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

  renderMarketStructure(tickerValue, marketState);
  await renderMarketStructureFile(
    result.market_structure_txt,
    result.market_structure_txt_url,
  );

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
