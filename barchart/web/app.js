const pairsContainer = document.getElementById('pairs-container');
const statusElement = document.getElementById('messages');
const template = document.getElementById('pair-card-template');

let openAiConfigured = false;

function setStatus(message, isError = false) {
  statusElement.textContent = message;
  statusElement.classList.toggle('error', Boolean(isError));
}

async function loadOpenAIStatus() {
  try {
    const response = await fetch('/api/config/openai');
    if (!response.ok) {
      throw new Error(`Request failed with status ${response.status}`);
    }

    const payload = await response.json();
    openAiConfigured = Boolean(payload.has_api_key);
  } catch (error) {
    console.error('Unable to determine OpenAI configuration status:', error);
    openAiConfigured = false;
  }
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

function renderPairs(pairs) {
  pairsContainer.innerHTML = '';
  if (!pairs || pairs.length === 0) {
    pairsContainer.classList.add('empty');
    return;
  }

  pairsContainer.classList.remove('empty');

  pairs.forEach((pair) => {
    const fragment = template.content.cloneNode(true);
    const card = fragment.querySelector('.pair-card');
    card.dataset.pairId = pair.id;

    card.querySelector('.pair-title').textContent = pair.label || `${pair.ticker} ${pair.expiry}`;
    card.querySelector('.pair-side').textContent = pair.side_by_side;
    card.querySelector('.pair-greeks').textContent = pair.greeks;
    card.querySelector('.pair-updated').textContent = formatTimestamp(pair.last_modified);

    const processButton = card.querySelector('.process-button');
    const input = card.querySelector('.spot-input');
    const ivDirectionSelect = card.querySelector('.iv-direction-input');
    const insightsToggle = card.querySelector('.insights-checkbox');
    const insightsNote = card.querySelector('.insights-note');

    if (insightsToggle) {
      insightsToggle.checked = false;
    }

    if (insightsNote) {
      if (openAiConfigured) {
        insightsNote.textContent = 'AI insights will be requested when this option is checked.';
        insightsNote.classList.remove('warning');
      } else {
        insightsNote.textContent = 'Configure your OpenAI API key before enabling AI insights.';
        insightsNote.classList.add('warning');
      }
    }

    processButton.addEventListener('click', () => {
      handleProcess(pair, input, ivDirectionSelect, processButton, insightsToggle);
    });

    pairsContainer.appendChild(fragment);
  });
}

async function handleProcess(pair, input, ivDirectionSelect, button, insightsToggle) {
  const value = input.value.trim();
  if (!value) {
    input.focus();
    setStatus(`Enter a spot price for ${pair.label || pair.id} before processing.`, true);
    return;
  }

  const spotPrice = Number(value);
  if (!Number.isFinite(spotPrice)) {
    input.focus();
    setStatus('Spot price must be a valid number.', true);
    return;
  }

  const ivDirection = ivDirectionSelect?.value?.trim().toLowerCase();
  if (!ivDirection || !['up', 'down'].includes(ivDirection)) {
    ivDirectionSelect?.focus();
    setStatus('Select whether implied volatility is trending up or down.', true);
    return;
  }

  const generateInsights = Boolean(insightsToggle?.checked);

  button.disabled = true;
  button.textContent = 'Processing…';
  setStatus(`Processing ${pair.label || pair.id}…`);

  try {
    const response = await fetch('/api/process', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        pair_id: pair.id,
        spot_price: spotPrice,
        iv_direction: ivDirection,
        generate_insights: generateInsights,
      }),
    });

    const payload = await response.json();
    if (!response.ok || payload.error) {
      throw new Error(payload.error || `Request failed with status ${response.status}`);
    }

    const result = payload.result || {};

    const fallbackLabel = [pair.ticker, pair.expiry].filter(Boolean).join(' ').trim();

    sessionStorage.setItem(
      'latestProcessResult',
      JSON.stringify({
        pairDisplay: pair.label || fallbackLabel || pair.id,
        spotPrice,
        ivDirection,
        processedAt: new Date().toISOString(),
        result,
        generateInsights,
      }),
    );

    const resultsUrl = buildRelativeUrl('results.html');
    window.location.assign(resultsUrl);
    return;
  } catch (error) {
    console.error(error);
    setStatus(`Failed to process ${pair.label || pair.id}: ${error.message}`, true);
  } finally {
    button.disabled = false;
    button.textContent = 'Process';
  }
}

async function loadPairs() {
  setStatus('Loading available file pairs…');
  try {
    const response = await fetch('/api/pairs');
    if (!response.ok) {
      throw new Error(`Request failed with status ${response.status}`);
    }

    const payload = await response.json();
    renderPairs(payload.pairs || []);

    if (!payload.pairs || payload.pairs.length === 0) {
      setStatus('No unprocessed file pairs were found. Drop new CSVs into the input folder to begin.');
    } else {
      let message = 'Select a pair and provide the spot price to process the files.';
      if (!openAiConfigured) {
        message += ' Configure your OpenAI API key before enabling AI insights.';
      }
      setStatus(message);
    }
  } catch (error) {
    console.error(error);
    setStatus(`Unable to load file pairs: ${error.message}`, true);
  }
}

window.addEventListener('DOMContentLoaded', () => {
  (async () => {
    await loadOpenAIStatus();
    await loadPairs();
  })().catch((error) => {
    console.error('Initialisation failed:', error);
    setStatus(`Unable to initialise the page: ${error.message}`, true);
  });
});
