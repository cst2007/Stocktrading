const statusElement = document.getElementById('config-status');
const form = document.getElementById('openai-form');
const apiKeyInput = document.getElementById('api-key');
const clearButton = document.getElementById('clear-key');

let hasStoredKey = false;

function setStatus(message, isError = false) {
  statusElement.textContent = message;
  statusElement.classList.toggle('error', Boolean(isError));
}

async function loadConfiguration() {
  try {
    const response = await fetch('/api/config/openai');
    if (!response.ok) {
      throw new Error(`Request failed with status ${response.status}`);
    }

    const payload = await response.json();
    hasStoredKey = Boolean(payload.has_api_key);

    if (hasStoredKey) {
      setStatus('An OpenAI API key is configured for insight generation.');
    } else {
      setStatus('No OpenAI API key is currently stored. Provide one to enable AI insights.');
    }

    form.hidden = false;
    apiKeyInput.value = '';
  } catch (error) {
    console.error('Failed to load OpenAI configuration:', error);
    setStatus(`Unable to load configuration: ${error.message}`, true);
    form.hidden = true;
  }
}

async function updateApiKey(newKey) {
  const buttons = form.querySelectorAll('button');
  buttons.forEach((button) => {
    button.disabled = true;
  });

  setStatus('Saving OpenAI configuration…');

  try {
    const response = await fetch('/api/config/openai', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ api_key: newKey }),
    });

    const payload = await response.json();
    if (!response.ok || payload.error) {
      throw new Error(payload.error || `Request failed with status ${response.status}`);
    }

    hasStoredKey = Boolean(payload.has_api_key);
    apiKeyInput.value = '';

    if (hasStoredKey) {
      setStatus('OpenAI API key saved successfully.');
    } else {
      setStatus('Stored OpenAI API key cleared.');
    }
  } catch (error) {
    console.error('Failed to update OpenAI configuration:', error);
    setStatus(`Unable to update configuration: ${error.message}`, true);
  } finally {
    buttons.forEach((button) => {
      button.disabled = false;
    });
  }
}

form?.addEventListener('submit', async (event) => {
  event.preventDefault();
  const value = apiKeyInput.value.trim();
  if (!value) {
    apiKeyInput.focus();
    setStatus('Enter your OpenAI API key before saving.', true);
    return;
  }

  await updateApiKey(value);
});

clearButton?.addEventListener('click', async () => {
  await updateApiKey('');
});

window.addEventListener('DOMContentLoaded', () => {
  loadConfiguration();
});
