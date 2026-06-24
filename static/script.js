const queryForm = document.getElementById('query-form');
const queryInput = document.getElementById('query-input');
const resultsEl = document.getElementById('results');
const emptyState = document.getElementById('empty-state');
const loadingEl = document.getElementById('loading');
const parsedInfoEl = document.getElementById('parsed-info');

const toggleManualBtn = document.getElementById('toggle-manual');
const manualPanel = document.getElementById('manual-panel');
const manualSubmitBtn = document.getElementById('manual-submit');

// --- Manual sliders ---
toggleManualBtn.addEventListener('click', () => {
  manualPanel.classList.toggle('hidden');
});

document.querySelectorAll('.slider-row input[type="range"]').forEach(input => {
  input.addEventListener('input', (e) => {
    e.target.parentElement.querySelector('.slider-val').textContent = e.target.value;
  });
});

manualSubmitBtn.addEventListener('click', () => {
  const weights = {};
  document.querySelectorAll('.slider-row').forEach(row => {
    const criterion = row.dataset.criterion;
    const value = row.querySelector('input[type="range"]').value;
    weights[criterion] = parseInt(value, 10);
  });
  fetchRecommendation({ weights });
});

// --- Free text form ---
queryForm.addEventListener('submit', (e) => {
  e.preventDefault();
  const message = queryInput.value.trim();
  if (!message) return;
  fetchRecommendation({ message });
});

async function fetchRecommendation(payload) {
  setLoading(true);
  parsedInfoEl.classList.add('hidden');
  resultsEl.innerHTML = '';
  emptyState.classList.add('hidden');

  try {
    const res = await fetch('/api/recommend', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    setLoading(false);

    if (data.error) {
      resultsEl.innerHTML = `<p style="color:#A9A192;">${data.error}</p>`;
      return;
    }

    if (data.parsed) {
      renderParsedInfo(data.parsed);
    }

    if (!data.results || data.results.length === 0) {
      resultsEl.innerHTML = `<p style="color:#A9A192;">${data.message || 'No matches found.'}</p>`;
      return;
    }

    renderResults(data.results);
  } catch (err) {
    setLoading(false);
    resultsEl.innerHTML = `<p style="color:#A9A192;">Something went wrong: ${err.message}</p>`;
  }
}

function setLoading(isLoading) {
  loadingEl.classList.toggle('hidden', !isLoading);
}

function renderParsedInfo(parsed) {
  const weightsStr = Object.entries(parsed.weights)
    .map(([k, v]) => `${k}: ${v}`)
    .join(' · ');
  let html = `<strong>Parsed priorities</strong> &nbsp; ${weightsStr}`;
  if (parsed.filters && Object.keys(parsed.filters).length > 0) {
    const filtersStr = Object.entries(parsed.filters)
      .map(([k, v]) => `${k}: ${v}`)
      .join(' · ');
    html += `<br><strong>Hard filters</strong> &nbsp; ${filtersStr}`;
  }
  parsedInfoEl.innerHTML = html;
  parsedInfoEl.classList.remove('hidden');
}

function renderResults(results) {
  resultsEl.innerHTML = results.map((r, i) => `
    <div class="result-card ${i === 0 ? 'rank-1' : ''}">
      <div class="result-top">
        <div>
          <div class="result-rank">Rank ${i + 1}</div>
          <div class="result-name">${r.name}</div>
        </div>
        <div class="result-score">${r.score}<span>/100</span></div>
      </div>
      <div class="breakdown">
        ${r.breakdown.map(b => `
          <div class="breakdown-row">
            <div class="breakdown-label">${b.criterion}</div>
            <div class="breakdown-bar-track">
              <div class="breakdown-bar-fill" style="width: ${b.suitability}%;"></div>
            </div>
            <div class="breakdown-value">${b.suitability.toFixed(0)}</div>
          </div>
        `).join('')}
      </div>
    </div>
  `).join('');
}
