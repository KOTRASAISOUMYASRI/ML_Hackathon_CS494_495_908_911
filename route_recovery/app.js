const $ = id => document.getElementById(id);
const partition = $('partition'), trip = $('trip'), button = $('run');
const modelLabels = {
  distance: 'Distance baseline', decision_tree: 'Decision Tree',
  logistic_regression: 'Logistic Regression', knn: 'KNN',
  random_forest: 'Random Forest (baseline)', random_forest_optimized: 'Random Forest (optimized)',
  hist_gradient_boosting: 'HistGradientBoosting (baseline)',
  hist_gradient_boosting_optimized: 'HistGradientBoosting (previous F1 tuned)',
  hist_gradient_boosting_candidate_tuned: 'HistGradientBoosting (accuracy tuned)'
};
let inventory = {}, busy = false, modelInitialized = false;
$('api').value = localStorage.getItem('pathsnap-api') ||
  (location.port === '8080' ? `http://${location.hostname}:8000` : location.origin);

function base() {
  const url = new URL($('api').value);
  if (!['http:', 'https:'].includes(url.protocol)) throw Error('Use http or https');
  return url.href.replace(/\/$/, '');
}
function message(text, error = false) {
  $('status').textContent = text;
  $('status').className = error ? 'error' : '';
}
function populate() {
  const selected = trip.value;
  trip.replaceChildren(...(inventory[partition.value] || []).map(id => new Option(id, id)));
  if ([...trip.options].some(option => option.value === selected)) trip.value = selected;
  button.disabled = busy || !trip.options.length;
}
function setBusy(value) {
  busy = value;
  for (const id of ['connect', 'api', 'partition', 'trip', 'model', 'evaluate']) $(id).disabled = value;
  button.disabled = value || !trip.options.length;
}
function clearResults() {
  for (const id of ['results', 'gaps', 'evaluation']) $(id).hidden = true;
  $('map').srcdoc = '';
}
async function request(path, options = {}) {
  const response = await fetch(base() + path, {...options, signal: AbortSignal.timeout(180000)});
  if (!response.ok) {
    let detail = await response.text();
    try { detail = JSON.stringify(JSON.parse(detail).detail); } catch {}
    throw Error(`HTTP ${response.status}: ${detail}`);
  }
  return response.json();
}
async function connect() {
  clearResults(); $('selectedModel').hidden = true; inventory = {}; populate(); setBusy(true);
  try {
    const health = await request('/health');
    const selected = $('model').value;
    $('model').replaceChildren(...health.models.map(name => new Option(modelLabels[name] || name, name)));
    if (modelInitialized && health.models.includes(selected)) $('model').value = selected;
    else if (health.models.includes(health.default_model)) $('model').value = health.default_model;
    modelInitialized = true;
    inventory = await request('/trips');
    const comparison = await request('/selected-model');
    renderSelectedComparison(comparison);
    localStorage.setItem('pathsnap-api', base()); populate();
    if (!health.default_model_available) {
      throw Error('Selected HistGradientBoosting artifact is missing. Run git lfs pull on Laptop 1.');
    }
    message(trip.options.length ? 'Connected. Select a trip and emission model.' :
      'Connected, but no trips found. Check the server dataset directory.');
  } catch (error) {
    inventory = {}; populate();
    message(`Connection failed: ${error.message}. Check the address, server and allowed UI origin.`, true);
  } finally { setBusy(false); }
}
function cell(row, value, tag = 'td') {
  const element = document.createElement(tag); element.textContent = value; row.append(element);
}
function comparisonTable(independent, hmm, aggregate = false) {
  const table = document.createElement('table'), heading = document.createElement('tr');
  for (const label of ['Method', 'Top-1 (all observations)', aggregate ? 'Mean trip road-arc Macro F1' : 'Road-arc Macro F1 (this trip)', 'Continuity', 'Exact trips']) cell(heading, label, 'th');
  table.append(heading);
  for (const [label, m] of [['Classifier alone', independent], ['Classifier + HMM', hmm]]) {
    const row = document.createElement('tr');
    const top1 = aggregate ? m.top1_all_observations : m.top1_accuracy;
    const f1 = aggregate ? m.mean_trip_road_macro_f1 : m.road_arc_macro_f1;
    const continuity = aggregate ? m.continuity : m.continuity.fraction;
    const exact = aggregate ? m.exact_trip_matches : (m.exact_trip_match ? 'Yes' : 'No');
    for (const value of [label, (100*top1).toFixed(2)+'%', f1.toFixed(4),
      continuity === null ? 'N/A' : (100*continuity).toFixed(2)+'%', exact]) cell(row, value);
    table.append(row);
  }
  return table;
}
function renderSelectedComparison(data) {
  $('selectedModel').hidden = false;
  $('selectedTitle').textContent = `Saved comparison: ${data.label} before and after HMM`;
  $('selectedValues').replaceChildren();
  const note = document.createElement('p');
  if (!data.report_available) {
    note.textContent = data.message;
    $('selectedValues').append(note); return;
  }
  const s = data.summary, delta = data.changes.top1_percentage_points;
  $('selectedValues').append(comparisonTable(s.independent, s.hmm, true));
  note.textContent = `${s.trips} trips, ${s.observations.toLocaleString()} observations. Accuracy change: ${delta >= 0 ? '+' : ''}${delta.toFixed(2)} percentage points. Continuity change: +${data.changes.continuity_percentage_points.toFixed(2)} percentage points. ${data.scope}. ${data.message}`;
  $('selectedValues').append(note);
}
function renderEvaluation(e) {
  $('evaluation').hidden = !e;
  if (!e) return;
  const table = comparisonTable(e.independent, e.hmm);
  const note = document.createElement('p');
  note.textContent = `Candidate misses: ${e.candidate_misses}/${e.observations}. ${e.reference_method}.`;
  const delta = 100 * (e.hmm.top1_accuracy - e.independent.top1_accuracy);
  note.textContent += ` Accuracy change on this trip: ${delta >= 0 ? '+' : ''}${delta.toFixed(2)} percentage points.`;
  if (e.candidate) note.textContent += ` Candidate binary Macro F1: ${e.candidate.macro_f1.toFixed(4)}; Log Loss: ${e.candidate.log_loss.toFixed(4)}.`;
  $('evaluationValues').replaceChildren(table, note);
}
function render(result) {
  $('results').hidden = false; $('metrics').replaceChildren();
  for (const [label, value] of [['Inference', result.elapsed_seconds.toFixed(2)+' s'],
    ['Classifier', result.classifier_seconds.toFixed(2)+' s'],
    ['HMM decoder', result.decoder_seconds.toFixed(2)+' s'],
    ['Matched', result.matched_observations], ['Unmatched', result.unmatched_observations],
    ['Road sections', result.section_count], ['Breaks', result.breaks.length]]) {
    const box = document.createElement('div'); box.className = 'metric';
    const strong = document.createElement('strong'), caption = document.createElement('span');
    strong.textContent = value; caption.textContent = label; box.append(strong, caption); $('metrics').append(box);
  }
  $('gaps').hidden = false;
  const counts = {};
  for (const gap of result.breaks) { const key = gap.cause || gap.reason; counts[key] = (counts[key] || 0) + 1; }
  $('counts').textContent = result.breaks.length ? Object.entries(counts).map(([k,v]) => `${k.replaceAll('_',' ')}: ${v}`).join(' | ') : 'No route breaks.';
  $('gaprows').replaceChildren();
  for (const gap of result.breaks) {
    const row = document.createElement('tr');
    for (const text of [result.observation_ids[gap.observation], gap.reason, gap.cause || '-']) cell(row, text);
    $('gaprows').append(row);
  }
  renderEvaluation(result.evaluation);
}
$('connect').onclick = connect;
partition.onchange = () => { clearResults(); populate(); };
button.onclick = async () => {
  // Snapshot BEFORE changing controls; rebuilding the dropdown used to reset the trip.
  const payload = {partition: partition.value, trip_id: trip.value, model: $('model').value,
    evaluate: $('evaluate').checked, diagnose: true};
  clearResults(); setBusy(true); const started = Date.now(); message('Reconstructing route...');
  const timer = setInterval(() => message(`Reconstructing route: ${Math.floor((Date.now()-started)/1000)} seconds`), 1000);
  try {
    const data = await request('/visualize', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload)});
    $('map').srcdoc = data.map_html; render(data.result);
    message(`${data.result.trip_id} | ${data.result.model} | ${data.result.section_count} section(s), ${data.result.breaks.length} break(s).`);
  } catch (error) {
    message(`Request failed: ${error.message}. A browser timeout does not cancel server computation.`, true);
  } finally { clearInterval(timer); setBusy(false); }
};
connect();
