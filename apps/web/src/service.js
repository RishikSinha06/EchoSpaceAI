import { buildGridScene } from './grid.js';
import { comparisonModels, showComparison } from './comparison.js';

// Reconstruction panel: talks to the local inference API through the Vite
// proxy at /svc (see vite.config.js). Only committed, hash-verified test
// samples can be chosen; the service refuses anything else.
const API = '/svc';

async function call(path, options) {
  const response = await fetch(API + path, options);
  const body = response.headers.get('content-type')?.includes('json') ? await response.json() : null;
  if (!response.ok) throw new Error(body?.detail ? (typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail)) : `HTTP ${response.status}`);
  return body;
}

function option(value, label) {
  const element = document.createElement('option');
  element.value = value; element.textContent = label;
  return element;
}

function download(url) {
  const a = document.createElement('a'); a.href = API + url; a.download = ''; a.click();
}

export async function attachReconstructionPanel(show, setStatus) {
  const anchor = [...document.querySelectorAll('.sidebar .section-label')].find((el) => el.textContent.includes('LAYERS'));
  const panel = document.createElement('div');
  panel.className = 'dataset-panel service-panel';
  panel.innerHTML = [
    '<div class="section-label">02 / RECONSTRUCTION</div>',
    '<label class="dataset-label" for="svc-model">Model</label><select id="svc-model"></select>',
    '<label class="dataset-label" for="svc-room">Held-out room</label><select id="svc-room"></select>',
    '<label class="dataset-label" for="svc-sample">Partial scan</label><select id="svc-sample"></select>',
    '<label class="dataset-label" for="svc-k">RIR measurements (K)</label>',
    '<select id="svc-k"><option value="1">1</option><option value="2">2</option><option value="4">4</option><option value="8" selected>8</option></select>',
    '<button class="primary" id="svc-run" type="button">Reconstruct <span>→</span></button>',
    '<button class="outline" id="svc-compare" type="button" disabled>Compare observation / A / D / truth</button>',
    '<div class="svc-downloads"><button class="subtle" id="svc-glb" disabled>GLB</button><button class="subtle" id="svc-grid" disabled>Grid JSON</button><button class="subtle" id="svc-scene" disabled>RoomScene</button><button class="subtle" id="svc-prov" disabled>Provenance</button></div>',
    '<p class="hint" id="svc-note">Connecting to the inference service…</p>',
  ].join('');
  anchor.before(panel);
  const $ = (id) => panel.querySelector('#' + id);
  let samples = [];
  let current = null;
  let registered = [];

  async function reconstruct(modelId, sampleId, k) {
    const result = await call('/v1/reconstructions', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ schema_version: 'echospace.reconstruction_request.v1', model_id: modelId, sample_id: sampleId, k }),
    });
    return { result, grid: await call(result.links.grid) };
  }
  function inspect(item) {
    const scene = buildGridScene(item.grid);
    current = item.result;
    show({ ...scene, kind: 'grid', name: item.grid.name, reconstruction: item.result });
    for (const id of ['svc-glb', 'svc-grid', 'svc-prov', 'svc-scene']) $(id).disabled = false;
  }

  function fillSamples() {
    const chosen = samples.filter((s) => s.room_id === $('svc-room').value);
    $('svc-sample').replaceChildren(...chosen.map((s) => option(s.sample_id, `${s.occlusion_type.replace('_', ' ')} · ${s.coverage_bin} coverage (${Math.round(s.coverage * 100)} %)`)));
  }
  $('svc-room').addEventListener('change', fillSamples);

  async function loadSamples() {
    const listing = await call(`/v1/samples?split=test&model_id=${encodeURIComponent($('svc-model').value)}&limit=5000`);
    samples = listing.samples;
    const rooms = [...new Set(samples.map((s) => s.room_id))];
    $('svc-room').replaceChildren(...rooms.map((room) => option(room, room)));
    fillSamples();
    return listing;
  }
  $('svc-model').addEventListener('change', () => loadSamples().catch((error) => setStatus(error.message, true)));

  $('svc-run').addEventListener('click', async () => {
    const button = $('svc-run');
    button.disabled = true;
    setStatus('Reconstructing…');
    try {
      const item = await reconstruct($('svc-model').value, $('svc-sample').value, Number($('svc-k').value));
      inspect(item);
      setStatus(`Reconstruction ${item.result.reconstruction_id} ready`);
    } catch (error) {
      setStatus(`Reconstruction failed: ${error.message}`, true);
    } finally {
      button.disabled = false;
    }
  });
  $('svc-compare').addEventListener('click', async () => {
    $('svc-compare').disabled = true;
    setStatus('Comparing A and D on the same verified scan…');
    try {
      const pair = comparisonModels(registered, $('svc-model').value);
      const sampleId = $('svc-sample').value, k = Number($('svc-k').value);
      const items = await Promise.all(pair.map((m) => reconstruct(m.model_id, sampleId, k)));
      showComparison(items, inspect);
      setStatus('Comparison ready: observation, A, D and ground truth');
    } catch (error) { setStatus(`Comparison failed: ${error.message}`, true); }
    finally { $('svc-compare').disabled = false; }
  });
  $('svc-glb').addEventListener('click', () => current && download(current.links.glb));
  $('svc-grid').addEventListener('click', () => current && download(current.links.grid));
  $('svc-prov').addEventListener('click', () => current && download(current.links.provenance));
  $('svc-scene').addEventListener('click', () => current && download(current.links.room_scene));

  try {
    const health = await call('/health');
    const { models } = await call('/v1/models');
    registered = models;
    const servable = models.filter((m) => m.available);
    $('svc-model').replaceChildren(...models.map((m) => {
      const element = option(m.model_id, `${m.family} ${m.variant} · fold ${m.fold}${m.available ? '' : ' (unavailable)'}`);
      element.disabled = !m.available;
      return element;
    }));
    if (!servable.length) throw new Error('No model can be served: ' + (models[0]?.unavailable_reason || 'registry empty'));
    $('svc-model').value = servable[0].model_id;
    try { comparisonModels(models, $('svc-model').value); $('svc-compare').disabled = false; } catch { /* Registry explains unavailable checkpoints. */ }
    const listing = await loadSamples();
    $('svc-note').textContent = `${listing.count} held-out test scans from fold ${listing.fold}; service ${health.inference_version} at ${health.code_revision}. Rooms the model trained on are refused.`;
  } catch (error) {
    $('svc-note').textContent = `Inference service unavailable (${error.message}). Start it with: uvicorn echospace.inference.api:app --port 8000`;
    for (const id of ['svc-model', 'svc-room', 'svc-sample', 'svc-k', 'svc-run']) $(id).disabled = true;
  }
}
