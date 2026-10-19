import * as THREE from 'three';
import { buildGridScene } from './grid.js';
import { showComparison } from './comparison.js';
import './simulation.css';
const PREFIX = '/svc/v1/simulation';
async function request(path, options = {}, simulation = true) {
  const response = await fetch((simulation ? PREFIX : '/svc') + path, options);
  const value = await response.json();
  if (!response.ok) throw new Error(typeof value.detail === 'string' ? value.detail : JSON.stringify(value.detail || value));
  return value;
}
const post = (body) => ({ method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
const number = (id, label, value, min, max, step = .1) => `<label>${label}<input id="lab-${id}" type="number" value="${value}" min="${min}" max="${max}" step="${step}"></label>`;
const defaults = { source_m: [2.8, 1.2, 2.5], receiver_m: [3.8, 1.2, 2.5] };
export function roomVertices(shape, w, d) {
  return shape === 'rectangle' ? [[0, 0], [w, 0], [w, d], [0, d]] : [[0, 0], [w, 0], [w, .55 * d], [.55 * w, .55 * d], [.55 * w, d], [0, d]];
}
export function observationGrid(capture, reference = null) {
  return { format: 'echospace.grid.v2', ...capture.grid, name: 'Sensor observation · unknown remains unknown',
    interior: capture.observed_free, observed: capture.observed, observed_free: capture.observed_free,
    observed_wall: capture.observed_wall, wall_height_m: 2.7, wall_height_assumed: true,
    measurements: capture.positions_scene_m.map((p) => ({ source_scene_m: p[0], receiver_scene_m: p[1] })),
    ...(reference ? { truth: { interior: reference.target_occupancy, boundary: reference.target_boundary } } : {}) };
}
function paintGrid(canvas, capture, reference = null) {
  canvas.width = canvas.height = 320; const c = canvas.getContext('2d');
  c.fillStyle = '#101f2a'; c.fillRect(0, 0, 320, 320); if (!capture) return;
  for (let row = 0; row < 64; row++) for (let col = 0; col < 64; col++) {
    c.fillStyle = reference ? reference.target_occupancy[row][col] ? '#f4f1de' : '#101f2a' :
      capture.observed_wall[row][col] ? '#20c8c1' : capture.observed_free[row][col] ? '#315b62' : '#101f2a';
    c.fillRect(col * 5, row * 5, 5, 5);
  }
}
function paintSignal(canvas, values = [], color = '#20c8c1') {
  canvas.width = 640; canvas.height = 140; const c = canvas.getContext('2d');
  c.fillStyle = '#101f2a'; c.fillRect(0, 0, 640, 140);
  c.strokeStyle = '#334752'; c.beginPath(); c.moveTo(0, 70); c.lineTo(640, 70); c.stroke();
  const peak = Math.max(1e-9, ...values.map(Math.abs)); c.strokeStyle = color; c.beginPath();
  const step = Math.max(1, Math.floor(values.length / 640));
  for (let i = 0; i < values.length; i += step) { const x = i / Math.max(1, values.length - 1) * 639; const y = 70 - values[i] / peak * 60; if (!i) c.moveTo(x, y); else c.lineTo(x, y); }
  c.stroke(); c.fillStyle = '#9db5bc'; c.font = '14px sans-serif';
  c.fillText(values.length ? `0–${(values.length / 16).toFixed(1)} ms · peak ${peak.toFixed(4)}` : 'Capture to inspect the signal', 10, 132);
}
function setupScene(config) {
  const group = new THREE.Group(); const room = config.room;
  const vertices = roomVertices(room.shape, room.width_m, room.depth_m);
  const shape = new THREE.Shape(vertices.map(([x, z]) => new THREE.Vector2(x, z)));
  const floor = new THREE.Mesh(new THREE.ShapeGeometry(shape), new THREE.MeshStandardMaterial({ color: 0x243746, side: THREE.DoubleSide })); floor.rotation.x = Math.PI / 2; group.add(floor);
  vertices.forEach((a, i) => { const b = vertices[(i + 1) % vertices.length]; const dx = b[0] - a[0], dz = b[1] - a[1];
    const wall = new THREE.Mesh(new THREE.BoxGeometry(Math.hypot(dx, dz), room.height_m, .06), new THREE.MeshStandardMaterial({ color: 0x7a9ca8, transparent: true, opacity: .35 }));
    wall.position.set((a[0] + b[0]) / 2, room.height_m / 2, (a[1] + b[1]) / 2); wall.rotation.y = -Math.atan2(dz, dx); group.add(wall);
  });
  const marker = (position, color) => { const mesh = new THREE.Mesh(new THREE.SphereGeometry(.11, 14, 10), new THREE.MeshStandardMaterial({ color })); mesh.position.set(...position); group.add(mesh); };
  marker(config.scanner.position_m, 0x20c8c1); config.pairs.forEach((p) => { marker(p.source_m, 0xff6b6b); marker(p.receiver_m, 0x5fa8ff); });
  return { group, layers: {}, kind: 'mesh', name: 'Virtual room reference · not model input', stats: { width_m: room.width_m, depth_m: room.depth_m, height_m: room.height_m } };
}
export function attachSimulationLab(show, setStatus) {
  const launch = document.createElement('button'); launch.className = 'lab-launch'; launch.textContent = 'Open sensor lab'; document.querySelector('.topbar').append(launch);
  let dialog;
  launch.addEventListener('click', async () => {
    if (dialog?.open) return;
    dialog = document.createElement('dialog'); dialog.className = 'sensor-lab'; dialog.setAttribute('aria-labelledby', 'lab-title');
    dialog.innerHTML = `<header class="lab-header"><div><span class="lab-eyebrow">SOFTWARE SENSOR LAB / V1</span><h2 id="lab-title">Capture the room. Inspect the evidence.</h2><p>Virtual sensors and audited replay. Synthetic captures are diagnostics, never held-out benchmark results.</p></div><button id="lab-close" aria-label="Close sensor lab">Close</button></header>
      <div class="lab-toolbar"><label>Acquisition mode<select id="lab-mode"><option value="virtual">Interactive virtual room</option><option value="replay">Audited dataset replay</option></select></label><button class="lab-primary" id="lab-capture">Capture sensors</button><button id="lab-next" hidden>Next sample</button><button id="lab-play" hidden>Play replay</button><label id="lab-speed-wrap" hidden>Playback<select id="lab-speed"><option value=".5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option></select></label><span id="lab-state" class="lab-chip">CONFIGURED</span></div>
      <p class="lab-status" id="lab-status" role="status">Loading the local simulation engine…</p>
      <div class="lab-grid"><section class="lab-controls">
        <div id="lab-replay-fields" hidden><h3>Verified replay sample</h3><label>Held-out sample<select id="lab-replay-sample"></select></label><p class="lab-note">Replays original masks, waveforms and positions. Playback speed never changes audio timing.</p></div>
        <div id="lab-virtual-fields"><h3>Room setup <span>metres</span></h3><label>Footprint<select id="lab-shape"><option value="rectangle">Rectangle</option><option value="l_room">L-shaped room</option></select></label>
          <div class="lab-fields">${number('width', 'Width X', 6, 3, 10)}${number('depth', 'Depth Z', 5, 3, 10)}${number('height', 'Height Y', 2.8, 2, 5)}${number('reflection', 'Reflection', .65, 0, .98, .05)}</div>
          <h3>Scanner</h3><div class="lab-fields">${number('scan-x', 'X', 2, 0, 10)}${number('scan-z', 'Z', 2.5, 0, 10)}${number('scan-y', 'Height', 1.1, .1, 4)}${number('heading', 'Heading °', 0, -360, 360, 5)}${number('fov', 'Field of view °', 120, 10, 360, 10)}${number('range', 'Range', 12, .5, 20)}${number('rays', 'Rays', 361, 16, 1441, 1)}${number('dropout', 'Dropout', 0, 0, .8, .05)}${number('range-noise', 'Range σ', 0, 0, .2, .01)}</div>
          <h3>Acoustic pair</h3><label>Selected pair<select id="lab-pair"></select></label><div class="lab-fields">${number('source-x', 'Speaker X', 2.8, 0, 10)}${number('source-z', 'Speaker Z', 2.5, 0, 10)}${number('source-y', 'Speaker height', 1.2, .1, 4)}${number('receiver-x', 'Microphone X', 3.8, 0, 10)}${number('receiver-z', 'Microphone Z', 2.5, 0, 10)}${number('receiver-y', 'Mic height', 1.2, .1, 4)}</div>
          <div class="lab-actions"><button id="lab-add-pair">Add pair</button><button id="lab-remove-pair">Remove pair</button></div>
          <details><summary>Device effects and timing</summary><label>Excitation<select id="lab-excitation"><option value="impulse">Impulse response capture</option><option value="sweep">Sweep + deconvolution</option></select></label><div class="lab-fields">${number('snr', 'SNR dB · 80=off', 80, 0, 80, 1)}${number('gain', 'ADC input gain dB', 0, -12, 12, 1)}${number('lowpass', 'Device low-pass Hz · 0=off', 0, 0, 7900, 100)}${number('offset', 'Timing offset ms', 0, -2, 2, .1)}${number('drift', 'Clock drift ppm', 0, -500, 500, 10)}${number('bits', 'ADC bits', 16, 8, 24, 1)}${number('clip', 'Clip level', 1, .05, 2, .05)}${number('pose-noise', 'Pose error ±m', 0, 0, .1, .01)}${number('seed', 'Random seed', 0, 0, 2147483647, 1)}</div><p class="lab-note">Gain precedes ADC and peak normalization. Linear gain may normalize away; clipping/noise can still alter the response.</p></details>
        </div><label>Requested K<select id="lab-k"><option>1</option><option>2</option><option>4</option><option>8</option></select></label>
      </section><section class="lab-stage"><div class="lab-card"><div class="lab-card-heading"><h3>Scene setup <span>reference, not model input</span></h3><button id="lab-room-3d">View room in 3D</button></div><svg id="lab-plan" role="img" aria-label="Virtual room plan: drag the scanner, speaker or microphone" viewBox="0 0 640 450"></svg><p class="lab-note">Drag markers or edit coordinates. Teal scanner · coral speaker · blue microphone. Canvas anchored to the first scanner pose.</p></div>
        <div class="lab-map-pair"><div class="lab-card"><h3>Captured observation</h3><canvas id="lab-observation" role="img" aria-label="Captured observation, unknown cells remain dark"></canvas><p class="lab-note">Dark: unknown · muted teal: free · bright teal: hits</p></div><div class="lab-card" id="lab-reference-card" hidden><h3>Reference target only</h3><canvas id="lab-reference" role="img" aria-label="Privileged simulation ground truth"></canvas><p class="lab-note">Evaluation only. Never model input.</p></div></div>
        <div class="lab-card"><div class="lab-result-line"><span id="lab-counts">No capture yet</span><label><input type="checkbox" id="lab-truth"> Show reference targets / paths</label></div><div class="lab-actions"><button id="lab-observation-3d" disabled>Inspect observation in 3D</button><button id="lab-compare" disabled>Compare A / D / truth</button></div><label>Model<select id="lab-model"></select></label><div class="lab-actions"><button id="lab-infer" disabled>Reconstruct</button><a id="lab-bundle" aria-disabled="true">Download capture bundle</a><a id="lab-wav" aria-disabled="true">RIR WAV</a></div><p id="lab-result" class="lab-note">Capture first. Both endpoints must be in observed free cells.</p></div>
      </section><section class="lab-readouts"><div class="lab-card"><h3>Model input RIR · first valid pair</h3><canvas id="lab-rir" role="img" aria-label="Processed 16 kHz room impulse response"></canvas><p class="lab-note">16 kHz · 80 ms · emission reference retained · peak normalized</p><h3>Recorded device signal</h3><canvas id="lab-recording" role="img" aria-label="Virtual microphone recording"></canvas><p class="lab-note">Plots scale amplitude to their own peak. Replay supplies the processed RIR only.</p></div><div class="lab-card"><h3>Virtual acquisition events</h3><ol id="lab-events"></ol><p id="lab-paths" class="lab-note"></p></div><p class="lab-note lab-limit">Direct and finite first-order wall/floor/ceiling paths. No higher-order reverb, scattering or diffraction. Software simulation, not validated hardware. Download bundles to retain captures.</p></section></div>`;
    document.body.append(dialog); dialog.showModal();
    const $ = (id) => dialog.querySelector('#lab-' + id), val = (id) => Number($(id).value);
    let pairs = [structuredClone(defaults)], pairIndex = 0, session = null, capture = null, reference = null, registered = [], samples = [], busy = false, playing = false, timer = null;
    const state = (text, error = false) => { $('status').textContent = text; $('status').classList.toggle('error', error); };
    function pause() { playing = false; clearTimeout(timer); $('play').textContent = 'Play replay'; }
    function invalidate() { capture = null; reference = null; $('state').textContent = 'CHANGED'; $('reference-card').hidden = true; paintGrid($('observation'), null); updateActions(); state('Settings changed. Capture sensors to update the observation.'); }
    function updateActions() {
      const ready = !!capture && !busy, valid = ready && capture.valid_pairs > 0;
      $('observation-3d').disabled = !ready;
      $('infer').disabled = !valid || !registered.some((m) => m.available);
      $('compare').disabled = !valid || !['A', 'D'].every((v) => registered.some((m) => m.available && m.family === 'EchoFusion' && m.variant === v && m.fold === 0));
      for (const id of ['bundle', 'wav']) { $(id).setAttribute('aria-disabled', String(!ready || id === 'wav' && !valid)); if (!capture) $(id).removeAttribute('href'); }
    }
    function pairFields() {
      const current = pairs[pairIndex];
      for (const key of ['source', 'receiver']) for (const [axis, i] of [['x', 0], ['y', 1], ['z', 2]]) $(key + '-' + axis).value = current[key + '_m'][i];
      $('pair').replaceChildren(...pairs.map((p, i) => new Option(`Pair ${i + 1}`, i))); $('pair').value = pairIndex;
      $('add-pair').disabled = pairs.length >= 8; $('remove-pair').disabled = pairs.length <= 1;
    }
    pairFields();
    function config() {
      if ($('mode').value === 'replay') return { mode: 'replay', sample_id: $('replay-sample').value, k: val('k') };
      return { mode: 'virtual', seed: val('seed'), k: val('k'), room: { shape: $('shape').value, width_m: val('width'), depth_m: val('depth'), height_m: val('height'), reflection: val('reflection') },
        scanner: { position_m: [val('scan-x'), val('scan-y'), val('scan-z')], heading_deg: val('heading'), fov_deg: val('fov'), range_m: val('range'), rays: val('rays'), dropout: val('dropout'), range_std_m: val('range-noise') },
        pairs, audio: { excitation: $('excitation').value, noise_snr_db: val('snr') >= 80 ? null : val('snr'), gain_db: val('gain'), lowpass_hz: val('lowpass') || null, offset_ms: val('offset'), clock_drift_ppm: val('drift'), adc_bits: val('bits'), clip_level: val('clip'), pose_noise_m: val('pose-noise') } };
    }
    let drag = null;
    function drawPlan() {
      const svg = $('plan'); if ($('mode').value !== 'virtual') { svg.innerHTML = '<text x="30" y="60" fill="#9db5bc">Audited replay: scanner-aligned observation below.</text>'; return; }
      const c = config(), room = c.room, scale = Math.min(530 / room.width_m, 340 / room.depth_m), tx = 55, ty = 45;
      const point = (p) => [tx + p[0] * scale, ty + p[2] * scale], vertices = roomVertices(room.shape, room.width_m, room.depth_m);
      let markup = `<defs><pattern id="lab-meter-grid" width="${scale}" height="${scale}" patternUnits="userSpaceOnUse" x="${tx}" y="${ty}"><path d="M ${scale} 0 L 0 0 0 ${scale}" fill="none" stroke="#29414f" stroke-width="1"/></pattern></defs><rect width="640" height="450" fill="#101f2a"/><polygon points="${vertices.map(([x, z]) => `${tx + x * scale},${ty + z * scale}`).join(' ')}" fill="url(#lab-meter-grid)" stroke="#89a5ad" stroke-width="3"/>`;
      const scanner = c.scanner.position_m, sp = point(scanner), angle = c.scanner.heading_deg * Math.PI / 180;
      if (capture?.beams?.length) for (let i = 0; i < capture.beams.length; i += Math.max(1, Math.floor(capture.beams.length / 80))) {
        const beam = capture.beams[i]; if (beam.range_m == null) continue;
        const end = point([scanner[0] + beam.range_m * Math.cos(beam.angle_rad), scanner[1], scanner[2] + beam.range_m * Math.sin(beam.angle_rad)]);
        markup += `<line x1="${sp[0]}" y1="${sp[1]}" x2="${end[0]}" y2="${end[1]}" stroke="#20c8c1" stroke-opacity=".2"/>`;
      }
      if (reference?.acoustic_paths) for (const path of reference.acoustic_paths.find((p) => p.pair_index === pairIndex)?.paths || []) markup += `<polyline points="${path.points_scene_m.map((p) => point(p).join(',')).join(' ')}" fill="none" stroke="#ffba66" stroke-opacity=".65" stroke-dasharray="5 4"/>`;
      const marker = (p, color, name, kind) => { const [x, z] = point(p); return `<g data-device="${kind}" tabindex="0" role="button" aria-label="Move ${name}"><circle cx="${x}" cy="${z}" r="11" fill="${color}" stroke="#07121b" stroke-width="3"/><text x="${x + 15}" y="${z + 4}" fill="${color}" font-size="13">${name}</text></g>`; };
      markup += `<line x1="${sp[0]}" y1="${sp[1]}" x2="${sp[0] + Math.cos(angle) * 38}" y2="${sp[1] + Math.sin(angle) * 38}" stroke="#20c8c1" stroke-width="3"/>`;
      markup += marker(scanner, '#20c8c1', 'Scanner', 'scan') + marker(pairs[pairIndex].source_m, '#ff6b6b', 'Speaker', 'source') + marker(pairs[pairIndex].receiver_m, '#5fa8ff', 'Microphone', 'receiver');
      markup += '<text x="55" y="425" fill="#9db5bc" font-size="13">X → · Z ↓ · grid = 1 m · Y is device height</text>';
      svg.innerHTML = markup; svg.dataset.scale = scale; svg.dataset.tx = tx; svg.dataset.ty = ty;
    }
    $('plan').addEventListener('pointerdown', (event) => { if (busy) return; const device = event.target.closest('[data-device]'); if (!device) return; drag = device.dataset.device; $('plan').setPointerCapture(event.pointerId); });
    $('plan').addEventListener('pointermove', (event) => {
      if (!drag || busy) return; const svg = $('plan'), rect = svg.getBoundingClientRect();
      const x = ((event.clientX - rect.left) / rect.width * 640 - Number(svg.dataset.tx)) / Number(svg.dataset.scale), z = ((event.clientY - rect.top) / rect.height * 450 - Number(svg.dataset.ty)) / Number(svg.dataset.scale);
      for (const [axis, value] of [['x', x], ['z', z]]) $(drag + '-' + axis).value = Math.max(.05, Math.min(9.95, value)).toFixed(2);
      if (drag !== 'scan') { pairs[pairIndex][drag + '_m'][0] = val(drag + '-x'); pairs[pairIndex][drag + '_m'][2] = val(drag + '-z'); }
      invalidate(); drawPlan();
    });
    $('plan').addEventListener('keydown', (event) => {
      const marker = event.target.closest('[data-device]');
      if (busy || !marker || !['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown'].includes(event.key)) return;
      event.preventDefault(); const device = marker.dataset.device, axis = ['ArrowLeft', 'ArrowRight'].includes(event.key) ? 'x' : 'z';
      const change = ['ArrowLeft', 'ArrowUp'].includes(event.key) ? -.1 : .1;
      $(device + '-' + axis).value = Math.max(.05, Math.min(9.95, val(device + '-' + axis) + change)).toFixed(2);
      if (device !== 'scan') pairs[pairIndex][device + '_m'][axis === 'x' ? 0 : 2] = val(device + '-' + axis);
      invalidate(); drawPlan(); $('plan').querySelector('[data-device="' + device + '"]').focus();
    });
    for (const kind of ['pointerup', 'pointercancel']) $('plan').addEventListener(kind, () => { drag = null; });
    $('virtual-fields').addEventListener('input', (event) => {
      if (event.target.tagName !== 'INPUT' && event.target.tagName !== 'SELECT') return;
      const match = event.target.id.match(/^lab-(source|receiver)-(x|y|z)$/);
      if (match) pairs[pairIndex][match[1] + '_m'][{ x: 0, y: 1, z: 2 }[match[2]]] = Number(event.target.value);
      if (event.target.id === 'lab-pair') return;
      invalidate(); drawPlan();
    });
    $('pair').addEventListener('change', () => { pairIndex = val('pair'); pairFields(); drawPlan(); });
    $('add-pair').onclick = () => { const p = structuredClone(pairs[pairIndex]); p.source_m[0] += .35; p.receiver_m[0] += .35; pairs.push(p); pairIndex = pairs.length - 1; pairFields(); invalidate(); drawPlan(); };
    $('remove-pair').onclick = () => { if (pairs.length > 1) { pairs.splice(pairIndex, 1); pairIndex = 0; pairFields(); invalidate(); drawPlan(); } };
    function modeChanged() { pause(); session = null; invalidate(); const replay = $('mode').value === 'replay'; $('virtual-fields').hidden = replay; $('replay-fields').hidden = !replay; for (const id of ['next', 'play', 'speed-wrap']) $(id).hidden = !replay; $('room-3d').hidden = replay; drawPlan(); }
    $('mode').onchange = modeChanged; $('replay-sample').onchange = () => { session = null; invalidate(); }; $('k').onchange = invalidate;
    async function run(action) { if (busy) return; busy = true; $('capture').disabled = true; const controls = [...dialog.querySelectorAll('input,select,#lab-add-pair,#lab-remove-pair,#lab-next,#lab-room-3d')]; const disabled = controls.map((el) => el.disabled); controls.forEach((el) => { el.disabled = true; }); updateActions(); try { await action(); } catch (error) { state(error.message, true); pause(); } finally { busy = false; controls.forEach((el, i) => { el.disabled = disabled[i]; }); $('capture').disabled = false; updateActions(); } }
    async function captureSensors() {
      state('Capturing sensor observations…'); const c = config();
      const changedRoom = session && (c.mode !== session.configuration.mode || (c.sample_id ?? null) !== (session.configuration.sample_id ?? null) || c.mode === 'virtual' && JSON.stringify(c.room) !== JSON.stringify(session.configuration.room));
      if (!session || changedRoom) session = await request('/sessions', post({ schema_version: 'echospace.simulation_config.v1', config: c }));
      else session = await request('/sessions/' + session.session_id, { ...post({ schema_version: 'echospace.simulation_config.v1', config: c }), method: 'PATCH' });
      capture = await request('/sessions/' + session.session_id + '/capture', post({})); reference = null;
      session = await request('/sessions/' + session.session_id); $('state').textContent = session.state.toUpperCase();
      paintGrid($('observation'), capture); paintSignal($('rir'), capture.waveforms[0] || []);
      let device = {}; if (capture.valid_pairs) device = (await request('/captures/' + capture.capture_id + '/signals')).observed_signals;
      paintSignal($('recording'), device.pair_0_recording || [], '#ffba66');
      $('counts').textContent = `${capture.observed_cell_count} observed cells · ${capture.valid_pairs} valid pairs · K used ${capture.k_used}`;
      $('events').replaceChildren(...session.events.slice(-12).map((e) => { const li = document.createElement('li'); li.textContent = `${e.time_s.toFixed(3)} s · ${e.type.replaceAll('_', ' ')}`; return li; }));
      $('result').textContent = capture.warning; $('bundle').href = PREFIX + '/sessions/' + session.session_id + '/bundle.zip'; $('bundle').download = '';
      $('wav').href = PREFIX + '/captures/' + capture.capture_id + '/audio.wav'; $('wav').download = '';
      state(capture.rejected_pairs?.length ? `Capture ready. ${capture.rejected_pairs.length} pair(s) rejected: move both endpoints into observed free cells.` : 'Capture ready. Select a model or compare A and D.');
      if ($('truth').checked) await referenceView(); drawPlan();
    }
    $('capture').onclick = () => run(captureSensors);
    async function referenceView() {
      $('reference-card').hidden = !$('truth').checked || !capture;
      if ($('truth').checked && capture) { reference = await request('/captures/' + capture.capture_id + '/reference.json'); paintGrid($('reference'), capture, reference);
        const paths = reference.acoustic_paths?.find((p) => p.pair_index === pairIndex)?.paths || []; $('paths').textContent = paths.length ? 'REFERENCE paths: ' + paths.map((p) => `${p.kind} ${p.arrival_ms.toFixed(2)} ms`).join(' · ') : 'Reference targets are evaluation only.';
      } else { reference = null; $('paths').textContent = ''; }
      drawPlan();
    }
    $('truth').onchange = () => run(referenceView);
    async function infer(modelId) { const result = await request('/sessions/' + session.session_id + '/infer', post({ model_id: modelId, k: val('k') })); return { result, grid: await request(result.links.grid, {}, false) }; }
    function inspect(item) { show({ ...buildGridScene(item.grid), kind: 'grid', name: item.grid.name, reconstruction: item.result }); dialog.close(); setStatus(item.result.warning || 'Simulation inference ready'); }
    $('infer').onclick = () => run(async () => { state('Running the registered checkpoint…'); const item = await infer($('model').value); $('result').textContent = `Missing-region IoU ${item.result.metrics.occupancy_iou.toFixed(3)} · ${item.result.warning}`; inspect(item); });
    $('compare').onclick = () => run(async () => { state('Comparing A and D on the same capture…'); const models = ['A', 'D'].map((v) => registered.find((m) => m.available && m.family === 'EchoFusion' && m.variant === v && m.fold === 0)); const items = []; for (const m of models) items.push(await infer(m.model_id)); showComparison(items, inspect); state('Comparison ready. Virtual scores are synthetic diagnostics.'); });
    $('observation-3d').onclick = () => { const scene = buildGridScene(observationGrid(capture)); const inferred = scene.layers.inferred; scene.group.remove(inferred); inferred.geometry.dispose(); inferred.material.dispose(); delete scene.layers.inferred; scene.stats.inferred_wall_m = 0; show({ ...scene, kind: 'observation', name: 'Captured observation · no hidden completion' }); dialog.close(); };
    $('room-3d').onclick = () => { show(setupScene(config())); dialog.close(); };
    async function nextSample() { if (!samples.length) throw new Error('No audited samples available'); $('replay-sample').selectedIndex = ($('replay-sample').selectedIndex + 1) % samples.length; session = null; invalidate(); await captureSensors(); }
    $('next').onclick = () => run(nextSample);
    $('play').onclick = () => { if (playing) { pause(); return; } playing = true; $('play').textContent = 'Pause replay'; const tick = async () => { if (!playing || !dialog.open) return; await run(nextSample); if (playing) timer = setTimeout(tick, 2000 / val('speed')); }; tick(); };
    $('close').onclick = () => dialog.close(); dialog.addEventListener('close', () => { pause(); dialog.remove(); }, { once: true });
    for (const id of ['bundle', 'wav']) $(id).onclick = (event) => { if ($(id).getAttribute('aria-disabled') === 'true') event.preventDefault(); };
    drawPlan(); paintGrid($('observation'), null); paintSignal($('rir')); paintSignal($('recording')); updateActions();
    try {
      await request('/capabilities'); const listing = await request('/v1/models', {}, false); registered = listing.models;
      $('model').replaceChildren(...registered.map((m) => { const o = new Option(`${m.family} ${m.variant}${m.available ? '' : ' (unavailable)'}`, m.model_id); o.disabled = !m.available; return o; }));
      const selected = registered.find((m) => m.available && m.variant === 'D') || registered.find((m) => m.available); if (selected) $('model').value = selected.model_id;
      const data = await request('/v1/samples?split=test&fold=0&limit=5000', {}, false); samples = data.samples;
      $('replay-sample').replaceChildren(...samples.map((s) => new Option(`${s.room_id} · ${s.occlusion_type} · ${s.coverage_bin}`, s.sample_id)));
      state('Engine ready. Place devices, then capture. First capture fixes the scanner anchor.'); updateActions();
    } catch (error) { state(`Local engine unavailable: ${error.message}. Start the API with the simulation branch.`, true); }
  });
}
