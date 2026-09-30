import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFExporter } from 'three/addons/exporters/GLTFExporter.js';
import { buildGridScene } from './grid.js';
import { loadFiles } from './loaders.js';
import './style.css';

document.querySelector('#app').innerHTML = `
  <header class="topbar"><div class="brand"><span class="brand-mark">◈</span><span>EchoSpace <em>AI</em></span></div><span class="top-label">ROOM EXPLORER / V0.1</span></header>
  <main class="layout">
    <aside class="sidebar">
      <div class="eyebrow">VISUAL WORKSPACE</div>
      <h1>See the room<br><span>take shape.</span></h1>
      <p class="intro">Inspect source geometry and model completions in a metric 3D view.</p>
      <div class="section-label">01 / SOURCE</div>
      <button class="primary" id="open">Open files <span>↗</span></button>
      <input id="file-input" type="file" multiple accept=".json,.glb,.gltf,.obj,.ply,.bin,.png,.jpg,.jpeg,.webp" hidden>
      <button class="subtle" id="example">Load example prediction</button>
      <p class="hint">GLB, glTF, OBJ, PLY, or EchoSpace grid JSON. For glTF, select its related files together.</p>
      <div class="section-label">03 / LAYERS</div>
      <label class="layer"><input type="checkbox" id="floor" checked><span class="swatch floor"></span> Interior floor</label>
      <label class="layer"><input type="checkbox" id="observed" checked><span class="swatch observed"></span> Observed boundary</label>
      <label class="layer"><input type="checkbox" id="inferred" checked><span class="swatch inferred"></span> Inferred boundary</label>
      <div class="section-label">04 / OUTPUT</div>
      <button class="outline" id="export" disabled>Export GLB <span>↓</span></button>
      <p class="hint">Open exported scenes in Blender or another native GLB viewer. The wall height is assumed for display.</p>
      <div class="sidebar-footer">2D inference → 3D inspection<br>EchoSpace AI research workspace</div>
    </aside>
    <section class="viewport-panel" id="drop-zone">
      <div class="view-head"><div><span class="live-dot"></span> <span id="scene-name">Example room</span></div><span id="scene-kind">PREDICTION PREVIEW</span></div>
      <div id="canvas-host"></div>
      <div class="view-tools"><button id="reset-view" title="Fit view">⌖ <span>Fit</span></button><button id="top-view" title="Top view">▦ <span>Top</span></button><button id="perspective-view" title="Perspective view">◇ <span>3D</span></button></div>
      <div class="view-foot"><span id="status">Loading example scene…</span><span>DRAG TO ORBIT · SCROLL TO ZOOM</span></div>
      <div class="drop-overlay" id="drop-overlay">Drop room files to inspect</div>
    </section>
    <aside class="inspector">
      <div class="section-label">SCENE INSPECTOR</div>
      <div class="info-card"><span class="card-label">ACTIVE SCENE</span><strong id="info-name">—</strong><span id="info-type">No scene</span></div>
      <div class="section-label">GEOMETRY</div>
      <div class="metric"><span>Width</span><b id="width">—</b></div>
      <div class="metric"><span>Depth</span><b id="depth">—</b></div>
      <div class="metric"><span>Floor area</span><b id="area">—</b></div>
      <div class="metric"><span>Cell size</span><b id="cell">—</b></div>
      <div class="metric"><span>Display height</span><b id="height">—</b></div>
      <div class="section-label">BOUNDARY</div>
      <div class="metric"><span>Observed</span><b id="obs-length">—</b></div>
      <div class="metric"><span>Inferred</span><b id="inf-length">—</b></div>
      <div class="inspector-note">Observed and inferred refer to the input mask. Source meshes are displayed without invented provenance.</div>
    </aside>
  </main>`;

const byId = (id) => document.getElementById(id);
const host = byId('canvas-host');
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x101c26);
scene.fog = new THREE.Fog(0x101c26, 30, 100);
scene.add(new THREE.HemisphereLight(0xddeeff, 0x25303a, 2.2));
const sun = new THREE.DirectionalLight(0xffffff, 2.2); sun.position.set(6, 12, 5); scene.add(sun);
const fill = new THREE.DirectionalLight(0x4bb9c8, 0.8); fill.position.set(-6, 4, -8); scene.add(fill);
const grid = new THREE.GridHelper(50, 50, 0x3c5561, 0x2a3e49); grid.position.y = -0.018; scene.add(grid);
const camera = new THREE.PerspectiveCamera(50, 1, 0.01, 10000);
camera.position.set(6, 6, 8);
const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: false });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.35;
host.append(renderer.domElement);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.minDistance = 0.1;
let active = null;
let bounds = null;

function resize() {
  const w = host.clientWidth, h = host.clientHeight;
  camera.aspect = w / Math.max(h, 1); camera.updateProjectionMatrix(); renderer.setSize(w, h, false);
}
new ResizeObserver(resize).observe(host); resize();
function render() { requestAnimationFrame(render); controls.update(); renderer.render(scene, camera); }
render();

function setStatus(message, error = false) { byId('status').textContent = message; byId('status').classList.toggle('error', error); }
function metric(id, value, unit = 'm') { byId(id).textContent = value == null ? '—' : `${value.toFixed(2)} ${unit}`; }
function updateInspector() {
  const stats = active?.stats;
  byId('info-name').textContent = active?.name || '—';
  byId('info-type').textContent = active?.kind === 'grid' ? 'GRID RECONSTRUCTION' : active ? 'SOURCE MESH' : 'No scene';
  metric('width', stats?.width_m); metric('depth', stats?.depth_m); metric('area', stats?.area_m2, 'm²');
  metric('cell', stats?.cell_m); metric('height', stats?.height_m);
  metric('obs-length', stats?.observed_wall_m); metric('inf-length', stats?.inferred_wall_m);
}
function fitView(mode = 'perspective') {
  if (!bounds) return;
  const center = bounds.getCenter(new THREE.Vector3());
  const size = bounds.getSize(new THREE.Vector3());
  const span = Math.max(size.x, size.z, size.y, 0.5);
  const distance = span / (2 * Math.tan(THREE.MathUtils.degToRad(camera.fov / 2))) * 1.65;
  controls.target.copy(center);
  if (mode === 'top') camera.position.set(center.x, center.y + distance, center.z + 0.0001);
  else camera.position.set(center.x + distance * 0.7, center.y + distance * 0.58, center.z + distance * 0.8);
  camera.near = Math.max(0.001, distance / 1000); camera.far = Math.max(100, distance * 100); camera.updateProjectionMatrix();
  controls.update();
}
function show(result) {
  if (active) {
    scene.remove(active.group);
    active.group.traverse((object) => { object.geometry?.dispose(); const materials = object.material ? (Array.isArray(object.material) ? object.material : [object.material]) : []; materials.forEach((m) => m.dispose()); });
  }
  active = result; scene.add(result.group);
  bounds = new THREE.Box3().setFromObject(result.group);
  if (bounds.isEmpty()) throw new Error('This file contains no visible geometry.');
  fitView();
  byId('scene-name').textContent = result.name || result.group.name;
  byId('scene-kind').textContent = result.kind === 'grid' ? 'PREDICTION / GRID' : 'SOURCE / MESH';
  for (const key of ['floor', 'observed', 'inferred']) { byId(key).disabled = !result.layers[key]; result.layers[key] && (result.layers[key].visible = byId(key).checked); }
  byId('export').disabled = false;
  updateInspector(); setStatus('Scene ready');
}

byId('open').onclick = () => byId('file-input').click();
byId('file-input').onchange = async (event) => {
  if (!event.target.files.length) return;
  try { setStatus('Opening files…'); show(await loadFiles(event.target.files)); } catch (error) { setStatus(error.message, true); }
  event.target.value = '';
};
byId('example').onclick = async () => {
  try { setStatus('Loading example…'); const response = await fetch('/examples/example-grid.json'); if (!response.ok) throw new Error('Example scene unavailable.'); const result = buildGridScene(await response.json()); show({ ...result, kind: 'grid', name: 'Example L-shaped room' }); } catch (error) { setStatus(error.message, true); }
};
for (const key of ['floor', 'observed', 'inferred']) byId(key).onchange = (event) => { if (active?.layers[key]) active.layers[key].visible = event.target.checked; };
byId('reset-view').onclick = () => fitView();
byId('top-view').onclick = () => fitView('top');
byId('perspective-view').onclick = () => fitView('perspective');
byId('export').onclick = async () => {
  if (!active) return;
  try {
    setStatus('Exporting GLB…');
    const bytes = await new GLTFExporter().parseAsync(active.group, { binary: true, onlyVisible: true });
    const url = URL.createObjectURL(new Blob([bytes], { type: 'model/gltf-binary' }));
    const a = document.createElement('a'); a.href = url; a.download = `${(active.name || 'echospace-room').replace(/\.[^.]+$/, '').replace(/[^a-z0-9_-]+/gi, '-')}.glb`; a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000); setStatus('GLB exported');
  } catch (error) { setStatus(`Export failed: ${error.message}`, true); }
};
const zone = byId('drop-zone');
let dragDepth = 0;
zone.addEventListener('dragenter', (e) => { e.preventDefault(); dragDepth++; byId('drop-overlay').classList.add('visible'); });
zone.addEventListener('dragover', (e) => e.preventDefault());
zone.addEventListener('dragleave', (e) => { e.preventDefault(); dragDepth--; if (dragDepth <= 0) { dragDepth = 0; byId('drop-overlay').classList.remove('visible'); } });
zone.addEventListener('drop', async (e) => { e.preventDefault(); dragDepth = 0; byId('drop-overlay').classList.remove('visible'); try { setStatus('Opening files…'); show(await loadFiles(e.dataTransfer.files)); } catch (error) { setStatus(error.message, true); } });
byId('example').click();

renderer.localClippingEnabled = true;
import('./dataset.js').then((module) => module.attachDatasetBrowser(show, setStatus));
