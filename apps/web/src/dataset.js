import * as THREE from 'three';
import { OBJLoader } from 'three/addons/loaders/OBJLoader.js';

export async function attachDatasetBrowser(show, setStatus) {
  const section = [...document.querySelectorAll('.sidebar .section-label')].find((el) => el.textContent.includes('LAYERS'));
  const panel = document.createElement('div');
  panel.className = 'dataset-panel';
  panel.innerHTML = [
    '<div class="section-label">02 / ACOUSTICROOMS</div>',
    '<label class="dataset-label" for="dataset-category">Room category</label>',
    '<select id="dataset-category"></select>',
    '<label class="dataset-label" for="dataset-room">Room mesh</label>',
    '<select id="dataset-room"></select>',
    '<button class="outline" id="dataset-open" type="button">View selected room <span>↗</span></button>',
    '<label class="dataset-label" for="dataset-cut">Section cut <span id="cut-value">75%</span></label>',
    '<input id="dataset-cut" type="range" min="30" max="100" value="75">',
    '<p class="hint" id="dataset-note">Reading local dataset…</p>'
  ].join('');
  section.before(panel);
  const categorySelect = panel.querySelector('#dataset-category');
  const roomSelect = panel.querySelector('#dataset-room');
  const openButton = panel.querySelector('#dataset-open');
  const cutInput = panel.querySelector('#dataset-cut');
  const note = panel.querySelector('#dataset-note');
  let rooms = [];
  let currentMesh = null;
  let cutPlane = null;
  let extent = null;

  function updateCut() {
    const percent = Number(cutInput.value);
    panel.querySelector('#cut-value').textContent = percent + '%';
    if (cutPlane && extent) cutPlane.constant = extent.min + extent.size * percent / 100 + 0.001;
  }
  cutInput.addEventListener('input', updateCut);

  function fillRooms() {
    const matching = rooms.filter((room) => room.category === categorySelect.value).sort((a, b) => {
      const na = Number(a.name.match(/_idx_(\d+)$/)?.[1]);
      const nb = Number(b.name.match(/_idx_(\d+)$/)?.[1]);
      return na - nb;
    });
    roomSelect.replaceChildren(...matching.map((room) => {
      const option = document.createElement('option');
      option.value = room.id; option.textContent = room.name;
      return option;
    }));
  }
  categorySelect.addEventListener('change', fillRooms);

  openButton.addEventListener('click', async () => {
    const selected = rooms.find((room) => room.id === roomSelect.value);
    if (!selected) return;
    openButton.disabled = true;
    setStatus('Loading ' + selected.name + '…');
    try {
      const url = '/api/acousticrooms/mesh?id=' + encodeURIComponent(selected.id);
      const object = await new OBJLoader().loadAsync(url);
      object.name = selected.name;
      object.rotation.x = -Math.PI / 2; // AcousticRooms OBJ coordinates are Z-up.
      object.updateMatrixWorld(true);
      const box = new THREE.Box3().setFromObject(object);
      if (box.isEmpty()) throw new Error('Room mesh is empty.');
      const size = box.getSize(new THREE.Vector3());
      extent = { min: box.min.y, size: size.y };
      cutPlane = new THREE.Plane(new THREE.Vector3(0, -1, 0), 0);
      updateCut();
      object.traverse((part) => {
        if (!part.isMesh) return;
        const materials = Array.isArray(part.material) ? part.material : [part.material];
        for (const material of materials) {
          material.side = THREE.DoubleSide;
          material.color?.setHex(0x84aebb);
          material.specular?.setHex(0x172b35);
          if ('shininess' in material) material.shininess = 12;
          material.clippingPlanes = [cutPlane];
          material.needsUpdate = true;
        }
      });
      currentMesh = object;
      show({
        group: object, layers: {}, kind: 'mesh', name: selected.name,
        stats: { width_m: size.x, depth_m: size.z, height_m: size.y }
      });
      note.textContent = 'AcousticRooms Z-up OBJ rotated into the Y-up viewer. Section cut hides upper geometry; RIR archives are absent locally.';
    } catch (error) {
      setStatus('Dataset load failed: ' + error.message, true);
    } finally {
      openButton.disabled = false;
    }
  });

  try {
    const response = await fetch('/api/acousticrooms');
    if (!response.ok) throw new Error('Local dataset endpoint unavailable.');
    const data = await response.json();
    if (!data.available) throw new Error(data.issue || 'Dataset unavailable.');
    rooms = data.rooms;
    const categories = [...new Set(rooms.map((room) => room.category))].sort();
    categorySelect.replaceChildren(...categories.map((category) => {
      const option = document.createElement('option');
      option.value = category;
      option.textContent = category + ' (' + rooms.filter((room) => room.category === category).length + ')';
      return option;
    }));
    categorySelect.value = categories.includes('Bedrooms') ? 'Bedrooms' : categories[0];
    fillRooms();
    note.textContent = rooms.length + ' OBJ room meshes available locally. Acoustic files are not present.';
    openButton.click();
  } catch (error) {
    note.textContent = error.message;
    openButton.disabled = true;
    categorySelect.disabled = true;
    roomSelect.disabled = true;
  }
}
