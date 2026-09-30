import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { OBJLoader } from 'three/addons/loaders/OBJLoader.js';
import { PLYLoader } from 'three/addons/loaders/PLYLoader.js';
import { buildGridScene } from './grid.js';

function extension(name) { return name.split('.').pop()?.toLowerCase(); }
function revokeAll(urls) { for (const url of urls) URL.revokeObjectURL(url); }

export async function loadFiles(files) {
  const all = Array.from(files);
  const main = all.find((f) => ['json', 'glb', 'gltf', 'obj', 'ply'].includes(extension(f.name)));
  if (!main) throw new Error('Choose an EchoSpace JSON, GLB, glTF, OBJ, or PLY file.');
  const ext = extension(main.name);
  if (ext === 'json') return { ...buildGridScene(JSON.parse(await main.text())), kind: 'grid', name: main.name };

  const urls = new Map(all.map((f) => [f.name.replaceAll('\\', '/').split('/').pop(), URL.createObjectURL(f)]));
  const manager = new THREE.LoadingManager();
  manager.setURLModifier((url) => urls.get(decodeURIComponent(url.split('/').pop().split('?')[0])) || url);
  try {
    let object;
    if (ext === 'glb' || ext === 'gltf') object = (await new GLTFLoader(manager).loadAsync(urls.get(main.name))).scene;
    else if (ext === 'obj') object = await new OBJLoader(manager).loadAsync(urls.get(main.name));
    else {
      const header = await main.slice(0, 8192).text();
      const hasFaces = /^element face [1-9][0-9]*/m.test(header);
      const geometry = await new PLYLoader(manager).loadAsync(urls.get(main.name));
      if (hasFaces) {
        if (!geometry.getAttribute('normal')) geometry.computeVertexNormals();
        object = new THREE.Mesh(geometry, new THREE.MeshStandardMaterial({ color: 0x92b9c9, vertexColors: !!geometry.getAttribute('color'), side: THREE.DoubleSide }));
      } else {
        object = new THREE.Points(geometry, new THREE.PointsMaterial({ color: 0x92b9c9, size: 0.025, vertexColors: !!geometry.getAttribute('color') }));
      }
    }
    if (!object) throw new Error('No visible geometry was found.');
    object.name = main.name;
    return { group: object, layers: {}, kind: 'mesh', name: main.name, stats: null };
  } finally {
    setTimeout(() => revokeAll(urls.values()), 1000);
  }
}
