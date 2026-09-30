import * as THREE from 'three';

function isBinaryRow(row, width) {
  return Array.isArray(row) && row.length === width && row.every((v) => v === 0 || v === 1);
}

export function validateGrid(value) {
  if (!value || value.format !== 'echospace.grid.v1') throw new Error('Expected format "echospace.grid.v1".');
  const { interior, observed, cell_size_m: cell, origin_m: origin } = value;
  if (!Array.isArray(interior) || !interior.length || !Array.isArray(interior[0]) || !interior[0].length) throw new Error('interior must be a nonempty 2D array.');
  const width = interior[0].length;
  if (width > 512 || interior.length > 512) throw new Error('Grid is too large for this viewer (max 512 × 512).');
  if (!interior.every((row) => isBinaryRow(row, width))) throw new Error('interior rows must have equal width and contain 0/1.');
  if (!Array.isArray(observed) || observed.length !== interior.length || !observed.every((row) => isBinaryRow(row, width))) throw new Error('observed must match interior and contain 0/1.');
  if (!Number.isFinite(cell) || cell <= 0 || cell > 10) throw new Error('cell_size_m must be a positive number in meters.');
  if (!Array.isArray(origin) || origin.length !== 2 || !origin.every(Number.isFinite)) throw new Error('origin_m must be [x, z] in meters.');
  if (value.wall_height_m != null && (!Number.isFinite(value.wall_height_m) || value.wall_height_m <= 0 || value.wall_height_m > 20)) throw new Error('wall_height_m must be positive.');
  return value;
}

function addQuad(vertices, a, b, c, d) {
  vertices.push(...a, ...b, ...c, ...a, ...c, ...d);
}

function makeMesh(vertices, color, opacity = 1) {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
  geometry.computeVertexNormals();
  const material = new THREE.MeshStandardMaterial({ color, side: THREE.DoubleSide, transparent: opacity < 1, opacity, roughness: 0.72, metalness: 0.05 });
  return new THREE.Mesh(geometry, material);
}

export function buildGridScene(raw) {
  const grid = validateGrid(raw);
  const group = new THREE.Group();
  group.name = grid.name || 'EchoSpace reconstruction';
  const floor = [], observedWalls = [], inferredWalls = [];
  const height = grid.wall_height_m ?? 2.7;
  const cell = grid.cell_size_m;
  const [ox, oz] = grid.origin_m;
  const rows = grid.interior.length, cols = grid.interior[0].length;
  let observedCount = 0, inferredCount = 0, interiorCount = 0;

  const inside = (r, c) => r >= 0 && r < rows && c >= 0 && c < cols && grid.interior[r][c] === 1;
  const measured = (r, c) => r >= 0 && r < rows && c >= 0 && c < cols && grid.observed[r][c] === 1;
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      if (!inside(r, c)) continue;
      interiorCount++;
      const x0 = ox + c * cell, x1 = x0 + cell, z0 = oz + r * cell, z1 = z0 + cell;
      addQuad(floor, [x0, 0, z0], [x0, 0, z1], [x1, 0, z1], [x1, 0, z0]);
      const edges = [
        [-1, 0, [x0, 0, z0], [x1, 0, z0]],
        [1, 0, [x1, 0, z1], [x0, 0, z1]],
        [0, -1, [x0, 0, z1], [x0, 0, z0]],
        [0, 1, [x1, 0, z0], [x1, 0, z1]],
      ];
      for (const [dr, dc, a, b] of edges) {
        if (inside(r + dr, c + dc)) continue;
        const isObserved = measured(r, c) && measured(r + dr, c + dc);
        const target = isObserved ? observedWalls : inferredWalls;
        addQuad(target, a, b, [b[0], height, b[2]], [a[0], height, a[2]]);
        if (isObserved) observedCount++; else inferredCount++;
      }
    }
  }
  if (!interiorCount) throw new Error('Grid contains no interior cells.');
  const floorMesh = makeMesh(floor, 0x243746, 0.78); floorMesh.name = 'Floor'; group.add(floorMesh);
  const observedMesh = makeMesh(observedWalls, 0x20c8c1, 0.88); observedMesh.name = 'Observed walls'; group.add(observedMesh);
  const inferredMesh = makeMesh(inferredWalls, 0xffba66, 0.72); inferredMesh.name = 'Inferred walls'; group.add(inferredMesh);
  return { group, layers: { floor: floorMesh, observed: observedMesh, inferred: inferredMesh }, stats: { width_m: cols * cell, depth_m: rows * cell, area_m2: interiorCount * cell * cell, observed_wall_m: observedCount * cell, inferred_wall_m: inferredCount * cell, cell_m: cell, height_m: height } };
}
