import * as THREE from 'three';

// Grid rendering for echospace.grid.v1 and echospace.grid.v2.
//
// Wall rule (shared with src/echospace/inference/export.py and checked against
// it by scripts/check-grid.mjs): a wall face stands on every edge between an
// interior cell and a non-interior cell or the canvas border. It is observed
// when the cell outside that edge was measured (v2: a measured wall hit; v1:
// any observed cell), otherwise inferred. A scanner inside the room never
// measures the far side of a wall, so the inside cell is not required to be
// observed. Scene frame: X right, Y up, Z forward; floor at Y = 0; the wall
// height is a display assumption, never a model output.

// [d_row, d_col, corner a (u, v), corner b (u, v)] for each cell edge
const EDGES = [
  [-1, 0, [0, 0], [1, 0]],
  [1, 0, [1, 1], [0, 1]],
  [0, -1, [0, 1], [0, 0]],
  [0, 1, [1, 0], [1, 1]],
];

function isBinaryRow(row, width) {
  return Array.isArray(row) && row.length === width && row.every((v) => v === 0 || v === 1);
}

function checkBinaryGrid(value, name, rows, cols) {
  if (!Array.isArray(value) || value.length !== rows || !value.every((row) => isBinaryRow(row, cols))) {
    throw new Error(`${name} must be a ${rows} × ${cols} grid of 0/1.`);
  }
}

function checkVector(value, name) {
  if (!Array.isArray(value) || value.length !== 3 || !value.every(Number.isFinite)) throw new Error(`${name} must be three numbers.`);
}

export function validateGrid(value) {
  if (!value || !['echospace.grid.v1', 'echospace.grid.v2'].includes(value.format)) {
    throw new Error('Expected format "echospace.grid.v1" or "echospace.grid.v2".');
  }
  const { interior, cell_size_m: cell } = value;
  if (!Array.isArray(interior) || !interior.length || !Array.isArray(interior[0]) || !interior[0].length) throw new Error('interior must be a nonempty 2D array.');
  const rows = interior.length, cols = interior[0].length;
  if (cols > 512 || rows > 512) throw new Error('Grid is too large for this viewer (max 512 × 512).');
  checkBinaryGrid(interior, 'interior', rows, cols);
  checkBinaryGrid(value.observed, 'observed', rows, cols);
  if (!Number.isFinite(cell) || cell <= 0 || cell > 10) throw new Error('cell_size_m must be a positive number in meters.');
  if (value.wall_height_m != null && (!Number.isFinite(value.wall_height_m) || value.wall_height_m <= 0 || value.wall_height_m > 20)) throw new Error('wall_height_m must be positive.');
  if (value.format === 'echospace.grid.v1') {
    const origin = value.origin_m;
    if (!Array.isArray(origin) || origin.length !== 2 || !origin.every(Number.isFinite)) throw new Error('origin_m must be [x, z] in meters.');
  } else {
    if (value.width !== cols || value.height !== rows) throw new Error('width/height must match the interior grid.');
    checkBinaryGrid(value.observed_wall, 'observed_wall', rows, cols);
    checkVector(value.origin_scene_m, 'origin_scene_m');
    checkVector(value.axis_u_scene, 'axis_u_scene');
    checkVector(value.axis_v_scene, 'axis_v_scene');
    const [u, v] = [value.axis_u_scene, value.axis_v_scene];
    const dot = u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
    if (Math.abs(Math.hypot(...u) - 1) > 1e-5 || Math.abs(Math.hypot(...v) - 1) > 1e-5 || Math.abs(dot) > 1e-5) throw new Error('Grid axes must be orthonormal.');
    if (value.truth) {
      checkBinaryGrid(value.truth.interior, 'truth.interior', rows, cols);
    }
  }
  if (!interior.some((row) => row.includes(1))) throw new Error('Grid contains no interior cells.');
  return value;
}

// Grid (col, row) corner -> scene point on the floor plane.
function cornerFunction(grid) {
  const cell = grid.cell_size_m;
  if (grid.format === 'echospace.grid.v1') {
    const [ox, oz] = grid.origin_m;
    return (col, row) => [ox + col * cell, 0, oz + row * cell];
  }
  const [o, u, v] = [grid.origin_scene_m, grid.axis_u_scene, grid.axis_v_scene];
  return (col, row) => [o[0] + cell * (col * u[0] + row * v[0]), 0, o[2] + cell * (col * u[2] + row * v[2])];
}

function pushQuad(target, a, b, c, d) {
  target.push(a, b, c, a, c, d);
}

// Triangles (as flat vertex lists) for floor, observed and inferred walls, plus stats.
export function gridGeometry(raw) {
  const grid = validateGrid(raw);
  const corner = cornerFunction(grid);
  const height = grid.wall_height_m ?? 2.7;
  const cell = grid.cell_size_m;
  const rows = grid.interior.length, cols = grid.interior[0].length;
  const hitGrid = grid.format === 'echospace.grid.v2' ? grid.observed_wall : grid.observed;
  const inside = (r, c) => r >= 0 && r < rows && c >= 0 && c < cols && grid.interior[r][c] === 1;
  const onCanvas = (r, c) => r >= 0 && r < rows && c >= 0 && c < cols;
  const floor = [], observed = [], inferred = [];
  let interiorCount = 0;
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      if (!inside(r, c)) continue;
      interiorCount++;
      pushQuad(floor, corner(c, r), corner(c, r + 1), corner(c + 1, r + 1), corner(c + 1, r));
      for (const [dr, dc, [ua, va], [ub, vb]] of EDGES) {
        const nr = r + dr, nc = c + dc;
        if (inside(nr, nc)) continue;
        const isObserved = onCanvas(nr, nc) && hitGrid[nr][nc] === 1;
        const a = corner(c + ua, r + va), b = corner(c + ub, r + vb);
        pushQuad(isObserved ? observed : inferred, a, b, [b[0], height, b[2]], [a[0], height, a[2]]);
      }
    }
  }
  const all = [...floor, ...observed, ...inferred];
  const min = [0, 1, 2].map((i) => Math.min(...all.map((p) => p[i])));
  const max = [0, 1, 2].map((i) => Math.max(...all.map((p) => p[i])));
  const stats = {
    floor_triangles: floor.length / 3,
    observed_wall_triangles: observed.length / 3,
    inferred_wall_triangles: inferred.length / 3,
    area_m2: interiorCount * cell * cell,
    observed_wall_m: (observed.length / 6) * cell,
    inferred_wall_m: (inferred.length / 6) * cell,
    bounds_min: min,
    bounds_max: max,
  };
  return { grid, floor, observed, inferred, stats, corner, height };
}

function makeMesh(points, color, opacity = 1) {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(points.flat(), 3));
  geometry.computeVertexNormals();
  const material = new THREE.MeshStandardMaterial({ color, side: THREE.DoubleSide, transparent: opacity < 1, opacity, roughness: 0.72, metalness: 0.05 });
  return new THREE.Mesh(geometry, material);
}

// Outline of a binary interior grid as line segments slightly above the floor.
function outline(cells, corner, color, y = 0.03) {
  const rows = cells.length, cols = cells[0].length, points = [];
  const inside = (r, c) => r >= 0 && r < rows && c >= 0 && c < cols && cells[r][c] === 1;
  for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
    if (!inside(r, c)) continue;
    for (const [dr, dc, [ua, va], [ub, vb]] of EDGES) {
      if (inside(r + dr, c + dc)) continue;
      const a = corner(c + ua, r + va), b = corner(c + ub, r + vb);
      points.push(a[0], y, a[2], b[0], y, b[2]);
    }
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(points, 3));
  return new THREE.LineSegments(geometry, new THREE.LineBasicMaterial({ color }));
}

function measurementMarkers(measurements) {
  const group = new THREE.Group();
  const source = new THREE.MeshStandardMaterial({ color: 0xff6b6b });
  const receiver = new THREE.MeshStandardMaterial({ color: 0x5fa8ff });
  const sphere = new THREE.SphereGeometry(0.09, 16, 12);
  for (const pair of measurements) {
    const s = new THREE.Mesh(sphere, source); s.position.set(...pair.source_scene_m); group.add(s);
    const r = new THREE.Mesh(sphere, receiver); r.position.set(...pair.receiver_scene_m); group.add(r);
  }
  return group;
}

export function buildGridScene(raw) {
  const { grid, floor, observed, inferred, stats, corner, height } = gridGeometry(raw);
  const group = new THREE.Group();
  group.name = grid.name || 'EchoSpace reconstruction';
  const layers = {};
  layers.floor = makeMesh(floor, 0x243746, 0.78); layers.floor.name = 'Floor';
  layers.observed = makeMesh(observed, 0x20c8c1, 0.88); layers.observed.name = 'Observed walls';
  layers.inferred = makeMesh(inferred, 0xffba66, 0.72); layers.inferred.name = 'Inferred walls';
  for (const key of ['floor', 'observed', 'inferred']) group.add(layers[key]);
  if (grid.format === 'echospace.grid.v2' && grid.truth) {
    layers.truth = outline(grid.truth.interior, corner, 0xf4f1de); layers.truth.name = 'Ground truth outline';
    group.add(layers.truth);
  }
  if (grid.format === 'echospace.grid.v2' && grid.measurements?.length) {
    layers.measurements = measurementMarkers(grid.measurements); layers.measurements.name = 'Sources and receivers';
    group.add(layers.measurements);
  }
  return {
    group,
    layers,
    stats: {
      width_m: grid.interior[0].length * grid.cell_size_m,
      depth_m: grid.interior.length * grid.cell_size_m,
      area_m2: stats.area_m2,
      observed_wall_m: stats.observed_wall_m,
      inferred_wall_m: stats.inferred_wall_m,
      cell_m: grid.cell_size_m,
      height_m: height,
    },
  };
}
