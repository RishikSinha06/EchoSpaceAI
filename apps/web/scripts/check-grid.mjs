// Agreement check: the browser renderer must reproduce the geometry the Python
// exporter writes into GLB files, for the shared echospace.grid.v2 fixture.
//   node apps/web/scripts/check-grid.mjs
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { gridGeometry } from '../src/grid.js';

const here = path.dirname(fileURLToPath(import.meta.url));
const fixture = JSON.parse(fs.readFileSync(path.join(here, '../../../tests/fixtures/grid_v2_agreement.json'), 'utf8'));
const { stats } = gridGeometry(fixture.document);
const expected = fixture.expected;
const problems = [];
for (const key of ['floor_triangles', 'observed_wall_triangles', 'inferred_wall_triangles']) {
  if (stats[key] !== expected[key]) problems.push(`${key}: browser ${stats[key]}, python ${expected[key]}`);
}
for (const key of ['area_m2', 'observed_wall_m', 'inferred_wall_m']) {
  if (Math.abs(stats[key] - expected[key]) > 1e-6) problems.push(`${key}: browser ${stats[key]}, python ${expected[key]}`);
}
for (const key of ['bounds_min', 'bounds_max']) {
  stats[key].forEach((value, i) => {
    if (Math.abs(value - expected[key][i]) > 1e-5) problems.push(`${key}[${i}]: browser ${value}, python ${expected[key][i]}`);
  });
}
if (problems.length) {
  console.error('Browser and GLB geometry disagree:\n  ' + problems.join('\n  '));
  process.exit(1);
}
console.log('grid v2 agreement: browser geometry matches the Python/GLB exporter', JSON.stringify(stats));
