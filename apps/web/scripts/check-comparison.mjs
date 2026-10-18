import assert from 'node:assert/strict';
import { comparisonModels, validateComparison, comparisonColour } from '../src/comparison.js';

const models = [
  { model_id: 'a', family: 'EchoFusion', variant: 'A', fold: 0, available: true },
  { model_id: 'd', family: 'EchoFusion', variant: 'D', fold: 0, available: true },
];
assert.deepEqual(comparisonModels(models, 'd').map((m) => m.model_id), ['a', 'd']);
assert.throws(() => comparisonModels([models[1]], 'd'), /available A and D/);
const grid = {
  width: 2, height: 2, cell_size_m: .2, origin_scene_m: [0, 0, 0], axis_u_scene: [1, 0, 0], axis_v_scene: [0, 0, 1],
  observed: [[1, 1], [0, 0]], observed_free: [[1, 0], [0, 0]], observed_wall: [[0, 1], [0, 0]],
  interior: [[1, 0], [1, 0]], occupancy_probability: [[.1, .9], [.8, .5]], boundary_probability: [[.1, .9], [.7, .5]],
  boundary_at_threshold: [[0, 1], [1, 0]], truth: { interior: [[1, 0], [1, 1]], boundary: [[1, 0], [1, 1]] },
};
const a = { result: { sample_id: 'scan', k_requested: 8, k_used: 4 }, grid };
const d = structuredClone(a);
validateComparison([a, d]);
d.result.sample_id = 'different';
assert.throws(() => validateComparison([a, d]), /sample_id/);
d.result.sample_id = 'scan'; d.grid.origin_scene_m = [1, 0, 0];
assert.throws(() => validateComparison([a, d]), /origin_scene_m/);
assert.equal(comparisonColour(grid, 1, 0, 'observation'), '#142632'); // unknown is never invented free/occupied
assert.equal(comparisonColour(grid, 0, 1, 'model'), '#20c8c1');
assert.equal(comparisonColour(grid, 1, 1, 'model'), '#142632'); // no flip from probability rounding
assert.equal(comparisonColour(grid, 1, 1, 'model', 'boundary'), '#142632');
assert.equal(comparisonColour(grid, 1, 1, 'truth'), '#f4f1de');
console.log('PASS: matched scan/frame/K, observation-only display and exact thresholded comparison maps');
