// Pure comparison checks/palettes are shared with the Node regression check.
export function comparisonModels(models, selectedId) {
  const selected = models.find((m) => m.model_id === selectedId);
  const candidates = models.filter((m) => m.available && m.fold === selected?.fold && m.family === 'EchoFusion');
  const a = candidates.find((m) => m.variant === 'A');
  const d = candidates.find((m) => m.variant === 'D');
  if (!a || !d) throw new Error('Comparison needs available A and D checkpoints from the same fold.');
  return [a, d];
}

export function validateComparison(items) {
  if (items.length !== 2) throw new Error('Expected A and D results.');
  const [a, d] = items;
  for (const key of ['sample_id', 'k_requested', 'k_used']) {
    if (a.result[key] !== d.result[key]) throw new Error(`Comparison ${key} differs.`);
  }
  for (const key of ['width', 'height', 'cell_size_m', 'origin_scene_m', 'axis_u_scene', 'axis_v_scene', 'observed', 'observed_free', 'observed_wall', 'truth']) {
    if (JSON.stringify(a.grid[key]) !== JSON.stringify(d.grid[key])) throw new Error(`Comparison ${key} differs.`);
  }
  if (!a.grid.truth) throw new Error('This benchmark sample has no truth labels.');
}

export function comparisonColour(grid, row, col, kind, head = 'occupancy', threshold = .5) {
  if (kind === 'truth') {
    const cells = head === 'occupancy' ? grid.truth.interior : grid.truth.boundary;
    return cells[row][col] ? '#f4f1de' : '#142632';
  }
  if (grid.observed_wall[row][col]) return '#20c8c1';
  if (grid.observed_free[row][col]) return '#315b62';
  if (kind === 'observation') return '#142632';
  // Binary maps were thresholded before JSON probability rounding, exactly as
  // the metric. Rounded .4999 -> .500 must not flip a displayed prediction.
  const positive = head === 'occupancy' ? grid.interior[row][col] :
    grid.boundary_at_threshold ? grid.boundary_at_threshold[row][col] : grid.boundary_probability[row][col] >= threshold;
  return positive ? '#ffba66' : '#142632';
}

export function showComparison(items, onInspect) {
  validateComparison(items);
  const [a, d] = items;
  const grid = a.grid;
  document.querySelector('#comparison-dialog')?.remove();
  const dialog = document.createElement('dialog');
  dialog.id = 'comparison-dialog'; dialog.className = 'comparison-dialog';
  dialog.setAttribute('aria-labelledby', 'comparison-title');
  dialog.innerHTML = '<header><div><h2 id="comparison-title">Compare the same partial scan</h2><p id="comparison-sample"></p></div><button id="comparison-close" type="button" autofocus>Close</button></header>' +
    '<div class="comparison-controls"><label for="comparison-head">Map</label><select id="comparison-head"><option value="occupancy">Room occupancy</option><option value="boundary">Wall boundary</option></select><span>Teal: observed wall · muted teal: observed free · amber: model · cream: truth</span></div><div class="comparison-panels"></div>' +
    '<p class="comparison-caption">All panels use the same scanner grid frame (u →, v ↓). Dark cells in the observation are unknown, not measured empty. Predictions are thresholded raw probabilities in unseen cells; measured cells are shown as observation context. Scores use valid unseen cells only.</p>';
  dialog.querySelector('#comparison-sample').textContent = `${a.result.room_id} · ${a.result.sample_id} · K=${a.result.k_used} · ${grid.cell_size_m.toFixed(2)} m/cell`;
  const panels = [
    { title: 'Observation only', kind: 'observation', item: a },
    { title: 'A · Geometry only', kind: 'model', item: a },
    { title: 'D · Geometry + acoustics', kind: 'model', item: d },
    { title: 'Ground truth · reference', kind: 'truth', item: a },
  ];
  for (const panel of panels) {
    const figure = document.createElement('figure');
    const heading = document.createElement('figcaption'); heading.textContent = panel.title; figure.append(heading);
    panel.canvas = document.createElement('canvas'); panel.canvas.width = grid.width * 5; panel.canvas.height = grid.height * 5;
    panel.canvas.setAttribute('aria-label', panel.title + ', scanner-aligned grid'); panel.canvas.setAttribute('role', 'img');
    figure.append(panel.canvas);
    const score = document.createElement('p'); score.className = 'comparison-score';
    score.textContent = panel.kind === 'model' ? `Unseen IoU: occupancy ${panel.item.result.metrics.occupancy_iou.toFixed(3)} · boundary ${panel.item.result.metrics.boundary_iou.toFixed(3)}` : panel.kind === 'observation' ? `${Math.round(a.result.coverage * 100)}% observed · no hidden labels` : 'Evaluation labels only · never model input';
    figure.append(score);
    if (panel.kind === 'model') {
      const inspect = document.createElement('button'); inspect.textContent = 'Inspect in 3D';
      inspect.addEventListener('click', () => { dialog.close(); onInspect(panel.item); }); figure.append(inspect);
    }
    dialog.querySelector('.comparison-panels').append(figure);
  }
  const draw = () => {
    const head = dialog.querySelector('#comparison-head').value;
    for (const panel of panels) {
      const context = panel.canvas.getContext('2d');
      for (let row = 0; row < grid.height; row++) for (let col = 0; col < grid.width; col++) {
        context.fillStyle = comparisonColour(panel.item.grid, row, col, panel.kind, head, panel.item.result.threshold ?? .5);
        context.fillRect(col * 5, row * 5, 5, 5);
      }
    }
  };
  dialog.querySelector('#comparison-head').addEventListener('change', draw);
  dialog.querySelector('#comparison-close').addEventListener('click', () => dialog.close());
  dialog.addEventListener('click', (event) => { if (event.target === dialog) dialog.close(); });
  dialog.addEventListener('close', () => dialog.remove(), { once: true });
  document.body.append(dialog); draw(); dialog.showModal();
}
